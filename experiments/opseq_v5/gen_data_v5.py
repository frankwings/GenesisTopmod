#!/usr/bin/env python3
"""
gen_data_v5.py — Generate opseq_v5 dataset: classification + shape regression.

Same as V4 but also saves shape_params [N, 3] = axis-aligned half-extents
[hx, hy, hz] of the normalized mesh. These represent the coarse proportions
(elongation/flattening) that the model should predict.

Shard layout (.npz):
  images       : [N, 4, 128, 128] uint8   (white-bg silhouettes, 255=bg, 0=fg)
  labels       : [N]              int8     (class ID, 0..19)
  shape_params : [N, 3]           float32  (axis-aligned half-extents)

Usage:
    python gen_data_v5.py [--n_train 50000] [--n_val 5000] [--seed 42]
                          [--out_dir experiments/opseq_v5/data]
                          [--shard_size 1000] [--device cuda]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

# ── Repo root on sys.path ────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from canonical_topos import CANONICAL_TOPOS, N_CLASSES, build_topo_mesh
from topmod.diffgeo  import _fan_triangulate
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

# ── Constants ────────────────────────────────────────────────────────────────

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0
COORD_LO      = -2.0
COORD_HI      =  2.0


# ═════════════════════════════════════════════════════════════════════════════
# Vertex deformations (same as V4)
# ═════════════════════════════════════════════════════════════════════════════

def _anisotropic_scale(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return verts * rng.uniform(0.5, 1.5, 3)


def _random_rotation(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.standard_normal(3); u /= np.linalg.norm(u) + 1e-8
    theta = float(rng.uniform(0, 2 * math.pi))
    c, s  = math.cos(theta / 2), math.sin(theta / 2)
    qw, qx, qy, qz = c, s*u[0], s*u[1], s*u[2]
    R = np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qw*qz),    2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),    1-2*(qx**2+qz**2),  2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx),    1-2*(qx**2+qy**2)],
    ], dtype=np.float64)
    return verts @ R.T


def _radial_bumps(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    dirs  = verts / np.maximum(norms, 1e-6)
    result = verts.copy()
    for _ in range(int(rng.integers(1, 4))):
        bump_dir  = rng.standard_normal(3); bump_dir /= np.linalg.norm(bump_dir) + 1e-8
        width     = float(rng.uniform(0.4, 1.5))
        amplitude = float(rng.uniform(0.05, 0.3))
        angle     = np.arccos(np.clip((dirs * bump_dir).sum(axis=1), -1.0, 1.0))
        result   += amplitude * np.exp(-(angle**2) / (2*width**2))[:, None] * dirs
    return result


def _twist(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = float(rng.uniform(-math.pi, math.pi))
    z_range = float(verts[:, 2].max() - verts[:, 2].min())
    if z_range < 1e-6:
        return verts
    z_norm = (verts[:, 2] - verts[:, 2].min()) / z_range
    angles = z_norm * rate
    result = verts.copy()
    result[:, 0] = np.cos(angles)*verts[:, 0] - np.sin(angles)*verts[:, 1]
    result[:, 1] = np.sin(angles)*verts[:, 0] + np.cos(angles)*verts[:, 1]
    return result


_DEFORM_FNS = [_anisotropic_scale, _random_rotation, _radial_bumps, _twist]


def apply_deformations(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_deform = int(rng.integers(2, 4))
    chosen   = rng.choice(len(_DEFORM_FNS), size=n_deform, replace=False)
    for idx in chosen:
        verts = _DEFORM_FNS[idx](verts, rng)
    return verts


def normalize_verts(verts: np.ndarray) -> np.ndarray:
    mn, mx = float(verts.min()), float(verts.max())
    extent = max(mx - mn, 1e-6)
    scale  = 0.8 * (COORD_HI - COORD_LO) / extent
    centre = (mn + mx) / 2.0
    return (verts - centre) * scale


def compute_half_extents(verts: np.ndarray) -> np.ndarray:
    """
    Compute axis-aligned half-extents: max absolute value along each axis.
    Returns [3] float32 array, values typically in [0, ~1.6].
    """
    return np.abs(verts).max(axis=0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Single sample generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample_v5(
    rng:    np.random.Generator,
    ctx:    dr.RasterizeCudaContext,
    device: str = 'cuda',
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Generate one v5 (images, class_label, shape_params) training sample.

    Returns
    -------
    images       : [4, 128, 128] uint8   (white-bg, 255=bg, 0=fg)
    label        : int                   (class ID, 0..N_CLASSES-1)
    shape_params : [3] float32           (axis-aligned half-extents)
    """
    # 1. Pick random canonical topology (uniform)
    label = int(rng.integers(0, N_CLASSES))
    topo  = CANONICAL_TOPOS[label]

    # 2. Build mesh
    verts_np, tris_np = build_topo_mesh(topo)

    if verts_np.shape[0] < 3 or tris_np.shape[0] == 0:
        raise ValueError(f"Degenerate mesh for topo {topo['label']}")

    # 3. Apply random deformations
    verts_np = apply_deformations(verts_np, rng)

    # 4. Normalize
    verts_np = normalize_verts(verts_np)

    # 5. Compute shape params (after deformation + normalization)
    shape_params = compute_half_extents(verts_np)

    # 6. Render 4-view silhouettes
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )

    sil_views: List[np.ndarray] = []
    for i in range(4):
        sil    = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                   resolution=(IMG_RES, IMG_RES))
        sil_np = sil[0, :, :, 0].detach().cpu().numpy()
        img    = ((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8)
        sil_views.append(img)

    images = np.stack(sil_views, axis=0)
    return images, label, shape_params


# ═════════════════════════════════════════════════════════════════════════════
# Shard I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_shard_v5(
    images_list: List[np.ndarray],
    labels_list: List[int],
    shapes_list: List[np.ndarray],
    path:        str,
) -> None:
    images = np.stack(images_list, axis=0)
    labels = np.array(labels_list, dtype=np.int8)
    shapes = np.stack(shapes_list, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels, shape_params=shapes)
    print(f"  saved {len(images_list)} samples -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opseq_v5 dataset")
    parser.add_argument('--n_train',    type=int, default=50000)
    parser.add_argument('--n_val',      type=int, default=5000)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--shard_size', type=int, default=1000)
    parser.add_argument('--out_dir',    type=str,
                        default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--device',     type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctx = dr.RasterizeCudaContext()

    n_total = args.n_train + args.n_val
    print(f"Generating {args.n_train} train + {args.n_val} val samples")
    print(f"  N_CLASSES: {N_CLASSES}")
    print(f"  Device: {args.device}")

    splits = {
        'train': list(range(args.n_train)),
        'val':   list(range(args.n_train, n_total)),
    }

    for split_name, indices in splits.items():
        split_dir  = os.path.join(args.out_dir, split_name)
        shard_size = args.shard_size if split_name == 'train' else len(indices) + 1

        images_buf: List[np.ndarray] = []
        labels_buf: List[int]        = []
        shapes_buf: List[np.ndarray] = []
        shard_idx  = 0
        n_ok = n_fail = 0
        t0 = time.time()

        for sample_idx, global_idx in enumerate(indices):
            rng = np.random.default_rng(args.seed + global_idx)

            try:
                images, label, shape_params = generate_sample_v5(
                    rng, ctx, device=args.device,
                )
            except Exception as exc:
                n_fail += 1
                if n_fail <= 10:
                    print(f"  [FAIL] {split_name}[{global_idx}]: {exc}")
                continue

            images_buf.append(images)
            labels_buf.append(label)
            shapes_buf.append(shape_params)
            n_ok += 1

            if len(images_buf) >= shard_size:
                path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
                save_shard_v5(images_buf, labels_buf, shapes_buf, path)
                images_buf, labels_buf, shapes_buf = [], [], []
                shard_idx += 1

            if (sample_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = (sample_idx + 1) / max(elapsed, 1e-6)
                eta     = (len(indices) - sample_idx - 1) / max(rate, 1e-6)
                print(f"  {split_name}: {sample_idx+1}/{len(indices)} "
                      f"| ok={n_ok} fail={n_fail} "
                      f"| {rate:.1f} smp/s | ETA {eta:.0f}s")

        if images_buf:
            path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
            save_shard_v5(images_buf, labels_buf, shapes_buf, path)

        elapsed = time.time() - t0
        print(f"{split_name}: {n_ok} ok, {n_fail} failed, {elapsed:.1f}s")

    print("V5 dataset generation complete.")


if __name__ == '__main__':
    main()
