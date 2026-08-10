#!/usr/bin/env python3
"""
gen_data_v2.py — Generate the opseq_v2 training dataset.

Phase A' paradigm: model predicts SHORT topology program (~10–50 tokens) +
coarse cage coordinates, then a differentiable executor refines the cage
against target silhouettes.

Grammar
-------
- Base primitive: uniformly from {cube, tetrahedron, icosahedron}
- 0–2 HDL ops  : adds genus handles before building DiffSequence
- 0–4 linear ops: sampled from the 17 globally-defined linear ops
- Cage         : verts0 of the DiffSequence (HDL-modified base)
- Deformations : global (rotation + anisotropic scale + shear) +
                 local (30–70 % of vertices perturbed ±0.3)
- Normalize    : 80 % of quantization range [−2, +2]
- Silhouettes  : 4 views (azimuths 0/90/180/270°), 128×128, WHITE background
                 (255 = background, 0 = foreground mesh)

Shard layout (.npz):
  images   : [N, 4, 128, 128] uint8   (white-background silhouettes)
  tokens   : [N, max_len]     int16   (padded with PAD_ID=355)
  lengths  : [N]              int16   (actual sequence length, includes EOS)
  genera   : [N]              int8    (ground-truth genus = # HDL ops)

Usage:
    python gen_data_v2.py [--n_train 20000] [--n_val 2000] [--seed 42]
                          [--out_dir experiments/opseq_v2/data]
                          [--shard_size 1000] [--device cuda]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr

from topmod.dlfl         import DLFLMesh
from topmod.primitives   import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.validate     import is_manifold
from topmod.io           import to_triangle_arrays
from topmod.diffgeo      import DiffSequence, mesh_to_arrays, LINEAR_OPS
from topmod.tokenizer    import (
    build_vocabulary_v2,
    encode_v2,
    DEFAULT_COORD_LO,
    DEFAULT_COORD_HI,
    dequantize_coord,
    _find_compatible_face_pair,
    _face_ordinal,
)
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

# ── Constants ─────────────────────────────────────────────────────────────────

N_COORD_BINS = 256
N_REF        = 64
VOCAB_V2     = build_vocabulary_v2(n_coord_bins=N_COORD_BINS, n_ref=N_REF)
VOCAB_SIZE_V2 = len(VOCAB_V2)       # 356

BOS_ID   = VOCAB_V2['BOS']          # 354
PAD_ID   = VOCAB_V2['PAD']          # 355
EOS_ID   = VOCAB_V2['EOS']          # 0

MAX_SEQ_LEN_V2 = 200                 # safe ceiling (topology ≤ ~50 + coords ≤ ~150)
AZIMUTHS       = [0.0, 90.0, 180.0, 270.0]
IMG_RES        = 128
CAMERA_RADIUS  = 3.0
COORD_LO       = DEFAULT_COORD_LO   # -2.0
COORD_HI       = DEFAULT_COORD_HI   # +2.0

# Base primitives map
_PRIM_FNS = {
    'cube':         make_cube,
    'tetrahedron':  make_tetrahedron,
    'icosahedron':  make_icosahedron,
}
_BASE_NAMES = list(_PRIM_FNS.keys())

# Only sample from the 17 verified-differentiable linear ops
_SAMPLE_OPS = list(LINEAR_OPS)   # ('CC','DUAL','DS','STA','SIMP','VC','LOOP',
                                  #  'SQRT3','HONEY','CCUT','LSTYLE','PENT','PENT2',
                                  #  'D1264','ROOT4','CHKB','DSBC')

# Operators that require all-triangle input — only valid after tri-producing ops
_TRI_ONLY_OPS = {'LOOP', 'SQRT3'}
# Operators that always produce all-triangle output
_TRI_PRODUCING_OPS = {'STA', 'LOOP', 'SQRT3', 'LSTYLE', 'FRAC'}
# Operators that always produce non-triangle faces (quads, pentagons, etc.)
_QUAD_PRODUCING_OPS = {'CC', 'DUAL', 'DS', 'PENT', 'PENT2', 'D1264',
                        'ROOT4', 'CHKB', 'DSBC', 'HONEY', 'CCUT', 'VC', 'SIMP'}


# ═════════════════════════════════════════════════════════════════════════════
# Smooth vertex deformations
# ═════════════════════════════════════════════════════════════════════════════

def _anisotropic_scale(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale each axis independently by a factor in [0.5, 1.5]."""
    scales = rng.uniform(0.5, 1.5, 3)
    return verts * scales


def _random_rotation(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a uniformly random 3-D rotation."""
    # Random axis-angle via quaternion
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u) + 1e-8
    theta = float(rng.uniform(0, 2 * math.pi))
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    qw, qx, qy, qz = c, s * u[0], s * u[1], s * u[2]
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),        1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),        2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    return verts @ R.T


def _shear(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a mild shear transformation."""
    S = np.eye(3, dtype=np.float64)
    S[0, 1] = float(rng.uniform(-0.3, 0.3))
    S[0, 2] = float(rng.uniform(-0.3, 0.3))
    S[1, 2] = float(rng.uniform(-0.3, 0.3))
    return verts @ S.T


def _radial_bumps(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian radial bumps on the surface."""
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    dirs  = verts / np.maximum(norms, 1e-6)
    result = verts.copy()
    for _ in range(int(rng.integers(1, 4))):
        bump_dir  = rng.standard_normal(3)
        bump_dir /= np.linalg.norm(bump_dir) + 1e-8
        width     = float(rng.uniform(0.4, 1.5))
        amplitude = float(rng.uniform(0.05, 0.3))
        dots      = np.clip((dirs * bump_dir).sum(axis=1), -1.0, 1.0)
        angle     = np.arccos(dots)
        bump      = amplitude * np.exp(-(angle ** 2) / (2.0 * width ** 2))
        result   += bump[:, None] * dirs
    return result


_GLOBAL_DEFORM_FNS = [_anisotropic_scale, _random_rotation, _shear, _radial_bumps]


def _local_perturbation(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb a random subset (30–70%) of vertices by up to ±0.3."""
    V        = verts.shape[0]
    frac     = float(rng.uniform(0.30, 0.70))
    n_perturb = max(1, int(frac * V))
    indices  = rng.choice(V, size=n_perturb, replace=False)
    offsets  = rng.uniform(-0.3, 0.3, (n_perturb, 3))
    result   = verts.copy()
    result[indices] += offsets
    return result


def apply_deformations_v2(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply global deformations (2–3 chosen randomly) then local perturbation.
    """
    # Global: pick 2 or 3 of the 4 global deformation functions
    n_global = int(rng.integers(2, 4))
    chosen   = rng.choice(len(_GLOBAL_DEFORM_FNS), size=n_global, replace=False)
    for idx in chosen:
        verts = _GLOBAL_DEFORM_FNS[idx](verts, rng)

    # Local perturbation always applied
    verts = _local_perturbation(verts, rng)
    return verts


# ═════════════════════════════════════════════════════════════════════════════
# Single sample generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample_v2(
    rng:    np.random.Generator,
    ctx:    dr.RasterizeCudaContext,
    device: str = 'cuda',
) -> Tuple[np.ndarray, List[int], int]:
    """
    Generate one V2 (images, token_ids, genus) training sample.

    Steps
    -----
    1. Sample base primitive, HDL count (0–2), and linear op sequence (0–4).
    2. Build DLFL mesh, apply HDL ops, record face ordinal pairs.
    3. Extract (positions, faces) → DiffSequence base (cage = verts0).
    4. Append linear ops to DiffSequence.
    5. Apply deformations to cage vertices.
    6. Normalize cage into 80% of quantization range.
    7. Encode V2 token sequence (topology + SEP + coords + EOS).
    8. Forward DiffSequence with normalized cage → final mesh.
    9. Render 4 white-background silhouettes at azimuths 0/90/180/270°.

    Returns
    -------
    images    : [4, 128, 128] uint8   (white-bg, 255=bg, 0=fg)
    token_ids : flat int list (no BOS, includes EOS)
    genus     : int (number of HDL ops actually applied)

    Raises
    ------
    ValueError / RuntimeError : on failure (caller should catch and retry).
    """
    # ── 1. Sample parameters ─────────────────────────────────────────────
    base_name = str(rng.choice(_BASE_NAMES))
    n_hdl     = int(rng.choice([0, 0, 0, 1, 1, 2]))   # bias toward genus-0
    # Op depth: geometric with mean ~1.5, clipped to [0, 4]
    raw_depth = int(rng.geometric(p=0.4)) - 1
    op_depth  = max(0, min(raw_depth, 4))

    # Sample ops with topology-aware filtering:
    # tri-only ops (LOOP, SQRT3) can only follow a tri-producing state
    op_names: List[str] = []
    # Base primitives: cube=quads, tetrahedron=tris, icosahedron=tris
    # HDL creates quad faces, so any HDL applied marks mesh as non-all-tri
    is_all_tri = (base_name != 'cube') and (n_hdl == 0)
    for _ in range(op_depth):
        available = [op for op in _SAMPLE_OPS
                     if not (op in _TRI_ONLY_OPS and not is_all_tri)]
        if not available:
            break
        chosen = str(rng.choice(available))
        op_names.append(chosen)
        # Update mesh state for next iteration
        if chosen in _TRI_PRODUCING_OPS:
            is_all_tri = True
        elif chosen in _QUAD_PRODUCING_OPS:
            is_all_tri = False
        # STA produces tris; keep is_all_tri as True after it

    # ── 2. Build DLFL mesh + HDL ops ─────────────────────────────────────
    mesh = _PRIM_FNS[base_name]()
    hdl_pairs: List[Tuple[int, int]] = []
    excluded_vids: Set[int] = set()

    for _ in range(n_hdl):
        try:
            f1, f2 = _find_compatible_face_pair(mesh, excluded_vids)
        except ValueError:
            break
        f1_ord = _face_ordinal(mesh, f1)
        f2_ord = _face_ordinal(mesh, f2)
        excluded_vids |= {v.id for v in f1.vertices()}
        excluded_vids |= {v.id for v in f2.vertices()}
        hdl_pairs.append((f1_ord, f2_ord))
        add_handle(mesh, f1, f2)

    actual_genus = len(hdl_pairs)

    # Check REF ordinals are within vocabulary range
    max_f_ord = max((max(p) for p in hdl_pairs), default=-1)
    if max_f_ord >= N_REF:
        raise ValueError(f"Face ordinal {max_f_ord} exceeds REF range {N_REF}")

    # ── 3. Extract (positions, faces) → DiffSequence base ────────────────
    positions, faces = mesh_to_arrays(mesh)

    seq = DiffSequence(
        (positions, faces),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    # ── 4. Append linear ops ──────────────────────────────────────────────
    for op in op_names:
        try:
            seq.append(op)
        except Exception as exc:
            raise ValueError(f"Failed to append op {op!r}: {exc}") from exc

    # ── 5. Get cage vertices, apply deformations ──────────────────────────
    cage_np = seq.verts0.detach().cpu().numpy().astype(np.float64)  # [V_cage, 3]
    cage_np = apply_deformations_v2(cage_np, rng)

    # ── 6. Normalize into 80% of quantization range ───────────────────────
    mn, mx = float(cage_np.min()), float(cage_np.max())
    extent = max(mx - mn, 1e-6)
    scale  = 0.8 * (COORD_HI - COORD_LO) / extent
    centre = (mn + mx) / 2.0
    cage_norm = (cage_np - centre) * scale   # fits in ≈ [−1.6, +1.6]

    # ── 7. Encode V2 token sequence ───────────────────────────────────────
    token_ids = encode_v2(
        base_name, hdl_pairs, op_names,
        cage_norm, VOCAB_V2,
        coord_lo=COORD_LO, coord_hi=COORD_HI, n_coord_bins=N_COORD_BINS,
    )

    if len(token_ids) > MAX_SEQ_LEN_V2:
        raise ValueError(
            f"Sequence too long: {len(token_ids)} > {MAX_SEQ_LEN_V2}"
        )

    # ── 8. Forward DiffSequence with normalized cage ──────────────────────
    cage_t = torch.tensor(cage_norm, dtype=torch.float32, device=device)
    seq.verts0 = cage_t

    with torch.no_grad():
        final_verts = seq.forward()   # [V_final, 3]

    tris = seq.triangles(device=device).to(torch.int32)  # [T, 3] int32 for nvdiffrast

    if final_verts.shape[0] < 3 or tris.shape[0] == 0:
        raise ValueError("Empty or degenerate mesh after DiffSequence forward")

    # ── 9. Render 4 white-background silhouettes ──────────────────────────
    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )
    sil_views: List[np.ndarray] = []
    for i in range(4):
        sil    = render_silhouette(ctx, final_verts, tris, mvps[i],
                                   resolution=(IMG_RES, IMG_RES))
        sil_np = sil[0, :, :, 0].detach().cpu().numpy()    # [H, W] float32, 1=fg
        # Convert to WHITE background: 255=bg, 0=fg
        img    = ((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8)
        sil_views.append(img)

    images = np.stack(sil_views, axis=0)   # [4, H, W] uint8

    return images, token_ids, actual_genus


# ═════════════════════════════════════════════════════════════════════════════
# Shard I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_shard_v2(
    images_list:    List[np.ndarray],
    token_ids_list: List[List[int]],
    genera_list:    List[int],
    path:           str,
) -> None:
    """Pad token sequences and write one .npz shard (int16 tokens for compactness)."""
    N       = len(images_list)
    max_len = max(len(ids) for ids in token_ids_list)

    images  = np.stack(images_list, axis=0)                       # [N, 4, H, W]
    tokens  = np.full((N, max_len), PAD_ID, dtype=np.int16)
    lengths = np.zeros(N, dtype=np.int16)
    genera  = np.array(genera_list, dtype=np.int8)

    for i, ids in enumerate(token_ids_list):
        L              = len(ids)
        tokens[i, :L]  = np.array(ids, dtype=np.int16)
        lengths[i]     = L

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        images=images, tokens=tokens, lengths=lengths, genera=genera,
    )
    print(f"  saved {N} samples → {path}  (seq_len≤{max_len})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opseq_v2 dataset")
    parser.add_argument('--n_train',    type=int, default=20000)
    parser.add_argument('--n_val',      type=int, default=2000)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--shard_size', type=int, default=1000,
                        help="Max samples per train shard")
    parser.add_argument('--out_dir',    type=str,
                        default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--device',     type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctx = dr.RasterizeCudaContext()

    n_total = args.n_train + args.n_val
    print(f"Generating {args.n_train} train + {args.n_val} val samples "
          f"using V2 grammar (seed={args.seed}) …")
    print(f"  Vocab size: {VOCAB_SIZE_V2}, MAX_SEQ_LEN: {MAX_SEQ_LEN_V2}")
    print(f"  Device: {args.device}")

    splits = {
        'train': list(range(args.n_train)),
        'val':   list(range(args.n_train, n_total)),
    }

    for split_name, indices in splits.items():
        split_dir  = os.path.join(args.out_dir, split_name)
        shard_size = args.shard_size if split_name == 'train' else len(indices) + 1

        images_buf: List[np.ndarray] = []
        tokens_buf: List[List[int]]  = []
        genera_buf: List[int]        = []
        shard_idx  = 0
        n_ok = n_fail = 0
        t0 = time.time()

        for sample_idx, global_idx in enumerate(indices):
            rng = np.random.default_rng(args.seed + global_idx)

            try:
                images, token_ids, genus = generate_sample_v2(
                    rng, ctx, device=args.device,
                )
            except Exception as exc:
                n_fail += 1
                if n_fail <= 10:
                    print(f"  [FAIL] {split_name}[{global_idx}]: {exc}")
                continue

            images_buf.append(images)
            tokens_buf.append(token_ids)
            genera_buf.append(genus)
            n_ok += 1

            # Flush shard
            if len(images_buf) >= shard_size:
                path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
                save_shard_v2(images_buf, tokens_buf, genera_buf, path)
                images_buf, tokens_buf, genera_buf = [], [], []
                shard_idx += 1

            if (sample_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = (sample_idx + 1) / max(elapsed, 1e-6)
                eta     = (len(indices) - sample_idx - 1) / max(rate, 1e-6)
                print(f"  {split_name}: {sample_idx+1}/{len(indices)} "
                      f"| ok={n_ok} fail={n_fail} "
                      f"| {rate:.1f} smp/s | ETA {eta:.0f}s")

        # Flush remaining
        if images_buf:
            path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
            save_shard_v2(images_buf, tokens_buf, genera_buf, path)

        elapsed = time.time() - t0
        print(f"{split_name}: {n_ok} ok, {n_fail} failed, {elapsed:.1f}s")

    print("V2 dataset generation complete.")


if __name__ == '__main__':
    main()
