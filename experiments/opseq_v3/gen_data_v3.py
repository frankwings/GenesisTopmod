#!/usr/bin/env python3
"""
gen_data_v3.py — Generate the opseq_v3 training dataset (Phase A'').

Topology-only paradigm: the model predicts ONLY the topology program
(base primitive + HDL ops + subdivision ops). Geometry is determined by
direct vertex optimization via nvdiffrast at inference time.

Grammar
-------
- Base primitive: uniformly from {cube, tetrahedron, icosahedron}
- 0–2 HDL ops  : adds genus handles
- 0–3 linear ops: from the 17 globally-defined linear ops
  (LOOP/SQRT3 only sampled when mesh is all-triangular)
- Cage deformations applied before rendering so silhouettes look distinct
  for the same topology (prevents the model from learning a trivial mapping)
- Silhouettes: 4 views (azimuths 0/90/180/270°), 128×128, WHITE background
  (255=bg, 0=fg object). render_silhouette returns 1=fg → invert to store.

Shard layout (.npz):
  images   : [N, 4, 128, 128] uint8   (white-bg silhouettes, 255=bg, 0=fg)
  tokens   : [N, max_len]     int16   (padded with PAD_ID=98)
  lengths  : [N]              int16   (actual length including EOS)
  genera   : [N]              int8    (ground-truth genus = # HDL ops applied)

Usage:
    python gen_data_v3.py [--n_train 20000] [--n_val 2000] [--seed 42]
                          [--out_dir experiments/opseq_v3/data]
                          [--shard_size 1000] [--device cuda]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Set, Tuple

import numpy as np
import torch

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr

from topmod.primitives     import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.diffgeo        import mesh_to_arrays, _fan_triangulate, LINEAR_OPS
from topmod.tokenizer      import (
    build_vocabulary_v3,
    encode_v3,
    DEFAULT_COORD_LO,
    DEFAULT_COORD_HI,
    _find_compatible_face_pair,
    _face_ordinal,
)
from topmod.subdivision import catmull_clark
from topmod.remeshing import (
    dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide, honeycomb_subdivide, corner_cutting,
    loop_style_subdivide, pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide, checkerboard_remesh, ds_bc_new_subdivide,
)
from topmod.high_level_ops import stellate_all
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

# ── Vocabulary constants ───────────────────────────────────────────────────────
N_REF    = 64
VOCAB_V3 = build_vocabulary_v3(n_ref=N_REF)

BOS_ID   = VOCAB_V3['BOS']   # 97
PAD_ID   = VOCAB_V3['PAD']   # 98
EOS_ID   = VOCAB_V3['EOS']   # 0

MAX_SEQ_LEN_V3 = 30   # 1 BASE + up to 3×(HDL+REF+REF) + up to 3 OPs + EOS = 17 max typical

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0
COORD_LO      = DEFAULT_COORD_LO   # -2.0
COORD_HI      = DEFAULT_COORD_HI   # +2.0

# Base primitives
_PRIM_FNS = {
    'cube':        make_cube,
    'tetrahedron': make_tetrahedron,
    'icosahedron': make_icosahedron,
}
_BASE_NAMES = list(_PRIM_FNS.keys())

# Only sample from the 17 verified-differentiable linear ops
_SAMPLE_OPS = list(LINEAR_OPS)

# Topology-aware operator sampling
_TRI_ONLY_OPS      = {'LOOP', 'SQRT3'}
_TRI_PRODUCING_OPS = {'STA', 'LOOP', 'SQRT3', 'LSTYLE', 'FRAC'}
_QUAD_PRODUCING_OPS = {'CC', 'DUAL', 'DS', 'PENT', 'PENT2', 'D1264',
                        'ROOT4', 'CHKB', 'DSBC', 'HONEY', 'CCUT', 'VC', 'SIMP'}

# Float operator dispatch (mirrors tokenizer._GLOBAL_OPS for applying ops to DLFLMesh)
def _sta(mesh):
    stellate_all(mesh)
    return mesh

_FLOAT_OPS = {
    'CC':    catmull_clark,
    'DUAL':  dual,
    'DS':    doo_sabin,
    'STA':   _sta,
    'SIMP':  simplest_subdivide,
    'VC':    vertex_cutting,
    'LOOP':  loop_subdivide,
    'SQRT3': sqrt3_subdivide,
    'HONEY': honeycomb_subdivide,
    'CCUT':  corner_cutting,
    'LSTYLE': loop_style_subdivide,
    'PENT':  pentagonal_subdivide,
    'PENT2': pentagonal2_subdivide,
    'D1264': dual1264_subdivide,
    'ROOT4': root4_subdivide,
    'CHKB':  checkerboard_remesh,
    'DSBC':  ds_bc_new_subdivide,
}


# ═════════════════════════════════════════════════════════════════════════════
# Mesh building
# ═════════════════════════════════════════════════════════════════════════════

def build_mesh_from_program(
    base_name: str,
    hdl_pairs: List[Tuple[int, int]],
    op_names:  List[str],
):
    """
    Execute a topology program on a fresh DLFL mesh.

    Parameters
    ----------
    base_name : 'cube' | 'tetrahedron' | 'icosahedron'
    hdl_pairs : list of (f1_ord, f2_ord) for HDL ops
    op_names  : list of zero-argument subdivision op names

    Returns
    -------
    DLFLMesh — the resulting mesh after all ops applied.
    """
    mesh = _PRIM_FNS[base_name]()

    # Apply HDL (face ordinals refer to the current face list at each step)
    for f1_ord, f2_ord in hdl_pairs:
        faces_list = list(mesh.faces.values())
        if f1_ord < len(faces_list) and f2_ord < len(faces_list):
            add_handle(mesh, faces_list[f1_ord], faces_list[f2_ord])

    # Apply subdivision ops
    for op in op_names:
        fn = _FLOAT_OPS.get(op)
        if fn is not None:
            mesh = fn(mesh)

    return mesh


# ═════════════════════════════════════════════════════════════════════════════
# Vertex deformations (applied to final mesh vertices before rendering)
# ═════════════════════════════════════════════════════════════════════════════

def _anisotropic_scale(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale each axis independently in [0.5, 1.5]."""
    return verts * rng.uniform(0.5, 1.5, 3)


def _random_rotation(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3-D rotation via quaternion."""
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
    """Gaussian radial bumps."""
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
    """Twist around Z axis proportionally to z coordinate."""
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
    """Apply 2–3 random global deformations."""
    n_deform = int(rng.integers(2, 4))
    chosen   = rng.choice(len(_DEFORM_FNS), size=n_deform, replace=False)
    for idx in chosen:
        verts = _DEFORM_FNS[idx](verts, rng)
    return verts


def normalize_verts(verts: np.ndarray) -> np.ndarray:
    """Normalize to 80% of the quantization range [-2, +2]."""
    mn, mx = float(verts.min()), float(verts.max())
    extent = max(mx - mn, 1e-6)
    scale  = 0.8 * (COORD_HI - COORD_LO) / extent
    centre = (mn + mx) / 2.0
    return (verts - centre) * scale


# ═════════════════════════════════════════════════════════════════════════════
# Mesh → renderable tensors
# ═════════════════════════════════════════════════════════════════════════════

def mesh_to_render_tensors(
    positions: list,
    faces:     list,
    verts_np:  np.ndarray,
    device:    str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (verts [V,3] float32, tris [T,3] int32) for nvdiffrast.

    Parameters
    ----------
    positions : list of (x,y,z) tuples from mesh_to_arrays (vertex ordering)
    faces     : polygon ring lists from mesh_to_arrays
    verts_np  : [V, 3] float64 deformed vertex positions (same ordering)
    device    : CUDA device string
    """
    tris = _fan_triangulate(faces)   # list of (i0, i1, i2) tuples
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(tris,    dtype=torch.int32,   device=device)
    return verts_t, faces_t


# ═════════════════════════════════════════════════════════════════════════════
# Single sample generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample_v3(
    rng:    np.random.Generator,
    ctx:    dr.RasterizeCudaContext,
    device: str = 'cuda',
) -> Tuple[np.ndarray, List[int], int]:
    """
    Generate one V3 (images, token_ids, genus) training sample.

    Steps
    -----
    1. Sample base primitive, HDL count, linear op sequence.
    2. Build DLFL mesh via float ops (not differentiable — topology only).
    3. Apply deformations to mesh vertex positions.
    4. Normalize deformed vertices.
    5. Render 4 white-background silhouettes.
    6. Encode V3 topology-only token sequence.

    Returns
    -------
    images    : [4, 128, 128] uint8   (white-bg, 255=bg, 0=fg)
    token_ids : flat int list (no BOS, includes EOS), max ~10-15 tokens
    genus     : int (number of HDL ops actually applied)

    Raises
    ------
    ValueError / RuntimeError on failure (caller should catch and retry).
    """
    # ── 1. Sample parameters ─────────────────────────────────────────────
    base_name = str(rng.choice(_BASE_NAMES))
    n_hdl     = int(rng.choice([0, 0, 0, 1, 1, 2]))   # bias toward genus-0

    # Op depth: geometric dist with mean ~1.5, clipped to [0, 3]
    raw_depth = int(rng.geometric(p=0.45)) - 1
    op_depth  = max(0, min(raw_depth, 3))

    # Topology-aware op sampling: skip LOOP/SQRT3 on non-triangular mesh
    op_names: List[str] = []
    # cube=quads, tetra/ico=tris; HDL adds quads
    is_all_tri = (base_name != 'cube') and (n_hdl == 0)
    for _ in range(op_depth):
        available = [op for op in _SAMPLE_OPS
                     if not (op in _TRI_ONLY_OPS and not is_all_tri)]
        if not available:
            break
        chosen = str(rng.choice(available))
        op_names.append(chosen)
        if chosen in _TRI_PRODUCING_OPS:
            is_all_tri = True
        elif chosen in _QUAD_PRODUCING_OPS:
            is_all_tri = False

    # ── 2. Build DLFL mesh + record HDL pairs ─────────────────────────────
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
        if max(f1_ord, f2_ord) >= N_REF:
            break   # out of REF range
        excluded_vids |= {v.id for v in f1.vertices()}
        excluded_vids |= {v.id for v in f2.vertices()}
        hdl_pairs.append((f1_ord, f2_ord))
        add_handle(mesh, f1, f2)

    actual_genus = len(hdl_pairs)

    # Apply subdivision ops to the mesh
    for op in op_names:
        fn = _FLOAT_OPS.get(op)
        if fn is not None:
            mesh = fn(mesh)

    # ── 3. Extract vertex positions ────────────────────────────────────────
    positions, faces = mesh_to_arrays(mesh)
    verts_np = np.array(positions, dtype=np.float64)   # [V, 3]

    if verts_np.shape[0] < 3 or len(faces) == 0:
        raise ValueError("Empty or degenerate mesh")

    # ── 4. Apply deformations + normalize ─────────────────────────────────
    verts_np = apply_deformations(verts_np, rng)
    verts_np = normalize_verts(verts_np)

    # ── 5. Render 4 white-background silhouettes ──────────────────────────
    verts_t, faces_t = mesh_to_render_tensors(positions, faces, verts_np, device)

    if faces_t.shape[0] == 0:
        raise ValueError("No triangles after fan triangulation")

    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )
    sil_views: List[np.ndarray] = []
    for i in range(4):
        sil    = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                   resolution=(IMG_RES, IMG_RES))
        sil_np = sil[0, :, :, 0].detach().cpu().numpy()   # [H,W], 1=fg
        # WHITE background: 255=bg, 0=fg  (invert from 1=fg, 0=bg)
        img    = ((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8)
        sil_views.append(img)

    images = np.stack(sil_views, axis=0)   # [4, H, W] uint8

    # ── 6. Encode V3 topology-only token sequence ─────────────────────────
    token_ids = encode_v3(base_name, hdl_pairs, op_names, VOCAB_V3)

    if len(token_ids) > MAX_SEQ_LEN_V3:
        raise ValueError(
            f"Sequence too long: {len(token_ids)} > {MAX_SEQ_LEN_V3}"
        )

    return images, token_ids, actual_genus


# ═════════════════════════════════════════════════════════════════════════════
# Shard I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_shard_v3(
    images_list:    List[np.ndarray],
    token_ids_list: List[List[int]],
    genera_list:    List[int],
    path:           str,
) -> None:
    """Pad token sequences and write one .npz shard."""
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
    print(f"  saved {N} samples → {path}  (max_seq_len={max_len})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opseq_v3 topology-only dataset")
    parser.add_argument('--n_train',    type=int, default=20000)
    parser.add_argument('--n_val',      type=int, default=2000)
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
    print(f"  Vocab size: {len(VOCAB_V3)} (topology-only)")
    print(f"  MAX_SEQ_LEN: {MAX_SEQ_LEN_V3}")
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
                images, token_ids, genus = generate_sample_v3(
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

            if len(images_buf) >= shard_size:
                path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
                save_shard_v3(images_buf, tokens_buf, genera_buf, path)
                images_buf, tokens_buf, genera_buf = [], [], []
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
            save_shard_v3(images_buf, tokens_buf, genera_buf, path)

        elapsed = time.time() - t0
        print(f"{split_name}: {n_ok} ok, {n_fail} failed, {elapsed:.1f}s")

    print("V3 topology-only dataset generation complete.")


if __name__ == '__main__':
    main()
