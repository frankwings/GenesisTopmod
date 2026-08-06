#!/usr/bin/env python3
"""
gen_data.py — Generate the opseq training dataset.

Samples random operator programs: genus g ∈ {0,1,2}, CC rounds k ∈ {1,2}.
Applies 2-4 random smooth vertex deformations, renders 4 silhouette views
(azimuths 0/90/180/270, elevation 0, 128×128), and saves as .npz shards.

Shard layout:
  images   : [N, 4, 128, 128] uint8
  tokens   : [N, max_len]     int32  (padded with PAD_ID=263)
  lengths  : [N]              int32  (actual sequence length, includes EOS)
  genera   : [N]              int32  (ground-truth genus)

Usage:
    python gen_data.py [--n_train 8000] [--n_val 500] [--seed 42]
                       [--out_dir experiments/opseq/data]
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

from topmod.dlfl        import DLFLMesh
from topmod.primitives  import make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.subdivision import catmull_clark
from topmod.validate    import is_manifold
from topmod.tokenizer   import (
    TopModToken,
    detokenize,
    build_vocabulary,
    encode_sequence,
    quantize_coord,
    DEFAULT_COORD_LO,
    DEFAULT_COORD_HI,
    _find_compatible_face_pair,
    _face_ordinal,
)
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

# ── Shared vocabulary constants ───────────────────────────────────────────────

N_BINS      = 128        # coordinate quantisation bins
MAX_ORDINAL = 128        # max face/edge ordinal reference
VOCAB       = build_vocabulary(n_position_bins=N_BINS, max_ordinal=MAX_ORDINAL)
VOCAB_SIZE  = len(VOCAB)  # 6 + 128 + 128 = 262
BOS_ID      = VOCAB_SIZE          # 262 — start token (not in tokeniser vocab)
PAD_ID      = VOCAB_SIZE + 1      # 263 — padding (ignore_index in loss)
EOS_ID      = VOCAB['EOS']        # 0

MAX_SEQ_LEN  = 1200      # safe ceiling (g=2,k=2 → ~1153 ids)
AZIMUTHS     = [0.0, 90.0, 180.0, 270.0]
IMG_RES      = 128
CAMERA_RADIUS = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Smooth vertex deformations
# ═════════════════════════════════════════════════════════════════════════════

def _anisotropic_scale(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale each axis by an independent random factor in [0.5, 2.0]."""
    scales = rng.uniform(0.5, 2.0, 3)
    return verts * scales


def _radial_bumps(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian bumps in the radial direction (spherical-harmonic flavour)."""
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    dirs  = verts / np.maximum(norms, 1e-6)
    result = verts.copy()
    n_bumps = int(rng.integers(2, 5))
    for _ in range(n_bumps):
        bump_dir   = rng.standard_normal(3)
        bump_dir  /= np.linalg.norm(bump_dir) + 1e-6
        width      = float(rng.uniform(0.3, 1.2))
        amplitude  = float(rng.uniform(0.05, 0.4))
        dots       = np.clip((dirs * bump_dir).sum(axis=1), -1.0, 1.0)
        angle      = np.arccos(dots)
        bump       = amplitude * np.exp(-(angle ** 2) / (2.0 * width ** 2))
        result    += bump[:, None] * dirs
    return result


def _twist(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate x,y by an angle proportional to z (twist)."""
    twist_rate = float(rng.uniform(-math.pi, math.pi))
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-6:
        return verts
    z_norm   = (verts[:, 2] - z_min) / z_range
    angles   = z_norm * twist_rate
    cos_a    = np.cos(angles)
    sin_a    = np.sin(angles)
    result   = verts.copy()
    result[:, 0] = cos_a * verts[:, 0] - sin_a * verts[:, 1]
    result[:, 1] = sin_a * verts[:, 0] + cos_a * verts[:, 1]
    return result


def _taper(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale x,y by a linear function of z (taper)."""
    base_scale = float(rng.uniform(0.5, 1.5))
    top_scale  = float(rng.uniform(0.5, 1.5))
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    z_range = z_max - z_min
    if z_range < 1e-6:
        return verts
    z_norm    = (verts[:, 2] - z_min) / z_range
    scale_xy  = base_scale + (top_scale - base_scale) * z_norm
    result    = verts.copy()
    result[:, 0] *= scale_xy
    result[:, 1] *= scale_xy
    return result


_DEFORM_FNS = [_anisotropic_scale, _radial_bumps, _twist, _taper]


def apply_deformations(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compose 2–4 randomly chosen deformations in sequence."""
    n      = int(rng.integers(2, 5))          # 2, 3, or 4
    chosen = rng.choice(len(_DEFORM_FNS), size=n, replace=False)
    for idx in chosen:
        verts = _DEFORM_FNS[idx](verts, rng)
    return verts


# ═════════════════════════════════════════════════════════════════════════════
# Mesh → render tensors
# ═════════════════════════════════════════════════════════════════════════════

def mesh_to_tensors(
    mesh:   DLFLMesh,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert DLFLMesh to (verts [V,3] float32, faces [F,3] int32) for
    nvdiffrast.  Quads and n-gons are fan-triangulated.
    """
    verts_list = list(mesh.vertices.values())
    vid_map    = {v.id: i for i, v in enumerate(verts_list)}

    verts_t = torch.tensor(
        [(v.x, v.y, v.z) for v in verts_list],
        dtype=torch.float32, device=device,
    )

    tris = []
    for f in mesh.faces.values():
        vs = list(f.vertices())
        n  = len(vs)
        if n < 3:
            continue
        elif n == 3:
            tris.append([vid_map[vs[0].id], vid_map[vs[1].id], vid_map[vs[2].id]])
        elif n == 4:
            tris.append([vid_map[vs[0].id], vid_map[vs[1].id], vid_map[vs[2].id]])
            tris.append([vid_map[vs[0].id], vid_map[vs[2].id], vid_map[vs[3].id]])
        else:
            for i in range(1, n - 1):
                tris.append([vid_map[vs[0].id], vid_map[vs[i].id], vid_map[vs[i+1].id]])

    faces_t = torch.tensor(tris, dtype=torch.int32, device=device)
    return verts_t, faces_t


# ═════════════════════════════════════════════════════════════════════════════
# Single sample generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample(
    g:      int,
    k:      int,
    rng:    np.random.Generator,
    ctx:    dr.RasterizeCudaContext,
    device: str = 'cuda',
) -> Tuple[np.ndarray, List[int], int]:
    """
    Generate one (images, token_ids, genus) training sample.

    Steps
    -----
    1. Build working mesh: icosahedron → g handles → k CC rounds.
    2. Emit HDL, CC tokens recording face ordinals from the live mesh state.
    3. Apply 2–4 random smooth deformations to vertex positions.
    4. Normalise positions into [−1.6, +1.6] ⊂ [coord_lo, coord_hi].
    5. Emit CV tokens (quantised normalised positions), then EOS.
    6. Verify: detokenize(tokens) ⟹ is_manifold.
    7. Render 4 silhouettes (azimuths 0/90/180/270, elevation 0, 128×128).

    Returns
    -------
    images    : [4, 128, 128] uint8
    token_ids : flat integer-encoded sequence (length varies, NOT padded)
    genus     : int ground-truth genus
    """
    lo, hi = DEFAULT_COORD_LO, DEFAULT_COORD_HI

    mesh   = make_icosahedron()
    tokens: List[TopModToken] = []
    excluded_vids: Set[int]   = set()

    # ── 1 & 2  Structural tokens ──────────────────────────────────────
    for _ in range(g):
        f1, f2   = _find_compatible_face_pair(mesh, excluded_vids)
        f1_ord   = _face_ordinal(mesh, f1)
        f2_ord   = _face_ordinal(mesh, f2)
        excluded_vids |= {v.id for v in f1.vertices()}
        excluded_vids |= {v.id for v in f2.vertices()}
        tokens.append(TopModToken(op='HDL',
                                   corner1=(f1_ord, 0),
                                   corner2=(f2_ord, 0)))
        add_handle(mesh, f1, f2)

    for _ in range(k):
        tokens.append(TopModToken(op='CC'))
        mesh = catmull_clark(mesh)

    # ── 3  Deform vertices ────────────────────────────────────────────
    verts_np = np.array(
        [(v.x, v.y, v.z) for v in mesh.vertices.values()],
        dtype=np.float64,
    )
    verts_np = apply_deformations(verts_np, rng)

    # ── 4  Normalise into quantisation range ──────────────────────────
    mn, mx = float(verts_np.min()), float(verts_np.max())
    extent = max(mx - mn, 1e-6)
    scale  = 0.8 * (hi - lo) / extent           # 80 % of [−2, +2]
    centre = (mn + mx) / 2.0
    verts_norm = (verts_np - centre) * scale     # fits in [−1.6, +1.6]

    # ── Apply normalised positions back to mesh ───────────────────────
    for i, v in enumerate(mesh.vertices.values()):
        v.x = float(verts_norm[i, 0])
        v.y = float(verts_norm[i, 1])
        v.z = float(verts_norm[i, 2])

    # ── 5  CV tokens ──────────────────────────────────────────────────
    for i in range(verts_norm.shape[0]):
        x, y, z = verts_norm[i]
        tokens.append(TopModToken(op='CV', pos=(
            quantize_coord(x, lo, hi, N_BINS),
            quantize_coord(y, lo, hi, N_BINS),
            quantize_coord(z, lo, hi, N_BINS),
        )))
    tokens.append(TopModToken(op='EOS'))

    # ── 6  Verify manifold ────────────────────────────────────────────
    reconstructed = detokenize(tokens)
    if not is_manifold(reconstructed):
        raise RuntimeError(f"Non-manifold after detokenize (g={g}, k={k})")

    # ── Encode ────────────────────────────────────────────────────────
    token_ids = encode_sequence(tokens, VOCAB)
    if len(token_ids) > MAX_SEQ_LEN:
        raise ValueError(
            f"Sequence too long: {len(token_ids)} > {MAX_SEQ_LEN} "
            f"(g={g}, k={k})"
        )

    # ── 7  Render 4 silhouettes ───────────────────────────────────────
    verts_t, faces_t = mesh_to_tensors(mesh, device)
    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )
    sil_views: List[np.ndarray] = []
    for i in range(4):
        sil     = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                     resolution=(IMG_RES, IMG_RES))
        sil_np  = sil[0, :, :, 0].detach().cpu().numpy()   # [H, W] float32
        sil_u8  = (sil_np * 255.0).clip(0, 255).astype(np.uint8)
        sil_views.append(sil_u8)

    images = np.stack(sil_views, axis=0)   # [4, H, W] uint8
    return images, token_ids, g


# ═════════════════════════════════════════════════════════════════════════════
# Shard I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_shard(
    images_list:    List[np.ndarray],
    token_ids_list: List[List[int]],
    genera_list:    List[int],
    path:           str,
) -> None:
    """Pad token sequences and write one .npz shard."""
    N       = len(images_list)
    max_len = max(len(ids) for ids in token_ids_list)

    images  = np.stack(images_list, axis=0)                  # [N, 4, H, W]
    tokens  = np.full((N, max_len), PAD_ID, dtype=np.int32)
    lengths = np.zeros(N, dtype=np.int32)
    genera  = np.array(genera_list, dtype=np.int32)

    for i, ids in enumerate(token_ids_list):
        L             = len(ids)
        tokens[i, :L] = ids
        lengths[i]    = L

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path,
                        images=images, tokens=tokens,
                        lengths=lengths, genera=genera)
    print(f"  saved {N} samples → {path}  (seq_len≤{max_len})")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opseq dataset")
    parser.add_argument('--n_train',    type=int,   default=8000)
    parser.add_argument('--n_val',      type=int,   default=500)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--shard_size', type=int,   default=1000,
                        help="Max samples per train shard")
    parser.add_argument('--out_dir',    type=str,
                        default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--device',     type=str,   default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctx = dr.RasterizeCudaContext()

    n_total = args.n_train + args.n_val
    print(f"Generating {args.n_train} train + {args.n_val} val samples "
          f"(seed={args.seed})…")

    splits = {
        'train': list(range(args.n_train)),
        'val':   list(range(args.n_train, n_total)),
    }

    genus_choices = [0, 1, 2]
    k_choices     = [1, 2]

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
            g   = int(rng.choice(genus_choices))
            k   = int(rng.choice(k_choices))

            try:
                images, token_ids, genus = generate_sample(
                    g, k, rng, ctx, device=args.device,
                )
            except Exception as exc:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  [FAIL] {split_name}[{global_idx}] g={g} k={k}: {exc}")
                continue

            images_buf.append(images)
            tokens_buf.append(token_ids)
            genera_buf.append(genus)
            n_ok += 1

            # Flush shard
            if len(images_buf) >= shard_size:
                path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
                save_shard(images_buf, tokens_buf, genera_buf, path)
                images_buf, tokens_buf, genera_buf = [], [], []
                shard_idx += 1

            if (sample_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = (sample_idx + 1) / elapsed
                eta     = (len(indices) - sample_idx - 1) / max(rate, 1e-6)
                print(f"  {split_name}: {sample_idx+1}/{len(indices)} "
                      f"| {rate:.1f} smp/s | ETA {eta:.0f}s | fails={n_fail}")

        if images_buf:
            path = os.path.join(split_dir, f'shard_{shard_idx:04d}.npz')
            save_shard(images_buf, tokens_buf, genera_buf, path)

        elapsed = time.time() - t0
        print(f"{split_name}: {n_ok} ok, {n_fail} failed, {elapsed:.1f}s")

    print("Dataset generation complete.")


if __name__ == '__main__':
    main()
