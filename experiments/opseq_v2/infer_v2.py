#!/usr/bin/env python3
"""
infer_v2.py — Inference pipeline for OpSeqModelV2 (Phase A').

Pipeline
--------
1. Load conditioning silhouette images (4 views, white-background).
2. Model generates a V2 token sequence (greedy or nucleus sampling).
3. Parse sequence with fault tolerance:
   - extract base name, HDL pairs, op names, and raw coord ints
   - apply HDL ops to DLFL base mesh
   - build DiffSequence from HDL-modified base
   - append op names to DiffSequence
   - truncate/pad cage coords to match actual cage size
4. Differentiable refinement: optimize cage via silhouette loss (500 steps).
5. Output final mesh: seq.forward(refined_cage), seq.triangles().

Usage:
    python infer_v2.py --ckpt experiments/opseq_v2/ckpt/best.pt
                       --images path/to/images.npz     # or directory with *.png
                       [--out_dir experiments/opseq_v2/infer_out]
                       [--n_refine_steps 500]
                       [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr

from topmod.primitives     import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.validate       import is_manifold
from topmod.diffgeo        import DiffSequence, mesh_to_arrays, LINEAR_OPS
from topmod.tokenizer      import (
    build_vocabulary_v2,
    decode_v2,
    dequantize_coord,
    DEFAULT_COORD_LO,
    DEFAULT_COORD_HI,
    _find_compatible_face_pair,
    _face_ordinal,
)
from model_v2 import OpSeqModelV2, BOS_ID, PAD_ID, EOS_ID, MAX_SEQ_LEN
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import optimize_through_chain

# ── Constants ─────────────────────────────────────────────────────────────────

N_COORD_BINS = 256
N_REF        = 64
VOCAB_V2     = build_vocabulary_v2(n_coord_bins=N_COORD_BINS, n_ref=N_REF)
VOCAB_INV_V2 = {v: k for k, v in VOCAB_V2.items()}

COORD_LO   = DEFAULT_COORD_LO   # -2.0
COORD_HI   = DEFAULT_COORD_HI   # +2.0
AZIMUTHS   = [0.0, 90.0, 180.0, 270.0]
IMG_RES    = 128
CAMERA_RADIUS = 3.0

# Fallback base when parsing fails
_DEFAULT_BASE = 'icosahedron'

_PRIM_FNS = {
    'cube':         make_cube,
    'tetrahedron':  make_tetrahedron,
    'icosahedron':  make_icosahedron,
}

# All supported ops (for validation during parsing)
_ALL_OPS = set(LINEAR_OPS)


# ═════════════════════════════════════════════════════════════════════════════
# Sequence parsing (fault-tolerant)
# ═════════════════════════════════════════════════════════════════════════════

def parse_v2_sequence(token_ids: List[int]) -> Dict:
    """
    Parse a V2 token sequence into its structural components.

    Fault-tolerant: unknown tokens and malformed subsequences are skipped.

    Returns
    -------
    dict with:
      'base'       : str       — base primitive name (default: 'icosahedron')
      'hdl_pairs'  : List[(int,int)]
      'ops'        : List[str] — op names (only linear ops accepted for safety)
      'coord_ints' : List[int] — raw COORD bin values (flattened x,y,z,x,y,z,…)
    """
    parsed = decode_v2(token_ids, VOCAB_INV_V2)

    # Fallback base
    base = parsed['base'] if parsed['base'] in _PRIM_FNS else _DEFAULT_BASE

    # Filter ops to linear-only for safety (nonlinear ops need extra params)
    ops = [op for op in parsed['ops'] if op in _ALL_OPS]

    return {
        'base':       base,
        'hdl_pairs':  parsed['hdl_pairs'],
        'ops':        ops,
        'coord_ints': parsed['coord_ints'],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Build DiffSequence from parsed components
# ═════════════════════════════════════════════════════════════════════════════

def build_seq_from_parsed(
    parsed: Dict,
    device: str = 'cuda',
) -> Tuple[DiffSequence, bool]:
    """
    Build a DiffSequence from a parsed V2 sequence dict.

    Steps
    -----
    1. Build DLFL base mesh.
    2. Apply HDL ops (fault-tolerant: silently stops if no valid face pair).
    3. Extract (positions, faces) → DiffSequence base.
    4. Append linear ops (fault-tolerant: stops at first failure).

    Returns
    -------
    seq      : DiffSequence ready for forward() and triangles()
    valid    : True if at least one op was applied or base is non-trivial
    """
    base_name = parsed['base']
    hdl_pairs = parsed['hdl_pairs']
    ops       = parsed['ops']

    # ── 1. DLFL base mesh ─────────────────────────────────────────────
    mesh = _PRIM_FNS[base_name]()

    # ── 2. Apply HDL ops ──────────────────────────────────────────────
    excluded_vids = set()
    faces_list    = list(mesh.faces.values())
    n_faces       = len(faces_list)

    for f1_ord, f2_ord in hdl_pairs:
        if f1_ord >= n_faces or f2_ord >= n_faces or f1_ord == f2_ord:
            continue
        f1 = faces_list[f1_ord]
        f2 = faces_list[f2_ord]
        vids1 = {v.id for v in f1.vertices()}
        vids2 = {v.id for v in f2.vertices()}
        if vids1 & excluded_vids or vids2 & excluded_vids or vids1 & vids2:
            continue
        try:
            add_handle(mesh, f1, f2)
            excluded_vids |= vids1 | vids2
            # Refresh face list after topology change
            faces_list = list(mesh.faces.values())
            n_faces    = len(faces_list)
        except Exception:
            pass

    # ── 3. Extract (positions, faces) → DiffSequence ──────────────────
    positions, faces = mesh_to_arrays(mesh)
    seq = DiffSequence(
        (positions, faces),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    # ── 4. Append linear ops ──────────────────────────────────────────
    for op in ops:
        try:
            seq.append(op)
        except Exception:
            break   # stop at first incompatible op

    valid = True   # always valid by construction
    return seq, valid


# ═════════════════════════════════════════════════════════════════════════════
# Coordinate fault-tolerant alignment
# ═════════════════════════════════════════════════════════════════════════════

def align_cage_coords(
    seq:         DiffSequence,
    coord_ints:  List[int],
    device:      str = 'cuda',
) -> torch.Tensor:
    """
    Convert raw COORD bin ints to a cage vertex tensor [V_cage, 3].

    Fault-tolerant:
    - If too many coords: truncate to V_cage * 3.
    - If too few coords:  pad with bin 128 (≈ 0.0, centre of range).

    Returns
    -------
    cage_verts : [V_cage, 3] float32 tensor on *device*.
    """
    V_cage    = seq.verts0.shape[0]
    needed    = V_cage * 3
    center_bin = N_COORD_BINS // 2   # 128 — maps to ≈ 0.0

    # Truncate or pad
    if len(coord_ints) > needed:
        coord_ints = coord_ints[:needed]
    elif len(coord_ints) < needed:
        coord_ints = coord_ints + [center_bin] * (needed - len(coord_ints))

    # Dequantize
    coords_f = np.array(
        [dequantize_coord(q, COORD_LO, COORD_HI, N_COORD_BINS) for q in coord_ints],
        dtype=np.float32,
    ).reshape(V_cage, 3)

    return torch.tensor(coords_f, dtype=torch.float32, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# Preprocess conditioning images
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_images(
    images: np.ndarray,   # [4, H, W] uint8, WHITE background (255=bg, 0=fg)
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare conditioning images for model input and silhouette optimization.

    Returns
    -------
    model_input : [1, 4, H, W] float32 in [0, 1] (for CNN encoder)
    targets     : [4, H, W, 1] float32 in 1=fg space (for silhouette loss)
    """
    # Float in [0,1], white-bg convention (0=fg, 1=bg)
    imgs_f = images.astype(np.float32) / 255.0   # [4, H, W]

    # Model input: [1, 4, H, W]
    model_input = torch.tensor(imgs_f, dtype=torch.float32, device=device).unsqueeze(0)

    # Targets: convert white-bg (0=fg, 1=bg) → 1=fg space
    # imgs_f is 0=fg, 1=bg; so 1-fg = target in 1=fg space
    targets_np = (1.0 - imgs_f)                           # [4, H, W]
    targets    = torch.tensor(targets_np, dtype=torch.float32, device=device)
    targets    = targets.unsqueeze(-1)                    # [4, H, W, 1]

    return model_input, targets


# ═════════════════════════════════════════════════════════════════════════════
# Full inference pipeline
# ═════════════════════════════════════════════════════════════════════════════

def infer_single(
    model:           OpSeqModelV2,
    images:          np.ndarray,        # [4, 128, 128] uint8, white-bg
    ctx:             dr.RasterizeCudaContext,
    device:          str   = 'cuda',
    use_nucleus:     bool  = False,
    top_p:           float = 0.9,
    temperature:     float = 1.0,
    n_refine_steps:  int   = 500,
    refine_lr:       float = 1e-2,
    lambda_lap:      float = 0.05,
    lambda_edge:     float = 0.01,
    log_refine:      int   = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Full inference for a single sample.

    Parameters
    ----------
    model       : Trained OpSeqModelV2.
    images      : [4, 128, 128] uint8, white-background silhouettes.
    ctx         : nvdiffrast CUDA context.
    device      : Torch device string.
    use_nucleus : Use nucleus sampling instead of greedy.
    n_refine_steps : Steps for optimize_through_chain (0 = skip refinement).

    Returns
    -------
    final_verts : [V, 3] float32 — final mesh vertices after refinement.
    tris        : [T, 3] int64  — fan-triangulated face indices.
    info        : dict with keys 'base', 'ops', 'n_hdl', 'n_cage', 'token_ids'.
    """
    # ── 1. Preprocess images ──────────────────────────────────────────
    model_input, targets = preprocess_images(images, device)

    # ── 2. Generate token sequence ────────────────────────────────────
    model.eval()
    if use_nucleus:
        token_ids = model.sample_nucleus(
            model_input, max_new_tokens=MAX_SEQ_LEN,
            top_p=top_p, temperature=temperature,
        )
    else:
        token_ids = model.sample_greedy(model_input, max_new_tokens=MAX_SEQ_LEN)

    # ── 3. Parse sequence ─────────────────────────────────────────────
    parsed = parse_v2_sequence(token_ids)

    # ── 4. Build DiffSequence ─────────────────────────────────────────
    seq, _ = build_seq_from_parsed(parsed, device=device)
    n_cage  = seq.verts0.shape[0]

    # ── 5. Fault-tolerant cage coordinate alignment ───────────────────
    cage_verts = align_cage_coords(seq, parsed['coord_ints'], device=device)
    seq.verts0 = cage_verts

    # ── 6. Differentiable refinement ──────────────────────────────────
    if n_refine_steps > 0:
        mvps, _ = orbit_cameras(
            4, elevation_deg=0.0, radius=CAMERA_RADIUS,
            azimuths_deg=AZIMUTHS, device=device,
        )
        refined_cage = optimize_through_chain(
            ctx, seq, targets, mvps,
            num_steps   = n_refine_steps,
            lr          = refine_lr,
            lambda_lap  = lambda_lap,
            lambda_edge = lambda_edge,
            resolution  = (IMG_RES, IMG_RES),
            log_every   = log_refine,
        )
        seq.verts0 = refined_cage

    # ── 7. Final mesh ─────────────────────────────────────────────────
    with torch.no_grad():
        final_verts = seq.forward()   # [V_final, 3]
    tris = seq.triangles(device=device)  # [T, 3] int64

    info = {
        'base':      parsed['base'],
        'ops':       parsed['ops'],
        'n_hdl':     len(parsed['hdl_pairs']),
        'n_cage':    n_cage,
        'token_ids': token_ids,
    }
    return final_verts.detach(), tris, info


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="OpSeq V2 inference")
    parser.add_argument('--ckpt',           required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument('--images',         required=True,
                        help="Path to .npz shard or directory of 4 PNG files")
    parser.add_argument('--out_dir',        default=os.path.join(_SCRIPT_DIR, 'infer_out'))
    parser.add_argument('--n_refine_steps', type=int,   default=500)
    parser.add_argument('--use_nucleus',    action='store_true')
    parser.add_argument('--top_p',          type=float, default=0.9)
    parser.add_argument('--temperature',    type=float, default=1.0)
    parser.add_argument('--device',         default='cuda')
    parser.add_argument('--sample_idx',     type=int,   default=0,
                        help="Sample index when --images is a .npz shard")
    args = parser.parse_args()

    device = args.device

    # ── Load model ────────────────────────────────────────────────────
    model = OpSeqModelV2().to(device)
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}  "
          f"(epoch={ckpt.get('epoch', '?')}, val={ckpt.get('val_loss', '?'):.4f})")

    # ── Load images ───────────────────────────────────────────────────
    if args.images.endswith('.npz'):
        data   = np.load(args.images, allow_pickle=False)
        images = data['images'][args.sample_idx]   # [4, 128, 128] uint8
        print(f"Loaded sample {args.sample_idx} from {args.images}")
    else:
        raise NotImplementedError(
            "Directory-based image loading not implemented; use a .npz shard"
        )

    # ── Inference ─────────────────────────────────────────────────────
    ctx = dr.RasterizeCudaContext()

    final_verts, tris, info = infer_single(
        model        = model,
        images       = images,
        ctx          = ctx,
        device       = device,
        use_nucleus  = args.use_nucleus,
        top_p        = args.top_p,
        temperature  = args.temperature,
        n_refine_steps = args.n_refine_steps,
    )

    print(f"\nInference complete:")
    print(f"  Base      : {info['base']}")
    print(f"  HDL ops   : {info['n_hdl']}")
    print(f"  Ops       : {info['ops']}")
    print(f"  Cage verts: {info['n_cage']}")
    print(f"  Final mesh: V={final_verts.shape[0]}, T={tris.shape[0]}")

    # ── Save output ───────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'mesh_sample{args.sample_idx}.npz')
    np.savez_compressed(
        out_path,
        verts=final_verts.cpu().numpy(),
        tris=tris.cpu().numpy(),
        token_ids=np.array(info['token_ids'], dtype=np.int32),
    )
    print(f"  Saved → {out_path}")


if __name__ == '__main__':
    main()
