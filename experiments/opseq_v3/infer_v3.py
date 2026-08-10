#!/usr/bin/env python3
"""
infer_v3.py — Inference pipeline for OpSeqModelV3 (Phase A'').

Key difference from infer_v2.py: no DiffSequence, no cage optimization.
Optimization is over ALL final mesh vertices directly via nvdiffrast.

Pipeline
--------
1. Model generates topology tokens (greedy or nucleus).
2. Parse with fault tolerance → {base, hdl_pairs, ops}.
3. Build DLFL mesh by executing the topology program (float, not differentiable).
4. Fan-triangulate → (verts [V,3], tris [T,3]).
5. Directly optimize ALL vertices via Adam + silhouette loss (500 steps).
6. Return optimized verts + tris.

Direct vertex optimization details:
  - Learnable: all V vertex positions
  - Loss: mean L1 silhouette loss over 4 views + Laplacian + edge-length regularizers
  - Optimizer: Adam, lr=0.01
  - Schedule: cosine annealing to lr*0.01

Usage:
    python infer_v3.py --ckpt experiments/opseq_v3/ckpt/best.pt
                       --images path/to/shard.npz
                       [--sample_idx 0]
                       [--n_refine_steps 500]
                       [--out_dir experiments/opseq_v3/infer_out]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr

from topmod.primitives     import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.diffgeo        import mesh_to_arrays, _fan_triangulate, LINEAR_OPS
from topmod.tokenizer      import build_vocabulary_v3, decode_v3
from topmod.subdivision    import catmull_clark
from topmod.remeshing import (
    dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide, honeycomb_subdivide, corner_cutting,
    loop_style_subdivide, pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide, checkerboard_remesh, ds_bc_new_subdivide,
)
from topmod.high_level_ops import stellate_all

from model_v3 import OpSeqModelV3, BOS_ID, PAD_ID, EOS_ID, MAX_SEQ_LEN
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette, laplacian_loss, edge_length_loss

# ── Constants ─────────────────────────────────────────────────────────────────

N_REF    = 64
VOCAB_V3 = build_vocabulary_v3(n_ref=N_REF)
VOCAB_INV_V3 = {v: k for k, v in VOCAB_V3.items()}

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0

_PRIM_FNS = {
    'cube':        make_cube,
    'tetrahedron': make_tetrahedron,
    'icosahedron': make_icosahedron,
}
_DEFAULT_BASE = 'icosahedron'
_ALL_LINEAR_OPS = set(LINEAR_OPS)

# Float operator dispatch
def _sta(mesh): stellate_all(mesh); return mesh

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
# Sequence parsing (fault-tolerant)
# ═════════════════════════════════════════════════════════════════════════════

def parse_v3_sequence(token_ids: List[int]) -> Dict:
    """
    Parse a V3 topology token sequence.

    Fault-tolerant: unknown tokens and malformed HDL params are skipped.
    Returns: {'base': str, 'hdl_pairs': [...], 'ops': [...]}
    """
    parsed = decode_v3(token_ids, VOCAB_INV_V3)

    # Fallback base if model predicted an unknown one
    base = parsed['base'] if parsed['base'] in _PRIM_FNS else _DEFAULT_BASE

    # Filter to linear ops only (nonlinear ops need extra params we don't have)
    ops = [op for op in parsed['ops'] if op in _ALL_LINEAR_OPS]

    return {
        'base':      base,
        'hdl_pairs': parsed['hdl_pairs'],
        'ops':       ops,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Mesh execution (float, non-differentiable)
# ═════════════════════════════════════════════════════════════════════════════

def execute_topology(parsed: Dict) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Build DLFL mesh from parsed topology program, fan-triangulate.

    Returns
    -------
    verts_np : [V, 3] float64 — default vertex positions
    tris     : list of (i0, i1, i2) face index tuples
    """
    base_name = parsed['base']
    hdl_pairs = parsed['hdl_pairs']
    ops       = parsed['ops']

    mesh = _PRIM_FNS[base_name]()

    # Apply HDL ops (fault-tolerant: skip invalid ordinals)
    for f1_ord, f2_ord in hdl_pairs:
        faces_list = list(mesh.faces.values())
        if f1_ord < len(faces_list) and f2_ord < len(faces_list) and f1_ord != f2_ord:
            try:
                add_handle(mesh, faces_list[f1_ord], faces_list[f2_ord])
            except Exception:
                pass   # skip incompatible handle

    # Apply subdivision ops (fault-tolerant: stop at first failure)
    for op in ops:
        fn = _FLOAT_OPS.get(op)
        if fn is not None:
            try:
                mesh = fn(mesh)
            except Exception:
                break

    positions, faces = mesh_to_arrays(mesh)
    verts_np = np.array(positions, dtype=np.float64)
    tris     = _fan_triangulate(faces)

    return verts_np, tris


# ═════════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_images(
    images: np.ndarray,   # [4, H, W] uint8, white-bg (255=bg, 0=fg)
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      model_input : [1, 4, H, W] float32 in [0,1]  (white-bg, 0=fg, 1=bg)
      targets     : [4, H, W, 1] float32 in 1=fg space  (1=fg, 0=bg)
    """
    imgs_f      = images.astype(np.float32) / 255.0          # [4,H,W] 0=fg,1=bg
    model_input = torch.tensor(imgs_f, device=device).unsqueeze(0)   # [1,4,H,W]
    # Convert white-bg (0=fg) → 1=fg space
    targets     = torch.tensor(1.0 - imgs_f, device=device).unsqueeze(-1)  # [4,H,W,1]
    return model_input, targets


# ═════════════════════════════════════════════════════════════════════════════
# Direct vertex optimization
# ═════════════════════════════════════════════════════════════════════════════

def optimize_vertices_direct(
    ctx:         dr.RasterizeCudaContext,
    verts_init:  np.ndarray,   # [V, 3] float64
    tris:        List[Tuple[int, int, int]],
    targets:     torch.Tensor,  # [4, H, W, 1] float32, 1=fg
    mvps:        torch.Tensor,  # [4, 4, 4]
    num_steps:   int   = 500,
    lr:          float = 0.01,
    lambda_lap:  float = 0.05,
    lambda_edge: float = 0.01,
    resolution:  Tuple[int, int] = (128, 128),
    log_every:   int  = 100,
) -> torch.Tensor:
    """
    Optimize ALL mesh vertices directly to match target silhouettes.

    This is the Phase A'' approach: topology is fixed (from LLM), geometry
    is learned by gradient descent through differentiable rasterization.

    Unlike optimize_through_chain (V2), there is NO differentiable operator
    chain — we optimize the final vertex positions directly.

    Parameters
    ----------
    verts_init  : [V, 3] initial positions (from float topology execution).
    tris        : list of (i0, i1, i2) triangle index tuples.
    targets     : [4, H, W, 1] silhouette targets in 1=fg space.
    mvps        : [4, 4, 4] MVP matrices.
    num_steps   : Adam optimization steps.
    lr          : Adam learning rate.
    lambda_lap  : Laplacian smoothness regularizer weight.
    lambda_edge : Edge-length regularizer weight.
    log_every   : Print progress every N steps (0=silent).

    Returns
    -------
    verts_opt : [V, 3] float32 optimized vertex positions (detached).
    """
    device = targets.device

    # Normalize initial positions to [-1.6, +1.6] (same as training distribution)
    verts_np  = verts_init.astype(np.float32)
    mn, mx    = float(verts_np.min()), float(verts_np.max())
    extent    = max(mx - mn, 1e-6)
    scale     = 0.8 * 4.0 / extent   # 80% of [-2, +2]
    centre    = (mn + mx) / 2.0
    verts_np  = (verts_np - centre) * scale

    # Learnable vertex tensor
    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    verts.requires_grad_(True)

    # Fixed triangle indices (int32 for nvdiffrast)
    tris_t = torch.tensor(tris, dtype=torch.int32, device=device)   # [T, 3]

    if tris_t.shape[0] == 0:
        return verts.detach()

    optimizer = torch.optim.Adam([verts], lr=lr)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01,
    )

    N = mvps.shape[0]   # number of views (4)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Silhouette loss across all views
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            rendered = render_silhouette(ctx, verts, tris_t, mvps[i], resolution)
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i+1])
        sil_loss = sil_loss / N

        # Regularizers
        reg_loss = torch.tensor(0.0, device=device)
        if lambda_lap > 0:
            reg_loss = reg_loss + lambda_lap  * laplacian_loss(verts, tris_t)
        if lambda_edge > 0:
            reg_loss = reg_loss + lambda_edge * edge_length_loss(verts, tris_t)

        total = sil_loss + reg_loss
        total.backward()
        optimizer.step()
        sched.step()

        if log_every > 0 and (step % log_every == 0 or step == num_steps - 1):
            print(
                f"  [vert_opt] step {step:4d}/{num_steps}"
                f"  sil={sil_loss.item():.4f}"
                f"  reg={reg_loss.item():.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.5f}"
            )

    return verts.detach()


# ═════════════════════════════════════════════════════════════════════════════
# Full inference pipeline
# ═════════════════════════════════════════════════════════════════════════════

def infer_single(
    model:          OpSeqModelV3,
    images:         np.ndarray,        # [4, 128, 128] uint8, white-bg
    ctx:            dr.RasterizeCudaContext,
    device:         str   = 'cuda',
    use_nucleus:    bool  = False,
    top_p:          float = 0.9,
    temperature:    float = 1.0,
    n_refine_steps: int   = 500,
    refine_lr:      float = 0.01,
    lambda_lap:     float = 0.05,
    lambda_edge:    float = 0.01,
    log_refine:     int   = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Full inference for one sample.

    Returns
    -------
    verts_opt : [V, 3] float32 — optimized vertex positions
    tris_t    : [T, 3] int64  — face indices
    info      : {'base', 'ops', 'n_hdl', 'n_verts', 'n_tris', 'token_ids'}
    """
    # ── 1. Preprocess images ──────────────────────────────────────────
    model_input, targets = preprocess_images(images, device)

    # ── 2. Generate topology tokens ───────────────────────────────────
    model.eval()
    if use_nucleus:
        token_ids = model.sample_nucleus(
            model_input, max_new_tokens=MAX_SEQ_LEN,
            top_p=top_p, temperature=temperature,
        )
    else:
        token_ids = model.sample_greedy(model_input, max_new_tokens=MAX_SEQ_LEN)

    # ── 3. Parse topology ─────────────────────────────────────────────
    parsed = parse_v3_sequence(token_ids)

    # ── 4. Execute topology → mesh ────────────────────────────────────
    verts_np, tris = execute_topology(parsed)

    if verts_np.shape[0] < 3 or len(tris) == 0:
        raise ValueError(
            f"Degenerate mesh from topology {parsed['base']} + {parsed['ops']}: "
            f"V={verts_np.shape[0]}, T={len(tris)}"
        )

    # ── 5. Compute orbit cameras ──────────────────────────────────────
    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )

    # ── 6. Direct vertex optimization ─────────────────────────────────
    verts_opt = optimize_vertices_direct(
        ctx         = ctx,
        verts_init  = verts_np,
        tris        = tris,
        targets     = targets,
        mvps        = mvps,
        num_steps   = n_refine_steps,
        lr          = refine_lr,
        lambda_lap  = lambda_lap,
        lambda_edge = lambda_edge,
        resolution  = (IMG_RES, IMG_RES),
        log_every   = log_refine,
    )

    # Return int64 tris for downstream use
    tris_t = torch.tensor(tris, dtype=torch.long, device=device)   # [T, 3]

    info = {
        'base':      parsed['base'],
        'ops':       parsed['ops'],
        'n_hdl':     len(parsed['hdl_pairs']),
        'n_verts':   verts_np.shape[0],
        'n_tris':    len(tris),
        'token_ids': token_ids,
    }
    return verts_opt, tris_t, info


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="OpSeq V3 (topology-only) inference")
    parser.add_argument('--ckpt',           required=True)
    parser.add_argument('--images',         required=True,
                        help="Path to .npz shard")
    parser.add_argument('--sample_idx',     type=int,   default=0)
    parser.add_argument('--out_dir',        default=os.path.join(_SCRIPT_DIR, 'infer_out'))
    parser.add_argument('--n_refine_steps', type=int,   default=500)
    parser.add_argument('--use_nucleus',    action='store_true')
    parser.add_argument('--top_p',          type=float, default=0.9)
    parser.add_argument('--temperature',    type=float, default=1.0)
    parser.add_argument('--device',         default='cuda')
    args = parser.parse_args()

    device = args.device

    model = OpSeqModelV3().to(device)
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded {args.ckpt}  epoch={ckpt.get('epoch','?')}  "
          f"val={ckpt.get('val_loss','?'):.4f}")

    data   = np.load(args.images, allow_pickle=False)
    images = data['images'][args.sample_idx]   # [4,128,128] uint8
    print(f"Sample {args.sample_idx} from {args.images}")

    ctx = dr.RasterizeCudaContext()

    verts_opt, tris_t, info = infer_single(
        model           = model,
        images          = images,
        ctx             = ctx,
        device          = device,
        use_nucleus     = args.use_nucleus,
        top_p           = args.top_p,
        temperature     = args.temperature,
        n_refine_steps  = args.n_refine_steps,
    )

    print(f"\nInference complete:")
    print(f"  Base       : {info['base']}")
    print(f"  HDL ops    : {info['n_hdl']}")
    print(f"  Ops        : {info['ops']}")
    print(f"  Vertices   : {info['n_verts']} → optimized")
    print(f"  Triangles  : {info['n_tris']}")
    print(f"  Token IDs  : {info['token_ids']}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'mesh_sample{args.sample_idx}.npz')
    np.savez_compressed(
        out_path,
        verts    = verts_opt.cpu().numpy(),
        tris     = tris_t.cpu().numpy(),
        token_ids= np.array(info['token_ids'], dtype=np.int32),
    )
    print(f"  Saved → {out_path}")


if __name__ == '__main__':
    main()
