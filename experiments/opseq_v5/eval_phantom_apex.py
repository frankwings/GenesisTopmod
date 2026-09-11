#!/usr/bin/env python3
"""
eval_phantom_apex.py — Phantom Apex Gradient where-to-refine.

Core mechanism (3-step)
-----------------------
1. Probe  — For every face f simultaneously insert an ε-lifted phantom apex
   a_f = c_f + ε_f · n_f (centroid + tiny normal displacement).  A SINGLE
   forward+backward on the phantom mesh gives ∇_{a_f}L for every face at once.
2. Select — Area-normalised gradient score  s_f = |∇_{a_f}L · n_f| / A_f,
   followed by greedy 1-ring topological NMS → top-K faces.
3. Execute — True DLFL stellate on selected faces, restart optimisation.

Key properties
--------------
• Causal probe: measures "would adding a DoF here reduce loss?" not "is this
  region already fitting badly?" (the distinction from cheap heuristics).
• Single backward: O(1) probe cost independent of F (renderer is the bottleneck
  regardless of face count).
• Gradient isolation: each apex is a distinct leaf tensor; sub-triangles of
  face f share only their apex → no cross-face gradient contamination.
• Float32 assert: after building the phantom mesh, check IoU(phantom)−IoU(orig)
  is above PHM_IOU_DIFF_MIN (1e-6).  If phantom is numerically identical to
  original the gradient is degenerate → double ε and retry.

Definitive design decisions (post-debate, do not re-design)
------------------------------------------------------------
ε-lift    : ε_f = max(5e-4 · diag_f,  1e-4),  diag_f = face bbox diagonal
Scoring   : s_f = |∇_{a_f}L · n_f| / (A_f + 1e-8),  absolute (concave=convex)
Top-K     : K=20 per round (small-step, frequent probing)
NMS       : 1-ring edge adjacency, greedy descending-score suppression
Trigger   : every PHM_INTERVAL steps (fixed schedule, same for all methods →
            fair comparison)
Plateau   : EMA(α=0.95) + vertex-displacement double-AND as secondary guard

H1 hypothesis (must pass before full matrix):
   phantom significantly > random at 800F on cow (p < 0.05, noise_floor=0.02)

Usage
-----
  python3 eval_phantom_apex.py --shape cow --budget 800 --trials 4 \
      --variants phantom random --total-steps 800 \
      --json eval_out/phantom_h1.json

Long run (bg_task):
  python3 -m framework.bg_task launch --id phantom_h1_cow --timeout 10800 \
      --cwd /home/kingy/Foundation/ZenithLoom -- \\
      python3 /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/\\
      eval_phantom_apex.py --shape cow --budget 800 --trials 4 \\
      --variants phantom random --total-steps 800 \\
      --json /home/kingy/Projects/Genesis/GenesisTopmod/eval_out/phantom_h1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from pipeline.geometry_optimizer import laplacian_loss, edge_length_loss
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

from eval_extrude_v3 import (
    make_6_cameras, render_sil_and_depth, render_views_n, render_depths_n,
    compute_iou_n, depth_loss_masked,
    dlfl_boundary_stats, _mesh_genus,
    topmod_stellate_cluster,
    N_VIEWS, IMG_RES, CAMERA_RADIUS, W_LAP, W_LAP_BOOST, W_EDGE, W_DEPTH,
    WARMUP_STEPS, LAP_WARMUP_STEPS, DEPTH_WARMUP_STEPS,
    MIN_STEP_FOR_INJ,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_STEPS   = 800
LR            = 3e-3
LR_MIN        = 3e-5

# ── Phantom-apex probe ────────────────────────────────────────────────────────
PHM_EPS_BASE     = 5e-4   # ε = max(PHM_EPS_BASE * diag_f,  PHM_EPS_FLOOR)
PHM_EPS_FLOOR    = 1e-4   # absolute lower bound so ε > float32 quant noise
PHM_IOU_DIFF_MIN = 1e-6   # if IoU(phantom) − IoU(orig) < this → ε too small
PHM_MAX_EPS_RETRY = 5     # max doublings of ε in the IoU assert

# ── Refinement schedule ───────────────────────────────────────────────────────
PHM_K_STELLATE  = 20      # faces stellated per probe round (small-step)
PHM_INTERVAL    = 50      # main-loop steps between probe triggers
PHM_MAX_ROUNDS  = 32      # safety ceiling (caps at face_budget anyway)

# ── Plateau guard (secondary condition, NOT required for trigger) ─────────────
PHM_EMA_ALPHA   = 0.95    # EMA smoothing for loss plateau detection
PHM_DISP_THRESH = 1e-5    # mean vertex displacement plateau threshold (unit-bbox)

# ── NMS ──────────────────────────────────────────────────────────────────────
PHM_NMS_RING    = 1       # 1-ring: faces sharing an edge are mutual-excluded

# ── Comparison methods ────────────────────────────────────────────────────────
VALID_STRATEGIES = ('phantom', 'random', 'area', 'heuristic')


# ─────────────────────────────────────────────────────────────────────────────
# Face geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_face_geometry(
    verts_np: np.ndarray,  # [V, 3] float64
    tris_np:  np.ndarray,  # [F, 3] int32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (unit_normals [F,3], areas [F], centroids [F,3], bbox_diags [F])."""
    v0 = verts_np[tris_np[:, 0]]   # [F, 3]
    v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]

    e01   = v1 - v0
    e02   = v2 - v0
    cross = np.cross(e01, e02)                          # [F, 3]
    cross_norms = np.linalg.norm(cross, axis=1)         # [F]
    normals = cross / (cross_norms[:, None] + 1e-12)    # [F, 3] unit normals
    areas   = cross_norms * 0.5                         # [F]

    centroids = (v0 + v1 + v2) / 3.0                   # [F, 3]

    bbox_min = np.minimum(np.minimum(v0, v1), v2)       # [F, 3]
    bbox_max = np.maximum(np.maximum(v0, v1), v2)       # [F, 3]
    diags    = np.linalg.norm(bbox_max - bbox_min, axis=1)  # [F]

    return normals, areas, centroids, diags


# ─────────────────────────────────────────────────────────────────────────────
# Phantom mesh construction
# ─────────────────────────────────────────────────────────────────────────────

def build_phantom_mesh(
    verts_np:  np.ndarray,  # [V, 3] float64
    tris_np:   np.ndarray,  # [F, 3] int32
    eps_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert ε-lifted phantom apices into every face simultaneously.

    For each face f = (v0, v1, v2):
        apex_f = centroid_f + ε_f · normal_f
        Replace f with three sub-triangles: (v0,v1,apex), (v1,v2,apex), (v2,v0,apex)

    Gradient isolation: apex_f is touched only by f's three sub-triangles →
    no cross-face gradient paths.

    Returns
    -------
    phantom_verts : [V+F, 3] float32 — original verts then apex positions
    phantom_tris  : [3*F, 3] int32  — sub-triangles
    apex_pos      : [F, 3]   float64 — apex positions (for tensor construction)
    """
    V = len(verts_np)
    F = len(tris_np)

    normals, _, centroids, diags = compute_face_geometry(verts_np, tris_np)

    eps_vec  = np.maximum(PHM_EPS_BASE * diags, PHM_EPS_FLOOR) * eps_scale  # [F]
    apex_pos = centroids + eps_vec[:, None] * normals                         # [F, 3]

    phantom_verts = np.concatenate(
        [verts_np.astype(np.float32), apex_pos.astype(np.float32)], axis=0)  # [V+F, 3]

    v0 = tris_np[:, 0]; v1 = tris_np[:, 1]; v2 = tris_np[:, 2]
    af = np.arange(V, V + F, dtype=np.int32)  # apex vertex indices

    # Sub-triangle winding preserves outward normals (apex above face centroid)
    t0 = np.stack([v0, v1, af], axis=1)   # [F, 3]
    t1 = np.stack([v1, v2, af], axis=1)
    t2 = np.stack([v2, v0, af], axis=1)
    phantom_tris = np.concatenate([t0, t1, t2], axis=0).astype(np.int32)  # [3F, 3]

    return phantom_verts, phantom_tris, apex_pos


# ─────────────────────────────────────────────────────────────────────────────
# Core probe: phantom apex gradient
# ─────────────────────────────────────────────────────────────────────────────

def phantom_apex_probe(
    ctx,
    verts_np:  np.ndarray,  # [V, 3] float64 — current optimised vertices
    tris_np:   np.ndarray,  # [F, 3] int32
    gt_uint8:  np.ndarray,  # [N_V, H, W] uint8   (0=fg)
    gt_depths: np.ndarray,  # [N_V, H, W] float32 NDC-z
    mvps:      torch.Tensor,
    device:    str,
    eps_scale: float = 1.0,
    _retry_depth: int = 0,
) -> np.ndarray:
    """Compute per-face area-normalised phantom apex gradient scores [F].

    Algorithm
    ---------
    1. Build phantom mesh (ε-lifted apices as leaf tensor).
    2. Assert phantom ≠ original (IoU diff ≥ PHM_IOU_DIFF_MIN); double ε if not.
    3. Single forward+backward over all N views using reconstruction loss only
       (no regularisation — regularisation is optimiser-specific, not a DoF signal).
    4. score_f = |grad_f · n_f| / (A_f + 1e-8).

    High score → adding a stellate apex here reduces the reconstruction loss.
    """
    V  = len(verts_np)
    F  = len(tris_np)
    N_v = mvps.shape[0]

    normals, areas, _, _ = compute_face_geometry(verts_np, tris_np)

    # ── Build phantom mesh ────────────────────────────────────────────────────
    ph_verts_np, ph_tris_np, apex_pos_np = build_phantom_mesh(
        verts_np, tris_np, eps_scale=eps_scale)

    # ── Float32 / ε assert ───────────────────────────────────────────────────
    # The phantom mesh must render DIFFERENTLY from the original (ε is effective).
    # If IoU diff < PHM_IOU_DIFF_MIN the apices are within float32 quant noise →
    # gradient is degenerate.  Double ε and retry.
    orig_verts_t = torch.tensor(verts_np.astype(np.float32), device=device)
    orig_tris_t  = torch.tensor(tris_np, dtype=torch.int32, device=device)
    ph_verts_t   = torch.tensor(ph_verts_np, device=device)
    ph_tris_t    = torch.tensor(ph_tris_np, dtype=torch.int32, device=device)

    with torch.no_grad():
        orig_sils = render_views_n(ctx, orig_verts_t, orig_tris_t, mvps)
        ph_sils   = render_views_n(ctx, ph_verts_t,   ph_tris_t,   mvps)

    iou_orig    = compute_iou_n(orig_sils, gt_uint8)
    iou_phantom = compute_iou_n(ph_sils,   gt_uint8)
    iou_diff    = abs(iou_phantom - iou_orig)

    del orig_verts_t, orig_tris_t, ph_verts_t, ph_tris_t
    if device != 'cpu':
        torch.cuda.empty_cache()

    if iou_diff < PHM_IOU_DIFF_MIN:
        if _retry_depth < PHM_MAX_EPS_RETRY:
            new_scale = eps_scale * 2.0
            print(f"  [PHM-assert] IoU_diff={iou_diff:.2e} < {PHM_IOU_DIFF_MIN:.0e} "
                  f"→ ε too small, retry with ×{new_scale:.1f}")
            return phantom_apex_probe(ctx, verts_np, tris_np, gt_uint8, gt_depths,
                                      mvps, device,
                                      eps_scale=new_scale,
                                      _retry_depth=_retry_depth + 1)
        else:
            print(f"  [PHM-assert WARN] max retries ({PHM_MAX_EPS_RETRY}) reached; "
                  f"proceeding with eps_scale={eps_scale}")

    # ── Gradient computation ──────────────────────────────────────────────────
    # Original vertices: detached (gradient must flow ONLY through apex_part)
    orig_part  = torch.tensor(verts_np.astype(np.float32), device=device)
    apex_part  = torch.tensor(apex_pos_np.astype(np.float32), device=device,
                              requires_grad=True)           # [F, 3] leaf node
    all_verts  = torch.cat([orig_part, apex_part], dim=0)  # [V+F, 3]
    ph_tris_t  = torch.tensor(ph_tris_np, dtype=torch.int32, device=device)

    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device) for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device) for i in range(N_v)]

    # Single forward pass (reconstruction loss only — no regularisation)
    loss_ph = torch.tensor(0.0, device=device)
    for i in range(N_v):
        sil, ndc_z, fg = render_sil_and_depth(
            ctx, all_verts, ph_tris_t, mvps[i], (IMG_RES, IMG_RES))
        loss_ph += F.l1_loss(sil[0], targets[i])
        loss_ph += W_DEPTH * depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])

    loss_ph.backward()

    if apex_part.grad is None:
        print("  [PHM WARN] apex_part.grad is None — zero scores returned")
        apex_grad = np.zeros((F, 3), dtype=np.float64)
    else:
        apex_grad = apex_part.grad.detach().cpu().numpy().astype(np.float64)  # [F, 3]

    # Cleanup GPU memory
    del orig_part, apex_part, all_verts, ph_tris_t, targets, gt_depth_t, gt_fg_t, loss_ph
    if device != 'cpu':
        torch.cuda.empty_cache()

    # ── Score: |∇_{a_f}L · n_f| / A_f ───────────────────────────────────────
    dot    = np.sum(apex_grad * normals, axis=1)        # [F] signed dot product
    scores = np.abs(dot) / (areas + 1e-8)               # [F] area-normalised

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 1-ring topological NMS
# ─────────────────────────────────────────────────────────────────────────────

def topological_nms_1ring(
    tris_np: np.ndarray,  # [F, 3] int32
    scores:  np.ndarray,  # [F] float — higher = better
    top_k:   int,
) -> np.ndarray:
    """Greedy 1-ring NMS: select up to top_k non-edge-adjacent faces by score.

    Returns [≤top_k] int32 face indices in score-descending order.
    """
    F = len(tris_np)

    # Build face-adjacency via shared directed edges
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(F):
        for k in range(3):
            a = int(tris_np[fi, k])
            b = int(tris_np[fi, (k + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_to_faces.setdefault(e, []).append(fi)

    neighbors: List[List[int]] = [[] for _ in range(F)]
    for faces_list in edge_to_faces.values():
        if len(faces_list) == 2:
            f0, f1 = faces_list
            neighbors[f0].append(f1)
            neighbors[f1].append(f0)

    order      = np.argsort(-scores)   # descending
    selected   = []
    suppressed = set()
    for fi_np in order:
        fi = int(fi_np)
        if fi in suppressed:
            continue
        selected.append(fi)
        if len(selected) >= top_k:
            break
        for nb in neighbors[fi]:
            suppressed.add(nb)

    return np.array(selected, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Main refinement loop (shared by phantom, random, area strategies)
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy_refine(
    ctx,
    verts_init:    np.ndarray,
    tris_init:     np.ndarray,
    gt_uint8:      np.ndarray,
    gt_depths:     np.ndarray,
    mvps:          torch.Tensor,
    device:        str,
    strategy:      str   = 'phantom',    # 'phantom'|'random'|'area'
    total_steps:   int   = TOTAL_STEPS,
    face_budget:   int   = 800,
    k_stellate:    int   = PHM_K_STELLATE,
    phm_interval:  int   = PHM_INTERVAL,
    max_rounds:    int   = PHM_MAX_ROUNDS,
) -> Tuple[float, List[dict]]:
    """Unified refinement loop for phantom / random / area strategies.

    All three strategies share:
      • The same main optimisation loop (identical loss, LR schedule, warmup).
      • The same fixed-schedule trigger (every phm_interval steps).
      • The same 1-ring NMS applied after face scoring.
      • The same face budget cap.

    The ONLY difference: how face scores [F] are assigned.
      phantom  → phantom_apex_probe (gradient-based causal probe)
      random   → uniform random scores (np.random.rand(F))
      area     → inverse area scores (smallest faces first = coarsest coverage)

    Returns (final_iou, refine_log).
    """
    assert strategy in VALID_STRATEGIES, f"Unknown strategy: {strategy!r}"
    N_v = mvps.shape[0]

    # GT tensors (fixed throughout)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device) for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device) for i in range(N_v)]
    mvps_np    = mvps.detach().cpu().numpy().astype(np.float64)

    # Init mesh (remesh from cc2 icosphere)
    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    init_F     = len(tris_np)
    init_genus = _mesh_genus(verts_np, tris_np)
    print(f"  [{strategy}] Init: V={len(verts_np)} F={init_F} genus={init_genus} "
          f"budget={face_budget}")

    # Validate manifold
    n_bnd0, _ = dlfl_boundary_stats(verts_np, tris_np)
    if n_bnd0 > 0:
        raise RuntimeError(f"Init mesh has {n_bnd0} boundary edges")

    def _make_tensors(v_np, t_np):
        vt = torch.tensor(v_np, dtype=torch.float32, device=device).requires_grad_(True)
        ft = torch.tensor(t_np, dtype=torch.int32,   device=device)
        return vt, ft

    def _rebuild(v_np, t_np, cur_step):
        vt, ft = _make_tensors(v_np, t_np)
        op     = torch.optim.Adam([vt], lr=LR)
        rem    = max(total_steps - cur_step - 1, 1)
        sc     = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=rem, eta_min=LR_MIN)
        return vt, ft, op, sc

    verts_t, faces_t = _make_tensors(verts_np, tris_np)
    targets          = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR_MIN)

    refine_log:     List[dict] = []
    n_rounds        = 0
    warmup_counter  = 0
    lap_boost_left  = 0
    depth_boost_left = 0
    loss_ema        = None
    prev_verts_np   = verts_np.copy()

    for step in range(total_steps):

        # ── Warmup LR ────────────────────────────────────────────────────────
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup_counter -= 1

        # ── Forward + backward ────────────────────────────────────────────────
        opt.zero_grad()
        l_sil  = torch.tensor(0., device=device)
        l_dep  = torch.tensor(0., device=device)
        w_lap  = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        w_dm   = ((DEPTH_WARMUP_STEPS - depth_boost_left) / DEPTH_WARMUP_STEPS
                  if depth_boost_left > 0 else 1.0)
        if lap_boost_left   > 0: lap_boost_left   -= 1
        if depth_boost_left > 0: depth_boost_left -= 1

        for i in range(N_v):
            sil, ndc_z, fg_mask = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            l_sil += F.l1_loss(sil[0], targets[i])
            l_dep += depth_loss_masked(ndc_z, fg_mask, gt_depth_t[i], gt_fg_t[i])
        l_sil /= N_v; l_dep /= N_v
        lap  = laplacian_loss(verts_t, faces_t)
        edge = edge_length_loss(verts_t, faces_t)
        loss = l_sil + W_DEPTH * w_dm * l_dep + w_lap * lap + W_EDGE * edge
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()

        # ── EMA loss + vertex displacement (plateau guard) ────────────────────
        lv = loss.item()
        loss_ema = lv if loss_ema is None else PHM_EMA_ALPHA * loss_ema + (1 - PHM_EMA_ALPHA) * lv
        cur_verts_np = verts_t.detach().cpu().numpy()
        mean_disp    = float(np.linalg.norm(cur_verts_np - prev_verts_np, axis=1).mean())
        prev_verts_np = cur_verts_np

        # ── Periodic eval ─────────────────────────────────────────────────────
        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(pred_sils, gt_uint8)
            n_cur = int(faces_t.shape[0])
            print(f"  [{strategy}] step={step:4d}  IoU={iou:.4f}  "
                  f"F={n_cur}  rounds={n_rounds}  ema={loss_ema:.4f}")

        # ── Refinement trigger ────────────────────────────────────────────────
        can_refine = (
            n_rounds < max_rounds
            and step >= MIN_STEP_FOR_INJ
            and step % phm_interval == 0
            and warmup_counter == 0
            and len(tris_np) < face_budget
        )
        if not can_refine:
            continue

        # ── Face scoring → NMS → stellate ────────────────────────────────────
        verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
        tris_cur  = faces_t.cpu().numpy()
        F_cur     = len(tris_cur)
        iou_before = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8) \
            if step % 100 != 0 else iou  # reuse if just computed

        budget_rem = face_budget - F_cur
        k_actual   = min(k_stellate, budget_rem // 2)  # each stellate +2F for triangle
        if k_actual <= 0:
            continue

        if strategy == 'phantom':
            scores = phantom_apex_probe(ctx, verts_cur, tris_cur,
                                         gt_uint8, gt_depths, mvps, device)
        elif strategy == 'random':
            scores = np.random.rand(F_cur).astype(np.float64)
        elif strategy == 'area':
            # Largest faces stellated last — gives coverage-uniform refinement
            _, areas, _, _ = compute_face_geometry(verts_cur, tris_cur)
            scores = -areas  # negate: NMS will pick highest score = smallest area
        elif strategy == 'heuristic':
            # Reuse existing eval_local_refine score_faces_by_projection
            from eval_local_refine import (score_faces_by_projection,
                                            compute_missing_error_maps,
                                            render_depths_n)
            from eval_extrude_v3 import compute_missing_error_maps as _cme
            with torch.no_grad():
                pred_sils_np  = render_views_n(ctx, verts_t, faces_t, mvps)
                pred_depths_np = render_depths_n(ctx, verts_t, faces_t, mvps)
            error_maps = _cme(pred_sils_np, gt_uint8)
            scores = score_faces_by_projection(
                verts_cur, tris_cur, error_maps, pred_depths_np,
                gt_depths, gt_uint8, mvps_np)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        selected = topological_nms_1ring(tris_cur, scores, k_actual)

        if len(selected) == 0:
            print(f"  [{strategy}] step={step}: NMS returned 0 faces — skip")
            continue

        print(f"  [{strategy}] step={step}: stellating {len(selected)} faces "
              f"(budget_rem={budget_rem})  F: {F_cur}→?")

        try:
            new_verts, new_tris, _ = topmod_stellate_cluster(
                verts_cur, tris_cur, selected)
        except Exception as e:
            print(f"  [{strategy}] WARN stellate failed: {e} — skip")
            continue

        n_bnd, _ = dlfl_boundary_stats(new_verts, new_tris)
        if n_bnd > 0:
            print(f"  [{strategy}] WARN stellate→{n_bnd} boundary edges — skip")
            continue
        new_genus = _mesh_genus(new_verts, new_tris)
        if new_genus != init_genus:
            print(f"  [{strategy}] WARN genus changed {init_genus}→{new_genus} — skip")
            continue

        # Rebuild optimiser on new mesh
        verts_t, faces_t, opt, sched = _rebuild(new_verts, new_tris, step)
        tris_np   = new_tris
        verts_np  = new_verts
        prev_verts_np = verts_np.copy()
        warmup_counter   = WARMUP_STEPS
        lap_boost_left   = LAP_WARMUP_STEPS
        depth_boost_left = DEPTH_WARMUP_STEPS
        n_rounds += 1

        # Post-refine eval
        with torch.no_grad():
            post_sils = render_views_n(ctx, verts_t, faces_t, mvps)
        iou_after = compute_iou_n(post_sils, gt_uint8)
        refine_log.append({
            'step':        step,
            'n_stellated': int(len(selected)),
            'score_max':   float(scores.max()),
            'iou_before':  float(iou_before),
            'iou_after':   float(iou_after),
            'total_faces': int(len(new_tris)),
        })
        print(f"  [{strategy}]   IoU {iou_before:.4f}→{iou_after:.4f}  "
              f"F={len(new_tris)}  genus={new_genus}")

    # ── Final eval ────────────────────────────────────────────────────────────
    with torch.no_grad():
        final_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(final_sils, gt_uint8)
    final_faces = int(faces_t.shape[0])
    print(f"  [{strategy}] DONE  IoU={final_iou:.4f}  F={final_faces}  rounds={n_rounds}")
    return final_iou, refine_log


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_scene(shape: str, device: str) -> dict:
    """Build nvdiffrast ctx, cameras, GT silhouettes/depths, cc2 init mesh."""
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ctx        = dr.RasterizeCudaContext()
    mvps, eyes = make_6_cameras(radius=CAMERA_RADIUS, device=device)

    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)
    shape_name  = shape

    if shape == 'torus':
        gt_verts, gt_tris = make_torus()
    elif shape == 'bunny' and os.path.exists(BUNNY_PATH):
        gt_verts, gt_tris = load_obj(BUNNY_PATH)
    else:
        _p = os.path.join(_SAMPLE_DIR, f'{shape}.obj')
        if os.path.exists(_p):
            gt_verts, gt_tris = load_obj(_p)
        elif os.path.exists(BUNNY_PATH):
            print(f"  WARNING: {shape!r} not found — falling back to bunny")
            gt_verts, gt_tris = load_obj(BUNNY_PATH); shape_name = 'bunny'
        else:
            print("  WARNING: no shape found — using torus")
            gt_verts, gt_tris = make_torus(); shape_name = 'torus'

    gt_verts = normalize_to_range(gt_verts)
    vt_gt = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    ft_gt = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)

    gt_sils, gt_deps = [], []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, vt_gt, ft_gt, mvps[i], (IMG_RES, IMG_RES))
            gt_sils.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255)
                .clip(0, 255).astype(np.uint8))
            gt_deps.append(ndc_z_i.cpu().numpy())
    del vt_gt, ft_gt
    gt_uint8  = np.stack(gt_sils, axis=0)
    gt_depths = np.stack(gt_deps, axis=0).astype(np.float32)

    # cc2 init mesh (~480F)
    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    pos, fcs = mesh_to_arrays(ico)
    init_verts = np.array(pos, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)

    return {
        'ctx': ctx, 'mvps': mvps,
        'gt_uint8': gt_uint8, 'gt_depths': gt_depths,
        'init_verts': init_verts, 'init_tris': init_tris,
        'shape_name': shape_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarize(values: List[float]) -> dict:
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': 0., 'std': 0., 'min': 0., 'max': 0., 'values': []}
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    return {'n': n, 'mean': mean, 'std': std,
            'min': min(values), 'max': max(values), 'values': list(values)}


def welch_significant(a_vals: List[float], b_vals: List[float],
                       noise_floor: float = 0.02) -> Tuple[bool, float, float]:
    """Welch t-test (1-tailed: b > a) + noise_floor guard.

    Returns (is_significant, delta_mean, p_value_approx).
    """
    from scipy import stats
    a, b = np.array(a_vals), np.array(b_vals)
    delta = float(b.mean() - a.mean())
    if abs(delta) <= noise_floor:
        return False, delta, 1.0
    if len(a) < 2 or len(b) < 2:
        return abs(delta) > noise_floor, delta, 0.5
    t_stat, p_two = stats.ttest_ind(b, a, equal_var=False)
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    return (delta > noise_floor and p_one < 0.05), delta, p_one


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phantom Apex Gradient where-to-refine")
    ap.add_argument('--shape',        default='cow',
                    help='Shape to reconstruct (cow, bunny, torus, …)')
    ap.add_argument('--trials',       type=int, default=4)
    ap.add_argument('--variants',     nargs='+', default=['phantom', 'random'],
                    choices=list(VALID_STRATEGIES))
    ap.add_argument('--budget',       type=int, default=800,
                    help='Face budget target (start=480F cc2)')
    ap.add_argument('--total-steps',  type=int, default=TOTAL_STEPS)
    ap.add_argument('--k-stellate',   type=int, default=PHM_K_STELLATE,
                    help='Faces to stellate per round')
    ap.add_argument('--interval',     type=int, default=PHM_INTERVAL,
                    help='Main-loop steps between refinement probes')
    ap.add_argument('--device',       default='cuda')
    ap.add_argument('--json',         default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 72)
    print(f"PHANTOM APEX GRADIENT  shape={args.shape}  budget={args.budget}F  "
          f"trials={args.trials}")
    print(f"variants={args.variants}  total_steps={args.total_steps}  "
          f"k={args.k_stellate}  interval={args.interval}")
    print("=" * 72)

    # Budget → max_rounds calculation
    # Each round: k_stellate stellates × 2F = +2k faces
    # Rounds needed: (budget - 480) / (2 * k_stellate)
    init_F     = 480   # approximate cc2 face count
    n_faces_up = max(args.budget - init_F, 0)
    faces_per_round = 2 * args.k_stellate
    max_rds = max(1, math.ceil(n_faces_up / max(faces_per_round, 1)) + 2)
    print(f"Est. rounds to budget: ≈{max_rds}")

    scene = setup_scene(args.shape, device)
    print(f"Scene: shape={scene['shape_name']}  "
          f"init V={len(scene['init_verts'])} F={len(scene['init_tris'])}")

    results: Dict[str, dict] = {}
    for variant in args.variants:
        ious, faces_list = [], []
        print(f"\n─── Variant '{variant}' × {args.trials} ───")
        for t in range(args.trials):
            t0 = time.time()
            iou, log = run_strategy_refine(
                scene['ctx'],
                scene['init_verts'].copy(),
                scene['init_tris'].copy(),
                scene['gt_uint8'],
                scene['gt_depths'],
                scene['mvps'],
                device,
                strategy     = variant,
                total_steps  = args.total_steps,
                face_budget  = args.budget,
                k_stellate   = args.k_stellate,
                phm_interval = args.interval,
                max_rounds   = max_rds,
            )
            final_faces = log[-1]['total_faces'] if log else len(scene['init_tris'])
            ious.append(iou)
            faces_list.append(final_faces)
            print(f"  trial {t+1}/{args.trials}: IoU={iou:.4f}  "
                  f"faces={final_faces}  ({time.time()-t0:.0f}s)")

        s = summarize(ious)
        results[variant] = {
            'iou':        s,
            'faces_mean': sum(faces_list) / len(faces_list),
        }
        print(f"  → IoU {s['mean']:.4f} ± {s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]  "
              f"faces_mean={results[variant]['faces_mean']:.0f}")

    # ── H1 significance test (phantom > random) ───────────────────────────────
    noise_floor = 0.02
    print("\n" + "=" * 72)
    print(f"PAIRWISE  noise_floor={noise_floor}  Welch 1-tailed p<0.05")
    print("=" * 72)
    vs = args.variants
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a_iou = results[vs[i]]['iou']['values']
            b_iou = results[vs[j]]['iou']['values']
            sig, dmean, p = welch_significant(a_iou, b_iou, noise_floor)
            verdict = ("SIGNIFICANT (H1 pass)" if sig
                       else f"NOT significant (p={p:.3f})")
            print(f"  {vs[j]} vs {vs[i]}:  Δmean={dmean:+.4f}  p≈{p:.3f}  → {verdict}")

    if 'phantom' in results and 'random' in results:
        ph_iou  = results['phantom']['iou']['values']
        rnd_iou = results['random']['iou']['values']
        sig_h1, d_h1, p_h1 = welch_significant(rnd_iou, ph_iou, noise_floor)
        print(f"\nH1 (phantom > random @{args.budget}F):  "
              f"{'PASS ✓' if sig_h1 else 'FAIL ✗'}  "
              f"Δ={d_h1:+.4f}  p≈{p_h1:.3f}")
        if not sig_h1:
            print("  → H1 FAIL: phantom gradient does not outperform random selection.")
            print("    Recommendation: revert to cheap heuristic (eval_local_refine Biso).")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({
                'shape':       scene['shape_name'],
                'budget':      args.budget,
                'trials':      args.trials,
                'total_steps': args.total_steps,
                'k_stellate':  args.k_stellate,
                'interval':    args.interval,
                'results':     {k: {**v,
                                    'iou': v['iou']}
                                for k, v in results.items()},
            }, f, indent=2)
        print(f"\nRaw results → {args.json}")


if __name__ == '__main__':
    main()
