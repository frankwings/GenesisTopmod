#!/usr/bin/env python3
"""
eval_local_refine.py — DMesh-inspired local error-driven face refinement.

Core idea
---------
Instead of growing new protrusions (extrude) when IoU plateaus, selectively
increase mesh DENSITY where the mesh most needs it: project every face to 6
views, score by (silhouette missing + depth residual) in its footprint, and
stellate the top-K faces. Faces in well-covered, flat regions stay coarse;
high-error silhouette regions accumulate resolution.

Comparison with existing approach
----------------------------------
  eval_extrude_v3  : plateau + topology-gap signal → extrude cluster (new volume)
  eval_local_refine: periodic per-face error scoring → stellate top-K (in-plane)
These are complementary; variant C uses BOTH.

Acceptance criterion: reach near cc3-baseline IoU (0.9692) using far fewer than
cc3's 1920 faces by concentrating the face budget in high-error regions.

Variants
--------
  A = cc2 baseline (480F, no refinement)
  B = local-stellate only  (starts cc2, adaptive density via stellate)
  C = local-stellate + extrude for topology gaps (best-of-both)

Usage
-----
  python3 eval_local_refine.py --shape cow --total-steps 400
  python3 eval_local_refine.py --shape cow --trials 4 --variants A B --total-steps 400

Long runs (>5 min) → launch via:
  python3 -m framework.bg_task launch --id local_refine_v1 --timeout 3600 \\
      --cwd /home/kingy/Foundation/ZenithLoom -- \\
      python3 /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/eval_local_refine.py \\
      --shape cow --trials 4 --variants A B C --total-steps 400 --json eval_out/local_refine.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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

from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

from eval_extrude_v3 import (
    # camera / rendering
    make_6_cameras, render_sil_and_depth, render_views_n, render_depths_n,
    # metrics
    compute_iou_n, compute_depth_l1_n, depth_loss_masked,
    # topology tools
    dlfl_boundary_stats, assert_watertight_v3, _mesh_genus,
    topmod_stellate_cluster, topmod_extrude_cluster, split_new_long_edges,
    # error / voting
    compute_missing_error_maps, detect_topology_bottleneck,
    vote_faces_multiview_n, select_face_cluster,
    # constants
    N_VIEWS, IMG_RES, CAMERA_RADIUS, W_LAP, W_LAP_BOOST, W_EDGE, W_DEPTH,
    WARMUP_STEPS, LAP_WARMUP_STEPS, DEPTH_WARMUP_STEPS,
    EXTRUDE_BBOX_FRAC, MIN_STEP_FOR_INJ,
    PLATEAU_STEPS, PLATEAU_EPS, EVAL_INTERVAL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (local-refine specific)
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_STEPS      = 400
LR               = 3e-3
LR_MIN           = 3e-5

# ── Face scoring ─────────────────────────────────────────────────────────────
SCORE_SIL_W   = 1.0    # weight of silhouette-missing signal in per-face score
SCORE_DEPTH_W = 50.0   # weight of mean depth residual (scaled to ~comparable range)
SCORE_PAD_PX  = 6      # px padding around face footprint bbox when sampling error

# ── Refinement schedule ──────────────────────────────────────────────────────
# Strategy: few large rounds (1-3) early in training, then long uninterrupted
# convergence. Too many small rounds waste budget on optimizer cold-start
# recovery (measured: 9×24 at interval=80 → 0.9329 = same as cc2 baseline).
REFINE_INTERVAL  = 200 # min steps between successive refinement rounds
TOP_K_REFINE     = 80  # faces to stellate per round (each adds +2F for triangles)
MIN_FACE_SCORE   = 2.0 # minimum per-face score to stellate (noise gate)
MIN_PROJ_PX2     = 4   # skip faces whose projected footprint area < this (px²)
FACE_BUDGET      = 1200 # hard cap on total faces (< cc3's 1920)
MAX_REFINE_ROUNDS = 3  # few large rounds, not many small ones

# ── Topology-gap extrude (variant C only) ────────────────────────────────────
EXTRUDE_MIN_CC   = 0.02  # min cc_fraction to attempt extrude
MAX_EXTRUDES     = 5     # max extrude injections in variant C

# ── Warmup after refinement ──────────────────────────────────────────────────
REFINE_WARMUP    = WARMUP_STEPS
REFINE_LAP_BOOST = LAP_WARMUP_STEPS


# ─────────────────────────────────────────────────────────────────────────────
# Per-face error scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_faces_by_projection(
    verts_np:       np.ndarray,   # [V, 3] float64
    tris_np:        np.ndarray,   # [F, 3] int32
    error_maps:     np.ndarray,   # [N_V, H, W] float32, >0.5 = missing pixel
    pred_depths_np: np.ndarray,   # [N_V, H, W] float32 NDC-z
    gt_depths:      np.ndarray,   # [N_V, H, W] float32 NDC-z
    gt_uint8:       np.ndarray,   # [N_V, H, W] uint8, 0=fg
    mvps_np:        np.ndarray,   # [N_V, 4, 4] float64
    sil_weight:     float = SCORE_SIL_W,
    depth_weight:   float = SCORE_DEPTH_W,
    pad_px:         int   = SCORE_PAD_PX,
    min_proj_px2:   int   = MIN_PROJ_PX2,
) -> np.ndarray:
    """Project each face to 6 views, accumulate per-face error score.

    For each face × view:
      - Project the face's 3 vertices to image space.
      - Take the padded footprint bbox.
      - Accumulate silhouette-missing pixels (error_maps > 0.5) in the bbox.
      - Accumulate mean depth residual |pred − gt| on GT fg pixels in the bbox.

    Returns face_scores [F] float64.  High score = face is near a region
    where geometry or resolution is lacking.
    """
    F_count = len(tris_np)
    N_V, H, W = error_maps.shape
    scores = np.zeros(F_count, dtype=np.float64)

    for fi in range(F_count):
        tri       = tris_np[fi]
        face_verts = verts_np[tri]   # [3, 3]
        pts_h     = np.concatenate(
            [face_verts, np.ones((3, 1), dtype=np.float64)], axis=1)  # [3, 4]

        for vi in range(N_V):
            mvp = mvps_np[vi]
            clip = pts_h @ mvp.T      # [3, 4]
            ws   = clip[:, 3]

            # Skip face if any vertex is behind camera
            if np.any(ws <= 0.0):
                continue

            ndc_x = clip[:, 0] / ws
            ndc_y = clip[:, 1] / ws
            px    = (ndc_x * 0.5 + 0.5) * W - 0.5
            py    = (1.0 - (ndc_y * 0.5 + 0.5)) * H - 0.5

            # Footprint bbox (padded)
            x0 = max(0, int(px.min()) - pad_px)
            x1 = min(W, int(px.max()) + pad_px + 1)
            y0 = max(0, int(py.min()) - pad_px)
            y1 = min(H, int(py.max()) + pad_px + 1)

            if (x1 - x0) * (y1 - y0) < min_proj_px2:
                continue  # degenerate projection

            # Silhouette missing signal
            sil_err = float(error_maps[vi][y0:y1, x0:x1].sum())
            scores[fi] += sil_weight * sil_err

            # Depth residual on GT fg pixels in footprint
            gt_fg_patch = gt_uint8[vi][y0:y1, x0:x1] < 128
            if gt_fg_patch.sum() >= 2:
                depth_resid = np.abs(
                    pred_depths_np[vi][y0:y1, x0:x1] - gt_depths[vi][y0:y1, x0:x1]
                )
                scores[fi] += depth_weight * float(depth_resid[gt_fg_patch].mean())

    return scores


def select_refine_faces(
    face_scores:    np.ndarray,   # [F] float64
    top_k:          int,
    min_score:      float,
    face_budget_rem: int,
) -> np.ndarray:
    """Return indices of at most top_k faces to stellate.

    - Only include faces with score >= min_score.
    - Limit to face_budget_rem to stay within the face budget.
    """
    eligible = np.where(face_scores >= min_score)[0]
    if len(eligible) == 0:
        return np.array([], dtype=np.int32)
    order   = np.argsort(-face_scores[eligible])
    top_idx = eligible[order[:top_k]]
    # Each stellate of a triangle face adds 2 faces (+2F per stellated face).
    # Allow at most face_budget_rem // 2 stellates.
    max_stellate = max(0, face_budget_rem // 2)
    top_idx = top_idx[:max_stellate]
    return top_idx.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation loop — local refinement
# ─────────────────────────────────────────────────────────────────────────────

def run_local_refine(
    ctx,
    verts_init:      np.ndarray,
    tris_init:       np.ndarray,
    gt_uint8:        np.ndarray,            # [N, H, W] uint8, 0=fg
    gt_depths:       np.ndarray,            # [N, H, W] float32 NDC-z
    mvps:            torch.Tensor,          # [N, 4, 4]
    device:          str,
    use_extrude:     bool  = False,         # variant C: also extrude topology gaps
    total_steps:     int   = TOTAL_STEPS,
    top_k_refine:    int   = TOP_K_REFINE,
    face_budget:     int   = FACE_BUDGET,
    refine_interval: int   = REFINE_INTERVAL,
    max_rounds:      int   = MAX_REFINE_ROUNDS,
) -> Tuple[float, np.ndarray, np.ndarray, List[dict]]:
    """Local error-driven refinement loop.

    Returns (final_iou, pred_sils, pred_depths, refine_log).
    refine_log: list of dicts per refinement round with step/n_stellated/
    score_max/iou_before/iou_after/total_faces.
    """
    N_v = mvps.shape[0]

    # Pre-compute GT depth tensors for depth loss
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]

    # Convert mvps to numpy once for projection
    mvps_np = mvps.detach().cpu().numpy().astype(np.float64)

    # Initial mesh
    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    init_F   = len(tris_np)
    init_genus = _mesh_genus(verts_np, tris_np)
    print(f"  Init mesh: V={len(verts_np)} F={init_F} genus={init_genus} "
          f"  budget={face_budget}")

    # Verify initial mesh is manifold
    n_bnd0, _ = dlfl_boundary_stats(verts_np, tris_np)
    if n_bnd0 > 0:
        raise RuntimeError(
            f"Init mesh has {n_bnd0} boundary edges — cannot use DLFL ops")

    # Build tensors
    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                            device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32, device=device)
    targets = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)

    def _rebuild(v_np, t_np, cur_step):
        """Cold-start Adam + cosine scheduler on new mesh."""
        vt = torch.tensor(v_np, dtype=torch.float32,
                          device=device).requires_grad_(True)
        ft = torch.tensor(t_np, dtype=torch.int32, device=device)
        op = torch.optim.Adam([vt], lr=LR)
        rem = max(total_steps - cur_step - 1, 1)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=rem, eta_min=LR_MIN)
        return vt, ft, op, sc

    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    refine_log:     List[dict] = []
    extrude_log:    List[dict] = []
    n_rounds        = 0
    n_extrudes      = 0
    last_refine_step = -refine_interval  # allow first round at step 0+interval
    warmup_counter  = 0
    lap_boost_left  = 0
    depth_boost_left = 0
    iou_history:    List[float] = []

    for step in range(total_steps):

        # ── Warmup lr scaling ────────────────────────────────────────────────
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup_counter -= 1

        # ── Forward pass ─────────────────────────────────────────────────────
        opt.zero_grad()
        loss_sil   = torch.tensor(0., device=device)
        loss_depth = torch.tensor(0., device=device)
        w_lap = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        w_depth_mult = (DEPTH_WARMUP_STEPS - depth_boost_left) / DEPTH_WARMUP_STEPS \
                       if depth_boost_left > 0 else 1.0
        if lap_boost_left  > 0: lap_boost_left  -= 1
        if depth_boost_left > 0: depth_boost_left -= 1

        for i in range(N_v):
            sil, ndc_z, fg_mask = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            loss_sil += F.l1_loss(sil[0], targets[i])
            loss_depth += depth_loss_masked(
                ndc_z, fg_mask, gt_depth_t[i], gt_fg_t[i])
        loss_sil   /= N_v
        loss_depth /= N_v
        lap  = laplacian_loss(verts_t, faces_t)
        edge = edge_length_loss(verts_t, faces_t)
        loss = (loss_sil + W_DEPTH * w_depth_mult * loss_depth
                + w_lap * lap + W_EDGE * edge)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
            with torch.no_grad():
                pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(pred_sils, gt_uint8)
            iou_history.append(iou)
            if step % 100 == 0:
                print(f"  step={step:4d}  IoU={iou:.4f}  "
                      f"F={faces_t.shape[0]}  rounds={n_rounds}  "
                      f"lap_boost={lap_boost_left}")

        # ── Refinement trigger ────────────────────────────────────────────────
        can_refine = (
            n_rounds < max_rounds
            and step >= MIN_STEP_FOR_INJ
            and (step - last_refine_step) >= refine_interval
            and warmup_counter == 0
            and step % EVAL_INTERVAL == 0
            and len(tris_np) < face_budget
        )
        if can_refine:
            # Check plateau (within the last PLATEAU_STEPS // EVAL_INTERVAL + 1 evals)
            n_needed = PLATEAU_STEPS // EVAL_INTERVAL + 1
            is_plateau = (
                len(iou_history) >= n_needed
                and iou_history[-1] - iou_history[-n_needed] < PLATEAU_EPS
            )
            # Also trigger on schedule (every 2×interval) regardless of plateau
            on_schedule = (step - last_refine_step) >= 2 * refine_interval

            if not (is_plateau or on_schedule):
                pass
            else:
                verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
                tris_cur  = faces_t.cpu().numpy()
                iou_before = iou_history[-1] if iou_history else 0.0

                # ── Per-face error scoring ─────────────────────────────────
                with torch.no_grad():
                    pred_sils      = render_views_n(ctx, verts_t, faces_t, mvps)
                    pred_depths_np = render_depths_n(ctx, verts_t, faces_t, mvps)
                error_maps = compute_missing_error_maps(pred_sils, gt_uint8)

                face_scores = score_faces_by_projection(
                    verts_cur, tris_cur,
                    error_maps, pred_depths_np, gt_depths, gt_uint8, mvps_np)

                score_max = float(face_scores.max())
                budget_rem = face_budget - len(tris_cur)
                refine_faces = select_refine_faces(
                    face_scores, top_k_refine, MIN_FACE_SCORE, budget_rem)

                if len(refine_faces) == 0:
                    print(f"  [REFINE] step={step}: no eligible faces "
                          f"(score_max={score_max:.1f} < {MIN_FACE_SCORE})")
                    last_refine_step = step
                else:
                    print(f"  [REFINE] step={step}: stellating "
                          f"{len(refine_faces)} faces  "
                          f"score_max={score_max:.1f}  "
                          f"F: {len(tris_cur)}→?")

                    try:
                        new_verts, new_tris, old_V = topmod_stellate_cluster(
                            verts_cur, tris_cur, refine_faces)
                    except Exception as e:
                        print(f"  [WARN] stellate failed: {e} — skipping")
                        last_refine_step = step
                        continue

                    # Manifold check
                    n_bnd, _ = dlfl_boundary_stats(new_verts, new_tris)
                    if n_bnd > 0:
                        print(f"  [WARN] stellate produced {n_bnd} boundary "
                              f"edges — skipping")
                        last_refine_step = step
                        continue
                    new_genus = _mesh_genus(new_verts, new_tris)
                    if new_genus != init_genus:
                        print(f"  [WARN] genus changed {init_genus}→{new_genus} "
                              f"— skipping")
                        last_refine_step = step
                        continue

                    # Rebuild optimizer on new mesh (cold-start)
                    verts_t, faces_t, opt, sched = _rebuild(
                        new_verts, new_tris, step)
                    tris_np, verts_np = new_tris, new_verts
                    warmup_counter  = REFINE_WARMUP
                    lap_boost_left  = REFINE_LAP_BOOST
                    depth_boost_left = DEPTH_WARMUP_STEPS
                    last_refine_step = step
                    n_rounds += 1

                    # Immediate post-refine eval
                    with torch.no_grad():
                        post_sils = render_views_n(ctx, verts_t, faces_t, mvps)
                    iou_after = compute_iou_n(post_sils, gt_uint8)
                    iou_history.clear()
                    iou_history.append(iou_after)

                    refine_log.append({
                        'step':        step,
                        'n_stellated': int(len(refine_faces)),
                        'score_max':   float(score_max),
                        'score_top5':  face_scores[refine_faces[:5]].tolist(),
                        'iou_before':  float(iou_before),
                        'iou_after':   float(iou_after),
                        'total_faces': int(len(new_tris)),
                    })
                    print(f"       IoU {iou_before:.4f} → {iou_after:.4f}  "
                          f"F={len(new_tris)}  genus={new_genus}")

        # ── Variant C: also extrude topology gaps ────────────────────────────
        if (use_extrude
                and n_extrudes < MAX_EXTRUDES
                and step >= MIN_STEP_FOR_INJ
                and warmup_counter == 0
                and step % EVAL_INTERVAL == 0
                and step != last_refine_step):
            # Quick plateau+CC check for extrude eligibility
            n_needed = PLATEAU_STEPS // EVAL_INTERVAL + 1
            if len(iou_history) >= n_needed:
                is_plateau_ex = (iou_history[-1] - iou_history[-n_needed]
                                 < PLATEAU_EPS)
                if is_plateau_ex:
                    with torch.no_grad():
                        pred_sils_ex = render_views_n(ctx, verts_t, faces_t, mvps)
                    error_maps_ex = compute_missing_error_maps(pred_sils_ex, gt_uint8)
                    is_bn, cc_frac = detect_topology_bottleneck(error_maps_ex)
                    if is_bn and cc_frac >= EXTRUDE_MIN_CC:
                        verts_ex = verts_t.detach().cpu().numpy().astype(np.float64)
                        tris_ex  = faces_t.cpu().numpy()
                        face_votes, face_dir = vote_faces_multiview_n(
                            error_maps_ex, verts_ex, tris_ex, mvps)
                        if face_votes.max() >= 2:
                            cluster = select_face_cluster(face_votes, tris_ex,
                                                           top_k=8, grow_rings=1)
                            if len(cluster) > 0:
                                d = face_dir[cluster].sum(axis=0)
                                dn = np.linalg.norm(d)
                                if dn > 1e-8:
                                    extrude_dir  = d / dn
                                    extrude_dist = EXTRUDE_BBOX_FRAC * float(
                                        verts_ex.max() - verts_ex.min())
                                    try:
                                        nv, nt, old_V = topmod_extrude_cluster(
                                            verts_ex, tris_ex, cluster,
                                            extrude_dir, extrude_dist)
                                        n_bnd_ex, _ = dlfl_boundary_stats(nv, nt)
                                        if n_bnd_ex == 0:
                                            nv, nt = split_new_long_edges(nv, nt, old_V)
                                            if dlfl_boundary_stats(nv, nt)[0] == 0:
                                                verts_t, faces_t, opt, sched = \
                                                    _rebuild(nv, nt, step)
                                                tris_np, verts_np = nt, nv
                                                warmup_counter   = WARMUP_STEPS
                                                lap_boost_left   = LAP_WARMUP_STEPS
                                                depth_boost_left = DEPTH_WARMUP_STEPS
                                                n_extrudes += 1
                                                iou_history.clear()
                                                print(f"  [EXTRUDE #{n_extrudes}] "
                                                      f"step={step}  cluster={len(cluster)}  "
                                                      f"F={len(nt)}")
                                    except Exception as e:
                                        print(f"  [WARN] extrude failed: {e}")

    # ── Final eval ────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_sils_f   = render_views_n(ctx, verts_t, faces_t, mvps)
        pred_depths_f = render_depths_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(pred_sils_f, gt_uint8)
    final_faces = int(faces_t.shape[0])
    print(f"  [DONE] IoU={final_iou:.4f}  F={final_faces}  "
          f"rounds={n_rounds}  extrudes={n_extrudes}")
    return final_iou, pred_sils_f, pred_depths_f, refine_log


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup (reused across trials)
# ─────────────────────────────────────────────────────────────────────────────

def setup_scene(shape: str, device: str) -> dict:
    """Build ctx, cameras, GT silhouettes/depths, and init cc2 icosphere."""
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
            print(f"WARNING: {shape} not found — falling back to bunny")
            gt_verts, gt_tris = load_obj(BUNNY_PATH); shape_name = 'bunny'
        else:
            print("WARNING: no shape found — using torus fallback")
            gt_verts, gt_tris = make_torus(); shape_name = 'torus'

    gt_verts = normalize_to_range(gt_verts)
    verts_gt_t = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    faces_gt_t = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)
    gt_sil_list, gt_dep_list = [], []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, verts_gt_t, faces_gt_t, mvps[i], (IMG_RES, IMG_RES))
            gt_sil_list.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255)
                .clip(0, 255).astype(np.uint8))
            gt_dep_list.append(ndc_z_i.cpu().numpy())
    gt_uint8  = np.stack(gt_sil_list, axis=0)
    gt_depths = np.stack(gt_dep_list, axis=0).astype(np.float32)
    del verts_gt_t, faces_gt_t

    # cc2 init mesh (2× Catmull-Clark, ~480 faces)
    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)

    # cc3 init mesh (3× Catmull-Clark, ~1920 faces)
    ico3 = make_icosahedron()
    ico3 = catmull_clark(ico3); ico3 = catmull_clark(ico3); ico3 = catmull_clark(ico3)
    pos3, fcs3 = mesh_to_arrays(ico3)
    cc3_verts = np.array(pos3, dtype=np.float64)
    mn3, mx3 = float(cc3_verts.min()), float(cc3_verts.max())
    cc3_verts = (cc3_verts - (mn3 + mx3) / 2.0) * (2.0 / max(mx3 - mn3, 1e-6))
    cc3_tris  = np.array(_fan_triangulate(fcs3), dtype=np.int32)

    return {
        'ctx': ctx, 'mvps': mvps, 'gt_uint8': gt_uint8, 'gt_depths': gt_depths,
        'init_verts': init_verts, 'init_tris': init_tris,
        'cc3_verts': cc3_verts, 'cc3_tris': cc3_tris,
        'shape_name': shape_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Variant runners
# ─────────────────────────────────────────────────────────────────────────────

def run_depth_baseline(
    ctx, init_verts, init_tris, gt_uint8, gt_depths, mvps,
    device, total_steps, label='cc2',
) -> Tuple[float, int]:
    """Baseline with depth loss but no injection/refinement.

    Returns (final_iou, n_faces).
    """
    N_v = mvps.shape[0]
    targets = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)

    verts_np, tris_np = adaptive_remesh(
        init_verts.copy().astype(np.float64), init_tris.copy())
    n_faces = len(tris_np)

    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]

    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)
    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    for step in range(total_steps):
        opt.zero_grad()
        sil_loss   = torch.tensor(0.0, device=device)
        d_loss_val = torch.tensor(0.0, device=device)
        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            sil_loss   += F.l1_loss(sil[0], targets[i])
            d_loss_val += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sil_loss   /= N_v
        d_loss_val /= N_v
        loss = (sil_loss + W_DEPTH * d_loss_val
                + W_LAP * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
            print(f"  [{label}] step {step:4d}/{total_steps}  "
                  f"sil={sil_loss.item():.4f}  iou={iou:.4f}")

    pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou = compute_iou_n(pred_sils, gt_uint8)
    print(f"  [{label}] FINAL  IoU={final_iou:.4f}  F={n_faces}")
    return final_iou, n_faces


def run_variant(variant: str, scene: dict, device: str,
                total_steps: int, **kw) -> dict:
    """Run one trial of a variant.

    Variants:
      A   — cc2 baseline (with depth, no refinement)
      cc3 — cc3 baseline (with depth, no refinement, ~1920 faces)
      B   — local stellate refinement (starts cc2, adaptive density)
      C   — local stellate + extrude for topology gaps
    """
    ctx       = scene['ctx']
    mvps      = scene['mvps']
    gt_uint8  = scene['gt_uint8']
    gt_depths = scene['gt_depths']
    iv        = scene['init_verts']
    it        = scene['init_tris']

    t0 = time.time()
    if variant == 'A':
        iou, n_f = run_depth_baseline(ctx, iv, it, gt_uint8, gt_depths, mvps,
                                      device, total_steps, label='cc2')
        return {'iou': float(iou), 'n_rounds': 0,
                'final_faces': n_f, 'time': time.time() - t0}

    if variant == 'cc3':
        cc3v = scene['cc3_verts']
        cc3t = scene['cc3_tris']
        iou, n_f = run_depth_baseline(ctx, cc3v, cc3t, gt_uint8, gt_depths, mvps,
                                      device, total_steps, label='cc3')
        return {'iou': float(iou), 'n_rounds': 0,
                'final_faces': n_f, 'time': time.time() - t0}

    use_extrude = (variant == 'C')
    iou, pred_sils, pred_depths, log = run_local_refine(
        ctx, iv, it, gt_uint8, gt_depths, mvps, device,
        use_extrude=use_extrude, total_steps=total_steps, **kw)
    # Count final faces from last log entry (or fallback to init)
    final_faces = log[-1]['total_faces'] if log else len(it)
    return {
        'iou':         float(iou),
        'n_rounds':    len(log),
        'final_faces': final_faces,
        'time':        time.time() - t0,
        'log':         log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers (mirror eval_multitrial.py)
# ─────────────────────────────────────────────────────────────────────────────

def summarize(values: List[float]) -> dict:
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': 0., 'std': 0., 'min': 0., 'max': 0., 'values': []}
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / max(n - 1, 1))
    return {'n': n, 'mean': mean, 'std': std,
            'min': min(values), 'max': max(values), 'values': list(values)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Local error-driven face refinement eval")
    ap.add_argument('--shape',        default='cow')
    ap.add_argument('--trials',       type=int, default=1)
    ap.add_argument('--variants',     nargs='+', default=['A', 'cc3', 'B'],
                    choices=['A', 'cc3', 'B', 'C'])
    ap.add_argument('--total-steps',  type=int, default=TOTAL_STEPS)
    ap.add_argument('--top-k',        type=int, default=TOP_K_REFINE,
                    help='faces to stellate per refinement round')
    ap.add_argument('--face-budget',  type=int, default=FACE_BUDGET,
                    help='hard cap on total faces')
    ap.add_argument('--refine-interval', type=int, default=REFINE_INTERVAL)
    ap.add_argument('--device',       default='cuda')
    ap.add_argument('--json',         default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    kw = dict(
        top_k_refine    = args.top_k,
        face_budget     = args.face_budget,
        refine_interval = args.refine_interval,
    )

    print("=" * 72)
    print(f"LOCAL REFINE EVAL  shape={args.shape}  trials={args.trials}  "
          f"variants={args.variants}  steps={args.total_steps}")
    print(f"top_k={args.top_k}  face_budget={args.face_budget}  "
          f"refine_interval={args.refine_interval}")
    print("=" * 72)

    scene = setup_scene(args.shape, device)
    print(f"Scene: shape={scene['shape_name']}  "
          f"init V={len(scene['init_verts'])} F={len(scene['init_tris'])}")

    results: Dict[str, dict] = {}
    for v in args.variants:
        ious, rounds_list, faces_list = [], [], []
        print(f"\n─── Variant {v} × {args.trials} ───")
        for t in range(args.trials):
            r = run_variant(v, scene, device, args.total_steps, **kw)
            ious.append(r['iou'])
            rounds_list.append(r.get('n_rounds', 0))
            faces_list.append(r.get('final_faces', len(scene['init_tris'])))
            print(f"  trial {t+1}/{args.trials}: "
                  f"IoU={r['iou']:.4f}  "
                  f"rounds={r.get('n_rounds',0)}  "
                  f"faces={r.get('final_faces',0)}  "
                  f"({r['time']:.0f}s)")
        s = summarize(ious)
        results[v] = {
            'iou':         s,
            'n_rounds_mean': sum(rounds_list)/len(rounds_list),
            'faces_mean':   sum(faces_list)/len(faces_list),
        }
        print(f"  → IoU mean={s['mean']:.4f} ± {s['std']:.4f}  "
              f"[min {s['min']:.4f}, max {s['max']:.4f}]")
        print(f"     faces_mean={results[v]['faces_mean']:.0f}  "
              f"(cc2=480, cc3=1920)")

    # ── Pairwise comparison ───────────────────────────────────────────────────
    noise_floor = 0.02
    print("\n" + "=" * 72)
    print("PAIRWISE (noise_floor=0.02 — nvdiffrast antialias backward is "
          "nondeterministic, need multi-trial to trust Δ<0.02)")
    print("=" * 72)
    vs = args.variants
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = results[vs[i]]['iou'], results[vs[j]]['iou']
            dmean = b['mean'] - a['mean']
            combined_sigma = math.sqrt(a['std'] ** 2 + b['std'] ** 2)
            real = abs(dmean) > noise_floor and abs(dmean) > combined_sigma
            verdict = "REAL" if real else (
                f"NOISE (|Δ|={abs(dmean):.4f}≤floor)" if abs(dmean) <= noise_floor
                else f"INCONCLUSIVE (|Δ|={abs(dmean):.4f}>floor but ≤σ_combined)")
            print(f"  {vs[j]} vs {vs[i]}: Δmean={dmean:+.4f}  → {verdict}")

    print("\nTIP: compare with cc3 baseline by running eval_extrude_v3.py with "
          "cc3 init (3× catmull-clark = 1920 faces). cc3 IoU≈0.9692 (from docs).")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'shape':       scene['shape_name'],
                'trials':      args.trials,
                'total_steps': args.total_steps,
                'top_k':       args.top_k,
                'face_budget': args.face_budget,
                'results':     results,
            }, f, indent=2)
        print(f"\nRaw results → {args.json}")


if __name__ == '__main__':
    main()
