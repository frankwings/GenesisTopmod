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
# REINFORCE / probe-and-keep hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

RL_N_CANDIDATES  = 720   # heuristic pre-filter pool for REINFORCE / probe-keep
RL_N_SELECT      = 360   # stellates per round  (iso-budget: 480 + 2×360×2 = 1920F)
RL_N_RL_STEPS    = 5     # REINFORCE iterations per refinement trigger
RL_BATCH         = 6     # B — samples per REINFORCE step
RL_K_PROBE       = 10    # K — gradient steps per probe sample
RL_LR_INIT       = 0.5   # logit learning rate
RL_EPS_CARD      = 0.001 # ε_card — cardinality penalty in reward signal
ISO_MAX_ROUNDS   = 2     # 2 rounds → 480 + 720 + 720 = 1920F (iso-budget with cc3)

PK_BATCH_SIZE    = 60    # probe-and-keep: candidate faces per probe batch
PK_K_PROBE       = 10    # gradient steps per batch probe


# ─────────────────────────────────────────────────────────────────────────────
# Shared probe helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_K(
    ctx,
    verts_np:  np.ndarray,
    tris_np:   np.ndarray,
    gt_uint8:  np.ndarray,
    gt_depths: np.ndarray,
    mvps:      torch.Tensor,
    device:    str,
    K:         int = 10,
) -> float:
    """Cold-start Adam, run K gradient steps, return mean loss of last 3 steps.

    Creates and immediately releases all GPU tensors so this can be called many
    times from REINFORCE / probe-and-keep inner loops without memory leaks.
    """
    N_v = mvps.shape[0]
    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]

    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32, device=device)
    opt     = torch.optim.Adam([verts_t], lr=LR)

    losses: List[float] = []
    for _ in range(K):
        opt.zero_grad()
        l_sil = torch.tensor(0.0, device=device)
        l_dep = torch.tensor(0.0, device=device)
        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            l_sil += F.l1_loss(sil[0], targets[i])
            l_dep += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        l_sil /= N_v; l_dep /= N_v
        total = (l_sil + W_DEPTH * l_dep
                 + W_LAP * laplacian_loss(verts_t, faces_t)
                 + W_EDGE * edge_length_loss(verts_t, faces_t))
        total.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        losses.append(total.item())

    del verts_t, faces_t, targets, gt_depth_t, gt_fg_t, opt
    if device != 'cpu':
        torch.cuda.empty_cache()

    tail = max(1, min(3, K))
    return float(np.mean(losses[-tail:]))


def _get_prefiltered_candidates(
    face_scores: np.ndarray,
    n_cands:     int,
) -> np.ndarray:
    """Return indices of top-n_cands faces by heuristic score."""
    n = min(n_cands, len(face_scores))
    return np.argsort(-face_scores)[:n].astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE selection
# ─────────────────────────────────────────────────────────────────────────────

def _reinforce_select_faces(
    ctx,
    verts_np:        np.ndarray,
    tris_np:         np.ndarray,
    gt_uint8:        np.ndarray,
    gt_depths:       np.ndarray,
    mvps:            torch.Tensor,
    device:          str,
    candidate_faces: np.ndarray,        # [C] pre-filtered face indices
    n_select:        int,
    n_rl_steps:      int   = RL_N_RL_STEPS,
    B:               int   = RL_BATCH,
    K_probe:         int   = RL_K_PROBE,
    rl_lr:           float = RL_LR_INIT,
    eps_card:        float = RL_EPS_CARD,
) -> np.ndarray:
    """REINFORCE-Ball: learn per-face Bernoulli logits to select which faces to stellate.

    Algorithm (faithful to DMesh++ Eq.8)
    -------------------------------------
    1.  z_f = 0   ∀f ∈ candidates   (logit; phi_f = sigmoid(z_f) = 0.5 initially)
    2.  For n_rl_steps iterations:
        a. phi  = sigmoid(z)
        b. Sample B binary masks  m_i ~ Bernoulli(phi),  i = 1..B
        c. For each mask i:
             - Apply stellates for all f with m_i_f = 1 to a COPY of the mesh
             - Probe K gradient steps → loss_i
             - reward_i = −loss_i − eps_card · |m_i|   (higher = better)
        d. Baseline  b = mean(reward_i)
        e. REINFORCE gradient ascent (maximise expected reward):
             z_f  +=  rl_lr · (1/B) · Σ_i (reward_i − b) · (m_i_f − phi_f)
    3.  Decode: top-n_select faces by sigmoid(z_final).

    Returns [≤n_select] int32 face indices selected from candidate_faces.
    """
    C = len(candidate_faces)
    if C == 0:
        return np.array([], dtype=np.int32)
    n_sel = min(n_select, C)

    z = np.zeros(C, dtype=np.float64)          # per-candidate logits

    for rl_iter in range(n_rl_steps):
        phi     = 1.0 / (1.0 + np.exp(-z))     # [C] sigmoid
        rewards = np.zeros(B, dtype=np.float64)
        masks   = np.zeros((B, C), dtype=np.float64)

        for bi in range(B):
            m = (np.random.rand(C) < phi).astype(np.float64)
            masks[bi] = m
            sel_idx   = candidate_faces[m > 0.5]

            if len(sel_idx) == 0:
                # No selection → probe base mesh
                loss_val = compute_loss_K(ctx, verts_np, tris_np,
                                          gt_uint8, gt_depths, mvps, device, K_probe)
                rewards[bi] = -loss_val
                continue

            try:
                pv, pt, _ = topmod_stellate_cluster(
                    verts_np.copy(), tris_np.copy(), sel_idx)
                n_bnd, _ = dlfl_boundary_stats(pv, pt)
                probe_v, probe_t = (verts_np, tris_np) if n_bnd > 0 else (pv, pt)
                loss_val = compute_loss_K(ctx, probe_v, probe_t,
                                          gt_uint8, gt_depths, mvps, device, K_probe)
            except Exception:
                loss_val = compute_loss_K(ctx, verts_np, tris_np,
                                          gt_uint8, gt_depths, mvps, device, K_probe)

            rewards[bi] = -loss_val - eps_card * float(m.sum())

        # Gradient ascent: z += rl_lr · (1/B) Σ_i (r_i − b) · (m_i − phi)
        baseline  = rewards.mean()
        advantage = rewards - baseline                       # [B]
        grad      = (advantage[:, None] * (masks - phi[None, :])).mean(axis=0)  # [C]
        z        += rl_lr * grad

        if rl_iter % 2 == 0 or rl_iter == n_rl_steps - 1:
            phi_cur = 1.0 / (1.0 + np.exp(-z))
            print(f"    [RL it={rl_iter}]  phi∈[{phi_cur.min():.3f},{phi_cur.max():.3f}]"
                  f"  r_mean={rewards.mean():.5f}  adv_std={advantage.std():.5f}")

    phi_final = 1.0 / (1.0 + np.exp(-z))
    top_idx   = np.argsort(-phi_final)[:n_sel]
    selected  = candidate_faces[top_idx]
    print(f"    [RL decode]  selected={len(selected)}  "
          f"phi_sel_mean={phi_final[top_idx].mean():.3f}")
    return selected.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Probe-and-keep greedy selection
# ─────────────────────────────────────────────────────────────────────────────

def _probe_keep_select_faces(
    ctx,
    verts_np:        np.ndarray,
    tris_np:         np.ndarray,
    gt_uint8:        np.ndarray,
    gt_depths:       np.ndarray,
    mvps:            torch.Tensor,
    device:          str,
    candidate_faces: np.ndarray,        # [C] pre-filtered face indices
    n_select:        int,
    batch_size:      int = PK_BATCH_SIZE,
    K_probe:         int = PK_K_PROBE,
) -> np.ndarray:
    """Greedy probe-and-keep: iterate batches, keep a batch if its ΔL < 0.

    Each probe applies the *accumulated accepted set ∪ current batch* to a
    COPY of the original verts_np/tris_np — this avoids face-index shifting
    that would occur if stellates were committed incrementally.

    Returns [≤n_select] int32 face indices.
    """
    C = len(candidate_faces)
    if C == 0:
        return np.array([], dtype=np.int32)

    # Baseline: probe the original unmodified mesh
    base_loss = compute_loss_K(ctx, verts_np, tris_np,
                               gt_uint8, gt_depths, mvps, device, K_probe)
    print(f"    [PK] base_loss={base_loss:.6f}")

    accepted: List[int] = []
    best_loss = base_loss
    n_batches = math.ceil(C / batch_size)

    for bi in range(n_batches):
        if len(accepted) >= n_select:
            break

        batch     = candidate_faces[bi * batch_size: (bi + 1) * batch_size]
        trial_set = np.array(accepted + batch.tolist(), dtype=np.int32)

        try:
            pv, pt, _ = topmod_stellate_cluster(
                verts_np.copy(), tris_np.copy(), trial_set)
            n_bnd, _ = dlfl_boundary_stats(pv, pt)
            if n_bnd > 0:
                continue   # topology break: skip batch
            probe_loss = compute_loss_K(ctx, pv, pt,
                                        gt_uint8, gt_depths, mvps, device, K_probe)
        except Exception:
            continue

        delta = probe_loss - best_loss
        if delta < 0:
            accepted.extend(batch.tolist())
            best_loss = probe_loss
            print(f"    [PK] batch {bi:2d}: KEEP  Δloss={delta:+.6f}  "
                  f"n_accepted={len(accepted)}")

    result = np.array(accepted[:n_select], dtype=np.int32)
    print(f"    [PK] done: {len(result)} accepted  best_loss={best_loss:.6f}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Phantom Apex Gradient — where-to-refine via marginal DOF benefit probe
# ═════════════════════════════════════════════════════════════════════════════
#
# Core idea: insert ε-lifted phantom vertices (apex) at every face centroid,
# render the phantom mesh in a single forward+backward pass, and extract the
# normal-projected gradient at each apex.  High gradient = "this face would
# USE the extra DOF if we stellated here" — a causal signal, unlike the cheap
# heuristic's correlational "this face fits badly".
#
# Three steps: Probe → Select (area-norm + 1-ring NMS + top-K) → Execute.

PH_K              = 20      # top-K faces per round (小步快跑)
PH_EMA_ALPHA      = 0.95    # loss EMA smoothing factor
PH_LOSS_FRAC      = 0.05    # plateau: EMA delta < 5% of post-refine 100-step drop
PH_DISP_THRESH    = 5e-5    # vertex displacement threshold (normalized to unit bbox)
PH_DISP_WINDOW    = 50      # steps for displacement average
PH_IOU_ASSERT_TOL = 5e-4    # IoU diff tolerance for phantom ↔ original
                             # (F→3F topology change causes ~1e-4 rasterisation noise
                             #  even at ε=0; 5e-4 catches gross errors, tolerates noise)
PH_MAX_EPS_RETRY  = 3       # max epsilon doubling retries
PH_MIN_STEPS_FIRST = 80     # min steps before first trigger
PH_REFINE_COOLDOWN = 30     # min steps between successive probes
PH_MAX_WAIT        = 80     # force-trigger if no refine in this many steps
                             # (K=20, 800F budget → 8 rounds × 80 = 640 steps min)

# ── Local-IoU-deficit where-to-refine (LIOU) ─────────────────────────────────
# Replaces phantom gradient + displacement trigger.
# Trigger: loss-EMA relative stall (single condition, no displacement).
# Signal:  per-face worst-view local silhouette IoU + depth error.
# Restart: partial warm restart (preserve Adam momentum for old vertices).
LIOU_STALL_WINDOW  = 50     # steps of loss_ema history for stall check
LIOU_STALL_FRAC    = 0.01   # stall if |ema[t]-ema[t-W]|/ema[t] < 1%
LIOU_STALL_MIN_STEP = 200   # don't trigger before this many steps
LIOU_MAX_WAIT      = 1200   # safety valve: force if no trigger in this many steps
LIOU_K             = 20     # top-K faces per round (same as PH_K)
LIOU_PAD_PX        = 8      # footprint bbox padding (px)
LIOU_MIN_PX2       = 4      # min projected area to score (px²)
LIOU_LAMBDA_DEPTH  = 50.0   # λ: depth error weight in deficit score
LIOU_LAP_BOOST     = LAP_WARMUP_STEPS  # lap boost after warm restart
MAX_LIOU_EXTRUDES  = 3   # max extrude injections per LIOU_EX trial


def _face_geometry_np(verts_np: np.ndarray, tris_np: np.ndarray):
    """Per-face centroids [F,3], normals [F,3], areas [F] for triangle mesh."""
    v0 = verts_np[tris_np[:, 0]]
    v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / (norms + 1e-12)
    areas = 0.5 * norms.squeeze(-1)
    return centroids, normals, areas


def phantom_apex_probe(
    ctx,
    verts_t:    torch.Tensor,       # [V, 3] float32, current optimized verts
    faces_t:    torch.Tensor,       # [F, 3] int32
    mvps:       torch.Tensor,       # [N_v, 4, 4]
    gt_uint8:   np.ndarray,         # [N_v, H, W] uint8
    gt_depths:  np.ndarray,         # [N_v, H, W] float32
    device:     str,
    iou_tol:    float = PH_IOU_ASSERT_TOL,
    max_retry:  int   = PH_MAX_EPS_RETRY,
) -> np.ndarray:
    """Probe per-face marginal DOF benefit via phantom epsilon-lifted apexes.

    Steps:
      1. a_f = centroid_f + ε·n_f   (ε = max(5e-4·diag_f, 1e-4))
      2. Build phantom mesh: all F faces → 3F triangles with apex leaf nodes
      3. Single 6-view forward+backward
      4. score_f = |∇_{a_f}L · n_f| / (A_f + 1e-8)   (area-normalized)

    Assert: |IoU_phantom − IoU_original| < iou_tol, else double ε and retry.

    Returns face_scores [F] float64.
    """
    N_v     = mvps.shape[0]
    F_count = faces_t.shape[0]
    V_orig  = verts_t.shape[0]

    verts_np = verts_t.detach().cpu().numpy().astype(np.float64)
    faces_np = faces_t.cpu().numpy()

    centroids, normals, areas = _face_geometry_np(verts_np, faces_np)

    # Per-face bounding box diagonal for adaptive epsilon
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    bbox_min = np.minimum(np.minimum(v0, v1), v2)
    bbox_max = np.maximum(np.maximum(v0, v1), v2)
    diag_f   = np.linalg.norm(bbox_max - bbox_min, axis=1)
    base_eps = np.maximum(5e-4 * diag_f, 1e-4)

    # Render original mesh for IoU assert
    with torch.no_grad():
        orig_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    orig_iou = compute_iou_n(orig_sils, gt_uint8)

    # GT targets for loss
    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]

    # Phantom face topology (constant): each face i → 3 tris with apex V_orig+i
    apex_idx = np.arange(F_count, dtype=np.int32) + V_orig
    pf = np.empty((3 * F_count, 3), dtype=np.int32)
    pf[0::3, 0] = faces_np[:, 0]; pf[0::3, 1] = faces_np[:, 1]; pf[0::3, 2] = apex_idx
    pf[1::3, 0] = faces_np[:, 1]; pf[1::3, 1] = faces_np[:, 2]; pf[1::3, 2] = apex_idx
    pf[2::3, 0] = faces_np[:, 2]; pf[2::3, 1] = faces_np[:, 0]; pf[2::3, 2] = apex_idx
    phantom_faces_t = torch.tensor(pf, dtype=torch.int32, device=device)

    epsilon = base_eps.copy()

    for retry in range(max_retry + 1):
        apex_pos = centroids + epsilon[:, None] * normals        # [F, 3]

        orig_verts_d = verts_t.detach()
        apex_t = torch.tensor(
            apex_pos, dtype=torch.float32, device=device).requires_grad_(True)
        phantom_verts = torch.cat([orig_verts_d, apex_t], dim=0)  # [V+F, 3]

        # ── IoU assert ──
        with torch.no_grad():
            phantom_sils = render_views_n(ctx, phantom_verts, phantom_faces_t, mvps)
        phantom_iou = compute_iou_n(phantom_sils, gt_uint8)
        iou_diff = abs(phantom_iou - orig_iou)

        if iou_diff > iou_tol:
            if retry < max_retry:
                print(f"    [PHANTOM ASSERT] IoU diff={iou_diff:.2e} > tol={iou_tol:.1e}"
                      f" — doubling ε (retry {retry + 1}/{max_retry})")
                epsilon *= 2.0
                del apex_t, phantom_verts
                continue
            else:
                print(f"    [PHANTOM WARN] IoU diff={iou_diff:.2e} persists "
                      f"after {max_retry} retries — proceeding anyway")

        # ── Forward + backward (single pass, 6 views) ──
        loss_total = torch.tensor(0.0, device=device)
        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, phantom_verts, phantom_faces_t, mvps[i], (IMG_RES, IMG_RES))
            loss_total = loss_total + F.l1_loss(sil[0], targets[i])
            loss_total = loss_total + W_DEPTH * depth_loss_masked(
                ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        loss_total = loss_total / N_v
        loss_total.backward()

        # ── Extract scores ──
        apex_grad = apex_t.grad
        if apex_grad is None:
            print("    [PHANTOM WARN] No gradient at apex vertices!")
            del apex_t, phantom_verts, phantom_faces_t, targets, gt_depth_t, gt_fg_t
            return np.zeros(F_count, dtype=np.float64)

        normals_t = torch.tensor(normals, dtype=torch.float32, device=device)
        proj  = (apex_grad * normals_t).sum(dim=1)       # [F]
        score = torch.abs(proj)                           # |∇·n| (凹凸同等)
        areas_t = torch.tensor(areas, dtype=torch.float32, device=device)
        score = score / (areas_t + 1e-8)                  # area-normalize

        result = score.detach().cpu().numpy().astype(np.float64)

        del apex_t, phantom_verts, phantom_faces_t, targets, gt_depth_t, gt_fg_t
        if device != 'cpu':
            torch.cuda.empty_cache()

        return result

    # Should never reach here
    return np.zeros(F_count, dtype=np.float64)


def _nms_1ring(
    scores:  np.ndarray,       # [F] float64
    tris_np: np.ndarray,       # [F, 3] int32
    K:       int,
) -> np.ndarray:
    """1-ring topological NMS: greedily pick top-K faces, suppress shared-edge
    neighbours (prevent slivers from adjacent stellates).

    Returns selected face indices [≤K] int32.
    """
    F_count = len(tris_np)

    # Build edge → face adjacency
    edge_to_faces: Dict[tuple, List[int]] = defaultdict(list)
    for fi in range(F_count):
        vs = [int(tris_np[fi, j]) for j in range(3)]
        for a, b in [(vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])]:
            edge_to_faces[(min(a, b), max(a, b))].append(fi)

    # Face → neighbour set
    neighbors: Dict[int, set] = defaultdict(set)
    for face_list in edge_to_faces.values():
        for a in face_list:
            for b in face_list:
                if a != b:
                    neighbors[a].add(b)

    # Greedy descent
    order      = np.argsort(-scores)
    selected   = []
    suppressed = set()
    for fi in order:
        fi = int(fi)
        if fi in suppressed:
            continue
        selected.append(fi)
        if len(selected) >= K:
            break
        for nb in neighbors[fi]:
            suppressed.add(nb)

    return np.array(selected, dtype=np.int32)


def score_faces_by_local_iou_deficit(
    verts_np:       np.ndarray,   # [V, 3] float64
    tris_np:        np.ndarray,   # [F, 3] int32
    pred_sils:      np.ndarray,   # [N, H, W] float32, >0.5=fg
    pred_depths_np: np.ndarray,   # [N, H, W] float32
    gt_uint8:       np.ndarray,   # [N, H, W] uint8, <128=fg
    gt_depths:      np.ndarray,   # [N, H, W] float32
    mvps_np:        np.ndarray,   # [N, 4, 4] float64
    pad_px:         int   = LIOU_PAD_PX,
    min_proj_px2:   int   = LIOU_MIN_PX2,
    lambda_depth:   float = LIOU_LAMBDA_DEPTH,
) -> np.ndarray:   # [F] float64
    """Score faces by worst-view local silhouette IoU deficit + depth error."""
    F_count = len(tris_np)
    N, H, W = pred_sils.shape
    scores = np.zeros(F_count, dtype=np.float64)

    for fi in range(F_count):
        tri        = tris_np[fi]
        face_verts = verts_np[tri]   # [3, 3]
        pts_h      = np.concatenate(
            [face_verts, np.ones((3, 1), dtype=np.float64)], axis=1)  # [3, 4]

        min_local_iou = 1.0
        max_depth_err = 0.0
        n_valid_views = 0

        for vi in range(N):
            mvp  = mvps_np[vi]
            clip = pts_h @ mvp.T      # [3, 4]
            ws   = clip[:, 3]

            if np.any(ws <= 0.0):
                continue

            ndc_x = clip[:, 0] / ws
            ndc_y = clip[:, 1] / ws
            px    = (ndc_x * 0.5 + 0.5) * W - 0.5
            py    = (1.0 - (ndc_y * 0.5 + 0.5)) * H - 0.5

            x0 = max(0, int(px.min()) - pad_px)
            x1 = min(W, int(px.max()) + pad_px + 1)
            y0 = max(0, int(py.min()) - pad_px)
            y1 = min(H, int(py.max()) + pad_px + 1)

            if (x1 - x0) * (y1 - y0) < min_proj_px2:
                continue

            pred_fg_patch = pred_sils[vi][y0:y1, x0:x1] > 0.5
            gt_fg_patch   = gt_uint8[vi][y0:y1, x0:x1] < 128

            intersection = (pred_fg_patch & gt_fg_patch).sum()
            union        = (pred_fg_patch | gt_fg_patch).sum()
            if union == 0:
                local_iou = 1.0
            else:
                local_iou = intersection / union

            if gt_fg_patch.sum() >= 2:
                depth_err       = np.abs(
                    pred_depths_np[vi][y0:y1, x0:x1] - gt_depths[vi][y0:y1, x0:x1])
                local_depth_err = float(depth_err[gt_fg_patch].mean())
            else:
                local_depth_err = 0.0

            min_local_iou = min(min_local_iou, local_iou)
            max_depth_err = max(max_depth_err, local_depth_err)
            n_valid_views += 1

        if n_valid_views == 0:
            deficit_f = 0.0
        else:
            deficit_f = (1.0 - min_local_iou) + lambda_depth * max_depth_err

        scores[fi] = deficit_f

    return scores


def _warm_restart(
    v_np:        np.ndarray,   # [new_V, 3] float64  (new mesh verts)
    t_np:        np.ndarray,   # [new_F, 3] int32
    old_opt:     torch.optim.Optimizer,
    old_V:       int,          # number of original vertices (before stellate)
    cur_step:    int,
    total_steps: int,
    device:      str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.optim.Optimizer,
           torch.optim.lr_scheduler._LRScheduler]:
    """Partial warm restart: copy Adam state for original vertices, zero-init for new."""
    new_V = len(v_np)
    verts_t = torch.tensor(v_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(t_np, dtype=torch.int32,  device=device)

    # Get current LR from old optimizer
    cur_lr = old_opt.param_groups[0]['lr']

    # Create new optimizer at current LR
    new_opt = torch.optim.Adam([verts_t], lr=cur_lr)

    # Get old param tensor (single param group, single param)
    old_param = list(old_opt.state.keys())[0]
    old_state = old_opt.state[old_param]

    if 'exp_avg' in old_state:
        # Build new state tensors: size [new_V, 3]
        new_exp_avg     = torch.zeros(new_V, 3, device=device, dtype=torch.float32)
        new_exp_avg_sq  = torch.zeros(new_V, 3, device=device, dtype=torch.float32)
        copy_V = min(old_V, new_V)
        new_exp_avg[:copy_V]    = old_state['exp_avg'][:copy_V].clone()
        new_exp_avg_sq[:copy_V] = old_state['exp_avg_sq'][:copy_V].clone()

        # Install state into new optimizer
        new_opt.state[verts_t] = {
            'step':         old_state.get('step', torch.tensor(0)),
            'exp_avg':      new_exp_avg,
            'exp_avg_sq':   new_exp_avg_sq,
        }

    # New cosine scheduler from current LR down to LR_MIN for remaining steps
    rem = max(total_steps - cur_step - 1, 1)
    new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        new_opt, T_max=rem, eta_min=LR_MIN)

    return verts_t, faces_t, new_opt, new_sched


# ─────────────────────────────────────────────────────────────────────────────
# Phantom-gradient optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_phantom_refine(
    ctx,
    verts_init:   np.ndarray,
    tris_init:    np.ndarray,
    gt_uint8:     np.ndarray,         # [N, H, W] uint8
    gt_depths:    np.ndarray,         # [N, H, W] float32
    mvps:         torch.Tensor,       # [N, 4, 4]
    device:       str,
    total_steps:  int   = 800,
    face_budget:  int   = 800,
    K:            int   = PH_K,
    strategy:     str   = 'phantom',  # 'phantom' | 'random' | 'heuristic'
) -> Tuple[float, int, List[dict]]:
    """Optimise mesh with phantom-gradient (or random/heuristic) face refinement.

    Plateau trigger (dual AND):
      A) Loss EMA(α=0.95) delta < 5 % of post-refine 100-step reference drop
      B) Mean vertex displacement over last 50 steps < 1e-5 (unit-bbox-normalised)

    Returns (final_iou, final_faces, refine_log).
    """
    N_v = mvps.shape[0]
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]
    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    mvps_np = mvps.detach().cpu().numpy().astype(np.float64)

    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    init_F     = len(tris_np)
    init_genus = _mesh_genus(verts_np, tris_np)
    n_bnd0, _  = dlfl_boundary_stats(verts_np, tris_np)
    if n_bnd0 > 0:
        raise RuntimeError(f"Init mesh has {n_bnd0} boundary edges")

    max_rounds = (face_budget - init_F) // (K * 2) + 1  # upper bound

    def _rebuild(v_np, t_np, cur_step):
        vt = torch.tensor(v_np, dtype=torch.float32,
                          device=device).requires_grad_(True)
        ft = torch.tensor(t_np, dtype=torch.int32, device=device)
        op = torch.optim.Adam([vt], lr=LR)
        rem = max(total_steps - cur_step - 1, 1)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=rem, eta_min=LR_MIN)
        return vt, ft, op, sc

    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)
    opt     = torch.optim.Adam([verts_t], lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    # ── Tracking state ──
    refine_log:      List[dict] = []
    n_rounds         = 0
    warmup_counter   = 0
    lap_boost_left   = 0
    depth_boost_left = 0
    last_refine_step = -PH_REFINE_COOLDOWN

    # Plateau tracking
    loss_ema:          Optional[float] = None
    ref_drop:          Optional[float] = None      # 100-step reference drop
    post_refine_step:  Optional[int]   = None      # step when last refine happened
    post_refine_loss:  Optional[float] = None      # loss_ema right after last refine
    verts_prev:        Optional[np.ndarray] = None # previous-step positions
    disp_history:      List[float] = []            # per-step vertex displacement

    print(f"  Init: V={len(verts_np)} F={init_F}  budget={face_budget}  "
          f"K={K}  strategy={strategy}  max_rounds≈{max_rounds}")

    for step in range(total_steps):
        # ── Warmup ──
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup_counter -= 1

        # ── Forward ──
        opt.zero_grad()
        loss_sil   = torch.tensor(0., device=device)
        loss_depth = torch.tensor(0., device=device)
        w_lap = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        w_depth_mult = ((DEPTH_WARMUP_STEPS - depth_boost_left) /
                        DEPTH_WARMUP_STEPS if depth_boost_left > 0 else 1.0)
        if lap_boost_left   > 0: lap_boost_left -= 1
        if depth_boost_left > 0: depth_boost_left -= 1

        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            loss_sil   = loss_sil + F.l1_loss(sil[0], targets[i])
            loss_depth = loss_depth + depth_loss_masked(
                ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        loss_sil   = loss_sil / N_v
        loss_depth = loss_depth / N_v
        lap  = laplacian_loss(verts_t, faces_t)
        edge = edge_length_loss(verts_t, faces_t)
        loss = (loss_sil + W_DEPTH * w_depth_mult * loss_depth
                + w_lap * lap + W_EDGE * edge)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

        cur_loss = loss.item()

        # ── EMA + displacement tracking ──
        if loss_ema is None:
            loss_ema = cur_loss
        else:
            loss_ema = PH_EMA_ALPHA * loss_ema + (1 - PH_EMA_ALPHA) * cur_loss

        cur_verts_np = verts_t.detach().cpu().numpy()
        if verts_prev is not None and cur_verts_np.shape == verts_prev.shape:
            bbox_diag = max(float(cur_verts_np.max() - cur_verts_np.min()), 1e-8)
            disp = float(np.mean(np.linalg.norm(
                cur_verts_np - verts_prev, axis=1))) / bbox_diag
            disp_history.append(disp)
        verts_prev = cur_verts_np.copy()

        # ── Reference drop tracking ──
        if (post_refine_step is not None and post_refine_loss is not None
                and step == post_refine_step + 100):
            ref_drop = max(post_refine_loss - loss_ema, 1e-8)

        # ── Periodic print ──
        if step % 100 == 0:
            with torch.no_grad():
                iou_now = compute_iou_n(
                    render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
            avg_disp = (float(np.mean(disp_history[-PH_DISP_WINDOW:]))
                        if disp_history else 0.0)
            print(f"  step={step:4d}  IoU={iou_now:.4f}  "
                  f"F={faces_t.shape[0]}  rounds={n_rounds}  "
                  f"ema={loss_ema:.5f}  disp={avg_disp:.2e}")

        # ── Plateau-based refinement trigger (dual AND) ──
        budget_rem = face_budget - len(tris_np)
        can_refine = (
            budget_rem >= K * 2
            and step >= PH_MIN_STEPS_FIRST
            and (step - last_refine_step) >= PH_REFINE_COOLDOWN
            and warmup_counter == 0
        )
        if not can_refine:
            continue

        # Safety valve: force trigger if no refine in PH_MAX_WAIT steps
        waited = step - max(last_refine_step, 0)
        force_trigger = waited >= PH_MAX_WAIT

        # Condition A: loss EMA stalled
        if ref_drop is not None:
            ema_delta = abs(cur_loss - loss_ema)
            cond_a = ema_delta < PH_LOSS_FRAC * ref_drop
        elif step >= PH_MIN_STEPS_FIRST and n_rounds == 0:
            cond_a = True
        else:
            cond_a = False

        # Condition B: vertex displacement stalled
        if len(disp_history) >= PH_DISP_WINDOW:
            avg_disp = float(np.mean(disp_history[-PH_DISP_WINDOW:]))
            cond_b = avg_disp < PH_DISP_THRESH
        elif n_rounds == 0 and step >= PH_MIN_STEPS_FIRST:
            cond_b = True
        else:
            cond_b = False

        if not ((cond_a and cond_b) or force_trigger):
            continue
        if force_trigger and not (cond_a and cond_b):
            avg_d = (float(np.mean(disp_history[-PH_DISP_WINDOW:]))
                     if disp_history else 0.0)
            print(f"  [FORCE TRIGGER] step={step}: waited {waited} steps  "
                  f"disp={avg_d:.2e}  ema_delta={abs(cur_loss-loss_ema):.2e}")

        # ── Probe + select ──
        verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
        tris_cur  = faces_t.cpu().numpy()

        with torch.no_grad():
            iou_before = compute_iou_n(
                render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)

        n_sel = min(K, budget_rem // 2)

        if strategy == 'phantom':
            scores = phantom_apex_probe(
                ctx, verts_t, faces_t, mvps, gt_uint8, gt_depths, device)
            refine_faces = _nms_1ring(scores, tris_cur, n_sel)
            score_info = (f"phantom score∈[{scores.min():.4f},{scores.max():.4f}]  "
                          f"sel_mean={scores[refine_faces].mean():.4f}"
                          if len(refine_faces) > 0 else "no faces")

        elif strategy == 'random':
            refine_faces = np.random.choice(
                len(tris_cur), size=min(n_sel, len(tris_cur)),
                replace=False).astype(np.int32)
            score_info = "random selection"

        elif strategy == 'heuristic':
            # Existing cheap heuristic: project-based per-face error score
            with torch.no_grad():
                pred_sils      = render_views_n(ctx, verts_t, faces_t, mvps)
                pred_depths_np = render_depths_n(ctx, verts_t, faces_t, mvps)
            error_maps = compute_missing_error_maps(pred_sils, gt_uint8)
            face_scores = score_faces_by_projection(
                verts_cur, tris_cur, error_maps, pred_depths_np,
                gt_depths, gt_uint8, mvps_np)
            # Use 1-ring NMS for fair comparison
            refine_faces = _nms_1ring(face_scores, tris_cur, n_sel)
            score_info = (f"heuristic score∈[{face_scores.min():.1f},"
                          f"{face_scores.max():.1f}]"
                          if len(refine_faces) > 0 else "no faces")

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        if len(refine_faces) == 0:
            print(f"  [REFINE] step={step}: no faces selected — skipping")
            last_refine_step = step
            continue

        print(f"  [REFINE] step={step}: stellating {len(refine_faces)} faces  "
              f"{score_info}  F: {len(tris_cur)}→?")

        try:
            new_verts, new_tris, old_V = topmod_stellate_cluster(
                verts_cur, tris_cur, refine_faces)
        except Exception as e:
            print(f"  [WARN] stellate failed: {e}")
            last_refine_step = step
            continue

        n_bnd, _ = dlfl_boundary_stats(new_verts, new_tris)
        if n_bnd > 0:
            print(f"  [WARN] stellate produced {n_bnd} boundary edges — skipping")
            last_refine_step = step
            continue
        new_genus = _mesh_genus(new_verts, new_tris)
        if new_genus != init_genus:
            print(f"  [WARN] genus changed {init_genus}→{new_genus} — skipping")
            last_refine_step = step
            continue

        # ── Commit ──
        verts_t, faces_t, opt, sched = _rebuild(new_verts, new_tris, step)
        tris_np  = new_tris
        verts_np = new_verts
        warmup_counter   = WARMUP_STEPS
        lap_boost_left   = LAP_WARMUP_STEPS
        depth_boost_left = DEPTH_WARMUP_STEPS
        last_refine_step = step
        n_rounds += 1

        # Reset plateau tracking for next round
        post_refine_step = step
        post_refine_loss = loss_ema
        verts_prev       = None
        disp_history.clear()

        with torch.no_grad():
            iou_after = compute_iou_n(
                render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
        refine_log.append({
            'step':         step,
            'n_stellated':  int(len(refine_faces)),
            'iou_before':   float(iou_before),
            'iou_after':    float(iou_after),
            'total_faces':  int(len(new_tris)),
            'strategy':     strategy,
        })
        print(f"       IoU {iou_before:.4f} → {iou_after:.4f}  "
              f"F={len(new_tris)}  genus={new_genus}")

    # ── Final ──
    with torch.no_grad():
        final_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(final_sils, gt_uint8)
    final_faces = int(faces_t.shape[0])
    print(f"  [DONE] IoU={final_iou:.4f}  F={final_faces}  rounds={n_rounds}")
    return final_iou, final_faces, refine_log


# ─────────────────────────────────────────────────────────────────────────────
# LIOU (Local-IoU-deficit) optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_liou_refine(
    ctx,
    verts_init:   np.ndarray,
    tris_init:    np.ndarray,
    gt_uint8:     np.ndarray,         # [N, H, W] uint8
    gt_depths:    np.ndarray,         # [N, H, W] float32
    mvps:         torch.Tensor,       # [N, 4, 4]
    device:       str,
    total_steps:  int   = 2400,
    face_budget:  int   = 800,
    K:            int   = LIOU_K,
    strategy:     str   = 'liou',     # 'liou' | 'random'
    use_extrude:  bool  = False,      # LIOU_EX: try topology extrude at stall
    out_obj:      str    = None,      # optional: dump final mesh to this .obj
) -> Tuple[float, int, List[dict], bool]:
    """Optimise mesh with LIOU-triggered face refinement + partial warm restart.

    Trigger: loss-EMA relative stall (|ema[t] - ema[t-W]| / ema[t] < LIOU_STALL_FRAC)
    Safety valve: fire at LIOU_MAX_WAIT steps if stall never triggers.

    Returns (final_iou, final_faces, refine_log, trial_valid).
    trial_valid = True iff ALL rounds used true-stall trigger (not force).
    """
    N_v = mvps.shape[0]
    mvps_np = mvps.detach().cpu().numpy().astype(np.float64)

    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]
    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)

    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    init_F   = len(tris_np)
    init_genus = _mesh_genus(verts_np, tris_np)
    print(f"  [LIOU] Init mesh: V={len(verts_np)} F={init_F} genus={init_genus} "
          f"budget={face_budget} K={K} strategy={strategy} use_extrude={use_extrude}")

    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                            device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32, device=device)

    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    refine_log:      List[dict] = []
    n_rounds         = 0
    n_extrudes       = 0
    lap_boost_left   = 0
    depth_boost_left = 0
    loss_ema         = None       # running EMA of loss
    loss_ema_history: List[float] = []
    last_refine_step = -1         # step at which last refinement fired
    warmup_counter   = 0

    for step in range(total_steps):

        # ── Warmup LR scaling ────────────────────────────────────────────────
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = opt.defaults['lr'] * max(frac, 0.05)
            warmup_counter -= 1

        # ── Forward pass ─────────────────────────────────────────────────────
        opt.zero_grad()
        loss_sil   = torch.tensor(0., device=device)
        loss_depth = torch.tensor(0., device=device)
        w_lap = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        w_depth_mult = ((DEPTH_WARMUP_STEPS - depth_boost_left) /
                        DEPTH_WARMUP_STEPS if depth_boost_left > 0 else 1.0)
        if lap_boost_left   > 0: lap_boost_left -= 1
        if depth_boost_left > 0: depth_boost_left -= 1

        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            loss_sil   = loss_sil + F.l1_loss(sil[0], targets[i])
            loss_depth = loss_depth + depth_loss_masked(
                ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        loss_sil   = loss_sil / N_v
        loss_depth = loss_depth / N_v
        lap  = laplacian_loss(verts_t, faces_t)
        edge = edge_length_loss(verts_t, faces_t)
        loss = (loss_sil + W_DEPTH * w_depth_mult * loss_depth
                + w_lap * lap + W_EDGE * edge)

        # ── Loss EMA ─────────────────────────────────────────────────────────
        loss_val = loss.item()
        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss_val
        loss_ema_history.append(loss_ema)

        # ── Backward ─────────────────────────────────────────────────────────
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % 200 == 0:
            with torch.no_grad():
                iou_now = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps),
                                        gt_uint8)
            cur_lr = opt.param_groups[0]['lr']
            print(f"  [LIOU] step {step:4d}/{total_steps}  "
                  f"loss_ema={loss_ema:.4f}  iou={iou_now:.4f}  "
                  f"lr={cur_lr:.2e}  F={faces_t.shape[0]}")

        # ── Refinement trigger ───────────────────────────────────────────────
        cur_F = int(faces_t.shape[0])
        budget_ok = (cur_F + 2 * K) <= face_budget
        min_step_ok = step >= LIOU_STALL_MIN_STEP

        if not budget_ok:
            continue   # hard cap reached

        # Check stall condition
        stall_triggered = False
        ema_rel_change  = float('inf')
        if (min_step_ok and len(loss_ema_history) > LIOU_STALL_WINDOW):
            ema_old = loss_ema_history[-LIOU_STALL_WINDOW - 1]
            if ema_old > 1e-8:
                ema_rel_change = abs(loss_ema - ema_old) / ema_old
                if ema_rel_change < LIOU_STALL_FRAC:
                    stall_triggered = True

        # Safety valve: force trigger if too many steps since last refine
        steps_since_last = step - last_refine_step
        force_triggered = (min_step_ok and
                           steps_since_last >= LIOU_MAX_WAIT and
                           not stall_triggered)

        if not (stall_triggered or force_triggered):
            continue

        trigger_type = 'stall' if stall_triggered else 'force'
        tag = '[TRUE STALL]' if stall_triggered else '[FORCE TRIGGER]'
        print(f"  {tag} step={step} ema_rel_change={ema_rel_change:.4f} "
              f"(threshold={LIOU_STALL_FRAC}) F={cur_F}")

        # ── Score faces ──────────────────────────────────────────────────────
        with torch.no_grad():
            pred_sils_np = render_views_n(
                ctx, verts_t, faces_t, mvps)                         # [N,H,W] numpy
            pred_depths_np2 = render_depths_n(
                ctx, verts_t, faces_t, mvps)                         # [N,H,W] numpy

        verts_np_cur = verts_t.detach().cpu().numpy().astype(np.float64)
        tris_np_cur  = faces_t.detach().cpu().numpy()

        # ── LIOU_EX: attempt topology extrude first ───────────────────────────
        extrude_done = False
        if use_extrude and n_extrudes < MAX_LIOU_EXTRUDES:
            error_maps_ex = compute_missing_error_maps(pred_sils_np, gt_uint8)
            is_bn, cc_frac = detect_topology_bottleneck(error_maps_ex)
            if is_bn and cc_frac >= EXTRUDE_MIN_CC:
                face_votes, face_dir = vote_faces_multiview_n(
                    error_maps_ex, verts_np_cur, tris_np_cur, mvps)
                if face_votes.max() >= 2:
                    cluster = select_face_cluster(
                        face_votes, tris_np_cur, top_k=8, grow_rings=1)
                    if len(cluster) > 0:
                        d  = face_dir[cluster].sum(axis=0)
                        dn = np.linalg.norm(d)
                        if dn > 1e-8:
                            extrude_dir  = d / dn
                            extrude_dist = EXTRUDE_BBOX_FRAC * float(
                                verts_np_cur.max() - verts_np_cur.min())
                            try:
                                with torch.no_grad():
                                    iou_before_ex = compute_iou_n(
                                        render_views_n(ctx, verts_t, faces_t, mvps),
                                        gt_uint8)
                                old_V_ex = len(verts_np_cur)
                                nv, nt, _ = topmod_extrude_cluster(
                                    verts_np_cur, tris_np_cur, cluster,
                                    extrude_dir, extrude_dist)
                                n_bnd_ex, _ = dlfl_boundary_stats(nv, nt)
                                if n_bnd_ex == 0:
                                    nv2, nt2 = split_new_long_edges(nv, nt, old_V_ex)
                                    if dlfl_boundary_stats(nv2, nt2)[0] == 0:
                                        nv, nt = nv2, nt2
                                    new_genus_ex = _mesh_genus(nv, nt)
                                    verts_t, faces_t, opt, sched = _warm_restart(
                                        nv, nt, opt, old_V_ex, step, total_steps, device)
                                    warmup_counter   = WARMUP_STEPS
                                    lap_boost_left   = LIOU_LAP_BOOST
                                    depth_boost_left = DEPTH_WARMUP_STEPS
                                    n_extrudes      += 1
                                    n_rounds        += 1
                                    last_refine_step = step
                                    loss_ema_history.clear()
                                    loss_ema = None
                                    with torch.no_grad():
                                        iou_after_ex = compute_iou_n(
                                            render_views_n(ctx, verts_t, faces_t, mvps),
                                            gt_uint8)
                                    refine_log.append({
                                        'step':           step,
                                        'trigger_type':   'extrude',
                                        'n_stellated':    0,
                                        'n_extruded':     int(len(cluster)),
                                        'extrude_dir':    extrude_dir.tolist(),
                                        'extrude_dist':   float(extrude_dist),
                                        'cc_frac':        float(cc_frac),
                                        'ema_rel_change': (float(ema_rel_change)
                                                           if ema_rel_change != float('inf')
                                                           else -1.0),
                                        'iou_before':     float(iou_before_ex),
                                        'iou_after':      float(iou_after_ex),
                                        'total_faces':    int(faces_t.shape[0]),
                                    })
                                    print(f"  [EXTRUDE #{n_extrudes}] step={step}  "
                                          f"cluster={len(cluster)}  "
                                          f"dist={extrude_dist:.4f}  "
                                          f"cc_frac={cc_frac:.3f}  "
                                          f"genus={new_genus_ex}")
                                    print(f"       IoU {iou_before_ex:.4f} → "
                                          f"{iou_after_ex:.4f}  F={faces_t.shape[0]}")
                                    extrude_done = True
                                else:
                                    print(f"  [WARN] extrude boundary {n_bnd_ex} — skip")
                            except Exception as e:
                                print(f"  [WARN] extrude failed: {e}")

        if extrude_done:
            # extrude handled this stall — skip stellate
            pass
        else:
            # ── Score + stellate faces ────────────────────────────────────────
            if strategy == 'liou':
                scores = score_faces_by_local_iou_deficit(
                    verts_np_cur, tris_np_cur, pred_sils_np, pred_depths_np2,
                    gt_uint8, gt_depths, mvps_np)
                refine_faces = _nms_1ring(scores, tris_np_cur, K)
            else:  # random
                rng = np.random.default_rng()
                refine_faces = rng.choice(len(tris_np_cur),
                                          size=min(K, len(tris_np_cur)),
                                          replace=False).astype(np.int32)

            if len(refine_faces) == 0:
                print(f"  [LIOU] No faces selected — skipping round")
                continue

            # ── Measure IoU before ───────────────────────────────────────────
            with torch.no_grad():
                iou_before = compute_iou_n(
                    render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)

            # ── Stellate selected faces ──────────────────────────────────────
            old_V = len(verts_np_cur)
            new_verts_np, new_tris_np, _ = topmod_stellate_cluster(
                verts_np_cur, tris_np_cur, list(refine_faces))
            new_genus = _mesh_genus(new_verts_np, new_tris_np)

            # ── Partial warm restart ─────────────────────────────────────────
            verts_t, faces_t, opt, sched = _warm_restart(
                new_verts_np, new_tris_np, opt, old_V, step, total_steps, device)

            # Post-refinement warmup & boosts
            warmup_counter   = WARMUP_STEPS
            lap_boost_left   = LIOU_LAP_BOOST
            depth_boost_left = DEPTH_WARMUP_STEPS

            # ── Measure IoU after ────────────────────────────────────────────
            with torch.no_grad():
                iou_after = compute_iou_n(
                    render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)

            n_rounds    += 1
            last_refine_step = step
            loss_ema_history.clear()
            loss_ema = None

            refine_log.append({
                'step':           step,
                'trigger_type':   trigger_type,
                'n_stellated':    int(len(refine_faces)),
                'ema_rel_change': float(ema_rel_change) if ema_rel_change != float('inf') else -1.0,
                'iou_before':     float(iou_before),
                'iou_after':      float(iou_after),
                'total_faces':    int(faces_t.shape[0]),
            })
            print(f"       IoU {iou_before:.4f} → {iou_after:.4f}  "
                  f"F={faces_t.shape[0]}  genus={new_genus}")

    # ── Final ─────────────────────────────────────────────────────────────────
    with torch.no_grad():
        final_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(final_sils, gt_uint8)
    final_faces = int(faces_t.shape[0])
    trial_valid = all(r['trigger_type'] in ('stall', 'extrude') for r in refine_log)
    if out_obj is not None:
        _v = verts_t.detach().cpu().numpy()
        _f = faces_t.cpu().numpy()
        with open(out_obj, 'w') as _fh:
            for p in _v:
                _fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for tri in _f:
                _fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        print(f"  [LIOU] saved mesh -> {out_obj}")
    print(f"  [LIOU DONE] IoU={final_iou:.4f}  F={final_faces}  "
          f"rounds={n_rounds}  extrudes={n_extrudes}  trial_valid={trial_valid}")
    return final_iou, final_faces, refine_log, trial_valid


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
    # ── REINFORCE / probe-and-keep strategy ──────────────────────────────────
    strategy:        str   = 'heuristic',  # 'heuristic' | 'reinforce' | 'probe_keep'
    n_rl_steps:      int   = RL_N_RL_STEPS,
    rl_batch:        int   = RL_BATCH,
    rl_k_probe:      int   = RL_K_PROBE,
    rl_lr:           float = RL_LR_INIT,
    rl_eps_card:     float = RL_EPS_CARD,
    pk_batch_size:   int   = PK_BATCH_SIZE,
    pk_k_probe:      int   = PK_K_PROBE,
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

                score_max  = float(face_scores.max())
                budget_rem = face_budget - len(tris_cur)

                if strategy == 'heuristic':
                    refine_faces = select_refine_faces(
                        face_scores, top_k_refine, MIN_FACE_SCORE, budget_rem)
                elif strategy in ('reinforce', 'probe_keep'):
                    n_sel = min(RL_N_SELECT, budget_rem // 2)
                    cands = _get_prefiltered_candidates(face_scores,
                                                        RL_N_CANDIDATES)
                    if n_sel <= 0 or len(cands) == 0:
                        refine_faces = np.array([], dtype=np.int32)
                    elif strategy == 'reinforce':
                        print(f"  [REINFORCE] step={step}: C={len(cands)}"
                              f"  n_sel={n_sel}"
                              f"  n_rl_steps={n_rl_steps}  B={rl_batch}")
                        refine_faces = _reinforce_select_faces(
                            ctx, verts_cur, tris_cur,
                            gt_uint8, gt_depths, mvps, device,
                            candidate_faces=cands,
                            n_select=n_sel,
                            n_rl_steps=n_rl_steps,
                            B=rl_batch,
                            K_probe=rl_k_probe,
                            rl_lr=rl_lr,
                            eps_card=rl_eps_card)
                    else:
                        print(f"  [PROBE-KEEP] step={step}: C={len(cands)}"
                              f"  n_sel={n_sel}")
                        refine_faces = _probe_keep_select_faces(
                            ctx, verts_cur, tris_cur,
                            gt_uint8, gt_depths, mvps, device,
                            candidate_faces=cands,
                            n_select=n_sel,
                            batch_size=pk_batch_size,
                            K_probe=pk_k_probe)
                else:
                    raise ValueError(f"Unknown strategy: {strategy!r}")

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
      A    — cc2 baseline (with depth, no refinement)
      cc3  — cc3 baseline (with depth, no refinement, ~1920 faces)
      B    — local stellate, heuristic selection (starts cc2, adaptive density)
      C    — local stellate + extrude for topology gaps
      Biso — heuristic selection iso-budget (360×2 rounds → ~1920F)
      R    — REINFORCE where-to-refine iso-budget (360×2 rounds → ~1920F)
      P    — probe-and-keep greedy iso-budget (360×2 rounds → ~1920F)
      PH   — phantom apex gradient (K=20 per round, plateau-triggered)
      RND  — random face selection (control for PH)
      HEUR — cheap heuristic with same K=20 (fair comparison for PH)
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

    # ── iso-budget variants (R, P, Biso): 360 stellates × 2 rounds → ~1920F ──
    _ISO_KW = dict(
        face_budget     = 1920,
        refine_interval = REFINE_INTERVAL,
        max_rounds      = ISO_MAX_ROUNDS,
        top_k_refine    = RL_N_SELECT,
    )
    if variant == 'Biso':
        iou, _, _, log = run_local_refine(
            ctx, iv, it, gt_uint8, gt_depths, mvps, device,
            use_extrude=False, total_steps=total_steps,
            strategy='heuristic', **_ISO_KW)
        final_faces = log[-1]['total_faces'] if log else len(it)
        return {'iou': float(iou), 'n_rounds': len(log),
                'final_faces': final_faces, 'time': time.time() - t0, 'log': log}

    if variant == 'R':
        iou, _, _, log = run_local_refine(
            ctx, iv, it, gt_uint8, gt_depths, mvps, device,
            use_extrude=False, total_steps=total_steps,
            strategy='reinforce', **_ISO_KW)
        final_faces = log[-1]['total_faces'] if log else len(it)
        return {'iou': float(iou), 'n_rounds': len(log),
                'final_faces': final_faces, 'time': time.time() - t0, 'log': log}

    if variant == 'P':
        iou, _, _, log = run_local_refine(
            ctx, iv, it, gt_uint8, gt_depths, mvps, device,
            use_extrude=False, total_steps=total_steps,
            strategy='probe_keep', **_ISO_KW)
        final_faces = log[-1]['total_faces'] if log else len(it)
        return {'iou': float(iou), 'n_rounds': len(log),
                'final_faces': final_faces, 'time': time.time() - t0, 'log': log}

    # ── Phantom apex gradient variants ──────────────────────────────────────
    _ph_budget = kw.get('face_budget', 800)
    _ph_K      = kw.get('ph_K', PH_K)
    if variant in ('PH', 'RND', 'HEUR'):
        strategy_map = {'PH': 'phantom', 'RND': 'random', 'HEUR': 'heuristic'}
        iou, n_f, log = run_phantom_refine(
            ctx, iv, it, gt_uint8, gt_depths, mvps, device,
            total_steps=total_steps,
            face_budget=_ph_budget,
            K=_ph_K,
            strategy=strategy_map[variant])
        return {'iou': float(iou), 'n_rounds': len(log),
                'final_faces': n_f, 'time': time.time() - t0, 'log': log}

    # ── LIOU (Local-IoU-deficit) variants ──────────────────────────────────
    _liou_budget = kw.get('face_budget', 800)
    _liou_K      = kw.get('liou_K', LIOU_K)
    if variant in ('LIOU', 'LIOU_RND', 'LIOU_EX'):
        strategy_map2 = {'LIOU': 'liou', 'LIOU_RND': 'random', 'LIOU_EX': 'liou'}
        iou, n_f, log, trial_valid = run_liou_refine(
            ctx, iv, it, gt_uint8, gt_depths, mvps, device,
            total_steps=total_steps,
            face_budget=_liou_budget,
            K=_liou_K,
            strategy=strategy_map2[variant],
            use_extrude=(variant == 'LIOU_EX'))
        n_extrudes = sum(1 for r in log if r.get('trigger_type') == 'extrude')
        return {'iou': float(iou), 'n_rounds': len(log),
                'final_faces': n_f, 'time': time.time() - t0, 'log': log,
                'trial_valid': trial_valid, 'n_extrudes': n_extrudes}

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
                    choices=['A', 'cc3', 'B', 'C', 'Biso', 'R', 'P',
                             'PH', 'RND', 'HEUR', 'LIOU', 'LIOU_RND', 'LIOU_EX'])
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
