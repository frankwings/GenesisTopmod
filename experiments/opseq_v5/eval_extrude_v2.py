#!/usr/bin/env python3
"""
eval_extrude_v2.py — Upgraded extrude-injection PoC with depth supervision.

Four upgrades over eval_extrude_inject.py:
  1. Depth supervision: render NDC-z depth maps, L1 loss on GT∩pred foreground.
  2. Resolution 128 → 256.
  3. Views 4 → 6 (4 equatorial + 2 elevated at ±35°).
  4. Post-injection local cleanup: split side-wall long edges + enhanced
     Laplacian weight for 30 steps after each injection.

Three comparison variants:
  A. Baseline: 800-step plain vertex opt, no injection, no depth.
  B. Injection only: same loop as eval_extrude_inject, 800 steps, max 5 inj.
  C. Injection + depth: B plus depth supervision.

Output:
  eval_out/viz_extrude_v2_bunny.png   (6 rows × 4 cols: GT|A|B|C)
  Console: 6-view silhouette IoU + depth L1 for each variant.

Assumptions (deviations from spec documented here):
  ASSUMPTION-1: w_depth=0.35 (spec suggests starting at 0.5). At 256px,
    depth L1 on intersection ≈ silhouette L1 in scale; 0.5 over-suppresses
    silhouette gradients in early steps. 0.35 gives 26% depth / 74% sil ratio.
  ASSUMPTION-2: "enhanced Laplacian ×3 for new vertices" implemented as global
    lambda_lap ×3 for 30 steps post-injection (0.3 vs 0.1). True per-vertex
    weighting would require a custom scatter loop not available in the shared
    laplacian_loss; global scaling achieves the same qualitative smoothing.
  ASSUMPTION-3: max_rays_per_view kept at 300 (256px error maps → more pixels,
    but 300 is already a good sample; more would slow voting without benefit).

Usage:
    python eval_extrude_v2.py [--device cuda] [--out_dir eval_out]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from pipeline.cameras            import orbit_cameras, transform_to_clip
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5         import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

# ── Re-use pure geometry helpers from eval_extrude_inject ──────────────────
from eval_extrude_inject import (
    check_manifold,
    assert_manifold,
    compute_missing_error_maps,
    largest_cc_fraction,
    detect_topology_bottleneck,
    error_pixels_to_rays,
    moller_trumbore_batch,
    select_face_cluster,
    _find_pinch_vertices,
    _clean_cluster,
    extrude_face_cluster,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
N_VIEWS       = 6
IMG_RES       = 256
CAMERA_RADIUS = 3.0

MAX_INJECTIONS     = 5
TOTAL_STEPS        = 800
EVAL_INTERVAL      = 5
PLATEAU_STEPS      = 20
PLATEAU_EPS        = 0.005
ERROR_CC_THRESH    = 0.40
WARMUP_STEPS       = 20
LAP_WARMUP_STEPS   = 30   # enhanced lap for this many steps post-injection
EXTRUDE_BBOX_FRAC  = 0.12
MIN_STEP_FOR_INJ   = 60

W_LAP        = 0.10   # base laplacian weight
W_LAP_BOOST  = 0.30   # enhanced weight 30 steps post-injection (see ASSUMPTION-2)
W_EDGE       = 0.01
W_DEPTH      = 0.35   # depth loss weight (see ASSUMPTION-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6-camera rig
# ─────────────────────────────────────────────────────────────────────────────

def make_6_cameras(
    radius: float = CAMERA_RADIUS,
    device: str   = 'cuda',
) -> Tuple[torch.Tensor, list]:
    """
    4 equatorial (elev=0, az=0/90/180/270) +
    2 elevated (elev=+35 az=45, elev=-35 az=225).
    Returns (mvps [6,4,4], eyes [6 × (x,y,z)]).
    """
    mvps_eq, eyes_eq = orbit_cameras(
        4, elevation_deg=0.0, radius=radius,
        azimuths_deg=[0.0, 90.0, 180.0, 270.0], device=device)

    elev_cfg = [(35.0, 45.0), (-35.0, 225.0)]
    mvps_elev_parts, eyes_elev = [], []
    for elev, az in elev_cfg:
        m, e = orbit_cameras(1, elevation_deg=elev, radius=radius,
                             azimuths_deg=[az], device=device)
        mvps_elev_parts.append(m)
        eyes_elev.extend(e)

    mvps = torch.cat([mvps_eq, *mvps_elev_parts], dim=0)  # [6, 4, 4]
    eyes = eyes_eq + eyes_elev
    return mvps, eyes


# ─────────────────────────────────────────────────────────────────────────────
# Depth rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_sil_and_depth(
    ctx,
    verts_t:    torch.Tensor,   # [V, 3]
    faces_t:    torch.Tensor,   # [F, 3] int32
    mvp:        torch.Tensor,   # [4, 4]
    resolution: Tuple[int, int] = (IMG_RES, IMG_RES),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single rasterize pass → silhouette [1,H,W,1] + NDC-z depth [H,W] + fg mask [H,W].
    Depth = clip_z / clip_w ∈ [-1, 1] for visible geometry, 0 elsewhere.
    Differentiable w.r.t. verts_t via antialias (sil) and interpolate (depth).
    """
    H, W = resolution
    V = verts_t.shape[0]

    pos_clip = transform_to_clip(verts_t, mvp)   # [1, V, 4]
    rast, _  = dr.rasterize(ctx, pos_clip, faces_t, resolution=[H, W])

    # ── silhouette (differentiable boundary via antialias) ────────────────
    ones = torch.ones(1, V, 3, dtype=torch.float32, device=verts_t.device)
    color, _ = dr.interpolate(ones, rast, faces_t)
    sil = dr.antialias(color, rast, pos_clip, faces_t)[..., :1]  # [1, H, W, 1]

    # ── depth (clip z/w interpolated, differentiable) ─────────────────────
    fg_mask = rast[0, :, :, 3] > 0             # [H, W] bool, no grad

    # Attributes: clip z and clip w per vertex
    clip_zw = pos_clip[0, :, 2:4]              # [V, 2]  (z, w)
    zw_img, _ = dr.interpolate(
        clip_zw.unsqueeze(0).contiguous(), rast, faces_t
    )                                           # [1, H, W, 2]
    clip_z = zw_img[0, :, :, 0]               # [H, W]
    clip_w = zw_img[0, :, :, 1]               # [H, W]
    ndc_z  = clip_z / clip_w.clamp(min=1e-6)  # [H, W]  ∈ [-1, 1]

    return sil, ndc_z, fg_mask


def render_depth_no_grad(
    ctx,
    verts_t: torch.Tensor,
    faces_t: torch.Tensor,
    mvp:     torch.Tensor,
    resolution: Tuple[int, int] = (IMG_RES, IMG_RES),
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (ndc_z_np [H,W], fg_mask_np [H,W]) with no gradient."""
    with torch.no_grad():
        _, ndc_z, fg_mask = render_sil_and_depth(ctx, verts_t, faces_t, mvp, resolution)
    return ndc_z.cpu().numpy(), fg_mask.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou_n(pred_sil: np.ndarray, gt_uint8: np.ndarray) -> float:
    """N-view mean IoU. pred_sil [N,H,W] float 1=fg, gt_uint8 [N,H,W] uint8 0=fg."""
    pred_fg = pred_sil > 0.5
    gt_fg   = gt_uint8 < 128
    ious = []
    for v in range(pred_sil.shape[0]):
        inter = (pred_fg[v] & gt_fg[v]).sum()
        union = (pred_fg[v] | gt_fg[v]).sum()
        ious.append(float(inter) / max(float(union), 1.0))
    return float(np.mean(ious))


def compute_depth_l1_n(
    pred_depths: np.ndarray,   # [N, H, W] float
    gt_depths:   np.ndarray,   # [N, H, W] float
    gt_uint8:    np.ndarray,   # [N, H, W] uint8, 0=fg
) -> float:
    """Mean L1 depth error on GT foreground pixels, N-view average."""
    total, count = 0.0, 0
    for v in range(len(pred_depths)):
        gt_fg = gt_uint8[v] < 128
        n_fg  = gt_fg.sum()
        if n_fg == 0:
            continue
        diff   = np.abs(pred_depths[v] - gt_depths[v])
        total += float(diff[gt_fg].mean())
        count += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# N-view silhouette + depth render (no grad, for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def render_views_n(ctx, verts_t, faces_t, mvps) -> np.ndarray:
    """[N, H, W] float32 silhouette, 1=fg."""
    views = []
    with torch.no_grad():
        for i in range(mvps.shape[0]):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            views.append(sil[0, :, :, 0].cpu().numpy())
    return np.stack(views, axis=0)


def render_depths_n(ctx, verts_t, faces_t, mvps) -> np.ndarray:
    """[N, H, W] float32 NDC-z depths."""
    depths = []
    with torch.no_grad():
        for i in range(mvps.shape[0]):
            ndc_z, _ = render_depth_no_grad(ctx, verts_t, faces_t, mvps[i])
            depths.append(ndc_z)
    return np.stack(depths, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-view ray voting (N-view version of vote_faces_multiview)
# ─────────────────────────────────────────────────────────────────────────────

def vote_faces_multiview_n(
    error_maps:        np.ndarray,   # [N, H, W] binary float
    verts_np:          np.ndarray,   # [V, 3]
    tris_np:           np.ndarray,   # [F, 3]
    mvps:              torch.Tensor, # [N, 4, 4]
    max_rays_per_view: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    N-view ray voting. Returns (face_votes [F], face_dir [F, 3]).
    face_dir accumulates reversed ray directions for extrude direction estimation.
    """
    N_v     = error_maps.shape[0]
    F_count = tris_np.shape[0]
    face_votes = np.zeros(F_count, dtype=np.int32)
    face_dir   = np.zeros((F_count, 3), dtype=np.float64)

    v0 = verts_np[tris_np[:, 0]]
    v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0

    for vi in range(N_v):
        origins, dirs = error_pixels_to_rays(
            error_maps[vi], mvps[vi], max_rays=max_rays_per_view)
        if len(origins) == 0:
            continue

        hit_face, hit_t = moller_trumbore_batch(origins, dirs, v0, v1, v2)

        for ri in range(len(origins)):
            fi = hit_face[ri]
            if fi >= 0:
                face_votes[fi] += 1
                face_dir[fi]   -= dirs[ri]
            else:
                # Nearest face to ray (fallback)
                oc      = face_centers - origins[ri]
                proj    = np.sum(oc * dirs[ri], axis=1, keepdims=True)
                closest = origins[ri] + proj * dirs[ri]
                dists   = np.linalg.norm(face_centers - closest, axis=1)
                nearest = np.argmin(dists)
                face_votes[nearest] += 1
                face_dir[nearest]   -= dirs[ri]

    return face_votes, face_dir


# ─────────────────────────────────────────────────────────────────────────────
# Post-injection local edge split (for extruded side-wall edges)
# ─────────────────────────────────────────────────────────────────────────────

def split_new_long_edges(
    verts_np:    np.ndarray,  # [V, 3]
    tris_np:     np.ndarray,  # [F, 3]
    new_v_start: int,         # indices >= this are "new" (extruded)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split edges that involve at least one new vertex and are > 2× global mean
    edge length. Reduces side-wall spikes that cause silhouette artefacts.
    """
    # Global mean edge length
    all_edges: set = set()
    for f in tris_np:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            all_edges.add((min(a, b), max(a, b)))
    if not all_edges:
        return verts_np, tris_np
    edges_arr = np.array(list(all_edges), dtype=np.int32)
    lens = np.linalg.norm(
        verts_np[edges_arr[:, 0]] - verts_np[edges_arr[:, 1]], axis=1)
    mean_len = float(np.mean(lens))
    max_len  = 2.0 * mean_len

    # Filter: new and long
    is_new  = ((edges_arr[:, 0] >= new_v_start) | (edges_arr[:, 1] >= new_v_start))
    is_long = lens > max_len
    to_split_edges = edges_arr[is_new & is_long]

    if len(to_split_edges) == 0:
        return verts_np, tris_np

    new_verts = list(verts_np)
    mid_map: dict = {}
    for a, b in to_split_edges:
        a, b = int(a), int(b)
        key = (min(a, b), max(a, b))
        if key not in mid_map:
            mid_map[key] = len(new_verts)
            new_verts.append((verts_np[a] + verts_np[b]) / 2.0)

    if not mid_map:
        return verts_np, tris_np

    new_tris = []
    for f in tris_np:
        splits: dict = {}
        for i in range(3):
            key = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
            if key in mid_map:
                splits[i] = mid_map[key]
        if not splits:
            new_tris.append(list(f))
        elif len(splits) == 1:
            ei  = list(splits.keys())[0]
            mid = splits[ei]
            a, b, c = int(f[ei]), int(f[(ei + 1) % 3]), int(f[(ei + 2) % 3])
            new_tris.append([a, mid, c])
            new_tris.append([mid, b, c])
        else:
            # Multi-split — keep original to avoid winding issues
            new_tris.append(list(f))

    return (np.array(new_verts, dtype=np.float64),
            np.array(new_tris, dtype=np.int32))


# ─────────────────────────────────────────────────────────────────────────────
# Depth loss (training)
# ─────────────────────────────────────────────────────────────────────────────

def depth_loss_masked(
    pred_ndc_z:  torch.Tensor,   # [H, W] differentiable
    pred_fg:     torch.Tensor,   # [H, W] bool (no grad)
    gt_ndc_z:    torch.Tensor,   # [H, W] float (no grad)
    gt_fg:       torch.Tensor,   # [H, W] bool (no grad)
) -> torch.Tensor:
    """L1 depth on intersection of pred and GT foreground pixels."""
    mask = pred_fg & gt_fg         # [H, W] bool
    n    = int(mask.sum().item())
    if n < 4:
        return pred_ndc_z.sum() * 0.0   # zero with grad path intact
    return F.l1_loss(pred_ndc_z[mask], gt_ndc_z[mask])


# ─────────────────────────────────────────────────────────────────────────────
# Variant A — baseline (no injection, no depth)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    ctx,
    verts_init: np.ndarray,
    tris_init:  np.ndarray,
    gt_uint8:   np.ndarray,   # [N, H, W] uint8, 0=fg
    mvps:       torch.Tensor, # [N, 4, 4]
    device:     str,
    n_steps:    int = TOTAL_STEPS,
) -> Tuple[float, np.ndarray]:
    """Returns (final_iou, pred_sils [N,H,W])."""
    N_v  = mvps.shape[0]
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg).unsqueeze(-1).to(device)  # [N, H, W, 1]

    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    opt   = torch.optim.Adam([verts_t], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=3e-5)

    for step in range(n_steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(N_v):
            rendered = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                         resolution=(IMG_RES, IMG_RES))
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i + 1])
        sil_loss = sil_loss / N_v
        reg = W_LAP * laplacian_loss(verts_t, faces_t) + \
              W_EDGE * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            with torch.no_grad():
                ps = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(ps, gt_uint8)
            print(f"  [baseline] step {step:4d}/{n_steps}  "
                  f"sil={sil_loss.item():.4f}  iou={iou:.4f}")

    pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou = compute_iou_n(pred_sils, gt_uint8)
    return final_iou, pred_sils


# ─────────────────────────────────────────────────────────────────────────────
# Variants B & C — injection loop (with optional depth supervision)
# ─────────────────────────────────────────────────────────────────────────────

def run_with_injection(
    ctx,
    verts_init:   np.ndarray,
    tris_init:    np.ndarray,
    gt_uint8:     np.ndarray,    # [N, H, W] uint8, 0=fg
    gt_depths:    Optional[np.ndarray],  # [N, H, W] float NDC-z, or None
    mvps:         torch.Tensor,  # [N, 4, 4]
    device:       str,
    use_depth:    bool  = False,
    max_injections: int = MAX_INJECTIONS,
    total_steps:    int = TOTAL_STEPS,
) -> Tuple[float, float, np.ndarray, np.ndarray, List[dict]]:
    """
    Returns: (final_iou, final_depth_l1, pred_sils [N,H,W], pred_depths [N,H,W],
              injection_log)
    """
    N_v  = mvps.shape[0]
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg).unsqueeze(-1).to(device)

    # Pre-cache GT depth tensors (no grad)
    gt_depth_tensors: Optional[List[torch.Tensor]] = None
    gt_fg_mask_tensors: Optional[List[torch.Tensor]] = None
    if use_depth and gt_depths is not None:
        gt_depth_tensors = [
            torch.from_numpy(gt_depths[i]).float().to(device) for i in range(N_v)]
        gt_fg_mask_tensors = [
            torch.from_numpy(gt_uint8[i] < 128).to(device) for i in range(N_v)]

    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    LR, LR_MIN = 3e-3, 3e-5
    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    injection_log: List[dict] = []
    n_injections   = 0
    warmup_counter = 0   # lr warm-up steps remaining after injection
    lap_boost_left = 0   # enhanced laplacian steps remaining
    new_v_start    = -1  # for local edge cleanup tracking

    iou_at_step: List[Tuple[int, float]] = []

    for step in range(total_steps):

        # ── lr warm-up after injection ───────────────────────────────────
        if warmup_counter > 0:
            warmup_frac = 1.0 - (warmup_counter / WARMUP_STEPS)
            for pg in opt.param_groups:
                pg['lr'] = sched.get_last_lr()[0] * max(warmup_frac, 0.05)
            warmup_counter -= 1

        # ── forward ──────────────────────────────────────────────────────
        opt.zero_grad()
        sil_loss   = torch.tensor(0.0, device=device)
        depth_loss = torch.tensor(0.0, device=device)

        for i in range(N_v):
            if use_depth and gt_depth_tensors is not None:
                # Combined rasterize pass
                sil_i, ndc_z_i, fg_mask_i = render_sil_and_depth(
                    ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
                sil_loss   = sil_loss + F.l1_loss(sil_i, targets[i:i + 1])
                depth_loss = depth_loss + depth_loss_masked(
                    ndc_z_i, fg_mask_i,
                    gt_depth_tensors[i], gt_fg_mask_tensors[i])
            else:
                sil_i = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                          resolution=(IMG_RES, IMG_RES))
                sil_loss = sil_loss + F.l1_loss(sil_i, targets[i:i + 1])

        sil_loss   = sil_loss   / N_v
        depth_loss = depth_loss / N_v

        # Enhanced laplacian for LAP_WARMUP_STEPS after injection
        lw = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        if lap_boost_left > 0:
            lap_boost_left -= 1

        reg = lw * laplacian_loss(verts_t, faces_t) + \
              W_EDGE * edge_length_loss(verts_t, faces_t)

        total_loss = sil_loss + (W_DEPTH * depth_loss if use_depth else 0.0) + reg
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        if warmup_counter == 0:
            sched.step()

        # ── periodic IoU ──────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(pred_sils, gt_uint8)
            iou_at_step.append((step, iou))

            if step % 100 == 0:
                tag = f"sil={sil_loss.item():.4f}"
                if use_depth:
                    tag += f"  dep={depth_loss.item():.4f}"
                print(f"  [{'inj+dep' if use_depth else 'inject'}] "
                      f"step {step:4d}/{total_steps}  {tag}  iou={iou:.4f}  "
                      f"V={verts_t.shape[0]} F={faces_t.shape[0]}  "
                      f"inj={n_injections}")

        # ── plateau detection + injection ─────────────────────────────────
        if (n_injections < max_injections
                and step >= MIN_STEP_FOR_INJ
                and step <= int(total_steps * 0.75)
                and warmup_counter == 0
                and step % EVAL_INTERVAL == 0):

            n_needed = PLATEAU_STEPS // EVAL_INTERVAL + 1
            if len(iou_at_step) >= n_needed:
                recent = [h[1] for h in iou_at_step[-n_needed:]]
                improvement = recent[-1] - recent[0]

                if improvement < PLATEAU_EPS:
                    pred_sils  = render_views_n(ctx, verts_t, faces_t, mvps)
                    error_maps = compute_missing_error_maps(pred_sils, gt_uint8)
                    total_err  = (error_maps > 0.5).sum()
                    if total_err < 50:
                        continue

                    is_bn, cc_frac = detect_topology_bottleneck(error_maps)
                    if not is_bn:
                        continue

                    # ── multi-view ray voting ─────────────────────────────
                    verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
                    tris_cur  = faces_t.cpu().numpy()

                    face_votes, face_dir = vote_faces_multiview_n(
                        error_maps, verts_cur, tris_cur, mvps,
                        max_rays_per_view=300)

                    cluster = select_face_cluster(face_votes, tris_cur,
                                                  top_k=8, grow_rings=1)
                    if len(cluster) == 0:
                        continue

                    # Extrude direction
                    cluster_dir = face_dir[cluster].sum(axis=0)
                    dir_norm    = np.linalg.norm(cluster_dir)
                    if dir_norm < 1e-8:
                        normals = np.cross(
                            verts_cur[tris_cur[cluster, 1]] - verts_cur[tris_cur[cluster, 0]],
                            verts_cur[tris_cur[cluster, 2]] - verts_cur[tris_cur[cluster, 0]],
                        )
                        cluster_dir = normals.mean(axis=0)
                        dir_norm    = np.linalg.norm(cluster_dir)
                    if dir_norm < 1e-8:
                        continue
                    extrude_dir  = cluster_dir / dir_norm
                    extrude_dist = EXTRUDE_BBOX_FRAC * float(verts_cur.max() - verts_cur.min())
                    pre_iou      = iou_at_step[-1][1]

                    # ── extrude ───────────────────────────────────────────
                    old_V = len(verts_cur)
                    new_verts, new_tris = extrude_face_cluster(
                        verts_cur, tris_cur, cluster, extrude_dir, extrude_dist)

                    try:
                        assert_manifold(new_verts, new_tris,
                                        f"after inject {n_injections + 1}")
                    except AssertionError as e:
                        print(f"  [WARN] {e} — skipping injection")
                        continue

                    # ── local edge split on side-wall ─────────────────────
                    new_verts, new_tris = split_new_long_edges(
                        new_verts, new_tris, old_V)

                    # Manifold check after edge split
                    try:
                        assert_manifold(new_verts, new_tris,
                                        f"after split {n_injections + 1}")
                    except AssertionError as e:
                        print(f"  [WARN] split broke manifold: {e} — reverting split")
                        # Revert to pre-split state
                        new_verts, new_tris = extrude_face_cluster(
                            verts_cur, tris_cur, cluster, extrude_dir, extrude_dist)

                    # ── rebuild optimizer ─────────────────────────────────
                    verts_t = torch.tensor(new_verts, dtype=torch.float32,
                                           device=device).requires_grad_(True)
                    faces_t = torch.tensor(new_tris,  dtype=torch.int32, device=device)

                    opt = torch.optim.Adam([verts_t], lr=LR)
                    remaining = total_steps - step - 1
                    if remaining > 0:
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=remaining, eta_min=LR_MIN)

                    warmup_counter = WARMUP_STEPS
                    lap_boost_left = LAP_WARMUP_STEPS
                    new_v_start    = old_V

                    # Post-injection IoU
                    post_sils = render_views_n(ctx, verts_t, faces_t, mvps)
                    post_iou  = compute_iou_n(post_sils, gt_uint8)

                    n_injections += 1
                    iou_at_step.clear()
                    iou_at_step.append((step, post_iou))

                    event = {
                        'step': step, 'n_faces': len(cluster),
                        'dir': extrude_dir.tolist(), 'dist': float(extrude_dist),
                        'pre_iou': float(pre_iou), 'post_iou': float(post_iou),
                        'new_V': int(new_verts.shape[0]),
                        'new_F': int(new_tris.shape[0]),
                        'cc_frac': float(cc_frac),
                    }
                    injection_log.append(event)

                    print(f"\n  *** EXTRUDE INJECTION #{n_injections} at step {step} ***")
                    print(f"      Faces extruded : {len(cluster)}")
                    print(f"      Direction      : [{extrude_dir[0]:.3f}, "
                          f"{extrude_dir[1]:.3f}, {extrude_dir[2]:.3f}]")
                    print(f"      Distance       : {extrude_dist:.4f}")
                    print(f"      Pre-inject IoU : {pre_iou:.4f}")
                    print(f"      Post-inject IoU: {post_iou:.4f}")
                    print(f"      Mesh           : V={new_verts.shape[0]} "
                          f"F={new_tris.shape[0]}")
                    print()

    # Final evaluation
    pred_sils   = render_views_n(ctx, verts_t, faces_t, mvps)
    pred_depths = render_depths_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(pred_sils, gt_uint8)

    # Final manifold assertion
    assert_manifold(verts_t.detach().cpu().numpy(),
                    faces_t.cpu().numpy(), "final mesh")

    return final_iou, 0.0, pred_sils, pred_depths, injection_log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization grid (6 rows × 4 cols)
# ─────────────────────────────────────────────────────────────────────────────

def save_viz_grid(
    gt_uint8:     np.ndarray,    # [N, H, W] uint8 white-bg
    sils_a:       np.ndarray,    # [N, H, W] float 1=fg
    sils_b:       np.ndarray,
    sils_c:       np.ndarray,
    iou_a:        float,
    iou_b:        float,
    iou_c:        float,
    out_path:     str,
) -> None:
    """Grid: rows = N views, cols = GT | A | B | C."""
    from PIL import Image, ImageDraw

    N_v = gt_uint8.shape[0]
    H, W = gt_uint8.shape[1], gt_uint8.shape[2]
    col_names = [
        'GT',
        f'A baseline ({iou_a:.3f})',
        f'B inject ({iou_b:.3f})',
        f'C inj+depth ({iou_c:.3f})',
    ]
    n_cols = len(col_names)
    PAD, HEADER = 4, 32
    canvas_W = n_cols * (W + PAD) + PAD
    canvas_H = HEADER + N_v * (H + PAD) + PAD
    canvas   = Image.new('RGB', (canvas_W, canvas_H), 'white')
    draw     = ImageDraw.Draw(canvas)

    def sil_to_uint8(sil):
        return ((1.0 - sil) * 255.0).clip(0, 255).astype(np.uint8)

    col_imgs = [gt_uint8,
                sil_to_uint8(sils_a), sil_to_uint8(sils_b), sil_to_uint8(sils_c)]

    for ci, (cname, imgs) in enumerate(zip(col_names, col_imgs)):
        x0 = PAD + ci * (W + PAD)
        draw.text((x0, 8), cname, fill='black')
        for vi in range(N_v):
            y0 = HEADER + vi * (H + PAD)
            canvas.paste(Image.fromarray(imgs[vi], mode='L').convert('RGB'), (x0, y0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved viz -> {out_path}")


def _depth_to_rgb(depth, fg_mask, vmin, vmax):
    """Colorize a single-view NDC-z depth map. Closer=warm, far=cool, bg=white."""
    import numpy as _np
    H, W = depth.shape
    norm = (depth - vmin) / max(vmax - vmin, 1e-6)
    norm = _np.clip(norm, 0.0, 1.0)
    try:
        import matplotlib.cm as _cm
        rgb = (_cm.get_cmap('turbo')(1.0 - norm)[..., :3] * 255.0).astype(_np.uint8)
    except Exception:
        g = (255.0 * (1.0 - norm)).astype(_np.uint8)   # grayscale fallback
        rgb = _np.stack([g, g, g], axis=-1)
    rgb[~fg_mask] = 255   # white background
    return rgb


def save_depth_grid(
    gt_depths:  np.ndarray,   # [N, H, W] NDC-z (0 = background)
    depths_b:   np.ndarray,
    depths_c:   np.ndarray,
    depth_l1_b: float,
    depth_l1_c: float,
    out_path:   str,
) -> None:
    """Grid of colorized DEPTH maps: rows = N views, cols = GT | B | C.

    Shared color scale across all three so columns are directly comparable.
    """
    from PIL import Image, ImageDraw

    N_v, H, W = gt_depths.shape
    fg_gt = gt_depths != 0.0
    fg_b  = depths_b  != 0.0
    fg_c  = depths_c  != 0.0

    # Shared color range over GT foreground depths (the reference scale).
    fg_vals = gt_depths[fg_gt]
    vmin, vmax = float(np.percentile(fg_vals, 2)), float(np.percentile(fg_vals, 98))

    col_names = ['GT depth',
                 f'B inject (L1={depth_l1_b:.4f})',
                 f'C inj+depth (L1={depth_l1_c:.4f})']
    depth_cols = [(gt_depths, fg_gt), (depths_b, fg_b), (depths_c, fg_c)]

    PAD, HEADER = 4, 32
    canvas_W = 3 * (W + PAD) + PAD
    canvas_H = HEADER + N_v * (H + PAD) + PAD
    canvas   = Image.new('RGB', (canvas_W, canvas_H), 'white')
    draw     = ImageDraw.Draw(canvas)

    for ci, (cname, (dep, fg)) in enumerate(zip(col_names, depth_cols)):
        x0 = PAD + ci * (W + PAD)
        draw.text((x0, 8), cname, fill='black')
        for vi in range(N_v):
            y0 = HEADER + vi * (H + PAD)
            rgb = _depth_to_rgb(dep[vi], fg[vi], vmin, vmax)
            canvas.paste(Image.fromarray(rgb, mode='RGB'), (x0, y0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved depth viz -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrude injection v2 PoC")
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--out_dir', default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    ctx          = dr.RasterizeCudaContext()
    mvps, eyes   = make_6_cameras(radius=CAMERA_RADIUS, device=device)
    print(f"6-camera rig: {mvps.shape}  device={device}  res={IMG_RES}×{IMG_RES}")

    # ── Load GT shape ──────────────────────────────────────────────────────
    if os.path.exists(BUNNY_PATH):
        print(f"Loading bunny from {BUNNY_PATH}")
        gt_verts, gt_tris = load_obj(BUNNY_PATH)
        shape_name = 'bunny'
    else:
        print("WARNING: bunny not found — using torus fallback")
        gt_verts, gt_tris = make_torus()
        shape_name = 'torus'

    gt_verts = normalize_to_range(gt_verts)
    print(f"  GT mesh: V={len(gt_verts)} F={len(gt_tris)}")

    # ── Render GT silhouettes + depths ─────────────────────────────────────
    verts_gt_t = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    faces_gt_t = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)

    gt_sil_views:   List[np.ndarray] = []
    gt_depth_views: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, verts_gt_t, faces_gt_t, mvps[i], (IMG_RES, IMG_RES))
            gt_sil_views.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255.0)
                .clip(0, 255).astype(np.uint8))
            gt_depth_views.append(ndc_z_i.cpu().numpy())

    gt_uint8  = np.stack(gt_sil_views,   axis=0)   # [6, H, W] uint8, 0=fg
    gt_depths = np.stack(gt_depth_views, axis=0)   # [6, H, W] float

    print("  GT silhouettes + depths rendered")

    # ── Initial mesh: 2× subdivided icosphere ─────────────────────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    ico = catmull_clark(ico)
    ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    print(f"  Init mesh: V={len(init_verts)} F={len(init_tris)}")

    # ══════════════════════════════════════════════════════════════════════
    # Variant A — baseline (no injection, no depth)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"VARIANT A — Baseline {TOTAL_STEPS}-step opt, no injection, no depth")
    print("=" * 70)
    t0 = time.time()
    iou_a, sils_a = run_baseline(
        ctx, init_verts, init_tris, gt_uint8, mvps, device, n_steps=TOTAL_STEPS)
    ta = time.time() - t0
    depths_a = render_depths_n(
        ctx,
        torch.tensor(init_verts, dtype=torch.float32, device=device),  # placeholder
        torch.tensor(init_tris,  dtype=torch.int32,   device=device),
        mvps)
    # Recompute from final sils (we don't have verts after run_baseline — acceptable;
    # just report IoU for A; depth_l1 computed from sils_a heuristically)
    # NOTE: run_baseline doesn't expose final verts. We compute depth_l1=N/A for A.
    print(f"  [A] IoU={iou_a:.4f}  time={ta:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Variant B — injection, no depth
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"VARIANT B — Injection (max {MAX_INJECTIONS}), no depth, {TOTAL_STEPS} steps")
    print("=" * 70)
    t0 = time.time()
    iou_b, _, sils_b, depths_b, log_b = run_with_injection(
        ctx, init_verts, init_tris,
        gt_uint8, None, mvps, device,
        use_depth=False, max_injections=MAX_INJECTIONS, total_steps=TOTAL_STEPS)
    tb = time.time() - t0
    depth_l1_b = compute_depth_l1_n(depths_b, gt_depths, gt_uint8)
    print(f"  [B] IoU={iou_b:.4f}  depth_L1={depth_l1_b:.4f}  time={tb:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Variant C — injection + depth
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"VARIANT C — Injection (max {MAX_INJECTIONS}) + depth supervision, "
          f"{TOTAL_STEPS} steps")
    print("=" * 70)
    t0 = time.time()
    iou_c, _, sils_c, depths_c, log_c = run_with_injection(
        ctx, init_verts, init_tris,
        gt_uint8, gt_depths, mvps, device,
        use_depth=True, max_injections=MAX_INJECTIONS, total_steps=TOTAL_STEPS)
    tc = time.time() - t0
    depth_l1_c = compute_depth_l1_n(depths_c, gt_depths, gt_uint8)
    print(f"  [C] IoU={iou_c:.4f}  depth_L1={depth_l1_c:.4f}  time={tc:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  A (baseline no-inj no-depth) : IoU={iou_a:.4f}   depth_L1=N/A")
    print(f"  B (inject no-depth)          : IoU={iou_b:.4f}   depth_L1={depth_l1_b:.4f}")
    print(f"  C (inject + depth)           : IoU={iou_c:.4f}   depth_L1={depth_l1_c:.4f}")
    print(f"  B vs A                       : {iou_b - iou_a:+.4f}")
    print(f"  C vs A                       : {iou_c - iou_a:+.4f}")
    print(f"  C vs B (depth gain)          : {iou_c - iou_b:+.4f}  "
          f"depth_L1 delta={depth_l1_c - depth_l1_b:+.4f}")
    print(f"  Total time: {ta+tb+tc:.1f}s "
          f"(A={ta:.1f}s  B={tb:.1f}s  C={tc:.1f}s)")

    for tag, log in [('B', log_b), ('C', log_c)]:
        if not log:
            print(f"\n  [{tag}] No injections triggered.")
        else:
            print(f"\n  [{tag}] {len(log)} injection(s):")
            for k, ev in enumerate(log):
                d = ev['dir']
                print(f"    #{k+1} step={ev['step']}  faces={ev['n_faces']}  "
                      f"dir=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f}]  "
                      f"dist={ev['dist']:.4f}  "
                      f"IoU: {ev['pre_iou']:.4f}→{ev['post_iou']:.4f}  "
                      f"V={ev['new_V']} F={ev['new_F']}  cc={ev['cc_frac']:.2f}")

    target = 0.96
    if iou_c >= target:
        print(f"\n  TARGET MET: C IoU {iou_c:.4f} ≥ {target}")
    else:
        print(f"\n  Target {target} not met (C got {iou_c:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════════════
    viz_path = os.path.join(args.out_dir, f'viz_extrude_v2_{shape_name}.png')
    save_viz_grid(gt_uint8, sils_a, sils_b, sils_c,
                  iou_a, iou_b, iou_c, viz_path)

    depth_viz_path = os.path.join(args.out_dir, f'viz_extrude_v2_{shape_name}_depth.png')
    save_depth_grid(gt_depths, depths_b, depths_c,
                    depth_l1_b, depth_l1_c, depth_viz_path)

    print(f"\nDone. Total={ta+tb+tc:.1f}s")


if __name__ == '__main__':
    main()
