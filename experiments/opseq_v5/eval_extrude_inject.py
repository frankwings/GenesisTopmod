#!/usr/bin/env python3
"""
eval_extrude_inject.py — Error-driven topology operator injection PoC.

When vertex optimization plateaus on a silhouette fitting task, this script:
1. Detects topology bottlenecks via silhouette error concentration analysis
2. Identifies responsible mesh face cluster via multi-view ray voting
3. Extrudes selected cluster in the error-guided direction (manifold-safe)
4. Continues optimizing with enriched topology

Demonstrated on Stanford Bunny — grows ear structures that pure vertex
optimization with Laplacian regularization cannot produce.

Usage:
    python eval_extrude_inject.py [--device cuda] [--out_dir eval_out]
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

from pipeline.cameras import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5 import adaptive_remesh, compute_iou
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0

# Injection hyper-parameters
MAX_INJECTIONS     = 3
TOTAL_STEPS        = 500
EVAL_INTERVAL      = 5     # evaluate IoU every N steps
PLATEAU_STEPS      = 20    # 20 consecutive steps with < eps improvement
PLATEAU_EPS        = 0.005 # detect slowdown (spec says 0.001; tuned to 0.005 for earlier
                           # injection timing — with 0.001, first inject is ~step 345,
                           # leaving too few steps for 3-round injection within 500 total)
ERROR_CC_THRESH    = 0.40  # largest CC > 40% of error pixels
WARMUP_STEPS       = 20
EXTRUDE_BBOX_FRAC  = 0.12  # extrude distance = 0.12 * bbox (spec says 0.3; tuned to 0.12
                           # for faster post-injection recovery in multi-round setting)
MIN_STEP_FOR_INJ   = 60    # don't inject in the first 60 steps


# ═════════════════════════════════════════════════════════════════════════════
# Manifold assertion
# ═════════════════════════════════════════════════════════════════════════════

def check_manifold(verts: np.ndarray, tris: np.ndarray) -> Tuple[int, int]:
    """
    Returns (n_boundary_edges, n_nonmanifold_edges).
    Boundary = shared by 1 face; non-manifold = shared by >2 faces.
    Watertight manifold has both == 0.
    """
    edge_count: Dict[Tuple[int, int], int] = {}
    for f in tris:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
    n_boundary = sum(1 for c in edge_count.values() if c == 1)
    n_nonmanifold = sum(1 for c in edge_count.values() if c > 2)
    return n_boundary, n_nonmanifold


def assert_manifold(verts: np.ndarray, tris: np.ndarray, label: str = "") -> None:
    """Assert no non-manifold edges (count > 2). Boundary edges are warned."""
    n_boundary, n_nonmanifold = check_manifold(verts, tris)
    tag = f" {label}" if label else ""
    if n_nonmanifold > 0:
        raise AssertionError(
            f"[manifold{tag}] {n_nonmanifold} non-manifold edges (count > 2)")
    if n_boundary > 0:
        print(f"  [manifold WARN{tag}] {n_boundary} boundary edges "
              f"(V={len(verts)} F={len(tris)})")
    else:
        print(f"  [manifold OK{tag}] V={len(verts)} F={len(tris)}")


# ═════════════════════════════════════════════════════════════════════════════
# Silhouette error analysis
# ═════════════════════════════════════════════════════════════════════════════

def render_views(ctx, verts_t, faces_t, mvps) -> np.ndarray:
    """Render 4 views, return [4, H, W] numpy float32 (1=fg)."""
    views = []
    with torch.no_grad():
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            views.append(sil[0, :, :, 0].cpu().numpy())
    return np.stack(views, axis=0)


def compute_missing_error_maps(
    pred_sils: np.ndarray,   # [4, H, W] float32, 1=fg
    gt_uint8:  np.ndarray,   # [4, H, W] uint8, 0=fg 255=bg
) -> np.ndarray:
    """
    Missing region = GT foreground minus rendered foreground (positive part).
    Returns [4, H, W] binary maps.
    """
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    pred_fg = (pred_sils > 0.5).astype(np.float32)
    return np.clip(gt_fg - pred_fg, 0, 1)


def largest_cc_fraction(binary_map: np.ndarray) -> float:
    """Fraction of True pixels in the largest 8-connected component."""
    H, W = binary_map.shape
    total = int(binary_map.sum())
    if total == 0:
        return 0.0
    visited = np.zeros((H, W), dtype=bool)
    max_size = 0
    for sy in range(H):
        for sx in range(W):
            if binary_map[sy, sx] > 0.5 and not visited[sy, sx]:
                queue = [(sy, sx)]
                visited[sy, sx] = True
                size = 0
                head = 0
                while head < len(queue):
                    y, x = queue[head]; head += 1
                    size += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if binary_map[ny, nx] > 0.5 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                max_size = max(max_size, size)
    return max_size / total


def detect_topology_bottleneck(error_maps: np.ndarray) -> Tuple[bool, float]:
    """
    Returns (is_bottleneck, max_cc_fraction) based on error concentration.
    Bottleneck if any view has largest CC > 40% of error pixels.
    """
    max_frac = 0.0
    for i in range(error_maps.shape[0]):
        total_err = (error_maps[i] > 0.5).sum()
        if total_err < 20:
            continue
        frac = largest_cc_fraction(error_maps[i])
        max_frac = max(max_frac, frac)
    return max_frac > ERROR_CC_THRESH, max_frac


# ═════════════════════════════════════════════════════════════════════════════
# Multi-view ray unprojection + Moller-Trumbore intersection
# ═════════════════════════════════════════════════════════════════════════════

def error_pixels_to_rays(
    error_map: np.ndarray,    # [H, W] binary
    mvp:       torch.Tensor,  # [4, 4]
    max_rays:  int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject error pixels to world-space rays via inverse MVP.
    Returns (origins [R,3], directions [R,3]).
    """
    H, W = error_map.shape
    ys, xs = np.where(error_map > 0.5)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # Subsample if too many
    if len(ys) > max_rays:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(ys), max_rays, replace=False)
        ys, xs = ys[idx], xs[idx]

    # Pixel center to NDC
    ndc_x = (xs + 0.5) / W * 2.0 - 1.0
    ndc_y = 1.0 - (ys + 0.5) / H * 2.0  # flip y for OpenGL

    mvp_np = mvp.cpu().numpy().astype(np.float64)
    inv_mvp = np.linalg.inv(mvp_np)

    def unproject(z_ndc):
        pts = np.stack([ndc_x, ndc_y,
                        np.full_like(ndc_x, z_ndc),
                        np.ones_like(ndc_x)], axis=1)  # [R,4]
        world = (inv_mvp @ pts.T).T  # [R,4]
        w = world[:, 3:4]
        return world[:, :3] / np.where(np.abs(w) > 1e-8, w, 1e-8)

    near_pts = unproject(-1.0)
    far_pts = unproject(+1.0)
    dirs = far_pts - near_pts
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(norms > 1e-8, norms, 1e-8)

    return near_pts, dirs


def moller_trumbore_batch(
    origins: np.ndarray,   # [R, 3]
    dirs:    np.ndarray,   # [R, 3]
    v0: np.ndarray,        # [F, 3]
    v1: np.ndarray,        # [F, 3]
    v2: np.ndarray,        # [F, 3]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-ray: find closest intersecting face.
    Returns:
        hit_face : [R] int (-1 if no hit)
        hit_t    : [R] float (distance, inf if no hit)
    """
    EPS = 1e-8
    R = len(origins)
    F = len(v0)
    hit_face = np.full(R, -1, dtype=np.int32)
    hit_t = np.full(R, np.inf, dtype=np.float64)

    e1 = v1 - v0  # [F, 3]
    e2 = v2 - v0

    # Process in chunks to manage memory: [R, F] is R*F which can be large
    CHUNK = 128
    for r_start in range(0, R, CHUNK):
        r_end = min(r_start + CHUNK, R)
        ro = origins[r_start:r_end]  # [r, 3]
        rd = dirs[r_start:r_end]     # [r, 3]
        r = len(ro)

        # h = cross(rd, e2) : [r, F, 3]
        h = np.cross(rd[:, None, :], e2[None, :, :])
        # a = dot(e1, h) : [r, F]
        a = np.einsum('fk,rfk->rf', e1, h)
        valid = np.abs(a) > EPS
        inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

        s = ro[:, None, :] - v0[None, :, :]  # [r, F, 3]
        u = inv_a * np.einsum('rfk,rfk->rf', s, h)
        valid = valid & (u >= 0) & (u <= 1.0)

        q = np.cross(s, e1[None, :, :])  # [r, F, 3]
        v = inv_a * np.einsum('rfk,rfk->rf', rd[:, None, :].repeat(F, axis=1).reshape(r, F, 3), q)
        valid = valid & (v >= 0) & ((u + v) <= 1.0)

        t = inv_a * np.einsum('fk,rfk->rf', e2, q)
        valid = valid & (t > EPS)

        # For each ray, find closest hit
        t_masked = np.where(valid, t, np.inf)
        best_f = np.argmin(t_masked, axis=1)  # [r]
        best_t = t_masked[np.arange(r), best_f]  # [r]
        has_hit = best_t < np.inf

        for ri in range(r):
            if has_hit[ri] and best_t[ri] < hit_t[r_start + ri]:
                hit_face[r_start + ri] = best_f[ri]
                hit_t[r_start + ri] = best_t[ri]

    return hit_face, hit_t


def vote_faces_multiview(
    error_maps: np.ndarray,   # [4, H, W] binary
    verts_np:   np.ndarray,   # [V, 3]
    tris_np:    np.ndarray,   # [F, 3]
    mvps:       torch.Tensor, # [4, 4, 4]
    eyes:       list,
    max_rays_per_view: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cast rays from error pixels in each view, vote on mesh faces.

    Returns:
        face_votes  : [F] int — ray hit count per face
        face_dir    : [F, 3] float — accumulated reverse ray direction per face
    """
    F_count = tris_np.shape[0]
    face_votes = np.zeros(F_count, dtype=np.int32)
    face_dir = np.zeros((F_count, 3), dtype=np.float64)

    v0 = verts_np[tris_np[:, 0]]
    v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]

    # Face centroids for nearest-face fallback
    face_centers = (v0 + v1 + v2) / 3.0

    for vi in range(4):
        origins, dirs = error_pixels_to_rays(error_maps[vi], mvps[vi],
                                              max_rays=max_rays_per_view)
        if len(origins) == 0:
            continue

        hit_face, hit_t = moller_trumbore_batch(origins, dirs, v0, v1, v2)

        for ri in range(len(origins)):
            fi = hit_face[ri]
            if fi >= 0:
                face_votes[fi] += 1
                face_dir[fi] -= dirs[ri]  # reverse ray direction
            else:
                # No hit — vote for nearest face to ray
                oc = face_centers - origins[ri]
                proj = np.sum(oc * dirs[ri], axis=1, keepdims=True)
                closest = origins[ri] + proj * dirs[ri]
                dists = np.linalg.norm(face_centers - closest, axis=1)
                nearest = np.argmin(dists)
                face_votes[nearest] += 1
                face_dir[nearest] -= dirs[ri]

    return face_votes, face_dir


def select_face_cluster(
    face_votes: np.ndarray,  # [F]
    tris_np:    np.ndarray,  # [F, 3]
    top_k:      int = 8,
    grow_rings: int = 1,
) -> np.ndarray:
    """Select cluster: top_k voted faces + grow by adjacency rings."""
    F_count = tris_np.shape[0]
    if face_votes.max() == 0:
        return np.array([], dtype=np.int32)

    # Build face adjacency via shared edges
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(F_count):
        for i in range(3):
            a, b = int(tris_np[fi, i]), int(tris_np[fi, (i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(fi)

    face_neighbors: Dict[int, Set[int]] = {fi: set() for fi in range(F_count)}
    for flist in edge_to_faces.values():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                face_neighbors[flist[i]].add(flist[j])
                face_neighbors[flist[j]].add(flist[i])

    # Start with top_k highest-voted faces
    sorted_faces = np.argsort(face_votes)
    seed_faces = set()
    for fi in reversed(sorted_faces):
        if face_votes[fi] == 0:
            break
        seed_faces.add(fi)
        if len(seed_faces) >= top_k:
            break

    # Grow by adjacency rings
    cluster = set(seed_faces)
    for _ in range(grow_rings):
        frontier = set()
        for fi in cluster:
            frontier |= face_neighbors.get(fi, set())
        cluster |= frontier

    return np.array(sorted(cluster), dtype=np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# Manifold-preserving face cluster extrusion
# ═════════════════════════════════════════════════════════════════════════════

def _find_pinch_vertices(cluster_faces: np.ndarray, tris_np: np.ndarray,
                          edge_to_faces: Dict) -> Set[int]:
    """
    Find 'pinch' vertices — boundary vertices where the cluster boundary
    is not a simple loop (vertex appears in multiple disconnected boundary
    segments). These cause non-manifold edges on extrusion.
    """
    cluster_set = set(cluster_faces.tolist())

    # Find boundary edges
    boundary_edges = []
    for edge, flist in edge_to_faces.items():
        has_in = any(fi in cluster_set for fi in flist)
        has_out = any(fi not in cluster_set for fi in flist)
        if has_in and has_out:
            boundary_edges.append(edge)

    # Count how many boundary edges each vertex participates in
    vert_boundary_count: Dict[int, int] = {}
    for a, b in boundary_edges:
        vert_boundary_count[a] = vert_boundary_count.get(a, 0) + 1
        vert_boundary_count[b] = vert_boundary_count.get(b, 0) + 1

    # A vertex on a simple boundary loop has exactly 2 boundary edges.
    # More than 2 = pinch vertex.
    pinch = set()
    for v, cnt in vert_boundary_count.items():
        if cnt > 2:
            pinch.add(v)
    return pinch


def _clean_cluster(cluster_faces: np.ndarray, tris_np: np.ndarray,
                    edge_to_faces: Dict) -> np.ndarray:
    """Remove faces adjacent to pinch vertices to ensure clean boundary."""
    max_iter = 5
    cluster = set(cluster_faces.tolist())

    for _ in range(max_iter):
        pinch = _find_pinch_vertices(np.array(sorted(cluster), dtype=np.int32),
                                      tris_np, edge_to_faces)
        if not pinch:
            break
        # Remove faces that touch pinch vertices
        to_remove = set()
        for fi in cluster:
            f = tris_np[fi]
            if any(int(f[i]) in pinch for i in range(3)):
                to_remove.add(fi)
        if not to_remove:
            break
        cluster -= to_remove

    return np.array(sorted(cluster), dtype=np.int32)


def extrude_face_cluster(
    verts_np:      np.ndarray,  # [V, 3] float64
    tris_np:       np.ndarray,  # [F, 3] int32
    cluster_faces: np.ndarray,  # face indices to extrude
    extrude_dir:   np.ndarray,  # [3] unit vector
    distance:      float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrude a cluster of faces along extrude_dir*distance, manifold-preserving.

    Strategy:
    - Clean cluster by removing faces around pinch vertices
    - Boundary vertices (shared by cluster and non-cluster) are duplicated
    - Interior vertices are offset in-place
    - Side-wall triangles connect boundary to offset copies
    """
    F_total = tris_np.shape[0]

    # Build edge-to-faces map
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(F_total):
        for i in range(3):
            a, b = int(tris_np[fi, i]), int(tris_np[fi, (i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(fi)

    # Clean cluster to remove pinch vertex issues
    cluster_faces = _clean_cluster(cluster_faces, tris_np, edge_to_faces)
    if len(cluster_faces) == 0:
        return verts_np, tris_np

    cluster_set = set(cluster_faces.tolist())

    # Boundary edges: shared between cluster and non-cluster
    boundary_edges = []
    for edge, flist in edge_to_faces.items():
        has_in = any(fi in cluster_set for fi in flist)
        has_out = any(fi not in cluster_set for fi in flist)
        if has_in and has_out:
            boundary_edges.append(edge)

    if not boundary_edges:
        return verts_np, tris_np

    # Vertex classification
    boundary_verts = set()
    for a, b in boundary_edges:
        boundary_verts.add(a)
        boundary_verts.add(b)

    cluster_verts = set()
    for fi in cluster_faces:
        for i in range(3):
            cluster_verts.add(int(tris_np[fi, i]))
    interior_verts = cluster_verts - boundary_verts

    # Create offset copies of boundary vertices
    new_verts = list(verts_np)
    offset = extrude_dir * distance
    bv_copy_map: Dict[int, int] = {}
    for vid in sorted(boundary_verts):
        new_vid = len(new_verts)
        bv_copy_map[vid] = new_vid
        new_verts.append(verts_np[vid] + offset)

    # Move interior vertices
    for vid in interior_verts:
        new_verts[vid] = verts_np[vid] + offset

    # Remap cluster faces: boundary verts -> their offset copies
    new_tris = []
    for fi in range(F_total):
        f = list(tris_np[fi])
        if fi in cluster_set:
            for i in range(3):
                if f[i] in bv_copy_map:
                    f[i] = bv_copy_map[f[i]]
        new_tris.append(f)

    # Side-wall triangles for each boundary edge
    for a, b in boundary_edges:
        a_top = bv_copy_map[a]
        b_top = bv_copy_map[b]

        # Find cluster face edge traversal order for correct winding
        edge_order = None
        for fi in edge_to_faces.get((min(a, b), max(a, b)), []):
            if fi in cluster_set:
                f = tris_np[fi]
                for i in range(3):
                    vi, vj = int(f[i]), int(f[(i + 1) % 3])
                    if vi == a and vj == b:
                        edge_order = (a, b); break
                    elif vi == b and vj == a:
                        edge_order = (b, a); break
                if edge_order is not None:
                    break

        if edge_order is None:
            new_tris.append([a, b, b_top])
            new_tris.append([a, b_top, a_top])
        else:
            ea, eb = edge_order
            # Side wall: reverse cluster edge direction for outward-facing normal
            new_tris.append([eb, ea, bv_copy_map[ea]])
            new_tris.append([eb, bv_copy_map[ea], bv_copy_map[eb]])

    return np.array(new_verts, dtype=np.float64), np.array(new_tris, dtype=np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# Optimization loops
# ═════════════════════════════════════════════════════════════════════════════

def run_baseline(
    ctx, verts_init, tris_init, gt_uint8, mvps, device, n_steps=TOTAL_STEPS,
) -> Tuple[float, np.ndarray]:
    """Standard 500-step optimization, no injection. Returns (iou, sils)."""
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg).unsqueeze(-1).to(device)

    verts_np, tris_np = adaptive_remesh(verts_init.copy().astype(np.float64),
                                         tris_init.copy())
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)

    opt = torch.optim.Adam([verts_t], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=3e-5)

    for step in range(n_steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(4):
            rendered = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                         resolution=(IMG_RES, IMG_RES))
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i + 1])
        sil_loss = sil_loss / 4
        reg = 0.1 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            pred_sil = render_views(ctx, verts_t, faces_t, mvps)
            iou = compute_iou(pred_sil, gt_uint8)
            print(f"  [baseline] step {step:4d}/{n_steps}  "
                  f"sil={sil_loss.item():.4f}  iou={iou:.4f}")

    pred_sils = render_views(ctx, verts_t, faces_t, mvps)
    final_iou = compute_iou(pred_sils, gt_uint8)
    return final_iou, pred_sils


def run_with_injection(
    ctx, verts_init, tris_init, gt_uint8, mvps, eyes, device,
    max_injections=MAX_INJECTIONS, total_steps=TOTAL_STEPS,
) -> Tuple[float, np.ndarray, List[dict]]:
    """
    Optimization with error-driven extrude injection.
    Returns (final_iou, final_sils, injection_log).
    """
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg).unsqueeze(-1).to(device)

    verts_np, tris_np = adaptive_remesh(verts_init.copy().astype(np.float64),
                                         tris_init.copy())
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)

    LR = 3e-3
    LR_MIN = 3e-5
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR_MIN)

    injection_log: List[dict] = []
    n_injections = 0
    warmup_counter = 0

    # IoU tracking for plateau detection
    iou_at_step: List[Tuple[int, float]] = []  # (step, iou)

    for step in range(total_steps):
        # ── Warm-up lr after injection ────────────────────────────────
        if warmup_counter > 0:
            warmup_frac = 1.0 - (warmup_counter / WARMUP_STEPS)
            for pg in opt.param_groups:
                pg['lr'] = sched.get_last_lr()[0] * max(warmup_frac, 0.05)
            warmup_counter -= 1

        # ── Forward pass ──────────────────────────────────────────────
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(4):
            rendered = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                         resolution=(IMG_RES, IMG_RES))
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i + 1])
        sil_loss = sil_loss / 4
        reg = 0.1 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        if warmup_counter == 0:
            sched.step()

        # ── Periodic IoU evaluation ───────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            pred_sils = render_views(ctx, verts_t, faces_t, mvps)
            iou = compute_iou(pred_sils, gt_uint8)
            iou_at_step.append((step, iou))

            if step % 50 == 0:
                print(f"  [inject] step {step:4d}/{total_steps}  "
                      f"sil={sil_loss.item():.4f}  iou={iou:.4f}  "
                      f"V={verts_t.shape[0]} F={faces_t.shape[0]}  "
                      f"inj={n_injections}")

        # ── Plateau detection + injection trigger ─────────────────────
        if (n_injections < max_injections
                and step >= MIN_STEP_FOR_INJ
                and step <= int(total_steps * 0.75)  # need recovery budget after inject
                and warmup_counter == 0
                and step % EVAL_INTERVAL == 0):

            # Need enough history: at least plateau_steps / eval_interval + 1 entries
            n_entries_needed = PLATEAU_STEPS // EVAL_INTERVAL + 1
            if len(iou_at_step) >= n_entries_needed:
                recent = [h[1] for h in iou_at_step[-n_entries_needed:]]
                improvement = recent[-1] - recent[0]

                if improvement < PLATEAU_EPS:
                    # Plateau detected — compute error maps
                    pred_sils = render_views(ctx, verts_t, faces_t, mvps)
                    error_maps = compute_missing_error_maps(pred_sils, gt_uint8)
                    total_err = (error_maps > 0.5).sum()

                    if total_err < 50:
                        continue  # too little error

                    is_bottleneck, cc_frac = detect_topology_bottleneck(error_maps)
                    if not is_bottleneck:
                        continue

                    # ── Multi-view ray voting ─────────────────────────
                    verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
                    tris_cur = faces_t.cpu().numpy()

                    face_votes, face_dir = vote_faces_multiview(
                        error_maps, verts_cur, tris_cur, mvps, eyes,
                        max_rays_per_view=300,
                    )

                    cluster = select_face_cluster(face_votes, tris_cur,
                                                   top_k=8, grow_rings=1)
                    if len(cluster) == 0:
                        continue

                    # Compute extrude direction
                    cluster_dir = face_dir[cluster].sum(axis=0)
                    dir_norm = np.linalg.norm(cluster_dir)
                    if dir_norm < 1e-8:
                        # Fallback: mean face normal of cluster
                        normals = np.cross(
                            verts_cur[tris_cur[cluster, 1]] - verts_cur[tris_cur[cluster, 0]],
                            verts_cur[tris_cur[cluster, 2]] - verts_cur[tris_cur[cluster, 0]],
                        )
                        cluster_dir = normals.mean(axis=0)
                        dir_norm = np.linalg.norm(cluster_dir)
                    if dir_norm < 1e-8:
                        continue
                    extrude_dir = cluster_dir / dir_norm

                    # Extrude distance
                    bbox = float(verts_cur.max() - verts_cur.min())
                    extrude_dist = EXTRUDE_BBOX_FRAC * bbox

                    pre_iou = iou_at_step[-1][1]

                    # ── Perform extrusion ──────────────────────────────
                    new_verts, new_tris = extrude_face_cluster(
                        verts_cur, tris_cur, cluster, extrude_dir, extrude_dist,
                    )

                    # Manifold check
                    try:
                        assert_manifold(new_verts, new_tris,
                                        f"after inject {n_injections + 1}")
                    except AssertionError as e:
                        print(f"  [WARN] {e} — skipping injection")
                        continue

                    # Rebuild optimizer
                    verts_t = torch.tensor(new_verts, dtype=torch.float32,
                                           device=device).requires_grad_(True)
                    faces_t = torch.tensor(new_tris, dtype=torch.int32, device=device)

                    opt = torch.optim.Adam([verts_t], lr=LR)
                    remaining = total_steps - step - 1
                    if remaining > 0:
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=remaining, eta_min=LR_MIN)
                    warmup_counter = WARMUP_STEPS

                    # Post-injection IoU
                    post_sils = render_views(ctx, verts_t, faces_t, mvps)
                    post_iou = compute_iou(post_sils, gt_uint8)

                    n_injections += 1
                    iou_at_step.clear()  # reset plateau detector
                    iou_at_step.append((step, post_iou))

                    event = {
                        'step': step,
                        'n_faces': len(cluster),
                        'dir': extrude_dir.tolist(),
                        'dist': float(extrude_dist),
                        'pre_iou': float(pre_iou),
                        'post_iou': float(post_iou),
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
                    print(f"      Mesh: V={new_verts.shape[0]} F={new_tris.shape[0]}")
                    print()

    # Final evaluation
    final_sils = render_views(ctx, verts_t, faces_t, mvps)
    final_iou = compute_iou(final_sils, gt_uint8)

    # Final manifold assertion
    verts_final = verts_t.detach().cpu().numpy()
    tris_final = faces_t.cpu().numpy()
    assert_manifold(verts_final, tris_final, "final mesh")

    return final_iou, final_sils, injection_log


# ═════════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════════

def save_viz_grid(
    gt_uint8:      np.ndarray,
    baseline_sils: np.ndarray,
    inject_sils:   np.ndarray,
    iou_baseline:  float,
    iou_inject:    float,
    out_path:      str,
) -> None:
    """Grid: cols = GT | baseline-500 | extrude-inject-500, rows = 4 views."""
    from PIL import Image, ImageDraw

    col_names = ['GT', f'baseline ({iou_baseline:.3f})', f'inject ({iou_inject:.3f})']
    PAD, HEADER = 4, 28
    W = len(col_names) * (IMG_RES + PAD) + PAD
    H = HEADER + 4 * (IMG_RES + PAD) + PAD
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)

    col_imgs = [
        gt_uint8,
        ((1.0 - baseline_sils) * 255.0).clip(0, 255).astype(np.uint8),
        ((1.0 - inject_sils) * 255.0).clip(0, 255).astype(np.uint8),
    ]
    for ci, (cname, imgs) in enumerate(zip(col_names, col_imgs)):
        x0 = PAD + ci * (IMG_RES + PAD)
        draw.text((x0, 6), cname, fill='black')
        for vi in range(4):
            y0 = HEADER + vi * (IMG_RES + PAD)
            canvas.paste(Image.fromarray(imgs[vi], mode='L').convert('RGB'), (x0, y0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved viz -> {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Error-driven extrude injection PoC")
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--out_dir', default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    ctx = dr.RasterizeCudaContext()
    mvps, eyes = orbit_cameras(4, elevation_deg=0.0, radius=CAMERA_RADIUS,
                                azimuths_deg=AZIMUTHS, device=device)

    # ── Load shape ────────────────────────────────────────────────────
    if os.path.exists(BUNNY_PATH):
        print(f"Loading bunny from {BUNNY_PATH}")
        verts_np, tris_np = load_obj(BUNNY_PATH)
        shape_name = 'bunny'
    else:
        print(f"WARNING: bunny not found, using torus fallback")
        verts_np, tris_np = make_torus()
        shape_name = 'torus'

    verts_np = normalize_to_range(verts_np)
    print(f"  GT mesh: V={len(verts_np)} F={len(tris_np)}")

    # ── Render GT silhouettes ─────────────────────────────────────────
    verts_gt = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces_gt = torch.tensor(tris_np, dtype=torch.int32, device=device)
    gt_views = []
    with torch.no_grad():
        for i in range(4):
            sil = render_silhouette(ctx, verts_gt, faces_gt, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            sil_np = sil[0, :, :, 0].cpu().numpy()
            gt_views.append(((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8))
    gt_uint8 = np.stack(gt_views, axis=0)
    print("  GT silhouettes rendered")

    # ── Initial mesh: subdivided icosahedron (genus-0) ────────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    ico = catmull_clark(ico)
    ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    sc = 2.0 / max(mx - mn, 1e-6)
    init_verts = (init_verts - (mn + mx) / 2.0) * sc
    init_tris = np.array(_fan_triangulate(fcs), dtype=np.int32)
    print(f"  Init mesh: V={len(init_verts)} F={len(init_tris)}")

    # ── Baseline ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BASELINE: Standard vertex optimization (500 steps, no injection)")
    print("=" * 65)
    t0 = time.time()
    iou_baseline, baseline_sils = run_baseline(
        ctx, init_verts, init_tris, gt_uint8, mvps, device)
    t_bl = time.time() - t0
    print(f"  Final IoU (baseline): {iou_baseline:.4f}  ({t_bl:.1f}s)")

    # ── With injection ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXTRUDE INJECTION (max 3 injections, 500 total steps)")
    print("=" * 65)
    t0 = time.time()
    iou_inject, inject_sils, inj_log = run_with_injection(
        ctx, init_verts, init_tris, gt_uint8, mvps, eyes, device)
    t_inj = time.time() - t0
    print(f"  Final IoU (inject): {iou_inject:.4f}  ({t_inj:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Baseline IoU  : {iou_baseline:.4f}")
    print(f"  Inject IoU    : {iou_inject:.4f}")
    print(f"  Delta         : {iou_inject - iou_baseline:+.4f}")
    print(f"  Injections    : {len(inj_log)}")
    for i, ev in enumerate(inj_log):
        print(f"    #{i+1} step={ev['step']}  faces={ev['n_faces']}  "
              f"dir=[{ev['dir'][0]:.2f},{ev['dir'][1]:.2f},{ev['dir'][2]:.2f}]  "
              f"dist={ev['dist']:.4f}  "
              f"IoU: {ev['pre_iou']:.4f} -> {ev['post_iou']:.4f}  "
              f"V={ev['new_V']} F={ev['new_F']}  cc={ev['cc_frac']:.2f}")

    target = 0.93
    if iou_inject > target:
        print(f"\n  TARGET MET: IoU {iou_inject:.4f} > {target}")
    else:
        print(f"\n  Target {target} not met (got {iou_inject:.4f}), "
              f"but delta {iou_inject - iou_baseline:+.4f} shows improvement")

    # ── Save visualization ────────────────────────────────────────────
    viz_path = os.path.join(args.out_dir, f'viz_extrude_{shape_name}.png')
    save_viz_grid(gt_uint8, baseline_sils, inject_sils,
                  iou_baseline, iou_inject, viz_path)

    print(f"\nTotal time: {t_bl + t_inj:.1f}s")


if __name__ == '__main__':
    main()
