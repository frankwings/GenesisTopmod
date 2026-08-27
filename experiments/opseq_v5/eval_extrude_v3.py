#!/usr/bin/env python3
"""
eval_extrude_v3.py — Extrude injection via TopMod DLFL half-edge operators.

Identical to eval_extrude_v2.py in all respects EXCEPT the extrusion mechanism:

  v2: hand-crafted numpy side-wall triangulation (extrude_face_cluster)
      → ~20 boundary edges, visible spike artefacts

  v3: topmod.high_level_ops.extrude_face (DLFL half-edge primitive)
      → 0 boundary edges guaranteed by construction, clean manifold

Conflict resolution (cluster → per-face sequential extrude):
  ASSUMPTION-1 (design choice): Option (c) — extrude each cluster face
    individually via DLFL extrude_face, sequentially on the same mesh.
    After each face extrusion, top-cap vertex positions are overridden from
    the face normal direction to the aggregated voting direction (extrude_dir).
    Rationale: sequential per-face extrusion in DLFL is safe because
    extrude_face properly rewires half-edge twins for adjacent faces —
    even when two cluster faces share an edge, the second extrusion sees
    correct adjacency from the first extrusion's side walls.  This gives
    much better geometric coverage than single-face (option b) while
    maintaining the manifold guarantee.  Confirmed by testing: 5 adjacent
    faces extruded sequentially → 0 boundary edges, χ preserved.
    Option (a) (polygon merge) requires deleting shared edges first, which
    is fragile on concave clusters.

Other assumptions (inherited from v2, unchanged):
  ASSUMPTION-2: w_depth=0.35 (see v2 for justification).
  ASSUMPTION-3: enhanced Laplacian = global ×3 weight for 30 steps post-inj.
  ASSUMPTION-4: max_rays_per_view=300 (adequate sample even at 256px).

Usage:
    python eval_extrude_v3.py [--device cuda] [--out_dir eval_out]

Outputs:
    eval_out/viz_extrude_v3_bunny.png        (6 rows × 4 cols: GT|A|B|C)
    eval_out/viz_extrude_v3_bunny_depth.png  (6 rows × 3 cols: GT|B|C depth)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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

# ── TopMod DLFL operators ───────────────────────────────────────────────────
from topmod.primitives    import _build_mesh
from topmod.io            import to_triangle_arrays
from topmod.high_level_ops import extrude_face as topmod_extrude_face

# ── Rendering + optimisation helpers ────────────────────────────────────────
from pipeline.cameras            import orbit_cameras, transform_to_clip
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

# ── Shared analysis helpers from v1 (no extrude_face_cluster) ───────────────
from eval_extrude_inject import (
    check_manifold,
    assert_manifold,
    compute_missing_error_maps,
    detect_topology_bottleneck,
    error_pixels_to_rays,
    moller_trumbore_batch,
    select_face_cluster,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────
N_VIEWS       = 6
IMG_RES       = 256
CAMERA_RADIUS = 3.0

MAX_INJECTIONS   = 20    # safety ceiling on ATTEMPTS (count now emerges via rollback)
TOTAL_STEPS      = 800
EVAL_INTERVAL    = 5
PLATEAU_STEPS    = 20
PLATEAU_EPS      = 0.005
WARMUP_STEPS     = 20
LAP_WARMUP_STEPS = 50
DEPTH_WARMUP_STEPS = 80   # phase-in depth loss after each injection (0->full)
EXTRUDE_BBOX_FRAC = 0.12
MIN_STEP_FOR_INJ  = 60
INJECT_CUTOFF_FRAC = 0.90 # allow injections up to 90% of the schedule

# ── Learnable-dist probe (Stage A, debate_design conclusion) ─────────────────
# After a cluster extrude, run a short gradient probe on a SINGLE scalar dist so
# each cluster grows a custom amount (horn 0.2·bbox, leg 0.08·bbox) instead of a
# fixed 0.12. All vertices frozen; only dist learns; top-cap = base + Δdist·dir.
# Then the cap is detached into leaves and Stage B (cold-start) runs unchanged.
DIST_PROBE_STEPS  = 30    # Stage-A gradient steps (<5% of schedule)
DIST_PROBE_LR_MULT = 5.0  # dist LR = 5x verts LR (single scalar, low variance)
DIST_FLOOR_FRAC   = 0.01  # double-sided ReLU-L2 reg lower bound (·bbox)
DIST_CEIL_FRAC    = 0.30  # ... upper bound (·bbox)
DIST_REG_LAMBDA   = 0.1   # reg strength (0 inside [floor,ceil], quadratic outside)

# ── dist-veto mode: use the dist probe as a MISVOTE FILTER, not a size knob ──
# The learnable-dist experiment (C=0.9423<0.9510) showed the probe learns tiny/
# negative dist for clusters that shouldn't grow (vote noise). So instead of
# growing at the learned dist, VETO the injection when the probe collapses
# toward zero, and commit surviving injections at the FIXED init dist (which
# gave the best 0.9510). Goal: prune misvotes -> fewer, higher-quality grows.
DIST_VETO_FRAC   = 0.5    # veto if dist_learned < 0.5 * init dist
VETO_STREAK_MAX  = 3      # consecutive vetoes -> growth done
# ── reverse-kick: seed new cap verts with inward initial Adam momentum ──
REVERSE_KICK_MULT = 3.0   # first inward step ≈ lr*mult along −extrude_dir

# ── Emergent-count control (debate_design conclusion) ────────────────────────
# Keep/rollback each extrude by the change in binary missing-pixel (error-ray)
# count — anti-aliasing-noise-free, unlike per-face IoU at 242v.
RECOVER_WINDOW    = 60    # steps to let an extrude converge before judging it
MISS_DROP_MIN     = 20    # min LOCAL-region missing-pixel drop to KEEP an extrude
MAX_WASTED_ROUNDS = 2     # consecutive rolled-back rounds -> stop growing
VOTE_MIN_VIEWS_FRAC = 0.3 # a face must be voted by >= max(2, 0.3*N_v) views

W_LAP       = 0.10
W_LAP_BOOST = 0.40
W_EDGE      = 0.01
W_DEPTH     = 0.35

# ── Operator dispatch thresholds (Step 3a — calibrate from experiment results) ──
OP_MISSING_SUM_T1    = 200   # missing pixel count above this → extrude (vs stellate)
OP_DEPTH_HF_T2       = 0.30  # depth residual high-freq fraction above this → stellate
STELLATE_RECOVER_WINDOW = 150  # verdict window for stellate rollback
FACE_CAP_FACTOR      = 2.0   # hard cap: reject injection if new_tris > 2× init_F


# ─────────────────────────────────────────────────────────────────────────────
# 6-camera rig (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

def make_6_cameras(radius=CAMERA_RADIUS, device='cuda'):
    mvps_eq, eyes_eq = orbit_cameras(
        4, elevation_deg=0.0, radius=radius,
        azimuths_deg=[0.0, 90.0, 180.0, 270.0], device=device)
    mvps_parts, eyes_elev = [], []
    for elev, az in [(35.0, 45.0), (-35.0, 225.0)]:
        m, e = orbit_cameras(1, elevation_deg=elev, radius=radius,
                             azimuths_deg=[az], device=device)
        mvps_parts.append(m); eyes_elev.extend(e)
    return torch.cat([mvps_eq, *mvps_parts], dim=0), eyes_eq + eyes_elev


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

def render_sil_and_depth(ctx, verts_t, faces_t, mvp,
                          resolution=(IMG_RES, IMG_RES)):
    H, W = resolution
    V    = verts_t.shape[0]
    pos_clip = transform_to_clip(verts_t, mvp)
    rast, _  = dr.rasterize(ctx, pos_clip, faces_t, resolution=[H, W])
    ones = torch.ones(1, V, 3, dtype=torch.float32, device=verts_t.device)
    color, _ = dr.interpolate(ones, rast, faces_t)
    sil = dr.antialias(color, rast, pos_clip, faces_t)[..., :1]
    fg_mask = rast[0, :, :, 3] > 0
    clip_zw = pos_clip[0, :, 2:4]
    zw_img, _ = dr.interpolate(clip_zw.unsqueeze(0).contiguous(), rast, faces_t)
    ndc_z = zw_img[0, :, :, 0] / zw_img[0, :, :, 1].clamp(min=1e-6)
    return sil, ndc_z, fg_mask


def render_views_n(ctx, verts_t, faces_t, mvps):
    views = []
    with torch.no_grad():
        for i in range(mvps.shape[0]):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            views.append(sil[0, :, :, 0].cpu().numpy())
    return np.stack(views, axis=0)


def render_depths_n(ctx, verts_t, faces_t, mvps):
    depths = []
    with torch.no_grad():
        for i in range(mvps.shape[0]):
            _, ndc_z, _ = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i])
            depths.append(ndc_z.cpu().numpy())
    return np.stack(depths, axis=0)


def compute_iou_n(pred_sil, gt_uint8):
    pred_fg = pred_sil > 0.5;  gt_fg = gt_uint8 < 128
    ious = []
    for v in range(pred_sil.shape[0]):
        inter = (pred_fg[v] & gt_fg[v]).sum()
        union = (pred_fg[v] | gt_fg[v]).sum()
        ious.append(float(inter) / max(float(union), 1.0))
    return float(np.mean(ious))


def compute_depth_l1_n(pred_depths, gt_depths, gt_uint8):
    total, count = 0.0, 0
    for v in range(len(pred_depths)):
        gt_fg = gt_uint8[v] < 128
        if not gt_fg.any():
            continue
        total += float(np.abs(pred_depths[v] - gt_depths[v])[gt_fg].mean())
        count += 1
    return total / max(count, 1)


def depth_loss_masked(pred_ndc_z, pred_fg, gt_ndc_z, gt_fg):
    mask = pred_fg & gt_fg
    if int(mask.sum().item()) < 4:
        return pred_ndc_z.sum() * 0.0
    return F.l1_loss(pred_ndc_z[mask], gt_ndc_z[mask])


def region_missing_count(error_maps, world_pts, mvps, radius=24):
    """
    Count binary missing pixels (GT fg not covered by render) that fall inside
    a per-view disk of `radius` px around the projection of `world_pts`.

    This is the debate's per-region error-ray metric: it isolates whether an
    extrude filled ITS local silhouette gap, immune to the transient GLOBAL
    perturbation caused by the optimizer reset + warm-up after injection.
    """
    N_v, H, W = error_maps.shape
    if world_pts is None or len(world_pts) == 0:
        return int((error_maps > 0.5).sum())
    pts_h = np.concatenate(
        [np.asarray(world_pts, dtype=np.float64),
         np.ones((len(world_pts), 1))], axis=1)          # [K,4]
    yy, xx = np.ogrid[:H, :W]
    r2 = float(radius * radius)
    total = 0
    for vi in range(N_v):
        em = error_maps[vi] > 0.5
        if not em.any():
            continue
        mvp  = mvps[vi].detach().cpu().numpy().astype(np.float64)
        clip = (mvp @ pts_h.T).T                          # [K,4]
        w    = clip[:, 3:4]
        ndc  = clip[:, :3] / np.where(np.abs(w) > 1e-8, w, 1e-8)
        px   = (ndc[:, 0] * 0.5 + 0.5) * W - 0.5
        py   = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H - 0.5
        region = np.zeros((H, W), dtype=bool)
        for cx, cy in zip(px, py):
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            region |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r2
        total += int((em & region).sum())
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Multi-view ray voting (N-view, same logic as v2)
# ─────────────────────────────────────────────────────────────────────────────

def vote_faces_multiview_n(error_maps, verts_np, tris_np, mvps,
                            max_rays_per_view=300):
    N_v     = error_maps.shape[0]
    F_count = tris_np.shape[0]
    face_votes = np.zeros(F_count, dtype=np.int32)
    face_dir   = np.zeros((F_count, 3), dtype=np.float64)
    v0 = verts_np[tris_np[:, 0]]; v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0

    for vi in range(N_v):
        origins, dirs = error_pixels_to_rays(
            error_maps[vi], mvps[vi], max_rays=max_rays_per_view)
        if len(origins) == 0:
            continue
        hit_face, _ = moller_trumbore_batch(origins, dirs, v0, v1, v2)
        for ri in range(len(origins)):
            fi = hit_face[ri]
            if fi >= 0:
                face_votes[fi] += 1
                face_dir[fi]   -= dirs[ri]
            else:
                oc  = face_centers - origins[ri]
                proj = np.sum(oc * dirs[ri], axis=1, keepdims=True)
                dists = np.linalg.norm(face_centers - origins[ri] - proj * dirs[ri], axis=1)
                nearest = np.argmin(dists)
                face_votes[nearest] += 1
                face_dir[nearest]   -= dirs[ri]
    return face_votes, face_dir


# ─────────────────────────────────────────────────────────────────────────────
# Post-injection local edge split (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

def split_new_long_edges(verts_np, tris_np, new_v_start):
    all_edges: set = set()
    for f in tris_np:
        for i in range(3):
            a, b = int(f[i]), int(f[(i+1)%3])
            all_edges.add((min(a,b), max(a,b)))
    if not all_edges:
        return verts_np, tris_np
    ea = np.array(list(all_edges), dtype=np.int32)
    lens = np.linalg.norm(verts_np[ea[:,0]] - verts_np[ea[:,1]], axis=1)
    max_len = 2.0 * float(np.mean(lens))
    is_new  = (ea[:,0] >= new_v_start) | (ea[:,1] >= new_v_start)
    to_split = ea[is_new & (lens > max_len)]
    if len(to_split) == 0:
        return verts_np, tris_np
    new_verts = list(verts_np); mid_map: dict = {}
    for a, b in to_split:
        a, b = int(a), int(b)
        key = (min(a,b), max(a,b))
        if key not in mid_map:
            mid_map[key] = len(new_verts)
            new_verts.append((verts_np[a] + verts_np[b]) / 2.0)
    new_tris = []
    for f in tris_np:
        splits: dict = {}
        for i in range(3):
            key = (min(f[i], f[(i+1)%3]), max(f[i], f[(i+1)%3]))
            if key in mid_map: splits[i] = mid_map[key]
        if not splits:
            new_tris.append(list(f))
        elif len(splits) == 1:
            ei = list(splits.keys())[0]; mid = splits[ei]
            a, b, c = int(f[ei]), int(f[(ei+1)%3]), int(f[(ei+2)%3])
            new_tris.append([a,mid,c]); new_tris.append([mid,b,c])
        else:
            new_tris.append(list(f))
    return np.array(new_verts, dtype=np.float64), np.array(new_tris, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# *** NEW: TopMod DLFL-based single-face extrusion ***
# ─────────────────────────────────────────────────────────────────────────────

def dlfl_boundary_stats(verts_np, tris_np):
    """
    Count boundary edges (count==1) and non-manifold edges (count>2)
    in a numpy triangle mesh.
    Returns (n_boundary, n_nonmanifold).
    """
    edge_cnt: Dict[Tuple[int,int], int] = {}
    for f in tris_np:
        for i in range(3):
            a, b = int(f[i]), int(f[(i+1)%3])
            key  = (min(a,b), max(a,b))
            edge_cnt[key] = edge_cnt.get(key, 0) + 1
    n_bnd = sum(1 for c in edge_cnt.values() if c == 1)
    n_nm  = sum(1 for c in edge_cnt.values() if c > 2)
    return n_bnd, n_nm


def topmod_extrude_single_face(
    verts_np:    np.ndarray,     # [V, 3] float64 current mesh vertices
    tris_np:     np.ndarray,     # [F, 3] int32  current mesh triangles
    face_idx:    int,            # index into tris_np (triangle to extrude)
    extrude_dir: np.ndarray,     # [3] unit vector — voting direction
    dist:        float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extrude a single triangle face via TopMod DLFL half-edge operator,
    then override top-cap vertex positions to align with extrude_dir.

    Algorithm (Option b — see ASSUMPTION-1 in docstring above):
      1. Build DLFL mesh from numpy via _build_mesh (requires closed mesh).
      2. Look up face face_idx by insertion order (== tris_np row order).
      3. Compute face normal BEFORE extrusion (saved for direction correction).
      4. Call topmod_extrude_face(mesh, face, dist) — wires all twins correctly.
      5. Override new top-cap vertices: nv += dist*(extrude_dir - face_normal).
      6. to_triangle_arrays → back to numpy.

    Returns:
        new_verts [V+3, 3] float64
        new_tris  [F+6, 3] int32
        old_V     int       — new vertices have indices old_V..old_V+2
    """
    old_V = len(verts_np)

    # ── 1. Build DLFL ──────────────────────────────────────────────────────
    positions    = [tuple(map(float, v)) for v in verts_np]
    face_indices = [list(map(int, tris_np[i])) for i in range(len(tris_np))]
    mesh = _build_mesh(positions, face_indices)

    # ── 2. Get target DLFL face by insertion order ─────────────────────────
    # _build_mesh creates one DLFL face per face_indices entry, in order.
    # Python dict (3.7+) preserves insertion order, so list()[face_idx] works.
    dlfl_faces = list(mesh.faces.values())
    if face_idx >= len(dlfl_faces):
        raise IndexError(f"face_idx {face_idx} >= n_faces {len(dlfl_faces)}")
    target_face = dlfl_faces[face_idx]

    # ── 3. Save face normal before extrusion ──────────────────────────────
    fn = np.array(target_face.normal(), dtype=np.float64)
    fn_norm = np.linalg.norm(fn)
    if fn_norm > 1e-8:
        fn /= fn_norm

    # ── 4. Extrude along face normal ───────────────────────────────────────
    new_faces = topmod_extrude_face(mesh, target_face, dist)
    top_face  = new_faces[0]   # first returned face is the top cap

    # ── 5. Override top-cap vertex positions ───────────────────────────────
    # New vertices were placed at: orig_v + dist * face_normal
    # Correct them to:            orig_v + dist * extrude_dir
    correction = dist * (extrude_dir - fn)          # [3]
    for he in top_face.halfedges():
        nv    = he.origin
        nv.x += correction[0]
        nv.y += correction[1]
        nv.z += correction[2]

    # ── 6. Back to numpy ───────────────────────────────────────────────────
    pos_out, tri_out = to_triangle_arrays(mesh)
    new_verts = np.array(pos_out, dtype=np.float64)
    new_tris  = np.array(tri_out, dtype=np.int32)

    # Sanity: exactly 3 new vertices (for a triangular face)
    assert new_verts.shape[0] == old_V + 3, \
        f"Expected {old_V+3} vertices, got {new_verts.shape[0]}"

    return new_verts, new_tris, old_V


def _extrude_one_face(mesh, dlfl_face, extrude_dir, dist):
    """Extrude a single DLFL face, then override its top cap to extrude_dir."""
    fn = np.array(dlfl_face.normal(), dtype=np.float64)
    fn_norm = np.linalg.norm(fn)
    if fn_norm > 1e-8:
        fn /= fn_norm
    new_faces = topmod_extrude_face(mesh, dlfl_face, dist)
    top_cap   = new_faces[0]
    correction = dist * (extrude_dir - fn)
    for he in top_cap.halfedges():
        nv = he.origin
        nv.x += correction[0]
        nv.y += correction[1]
        nv.z += correction[2]


def _extrude_cluster_perface(verts_np, tris_np, cluster, extrude_dir, dist, old_V):
    """FALLBACK: extrude each cluster triangle individually (produces spikes).

    Only used when the cluster boundary is not a single simple loop and cannot
    be merged into one n-gon face. Kept manifold, but visually spiky.
    """
    positions    = [tuple(map(float, v)) for v in verts_np]
    face_indices = [list(map(int, tris_np[i])) for i in range(len(tris_np))]
    mesh = _build_mesh(positions, face_indices)
    dlfl_faces = list(mesh.faces.values())
    for fi in cluster:
        f = dlfl_faces[int(fi)]
        if f.id not in mesh.faces:
            continue
        _extrude_one_face(mesh, f, extrude_dir, dist)
    pos_out, tri_out = to_triangle_arrays(mesh)
    return (np.array(pos_out, dtype=np.float64),
            np.array(tri_out, dtype=np.int32), old_V)


def topmod_extrude_cluster(
    verts_np:      np.ndarray,     # [V, 3] float64
    tris_np:       np.ndarray,     # [F, 3] int32
    cluster_faces: np.ndarray,     # face indices to extrude
    extrude_dir:   np.ndarray,     # [3] unit direction
    dist:          float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extrude a cluster of faces via TopMod DLFL half-edge operators (approach a).

    Instead of extruding each triangle individually (which grows a forest of
    thin spikes), the largest edge-connected component of the voted cluster is
    MERGED into a single n-gon face, then extruded ONCE along extrude_dir. This
    grows one coherent volume (e.g. a whole ear) with a clean side wall.

    Steps:
      1. Restrict cluster to its largest edge-connected component.
      2. Compute the component's boundary as directed edges; walk them into a
         single simple loop (the n-gon boundary), preserving orientation.
      3. Rebuild face list = all non-component triangles + one n-gon (the loop).
      4. Build DLFL, extrude that single n-gon face once, override top cap to
         extrude_dir.
      5. Convert back to triangle arrays.

    Falls back to per-face extrusion only if the boundary is not a simple loop.

    Returns (new_verts, new_tris, old_V); new top-cap verts are at old_V..V'-1.
    """
    old_V = len(verts_np)

    if len(cluster_faces) == 0:
        return verts_np.copy(), tris_np.copy(), old_V

    cluster_set = set(int(fi) for fi in cluster_faces)

    def tri_dir_edges(f):
        a, b, c = (int(x) for x in tris_np[f])
        return [(a, b), (b, c), (c, a)]

    # ── 1. Largest edge-connected component of the cluster ──────────────
    ue_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for f in cluster_set:
        for (a, b) in tri_dir_edges(f):
            ue = (a, b) if a < b else (b, a)
            ue_to_faces.setdefault(ue, []).append(f)
    adj: Dict[int, set] = {f: set() for f in cluster_set}
    for fs in ue_to_faces.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                adj[fs[i]].add(fs[j])
                adj[fs[j]].add(fs[i])
    seen: set = set()
    comps: List[List[int]] = []
    for f in cluster_set:
        if f in seen:
            continue
        stack, comp = [f], []
        seen.add(f)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    comp = max(comps, key=len)
    comp_set = set(comp)

    # ── 2. Boundary directed edges → single simple loop ─────────────────
    dir_edges = set()
    for f in comp_set:
        for e in tri_dir_edges(f):
            dir_edges.add(e)
    boundary = [(a, b) for (a, b) in dir_edges if (b, a) not in dir_edges]

    loop = None
    if len(boundary) >= 3:
        nxt: Dict[int, int] = {}
        ok = True
        for (a, b) in boundary:
            if a in nxt:            # vertex starts two boundary edges → pinch
                ok = False
                break
            nxt[a] = b
        if ok and len(nxt) == len(boundary):
            start = boundary[0][0]
            walk = [start]
            cur = nxt[start]
            while cur != start and cur in nxt and len(walk) <= len(boundary):
                walk.append(cur)
                cur = nxt[cur]
            if cur == start and len(walk) == len(boundary):
                loop = walk

    if loop is None:
        print(f"  [WARN] cluster boundary not a simple loop "
              f"({len(boundary)} bnd edges) — per-face fallback")
        return _extrude_cluster_perface(
            verts_np, tris_np, comp, extrude_dir, dist, old_V)

    # ── 3. Rebuild faces: non-component triangles + one merged n-gon ────
    positions    = [tuple(map(float, v)) for v in verts_np]
    face_indices = [list(map(int, tris_np[i]))
                    for i in range(len(tris_np)) if i not in comp_set]
    ngon_index = len(face_indices)
    face_indices.append(list(loop))
    mesh = _build_mesh(positions, face_indices)

    # ── 4. Extrude that single n-gon face once ──────────────────────────
    dlfl_faces  = list(mesh.faces.values())
    target_face = dlfl_faces[ngon_index]
    _extrude_one_face(mesh, target_face, extrude_dir, dist)

    # ── 4b. Drop isolated interior verts left by the merge ──────────────
    # Merging the component into one n-gon removes the interior triangles,
    # leaving the interior vertices unreferenced (v.he is None). If kept, they
    # inflate V and corrupt χ = V − E + F (and thus genus). Remove them so the
    # topology accounting stays honest (extrude must NOT change genus).
    isolated = [v for v in list(mesh.vertices.values()) if v.he is None]
    for v in isolated:
        mesh._remove_vertex(v)

    # ── 5. Back to numpy ────────────────────────────────────────────────
    # After cleanup, DLFL vertex order = [surviving originals..., n new
    # top-cap verts]. extrude_face creates exactly len(loop) new vertices, so
    # the new-vertex block starts at old_V_ret = V_total − len(loop).
    pos_out, tri_out = to_triangle_arrays(mesh)
    new_verts = np.array(pos_out, dtype=np.float64)
    new_tris  = np.array(tri_out, dtype=np.int32)
    old_V_ret = mesh.V() - len(loop)

    print(f"    DLFL merged-face extrude ({len(comp_set)} tris → 1 "
          f"{len(loop)}-gon, {len(isolated)} iso-verts dropped): "
          f"V={mesh.V()} E={mesh.E()} F={mesh.F()} "
          f"χ={mesh.euler_characteristic()} genus={mesh.genus()}")

    return new_verts, new_tris, old_V_ret


def assert_watertight_v3(verts_np, tris_np, label=""):
    """
    Hard assert: boundary_edges == 0 AND non-manifold_edges == 0.
    Also prints V/E/F/euler/genus using DLFL (only when mesh is closed).
    Raises AssertionError on any failure (never raises ValueError).
    """
    n_bnd, n_nm = dlfl_boundary_stats(verts_np, tris_np)
    tag = f" [{label}]" if label else ""

    # Fail fast on topology errors BEFORE attempting DLFL build
    # (_build_mesh raises ValueError on open meshes, so we check first)
    if n_nm > 0:
        raise AssertionError(
            f"Non-manifold mesh{tag}: {n_nm} edges with count>2")
    if n_bnd > 0:
        raise AssertionError(
            f"Non-watertight mesh{tag}: {n_bnd} boundary edges (count==1)")

    # Mesh is closed — build DLFL for topology stats
    positions    = [tuple(map(float, v)) for v in verts_np]
    face_indices = [list(map(int, tris_np[i])) for i in range(len(tris_np))]
    mesh = _build_mesh(positions, face_indices)
    print(f"  [manifold OK{tag}] V={mesh.V()} E={mesh.E()} F={mesh.F()}"
          f"  χ={mesh.euler_characteristic()} genus={mesh.genus()}"
          f"  boundary_edges=0  nonmanifold=0")


# ─────────────────────────────────────────────────────────────────────────────
# *** NEW (Step 3a): TopMod DLFL stellate cluster ***
# ─────────────────────────────────────────────────────────────────────────────

def topmod_stellate_cluster(
    verts_np:      np.ndarray,     # [V, 3] float64
    tris_np:       np.ndarray,     # [F, 3] int32
    cluster_faces: np.ndarray,     # face indices to stellate
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Stellate each face in cluster_faces via TopMod DLFL stellate operator.

    For each face, stellate adds a center vertex at the face centroid and splits
    the face into n triangles (one per edge). Face objects are invalidated after
    each stellate call, so all targets are snapshotted BEFORE any call.

    Topology per stellate of an n-gon: V+1, E+n, F+n-1.
    For a triangle (n=3): V+1, E+3, F+2  →  χ unchanged (1-3+2=0).

    Returns (new_verts, new_tris, old_V); new center vertices are appended after
    original ones in mesh.vertices insertion order, so new_v_start = old_V.
    """
    from topmod.high_level_ops import stellate as topmod_stellate

    positions    = [tuple(v) for v in verts_np.tolist()]
    face_indices = [tuple(int(x) for x in tri) for tri in tris_np.tolist()]
    mesh = _build_mesh(positions, face_indices)
    old_V = mesh.V()

    # Snapshot face objects BEFORE any stellate (stellate invalidates each face)
    all_faces    = list(mesh.faces.values())
    target_faces = [all_faces[int(fi)] for fi in cluster_faces]

    for face in target_faces:
        topmod_stellate(mesh, face)

    positions_out, triangles_out = to_triangle_arrays(mesh)
    new_verts = np.array(positions_out, dtype=np.float64)
    new_tris  = (np.array(triangles_out, dtype=np.int32)
                 if triangles_out else np.zeros((0, 3), dtype=np.int32))

    print(f"    DLFL stellate ({len(cluster_faces)} face(s)): "
          f"V={mesh.V()} E={mesh.E()} F={mesh.F()} "
          f"χ={mesh.euler_characteristic()} genus={mesh.genus()}")

    return new_verts, new_tris, old_V


# ─────────────────────────────────────────────────────────────────────────────
# *** NEW (Step 3a): Operator dispatcher ***
# ─────────────────────────────────────────────────────────────────────────────

def choose_operator(
    cluster:     np.ndarray,        # [K] face indices
    tris_cur:    np.ndarray,        # [F, 3]
    verts_cur:   np.ndarray,        # [V, 3]
    error_maps:  np.ndarray,        # [N_V, H, W] float32  (>0.5 = missing)
    pred_depths: np.ndarray,        # [N_V, H, W] float32  NDC-z
    gt_depths:   np.ndarray,        # [N_V, H, W] float32  NDC-z
    gt_uint8:    np.ndarray,        # [N_V, H, W] uint8    (0=fg)
    mvps:        torch.Tensor,      # [N_V, 4, 4]
) -> Tuple[str, dict]:              # (op_type, signals)
    """
    Decide whether to extrude or stellate the cluster.

    Signal 1 — missing_sum:
      Project each cluster face's 3 vertices into each view.  Accumulate missing
      error pixels inside a bbox padded by 12 px around the projected footprint.
      Sum across all views.

    Signal 2 — depth_resid_hf:
      For each view, compute residual = pred_depths[vi] - gt_depths[vi] on GT fg
      pixels restricted to the cluster bbox.  FFT the patch; high-freq mask
      excludes the centre 1/4 (DC + low-freq).  Take median over valid views.

    Hard rules (in order):
      depth_resid_hf > OP_DEPTH_HF_T2  →  'stellate'
      missing_sum    > OP_MISSING_SUM_T1 →  'extrude'
      else                               →  'extrude'
    """
    import numpy.fft as npfft

    N_v, H, W = error_maps.shape
    PAD_PX    = 12

    # ── Project cluster vertices to each view, collect bbox ──────────────────
    cluster_vids = set()
    for fi in cluster:
        for vi in tris_cur[int(fi)]:
            cluster_vids.add(int(vi))
    cluster_v = verts_cur[list(cluster_vids)]   # [K, 3]
    pts_h = np.concatenate(
        [cluster_v, np.ones((len(cluster_v), 1))], axis=1)  # [K, 4]

    missing_sum = 0
    hf_ratios   = []

    for vi in range(N_v):
        mvp  = mvps[vi].detach().cpu().numpy().astype(np.float64)
        clip = (mvp @ pts_h.T).T                             # [K, 4]
        w    = clip[:, 3:4]
        ndc  = clip[:, :3] / np.where(np.abs(w) > 1e-8, w, 1e-8)
        px   = ((ndc[:, 0] * 0.5 + 0.5) * W - 0.5).astype(np.float32)
        py   = ((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H - 0.5).astype(np.float32)

        valid = np.isfinite(px) & np.isfinite(py)
        if not valid.any():
            continue

        px_v, py_v = px[valid], py[valid]
        x0 = int(np.clip(px_v.min() - PAD_PX, 0, W - 1))
        x1 = int(np.clip(px_v.max() + PAD_PX, 0, W - 1)) + 1
        y0 = int(np.clip(py_v.min() - PAD_PX, 0, H - 1))
        y1 = int(np.clip(py_v.max() + PAD_PX, 0, H - 1)) + 1

        if x1 <= x0 or y1 <= y0:
            continue

        # missing pixels inside bbox
        em_patch = error_maps[vi, y0:y1, x0:x1]
        missing_sum += int((em_patch > 0.5).sum())

        # depth residual high-freq fraction
        gt_fg_mask = gt_uint8[vi] < 128
        roi_mask   = gt_fg_mask[y0:y1, x0:x1]
        if not roi_mask.any():
            continue

        pred_patch = pred_depths[vi, y0:y1, x0:x1].astype(np.float32)
        gt_patch   = gt_depths[vi, y0:y1, x0:x1].astype(np.float32)
        resid      = pred_patch - gt_patch
        resid[~roi_mask] = 0.0

        fft2    = npfft.fft2(resid)
        power   = np.abs(fft2) ** 2
        total_p = float(power.sum())
        if total_p < 1e-12:
            continue

        fh, fw   = resid.shape
        # high-freq mask: exclude central 1/4 (DC + low-freq)
        cy, cx   = fh // 4, fw // 4
        hf_mask  = np.ones((fh, fw), dtype=bool)
        hf_mask[:cy, :cx]   = False   # top-left
        hf_mask[:cy, -cx:]  = False   # top-right
        hf_mask[-cy:, :cx]  = False   # bottom-left
        hf_mask[-cy:, -cx:] = False   # bottom-right
        hf_p    = float(power[hf_mask].sum())
        hf_ratios.append(hf_p / total_p)

    depth_resid_hf = float(np.median(hf_ratios)) if hf_ratios else 0.0
    signals = {'missing_sum': int(missing_sum), 'depth_resid_hf': depth_resid_hf}

    if depth_resid_hf > OP_DEPTH_HF_T2:
        return 'stellate', signals
    elif missing_sum > OP_MISSING_SUM_T1:
        return 'extrude', signals
    else:
        return 'extrude', signals


# ─────────────────────────────────────────────────────────────────────────────
# *** NEW (Step 3a): Mesh genus helper ***
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_genus(verts_np: np.ndarray, tris_np: np.ndarray) -> int:
    """Compute genus via Euler characteristic χ = V - E + F, genus = (2 - χ) / 2."""
    V = len(verts_np)
    F = len(tris_np)
    edges: set = set()
    for tri in tris_np:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    E = len(edges)
    chi = V - E + F
    return int(round((2 - chi) / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Variant A — baseline (no injection, no depth)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(ctx, verts_init, tris_init, gt_uint8, mvps, device,
                 n_steps=TOTAL_STEPS):
    N_v = mvps.shape[0]
    targets = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device
                           ).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)
    opt   = torch.optim.Adam([verts_t], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=3e-5)

    for step in range(n_steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(N_v):
            sil_loss = sil_loss + F.l1_loss(
                render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                  resolution=(IMG_RES, IMG_RES)),
                targets[i:i+1])
        sil_loss = sil_loss / N_v
        (sil_loss + W_LAP*laplacian_loss(verts_t, faces_t) +
         W_EDGE*edge_length_loss(verts_t, faces_t)).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
            print(f"  [baseline] step {step:4d}/{n_steps}  "
                  f"sil={sil_loss.item():.4f}  iou={iou:.4f}")

    pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
    return compute_iou_n(pred_sils, gt_uint8), pred_sils


# ─────────────────────────────────────────────────────────────────────────────
# Variants B & C — injection via TopMod (with optional depth)
# ─────────────────────────────────────────────────────────────────────────────

def run_with_injection_v3(
    ctx,
    verts_init:     np.ndarray,
    tris_init:      np.ndarray,
    gt_uint8:       np.ndarray,           # [N, H, W] uint8, 0=fg
    gt_depths:      Optional[np.ndarray], # [N, H, W] float NDC-z, or None
    mvps:           torch.Tensor,         # [N, 4, 4]
    device:         str,
    use_depth:      bool = False,
    max_injections: int  = MAX_INJECTIONS,
    total_steps:    int  = TOTAL_STEPS,
    use_rollback:   bool = False,
    recover_window: int  = RECOVER_WINDOW,
    use_dist_veto:  bool = False,
    use_warm_start: bool = False,
    use_learnable_dist: bool = False,
    use_reverse_kick: bool = False,
    use_operator_menu: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray, List[dict]]:
    """
    Returns (final_iou, final_depth_l1, pred_sils, pred_depths, injection_log).

    Key difference from v2: extrusion uses topmod_extrude_single_face()
    instead of extrude_face_cluster(), guaranteeing zero boundary edges.
    """
    N_v   = mvps.shape[0]
    gt_fg = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg).unsqueeze(-1).to(device)

    gt_depth_t = gt_fg_t = None
    if use_depth and gt_depths is not None:
        gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                      for i in range(N_v)]
        gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                      for i in range(N_v)]

    verts_np, tris_np = adaptive_remesh(
        verts_init.copy().astype(np.float64), tris_init.copy())

    # Verify initial mesh is watertight (required for _build_mesh round-trip)
    n_bnd0, _ = dlfl_boundary_stats(verts_np, tris_np)
    if n_bnd0 > 0:
        raise RuntimeError(
            f"Initial mesh has {n_bnd0} boundary edges — cannot use DLFL extrude")

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device
                           ).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)

    LR, LR_MIN = 3e-3, 3e-5
    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=LR_MIN)

    injection_log:  List[dict] = []
    n_injections   = 0      # committed extrusions
    n_attempts     = 0      # extrude attempts (>= n_injections if rollback on)
    weak_rounds    = 0      # consecutive plateaus with too-weak multi-view votes
    growth_done    = False  # set True once votes dry up -> stop growing
    veto_streak    = 0      # dist-veto mode: consecutive vetoed injections
    pending        = None   # rollback mode: trial extrude awaiting verdict
    warmup_counter = 0
    lap_boost_left = 0
    depth_warmup_left = 0
    new_v_start    = -1
    iou_at_step:   List[Tuple[int, float]] = []
    loss_history:  List[float] = []          # total loss per step (for stellate verdict)

    # ── operator-menu one-time setup ─────────────────────────────────────────
    init_F     = len(tris_np)                # for FACE_CAP_FACTOR check
    init_genus = _mesh_genus(verts_np, tris_np)
    gt_depths_np = gt_depths   # alias for choose_operator (already numpy or None)

    def _rebuild_opt(v_np, t_np, cur_step, warm_opt=None, warm_param=None,
                     warm_map=None, kick=None):
        """Rebuild leaf-tensor verts + Adam + cosine sched for remaining steps.

        Momentum-preserving mode: if `warm_opt`/`warm_param` (the pre-extrude
        Adam + its verts tensor) are given, copy the old vertices' Adam state
        (step count + exp_avg + exp_avg_sq) into the surviving vertices' rows of
        the new optimizer and leave the k new vertices at zero momentum.

        CRITICAL: extrude MERGES the cluster into one n-gon and drops the now
        isolated interior verts, and `to_triangle_arrays` then RENUMBERS the
        survivors (compacting the dropped slots). So new row `i` does NOT map to
        old row `i`. `warm_map=(new_idx, old_idx)` gives the correct per-survivor
        remap (built by position match at the call site); without it the copy is
        index-aligned (only valid when no verts were dropped/renumbered).

        Reverse-kick mode: `kick=(kidx, kdir, mult)` seeds the NEW vertices with
        an Adam state whose first update points INWARD (−kdir), softening the
        extrude overshoot. We set exp_avg=kdir (outward, unit) and exp_avg_sq=
        (1/mult)^2 so that m/sqrt(v)=kdir*mult exactly → update=−lr*mult*kdir
        (inward, exact direction). It decays as real gradients accumulate.
        Composes with cold-start (survivors stay at zero momentum + warmup).
        """
        vt = torch.tensor(v_np, dtype=torch.float32,
                          device=device).requires_grad_(True)
        ft = torch.tensor(t_np, dtype=torch.int32, device=device)
        o  = torch.optim.Adam([vt], lr=LR)
        rem = max(total_steps - cur_step - 1, 1)
        s  = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=rem, eta_min=LR_MIN)

        exp_avg = exp_avg_sq = st = None

        if warm_opt is not None and warm_param is not None:
            old_state = warm_opt.state.get(warm_param, {})
            if 'exp_avg' in old_state:
                exp_avg    = torch.zeros_like(vt)
                exp_avg_sq = torch.zeros_like(vt)
                if warm_map is not None:
                    new_i, old_i = warm_map
                    exp_avg[new_i]    = old_state['exp_avg'][old_i]
                    exp_avg_sq[new_i] = old_state['exp_avg_sq'][old_i]
                else:
                    n_old = old_state['exp_avg'].shape[0]
                    exp_avg[:n_old]    = old_state['exp_avg']
                    exp_avg_sq[:n_old] = old_state['exp_avg_sq']
                # Preserve the step count so Adam bias-correction stays warm for
                # the old vertices (clone to avoid aliasing the freed optimizer).
                st = old_state['step']
                st = st.clone() if torch.is_tensor(st) else torch.tensor(
                    float(st))

        if kick is not None:
            kidx, kdir, kmult = kick
            if exp_avg is None:
                exp_avg    = torch.zeros_like(vt)
                exp_avg_sq = torch.zeros_like(vt)
            # exp_avg = outward unit dir → Adam steps inward (−kdir); exp_avg_sq
            # = (1/mult)^2 makes m/sqrt(v) = kdir*mult (exact inward direction).
            exp_avg[kidx]    = kdir
            exp_avg_sq[kidx] = (1.0 / float(kmult)) ** 2
            if st is None:
                st = torch.tensor(1.0)

        if exp_avg is not None:
            o.state[vt] = {
                'step':       st if st is not None else torch.tensor(1.0),
                'exp_avg':    exp_avg,
                'exp_avg_sq': exp_avg_sq,
            }
        return vt, ft, o, s

    def _refine_dist(base_verts, tris, new_idx, dir_np, dist_init, bbox,
                     n_steps=DIST_PROBE_STEPS):
        """Stage A — gradient-probe the per-cluster extrude distance.

        All vertices are FROZEN; only a single scalar `dist_k` learns. The
        top-cap vertices (base_verts[new_idx]) slide along `dir_np` to best fill
        the silhouette/depth targets:  top = base_top + (dist_k - dist_init)*dir.
        Returns (updated_verts_float64, dist_learned). Cheap (<5% of schedule)
        and side-effect-free: the cap is baked into positions, no graph survives.
        """
        if len(new_idx) == 0:
            return base_verts, float(dist_init)
        vbase  = torch.tensor(base_verts, dtype=torch.float32, device=device)
        tris_t = torch.tensor(tris, dtype=torch.int32, device=device)
        dir_t  = torch.tensor(dir_np, dtype=torch.float32, device=device)
        idx_t  = torch.tensor(new_idx, dtype=torch.long, device=device)
        base_top = vbase[idx_t].clone()
        dist_k = torch.tensor(float(dist_init), device=device,
                              requires_grad=True)
        floor = DIST_FLOOR_FRAC * bbox
        ceil  = DIST_CEIL_FRAC * bbox
        o = torch.optim.Adam([dist_k], lr=LR * DIST_PROBE_LR_MULT)
        for _ in range(n_steps):
            o.zero_grad()
            vfull = vbase.clone()
            vfull = vfull.index_copy(
                0, idx_t, base_top + (dist_k - dist_init) * dir_t)
            sl = dl = torch.tensor(0.0, device=device)
            for i in range(N_v):
                if use_depth and gt_depth_t is not None:
                    sil_i, ndc_z_i, fg_i = render_sil_and_depth(
                        ctx, vfull, tris_t, mvps[i], (IMG_RES, IMG_RES))
                    sl = sl + F.l1_loss(sil_i, targets[i:i+1])
                    dl = dl + depth_loss_masked(
                        ndc_z_i, fg_i, gt_depth_t[i], gt_fg_t[i])
                else:
                    sl = sl + F.l1_loss(
                        render_silhouette(ctx, vfull, tris_t, mvps[i],
                                          resolution=(IMG_RES, IMG_RES)),
                        targets[i:i+1])
            sl = sl / N_v
            dl = dl / N_v
            reg = (DIST_REG_LAMBDA * torch.relu(floor - dist_k).pow(2)
                   + DIST_REG_LAMBDA * torch.relu(dist_k - ceil).pow(2))
            loss = sl + (W_DEPTH * dl if use_depth else 0.0) + reg
            loss.backward()
            o.step()
        dist_final = float(dist_k.detach())
        out = base_verts.copy()
        out[new_idx] = (base_top.cpu().numpy()
                        + (dist_final - dist_init) * dir_np)
        return out, dist_final

    for step in range(total_steps):

        # ── rollback mode: keep/rollback verdict on a pending trial extrude ──
        # Only judged after `recover_window` steps (default 150) so the extrude
        # has time to complete its delayed+compounding recovery — a short window
        # (60) sees only the transient optimizer-reset crater and wrongly kills
        # good protrusions.
        if use_rollback and pending is not None and step >= pending['eval_step']:
            pred_sils_e = render_views_n(ctx, verts_t, faces_t, mvps)
            err_e = compute_missing_error_maps(pred_sils_e, gt_uint8)
            iou_e = compute_iou_n(pred_sils_e, gt_uint8)

            pending_op = pending.get('op_type', 'extrude')

            if use_operator_menu and pending_op == 'stellate':
                # Stellate verdict: loss must drop by 2σ below trigger
                opt.zero_grad()
                sl_now = dl_now = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    for i in range(N_v):
                        if use_depth and gt_depth_t is not None:
                            sil_i, ndc_z_i, fg_i = render_sil_and_depth(
                                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
                            sl_now = sl_now + F.l1_loss(sil_i, targets[i:i+1])
                            dl_now = dl_now + depth_loss_masked(
                                ndc_z_i, fg_i, gt_depth_t[i], gt_fg_t[i])
                        else:
                            sl_now = sl_now + F.l1_loss(
                                render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                                  resolution=(IMG_RES, IMG_RES)),
                                targets[i:i+1])
                lw_now = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
                dw_now = W_DEPTH if (not use_depth or depth_warmup_left == 0) else 0.0
                loss_now = float((sl_now / N_v
                                  + (dw_now * dl_now / N_v if use_depth else 0.0)
                                  + lw_now * laplacian_loss(verts_t, faces_t)
                                  + W_EDGE * edge_length_loss(verts_t, faces_t)
                                  ).item())
                threshold = (pending['loss_trigger']
                             - 2.0 * pending['loss_sigma'])
                keep = loss_now < threshold
                if keep:
                    n_injections += 1
                    weak_rounds   = 0
                    ev = pending['event']
                    ev['post_iou'] = float(iou_e)
                    injection_log.append(ev)
                    print(f"  [KEEP stellate #{n_injections}] "
                          f"@step{pending['step']}: "
                          f"loss {loss_now:.4f} < trigger "
                          f"{pending['loss_trigger']:.4f} - 2σ "
                          f"({threshold:.4f})  iou={iou_e:.4f}")
                else:
                    verts_t, faces_t, opt, sched = _rebuild_opt(
                        pending['snap_verts'], pending['snap_tris'], step)
                    warmup_counter = depth_warmup_left = lap_boost_left = 0
                    print(f"  [ROLLBACK stellate] @step{pending['step']}: "
                          f"loss {loss_now:.4f} >= threshold {threshold:.4f} "
                          f"— reverted")
            else:
                # Standard extrude verdict: missing-pixel drop
                missing_after = region_missing_count(
                    err_e, pending['region_pts'], mvps)
                drop  = pending['missing_before'] - missing_after
                if drop >= MISS_DROP_MIN:
                    # KEEP: the extrude filled a genuine silhouette gap
                    n_injections += 1
                    weak_rounds   = 0
                    ev = pending['event']
                    ev['post_iou']       = float(iou_e)
                    ev['missing_before'] = int(pending['missing_before'])
                    ev['missing_after']  = int(missing_after)
                    injection_log.append(ev)
                    print(f"  [KEEP #{n_injections}] extrude @step{pending['step']}: "
                          f"local-missing {pending['missing_before']}->{missing_after} "
                          f"(drop {drop}>= {MISS_DROP_MIN})  iou={iou_e:.4f}  "
                          f"V={verts_t.shape[0]}")
                else:
                    # ROLLBACK: restore pre-extrude snapshot
                    verts_t, faces_t, opt, sched = _rebuild_opt(
                        pending['snap_verts'], pending['snap_tris'], step)
                    warmup_counter = depth_warmup_left = lap_boost_left = 0
                    print(f"  [ROLLBACK] extrude @step{pending['step']}: "
                          f"local-missing drop {drop} < {MISS_DROP_MIN} — reverted")

            pending = None
            iou_at_step.clear()
            iou_at_step.append((step, iou_e))

        # ── lr warm-up ──────────────────────────────────────────────────────
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = sched.get_last_lr()[0] * max(frac, 0.05)
            warmup_counter -= 1

        # ── forward pass ────────────────────────────────────────────────────
        opt.zero_grad()
        sil_loss = depth_loss_val = torch.tensor(0.0, device=device)

        for i in range(N_v):
            if use_depth and gt_depth_t is not None:
                sil_i, ndc_z_i, fg_i = render_sil_and_depth(
                    ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
                sil_loss      = sil_loss + F.l1_loss(sil_i, targets[i:i+1])
                depth_loss_val = depth_loss_val + depth_loss_masked(
                    ndc_z_i, fg_i, gt_depth_t[i], gt_fg_t[i])
            else:
                sil_loss = sil_loss + F.l1_loss(
                    render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                      resolution=(IMG_RES, IMG_RES)),
                    targets[i:i+1])

        sil_loss       = sil_loss / N_v
        depth_loss_val = depth_loss_val / N_v

        lw = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        if lap_boost_left > 0: lap_boost_left -= 1

        # depth phase-in: ramp 0->full over DEPTH_WARMUP_STEPS after each inject
        if depth_warmup_left > 0:
            dw = W_DEPTH * (1.0 - depth_warmup_left / DEPTH_WARMUP_STEPS)
            depth_warmup_left -= 1
        else:
            dw = W_DEPTH

        total_loss = (sil_loss
                      + (dw * depth_loss_val if use_depth else 0.0)
                      + lw * laplacian_loss(verts_t, faces_t)
                      + W_EDGE * edge_length_loss(verts_t, faces_t))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        if warmup_counter == 0: sched.step()

        loss_history.append(float(total_loss.item()))

        # ── periodic IoU ─────────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou       = compute_iou_n(pred_sils, gt_uint8)
            iou_at_step.append((step, iou))
            if step % 100 == 0:
                tag = f"sil={sil_loss.item():.4f}"
                if use_depth: tag += f"  dep={depth_loss_val.item():.4f}"
                print(f"  [{'inj+dep' if use_depth else 'inject '}] "
                      f"step {step:4d}/{total_steps}  {tag}  iou={iou:.4f}  "
                      f"V={verts_t.shape[0]} F={faces_t.shape[0]}  inj={n_injections}")

        # ── plateau detection + TopMod injection ─────────────────────────────
        if (not growth_done
                and not (use_rollback and pending is not None)
                and n_attempts < max_injections
                and step >= MIN_STEP_FOR_INJ
                and step <= int(total_steps * INJECT_CUTOFF_FRAC)
                and warmup_counter == 0
                and step % EVAL_INTERVAL == 0):

            n_needed = PLATEAU_STEPS // EVAL_INTERVAL + 1
            if len(iou_at_step) < n_needed:
                continue
            recent = [h[1] for h in iou_at_step[-n_needed:]]
            if recent[-1] - recent[0] >= PLATEAU_EPS:
                continue

            pred_sils  = render_views_n(ctx, verts_t, faces_t, mvps)
            error_maps = compute_missing_error_maps(pred_sils, gt_uint8)
            if (error_maps > 0.5).sum() < 50:
                continue
            is_bn, cc_frac = detect_topology_bottleneck(error_maps)
            if not is_bn:
                continue

            # ── ray voting ─────────────────────────────────────────────────
            verts_cur = verts_t.detach().cpu().numpy().astype(np.float64)
            tris_cur  = faces_t.cpu().numpy()

            face_votes, face_dir = vote_faces_multiview_n(
                error_maps, verts_cur, tris_cur, mvps, max_rays_per_view=300)

            # Emergent-count gate: a genuine missing protrusion is voted by
            # multiple views (>= max(2, 0.3*N_v)). If the strongest face is too
            # weakly voted, the plateau is NOT a topology gap — count consecutive
            # weak plateaus and terminate growth once votes dry up. This is what
            # makes the injection count emerge from the geometry instead of a
            # hand-set budget: keep growing while there is real evidence, stop
            # when there is none.
            vote_min = max(2, int(round(N_v * VOTE_MIN_VIEWS_FRAC)))
            if int(face_votes.max()) < vote_min:
                weak_rounds += 1
                if weak_rounds >= MAX_WASTED_ROUNDS:
                    growth_done = True
                    print(f"  [GROWTH DONE] {weak_rounds} consecutive weak-vote "
                          f"plateaus (max_votes={int(face_votes.max())} < "
                          f"{vote_min}) — no more topology gaps; stop growing.")
                continue
            weak_rounds = 0

            cluster = select_face_cluster(face_votes, tris_cur,
                                          top_k=8, grow_rings=1)
            if len(cluster) == 0:
                continue

            # Extrude direction from aggregate cluster voting
            cluster_dir = face_dir[cluster].sum(axis=0)
            dir_norm    = np.linalg.norm(cluster_dir)
            if dir_norm < 1e-8:
                # Fallback: average face normal of cluster
                normals = np.cross(
                    verts_cur[tris_cur[cluster, 1]] - verts_cur[tris_cur[cluster, 0]],
                    verts_cur[tris_cur[cluster, 2]] - verts_cur[tris_cur[cluster, 0]])
                cluster_dir = normals.mean(axis=0)
                dir_norm    = np.linalg.norm(cluster_dir)
                if dir_norm < 1e-8: continue
            extrude_dir  = cluster_dir / dir_norm
            extrude_dist = EXTRUDE_BBOX_FRAC * float(
                verts_cur.max() - verts_cur.min())
            pre_iou = iou_at_step[-1][1]

            # ── Operator dispatch (menu path) ───────────────────────────────
            if use_operator_menu:
                pred_depths_np = render_depths_n(ctx, verts_t, faces_t, mvps)
                op_type, op_signals = choose_operator(
                    cluster, tris_cur, verts_cur, error_maps,
                    pred_depths_np,
                    gt_depths_np if gt_depths_np is not None
                        else np.zeros_like(pred_depths_np),
                    gt_uint8, mvps)
                print(f"  [operator-menu] op={op_type}  signals={op_signals}")
            else:
                op_type    = 'extrude'
                op_signals = {}

            # ── TopMod DLFL injection (branch on op_type) ───────────────────
            if op_type == 'stellate':
                # Face-cap check BEFORE stellate
                stellate_F_estimate = (len(tris_cur)
                                       + 2 * len(cluster))
                if stellate_F_estimate > FACE_CAP_FACTOR * init_F:
                    print(f"  [WARN] stellate would exceed face cap "
                          f"({stellate_F_estimate} > "
                          f"{FACE_CAP_FACTOR}×{init_F}) — skipping")
                    continue
                try:
                    new_verts, new_tris, old_V = topmod_stellate_cluster(
                        verts_cur, tris_cur, cluster)
                except Exception as e:
                    print(f"  [WARN] stellate failed: {e} — falling back to extrude")
                    op_type = 'extrude'

            if op_type == 'extrude':
                # Face-cap check BEFORE extrude
                if len(tris_cur) > FACE_CAP_FACTOR * init_F:
                    print(f"  [WARN] extrude would exceed face cap "
                          f"({len(tris_cur)} > "
                          f"{FACE_CAP_FACTOR}×{init_F}) — skipping")
                    continue
                try:
                    new_verts, new_tris, old_V = topmod_extrude_cluster(
                        verts_cur, tris_cur,
                        cluster, extrude_dir, extrude_dist)
                except Exception as e:
                    print(f"  [WARN] TopMod cluster extrude failed: {e} — skipping")
                    continue

            # ── Genus check (menu path) ─────────────────────────────────────
            if use_operator_menu:
                post_genus = _mesh_genus(new_verts, new_tris)
                if post_genus != init_genus:
                    print(f"  [WARN] injection changed genus "
                          f"{init_genus}→{post_genus} — skipping")
                    continue

            # ── Hard watertight assertion ───────────────────────────────────
            n_bnd_post, n_nm_post = dlfl_boundary_stats(new_verts, new_tris)
            try:
                assert_watertight_v3(new_verts, new_tris,
                                     f"after inject {n_injections+1}")
            except AssertionError as e:
                print(f"  [WARN] {e} — skipping injection")
                continue

            # ── Stage A: gradient-probe the extrude distance (opt-in only) ──
            # The dist probe is EXPENSIVE and only two modes consume it:
            #   • dist-veto     — probe as a misvote filter (commit at fixed dist)
            #   • learnable-dist — bake the learned cap position (size knob)
            # Default / warm-start use the FIXED init dist (the 0.9510 champion
            # setup), so the probe is skipped entirely — keeping those runs a
            # clean apples-to-apples test of the optimizer-rebuild strategy.
            bbox_diag = float(verts_cur.max() - verts_cur.min())
            new_idx   = list(range(old_V, int(new_verts.shape[0])))
            dist_learned = extrude_dist
            dist_probe   = extrude_dist

            if use_dist_veto or use_learnable_dist:
                probe_verts, dist_probe = _refine_dist(
                    new_verts, new_tris, new_idx, extrude_dir, extrude_dist,
                    bbox_diag)

            if use_dist_veto:
                # Use the dist probe purely as a MISVOTE FILTER, not a size knob.
                # If the cluster doesn't "want" to grow (learned dist collapses
                # toward 0), the error-ray vote was spurious -> undo this whole
                # injection. Survivors commit at the FIXED init dist.
                if dist_probe < DIST_VETO_FRAC * extrude_dist:
                    veto_streak += 1
                    n_attempts  += 1
                    iou_at_step.clear()
                    print(f"\n  [VETO] extrude @step{step}: probe dist "
                          f"{dist_probe:.4f} < {DIST_VETO_FRAC:.2f}*"
                          f"{extrude_dist:.4f} — misvote, undone "
                          f"(streak {veto_streak}/{VETO_STREAK_MAX})")
                    if veto_streak >= VETO_STREAK_MAX:
                        growth_done = True
                        print(f"  [growth_done] {veto_streak} consecutive "
                              f"vetoes — votes exhausted, stop growing")
                    continue
                veto_streak = 0
                # keep new_verts at the fixed init dist (discard probe geometry).
            elif use_learnable_dist:
                # learnable-dist size-knob mode: bake the learned cap position.
                dist_learned = dist_probe
                new_verts    = probe_verts

            # ── Local edge split on side-wall ───────────────────────────────
            split_verts, split_tris = split_new_long_edges(
                new_verts, new_tris, old_V)

            # Verify split didn't break manifold
            n_bnd_s, _ = dlfl_boundary_stats(split_verts, split_tris)
            if n_bnd_s == 0:
                new_verts, new_tris = split_verts, split_tris
            else:
                print(f"  [WARN] edge split created {n_bnd_s} boundary edges — reverting split")

            # ── Rebuild optimizer ───────────────────────────────────────────
            # Cold-start (zero momentum + warmup) was 0.9510. Warm-start scored
            # 0.9358 but that run had a momentum-misalignment BUG (extrude drops
            # iso-verts and renumbers survivors, so index-aligned copy scrambled
            # momentum). With `use_warm_start`, remap momentum by POSITION match.
            snap_verts, snap_tris = verts_cur.copy(), tris_cur.copy()
            if use_warm_start:
                # Build survivor remap: new survivor row i (i < old_V) -> its
                # original row in verts_cur, matched by exact position (extrude
                # never moves surviving verts). Cap verts [old_V:] stay at zero.
                old_pos = {tuple(np.round(verts_cur[j], 8)): j
                           for j in range(verts_cur.shape[0])}
                new_i, old_i = [], []
                for i in range(old_V):
                    j = old_pos.get(tuple(np.round(new_verts[i], 8)))
                    if j is not None:
                        new_i.append(i); old_i.append(j)
                warm_map = (
                    torch.tensor(new_i, dtype=torch.long, device=device),
                    torch.tensor(old_i, dtype=torch.long, device=device))
                print(f"      warm-start: remapped {len(new_i)}/{old_V} survivor "
                      f"momenta by position")
                verts_t, faces_t, opt, sched = _rebuild_opt(
                    new_verts, new_tris, step,
                    warm_opt=opt, warm_param=verts_t, warm_map=warm_map)
            else:
                kick = None
                if use_reverse_kick:
                    n_new = int(new_verts.shape[0]) - old_V
                    if n_new > 0:
                        kidx = torch.arange(old_V, int(new_verts.shape[0]),
                                            device=device)
                        kdir = torch.tensor(extrude_dir, dtype=torch.float32,
                                            device=device).unsqueeze(0).expand(
                                                n_new, 3).contiguous()
                        kick = (kidx, kdir, REVERSE_KICK_MULT)
                verts_t, faces_t, opt, sched = _rebuild_opt(
                    new_verts, new_tris, step, kick=kick)

            warmup_counter    = WARMUP_STEPS
            lap_boost_left    = LAP_WARMUP_STEPS
            depth_warmup_left = DEPTH_WARMUP_STEPS
            new_v_start       = old_V

            n_attempts += 1
            iou_at_step.clear()

            event = {
                'step':      step,
                'n_faces':   len(cluster),
                'dir':       extrude_dir.tolist(),
                'dist':       float(dist_learned),
                'dist_init':  float(extrude_dist),
                'dist_probe': float(dist_probe),
                'pre_iou':   float(pre_iou),
                'new_V':     int(new_verts.shape[0]),
                'new_F':     int(new_tris.shape[0]),
                'bnd_edges': int(n_bnd_post),
                'cc_frac':   float(cc_frac),
                'op_type':   op_type,
                'op_signals': op_signals,
            }

            if use_rollback:
                # Defer verdict by `recover_window` steps: let the injection
                # complete its delayed+compounding recovery, then judge.
                region_pts = (verts_cur[tris_cur[cluster, 0]]
                              + verts_cur[tris_cur[cluster, 1]]
                              + verts_cur[tris_cur[cluster, 2]]) / 3.0
                missing_before = region_missing_count(error_maps, region_pts, mvps)

                # Stellate rollback uses 150-step window + loss-sigma verdict
                if use_operator_menu and op_type == 'stellate':
                    loss_trigger = loss_history[-1] if loss_history else 1e9
                    recent_losses = loss_history[-50:] if len(loss_history) >= 2 else [loss_trigger]
                    loss_sigma = float(np.std(recent_losses)) if len(recent_losses) > 1 else 0.0
                    eval_win = STELLATE_RECOVER_WINDOW
                else:
                    loss_trigger = 0.0
                    loss_sigma   = 0.0
                    eval_win     = recover_window

                pending = {
                    'step':           step,
                    'eval_step':      min(step + eval_win, total_steps - 1),
                    'missing_before': missing_before,
                    'region_pts':     region_pts,
                    'snap_verts':     snap_verts,
                    'snap_tris':      snap_tris,
                    'event':          event,
                    'op_type':        op_type,
                    'loss_trigger':   float(loss_trigger),
                    'loss_sigma':     float(loss_sigma),
                }
                print(f"\n  *** TRIAL {op_type.upper()} (attempt {n_attempts}) at step {step} ***")
                print(f"      Cluster faces  : {len(cluster)}  dir="
                      f"[{extrude_dir[0]:.3f},{extrude_dir[1]:.3f},{extrude_dir[2]:.3f}]"
                      f"  dist={dist_learned:.4f} (init {extrude_dist:.4f})")
                print(f"      Mesh           : V={new_verts.shape[0]} F={new_tris.shape[0]}"
                      f"  boundary_edges={n_bnd_post} (MUST be 0)")
                print(f"      local-missing  : {missing_before}  "
                      f"(verdict at step {pending['eval_step']})")
                print()
            else:
                # Emergent-count: commit immediately, no rollback.
                n_injections += 1
                post_sils = render_views_n(ctx, verts_t, faces_t, mvps)
                post_iou  = compute_iou_n(post_sils, gt_uint8)
                event['post_iou'] = float(post_iou)
                injection_log.append(event)
                print(f"\n  *** EXTRUDE #{n_injections} at step {step} ***")
                print(f"      Cluster faces  : {len(cluster)}  dir="
                      f"[{extrude_dir[0]:.3f},{extrude_dir[1]:.3f},{extrude_dir[2]:.3f}]"
                      f"  dist={dist_learned:.4f} (init {extrude_dist:.4f})")
                print(f"      Mesh           : V={new_verts.shape[0]} F={new_tris.shape[0]}"
                      f"  boundary_edges={n_bnd_post} (MUST be 0)")
                print(f"      IoU pre->post  : {pre_iou:.4f} -> {post_iou:.4f}  "
                      f"(post is transient, recovers over next steps)")
                print()

    # ── Final evaluation ─────────────────────────────────────────────────────
    pred_sils   = render_views_n(ctx, verts_t, faces_t, mvps)
    pred_depths = render_depths_n(ctx, verts_t, faces_t, mvps)
    final_iou   = compute_iou_n(pred_sils, gt_uint8)

    verts_final = verts_t.detach().cpu().numpy().astype(np.float64)
    tris_final  = faces_t.cpu().numpy()
    assert_watertight_v3(verts_final, tris_final, "final mesh")

    return final_iou, 0.0, pred_sils, pred_depths, injection_log


# ─────────────────────────────────────────────────────────────────────────────
# Visualization (identical to v2, output names changed to v3)
# ─────────────────────────────────────────────────────────────────────────────

def save_viz_grid(gt_uint8, sils_a, sils_b, sils_c,
                  iou_a, iou_b, iou_c, out_path):
    from PIL import Image, ImageDraw
    N_v = gt_uint8.shape[0]; H, W = gt_uint8.shape[1], gt_uint8.shape[2]
    col_names = ['GT', f'A baseline ({iou_a:.3f})',
                 f'B inject ({iou_b:.3f})', f'C inj+depth ({iou_c:.3f})']
    PAD, HEADER = 4, 32
    canvas = Image.new('RGB', (4*(W+PAD)+PAD, HEADER+N_v*(H+PAD)+PAD), 'white')
    draw   = ImageDraw.Draw(canvas)
    def s2u8(s): return ((1.0-s)*255).clip(0,255).astype(np.uint8)
    col_imgs = [gt_uint8, s2u8(sils_a), s2u8(sils_b), s2u8(sils_c)]
    for ci, (cn, imgs) in enumerate(zip(col_names, col_imgs)):
        x0 = PAD + ci*(W+PAD); draw.text((x0, 8), cn, fill='black')
        for vi in range(N_v):
            canvas.paste(Image.fromarray(imgs[vi], mode='L').convert('RGB'),
                         (x0, HEADER+vi*(H+PAD)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path); print(f"  Saved viz -> {out_path}")


def _depth_to_rgb(dep, fg, vmin, vmax):
    norm = np.clip((dep - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    try:
        import matplotlib.cm as _cm
        rgb = (_cm.get_cmap('turbo')(1.0 - norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        g   = (255 * (1.0 - norm)).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    rgb[~fg] = 255; return rgb


def save_depth_grid(gt_depths, depths_b, depths_c,
                    depth_l1_b, depth_l1_c, out_path):
    from PIL import Image, ImageDraw
    N_v, H, W = gt_depths.shape
    fg_gt = gt_depths != 0; fg_b = depths_b != 0; fg_c = depths_c != 0
    fg_vals = gt_depths[fg_gt]
    vmin, vmax = float(np.percentile(fg_vals,2)), float(np.percentile(fg_vals,98))
    col_names  = ['GT depth', f'B inject (L1={depth_l1_b:.4f})',
                  f'C inj+depth (L1={depth_l1_c:.4f})']
    depth_cols = [(gt_depths, fg_gt), (depths_b, fg_b), (depths_c, fg_c)]
    PAD, HEADER = 4, 32
    canvas = Image.new('RGB', (3*(W+PAD)+PAD, HEADER+N_v*(H+PAD)+PAD), 'white')
    draw   = ImageDraw.Draw(canvas)
    for ci, (cn, (dep, fg)) in enumerate(zip(col_names, depth_cols)):
        x0 = PAD + ci*(W+PAD); draw.text((x0, 8), cn, fill='black')
        for vi in range(N_v):
            rgb = _depth_to_rgb(dep[vi], fg[vi], vmin, vmax)
            canvas.paste(Image.fromarray(rgb, mode='RGB'),
                         (x0, HEADER+vi*(H+PAD)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path); print(f"  Saved depth viz -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrude injection v3 — TopMod DLFL")
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--out_dir', default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--shape',   default='bunny',
                        help='bunny | cow | airplane | torus')
    parser.add_argument('--rollback', action='store_true',
                        help='enable keep/rollback verdict on each extrude '
                             '(default: emergent-count immediate commit)')
    parser.add_argument('--total-steps', type=int, default=TOTAL_STEPS,
                        help='optimization steps per variant')
    parser.add_argument('--recover-window', type=int, default=RECOVER_WINDOW,
                        help='rollback mode: steps before judging an extrude')
    parser.add_argument('--dist-veto', action='store_true',
                        help='use the dist probe as a misvote filter: veto an '
                             'injection when learned dist collapses toward 0, '
                             'commit survivors at the fixed init dist')
    parser.add_argument('--warm-start', action='store_true',
                        help='momentum-preserving optimizer rebuild after '
                             'extrude (position-remapped, bug-fixed)')
    parser.add_argument('--learnable-dist', action='store_true',
                        help='bake the gradient-probed per-cluster extrude '
                             'distance (size-knob mode)')
    parser.add_argument('--reverse-kick', action='store_true',
                        help='seed new cap verts with inward initial Adam '
                             'momentum to soften extrude overshoot (cold-start)')
    parser.add_argument('--menu', action='store_true',
                        help='enable operator menu (choose_operator: extrude vs '
                             'stellate) for variants B, C, and D')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    ctx        = dr.RasterizeCudaContext()
    mvps, eyes = make_6_cameras(radius=CAMERA_RADIUS, device=device)
    print(f"6-camera rig: {mvps.shape}  device={device}  res={IMG_RES}×{IMG_RES}")

    # ── GT shape ────────────────────────────────────────────────────────────
    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)
    shape_name = args.shape
    if args.shape == 'torus':
        gt_verts, gt_tris = make_torus()
    elif args.shape == 'bunny' and os.path.exists(BUNNY_PATH):
        print(f"Loading bunny from {BUNNY_PATH}")
        gt_verts, gt_tris = load_obj(BUNNY_PATH)
    else:
        _p = os.path.join(_SAMPLE_DIR, f'{args.shape}.obj')
        if os.path.exists(_p):
            print(f"Loading {args.shape} from {_p}")
            gt_verts, gt_tris = load_obj(_p)
        elif os.path.exists(BUNNY_PATH):
            print(f"WARNING: {args.shape} not found — falling back to bunny")
            gt_verts, gt_tris = load_obj(BUNNY_PATH); shape_name = 'bunny'
        else:
            print("WARNING: no shape found — using torus fallback")
            gt_verts, gt_tris = make_torus(); shape_name = 'torus'

    gt_verts = normalize_to_range(gt_verts)
    print(f"  GT mesh: V={len(gt_verts)} F={len(gt_tris)}")

    # ── GT silhouettes + depths ──────────────────────────────────────────────
    verts_gt_t = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    faces_gt_t = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)
    gt_sil_views, gt_dep_views = [], []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, verts_gt_t, faces_gt_t, mvps[i], (IMG_RES, IMG_RES))
            gt_sil_views.append(
                ((1.0 - sil_i[0,:,:,0].cpu().numpy())*255).clip(0,255).astype(np.uint8))
            gt_dep_views.append(ndc_z_i.cpu().numpy())
    gt_uint8  = np.stack(gt_sil_views, axis=0)
    gt_depths = np.stack(gt_dep_views, axis=0)
    print("  GT silhouettes + depths rendered")

    # ── Initial mesh (2× CC icosphere) ──────────────────────────────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn+mx)/2.0) * (2.0/max(mx-mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    print(f"  Init mesh: V={len(init_verts)} F={len(init_tris)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant A
    # ══════════════════════════════════════════════════════════════════════════
    TS   = args.total_steps
    menu_sfx = "+menu" if args.menu else ""
    mode = (f"rollback (window={args.recover_window}){menu_sfx}"
            if args.rollback else
            f"dist-veto (misvote filter, fixed dist){menu_sfx}" if args.dist_veto else
            f"warm-start (position-remapped momentum){menu_sfx}" if args.warm_start else
            f"learnable-dist (baked size knob){menu_sfx}" if args.learnable_dist else
            f"reverse-kick (cold-start + inward new-vert momentum){menu_sfx}"
            if args.reverse_kick else
            f"emergent-count (cold-start, fixed dist){menu_sfx}")
    print("\n" + "="*70)
    print(f"VARIANT A — Baseline {TS}-step, no injection, no depth")
    print("="*70)
    t0 = time.time()
    iou_a, sils_a = run_baseline(
        ctx, init_verts, init_tris, gt_uint8, mvps, device, n_steps=TS)
    ta = time.time() - t0
    print(f"  [A] IoU={iou_a:.4f}  time={ta:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant B
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"VARIANT B — TopMod injection (max {MAX_INJECTIONS}), "
          f"no depth, {TS} steps, {mode}")
    print("="*70)
    t0 = time.time()
    iou_b, _, sils_b, depths_b, log_b = run_with_injection_v3(
        ctx, init_verts, init_tris,
        gt_uint8, None, mvps, device,
        use_depth=False, max_injections=MAX_INJECTIONS, total_steps=TS,
        use_rollback=args.rollback, recover_window=args.recover_window,
        use_dist_veto=args.dist_veto, use_warm_start=args.warm_start,
        use_learnable_dist=args.learnable_dist,
        use_reverse_kick=args.reverse_kick,
        use_operator_menu=args.menu)
    tb = time.time() - t0
    depth_l1_b = compute_depth_l1_n(depths_b, gt_depths, gt_uint8)
    print(f"  [B] IoU={iou_b:.4f}  depth_L1={depth_l1_b:.4f}  time={tb:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant C
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"VARIANT C — TopMod injection + depth supervision, {TS} steps, {mode}")
    print("="*70)
    t0 = time.time()
    iou_c, _, sils_c, depths_c, log_c = run_with_injection_v3(
        ctx, init_verts, init_tris,
        gt_uint8, gt_depths, mvps, device,
        use_depth=True, max_injections=MAX_INJECTIONS, total_steps=TS,
        use_rollback=args.rollback, recover_window=args.recover_window,
        use_dist_veto=args.dist_veto, use_warm_start=args.warm_start,
        use_learnable_dist=args.learnable_dist,
        use_reverse_kick=args.reverse_kick,
        use_operator_menu=args.menu)
    tc = time.time() - t0
    depth_l1_c = compute_depth_l1_n(depths_c, gt_depths, gt_uint8)
    print(f"  [C] IoU={iou_c:.4f}  depth_L1={depth_l1_c:.4f}  time={tc:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant D (inject+depth+menu) — only when --menu is active
    # ══════════════════════════════════════════════════════════════════════════
    iou_d = depth_l1_d = None
    sils_d = depths_d = log_d = None
    td = 0.0
    if args.menu:
        print("\n" + "="*70)
        print(f"VARIANT D — TopMod inject+depth+menu, {TS} steps, {mode}")
        print("="*70)
        t0 = time.time()
        iou_d, _, sils_d, depths_d, log_d = run_with_injection_v3(
            ctx, init_verts, init_tris,
            gt_uint8, gt_depths, mvps, device,
            use_depth=True, max_injections=MAX_INJECTIONS, total_steps=TS,
            use_rollback=args.rollback, recover_window=args.recover_window,
            use_dist_veto=args.dist_veto, use_warm_start=args.warm_start,
            use_learnable_dist=args.learnable_dist,
            use_reverse_kick=args.reverse_kick,
            use_operator_menu=True)
        td = time.time() - t0
        depth_l1_d = compute_depth_l1_n(depths_d, gt_depths, gt_uint8)
        print(f"  [D] IoU={iou_d:.4f}  depth_L1={depth_l1_d:.4f}  time={td:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("RESULTS SUMMARY (v3 — TopMod DLFL extrude)")
    print("="*70)
    print(f"  A baseline no-inj  : IoU={iou_a:.4f}")
    print(f"  B inject no-depth  : IoU={iou_b:.4f}  depth_L1={depth_l1_b:.4f}")
    print(f"  C inject+depth     : IoU={iou_c:.4f}  depth_L1={depth_l1_c:.4f}")
    if iou_d is not None:
        print(f"  D inject+dep+menu  : IoU={iou_d:.4f}  depth_L1={depth_l1_d:.4f}")
    print(f"  B vs A             : {iou_b-iou_a:+.4f}")
    print(f"  C vs A             : {iou_c-iou_a:+.4f}")
    if iou_d is not None:
        print(f"  D vs A             : {iou_d-iou_a:+.4f}")
    print(f"  C depth vs B depth : {depth_l1_c-depth_l1_b:+.4f}")
    t_total = ta + tb + tc + (td if iou_d is not None else 0.0)
    td_str  = f" D={td:.1f}" if iou_d is not None else ""
    print(f"  Total time         : {t_total:.1f}s  (A={ta:.1f} B={tb:.1f} C={tc:.1f}{td_str})")

    log_tags = [('B', log_b), ('C', log_c)]
    if log_d is not None:
        log_tags.append(('D', log_d))
    for tag, log in log_tags:
        if not log: print(f"\n  [{tag}] No injections.")
        else:
            print(f"\n  [{tag}] {len(log)} injection(s):")
            for k, ev in enumerate(log):
                d = ev['dir']
                print(f"    #{k+1} step={ev['step']}  faces={ev['n_faces']}  "
                      f"dir=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f}]  "
                      f"dist={ev['dist']:.4f}  "
                      f"IoU:{ev['pre_iou']:.4f}→{ev['post_iou']:.4f}  "
                      f"V={ev['new_V']} F={ev['new_F']}  "
                      f"bnd={ev['bnd_edges']}  cc={ev['cc_frac']:.2f}")

    if iou_c >= 0.96:
        print(f"\n  TARGET MET: C IoU {iou_c:.4f} ≥ 0.96  ✓")
    else:
        print(f"\n  Target 0.96 not met (C={iou_c:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════════════════
    save_viz_grid(gt_uint8, sils_a, sils_b, sils_c, iou_a, iou_b, iou_c,
                  os.path.join(args.out_dir, f'viz_extrude_v3_{shape_name}.png'))
    save_depth_grid(gt_depths, depths_b, depths_c, depth_l1_b, depth_l1_c,
                    os.path.join(args.out_dir,
                                 f'viz_extrude_v3_{shape_name}_depth.png'))
    print(f"\nDone. Total={ta+tb+tc:.1f}s")


if __name__ == '__main__':
    main()
