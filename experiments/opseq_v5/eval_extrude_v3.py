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

MAX_INJECTIONS   = 5
TOTAL_STEPS      = 800
EVAL_INTERVAL    = 5
PLATEAU_STEPS    = 20
PLATEAU_EPS      = 0.005
WARMUP_STEPS     = 20
LAP_WARMUP_STEPS = 50
DEPTH_WARMUP_STEPS = 80   # phase-in depth loss after each injection (0->full)
EXTRUDE_BBOX_FRAC = 0.12
MIN_STEP_FOR_INJ  = 60

W_LAP       = 0.10
W_LAP_BOOST = 0.40
W_EDGE      = 0.01
W_DEPTH     = 0.35


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
    n_injections   = 0
    warmup_counter = 0
    lap_boost_left = 0
    depth_warmup_left = 0
    new_v_start    = -1
    iou_at_step:   List[Tuple[int, float]] = []

    for step in range(total_steps):

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
        if (n_injections < max_injections
                and step >= MIN_STEP_FOR_INJ
                and step <= int(total_steps * 0.65)
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

            # ── TopMod DLFL cluster extrusion ──────────────────────────────
            try:
                new_verts, new_tris, old_V = topmod_extrude_cluster(
                    verts_cur, tris_cur,
                    cluster, extrude_dir, extrude_dist)
            except Exception as e:
                print(f"  [WARN] TopMod cluster extrude failed: {e} — skipping")
                continue

            # ── Hard watertight assertion ───────────────────────────────────
            n_bnd_post, n_nm_post = dlfl_boundary_stats(new_verts, new_tris)
            try:
                assert_watertight_v3(new_verts, new_tris,
                                     f"after inject {n_injections+1}")
            except AssertionError as e:
                print(f"  [WARN] {e} — skipping injection")
                continue

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
            verts_t = torch.tensor(new_verts, dtype=torch.float32,
                                   device=device).requires_grad_(True)
            faces_t = torch.tensor(new_tris,  dtype=torch.int32, device=device)

            opt = torch.optim.Adam([verts_t], lr=LR)
            remaining = total_steps - step - 1
            if remaining > 0:
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=remaining, eta_min=LR_MIN)

            warmup_counter    = WARMUP_STEPS
            lap_boost_left    = LAP_WARMUP_STEPS
            depth_warmup_left = DEPTH_WARMUP_STEPS
            new_v_start       = old_V

            post_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            post_iou  = compute_iou_n(post_sils, gt_uint8)

            n_injections += 1
            iou_at_step.clear()
            iou_at_step.append((step, post_iou))

            event = {
                'step':       step,
                'n_faces':    len(cluster),
                'dir':        extrude_dir.tolist(),
                'dist':       float(extrude_dist),
                'pre_iou':    float(pre_iou),
                'post_iou':   float(post_iou),
                'new_V':      int(new_verts.shape[0]),
                'new_F':      int(new_tris.shape[0]),
                'bnd_edges':  int(n_bnd_post),
                'cc_frac':    float(cc_frac),
            }
            injection_log.append(event)

            print(f"\n  *** TOPMOD CLUSTER EXTRUDE #{n_injections} at step {step} ***")
            print(f"      Cluster faces  : {len(cluster)}")
            print(f"      Direction      : [{extrude_dir[0]:.3f},"
                  f"{extrude_dir[1]:.3f},{extrude_dir[2]:.3f}]")
            print(f"      Distance       : {extrude_dist:.4f}")
            print(f"      Pre-inject IoU : {pre_iou:.4f}")
            print(f"      Post-inject IoU: {post_iou:.4f}")
            print(f"      Mesh           : V={new_verts.shape[0]} "
                  f"F={new_tris.shape[0]}")
            print(f"      boundary_edges : {n_bnd_post}  "
                  f"(MUST be 0)")
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
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    ctx        = dr.RasterizeCudaContext()
    mvps, eyes = make_6_cameras(radius=CAMERA_RADIUS, device=device)
    print(f"6-camera rig: {mvps.shape}  device={device}  res={IMG_RES}×{IMG_RES}")

    # ── GT shape ────────────────────────────────────────────────────────────
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
    print("\n" + "="*70)
    print(f"VARIANT A — Baseline {TOTAL_STEPS}-step, no injection, no depth")
    print("="*70)
    t0 = time.time()
    iou_a, sils_a = run_baseline(
        ctx, init_verts, init_tris, gt_uint8, mvps, device)
    ta = time.time() - t0
    print(f"  [A] IoU={iou_a:.4f}  time={ta:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant B
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"VARIANT B — TopMod injection (max {MAX_INJECTIONS}), "
          f"no depth, {TOTAL_STEPS} steps")
    print("="*70)
    t0 = time.time()
    iou_b, _, sils_b, depths_b, log_b = run_with_injection_v3(
        ctx, init_verts, init_tris,
        gt_uint8, None, mvps, device,
        use_depth=False, max_injections=MAX_INJECTIONS, total_steps=TOTAL_STEPS)
    tb = time.time() - t0
    depth_l1_b = compute_depth_l1_n(depths_b, gt_depths, gt_uint8)
    print(f"  [B] IoU={iou_b:.4f}  depth_L1={depth_l1_b:.4f}  time={tb:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Variant C
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"VARIANT C — TopMod injection + depth supervision, {TOTAL_STEPS} steps")
    print("="*70)
    t0 = time.time()
    iou_c, _, sils_c, depths_c, log_c = run_with_injection_v3(
        ctx, init_verts, init_tris,
        gt_uint8, gt_depths, mvps, device,
        use_depth=True, max_injections=MAX_INJECTIONS, total_steps=TOTAL_STEPS)
    tc = time.time() - t0
    depth_l1_c = compute_depth_l1_n(depths_c, gt_depths, gt_uint8)
    print(f"  [C] IoU={iou_c:.4f}  depth_L1={depth_l1_c:.4f}  time={tc:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("RESULTS SUMMARY (v3 — TopMod DLFL extrude)")
    print("="*70)
    print(f"  A baseline no-inj  : IoU={iou_a:.4f}")
    print(f"  B inject no-depth  : IoU={iou_b:.4f}  depth_L1={depth_l1_b:.4f}")
    print(f"  C inject+depth     : IoU={iou_c:.4f}  depth_L1={depth_l1_c:.4f}")
    print(f"  B vs A             : {iou_b-iou_a:+.4f}")
    print(f"  C vs A             : {iou_c-iou_a:+.4f}")
    print(f"  C depth vs B depth : {depth_l1_c-depth_l1_b:+.4f}")
    print(f"  Total time         : {ta+tb+tc:.1f}s  (A={ta:.1f} B={tb:.1f} C={tc:.1f})")

    for tag, log in [('B', log_b), ('C', log_c)]:
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
