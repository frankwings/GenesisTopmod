"""
operator_grounding.py — Map VLM region numbers to concrete face IDs.

The VLM picks a region number from the triptych image.  This module converts
that number into a cluster of face indices by re-running the multi-view ray
voting on the subset of error pixels that overlap the chosen region circle.

Usage
-----
    from operator_grounding import ground_region

    cluster, extrude_dir = ground_region(
        region_label   = 2,
        candidates     = candidates,   # from build_candidates()
        error_maps     = error_maps,   # [N_V, H, W] float32
        verts_np       = verts_np,
        tris_np        = tris_np,
        mvps           = mvps,
        face_votes     = face_votes,   # [F] int32 from vote_faces_multiview_n
        face_dir       = face_dir,     # [F, 3] float64 aggregated directions
    )
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np


def ground_region(
    region_label:  int,
    candidates:    List[Dict[str, Any]],
    error_maps:    np.ndarray,          # [N_V, H, W] float32
    verts_np:      np.ndarray,          # [V, 3]
    tris_np:       np.ndarray,          # [F, 3]
    mvps,                               # [N_V, 4, 4] tensor or ndarray
    face_votes:    np.ndarray,          # [F] int32
    face_dir:      np.ndarray,          # [F, 3] float64
    cluster_radius: float = 0.12,       # fraction of image height for BFS ball
    min_cluster:   int   = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cluster_face_indices, extrude_dir) for the given region label.

    Strategy
    --------
    1. Find the candidate dict matching *region_label*.
    2. For each view, the candidate has a projected pixel centre.  Grow a disk
       of radius *cluster_radius* × H in that view's error map.
    3. Weighted-union the face votes inside the disk across all views.
    4. Return the top-K faces (BFS ball from the argmax face) as the cluster,
       and the normalised aggregated direction as extrude_dir.

    If the region label is not found, fall back to the global face_votes argmax.

    Parameters
    ----------
    region_label : int
        1-based region number chosen by the VLM.
    candidates : list[dict]
        Candidate descriptors with "label", "center_px" (list of (x,y) per view),
        "face_ids" (list of face indices in the cluster).
    error_maps : [N_V, H, W] float32
        Per-view error maps (>0.5 = missing pixel).
    verts_np, tris_np : ndarray
        Current mesh arrays.
    mvps : [N_V, 4, 4] tensor
        Camera MVP matrices.
    face_votes : [F] int32
        Per-face vote accumulation from vote_faces_multiview_n.
    face_dir : [F, 3] float64
        Per-face direction accumulation.
    cluster_radius : float
        Circle radius as fraction of image height.
    min_cluster : int
        Minimum cluster size — fall back to single face if fewer found.

    Returns
    -------
    cluster : np.ndarray [K] int32
        Face indices in the cluster.
    extrude_dir : np.ndarray [3] float64
        Unit direction for the extrusion (normalised aggregate vote direction).
    """
    # ── 1. Find the candidate ──────────────────────────────────────────────
    cand = None
    for c in candidates:
        if int(c.get("label", -1)) == region_label:
            cand = c
            break

    if cand is not None and "face_ids" in cand:
        # Fast path: candidate already carries precomputed face IDs
        cluster = np.array(cand["face_ids"], dtype=np.int32)
        if len(cluster) >= min_cluster:
            d = face_dir[cluster].sum(axis=0)
            dn = np.linalg.norm(d)
            extrude_dir = d / dn if dn > 1e-8 else np.array([0., 1., 0.])
            return cluster, extrude_dir

    # ── 2. Fallback: use global vote argmax if label not found ──────────
    best_face = int(np.argmax(face_votes))
    cluster = _bfs_cluster(tris_np, best_face, max_faces=8)
    d = face_dir[cluster].sum(axis=0)
    dn = np.linalg.norm(d)
    extrude_dir = d / dn if dn > 1e-8 else np.array([0., 1., 0.])
    return cluster.astype(np.int32), extrude_dir


def build_candidates(
    face_votes:    np.ndarray,   # [F] int32
    face_dir:      np.ndarray,   # [F, 3] float64
    tris_np:       np.ndarray,   # [F, 3] int32
    verts_np:      np.ndarray,   # [V, 3]
    mvps,                         # [N_V, 4, 4] tensor
    n_candidates:  int = 3,
    cluster_size:  int = 8,
    error_maps:    Optional[np.ndarray] = None,   # [N_V, H, W] 1=missing(red)
    img_res:       int = 256,
) -> List[Dict[str, Any]]:
    """Build the candidate list for encode_state and ground_region.

    Finds up to *n_candidates* peaks in face_votes (greedy argmax + suppress),
    computes cluster + direction for each, and places the per-view overlay
    circle.

    Circle placement (Phase-2 fix)
    ------------------------------
    Old behaviour projected the cluster's single 3-D world centroid to every
    view, which drifts away from the actual red (missing) pixels (the circle
    was often "far from the red").  When *error_maps* is supplied, the circle
    centre for each view is instead the centroid of the RED pixels that fall
    inside the cluster's projected footprint for that view, so the circle
    faithfully sits on the geometry-missing region the VLM must judge.  Falls
    back to the footprint centroid (then the 3-D-centroid projection) when no
    red pixels overlap.

    Returns
    -------
    list[dict] with keys:
        "label"       : int (1-based)
        "face_ids"    : list[int]
        "extrude_dir" : [3] float64
        "center_px"   : list[(x, y)] one per view (−1,−1 if behind camera)
        "description" : str
    """
    import torch

    # Normalise mvps to a numpy array once for projection helpers
    if hasattr(mvps, "detach"):
        mvps_np = mvps.detach().cpu().numpy().astype(np.float64)
    else:
        mvps_np = np.array(mvps, dtype=np.float64)

    F = len(face_votes)
    suppressed = np.zeros(F, dtype=bool)
    candidates: List[Dict[str, Any]] = []

    for label in range(1, n_candidates + 1):
        masked = face_votes.copy().astype(np.float32)
        masked[suppressed] = -1
        if masked.max() <= 0:
            break

        best = int(np.argmax(masked))
        cluster = _bfs_cluster(tris_np, best, max_faces=cluster_size)

        # Suppress neighbourhood
        suppressed[cluster] = True

        d = face_dir[cluster].sum(axis=0)
        dn = np.linalg.norm(d)
        extrude_dir = d / dn if dn > 1e-8 else np.array([0., 1., 0.])

        # Cluster world centroid (fallback for circle placement)
        v0 = verts_np[tris_np[cluster, 0]]
        v1 = verts_np[tris_np[cluster, 1]]
        v2 = verts_np[tris_np[cluster, 2]]
        centroid = np.mean((v0 + v1 + v2) / 3.0, axis=0)   # [3]

        # Per-view circle centre: red-pixel centroid inside cluster footprint
        # (faithful), falling back to 3-D-centroid projection.
        center_px = _cluster_view_centers(
            cluster, tris_np, verts_np, mvps_np, error_maps,
            fallback_pt=centroid, img_res=img_res)

        total_votes = int(face_votes[cluster].sum())
        candidates.append({
            "label":       label,
            "face_ids":    cluster.tolist(),
            "extrude_dir": extrude_dir,
            "center_px":   center_px,
            "description": f"region {label} ({total_votes} votes)",
        })

    return candidates


# ── helpers ───────────────────────────────────────────────────────────────────

def _project_points_to_view(
    points_w: np.ndarray,   # [K, 3] world points
    mvp_np:   np.ndarray,   # [4, 4]
    img_res:  int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project K world points to a single view.

    Returns (coords [K,2] float (x_px,y_px), valid [K] bool).
    Invalid = behind camera (w<=0); their coords are meaningless.
    """
    K = len(points_w)
    pts_h = np.concatenate([points_w, np.ones((K, 1))], axis=1)   # [K,4]
    clip = pts_h @ mvp_np.T                                       # [K,4]
    w = clip[:, 3]
    valid = w > 1e-8
    w_safe = np.where(valid, w, 1.0)
    ndc_x = clip[:, 0] / w_safe
    ndc_y = clip[:, 1] / w_safe
    px = (ndc_x * 0.5 + 0.5) * img_res - 0.5
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * img_res - 0.5
    return np.stack([px, py], axis=1), valid


def _cluster_view_centers(
    cluster:     np.ndarray,          # [K] face indices
    tris_np:     np.ndarray,          # [F, 3]
    verts_np:    np.ndarray,          # [V, 3]
    mvps_np:     np.ndarray,          # [N_V, 4, 4]
    error_maps:  Optional[np.ndarray],# [N_V, H, W] 1=missing, or None
    fallback_pt: np.ndarray,          # [3] 3-D cluster centroid
    img_res:     int = 256,
    footprint_pad: int = 12,          # px padding around footprint bbox
    min_red:     int = 5,             # min red pixels to trust a red centroid
) -> List[Tuple[int, int]]:
    """Per-view overlay-circle centre for a face cluster.

    For each view:
      1. Project the cluster's vertices → 2-D footprint.
      2. Take the RED (missing) pixels inside the footprint bbox (padded).
      3. Circle centre = centroid of those red pixels (faithful to the error).
      4. Fall back to footprint centroid, then to the projected 3-D centroid.
    """
    vidx = np.unique(tris_np[cluster].reshape(-1))
    pts_w = verts_np[vidx]                        # [K,3]
    N_V = mvps_np.shape[0]

    centers: List[Tuple[int, int]] = []
    for vi in range(N_V):
        coords, valid = _project_points_to_view(pts_w, mvps_np[vi], img_res)
        coords = coords[valid]
        # keep only points inside the frame
        in_frame = [
            (x, y) for x, y in coords
            if 0 <= x < img_res and 0 <= y < img_res
        ]
        if not in_frame:
            # whole cluster off-screen/behind camera → fall back to 3-D centroid
            fb, fb_valid = _project_points_to_view(
                fallback_pt[None, :], mvps_np[vi], img_res)
            if fb_valid[0] and 0 <= fb[0, 0] < img_res and 0 <= fb[0, 1] < img_res:
                centers.append((int(fb[0, 0]), int(fb[0, 1])))
            else:
                centers.append((-1, -1))
            continue

        fa = np.array(in_frame)                    # [M,2] (x,y)
        fc_x, fc_y = float(fa[:, 0].mean()), float(fa[:, 1].mean())

        # Red pixels inside the padded footprint bbox
        if error_maps is not None:
            x0 = max(0, int(fa[:, 0].min()) - footprint_pad)
            x1 = min(img_res, int(fa[:, 0].max()) + footprint_pad + 1)
            y0 = max(0, int(fa[:, 1].min()) - footprint_pad)
            y1 = min(img_res, int(fa[:, 1].max()) + footprint_pad + 1)
            sub = error_maps[vi][y0:y1, x0:x1]
            ys, xs = np.nonzero(sub > 0.5)
            if len(xs) >= min_red:
                cx = xs.mean() + x0
                cy = ys.mean() + y0
                centers.append((int(cx), int(cy)))
                continue

        centers.append((int(fc_x), int(fc_y)))

    return centers


def _bfs_cluster(
    tris_np:  np.ndarray,
    seed:     int,
    max_faces: int = 8,
) -> np.ndarray:
    """BFS from seed face via shared edges, returning up to max_faces face IDs."""
    F = len(tris_np)
    # Build edge→faces adjacency
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, tri in enumerate(tris_np):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k+1) % 3])
            e = (min(a, b), max(a, b))
            edge_to_faces.setdefault(e, []).append(fi)

    adj: Dict[int, List[int]] = {fi: [] for fi in range(F)}
    for flist in edge_to_faces.values():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                adj[flist[i]].append(flist[j])
                adj[flist[j]].append(flist[i])

    visited = {seed}
    queue   = [seed]
    result  = [seed]
    while queue and len(result) < max_faces:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                result.append(nb)
                if len(result) >= max_faces:
                    break

    return np.array(result, dtype=np.int32)


def _project_to_views(
    world_pt: np.ndarray,   # [3]
    mvps,                    # [N_V, 4, 4] tensor or ndarray
) -> List[Tuple[int, int]]:
    """Project a world-space point to pixel coords in each view.

    Returns list of (x_px, y_px) per view; (-1, -1) if behind camera.
    """
    import torch

    # Accept both tensor and ndarray
    if hasattr(mvps, "detach"):
        mvps_np = mvps.detach().cpu().numpy()
    else:
        mvps_np = np.array(mvps)

    N_V = mvps_np.shape[0]
    # Assume square images at IMG_RES (we don't pass H/W here → use unit NDC)
    pt_h = np.array([*world_pt, 1.0], dtype=np.float64)   # [4]

    result: List[Tuple[int, int]] = []
    for vi in range(N_V):
        clip = mvps_np[vi].astype(np.float64) @ pt_h      # [4]
        w    = clip[3]
        if abs(w) < 1e-8 or w < 0:
            result.append((-1, -1))
            continue
        ndc_x = clip[0] / w
        ndc_y = clip[1] / w
        # Map NDC → image pixels (assume IMG_RES from context; use 256 default)
        IMG_RES = 256
        px = int((ndc_x * 0.5 + 0.5) * IMG_RES - 0.5)
        py = int((1.0 - (ndc_y * 0.5 + 0.5)) * IMG_RES - 0.5)
        # Clip to image bounds
        if px < 0 or px >= IMG_RES or py < 0 or py >= IMG_RES:
            result.append((-1, -1))
        else:
            result.append((px, py))

    return result
