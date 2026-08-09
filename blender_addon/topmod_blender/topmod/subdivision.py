"""
Catmull-Clark subdivision for DLFLMesh.

catmull_clark(mesh) -> DLFLMesh

Produces a new DLFLMesh with the subdivided topology.  The original mesh
is not modified.

Algorithm (standard Catmull-Clark):
1. For each face f, compute face point F = centroid of f's vertices.
2. For each edge e, compute edge point E = avg(endpoints + adjacent face points).
3. For each vertex v, compute new position using the standard CC formula.
4. Reconstruct the mesh: each n-gon face becomes n quads.

The resulting mesh is always all-quads (4-valent faces).
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from .dlfl import DLFLMesh, Vertex
from .primitives import _build_mesh


# ─────────────────────────────────────────────────────────────────────────────

def catmull_clark(mesh: DLFLMesh) -> DLFLMesh:
    """
    Apply one round of Catmull-Clark subdivision.

    Returns a brand-new DLFLMesh (the input mesh is unchanged).
    """
    # ── Step 1: face points ────────────────────────────────────────────
    face_points: Dict[int, Tuple[float, float, float]] = {}
    for face in mesh.iter_faces():
        verts = face.vertices()
        n = len(verts)
        if n == 0:
            face_points[face.id] = (0.0, 0.0, 0.0)
            continue
        fx = sum(v.x for v in verts) / n
        fy = sum(v.y for v in verts) / n
        fz = sum(v.z for v in verts) / n
        face_points[face.id] = (fx, fy, fz)

    # ── Step 2: edge points ────────────────────────────────────────────
    edge_points: Dict[int, Tuple[float, float, float]] = {}
    for edge in mesh.iter_edges():
        v0, v1 = edge.vertices()
        f0, f1 = edge.faces()
        ep = [v0.x, v0.y, v0.z]
        ep[0] += v1.x; ep[1] += v1.y; ep[2] += v1.z
        count = 2
        if f0 is not None:
            fp = face_points[f0.id]
            ep[0] += fp[0]; ep[1] += fp[1]; ep[2] += fp[2]
            count += 1
        if f1 is not None:
            fp = face_points[f1.id]
            ep[0] += fp[0]; ep[1] += fp[1]; ep[2] += fp[2]
            count += 1
        edge_points[edge.id] = (ep[0] / count, ep[1] / count, ep[2] / count)

    # ── Step 3: new vertex positions (CC formula) ──────────────────────
    # For interior vertex v of valence n:
    #   new_v = (Q/n) + (2R/n) + ((n-3)/n) * v
    # where Q = avg of adjacent face points, R = avg of adjacent edge midpoints
    new_vertex_pos: Dict[int, Tuple[float, float, float]] = {}

    for v in mesh.iter_vertices():
        outgoing = v.outgoing_halfedges()
        n = len(outgoing)
        if n == 0:
            new_vertex_pos[v.id] = (v.x, v.y, v.z)
            continue

        # Collect adjacent face points
        adj_faces = []
        for he in outgoing:
            if he.face is not None:
                adj_faces.append(face_points[he.face.id])

        # Collect adjacent edge midpoints (midpoint of each edge from v)
        adj_edge_mids = []
        for he in outgoing:
            if he.destination is not None:
                dst = he.destination
                adj_edge_mids.append(((v.x + dst.x) / 2,
                                      (v.y + dst.y) / 2,
                                      (v.z + dst.z) / 2))

        if not adj_faces or not adj_edge_mids:
            new_vertex_pos[v.id] = (v.x, v.y, v.z)
            continue

        nf = len(adj_faces)
        Q = (sum(f[0] for f in adj_faces) / nf,
             sum(f[1] for f in adj_faces) / nf,
             sum(f[2] for f in adj_faces) / nf)

        ne = len(adj_edge_mids)
        R = (sum(m[0] for m in adj_edge_mids) / ne,
             sum(m[1] for m in adj_edge_mids) / ne,
             sum(m[2] for m in adj_edge_mids) / ne)

        # Use n = average of face-adjacency count (= valence for interior vertex)
        val = n
        nx = (Q[0] + 2 * R[0] + (val - 3) * v.x) / val
        ny = (Q[1] + 2 * R[1] + (val - 3) * v.y) / val
        nz = (Q[2] + 2 * R[2] + (val - 3) * v.z) / val
        new_vertex_pos[v.id] = (nx, ny, nz)

    # ── Step 4: build new mesh ─────────────────────────────────────────
    # Index scheme:
    #   0 .. V-1               : moved original vertices
    #   V .. V+E-1             : edge points
    #   V+E .. V+E+F-1         : face points

    orig_verts = list(mesh.vertices.values())
    orig_edges = list(mesh.edges.values())
    orig_faces = list(mesh.faces.values())

    V = len(orig_verts)
    E = len(orig_edges)
    F = len(orig_faces)

    # Build position list
    positions: List[Tuple[float, float, float]] = []

    vid_to_idx: Dict[int, int] = {}
    for i, v in enumerate(orig_verts):
        vid_to_idx[v.id] = i
        positions.append(new_vertex_pos[v.id])

    eid_to_idx: Dict[int, int] = {}
    for i, e in enumerate(orig_edges):
        eid_to_idx[e.id] = V + i
        positions.append(edge_points[e.id])

    fid_to_idx: Dict[int, int] = {}
    for i, f in enumerate(orig_faces):
        fid_to_idx[f.id] = V + E + i
        positions.append(face_points[f.id])

    # Each original n-gon face becomes n quads.
    # Quad vertices: face_point, edge_point(prev_edge), orig_vertex, edge_point(next_edge)
    new_faces: List[List[int]] = []

    for face in orig_faces:
        hes = face.halfedges()
        fp_idx = fid_to_idx[face.id]

        for he in hes:
            v_idx  = vid_to_idx[he.origin.id]
            ep_idx = eid_to_idx[he.edge.id]
            pe_idx = eid_to_idx[he.prev.edge.id]

            # Quad: face_point → prev_edge_point → orig_vertex → curr_edge_point
            new_faces.append([fp_idx, pe_idx, v_idx, ep_idx])

    return _build_mesh(positions, new_faces)
