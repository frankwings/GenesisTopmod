"""
Global remeshing schemes: dual and Doo-Sabin subdivision.

Clean-room implementations from documented semantics only
(DLFLDual.hh / DLFLSubdiv.hh headers + Doo & Sabin 1978) — no GPL code.

dual(mesh)      -> DLFLMesh   V'=F, E'=E, F'=V; involution up to isomorphism
doo_sabin(mesh) -> DLFLMesh   V'=2E, E'=4E, F'=V+E+F; all corners cut

Both produce a brand-new DLFLMesh; the input mesh is unchanged.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from .dlfl import DLFLMesh
from .primitives import _build_mesh


# ─────────────────────────────────────────────────────────────────────────────
# Dual
# ─────────────────────────────────────────────────────────────────────────────

def dual(mesh: DLFLMesh) -> DLFLMesh:
    """
    Combinatorial dual: one vertex per face (at the centroid), one face per
    vertex (the ring of adjacent face-vertices around it).

    Oracle: V' = F, E' = E, F' = V; χ and genus preserved.
    dual(dual(M)) is combinatorially isomorphic to M.
    """
    # One dual vertex per primal face (centroid)
    positions: List[Tuple[float, float, float]] = []
    fid_to_idx: Dict[int, int] = {}
    for f in mesh.iter_faces():
        fid_to_idx[f.id] = len(positions)
        positions.append(f.centroid())

    # One dual face per primal vertex: adjacent faces around the vertex.
    # Vertex fan order (he -> he.twin.next) walks clockwise around the
    # vertex for CCW-wound faces, so reverse it to keep CCW output winding.
    faces: List[List[int]] = []
    for v in mesh.iter_vertices():
        ring = [fid_to_idx[he.face.id] for he in v.outgoing_halfedges()]
        ring.reverse()
        faces.append(ring)

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Doo-Sabin
# ─────────────────────────────────────────────────────────────────────────────

def doo_sabin(mesh: DLFLMesh) -> DLFLMesh:
    """
    One round of Doo-Sabin subdivision (corner-cutting).

    For every corner (half-edge he = corner of he.face at he.origin) a new
    vertex is created at the average of: the corner vertex, the face
    centroid, and the midpoints of the two face edges incident to that
    corner.

    New faces:
      face-face   : per primal face, its corner points in face order
      edge-face   : per primal edge, quad joining the 4 corner points of
                    the two incident half-edges
      vertex-face : per primal vertex, its corner points around the vertex

    Oracle: V' = 2E (one per half-edge), E' = 4E, F' = V + E + F;
    χ and genus preserved.
    """
    positions: List[Tuple[float, float, float]] = []
    corner_idx: Dict[int, int] = {}   # halfedge id -> new vertex index

    # ── Corner points ──────────────────────────────────────────────────
    for f in mesh.iter_faces():
        cx, cy, cz = f.centroid()
        for he in f.halfedges():
            v = he.origin
            p = he.prev.origin          # previous vertex in face loop
            nx = he.twin.origin         # next vertex (= he.destination)
            m1 = ((v.x + p.x) / 2, (v.y + p.y) / 2, (v.z + p.z) / 2)
            m2 = ((v.x + nx.x) / 2, (v.y + nx.y) / 2, (v.z + nx.z) / 2)
            corner_idx[he.id] = len(positions)
            positions.append((
                (v.x + cx + m1[0] + m2[0]) / 4.0,
                (v.y + cy + m1[1] + m2[1]) / 4.0,
                (v.z + cz + m1[2] + m2[2]) / 4.0,
            ))

    faces: List[List[int]] = []

    # ── Face-faces: same winding as the primal face ────────────────────
    for f in mesh.iter_faces():
        faces.append([corner_idx[he.id] for he in f.halfedges()])

    # ── Edge-faces: quad per primal edge ───────────────────────────────
    # For edge (h, t=h.twin): corners C(h.next), C(h) on h's side and
    # C(t.next), C(t) on t's side — wound opposite to the face-faces.
    for e in mesh.iter_edges():
        h = e.he0
        t = e.he1
        faces.append([
            corner_idx[h.next.id],
            corner_idx[h.id],
            corner_idx[t.next.id],
            corner_idx[t.id],
        ])

    # ── Vertex-faces: corners around each vertex ───────────────────────
    # Fan order (he -> he.twin.next) pairs C(h)->C(t.next) in the same
    # direction as the edge-faces, so reverse it (as in dual()) to give
    # each shared edge opposite directions in its two incident faces.
    for v in mesh.iter_vertices():
        ring = [corner_idx[he.id] for he in v.outgoing_halfedges()]
        ring.reverse()
        faces.append(ring)

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Simplest (mid-edge / Peters-Reif) subdivision
# ─────────────────────────────────────────────────────────────────────────────

def simplest_subdivide(mesh: DLFLMesh) -> DLFLMesh:
    """
    Mid-edge ("simplest") subdivision: one new vertex per edge midpoint;
    faces = shrunk original faces + one face per original vertex.

    Oracle: V' = E, E' = 2E, F' = F + V; χ and genus preserved.
    (cube → cuboctahedron)
    """
    positions: List[Tuple[float, float, float]] = []
    eid_to_idx: Dict[int, int] = {}
    for e in mesh.iter_edges():
        v0, v1 = e.vertices()
        eid_to_idx[e.id] = len(positions)
        positions.append(((v0.x + v1.x) / 2,
                          (v0.y + v1.y) / 2,
                          (v0.z + v1.z) / 2))

    faces: List[List[int]] = []

    # Face-faces: edge midpoints in face-loop order (same winding)
    for f in mesh.iter_faces():
        faces.append([eid_to_idx[he.edge.id] for he in f.halfedges()])

    # Vertex-faces: midpoints of incident edges, fan order reversed
    # (fan order pairs same-direction with the face-faces; see dual()).
    for v in mesh.iter_vertices():
        ring = [eid_to_idx[he.edge.id] for he in v.outgoing_halfedges()]
        ring.reverse()
        faces.append(ring)

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Vertex cutting (truncation)
# ─────────────────────────────────────────────────────────────────────────────

def vertex_cutting(mesh: DLFLMesh, offset: float = 0.25) -> DLFLMesh:
    """
    Vertex truncation: every vertex is cut off; each corner of each edge
    yields a new vertex at *offset* along the edge from its origin.

    New faces: each original n-gon → 2n-gon; each original valence-k
    vertex → k-gon.

    Oracle: V' = 2E (one per half-edge), E' = 3E, F' = F + V;
    χ and genus preserved.  (cube → truncated cube for offset < 0.5)
    """
    positions: List[Tuple[float, float, float]] = []
    he_idx: Dict[int, int] = {}
    for he in mesh.iter_halfedges():
        u = he.origin
        w = he.twin.origin
        he_idx[he.id] = len(positions)
        positions.append((u.x + offset * (w.x - u.x),
                          u.y + offset * (w.y - u.y),
                          u.z + offset * (w.z - u.z)))

    faces: List[List[int]] = []

    # Face-faces: each n-gon → 2n-gon, alternating the two points of
    # every boundary edge (near-origin point, then near-destination point).
    for f in mesh.iter_faces():
        ring: List[int] = []
        for he in f.halfedges():
            ring.append(he_idx[he.id])        # near he.origin
            ring.append(he_idx[he.twin.id])   # near he.destination
        faces.append(ring)

    # Vertex-faces: the k cut points around each vertex, fan reversed.
    for v in mesh.iter_vertices():
        ring = [he_idx[he.id] for he in v.outgoing_halfedges()]
        ring.reverse()
        faces.append(ring)

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Triangle-only schemes: Loop and sqrt(3)
# ─────────────────────────────────────────────────────────────────────────────

def _require_all_triangles(mesh: DLFLMesh, op_name: str) -> None:
    for f in mesh.iter_faces():
        if f.degree() != 3:
            raise ValueError(
                f"{op_name} requires an all-triangle mesh "
                f"(face {f.id} has degree {f.degree()})"
            )


def loop_subdivide(mesh: DLFLMesh) -> DLFLMesh:
    """
    One round of Loop subdivision (triangle meshes only).

    Each triangle splits into 4.  Positions follow the standard Loop
    rules: edge points 3/8·(endpoints) + 1/8·(opposite vertices);
    original vertices smoothed with β(n) = (5/8 − (3/8 + cos(2πn)/4)²)/n.

    Oracle: V' = V + E, E' = 4E, F' = 4F; χ and genus preserved.
    Raises ValueError on non-triangular input.
    """
    import math
    _require_all_triangles(mesh, "loop_subdivide")

    positions: List[Tuple[float, float, float]] = []

    # Smoothed original vertices
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        ring = [he.twin.origin for he in v.outgoing_halfedges()]
        n = len(ring)
        beta = (5.0 / 8.0 - (3.0 / 8.0 + math.cos(2 * math.pi / n) / 4.0) ** 2) / n
        sx = sum(u.x for u in ring)
        sy = sum(u.y for u in ring)
        sz = sum(u.z for u in ring)
        w = 1.0 - n * beta
        vid_to_idx[v.id] = len(positions)
        positions.append((w * v.x + beta * sx,
                          w * v.y + beta * sy,
                          w * v.z + beta * sz))

    # Edge points: 3/8 endpoints + 1/8 opposite vertices
    eid_to_idx: Dict[int, int] = {}
    for e in mesh.iter_edges():
        v0, v1 = e.vertices()
        o0 = e.he0.prev.origin   # vertex opposite in he0's triangle
        o1 = e.he1.prev.origin   # vertex opposite in he1's triangle
        eid_to_idx[e.id] = len(positions)
        positions.append((
            0.375 * (v0.x + v1.x) + 0.125 * (o0.x + o1.x),
            0.375 * (v0.y + v1.y) + 0.125 * (o0.y + o1.y),
            0.375 * (v0.z + v1.z) + 0.125 * (o0.z + o1.z),
        ))

    # 1 → 4 split, winding inherited from the parent triangle
    faces: List[List[int]] = []
    for f in mesh.iter_faces():
        h0, h1, h2 = f.halfedges()
        c0, c1, c2 = (vid_to_idx[h0.origin.id],
                      vid_to_idx[h1.origin.id],
                      vid_to_idx[h2.origin.id])
        m0, m1, m2 = (eid_to_idx[h0.edge.id],
                      eid_to_idx[h1.edge.id],
                      eid_to_idx[h2.edge.id])
        faces.append([c0, m0, m2])
        faces.append([c1, m1, m0])
        faces.append([c2, m2, m1])
        faces.append([m0, m1, m2])

    return _build_mesh(positions, faces)


def sqrt3_subdivide(mesh: DLFLMesh) -> DLFLMesh:
    """
    One round of sqrt(3) subdivision (Kobbelt 2000; triangle meshes only).

    Insert a centroid vertex in every triangle, then flip all original
    edges.  Original vertices smoothed with α(n) = (4 − 2cos(2π/n))/9.

    Oracle: V' = V + F, E' = 3E, F' = 3F; χ and genus preserved.
    Raises ValueError on non-triangular input.
    """
    import math
    _require_all_triangles(mesh, "sqrt3_subdivide")

    positions: List[Tuple[float, float, float]] = []

    # Smoothed original vertices
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        ring = [he.twin.origin for he in v.outgoing_halfedges()]
        n = len(ring)
        alpha = (4.0 - 2.0 * math.cos(2.0 * math.pi / n)) / 9.0
        sx = sum(u.x for u in ring) / n
        sy = sum(u.y for u in ring) / n
        sz = sum(u.z for u in ring) / n
        vid_to_idx[v.id] = len(positions)
        positions.append(((1 - alpha) * v.x + alpha * sx,
                          (1 - alpha) * v.y + alpha * sy,
                          (1 - alpha) * v.z + alpha * sz))

    # Face centroids
    fid_to_idx: Dict[int, int] = {}
    for f in mesh.iter_faces():
        fid_to_idx[f.id] = len(positions)
        positions.append(f.centroid())

    # Flip every original edge: for edge (h in A, t in B) with
    # u = h.origin, w = t.origin, emit CCW triangles
    # (u, cB, cA) and (w, cA, cB).
    faces: List[List[int]] = []
    for e in mesh.iter_edges():
        h, t = e.he0, e.he1
        ca = fid_to_idx[h.face.id]
        cb = fid_to_idx[t.face.id]
        u = vid_to_idx[h.origin.id]
        w = vid_to_idx[t.origin.id]
        faces.append([u, cb, ca])
        faces.append([w, ca, cb])

    return _build_mesh(positions, faces)
