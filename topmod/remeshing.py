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
