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

def _corner_cut_topology(
    mesh: DLFLMesh,
    corner_pos: Dict[int, Tuple[float, float, float]],
) -> DLFLMesh:
    """
    Shared topology builder for Doo-Sabin-class corner-cutting schemes.

    *corner_pos* maps each half-edge id (= corner of he.face at he.origin)
    to the position of that corner's new vertex.  The output topology is
    always: face-faces + edge-quads + vertex-faces →
    V' = 2E, E' = 4E, F' = V + E + F.
    """
    positions: List[Tuple[float, float, float]] = []
    corner_idx: Dict[int, int] = {}   # halfedge id -> new vertex index

    for f in mesh.iter_faces():
        for he in f.halfedges():
            corner_idx[he.id] = len(positions)
            positions.append(corner_pos[he.id])

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


def doo_sabin(mesh: DLFLMesh) -> DLFLMesh:
    """
    One round of Doo-Sabin subdivision (corner-cutting).

    Each corner's new vertex is the average of: the corner vertex, the
    face centroid, and the midpoints of the two face edges incident to
    that corner.

    Oracle: V' = 2E, E' = 4E, F' = V + E + F; χ and genus preserved.
    """
    corner_pos: Dict[int, Tuple[float, float, float]] = {}
    for f in mesh.iter_faces():
        cx, cy, cz = f.centroid()
        for he in f.halfedges():
            v = he.origin
            p = he.prev.origin
            nx = he.twin.origin
            m1 = ((v.x + p.x) / 2, (v.y + p.y) / 2, (v.z + p.z) / 2)
            m2 = ((v.x + nx.x) / 2, (v.y + nx.y) / 2, (v.z + nx.z) / 2)
            corner_pos[he.id] = (
                (v.x + cx + m1[0] + m2[0]) / 4.0,
                (v.y + cy + m1[1] + m2[1]) / 4.0,
                (v.z + cz + m1[2] + m2[2]) / 4.0,
            )
    return _corner_cut_topology(mesh, corner_pos)


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


# ─────────────────────────────────────────────────────────────────────────────
# Batch 2 — schemes from docs/reference_semantics.md (clean-room)
# ─────────────────────────────────────────────────────────────────────────────

def _copy_mesh(mesh: DLFLMesh) -> DLFLMesh:
    """Rebuild an identical DLFLMesh (fresh ids)."""
    positions: List[Tuple[float, float, float]] = []
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        vid_to_idx[v.id] = len(positions)
        positions.append((v.x, v.y, v.z))
    faces = [[vid_to_idx[v.id] for v in f.vertices()] for f in mesh.iter_faces()]
    return _build_mesh(positions, faces)


def honeycomb_subdivide(mesh: DLFLMesh) -> DLFLMesh:
    """
    Honeycomb subdivision (TopMod honeycombSubdivide).

    Topologically identical to dual(stellate_all(M)) — per reference
    semantics (docs/reference_semantics.md §1); implemented as exactly
    that composition.  Geometry uses stellated-triangle centroids
    (the reference uses a trigonometric averaging mask — geometric
    difference only, topology identical).

    Oracle: V' = 2E, E' = 3E, F' = F + V; χ and genus preserved.
    """
    from .high_level_ops import stellate_all
    tmp = _copy_mesh(mesh)
    stellate_all(tmp)
    return dual(tmp)


def star_subdivide(mesh: DLFLMesh, offset: float = 0.0) -> None:
    """
    Star subdivision (TopMod starSubdivide), in place.

    = stellate_all applied twice; the first-round apexes are then
    displaced by *offset* along their original face normals.

    Oracle: V' = V + F + 2E, E' = 9E, F' = 6E (all-tri);
    χ and genus preserved.
    """
    from .high_level_ops import stellate_all
    normals = [f.normal() for f in mesh.iter_faces()]
    apexes = stellate_all(mesh)           # aligned with the face order above
    stellate_all(mesh)
    if offset:
        for apex, n in zip(apexes, normals):
            apex.x += offset * n[0]
            apex.y += offset * n[1]
            apex.z += offset * n[2]


def corner_cutting(mesh: DLFLMesh, alpha: float = 0.5) -> DLFLMesh:
    """
    Corner-cutting subdivision (TopMod cornerCuttingSubdivide).

    Geometric variant of Doo-Sabin: identical topology, but each face's
    new corner i is a weighted average of the face's vertices with the
    diagonal weight exposed as the tension parameter *alpha*:

        w_ii = alpha
        w_ij = (1 - alpha) * (3 + 2 cos(2π(i-j)/d)) / (3d - 5)

    Oracle: V' = 2E, E' = 4E, F' = V + E + F; χ and genus preserved.
    *alpha* affects geometry only.
    """
    import math
    corner_pos: Dict[int, Tuple[float, float, float]] = {}
    for f in mesh.iter_faces():
        hes = f.halfedges()
        verts = [he.origin for he in hes]
        d = len(verts)
        denom = 3 * d - 5
        for i, he in enumerate(hes):
            x = y = z = 0.0
            for j, v in enumerate(verts):
                if i == j:
                    w = alpha
                else:
                    w = (1.0 - alpha) * (3.0 + 2.0 * math.cos(
                        2.0 * math.pi * (i - j) / d)) / denom
                x += w * v.x
                y += w * v.y
                z += w * v.z
            corner_pos[he.id] = (x, y, z)
    return _corner_cut_topology(mesh, corner_pos)


def _split_faces_positions(
    mesh: DLFLMesh, length: float,
) -> Tuple[List[Tuple[float, float, float]], Dict[int, int], Dict[int, int]]:
    """
    Shared vertex layout for loop_style / fractal: blended original
    vertices + one midpoint per edge.

    Returns (positions, vid_to_idx, eid_to_idx).
    """
    positions: List[Tuple[float, float, float]] = []
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        mids = []
        for he in v.outgoing_halfedges():
            w = he.twin.origin
            mids.append(((v.x + w.x) / 2, (v.y + w.y) / 2, (v.z + w.z) / 2))
        n = len(mids)
        ax = sum(m[0] for m in mids) / n
        ay = sum(m[1] for m in mids) / n
        az = sum(m[2] for m in mids) / n
        vid_to_idx[v.id] = len(positions)
        positions.append((length * v.x + (1 - length) * ax,
                          length * v.y + (1 - length) * ay,
                          length * v.z + (1 - length) * az))

    eid_to_idx: Dict[int, int] = {}
    for e in mesh.iter_edges():
        v0, v1 = e.vertices()
        eid_to_idx[e.id] = len(positions)
        positions.append(((v0.x + v1.x) / 2,
                          (v0.y + v1.y) / 2,
                          (v0.z + v1.z) / 2))
    return positions, vid_to_idx, eid_to_idx


def loop_style_subdivide(mesh: DLFLMesh, length: float = 1.0) -> DLFLMesh:
    """
    Loop-style subdivision (TopMod loopStyleSubdivide) — the polygonal
    generalization of Loop connectivity: split every edge at its
    midpoint; per d-gon face cut off each old-vertex corner as a
    triangle, leaving a central midpoint d-gon.

    Original vertices are blended: length·p + (1−length)·avg(midpoints).

    Oracle: V' = V + E, E' = 4E, F' = F + 2E; χ and genus preserved.
    On all-tri input the connectivity equals Loop's (geometry differs).
    """
    positions, vid_to_idx, eid_to_idx = _split_faces_positions(mesh, length)

    faces: List[List[int]] = []
    for f in mesh.iter_faces():
        hes = f.halfedges()
        # corner triangle per old vertex: [m(prev), v, m(cur)]
        for he in hes:
            faces.append([eid_to_idx[he.prev.edge.id],
                          vid_to_idx[he.origin.id],
                          eid_to_idx[he.edge.id]])
        # central midpoint d-gon
        faces.append([eid_to_idx[he.edge.id] for he in hes])

    return _build_mesh(positions, faces)


def fractal_subdivide(mesh: DLFLMesh, offset: float = 1.0) -> DLFLMesh:
    """
    Fractal subdivision (TopMod fractalSubdivide).

    = loop_style split, then stellate each central midpoint polygon with
    an apex displaced along the face normal by
    h = offset · sqrt(max(L2² − L1², 0)), where L2 = distance between
    consecutive midpoints and L1 = half the across-face corner distance
    (reference heuristic; geometric only).

    Oracle: V' = V + E + F, E' = 6E, F' = 4E (all-tri);
    χ and genus preserved.
    """
    import math
    positions, vid_to_idx, eid_to_idx = _split_faces_positions(mesh, 1.0)

    # Apex per face
    fid_to_idx: Dict[int, int] = {}
    for f in mesh.iter_faces():
        cx, cy, cz = f.centroid()
        nx, ny, nz = f.normal()
        verts = f.vertices()
        d = len(verts)
        # L2: distance between two consecutive edge midpoints
        e0, e1 = f.halfedges()[0], f.halfedges()[1]
        m0 = positions[eid_to_idx[e0.edge.id]]
        m1 = positions[eid_to_idx[e1.edge.id]]
        L2 = math.dist(m0, m1)
        # L1: half the distance between opposite corners
        va, vb = verts[0], verts[d // 2]
        L1 = math.dist((va.x, va.y, va.z), (vb.x, vb.y, vb.z)) / 2.0
        h = offset * math.sqrt(max(L2 * L2 - L1 * L1, 0.0))
        fid_to_idx[f.id] = len(positions)
        positions.append((cx + h * nx, cy + h * ny, cz + h * nz))

    faces: List[List[int]] = []
    for f in mesh.iter_faces():
        hes = f.halfedges()
        apex = fid_to_idx[f.id]
        for he in hes:
            # corner triangle
            faces.append([eid_to_idx[he.prev.edge.id],
                          vid_to_idx[he.origin.id],
                          eid_to_idx[he.edge.id]])
            # apex triangle (stellated central polygon)
            faces.append([eid_to_idx[he.prev.edge.id],
                          eid_to_idx[he.edge.id],
                          apex])

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Batch 3 — schemes from docs/reference_semantics.md §§2a, 2b, 8, 4
# ─────────────────────────────────────────────────────────────────────────────

def _honeycomb_mask_weights(d: int) -> List[List[float]]:
    """Honeycomb averaging mask (docs/reference_semantics.md §1):
    diagonal α = 1/√2 − 1/4 + 5/(4d); off-diag (1−α)(3+2cos(2π(i−j)/d))/(3d−5)."""
    import math
    alpha = 1.0 / math.sqrt(2.0) - 0.25 + 5.0 / (4.0 * d)
    denom = 3 * d - 5
    rows: List[List[float]] = []
    for i in range(d):
        row = []
        for j in range(d):
            if i == j:
                row.append(alpha)
            else:
                row.append((1.0 - alpha)
                           * (3.0 + 2.0 * math.cos(2.0 * math.pi * (i - j) / d))
                           / denom)
        rows.append(row)
    return rows


def pentagonal_subdivide(mesh: DLFLMesh, offset: float = 0.0) -> DLFLMesh:
    """
    Pentagonal subdivision (TopMod pentagonalSubdivide).

    Trisect every edge, add a centroid vertex per face, and fan spokes
    from the centroid to one trisection point per boundary edge (the one
    nearer each corner, chosen consistently so adjacent faces never pick
    the same point).  Every old d-gon becomes d pentagons.

    Oracle: V' = V + 2E + F, E' = 5E, F' = 2E (all pentagons);
    χ and genus preserved.  (tetrahedron → dodecahedron combinatorics)

    *offset* pulls each face's spoke-target trisection point toward that
    face's centroid — geometric only.
    """
    positions: List[Tuple[float, float, float]] = []

    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        vid_to_idx[v.id] = len(positions)
        positions.append((v.x, v.y, v.z))

    # Two trisection points per edge, keyed by half-edge:
    # near(he) = point at 1/3 from he.origin (shared: near(he) is the
    # same vertex as the point at 2/3 along the twin).
    near: Dict[int, int] = {}
    for e in mesh.iter_edges():
        h, t = e.he0, e.he1
        u, w = h.origin, t.origin
        near[h.id] = len(positions)
        positions.append((u.x + (w.x - u.x) / 3.0,
                          u.y + (w.y - u.y) / 3.0,
                          u.z + (w.z - u.z) / 3.0))
        near[t.id] = len(positions)
        positions.append((w.x + (u.x - w.x) / 3.0,
                          w.y + (u.y - w.y) / 3.0,
                          w.z + (u.z - w.z) / 3.0))

    fid_to_idx: Dict[int, int] = {}
    for f in mesh.iter_faces():
        fid_to_idx[f.id] = len(positions)
        positions.append(f.centroid())

    # offset: each face pulls its own spoke targets (near(he) for its
    # half-edges) toward its centroid.  Each trisection point is the
    # spoke target of exactly one face, so this is well defined.
    if offset:
        for f in mesh.iter_faces():
            cx, cy, cz = positions[fid_to_idx[f.id]]
            for he in f.halfedges():
                i = near[he.id]
                x, y, z = positions[i]
                positions[i] = (x + offset * (cx - x),
                                y + offset * (cy - y),
                                z + offset * (cz - z))

    # Face boundary (CCW) reads per half-edge: origin, near(he), far(he)
    # where far(he) = near(he.twin).  Pentagon between consecutive spokes
    # at near(he) and near(he.next):
    #   [near(he), far(he), he.next.origin, near(he.next), centroid]
    faces: List[List[int]] = []
    for f in mesh.iter_faces():
        c = fid_to_idx[f.id]
        for he in f.halfedges():
            faces.append([near[he.id],
                          near[he.twin.id],
                          vid_to_idx[he.next.origin.id],
                          near[he.next.id],
                          c])

    return _build_mesh(positions, faces)


def pentagonal2_subdivide(mesh: DLFLMesh, scale_factor: float = 0.75) -> DLFLMesh:
    """
    Pentagonal subdivision, variant 2 (TopMod pentagonalSubdivide2).

    Split every edge at its midpoint; per face insert a scaled inner
    d-gon at the edge midpoints (new vertices) and connect each midpoint
    to its inner copy.  Yields the inner d-gon plus d pentagons per face.

    Oracle: V' = V + 3E, E' = 6E, F' = F + 2E; χ and genus preserved.
    *scale_factor* shrinks the inner polygon about its centroid
    (geometric only).
    """
    positions: List[Tuple[float, float, float]] = []

    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        vid_to_idx[v.id] = len(positions)
        positions.append((v.x, v.y, v.z))

    eid_to_idx: Dict[int, int] = {}
    for e in mesh.iter_edges():
        v0, v1 = e.vertices()
        eid_to_idx[e.id] = len(positions)
        positions.append(((v0.x + v1.x) / 2,
                          (v0.y + v1.y) / 2,
                          (v0.z + v1.z) / 2))

    # Per-face inner corners: scaled copies of the edge midpoints,
    # keyed by half-edge (inner corner matching midpoint of he.edge).
    inner: Dict[int, int] = {}
    for f in mesh.iter_faces():
        hes = f.halfedges()
        mids = [positions[eid_to_idx[he.edge.id]] for he in hes]
        d = len(mids)
        cx = sum(m[0] for m in mids) / d
        cy = sum(m[1] for m in mids) / d
        cz = sum(m[2] for m in mids) / d
        for he, m in zip(hes, mids):
            inner[he.id] = len(positions)
            positions.append((cx + scale_factor * (m[0] - cx),
                              cy + scale_factor * (m[1] - cy),
                              cz + scale_factor * (m[2] - cz)))

    faces: List[List[int]] = []
    for f in mesh.iter_faces():
        hes = f.halfedges()
        # inner d-gon, same winding as the face
        faces.append([inner[he.id] for he in hes])
        # pentagons: outer arc m(he) → v(he.next) → m(he.next), then
        # inward to inner(he.next) and back along the inner polygon.
        for he in hes:
            faces.append([eid_to_idx[he.edge.id],
                          vid_to_idx[he.next.origin.id],
                          eid_to_idx[he.next.edge.id],
                          inner[he.next.id],
                          inner[he.id]])

    return _build_mesh(positions, faces)


def dual1264_subdivide(mesh: DLFLMesh, sf: float = 1.0) -> DLFLMesh:
    """
    Dual 12.6.4 subdivision (TopMod dual1264Subdivide).

    Doo-Sabin-like, but the inner face per old d-gon is a 2d-gon whose
    corners sit at 1/3 and 2/3 along each boundary edge (per-face
    copies), scaled by *sf* about their centroid.

    Faces: one 2d-gon per old face, one quad per old edge, one 2n-gon
    per old valence-n vertex.

    Oracle: V' = 4E, E' = 6E, F' = F + E + V; χ and genus preserved.
    """
    positions: List[Tuple[float, float, float]] = []
    a_idx: Dict[int, int] = {}   # per half-edge: point at 1/3 from origin
    b_idx: Dict[int, int] = {}   # per half-edge: point at 2/3 from origin

    for f in mesh.iter_faces():
        hes = f.halfedges()
        pts: List[Tuple[float, float, float]] = []
        for he in hes:
            u, w = he.origin, he.twin.origin
            pts.append((u.x + (w.x - u.x) / 3.0,
                        u.y + (w.y - u.y) / 3.0,
                        u.z + (w.z - u.z) / 3.0))
            pts.append((u.x + 2.0 * (w.x - u.x) / 3.0,
                        u.y + 2.0 * (w.y - u.y) / 3.0,
                        u.z + 2.0 * (w.z - u.z) / 3.0))
        n = len(pts)
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        cz = sum(p[2] for p in pts) / n
        for he, k in zip(hes, range(0, n, 2)):
            pa, pb = pts[k], pts[k + 1]
            a_idx[he.id] = len(positions)
            positions.append((cx + sf * (pa[0] - cx),
                              cy + sf * (pa[1] - cy),
                              cz + sf * (pa[2] - cz)))
            b_idx[he.id] = len(positions)
            positions.append((cx + sf * (pb[0] - cx),
                              cy + sf * (pb[1] - cy),
                              cz + sf * (pb[2] - cz)))

    faces: List[List[int]] = []

    # Face-faces: 2d-gon per old face, same winding
    for f in mesh.iter_faces():
        ring: List[int] = []
        for he in f.halfedges():
            ring.append(a_idx[he.id])
            ring.append(b_idx[he.id])
        faces.append(ring)

    # Edge-faces: quad bridging the two aligned inner segments.
    # Face A's inner polygon uses a(h)→b(h); the quad traverses b(h)→a(h),
    # crossing to the twin side at matching ends: a(h)~b(t) near h.origin.
    for e in mesh.iter_edges():
        h, t = e.he0, e.he1
        faces.append([b_idx[h.id], a_idx[h.id], b_idx[t.id], a_idx[t.id]])

    # Vertex-faces: 2n-gon per old vertex.  Walk outgoing half-edges in
    # CCW order (reversed fan, as in dual()); per corner append
    # [a(he), b(he.prev)] — the inner-polygon corner-span reversed plus
    # the cross edge to the next corner.
    for v in mesh.iter_vertices():
        ring = []
        outgoing = v.outgoing_halfedges()
        outgoing.reverse()
        for he in outgoing:
            ring.append(a_idx[he.id])
            ring.append(b_idx[he.prev.id])
        faces.append(ring)

    return _build_mesh(positions, faces)


def root4_subdivide(mesh: DLFLMesh, a: float = 0.0, twist: float = 0.0) -> DLFLMesh:
    """
    Root-4 subdivision (TopMod root4Subdivide).

    Per face: inner d-gon from the honeycomb mask applied to twisted
    boundary samples; bridge to the old boundary with a prism ring; then
    delete all original edges (each deletion merges the two flanking
    bridge quads into a hexagon).  Old vertices survive with unchanged
    valence and are smoothed by blend *a* toward their neighbor average.

    Oracle: V' = V + 2E, E' = 4E, F' = F + E (inner d-gons + edge
    hexagons); χ and genus preserved.

    *twist* slides the inner-ring samples along the boundary edges;
    *a* is the old-vertex smoothing blend.  Both geometric only.
    """
    positions: List[Tuple[float, float, float]] = []

    # Old vertices, smoothed: p' = a·q + (1−a)·p with q = average of the
    # opposite endpoints of the incident edges.
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        ring = [he.twin.origin for he in v.outgoing_halfedges()]
        n = len(ring)
        qx = sum(u.x for u in ring) / n
        qy = sum(u.y for u in ring) / n
        qz = sum(u.z for u in ring) / n
        vid_to_idx[v.id] = len(positions)
        positions.append((a * qx + (1 - a) * v.x,
                          a * qy + (1 - a) * v.y,
                          a * qz + (1 - a) * v.z))

    # Inner d-gon corners per face (keyed by half-edge: corner at
    # he.origin): honeycomb mask over twisted samples
    # m_i = (1−twist)·p_i + twist·p_{i+1}.
    inner: Dict[int, int] = {}
    for f in mesh.iter_faces():
        hes = f.halfedges()
        d = len(hes)
        samples = []
        for he in hes:
            u, w = he.origin, he.twin.origin
            samples.append(((1 - twist) * u.x + twist * w.x,
                            (1 - twist) * u.y + twist * w.y,
                            (1 - twist) * u.z + twist * w.z))
        weights = _honeycomb_mask_weights(d)
        for i, he in enumerate(hes):
            x = sum(weights[i][j] * samples[j][0] for j in range(d))
            y = sum(weights[i][j] * samples[j][1] for j in range(d))
            z = sum(weights[i][j] * samples[j][2] for j in range(d))
            inner[he.id] = len(positions)
            positions.append((x, y, z))

    faces: List[List[int]] = []

    # Inner d-gon per face, same winding
    for f in mesh.iter_faces():
        faces.append([inner[he.id] for he in f.halfedges()])

    # Hexagon per old edge: the two zero-height-extrusion side quads
    # merged by deleting the old edge u—w:
    # [u, n_B(t.next), n_B(t), w, n_A(h.next), n_A(h)]
    for e in mesh.iter_edges():
        h, t = e.he0, e.he1
        u, w = h.origin, t.origin
        faces.append([vid_to_idx[u.id],
                      inner[t.next.id],
                      inner[t.id],
                      vid_to_idx[w.id],
                      inner[h.next.id],
                      inner[h.id]])

    return _build_mesh(positions, faces)


# ─────────────────────────────────────────────────────────────────────────────
# Batch 4a — schemes from docs/reference_semantics.md §§9, 11
# ─────────────────────────────────────────────────────────────────────────────

def checkerboard_remesh(mesh: DLFLMesh, thickness: float = 0.25) -> DLFLMesh:
    """
    Checkerboard remeshing (TopMod checkerBoardRemeshing).

    Reference pipeline: inset every face, trisect every original edge at
    *thickness*·length from each end, chord every corner around each old
    vertex, then delete the inset spokes.  Final connectivity (derived,
    χ-verified):

      - inner d-gon per old face:        [q(he) …]
      - central quad per half-edge:      [c(he), c(he.twin), q(he.next), q(he)]
      - corner quad per half-edge:       [c(he.prev.twin), v, c(he), q(he)]

    where q(he) = inset corner of he.face at he.origin and c(he) = point
    at *thickness* along he from he.origin.

    Oracle: V' = V + 4E, E' = 9E, F' = F + 4E; χ and genus preserved.
    All-quad output on all-quad input.  *thickness* ∈ (0, 0.5): geometric.
    """
    t = thickness
    positions: List[Tuple[float, float, float]] = []

    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        vid_to_idx[v.id] = len(positions)
        positions.append((v.x, v.y, v.z))

    # c(he): checker point at t along he from he.origin (2 per edge)
    c_idx: Dict[int, int] = {}
    for he in mesh.iter_halfedges():
        u, w = he.origin, he.twin.origin
        c_idx[he.id] = len(positions)
        positions.append((u.x + t * (w.x - u.x),
                          u.y + t * (w.y - u.y),
                          u.z + t * (w.z - u.z)))

    # q(he): inset corner — he.origin pulled inward along both incident
    # boundary directions of the face by t.
    q_idx: Dict[int, int] = {}
    for f in mesh.iter_faces():
        for he in f.halfedges():
            v = he.origin
            wn = he.twin.origin      # next vertex along the face
            wp = he.prev.origin      # previous vertex along the face
            q_idx[he.id] = len(positions)
            positions.append((v.x + t * (wn.x - v.x) + t * (wp.x - v.x),
                              v.y + t * (wn.y - v.y) + t * (wp.y - v.y),
                              v.z + t * (wn.z - v.z) + t * (wp.z - v.z)))

    faces: List[List[int]] = []

    # Inner d-gon per face (same winding)
    for f in mesh.iter_faces():
        faces.append([q_idx[he.id] for he in f.halfedges()])

    # Central + corner quads per half-edge
    for f in mesh.iter_faces():
        for he in f.halfedges():
            faces.append([c_idx[he.id],
                          c_idx[he.twin.id],
                          q_idx[he.next.id],
                          q_idx[he.id]])
            faces.append([c_idx[he.prev.twin.id],
                          vid_to_idx[he.origin.id],
                          c_idx[he.id],
                          q_idx[he.id]])

    return _build_mesh(positions, faces)


def ds_bc_new_subdivide(mesh: DLFLMesh, sf: float = 1.0,
                        length: float = 1.0) -> DLFLMesh:
    """
    Doo-Sabin BC-new subdivision (TopMod dooSabinSubdivideBCNew).

    DS polygon computed from the mid-edge-refined boundary (a 2d-gon per
    face), original vertices survive; the reference's midpoint vertices
    are eliminated by bypass chords.  Final connectivity (derived,
    χ-verified):

      - top 2d-gon per face:  [tv(he), tm(he) …]
      - v-side pentagon per edge: [tmA(h), tvA(h), v, tvB(t.next), tmB(t)]
      - w-side pentagon per edge: [w, tvA(h.next), tmA(h), tmB(t), tvB(t)]

    where tv/tm are the per-face DS corners at old vertices / edge
    midpoints, scaled by *sf* about the face centroid.

    Oracle: V' = V + 4E, E' = 7E, F' = F + 2E; χ and genus preserved.
    *sf* = DS scale, *length* = old-vertex blend (1 = keep): geometric.
    """
    positions: List[Tuple[float, float, float]] = []

    # Per-face DS corners on the refined 2d-gon boundary
    # (alternating old corners and edge midpoints), DS mask =
    # (point + centroid + two adjacent refined midpoints)/4, then scaled
    # by sf about the refined-boundary centroid.
    tv_idx: Dict[int, int] = {}
    tm_idx: Dict[int, int] = {}
    top_of_vertex: Dict[int, List[int]] = {}   # old vid -> its top corners

    for f in mesh.iter_faces():
        hes = f.halfedges()
        ring: List[Tuple[float, float, float]] = []
        for he in hes:
            v = he.origin
            w = he.twin.origin
            ring.append((v.x, v.y, v.z))
            ring.append(((v.x + w.x) / 2, (v.y + w.y) / 2, (v.z + w.z) / 2))
        n = len(ring)
        cx = sum(p[0] for p in ring) / n
        cy = sum(p[1] for p in ring) / n
        cz = sum(p[2] for p in ring) / n

        def ds_point(k: int) -> Tuple[float, float, float]:
            p = ring[k]
            a = ring[(k - 1) % n]
            b = ring[(k + 1) % n]
            m1 = ((p[0] + a[0]) / 2, (p[1] + a[1]) / 2, (p[2] + a[2]) / 2)
            m2 = ((p[0] + b[0]) / 2, (p[1] + b[1]) / 2, (p[2] + b[2]) / 2)
            x = (p[0] + cx + m1[0] + m2[0]) / 4.0
            y = (p[1] + cy + m1[1] + m2[1]) / 4.0
            z = (p[2] + cz + m1[2] + m2[2]) / 4.0
            return (cx + sf * (x - cx), cy + sf * (y - cy), cz + sf * (z - cz))

        for i, he in enumerate(hes):
            tv_idx[he.id] = len(positions)
            positions.append(ds_point(2 * i))
            top_of_vertex.setdefault(he.origin.id, []).append(tv_idx[he.id])
            tm_idx[he.id] = len(positions)
            positions.append(ds_point(2 * i + 1))

    # Old vertices, blended toward the average of their top corners
    vid_to_idx: Dict[int, int] = {}
    for v in mesh.iter_vertices():
        tops = top_of_vertex[v.id]
        k = len(tops)
        ax = sum(positions[i][0] for i in tops) / k
        ay = sum(positions[i][1] for i in tops) / k
        az = sum(positions[i][2] for i in tops) / k
        vid_to_idx[v.id] = len(positions)
        positions.append((length * v.x + (1 - length) * ax,
                          length * v.y + (1 - length) * ay,
                          length * v.z + (1 - length) * az))

    faces: List[List[int]] = []

    # Top 2d-gon per face (same winding)
    for f in mesh.iter_faces():
        ring2: List[int] = []
        for he in f.halfedges():
            ring2.append(tv_idx[he.id])
            ring2.append(tm_idx[he.id])
        faces.append(ring2)

    # Two pentagons per old edge
    for e in mesh.iter_edges():
        h, t = e.he0, e.he1
        v, w = h.origin, t.origin
        faces.append([tm_idx[h.id],
                      tv_idx[h.id],
                      vid_to_idx[v.id],
                      tv_idx[t.next.id],
                      tm_idx[t.id]])
        faces.append([vid_to_idx[w.id],
                      tv_idx[h.next.id],
                      tm_idx[h.id],
                      tm_idx[t.id],
                      tv_idx[t.id]])

    return _build_mesh(positions, faces)
