"""
High-level mesh operations, composed from the 4 fundamental operators.

extrude_face(mesh, face, dist)           -> list[Face]
add_handle(mesh, face1, face2)           -> list[Edge]
stellate(mesh, face)                     -> Vertex
subdivide_edge(mesh, edge)               -> Vertex
subdivide_face(mesh, face)               -> Vertex
stellate_all(mesh)                       -> list[Vertex]
collapse_edge(mesh, edge)                -> Vertex
trisect_edge(mesh, edge, t1, t2)         -> (Vertex, Vertex)
subdivide_all_edges(mesh, n)             -> None
subdivide_all_faces(mesh)                -> list[Vertex]
triangulate_face(mesh, face)             -> None
triangulate_all(mesh)                    -> None
double_stellate_face(mesh, face, d)      -> Vertex
stellate_subdivide(mesh)                 -> None
punch_hole(mesh, face1, face2)           -> list[Edge]  (alias for add_handle)
"""

from __future__ import annotations
import math
from typing import List, Tuple

from .dlfl import DLFLMesh, Vertex, HalfEdge, Face, Edge
from .operators import create_vertex, delete_vertex, insert_edge, delete_edge


# ── helpers ───────────────────────────────────────────────────────────────────

def _vec_add(a: Tuple, b: Tuple) -> Tuple:
    return tuple(ai + bi for ai, bi in zip(a, b))

def _vec_scale(a: Tuple, s: float) -> Tuple:
    return tuple(ai * s for ai in a)

def _vec_normalize(a: Tuple) -> Tuple:
    length = math.sqrt(sum(x * x for x in a))
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(x / length for x in a)


# ── extrude_face ──────────────────────────────────────────────────────────────

def extrude_face(mesh: DLFLMesh, face: Face, dist: float = 1.0) -> List[Face]:
    """
    Extrude *face* outward by *dist* along its normal.

    Algorithm:
    1. Snapshot the face boundary (vertices, half-edges, exterior twins).
    2. Remove the original face and its boundary half-edges/edges.
    3. Create new (top) vertices offset by dist * normal.
    4. Build top cap + n side quads, wiring all twins properly.

    The original face is removed; the top cap and side walls are created.
    Original vertices stay in place (become bottom ring of side walls).
    New vertices form the top cap.

    Returns a list of all newly created faces (top + side quads).
    """
    normal = face.normal()
    dx, dy, dz = (dist * normal[0], dist * normal[1], dist * normal[2])

    # Snapshot before mutation
    orig_hes: List[HalfEdge] = list(face.halfedges())  # CCW
    orig_verts: List[Vertex] = [he.origin for he in orig_hes]
    n = len(orig_verts)

    if n < 3:
        raise ValueError("extrude_face: face must have at least 3 vertices")

    # Save exterior twins (the half-edges on adjacent faces pointing inward)
    exterior_twins: List[HalfEdge] = [he.twin for he in orig_hes]

    # 1. Remove original face and its boundary half-edges/edges
    mesh._remove_face(face)
    for he in orig_hes:
        e = he.edge
        if e is not None:
            mesh._remove_edge(e)
        mesh._remove_halfedge(he)

    # 2. Create new top vertices
    new_verts: List[Vertex] = []
    for v in orig_verts:
        nv = mesh._new_vertex(v.x + dx, v.y + dy, v.z + dz)
        new_verts.append(nv)

    # 3. Build the top face: new_verts[0], new_verts[1], ..., new_verts[n-1] (CCW)
    top_face = mesh._new_face()
    top_hes: List[HalfEdge] = []
    for i in range(n):
        he = mesh._new_halfedge()
        he.origin = new_verts[i]
        he.face = top_face
        top_hes.append(he)
    for i in range(n):
        top_hes[i].next = top_hes[(i + 1) % n]
        top_hes[i].prev = top_hes[(i - 1) % n]
    top_face.he = top_hes[0]
    for i, nv in enumerate(new_verts):
        nv.he = top_hes[i]

    # 4. Build n side-quad faces
    # Side quad[i] connects:
    #   orig_verts[i] → orig_verts[(i+1)%n] → new_verts[(i+1)%n] → new_verts[i]
    # This matches the winding so that:
    #   - bottom edge (orig[i]→orig[i+1]) twins with exterior_twins[i]
    #   - top edge (new[i+1]→new[i]) twins with top_hes[i] (which goes new[i]→new[i+1])
    #   - left edge (new[i]→orig[i]) twins with right edge of quad[i-1]
    #   - right edge (orig[i+1]→new[i+1]) twins with left edge of quad[i+1]

    side_faces: List[Face] = []
    # Each side quad has 4 half-edges: bottom, right, top, left
    side_bottom: List[HalfEdge] = []
    side_right: List[HalfEdge] = []
    side_top: List[HalfEdge] = []
    side_left: List[HalfEdge] = []

    for i in range(n):
        sf = mesh._new_face()
        side_faces.append(sf)

        bot  = mesh._new_halfedge()  # orig[i]    → orig[i+1]
        rt   = mesh._new_halfedge()  # orig[i+1]  → new[i+1]
        top  = mesh._new_halfedge()  # new[i+1]   → new[i]
        lt   = mesh._new_halfedge()  # new[i]     → orig[i]

        bot.origin = orig_verts[i]
        rt.origin  = orig_verts[(i + 1) % n]
        top.origin = new_verts[(i + 1) % n]
        lt.origin  = new_verts[i]

        bot.face = rt.face = top.face = lt.face = sf
        sf.he = bot

        bot.next = rt;  rt.prev = bot
        rt.next  = top; top.prev = rt
        top.next = lt;  lt.prev = top
        lt.next  = bot; bot.prev = lt

        side_bottom.append(bot)
        side_right.append(rt)
        side_top.append(top)
        side_left.append(lt)

    # 5. Wire all twin pairs and create Edge objects

    # (a) Bottom edges: side_bottom[i] (orig[i]→orig[i+1]) ↔ exterior_twins[i] (orig[i+1]→orig[i])
    for i in range(n):
        ext = exterior_twins[i]
        mesh._new_edge(side_bottom[i], ext)

    # (b) Top edges: side_top[i] (new[i+1]→new[i]) ↔ top_hes[i] (new[i]→new[i+1])
    for i in range(n):
        mesh._new_edge(side_top[i], top_hes[i])

    # (c) Vertical edges: side_right[i] (orig[i+1]→new[i+1]) ↔ side_left[(i+1)%n] (new[i+1]→orig[i+1])
    for i in range(n):
        mesh._new_edge(side_right[i], side_left[(i + 1) % n])

    # 6. Fix vertex.he pointers for original vertices
    for i, v in enumerate(orig_verts):
        if v.he is None or v.he.id not in mesh.halfedges or v.he.origin is not v:
            v.he = side_bottom[i]

    new_faces_list = [top_face] + side_faces
    return new_faces_list


# ── add_handle ────────────────────────────────────────────────────────────────

def add_handle(mesh: DLFLMesh, face1: Face, face2: Face) -> List[Edge]:
    """
    Connect two faces with a tube (topological handle), increasing genus by 1.

    Both faces must be on the same connected component and must not be the
    same face.  Both faces are removed, and n side-quad faces are created
    connecting corresponding vertices of the two faces.

    If the faces have different valences, the smaller face's vertex count
    is used (extra vertices on the larger face are left as-is).

    Returns the list of newly created edges (the tube walls).
    """
    if face1 is face2:
        raise ValueError("add_handle: face1 and face2 must be different faces")

    hes1 = list(face1.halfedges())
    hes2 = list(face2.halfedges())

    n = min(len(hes1), len(hes2))
    if n < 3:
        raise ValueError("add_handle: faces must have at least 3 vertices")

    verts1 = [he.origin for he in hes1[:n]]
    verts2 = [he.origin for he in hes2[:n]]

    # Save exterior twins (half-edges on adjacent faces)
    ext_twins1 = [he.twin for he in hes1[:n]]
    ext_twins2 = [he.twin for he in hes2[:n]]

    # Remove both faces and their boundary half-edges/edges
    mesh._remove_face(face1)
    for he in hes1[:n]:
        if he.edge is not None:
            mesh._remove_edge(he.edge)
        mesh._remove_halfedge(he)

    mesh._remove_face(face2)
    for he in hes2[:n]:
        if he.edge is not None:
            mesh._remove_edge(he.edge)
        mesh._remove_halfedge(he)

    # Build n side-quad faces connecting the two face boundaries.
    # Quad[i] = verts1[i] → verts1[(i+1)%n] → verts2[(i+1)%n] → verts2[i]
    #
    # Winding: face1 was CCW viewed from outside.  face2 is also CCW from
    # outside but we want the tube interior to be consistent.
    # face2's vertices are traversed in REVERSE order for consistent winding.
    # So: quad[i] = verts1[i] → verts1[i+1] → verts2[n-1-i] → verts2[n-i]
    #
    # Actually simpler: we reverse verts2 so they match up CCW.
    verts2_rev = list(reversed(verts2))
    ext_twins2_rev = list(reversed(ext_twins2))
    # Shift ext_twins2_rev: ext_twins2 was for edges verts2[i]→verts2[i+1]
    # After reversing verts2, the edge verts2_rev[i]→verts2_rev[i+1]
    # corresponds to the original edge verts2[n-1-i]→verts2[n-2-i]
    # which had twin ext_twins2[n-2-i].
    # So ext for reversed: ext_twins2_rev[i] corresponds to edge
    # going verts2_rev[i] → verts2_rev[i+1].
    # Original: ext_twins2[j] is twin of edge verts2[j]→verts2[j+1].
    # After reverse: verts2_rev[i] = verts2[n-1-i].
    # Edge verts2_rev[i]→verts2_rev[i+1] = verts2[n-1-i]→verts2[n-2-i].
    # This is the REVERSE of original edge verts2[n-2-i]→verts2[n-1-i].
    # Original twin of verts2[n-2-i]→verts2[n-1-i] is ext_twins2[n-2-i].
    # But that twin goes verts2[n-1-i]→verts2[n-2-i], which matches our direction.
    # So: ext for reversed edge i = ext_twins2[n-2-i].
    ext2_for_rev = [ext_twins2[n - 2 - i] for i in range(n)]

    new_edges: List[Edge] = []
    side_faces: List[Face] = []

    # For each quad: 4 half-edges
    quad_bot: List[HalfEdge] = []   # verts1[i] → verts1[i+1]
    quad_right: List[HalfEdge] = [] # verts1[i+1] → verts2_rev[i+1]
    quad_top: List[HalfEdge] = []   # verts2_rev[i+1] → verts2_rev[i]
    quad_left: List[HalfEdge] = []  # verts2_rev[i] → verts1[i]

    for i in range(n):
        sf = mesh._new_face()
        side_faces.append(sf)

        bot = mesh._new_halfedge()
        rt  = mesh._new_halfedge()
        top = mesh._new_halfedge()
        lt  = mesh._new_halfedge()

        bot.origin = verts1[i]
        rt.origin  = verts1[(i + 1) % n]
        top.origin = verts2_rev[(i + 1) % n]
        lt.origin  = verts2_rev[i]

        bot.face = rt.face = top.face = lt.face = sf
        sf.he = bot

        bot.next = rt;  rt.prev = bot
        rt.next  = top; top.prev = rt
        top.next = lt;  lt.prev = top
        lt.next  = bot; bot.prev = lt

        quad_bot.append(bot)
        quad_right.append(rt)
        quad_top.append(top)
        quad_left.append(lt)

    # Wire twin pairs:
    # (a) Bottom edges: quad_bot[i] ↔ ext_twins1[i]
    for i in range(n):
        e = mesh._new_edge(quad_bot[i], ext_twins1[i])
        new_edges.append(e)

    # (b) Top edges: quad_top[i] ↔ ext2_for_rev[i]
    for i in range(n):
        e = mesh._new_edge(quad_top[i], ext2_for_rev[i])
        new_edges.append(e)

    # (c) Vertical edges: quad_right[i] ↔ quad_left[(i+1)%n]
    for i in range(n):
        e = mesh._new_edge(quad_right[i], quad_left[(i + 1) % n])
        new_edges.append(e)

    # Fix vertex.he pointers
    for i, v in enumerate(verts1):
        if v.he is None or v.he.id not in mesh.halfedges or v.he.origin is not v:
            v.he = quad_bot[i]
    for i, v in enumerate(verts2_rev):
        if v.he is None or v.he.id not in mesh.halfedges or v.he.origin is not v:
            v.he = quad_left[i]

    return new_edges


# ── stellate ─────────────────────────────────────────────────────────────────

def stellate(mesh: DLFLMesh, face: Face) -> Vertex:
    """
    Stellate a face: add a new center vertex and split the face into n
    triangles (one per edge of the original face).

    Returns the new center vertex.

    Topology: V+1, E+n, F+n-1.
    """
    hes = face.halfedges()
    n   = len(hes)
    if n < 3:
        raise ValueError("stellate: face must have at least 3 vertices")

    cx, cy, cz = face.centroid()
    center = mesh._new_vertex(cx, cy, cz)

    # Snapshot: original boundary half-edges and their exterior twins.
    orig_hes: List[HalfEdge] = list(hes)
    orig_verts: List[Vertex]  = [he.origin for he in orig_hes]
    # Each orig_hes[i].twin is on an adjacent face (exterior).
    exterior_twins: List[HalfEdge] = [he.twin for he in orig_hes]

    # Remove the original face and its boundary half-edges.
    mesh._remove_face(face)
    for he in orig_hes:
        # Remove from the edge object (will be replaced by new triangle base he)
        e = he.edge
        if e is not None:
            mesh._remove_edge(e)
        mesh._remove_halfedge(he)

    # Build n triangle faces: triangle[i] = (v[i], v[i+1], center)
    #   he_base[i]  : v[i]   → v[i+1]  (needs twin = exterior_twins[i])
    #   he_right[i] : v[i+1] → center
    #   he_left[i]  : center → v[i]
    he_base:  List[HalfEdge] = []
    he_right: List[HalfEdge] = []
    he_left:  List[HalfEdge] = []
    new_faces_list: List[Face] = []

    for i in range(n):
        tf = mesh._new_face()
        new_faces_list.append(tf)

        b = mesh._new_halfedge()   # v[i]   → v[i+1]
        r = mesh._new_halfedge()   # v[i+1] → center
        l = mesh._new_halfedge()   # center → v[i]

        b.origin = orig_verts[i]
        r.origin = orig_verts[(i + 1) % n]
        l.origin = center

        b.face = r.face = l.face = tf
        tf.he = b

        b.next = r; r.prev = b
        r.next = l; l.prev = r
        l.next = b; b.prev = l

        he_base.append(b)
        he_right.append(r)
        he_left.append(l)

    # Wire base half-edges to their exterior twins (existing outer half-edges).
    for i in range(n):
        ext = exterior_twins[i]
        new_edge = mesh._new_edge(he_base[i], ext)
        # he_base[i] goes v[i]→v[i+1]; ext goes v[i+1]→v[i] ✓

    # Wire center spokes: he_right[i] (v[i+1]→center) twins with he_left[i+1] (center→v[i+1])
    for i in range(n):
        r_he = he_right[i]        # v[i+1] → center
        l_he = he_left[(i + 1) % n]  # center → v[i+1]
        mesh._new_edge(r_he, l_he)

    # Fix vertex.he pointers
    center.he = he_left[0]   # center → v[0]
    for i, v in enumerate(orig_verts):
        if v.he is None or not (v.he.id in mesh.halfedges and v.he.origin is v):
            v.he = he_base[i]

    return center


def _find_outgoing(mesh: DLFLMesh, v: Vertex) -> HalfEdge | None:
    for he in mesh.halfedges.values():
        if he.origin is v:
            return he
    return None


# ── subdivide_edge ────────────────────────────────────────────────────────────

def subdivide_edge(mesh: DLFLMesh, edge: Edge) -> Vertex:
    """
    Split an edge at its midpoint by inserting a new vertex.

    Before: v0 ─── v1    (one edge, two half-edges, two faces)
    After:  v0 ─ m ─ v1  (two edges, four half-edges, same faces)

    Returns the new midpoint vertex.

    Topology: V+1, E+1, F unchanged.
    """
    he_ab = edge.he0  # v0 → v1
    he_ba = edge.he1  # v1 → v0

    v0 = he_ab.origin
    v1 = he_ba.origin

    mx = (v0.x + v1.x) / 2.0
    my = (v0.y + v1.y) / 2.0
    mz = (v0.z + v1.z) / 2.0

    mid = mesh._new_vertex(mx, my, mz)

    # We split by:
    #   1. Reroute he_ab: now goes v0 → mid  (keep he_ab, change destination
    #      by inserting new half-edges)
    #   2. Insert new half-edges mid → v1 in both faces.

    # New half-edges for the second segment
    new_he_fwd = mesh._new_halfedge()   # mid → v1 on face of he_ab
    new_he_bwd = mesh._new_halfedge()   # v1  → mid on face of he_ba (twin of above)

    new_he_fwd.origin = mid
    new_he_bwd.origin = v1

    new_he_fwd.face = he_ab.face
    new_he_bwd.face = he_ba.face

    # Wire new edge
    mesh._new_edge(new_he_fwd, new_he_bwd)

    # Reroute he_ab's next/prev: insert new_he_fwd after he_ab
    next_ab = he_ab.next
    he_ab.next       = new_he_fwd
    new_he_fwd.prev  = he_ab
    new_he_fwd.next  = next_ab
    next_ab.prev     = new_he_fwd

    # Reroute he_ba: change origin to mid, insert new_he_bwd before he_ba
    prev_ba = he_ba.prev
    prev_ba.next     = new_he_bwd
    new_he_bwd.prev  = prev_ba
    new_he_bwd.next  = he_ba
    he_ba.prev       = new_he_bwd

    he_ba.origin = mid   # he_ba now goes mid → v0 (it's the twin of he_ab which ends at mid now)

    # The old edge (he_ab / he_ba) now represents v0—mid.
    # he_ba still starts at v1... wait, let me re-think.
    #
    # he_ab: v0 → v1 (twin = he_ba: v1 → v0)
    # After split:
    #   Segment 1: v0 → mid  (use existing he_ab & he_ba, repoint he_ba origin)
    #   Segment 2: mid → v1  (new_he_fwd, new_he_bwd)
    #
    # he_ba should now go mid → v0 (origin was v1, change to mid).
    # Already done above: he_ba.origin = mid

    mid.he = new_he_fwd   # mid's outgoing: mid → v1

    # Fix v1.he if it pointed to he_ba (now starting at mid)
    if v1.he is he_ba:
        v1.he = new_he_bwd   # new_he_bwd: v1 → mid (origin = v1)
    # Actually new_he_bwd.origin = v1, so that's fine.

    return mid


# ── subdivide_face ────────────────────────────────────────────────────────────

def subdivide_face(mesh: DLFLMesh, face: Face) -> Vertex:
    """
    Subdivide a face by connecting its centroid to all its vertices.

    Equivalent to stellate().  Returns the new center vertex.
    """
    return stellate(mesh, face)


def stellate_all(mesh: DLFLMesh) -> List[Vertex]:
    """
    Stellate every face of the mesh (global stellation).

    Oracle: per n-gon face (+1, +n, +n−1) summed over all faces →
    V' = V + F, E' = 3E, F' = 2E.  Output is all-triangle.
    χ and genus preserved.

    Returns the list of new apex vertices (one per original face).
    """
    apexes: List[Vertex] = []
    for face in list(mesh.faces.values()):
        apexes.append(stellate(mesh, face))
    return apexes


# ── collapse_edge ────────────────────────────────────────────────────────────

def collapse_edge(mesh: DLFLMesh, edge: Edge) -> Vertex:
    """
    Collapse an edge by merging its two endpoints into one vertex
    at their midpoint.

    Before: ... ─ v0 ── v1 ─ ...  (edge e, two flanking faces)
    After:  ... ─── m ─── ...     (v0 & v1 merged into m)

    The two faces adjacent to the edge are removed (they degenerate to
    lines).  All half-edges that used v0 or v1 are repointed to m.

    Topology: V−1, E−(1+k), F−2 for an interior edge where k is the
    number of additional edges that become duplicates after merging
    (typically k=2 for a manifold interior edge → V−1, E−3, F−2).

    Returns the surviving (midpoint) vertex.
    """
    v0, v1 = edge.vertices()
    mx = (v0.x + v1.x) / 2.0
    my = (v0.y + v1.y) / 2.0
    mz = (v0.z + v1.z) / 2.0

    # Repoint all half-edges from v1 to v0 (v0 becomes the surviving vertex)
    for he in list(mesh.halfedges.values()):
        if he.origin is v1:
            he.origin = v0

    # Set the surviving vertex to the midpoint
    v0.x, v0.y, v0.z = mx, my, mz

    # Now the edge's two half-edges both originate from v0 or point to v0,
    # making it a self-loop. Delete the edge and its flanking faces.
    he0 = edge.he0
    he1 = edge.he1

    # The two flanking faces degenerate (contain repeated vertex v0).
    # Remove them and their degenerate edges.
    for face_he in [he0, he1]:
        f = face_he.face
        if f is None:
            continue
        # Collect all half-edges of this face
        ring = f.halfedges()
        # Remove face
        mesh._remove_face(f)
        for h in ring:
            e = h.edge
            if e is not None and e.id in mesh.edges:
                # Check if this edge is now degenerate (self-loop)
                ev0, ev1 = e.vertices()
                if ev0 is ev1:
                    mesh._remove_edge(e)
                    mesh._remove_halfedge(e.he0)
                    mesh._remove_halfedge(e.he1)

    # Remove the original collapsed edge if still present
    if edge.id in mesh.edges:
        mesh._remove_edge(edge)
        if he0.id in mesh.halfedges:
            mesh._remove_halfedge(he0)
        if he1.id in mesh.halfedges:
            mesh._remove_halfedge(he1)

    # Remove v1
    if v1.id in mesh.vertices:
        mesh._remove_vertex(v1)

    # Fix v0.he
    v0.he = None
    for he in mesh.halfedges.values():
        if he.origin is v0:
            v0.he = he
            break

    # Rewire: some half-edge pairs may now be duplicates (same
    # origin/destination).  For each pair of half-edges that share the
    # same two vertices, merge them.  This is complex in general;
    # for now, just fix next/prev pointers around removed faces.

    # Fix next/prev chains: for each remaining half-edge, if its
    # next or prev was removed, skip to the next valid one.
    valid_ids = set(mesh.halfedges.keys())
    for he in list(mesh.halfedges.values()):
        while he.next is not None and he.next.id not in valid_ids:
            he.next = he.next.next
        while he.prev is not None and he.prev.id not in valid_ids:
            he.prev = he.prev.prev

    return v0


# ── trisect_edge ─────────────────────────────────────────────────────────────

def trisect_edge(mesh: DLFLMesh, edge: Edge,
                 t1: float = 1.0/3.0,
                 t2: float = 2.0/3.0) -> Tuple[Vertex, Vertex]:
    """
    Split an edge into three segments by inserting two vertices.

    Parameters t1, t2 ∈ (0,1) control the split positions along v0→v1.
    Default: equal trisection (1/3, 2/3).

    Before: v0 ──────── v1
    After:  v0 ─ m1 ─ m2 ─ v1

    Topology: V+2, E+2, F unchanged.

    Returns (m1, m2) — the two new vertices.
    """
    # First split: v0 — m2 — v1 at parameter t2
    he_ab = edge.he0
    v0 = he_ab.origin
    v1 = edge.he1.origin

    m2 = subdivide_edge(mesh, edge)
    # Position m2 at t2 along v0→v1
    m2.x = v0.x + t2 * (v1.x - v0.x)
    m2.y = v0.y + t2 * (v1.y - v0.y)
    m2.z = v0.z + t2 * (v1.z - v0.z)

    # Now the original edge is v0—m2.  Split it at t1/t2.
    # Find the edge connecting v0 and m2
    e_v0_m2 = mesh.find_edge(v0, m2)
    if e_v0_m2 is None:
        raise RuntimeError("trisect_edge: cannot find v0—m2 edge")

    m1 = subdivide_edge(mesh, e_v0_m2)
    # Position m1 at t1 along v0→v1
    m1.x = v0.x + t1 * (v1.x - v0.x)
    m1.y = v0.y + t1 * (v1.y - v0.y)
    m1.z = v0.z + t1 * (v1.z - v0.z)

    return (m1, m2)


# ── subdivide_all_edges ──────────────────────────────────────────────────────

def subdivide_all_edges(mesh: DLFLMesh, n_divs: int = 2) -> List[Vertex]:
    """
    Subdivide every edge into *n_divs* equal segments.

    n_divs=2 (default): midpoint split, same as topmod3d's subdivideAllEdges.

    Topology: V + (n_divs−1)·E_old, E_old·n_divs, F unchanged.

    Returns all newly created vertices.
    """
    new_verts: List[Vertex] = []
    orig_edges = list(mesh.edges.values())
    for e in orig_edges:
        if n_divs == 2:
            new_verts.append(subdivide_edge(mesh, e))
        else:
            # General: split into n_divs by repeated bisection is tricky;
            # instead, insert (n_divs−1) vertices at equal parameters.
            he = e.he0
            v0_pos = (he.origin.x, he.origin.y, he.origin.z)
            v1 = e.he1.origin
            v1_pos = (v1.x, v1.y, v1.z)
            cur_edge = e
            for k in range(1, n_divs):
                t = k / n_divs
                mid = subdivide_edge(mesh, cur_edge)
                mid.x = v0_pos[0] + t * (v1_pos[0] - v0_pos[0])
                mid.y = v0_pos[1] + t * (v1_pos[1] - v0_pos[1])
                mid.z = v0_pos[2] + t * (v1_pos[2] - v0_pos[2])
                new_verts.append(mid)
                # The newly created second edge (mid—v1) is the one to
                # split next.  After subdivide_edge, mid.he points mid→v1.
                cur_edge = mid.he.edge
    return new_verts


# ── subdivide_all_faces ──────────────────────────────────────────────────────

def subdivide_all_faces(mesh: DLFLMesh) -> List[Vertex]:
    """
    Subdivide every face by connecting its centroid to all corners
    (= stellate_all with apex at centroid height = 0).

    Oracle: same as stellate_all: V'=V+F, E'=3E, F'=2E.

    Returns list of new center vertices.
    """
    return stellate_all(mesh)


# ── triangulate ──────────────────────────────────────────────────────────────

def triangulate_face(mesh: DLFLMesh, face: Face) -> None:
    """
    Triangulate one face by fan from its first vertex.

    An n-gon becomes n−2 triangles (n−3 new edges inserted).
    V unchanged, E + (n−3), F + (n−3).
    """
    hes = face.halfedges()
    n = len(hes)
    if n <= 3:
        return  # already a triangle (or degenerate)

    # Fan: insert edges from hes[0] to hes[2], hes[3], ..., hes[n−2]
    for k in range(2, n - 1):
        # After each insertion, hes[0] may be on a new (smaller) face.
        # Re-fetch the face from hes[0].
        f = hes[0].face
        f_hes = f.halfedges()
        # Find hes[0] in f_hes to locate the correct target corner
        idx_0 = None
        for i, h in enumerate(f_hes):
            if h is hes[0]:
                idx_0 = i
                break
        if idx_0 is None:
            break
        # Target is the corner at offset k from hes[0] in the original
        # numbering.  In the current face it may be at a different index
        # because earlier splits shortened the face.
        # The target vertex is hes[k].origin — find the half-edge in
        # the current face that starts at that vertex.
        target_v = hes[k].origin
        target_he = None
        for h in f_hes:
            if h.origin is target_v and h is not hes[0]:
                target_he = h
                break
        if target_he is None:
            break
        insert_edge(mesh, hes[0], target_he)


def triangulate_all(mesh: DLFLMesh) -> None:
    """
    Triangulate every face of the mesh by fan from the first vertex.

    Oracle: V unchanged, E + Σ(d_i − 3) for each face of degree d_i,
    F + Σ(d_i − 3).
    """
    for face in list(mesh.faces.values()):
        triangulate_face(mesh, face)


# ── double_stellate_face ─────────────────────────────────────────────────────

def double_stellate_face(mesh: DLFLMesh, face: Face,
                         dist: float = 0.0) -> Vertex:
    """
    Double-stellate a face: stellate it, then stellate each resulting
    triangle from the first stellation.

    The first apex is placed at face centroid + dist * normal.
    The second-round apexes are at the centroids of the first-round
    triangles (no additional displacement).

    Returns the first-round apex vertex.
    """
    # Record original edges to know which faces are "first-round"
    apex = stellate(mesh, face)
    if dist != 0.0:
        normal = face.normal() if face.id in mesh.faces else (0, 0, 1)
        # face was removed by stellate; use the average of the new
        # triangle normals instead
        out_hes = apex.outgoing_halfedges()
        if out_hes:
            nx = ny = nz = 0.0
            for he in out_hes:
                n = he.face.normal()
                nx += n[0]; ny += n[1]; nz += n[2]
            mag = (nx*nx + ny*ny + nz*nz) ** 0.5
            if mag > 1e-12:
                nx, ny, nz = nx/mag, ny/mag, nz/mag
            apex.x += dist * nx
            apex.y += dist * ny
            apex.z += dist * nz

    # Second round: stellate each triangle that was just created
    new_faces = [he.face for he in apex.outgoing_halfedges()]
    for f in new_faces:
        stellate(mesh, f)

    return apex


# ── stellate_subdivide ───────────────────────────────────────────────────────

def stellate_subdivide(mesh: DLFLMesh) -> None:
    """
    Stellate subdivision: stellate every face, then delete all original
    edges.  This is topmod3d's ``stellateSubdivide``.

    Different from plain ``stellate_all`` (which keeps original edges).

    Oracle (cube): V=14, E=24, F=12.
    General: V' = V + F, E' = E + 2·Σ(d_i) = E + 4E = ... (complex;
    depends on face degrees).  In practice for regular meshes:
    after stellate_all: V+F, 3E, 2E; after deleting E old edges:
    each old edge deletion merges two triangles → V+F, 2E, E+F.
    Wait — topmod3d shows V=14, E=24, F=12 on cube (V=8,E=12,F=6):
    stellate_all → 14, 36, 24; delete 12 old edges → 14, 24, 12.
    Each deletion: E−1, F−1 (merge two triangles sharing that edge).
    So: V'=V+F, E'=3E−E=2E, F'=2E−E=E.
    """
    orig_vert_ids = set(v.id for v in mesh.iter_vertices())
    stellate_all(mesh)
    # After stellate_all, "base edges" (connecting two original vertices)
    # correspond to the original edges — delete them.
    base_edges = [e for e in list(mesh.edges.values())
                  if all(v.id in orig_vert_ids for v in e.vertices())]
    for e in base_edges:
        if e.id in mesh.edges:
            delete_edge(mesh, e)


# ── punch_hole (alias for add_handle) ────────────────────────────────────────

def punch_hole(mesh: DLFLMesh, face1: Face, face2: Face) -> List[Edge]:
    """Alias for add_handle — punch a hole/tunnel between two faces."""
    return add_handle(mesh, face1, face2)


# ── extrude_face_dome ────────────────────────────────────────────────────────

_FACE_DOME_HEIGHTS = (0.3, 0.18, 0.1, 0.05, 0.025)
_FACE_DOME_SCALES  = (1.7, 1.6, 1.4, 1.2, 1.1)


def extrude_face_dome(mesh: DLFLMesh, face: Face,
                      length: float = 1.0, sf: float = 1.0) -> Face:
    """
    Dome-shaped extrusion of a single face (topmod3d's extrudeFaceDome).

    5 successive DS-style extrusions with decreasing height and increasing
    scale, followed by a final stellation to close the dome apex.

    In-place.  Returns the final apex face (tiny top cap).
    """
    import math

    # Average boundary edge length for height unit
    vs = face.vertices()
    d = len(vs)
    per = 0.0
    for i in range(d):
        a, b = vs[i], vs[(i + 1) % d]
        per += math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
    unit = per / d

    current = face
    for h, s in zip(_FACE_DOME_HEIGHTS, _FACE_DOME_SCALES):
        new_faces = extrude_face(mesh, current, dist=h * length * unit)
        top = new_faces[0]
        # DS ring repositioning
        ring = top.vertices()
        n = len(ring)
        pts = [(v.x, v.y, v.z) for v in ring]
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        cz = sum(p[2] for p in pts) / n
        scale = s * sf
        for k, v in enumerate(ring):
            p = pts[k]
            a = pts[(k - 1) % n]
            b = pts[(k + 1) % n]
            dsx = (p[0] + cx + (p[0] + a[0]) / 2 + (p[0] + b[0]) / 2) / 4.0
            dsy = (p[1] + cy + (p[1] + a[1]) / 2 + (p[1] + b[1]) / 2) / 4.0
            dsz = (p[2] + cz + (p[2] + a[2]) / 2 + (p[2] + b[2]) / 2) / 4.0
            v.x = cx + scale * (dsx - cx)
            v.y = cy + scale * (dsy - cy)
            v.z = cz + scale * (dsz - cz)
        current = top

    # Final stellation to close dome
    stellate(mesh, current)
    return current


# ── makeWireframe ────────────────────────────────────────────────────────────

def make_wireframe(mesh: DLFLMesh, thickness: float = 0.1) -> DLFLMesh:
    """
    Wireframe generation (topmod3d's makeWireframe).

    Pipeline: modified_corner_cutting → create_crust → punch matching holes.
    The result is a hollow wireframe — each original edge becomes a beam,
    each original face becomes a hole.

    Returns a new mesh.
    """
    from .remeshing import modified_corner_cutting, create_crust

    # Step 1: modified corner cutting (insets faces, creates quad bridges)
    wire = modified_corner_cutting(mesh, thickness=thickness)

    # Step 2: create crust (double-wall shell)
    shelled, pairs = create_crust(wire, thickness=thickness)

    # Step 3: punch holes at faces that correspond to the inner inset faces
    # In topmod3d this uses FTHole marking; we punch all face pairs that
    # correspond to the original inset faces (the first F faces of wire).
    n_orig_faces = mesh.F()
    for i in range(n_orig_faces):
        if i < len(pairs):
            outer, inner = pairs[i]
            if outer.degree() == inner.degree():
                add_handle(shelled, outer, inner)

    return shelled
