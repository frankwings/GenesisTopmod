"""
High-level mesh operations, composed from the 4 fundamental operators.

extrude_face(mesh, face, dist)           -> list[Face]
add_handle(mesh, face1, face2)           -> list[Edge]
stellate(mesh, face)                     -> Vertex
subdivide_edge(mesh, edge)               -> Vertex
subdivide_face(mesh, face)               -> Vertex
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
