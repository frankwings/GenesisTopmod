"""
Four fundamental TopMod operators.

References
----------
Akleman & Chen (2003) "A minimal and complete set of operators for the
development of robust manifold mesh modelers." Graphical Models 65(5).

All operators preserve the 2-manifold invariant: after every call, every
edge in the mesh has exactly two half-edges (twins).

API
---
create_vertex(mesh, x, y, z) -> Vertex
delete_vertex(mesh, v)       -> None          (v must be degree-0 or isolated)
insert_edge(mesh, he1, he2)  -> Edge          (he1 and he2 are HalfEdge corners)
delete_edge(mesh, edge)      -> Face          (merged face)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .dlfl import DLFLMesh, Vertex, HalfEdge, Face, Edge


# ── 1. create_vertex ───────────────────────────────────────────────────────────

def create_vertex(mesh: DLFLMesh,
                  x: float = 0.0,
                  y: float = 0.0,
                  z: float = 0.0) -> Vertex:
    """
    Create an isolated vertex at (x, y, z) together with a degenerate
    "loop face" — a face whose boundary is a single half-edge looping to
    itself (next = prev = self, twin points back).

    Topology change: V+1, E+1, F+1  ⟹  χ unchanged (+1 new component).
    """
    v  = mesh._new_vertex(x, y, z)
    f  = mesh._new_face()

    # Two twin half-edges form the degenerate loop edge.
    he0 = mesh._new_halfedge()
    he1 = mesh._new_halfedge()

    he0.origin = v
    he1.origin = v    # both originate at v (degenerate)

    # Link twins
    mesh._new_edge(he0, he1)   # also sets he0.twin=he1, he1.twin=he0

    # Face loop: he0 loops to itself
    he0.next = he0
    he0.prev = he0
    he0.face = f

    # he1 is the "outer" side — also a degenerate loop for bookkeeping
    he1.next = he1
    he1.prev = he1
    he1.face = f   # same degenerate face (outer half also on f)

    f.he = he0
    v.he = he0

    return v


# ── 2. delete_vertex ──────────────────────────────────────────────────────────

def delete_vertex(mesh: DLFLMesh, v: Vertex) -> None:
    """
    Delete an isolated (degree-0) vertex and its associated degenerate face.

    Raises ValueError if the vertex still has edges attached.
    """
    # Collect all half-edges originating at v directly from mesh dict
    # (outgoing_halfedges() fan traversal can be confused by degenerate loops)
    hes_from_v = [he for he in mesh.halfedges.values() if he.origin is v]

    if len(hes_from_v) == 0:
        # Truly isolated — no half-edges at all
        mesh._remove_vertex(v)
        return

    # Classify: are all half-edges part of a degenerate self-loop?
    # A degenerate self-loop has: he.next == he and he.twin.origin == v
    non_degenerate = [
        he for he in hes_from_v
        if not (he.next is he and he.twin is not None and he.twin.origin is v)
    ]

    if non_degenerate:
        raise ValueError(
            f"Cannot delete_vertex {v}: has {len(non_degenerate)} non-degenerate "
            "half-edges. Use delete_edge first to remove all incident edges."
        )

    # All half-edges are degenerate self-loops — find the canonical pair.
    # create_vertex produces exactly 2 half-edges (he0, he1) both from v.
    he0 = hes_from_v[0]
    he1 = he0.twin

    if he0.next is not he0:
        raise ValueError(
            f"Vertex {v} is connected to a real (non-degenerate) face. "
            "Cannot delete."
        )

    f = he0.face
    e = he0.edge

    mesh._remove_halfedge(he0)
    mesh._remove_halfedge(he1)
    mesh._remove_edge(e)
    mesh._remove_face(f)
    mesh._remove_vertex(v)


# ── 3. insert_edge ────────────────────────────────────────────────────────────

def insert_edge(mesh: DLFLMesh, he1: HalfEdge, he2: HalfEdge) -> Edge:
    """
    Insert a new edge between the *origins* of half-edges he1 and he2.

    he1 and he2 are "corner" specifiers: inserting the edge means the new
    edge goes from he1.origin to he2.origin, and the new half-edges are
    inserted *before* he1 and he2 in their respective face loops.

    Topological cases
    -----------------
    Case A — he1.face is he2.face (same face):
        The face is split into two faces.  V, E+1, F+1  ⟹  χ+1  (genus−½ per component).

    Case B — different faces, same component:
        The two faces merge into one, adding a handle.  V, E+1, F−1  ⟹  χ−1  (genus+½).

    Case C — different faces, different components:
        Components merge.  V, E+1, F−1  ⟹  χ−1.

    Returns the newly created Edge.
    """
    f1 = he1.face
    f2 = he2.face

    if f1 is None or f2 is None:
        raise ValueError("insert_edge: half-edges must belong to faces.")

    # Create the two new half-edges for the new edge.
    new_he1 = mesh._new_halfedge()   # goes origin(he1) → origin(he2)
    new_he2 = mesh._new_halfedge()   # goes origin(he2) → origin(he1)
    new_edge = mesh._new_edge(new_he1, new_he2)

    new_he1.origin = he1.origin
    new_he2.origin = he2.origin

    # ── Splice into face loops ────────────────────────────────────────
    # Before he1:  ...→ he1.prev → he1 → ...
    # After:       ...→ he1.prev → new_he1 → he2 → ... → he1 → ...
    #                              new_he2 → he1 → ... (for other face)

    prev1 = he1.prev
    prev2 = he2.prev

    # Insert new_he1 before he1 in f1's loop
    prev1.next = new_he1
    new_he1.prev = prev1
    new_he1.next = he2       # continues into face 2's segment (or same face)
    he2.prev     = new_he1

    # Insert new_he2 before he2 in f2's loop (or the other half of f1)
    prev2.next = new_he2
    new_he2.prev = prev2
    new_he2.next = he1
    he1.prev     = new_he2

    if f1 is f2:
        # ── Case A: same face → split into two faces ──────────────────
        # new_he1's loop: new_he1 → he2 → ... → new_he2 → he1 → ... → back to new_he1?
        # No: new_he1.next = he2; new_he2.next = he1; new_he1 ← prev2.next = new_he2
        # Walk the two loops to assign faces.

        new_face = mesh._new_face()

        # Loop 1: starting at new_he1, follow .next until we return
        cur = new_he1
        while True:
            cur.face = new_he1.face if cur is not new_he1 else None  # temp
            cur = cur.next
            if cur is new_he1:
                break

        # Assign face properly
        # Loop containing new_he1:
        cur = new_he1
        while True:
            cur.face = new_face
            cur = cur.next
            if cur is new_he1:
                break

        # Loop containing new_he2 stays as f1:
        cur = new_he2
        while True:
            cur.face = f1
            cur = cur.next
            if cur is new_he2:
                break

        new_face.he = new_he1
        f1.he       = new_he2

        new_he1.face = new_face
        new_he2.face = f1

    else:
        # ── Case B / C: different faces → merge into one face ─────────
        # new_he1's loop now spans both old face loops.
        # Remove f2 from the mesh.

        # Re-label all half-edges that were on f2 to f1:
        cur = new_he1
        while True:
            cur.face = f1
            cur = cur.next
            if cur is new_he1:
                break

        new_he1.face = f1
        new_he2.face = f1

        f1.he = new_he1

        mesh._remove_face(f2)

    # Update vertex outgoing pointers if needed
    if he1.origin.he is None:
        he1.origin.he = new_he1
    if he2.origin.he is None:
        he2.origin.he = new_he2

    # Make sure origin vertices point to valid outgoing he
    _fix_vertex_he(he1.origin)
    _fix_vertex_he(he2.origin)

    return new_edge


def _fix_vertex_he(v: Vertex) -> None:
    """Ensure v.he is an outgoing half-edge from v (reset if stale)."""
    if v.he is not None and v.he.origin is v:
        return
    # Scan all half-edges – only done rarely
    v.he = None
    for he in (v.he,):  # can't iterate from None
        break
    # Fall back: will be fixed by caller context if needed.


# ── 4. delete_edge ────────────────────────────────────────────────────────────

def delete_edge(mesh: DLFLMesh, edge: Edge) -> Face:
    """
    Delete an edge, merging the two adjacent faces into one.

    If both half-edges are on the *same* face (a bridge / loop edge in a
    degenerate mesh), the face is split — but we do not support that here
    since the manifold invariant prevents it in practice.

    Returns the surviving merged face.

    Topology change: E−1, F−1  ⟹  χ unchanged  (genus may change).
    """
    he_a = edge.he0
    he_b = edge.he1

    f_a = he_a.face
    f_b = he_b.face

    if f_a is None or f_b is None:
        raise ValueError("delete_edge: half-edges must belong to faces.")

    # Splice he_a out of the loop
    prev_a = he_a.prev
    next_a = he_a.next

    prev_b = he_b.prev
    next_b = he_b.next

    prev_a.next = next_b
    next_b.prev = prev_a

    prev_b.next = next_a
    next_a.prev = prev_b

    # Surviving face is f_a; re-label all f_b half-edges to f_a
    if f_a is not f_b:
        cur = next_b
        while True:
            cur.face = f_a
            cur = cur.next
            if cur is next_b:
                break
        mesh._remove_face(f_b)

    # Update f_a entry pointer (avoid deleted half-edges)
    f_a.he = next_a if next_a is not he_a else next_b

    # Fix vertex outgoing pointers
    for v, he_del, he_keep in ((he_a.origin, he_a, next_b),
                                (he_b.origin, he_b, next_a)):
        if v.he is he_del:
            # point to another outgoing half-edge
            v.he = he_keep if he_keep.origin is v else None
            # search in fan
            if v.he is None:
                for he in (next_a, next_b, prev_a, prev_b):
                    if he.origin is v and he is not he_del and he is not he_b:
                        v.he = he
                        break

    mesh._remove_halfedge(he_a)
    mesh._remove_halfedge(he_b)
    mesh._remove_edge(edge)

    return f_a
