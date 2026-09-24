"""
Corner-level topology shared by the interactive picker and the converter.

A *corner* is a (face, vertex) pair — in DLFL terms exactly one half-edge:
the one originating at that vertex inside that face's boundary loop.

This module also owns the lossless description of a DLFL mesh that the
converter writes to and reads from Blender.  A face-vertex list is *not*
enough: once a mesh has two parallel edges between the same pair of
vertices — which is what inserting an edge between already-joined corners
produces, and which TopMod allows — a vertex list cannot say which of them
a corner uses.  So a face is described as a list of ``(vertex index, edge
index)`` corners instead, which is exactly Blender's own loop array
(``loop.vertex_index`` / ``loop.edge_index``).

Verified against Blender 4.4 and 5.3: two edges sharing a vertex pair, and
faces that repeat a vertex, survive Edit Mode round trips and ``.blend``
save/reload.  Two shapes do not survive contact with Blender and are refused
instead:

* a *one*-corner polygon crashes the Mesh→BMesh conversion outright, so the
  writer drops those;
* a degenerate ``(v, v)`` edge is *stored* correctly but then hangs routine
  BMesh work — normal computation and clearing selection both spin forever
  — so :func:`corner_pair_error` refuses the pick that would make one.

This module deliberately imports no ``bpy``/``bmesh`` so the topology can be
unit-tested headlessly (see ``tests/test_corner_rules.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:                                   # inside the addon package
    from .topmod.dlfl import DLFLMesh, Edge
except ImportError:                    # loaded standalone by the test suite
    from topmod.dlfl import DLFLMesh, Edge


# Blender's Mesh→BMesh conversion crashes on a polygon with a single corner.
MIN_FACE_CORNERS = 2


def corner_pair_error(face_a: int, corner_a: int, vert_a: int,
                      face_b: int, corner_b: int, vert_b: int) -> Optional[str]:
    """
    Return why an edge may not be inserted between two corners, or None.

    Two pairs are refused, both because the result hangs rather than because
    the topology is wrong:

    * the **same corner** twice — ``insert_edge`` splices both new
      half-edges against one half-edge, leaving a face loop that never
      closes, and the traversal afterwards spins;
    * **two corners on the same vertex** — a valid DLFL self-loop, and
      Blender stores it, but a degenerate ``(v, v)`` edge then hangs
      ``bm.normal_update()`` and selection clearing.  TopMod proper allows
      this; the Blender bridge cannot.

    Everything else goes through, including pairs Blender's own modelling
    tools could not make:

    * corners on two different faces (the faces merge, genus +1),
    * corners whose vertices already share an edge (a second, parallel edge
      and a 2-gon).

    The message is a lower-case sentence fragment, so callers can embed it
    ("Cannot insert edge: ...") or append it to viewport header text.
    """
    if face_a == face_b and corner_a == corner_b:
        return ("both picks are the same corner, which would leave a face "
                "loop that never closes")
    if vert_a == vert_b:
        return ("both corners are on the same vertex, and the self-loop edge "
                "that makes hangs Blender")
    return None


def resolve_corner_halfedge(face, corner: int):
    """
    Return the half-edge at position *corner* of a DLFL *face*.

    Resolved by position rather than by vertex: a merged face visits some
    vertices more than once, so the vertex alone no longer names a corner.
    The position is the same one Blender's loop array uses, because
    :func:`build_dlfl_from_corners` wires each face's half-edges in loop
    order.
    """
    halfedges = face.halfedges()
    if not halfedges:
        return None
    return halfedges[corner % len(halfedges)]


def dlfl_corner_arrays(mesh: DLFLMesh):
    """
    Flatten a DLFLMesh into ``(coords, edge_pairs, faces_corners)``.

    ``faces_corners`` holds one list of ``(vertex index, edge index)`` per
    face, in boundary order.  A corner's edge is the one leaving it along
    the face — ``half-edge.edge`` — which is precisely what Blender stores
    in ``loop.edge_index``.

    Indices are positions in the DLFL's own insertion order, so a mesh
    written from this and read straight back keeps every index.
    """
    vertex_list = list(mesh.vertices.values())
    vertex_pos = {v.id: i for i, v in enumerate(vertex_list)}
    edge_list = list(mesh.edges.values())
    edge_pos = {e.id: i for i, e in enumerate(edge_list)}

    coords = [(v.x, v.y, v.z) for v in vertex_list]
    edge_pairs = [(vertex_pos[e.he0.origin.id], vertex_pos[e.he1.origin.id])
                  for e in edge_list]

    faces_corners: List[List[Tuple[int, int]]] = []
    for face in mesh.faces.values():
        corners = []
        for he in face.halfedges():
            if he.edge is None or he.origin is None:
                raise ValueError(
                    f"half-edge {he.id} has no edge or origin; the DLFL mesh "
                    "is malformed")
            corners.append((vertex_pos[he.origin.id], edge_pos[he.edge.id]))
        faces_corners.append(corners)

    return coords, edge_pairs, faces_corners


def build_dlfl_from_corners(
        positions: Sequence[Sequence[float]],
        faces_corners: Sequence[Sequence[Tuple[int, int]]]
) -> Tuple[DLFLMesh, Dict[int, Edge]]:
    """
    Rebuild a DLFLMesh from per-corner ``(vertex index, edge index)`` lists.

    The counterpart of :func:`dlfl_corner_arrays`, and the reason the
    converter no longer goes through ``primitives._build_mesh``: that one
    keys half-edges by *directed vertex pair*, so two parallel edges
    overwrite each other and one of them silently loses its twin.  Pairing
    by edge index instead is unambiguous however many edges share a pair.

    Returns the mesh and a map from edge index to the DLFL ``Edge``, which
    the converter uses to key its BMesh-edge lookup table exactly.

    Raises ValueError if an edge is not used by exactly two corners, i.e.
    the mesh is not a closed 2-manifold.
    """
    mesh = DLFLMesh()
    verts = [mesh._new_vertex(*position) for position in positions]

    sides: Dict[int, List] = {}
    for corners in faces_corners:
        face = mesh._new_face()
        halfedges = []
        for vertex_index, edge_index in corners:
            he = mesh._new_halfedge()
            he.origin = verts[vertex_index]
            he.face = face
            halfedges.append(he)
            sides.setdefault(edge_index, []).append(he)

        count = len(halfedges)
        for k in range(count):
            halfedges[k].next = halfedges[(k + 1) % count]
            halfedges[k].prev = halfedges[(k - 1) % count]
        face.he = halfedges[0]
        for he in halfedges:
            if he.origin.he is None:
                he.origin.he = he

    edges_by_index: Dict[int, Edge] = {}
    for edge_index in sorted(sides):
        pair = sides[edge_index]
        if len(pair) != 2:
            raise ValueError(
                f"edge {edge_index} is used by {len(pair)} face corner(s), "
                "not 2: the mesh is not closed or is non-manifold")
        edges_by_index[edge_index] = mesh._new_edge(pair[0], pair[1])

    return mesh, edges_by_index


def unwritable_faces(mesh: DLFLMesh) -> List[Tuple[int, int]]:
    """
    Faces Blender cannot take, as ``(face id, corner count)``.

    Only one kind: a polygon with fewer than two corners.  Blender's
    Mesh→BMesh conversion crashes outright on a single-corner polygon, so
    the writer must never hand one over.  ``insert_edge`` cannot produce
    one; ``create_vertex`` can, as its degenerate loop face.
    """
    return [(face.id, len(face.halfedges()))
            for face in mesh.faces.values()
            if len(face.halfedges()) < MIN_FACE_CORNERS]


def degenerate_edges(mesh: DLFLMesh) -> List[Tuple[int, int]]:
    """
    Edges running from a vertex to itself, as ``(edge id, vertex id)``.

    Blender stores these, but they hang ``bm.normal_update()`` and selection
    clearing, so the converter refuses a result containing one rather than
    handing the user a frozen viewport.
    """
    return [(edge.id, edge.he0.origin.id)
            for edge in mesh.edges.values()
            if edge.he0.origin is edge.he1.origin]
