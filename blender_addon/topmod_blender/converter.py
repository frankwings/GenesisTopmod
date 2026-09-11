"""
BMesh ↔ DLFLMesh converter.

This is the only file that touches both the Blender API (bmesh) and
the topmod core.  The topmod package itself never imports bpy.

Both directions go through per-corner ``(vertex index, edge index)`` lists
rather than face-vertex lists, because a DLFL mesh can hold topology a
vertex list cannot describe:

* a face may walk the same edge twice — the cross-face ``insert_edge``
  merge, whose two faces become one;
* two edges may join the same pair of vertices — inserting between corners
  that are already joined, which yields a 2-gon;
* an edge may run from a vertex to itself.

Blender stores all three quite happily (verified on 4.4 and 5.3, through
Edit Mode round trips and ``.blend`` save/reload), but neither
``bmesh.faces.new`` nor ``bmesh.edges.new`` will *build* them, and
``Mesh.from_pydata`` resolves a corner's edge by looking its vertex pair
up, which collapses parallel edges.  So the writer fills Blender's mesh
arrays directly — including each loop's ``edge_index`` — and the reader
takes corners straight off the BMesh loops.

Never call ``mesh.validate()`` on the result: it deletes duplicate edges and
faces that repeat a vertex.

Public API
----------
bmesh_to_dlfl(bm) -> DLFLMesh                 # with _bv/_bf/_be index maps
dlfl_to_bmesh(mesh, bm, obj=None)
apply_op(context, op_fn, ...) -> DLFLMesh     # global ops
apply_local_face_op(context, op_fn, ...)      # selected-face ops
apply_local_edge_op(context, op_fn, ...)      # selected-edge ops
apply_two_face_op(context, op_fn, ...)        # exactly-2-selected-face ops
apply_insert_edge_corners(context, a, b)      # two picked corners
apply_delete_vertex(context)                  # one selected isolated vertex
"""

from __future__ import annotations

from typing import Callable, List, Optional

import bmesh
import bpy
from mathutils import Vector

# topmod core is shipped as a sub-package inside the addon
from .topmod.dlfl import DLFLMesh
from .corner_rules import (MIN_FACE_CORNERS, build_dlfl_from_corners,
                           degenerate_edges, dlfl_corner_arrays,
                           unwritable_faces)


def bmesh_to_dlfl(bm: bmesh.types.BMesh) -> DLFLMesh:
    """
    Convert a Blender BMesh to a DLFLMesh.

    Each face is read as its loops' ``(vertex index, edge index)`` pairs, so
    two parallel edges stay distinct.  The BMesh must be a closed,
    orientable 2-manifold; ValueError otherwise.

    The result carries three lookup tables used by the operators:
    ``_bv_map`` (BMesh vertex index -> Vertex), ``_bf_map`` (face index ->
    Face) and ``_be_map`` (edge index -> Edge).  All three are exact, keyed
    on Blender's own indices rather than resolved by position or by vertex
    pair.
    """
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    positions = [(v.co.x, v.co.y, v.co.z) for v in bm.verts]
    faces_corners = [[(loop.vert.index, loop.edge.index) for loop in f.loops]
                     for f in bm.faces]

    mesh, edges_by_index = build_dlfl_from_corners(positions, faces_corners)

    dlfl_verts = list(mesh.vertices.values())
    dlfl_faces = list(mesh.faces.values())
    mesh._bv_map = {i: dlfl_verts[i] for i in range(len(dlfl_verts))}
    mesh._bf_map = {f.index: dlfl_faces[i] for i, f in enumerate(bm.faces)}
    mesh._be_map = dict(edges_by_index)

    return mesh


def _load_dlfl_into_bmesh(bm: bmesh.types.BMesh, mesh: DLFLMesh) -> None:
    """
    Replace the contents of *bm* with *mesh*.

    Fills a throwaway ``Mesh``'s arrays directly — vertices, edges, and one
    loop per corner carrying both its vertex *and* its edge — then hands
    that to ``bm.from_mesh``.  Writing the loop's edge explicitly is what
    keeps two parallel edges apart; ``Mesh.from_pydata`` would look the edge
    up by vertex pair and merge them.

    ``mesh.validate()`` is never called: it deletes duplicate edges and any
    face that repeats a vertex.
    """
    coords, edge_pairs, faces_corners = dlfl_corner_arrays(mesh)

    # A single-corner polygon crashes Blender's Mesh->BMesh conversion.
    # create_vertex's degenerate loop face is the only source of one.
    writable = [corners for corners in faces_corners
                if len(corners) >= MIN_FACE_CORNERS]

    corner_verts: List[int] = []
    corner_edges: List[int] = []
    loop_starts: List[int] = []
    for corners in writable:
        loop_starts.append(len(corner_verts))
        for vertex_index, edge_index in corners:
            corner_verts.append(vertex_index)
            corner_edges.append(edge_index)

    scratch = bpy.data.meshes.new("_topmod_scratch")
    try:
        scratch.vertices.add(len(coords))
        scratch.vertices.foreach_set(
            "co", [value for co in coords for value in co])
        scratch.edges.add(len(edge_pairs))
        scratch.edges.foreach_set(
            "vertices", [index for pair in edge_pairs for index in pair])
        scratch.loops.add(len(corner_verts))
        scratch.loops.foreach_set("vertex_index", corner_verts)
        scratch.loops.foreach_set("edge_index", corner_edges)
        scratch.polygons.add(len(writable))
        scratch.polygons.foreach_set("loop_start", loop_starts)
        scratch.update()

        bm.clear()
        bm.from_mesh(scratch)
    finally:
        bpy.data.meshes.remove(scratch)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.normal_update()


def dlfl_to_bmesh(mesh: DLFLMesh, bm: bmesh.types.BMesh,
                  obj: Optional[bpy.types.Object] = None) -> None:
    """
    Replace the contents of *bm* with the geometry from *mesh*.

    If *obj* is given, the mesh data is written back to the Blender
    object and the BMesh is freed.
    """
    _load_dlfl_into_bmesh(bm, mesh)

    if obj is not None:
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()


def apply_op(context: bpy.types.Context,
             op_fn: Callable,
             returns_new: bool = True,
             **kwargs) -> Optional[DLFLMesh]:
    """
    Convenience wrapper: get the active mesh as DLFL, apply an operator,
    write the result back.

    Parameters
    ----------
    context    : Blender context (from operator.execute)
    op_fn      : a topmod operator function
    returns_new: True if op_fn returns a new DLFLMesh (e.g. catmull_clark);
                 False if it mutates in place (e.g. stellate_all, dome)
    **kwargs   : forwarded to op_fn

    Returns the resulting DLFLMesh (or None on error).
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None

    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    dlfl = bmesh_to_dlfl(bm)

    result = op_fn(dlfl, **kwargs)
    out = result if returns_new else dlfl

    # For ops that return a tuple (e.g. create_crust -> (mesh, pairs))
    if isinstance(out, tuple):
        out = out[0]

    # Clear the existing edit-mode BMesh and rebuild from the DLFL result
    _load_dlfl_into_bmesh(bm, out)
    bmesh.update_edit_mesh(me)

    return out


def _rebuild_bmesh(bm, dlfl_mesh, me):
    """Clear bm and rebuild from DLFL mesh, then update edit mesh."""
    _load_dlfl_into_bmesh(bm, dlfl_mesh)
    bmesh.update_edit_mesh(me)


def apply_local_face_op(context, op_fn, **kwargs):
    """
    Apply an operator to each SELECTED face.

    The op_fn signature must be op_fn(mesh, face, **kwargs).
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()

    selected = [f.index for f in bm.faces if f.select]
    if not selected:
        return None

    dlfl = bmesh_to_dlfl(bm)
    bf_map = dlfl._bf_map

    for fi in selected:
        df = bf_map.get(fi)
        if df is not None and df.id in dlfl.faces:
            op_fn(dlfl, df, **kwargs)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl


def apply_local_edge_op(context, op_fn, **kwargs):
    """
    Apply an operator to each SELECTED edge.

    The op_fn signature must be op_fn(mesh, edge, **kwargs).
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.edges.ensure_lookup_table()

    selected = [e.index for e in bm.edges if e.select]
    if not selected:
        return None

    dlfl = bmesh_to_dlfl(bm)
    be_map = dlfl._be_map

    for ei in selected:
        de = be_map.get(ei)
        if de is not None and de.id in dlfl.edges:
            op_fn(dlfl, de, **kwargs)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl


def apply_two_face_op(context, op_fn, **kwargs):
    """
    Apply an operator that needs exactly 2 selected faces.

    The op_fn signature must be op_fn(mesh, face1, face2, **kwargs).
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()

    selected = [f.index for f in bm.faces if f.select]
    if len(selected) != 2:
        return "select_error"

    dlfl = bmesh_to_dlfl(bm)
    bf_map = dlfl._bf_map
    f1 = bf_map.get(selected[0])
    f2 = bf_map.get(selected[1])
    if f1 is None or f2 is None:
        return None

    op_fn(dlfl, f1, f2, **kwargs)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl


def apply_insert_edge_corners(context, corner_a, corner_b):
    """
    insert_edge from two picked *corners*.

    Each corner is a ``(face_index, corner_position)`` pair as recorded by
    the interactive picker in ``corner_pick.py``.  ``corner_position`` is a
    position in the face's loop order, which is also the DLFL face's
    half-edge order because ``build_dlfl_from_corners`` wires them in that
    order.  Resolving by position rather than by vertex matters once a face
    visits a vertex more than once, which a merged face does.

    A corner is precisely one half-edge, which is what ``insert_edge``
    takes.  Corners on one face split it; corners on two faces merge them
    into a single face whose boundary runs through the new edge once per
    direction (genus +1).  Corners whose vertices are already joined add a
    second, parallel edge and a 2-gon.  All of those round-trip through
    Blender; a self-loop would not, and is refused.

    Returns ``(dlfl_mesh, new_edge_index)`` on success, where the index
    locates the inserted edge in the rebuilt BMesh, or a human-readable
    error string on failure. The index is what identifies it: with a
    parallel edge alongside, its vertex pair no longer would.
    """
    from .topmod.operators import insert_edge as _insert_edge
    from .corner_rules import corner_pair_error, resolve_corner_halfedge

    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return "No mesh in Edit Mode"
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()

    picked = []           # (face_index, corner, vert_index)
    for face_index, corner in (corner_a, corner_b):
        if not 0 <= face_index < len(bm.faces):
            return f"Face {face_index} no longer exists"
        face = bm.faces[face_index]
        corner %= len(face.loops)
        picked.append((face_index, corner,
                       face.loops[corner].vert.index))

    (fa, ca, va), (fb, cb, vb) = picked
    reason = corner_pair_error(fa, ca, va, fb, cb, vb)
    if reason is not None:
        return f"Cannot insert edge: {reason}"

    try:
        dlfl = bmesh_to_dlfl(bm)
    except ValueError as exc:
        return f"Mesh must be a closed 2-manifold ({exc})"

    halfedges = []
    for face_index, corner, _vert_index in picked:
        dlfl_face = dlfl._bf_map.get(face_index)
        if dlfl_face is None:
            return "Could not map the picked corner onto the DLFL mesh"
        he = resolve_corner_halfedge(dlfl_face, corner)
        if he is None:
            return f"Face {face_index} has no corner {corner}"
        halfedges.append(he)

    new_edge = _insert_edge(dlfl, halfedges[0], halfedges[1])

    # Safety net behind corner_pair_error. Blender takes duplicate edges and
    # faces that repeat a vertex, but a single-corner polygon crashes its
    # Mesh->BMesh conversion and a self-loop edge hangs it. The BMesh is
    # still untouched here, so refusing costs nothing.
    broken = unwritable_faces(dlfl)
    if broken:
        return ("Cannot insert edge: the result contains "
                f"{len(broken)} face(s) with fewer than {MIN_FACE_CORNERS} "
                "corners, which Blender cannot store")
    loops = degenerate_edges(dlfl)
    if loops:
        return ("Cannot insert edge: the result contains "
                f"{len(loops)} self-loop edge(s), which hang Blender")

    # dlfl_corner_arrays numbers edges by DLFL insertion order and the
    # writer keeps that order, so the edge just appended is the last one.
    edge_order = list(dlfl.edges.values())
    new_edge_index = edge_order.index(new_edge)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl, new_edge_index


def apply_delete_vertex(context):
    """delete_vertex: user selects exactly 1 isolated vertex."""
    from .topmod.operators import delete_vertex as _delete_vertex

    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()

    selected = [v.index for v in bm.verts if v.select]
    if len(selected) != 1:
        return "select_error"

    dlfl = bmesh_to_dlfl(bm)
    dv = dlfl._bv_map[selected[0]]
    _delete_vertex(dlfl, dv)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl
