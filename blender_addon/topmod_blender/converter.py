"""
BMesh ↔ DLFLMesh converter.

This is the only file that touches both the Blender API (bmesh) and
the topmod core.  The topmod package itself never imports bpy.

Public API
----------
bmesh_to_dlfl(bm) -> (DLFLMesh, idx_maps)
dlfl_to_bmesh(mesh, bm, obj=None)
apply_op(context, op_fn, ...) -> DLFLMesh     # global ops
apply_local_face_op(context, op_fn, ...)      # selected-face ops
apply_local_edge_op(context, op_fn, ...)      # selected-edge ops
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Vector

# topmod core is shipped as a sub-package inside the addon
from .topmod.dlfl import DLFLMesh
from .topmod.primitives import _build_mesh


def bmesh_to_dlfl(bm: bmesh.types.BMesh) -> DLFLMesh:
    """
    Convert a Blender BMesh to a DLFLMesh.

    The BMesh must be a closed, orientable 2-manifold (no loose verts,
    no boundary edges, no non-manifold edges).  Raises ValueError
    otherwise (_build_mesh will fail on unpaired half-edges).
    """
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    positions: List[Tuple[float, float, float]] = [
        (v.co.x, v.co.y, v.co.z) for v in bm.verts
    ]
    face_indices: List[List[int]] = [
        [v.index for v in f.verts] for f in bm.faces
    ]
    mesh = _build_mesh(positions, face_indices)

    # Build index maps: BMesh index → DLFL element
    dlfl_verts = list(mesh.vertices.values())
    dlfl_faces = list(mesh.faces.values())
    dlfl_edges = list(mesh.edges.values())

    bv_to_dlfl_v = {i: dlfl_verts[i] for i in range(len(dlfl_verts))}
    bf_to_dlfl_f = {i: dlfl_faces[i] for i in range(len(dlfl_faces))}

    # Edge map: find DLFL edge by endpoint vertex indices
    be_to_dlfl_e: Dict[int, object] = {}
    for i, be in enumerate(bm.edges):
        v0_idx, v1_idx = be.verts[0].index, be.verts[1].index
        dv0, dv1 = dlfl_verts[v0_idx], dlfl_verts[v1_idx]
        de = mesh.find_edge(dv0, dv1)
        if de is not None:
            be_to_dlfl_e[i] = de

    mesh._bv_map = bv_to_dlfl_v
    mesh._bf_map = bf_to_dlfl_f
    mesh._be_map = be_to_dlfl_e

    return mesh


def dlfl_to_bmesh(mesh: DLFLMesh, bm: bmesh.types.BMesh,
                  obj: Optional[bpy.types.Object] = None) -> None:
    """
    Replace the contents of *bm* with the geometry from *mesh*.

    If *obj* is given, the mesh data is written back to the Blender
    object and the BMesh is freed.
    """
    bm.clear()

    # Vertices
    vid_to_bv: Dict[int, bmesh.types.BMVert] = {}
    for v in mesh.vertices.values():
        bv = bm.verts.new((v.x, v.y, v.z))
        vid_to_bv[v.id] = bv
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    # Faces
    for f in mesh.faces.values():
        verts = f.vertices()
        if len(verts) < 3:
            continue
        try:
            bm.faces.new([vid_to_bv[v.id] for v in verts])
        except ValueError:
            # Duplicate face (shouldn't happen with valid DLFL, but guard)
            pass
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()

    # Normals
    bm.normal_update()

    # Write back
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

    # Clear the existing edit-mode BMesh and rebuild from DLFL result
    bm.clear()

    vid_to_bv: Dict[int, bmesh.types.BMVert] = {}
    for v in out.vertices.values():
        bv = bm.verts.new((v.x, v.y, v.z))
        vid_to_bv[v.id] = bv
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for f in out.faces.values():
        verts = f.vertices()
        if len(verts) < 3:
            continue
        try:
            bm.faces.new([vid_to_bv[v.id] for v in verts])
        except ValueError:
            pass
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    bmesh.update_edit_mesh(me)

    return out


def _rebuild_bmesh(bm, dlfl_mesh, me):
    """Clear bm and rebuild from DLFL mesh, then update edit mesh."""
    bm.clear()
    vid_to_bv: Dict[int, bmesh.types.BMVert] = {}
    for v in dlfl_mesh.vertices.values():
        bv = bm.verts.new((v.x, v.y, v.z))
        vid_to_bv[v.id] = bv
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    for f in dlfl_mesh.faces.values():
        verts = f.vertices()
        if len(verts) < 3:
            continue
        try:
            bm.faces.new([vid_to_bv[v.id] for v in verts])
        except ValueError:
            pass
    bm.faces.ensure_lookup_table()
    bm.normal_update()
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


def _find_halfedge(dlfl, bv_map, vi_from, vi_to):
    """Find the DLFL half-edge from vi_from toward vi_to.

    Two vertices A→B define a directed edge, which corresponds to
    exactly one half-edge: the one originating at A whose face contains
    both A and B, and whose next walk goes toward B.

    Returns the half-edge, or None if not found.
    """
    dv_from = bv_map[vi_from]
    dv_to   = bv_map[vi_to]
    for he in dv_from.outgoing_halfedges():
        # Walk the face to check if dv_to is the next vertex
        if he.next.origin is dv_to:
            return he
    # Fallback: dv_to is on the same face but not immediately next
    for he in dv_from.outgoing_halfedges():
        face_verts = set(id(v) for v in he.face.vertices())
        if id(dv_to) in face_verts:
            return he
    return None


def apply_insert_edge(context):
    """
    insert_edge: user selects exactly 4 vertices **in order** (via
    select history).

    Vertices 1→2 define half-edge 1 (origin = V1, direction toward V2).
    Vertices 3→4 define half-edge 2 (origin = V3, direction toward V4).
    The new edge connects V1 and V3.

    The direction (A→B) determines which face the half-edge belongs to,
    eliminating all ambiguity in the cross-face case.
    """
    from .topmod.operators import insert_edge as _insert_edge

    obj = context.edit_object
    if obj is None or obj.type != 'MESH':
        return None
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Use select history to get ordered selection
    history = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert)]
    if len(history) != 4:
        return "select_error"

    v1, v2, v3, v4 = [v.index for v in history]

    dlfl = bmesh_to_dlfl(bm)
    bv_map = dlfl._bv_map

    # V1→V2 defines half-edge 1
    he0 = _find_halfedge(dlfl, bv_map, v1, v2)
    # V3→V4 defines half-edge 2
    he1 = _find_halfedge(dlfl, bv_map, v3, v4)

    if he0 is None:
        return "he1_error"
    if he1 is None:
        return "he2_error"

    _insert_edge(dlfl, he0, he1)

    _rebuild_bmesh(bm, dlfl, me)
    return dlfl


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
