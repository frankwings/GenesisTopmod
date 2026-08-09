"""
BMesh ↔ DLFLMesh converter.

This is the only file that touches both the Blender API (bmesh) and
the topmod core.  The topmod package itself never imports bpy.

Public API
----------
bmesh_to_dlfl(bm) -> DLFLMesh
dlfl_to_bmesh(mesh, bm, obj=None)
apply_op(obj, op_fn, **kwargs) -> DLFLMesh   # convenience wrapper
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
    return _build_mesh(positions, face_indices)


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

    bm_new = bmesh.new()
    dlfl_to_bmesh(out, bm_new)

    # Replace edit mesh
    bm_new.to_mesh(me)
    bm_new.free()
    bmesh.update_edit_mesh(me)

    return out
