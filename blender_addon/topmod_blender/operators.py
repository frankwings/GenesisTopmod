"""
Blender operators for all TopMod mesh operations.

Each topmod operator becomes a ``bpy.types.Operator`` with exposed
parameters (FloatProperty, IntProperty, etc.).  All operators:

1. Require Edit Mode on a mesh object.
2. Convert the BMesh to DLFLMesh, apply the topmod op, convert back.
3. Support Undo.

Operators are grouped into categories matching docs/operators.md:
  1. Fundamental   (insert_edge, delete_edge — local, need selection)
  2. High-level    (extrude_face, stellate, subdivide_edge/face,
                    add_handle, stellate_all)
  3. Classic subdivision (catmull_clark, dual, doo_sabin, simplest,
                          vertex_cutting, loop, sqrt3)
  4. TopMod remeshing (honeycomb, star, corner_cutting, loop_style,
                       fractal, pentagonal, pentagonal2, dual1264,
                       root4, checkerboard, ds_bc_new, dome)
  5. Structural    (create_crust)
"""

from __future__ import annotations

import bpy
from bpy.props import FloatProperty, IntProperty

from .converter import (apply_op, apply_local_face_op, apply_local_edge_op,
                        apply_two_face_op, apply_insert_edge,
                        apply_delete_vertex)

# Import topmod ops via the bundled sub-package
from .topmod import (
    catmull_clark, dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide,
    honeycomb_subdivide, star_subdivide, corner_cutting,
    loop_style_subdivide, fractal_subdivide,
    pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide,
    checkerboard_remesh, ds_bc_new_subdivide, dome_subdivide,
    create_crust,
    stellate_all,
    extrude_face, stellate, subdivide_face,
    # Batch 5
    stellate_subdivide, two_stellate_subdivide,
    subdivide_all_edges, subdivide_all_faces,
    triangulate_all, double_stellate_face,
    extrude_face_dome, make_wireframe,
    doo_sabin_bc, modified_corner_cutting, modified_corner_cutting2,
    create_crust_with_scaling,
    # Local ops
    extrude_face, stellate, subdivide_face, subdivide_edge,
    delete_edge, collapse_edge, trisect_edge, triangulate_face,
    add_handle, punch_hole,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a simple global operator (no face/edge selection needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_global_op(idname: str, label: str, description: str,
                    op_fn, returns_new: bool = True,
                    props: dict = None):
    """
    Factory for a Blender operator that applies a global topmod op.
    ``props`` is a dict of {attr_name: bpy.props.*Property(...)}.
    """
    # Capture op_fn and returns_new in closures — do NOT store as class
    # attributes, because Python binds functions-as-class-attrs via the
    # descriptor protocol (self is injected as the first argument).
    _captured_fn = op_fn
    _captured_returns_new = returns_new

    attrs = {
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": description,
        "bl_options": {'REGISTER', 'UNDO'},
    }

    # Blender 2.8+ uses __annotations__ for property registration.
    # bpy.props.*Property() returns tuples that Blender's metaclass
    # picks up from the class __annotations__ dict during register_class.
    if props:
        attrs["__annotations__"] = dict(props)

    def execute(self, context):
        kwargs = {}
        if props:
            for k in props:
                kwargs[k] = getattr(self, k)
        try:
            result = apply_op(context, _captured_fn,
                              returns_new=_captured_returns_new, **kwargs)
            if result is None:
                self.report({'ERROR'}, "Failed — is the mesh a closed "
                            "2-manifold?")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.edit_object is not None)

    attrs["execute"] = execute
    attrs["poll"] = classmethod(poll)

    return type(idname.replace(".", "_").upper(), (bpy.types.Operator,), attrs)


def _make_face_op(idname, label, description, op_fn, props=None):
    """Factory for operators that act on SELECTED FACES."""
    _captured_fn = op_fn

    attrs = {
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": description + " (apply to selected faces)",
        "bl_options": {'REGISTER', 'UNDO'},
    }
    if props:
        attrs["__annotations__"] = dict(props)

    def execute(self, context):
        kwargs = {}
        if props:
            for k in props:
                kwargs[k] = getattr(self, k)
        try:
            result = apply_local_face_op(context, _captured_fn, **kwargs)
            if result is None:
                self.report({'ERROR'}, "Select at least one face")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.edit_object is not None)

    attrs["execute"] = execute
    attrs["poll"] = classmethod(poll)
    return type(idname.replace(".", "_").upper(), (bpy.types.Operator,), attrs)


def _make_edge_op(idname, label, description, op_fn, props=None):
    """Factory for operators that act on SELECTED EDGES."""
    _captured_fn = op_fn

    attrs = {
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": description + " (apply to selected edges)",
        "bl_options": {'REGISTER', 'UNDO'},
    }
    if props:
        attrs["__annotations__"] = dict(props)

    def execute(self, context):
        kwargs = {}
        if props:
            for k in props:
                kwargs[k] = getattr(self, k)
        try:
            result = apply_local_edge_op(context, _captured_fn, **kwargs)
            if result is None:
                self.report({'ERROR'}, "Select at least one edge")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.edit_object is not None)

    attrs["execute"] = execute
    attrs["poll"] = classmethod(poll)
    return type(idname.replace(".", "_").upper(), (bpy.types.Operator,), attrs)


def _make_two_face_op(idname, label, description, op_fn):
    """Factory for operators that need exactly 2 selected faces."""
    _captured_fn = op_fn

    attrs = {
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": description + " (select exactly 2 faces)",
        "bl_options": {'REGISTER', 'UNDO'},
    }

    def execute(self, context):
        try:
            result = apply_two_face_op(context, _captured_fn)
            if result == "select_error":
                self.report({'ERROR'}, "Select exactly 2 faces")
                return {'CANCELLED'}
            if result is None:
                self.report({'ERROR'}, "Failed — check face selection")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}

    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.edit_object is not None)

    attrs["execute"] = execute
    attrs["poll"] = classmethod(poll)
    return type(idname.replace(".", "_").upper(), (bpy.types.Operator,), attrs)


def _make_special_op(idname, label, description, execute_fn):
    """Factory for ops with custom execute (insert_edge, delete_vertex)."""
    attrs = {
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": description,
        "bl_options": {'REGISTER', 'UNDO'},
    }
    attrs["execute"] = execute_fn

    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.edit_object is not None)

    attrs["poll"] = classmethod(poll)
    return type(idname.replace(".", "_").upper(), (bpy.types.Operator,), attrs)


# ─────────────────────────────────────────────────────────────────────────────
# Operator classes
# ─────────────────────────────────────────────────────────────────────────────

# -- 2. High-level (global) ------------------------------------------------

TOPMOD_OT_stellate_all = _make_global_op(
    "topmod.stellate_all",
    "Stellate All",
    "Stellate every face (pyramid on each face → all-triangle mesh)",
    stellate_all, returns_new=False,
)

# -- 3. Classic subdivision ------------------------------------------------

TOPMOD_OT_catmull_clark = _make_global_op(
    "topmod.catmull_clark",
    "Catmull-Clark",
    "Catmull-Clark subdivision (all-quad output)",
    catmull_clark,
)

TOPMOD_OT_dual = _make_global_op(
    "topmod.dual",
    "Dual",
    "Combinatorial dual (faces ↔ vertices)",
    dual,
)

TOPMOD_OT_doo_sabin = _make_global_op(
    "topmod.doo_sabin",
    "Doo-Sabin",
    "Doo-Sabin subdivision (corner-cutting)",
    doo_sabin,
)

TOPMOD_OT_simplest = _make_global_op(
    "topmod.simplest",
    "Simplest (Mid-Edge)",
    "Mid-edge / Peters-Reif subdivision",
    simplest_subdivide,
)

TOPMOD_OT_vertex_cutting = _make_global_op(
    "topmod.vertex_cutting",
    "Vertex Cutting",
    "Truncate every vertex",
    vertex_cutting,
    props={"offset": FloatProperty(
        name="Offset", default=0.25, min=0.01, max=0.49,
        description="Corner-cut depth")},
)

TOPMOD_OT_loop = _make_global_op(
    "topmod.loop_subdivide",
    "Loop",
    "Loop subdivision (triangle meshes only)",
    loop_subdivide,
)

TOPMOD_OT_sqrt3 = _make_global_op(
    "topmod.sqrt3",
    "√3",
    "√3 subdivision / Kobbelt (triangle meshes only)",
    sqrt3_subdivide,
)

# -- 4. TopMod remeshing ---------------------------------------------------

TOPMOD_OT_honeycomb = _make_global_op(
    "topmod.honeycomb",
    "Honeycomb",
    "Honeycomb subdivision (dual ∘ stellate_all)",
    honeycomb_subdivide,
)

TOPMOD_OT_star = _make_global_op(
    "topmod.star",
    "Star",
    "Star subdivision (stellate_all × 2, with optional spike offset)",
    star_subdivide, returns_new=False,
    props={"offset": FloatProperty(
        name="Offset", default=0.0, min=0.0, max=2.0,
        description="Spike height along face normals")},
)

TOPMOD_OT_corner_cutting = _make_global_op(
    "topmod.corner_cutting",
    "Corner Cutting",
    "Corner-cutting subdivision (parameterized Doo-Sabin variant)",
    corner_cutting,
    props={"alpha": FloatProperty(
        name="Alpha", default=0.5, min=0.01, max=0.99,
        description="Tension (diagonal weight)")},
)

TOPMOD_OT_loop_style = _make_global_op(
    "topmod.loop_style",
    "Loop-Style",
    "Loop connectivity generalized to arbitrary polygons",
    loop_style_subdivide,
    props={"length": FloatProperty(
        name="Length", default=1.0, min=0.0, max=1.0,
        description="Old-vertex blend (1 = keep position)")},
)

TOPMOD_OT_fractal = _make_global_op(
    "topmod.fractal",
    "Fractal",
    "Fractal subdivision (loop_style + stellated spikes)",
    fractal_subdivide,
    props={"offset": FloatProperty(
        name="Offset", default=1.0, min=0.0, max=5.0,
        description="Spike height factor")},
)

TOPMOD_OT_pentagonal = _make_global_op(
    "topmod.pentagonal",
    "Pentagonal",
    "Pentagonal subdivision (all-pentagon output)",
    pentagonal_subdivide,
    props={"offset": FloatProperty(
        name="Offset", default=0.0, min=0.0, max=1.0,
        description="Pull spoke neighbors toward centroid")},
)

TOPMOD_OT_pentagonal2 = _make_global_op(
    "topmod.pentagonal2",
    "Pentagonal 2",
    "Second pentagonal variant (inner d-gon + pentagons)",
    pentagonal2_subdivide,
    props={"scale_factor": FloatProperty(
        name="Scale Factor", default=0.75, min=0.1, max=1.0,
        description="Inner-polygon shrink factor")},
)

TOPMOD_OT_dual1264 = _make_global_op(
    "topmod.dual1264",
    "Dual 12.6.4",
    "Dual 12.6.4 subdivision (dodecagon/hexagon/quad tiling)",
    dual1264_subdivide,
    props={"sf": FloatProperty(
        name="Scale", default=1.0, min=0.1, max=2.0,
        description="Inner-polygon scale")},
)

TOPMOD_OT_root4 = _make_global_op(
    "topmod.root4",
    "Root-4",
    "Root-4 subdivision (honeycomb-mask inner polygons + hexagon bridges)",
    root4_subdivide,
    props={
        "a": FloatProperty(
            name="Smoothing", default=0.0, min=0.0, max=1.0,
            description="Old-vertex smoothing blend"),
        "twist": FloatProperty(
            name="Twist", default=0.0, min=-1.0, max=1.0,
            description="Inner-ring sampling shift"),
    },
)

TOPMOD_OT_checkerboard = _make_global_op(
    "topmod.checkerboard",
    "Checkerboard",
    "Checkerboard remeshing (inset + trisect + chamfer)",
    checkerboard_remesh,
    props={"thickness": FloatProperty(
        name="Thickness", default=0.25, min=0.01, max=0.49,
        description="Inset / trisection ratio")},
)

TOPMOD_OT_ds_bc_new = _make_global_op(
    "topmod.ds_bc_new",
    "DS BC-New",
    "Doo-Sabin BC-new variant",
    ds_bc_new_subdivide,
    props={
        "sf": FloatProperty(
            name="Scale", default=1.0, min=0.1, max=2.0,
            description="DS corner scale"),
        "length": FloatProperty(
            name="Length", default=1.0, min=0.0, max=1.0,
            description="Old-vertex blend"),
    },
)

TOPMOD_OT_dome = _make_global_op(
    "topmod.dome",
    "Dome",
    "Dome subdivision (7-layer extrusion domes on every face)",
    dome_subdivide, returns_new=False,
    props={
        "length": FloatProperty(
            name="Length", default=1.0, min=0.0, max=3.0,
            description="Height profile scale"),
        "sf": FloatProperty(
            name="Scale", default=1.0, min=0.0, max=3.0,
            description="Ring scale profile"),
    },
)

# -- 4b. TopMod remeshing (new batch 5) ------------------------------------

TOPMOD_OT_stellate_subdivide = _make_global_op(
    "topmod.stellate_subdivide",
    "Stellate Subdivide",
    "Stellate all faces then delete original edges",
    stellate_subdivide, returns_new=False,
)

TOPMOD_OT_two_stellate = _make_global_op(
    "topmod.two_stellate",
    "Two-Stellate",
    "Two-pass stellate subdivision with height and curvature control",
    two_stellate_subdivide, returns_new=False,
    props={
        "offset": FloatProperty(
            name="Offset", default=0.0, min=0.0, max=2.0,
            description="First-pass spike height"),
        "curve": FloatProperty(
            name="Curve", default=0.0, min=0.0, max=2.0,
            description="Second-pass curvature"),
    },
)

TOPMOD_OT_doo_sabin_bc = _make_global_op(
    "topmod.doo_sabin_bc",
    "Doo-Sabin BC",
    "Doo-Sabin BC: subdivide all edges then Doo-Sabin",
    doo_sabin_bc,
)

TOPMOD_OT_modified_cc = _make_global_op(
    "topmod.modified_cc",
    "Modified Corner Cutting",
    "Bisector-inset corner-cutting subdivision",
    modified_corner_cutting,
    props={"thickness": FloatProperty(
        name="Thickness", default=0.25, min=0.01, max=0.49,
        description="Inset depth along bisectors")},
)

TOPMOD_OT_modified_cc2 = _make_global_op(
    "topmod.modified_cc2",
    "Modified Corner Cutting 2",
    "Scale-based corner-cutting subdivision",
    modified_corner_cutting2,
    props={"scale": FloatProperty(
        name="Scale", default=0.25, min=0.01, max=0.99,
        description="Displacement scale along neighbor directions")},
)

# -- 2b. High-level (global, new batch 5) ---------------------------------

TOPMOD_OT_subdivide_all_edges = _make_global_op(
    "topmod.subdivide_all_edges",
    "Subdivide All Edges",
    "Split every edge at its midpoint",
    subdivide_all_edges, returns_new=False,
)

TOPMOD_OT_subdivide_all_faces = _make_global_op(
    "topmod.subdivide_all_faces",
    "Subdivide All Faces",
    "Fan-subdivide every face from centroid (= stellate all)",
    subdivide_all_faces, returns_new=False,
)

TOPMOD_OT_triangulate_all = _make_global_op(
    "topmod.triangulate_all",
    "Triangulate All",
    "Fan-triangulate every face from its first vertex",
    triangulate_all, returns_new=False,
)

TOPMOD_OT_make_wireframe = _make_global_op(
    "topmod.make_wireframe",
    "Make Wireframe",
    "Wireframe: corner-cutting → crust → punch holes",
    make_wireframe,
    props={"thickness": FloatProperty(
        name="Thickness", default=0.1, min=0.01, max=0.49,
        description="Beam width and shell thickness")},
)

# -- 5. Structural ---------------------------------------------------------

TOPMOD_OT_crust = _make_global_op(
    "topmod.create_crust",
    "Create Crust",
    "Shell creation (duplicate + inward offset → hollow double wall)",
    create_crust, returns_new=True,
    props={"thickness": FloatProperty(
        name="Thickness", default=0.1, min=-2.0, max=2.0,
        description="Shell thickness (negative = outward)")},
)

TOPMOD_OT_crust_scaling = _make_global_op(
    "topmod.crust_scaling",
    "Crust (Scaling)",
    "Shell via scaling toward centroid (not normal offset)",
    create_crust_with_scaling, returns_new=True,
    props={"scale_factor": FloatProperty(
        name="Scale", default=0.9, min=0.01, max=0.99,
        description="Inner shell scale relative to centroid")},
)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

_classes = [
    # High-level (global)
    TOPMOD_OT_stellate_all,
    TOPMOD_OT_subdivide_all_edges,
    TOPMOD_OT_subdivide_all_faces,
    TOPMOD_OT_triangulate_all,
    # Classic subdivision
    TOPMOD_OT_catmull_clark,
    TOPMOD_OT_dual,
    TOPMOD_OT_doo_sabin,
    TOPMOD_OT_simplest,
    TOPMOD_OT_vertex_cutting,
    TOPMOD_OT_loop,
    TOPMOD_OT_sqrt3,
    # TopMod remeshing (original)
    TOPMOD_OT_honeycomb,
    TOPMOD_OT_star,
    TOPMOD_OT_corner_cutting,
    TOPMOD_OT_loop_style,
    TOPMOD_OT_fractal,
    TOPMOD_OT_pentagonal,
    TOPMOD_OT_pentagonal2,
    TOPMOD_OT_dual1264,
    TOPMOD_OT_root4,
    TOPMOD_OT_checkerboard,
    TOPMOD_OT_ds_bc_new,
    TOPMOD_OT_dome,
    # TopMod remeshing (batch 5)
    TOPMOD_OT_stellate_subdivide,
    TOPMOD_OT_two_stellate,
    TOPMOD_OT_doo_sabin_bc,
    TOPMOD_OT_modified_cc,
    TOPMOD_OT_modified_cc2,
    # Structural
    TOPMOD_OT_crust,
    TOPMOD_OT_crust_scaling,
    # Composite
    TOPMOD_OT_make_wireframe,
]

# ── Local operators (need face/edge/vertex selection) ─────────────────────

# Face ops
TOPMOD_OT_extrude_face = _make_face_op(
    "topmod.extrude_face", "Extrude Face",
    "Extrude selected faces along their normals",
    extrude_face,
    props={"dist": FloatProperty(
        name="Distance", default=0.5, min=-5.0, max=5.0,
        description="Extrusion distance along face normal")},
)

TOPMOD_OT_stellate_face = _make_face_op(
    "topmod.stellate_face", "Stellate Face",
    "Add apex vertex at face centroid, split into triangles",
    stellate,
)

TOPMOD_OT_subdivide_face = _make_face_op(
    "topmod.subdivide_face", "Subdivide Face",
    "Fan-subdivide selected faces from centroid",
    subdivide_face,
)

TOPMOD_OT_triangulate_face = _make_face_op(
    "topmod.triangulate_face", "Triangulate Face",
    "Fan-triangulate selected faces",
    triangulate_face,
)

TOPMOD_OT_double_stellate_face = _make_face_op(
    "topmod.double_stellate_face", "Double Stellate Face",
    "Two-round stellate on selected faces",
    double_stellate_face,
    props={"dist": FloatProperty(
        name="Distance", default=0.0, min=0.0, max=2.0,
        description="Apex displacement along face normal")},
)

TOPMOD_OT_extrude_face_dome = _make_face_op(
    "topmod.extrude_face_dome_local", "Extrude Dome (Face)",
    "Dome-shaped extrusion on selected faces",
    extrude_face_dome,
    props={
        "length": FloatProperty(
            name="Length", default=1.0, min=0.0, max=3.0,
            description="Dome height"),
        "sf": FloatProperty(
            name="Scale", default=1.0, min=0.0, max=3.0,
            description="Dome ring scale"),
    },
)

# Edge ops
TOPMOD_OT_delete_edge = _make_edge_op(
    "topmod.delete_edge", "Delete Edge",
    "Delete selected edges (merge flanking faces)",
    delete_edge,
)

TOPMOD_OT_subdivide_edge = _make_edge_op(
    "topmod.subdivide_edge", "Subdivide Edge",
    "Split selected edges at midpoint",
    subdivide_edge,
)

TOPMOD_OT_collapse_edge = _make_edge_op(
    "topmod.collapse_edge", "Collapse Edge",
    "Merge endpoints of selected edges to midpoint",
    collapse_edge,
)

TOPMOD_OT_trisect_edge = _make_edge_op(
    "topmod.trisect_edge", "Trisect Edge",
    "Split selected edges into 3 equal segments",
    trisect_edge,
)

# Two-face ops
TOPMOD_OT_add_handle = _make_two_face_op(
    "topmod.add_handle", "Add Handle",
    "Connect 2 selected faces with a tunnel (genus +1)",
    add_handle,
)

TOPMOD_OT_punch_hole = _make_two_face_op(
    "topmod.punch_hole", "Punch Hole",
    "Punch a hole between 2 selected faces",
    punch_hole,
)

# Special ops
def _exec_insert_edge(self, context):
    try:
        result = apply_insert_edge(context)
        if result == "select_error":
            self.report({'ERROR'}, "Select exactly 2 vertices")
            return {'CANCELLED'}
        if result is None:
            self.report({'ERROR'}, "Failed — vertices must share a face "
                        "or each have an outgoing half-edge")
            return {'CANCELLED'}
    except Exception as e:
        self.report({'ERROR'}, str(e))
        return {'CANCELLED'}
    return {'FINISHED'}

TOPMOD_OT_insert_edge = _make_special_op(
    "topmod.insert_edge", "Insert Edge",
    "Insert edge between 2 selected vertices (select 2 verts on same face for diagonal, different faces for merge)",
    _exec_insert_edge,
)

def _exec_delete_vertex(self, context):
    try:
        result = apply_delete_vertex(context)
        if result == "select_error":
            self.report({'ERROR'}, "Select exactly 1 isolated vertex")
            return {'CANCELLED'}
        if result is None:
            self.report({'ERROR'}, "Failed — vertex must be isolated (no edges)")
            return {'CANCELLED'}
    except Exception as e:
        self.report({'ERROR'}, str(e))
        return {'CANCELLED'}
    return {'FINISHED'}

TOPMOD_OT_delete_vertex = _make_special_op(
    "topmod.delete_vertex", "Delete Vertex",
    "Delete an isolated vertex (no edges)",
    _exec_delete_vertex,
)

_local_classes = [
    # Face ops
    TOPMOD_OT_extrude_face,
    TOPMOD_OT_stellate_face,
    TOPMOD_OT_subdivide_face,
    TOPMOD_OT_triangulate_face,
    TOPMOD_OT_double_stellate_face,
    TOPMOD_OT_extrude_face_dome,
    # Edge ops
    TOPMOD_OT_delete_edge,
    TOPMOD_OT_subdivide_edge,
    TOPMOD_OT_collapse_edge,
    TOPMOD_OT_trisect_edge,
    # Two-face ops
    TOPMOD_OT_add_handle,
    TOPMOD_OT_punch_hole,
    # Special
    TOPMOD_OT_insert_edge,
    TOPMOD_OT_delete_vertex,
]

_classes = _classes + _local_classes


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
