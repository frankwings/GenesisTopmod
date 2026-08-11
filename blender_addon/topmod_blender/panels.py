"""
Blender UI panels and menus for TopMod.

- Mesh menu (Edit Mode → Mesh → TopMod)
- N-panel sidebar (View3D → Sidebar → TopMod)
"""

from __future__ import annotations

import bpy


# ─────────────────────────────────────────────────────────────────────────────
# Mesh menu (Edit Mode → Mesh → TopMod)
# ─────────────────────────────────────────────────────────────────────────────

class TOPMOD_MT_subdivision_classic(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_subdivision_classic"
    bl_label = "Classic Subdivision"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.catmull_clark")
        layout.operator("topmod.dual")
        layout.operator("topmod.doo_sabin")
        layout.operator("topmod.simplest")
        layout.operator("topmod.vertex_cutting")
        layout.operator("topmod.loop_subdivide")
        layout.operator("topmod.sqrt3")


class TOPMOD_MT_remeshing(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_remeshing"
    bl_label = "TopMod Remeshing"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.honeycomb")
        layout.operator("topmod.star")
        layout.operator("topmod.corner_cutting")
        layout.operator("topmod.loop_style")
        layout.operator("topmod.fractal")
        layout.separator()
        layout.operator("topmod.pentagonal")
        layout.operator("topmod.pentagonal2")
        layout.operator("topmod.dual1264")
        layout.operator("topmod.root4")
        layout.separator()
        layout.operator("topmod.checkerboard")
        layout.operator("topmod.ds_bc_new")
        layout.operator("topmod.dome")
        layout.separator()
        layout.operator("topmod.stellate_subdivide")
        layout.operator("topmod.two_stellate")
        layout.operator("topmod.doo_sabin_bc")
        layout.operator("topmod.modified_cc")
        layout.operator("topmod.modified_cc2")


class TOPMOD_MT_structural(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_structural"
    bl_label = "Structural"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.create_crust")
        layout.operator("topmod.crust_scaling")
        layout.separator()
        layout.operator("topmod.make_wireframe")


class TOPMOD_MT_local_face(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_local_face"
    bl_label = "Face Operations (selection)"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.extrude_face")
        layout.operator("topmod.stellate_face")
        layout.operator("topmod.subdivide_face")
        layout.operator("topmod.triangulate_face")
        layout.operator("topmod.double_stellate_face")
        layout.operator("topmod.extrude_face_dome_local")
        layout.separator()
        layout.operator("topmod.add_handle")
        layout.operator("topmod.punch_hole")


class TOPMOD_MT_local_edge(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_local_edge"
    bl_label = "Edge Operations (selection)"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.subdivide_edge")
        layout.operator("topmod.trisect_edge")
        layout.operator("topmod.delete_edge")
        layout.operator("topmod.collapse_edge")
        layout.separator()
        layout.operator("topmod.insert_edge")


class TOPMOD_MT_tools(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_tools"
    bl_label = "Global Tools"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.stellate_all")
        layout.operator("topmod.subdivide_all_edges")
        layout.operator("topmod.subdivide_all_faces")
        layout.operator("topmod.triangulate_all")


class TOPMOD_MT_main(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_main"
    bl_label = "TopMod"

    def draw(self, _context):
        layout = self.layout
        layout.menu("TOPMOD_MT_local_face")
        layout.menu("TOPMOD_MT_local_edge")
        layout.separator()
        layout.menu("TOPMOD_MT_tools")
        layout.separator()
        layout.menu("TOPMOD_MT_subdivision_classic")
        layout.menu("TOPMOD_MT_remeshing")
        layout.separator()
        layout.menu("TOPMOD_MT_structural")


def _mesh_menu_draw(self, _context):
    self.layout.separator()
    self.layout.menu("TOPMOD_MT_main")


# ─────────────────────────────────────────────────────────────────────────────
# N-panel sidebar (View3D → Sidebar → TopMod tab)
# ─────────────────────────────────────────────────────────────────────────────

class TOPMOD_PT_main(bpy.types.Panel):
    bl_idname = "TOPMOD_PT_main"
    bl_label = "TopMod"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "TopMod"
    bl_context = "mesh_edit"

    def draw(self, _context):
        layout = self.layout

        # -- Face operations (selection) --
        box = layout.box()
        box.label(text="Face Ops (selection)", icon='FACESEL')
        col = box.column(align=True)
        col.operator("topmod.extrude_face")
        col.operator("topmod.stellate_face")
        col.operator("topmod.subdivide_face")
        col.operator("topmod.triangulate_face")
        col.operator("topmod.double_stellate_face")
        col.operator("topmod.extrude_face_dome_local")
        col.separator()
        col.operator("topmod.add_handle")
        col.operator("topmod.punch_hole")

        # -- Edge operations (selection) --
        box = layout.box()
        box.label(text="Edge Ops (selection)", icon='EDGESEL')
        col = box.column(align=True)
        col.operator("topmod.subdivide_edge")
        col.operator("topmod.trisect_edge")
        col.operator("topmod.delete_edge")
        col.operator("topmod.collapse_edge")
        col.separator()
        col.operator("topmod.insert_edge")
        col.operator("topmod.delete_vertex")

        # -- Global tools --
        box = layout.box()
        box.label(text="Global Tools", icon='MESH_DATA')
        col = box.column(align=True)
        col.operator("topmod.stellate_all", icon='MESH_ICOSPHERE')
        col.operator("topmod.subdivide_all_edges")
        col.operator("topmod.subdivide_all_faces")
        col.operator("topmod.triangulate_all")

        # -- Classic subdivision --
        box = layout.box()
        box.label(text="Classic Subdivision", icon='MOD_SUBSURF')
        col = box.column(align=True)
        col.operator("topmod.catmull_clark")
        col.operator("topmod.dual")
        col.operator("topmod.doo_sabin")
        col.operator("topmod.simplest")
        col.operator("topmod.vertex_cutting")
        col.operator("topmod.loop_subdivide")
        col.operator("topmod.sqrt3")

        # -- TopMod remeshing --
        box = layout.box()
        box.label(text="TopMod Remeshing", icon='OUTLINER_OB_MESH')
        col = box.column(align=True)
        col.operator("topmod.honeycomb")
        col.operator("topmod.star")
        col.operator("topmod.corner_cutting")
        col.operator("topmod.loop_style")
        col.operator("topmod.fractal")
        col.separator()
        col.operator("topmod.pentagonal")
        col.operator("topmod.pentagonal2")
        col.operator("topmod.dual1264")
        col.operator("topmod.root4")
        col.separator()
        col.operator("topmod.checkerboard")
        col.operator("topmod.ds_bc_new")
        col.operator("topmod.dome")
        col.separator()
        col.operator("topmod.stellate_subdivide")
        col.operator("topmod.two_stellate")
        col.operator("topmod.doo_sabin_bc")
        col.operator("topmod.modified_cc")
        col.operator("topmod.modified_cc2")

        # -- Structural --
        box = layout.box()
        box.label(text="Structural", icon='MOD_SOLIDIFY')
        col = box.column(align=True)
        col.operator("topmod.create_crust")
        col.operator("topmod.crust_scaling")
        col.separator()
        col.operator("topmod.make_wireframe")


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

_classes = [
    TOPMOD_MT_subdivision_classic,
    TOPMOD_MT_remeshing,
    TOPMOD_MT_structural,
    TOPMOD_MT_tools,
    TOPMOD_MT_local_face,
    TOPMOD_MT_local_edge,
    TOPMOD_MT_main,
    TOPMOD_PT_main,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_edit_mesh.append(_mesh_menu_draw)


def unregister():
    bpy.types.VIEW3D_MT_edit_mesh.remove(_mesh_menu_draw)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
