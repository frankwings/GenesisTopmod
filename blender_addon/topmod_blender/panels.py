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


class TOPMOD_MT_main(bpy.types.Menu):
    bl_idname = "TOPMOD_MT_main"
    bl_label = "TopMod"

    def draw(self, _context):
        layout = self.layout
        layout.operator("topmod.stellate_all")
        layout.separator()
        layout.menu("TOPMOD_MT_subdivision_classic")
        layout.menu("TOPMOD_MT_remeshing")
        layout.separator()
        layout.operator("topmod.create_crust")


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

        # -- High-level --
        box = layout.box()
        box.label(text="High-Level", icon='MESH_DATA')
        box.operator("topmod.stellate_all", icon='MESH_ICOSPHERE')

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

        # -- Structural --
        box = layout.box()
        box.label(text="Structural", icon='MOD_SOLIDIFY')
        box.operator("topmod.create_crust")


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

_classes = [
    TOPMOD_MT_subdivision_classic,
    TOPMOD_MT_remeshing,
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
