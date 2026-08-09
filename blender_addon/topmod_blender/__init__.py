"""
TopMod for Blender — DLFL mesh operators as a Blender addon.

Install: Blender → Edit → Preferences → Add-ons → Install from Disk →
select the 'topmod_blender' folder (or its parent zip).

All operators appear in Edit Mode → Mesh → TopMod submenu, and in the
sidebar panel (N-panel → TopMod).
"""

bl_info = {
    "name": "TopMod (DLFL Mesh Operators)",
    "author": "Zengyn42 / GenesisTopmod",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Mesh > TopMod / Sidebar > TopMod",
    "description": "29 topology-preserving 2-manifold mesh operators based "
                   "on Akleman & Chen's DLFL theory",
    "category": "Mesh",
}

import bpy

from . import operators
from . import panels


def register():
    operators.register()
    panels.register()


def unregister():
    panels.unregister()
    operators.unregister()


if __name__ == "__main__":
    register()
