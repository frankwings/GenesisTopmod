"""
Interactive corner picking for ``insert_edge``.

``insert_edge`` needs two *corners* — (face, vertex) pairs, i.e. half-edges.
Naming those by typing indices is hopeless, so this module drives the pick
from the 3D Viewport: for each corner the user clicks a face, then one of
that face's corners, with live GPU overlays showing what is under the
cursor and what has already been recorded.

The operator lives here rather than in ``operators.py`` because it is the
only modal one in the addon; ``operators.py`` imports it and registers it
alongside the rest.

Layering
--------
This file is UI only.  ``execute()`` is a thin adapter that hands the two
recorded corners to ``converter.apply_insert_edge_corners``, which does the
BMesh <-> DLFL round trip and calls the topmod core.

The two corners may sit on the same face, which splits it, or on two
different faces, which merges them. Their vertices may already be joined,
which adds a second edge alongside the first — TopMod allows that and so
does this picker. Only two picks on the same vertex are refused, because the
self-loop edge they make hangs Blender.
"""

# NB: no ``from __future__ import annotations`` here. PEP 563 turns class
# annotations into strings, and Blender registers operator properties by
# reading ``__annotations__`` — the bpy.props values below have to survive
# as real objects. The other modules in this addon build their property
# dicts at runtime, so they are unaffected.

from dataclasses import dataclass

import bpy
import blf
import bmesh
import gpu
from bpy.props import FloatProperty, IntVectorProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

from .converter import apply_insert_edge_corners
from .corner_rules import corner_pair_error


# ─────────────────────────────────────────────────────────────────────────────
# Appearance
# ─────────────────────────────────────────────────────────────────────────────

CORNERS_TO_PICK = 2

COL_HOVER_FACE_FILL     = (1.0, 0.55, 0.10, 0.25)
COL_HOVER_FACE_EDGE     = (1.0, 0.55, 0.10, 1.00)
COL_CURRENT_FACE_FILL   = (0.20, 0.60, 1.00, 0.12)
COL_CURRENT_FACE_EDGE   = (0.20, 0.60, 1.00, 0.90)
COL_STORED_FACE_EDGE    = (0.25, 1.00, 0.45, 0.45)
COL_CORNER_IDLE         = (1.00, 1.00, 1.00, 0.85)
COL_CORNER_HOVER        = (1.00, 0.85, 0.00, 1.00)
COL_CORNER_HOVER_FILL   = (1.00, 0.85, 0.00, 0.35)
COL_CORNER_STORED       = (0.25, 1.00, 0.45, 1.00)
COL_CORNER_STORED_FILL  = (0.25, 1.00, 0.45, 0.35)
COL_CORNER_BLOCKED      = (1.00, 0.30, 0.30, 1.00)
COL_CORNER_BLOCKED_FILL = (1.00, 0.30, 0.30, 0.30)
COL_LINK_LINE           = (1.00, 1.00, 1.00, 0.60)
COL_LINK_INVALID        = (1.00, 0.30, 0.30, 0.80)

# How far along each adjacent edge the corner "L" highlight extends (0..1).
CORNER_ARM_FRACTION = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Recorded data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CornerPick:
    """One stored corner, kept for the whole operation."""

    face_index: int
    corner: int          # position in the face's loop / winding order
    vert_index: int
    face_coords: list    # world-space corner positions of that face
    face_lines: list     # world-space outline segments of that face

    @property
    def co(self):
        return self.face_coords[self.corner]


# ─────────────────────────────────────────────────────────────────────────────
# GPU drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

_shader_cache = {}


def _builtin_shader(*names):
    """
    Return the first builtin shader that exists.

    Blender 4.0 dropped the ``3D_`` prefix and added ``POINT_UNIFORM_COLOR``,
    so the addon's minimum (3.6) needs the older spellings as fallbacks.
    Shaders are created lazily, inside the draw callback, where a GPU
    context is guaranteed.
    """
    shader = _shader_cache.get(names)
    if shader is None:
        for name in names:
            try:
                shader = gpu.shader.from_builtin(name)
                break
            except (ValueError, TypeError, RuntimeError):
                continue
        else:
            raise RuntimeError(f"No builtin shader among {names}")
        _shader_cache[names] = shader
    return shader


def _blf_size(font_id, size):
    """``blf.size`` lost its ``dpi`` argument in Blender 4.0."""
    try:
        blf.size(font_id, size)
    except TypeError:
        blf.size(font_id, size, 72)


def _draw_tris(coords, color):
    if not coords:
        return
    shader = _builtin_shader('UNIFORM_COLOR', '3D_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_lines(coords, color, width):
    if not coords:
        return
    shader = _builtin_shader('POLYLINE_UNIFORM_COLOR', '3D_POLYLINE_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.bind()
    viewport = gpu.state.viewport_get()
    shader.uniform_float("viewportSize", (viewport[2], viewport[3]))
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_points(coords, color, size):
    if not coords:
        return
    shader = _builtin_shader('POINT_UNIFORM_COLOR', 'UNIFORM_COLOR',
                             '3D_UNIFORM_COLOR')
    gpu.state.point_size_set(size)
    batch = batch_for_shader(shader, 'POINTS', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


class _OverlayState:
    """
    Plain-Python container read by the draw callbacks.

    Kept separate from the operator on purpose: Blender frees an operator
    when it finishes, and a draw handler still holding it would raise
    ``ReferenceError`` on every redraw.  The operator writes this object,
    the callbacks only read it.
    """

    def __init__(self, ui_scale):
        self.ui_scale = ui_scale
        self.reset()

    def reset(self):
        self.phase = 'FACE'
        # Face phase
        self.hover_face_tris = []
        self.hover_face_lines = []
        # Corner phase
        self.current_face_tris = []
        self.current_face_lines = []
        self.corner_points = []
        self.hover_tris = []
        self.hover_lines = []
        self.hover_point = []
        self.hover_blocked = False
        self.link_lines = []
        self.link_invalid = False
        # Stored picks (every phase)
        self.stored_face_lines = []
        self.stored_tris = []
        self.stored_lines = []
        self.stored_points = []
        self.labels = []   # (world_co, text, rgba)


def _draw_overlay(state):
    """POST_VIEW: world-space shapes."""
    s = state.ui_scale
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')     # always draw on top of the mesh
    gpu.state.depth_mask_set(False)

    _draw_lines(state.stored_face_lines, COL_STORED_FACE_EDGE, 1.5 * s)

    if state.phase == 'FACE':
        _draw_tris(state.hover_face_tris, COL_HOVER_FACE_FILL)
        _draw_lines(state.hover_face_lines, COL_HOVER_FACE_EDGE, 3.0 * s)
    else:
        _draw_tris(state.current_face_tris, COL_CURRENT_FACE_FILL)
        _draw_lines(state.current_face_lines, COL_CURRENT_FACE_EDGE, 2.0 * s)
        _draw_lines(state.link_lines,
                    COL_LINK_INVALID if state.link_invalid else COL_LINK_LINE,
                    2.5 * s)
        _draw_points(state.corner_points, COL_CORNER_IDLE, 9.0 * s)

    _draw_tris(state.stored_tris, COL_CORNER_STORED_FILL)
    _draw_lines(state.stored_lines, COL_CORNER_STORED, 4.0 * s)
    _draw_points(state.stored_points, COL_CORNER_STORED, 16.0 * s)

    if state.hover_blocked:
        fill, edge = COL_CORNER_BLOCKED_FILL, COL_CORNER_BLOCKED
    else:
        fill, edge = COL_CORNER_HOVER_FILL, COL_CORNER_HOVER
    _draw_tris(state.hover_tris, fill)
    _draw_lines(state.hover_lines, edge, 4.0 * s)
    _draw_points(state.hover_point, edge, 16.0 * s)

    gpu.state.point_size_set(1.0)
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')


def _draw_labels(state):
    """POST_PIXEL: numbered labels next to stored / hovered corners."""
    if not state.labels:
        return
    # The region currently being drawn, so labels project correctly in
    # every 3D Viewport, including quad view.
    region = bpy.context.region
    rv3d = bpy.context.region_data
    if region is None or rv3d is None:
        return

    s = state.ui_scale
    font_id = 0
    _blf_size(font_id, 16 * s)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.9)
    blf.shadow_offset(font_id, 1, -1)
    for co, text, color in state.labels:
        pos = view3d_utils.location_3d_to_region_2d(region, rv3d, co)
        if pos is None:
            continue
        blf.color(font_id, *color)
        blf.position(font_id, pos.x + 12 * s, pos.y + 12 * s, 0)
        blf.draw(font_id, text)
    blf.disable(font_id, blf.SHADOW)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / BMesh helpers
# ─────────────────────────────────────────────────────────────────────────────

def _face_world_geometry(face, matrix_world):
    """World-space corner coords, fill triangles and outline of a face."""
    coords = [matrix_world @ v.co for v in face.verts]
    tris = [coords[i] for tri in tessellate_polygon([coords]) for i in tri]
    lines = []
    for i, co in enumerate(coords):
        lines.extend((co, coords[(i + 1) % len(coords)]))
    return coords, tris, lines


def _corner_wedge(coords, i):
    """
    Triangle + two short edge segments visualising corner *i* of a face.

    The arms run from the corner vertex along its two face edges, which is
    what makes it readable *which face* a corner belongs to when several
    faces share the vertex.
    """
    n = len(coords)
    c = coords[i]
    a = c.lerp(coords[(i - 1) % n], CORNER_ARM_FRACTION)
    b = c.lerp(coords[(i + 1) % n], CORNER_ARM_FRACTION)
    return [c, a, b], [c, a, c, b]


def _deselect_all(bm):
    for seq in (bm.verts, bm.edges, bm.faces):
        for elem in seq:
            elem.select = False
    bm.select_history.clear()


def _elem_kind(elem):
    if isinstance(elem, bmesh.types.BMVert):
        return 'VERT'
    if isinstance(elem, bmesh.types.BMEdge):
        return 'EDGE'
    return 'FACE'


def _snapshot_selection(bm):
    for seq in (bm.verts, bm.edges, bm.faces):
        seq.index_update()
    active = bm.faces.active
    return {
        "verts": [v.index for v in bm.verts if v.select],
        "edges": [e.index for e in bm.edges if e.select],
        "faces": [f.index for f in bm.faces if f.select],
        "history": [(_elem_kind(e), e.index) for e in bm.select_history],
        "active_face": active.index if active is not None else -1,
    }


def _restore_selection(bm, snap):
    seqs = {'VERT': bm.verts, 'EDGE': bm.edges, 'FACE': bm.faces}
    for seq in seqs.values():
        seq.ensure_lookup_table()
    _deselect_all(bm)
    # Bounds-checked: an insertion that failed part way could leave fewer
    # elements than the snapshot recorded, and a stale index would raise
    # in the middle of the restore.
    for kind, indices in (('VERT', snap["verts"]),
                          ('EDGE', snap["edges"]),
                          ('FACE', snap["faces"])):
        seq = seqs[kind]
        for i in indices:
            if 0 <= i < len(seq):
                seq[i].select = True
    for kind, i in snap["history"]:
        seq = seqs[kind]
        if 0 <= i < len(seq):
            bm.select_history.add(seq[i])
    active = snap["active_face"]
    bm.faces.active = bm.faces[active] if 0 <= active < len(bm.faces) else None


def _open_edge_count(bm):
    """Edges not shared by exactly two faces (0 for a closed 2-manifold)."""
    return sum(1 for e in bm.edges if len(e.link_faces) != 2)


# ─────────────────────────────────────────────────────────────────────────────
# Operator
# ─────────────────────────────────────────────────────────────────────────────

_NAV_EVENT_TYPES = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE',
    'WHEELOUTMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
    'MOUSESMARTZOOM',
}


class TOPMOD_OT_insert_edge(bpy.types.Operator):
    """Insert an edge between two corners picked in the viewport"""

    bl_idname = "topmod.insert_edge"
    bl_label = "Insert Edge"
    bl_description = ("Insert an edge between two corners: click a face, then "
                      "one of its corners, twice. Corners on one face split "
                      "it; corners on two faces merge them")
    bl_options = {'REGISTER', 'UNDO'}

    face_indices: IntVectorProperty(
        name="Faces",
        description="Face each corner was picked on, in picking order",
        size=CORNERS_TO_PICK,
        default=(-1,) * CORNERS_TO_PICK,
        min=-1,
    )
    corners: IntVectorProperty(
        name="Corners",
        description="Corner picked on each face (position in that face's "
                    "winding order)",
        size=CORNERS_TO_PICK,
        default=(0,) * CORNERS_TO_PICK,
        min=0,
    )
    hover_radius: FloatProperty(
        name="Hover Radius",
        description="How close the cursor must be to a corner to highlight it",
        default=24.0,
        min=4.0,
        max=200.0,
        subtype='PIXEL',
    )

    @classmethod
    def poll(cls, context):
        obj = context.edit_object
        return (context.mode == 'EDIT_MESH' and
                obj is not None and obj.type == 'MESH')

    # ── entry points ──────────────────────────────────────────────────────

    def invoke(self, context, event):
        self._handles = []
        area = context.area
        if area is None or area.type != 'VIEW_3D':
            self.report({'WARNING'},
                        "Insert Edge must be run from the 3D Viewport")
            return {'CANCELLED'}

        region = context.region
        rv3d = context.region_data
        if region is None or region.type != 'WINDOW':
            # Launched from the sidebar or a menu: fall back to the area's
            # main region.
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            rv3d = area.spaces.active.region_3d
        if region is None or rv3d is None:
            self.report({'WARNING'}, "Could not find a 3D Viewport region")
            return {'CANCELLED'}

        self._area = area
        self._region = region
        self._rv3d = rv3d
        self._obj = context.edit_object
        self._obj_name = self._obj.name
        self._matrix_world = self._obj.matrix_world.copy()
        self._matrix_inv = self._matrix_world.inverted_safe()
        prefs = context.preferences
        self._radius = self.hover_radius * prefs.system.ui_scale
        self._emulate_3_button = prefs.inputs.use_mouse_emulate_3_button
        self._track_mouse(event)

        bm = self._bmesh()

        # Fail before the user picks anything rather than after: the DLFL
        # conversion needs a closed 2-manifold.
        open_edges = _open_edge_count(bm)
        if open_edges:
            self.report({'WARNING'},
                        f"Mesh must be a closed 2-manifold — {open_edges} "
                        "edge(s) are boundary or non-manifold")
            return {'CANCELLED'}

        self._saved_select_mode = tuple(context.tool_settings.mesh_select_mode)
        self._saved_selection = _snapshot_selection(bm)
        self._build_bvh(bm)
        if self._bvh is None:
            self.report({'WARNING'}, "Mesh has no visible faces")
            return {'CANCELLED'}

        # Stored for the whole operation.
        self._picks = []

        # Per-step state.
        self._phase = 'FACE'
        self._hover_face = -1
        self._face_index = -1
        self._face_vert_indices = []
        self._face_coords = []
        self._face_tris = []
        self._face_lines = []
        self._hover_corner = -1
        self._hover_reason = None

        self._state = _OverlayState(prefs.system.ui_scale)
        self._handles = [
            bpy.types.SpaceView3D.draw_handler_add(
                _draw_overlay, (self._state,), 'WINDOW', 'POST_VIEW'),
            bpy.types.SpaceView3D.draw_handler_add(
                _draw_labels, (self._state,), 'WINDOW', 'POST_PIXEL'),
        ]
        context.window_manager.modal_handler_add(self)

        self._enter_face_phase(context, clear_selection=False)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        try:
            return self._handle_event(context, event)
        except Exception:
            # A modal that throws must never leave draw handlers behind.
            self._finish(context, restore=False)
            raise

    def execute(self, context):
        """
        Apply the recorded corners.  Also the Redo-panel path.

        The recorded indices are only meaningful against the mesh *before*
        the insertion — which is what redo re-runs against, because Blender
        undoes the operator before calling ``execute()`` again.
        """
        obj = context.edit_object
        if obj is None or obj.type != 'MESH':
            self.report({'WARNING'}, "No mesh in Edit Mode")
            return {'CANCELLED'}

        corner_a, corner_b = [(int(f), int(c))
                              for f, c in zip(self.face_indices, self.corners)]
        if corner_a[0] < 0 or corner_b[0] < 0:
            self.report({'WARNING'}, "No corners recorded")
            return {'CANCELLED'}

        try:
            result = apply_insert_edge_corners(context, corner_a, corner_b)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if isinstance(result, str):
            self.report({'WARNING'}, result)
            return {'CANCELLED'}

        _dlfl, new_edge_index = result
        self._select_new_edge(context, obj, new_edge_index)
        return {'FINISHED'}

    @staticmethod
    def _select_new_edge(context, obj, edge_index):
        """
        Select the inserted edge so the result is visible.

        Located by index, not by its two vertices: the insertion may have
        added a second edge alongside an existing one, and a vertex pair
        cannot tell those apart.
        """
        context.tool_settings.mesh_select_mode = (False, True, False)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        if not 0 <= edge_index < len(bm.edges):
            return
        _deselect_all(bm)
        edge = bm.edges[edge_index]
        edge.select_set(True)
        bm.select_history.add(edge)
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    # ── event handling ────────────────────────────────────────────────────

    def _handle_event(self, context, event):
        obj = context.edit_object
        if (context.mode != 'EDIT_MESH' or obj is None
                or obj.name != self._obj_name):
            self._finish(context, restore=False)
            return {'CANCELLED'}

        etype, evalue = event.type, event.value
        self._track_mouse(event)

        if etype in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hover()
            return {'RUNNING_MODAL'}

        if self._is_navigation(event):
            return {'PASS_THROUGH'}

        if etype == 'ESC' and evalue == 'PRESS':
            self._finish(context, restore=True)
            return {'CANCELLED'}

        # Let clicks on headers, toolbars and other editors work normally.
        if not self._mouse_in_region():
            return {'PASS_THROUGH'}

        if evalue != 'PRESS':
            return {'RUNNING_MODAL'}

        if etype == 'RIGHTMOUSE':
            self._finish(context, restore=True)
            return {'CANCELLED'}

        if etype == 'BACK_SPACE':
            self._step_back(context)
            return {'RUNNING_MODAL'}

        if etype == 'LEFTMOUSE':
            self._update_hover()
            if self._phase == 'FACE':
                if self._hover_face >= 0:
                    self._enter_corner_phase(context, self._hover_face)
            else:
                h = self._hover_corner
                if h >= 0 and self._corner_problem(h) is None:
                    return self._store_corner(context, h)

        # Swallow everything else so stray hotkeys don't fire mid-pick.
        return {'RUNNING_MODAL'}

    def _is_navigation(self, event):
        etype = event.type
        if etype in _NAV_EVENT_TYPES or etype.startswith('NDOF'):
            return True
        if etype.startswith('NUMPAD') and etype != 'NUMPAD_ENTER':
            return True
        if self._emulate_3_button and etype == 'LEFTMOUSE' and event.alt:
            return True
        return False

    # ── phases ────────────────────────────────────────────────────────────
    #
    # For each corner:  FACE phase (face select mode) → CORNER phase (vertex
    # select mode).  Storing a corner loops back to the FACE phase until
    # CORNERS_TO_PICK corners are stored, then the operator confirms.

    def _enter_face_phase(self, context, clear_selection=True):
        if clear_selection:
            bm = self._bmesh()
            _deselect_all(bm)
            bmesh.update_edit_mesh(self._obj.data,
                                   loop_triangles=False, destructive=False)
        context.tool_settings.mesh_select_mode = (False, False, True)

        self._phase = 'FACE'
        self._face_index = -1
        self._face_vert_indices = []
        self._face_coords = []
        self._face_tris = []
        self._face_lines = []
        self._hover_corner = -1
        self._hover_reason = None
        self._hover_face = -1
        self._update_hover(force=True)

    def _enter_corner_phase(self, context, face_index):
        bm = self._bmesh()
        face = bm.faces[face_index]
        _deselect_all(bm)
        face.select_set(True)
        bm.faces.active = face
        bmesh.update_edit_mesh(self._obj.data,
                               loop_triangles=False, destructive=False)

        context.tool_settings.mesh_select_mode = (True, False, False)

        # The select-mode switch rewrites the edit mesh — re-fetch the
        # wrapper rather than holding one across it.
        bm = self._bmesh()
        bm.verts.index_update()
        face = bm.faces[face_index]
        coords, tris, lines = _face_world_geometry(face, self._matrix_world)

        self._phase = 'CORNER'
        self._face_index = face_index
        self._face_vert_indices = [v.index for v in face.verts]
        self._face_coords = coords
        self._face_tris = tris
        self._face_lines = lines
        self._hover_face = -1
        self._hover_corner = -1
        self._hover_reason = None
        self._update_hover(force=True)

    def _store_corner(self, context, corner):
        self._picks.append(_CornerPick(
            face_index=self._face_index,
            corner=corner,
            vert_index=self._face_vert_indices[corner],
            face_coords=self._face_coords,
            face_lines=self._face_lines,
        ))
        if len(self._picks) == CORNERS_TO_PICK:
            return self._confirm(context)
        self._enter_face_phase(context)
        return {'RUNNING_MODAL'}

    def _step_back(self, context):
        if self._phase == 'CORNER':
            # Re-pick the face for the current corner.
            self._enter_face_phase(context)
        elif self._picks:
            # Un-store the previous corner and return to its corner phase.
            last = self._picks.pop()
            self._enter_corner_phase(context, last.face_index)

    def _confirm(self, context):
        self.face_indices = [p.face_index for p in self._picks]
        self.corners = [p.corner for p in self._picks]
        # Finish first: execute() rebuilds the BMesh from scratch, so no
        # cached index, coordinate or draw handler may still be in play.
        self._finish(context, restore=False)
        result = self.execute(context)
        if 'CANCELLED' in result:
            self._restore_original_selection(context)
        return result

    def _finish(self, context, restore):
        for handle in self._handles:
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        self._handles = []
        self._area.header_text_set(None)
        if restore:
            self._restore_original_selection(context)
        self._area.tag_redraw()

    def _restore_original_selection(self, context):
        context.tool_settings.mesh_select_mode = self._saved_select_mode
        bm = self._bmesh()
        _restore_selection(bm, self._saved_selection)
        bmesh.update_edit_mesh(self._obj.data,
                               loop_triangles=False, destructive=False)

    # ── hover logic ───────────────────────────────────────────────────────

    def _track_mouse(self, event):
        # Window coordinates minus the captured region's origin — not
        # mouse_region_x, which belongs to whichever region happens to be
        # under the cursor.
        self._mouse = Vector((event.mouse_x - self._region.x,
                              event.mouse_y - self._region.y))

    def _mouse_in_region(self):
        mx, my = self._mouse
        return 0 <= mx < self._region.width and 0 <= my < self._region.height

    def _update_hover(self, force=False):
        inside = self._mouse_in_region()
        if self._phase == 'FACE':
            new = self._ray_pick_face() if inside else -1
            changed = new != self._hover_face
            self._hover_face = new
        else:
            new = self._nearest_corner() if inside else -1
            changed = new != self._hover_corner
            self._hover_corner = new
            self._hover_reason = self._corner_problem(new) if new >= 0 else None
        if changed or force:
            self._refresh_overlay()

    def _ray_pick_face(self):
        coord = tuple(self._mouse)
        origin = view3d_utils.region_2d_to_origin_3d(
            self._region, self._rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(
            self._region, self._rv3d, coord)

        # Transform the ray into object space (handles non-uniform scale).
        local_origin = self._matrix_inv @ origin
        local_dir = (self._matrix_inv @ (origin + direction)) - local_origin
        if local_dir.length_squared == 0.0:
            return -1

        _, _, poly_index, _ = self._bvh.ray_cast(local_origin,
                                                 local_dir.normalized())
        if poly_index is None:
            return -1
        return self._poly_to_face[poly_index]

    def _nearest_corner(self):
        best, best_dist = -1, self._radius
        for i, co in enumerate(self._face_coords):
            screen = view3d_utils.location_3d_to_region_2d(
                self._region, self._rv3d, co)
            if screen is None:
                continue
            dist = (screen - self._mouse).length
            if dist <= best_dist:
                best, best_dist = i, dist
        return best

    def _corner_problem(self, corner):
        """
        Why this corner cannot be the next pick, or None.

        Index comparisons only, no side effects, so it is cheap enough to
        run on every hover change.  A corner on a different face is fine
        (the faces merge), and so is one whose vertex is already joined to
        the stored one (a second, parallel edge).
        """
        if not self._picks:
            return None
        prev = self._picks[-1]
        return corner_pair_error(prev.face_index, prev.corner, prev.vert_index,
                                 self._face_index, corner,
                                 self._face_vert_indices[corner])

    # ── overlay ───────────────────────────────────────────────────────────

    def _refresh_overlay(self):
        """The single place that writes the overlay state."""
        st = self._state
        st.reset()
        st.phase = self._phase

        # Stored picks: green face outline + corner wedge + number.
        for number, pick in enumerate(self._picks, start=1):
            st.stored_face_lines.extend(pick.face_lines)
            tris, lines = _corner_wedge(pick.face_coords, pick.corner)
            st.stored_tris.extend(tris)
            st.stored_lines.extend(lines)
            st.stored_points.append(pick.co)
            st.labels.append((pick.co, str(number), COL_CORNER_STORED))

        if self._phase == 'FACE':
            if self._hover_face >= 0:
                face = self._bmesh().faces[self._hover_face]
                _, tris, lines = _face_world_geometry(face, self._matrix_world)
                st.hover_face_tris = tris
                st.hover_face_lines = lines
        else:
            st.current_face_tris = self._face_tris
            st.current_face_lines = self._face_lines
            st.corner_points = self._face_coords

            h = self._hover_corner
            if h >= 0:
                co = self._face_coords[h]
                tris, lines = _corner_wedge(self._face_coords, h)
                st.hover_tris, st.hover_lines, st.hover_point = tris, lines, [co]
                blocked = self._hover_reason is not None
                st.hover_blocked = blocked
                if not blocked:
                    st.labels.append((co, str(len(self._picks) + 1),
                                      COL_CORNER_HOVER))
                if self._picks:
                    st.link_lines = [self._picks[-1].co, co]
                    st.link_invalid = blocked

        self._update_header()
        self._area.tag_redraw()

    def _update_header(self):
        number = len(self._picks) + 1
        prefix = f"Insert Edge - corner {number}/{CORNERS_TO_PICK}"
        cancel = "Esc / RMB: cancel"

        if self._phase == 'FACE':
            step = f"{prefix}: click a face"
            if self._picks:
                step += f" (same face as corner {number - 1} splits it, "
                step += "another face merges the two)"
            parts = [step]
            if self._picks:
                parts.append(f"Backspace: back to corner {number - 1}")
        else:
            step = f"{prefix}: face {self._face_index} recorded, click a corner"
            if self._hover_reason:
                step += f"  (blocked: {self._hover_reason})"
            parts = [step, "Backspace: re-pick face"]
        parts.append(cancel)
        self._area.header_text_set("    |    ".join(parts))

    # ── misc ──────────────────────────────────────────────────────────────

    def _bmesh(self):
        bm = bmesh.from_edit_mesh(self._obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        return bm

    def _build_bvh(self, bm):
        """BVH of visible faces only, with a map back to BMesh face indices."""
        bm.verts.index_update()
        bm.faces.index_update()
        coords = [v.co.copy() for v in bm.verts]
        polys = []
        self._poly_to_face = []
        for f in bm.faces:
            if not f.hide:
                polys.append([v.index for v in f.verts])
                self._poly_to_face.append(f.index)
        self._bvh = BVHTree.FromPolygons(coords, polys) if polys else None
