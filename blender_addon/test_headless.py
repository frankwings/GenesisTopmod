"""
Headless checks for the addon's corner-picking Insert Edge operator.

Run from the repo root::

    blender -b --factory-startup --python blender_addon/test_headless.py

Exits non-zero if anything fails, so it can be dropped into CI wherever a
Blender binary is available.

Scope
-----
Everything reachable through ``execute()``: registration, the corner ->
half-edge adapter, same-face splits, cross-face merges, the rules that
refuse a pair, and a regression pass over other operators (they all share
the ``Mesh.from_pydata`` write path).  The modal picker itself needs a real
3D Viewport (events, draw handlers, hover) and undo needs a window, so both
are covered by the manual checklist in ``blender_addon/README.md``.

A note on BMesh lifetime: never keep a ``bmesh.from_edit_mesh`` wrapper
past the point where its object is deleted.  Freeing one afterwards is an
access violation that takes Blender down with it, so every helper below
builds its own and drops it before returning.
"""

import os
import sys
import traceback

import bpy
import bmesh

# Unbuffered, so a hard crash still leaves the progress visible.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(label, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" +
          (f"  -- {extra}" if extra else ""))
    if not cond:
        FAILURES.append(label)


def fresh_cube():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    return obj


def _bm(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()
    return bm


def counts(obj):
    bm = _bm(obj)
    result = (len(bm.verts), len(bm.edges), len(bm.faces))
    del bm
    return result


def face_lists(obj):
    bm = _bm(obj)
    result = [[v.index for v in f.verts] for f in bm.faces]
    del bm
    return result


def open_edges(obj):
    bm = _bm(obj)
    result = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    del bm
    return result


def parallel_pairs(obj):
    """Vertex pairs carrying more than one edge."""
    bm = _bm(obj)
    seen, doubled = set(), set()
    for e in bm.edges:
        key = tuple(sorted(v.index for v in e.verts))
        if key in seen:
            doubled.add(key)
        seen.add(key)
    del bm
    return doubled


def degenerate_edges(obj):
    """Edges whose two endpoints are the same vertex."""
    bm = _bm(obj)
    result = [e.index for e in bm.edges if e.verts[0].index == e.verts[1].index]
    del bm
    return result


def selected_edge(obj):
    bm = _bm(obj)
    sel = [e.index for e in bm.edges if e.select]
    del bm
    return sel


def edge_state(obj, vert_a, vert_b):
    """(exists, selected) for the edge between two vertex indices."""
    bm = _bm(obj)
    exists = selected = False
    if 0 <= vert_a < len(bm.verts) and 0 <= vert_b < len(bm.verts):
        edge = bm.edges.get((bm.verts[vert_a], bm.verts[vert_b]))
        exists = edge is not None
        selected = bool(edge and edge.select)
    del bm
    return exists, selected


def free_pair(obj, same_face):
    """
    Two corners the rules allow: distinct vertices with no edge between them.

    ``same_face`` picks both corners off one face (a split) or off two
    different faces (a merge).
    """
    bm = _bm(obj)
    found = (None, None, None, None)
    for fa in bm.faces:
        for ca, va in enumerate(fa.verts):
            joined = {e.other_vert(va).index for e in va.link_edges}
            others = [fa] if same_face else [f for f in bm.faces if f is not fa]
            for fb in others:
                for cb, vb in enumerate(fb.verts):
                    if vb.index != va.index and vb.index not in joined:
                        found = ((fa.index, ca), (fb.index, cb),
                                 va.index, vb.index)
                        break
                if found[0]:
                    break
            if found[0]:
                break
        if found[0]:
            break
    del bm
    return found


def joined_pair(obj):
    """Two corners on different faces whose vertices already share an edge."""
    bm = _bm(obj)
    found = None
    for fa in bm.faces:
        for ca, va in enumerate(fa.verts):
            joined = {e.other_vert(va).index for e in va.link_edges}
            for fb in bm.faces:
                if fb is fa:
                    continue
                for cb, vb in enumerate(fb.verts):
                    if vb.index in joined:
                        found = ((fa.index, ca), (fb.index, cb))
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    del bm
    return found


def shared_vertex_pair(obj):
    """The same vertex reached as a corner of two different faces."""
    bm = _bm(obj)
    found = None
    for ca, va in enumerate(bm.faces[0].verts):
        for face_index in range(1, len(bm.faces)):
            for cb, vb in enumerate(bm.faces[face_index].verts):
                if va.index == vb.index:
                    found = (ca, face_index, cb)
                    break
            if found:
                break
        if found:
            break
    del bm
    return found


def insert(corner_a, corner_b):
    return bpy.ops.topmod.insert_edge(
        face_indices=(corner_a[0], corner_b[0]),
        corners=(corner_a[1], corner_b[1]))


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


section("1. REGISTRATION")
import topmod_blender

topmod_blender.register()
check("addon registers", True)
check("topmod.insert_edge exists", hasattr(bpy.ops.topmod, "insert_edge"))

op_cls = bpy.types.TOPMOD_OT_insert_edge
# Operator properties live in their own srna, not on the class bl_rna.
rna = bpy.ops.topmod.insert_edge.get_rna_type()
props = list(rna.properties.keys())
check("face_indices property registered", "face_indices" in props, str(props))
check("corners property registered", "corners" in props)
check("hover_radius property registered", "hover_radius" in props)
check("face_indices is a 2-vector",
      rna.properties["face_indices"].array_length == 2)
check("has UNDO", 'UNDO' in op_cls.bl_options)
check("is modal (has invoke)", callable(getattr(op_cls, "invoke", None)))
del op_cls, rna


section("2. SAME-FACE PICK: the face splits in two")
obj = fresh_cube()
v0, e0, f0 = counts(obj)
corner_a, corner_b, vert_a, vert_b = free_pair(obj, same_face=True)
print(f"  before: V={v0} E={e0} F={f0}")
print(f"  picking {corner_a} and {corner_b}  (verts {vert_a}, {vert_b})")

res = insert(corner_a, corner_b)
check("operator FINISHED", res == {'FINISHED'}, str(res))

v1, e1, f1 = counts(obj)
print(f"  after:  V={v1} E={e1} F={f1}")
check("V unchanged", v1 == v0)
check("E + 1", e1 == e0 + 1)
check("F + 1 (face split)", f1 == f0 + 1)
faces = face_lists(obj)
check("all faces have >= 3 verts", all(len(f) >= 3 for f in faces))
check("no face repeats a vertex",
      all(len(set(f)) == len(f) for f in faces))
check("mesh still closed", open_edges(obj) == 0, f"{open_edges(obj)} open edges")

exists, selected = edge_state(obj, vert_a, vert_b)
check("the new edge exists", exists)
check("the new edge is selected", selected)
check("exactly one edge is selected", len(selected_edge(obj)) == 1,
      str(selected_edge(obj)))
check("edge select mode is active",
      tuple(bpy.context.tool_settings.mesh_select_mode) == (False, True, False),
      str(tuple(bpy.context.tool_settings.mesh_select_mode)))


section("3. CROSS-FACE PICK: the two faces merge (genus +1)")
obj = fresh_cube()
v0, e0, f0 = counts(obj)
corner_a, corner_b, vert_a, vert_b = free_pair(obj, same_face=False)
print(f"  before: V={v0} E={e0} F={f0}")
print(f"  picking {corner_a} and {corner_b}  (verts {vert_a}, {vert_b})")
check("the two corners really are on different faces", corner_a[0] != corner_b[0])

res = insert(corner_a, corner_b)
check("operator FINISHED", res == {'FINISHED'}, str(res))

v1, e1, f1 = counts(obj)
print(f"  after:  V={v1} E={e1} F={f1}")
faces = face_lists(obj)
for i, f in enumerate(faces):
    mark = "   <-- merged face" if len(set(f)) != len(f) else ""
    print(f"     face {i}: {f}{mark}")

check("V unchanged", v1 == v0, f"V={v1}")
check("E + 1", e1 == e0 + 1, f"E={e1}")
check("F - 1 (the two faces merged)", f1 == f0 - 1, f"F={f1}")
check("Euler characteristic is 0, i.e. genus 1",
      v1 - e1 + f1 == 0, f"chi={v1 - e1 + f1}")

merged = [f for f in faces if len(set(f)) != len(f)]
check("exactly one face repeats a vertex (the merged one)", len(merged) == 1,
      f"{len(merged)} such faces")
if merged:
    # The new edge appears once in each direction. Counting the picked
    # vertices alone would be wrong when the two faces share a vertex, which
    # then shows up a third time.
    loop = merged[0]
    steps = [(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
    check("the merged face walks the new edge once in each direction",
          steps.count((vert_a, vert_b)) == 1 and
          steps.count((vert_b, vert_a)) == 1, str(loop))
    check("both picked vertices appear at least twice",
          loop.count(vert_a) >= 2 and loop.count(vert_b) >= 2, str(loop))
    check("the merged face is the two originals plus the doubled edge",
          len(loop) == 10, f"{len(loop)} corners")
check("the new edge exists", edge_state(obj, vert_a, vert_b)[0])


section("4. THE MERGED FACE SURVIVES EVERY ROUND TRIP")
before = sorted(tuple(f) for f in face_lists(obj))
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data
print(f"  object mode: V={len(me.vertices)} E={len(me.edges)} "
      f"P={len(me.polygons)}")
poly_lists = [[me.loops[i].vertex_index
               for i in range(p.loop_start, p.loop_start + p.loop_total)]
              for p in me.polygons]
check("the merged face survives leaving Edit Mode",
      any(len(set(pl)) != len(pl) for pl in poly_lists), str(poly_lists))
bpy.ops.object.mode_set(mode='EDIT')
check("and survives re-entering Edit Mode",
      sorted(tuple(f) for f in face_lists(obj)) == before)
del me, poly_lists

# The real proof the topology round-tripped: insert a second edge, which
# means reading the merged mesh back through bmesh_to_dlfl first.
corner_a, corner_b, vert_a, vert_b = free_pair(obj, same_face=False)
if corner_a is None:
    check("a second insertion is possible on the merged mesh", False,
          "no legal pair found")
else:
    v0, e0, f0 = counts(obj)
    res = insert(corner_a, corner_b)
    v1, e1, f1 = counts(obj)
    check("a second insertion reads the merged mesh back and applies",
          res == {'FINISHED'}, str(res))
    check("  E + 1 again", e1 == e0 + 1, f"E={e0} -> {e1}")
    check("  genus went up again (chi -2 in two steps)",
          v1 - e1 + f1 == -2, f"chi={v1 - e1 + f1}")

print()
print("  KNOWN LIMITATION (topmod core, not the converter): the subdivision")
print("  operators do not handle a face that visits a vertex twice. Running")
print("  catmull_clark on a merged mesh yields a non-manifold result -- this")
print("  reproduces in pure Python with no Blender involved.")


section("5. NEIGHBOURING CORNERS: a double edge and a 2-gon")
obj = fresh_cube()
v0, e0, f0 = counts(obj)
joined = joined_pair(obj)
print(f"  before: V={v0} E={e0} F={f0}")
print("  picking neighbouring corners (0, 0) and (0, 1)")
res = insert((0, 0), (0, 1))
check("operator FINISHED", res == {'FINISHED'}, str(res))
v1, e1, f1 = counts(obj)
faces = face_lists(obj)
print(f"  after:  V={v1} E={e1} F={f1}")
for i, f in enumerate(faces):
    mark = "   <-- 2-gon" if len(f) == 2 else ""
    print(f"     face {i}: {f}{mark}")
check("V unchanged", v1 == v0, f"V={v1}")
check("E + 1", e1 == e0 + 1, f"E={e1}")
check("F + 1 (the face split)", f1 == f0 + 1, f"F={f1}")
doubled = parallel_pairs(obj)
check("a genuine double edge exists", len(doubled) == 1, str(doubled))
check("one face is a 2-gon", any(len(f) == 2 for f in faces),
      str(sorted(len(f) for f in faces)))
check("Euler characteristic unchanged (still a sphere)",
      v1 - e1 + f1 == 2, f"chi={v1 - e1 + f1}")
check("exactly one edge is selected", len(selected_edge(obj)) == 1,
      str(selected_edge(obj)))

print()
print("  -- and it survives leaving + re-entering Edit Mode --")
before = sorted(tuple(f) for f in face_lists(obj))
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data
print(f"     object mode: V={len(me.vertices)} E={len(me.edges)} "
      f"P={len(me.polygons)}")
bpy.ops.object.mode_set(mode='EDIT')
check("faces unchanged by the round trip",
      sorted(tuple(f) for f in face_lists(obj)) == before)
check("the double edge is still there", len(parallel_pairs(obj)) == 1)
del me

print()
print("  -- a later operator can read it back --")
try:
    res = bpy.ops.topmod.dual()
    ok = res == {'FINISHED'}
except Exception as exc:                            # noqa: BLE001
    res, ok = f"raised {type(exc).__name__}: {exc}", False
check("topmod.dual accepts a mesh with a double edge", ok, str(res))


section("6. TWO CORNERS ON ONE VERTEX IS REFUSED (self-loop hangs Blender)")
obj = fresh_cube()
before_counts = counts(obj)
shared = shared_vertex_pair(obj)
print(f"  shared vertex: face 0 corner {shared[0]} == "
      f"face {shared[1]} corner {shared[2]}")
print("  NOTE: Blender stores a (v,v) edge fine, but bm.normal_update() and")
print("        clearing selection then spin forever, so this is refused.")
res = insert((0, shared[0]), (shared[1], shared[2]))
check("refused: two corners on one vertex", res == {'CANCELLED'}, str(res))
check("  mesh untouched", counts(bpy.context.edit_object) == before_counts)
check("  no degenerate edge was written", degenerate_edges(obj) == [],
      str(degenerate_edges(obj)))


section("6b. THE ONLY REFUSALS")
cases = [
    ("the same corner twice (would hang insert_edge)", ((0, 2), (0, 2))),
]
print("  (neighbouring corners and cross-face picks are legal - see 3 and 5)")
for label, (ca, cb) in cases:
    obj = fresh_cube()
    before_counts = counts(obj)
    try:
        res = insert(ca, cb)
    except Exception as exc:                        # noqa: BLE001
        res = f"raised {type(exc).__name__}: {exc}"
    check(f"refused: {label}", res == {'CANCELLED'}, str(res))
    check(f"  mesh untouched after: {label}",
          counts(bpy.context.edit_object) == before_counts,
          str(counts(bpy.context.edit_object)))

obj = fresh_cube()
before_counts = counts(obj)
try:
    res = bpy.ops.topmod.insert_edge(face_indices=(0, 99), corners=(0, 0))
except Exception as exc:                            # noqa: BLE001
    res = f"raised {type(exc).__name__}: {exc}"
check("refused: a face index that does not exist", res == {'CANCELLED'}, str(res))
check("  mesh untouched", counts(bpy.context.edit_object) == before_counts)


section("7. RUNNING WITH NO RECORDED CORNERS FAILS CLEANLY")
print("  NOTE: background mode has no window, so 'INVOKE_DEFAULT' falls through")
print("        to execute() -- the modal picker itself is on the GUI checklist.")
obj = fresh_cube()
before_counts = counts(obj)
try:
    res = bpy.ops.topmod.insert_edge('INVOKE_DEFAULT')
    check("defaults (-1, -1) are refused, not applied", res == {'CANCELLED'},
          str(res))
except Exception as exc:                            # noqa: BLE001
    check("defaults (-1, -1) are refused, not applied", False,
          f"raised {type(exc).__name__}: {exc}")
check("  mesh untouched", counts(bpy.context.edit_object) == before_counts)


section("8. NON-MANIFOLD INPUT IS REJECTED BY execute()")
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_plane_add()          # open mesh: 4 boundary edges
bpy.ops.object.mode_set(mode='EDIT')
try:
    res = bpy.ops.topmod.insert_edge(face_indices=(0, 0), corners=(0, 2))
except Exception as exc:                            # noqa: BLE001
    res = f"raised {type(exc).__name__}: {exc}"
check("open mesh is refused, not crashed", res == {'CANCELLED'}, str(res))


section("9. THE PICKER'S HOVER GATE (what turns a corner red)")
# execute() is only half the story: the modal picker refuses a click through
# _corner_problem, and that path never goes through execute(). Exercised
# here on a stand-in, since a bpy Operator cannot be instantiated directly
# and the method only reads three plain attributes.
from topmod_blender.corner_pick import TOPMOD_OT_insert_edge, _CornerPick

gate_fn = TOPMOD_OT_insert_edge._corner_problem


class _GateState:
    pass


def gate(stored_face, stored_corner, stored_vert,
         hover_face, hover_face_verts, hover_corner):
    state = _GateState()
    state._picks = [_CornerPick(face_index=stored_face, corner=stored_corner,
                                vert_index=stored_vert,
                                face_coords=[], face_lines=[])]
    state._face_index = hover_face
    state._face_vert_indices = hover_face_verts
    return gate_fn(state, hover_corner)


obj = fresh_cube()
faces = face_lists(obj)
face0, face1 = faces[0], faces[1]
print(f"  face 0 verts: {face0}")
print(f"  face 1 verts: {face1}")
stored_vert = face0[0]

check("the first corner is always allowed",
      gate_fn(type("S", (), {"_picks": []})(), 0) is None)
check("same face, the identical corner: BLOCKED",
      gate(0, 0, stored_vert, 0, face0, 0) is not None)
check("same face, a neighbouring corner (already joined): allowed",
      gate(0, 0, stored_vert, 0, face0, 1) is None,
      str(gate(0, 0, stored_vert, 0, face0, 1)))
check("same face, the far corner: allowed",
      gate(0, 0, stored_vert, 0, face0, 2) is None,
      str(gate(0, 0, stored_vert, 0, face0, 2)))
check("same face, the other neighbour: allowed",
      gate(0, 0, stored_vert, 0, face0, len(face0) - 1) is None)

for corner, vert in enumerate(face1):
    allowed = gate(0, 0, stored_vert, 1, face1, corner) is None
    if vert == stored_vert:
        check(f"cross-face onto the SAME vertex {vert}: BLOCKED", not allowed)
    else:
        check(f"cross-face onto vertex {vert}: allowed", allowed,
              str(gate(0, 0, stored_vert, 1, face1, corner)))

# a face that genuinely shares the stored vertex, for the self-loop rule
shared = shared_vertex_pair(obj)
shared_face_verts = face_lists(obj)[shared[1]]
check("a corner on the stored vertex from another face: BLOCKED",
      gate(0, shared[0], face0[shared[0]],
           shared[1], shared_face_verts, shared[2]) is not None)

# the gate must agree with what execute() then does
print()
print("  -- the gate and execute() agree --")
obj = fresh_cube()
allowed_by_gate = gate(0, 0, face_lists(obj)[0][0], 0, face_lists(obj)[0], 1) is None
res = insert((0, 0), (0, 1))
check("neighbouring corners: gate allows and execute() finishes",
      allowed_by_gate and res == {'FINISHED'},
      f"gate_allowed={allowed_by_gate} execute={res}")


section("10. OTHER OPERATORS STILL WORK (shared write path)")
for op_name in ("catmull_clark", "dual", "doo_sabin", "simplest",
                "stellate_all", "triangulate_all", "subdivide_all_edges",
                "honeycomb", "root4", "create_crust"):
    obj = fresh_cube()
    before_counts = counts(obj)
    try:
        res = getattr(bpy.ops.topmod, op_name)()
    except Exception as exc:                        # noqa: BLE001
        res = f"raised {type(exc).__name__}: {exc}"
    after_counts = counts(bpy.context.edit_object)
    opened = open_edges(bpy.context.edit_object)
    ok = (res == {'FINISHED'} and after_counts[2] > 0 and opened == 0
          and after_counts != before_counts)
    check(f"topmod.{op_name}", ok,
          f"{res} V={after_counts[0]} E={after_counts[1]} "
          f"F={after_counts[2]} open={opened}")


section("11. UNREGISTER IS CLEAN")
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
try:
    topmod_blender.unregister()
    check("addon unregisters", True)
    check("operator gone", not hasattr(bpy.types, "TOPMOD_OT_insert_edge"))
except Exception:                                   # noqa: BLE001
    traceback.print_exc()
    check("addon unregisters", False)


print()
print("=" * 70)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for failure in FAILURES:
        print("   -", failure)
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)

globals().pop("obj", None)

sys.exit(1 if FAILURES else 0)
