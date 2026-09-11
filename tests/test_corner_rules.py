"""
Tests for the addon's corner topology (blender_addon/topmod_blender/corner_rules.py).

A *corner* is a (face, vertex) pair, i.e. one half-edge.  The interactive
picker in ``corner_pick.py`` records corners as (face_index, corner_position)
and gates the second pick with ``corner_pair_error``; the converter flattens
a DLFL mesh with ``dlfl_corner_arrays`` and rebuilds it with
``build_dlfl_from_corners``.

The point of describing a face as ``(vertex index, edge index)`` corners
rather than a vertex list is that ``insert_edge`` can produce topology a
vertex list cannot describe: two parallel edges between one pair of
vertices, and faces that visit a vertex more than once.  Blender stores
both (verified on 4.4 and 5.3), so the round trip has to preserve them.

A self-loop edge is the exception: DLFL and Blender's mesh arrays both hold
one, but it hangs ``bm.normal_update()`` and selection clearing, so the
rules refuse the pick that would create it.

``corner_rules`` imports no ``bpy``, so it can be loaded straight from the
addon tree and tested headlessly — but the addon *package* does import bpy,
hence the load-by-path below rather than a normal import.
"""

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.operators import insert_edge
from topmod.primitives import make_cube, make_icosahedron, make_tetrahedron
from topmod.validate import is_manifold


def _load_corner_rules():
    path = os.path.join(os.path.dirname(__file__), "..", "blender_addon",
                        "topmod_blender", "corner_rules.py")
    spec = importlib.util.spec_from_file_location("corner_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corner_rules = _load_corner_rules()
corner_pair_error = corner_rules.corner_pair_error
resolve_corner_halfedge = corner_rules.resolve_corner_halfedge
degenerate_edges = corner_rules.degenerate_edges
dlfl_corner_arrays = corner_rules.dlfl_corner_arrays
build_dlfl_from_corners = corner_rules.build_dlfl_from_corners
unwritable_faces = corner_rules.unwritable_faces


# ── helpers ───────────────────────────────────────────────────────────────────

def _corner(mesh, face_index, corner):
    """(face, half-edge) for a corner position, plus its vertex."""
    face = list(mesh.faces.values())[face_index]
    he = resolve_corner_halfedge(face, corner)
    return face, he, he.origin


def _round_trip(mesh):
    """
    Flatten a DLFL mesh and rebuild it, exactly as the converter does.

    This is the whole contract with Blender: the converter writes these
    arrays into a Mesh and reads the same corners back off the BMesh, and
    Blender was verified to preserve them unchanged.
    """
    coords, _edge_pairs, faces_corners = dlfl_corner_arrays(mesh)
    rebuilt, _edges_by_index = build_dlfl_from_corners(coords, faces_corners)
    return rebuilt


def _signature(mesh):
    coords, edge_pairs, faces_corners = dlfl_corner_arrays(mesh)
    return (mesh.V(), mesh.E(), mesh.F(), mesh.genus(),
            sorted(tuple(p) if p[0] < p[1] else (p[1], p[0])
                   for p in edge_pairs),
            sorted(tuple(v for v, _e in face) for face in faces_corners))


def _parallel_pairs(mesh):
    """Vertex pairs joined by more than one edge."""
    _coords, edge_pairs, _faces = dlfl_corner_arrays(mesh)
    seen, doubled = set(), set()
    for a, b in edge_pairs:
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            doubled.add(key)
        seen.add(key)
    return doubled


# ── corner_pair_error ─────────────────────────────────────────────────────────

class TestCornerPairError:
    def test_two_corners_of_one_face_are_allowed(self):
        assert corner_pair_error(0, 0, 10, 0, 2, 12) is None

    def test_neighbouring_corners_are_allowed(self):
        # A double edge and a 2-gon. TopMod permits it; so does Blender.
        assert corner_pair_error(0, 0, 10, 0, 1, 11) is None

    def test_corners_on_different_faces_are_allowed(self):
        assert corner_pair_error(0, 0, 10, 3, 2, 22) is None

    def test_the_same_corner_twice_is_blocked(self):
        # insert_edge leaves a face loop that never closes and then hangs.
        assert corner_pair_error(0, 2, 12, 0, 2, 12) is not None

    def test_two_corners_on_one_vertex_are_blocked(self):
        # A DLFL self-loop. Blender stores it, then hangs on it.
        assert corner_pair_error(0, 0, 10, 3, 2, 10) is not None

    def test_same_position_on_another_face_is_fine(self):
        assert corner_pair_error(0, 2, 12, 1, 2, 22) is None

    def test_message_is_a_lowercase_fragment(self):
        # Callers embed it: "Cannot insert edge: ..." / header "(blocked: ...)".
        for reason in (corner_pair_error(0, 0, 10, 0, 0, 10),
                       corner_pair_error(0, 0, 10, 3, 2, 10)):
            assert reason and reason[0].islower() and not reason.endswith(".")


# ── resolve_corner_halfedge ───────────────────────────────────────────────────

class TestResolveCornerHalfedge:
    def test_returns_the_halfedge_at_that_position(self):
        mesh = make_cube()
        face = list(mesh.faces.values())[0]
        for position, he in enumerate(face.halfedges()):
            assert resolve_corner_halfedge(face, position) is he

    def test_positions_wrap(self):
        mesh = make_cube()
        face = list(mesh.faces.values())[0]
        size = len(face.halfedges())
        assert (resolve_corner_halfedge(face, size + 1)
                is resolve_corner_halfedge(face, 1))

    def test_resolves_by_position_not_by_vertex(self):
        """
        A merged face visits some vertices twice, so only the position names
        a corner unambiguously. Both corners on the repeated vertex must be
        reachable.
        """
        mesh = make_cube()
        _fa, ha, va = _corner(mesh, 0, 0)
        _fb, hb, _vb = _corner(mesh, 1, 1)
        insert_edge(mesh, ha, hb)

        merged = max(mesh.faces.values(), key=lambda f: len(f.halfedges()))
        positions = [i for i, he in enumerate(merged.halfedges())
                     if he.origin is va]
        assert len(positions) >= 2
        found = {id(resolve_corner_halfedge(merged, p)) for p in positions}
        assert len(found) == len(positions)      # distinct half-edges


# ── the corner round trip ─────────────────────────────────────────────────────

class TestRoundTrip:
    def test_a_plain_cube(self):
        mesh = make_cube()
        assert _signature(_round_trip(mesh)) == _signature(mesh)
        assert is_manifold(_round_trip(mesh))

    def test_same_face_split(self):
        mesh = make_cube()
        insert_edge(mesh, _corner(mesh, 0, 0)[1], _corner(mesh, 0, 2)[1])
        assert _signature(_round_trip(mesh)) == _signature(mesh)

    def test_cross_face_merge(self):
        mesh = make_cube()
        insert_edge(mesh, _corner(mesh, 0, 0)[1], _corner(mesh, 1, 1)[1])
        assert mesh.genus() == 1
        assert _signature(_round_trip(mesh)) == _signature(mesh)

    def test_double_edge_between_already_joined_corners(self):
        """
        The case TopMod allows and a vertex list cannot express: the two
        corners are neighbours, so they already share an edge and the split
        needs a second one.
        """
        mesh = make_cube()
        _fa, ha, va = _corner(mesh, 0, 0)
        _fb, hb, vb = _corner(mesh, 0, 1)
        assert mesh.find_edge(va, vb) is not None     # already joined
        before_e = mesh.E()
        insert_edge(mesh, ha, hb)

        assert mesh.E() == before_e + 1
        assert _parallel_pairs(mesh)                  # a genuine double edge
        assert is_manifold(mesh)
        assert _signature(_round_trip(mesh)) == _signature(mesh)

    def test_the_double_edge_bounds_a_2gon(self):
        mesh = make_cube()
        insert_edge(mesh, _corner(mesh, 0, 0)[1], _corner(mesh, 0, 1)[1])
        sizes = sorted(len(f.halfedges()) for f in mesh.faces.values())
        assert sizes[0] == 2                          # the bigon
        assert unwritable_faces(mesh) == []           # 2 corners is storable

    def test_a_self_loop_is_refused_before_it_is_built(self):
        """
        Two corners of one vertex make a DLFL self-loop. It round-trips on
        paper, but the degenerate (v, v) edge hangs Blender's normal update
        and selection clearing, so the rules stop it at the pick.
        """
        mesh = make_cube()
        faces = list(mesh.faces.values())
        vertex = faces[0].vertices()[0]
        other_index, other = next((i, f) for i, f in enumerate(faces)
                                  if i and vertex in f.vertices())
        corner_a = faces[0].vertices().index(vertex)
        corner_b = other.vertices().index(vertex)
        assert corner_pair_error(0, corner_a, vertex.id,
                                 other_index, corner_b, vertex.id) is not None

        # and the guard behind it recognises the shape, had it been built
        ha = next(h for h in faces[0].halfedges() if h.origin is vertex)
        hb = next(h for h in other.halfedges() if h.origin is vertex)
        insert_edge(mesh, ha, hb)
        assert len(degenerate_edges(mesh)) == 1

    def test_two_insertions_chain(self):
        mesh = make_cube()
        insert_edge(mesh, _corner(mesh, 0, 0)[1], _corner(mesh, 1, 1)[1])
        rebuilt = _round_trip(mesh)
        # a second insertion, applied to the rebuilt mesh
        faces = list(rebuilt.faces.values())
        insert_edge(rebuilt, faces[1].halfedges()[0], faces[2].halfedges()[1])
        assert rebuilt.genus() == 2
        assert _signature(_round_trip(rebuilt)) == _signature(rebuilt)

    def test_icosahedron_cross_face(self):
        mesh = make_icosahedron()
        faces = list(mesh.faces.values())
        insert_edge(mesh, faces[0].halfedges()[0], faces[7].halfedges()[1])
        assert _signature(_round_trip(mesh)) == _signature(mesh)

    def test_tetrahedron_now_has_legal_insertions(self):
        """
        Every pair of tetrahedron vertices is already joined, so this used to
        have no legal pick at all. With double edges allowed it does.
        """
        mesh = make_tetrahedron()
        face = list(mesh.faces.values())[0]
        verts = face.vertices()
        assert corner_pair_error(0, 0, verts[0].id, 0, 1, verts[1].id) is None
        insert_edge(mesh, _corner(mesh, 0, 0)[1], _corner(mesh, 0, 1)[1])
        assert is_manifold(mesh)
        assert unwritable_faces(mesh) == []
        assert _signature(_round_trip(mesh)) == _signature(mesh)


# ── the reader rejects what it should ─────────────────────────────────────────

class TestBuilderValidation:
    def test_an_edge_used_once_is_rejected(self):
        """An open mesh: the converter must raise, not build a broken DLFL."""
        with pytest.raises(ValueError, match="not closed"):
            build_dlfl_from_corners(
                [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                [[(0, 0), (1, 1), (2, 2)]])          # one triangle, 3 edges x1

    def test_parallel_edges_do_not_collide(self):
        """
        The reason for edge indices: keyed by vertex pair, these two edges
        would overwrite each other and one would lose its twin.
        """
        mesh, edges_by_index = build_dlfl_from_corners(
            [(0, 0, 0), (1, 0, 0)],
            [[(0, 0), (1, 1)],                       # bigon, edges 0 and 1
             [(1, 0), (0, 1)]])                      # its other side
        assert mesh.E() == 2
        assert len(edges_by_index) == 2
        for edge in mesh.edges.values():
            assert edge.he0.twin is edge.he1
            assert edge.he1.twin is edge.he0


class TestUnwritableFaces:
    def test_a_cube_is_writable(self):
        assert unwritable_faces(make_cube()) == []

    def test_a_one_corner_face_is_reported(self):
        """
        A single-corner polygon crashes Blender's Mesh->BMesh conversion, so
        the writer has to drop it. create_vertex makes exactly one.
        """
        from topmod.operators import create_vertex
        from topmod.dlfl import DLFLMesh

        mesh = DLFLMesh()
        create_vertex(mesh, 0, 0, 0)
        reported = unwritable_faces(mesh)
        assert len(reported) == 1
        assert reported[0][1] < 2
