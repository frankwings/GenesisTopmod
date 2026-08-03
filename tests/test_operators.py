"""Tests for the 4 fundamental TopMod operators."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.dlfl import DLFLMesh
from topmod.operators import create_vertex, delete_vertex, insert_edge, delete_edge
from topmod.validate import is_manifold, check_all, twin_check, face_loop_check
from topmod.primitives import make_cube, make_tetrahedron


# ── create_vertex ──────────────────────────────────────────────────────────────

class TestCreateVertex:
    def test_returns_vertex(self):
        mesh = DLFLMesh()
        v = create_vertex(mesh, 1, 2, 3)
        assert v is not None
        assert v.id in mesh.vertices

    def test_position_correct(self):
        mesh = DLFLMesh()
        v = create_vertex(mesh, 1.5, -2.0, 3.7)
        assert abs(v.x - 1.5) < 1e-9
        assert abs(v.y + 2.0) < 1e-9
        assert abs(v.z - 3.7) < 1e-9

    def test_increments_VEF(self):
        mesh = DLFLMesh()
        v1 = create_vertex(mesh)
        assert mesh.V() == 1
        assert mesh.E() == 1   # degenerate loop edge
        assert mesh.F() == 1   # degenerate face

    def test_two_vertices_two_components(self):
        mesh = DLFLMesh()
        create_vertex(mesh, 0, 0, 0)
        create_vertex(mesh, 1, 0, 0)
        assert mesh.V() == 2
        assert mesh.component_count() == 2

    def test_degenerate_has_twin(self):
        """The degenerate loop edge created by create_vertex has valid twins."""
        mesh = DLFLMesh()
        v = create_vertex(mesh)
        assert v.he is not None
        assert v.he.twin is not None
        assert v.he.twin.twin is v.he

    def test_twin_check_passes(self):
        mesh = DLFLMesh()
        create_vertex(mesh)
        ok, errs = twin_check(mesh)
        assert ok, errs

    def test_face_loop_check_passes(self):
        mesh = DLFLMesh()
        create_vertex(mesh)
        ok, errs = face_loop_check(mesh)
        assert ok, errs


# ── delete_vertex ──────────────────────────────────────────────────────────────

class TestDeleteVertex:
    def test_delete_isolated_vertex(self):
        mesh = DLFLMesh()
        v = create_vertex(mesh)
        V0, E0, F0 = mesh.V(), mesh.E(), mesh.F()
        delete_vertex(mesh, v)
        assert mesh.V() == V0 - 1
        assert mesh.E() == E0 - 1
        assert mesh.F() == F0 - 1

    def test_vertex_removed_from_dict(self):
        mesh = DLFLMesh()
        v = create_vertex(mesh)
        vid = v.id
        delete_vertex(mesh, v)
        assert vid not in mesh.vertices

    def test_delete_non_isolated_raises(self):
        mesh = make_cube()
        v = next(iter(mesh.vertices.values()))
        with pytest.raises(ValueError):
            delete_vertex(mesh, v)

    def test_create_then_delete_empty(self):
        mesh = DLFLMesh()
        v = create_vertex(mesh)
        delete_vertex(mesh, v)
        assert mesh.V() == 0
        assert mesh.E() == 0
        assert mesh.F() == 0


# ── insert_edge (case A: same face) ───────────────────────────────────────────

class TestInsertEdgeSameFace:
    def _make_quad_mesh(self):
        """Single quad face with 4 vertices."""
        from topmod.primitives import _build_mesh
        mesh = _build_mesh(
            [(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
            [[0,1,2,3]]
        )
        # _build_mesh requires closed mesh; this is a single face which
        # needs an outer face.  Use the outer half-edges trick by building
        # a two-face mesh (top/bottom) instead.
        return None

    def test_split_square_into_two_triangles(self):
        """Insert a diagonal into a quad face → two triangles."""
        from topmod.primitives import _build_mesh
        # Build a cube face: take the bottom face of a cube and find a face
        mesh = make_cube()
        # Pick any face
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        assert len(hes) == 4

        V0, E0, F0 = mesh.V(), mesh.E(), mesh.F()
        chi0 = mesh.euler_characteristic()

        # Insert diagonal across the face (skip adjacent vertices)
        edge = insert_edge(mesh, hes[0], hes[2])
        assert edge is not None

        # V unchanged, E+1, F+1  → χ unchanged
        assert mesh.V() == V0
        assert mesh.E() == E0 + 1
        assert mesh.F() == F0 + 1
        assert mesh.euler_characteristic() == chi0

    def test_manifold_after_same_face_insert(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_new_edge_has_twin(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        edge = insert_edge(mesh, hes[0], hes[2])
        assert edge.he0.twin is edge.he1
        assert edge.he1.twin is edge.he0

    def test_face_count_increases(self):
        mesh = make_cube()
        F0 = mesh.F()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        assert mesh.F() == F0 + 1


# ── insert_edge (case B/C: different faces) ────────────────────────────────────

class TestInsertEdgeDifferentFaces:
    def test_merge_two_faces_on_cube(self):
        """Two adjacent faces on a cube share an edge; deleting that edge
        and re-inserting across the merged face tests cross-face insert."""
        mesh = make_cube()
        # Delete an edge to merge two faces
        e = next(iter(mesh.edges.values()))
        f_merged = delete_edge(mesh, e)
        # Now f_merged is a 6-gon or similar merged face.
        # Find another face (not merged)
        other_face = None
        for f in mesh.faces.values():
            if f is not f_merged:
                other_face = f
                break
        assert other_face is not None

        he1 = f_merged.halfedges()[0]
        he2 = other_face.halfedges()[0]

        # These are on different faces → cross-face insert
        F0 = mesh.F()
        E0 = mesh.E()
        edge = insert_edge(mesh, he1, he2)
        # Face count decreases by 1 (two faces merge)
        assert mesh.F() == F0 - 1
        assert mesh.E() == E0 + 1

    def test_manifold_after_cross_face_insert(self):
        mesh = make_cube()
        e = next(iter(mesh.edges.values()))
        f_merged = delete_edge(mesh, e)
        other_face = next(f for f in mesh.faces.values() if f is not f_merged)
        he1 = f_merged.halfedges()[0]
        he2 = other_face.halfedges()[0]
        insert_edge(mesh, he1, he2)
        ok, errs = check_all(mesh)
        assert ok, errs


# ── delete_edge ────────────────────────────────────────────────────────────────

class TestDeleteEdge:
    def test_merge_two_faces(self):
        mesh = make_cube()
        F0 = mesh.F()
        E0 = mesh.E()
        e = next(iter(mesh.edges.values()))
        merged_face = delete_edge(mesh, e)
        assert mesh.F() == F0 - 1
        assert mesh.E() == E0 - 1
        assert merged_face is not None

    def test_euler_unchanged_after_delete_edge(self):
        mesh = make_cube()
        chi0 = mesh.euler_characteristic()
        e = next(iter(mesh.edges.values()))
        delete_edge(mesh, e)
        assert mesh.euler_characteristic() == chi0

    def test_manifold_after_delete_edge(self):
        mesh = make_cube()
        e = next(iter(mesh.edges.values()))
        delete_edge(mesh, e)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_edge_removed_from_dict(self):
        mesh = make_cube()
        e = next(iter(mesh.edges.values()))
        eid = e.id
        delete_edge(mesh, e)
        assert eid not in mesh.edges

    def test_delete_multiple_edges_valid(self):
        """Delete two edges from a tetrahedron; mesh remains valid each time."""
        mesh = make_tetrahedron()
        edges = list(mesh.edges.values())
        # Delete first two edges (they may or may not be adjacent; both valid)
        for e in edges[:2]:
            if e.id in mesh.edges:
                delete_edge(mesh, e)
                ok, errs = check_all(mesh)
                assert ok, errs

    def test_insert_then_delete_roundtrip(self):
        """Insert an edge into a face, then delete it → same topology."""
        mesh = make_cube()
        V0, E0, F0 = mesh.V(), mesh.E(), mesh.F()
        chi0 = mesh.euler_characteristic()

        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        new_edge = insert_edge(mesh, hes[0], hes[2])

        delete_edge(mesh, new_edge)
        assert mesh.V() == V0
        assert mesh.E() == E0
        assert mesh.F() == F0
        assert mesh.euler_characteristic() == chi0
        ok, errs = check_all(mesh)
        assert ok, errs
