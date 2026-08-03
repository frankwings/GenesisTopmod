"""
Manifold invariant checks — test all four check functions on known good/bad meshes.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.dlfl import DLFLMesh, HalfEdge
from topmod.validate import (
    is_manifold, check_all,
    twin_check, face_loop_check, vertex_fan_check, euler_check,
)
from topmod.primitives import make_cube, make_tetrahedron, make_icosahedron, make_octahedron
from topmod.operators import insert_edge, delete_edge


class TestTwinCheck:
    def test_cube_passes(self):
        ok, errs = twin_check(make_cube())
        assert ok, errs

    def test_tetra_passes(self):
        ok, errs = twin_check(make_tetrahedron())
        assert ok, errs

    def test_icosahedron_passes(self):
        ok, errs = twin_check(make_icosahedron())
        assert ok, errs

    def test_detects_missing_twin(self):
        mesh = DLFLMesh()
        he = mesh._new_halfedge()
        # No twin set → should fail
        ok, errs = twin_check(mesh)
        assert not ok
        assert any("twin" in e.lower() for e in errs)

    def test_detects_asymmetric_twin(self):
        mesh = DLFLMesh()
        he0 = mesh._new_halfedge()
        he1 = mesh._new_halfedge()
        he2 = mesh._new_halfedge()
        # he0.twin = he1 but he1.twin = he2 (not he0)
        he0.twin = he1
        he1.twin = he2
        he2.twin = he0
        e = mesh._new_edge(he0, he1)   # this will set he0↔he1 properly
        # Corrupt he1.twin manually
        he1.twin = he2
        ok, errs = twin_check(mesh)
        assert not ok


class TestFaceLoopCheck:
    def test_cube_passes(self):
        ok, errs = face_loop_check(make_cube())
        assert ok, errs

    def test_octahedron_passes(self):
        ok, errs = face_loop_check(make_octahedron())
        assert ok, errs

    def test_detects_broken_next(self):
        """If a face's next chain leads to wrong face, detect it."""
        mesh = make_cube()
        # Corrupt a face's half-edge face pointer
        he = next(iter(mesh.halfedges.values()))
        original_face = he.face
        # Point it to a different face
        other_face = next(f for f in mesh.faces.values() if f is not original_face)
        he.face = other_face
        ok, errs = face_loop_check(mesh)
        assert not ok

    def test_after_insert_edge_passes(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        ok, errs = face_loop_check(mesh)
        assert ok, errs

    def test_after_delete_edge_passes(self):
        mesh = make_cube()
        e = next(iter(mesh.edges.values()))
        delete_edge(mesh, e)
        ok, errs = face_loop_check(mesh)
        assert ok, errs


class TestVertexFanCheck:
    def test_cube_passes(self):
        ok, errs = vertex_fan_check(make_cube())
        assert ok, errs

    def test_icosahedron_passes(self):
        ok, errs = vertex_fan_check(make_icosahedron())
        assert ok, errs

    def test_tetrahedron_passes(self):
        ok, errs = vertex_fan_check(make_tetrahedron())
        assert ok, errs


class TestEulerCheck:
    def test_cube_chi2(self):
        mesh = make_cube()
        ok, errs = euler_check(mesh)
        assert ok, errs
        assert mesh.euler_characteristic() == 2

    def test_tetrahedron_chi2(self):
        mesh = make_tetrahedron()
        ok, errs = euler_check(mesh)
        assert ok, errs
        assert mesh.euler_characteristic() == 2

    def test_icosahedron_chi2(self):
        mesh = make_icosahedron()
        ok, errs = euler_check(mesh)
        assert ok, errs
        assert mesh.euler_characteristic() == 2

    def test_after_split_chi_preserved(self):
        mesh = make_cube()
        chi0 = mesh.euler_characteristic()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        ok, errs = euler_check(mesh)
        assert ok, errs
        assert mesh.euler_characteristic() == chi0


class TestIsManifold:
    def test_cube_is_manifold(self):
        assert is_manifold(make_cube())

    def test_tetrahedron_is_manifold(self):
        assert is_manifold(make_tetrahedron())

    def test_icosahedron_is_manifold(self):
        assert is_manifold(make_icosahedron())

    def test_octahedron_is_manifold(self):
        assert is_manifold(make_octahedron())

    def test_after_insert_edge_still_manifold(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        assert is_manifold(mesh)

    def test_after_delete_edge_still_manifold(self):
        mesh = make_cube()
        e = next(iter(mesh.edges.values()))
        delete_edge(mesh, e)
        assert is_manifold(mesh)

    def test_after_multiple_operations_manifold(self):
        mesh = make_cube()
        # Insert diagonal in face 1
        faces = list(mesh.faces.values())
        face = faces[0]
        hes = face.halfedges()
        insert_edge(mesh, hes[0], hes[2])
        # Delete another edge
        e = next(iter(mesh.edges.values()))
        delete_edge(mesh, e)
        assert is_manifold(mesh)


class TestCheckAll:
    def test_returns_tuple(self):
        result = check_all(make_cube())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cube_no_errors(self):
        ok, errs = check_all(make_cube())
        assert ok
        assert errs == []

    def test_returns_all_errors(self):
        """A corrupted mesh should return errors from multiple checks."""
        mesh = make_cube()
        he = next(iter(mesh.halfedges.values()))
        he.twin = None   # corrupt twin
        ok, errs = check_all(mesh)
        assert not ok
        assert len(errs) > 0
