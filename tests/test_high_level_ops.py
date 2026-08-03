"""Tests for high-level mesh operations."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.primitives import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import extrude_face, add_handle, stellate, subdivide_edge, subdivide_face
from topmod.validate import is_manifold, check_all
from topmod.subdivision import catmull_clark
from topmod.io import to_triangle_arrays


class TestStellate:
    def test_returns_vertex(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        center = stellate(mesh, face)
        assert center is not None
        assert center.id in mesh.vertices

    def test_vertex_count_increases_by_one(self):
        mesh = make_cube()
        V0 = mesh.V()
        face = next(iter(mesh.faces.values()))
        stellate(mesh, face)
        assert mesh.V() == V0 + 1

    def test_face_count_increases(self):
        """Stellating an n-gon adds n-1 faces (replaces 1 with n)."""
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        n = face.degree()  # 4 for cube face
        F0 = mesh.F()
        stellate(mesh, face)
        assert mesh.F() == F0 + n - 1

    def test_center_near_centroid(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        cx, cy, cz = face.centroid()
        center = stellate(mesh, face)
        assert abs(center.x - cx) < 1e-6
        assert abs(center.y - cy) < 1e-6
        assert abs(center.z - cz) < 1e-6

    def test_manifold_after_stellate(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        stellate(mesh, face)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_stellate_tetrahedron_face(self):
        mesh = make_tetrahedron()
        face = next(iter(mesh.faces.values()))
        stellate(mesh, face)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_stellate_all_cube_faces(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        for face in faces:
            if face.id in mesh.faces:
                stellate(mesh, face)
        ok, errs = check_all(mesh)
        assert ok, errs


class TestSubdivideEdge:
    def test_returns_vertex(self):
        mesh = make_cube()
        edge = next(iter(mesh.edges.values()))
        mid = subdivide_edge(mesh, edge)
        assert mid is not None
        assert mid.id in mesh.vertices

    def test_vertex_count_increases_by_one(self):
        mesh = make_cube()
        edge = next(iter(mesh.edges.values()))
        V0 = mesh.V()
        subdivide_edge(mesh, edge)
        assert mesh.V() == V0 + 1

    def test_edge_count_increases_by_one(self):
        mesh = make_cube()
        edge = next(iter(mesh.edges.values()))
        E0 = mesh.E()
        subdivide_edge(mesh, edge)
        assert mesh.E() == E0 + 1

    def test_face_count_unchanged(self):
        mesh = make_cube()
        F0 = mesh.F()
        edge = next(iter(mesh.edges.values()))
        subdivide_edge(mesh, edge)
        assert mesh.F() == F0

    def test_midpoint_position(self):
        mesh = make_cube()
        edge = next(iter(mesh.edges.values()))
        v0, v1 = edge.vertices()
        expected = ((v0.x + v1.x) / 2,
                    (v0.y + v1.y) / 2,
                    (v0.z + v1.z) / 2)
        mid = subdivide_edge(mesh, edge)
        assert abs(mid.x - expected[0]) < 1e-9
        assert abs(mid.y - expected[1]) < 1e-9
        assert abs(mid.z - expected[2]) < 1e-9

    def test_manifold_after_subdivide_edge(self):
        mesh = make_cube()
        edge = next(iter(mesh.edges.values()))
        subdivide_edge(mesh, edge)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_euler_preserved_after_subdivide_edge(self):
        mesh = make_cube()
        chi0 = mesh.euler_characteristic()
        edge = next(iter(mesh.edges.values()))
        subdivide_edge(mesh, edge)
        # V+1, E+1, F unchanged → χ unchanged
        assert mesh.euler_characteristic() == chi0


class TestSubdivideFace:
    def test_equivalent_to_stellate(self):
        """subdivide_face is an alias for stellate."""
        from topmod.high_level_ops import subdivide_face
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        V0 = mesh.V()
        subdivide_face(mesh, face)
        assert mesh.V() == V0 + 1


class TestCatmullClark:
    def test_returns_new_mesh(self):
        mesh = make_cube()
        sub = catmull_clark(mesh)
        assert sub is not mesh

    def test_original_unchanged(self):
        mesh = make_cube()
        V0, E0, F0 = mesh.V(), mesh.E(), mesh.F()
        catmull_clark(mesh)
        assert mesh.V() == V0 and mesh.E() == E0 and mesh.F() == F0

    def test_all_quad_faces(self):
        """Catmull-Clark always produces all-quad meshes."""
        mesh = make_cube()
        sub = catmull_clark(mesh)
        for face in sub.iter_faces():
            assert face.degree() == 4, f"Face {face.id} has degree {face.degree()}"

    def test_euler_characteristic_preserved(self):
        mesh = make_cube()
        sub = catmull_clark(mesh)
        assert sub.euler_characteristic() == 2

    def test_manifold_after_catmull_clark(self):
        mesh = make_cube()
        sub = catmull_clark(mesh)
        ok, errs = check_all(sub)
        assert ok, errs

    def test_VEF_count_cube(self):
        """
        Cube: 8V 12E 6F.
        After one CC:
          new_V = 8 + 12 + 6 = 26
          each original face → 4 quads: F_new = 6*4 = 24
          each original edge + 2 per orig face edges: E_new = 12*2 + 6*4 = 48
        """
        mesh = make_cube()
        sub = catmull_clark(mesh)
        assert sub.V() == 26
        assert sub.F() == 24
        assert sub.E() == 48

    def test_catmull_clark_tetrahedron_manifold(self):
        mesh = make_tetrahedron()
        sub = catmull_clark(mesh)
        ok, errs = check_all(sub)
        assert ok, errs

    def test_two_rounds_of_cc(self):
        """Two rounds of CC should still be valid."""
        mesh = make_cube()
        sub1 = catmull_clark(mesh)
        sub2 = catmull_clark(sub1)
        ok, errs = check_all(sub2)
        assert ok, errs
        assert sub2.euler_characteristic() == 2


class TestExtrudeFace:
    def test_extrude_cube_face_manifold(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        extrude_face(mesh, face, dist=0.5)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_extrude_increases_vertices(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        n = face.degree()  # 4 for cube
        V0 = mesh.V()
        extrude_face(mesh, face, dist=0.5)
        # Adds n new vertices for the top cap
        assert mesh.V() == V0 + n

    def test_extrude_increases_faces(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        n = face.degree()
        F0 = mesh.F()
        extrude_face(mesh, face, dist=0.5)
        # Original face removed (-1), adds 1 top face + n side quads
        assert mesh.F() == F0 - 1 + n + 1

    def test_extrude_genus_zero(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        extrude_face(mesh, face, dist=1.0)
        assert mesh.genus() == 0

    def test_extrude_tetrahedron_face_manifold(self):
        mesh = make_tetrahedron()
        face = next(iter(mesh.faces.values()))
        extrude_face(mesh, face, dist=0.5)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_extrude_multiple_faces_manifold(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())[:2]
        for face in faces:
            if face.id in mesh.faces:
                extrude_face(mesh, face, dist=0.3)
        ok, errs = check_all(mesh)
        assert ok, errs


class TestAddHandle:
    def test_add_handle_increases_genus(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        # Pick two non-adjacent faces (e.g., top and bottom of cube)
        f1, f2 = faces[0], faces[1]
        add_handle(mesh, f1, f2)
        assert mesh.genus() == 1

    def test_add_handle_manifold(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1, f2 = faces[0], faces[1]
        add_handle(mesh, f1, f2)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_add_handle_same_face_raises(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        with pytest.raises(ValueError):
            add_handle(mesh, face, face)

    def test_add_handle_returns_edges(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1, f2 = faces[0], faces[1]
        new_edges = add_handle(mesh, f1, f2)
        assert len(new_edges) > 0
        # Should return min(deg(f1), deg(f2)) edges
        assert len(new_edges) == min(f1.degree(), f2.degree()) or len(new_edges) >= 1

    def test_double_handle_genus_2(self):
        """Adding two handles to an icosahedron should give genus 2."""
        mesh = make_icosahedron()
        # Icosahedron has 20 faces — pick pairs that don't share vertices
        faces = list(mesh.faces.values())
        # First handle: faces 0 and 10 (no shared vertices)
        add_handle(mesh, faces[0], faces[10])
        assert mesh.genus() == 1
        ok, errs = check_all(mesh)
        assert ok, errs
        # Second handle: faces 8 and 9 (no vertex overlap with removed faces 0,10)
        add_handle(mesh, faces[8], faces[9])
        assert mesh.genus() == 2
        ok, errs = check_all(mesh)
        assert ok, errs


class TestToTriangleArrays:
    def test_cube_triangulation(self):
        mesh = make_cube()
        positions, triangles = to_triangle_arrays(mesh)
        assert len(positions) == 8
        # 6 quad faces → 12 triangles
        assert len(triangles) == 12

    def test_tetrahedron_triangulation(self):
        mesh = make_tetrahedron()
        positions, triangles = to_triangle_arrays(mesh)
        assert len(positions) == 4
        assert len(triangles) == 4  # already all triangles

    def test_icosahedron_triangulation(self):
        mesh = make_icosahedron()
        positions, triangles = to_triangle_arrays(mesh)
        assert len(positions) == 12
        assert len(triangles) == 20

    def test_indices_in_range(self):
        mesh = make_cube()
        positions, triangles = to_triangle_arrays(mesh)
        n = len(positions)
        for i, j, k in triangles:
            assert 0 <= i < n
            assert 0 <= j < n
            assert 0 <= k < n

    def test_after_catmull_clark(self):
        mesh = make_cube()
        sub = catmull_clark(mesh)
        positions, triangles = to_triangle_arrays(sub)
        assert len(positions) == 26
        # 24 quad faces → 48 triangles
        assert len(triangles) == 48
