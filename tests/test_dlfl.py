"""Tests for the DLFL data structure."""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.dlfl import DLFLMesh, Vertex, HalfEdge, Face, Edge


class TestVertexBasics:
    def test_vertex_creation(self):
        v = Vertex(1.0, 2.0, 3.0)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_vertex_position_property(self):
        v = Vertex()
        v.position = (4.0, 5.0, 6.0)
        assert v.position == (4.0, 5.0, 6.0)

    def test_vertex_unique_ids(self):
        v1 = Vertex()
        v2 = Vertex()
        assert v1.id != v2.id

    def test_vertex_repr(self):
        v = Vertex(1.0, 2.0, 3.0)
        assert "Vertex" in repr(v)


class TestHalfEdgeBasics:
    def test_halfedge_creation(self):
        he = HalfEdge()
        assert he.id is not None
        assert he.origin is None
        assert he.twin is None

    def test_face_loop_single(self):
        """face_loop on an isolated he (loops to itself) returns [self]."""
        he = HalfEdge()
        he.next = he
        assert he.face_loop() == [he]

    def test_unique_ids(self):
        h1, h2 = HalfEdge(), HalfEdge()
        assert h1.id != h2.id


class TestDLFLMeshStructure:
    def test_empty_mesh(self):
        mesh = DLFLMesh()
        assert mesh.V() == 0
        assert mesh.E() == 0
        assert mesh.F() == 0
        assert mesh.euler_characteristic() == 0

    def test_new_vertex_registered(self):
        mesh = DLFLMesh()
        v = mesh._new_vertex(1, 2, 3)
        assert v.id in mesh.vertices
        assert mesh.V() == 1

    def test_new_halfedge_registered(self):
        mesh = DLFLMesh()
        he = mesh._new_halfedge()
        assert he.id in mesh.halfedges

    def test_new_face_registered(self):
        mesh = DLFLMesh()
        f = mesh._new_face()
        assert f.id in mesh.faces

    def test_new_edge_sets_twins(self):
        mesh = DLFLMesh()
        he0 = mesh._new_halfedge()
        he1 = mesh._new_halfedge()
        e = mesh._new_edge(he0, he1)
        assert he0.twin is he1
        assert he1.twin is he0
        assert e.id in mesh.edges

    def test_euler_characteristic_sphere(self):
        """A cube has χ=2 (sphere topology)."""
        from topmod.primitives import make_cube
        mesh = make_cube()
        assert mesh.euler_characteristic() == 2

    def test_genus_zero(self):
        from topmod.primitives import make_cube
        mesh = make_cube()
        assert mesh.genus() == 0

    def test_component_count_single(self):
        from topmod.primitives import make_cube
        mesh = make_cube()
        assert mesh.component_count() == 1

    def test_find_edge(self):
        from topmod.primitives import make_tetrahedron
        mesh = make_tetrahedron()
        verts = list(mesh.vertices.values())
        # find_edge between adjacent vertices
        v0, v1 = verts[0], verts[1]
        e = mesh.find_edge(v0, v1)
        if e is None:
            e = mesh.find_edge(v1, v0)
        assert e is not None

    def test_find_halfedge(self):
        from topmod.primitives import make_tetrahedron
        mesh = make_tetrahedron()
        verts = list(mesh.vertices.values())
        v0, v1 = verts[0], verts[1]
        he = mesh.find_halfedge(v0, v1)
        if he is None:
            he = mesh.find_halfedge(v1, v0)
        assert he is not None
