"""Tests for primitive generators and OBJ import/export."""

import pytest
import os
import tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.primitives import make_cube, make_tetrahedron, make_icosahedron, make_octahedron
from topmod.validate import is_manifold, check_all
from topmod.io import to_obj, from_obj


class TestCube:
    def test_VEF(self):
        m = make_cube()
        assert m.V() == 8
        assert m.E() == 12
        assert m.F() == 6

    def test_euler(self):
        assert make_cube().euler_characteristic() == 2

    def test_genus_zero(self):
        assert make_cube().genus() == 0

    def test_manifold(self):
        assert is_manifold(make_cube())

    def test_all_quad_faces(self):
        m = make_cube()
        for face in m.iter_faces():
            assert face.degree() == 4

    def test_vertex_positions_bounded(self):
        m = make_cube(size=2.0)
        for v in m.iter_vertices():
            assert abs(v.x) <= 1.0 + 1e-9
            assert abs(v.y) <= 1.0 + 1e-9
            assert abs(v.z) <= 1.0 + 1e-9


class TestTetrahedron:
    def test_VEF(self):
        m = make_tetrahedron()
        assert m.V() == 4
        assert m.E() == 6
        assert m.F() == 4

    def test_euler(self):
        assert make_tetrahedron().euler_characteristic() == 2

    def test_manifold(self):
        assert is_manifold(make_tetrahedron())

    def test_all_triangle_faces(self):
        m = make_tetrahedron()
        for face in m.iter_faces():
            assert face.degree() == 3


class TestIcosahedron:
    def test_VEF(self):
        m = make_icosahedron()
        assert m.V() == 12
        assert m.E() == 30
        assert m.F() == 20

    def test_euler(self):
        assert make_icosahedron().euler_characteristic() == 2

    def test_genus_zero(self):
        assert make_icosahedron().genus() == 0

    def test_manifold(self):
        assert is_manifold(make_icosahedron())

    def test_all_triangle_faces(self):
        m = make_icosahedron()
        for face in m.iter_faces():
            assert face.degree() == 3


class TestOctahedron:
    def test_VEF(self):
        m = make_octahedron()
        assert m.V() == 6
        assert m.E() == 12
        assert m.F() == 8

    def test_euler(self):
        assert make_octahedron().euler_characteristic() == 2

    def test_manifold(self):
        assert is_manifold(make_octahedron())


class TestObjIO:
    def test_roundtrip_cube(self):
        m = make_cube()
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            path = f.name
        try:
            to_obj(m, path)
            m2 = from_obj(path)
            assert m2.V() == m.V()
            assert m2.E() == m.E()
            assert m2.F() == m.F()
            assert m2.euler_characteristic() == m.euler_characteristic()
        finally:
            os.unlink(path)

    def test_roundtrip_tetrahedron(self):
        m = make_tetrahedron()
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            path = f.name
        try:
            to_obj(m, path)
            m2 = from_obj(path)
            assert m2.V() == 4 and m2.F() == 4
            assert is_manifold(m2)
        finally:
            os.unlink(path)

    def test_obj_file_has_vertices(self):
        m = make_cube()
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as f:
            path = f.name
        try:
            to_obj(m, path)
            with open(path) as f:
                content = f.read()
            v_lines = [l for l in content.splitlines() if l.startswith("v ")]
            f_lines = [l for l in content.splitlines() if l.startswith("f ")]
            assert len(v_lines) == 8
            assert len(f_lines) == 6
        finally:
            os.unlink(path)

    def test_from_obj_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            from_obj("/tmp/nonexistent_topmod_test_123456.obj")

    def test_roundtrip_icosahedron_manifold(self):
        m = make_icosahedron()
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            path = f.name
        try:
            to_obj(m, path)
            m2 = from_obj(path)
            assert is_manifold(m2)
        finally:
            os.unlink(path)
