"""
Semantic cross-validation against the TopMod reference (davyrisso/topmod3d).

We cannot link the GPL C++ code, so the oracle is the *documented semantics*
of each operator (Akleman & Chen 2003; DLFLExtrude.hh / DLFLConnect.hh /
DLFLSubdiv.hh): every TopMod operator has a closed-form effect on the mesh
element counts.  For each of our operators, applied to each primitive, we
assert the exact predicted ΔV / ΔE / ΔF / Δχ / Δgenus — same mesh, same
operation, compared element-by-element against the reference semantics.

Oracle formulas (n = degree of the operated face):

| Operation                | ΔV      | ΔE  | ΔF    | Δχ | Δgenus |
|--------------------------|---------|-----|-------|----|--------|
| extrude_face (n-gon)     | +n      | +2n | +n    | 0  | 0      |
| add_handle (two n-gons)  | 0       | +n  | +n−2  | −2 | +1     |
| stellate (n-gon)         | +1      | +n  | +n−1  | 0  | 0      |
| subdivide_edge           | +1      | +1  | 0     | 0  | 0      |
| subdivide_face (n-gon)   | +1      | +n  | +n−1  | 0  | 0      |
| catmull_clark            | V'=V+E+F, E'=4E, F'=2E (all quads) | 0 | 0 |
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.primitives import (
    make_cube, make_tetrahedron, make_octahedron, make_icosahedron,
)
from topmod.high_level_ops import (
    extrude_face, add_handle, stellate, subdivide_edge, subdivide_face,
)
from topmod.subdivision import catmull_clark
from topmod.validate import check_all


PRIMITIVES = {
    "cube": make_cube,
    "tetrahedron": make_tetrahedron,
    "octahedron": make_octahedron,
    "icosahedron": make_icosahedron,
}


def counts(mesh):
    return (len(mesh.vertices), len(mesh.edges), len(mesh.faces))


def assert_valid(mesh, ctx):
    ok, errs = check_all(mesh)
    assert ok, f"{ctx}: manifold invariants violated: {errs}"


@pytest.fixture(params=PRIMITIVES.keys())
def named_mesh(request):
    return request.param, PRIMITIVES[request.param]()


# ─────────────────────────────────────────────────────────────────────────────
# extrude_face: n-gon face → n new verts, 2n new edges, n new faces, χ const
# (DLFLExtrude.hh: extrudeFace — top face translated, n side quads inserted)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtrudeOracle:
    def test_delta_counts(self, named_mesh):
        name, mesh = named_mesh
        face = next(iter(mesh.faces.values()))
        n = len(face.halfedges())
        V0, E0, F0 = counts(mesh)
        chi0, g0 = mesh.euler_characteristic(), mesh.genus()

        extrude_face(mesh, face, dist=0.5)

        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (n, 2 * n, n), name
        assert mesh.euler_characteristic() == chi0, name
        assert mesh.genus() == g0, name
        assert_valid(mesh, f"extrude on {name}")

    def test_double_extrude(self):
        mesh = make_cube()
        face = next(iter(mesh.faces.values()))
        new_faces = extrude_face(mesh, face, dist=0.5)
        # Extrude again from the new top face (a quad)
        top = new_faces[0]
        n = len(top.halfedges())
        V0, E0, F0 = counts(mesh)
        extrude_face(mesh, top, dist=0.5)
        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (n, 2 * n, n)
        assert_valid(mesh, "double extrude on cube")


# ─────────────────────────────────────────────────────────────────────────────
# add_handle: two n-gons removed, n side quads added → Δχ=−2, genus +1
# (DLFLConnect.hh: connectFaces — the handle/hole operator)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddHandleOracle:
    def _opposite_faces(self, mesh):
        """Pick two faces with no shared vertices (opposite faces work)."""
        faces = list(mesh.faces.values())
        for i, f1 in enumerate(faces):
            v1 = {v.id for v in f1.vertices()}
            for f2 in faces[i + 1:]:
                v2 = {v.id for v in f2.vertices()}
                if not (v1 & v2) and len(v1) == len(v2):
                    return f1, f2
        pytest.skip("no disjoint equal-degree face pair")

    @pytest.mark.parametrize("prim", ["cube", "octahedron", "icosahedron"])
    def test_genus_increment(self, prim):
        mesh = PRIMITIVES[prim]()
        f1, f2 = self._opposite_faces(mesh)
        n = len(f1.halfedges())
        V0, E0, F0 = counts(mesh)
        g0 = mesh.genus()

        add_handle(mesh, f1, f2)

        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (0, n, n - 2), prim
        assert mesh.euler_characteristic() == V0 - E0 + F0 - 2, prim
        assert mesh.genus() == g0 + 1, prim
        assert_valid(mesh, f"add_handle on {prim}")

    def test_genus_2(self):
        """Two handles on a cube → genus 2 (χ = −2)."""
        mesh = make_cube()
        f1, f2 = self._opposite_faces(mesh)
        add_handle(mesh, f1, f2)
        assert mesh.genus() == 1
        f3, f4 = self._opposite_faces(mesh)
        add_handle(mesh, f3, f4)
        assert mesh.genus() == 2
        assert mesh.euler_characteristic() == -2
        assert_valid(mesh, "genus-2 cube")


# ─────────────────────────────────────────────────────────────────────────────
# stellate / subdivide_face: n-gon → +1 vert, +n edges, +(n−1) faces
# (DLFLSubdiv.hh: stellateSubdivide semantics, single face)
# ─────────────────────────────────────────────────────────────────────────────

class TestStellateOracle:
    def test_delta_counts(self, named_mesh):
        name, mesh = named_mesh
        face = next(iter(mesh.faces.values()))
        n = len(face.halfedges())
        V0, E0, F0 = counts(mesh)
        chi0 = mesh.euler_characteristic()

        stellate(mesh, face)

        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (1, n, n - 1), name
        assert mesh.euler_characteristic() == chi0, name
        assert_valid(mesh, f"stellate on {name}")

    def test_subdivide_face_same_oracle(self, named_mesh):
        name, mesh = named_mesh
        face = next(iter(mesh.faces.values()))
        n = len(face.halfedges())
        V0, E0, F0 = counts(mesh)

        subdivide_face(mesh, face)

        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (1, n, n - 1), name
        assert_valid(mesh, f"subdivide_face on {name}")


# ─────────────────────────────────────────────────────────────────────────────
# subdivide_edge: +1 vert, +1 edge, faces unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestSubdivideEdgeOracle:
    def test_delta_counts(self, named_mesh):
        name, mesh = named_mesh
        edge = next(iter(mesh.edges.values()))
        V0, E0, F0 = counts(mesh)
        chi0 = mesh.euler_characteristic()

        subdivide_edge(mesh, edge)

        V1, E1, F1 = counts(mesh)
        assert (V1 - V0, E1 - E0, F1 - F0) == (1, 1, 0), name
        assert mesh.euler_characteristic() == chi0, name
        assert_valid(mesh, f"subdivide_edge on {name}")


# ─────────────────────────────────────────────────────────────────────────────
# catmull_clark: V' = V+E+F, E' = 4E, F' = 2E, all-quad, χ & genus preserved
# (DLFLSubdiv.hh: catmullClarkSubdivide)
# ─────────────────────────────────────────────────────────────────────────────

class TestCatmullClarkOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        g0 = mesh.genus()

        out = catmull_clark(mesh)

        assert counts(out) == (V + E + F, 4 * E, 2 * E), name
        assert out.euler_characteristic() == mesh.euler_characteristic(), name
        assert out.genus() == g0, name
        assert all(len(f.halfedges()) == 4 for f in out.faces.values()), name
        assert_valid(out, f"catmull_clark on {name}")

    def test_on_genus_1(self):
        """CC must preserve genus on a handle-bearing mesh."""
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1 = faces[0]
        v1 = {v.id for v in f1.vertices()}
        f2 = next(f for f in faces[1:]
                  if not (v1 & {v.id for v in f.vertices()}))
        add_handle(mesh, f1, f2)
        assert mesh.genus() == 1

        V, E, F = counts(mesh)
        out = catmull_clark(mesh)
        assert counts(out) == (V + E + F, 4 * E, 2 * E)
        assert out.genus() == 1
        assert_valid(out, "catmull_clark on genus-1")

    def test_two_rounds(self):
        mesh = make_tetrahedron()
        once = catmull_clark(mesh)
        V, E, F = counts(once)
        twice = catmull_clark(once)
        assert counts(twice) == (V + E + F, 4 * E, 2 * E)
        assert_valid(twice, "catmull_clark ×2 on tetrahedron")


# ─────────────────────────────────────────────────────────────────────────────
# Composition: oracle deltas must accumulate exactly over an operator sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestSequenceOracle:
    def test_extrude_then_stellate_then_handle(self):
        mesh = make_cube()                       # V8 E12 F6, g0
        face = next(iter(mesh.faces.values()))   # quad
        extrude_face(mesh, face, 0.5)            # +4V +8E +4F
        assert counts(mesh) == (12, 20, 10)

        f = next(iter(mesh.faces.values()))
        n = len(f.halfedges())
        stellate(mesh, f)                        # +1V +nE +(n−1)F
        assert counts(mesh) == (13, 20 + n, 10 + n - 1)

        # find disjoint equal-degree pair for a handle
        faces = list(mesh.faces.values())
        pair = None
        for i, f1 in enumerate(faces):
            s1 = {v.id for v in f1.vertices()}
            for f2 in faces[i + 1:]:
                s2 = {v.id for v in f2.vertices()}
                if not (s1 & s2) and len(s1) == len(s2):
                    pair = (f1, f2)
                    break
            if pair:
                break
        assert pair is not None
        m = len(pair[0].halfedges())
        V0, E0, F0 = counts(mesh)
        add_handle(mesh, *pair)                  # +0V +mE +(m−2)F, genus 1
        assert counts(mesh) == (V0, E0 + m, F0 + m - 2)
        assert mesh.genus() == 1
        assert_valid(mesh, "composed sequence")
