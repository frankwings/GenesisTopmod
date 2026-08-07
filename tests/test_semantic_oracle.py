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


# ─────────────────────────────────────────────────────────────────────────────
# dual: V'=F, E'=E, F'=V; χ & genus preserved; involution dual(dual(M)) ≅ M
# (DLFLDual.hh: createDual)
# ─────────────────────────────────────────────────────────────────────────────

from topmod.remeshing import dual, doo_sabin  # noqa: E402


class TestDualOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = dual(mesh)
        assert counts(out) == (F, E, V), name
        assert out.euler_characteristic() == mesh.euler_characteristic(), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"dual of {name}")

    def test_degree_valence_swap(self, named_mesh):
        """Face degrees of dual == vertex valences of primal (as multisets)."""
        name, mesh = named_mesh
        primal_valences = sorted(v.degree() for v in mesh.vertices.values())
        out = dual(mesh)
        dual_degrees = sorted(f.degree() for f in out.faces.values())
        assert dual_degrees == primal_valences, name

    def test_involution_counts(self, named_mesh):
        """dual(dual(M)) has exactly M's element counts and degree spectrum."""
        name, mesh = named_mesh
        dd = dual(dual(mesh))
        assert counts(dd) == counts(mesh), name
        assert (sorted(f.degree() for f in dd.faces.values())
                == sorted(f.degree() for f in mesh.faces.values())), name
        assert_valid(dd, f"dual^2 of {name}")

    def test_cube_octahedron_duality(self):
        """dual(cube) is combinatorially an octahedron (8 tri faces, 6 verts)."""
        out = dual(make_cube())
        assert counts(out) == (6, 12, 8)
        assert all(f.degree() == 3 for f in out.faces.values())

    def test_on_genus_1(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1 = faces[0]
        v1 = {v.id for v in f1.vertices()}
        f2 = next(f for f in faces[1:]
                  if not (v1 & {v.id for v in f.vertices()}))
        add_handle(mesh, f1, f2)
        V, E, F = counts(mesh)
        out = dual(mesh)
        assert counts(out) == (F, E, V)
        assert out.genus() == 1
        assert_valid(out, "dual of genus-1")


# ─────────────────────────────────────────────────────────────────────────────
# doo_sabin: V'=2E, E'=4E, F'=V+E+F; χ & genus preserved
# (DLFLSubdiv.hh: dooSabinSubdivideBC — face-face + edge-face + vertex-face)
# ─────────────────────────────────────────────────────────────────────────────

class TestDooSabinOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = doo_sabin(mesh)
        assert counts(out) == (2 * E, 4 * E, V + E + F), name
        assert out.euler_characteristic() == mesh.euler_characteristic(), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"doo_sabin on {name}")

    def test_cube_known_counts(self):
        """DS on cube: 24 verts, 48 edges, 26 faces (classic result)."""
        out = doo_sabin(make_cube())
        assert counts(out) == (24, 48, 26)

    def test_face_types(self):
        """DS on cube: 6 quads (face-face) + 12 quads (edge) + 8 tris (vertex)."""
        out = doo_sabin(make_cube())
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [3] * 8 + [4] * 18

    def test_on_genus_1(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1 = faces[0]
        v1 = {v.id for v in f1.vertices()}
        f2 = next(f for f in faces[1:]
                  if not (v1 & {v.id for v in f.vertices()}))
        add_handle(mesh, f1, f2)
        V, E, F = counts(mesh)
        out = doo_sabin(mesh)
        assert counts(out) == (2 * E, 4 * E, V + E + F)
        assert out.genus() == 1
        assert_valid(out, "doo_sabin on genus-1")

    def test_two_rounds(self):
        mesh = make_tetrahedron()
        once = doo_sabin(mesh)
        V, E, F = counts(once)
        twice = doo_sabin(once)
        assert counts(twice) == (2 * E, 4 * E, V + E + F)
        assert_valid(twice, "doo_sabin ×2 on tetrahedron")

    def test_ds_of_dual_equals_ds_counts(self):
        """DS is self-dual in counts: DS(M) and DS(dual(M)) have equal V/E/F."""
        mesh = make_cube()
        assert counts(doo_sabin(mesh)) == counts(doo_sabin(dual(mesh)))


# ─────────────────────────────────────────────────────────────────────────────
# stellate_all: per n-gon (+1, +n, +n−1) summed → V'=V+F, E'=3E, F'=2E, all-tri
# ─────────────────────────────────────────────────────────────────────────────

from topmod.high_level_ops import stellate_all  # noqa: E402
from topmod.remeshing import (                   # noqa: E402
    simplest_subdivide, vertex_cutting, loop_subdivide, sqrt3_subdivide,
)


class TestStellateAllOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        chi0 = mesh.euler_characteristic()
        stellate_all(mesh)
        assert counts(mesh) == (V + F, 3 * E, 2 * E), name
        assert mesh.euler_characteristic() == chi0, name
        assert all(f.degree() == 3 for f in mesh.faces.values()), name
        assert_valid(mesh, f"stellate_all on {name}")

    def test_on_genus_1(self):
        mesh = make_cube()
        faces = list(mesh.faces.values())
        f1 = faces[0]
        v1 = {v.id for v in f1.vertices()}
        f2 = next(f for f in faces[1:]
                  if not (v1 & {v.id for v in f.vertices()}))
        add_handle(mesh, f1, f2)
        V, E, F = counts(mesh)
        stellate_all(mesh)
        assert counts(mesh) == (V + F, 3 * E, 2 * E)
        assert mesh.genus() == 1
        assert_valid(mesh, "stellate_all on genus-1")


# ─────────────────────────────────────────────────────────────────────────────
# simplest_subdivide (mid-edge / Peters-Reif): V'=E, E'=2E, F'=F+V
# cube → cuboctahedron
# ─────────────────────────────────────────────────────────────────────────────

class TestSimplestOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = simplest_subdivide(mesh)
        assert counts(out) == (E, 2 * E, F + V), name
        assert out.euler_characteristic() == mesh.euler_characteristic(), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"simplest on {name}")

    def test_cube_gives_cuboctahedron(self):
        out = simplest_subdivide(make_cube())
        assert counts(out) == (12, 24, 14)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [3] * 8 + [4] * 6   # 8 vertex-tris + 6 face-quads


# ─────────────────────────────────────────────────────────────────────────────
# vertex_cutting (truncation): V'=2E, E'=3E, F'=F+V
# cube → truncated cube
# ─────────────────────────────────────────────────────────────────────────────

class TestVertexCuttingOracle:
    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = vertex_cutting(mesh)
        assert counts(out) == (2 * E, 3 * E, F + V), name
        assert out.euler_characteristic() == mesh.euler_characteristic(), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"vertex_cutting on {name}")

    def test_cube_gives_truncated_cube(self):
        out = vertex_cutting(make_cube())
        assert counts(out) == (24, 36, 14)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [3] * 8 + [8] * 6   # 8 vertex-tris + 6 octagons


# ─────────────────────────────────────────────────────────────────────────────
# loop_subdivide (triangles only): V'=V+E, E'=4E, F'=4F; precondition all-tri
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopOracle:
    @pytest.mark.parametrize("prim", ["tetrahedron", "octahedron", "icosahedron"])
    def test_counts(self, prim):
        mesh = PRIMITIVES[prim]()
        V, E, F = counts(mesh)
        out = loop_subdivide(mesh)
        assert counts(out) == (V + E, 4 * E, 4 * F), prim
        assert out.euler_characteristic() == mesh.euler_characteristic(), prim
        assert all(f.degree() == 3 for f in out.faces.values()), prim
        assert_valid(out, f"loop on {prim}")

    def test_rejects_non_triangular(self):
        with pytest.raises(ValueError):
            loop_subdivide(make_cube())

    def test_two_rounds(self):
        once = loop_subdivide(make_tetrahedron())
        V, E, F = counts(once)
        twice = loop_subdivide(once)
        assert counts(twice) == (V + E, 4 * E, 4 * F)
        assert_valid(twice, "loop ×2 on tetrahedron")


# ─────────────────────────────────────────────────────────────────────────────
# sqrt3_subdivide (triangles only): V'=V+F, E'=3E, F'=3F; precondition all-tri
# ─────────────────────────────────────────────────────────────────────────────

class TestSqrt3Oracle:
    @pytest.mark.parametrize("prim", ["tetrahedron", "octahedron", "icosahedron"])
    def test_counts(self, prim):
        mesh = PRIMITIVES[prim]()
        V, E, F = counts(mesh)
        out = sqrt3_subdivide(mesh)
        assert counts(out) == (V + F, 3 * E, 3 * F), prim
        assert out.euler_characteristic() == mesh.euler_characteristic(), prim
        assert all(f.degree() == 3 for f in out.faces.values()), prim
        assert_valid(out, f"sqrt3 on {prim}")

    def test_rejects_non_triangular(self):
        with pytest.raises(ValueError):
            sqrt3_subdivide(make_cube())

    def test_two_rounds(self):
        once = sqrt3_subdivide(make_octahedron())
        V, E, F = counts(once)
        twice = sqrt3_subdivide(once)
        assert counts(twice) == (V + F, 3 * E, 3 * F)
        assert_valid(twice, "sqrt3 ×2 on octahedron")
