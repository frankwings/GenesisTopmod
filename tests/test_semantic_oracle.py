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


# ─────────────────────────────────────────────────────────────────────────────
# Batch 2 (reference_semantics.md): honeycomb, star, corner_cutting,
# loop_style, fractal — compositions / geometric variants of existing ops
# ─────────────────────────────────────────────────────────────────────────────

from topmod.remeshing import (   # noqa: E402
    honeycomb_subdivide, star_subdivide, corner_cutting,
    loop_style_subdivide, fractal_subdivide,
)


class TestHoneycombOracle:
    """honeycomb = dual ∘ stellate_all topologically: V'=2E, E'=3E, F'=F+V."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = honeycomb_subdivide(mesh)
        assert counts(out) == (2 * E, 3 * E, F + V), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"honeycomb on {name}")

    def test_cube_face_types(self):
        """Cube → 6 quads (per face) + 8 hexagons (per valence-3 vertex)."""
        out = honeycomb_subdivide(make_cube())
        assert counts(out) == (24, 36, 14)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [4] * 6 + [6] * 8

    def test_matches_dual_stellate_composition(self):
        """Same element counts as dual(stellate_all(M))."""
        m1 = make_icosahedron()
        out = honeycomb_subdivide(m1)
        m2 = make_icosahedron()
        stellate_all(m2)
        comp = dual(m2)
        assert counts(out) == counts(comp)
        assert (sorted(f.degree() for f in out.faces.values())
                == sorted(f.degree() for f in comp.faces.values()))


class TestStarOracle:
    """star = stellate_all ∘ stellate_all: V'=V+F+2E, E'=9E, F'=6E."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        chi0 = mesh.euler_characteristic()
        star_subdivide(mesh)
        assert counts(mesh) == (V + F + 2 * E, 9 * E, 6 * E), name
        assert mesh.euler_characteristic() == chi0, name
        assert all(f.degree() == 3 for f in mesh.faces.values()), name
        assert_valid(mesh, f"star on {name}")

    def test_cube(self):
        mesh = make_cube()
        star_subdivide(mesh, offset=0.3)
        assert counts(mesh) == (38, 108, 72)


class TestCornerCuttingOracle:
    """corner_cutting(alpha): geometric variant of doo_sabin — same topology."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = corner_cutting(mesh, alpha=0.5)
        assert counts(out) == (2 * E, 4 * E, V + E + F), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"corner_cutting on {name}")

    def test_same_topology_as_doo_sabin(self):
        mesh = make_cube()
        cc = corner_cutting(mesh, alpha=0.7)
        ds = doo_sabin(mesh)
        assert counts(cc) == counts(ds)
        assert (sorted(f.degree() for f in cc.faces.values())
                == sorted(f.degree() for f in ds.faces.values()))

    def test_alpha_changes_geometry_not_topology(self):
        a = corner_cutting(make_cube(), alpha=0.3)
        b = corner_cutting(make_cube(), alpha=0.9)
        assert counts(a) == counts(b)
        pa = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in a.vertices.values())
        pb = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in b.vertices.values())
        assert pa != pb


class TestLoopStyleOracle:
    """loop_style: polygonal Loop connectivity: V'=V+E, E'=4E, F'=F+2E."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = loop_style_subdivide(mesh)
        assert counts(out) == (V + E, 4 * E, F + 2 * E), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"loop_style on {name}")

    def test_cube(self):
        out = loop_style_subdivide(make_cube())
        assert counts(out) == (20, 48, 30)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [3] * 24 + [4] * 6   # 24 corner tris + 6 central quads

    def test_matches_loop_counts_on_triangles(self):
        """On an all-tri mesh, connectivity counts equal Loop's."""
        m = make_icosahedron()
        assert counts(loop_style_subdivide(m)) == counts(loop_subdivide(m))


class TestFractalOracle:
    """fractal = loop_style split + stellate central faces:
    V'=V+E+F, E'=6E, F'=4E, all-tri."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = fractal_subdivide(mesh, offset=1.0)
        assert counts(out) == (V + E + F, 6 * E, 4 * E), name
        assert out.genus() == mesh.genus(), name
        assert all(f.degree() == 3 for f in out.faces.values()), name
        assert_valid(out, f"fractal on {name}")

    def test_cube(self):
        out = fractal_subdivide(make_cube())
        assert counts(out) == (26, 72, 48)


# ─────────────────────────────────────────────────────────────────────────────
# Batch 3 — pentagonal / pentagonal2 / dual1264 / root4
# (docs/reference_semantics.md §§2a, 2b, 8, 4)
# ─────────────────────────────────────────────────────────────────────────────

from topmod.remeshing import (   # noqa: E402
    pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide,
)


class TestPentagonalOracle:
    """pentagonal: trisect edges + centroid spokes to every third corner:
    V'=V+2E+F, E'=5E, F'=2E, all pentagons."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = pentagonal_subdivide(mesh)
        assert counts(out) == (V + 2 * E + F, 5 * E, 2 * E), name
        assert out.genus() == mesh.genus(), name
        assert all(f.degree() == 5 for f in out.faces.values()), name
        assert_valid(out, f"pentagonal on {name}")

    def test_cube(self):
        out = pentagonal_subdivide(make_cube())
        assert counts(out) == (38, 60, 24)

    def test_tetrahedron_is_dodecahedron(self):
        """Reference semantics: tetra → dodecahedron combinatorics."""
        out = pentagonal_subdivide(make_tetrahedron())
        assert counts(out) == (20, 30, 12)
        assert all(f.degree() == 5 for f in out.faces.values())

    def test_offset_changes_geometry_not_topology(self):
        a = pentagonal_subdivide(make_cube(), offset=0.0)
        b = pentagonal_subdivide(make_cube(), offset=0.5)
        assert counts(a) == counts(b)
        pa = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in a.vertices.values())
        pb = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in b.vertices.values())
        assert pa != pb


class TestPentagonal2Oracle:
    """pentagonal2: midpoint split + scaled inner d-gon + connectors:
    V'=V+3E, E'=6E, F'=F+2E (inner d-gons + pentagons)."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = pentagonal2_subdivide(mesh)
        assert counts(out) == (V + 3 * E, 6 * E, F + 2 * E), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"pentagonal2 on {name}")

    def test_cube_face_degrees(self):
        out = pentagonal2_subdivide(make_cube())
        assert counts(out) == (44, 72, 30)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [4] * 6 + [5] * 24   # 6 inner quads + 24 pentagons


class TestDual1264Oracle:
    """dual1264: DS-like with a 2d-gon inner face per old face:
    V'=4E, E'=6E, F'=F+E+V."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = dual1264_subdivide(mesh)
        assert counts(out) == (4 * E, 6 * E, F + E + V), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"dual1264 on {name}")

    def test_cube_face_degrees(self):
        out = dual1264_subdivide(make_cube())
        assert counts(out) == (48, 72, 26)
        degs = sorted(f.degree() for f in out.faces.values())
        # 12 edge quads + 8 vertex hexagons (valence 3 → 2n=6) + 6 face octagons
        assert degs == [4] * 12 + [6] * 8 + [8] * 6


class TestRoot4Oracle:
    """root4: inner d-gon + prism bridge, old edges deleted:
    V'=V+2E, E'=4E, F'=F+E (inner d-gons + edge hexagons)."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = root4_subdivide(mesh)
        assert counts(out) == (V + 2 * E, 4 * E, F + E), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"root4 on {name}")

    def test_cube_face_degrees(self):
        out = root4_subdivide(make_cube())
        assert counts(out) == (32, 48, 18)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [4] * 6 + [6] * 12   # 6 inner quads + 12 edge hexagons

    def test_params_change_geometry_not_topology(self):
        a = root4_subdivide(make_cube(), a=0.0, twist=0.0)
        b = root4_subdivide(make_cube(), a=0.5, twist=0.3)
        assert counts(a) == counts(b)
        pa = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in a.vertices.values())
        pb = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in b.vertices.values())
        assert pa != pb


# ─────────────────────────────────────────────────────────────────────────────
# Batch 4a — checkerboard / doo-sabin BC-new
# (docs/reference_semantics.md §§9, 11)
# ─────────────────────────────────────────────────────────────────────────────

from topmod.remeshing import (   # noqa: E402
    checkerboard_remesh, ds_bc_new_subdivide,
)


class TestCheckerboardOracle:
    """checkerboard: inset + edge trisection + corner chords − spokes:
    V'=V+4E, E'=9E, F'=F+4E; all-quad on quad input."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = checkerboard_remesh(mesh)
        assert counts(out) == (V + 4 * E, 9 * E, F + 4 * E), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"checkerboard on {name}")

    def test_cube_all_quads(self):
        out = checkerboard_remesh(make_cube())
        assert counts(out) == (56, 108, 54)
        assert all(f.degree() == 4 for f in out.faces.values())

    def test_thickness_changes_geometry_not_topology(self):
        a = checkerboard_remesh(make_cube(), thickness=0.2)
        b = checkerboard_remesh(make_cube(), thickness=0.4)
        assert counts(a) == counts(b)
        pa = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in a.vertices.values())
        pb = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in b.vertices.values())
        assert pa != pb


class TestDsBCNewOracle:
    """ds_bc_new: DS on the mid-edge-refined boundary, old vertices
    survive: V'=V+4E, E'=7E, F'=F+2E (2d-gon per face + 2 pentagons
    per edge)."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        out = ds_bc_new_subdivide(mesh)
        assert counts(out) == (V + 4 * E, 7 * E, F + 2 * E), name
        assert out.genus() == mesh.genus(), name
        assert_valid(out, f"ds_bc_new on {name}")

    def test_cube_face_degrees(self):
        out = ds_bc_new_subdivide(make_cube())
        assert counts(out) == (56, 84, 30)
        degs = sorted(f.degree() for f in out.faces.values())
        assert degs == [5] * 24 + [8] * 6   # 24 pentagons + 6 face octagons

    def test_params_change_geometry_not_topology(self):
        a = ds_bc_new_subdivide(make_cube(), sf=1.0, length=1.0)
        b = ds_bc_new_subdivide(make_cube(), sf=0.7, length=0.5)
        assert counts(a) == counts(b)
        pa = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in a.vertices.values())
        pb = sorted((round(v.x, 6), round(v.y, 6), round(v.z, 6))
                    for v in b.vertices.values())
        assert pa != pb


# ─────────────────────────────────────────────────────────────────────────────
# Batch 4b — dome (docs/reference_semantics.md §7)
# ─────────────────────────────────────────────────────────────────────────────

from topmod.remeshing import dome_subdivide   # noqa: E402


class TestDomeOracle:
    """dome = subdivide_all_edges(4) + 7 DS-extrusions per old face:
    V'=V+59E, E'=116E, F'=F+56E (in place)."""

    def test_counts(self, named_mesh):
        name, mesh = named_mesh
        V, E, F = counts(mesh)
        g = mesh.genus()
        dome_subdivide(mesh)
        assert counts(mesh) == (V + 59 * E, 116 * E, F + 56 * E), name
        assert mesh.genus() == g, name
        assert_valid(mesh, f"dome on {name}")

    def test_cube(self):
        mesh = make_cube()
        dome_subdivide(mesh)
        assert counts(mesh) == (716, 1392, 678)
