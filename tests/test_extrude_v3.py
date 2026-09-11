"""
test_extrude_v3.py — QA assertions for the TopMod DLFL extrusion mechanism
used in eval_extrude_v3.py.

Tests verify:
  1. topmod_extrude_single_face produces zero boundary edges (watertight).
  2. topmod_extrude_single_face preserves Euler characteristic (genus unchanged).
  3. Vertex count increases by exactly 3 for a triangular face.
  4. Triangle count increases by exactly 6 (1 top tri + 3 quads → 6 new tris).
  5. Direction override correctness: new top-cap centroid displaced in extrude_dir.
  6. assert_watertight_v3 raises on a mesh with boundary edges.
  7. assert_watertight_v3 raises on a mesh with non-manifold edges.
  8. Round-trip: _build_mesh → to_triangle_arrays is lossless (V, F counts).
  9. Multiple sequential injections on same mesh remain watertight.
 10. dlfl_boundary_stats returns (0, 0) for a closed mesh, (>0, 0) after
     artificially opening a boundary.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "opseq_v4"))

from topmod.primitives  import make_icosahedron, make_cube, _build_mesh
from topmod.subdivision import catmull_clark
from topmod.io          import to_triangle_arrays

# Import the functions under test from eval_extrude_v3
_V3_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "opseq_v5")
sys.path.insert(0, _V3_DIR)

from eval_extrude_v3 import (
    topmod_extrude_single_face,
    dlfl_boundary_stats,
    assert_watertight_v3,
    split_new_long_edges,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_ico_tri_mesh(n_subdiv: int = 2):
    """Catmull-Clark subdivided icosphere → numpy triangle mesh (closed)."""
    mesh = make_icosahedron()
    for _ in range(n_subdiv):
        mesh = catmull_clark(mesh)
    positions, tris = to_triangle_arrays(mesh)
    return np.array(positions, dtype=np.float64), np.array(tris, dtype=np.int32)


def extrude_dir_up():
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Watertight after extrusion
# ─────────────────────────────────────────────────────────────────────────────

class TestWatertightAfterExtrude:
    def test_zero_boundary_edges_face0(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, face_idx=0, extrude_dir=extrude_dir_up(), dist=0.2)
        bnd, nm = dlfl_boundary_stats(new_v, new_t)
        assert bnd == 0, f"Expected 0 boundary edges, got {bnd}"
        assert nm  == 0, f"Expected 0 non-manifold edges, got {nm}"

    def test_zero_boundary_edges_face_middle(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        mid = len(tris) // 2
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, face_idx=mid, extrude_dir=extrude_dir_up(), dist=0.3)
        bnd, nm = dlfl_boundary_stats(new_v, new_t)
        assert bnd == 0, f"Expected 0 boundary edges, got {bnd}"

    def test_zero_boundary_edges_last_face(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, face_idx=len(tris)-1,
            extrude_dir=extrude_dir_up(), dist=0.15)
        bnd, nm = dlfl_boundary_stats(new_v, new_t)
        assert bnd == 0, f"Expected 0 boundary edges, got {bnd}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Euler characteristic preserved
# ─────────────────────────────────────────────────────────────────────────────

class TestEulerPreserved:
    def _euler(self, verts, tris):
        V = len(verts)
        edges: set = set()
        for f in tris:
            for i in range(3):
                a, b = int(f[i]), int(f[(i+1)%3])
                edges.add((min(a,b), max(a,b)))
        E = len(edges); F = len(tris)
        return V - E + F

    def test_euler_unchanged(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        chi_before = self._euler(verts, tris)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), 0.2)
        chi_after = self._euler(new_v, new_t)
        assert chi_before == chi_after, \
            f"Euler changed: {chi_before} → {chi_after}"

    def test_genus_zero_preserved(self):
        """Icosphere has genus 0 (χ=2); extrude must keep it."""
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        chi_before = self._euler(verts, tris)
        assert chi_before == 2, f"Unexpected initial χ={chi_before}"
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, 5, extrude_dir_up(), 0.1)
        assert self._euler(new_v, new_t) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vertex count delta
# ─────────────────────────────────────────────────────────────────────────────

class TestVertexDelta:
    def test_exactly_3_new_vertices(self):
        """Extruding a triangle face adds exactly 3 top-cap vertices."""
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        V0 = len(verts)
        new_v, _, old_V = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), 0.2)
        assert old_V == V0, f"old_V mismatch: {old_V} != {V0}"
        assert new_v.shape[0] == V0 + 3, \
            f"Expected V+3={V0+3}, got {new_v.shape[0]}"

    def test_old_V_return_value(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        _, _, old_V = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), 0.1)
        assert old_V == len(verts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Triangle count delta
# ─────────────────────────────────────────────────────────────────────────────

class TestTriangleDelta:
    def test_exactly_6_new_triangles(self):
        """
        Extruding triangle face: remove 1, add 1 top-tri + 3 quads×2 tris = +6 net.
        """
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        F0 = len(tris)
        _, new_t, _ = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), 0.2)
        assert new_t.shape[0] == F0 + 6, \
            f"Expected F+6={F0+6}, got {new_t.shape[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Direction override
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionOverride:
    def test_top_cap_displaced_in_voting_direction(self):
        """
        New top-cap centroid should be ~ base centroid + dist * extrude_dir.
        """
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        face_idx = 3
        extrude_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        dist = 0.5

        # Base centroid of selected face
        f = tris[face_idx]
        base_centroid = verts[f].mean(axis=0)

        old_V = len(verts)
        new_v, _, _ = topmod_extrude_single_face(
            verts, tris, face_idx, extrude_dir, dist)

        # New vertices are at indices old_V..old_V+2
        top_centroid = new_v[old_V: old_V + 3].mean(axis=0)
        expected     = base_centroid + dist * extrude_dir

        np.testing.assert_allclose(
            top_centroid, expected, atol=1e-4,
            err_msg="Top-cap centroid should equal base + dist*extrude_dir")

    def test_different_directions_different_positions(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        old_V  = len(verts)
        dir_up = np.array([0., 1., 0.])
        dir_x  = np.array([1., 0., 0.])

        new_v_up, _, _ = topmod_extrude_single_face(verts, tris, 0, dir_up, 0.3)
        new_v_x,  _, _ = topmod_extrude_single_face(verts, tris, 0, dir_x,  0.3)

        up_cap = new_v_up[old_V: old_V + 3]
        x_cap  = new_v_x [old_V: old_V + 3]
        # They should not be identical
        assert not np.allclose(up_cap, x_cap), \
            "Different directions should produce different top-cap positions"


# ─────────────────────────────────────────────────────────────────────────────
# 6. assert_watertight_v3 raises on boundary mesh
# ─────────────────────────────────────────────────────────────────────────────

class TestAssertWatertight:
    def _make_open_mesh(self):
        """Remove one triangle from the icosphere to create a boundary."""
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        # Drop the last triangle → 3 boundary edges appear
        open_tris = tris[:-1]
        return verts, open_tris

    def test_raises_on_boundary(self):
        verts, open_tris = self._make_open_mesh()
        with pytest.raises(AssertionError, match="boundary"):
            assert_watertight_v3(verts, open_tris, "test boundary")

    def test_passes_on_closed_mesh(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        # Should not raise
        assert_watertight_v3(verts, tris, "test closed")

    def test_raises_on_nonmanifold(self):
        """Duplicating a face creates count=2 for inner edges and count=4 for shared."""
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        # Duplicate last face → 3 edges with count=3 (non-manifold)
        bad_tris = np.vstack([tris, tris[-1:]])
        with pytest.raises(AssertionError, match="[Nn]on"):
            assert_watertight_v3(verts, bad_tris, "test nonmanifold")


# ─────────────────────────────────────────────────────────────────────────────
# 7. _build_mesh → to_triangle_arrays round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_vertex_face_counts_preserved(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        positions  = [tuple(map(float, v)) for v in verts]
        face_idx   = [list(map(int, tris[i])) for i in range(len(tris))]
        mesh2      = _build_mesh(positions, face_idx)
        pos2, tri2 = to_triangle_arrays(mesh2)
        assert len(pos2) == len(verts), "V mismatch after round-trip"
        assert len(tri2) == len(tris), "F mismatch after round-trip"

    def test_positions_preserved(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        positions = [tuple(map(float, v)) for v in verts]
        face_idx  = [list(map(int, tris[i])) for i in range(len(tris))]
        mesh2     = _build_mesh(positions, face_idx)
        pos2, _   = to_triangle_arrays(mesh2)
        verts2    = np.array(pos2)
        # Positions should be identical (same insertion order)
        np.testing.assert_allclose(verts, verts2, atol=1e-10,
                                   err_msg="Positions changed across round-trip")

    def test_watertight_after_roundtrip(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        positions = [tuple(map(float, v)) for v in verts]
        face_idx  = [list(map(int, tris[i])) for i in range(len(tris))]
        mesh2     = _build_mesh(positions, face_idx)
        pos2, tri2 = to_triangle_arrays(mesh2)
        bnd, nm   = dlfl_boundary_stats(np.array(pos2), np.array(tri2))
        assert bnd == 0 and nm == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sequential multiple injections stay watertight
# ─────────────────────────────────────────────────────────────────────────────

class TestSequentialInjections:
    def test_three_injections_all_watertight(self):
        """Three consecutive injections on the same mesh — all must be watertight."""
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        ed = extrude_dir_up()

        for inj in range(3):
            face_to_extrude = (inj * 17) % len(tris)
            verts, tris, _ = topmod_extrude_single_face(
                verts, tris, face_to_extrude, ed, dist=0.1)
            bnd, nm = dlfl_boundary_stats(verts, tris)
            assert bnd == 0, f"Injection {inj+1}: {bnd} boundary edges"
            assert nm  == 0, f"Injection {inj+1}: {nm} non-manifold edges"

    def test_v_and_f_counts_after_n_injections(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        V0, F0 = len(verts), len(tris)
        ed = extrude_dir_up()
        for k in range(4):
            face_idx = (k * 13) % len(tris)
            verts, tris, _ = topmod_extrude_single_face(
                verts, tris, face_idx, ed, 0.1)
        assert len(verts) == V0 + 4 * 3,  f"Expected V+12, got {len(verts)}"
        assert len(tris)  == F0 + 4 * 6,  f"Expected F+24, got {len(tris)}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. dlfl_boundary_stats correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestDlflBoundaryStats:
    def test_closed_mesh_zero_boundary(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        bnd, nm = dlfl_boundary_stats(verts, tris)
        assert bnd == 0 and nm == 0

    def test_open_mesh_positive_boundary(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        open_tris = tris[:-1]   # drop 1 face → 3 boundary edges
        bnd, nm = dlfl_boundary_stats(verts, open_tris)
        assert bnd > 0, "Expected boundary edges in open mesh"
        assert nm == 0

    def test_nonmanifold_detected(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=1)
        bad_tris = np.vstack([tris, tris[-1:]])  # duplicate one face
        _, nm = dlfl_boundary_stats(verts, bad_tris)
        assert nm > 0, "Expected non-manifold edges with duplicate face"


# ─────────────────────────────────────────────────────────────────────────────
# 10. split_new_long_edges keeps mesh watertight
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitNewLongEdges:
    def test_split_keeps_watertight(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        old_V = len(verts)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), dist=1.5)  # large dist → long side-wall edges
        split_v, split_t = split_new_long_edges(new_v, new_t, old_V)
        bnd, nm = dlfl_boundary_stats(split_v, split_t)
        assert bnd == 0, f"split_new_long_edges broke watertightness: {bnd} bnd"
        assert nm  == 0, f"split_new_long_edges created non-manifold: {nm}"

    def test_split_increases_or_maintains_triangles(self):
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        old_V = len(verts)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), dist=2.0)
        split_v, split_t = split_new_long_edges(new_v, new_t, old_V)
        assert len(split_t) >= len(new_t), \
            "split_new_long_edges should not reduce triangle count"

    def test_no_split_needed_for_small_dist(self):
        """With tiny dist the side walls are short — no split expected."""
        verts, tris = make_ico_tri_mesh(n_subdiv=2)
        old_V = len(verts)
        new_v, new_t, _ = topmod_extrude_single_face(
            verts, tris, 0, extrude_dir_up(), dist=0.001)
        split_v, split_t = split_new_long_edges(new_v, new_t, old_V)
        # With tiny dist, nothing long enough to split
        assert len(split_t) == len(new_t)
