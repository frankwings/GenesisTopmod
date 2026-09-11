#!/usr/bin/env python3
"""
test_stellate_cluster.py — Standalone smoke test for topmod_stellate_cluster.

No GPU needed.  Tests:
  1. Build icosphere (make_icosahedron + 1 catmull_clark).
  2. Stellate first 3 faces via topmod_stellate_cluster.
  3. Assert boundary_edges == 0  (watertight).
  4. Assert face count increased by exactly 2 per stellated face:
     each triangle stellate: V+1, E+3, F+2  (so 3 faces → old_F + 6).
  5. Assert χ is unchanged per stellate  (each: 1-3+2=0 net, so total χ fixed).
     χ = V - E + F
  6. Print PASS.
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V5_DIR     = os.path.dirname(_SCRIPT_DIR)          # opseq_v5/
_REPO_ROOT  = os.path.dirname(os.path.dirname(_V5_DIR))
for _p in (_REPO_ROOT, _V5_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from topmod.primitives  import make_icosahedron
from topmod.subdivision import catmull_clark
from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

# Import the function under test from the parent opseq_v5 package
sys.path.insert(0, _V5_DIR)
from eval_extrude_v3 import topmod_stellate_cluster, dlfl_boundary_stats


def euler_characteristic(verts_np: np.ndarray, tris_np: np.ndarray) -> int:
    V = len(verts_np)
    F = len(tris_np)
    edges: set = set()
    for tri in tris_np:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    E = len(edges)
    return V - E + F


def main():
    print("Building icosphere (make_icosahedron + 1 catmull_clark)...")
    ico = make_icosahedron()
    ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    verts_np = np.array(positions, dtype=np.float64)
    tris_np  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    old_F = len(tris_np)
    old_V = len(verts_np)
    chi_before = euler_characteristic(verts_np, tris_np)
    print(f"  Init mesh: V={old_V} F={old_F}  χ={chi_before}")

    # Verify it starts watertight
    n_bnd0, _ = dlfl_boundary_stats(verts_np, tris_np)
    assert n_bnd0 == 0, f"Initial mesh has {n_bnd0} boundary edges — not watertight"

    # ── Stellate first 3 faces ───────────────────────────────────────────────
    cluster = np.array([0, 1, 2], dtype=np.int32)
    n_stellated = len(cluster)
    print(f"Stellating {n_stellated} faces (indices {cluster.tolist()})...")

    new_verts, new_tris, ret_old_V = topmod_stellate_cluster(
        verts_np, tris_np, cluster)

    new_F = len(new_tris)
    new_V = len(new_verts)
    chi_after = euler_characteristic(new_verts, new_tris)

    print(f"  After stellate: V={new_V} F={new_F}  χ={chi_after}")

    # ── Assertions ───────────────────────────────────────────────────────────
    # 1. Watertight
    n_bnd, n_nm = dlfl_boundary_stats(new_verts, new_tris)
    assert n_bnd == 0, f"FAIL: boundary_edges={n_bnd} (expected 0)"
    assert n_nm  == 0, f"FAIL: non-manifold_edges={n_nm} (expected 0)"
    print("  [OK] boundary_edges == 0  and  non_manifold == 0")

    # 2. Face count: each triangle stellate adds 2 faces (F+2 per face)
    expected_F = old_F + 2 * n_stellated
    assert new_F == expected_F, (
        f"FAIL: face count {new_F} != expected {expected_F} "
        f"(old_F={old_F} + 2*{n_stellated})")
    print(f"  [OK] face count {old_F} → {new_F}  (+{new_F - old_F} = 2×{n_stellated})")

    # 3. Vertex count: each stellate adds 1 center vertex
    expected_V = old_V + n_stellated
    assert new_V == expected_V, (
        f"FAIL: vertex count {new_V} != expected {expected_V}")
    print(f"  [OK] vertex count {old_V} → {new_V}  (+{n_stellated})")

    # 4. old_V returned correctly (new center verts at old_V..new_V-1)
    assert ret_old_V == old_V, (
        f"FAIL: returned old_V={ret_old_V} != actual old_V={old_V}")
    print(f"  [OK] returned old_V == {old_V}")

    # 5. χ unchanged (each triangle stellate: ΔV=1, ΔE=3, ΔF=2 → Δχ=0)
    assert chi_after == chi_before, (
        f"FAIL: χ changed {chi_before} → {chi_after} (should be unchanged)")
    print(f"  [OK] Euler characteristic χ={chi_before} unchanged")

    print("\nPASS — all stellate_cluster assertions satisfied")


if __name__ == '__main__':
    main()
