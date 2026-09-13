"""
Equivalence test: topmod_core.self_intersecting_pairs vs Open3D.

Compares exact pair sets (i<j) on synthetic meshes and on real npz meshes.
Run from the repo root:
    python3 topmod/cpp/test_si_equivalence.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import topmod_core as tc

_REPO    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_RESULTS = os.path.join(_REPO, 'experiments', 'opseq_v5', 'despike', 'results_genus')

PASS = 0
FAIL = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def open3d_si_set(V, F):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(V, float)),
        o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    pairs = np.asarray(om.get_self_intersecting_triangles())
    if len(pairs) == 0:
        return set()
    return {(int(min(a, b)), int(max(a, b))) for a, b in pairs}


def cpp_si_set(V, F):
    pairs = tc.self_intersecting_pairs(
        np.asarray(V, np.float64), np.asarray(F, np.int64))
    return {(int(p[0]), int(p[1])) for p in pairs}


def check(name, V, F, expected_n=None):
    global PASS, FAIL
    t0 = time.perf_counter()
    py_set  = open3d_si_set(V, F)
    t_py    = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    cpp_set = cpp_si_set(V, F)
    t_cpp   = (time.perf_counter() - t0) * 1000

    if expected_n is not None and len(py_set) != expected_n:
        print(f"[NOTE] {name}: Open3D got {len(py_set)} pairs (expected hint {expected_n})")

    only_py  = py_set  - cpp_set
    only_cpp = cpp_set - py_set

    if not only_py and not only_cpp:
        PASS += 1
        print(f"[PASS] {name}: {len(cpp_set)} pairs  "
              f"Open3D={t_py:.1f}ms  C++={t_cpp:.1f}ms")
        return True
    else:
        FAIL += 1
        print(f"[FAIL] {name}: Open3D={len(py_set)} C++={len(cpp_set)}")
        if only_py:  print(f"  Missing from C++ (first 5): {sorted(only_py)[:5]}")
        if only_cpp: print(f"  Extra in C++ (first 5):    {sorted(only_cpp)[:5]}")
        return False


# ── Synthetic meshes ───────────────────────────────────────────────────────────

def test_empty():
    V = np.zeros((3, 3), dtype=np.float64)
    F = np.zeros((0, 3), dtype=np.int64)
    pairs = tc.self_intersecting_pairs(V, F)
    assert pairs.shape == (0, 2), f"Expected (0,2), got {pairs.shape}"
    global PASS; PASS += 1
    print(f"[PASS] empty mesh: shape={pairs.shape}")


def test_clean_icos():
    """Icosphere from topmod: should have 0 SI pairs."""
    import topmod
    from topmod.io import to_triangle_arrays
    m = topmod.make_icosahedron()
    vv, ff = to_triangle_arrays(m)
    V = np.array(vv, float); F = np.array(ff, np.int64)
    check("icosahedron (clean)", V, F, expected_n=0)


def test_two_crossing():
    """Two triangles in X-shape: no shared vertex → 1 pair."""
    V = np.array([
        [-1, 0, 0], [1, 0, 0], [0, 0, 1],   # T0: horizontal
        [0, -1, 0.5], [0, 1, 0.5], [0, 0, -0.5]  # T1: crossing
    ], dtype=np.float64)
    F = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    check("two crossing triangles", V, F, expected_n=1)


def test_adjacent_no_si():
    """Two triangles sharing an edge: adjacent, excluded."""
    V = np.array([[0,0,0],[1,0,0],[0.5,1,0],[0.5,-1,0]], dtype=np.float64)
    F = np.array([[0,1,2],[0,1,3]], dtype=np.int64)
    check("adjacent triangles (shared edge)", V, F, expected_n=0)


def test_adjacent_vertex_only():
    """Two triangles sharing only one vertex: excluded."""
    V = np.array([[0,0,0],[1,0,0],[0.5,1,0],[0,-1,0],[-1,-1,0]], dtype=np.float64)
    F = np.array([[0,1,2],[0,3,4]], dtype=np.int64)
    check("triangles sharing one vertex (excluded)", V, F, expected_n=0)


def test_coplanar_overlap():
    """Two coplanar overlapping triangles: Open3D returns 0 (coplanar excluded)."""
    V = np.array([[0,0,0],[2,0,0],[1,2,0],[1,0,0],[3,0,0],[2,2,0]], dtype=np.float64)
    F = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    check("coplanar overlapping (excluded)", V, F, expected_n=0)


def test_parallel_non_touching():
    """Two non-intersecting triangles in parallel planes."""
    V = np.array([[0,0,0],[1,0,0],[0.5,1,0],[0,0,0.5],[1,0,0.5],[0.5,1,0.5]], dtype=np.float64)
    F = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    check("parallel planes no touch", V, F, expected_n=0)


def test_many_random_intersecting():
    """Random soup of triangles — compare pair-for-pair."""
    rng = np.random.default_rng(7)
    n_tri = 40
    V = rng.uniform(-1, 1, (n_tri * 3, 3))
    F = np.arange(n_tri * 3, dtype=np.int64).reshape(n_tri, 3)
    check("40 random non-sharing triangles", V, F)


def test_cc2_icosphere():
    """2×CC icosphere (960 faces): should be clean."""
    import topmod
    from topmod.io import to_triangle_arrays
    from topmod.subdivision import catmull_clark as cc
    m = topmod.make_icosahedron()
    m = cc(m); m = cc(m)
    vv, ff = to_triangle_arrays(m)
    V = np.array(vv, float); F = np.array(ff, np.int64)
    check("CC2 icosphere (960F, clean)", V, F, expected_n=0)


# ── Real meshes (npz) ─────────────────────────────────────────────────────────

def test_armadillo():
    """armadillo_g3chain_raw: clean mesh → 0 pairs."""
    path = os.path.join(_RESULTS, 'armadillo_g3chain_raw.npz')
    if not os.path.exists(path):
        print(f"[SKIP] armadillo not found: {path}"); return
    npz = np.load(path)
    V = npz['verts'].astype(np.float64); F = npz['tris'].astype(np.int64)
    check(f"armadillo (V={len(V):,} F={len(F):,}, clean)", V, F, expected_n=0)


def test_kitten():
    """kitten_g3ccchain_raw: may have SI pairs."""
    path = os.path.join(_RESULTS, 'kitten_g3ccchain_raw.npz')
    if not os.path.exists(path):
        print(f"[SKIP] kitten not found: {path}"); return
    npz = np.load(path)
    V = npz['verts'].astype(np.float64); F = npz['tris'].astype(np.int64)
    check(f"kitten_g3ccchain (V={len(V):,} F={len(F):,})", V, F)


def test_fertility():
    """fertility_g3ccchain_raw: ~270 SI pairs expected."""
    path = os.path.join(_RESULTS, 'fertility_g3ccchain_raw.npz')
    if not os.path.exists(path):
        print(f"[SKIP] fertility not found: {path}"); return
    npz = np.load(path)
    V = npz['verts'].astype(np.float64); F = npz['tris'].astype(np.int64)
    check(f"fertility_g3ccchain (V={len(V):,} F={len(F):,})", V, F, expected_n=270)


def test_perturbed_armadillo():
    """Perturbed armadillo (Gaussian noise): may create self-intersections."""
    path = os.path.join(_RESULTS, 'armadillo_g3chain_raw.npz')
    if not os.path.exists(path):
        print(f"[SKIP] armadillo not found"); return
    npz = np.load(path)
    V0 = npz['verts'].astype(np.float64); F = npz['tris'].astype(np.int64)
    rng = np.random.default_rng(99)
    el  = np.linalg.norm(V0[F[:,0]] - V0[F[:,1]], axis=1).mean()
    V   = V0 + rng.normal(0, 0.02 * el, V0.shape)
    check(f"perturbed armadillo (2% noise, F={len(F):,})", V, F)


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("SI equivalence: topmod_core vs Open3D")
    print("=" * 65)
    test_empty()
    test_clean_icos()
    test_two_crossing()
    test_adjacent_no_si()
    test_adjacent_vertex_only()
    test_coplanar_overlap()
    test_parallel_non_touching()
    test_many_random_intersecting()
    test_cc2_icosphere()
    test_armadillo()
    test_kitten()
    test_fertility()
    test_perturbed_armadillo()
    print("=" * 65)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
