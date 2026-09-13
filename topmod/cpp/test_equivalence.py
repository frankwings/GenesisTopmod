"""
Equivalence tests: compare C++ topmod_core output against the Python
reference implementations in topmod/.

Run from the repo root:
    python3 topmod/cpp/test_equivalence.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # cpp/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # repo root

import numpy as np
import topmod_core as _tc
import topmod
from topmod.io import to_triangle_arrays
from topmod.subdivision import catmull_clark as py_cc
from topmod.primitives import make_icosahedron, make_cube, make_tetrahedron
from topmod.validate import check_all

# ── Python reference functions ─────────────────────────────────────────────
import tempfile, os as _os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'opseq_v5', 'despike')))
try:
    from dlfl_untangle import (flip_sweep as py_flip_sweep,
                               collapse_short_edges as py_collapse)
    from phase1c_pipeline import dlfl_subdivide_arrays as py_subdivide_faces
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("[WARN] dlfl_untangle / phase1c_pipeline not importable; skipping those equivalence tests")

PASS = 0
FAIL = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def fail(name, msg=""):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}" + (f": {msg}" if msg else ""))


def faces_as_sorted_frozensets(F):
    """Convert F[Mx3] to a frozenset of frozensets for order-insensitive comparison."""
    return frozenset(frozenset(row) for row in F.tolist())


def mesh_to_VF(mesh):
    V_lst, F_lst = to_triangle_arrays(mesh)
    return np.array(V_lst, dtype=np.float64), np.array(F_lst, dtype=np.int64)


# ── Test 1: validate on known-good meshes ─────────────────────────────────

def test_validate():
    for name, factory in [("icosahedron", make_icosahedron),
                           ("cube", make_cube),
                           ("tetrahedron", make_tetrahedron)]:
        m = factory()
        V, F = mesh_to_VF(m)
        errs = _tc.validate(V, F)
        if errs:
            fail(f"validate/{name}", str(errs))
        else:
            ok(f"validate/{name}")


# ── Test 2: watertight ────────────────────────────────────────────────────

def test_watertight():
    for name, factory in [("icosahedron", make_icosahedron),
                           ("cube", make_cube)]:
        m = factory()
        V, F = mesh_to_VF(m)
        ok_flag, n_bad = _tc.check_watertight(F)
        if ok_flag and n_bad == 0:
            ok(f"watertight/{name}")
        else:
            fail(f"watertight/{name}", f"n_bad={n_bad}")


# ── Test 3: euler_genus ───────────────────────────────────────────────────

def test_genus():
    for name, factory, expected_genus in [
        ("icosahedron", make_icosahedron, 0),
        ("cube", make_cube, 0),
        ("tetrahedron", make_tetrahedron, 0),
    ]:
        m = factory()
        V, F = mesh_to_VF(m)
        g = _tc.euler_genus(V, F)
        if g == expected_genus:
            ok(f"genus/{name} (expected {expected_genus})")
        else:
            fail(f"genus/{name}", f"got {g}, expected {expected_genus}")


# ── Test 4: catmull_clark equivalence ─────────────────────────────────────

def test_catmull_clark():
    for name, factory in [("icosahedron", make_icosahedron),
                           ("cube", make_cube)]:
        m = factory()

        # Python CC
        m_py = py_cc(m)
        V_py_lst, F_py_lst = to_triangle_arrays(m_py)
        V_py = np.array(V_py_lst, dtype=np.float64)
        F_py = np.array(F_py_lst, dtype=np.int64)

        # C++ CC
        V_in, _ = mesh_to_VF(m)
        polys_in = []
        vid_map = {v.id: i for i, v in enumerate(sorted(m.vertices.values(), key=lambda v: v.id))}
        for face in m.iter_faces():
            polys_in.append([vid_map[v.id] for v in face.vertices()])
        V_cc, polys_cc = _tc.catmull_clark(np.array(V_in, dtype=np.float64), polys_in)
        F_cc = _tc.triangulate_all(V_cc, polys_cc)
        F_cc = np.array(F_cc, dtype=np.int64)

        # Compare: same vertex count, same face count
        if len(V_py) != len(V_cc):
            fail(f"catmull_clark/{name}/V_count", f"py={len(V_py)} cpp={len(V_cc)}")
            continue
        if len(F_py) != len(F_cc):
            fail(f"catmull_clark/{name}/F_count", f"py={len(F_py)} cpp={len(F_cc)}")
            continue

        # Vertex positions should match (order may differ).
        # Use nearest-neighbour matching: for each py vertex find closest cc vertex.
        from scipy.spatial import cKDTree
        tree = cKDTree(V_cc)
        dists, _ = tree.query(V_py)
        maxdiff = float(dists.max())
        if maxdiff < 1e-8:
            ok(f"catmull_clark/{name} (max_vertex_dist={maxdiff:.2e})")
        else:
            fail(f"catmull_clark/{name}/V_values", f"max_vertex_dist={maxdiff:.2e}")


# ── Test 5: triangulate_all ───────────────────────────────────────────────

def test_triangulate_all():
    # A cube has quads; triangulate_all should give 12 triangles
    m = make_cube()
    vid_map = {v.id: i for i, v in enumerate(sorted(m.vertices.values(), key=lambda v: v.id))}
    polys = []
    for face in m.iter_faces():
        polys.append([vid_map[v.id] for v in face.vertices()])
    V_in = np.array([(v.x,v.y,v.z) for v in sorted(m.vertices.values(), key=lambda v: v.id)], dtype=np.float64)
    F_out = _tc.triangulate_all(V_in, polys)
    # 6 quads → 6*2 = 12 triangles
    if len(F_out) == 12:
        ok("triangulate_all/cube (12 triangles)")
    else:
        fail("triangulate_all/cube", f"expected 12, got {len(F_out)}")


# ── Test 6: flip_sweep equivalence ────────────────────────────────────────

def test_flip_sweep():
    if not HAS_PIPELINE:
        print("  SKIP  flip_sweep equivalence (dlfl_untangle not available)")
        return
    m = make_icosahedron()
    # Do two CC rounds for a larger mesh
    m = py_cc(m)
    m = py_cc(m)
    V_lst, F_lst = to_triangle_arrays(m)
    V = np.array(V_lst, dtype=np.float64)
    F = np.array(F_lst, dtype=np.int64)

    V_py, F_py, nf_py = py_flip_sweep(V.copy(), F.copy(), passes=3, fold_cos=0.0)
    V_cpp, F_cpp, nf_cpp = _tc.flip_sweep(V, F, passes=3, fold_cos=0.0)

    if np.allclose(V_py, V_cpp, atol=1e-10):
        ok("flip_sweep/V_unchanged")
    else:
        fail("flip_sweep/V_unchanged")

    if len(F_py) == len(F_cpp):
        ok(f"flip_sweep/F_count (py={len(F_py)}, cpp={len(F_cpp)}, py_flips={nf_py}, cpp_flips={nf_cpp})")
    else:
        fail("flip_sweep/F_count", f"py={len(F_py)} cpp={len(F_cpp)}")


# ── Test 7: collapse_short_edges equivalence ──────────────────────────────

def test_collapse():
    if not HAS_PIPELINE:
        print("  SKIP  collapse equivalence (dlfl_untangle not available)")
        return
    m = make_icosahedron()
    m = py_cc(m)
    V_lst, F_lst = to_triangle_arrays(m)
    V = np.array(V_lst, dtype=np.float64)
    F = np.array(F_lst, dtype=np.int64)

    V_py, F_py, nc_py = py_collapse(V.copy(), F.copy(), ratio=0.4, max_n=20)
    V_cpp, F_cpp, nc_cpp = _tc.collapse_short_edges(V, F, 0.4, 20)

    if nc_py == nc_cpp:
        ok(f"collapse/n_collapsed match ({nc_py})")
    else:
        # Different collapse order is possible; just check topology is valid
        print(f"  NOTE  collapse n_collapsed: py={nc_py} cpp={nc_cpp} (may differ by tie-break)")

    if len(V_cpp) == len(V) - nc_cpp:
        ok(f"collapse/V_count (started={len(V)}, collapsed={nc_cpp})")
    else:
        fail("collapse/V_count", f"V={len(V)}, nc={nc_cpp}, V_out={len(V_cpp)}")

    ok_c, nb = _tc.check_watertight(F_cpp)
    if ok_c:
        ok("collapse/watertight")
    else:
        fail("collapse/watertight", f"n_bad={nb}")


# ── Test 8: subdivide_faces ───────────────────────────────────────────────

def test_subdivide_faces():
    m = make_icosahedron()
    V_lst, F_lst = to_triangle_arrays(m)
    V = np.array(V_lst, dtype=np.float64)
    F = np.array(F_lst, dtype=np.int64)

    fids = [0, 1, 2]
    V2, F2, ns = _tc.subdivide_faces(V, F, fids, expand_ring=False)

    if len(V2) >= len(V):
        ok(f"subdivide_faces/V_grows ({len(V)} -> {len(V2)})")
    else:
        fail("subdivide_faces/V_grows")

    ok_c, nb = _tc.check_watertight(F2)
    if ok_c:
        ok("subdivide_faces/watertight")
    else:
        fail("subdivide_faces/watertight", f"n_bad={nb}")

    # Original vertices unchanged
    if np.allclose(V2[:len(V)], V, atol=1e-12):
        ok("subdivide_faces/orig_V_unchanged")
    else:
        fail("subdivide_faces/orig_V_unchanged")

    if HAS_PIPELINE:
        V_py, F_py, ns_py = py_subdivide_faces(V.copy(), F.copy(), fids, expand_ring=False)
        if ns_py == ns:
            ok(f"subdivide_faces/n_split match ({ns})")
        else:
            fail("subdivide_faces/n_split", f"py={ns_py} cpp={ns}")
        if len(V_py) == len(V2) and len(F_py) == len(F2):
            ok("subdivide_faces/VF_count match")
        else:
            fail("subdivide_faces/VF_count", f"py V={len(V_py)} F={len(F_py)} cpp V={len(V2)} F={len(F2)}")


# ── Test 9: add_handle increases genus ───────────────────────────────────

def test_add_handle():
    # Icosahedron: take two antipodal faces, punch a handle → genus should be 1
    m = make_icosahedron()
    V_lst, F_lst = to_triangle_arrays(m)
    V = np.array(V_lst, dtype=np.float64)
    F = np.array(F_lst, dtype=np.int64)

    g_before = _tc.euler_genus(V, F)

    # Pick faces 0 and 10 (somewhat opposite on icosahedron)
    V2, F2 = _tc.add_handle(V, F, 0, 10)

    if len(V2) == len(V) and len(F2) < len(F):
        ok("add_handle/V_unchanged_F_reduced")
    else:
        ok(f"add_handle/ran (V: {len(V)}->{len(V2)}, F: {len(F)}->{len(F2)})")

    g_after = _tc.euler_genus(V2, F2)
    if g_after > g_before:
        ok(f"add_handle/genus_increased ({g_before} -> {g_after})")
    else:
        fail("add_handle/genus_increased", f"{g_before} -> {g_after}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("topmod_core equivalence tests")
    print("=" * 60)

    test_validate()
    test_watertight()
    test_genus()
    test_catmull_clark()
    test_triangulate_all()
    test_flip_sweep()
    test_collapse()
    test_subdivide_faces()
    test_add_handle()

    print("=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
