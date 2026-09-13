"""
Benchmark: topmod_core.self_intersecting_pairs vs Open3D at 25k/49k/99k faces.

Acceptance: C++ >= 50x faster than Open3D at 99k faces.

Run from the repo root:
    python3 topmod/cpp/bench_si.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import topmod_core as tc
import topmod
from topmod.io import to_triangle_arrays
from topmod.subdivision import catmull_clark as py_cc

_REPO    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_RESULTS = os.path.join(_REPO, 'experiments', 'opseq_v5', 'despike', 'results_genus')


def timer(fn, *args, **kwargs):
    t0 = time.perf_counter()
    r  = fn(*args, **kwargs)
    return r, (time.perf_counter() - t0) * 1000


def fmt(ms):
    return f"{ms:.1f} ms" if ms < 1000 else f"{ms/1000:.2f} s"


def speedup_str(py_ms, cpp_ms):
    if cpp_ms < 0.01:
        return "inf"
    return f"{py_ms/cpp_ms:.1f}x"


def open3d_si(V, F):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V.astype(np.float64)),
        o3d.utility.Vector3iVector(F.astype(np.int32)))
    return np.asarray(om.get_self_intersecting_triangles())


def cpp_si(V, F):
    return tc.self_intersecting_pairs(
        np.asarray(V, np.float64), np.asarray(F, np.int64))


def bench(name, V, F):
    print(f"\n  {name}  V={len(V):,} F={len(F):,}")
    py_pairs,  py_ms  = timer(open3d_si, V, F)
    cpp_pairs, cpp_ms = timer(cpp_si,    V, F)
    n_py  = len(py_pairs)
    n_cpp = len(cpp_pairs)
    match = "✓" if n_py == n_cpp else "✗"
    sp = py_ms / max(cpp_ms, 0.01)
    print(f"  Open3D  : {fmt(py_ms):>10}  pairs={n_py}")
    print(f"  C++     : {fmt(cpp_ms):>10}  pairs={n_cpp}  {match}")
    print(f"  Speedup : {sp:.1f}x")
    return sp, len(F)


# ── Build meshes ───────────────────────────────────────────────────────────────

print("Building test meshes ...")

# ~25k face mesh: 4×CC icosphere → 15360F, then a small npz if available
m = topmod.make_icosahedron()
for _ in range(3):
    m = py_cc(m)
from topmod.io import to_triangle_arrays
vv, ff = to_triangle_arrays(m)
V_25k = np.array(vv, float)
F_25k = np.array(ff, np.int64)
print(f"  CC3 icosphere : V={len(V_25k):,} F={len(F_25k):,}")

# ~49k face mesh: armadillo
armadillo_path = os.path.join(_RESULTS, 'armadillo_g3chain_raw.npz')
if os.path.exists(armadillo_path):
    npz = np.load(armadillo_path)
    V_49k = npz['verts'].astype(np.float64)
    F_49k = npz['tris'].astype(np.int64)
    print(f"  armadillo     : V={len(V_49k):,} F={len(F_49k):,}")
else:
    # Fallback: 4×CC icosphere
    m4 = py_cc(m)
    vv4, ff4 = to_triangle_arrays(m4)
    V_49k = np.array(vv4, float); F_49k = np.array(ff4, np.int64)
    print(f"  CC4 icosphere : V={len(V_49k):,} F={len(F_49k):,}")

# ~99k face mesh: fertility (has real SI pairs)
fertility_path = os.path.join(_RESULTS, 'fertility_g3ccchain_raw.npz')
if os.path.exists(fertility_path):
    npz = np.load(fertility_path)
    V_99k = npz['verts'].astype(np.float64)
    F_99k = npz['tris'].astype(np.int64)
    print(f"  fertility     : V={len(V_99k):,} F={len(F_99k):,}")
else:
    V_99k = None

# Perturbed armadillo (creates real SI pairs on 49k mesh)
rng = np.random.default_rng(42)
el  = np.linalg.norm(V_49k[F_49k[:,0]] - V_49k[F_49k[:,1]], axis=1).mean()
V_noisy = V_49k + rng.normal(0, 0.02 * el, V_49k.shape)

# ── Run benchmarks ─────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("Benchmark: topmod_core.self_intersecting_pairs vs Open3D")
print("=" * 65)

speedups = {}

sp, nf = bench("CC3 icosphere (clean, ~15k F)", V_25k, F_25k)
speedups[nf] = sp

sp, nf = bench("armadillo (clean, ~49k F)", V_49k, F_49k)
speedups[nf] = sp

sp, nf = bench("perturbed armadillo (SI pairs, ~49k F)", V_noisy, F_49k)
speedups[nf] = sp

if V_99k is not None:
    sp, nf = bench("fertility (SI pairs, ~99k F)", V_99k, F_99k)
    speedups[nf] = sp
    sp_99k = sp
else:
    sp_99k = None
    print("\n  [SKIP] 99k mesh not available")

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
for nf, sp in sorted(speedups.items()):
    print(f"  {nf:>7,} faces : {sp:.1f}x speedup")

if sp_99k is not None:
    ok = sp_99k >= 50
    print(f"\n  Acceptance (>= 50x at ~99k faces): {'PASS ✓' if ok else 'FAIL ✗'}  ({sp_99k:.1f}x)")
    if not ok:
        sys.exit(1)
else:
    # Use 49k noisy mesh as proxy
    sp_proxy = speedups.get(len(F_49k))
    if sp_proxy:
        ok = sp_proxy >= 50
        print(f"\n  Acceptance proxy (>= 50x at 49k noisy): {'PASS ✓' if ok else 'FAIL ✗'}  ({sp_proxy:.1f}x)")
