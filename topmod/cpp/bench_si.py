"""
Benchmark: topmod_core.self_intersecting_pairs vs Open3D at ~25k/49k/99k faces.

Uses real archived meshes and subsampled versions for consistent size tiers.
Perturbed copies inject real SI pairs so both broad+narrow phases are exercised.

Acceptance: C++ >= 50x faster than Open3D at ~99k faces.

Run from the repo root:
    python3 topmod/cpp/bench_si.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import topmod_core as tc

_REPO    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_RESULTS = os.path.join(_REPO, 'experiments', 'opseq_v5', 'despike', 'results_genus')


def timer(fn, *args, **kwargs):
    t0 = time.perf_counter()
    r  = fn(*args, **kwargs)
    return r, (time.perf_counter() - t0) * 1000


def fmt(ms):
    return f"{ms:.1f} ms" if ms < 1000 else f"{ms/1000:.2f} s"


def open3d_si(V, F):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V.astype(np.float64)),
        o3d.utility.Vector3iVector(F.astype(np.int32)))
    return np.asarray(om.get_self_intersecting_triangles())


def cpp_si(V, F):
    return tc.self_intersecting_pairs(
        np.asarray(V, np.float64), np.asarray(F, np.int64))


def subsample_mesh(V, F, target_nf):
    """Take first target_nf faces; remap vertices to compact range."""
    F_sub = F[:target_nf].copy()
    used = np.unique(F_sub)
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V_sub = V[used].copy()
    F_sub = remap[F_sub]
    return V_sub, F_sub


def bench(name, V, F):
    """Run both engines, compare pair counts, report timing."""
    print(f"\n  {name}  V={len(V):,} F={len(F):,}")

    cpp_pairs, cpp_ms = timer(cpp_si, V, F)
    py_pairs,  py_ms  = timer(open3d_si, V, F)

    n_cpp = len(cpp_pairs)
    n_py  = len(py_pairs)
    match = "✓" if n_py == n_cpp else "✗"
    sp = py_ms / max(cpp_ms, 0.01)

    print(f"  C++     : {fmt(cpp_ms):>10}  pairs={n_cpp}")
    print(f"  Open3D  : {fmt(py_ms):>10}  pairs={n_py}  {match}")
    print(f"  Speedup : {sp:.1f}x")
    return sp, len(F), n_cpp


# ── Load meshes ───────────────────────────────────────────────────────────────

print("Loading test meshes ...")

arm_path  = os.path.join(_RESULTS, 'armadillo_g3chain_raw.npz')
fert_path = os.path.join(_RESULTS, 'fertility_g3ccchain_raw.npz')

if not os.path.exists(arm_path):
    print(f"[ERROR] Armadillo not found: {arm_path}"); sys.exit(1)
if not os.path.exists(fert_path):
    print(f"[ERROR] Fertility not found: {fert_path}"); sys.exit(1)

npz_arm  = np.load(arm_path)
V_arm = npz_arm['verts'].astype(np.float64)
F_arm = npz_arm['tris'].astype(np.int64)

npz_fert = np.load(fert_path)
V_fert = npz_fert['verts'].astype(np.float64)
F_fert = npz_fert['tris'].astype(np.int64)

# Perturbed armadillo (0.20 * mean_edge → ~192 SI pairs at 99k faces)
rng = np.random.default_rng(42)
el  = np.linalg.norm(V_arm[F_arm[:,0]] - V_arm[F_arm[:,1]], axis=1).mean()
V_arm_noisy = V_arm + rng.normal(0, 0.20 * el, V_arm.shape)

# Create size tiers by subsampling armadillo (shuffled for spatial spread)
rng_idx = np.random.default_rng(0)
perm = rng_idx.permutation(len(F_arm))
F_arm_shuffled = F_arm[perm]

V_25k, F_25k = subsample_mesh(V_arm_noisy, F_arm_shuffled, 25000)
V_49k, F_49k = subsample_mesh(V_arm_noisy, F_arm_shuffled, 49000)
# 99k tier: use full perturbed armadillo (98904 faces)

print(f"  armadillo     : V={len(V_arm):,}  F={len(F_arm):,}")
print(f"  fertility     : V={len(V_fert):,} F={len(F_fert):,}")
print(f"  perturbed arm : noise = 0.20 * {el:.5f} = {0.20*el:.5f}")
print(f"  25k tier      : V={len(V_25k):,}  F={len(F_25k):,}")
print(f"  49k tier      : V={len(V_49k):,}  F={len(F_49k):,}")
print(f"  99k tier      : V={len(V_arm):,}  F={len(F_arm):,}  (full perturbed)")

# ── Run benchmarks ─────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("Benchmark: topmod_core.self_intersecting_pairs vs Open3D")
print("=" * 70)

results = {}

sp, nf, n = bench("~25k faces (perturbed armadillo subset)", V_25k, F_25k)
results['25k'] = (sp, nf, n)

sp, nf, n = bench("~49k faces (perturbed armadillo subset)", V_49k, F_49k)
results['49k'] = (sp, nf, n)

sp, nf, n = bench("~99k faces (perturbed armadillo full)", V_arm_noisy, F_arm)
results['99k_pert'] = (sp, nf, n)

sp, nf, n = bench("~99k faces (fertility, real SI)", V_fert, F_fert)
results['99k_fert'] = (sp, nf, n)

sp, nf, n = bench("~99k faces (clean armadillo, no SI)", V_arm, F_arm)
results['99k_clean'] = (sp, nf, n)

# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for label, (sp, nf, n) in sorted(results.items()):
    print(f"  {label:>12}: {nf:>7,} faces, {n:>6} SI pairs, {sp:>7.1f}x speedup")

# Acceptance check: >= 50x at 99k faces
sp_99k = max(results['99k_pert'][0], results['99k_fert'][0])
ok = sp_99k >= 50
print(f"\n  Acceptance (>= 50x at ~99k faces): {'PASS ✓' if ok else 'FAIL ✗'}  "
      f"(best 99k: {sp_99k:.1f}x)")
if not ok:
    sys.exit(1)
