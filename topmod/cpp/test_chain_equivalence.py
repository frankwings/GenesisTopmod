"""Ordered equivalence of the chain wrappers (dlfl_untangle / phase1c) between DLFL_BACKEND=py and =cpp:
a 5-op sequence (flip x3, collapse, subdivide 1/step faces, flip x2, collapse 20%) must give bit-identical
V and F (same ORDER, not just the same set) on a perturbed cc3 sphere and the 49k-V archived armadillo."""
import os, sys, numpy as np, time
R = "/home/kingy/Projects/Genesis/GenesisTopmod"; D = f"{R}/experiments/opseq_v5/despike"
sys.path[:0] = [R, D, f"{R}/experiments/opseq_v5"]
os.chdir(D)
import cc_subdiv, dlfl_untangle as du, phase1c_pipeline as p1c

def run(be, V, F, step):
    os.environ["DLFL_BACKEND"] = be
    V1, F1, nf = du.flip_sweep(V, F, passes=3); V2, F2, nc = du.collapse_short_edges(V1, F1, 0.5, 400)
    V3, F3, ns = p1c.dlfl_subdivide_arrays(V2, F2, list(range(0, len(F2), step)), expand_ring=False)
    V4, F4, nf2 = du.flip_sweep(V3, F3, passes=2); V5, F5, nc2 = du.collapse_short_edges(V4, F4, 0.4, int(0.2 * len(F4)))
    return (nf, nc, ns, nf2, nc2), V5, F5

def same(a, b): return a[0] == b[0] and a[1].shape == b[1].shape and np.allclose(a[1], b[1]) and np.array_equal(a[2], b[2])

rng = np.random.default_rng(0); fails = 0
v, p, t = cc_subdiv.icosphere_cc2(); v2, p2, t2 = cc_subdiv.cc_subdivide(v, p)
cases = [("cc3_perturbed", v2 + 0.02 * rng.standard_normal(v2.shape), np.asarray(t2, np.int64), 7)]
z = np.load(f"{D}/results_genus/armadillo_g3chain_raw.npz"); Va, Fa = z["verts"], z["tris"].astype(np.int64)
cases.append(("armadillo49k_perturbed", Va + 0.003 * rng.standard_normal(Va.shape), Fa, 20))
for name, V, F, step in cases:
    t0 = time.time(); a = run("py", V, F, step); tp = time.time() - t0
    t0 = time.time(); b = run("cpp", V, F, step); tc = time.time() - t0
    ok = same(a, b); fails += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: counts {a[0]} vs {b[0]} | py {tp:.1f}s cpp {tc:.2f}s ({tp/max(tc,1e-9):.0f}x)")
print("ALL PASS" if fails == 0 else f"{fails} FAILED"); sys.exit(fails)
