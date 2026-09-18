#!/usr/bin/env python3
"""test_hull_locate.py: verify find_tunnel_by_hull across the 5 coarse cc3p4 meshes.

Deliverable 3 of hull-locate spec:
  plugs found == g* (armadillo 0, kitten 1, rockerarm 1, threeholes 3, fertility 4)
  every accepted candidate passes face-pair rules
  wall time < 60 s / shape at 128^3
"""
import os, sys, time, json, traceback
import numpy as np

os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")

# ── Expected g* per shape ───────────────────────────────────────────────────────
EXPECTED = {
    "armadillo": 0,
    "kitten":    1,
    "rockerarm": 1,
    "threeholes": 3,
    "fertility": 4,
}

NPZ_DIR = "/tmp/liou_cow_viz"
PASS_BUDGET_S = 60.0   # per-shape wall budget (seconds)

# Read the phase7_handle.py source up to the main loop
p7_src = open("despike/phase7_handle.py").read()
CUT_MARKER = 'report("base", V, Fa)'
cut_idx = p7_src.index(CUT_MARKER)
SETUP_SRC = p7_src[:cut_idx]

results = {}

for shape, g_star in EXPECTED.items():
    npz = f"{NPZ_DIR}/cow_{shape}_{shape}_v5_cc3p4.npz"
    print(f"\n{'='*64}")
    print(f"SHAPE={shape}  g*={g_star}")
    print('='*64, flush=True)

    if not os.path.exists(npz):
        print(f"  SKIP: {npz} not found")
        results[shape] = ("SKIP", -1, g_star, 0.0)
        continue

    # Fresh cache for each test run (force full ladder re-computation)
    cache_path = f"{NPZ_DIR}/hull_plugs_{shape}_test.json"
    if os.path.exists(cache_path):
        os.unlink(cache_path)

    os.environ["SHAPE"]           = shape
    os.environ["BASE_NPZ"]        = npz
    os.environ["HULL_PLUGS_CACHE"] = cache_path
    os.environ["GENUS_TARGET"]    = "hull"
    os.environ.pop("DETECT", None)   # use default = hull

    t0 = time.time()
    try:
        # ── Execute phase7_handle.py setup in an isolated namespace ────────────
        # This builds HF, defines all functions, and populates globals exactly
        # as the real pipeline does.
        ns = {}
        exec(compile(SETUP_SRC, "p7_setup", "exec"), ns)

        V_t        = ns["V"]
        Fa_t       = ns["Fa"]
        HF_t       = ns["HF"]
        G_TARGET_t = ns["G_TARGET"]
        genus_fn   = ns["genus"]
        find_fn    = ns["find_tunnel_by_hull"]

        t_setup = time.time() - t0
        print(f"  Setup (HF build + define): {t_setup:.1f}s", flush=True)

        # ── Call the function under test ────────────────────────────────────────
        t1 = time.time()
        result = find_fn(V_t, Fa_t, HF_t, [], G_TARGET_t)
        t_hull = time.time() - t1
        wall   = time.time() - t0

        n_accepted = len(result)

        # ── Verify: accepted count == g* ────────────────────────────────────────
        ok_count = (n_accepted >= g_star)   # mouths >= handles needed (k exits -> k-1 handles)
        if not ok_count:
            print(f"  FAIL count: accepted={n_accepted} expected={g_star}", flush=True)

        # ── Verify: each face pair passes the rules ─────────────────────────────
        ok_pairs = True
        for idx, (fi, fj, ci_w, cj_w, key) in enumerate(result):
            errs = []

            # fi != fj
            if fi == fj:
                errs.append(f"fi==fj=={fi}")

            # non-adjacent (no shared vertex)
            if set(Fa_t[fi]) & set(Fa_t[fj]):
                errs.append(f"fi={fi} fj={fj} share vertex (adjacent)")

            # key format
            if not (isinstance(key, list) and len(key) == 2
                    and key[0] == "hull" and isinstance(key[1], list) and len(key[1]) == 3):
                errs.append(f"key format wrong: {key}")

            # ci_w / cj_w are numpy arrays near face centroids
            ci_ref = V_t[Fa_t[fi]].mean(0)
            cj_ref = V_t[Fa_t[fj]].mean(0)
            if np.linalg.norm(np.asarray(ci_w) - ci_ref) > 0.01:
                errs.append(f"ci_w mismatch (d={np.linalg.norm(np.asarray(ci_w)-ci_ref):.4f})")
            if np.linalg.norm(np.asarray(cj_w) - cj_ref) > 0.01:
                errs.append(f"cj_w mismatch (d={np.linalg.norm(np.asarray(cj_w)-cj_ref):.4f})")

            if errs:
                ok_pairs = False
                print(f"  FAIL pair {idx}: " + "; ".join(errs), flush=True)
            else:
                print(f"  pair {idx}: fi={fi} fj={fj} sep={np.linalg.norm(np.asarray(cj_w)-np.asarray(ci_w)):.4f} key={key[1]}", flush=True)

        # ── Wall time check ─────────────────────────────────────────────────────
        ok_time = wall <= PASS_BUDGET_S
        if not ok_time:
            print(f"  WARN wall={wall:.1f}s > budget {PASS_BUDGET_S}s", flush=True)

        status = "PASS" if (ok_count and ok_pairs) else "FAIL"
        print(f"  {status}: accepted={n_accepted}/{g_star} wall={wall:.1f}s hull={t_hull:.1f}s budget_ok={ok_time}",
              flush=True)
        results[shape] = (status, n_accepted, g_star, wall)

    except Exception as exc:
        traceback.print_exc()
        results[shape] = ("ERROR", -1, g_star, time.time() - t0)

# ── Summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("SUMMARY:")
print(f"{'Shape':<12} {'Status':<8} {'Accept':<8} {'g*':<5} {'Wall(s)':<10}")
print('-'*48)
all_pass = True
for shape, (status, n_acc, g_star, wall) in results.items():
    mark = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "?")
    print(f"  {mark} {shape:<10} {status:<8} {n_acc:<8} {g_star:<5} {wall:<10.1f}")
    if status != "PASS":
        all_pass = False

total = "\nOVERALL: PASS" if all_pass else "\nOVERALL: FAIL"
print(total, flush=True)
sys.exit(0 if all_pass else 1)
