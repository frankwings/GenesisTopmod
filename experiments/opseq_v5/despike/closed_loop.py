"""Closed-loop iterations: manifold_k -> DMesh init -> soup_k+1 -> pull -> manifold_k+1.
Usage: python3 despike/closed_loop.py <start_iter> <end_iter> <start_manifold_npz>
"""
import sys, os, subprocess, numpy as np
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
from eval_dmesh import load_any_mesh, similarity_from_bbox

DM = "/home/kingy/Projects/Genesis/GenesisExp/GenesisDMesh"
OUT = "/tmp/liou_cow_viz"
start, end, cur = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
our_gt_v, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "armadillo.obj"))
our_gt_v = normalize_to_range(our_gt_v)
dgt_v, _ = load_any_mesh(f"{DM}/exp_result/exp_3/armadillo_6v/2026_08_31_18_11_33/gt_mesh.obj")
scale, off = similarity_from_bbox(dgt_v, our_gt_v)
base_yaml = open(f"{DM}/exp/config/exp_3/armadillo_6v_ourinit.yaml").read()

for it in range(start, end + 1):
    print(f"\n########## LOOP {it}: init from {cur}", flush=True)
    d = np.load(cur); V, Fa = d["verts"], d["tris"]
    Vd = (V - off) / scale
    init_obj = f"{DM}/dataset/test/armadillo_ourinit6v_it{it}.obj"
    with open(init_obj, "w") as fh:
        for x, y, z in Vd: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
    yaml = (base_yaml.replace("armadillo_ourinit6v.obj", f"armadillo_ourinit6v_it{it}.obj")
                     .replace("log_dir: exp_result/exp_3/armadillo_6v_ourinit",
                              f"log_dir: exp_result/exp_3/armadillo_6v_ourinit_it{it}"))
    ycfg = f"{DM}/exp/config/exp_3/armadillo_6v_ourinit_it{it}.yaml"
    open(ycfg, "w").write(yaml)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{DM}/external/oneTBB/install/lib",
               PYTHONPATH=DM, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    r = subprocess.run([f"{DM}/.venv/bin/python", "exp/3_mv_recon.py",
                        f"--config=exp/config/exp_3/armadillo_6v_ourinit_it{it}.yaml",
                        "--no-log-time"], cwd=DM, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"DMesh failed at loop {it}"
    D = f"{DM}/exp_result/exp_3/armadillo_6v_ourinit_it{it}"
    ev = subprocess.run(["python3", "despike/eval_dmesh.py", "--pred",
                         f"{D}/save/epoch_3/phase_1/last/mesh.obj", "--dmesh-gt",
                         f"{D}/gt_mesh.obj", "--shape", "armadillo", "--tag",
                         f"dmesh6_it{it}"], capture_output=True, text=True)
    soup_line = [l for l in ev.stdout.splitlines() if "heldout16" in l]
    print(f"[loop {it}] DMesh soup: {soup_line[-1] if soup_line else ev.stdout[-300:]}", flush=True)
    env2 = dict(os.environ, MODE="6v", TAG=f"p2a6_it{it}", BASE_NPZ=cur,
                TARGET_OBJ=f"{OUT}/dmesh6_it{it}_aligned.obj")
    r2 = subprocess.run(["python3", "despike/phase2a_reverse.py"], env=env2,
                        capture_output=True, text=True)
    fin = [l for l in r2.stdout.splitlines() if "FINAL" in l or "delta" in l]
    print(f"[loop {it}] manifold: " + " | ".join(fin), flush=True)
    if r2.returncode != 0: print(r2.stdout[-2000:], r2.stderr[-2000:]); break
    cur = f"{OUT}/cow_armadillo_p2a6_it{it}_best.npz"
print("\n########## CLOSED LOOP DONE", flush=True)
