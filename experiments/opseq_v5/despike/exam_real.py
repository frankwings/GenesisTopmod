"""Real-data exam: heldout silhouette IoU on the 16 held-out views.

Usage: SHAPE=dino python3 despike/exam_real.py raw=results_genus/dino_real1_raw.npz taubin=results_genus/dino_real1_auto.npz

Prints: [exam_real] <tag> V= F= watertight genus ho16=<iou> hair= maxblob=
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np
import nvdiffrast.torch as dr
from phase1b_pipeline import check_watertight

REAL_DATA = os.environ.get("REAL_DATA", "")
SHAPE = os.environ.get("SHAPE", "dino")


def genus(V, F):
    F = np.asarray(F, np.int64)
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2


def exam_mesh(tag, npz_path, scene, ctx):
    z = np.load(npz_path)
    V, F = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
    wt, _ = check_watertight(F)
    g = genus(V, F)
    ho, hair, maxblob = scene.heldout_exam(ctx, V, F)
    print(f"[exam_real] {tag} V={len(V)} F={len(F)} watertight={wt} genus={g} "
          f"ho16={ho:.4f} hair={hair} maxblob={maxblob}", flush=True)
    return ho


if __name__ == "__main__":
    args = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            args[k] = v

    if not REAL_DATA:
        print("[exam_real] ERROR: REAL_DATA env not set", file=sys.stderr)
        sys.exit(1)

    from real_scene import load_real_scene
    from cow_v13 import DEVICE
    scene = load_real_scene(REAL_DATA, DEVICE)
    ctx = dr.RasterizeCudaContext()

    for tag in ("raw", "taubin"):
        if tag in args:
            exam_mesh(f"{SHAPE}_{tag}", args[tag], scene, ctx)
