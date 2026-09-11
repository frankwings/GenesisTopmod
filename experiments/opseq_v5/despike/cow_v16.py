"""V16: full pipeline = v13 training (6-view-only, online priors)
  -> TopMod amputation surgery (thin|spike comps, ring-grown whole-fin
     deletion, per-comp train6-IoU budget + global cap)
  -> settle re-fit (6 views, warmup + lap boost + depth phase-in)
  -> surgery pass 2 (catch regrowth)
All supervision = the original 6 training views. Held-out 16 views = exam only.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
from cow_v13 import (optimize_phase, heldout_exam, midpoint_subdivide,
                     DEVICE, OUT)
from surgery_lib import surgery
from eval_local_refine import setup_scene, render_views_n, compute_iou_n


def save_obj(path, v, t):
    with open(path, "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]

    def iou_fn(v, f):
        vt = torch.tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(f, dtype=np.int32), dtype=torch.int32,
                          device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2")
    v, t = midpoint_subdivide(v, t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3",
                          settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, iou_train = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                                  settle=True, use_fold=True, use_tube=True)
    save_obj(f"{OUT}/cow_v16_pre.obj", v, t)
    print(f"[train] done iou={iou_train:.4f} V={len(v)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[surgery pass 1]", flush=True)
    v, t = surgery(np.asarray(v, dtype=np.float64),
                   np.asarray(t, dtype=np.int64),
                   iou_fn=iou_fn, iou_budget=8e-4, global_cap=3e-3,
                   max_rounds=10, max_grow=4)
    print(f"[surgery1] done iou={iou_fn(v, t):.4f} V={len(v)}", flush=True)

    print("[settle re-fit]", flush=True)
    v, iou_settle = optimize_phase(ctx, v, np.asarray(t, dtype=np.int32),
                                   gt, gtd, mvps, 400, "settle",
                                   settle=True, use_fold=True, use_tube=True)
    print(f"[settle] done iou={iou_settle:.4f}", flush=True)

    print("[surgery pass 2]", flush=True)
    v, t = surgery(np.asarray(v, dtype=np.float64),
                   np.asarray(t, dtype=np.int64),
                   iou_fn=iou_fn, iou_budget=8e-4, global_cap=1.5e-3,
                   max_rounds=6, max_grow=4)
    iou_final = iou_fn(v, t)
    print(f"[surgery2] done iou={iou_final:.4f} V={len(v)}", flush=True)

    t = np.asarray(t, dtype=np.int32)
    save_obj(f"{OUT}/cow_v16.obj", v, t)
    np.savez(f"{OUT}/cow_v16.npz", verts=v, tris=t)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t, scene)
    print(f"\n=== V16 RESULT (train -> surgery -> settle -> surgery) ===")
    print(f"train6 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v13 train6=0.9779 ho=0.8981 | v15 train6=0.9736 ho=0.9046")
    print(f"time: {time.time()-t0:.0f}s", flush=True)

    # visuals: GT vs pre-surgery vs final, full + needle-site zoom
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "cow.obj"))
    gv = normalize_to_range(gv)
    pv, pf = [], []
    for line in open(f"{OUT}/cow_v16_pre.obj"):
        p = line.split()
        if not p: continue
        if p[0] == "v": pv.append([float(x) for x in p[1:4]])
        elif p[0] == "f": pf.append([int(x)-1 for x in p[1:4]])
    pv, pf = np.array(pv), np.array(pf)
    c = np.array([-0.056, 0.639, 0.49])
    fig = plt.figure(figsize=(13, 7), dpi=120)
    cols = [("GT cow", gv, gf), ("v16 pre-surgery", pv, pf),
            ("v16 final", v, t)]
    for j, (name, vv, ff) in enumerate(cols):
        for r, (elev, azim, zoom) in enumerate(
                [(0, 0, None), (0, 0, 0.35), (30, -60, 0.35)]):
            ax = fig.add_subplot(3, 3, r*3+j+1, projection="3d")
            ax.add_collection3d(Poly3DCollection(
                vv[ff], facecolor="#5cb85c", edgecolor="none"))
            lo = np.array([-1, -1, -1]) if zoom is None else c - zoom
            hi = np.array([1, 1, 1]) if zoom is None else c + zoom
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=elev, azim=azim)
            ax.axis("off")
            if r == 0: ax.set_title(name, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cow_v16_compare.png", bbox_inches="tight")
    # 16-angle turntable of final
    fig = plt.figure(figsize=(16, 4.5), dpi=110)
    for k in range(16):
        ax = fig.add_subplot(2, 8, k+1, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            v[t], facecolor="#5cb85c", edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2))
        ax.view_init(elev=20, azim=22.5 * k); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cow_v16_360.png", bbox_inches="tight")
    print("saved cow_v16_compare.png, cow_v16_360.png", flush=True)


if __name__ == "__main__":
    main()
