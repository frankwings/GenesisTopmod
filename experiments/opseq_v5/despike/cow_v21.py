"""V21: GT-consistency needle removal (approach B, 6 TRAINING views only).

Boss's redirect: change the needle-removal LOGIC from geometric
("protrudes -> delete") to GT-consistency ("project the needle to the 6
training views, delete ONLY the part that lands OUTSIDE the GT silhouette").

Two escape gates, both fed by escape_util.escape_mask (project verts to the 6
training-view pixels, flag those outside dilate(GT_fg)):
  (a) TRAINING tube prior: tube_mask is monkeypatched so only thin verts that
      ALSO escape GT get pulled to centroid.  Because escape now supplies the
      discrimination, TUBE_THR goes back to 0.4 (blanket thin) -> the ear is
      thin but stays INSIDE GT, so it is spared; the needle pokes outside GT,
      so it is pulled.  v20 needed TUBE_THR=0.25 (a thickness proxy); v21
      replaces the proxy with the real GT test.
  (b) SURGERY detector: comps condemned only where (thin|spike) AND escape.

No held-out view is ever used for supervision.
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
sys.path.insert(0, "/tmp")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import cow_v13
cow_v13.TUBE_THR = float(os.environ.get("TUBE_THR", "0.4"))
print(f"[v21] TUBE_THR = {cow_v13.TUBE_THR} (escape-gated)", flush=True)

import numpy as np, torch, time
from cow_v13 import optimize_phase, heldout_exam, midpoint_subdivide, DEVICE, OUT
from surgery_lib import surgery
from eval_local_refine import setup_scene, render_views_n, compute_iou_n
from escape_util import escape_mask

TAG = os.environ.get("TAG", "v21")
DILATE = int(os.environ.get("DILATE", "2"))

# --- globals the patched tube_mask consults (set in main once scene is up) ---
_MVPS = None
_GT = None
_orig_tube_mask = cow_v13.tube_mask


@torch.no_grad()
def _tube_mask_gated(v, excl, mean_edge):
    """thin AND escape(GT).  v is a torch tensor [V,3] on DEVICE."""
    thin = _orig_tube_mask(v, excl, mean_edge)          # torch bool[V]
    if _MVPS is None:
        return thin
    esc = escape_mask(v, _MVPS, _GT, dilate=DILATE)      # torch bool[V]
    return thin & esc


cow_v13.tube_mask = _tube_mask_gated


def save_obj(path, v, t):
    with open(path, "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")


def main():
    global _MVPS, _GT
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    _MVPS, _GT = mvps, gt

    def iou_fn(v, f):
        vt = torch.tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(f, dtype=np.int32), dtype=torch.int32, device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    def escape_fn(Vnp):
        """surgery detector gate: numpy V -> numpy bool[V]."""
        vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
        return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2", use_fold=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3", settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, iou_train = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                                  settle=True, use_fold=True, use_tube=True)
    save_obj(f"{OUT}/cow_{TAG}_pre.obj", v, t)
    print(f"[train] iou={iou_train:.4f} V={len(v)} ({time.time()-t0:.0f}s)", flush=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=3e-4, global_cap=1.5e-3,
                   max_rounds=8, max_grow=4, escape_fn=escape_fn)
    print(f"[surgery] iou={iou_fn(v,t):.4f} V={len(v)}", flush=True)
    v, _ = optimize_phase(ctx, v, np.asarray(t, np.int32), gt, gtd, mvps, 400,
                          "settle", settle=True, use_fold=True, use_tube=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=2e-4, global_cap=6e-4,
                   max_rounds=4, max_grow=4, escape_fn=escape_fn)
    iou_final = iou_fn(v, t); t = np.asarray(t, np.int32)
    save_obj(f"{OUT}/cow_{TAG}.obj", v, t)
    np.savez(f"{OUT}/cow_{TAG}.npz", verts=v, tris=t)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t, scene)
    print(f"\n=== {TAG.upper()} RESULT (TUBE_THR={cow_v13.TUBE_THR}, escape-gated) ===")
    print(f"train6 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v20 train6=0.9789 ho=0.9438/9544 (thr0.25) | v17 ho=0.9477/11332")
    print(f"time: {time.time()-t0:.0f}s", flush=True)
    # visuals
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "cow.obj")); gv = normalize_to_range(gv)
    z20 = np.load(f"{OUT}/cow_v20.npz")
    cols = [("GT cow", gv, gf), ("v20 (thr0.25)", z20["verts"], z20["tris"]),
            (f"{TAG} (escape)", v, t)]
    fin = np.array([-0.056,0.639,0.49]); head = np.array([0.008,0.519,1.311])
    views = [("full", None,(0,0)), ("fin", fin,(0,0)), ("ear", head,(20,120)), ("horn", head,(60,90))]
    fig = plt.figure(figsize=(12,13), dpi=120)
    for j,(name,vv,ff) in enumerate(cols):
        for r,(lbl,cc,(elev,azim)) in enumerate(views):
            ax = fig.add_subplot(4,3,r*3+j+1, projection="3d")
            ax.add_collection3d(Poly3DCollection(np.asarray(vv)[np.asarray(ff)], facecolor="#5cb85c", edgecolor="none"))
            lo,hi = (np.array([-1,-1,-1]),np.array([1,1,1])) if cc is None else (cc-0.4,cc+0.4)
            ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])
            ax.set_box_aspect((2,2,2)); ax.view_init(elev=elev, azim=azim); ax.axis("off")
            if r==0: ax.set_title(name, fontsize=11)
            if j==0: ax.text2D(-0.1,0.5,lbl, transform=ax.transAxes, rotation=90, va="center", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/cow_{TAG}_compare.png", bbox_inches="tight")
    fig = plt.figure(figsize=(16,4.5), dpi=110)
    for k in range(16):
        ax = fig.add_subplot(2,8,k+1, projection="3d")
        ax.add_collection3d(Poly3DCollection(v[t], facecolor="#5cb85c", edgecolor="none"))
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.set_box_aspect((2,2,2)); ax.view_init(elev=20, azim=22.5*k); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/cow_{TAG}_360.png", bbox_inches="tight")
    print(f"saved cow_{TAG}_compare.png cow_{TAG}_360.png", flush=True)


if __name__ == "__main__":
    main()
