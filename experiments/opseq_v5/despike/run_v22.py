"""Generic (any-shape) driver for the v22 needle-removal pipeline.
SHAPE env selects the GT mesh ({SHAPE}.obj in the pymeshlab sample dir).
Same algorithm as cow_v22.py: escape-seed + thin-propagate, 6 training views only.
Generic viz (GT vs v22 turntable) + generic held-out exam (loads {SHAPE}.obj).
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import cow_v13
cow_v13.TUBE_THR = float(os.environ.get("TUBE_THR", "0.4"))
SHAPE = os.environ.get("SHAPE", "cow")
TAG = os.environ.get("TAG", f"{SHAPE}_v22")
DILATE = int(os.environ.get("DILATE", "2"))
print(f"[run_v22] SHAPE={SHAPE} TAG={TAG} TUBE_THR={cow_v13.TUBE_THR}", flush=True)

import numpy as np, torch, time, collections
from scipy.ndimage import binary_dilation, label as cc_label
from cow_v13 import optimize_phase, midpoint_subdivide, DEVICE, OUT
from surgery_lib import surgery, _propagate_flag
from eval_local_refine import setup_scene, render_views_n, compute_iou_n, load_obj, normalize_to_range, BUNNY_PATH
from eval_extrude_v3 import orbit_cameras
from escape_util import escape_mask

_MVPS = None; _GT = None; _CUR_ADJ = None
_orig_tube_mask = cow_v13.tube_mask


def _set_faces(F):
    global _CUR_ADJ
    adj = collections.defaultdict(set)
    for a, b, c in np.asarray(F):
        a, b, c = int(a), int(b), int(c)
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    _CUR_ADJ = adj


@torch.no_grad()
def _tube_mask_gated(v, excl, mean_edge):
    thin = _orig_tube_mask(v, excl, mean_edge)
    if _MVPS is None or _CUR_ADJ is None:
        return thin
    esc = escape_mask(v, _MVPS, _GT, dilate=DILATE)
    keep = _propagate_flag(thin.cpu().numpy(), esc.cpu().numpy(), _CUR_ADJ)
    return torch.from_numpy(keep).to(v.device)


cow_v13.tube_mask = _tube_mask_gated


def heldout_exam(ctx, v, t):
    gt_v, gt_f = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gt_v = normalize_to_range(gt_v)
    gvt = torch.tensor(gt_v, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gt_f, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5, azimuths_deg=azs, device=DEVICE)
    if isinstance(mv, tuple): mv = mv[0]
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    gt_sils = render_views_n(ctx, gvt, gft, mv)
    pr_sils = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; maxblob = 0
    for i in range(16):
        g = gt_sils[i] > 0.5; p = pr_sils[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2); hair += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return inter / max(un, 1), hair, maxblob


def save_obj(path, v, t):
    with open(path, "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")


def main():
    global _MVPS, _GT
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene(SHAPE, DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    _MVPS, _GT = mvps, gt

    def iou_fn(v, f):
        vt = torch.tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(f, dtype=np.int32), dtype=torch.int32, device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    def escape_fn(Vnp):
        vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
        return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]; _set_faces(t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2", use_fold=True, use_tube=True)
    v, t = midpoint_subdivide(v, t); _set_faces(t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3", settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t); _set_faces(t)
    v, iou_train = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4", settle=True, use_fold=True, use_tube=True)
    print(f"[train] iou={iou_train:.4f} V={len(v)} ({time.time()-t0:.0f}s)", flush=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=3e-4, global_cap=1.5e-3, max_rounds=8, max_grow=4, escape_fn=escape_fn)
    _set_faces(t)
    v, _ = optimize_phase(ctx, v, np.asarray(t, np.int32), gt, gtd, mvps, 400, "settle", settle=True, use_fold=True, use_tube=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=2e-4, global_cap=6e-4, max_rounds=4, max_grow=4, escape_fn=escape_fn)
    iou_final = iou_fn(v, t); t = np.asarray(t, np.int32)
    save_obj(f"{OUT}/cow_{TAG}.obj", v, t)
    np.savez(f"{OUT}/cow_{TAG}.npz", verts=v, tris=t)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t)
    print(f"\n=== {TAG.upper()} RESULT ===")
    print(f"train6 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"V={len(v)} time={time.time()-t0:.0f}s", flush=True)
    # generic viz: GT vs v22 turntable
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj")); gv = normalize_to_range(gv)
    fig = plt.figure(figsize=(16, 4.5), dpi=110)
    for k in range(8):
        ax = fig.add_subplot(2, 8, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(np.asarray(gv)[np.asarray(gf)], facecolor="#888", edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=20, azim=45 * k); ax.axis("off")
        if k == 0: ax.text2D(-0.1, 0.5, "GT", transform=ax.transAxes, rotation=90, va="center")
        ax = fig.add_subplot(2, 8, k + 9, projection="3d")
        ax.add_collection3d(Poly3DCollection(v[t], facecolor="#5cb85c", edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=20, azim=45 * k); ax.axis("off")
        if k == 0: ax.text2D(-0.1, 0.5, "v22", transform=ax.transAxes, rotation=90, va="center")
    plt.tight_layout(); plt.savefig(f"{OUT}/cow_{TAG}_compare.png", bbox_inches="tight")
    print(f"saved cow_{TAG}_compare.png", flush=True)


if __name__ == "__main__":
    main()
