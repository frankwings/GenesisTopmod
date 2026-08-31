"""V23: v22 needle-removal + in-training NORMAL-CONSISTENCY smoothing.

Boss wants a DMesh-smooth surface. Diagnosis: the pipeline has NO surface-
smoothness term (only a weak uniform Laplacian W_LAP=0.1 and a tiny antiparallel
fold penalty). Per-pixel depth L1 (W_DEPTH=0.35) actively injects high-freq
facet noise on camera-facing surfaces. Post-hoc Taubin smooths but costs 2.5-4.4
IoU points (it ignores the silhouette).

Fix: add a normal-consistency loss  W_NC * mean(1 - n_i . n_j)  over adjacent
face pairs (pushes neighbours coplanar) DURING training, so silhouette+depth
keep it accurate while the term removes bumps -> smooth AND on-target.

Injection: monkeypatch cow_v13.edge_length_loss to append (W_NC/W_EDGE)*nc, so
the effective added weight is exactly W_NC (edge term is W_EDGE*edge_len).
Gated to V>2000 (cc4 + settle) so coarse fitting is untouched. The v22 escape-
seed + thin-propagate needle gates are kept unchanged.
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import cow_v13
cow_v13.TUBE_THR = float(os.environ.get("TUBE_THR", "0.4"))
W_NC = float(os.environ.get("W_NC", "0.10"))
NC_MIN_V = int(os.environ.get("NC_MIN_V", "2000"))
TAG = os.environ.get("TAG", "v23")
DILATE = int(os.environ.get("DILATE", "2"))
print(f"[v23] TUBE_THR={cow_v13.TUBE_THR} W_NC={W_NC} NC_MIN_V={NC_MIN_V}", flush=True)

import numpy as np, torch, time, collections
import torch.nn.functional as F
from cow_v13 import (optimize_phase, heldout_exam, midpoint_subdivide, DEVICE, OUT,
                     build_pairs)
from eval_local_refine import W_EDGE
from surgery_lib import surgery, _propagate_flag
from eval_local_refine import setup_scene, render_views_n, compute_iou_n, load_obj, normalize_to_range, BUNNY_PATH
from escape_util import escape_mask

# ---- normal-consistency patch on edge_length_loss ----
_orig_edge = cow_v13.edge_length_loss
_NC_CACHE = {}


def _pairs_for(faces_t):
    key = faces_t.data_ptr()
    p = _NC_CACHE.get(key)
    if p is None:
        p = torch.tensor(build_pairs(faces_t.detach().cpu().numpy()),
                         device=faces_t.device)
        _NC_CACHE.clear(); _NC_CACHE[key] = p
    return p


def _edge_with_nc(verts_t, faces_t):
    base = _orig_edge(verts_t, faces_t)
    if verts_t.shape[0] < NC_MIN_V:
        return base
    pairs = _pairs_for(faces_t)
    tri = verts_t[faces_t.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    d = (n[pairs[:, 0]] * n[pairs[:, 1]]).sum(-1)
    nc = (1.0 - d).mean()
    return base + (W_NC / max(W_EDGE, 1e-9)) * nc


cow_v13.edge_length_loss = _edge_with_nc

# ---- v22 escape-seed + thin-propagate tube-prior patch ----
_MVPS = None; _GT = None; _CUR_ADJ = None
_orig_tube_mask = cow_v13.tube_mask


def _set_faces(Fc):
    global _CUR_ADJ
    adj = collections.defaultdict(set)
    for a, b, c in np.asarray(Fc):
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


def save_obj(path, v, t):
    with open(path, "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")


def _shaded(ax, v, F_, az, color):
    tri = v[F_]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
    light = np.array([0.3, 0.5, 0.8]); light = light / np.linalg.norm(light)
    sh = np.clip(n @ light, 0.05, 1.0)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cols = np.stack([sh * color[0], sh * color[1], sh * color[2]], 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor="none"))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=15, azim=az); ax.axis("off")


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
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t, scene)
    print(f"\n=== {TAG.upper()} RESULT (W_NC={W_NC}) ===")
    print(f"train6 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v22 train6=0.9813 ho=0.9568/9100/721 (no NC)")
    print(f"time: {time.time()-t0:.0f}s", flush=True)
    # shaded comparison v22 vs v23
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z22 = np.load(f"{OUT}/cow_v22.npz"); v22, t22 = z22["verts"], z22["tris"]
    angles = [0, 90, 200]
    fig = plt.figure(figsize=(8, 4 * len(angles)), dpi=100)
    for r, az in enumerate(angles):
        ax = fig.add_subplot(len(angles), 2, r * 2 + 1, projection="3d")
        _shaded(ax, v22, t22, az, (0.55, 0.78, 0.42))
        if r == 0: ax.set_title("v22 (no NC)", fontsize=11)
        ax = fig.add_subplot(len(angles), 2, r * 2 + 2, projection="3d")
        _shaded(ax, v, t, az, (0.42, 0.68, 0.85))
        if r == 0: ax.set_title(f"{TAG} (W_NC={W_NC})", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/cow_{TAG}_smooth.png", bbox_inches="tight")
    print(f"saved cow_{TAG}_smooth.png", flush=True)


if __name__ == "__main__":
    main()
