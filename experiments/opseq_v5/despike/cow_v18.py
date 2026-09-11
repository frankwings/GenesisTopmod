"""V18: v17 + DATA-AWARE tube prior ("save the ears, kill the needles").
Every TUBE_EVERY steps the thin-tube mask is split into connected comps and
each comp is verified by a 6-view trial render with the comp blended to its
neighbor centroid:
  - IoU unchanged  -> zombie structure (fin/needle): keep collapse pull
  - IoU drops      -> silhouette-supported (ear/horn): EXEMPT from pull
Surgery passes are also gentler (tighter budgets) and the final surgery runs
AFTER settle.  All supervision = the original 6 training views.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import torch.nn.functional as Fn
from cow_v13 import (
    midpoint_subdivide, build_pairs, fold_loss, build_adj, centroid,
    spike_pen, sliver_pen, tube_mask, mean_edge_of, heldout_exam,
    DEVICE, OUT, WARMUP_STEPS, LAP_WARMUP_STEPS, DEPTH_WARMUP_STEPS,
    W_LAP_BOOST, W_FOLD, FOLD_START, W_SPIKE, W_SLIVER, W_TUBE,
    TUBE_EVERY,
)
from surgery_lib import surgery
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
)

ZOMBIE_TOL = 5e-5   # trial-render IoU drop below this => zombie comp


@torch.no_grad()
def gate_zombie(tmask, verts, faces_t, src, dst, deg, ctx, mvps, gt, adj_np):
    """Exempt silhouette-supported comps from the tube pull (6 views only)."""
    flagged = tmask.cpu().numpy()
    if not flagged.any():
        return tmask
    iou_cur = compute_iou_n(render_views_n(ctx, verts, faces_t, mvps), gt)
    seen = np.zeros(len(flagged), bool)
    keep = torch.zeros_like(tmask)
    for i in np.where(flagged)[0]:
        if seen[i]: continue
        stack = [i]; seen[i] = True; comp = [i]
        while stack:
            u = stack.pop()
            for w in adj_np[u]:
                if flagged[w] and not seen[w]:
                    seen[w] = True; comp.append(w); stack.append(w)
        vt = verts.clone()
        idx = torch.tensor(comp, device=verts.device)
        for _ in range(4):
            cen = centroid(vt, src, dst, deg)
            vt[idx] = 0.3 * vt[idx] + 0.7 * cen[idx]
        iou_trial = compute_iou_n(render_views_n(ctx, vt, faces_t, mvps), gt)
        if iou_cur - iou_trial < ZOMBIE_TOL:
            keep[idx] = True          # zombie: pull it in
    return keep


def adj_list(tris_np, nv):
    import collections
    adj = collections.defaultdict(list)
    es = set()
    for a, b, c in tris_np:
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            if k in es: continue
            es.add(k)
            adj[e[0]].append(e[1]); adj[e[1]].append(e[0])
    return adj


def optimize_phase(ctx, verts_np, tris_np, gt, gtd, mvps, steps, label,
                   settle=False, use_fold=False, use_tube=False):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    src, dst, deg, excl = build_adj(tris_np, len(verts_np))
    adj_np = adj_list(tris_np, len(verts_np))
    pairs_t = (torch.tensor(build_pairs(tris_np), device=DEVICE)
               if use_fold else None)
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_MIN)
    warmup = WARMUP_STEPS if settle else 0
    lap_boost = LAP_WARMUP_STEPS if settle else 0
    depth_warm = DEPTH_WARMUP_STEPS if settle else 0
    tmask = None
    for step in range(steps):
        if warmup > 0:
            frac = 1.0 - warmup / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup -= 1
        w_lap = W_LAP_BOOST if lap_boost > 0 else W_LAP
        dw = W_DEPTH * (1.0 - depth_warm / DEPTH_WARMUP_STEPS) if depth_warm > 0 else W_DEPTH
        if lap_boost > 0: lap_boost -= 1
        if depth_warm > 0: depth_warm -= 1
        me = mean_edge_of(verts_t.detach(), src, dst)
        if use_tube and step % TUBE_EVERY == 0:
            raw = tube_mask(verts_t.detach(), excl, me)
            tmask = gate_zombie(raw, verts_t.detach(), faces_t, src, dst,
                                deg, ctx, mvps, gt, adj_np)
        opt.zero_grad()
        sl = torch.tensor(0.0, device=DEVICE)
        dl = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i],
                                                  (IMG_RES, IMG_RES))
            sl += Fn.l1_loss(sil[0], targets[i])
            dl += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sl /= N_VIEWS; dl /= N_VIEWS
        loss = (sl + dw * dl + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t)
                + W_SPIKE * spike_pen(verts_t, src, dst, deg, me)
                + W_SLIVER * sliver_pen(verts_t, faces_l, me))
        if pairs_t is not None and step >= FOLD_START:
            ramp = min(1.0, (step - FOLD_START) / max(1, (steps - FOLD_START) * 0.5))
            loss = loss + W_FOLD * ramp * fold_loss(verts_t, faces_l, pairs_t)
        if tmask is not None and tmask.any():
            cen = centroid(verts_t, src, dst, deg)
            loss = loss + W_TUBE * (verts_t[tmask] - cen[tmask].detach()
                                    ).norm(dim=-1).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
            nt = int(tmask.sum()) if tmask is not None else -1
            print(f"  [{label}] {step:4d}/{steps} iou={iou:.4f} zombie_v={nt}",
                  flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    v_out = verts_t.detach()
    print(f"  [{label}] END iou={iou:.4f}", flush=True)
    return v_out.cpu().numpy().astype(np.float64), iou


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
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2",
                          use_fold=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3",
                          settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, iou_train = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                                  settle=True, use_fold=True, use_tube=True)
    save_obj(f"{OUT}/cow_v18_pre.obj", v, t)
    print(f"[train] done iou={iou_train:.4f} V={len(v)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[surgery]", flush=True)
    v, t = surgery(np.asarray(v, dtype=np.float64),
                   np.asarray(t, dtype=np.int64),
                   iou_fn=iou_fn, iou_budget=3e-4, global_cap=1.5e-3,
                   max_rounds=8, max_grow=4)
    print(f"[surgery] done iou={iou_fn(v, t):.4f} V={len(v)}", flush=True)

    print("[settle re-fit]", flush=True)
    v, iou_settle = optimize_phase(ctx, v, np.asarray(t, dtype=np.int32),
                                   gt, gtd, mvps, 400, "settle",
                                   settle=True, use_fold=True, use_tube=True)

    print("[final light surgery]", flush=True)
    v, t = surgery(np.asarray(v, dtype=np.float64),
                   np.asarray(t, dtype=np.int64),
                   iou_fn=iou_fn, iou_budget=2e-4, global_cap=6e-4,
                   max_rounds=4, max_grow=4)
    iou_final = iou_fn(v, t)
    print(f"[final] iou={iou_final:.4f} V={len(v)}", flush=True)

    t = np.asarray(t, dtype=np.int32)
    save_obj(f"{OUT}/cow_v18.obj", v, t)
    np.savez(f"{OUT}/cow_v18.npz", verts=v, tris=t)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t, scene)
    print(f"\n=== V18 RESULT (gated tube prior) ===")
    print(f"train6 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v17 train6=0.9761 ho=0.9477/11332/1156")
    print(f"time: {time.time()-t0:.0f}s", flush=True)

    # visuals
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "cow.obj"))
    gv = normalize_to_range(gv)
    z17 = np.load(f"{OUT}/cow_v17.npz")
    cols = [("GT cow", gv, gf), ("v17", z17["verts"], z17["tris"]),
            ("v18 final", v, t)]
    fin = np.array([-0.056, 0.639, 0.49])
    head = np.array([0.008, 0.519, 1.311])
    views = [("full", None, (0, 0)), ("fin zoom", fin, (0, 0)),
             ("ear zoom", head, (20, 120)), ("horn top", head, (60, 90))]
    fig = plt.figure(figsize=(12, 13), dpi=120)
    for j, (name, vv, ff) in enumerate(cols):
        for r, (lbl, cc, (elev, azim)) in enumerate(views):
            ax = fig.add_subplot(4, 3, r*3+j+1, projection="3d")
            ax.add_collection3d(Poly3DCollection(
                np.asarray(vv)[np.asarray(ff)], facecolor="#5cb85c",
                edgecolor="none"))
            if cc is None:
                lo, hi = np.array([-1, -1, -1]), np.array([1, 1, 1])
            else:
                lo, hi = cc - 0.4, cc + 0.4
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=elev, azim=azim)
            ax.axis("off")
            if r == 0: ax.set_title(name, fontsize=11)
            if j == 0: ax.text2D(-0.1, 0.5, lbl, transform=ax.transAxes,
                                 rotation=90, va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cow_v18_compare.png", bbox_inches="tight")
    fig = plt.figure(figsize=(16, 4.5), dpi=110)
    for k in range(16):
        ax = fig.add_subplot(2, 8, k+1, projection="3d")
        ax.add_collection3d(Poly3DCollection(v[t], facecolor="#5cb85c",
                                             edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2))
        ax.view_init(elev=20, azim=22.5 * k); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cow_v18_360.png", bbox_inches="tight")
    print("saved cow_v18_compare.png, cow_v18_360.png", flush=True)


if __name__ == "__main__":
    main()
