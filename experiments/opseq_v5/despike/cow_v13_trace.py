"""V13-trace: identical recipe to v13, but snapshot verts every 50 steps.
After training: detect residual thin-tube verts on the final mesh, trace them
back through cc4 snapshots, render a step-by-step timeline from the needle
viewpoint with the needle verts highlighted red.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import torch.nn.functional as F
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
SNAP_EVERY = 50
WARMUP_STEPS = 20; LAP_WARMUP_STEPS = 50; DEPTH_WARMUP_STEPS = 80
W_LAP_BOOST = 0.40
W_FOLD = 0.02; FOLD_START = 400
W_SPIKE = 400.0; SPIKE_THR = 0.85
W_SLIVER = 2000.0
W_TUBE = 100.0; TUBE_EVERY = 50; TUBE_THR = 0.4

from cow_v13_lib import (midpoint_subdivide, build_pairs, fold_loss, build_adj,
                         centroid, spike_pen, sliver_pen, tube_mask,
                         mean_edge_of)


def optimize_phase(ctx, verts_np, tris_np, gt, gtd, mvps, steps, label,
                   settle=False, use_fold=False, use_tube=False):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    src, dst, deg, excl = build_adj(tris_np, len(verts_np), DEVICE)
    pairs_t = (torch.tensor(build_pairs(tris_np), device=DEVICE) if use_fold else None)
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_MIN)
    warmup = WARMUP_STEPS if settle else 0
    lap_boost = LAP_WARMUP_STEPS if settle else 0
    depth_warm = DEPTH_WARMUP_STEPS if settle else 0
    tmask = None
    snaps = {}
    for step in range(steps):
        if step % SNAP_EVERY == 0:
            snaps[step] = verts_t.detach().cpu().numpy().copy()
        if warmup > 0:
            frac = 1.0 - warmup / WARMUP_STEPS
            for pg in opt.param_groups: pg['lr'] = LR * max(frac, 0.05)
            warmup -= 1
        w_lap = W_LAP_BOOST if lap_boost > 0 else W_LAP
        dw = W_DEPTH * (1.0 - depth_warm / DEPTH_WARMUP_STEPS) if depth_warm > 0 else W_DEPTH
        if lap_boost > 0: lap_boost -= 1
        if depth_warm > 0: depth_warm -= 1
        me = mean_edge_of(verts_t.detach(), src, dst)
        if use_tube and step % TUBE_EVERY == 0:
            tmask = tube_mask(verts_t.detach(), excl, me, TUBE_THR)
        opt.zero_grad()
        sl = torch.tensor(0.0, device=DEVICE); dl = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i],
                                                  (IMG_RES, IMG_RES))
            sl += F.l1_loss(sil[0], targets[i])
            dl += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sl /= N_VIEWS; dl /= N_VIEWS
        loss = (sl + dw * dl + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t)
                + W_SPIKE * spike_pen(verts_t, src, dst, deg, me, SPIKE_THR)
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
            print(f"  [{label}] {step:4d}/{steps} iou={iou:.4f}", flush=True)
    snaps[steps] = verts_t.detach().cpu().numpy().copy()
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    print(f"  [{label}] END iou={iou:.4f}", flush=True)
    return snaps[steps].astype(np.float64), iou, snaps, excl, (src, dst)


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _, s2, _, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2")
    np.savez(f"{OUT}/trace_cc2.npz", tris=t, **{f"s{k}": a for k, a in s2.items()})
    v, t = midpoint_subdivide(v, t)
    v, _, s3, _, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3",
                                    settle=True, use_tube=True)
    np.savez(f"{OUT}/trace_cc3.npz", tris=t, **{f"s{k}": a for k, a in s3.items()})
    v, t = midpoint_subdivide(v, t)
    v, iou, s4, excl, (src, dst) = optimize_phase(
        ctx, v, t, gt, gtd, mvps, 800, "cc4",
        settle=True, use_fold=True, use_tube=True)
    np.savez(f"{OUT}/trace_cc4.npz", tris=t, **{f"s{k}": a for k, a in s4.items()})
    # residual tube verts on final mesh
    vt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    me = mean_edge_of(vt, src, dst)
    tm = tube_mask(vt, excl, me, TUBE_THR).cpu().numpy()
    np.save(f"{OUT}/trace_tubemask.npy", tm)
    print(f"train6 iou={iou:.4f}  final tube verts={int(tm.sum())}")
    with open(f"{OUT}/cow_v13t.obj", "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"time {time.time()-t0:.0f}s; saved trace npz + cow_v13t.obj", flush=True)


if __name__ == "__main__":
    main()
