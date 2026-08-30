"""V14: v13 recipe with two root-cause fixes for the shoulder-fin needle:
  1. Tube prior active from cc2 (the birth phase) — kill fins at ~5 verts.
  2. Collapse penalty target changed: pull each tube vert toward the NEAREST
     NON-TUBE vertex (anchor), not its neighbor centroid.  A straight
     degenerate chain is a fixed point of neighbor-centroid pull (only the
     tip feels force); anchor pull retracts the whole fin into the body.
  3. Per-vert penalty uses sum/nv normalization so fin verts aren't diluted
     by legit-thin anatomy verts.
Still 6-view-only supervision. Held-out 16 views used for exam only.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as cc_label
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
    load_obj, normalize_to_range, BUNNY_PATH,
)
from pipeline.cameras import orbit_cameras
from cow_v13_lib import (midpoint_subdivide, build_pairs, fold_loss, build_adj,
                         centroid, spike_pen, sliver_pen, tube_mask,
                         mean_edge_of)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
WARMUP_STEPS = 20; LAP_WARMUP_STEPS = 50; DEPTH_WARMUP_STEPS = 80
W_LAP_BOOST = 0.40
W_FOLD = 0.02; FOLD_START = 400
W_SPIKE = 400.0; SPIKE_THR = 0.85
W_SLIVER = 2000.0
W_TUBE = 100.0; TUBE_EVERY = 50; TUBE_THR = 0.4


@torch.no_grad()
def tube_anchor(v, excl, mean_edge):
    """Return (mask, anchor_pos): anchor = nearest non-tube vertex position."""
    D = torch.cdist(v, v)
    D[excl] = 1e9
    mask = D.min(1).values < TUBE_THR * mean_edge
    if not mask.any():
        return mask, None
    # nearest NON-tube vertex for every tube vert (exclude tube columns)
    D2 = torch.cdist(v[mask], v[~mask])
    idx = D2.argmin(1)
    return mask, v[~mask][idx].clone()


def optimize_phase(ctx, verts_np, tris_np, gt, gtd, mvps, steps, label,
                   settle=False, use_fold=False, use_tube=False):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    nv = len(verts_np)
    src, dst, deg, excl = build_adj(tris_np, nv, DEVICE)
    pairs_t = (torch.tensor(build_pairs(tris_np), device=DEVICE) if use_fold else None)
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_MIN)
    warmup = WARMUP_STEPS if settle else 0
    lap_boost = LAP_WARMUP_STEPS if settle else 0
    depth_warm = DEPTH_WARMUP_STEPS if settle else 0
    tmask, tanchor = None, None
    for step in range(steps):
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
            tmask, tanchor = tube_anchor(verts_t.detach(), excl, me)
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
        if tmask is not None and tanchor is not None and tmask.any():
            # per-vert sum normalized by TOTAL verts: no dilution across CCs
            loss = loss + W_TUBE * (verts_t[tmask] - tanchor
                                    ).norm(dim=-1).pow(2).sum() / nv
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
            nt = int(tmask.sum()) if tmask is not None else -1
            print(f"  [{label}] {step:4d}/{steps} iou={iou:.4f} tube_v={nt}", flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    v_out = verts_t.detach()
    tm, _ = tube_anchor(v_out, excl, mean_edge_of(v_out, src, dst))
    print(f"  [{label}] END iou={iou:.4f} final_tube_v={int(tm.sum())}", flush=True)
    return v_out.cpu().numpy().astype(np.float64), iou, tm.cpu().numpy()


def heldout_exam(ctx, v, t):
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "cow.obj"))
    gv = normalize_to_range(gv)
    gvt = torch.tensor(gv, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gf, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv, _ = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5,
                          azimuths_deg=azs, device=DEVICE)
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    gs = render_views_n(ctx, gvt, gft, mv)
    ps = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; mb = 0
    for i in range(16):
        g = gs[i] > 0.5; p = ps[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2)
        hair += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1): mb = max(mb, int((lab == k).sum()))
    return inter / max(un, 1), hair, mb


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2", use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, _, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3",
                             settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, iou, tm = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                                settle=True, use_fold=True, use_tube=True)
    with open(f"{OUT}/cow_v14.obj", "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")
    np.save(f"{OUT}/v14_tubemask.npy", tm)
    print(f"saved {OUT}/cow_v14.obj", flush=True)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t)
    print(f"\n=== V14 RESULT (anchor-pull tube prior from cc2) ===")
    print(f"train6 IoU={iou:.4f}")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v13 train6=0.9754 ho=0.8981/40048/2898; v10 0.9827/0.8714")
    print(f"time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
