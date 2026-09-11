"""C2F v5: coarse-to-fine + the ORIGINAL 2026-08-22 anti-spike recipe at each
subdivision boundary (exactly what killed the C-vs-B spikes after extrude):

  On each subdivision (analogous to an injection event):
    - LR warmup: 20 steps ramp
    - Laplacian boost: W_LAP_BOOST=0.40 for 50 steps
    - Depth phase-in: ramp 0 -> W_DEPTH over 80 steps

Plus late-phase fold penalty in cc4 (v2's proven win). NO burn loss —
test whether the settle-first recipe prevents hairs at the root.
Budget: 800+800+800.
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
)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
WARMUP_STEPS = 20
LAP_WARMUP_STEPS = 50
DEPTH_WARMUP_STEPS = 80
W_LAP_BOOST = 0.40
W_FOLD = 0.02
FOLD_START = 400


def midpoint_subdivide(verts, tris):
    verts = list(map(tuple, verts))
    em = {}
    def mid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in em:
            va, vb = verts[a], verts[b]
            verts.append(((va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2))
            em[k] = len(verts) - 1
        return em[k]
    nt = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(nt, dtype=np.int32)


def build_pairs(tris_np):
    e2f = {}
    for fi, (a, b, c) in enumerate(tris_np):
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            e2f.setdefault(k, []).append(fi)
    return np.array([p for p in e2f.values() if len(p) == 2], dtype=np.int64)


def fold_loss(verts_t, faces_l, pairs_t):
    tri = verts_t[faces_l]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    d = (n[pairs_t[:, 0]] * n[pairs_t[:, 1]]).sum(-1)
    return F.relu(-d).pow(2).mean()


def hair_stats(ctx, verts_t, faces_t, mvps, gt_uint8):
    sils = render_views_n(ctx, verts_t, faces_t, mvps)
    total, maxblob = 0, 0
    for i in range(N_VIEWS):
        out = (sils[i] > 0.5) & ~binary_dilation(gt_uint8[i] < 128, iterations=2)
        total += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return total, maxblob


def optimize_phase(ctx, verts_np, tris_np, gt, gtd, mvps, steps, label,
                   settle=False, pairs_np=None, w_fold=0.0, fold_start=0):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    pairs_t = torch.tensor(pairs_np, device=DEVICE) if pairs_np is not None else None
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_MIN)
    warmup = WARMUP_STEPS if settle else 0
    lap_boost = LAP_WARMUP_STEPS if settle else 0
    depth_warm = DEPTH_WARMUP_STEPS if settle else 0
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
        opt.zero_grad()
        sl = torch.tensor(0.0, device=DEVICE)
        dl = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i],
                                                  (IMG_RES, IMG_RES))
            sl += F.l1_loss(sil[0], targets[i])
            dl += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sl /= N_VIEWS; dl /= N_VIEWS
        loss = (sl + dw * dl + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t))
        if pairs_t is not None and w_fold > 0 and step >= fold_start:
            ramp = min(1.0, (step - fold_start) / max(1, (steps - fold_start) * 0.5))
            loss = loss + w_fold * ramp * fold_loss(verts_t, faces_l, pairs_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
            hp, mb = hair_stats(ctx, verts_t.detach(), faces_t, mvps, gt)
            print(f"  [{label}] {step:4d}/{steps} sil={sl.item():.4f} iou={iou:.4f} "
                  f"hair={hp} maxblob={mb}", flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    hp, mb = hair_stats(ctx, verts_t.detach(), faces_t, mvps, gt)
    print(f"  [{label}] END iou={iou:.4f} hair={hp} maxblob={mb}", flush=True)
    return verts_t.detach().cpu().numpy().astype(np.float64), iou, hp, mb


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, *_ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2")
    v, t = midpoint_subdivide(v, t)
    v, *_ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3", settle=True)
    v, t = midpoint_subdivide(v, t)
    pairs = build_pairs(t)
    v, iou, hp, mb = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                                    settle=True, pairs_np=pairs,
                                    w_fold=W_FOLD, fold_start=FOLD_START)
    print(f"\n=== V5 RESULT ===")
    print(f"IoU={iou:.4f} hair={hp} maxblob={mb}  (refs: direct=0.9804/1690, "
          f"v2=0.9835/463, v4-burn=0.9827/119/19)")
    print(f"time: {time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT}/cow_v5.obj", "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")
    import imageio.v2 as imageio
    vt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    sils = render_views_n(ctx, vt, ft, mvps)
    rows = []
    for i in range(N_VIEWS):
        pred = sils[i] > 0.5; g = gt[i] < 128
        img = np.full((IMG_RES, IMG_RES, 3), 255, np.uint8)
        img[g & ~pred] = (220, 60, 60); img[pred & ~g] = (60, 200, 60)
        img[g & pred] = (200, 200, 80)
        rows.append(img)
    imageio.imwrite(f"{OUT}/cow_v5.png", np.concatenate(
        [np.concatenate(rows[:3], 1), np.concatenate(rows[3:], 1)], 0))
    print(f"saved {OUT}/cow_v5.obj / .png", flush=True)


if __name__ == "__main__":
    main()
