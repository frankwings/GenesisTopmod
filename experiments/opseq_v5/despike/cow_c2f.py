"""Coarse-to-fine progressive subdivision: cc2 -> cc3 -> cc4.

Hypothesis: fold-flap hairs form because 7680 faces dumped on a coarse fit
crumple freely. If we fit coarse first and only subdivide once the surface
is already near-correct, the fine verts start ON the surface and never fold.

Budget: 800 + 800 + 800 = 2400 steps (same total as cc4 baseline 0.9804).
Outputs: /tmp/liou_cow_viz/cow_c2f.obj / .png + fold census + hair pixels.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np
import torch
import torch.nn.functional as F
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
os.makedirs(OUT, exist_ok=True)


def midpoint_subdivide(verts: np.ndarray, tris: np.ndarray):
    """Interpolating 1->4 triangle subdivision (preserves fitted shape)."""
    verts = list(map(tuple, verts))
    edge_mid = {}

    def mid(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in edge_mid:
            va, vb = verts[a], verts[b]
            verts.append(((va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2))
            edge_mid[key] = len(verts) - 1
        return edge_mid[key]

    new_tris = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_tris += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(new_tris, dtype=np.int32)


def optimize_phase(ctx, verts_np, tris_np, gt_uint8, gt_depths, mvps,
                   steps, label, lr=LR):
    targets = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(DEVICE)
                  for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt_uint8[i] < 128).to(DEVICE)
               for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    opt = torch.optim.Adam([verts_t], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=LR_MIN)
    for step in range(steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=DEVICE)
        d_loss = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            sil_loss += F.l1_loss(sil[0], targets[i])
            d_loss += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sil_loss /= N_VIEWS
        d_loss /= N_VIEWS
        loss = (sil_loss + W_DEPTH * d_loss
                + W_LAP * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps),
                                gt_uint8)
            print(f"  [{label}] step {step:4d}/{steps}  "
                  f"sil={sil_loss.item():.4f}  iou={iou:.4f}", flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
    print(f"  [{label}] PHASE END  IoU={iou:.4f}  F={len(tris_np)}", flush=True)
    return verts_t.detach().cpu().numpy().astype(np.float64), iou


def fold_census(verts_np, tris_np):
    v = torch.tensor(verts_np, dtype=torch.float32)
    f = torch.tensor(tris_np, dtype=torch.int64)
    tri = v[f]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    edge2f = {}
    for fi, (a, b, c) in enumerate(tris_np):
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            edge2f.setdefault(k, []).append(fi)
    pairs = [p for p in edge2f.values() if len(p) == 2]
    pt = torch.tensor(pairs, dtype=torch.int64)
    d = (n[pt[:, 0]] * n[pt[:, 1]]).sum(-1)
    return int((d < 0).sum()), int((d < -0.5).sum()), len(pairs)


def hair_pixels(ctx, verts_np, tris_np, mvps, gt_uint8):
    from scipy.ndimage import binary_dilation
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    sils = render_views_n(ctx, verts_t, faces_t, mvps)
    total = 0
    for i in range(N_VIEWS):
        pred = sils[i] > 0.5
        gt = gt_uint8[i] < 128
        gt_dil = binary_dilation(gt, iterations=2)
        total += int((pred & ~gt_dil).sum())
    return total


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt_uint8, gt_depths = scene["gt_uint8"], scene["gt_depths"]

    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    print(f"=== phase cc2: F={len(t)} ===", flush=True)
    v, iou2 = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc2")

    v, t = midpoint_subdivide(v, t)
    print(f"=== phase cc3: F={len(t)} ===", flush=True)
    v, iou3 = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc3")

    v, t = midpoint_subdivide(v, t)
    print(f"=== phase cc4: F={len(t)} ===", flush=True)
    v, iou4 = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc4")

    neg, hard, npairs = fold_census(v, t)
    hp = hair_pixels(ctx, v, t, mvps, gt_uint8)
    print(f"\n=== C2F RESULT ===")
    print(f"IoU: cc2={iou2:.4f} -> cc3={iou3:.4f} -> cc4={iou4:.4f}")
    print(f"folds: neg={neg}/{npairs}  hard(dot<-0.5)={hard}")
    print(f"hair pixels (outside 2-dilated GT, 6 views): {hp}")
    print(f"baseline cc4 direct: IoU=0.9804  neg=2636/11520  hard=2123  hair~1690")
    print(f"time: {time.time()-t0:.0f}s", flush=True)

    # save obj
    with open(f"{OUT}/cow_c2f.obj", "w") as fh:
        for x, y, z in v:
            fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t:
            fh.write(f"f {a+1} {b+1} {c+1}\n")

    # overlay png: red=GT only, green=pred only, yellow=both
    import imageio.v2 as imageio
    verts_t = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    faces_t = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    sils = render_views_n(ctx, verts_t, faces_t, mvps)
    rows = []
    for i in range(N_VIEWS):
        pred = sils[i] > 0.5
        gt = gt_uint8[i] < 128
        img = np.full((IMG_RES, IMG_RES, 3), 255, np.uint8)
        img[gt & ~pred] = (220, 60, 60)
        img[pred & ~gt] = (60, 200, 60)
        img[gt & pred] = (200, 200, 80)
        rows.append(img)
    grid = np.concatenate(
        [np.concatenate(rows[:3], axis=1), np.concatenate(rows[3:], axis=1)],
        axis=0)
    imageio.imwrite(f"{OUT}/cow_c2f.png", grid)
    print(f"saved {OUT}/cow_c2f.obj / .png", flush=True)


if __name__ == "__main__":
    main()
