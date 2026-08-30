"""V3: polish cow_c2f_v2.obj with hair-burn loss.

Extra loss: predicted coverage on pixels OUTSIDE the 3-dilated GT mask,
heavily weighted. Normal sil/depth/reg losses keep the fit. Low LR, 400 steps.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, W_DEPTH, W_LAP, W_EDGE,
)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
STEPS = 400
LR_POLISH = 2e-3
W_BURN = 20.0

scene = setup_scene("cow", DEVICE)
ctx, mvps = scene["ctx"], scene["mvps"]
gt_uint8, gt_depths = scene["gt_uint8"], scene["gt_depths"]

vs, fs = [], []
for line in open(f"{OUT}/cow_c2f_v2.obj"):
    p = line.split()
    if not p: continue
    if p[0] == "v": vs.append([float(x) for x in p[1:4]])
    elif p[0] == "f": fs.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
verts_t = torch.tensor(np.array(vs), dtype=torch.float32,
                       device=DEVICE).requires_grad_(True)
faces_t = torch.tensor(np.array(fs), dtype=torch.int32, device=DEVICE)

targets = torch.from_numpy((gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(DEVICE) for i in range(N_VIEWS)]
gt_fg_t = [torch.from_numpy(gt_uint8[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
# burn mask: 1 where pixel is OUTSIDE 3-dilated GT
burn_masks = [torch.from_numpy(
    (~binary_dilation(gt_uint8[i] < 128, iterations=3)).astype(np.float32)
    ).to(DEVICE) for i in range(N_VIEWS)]

def hair_px():
    sils = render_views_n(ctx, verts_t.detach(), faces_t, mvps)
    return sum(int(((s > 0.5) & ~binary_dilation(gt_uint8[i] < 128, iterations=2)).sum())
               for i, s in enumerate(sils))

iou0 = compute_iou_n(render_views_n(ctx, verts_t.detach(), faces_t, mvps), gt_uint8)
print(f"start: IoU={iou0:.4f} hair={hair_px()}", flush=True)

opt = torch.optim.Adam([verts_t], lr=LR_POLISH)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-4)
t0 = time.time()
for step in range(STEPS):
    opt.zero_grad()
    sil_loss = torch.tensor(0.0, device=DEVICE)
    d_loss = torch.tensor(0.0, device=DEVICE)
    burn = torch.tensor(0.0, device=DEVICE)
    for i in range(N_VIEWS):
        sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i],
                                              (IMG_RES, IMG_RES))
        sil_loss += F.l1_loss(sil[0], targets[i])
        d_loss += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        burn += (sil[0, :, :, 0] * burn_masks[i]).sum() / (IMG_RES * IMG_RES)
    sil_loss /= N_VIEWS; d_loss /= N_VIEWS; burn /= N_VIEWS
    loss = (sil_loss + W_DEPTH * d_loss + W_BURN * burn
            + W_LAP * laplacian_loss(verts_t, faces_t)
            + W_EDGE * edge_length_loss(verts_t, faces_t))
    loss.backward()
    torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
    opt.step(); sched.step()
    if step % 100 == 0:
        iou = compute_iou_n(render_views_n(ctx, verts_t.detach(), faces_t, mvps), gt_uint8)
        print(f"  step {step:3d}/{STEPS} sil={sil_loss.item():.4f} "
              f"burn={burn.item()*1e4:.2f}e-4 iou={iou:.4f} hair={hair_px()}", flush=True)

iou1 = compute_iou_n(render_views_n(ctx, verts_t.detach(), faces_t, mvps), gt_uint8)
hp1 = hair_px()
print(f"\n=== V3 RESULT ===")
print(f"before: IoU={iou0:.4f} hair=463   after: IoU={iou1:.4f} hair={hp1}")
print(f"time: {time.time()-t0:.0f}s", flush=True)

v = verts_t.detach().cpu().numpy()
with open(f"{OUT}/cow_v3.obj", "w") as fh:
    for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
    for a, b, c in np.array(fs): fh.write(f"f {a+1} {b+1} {c+1}\n")

import imageio.v2 as imageio
sils = render_views_n(ctx, verts_t.detach(), faces_t, mvps)
rows = []
for i in range(N_VIEWS):
    pred = sils[i] > 0.5; gt = gt_uint8[i] < 128
    img = np.full((IMG_RES, IMG_RES, 3), 255, np.uint8)
    img[gt & ~pred] = (220, 60, 60); img[pred & ~gt] = (60, 200, 60)
    img[gt & pred] = (200, 200, 80)
    rows.append(img)
imageio.imwrite(f"{OUT}/cow_v3.png", np.concatenate(
    [np.concatenate(rows[:3], 1), np.concatenate(rows[3:], 1)], 0))
print(f"saved {OUT}/cow_v3.obj / .png", flush=True)
