#!/usr/bin/env python3
"""
viz_compare.py — Visual comparison: GT vs canonical-init vs shape-init results.

For each real shape, produce a grid image:
  rows    = 4 views
  columns = GT | plain-100 | shape-100 | plain-500 | shape-500

Output: eval_out/viz_<shape>.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from canonical_topos import CANONICAL_TOPOS, N_CLASSES, build_topo_mesh
from model_v5 import TopoShapeNet
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5 import adaptive_remesh, compute_iou
from eval_real_shapes import (
    load_obj, make_torus, make_stretched_bar, make_flat_plate,
    normalize_to_range, BUNNY_PATH,
)

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0


def optimize_and_render(ctx, class_id, gt_uint8, mvps, device, n_steps, shape_params):
    """Optimize and return (iou, final silhouettes [4,H,W] float 1=fg)."""
    topo = CANONICAL_TOPOS[class_id]
    verts_np, tris_np = build_topo_mesh(topo)

    mn, mx = float(verts_np.min()), float(verts_np.max())
    scale  = 2.0 / max(mx - mn, 1e-6)
    verts_np = (verts_np - (mn + mx) / 2.0) * scale
    if shape_params is not None:
        verts_np = verts_np * shape_params[np.newaxis, :]

    gt_fg_f = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg_f).unsqueeze(-1).to(device)

    verts_np, tris_np = adaptive_remesh(verts_np, tris_np)
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    opt   = torch.optim.Adam([verts_t], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=3e-5)

    for step in range(n_steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(4):
            rendered = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                         resolution=(IMG_RES, IMG_RES))
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i+1])
        sil_loss = sil_loss / 4
        reg = 0.1 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

    with torch.no_grad():
        views = []
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            views.append(sil[0, :, :, 0].cpu().numpy())
    pred_sil = np.stack(views, axis=0)
    return compute_iou(pred_sil, gt_uint8), pred_sil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'))
    parser.add_argument('--out_dir', default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--device',  default='cuda')
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    device = args.device if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = TopoShapeNet(n_classes=N_CLASSES).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict({k: v for k, v in state.items()
                           if not k.startswith('dropout')}, strict=True)
    model.eval()

    ctx = dr.RasterizeCudaContext()
    mvps, _ = orbit_cameras(4, elevation_deg=0.0, radius=CAMERA_RADIUS,
                             azimuths_deg=AZIMUTHS, device=device)

    shapes = {}
    if os.path.exists(BUNNY_PATH):
        shapes['bunny'] = load_obj(BUNNY_PATH)
    shapes['torus']         = make_torus()
    shapes['stretched_bar'] = make_stretched_bar()
    shapes['flat_plate']    = make_flat_plate()

    os.makedirs(args.out_dir, exist_ok=True)

    for name, (verts_np, tris_np) in shapes.items():
        print(f"=== {name} ===")
        verts_np = normalize_to_range(verts_np)

        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
        faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)
        gt_views = []
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            sil_np = sil[0, :, :, 0].detach().cpu().numpy()
            gt_views.append(((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8))
        gt_uint8 = np.stack(gt_views, axis=0)

        img_f = gt_uint8.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cls_logits, shape_pred = model(img_t)
            top1 = int(cls_logits[0].argmax().item())
            pred_shape = shape_pred[0].float().cpu().numpy()

        # Run 4 variants
        variants = {}
        for n_steps in (100, 500):
            iou_p, sil_p = optimize_and_render(ctx, top1, gt_uint8, mvps, device, n_steps, None)
            iou_s, sil_s = optimize_and_render(ctx, top1, gt_uint8, mvps, device, n_steps, pred_shape)
            variants[f'plain_{n_steps}'] = (iou_p, sil_p)
            variants[f'shape_{n_steps}'] = (iou_s, sil_s)
            print(f"  {n_steps}: plain={iou_p:.4f} shape={iou_s:.4f}")

        # Build grid: rows=4 views, cols=GT + 4 variants
        col_names = ['GT', 'plain_100', 'shape_100', 'plain_500', 'shape_500']
        PAD, HEADER = 4, 24
        W = len(col_names) * (IMG_RES + PAD) + PAD
        H = HEADER + 4 * (IMG_RES + PAD) + PAD
        canvas = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(canvas)

        for ci, cname in enumerate(col_names):
            x0 = PAD + ci * (IMG_RES + PAD)
            if cname == 'GT':
                label = 'GT'
                col_imgs = gt_uint8  # white-bg already
            else:
                iou, sil = variants[cname]
                label = f"{cname} ({iou:.3f})"
                col_imgs = ((1.0 - sil) * 255.0).clip(0, 255).astype(np.uint8)
            draw.text((x0, 6), label, fill='black')
            for vi in range(4):
                y0 = HEADER + vi * (IMG_RES + PAD)
                canvas.paste(Image.fromarray(col_imgs[vi], mode='L').convert('RGB'),
                             (x0, y0))

        out_path = os.path.join(args.out_dir, f'viz_{name}.png')
        canvas.save(out_path)
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
