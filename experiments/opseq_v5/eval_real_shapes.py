#!/usr/bin/env python3
"""
eval_real_shapes.py — Test 2: shape-aware init vs canonical init on REAL shapes.

Real shapes (not from the synthetic training distribution):
  1. Stanford Bunny  (genus 0, complex geometry, ears = local minima risk)
  2. Torus           (genus 1, parametric)
  3. Stretched bar   (genus 0, extreme aspect ratio 3:1:0.5)
  4. Flat plate      (genus 0, extreme aspect ratio 1:0.15:1)

Protocol per shape:
  1. Load/build GT mesh, normalize, render 4-view GT silhouettes
  2. Run TopoShapeNet -> top-1 topo class + shape params
  3. Optimize 4 variants:
       a. canonical init, 500 steps
       b. shape init,     500 steps
       c. canonical init, 100 steps
       d. shape init,     100 steps
  4. Report IoU for each

Usage:
    python eval_real_shapes.py [--ckpt experiments/opseq_v5/ckpt/best.pt]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

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

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0

BUNNY_PATH = ("/home/kingy/Projects/Genesis/GenesisExp/GenesisHunyuan/.venv/"
              "lib/python3.12/site-packages/pymeshlab/tests/sample_meshes/bunny.obj")


# ═════════════════════════════════════════════════════════════════════════════
# Real shape builders
# ═════════════════════════════════════════════════════════════════════════════

def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i+1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def make_torus(R=1.0, r=0.4, n_u=32, n_v=16) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for i in range(n_u):
        u = 2 * math.pi * i / n_u
        for j in range(n_v):
            v = 2 * math.pi * j / n_v
            x = (R + r * math.cos(v)) * math.cos(u)
            y = r * math.sin(v)
            z = (R + r * math.cos(v)) * math.sin(u)
            verts.append([x, y, z])
    for i in range(n_u):
        for j in range(n_v):
            a = i * n_v + j
            b = ((i + 1) % n_u) * n_v + j
            c = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            d = i * n_v + (j + 1) % n_v
            faces.append([a, b, c]); faces.append([a, c, d])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def make_stretched_bar() -> Tuple[np.ndarray, np.ndarray]:
    """Subdivided icosphere stretched to 3:1:0.5."""
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate
    mesh = make_icosahedron()
    mesh = catmull_clark(mesh)
    mesh = catmull_clark(mesh)
    positions, fcs = mesh_to_arrays(mesh)
    verts = np.array(positions, dtype=np.float64)
    verts = verts * np.array([3.0, 1.0, 0.5])
    tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    return verts, tris


def make_flat_plate() -> Tuple[np.ndarray, np.ndarray]:
    """Subdivided icosphere flattened to 1:0.15:1."""
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate
    mesh = make_icosahedron()
    mesh = catmull_clark(mesh)
    mesh = catmull_clark(mesh)
    positions, fcs = mesh_to_arrays(mesh)
    verts = np.array(positions, dtype=np.float64)
    verts = verts * np.array([1.0, 0.15, 1.0])
    tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    return verts, tris


def normalize_to_range(verts: np.ndarray) -> np.ndarray:
    """Normalize to 80% of [-2, 2] (same as training data)."""
    mn, mx = float(verts.min()), float(verts.max())
    extent = max(mx - mn, 1e-6)
    scale  = 0.8 * 4.0 / extent
    centre = (mn + mx) / 2.0
    return (verts - centre) * scale


# ═════════════════════════════════════════════════════════════════════════════
# Optimization (same core as eval_v5)
# ═════════════════════════════════════════════════════════════════════════════

def optimize(
    ctx, class_id, gt_uint8, mvps, device,
    n_steps, shape_params=None,
) -> float:
    topo = CANONICAL_TOPOS[class_id]
    try:
        verts_np, tris_np = build_topo_mesh(topo)
    except Exception:
        return 0.0
    if verts_np.shape[0] < 3 or tris_np.shape[0] == 0:
        return 0.0

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
        reg = 0.05 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

    with torch.no_grad():
        rendered_views = []
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            rendered_views.append(sil[0, :, :, 0].cpu().numpy())
    pred_sil = np.stack(rendered_views, axis=0)
    return compute_iou(pred_sil, gt_uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',    default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'))
    parser.add_argument('--out_dir', default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--device',  default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = TopoShapeNet(n_classes=N_CLASSES).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    base_state = {k: v for k, v in state.items() if not k.startswith('dropout')}
    model.load_state_dict(base_state, strict=True)
    model.eval()
    print(f"Loaded: {args.ckpt}")

    ctx = dr.RasterizeCudaContext()
    mvps, _ = orbit_cameras(4, elevation_deg=0.0, radius=CAMERA_RADIUS,
                             azimuths_deg=AZIMUTHS, device=device)

    # Build real shapes
    shapes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if os.path.exists(BUNNY_PATH):
        shapes['bunny'] = load_obj(BUNNY_PATH)
    else:
        print(f"WARNING: bunny not found at {BUNNY_PATH}")
    shapes['torus']         = make_torus()
    shapes['stretched_bar'] = make_stretched_bar()
    shapes['flat_plate']    = make_flat_plate()

    results = []

    for name, (verts_np, tris_np) in shapes.items():
        print(f"\n=== {name}  V={len(verts_np)} T={len(tris_np)} ===")
        verts_np = normalize_to_range(verts_np)

        # Render GT silhouettes
        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
        faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)
        gt_views = []
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            sil_np = sil[0, :, :, 0].detach().cpu().numpy()
            gt_views.append(((1.0 - sil_np) * 255.0).clip(0, 255).astype(np.uint8))
        gt_uint8 = np.stack(gt_views, axis=0)  # [4, H, W] white-bg

        # Model prediction
        img_f = gt_uint8.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cls_logits, shape_pred = model(img_t)
            top1 = int(cls_logits[0].argmax().item())
            pred_shape = shape_pred[0].float().cpu().numpy()

        gt_extents = np.abs(verts_np).max(axis=0)
        print(f"  predicted class: {top1} ({CANONICAL_TOPOS[top1]['label']})")
        print(f"  predicted shape: {pred_shape}")
        print(f"  GT half-extents: {gt_extents}")

        # 4 optimization variants
        row = {'name': name, 'class': CANONICAL_TOPOS[top1]['label'],
               'pred_shape': pred_shape.tolist(), 'gt_extents': gt_extents.tolist()}
        for n_steps in (500, 100):
            iou_plain = optimize(ctx, top1, gt_uint8, mvps, device, n_steps, None)
            iou_shape = optimize(ctx, top1, gt_uint8, mvps, device, n_steps, pred_shape)
            row[f'plain_{n_steps}'] = iou_plain
            row[f'shape_{n_steps}'] = iou_shape
            print(f"  {n_steps} steps:  plain={iou_plain:.4f}  shape={iou_shape:.4f}  "
                  f"delta={iou_shape-iou_plain:+.4f}")
        results.append(row)

    # Report
    lines = [
        "# Real Shapes: Shape-Aware Init vs Canonical Init",
        "",
        "| Shape | Pred Class | 500p plain | 500p shape | Δ500 | 100p plain | 100p shape | Δ100 |",
        "|-------|-----------|-----------|-----------|------|-----------|-----------|------|",
    ]
    for r in results:
        d500 = r['shape_500'] - r['plain_500']
        d100 = r['shape_100'] - r['plain_100']
        lines.append(
            f"| {r['name']} | {r['class']} | {r['plain_500']:.4f} | {r['shape_500']:.4f} "
            f"| {d500:+.4f} | {r['plain_100']:.4f} | {r['shape_100']:.4f} | {d100:+.4f} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"- **{r['name']}**: pred_shape={[f'{x:.2f}' for x in r['pred_shape']]}, "
                     f"gt_extents={[f'{x:.2f}' for x in r['gt_extents']]}")
    lines.append("")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'results_real_shapes.md')
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nResults -> {out_path}")


if __name__ == '__main__':
    main()
