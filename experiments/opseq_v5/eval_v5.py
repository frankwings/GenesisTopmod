#!/usr/bin/env python3
"""
eval_v5.py — Evaluation for TopoShapeNet (Plan C).

Key difference from V4: the initial mesh is pre-shaped using predicted
half-extents before inverse rendering optimization. This should make
convergence faster and improve final IoU.

Metrics:
1. Classification: top-1/3/5 accuracy (same as V4)
2. Shape regression: MAE on [hx, hy, hz]
3. IoU evaluation: compare V5 (shape-aware init) vs V4 (canonical init)

Usage:
    python eval_v5.py [--ckpt experiments/opseq_v5/ckpt/best.pt]
                      [--data_dir experiments/opseq_v5/data]
                      [--n_iou 100] [--opt_steps 500] [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from canonical_topos import CANONICAL_TOPOS, N_CLASSES, build_topo_mesh
from model_v5 import TopoShapeNet, count_params
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)

AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive remeshing (same as V4)
# ═════════════════════════════════════════════════════════════════════════════

def collapse_short_edges(verts, tris, min_len):
    edges = set()
    for f in tris:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i+1)%3]])))
    edges = np.array(list(edges))
    lengths = np.linalg.norm(verts[edges[:,0]]-verts[edges[:,1]], axis=1)
    short_idx = np.where(lengths < min_len)[0]
    if len(short_idx) == 0: return verts, tris, False
    short_idx = short_idx[np.argsort(lengths[short_idx])]
    merge_map = np.arange(len(verts))
    collapsed = set()
    for idx in short_idx:
        a, b = int(edges[idx][0]), int(edges[idx][1])
        while merge_map[a] != a: a = merge_map[a]
        while merge_map[b] != b: b = merge_map[b]
        if a == b or a in collapsed or b in collapsed: continue
        verts[a] = (verts[a]+verts[b])/2; merge_map[b] = a; collapsed.add(b)
    if not collapsed: return verts, tris, False
    for i in range(len(merge_map)):
        v = i
        while merge_map[v] != v: v = merge_map[v]
        merge_map[i] = v
    new_tris = []
    for f in tris:
        nf = [merge_map[f[0]], merge_map[f[1]], merge_map[f[2]]]
        if nf[0]!=nf[1] and nf[1]!=nf[2] and nf[0]!=nf[2]: new_tris.append(nf)
    if not new_tris: return verts, tris, False
    new_tris = np.array(new_tris)
    used = np.unique(new_tris)
    remap = np.full(len(verts), -1, dtype=int); remap[used] = np.arange(len(used))
    return verts[used], remap[new_tris], True


def split_long_edges(verts, tris, max_len):
    edges = set()
    for f in tris:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i+1)%3]])))
    edges = np.array(list(edges))
    lengths = np.linalg.norm(verts[edges[:,0]]-verts[edges[:,1]], axis=1)
    long_idx = np.where(lengths > max_len)[0]
    if len(long_idx) == 0: return verts, tris, False
    long_idx = long_idx[np.argsort(-lengths[long_idx])][:len(verts)//4]
    new_verts = list(verts)
    mid_map = {}
    for idx in long_idx:
        a, b = int(edges[idx][0]), int(edges[idx][1])
        key = (min(a,b), max(a,b))
        if key not in mid_map:
            mid_map[key] = len(new_verts)
            new_verts.append((verts[a]+verts[b])/2)
    if not mid_map: return verts, tris, False
    final_tris = []
    for f in tris:
        splits = {}
        for i in range(3):
            key = (min(f[i],f[(i+1)%3]), max(f[i],f[(i+1)%3]))
            if key in mid_map: splits[i] = mid_map[key]
        if not splits: final_tris.append(f)
        elif len(splits) == 1:
            ei = list(splits.keys())[0]; mid = splits[ei]
            a, b, c = f[ei], f[(ei+1)%3], f[(ei+2)%3]
            final_tris.append([a,mid,c]); final_tris.append([mid,b,c])
        else: final_tris.append(f)
    return np.array(new_verts), np.array(final_tris), True


def adaptive_remesh(v, t):
    edges = set()
    for f in t:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i+1)%3]])))
    edges_arr = np.array(list(edges))
    lengths = np.linalg.norm(v[edges_arr[:,0]]-v[edges_arr[:,1]], axis=1)
    ml = np.mean(lengths)
    v, t, _ = collapse_short_edges(v.copy(), t.copy(), 0.3*ml)
    v, t, _ = split_long_edges(v.copy(), t.copy(), 2.0*ml)
    return v, t


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class TopoDatasetV5(Dataset):
    def __init__(self, shard_paths: List[str]):
        images_list, labels_list, shapes_list = [], [], []
        for p in shard_paths:
            d = np.load(p, allow_pickle=False)
            images_list.append(d['images'])
            labels_list.append(d['labels'].astype(np.int64))
            shapes_list.append(d['shape_params'].astype(np.float32))
        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        self.shapes = np.concatenate(shapes_list, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img   = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])
        shape = self.shapes[idx]
        return img, label, shape


def collate_fn(batch):
    imgs, labels, shapes = zip(*batch)
    return (
        torch.from_numpy(np.stack(imgs, 0)),
        torch.tensor(labels, dtype=torch.long),
        torch.from_numpy(np.stack(shapes, 0)),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Classification + regression eval
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_classification(model, loader, device):
    model.eval()
    per_class_correct = [0] * N_CLASSES
    per_class_total   = [0] * N_CLASSES
    top1_correct = top3_correct = top5_correct = total = 0
    total_shape_mae = 0.0

    for imgs, labels, shapes in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        shapes = shapes.to(device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls_logits, shape_pred = model(imgs)

        B = labels.shape[0]
        total += B

        _, topk_idx = cls_logits.topk(5, dim=1)
        match = topk_idx.eq(labels.unsqueeze(1))
        top1_correct += match[:, :1].any(dim=1).sum().item()
        top3_correct += match[:, :3].any(dim=1).sum().item()
        top5_correct += match[:, :5].any(dim=1).sum().item()

        preds = cls_logits.argmax(dim=1)
        for b in range(B):
            gt = int(labels[b].item())
            per_class_total[gt]   += 1
            per_class_correct[gt] += int(preds[b].item() == gt)

        total_shape_mae += (shape_pred - shapes).abs().mean().item() * B

    return {
        'top1':          top1_correct / max(total, 1),
        'top3':          top3_correct / max(total, 1),
        'top5':          top5_correct / max(total, 1),
        'shape_mae':     total_shape_mae / max(total, 1),
        'per_class_acc': [c/max(t,1) for c,t in zip(per_class_correct, per_class_total)],
        'n_samples':     total,
    }


# ═════════════════════════════════════════════════════════════════════════════
# IoU evaluation with shape-aware init
# ═════════════════════════════════════════════════════════════════════════════

def compute_iou(pred_sil, gt_uint8):
    pred_fg = pred_sil > 0.5
    gt_fg   = gt_uint8 < 128
    ious = []
    for v in range(4):
        inter = (pred_fg[v] & gt_fg[v]).sum()
        union = (pred_fg[v] | gt_fg[v]).sum()
        ious.append(float(inter) / max(float(union), 1.0))
    return float(np.mean(ious))


def optimize_topology_for_iou(
    ctx, class_id, gt_uint8, mvps, device,
    n_steps=500, shape_params=None,
):
    """
    Build topology, optionally apply predicted shape, then optimize.

    shape_params: [3] numpy array of half-extents, or None (canonical init).
    """
    topo = CANONICAL_TOPOS[class_id]
    try:
        verts_np, tris_np = build_topo_mesh(topo)
    except Exception:
        return 0.0

    if verts_np.shape[0] < 3 or tris_np.shape[0] == 0:
        return 0.0

    # Normalize canonical mesh to unit sphere
    mn, mx = float(verts_np.min()), float(verts_np.max())
    scale  = 2.0 / max(mx - mn, 1e-6)
    verts_np = (verts_np - (mn + mx) / 2.0) * scale  # now in [-1, 1]

    # ★ Apply predicted shape params: scale each axis by half-extent
    if shape_params is not None:
        # shape_params = [hx, hy, hz], typically in [0, ~1.6]
        # Canonical mesh is in [-1, 1]. Scale to match target proportions.
        verts_np = verts_np * shape_params[np.newaxis, :]

    # GT targets
    gt_fg_f = (gt_uint8 < 128).astype(np.float32)
    targets = torch.from_numpy(gt_fg_f).unsqueeze(-1).to(device)

    # Adaptive remeshing
    verts_np, tris_np = adaptive_remesh(verts_np, tris_np)

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    opt   = torch.optim.Adam([verts_t], lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=3e-5)

    N_VIEWS = mvps.shape[0]
    for step in range(n_steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(N_VIEWS):
            rendered = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                         resolution=(IMG_RES, IMG_RES))
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i+1])
        sil_loss = sil_loss / N_VIEWS
        reg = 0.05 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)
        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

    with torch.no_grad():
        rendered_views = []
        for i in range(N_VIEWS):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            rendered_views.append(sil[0, :, :, 0].cpu().numpy())
    pred_sil = np.stack(rendered_views, axis=0)
    return compute_iou(pred_sil, gt_uint8)


def eval_iou(
    model, dataset, ctx, device,
    n_samples=100, opt_steps=500,
):
    model.eval()
    mvps, _ = orbit_cameras(4, elevation_deg=0.0, radius=CAMERA_RADIUS,
                             azimuths_deg=AZIMUTHS, device=device)

    # V5 results (shape-aware init)
    v5_best1: List[float] = []
    v5_best5: List[float] = []
    # V4 baseline (canonical init, no shape params)
    v4_best1: List[float] = []

    n_samples = min(n_samples, len(dataset))
    rng = np.random.default_rng(1234)
    indices = rng.choice(len(dataset), size=n_samples, replace=False).tolist()

    for rank, idx in enumerate(indices):
        img_f, gt_label, gt_shape = dataset[idx]
        gt_uint8 = (img_f * 255.0).astype(np.uint8)

        img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cls_logits, shape_pred = model(img_t)
            _, top5 = cls_logits[0].topk(5)
            top5 = top5.cpu().tolist()
            pred_shape = shape_pred[0].cpu().numpy()

        print(f"  IoU sample {rank+1}/{n_samples}  gt={gt_label}  "
              f"top5={top5}  shape_pred={pred_shape}  shape_gt={gt_shape}")

        # V5: top-1 with shape-aware init
        iou1_v5 = optimize_topology_for_iou(
            ctx, top5[0], gt_uint8, mvps, device, opt_steps,
            shape_params=pred_shape,
        )
        v5_best1.append(iou1_v5)

        # V5: best-of-5 with shape-aware init
        ious_5 = [iou1_v5]
        for cid in top5[1:]:
            iou = optimize_topology_for_iou(
                ctx, cid, gt_uint8, mvps, device, opt_steps,
                shape_params=pred_shape,
            )
            ious_5.append(iou)
        v5_best5.append(max(ious_5))

        # V4 baseline: top-1 WITHOUT shape params
        iou1_v4 = optimize_topology_for_iou(
            ctx, top5[0], gt_uint8, mvps, device, opt_steps,
            shape_params=None,
        )
        v4_best1.append(iou1_v4)

        print(f"    V5 best1={iou1_v5:.4f}  best5={v5_best5[-1]:.4f}  "
              f"V4 best1={iou1_v4:.4f}  delta={iou1_v5-iou1_v4:+.4f}")

    return {
        'v5_best1': v5_best1, 'v5_best5': v5_best5, 'v4_best1': v4_best1,
        'mean_v5_best1': float(np.mean(v5_best1)),
        'mean_v5_best5': float(np.mean(v5_best5)),
        'mean_v4_best1': float(np.mean(v4_best1)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════

def write_report(out_path, clf, iou, ckpt_path, n_iou):
    lines = [
        "# V5 TopoShapeNet — Evaluation Results (Plan C)",
        "",
        f"**Checkpoint:** `{ckpt_path}`",
        f"**Val samples (classification):** {clf['n_samples']}",
        f"**Val samples (IoU eval):** {n_iou}",
        "",
        "## Classification Accuracy",
        "",
        f"| Metric   | Value  |",
        f"|----------|--------|",
        f"| Top-1    | {clf['top1']*100:.2f}% |",
        f"| Top-3    | {clf['top3']*100:.2f}% |",
        f"| Top-5    | {clf['top5']*100:.2f}% |",
        f"| Shape MAE | {clf['shape_mae']:.4f} |",
        "",
        "## IoU Comparison: V5 (shape-aware) vs V4 (canonical)",
        "",
        f"| Metric               | Mean IoU |",
        f"|----------------------|----------|",
        f"| V5 Best-of-1 (shape) | {iou['mean_v5_best1']:.4f} |",
        f"| V5 Best-of-5 (shape) | {iou['mean_v5_best5']:.4f} |",
        f"| V4 Best-of-1 (plain) | {iou['mean_v4_best1']:.4f} |",
        f"| **V5 - V4 delta**    | {iou['mean_v5_best1']-iou['mean_v4_best1']:+.4f} |",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"\nResults -> {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate V5 TopoShapeNet")
    parser.add_argument('--ckpt',       default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'))
    parser.add_argument('--data_dir',   default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--out_dir',    default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--n_iou',      type=int, default=100)
    parser.add_argument('--opt_steps',  type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = TopoShapeNet(n_classes=N_CLASSES).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    base_state = {k: v for k, v in state.items() if not k.startswith('dropout')}
    model.load_state_dict(base_state, strict=True)
    model.eval()
    print(f"Loaded: {args.ckpt} (epoch {ckpt.get('epoch','?')}, "
          f"val_acc={ckpt.get('val_acc','?')})")

    # Load data
    val_shards = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.npz')))
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")
    val_ds = TopoDatasetV5(val_shards)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    # Classification + regression metrics
    print("\n=== Classification + Shape Regression ===")
    clf = eval_classification(model, val_loader, device)
    print(f"Top-1: {clf['top1']*100:.2f}%  Top-5: {clf['top5']*100:.2f}%  "
          f"Shape MAE: {clf['shape_mae']:.4f}")

    # IoU evaluation
    print(f"\n=== IoU evaluation ({args.n_iou} samples) ===")
    ctx = dr.RasterizeCudaContext()
    iou = eval_iou(model, val_ds, ctx, device,
                   n_samples=args.n_iou, opt_steps=args.opt_steps)
    print(f"\nV5 best1={iou['mean_v5_best1']:.4f}  "
          f"V5 best5={iou['mean_v5_best5']:.4f}  "
          f"V4 best1={iou['mean_v4_best1']:.4f}")

    # Write report
    os.makedirs(args.out_dir, exist_ok=True)
    write_report(os.path.join(args.out_dir, 'results_v5.md'),
                 clf, iou, args.ckpt, args.n_iou)


if __name__ == '__main__':
    main()
