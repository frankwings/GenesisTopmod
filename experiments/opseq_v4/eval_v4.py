#!/usr/bin/env python3
"""
eval_v4.py — Evaluation script for the V4 topology classifier.

Metrics reported
----------------
1. Classification: top-1 / top-3 / top-5 accuracy, per-class accuracy
2. IoU evaluation (100 val samples, expensive):
   For each sample the top-5 predicted topologies are each optimised via 500
   steps of direct vertex optimisation with nvdiffrast + adaptive remeshing.
   Reports best-of-1 IoU (using only the top-1 prediction) and best-of-5 IoU
   (best result across all 5 candidate topologies).

   Foreground IoU convention:
     - Rendered silhouette: 1 = foreground  (render_silhouette output)
     - GT images stored as white-bg uint8:  0 = fg, 255 = bg
     pred_fg = rendered > 0.5
     gt_fg   = stored_uint8 < 128  (white-bg: fg pixels are dark)

Results are written to eval_out/results_v4.md next to this script.

Usage:
    python eval_v4.py [--ckpt     experiments/opseq_v4/ckpt/best.pt]
                      [--data_dir experiments/opseq_v4/data]
                      [--out_dir  experiments/opseq_v4/eval_out]
                      [--n_iou    100]
                      [--opt_steps 500]
                      [--batch_size 256]
                      [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Repo / script paths ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from canonical_topos import CANONICAL_TOPOS, N_CLASSES, build_topo_mesh
from model_v4        import TopoClassifier, count_params
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)

# ── Constants ────────────────────────────────────────────────────────────────
AZIMUTHS      = [0.0, 90.0, 180.0, 270.0]
IMG_RES       = 128
CAMERA_RADIUS = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive remeshing helpers (verbatim from spec)
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

class TopoDatasetV4(Dataset):
    """Load v4 .npz shards. Returns (img_float32 [4,128,128], label int64)."""

    def __init__(self, shard_paths: List[str]):
        images_list, labels_list = [], []
        for p in shard_paths:
            d = np.load(p, allow_pickle=False)
            images_list.append(d['images'])
            labels_list.append(d['labels'].astype(np.int64))
        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img   = self.images[idx].astype(np.float32) / 255.0
        label = int(self.labels[idx])
        return img, label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return (torch.from_numpy(np.stack(imgs, 0)),
            torch.tensor(labels, dtype=torch.long))


# ═════════════════════════════════════════════════════════════════════════════
# Classification evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_classification(
    model:       TopoClassifier,
    loader:      DataLoader,
    device:      str,
) -> Dict:
    """
    Compute top-1/3/5 accuracy and per-class accuracy over the full val set.

    Returns a dict with keys:
        top1, top3, top5            : float (0..1)
        per_class_acc               : list[float] length N_CLASSES
        per_class_correct / total   : list[int]
    """
    model.eval()
    per_class_correct = [0] * N_CLASSES
    per_class_total   = [0] * N_CLASSES
    top1_correct = top3_correct = top5_correct = total = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(imgs)

        B = labels.shape[0]
        total += B

        # top-k
        _, topk_idx = logits.topk(5, dim=1)  # [B, 5]
        match = topk_idx.eq(labels.unsqueeze(1))
        top1_correct += match[:, :1].any(dim=1).sum().item()
        top3_correct += match[:, :3].any(dim=1).sum().item()
        top5_correct += match[:, :5].any(dim=1).sum().item()

        # per-class
        preds = logits.argmax(dim=1)
        for b in range(B):
            gt = int(labels[b].item())
            per_class_total[gt]   += 1
            per_class_correct[gt] += int(preds[b].item() == gt)

    per_class_acc = [
        per_class_correct[c] / max(per_class_total[c], 1)
        for c in range(N_CLASSES)
    ]

    return {
        'top1':              top1_correct / max(total, 1),
        'top3':              top3_correct / max(total, 1),
        'top5':              top5_correct / max(total, 1),
        'per_class_acc':     per_class_acc,
        'per_class_correct': per_class_correct,
        'per_class_total':   per_class_total,
        'n_samples':         total,
    }


# ═════════════════════════════════════════════════════════════════════════════
# IoU helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_iou(pred_sil: np.ndarray, gt_uint8: np.ndarray) -> float:
    """
    Foreground IoU between a rendered silhouette and a white-bg gt image.

    pred_sil : [4, H, W] float32  (render output, 1=fg, 0=bg)
    gt_uint8 : [4, H, W] uint8   (white-bg: 0=fg, 255=bg)

    IoU is averaged over the 4 views.
    """
    pred_fg = pred_sil > 0.5                   # [4, H, W] bool
    gt_fg   = gt_uint8 < 128                   # [4, H, W] bool  (fg=dark)

    ious = []
    for v in range(4):
        inter = (pred_fg[v] & gt_fg[v]).sum()
        union = (pred_fg[v] | gt_fg[v]).sum()
        ious.append(float(inter) / max(float(union), 1.0))
    return float(np.mean(ious))


def optimize_topology_for_iou(
    ctx:        dr.RasterizeCudaContext,
    class_id:   int,
    gt_uint8:   np.ndarray,   # [4, H, W] uint8
    mvps:       torch.Tensor, # [4, 4, 4]
    device:     str,
    n_steps:    int = 500,
) -> float:
    """
    Build the topology for *class_id*, run 500-step vertex optimisation with
    adaptive remeshing, return foreground IoU.

    Returns final mean IoU (averaged across 4 views).
    """
    topo = CANONICAL_TOPOS[class_id]
    try:
        verts_np, tris_np = build_topo_mesh(topo)
    except Exception:
        return 0.0

    if verts_np.shape[0] < 3 or tris_np.shape[0] == 0:
        return 0.0

    # Normalise initial mesh to [-1.6, 1.6]
    mn, mx = float(verts_np.min()), float(verts_np.max())
    scale  = 3.2 / max(mx - mn, 1e-6)
    verts_np = (verts_np - (mn + mx) / 2.0) * scale

    # GT targets in 1=fg space: shape [4, H, W, 1]
    gt_fg_f = (gt_uint8 < 128).astype(np.float32)   # [4, H, W]
    targets = torch.from_numpy(gt_fg_f).unsqueeze(-1).to(device)  # [4,H,W,1]

    # Adaptive remeshing (once before optimisation)
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
                                         resolution=(IMG_RES, IMG_RES))  # [1,H,W,1]
            sil_loss = sil_loss + F.l1_loss(rendered, targets[i:i+1])
        sil_loss = sil_loss / N_VIEWS

        reg = 0.05 * laplacian_loss(verts_t, faces_t) + \
              0.01 * edge_length_loss(verts_t, faces_t)

        (sil_loss + reg).backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

    # Render final silhouettes and compute IoU
    with torch.no_grad():
        rendered_views = []
        for i in range(N_VIEWS):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i],
                                    resolution=(IMG_RES, IMG_RES))
            rendered_views.append(sil[0, :, :, 0].cpu().numpy())  # [H, W]
    pred_sil = np.stack(rendered_views, axis=0)  # [4, H, W]
    return compute_iou(pred_sil, gt_uint8)


# ═════════════════════════════════════════════════════════════════════════════
# IoU evaluation (top-5 candidates)
# ═════════════════════════════════════════════════════════════════════════════

def eval_iou(
    model:      TopoClassifier,
    dataset:    TopoDatasetV4,
    ctx:        dr.RasterizeCudaContext,
    device:     str,
    n_samples:  int = 100,
    opt_steps:  int = 500,
) -> Dict:
    """
    For *n_samples* val samples:
      1. Get model top-5 predictions.
      2. Optimise each of the 5 topologies (500 steps).
      3. Record best-of-1 IoU (top-1 pred only) and best-of-5 IoU.

    Returns dict with keys:
        best1_ious : list[float]  — IoU using only top-1 prediction
        best5_ious : list[float]  — IoU using best of top-5 predictions
        mean_best1 : float
        mean_best5 : float
    """
    model.eval()
    mvps, _ = orbit_cameras(4, elevation_deg=0.0, radius=CAMERA_RADIUS,
                             azimuths_deg=AZIMUTHS, device=device)

    best1_ious: List[float] = []
    best5_ious: List[float] = []

    n_samples = min(n_samples, len(dataset))
    rng = np.random.default_rng(1234)
    indices = rng.choice(len(dataset), size=n_samples, replace=False).tolist()

    for rank, idx in enumerate(indices):
        img_f, gt_label = dataset[idx]
        gt_uint8 = (img_f * 255.0).astype(np.uint8)  # [4, H, W] uint8

        img_t = torch.from_numpy(img_f).unsqueeze(0).to(device)  # [1, 4, H, W]
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(img_t)
            _, top5 = logits[0].topk(5)
            top5 = top5.cpu().tolist()

        print(f"  IoU sample {rank+1}/{n_samples}  gt={gt_label}  "
              f"top5={top5}")

        # Optimise top-1 only
        iou1 = optimize_topology_for_iou(ctx, top5[0], gt_uint8, mvps, device, opt_steps)
        best1_ious.append(iou1)

        # Optimise all top-5
        ious_5 = [iou1]
        for cid in top5[1:]:
            iou = optimize_topology_for_iou(ctx, cid, gt_uint8, mvps, device, opt_steps)
            ious_5.append(iou)
        best5_ious.append(max(ious_5))

        print(f"    best-of-1={iou1:.4f}  best-of-5={best5_ious[-1]:.4f}")

    return {
        'best1_ious': best1_ious,
        'best5_ious': best5_ious,
        'mean_best1': float(np.mean(best1_ious)),
        'mean_best5': float(np.mean(best5_ious)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report writer
# ═════════════════════════════════════════════════════════════════════════════

def write_report(
    out_path:    str,
    clf_results: Dict,
    iou_results: Dict,
    ckpt_path:   str,
    n_iou:       int,
) -> None:
    lines = [
        "# V4 Topology Classifier — Evaluation Results",
        "",
        f"**Checkpoint:** `{ckpt_path}`",
        f"**Val samples (classification):** {clf_results['n_samples']}",
        f"**Val samples (IoU eval):** {n_iou}",
        "",
        "## Classification Accuracy",
        "",
        f"| Metric   | Value  |",
        f"|----------|--------|",
        f"| Top-1    | {clf_results['top1']*100:.2f}% |",
        f"| Top-3    | {clf_results['top3']*100:.2f}% |",
        f"| Top-5    | {clf_results['top5']*100:.2f}% |",
        "",
        "### Per-Class Accuracy",
        "",
        "| Class | Label | Correct | Total | Acc |",
        "|-------|-------|---------|-------|-----|",
    ]

    from canonical_topos import CANONICAL_TOPOS
    for c in range(N_CLASSES):
        entry   = CANONICAL_TOPOS[c]
        correct = clf_results['per_class_correct'][c]
        total   = clf_results['per_class_total'][c]
        acc     = clf_results['per_class_acc'][c]
        lines.append(
            f"| {c:2d} | {entry['label']:25s} | {correct:5d} | {total:5d} "
            f"| {acc*100:.1f}% |"
        )

    lines += [
        "",
        "## IoU Evaluation (500-step vertex optimisation)",
        "",
        f"| Metric        | Mean IoU |",
        f"|---------------|----------|",
        f"| Best-of-1     | {iou_results['mean_best1']:.4f} |",
        f"| Best-of-5     | {iou_results['mean_best5']:.4f} |",
        "",
        "### Per-Sample IoU",
        "",
        "| Sample | Best-of-1 | Best-of-5 |",
        "|--------|-----------|-----------|",
    ]

    for i, (b1, b5) in enumerate(
        zip(iou_results['best1_ious'], iou_results['best5_ious'])
    ):
        lines.append(f"| {i+1:6d} | {b1:.4f}    | {b5:.4f}    |")

    lines.append("")
    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(content)
    print(f"\nResults written to {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V4 TopoClassifier")
    parser.add_argument('--ckpt',       default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'))
    parser.add_argument('--data_dir',   default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--out_dir',    default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--n_iou',      type=int,   default=100,
                        help='Number of val samples for IoU evaluation (expensive)')
    parser.add_argument('--opt_steps',  type=int,   default=500,
                        help='Optimisation steps per candidate topology')
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = TopoClassifier(n_classes=N_CLASSES).to(device)
    # Handle checkpoints saved from TopoClassifierWithDropout wrapper
    # (same encoder/head keys, just with an extra 'dropout.*' prefix stripped)
    state = ckpt.get('model_state_dict', ckpt)
    # Strip the 'dropout.*' entries (not part of base TopoClassifier)
    base_state = {k: v for k, v in state.items() if not k.startswith('dropout')}
    model.load_state_dict(base_state, strict=True)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}  "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', '?')})")
    print(f"Parameters: {count_params(model):,}")

    # ── Load val dataset ─────────────────────────────────────────────────────
    val_shards = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.npz')))
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")

    val_ds = TopoDatasetV4(val_shards)
    print(f"Val dataset: {len(val_ds)} samples")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    # ── Classification metrics ───────────────────────────────────────────────
    print("\n=== Classification evaluation ===")
    t0 = time.time()
    clf_results = eval_classification(model, val_loader, device)
    print(f"Top-1 acc: {clf_results['top1']*100:.2f}%")
    print(f"Top-3 acc: {clf_results['top3']*100:.2f}%")
    print(f"Top-5 acc: {clf_results['top5']*100:.2f}%")
    print(f"Classification eval done in {time.time()-t0:.1f}s")

    # ── IoU evaluation ───────────────────────────────────────────────────────
    print(f"\n=== IoU evaluation ({args.n_iou} samples, "
          f"{args.opt_steps} opt steps each) ===")
    ctx = dr.RasterizeCudaContext()
    t0  = time.time()
    iou_results = eval_iou(
        model, val_ds, ctx, device,
        n_samples=args.n_iou,
        opt_steps=args.opt_steps,
    )
    print(f"\nMean best-of-1 IoU: {iou_results['mean_best1']:.4f}")
    print(f"Mean best-of-5 IoU: {iou_results['mean_best5']:.4f}")
    print(f"IoU eval done in {time.time()-t0:.1f}s")

    # ── Write report ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, 'results_v4.md')
    write_report(report_path, clf_results, iou_results, args.ckpt, args.n_iou)


if __name__ == '__main__':
    main()
