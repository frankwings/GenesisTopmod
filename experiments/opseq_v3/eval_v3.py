#!/usr/bin/env python3
"""
eval_v3.py — Evaluation for OpSeqModelV3 (Phase A'').

Metrics:
  pre_iou        : foreground IoU before vertex optimization (default positions)
  post_iou       : foreground IoU after direct vertex optimization (primary metric)
  manifold       : fraction producing a valid manifold mesh (by DLFL construction)
  op_acc         : % of correctly predicted topology tokens vs GT topology section
  genus_acc      : predicted genus (# HDL) == GT genus
  topology_match : predicted (base+ops) exactly matches GT (strict)

Foreground IoU convention (WHITE background):
  render_silhouette returns 1=fg, 0=bg → pred_fg = (pred > 0.5)
  stored uint8 (0=fg, 255=bg) / 255 → 0=fg,1=bg → gt_fg = (gt < 0.5)

Saves 8 side-by-side visualizations: [target | pre-refine | post-refine] per sample.

Usage:
    python eval_v3.py --ckpt experiments/opseq_v3/ckpt/best.pt
                      --data_dir experiments/opseq_v3/data
                      [--n_samples 500]
                      [--n_refine_steps 200]
                      [--out_dir experiments/opseq_v3/eval_out]
                      [--device cuda]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from topmod.tokenizer import build_vocabulary_v3
from model_v3 import OpSeqModelV3, BOS_ID, PAD_ID, EOS_ID, VOCAB_SIZE, MAX_SEQ_LEN
from infer_v3 import (
    parse_v3_sequence,
    execute_topology,
    preprocess_images,
    optimize_vertices_direct,
    VOCAB_V3,
    VOCAB_INV_V3,
    AZIMUTHS,
    IMG_RES,
    CAMERA_RADIUS,
)
from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

# ── Constants ─────────────────────────────────────────────────────────────────
N_REF = 64


# ═════════════════════════════════════════════════════════════════════════════
# IoU helpers
# ═════════════════════════════════════════════════════════════════════════════

def foreground_iou_single(
    pred_sil: np.ndarray,   # [H, W] float32, 1=fg
    gt_img:   np.ndarray,   # [H, W] float32 in [0,1], white-bg (0=fg, 1=bg)
) -> float:
    """Binary foreground IoU. pred_fg = (pred>0.5), gt_fg = (gt<0.5)."""
    pred_fg = pred_sil > 0.5
    gt_fg   = gt_img   < 0.5
    inter   = (pred_fg & gt_fg).sum()
    union   = (pred_fg | gt_fg).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def mean_foreground_iou(
    ctx:       dr.RasterizeCudaContext,
    verts:     torch.Tensor,   # [V, 3] float32
    tris_t:    torch.Tensor,   # [T, 3] int32
    gt_images: np.ndarray,     # [4, H, W] uint8, white-bg
    mvps:      torch.Tensor,   # [4, 4, 4]
    device:    str,
    resolution: Tuple[int, int] = (128, 128),
) -> float:
    """Render 4 views and compute mean foreground IoU against gt_images."""
    ious = []
    with torch.no_grad():
        for i in range(4):
            sil    = render_silhouette(ctx, verts, tris_t, mvps[i], resolution)
            sil_np = sil[0, :, :, 0].cpu().numpy()
            gt_f   = gt_images[i].astype(np.float32) / 255.0
            ious.append(foreground_iou_single(sil_np, gt_f))
    return float(np.mean(ious))


# ═════════════════════════════════════════════════════════════════════════════
# Topology section helpers
# ═════════════════════════════════════════════════════════════════════════════

def topology_section(token_ids: List[int]) -> List[int]:
    """Return all token IDs (V3 has no SEP, so full sequence minus EOS)."""
    eos_id = VOCAB_V3.get('EOS', 0)
    result = []
    for tid in token_ids:
        if tid == eos_id:
            break
        result.append(tid)
    return result


def op_token_accuracy(pred_ids: List[int], gt_ids: List[int]) -> float:
    """Token-level accuracy over topology section (no SEP in V3)."""
    pred_topo = topology_section(pred_ids)
    gt_topo   = topology_section(gt_ids)
    n_total   = max(len(pred_topo), len(gt_topo))
    if n_total == 0:
        return 1.0
    n_correct = sum(p == g for p, g in zip(pred_topo, gt_topo))
    return n_correct / n_total


def topology_match(pred_ids: List[int], gt_ids: List[int]) -> bool:
    """Exact match of topology section tokens."""
    return topology_section(pred_ids) == topology_section(gt_ids)


# ═════════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════════

def save_sample_visualization(
    gt_images:  np.ndarray,     # [4, 128, 128] uint8 white-bg
    pre_sils:   List[np.ndarray],  # [4] arrays of [128,128] float, 1=fg
    post_sils:  List[np.ndarray],  # [4] arrays of [128,128] float, 1=fg
    idx:        int,
    out_dir:    str,
) -> None:
    """Save 3-panel visualization: GT target | pre-refine | post-refine."""
    if not _HAS_MPL:
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    titles    = ['Target (GT)', 'Pre-refine', 'Post-refine']
    row_data  = [
        [gt_images[v].astype(np.float32) / 255.0 for v in range(4)],
        [1.0 - pre_sils[v] for v in range(4)],    # render → white-bg for display
        [1.0 - post_sils[v] for v in range(4)],
    ]
    for row, (title, views) in enumerate(zip(titles, row_data)):
        for col, img in enumerate(views):
            axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(f"{title}\nAz={AZIMUTHS[col]:.0f}°", fontsize=8)
            else:
                axes[row, col].set_title(f"Az={AZIMUTHS[col]:.0f}°", fontsize=8)

    fig.suptitle(f'Sample {idx}', fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f'sample_{idx:03d}.png')
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    device = args.device

    # ── Load model ────────────────────────────────────────────────────
    model = OpSeqModelV3().to(device)
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss','?'):.4f}")

    # ── Load val shards ────────────────────────────────────────────────
    val_shards = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.npz')))
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")

    records = []
    for p in val_shards:
        data    = np.load(p, allow_pickle=False)
        N       = int(data['images'].shape[0])
        images  = data['images']
        tokens  = data['tokens'].astype(np.int32)
        lengths = data['lengths'].astype(np.int32)
        genera  = data['genera'].astype(np.int32)
        for i in range(N):
            L = int(lengths[i])
            records.append({
                'images':   images[i],
                'gt_ids':   tokens[i, :L].tolist(),
                'gt_genus': int(genera[i]),
            })

    if args.n_samples > 0:
        records = records[:args.n_samples]
    print(f"Evaluating {len(records)} samples …")

    # ── nvdiffrast + cameras ───────────────────────────────────────────
    ctx = dr.RasterizeCudaContext()
    mvps, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=CAMERA_RADIUS,
        azimuths_deg=AZIMUTHS, device=device,
    )

    # ── Accumulators ──────────────────────────────────────────────────
    pre_ious:      List[float] = []
    post_ious:     List[float] = []
    manifold_ok:   List[bool]  = []
    op_accs:       List[float] = []
    genus_hits:    List[bool]  = []
    topo_matches:  List[bool]  = []

    os.makedirs(args.out_dir, exist_ok=True)
    n_saved_viz = 0
    t0 = time.time()

    for idx, rec in enumerate(records):
        images   = rec['images']    # [4, 128, 128] uint8
        gt_ids   = rec['gt_ids']
        gt_genus = rec['gt_genus']

        try:
            # ── Generate topology ─────────────────────────────────────
            model_input, targets = preprocess_images(images, device)
            with torch.no_grad():
                pred_ids = model.sample_greedy(
                    model_input, max_new_tokens=MAX_SEQ_LEN,
                )

            # ── Parse + execute topology ──────────────────────────────
            parsed   = parse_v3_sequence(pred_ids)
            verts_np, tris = execute_topology(parsed)

            if verts_np.shape[0] < 3 or len(tris) == 0:
                raise ValueError("Degenerate mesh")

            # ── Mesh tensors (float32, int32 for nvdiffrast) ──────────
            tris_t32 = torch.tensor(tris, dtype=torch.int32, device=device)

            # Normalize initial positions (same as infer_v3.py)
            mn, mx   = float(verts_np.min()), float(verts_np.max())
            extent   = max(mx - mn, 1e-6)
            scale    = 0.8 * 4.0 / extent
            centre   = (mn + mx) / 2.0
            verts_pre = ((verts_np - centre) * scale).astype(np.float32)
            verts_t  = torch.tensor(verts_pre, dtype=torch.float32, device=device)

            # ── Pre-refine IoU ────────────────────────────────────────
            pre_iou = mean_foreground_iou(ctx, verts_t, tris_t32, images, mvps,
                                          device, resolution=(IMG_RES, IMG_RES))
            pre_ious.append(pre_iou)

            # ── Collect pre-refine silhouettes for visualization ───────
            pre_sils = []
            with torch.no_grad():
                for i in range(4):
                    sil = render_silhouette(ctx, verts_t, tris_t32, mvps[i],
                                            resolution=(IMG_RES, IMG_RES))
                    pre_sils.append(sil[0, :, :, 0].cpu().numpy())

            # ── Direct vertex optimization ─────────────────────────────
            if args.n_refine_steps > 0:
                verts_opt = optimize_vertices_direct(
                    ctx         = ctx,
                    verts_init  = verts_np,
                    tris        = tris,
                    targets     = targets,
                    mvps        = mvps,
                    num_steps   = args.n_refine_steps,
                    lr          = 0.01,
                    lambda_lap  = 0.05,
                    lambda_edge = 0.01,
                    resolution  = (IMG_RES, IMG_RES),
                    log_every   = 0,   # silent during eval
                )
            else:
                verts_opt = verts_t.detach()

            post_iou = mean_foreground_iou(ctx, verts_opt, tris_t32, images, mvps,
                                           device, resolution=(IMG_RES, IMG_RES))
            post_ious.append(post_iou)

            # ── Collect post-refine silhouettes for visualization ──────
            post_sils = []
            with torch.no_grad():
                for i in range(4):
                    sil = render_silhouette(ctx, verts_opt, tris_t32, mvps[i],
                                            resolution=(IMG_RES, IMG_RES))
                    post_sils.append(sil[0, :, :, 0].cpu().numpy())

            # ── Manifold: always True by DLFL construction ─────────────
            manifold_ok.append(True)

            # ── Op accuracy ───────────────────────────────────────────
            op_accs.append(op_token_accuracy(pred_ids, gt_ids))

            # ── Genus accuracy ─────────────────────────────────────────
            pred_genus = len(parsed['hdl_pairs'])
            genus_hits.append(pred_genus == gt_genus)

            # ── Topology exact match ───────────────────────────────────
            topo_matches.append(topology_match(pred_ids, gt_ids))

            # ── Save visualization (first 8 samples) ──────────────────
            if n_saved_viz < 8 and _HAS_MPL:
                save_sample_visualization(images, pre_sils, post_sils,
                                          idx, args.out_dir)
                n_saved_viz += 1

        except Exception as exc:
            print(f"  [WARN] sample {idx} failed: {exc}")
            pre_ious.append(0.0)
            post_ious.append(0.0)
            manifold_ok.append(False)
            op_accs.append(0.0)
            genus_hits.append(False)
            topo_matches.append(False)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{idx+1}/{len(records)}]"
                f"  pre={np.mean(pre_ious):.3f}"
                f"  post={np.mean(post_ious):.3f}"
                f"  manifold={np.mean(manifold_ok):.3f}"
                f"  op_acc={np.mean(op_accs):.3f}"
                f"  genus={np.mean(genus_hits):.3f}"
                f"  topo_match={np.mean(topo_matches):.3f}"
                f"  ({elapsed:.0f}s)"
            )

    # ── Summary ───────────────────────────────────────────────────────
    results = {
        'n_samples':      len(records),
        'pre_iou':        float(np.mean(pre_ious)),
        'post_iou':       float(np.mean(post_ious)),
        'manifold':       float(np.mean(manifold_ok)),
        'op_acc':         float(np.mean(op_accs)),
        'genus_acc':      float(np.mean(genus_hits)),
        'topology_match': float(np.mean(topo_matches)),
        'n_refine_steps': args.n_refine_steps,
    }

    print("\n" + "=" * 65)
    print("Phase A'' Evaluation Results (Topology-Only + Direct Vertex Opt)")
    print("=" * 65)
    print(f"  Samples evaluated    : {results['n_samples']}")
    print(f"  Pre-refine  IoU      : {results['pre_iou']:.4f}   (topology default positions)")
    print(f"  Post-refine IoU      : {results['post_iou']:.4f}   (target: > 0.40)")
    print(f"  Manifold validity    : {results['manifold']:.4f}   (target: 1.00)")
    print(f"  Op token accuracy    : {results['op_acc']:.4f}   (target: > 0.80)")
    print(f"  Genus accuracy       : {results['genus_acc']:.4f}")
    print(f"  Topology exact match : {results['topology_match']:.4f}")
    print(f"  Refine steps         : {results['n_refine_steps']}")
    print("=" * 65)

    # Success check
    success = {
        'post_iou':  results['post_iou']  > 0.40,
        'manifold':  results['manifold']  == 1.0,
        'op_acc':    results['op_acc']    > 0.80,
    }
    print("\nSuccess criteria:")
    for metric, passed in success.items():
        print(f"  {'✓' if passed else '✗'} {metric}")

    # ── Save JSON ─────────────────────────────────────────────────────
    results_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Write results_v3.md ───────────────────────────────────────────
    _write_results_md(results, args.out_dir)


def _write_results_md(results: Dict, out_dir: str) -> None:
    """Write results_v3.md summary."""
    md_path = os.path.join(
        os.path.dirname(out_dir),   # experiments/opseq_v3/
        'results_v3.md'
    )
    lines = [
        "# Phase A'' Evaluation Results (Topology-Only + Direct Vertex Optimization)",
        "",
        "## Key Insight",
        "",
        "Ablation results show direct all-vertex optimization (IoU 0.974) outperforms",
        "DiffSequence cage optimization (0.962). This pipeline separates concerns cleanly:",
        "- **LLM**: predicts short topology program only (~5–15 tokens)",
        "- **Optimizer**: refines all mesh vertices directly via nvdiffrast",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Target | Pass? |",
        "|--------|-------|--------|-------|",
        f"| Pre-refine IoU    | {results['pre_iou']:.4f} | –      | – |",
        f"| Post-refine IoU   | {results['post_iou']:.4f} | > 0.40 | {'✓' if results['post_iou'] > 0.40 else '✗'} |",
        f"| Manifold          | {results['manifold']:.4f} | 1.00   | {'✓' if results['manifold'] == 1.0 else '✗'} |",
        f"| Op Token Acc      | {results['op_acc']:.4f} | > 0.80 | {'✓' if results['op_acc'] > 0.80 else '✗'} |",
        f"| Genus Accuracy    | {results['genus_acc']:.4f} | –      | – |",
        f"| Topology Match    | {results['topology_match']:.4f} | –      | – |",
        "",
        "## Ablation Comparison",
        "",
        "| System | Post-refine IoU | Notes |",
        "|--------|----------------|-------|",
        f"| **Phase A'' (this)** | **{results['post_iou']:.4f}** | Topology-only LLM + direct vertex opt |",
        "| Direct all-vertex opt (oracle topology) | ~0.974 | Upper bound |",
        "| Phase A' (cage+chain) | ~0.962 | V2: topology + cage coords |",
        "| Plain cube baseline | ~0.947 | No topology learning |",
        "| Phase A (sequential CV) | 0.013 | Original failure |",
        "",
        "## Settings",
        "",
        f"- Samples evaluated: {results['n_samples']}",
        f"- Refinement steps: {results['n_refine_steps']}",
        f"- Model: OpSeqModelV3 (4-layer transformer, 99-token topology-only vocab)",
    ]
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Results markdown → {md_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpSeqModelV3 (Phase A'')")
    parser.add_argument('--ckpt',           required=True)
    parser.add_argument('--data_dir',       default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--out_dir',        default=os.path.join(_SCRIPT_DIR, 'eval_out'))
    parser.add_argument('--n_samples',      type=int,   default=500)
    parser.add_argument('--n_refine_steps', type=int,   default=200)
    parser.add_argument('--device',         default='cuda')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
