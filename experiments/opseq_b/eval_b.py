#!/usr/bin/env python3
"""
eval_b.py — Evaluate a trained Phase B OpSeqModel.

Metrics
-------
  (i)   Manifold validity rate (greedy decode)
  (ii)  Token accuracy + exact-match rate
  (iii) Silhouette IoU vs conditioning images (4 views, 128×128)
  (iv)  Genus accuracy
  (v)   Mean distillation IoU (from distill_log.jsonl)
  (vi)  Rejection rate (from distill_log.jsonl)
  (vii) Comparison table vs MeshGPT published numbers

Also saves 8 side-by-side sample PNGs (target 4 views | generated mesh 4 views)
to samples/sample_NNN.png.

Results written to experiments/opseq_b/results.md.

Usage:
    python3 eval_b.py [--checkpoint PATH] [--data_dir PATH]
                      [--distill_log PATH] [--n_eval 100]
                      [--samples_dir PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'opseq'))

import nvdiffrast.torch as dr

from topmod.validate  import is_manifold
from topmod.tokenizer import build_vocabulary, decode_sequence, detokenize
from topmod.io        import to_triangle_arrays

from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

from model import OpSeqModel, count_params, BOS_ID, PAD_ID, EOS_ID, VOCAB_SIZE

# ── Vocab ──────────────────────────────────────────────────────────────────────
VOCAB     = build_vocabulary(n_position_bins=128, max_ordinal=128)
VOCAB_INV = {v: k for k, v in VOCAB.items()}

AZIMUTHS   = [0.0, 90.0, 180.0, 270.0]
IMG_RES    = 128
CAM_RADIUS = 3.0

MAX_SEQ_LEN_B = 5000


# ═════════════════════════════════════════════════════════════════════════════
# Mesh rendering helper
# ═════════════════════════════════════════════════════════════════════════════

def _mesh_to_tensors(mesh, device: str = 'cuda'):
    """DLFLMesh → (verts [V,3], faces [F,3]) float32/int32 tensors."""
    verts_list = list(mesh.vertices.values())
    vid_map    = {v.id: i for i, v in enumerate(verts_list)}
    verts_t = torch.tensor(
        [(v.x, v.y, v.z) for v in verts_list],
        dtype=torch.float32, device=device,
    )
    tris = []
    for f in mesh.faces.values():
        vs = list(f.vertices())
        n  = len(vs)
        if n < 3:
            continue
        elif n == 3:
            tris.append([vid_map[vs[0].id], vid_map[vs[1].id], vid_map[vs[2].id]])
        elif n == 4:
            tris.append([vid_map[vs[0].id], vid_map[vs[1].id], vid_map[vs[2].id]])
            tris.append([vid_map[vs[0].id], vid_map[vs[2].id], vid_map[vs[3].id]])
        else:
            for i in range(1, n - 1):
                tris.append([vid_map[vs[0].id], vid_map[vs[i].id], vid_map[vs[i+1].id]])
    faces_t = torch.tensor(tris, dtype=torch.int32, device=device)
    return verts_t, faces_t


def compute_silhouette_iou(
    ctx:       dr.RasterizeCudaContext,
    mesh,
    gt_images: np.ndarray,   # [4, H, W] uint8
    device:    str = 'cuda',
) -> float:
    """Render mesh from 4 cond views, compute mean binary IoU vs gt_images."""
    try:
        verts_t, faces_t = _mesh_to_tensors(mesh, device)
        if faces_t.shape[0] == 0:
            return 0.0

        mvps, _ = orbit_cameras(
            4, elevation_deg=0.0, radius=CAM_RADIUS,
            azimuths_deg=AZIMUTHS, device=device,
        )
        gt_f = torch.from_numpy(gt_images.astype(np.float32) / 255.0).to(device)

        ious: List[float] = []
        for i in range(4):
            pred_sil = render_silhouette(
                ctx, verts_t, faces_t, mvps[i],
                resolution=(IMG_RES, IMG_RES),
            )
            pred = pred_sil[0, :, :, 0]
            gt   = gt_f[i]

            pred_bin = (pred > 0.5).float()
            gt_bin   = (gt   > 0.5).float()
            inter    = (pred_bin * gt_bin).sum().item()
            union    = ((pred_bin + gt_bin) > 0).float().sum().item()
            ious.append(inter / max(union, 1.0))

        return float(np.mean(ious))
    except Exception:
        return 0.0


def render_mesh_views(
    ctx:     dr.RasterizeCudaContext,
    mesh,
    device:  str,
    res:     int = 128,
) -> Optional[np.ndarray]:
    """
    Render mesh from 4 conditioning views.
    Returns [4, H, W] uint8 or None on error.
    """
    try:
        verts_t, faces_t = _mesh_to_tensors(mesh, device)
        if faces_t.shape[0] == 0:
            return None
        mvps, _ = orbit_cameras(
            4, elevation_deg=0.0, radius=CAM_RADIUS,
            azimuths_deg=AZIMUTHS, device=device,
        )
        views = []
        for i in range(4):
            sil = render_silhouette(ctx, verts_t, faces_t, mvps[i], (res, res))
            arr = (sil[0, :, :, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            views.append(arr)
        return np.stack(views, axis=0)   # [4, H, W]
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 2×2 grid image helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_2x2_grid(views: np.ndarray) -> np.ndarray:
    """
    views: [4, H, W] uint8 grayscale
    Returns [2H, 2W] uint8 2×2 grid.
    """
    H, W = views.shape[1], views.shape[2]
    grid = np.zeros((2 * H, 2 * W), dtype=np.uint8)
    grid[:H,  :W]  = views[0]
    grid[:H,  W:]  = views[1]
    grid[H:,  :W]  = views[2]
    grid[H:,  W:]  = views[3]
    return grid


def save_side_by_side(
    target_views:    np.ndarray,   # [4, H, W] uint8
    generated_views: np.ndarray,   # [4, H, W] uint8  or None
    out_path:        str,
    idx:             int,
    iou:             float,
    genus:           int,
) -> None:
    """Save side-by-side PNG: left = target 2×2, right = generated 2×2."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        H, W = target_views.shape[1], target_views.shape[2]

        left  = _make_2x2_grid(target_views)
        if generated_views is not None:
            right = _make_2x2_grid(generated_views)
        else:
            right = np.zeros_like(left)

        combined = np.concatenate([left, right], axis=1)   # [2H, 4W]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.imshow(combined, cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f"Sample {idx}  |  genus={genus}  sil_iou={iou:.3f}\n"
            f"Left: target (4 views)    Right: generated (4 views)"
        )
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        # Fallback: raw numpy save without matplotlib
        try:
            from PIL import Image
            combined_pil = Image.fromarray(combined, mode='L')
            combined_pil.save(out_path)
        except Exception:
            pass   # silently skip if PIL also unavailable


# ═════════════════════════════════════════════════════════════════════════════
# Distill log stats
# ═════════════════════════════════════════════════════════════════════════════

def load_distill_stats(log_path: str) -> dict:
    """
    Parse distill_log.jsonl and return:
      mean_distill_iou, rejection_rate, n_total, n_accepted, n_rejected
    """
    if not os.path.exists(log_path):
        return {
            'mean_distill_iou': None,
            'rejection_rate':   None,
            'n_total':          0,
            'n_accepted':       0,
            'n_rejected':       0,
        }

    ious: List[float] = []
    n_accepted = 0
    n_rejected = 0

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            status = entry.get('status', '')
            if status == 'accepted':
                n_accepted += 1
                if 'iou' in entry:
                    ious.append(float(entry['iou']))
            elif status == 'rejected':
                n_rejected += 1

    n_total = n_accepted + n_rejected
    return {
        'mean_distill_iou': float(np.mean(ious)) if ious else None,
        'rejection_rate':   n_rejected / max(n_total, 1) * 100,
        'n_total':          n_total,
        'n_accepted':       n_accepted,
        'n_rejected':       n_rejected,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Val data loader
# ═════════════════════════════════════════════════════════════════════════════

def load_val_records(shard_paths: List[str]) -> List[dict]:
    records = []
    for p in shard_paths:
        data    = np.load(p, allow_pickle=False)
        N       = int(data['images'].shape[0])
        images  = data['images']
        tokens  = data['tokens']
        lengths = data['lengths']
        genera  = data['genera']
        for i in range(N):
            L = int(lengths[i])
            records.append({
                'image':  images[i],
                'tokens': tokens[i, :L].astype(np.int64),
                'length': L,
                'genus':  int(genera[i]),
            })
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.samples_dir, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = OpSeqModel(max_seq_len=MAX_SEQ_LEN_B).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Parameters  : {count_params(model):,}")
    best_epoch = ckpt.get('epoch', '?')
    best_val   = ckpt.get('val_loss', float('nan'))
    ckpt_label = ckpt.get('label', 'unknown')
    print(f"Saved at epoch {best_epoch}, val_loss={best_val:.4f}, label={ckpt_label}")

    # ── Load val data ──────────────────────────────────────────────────
    val_shards = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.npz')))
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")
    records = load_val_records(val_shards)
    n_eval  = min(args.n_eval, len(records))
    print(f"Evaluating on {n_eval} / {len(records)} val samples …")

    ctx = dr.RasterizeCudaContext()

    # ── Distill log stats ──────────────────────────────────────────────
    distill_stats = load_distill_stats(args.distill_log)

    # ── Metrics ────────────────────────────────────────────────────────
    n_manifold    = 0
    n_parse_fail  = 0
    n_genus_ok    = 0
    token_accs:   List[float] = []
    exact_matches = 0
    sil_ious:     List[float] = []

    n_samples_saved = 0
    N_SAMPLES_TO_SAVE = 8

    t0 = time.time()

    for idx in range(n_eval):
        r      = records[idx]
        img_np = r['image']        # [4, 128, 128] uint8
        gt_ids = r['tokens']       # [L] int64
        L      = r['length']
        genus  = r['genus']

        img_t = torch.from_numpy(
            img_np.astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)   # [1, 4, 128, 128]

        # ── Greedy decode ───────────────────────────────────────────────
        generated_ids = model.sample_greedy(img_t, max_new_tokens=MAX_SEQ_LEN_B)

        # ── Attempt detokenize ──────────────────────────────────────────
        try:
            toks = decode_sequence(generated_ids, VOCAB_INV)
            mesh = detokenize(toks)
        except Exception as exc:
            n_parse_fail += 1
            if n_parse_fail <= 5:
                print(f"  [parse fail {idx}] {exc!r}")
            sil_ious.append(0.0)
            token_accs.append(0.0)
            continue

        # Manifold
        if is_manifold(mesh):
            n_manifold += 1

        # Genus
        try:
            gen_genus = mesh.genus()
        except Exception:
            gen_genus = -999
        if gen_genus == genus:
            n_genus_ok += 1

        # Token accuracy
        min_L   = min(len(generated_ids), L)
        gen_arr = np.array(generated_ids[:min_L])
        gt_arr  = gt_ids[:min_L]
        n_match = int((gen_arr == gt_arr).sum())
        token_accs.append(n_match / L)
        if len(generated_ids) == L and (gen_arr == gt_arr).all():
            exact_matches += 1

        # Silhouette IoU
        iou = compute_silhouette_iou(ctx, mesh, img_np, device)
        sil_ious.append(iou)

        # Save side-by-side sample images
        if n_samples_saved < N_SAMPLES_TO_SAVE:
            gen_views = render_mesh_views(ctx, mesh, device, res=IMG_RES)
            sample_path = os.path.join(
                args.samples_dir, f'sample_{n_samples_saved:03d}.png'
            )
            save_side_by_side(
                target_views=img_np,
                generated_views=gen_views,
                out_path=sample_path,
                idx=idx,
                iou=iou,
                genus=genus,
            )
            n_samples_saved += 1

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n_eval}] manifold={n_manifold}  "
                  f"parse_fail={n_parse_fail}  "
                  f"iou={np.mean(sil_ious):.3f}  ({elapsed:.0f}s)")

    # ── Aggregate ──────────────────────────────────────────────────────
    n_decoded        = n_eval - n_parse_fail
    manifold_rate    = n_manifold    / max(n_decoded, 1) * 100
    genus_acc_rate   = n_genus_ok    / max(n_decoded, 1) * 100
    mean_token_acc   = float(np.mean(token_accs)) * 100 if token_accs else 0.0
    exact_match_rate = exact_matches / n_eval * 100
    mean_sil_iou     = float(np.mean(sil_ious)) if sil_ious else 0.0
    parse_fail_rate  = n_parse_fail  / n_eval * 100
    elapsed          = time.time() - t0

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluation summary  ({n_eval} greedy samples, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"(i)   Manifold validity : {manifold_rate:.1f}%  "
          f"({n_manifold}/{n_decoded} decoded)")
    print(f"      Parse failures    : {n_parse_fail}/{n_eval} "
          f"({parse_fail_rate:.1f}%)")
    print(f"(ii)  Token accuracy    : {mean_token_acc:.1f}%")
    print(f"      Exact match       : {exact_match_rate:.1f}%  "
          f"({exact_matches}/{n_eval})")
    print(f"(iii) Silhouette IoU    : {mean_sil_iou:.4f}")
    print(f"(iv)  Genus accuracy    : {genus_acc_rate:.1f}%  "
          f"({n_genus_ok}/{n_decoded})")

    if distill_stats['mean_distill_iou'] is not None:
        print(f"(v)   Mean distill IoU  : {distill_stats['mean_distill_iou']:.4f}")
    else:
        print(f"(v)   Mean distill IoU  : N/A (distill_log not found)")
    if distill_stats['rejection_rate'] is not None:
        print(f"(vi)  Rejection rate    : {distill_stats['rejection_rate']:.1f}%  "
              f"({distill_stats['n_rejected']}/{distill_stats['n_total']})")
    print(f"{'='*60}")
    print(f"Samples saved: {n_samples_saved} → {args.samples_dir}")

    # ── Write results.md ───────────────────────────────────────────────
    results_path = os.path.join(_SCRIPT_DIR, 'results.md')
    with open(results_path, 'w') as f:
        f.write("# OpSeq Phase B — Evaluation Results\n\n")
        f.write(f"**Checkpoint**: `{args.checkpoint}`\n\n")
        f.write(f"**Run type**: {ckpt_label}\n\n")
        f.write(f"**Epoch**: {best_epoch}  |  **Best val loss**: {best_val:.4f}\n\n")
        f.write(f"**Samples evaluated**: {n_eval}  |  "
                f"**Eval time**: {elapsed:.1f}s\n\n")
        f.write(f"**Max seq len**: {MAX_SEQ_LEN_B} "
                f"(genus=0: ~3852, genus=1: ~4231, genus=2: ~4610 tokens)\n\n")

        f.write("## Greedy Sampling Metrics\n\n")
        f.write("| # | Metric | Value |\n")
        f.write("|---|--------|-------|\n")
        f.write(f"| (i) | **Manifold validity rate** | "
                f"**{manifold_rate:.1f}%** ({n_manifold}/{n_decoded} decoded) |\n")
        f.write(f"|     | Parse failures | "
                f"{n_parse_fail}/{n_eval} ({parse_fail_rate:.1f}%) |\n")
        f.write(f"| (ii) | **Token accuracy** | **{mean_token_acc:.1f}%** |\n")
        f.write(f"|      | Exact-match rate | "
                f"{exact_match_rate:.1f}% ({exact_matches}/{n_eval}) |\n")
        f.write(f"| (iii) | **Silhouette IoU** | **{mean_sil_iou:.4f}** |\n")
        f.write(f"| (iv) | **Genus accuracy** | "
                f"**{genus_acc_rate:.1f}%** ({n_genus_ok}/{n_decoded}) |\n")

        if distill_stats['mean_distill_iou'] is not None:
            f.write(f"| (v) | **Mean distill IoU** | "
                    f"**{distill_stats['mean_distill_iou']:.4f}** |\n")
        else:
            f.write(f"| (v) | **Mean distill IoU** | N/A |\n")

        if distill_stats['rejection_rate'] is not None:
            f.write(f"| (vi) | **Rejection rate** | "
                    f"**{distill_stats['rejection_rate']:.1f}%** "
                    f"({distill_stats['n_rejected']}/{distill_stats['n_total']}) |\n\n")
        else:
            f.write(f"| (vi) | **Rejection rate** | N/A |\n\n")

        f.write("## Distillation Pipeline Stats\n\n")
        f.write("| Stat | Value |\n")
        f.write("|------|-------|\n")
        f.write(f"| Total attempted | {distill_stats['n_total']} |\n")
        f.write(f"| Accepted | {distill_stats['n_accepted']} |\n")
        f.write(f"| Rejected | {distill_stats['n_rejected']} |\n")
        iou_str = (
            f"{distill_stats['mean_distill_iou']:.4f}"
            if distill_stats['mean_distill_iou'] is not None
            else "N/A"
        )
        f.write(f"| Mean distill IoU | {iou_str} |\n\n")

        f.write("## Comparison with MeshGPT (Published Numbers)\n\n")
        f.write("> **Important caveat**: This comparison is provided for context only. "
                "Direct comparison is not meaningful due to fundamental differences:\n")
        f.write("> - MeshGPT uses a VQ-VAE + GPT architecture; we use "
                "DLFL topology tokens + silhouette conditioning.\n")
        f.write("> - MeshGPT is trained/evaluated on ShapeNet chairs; "
                "we use Thingi10K (multi-category, genus 0–2).\n")
        f.write("> - MeshGPT uses raw triangle mesh tokens; we use DLFL "
                "structural operators + quantized vertex coordinates.\n")
        f.write("> - The metrics (Coverage, Quality) are different from ours "
                "(Silhouette IoU, Manifold validity, Genus accuracy).\n\n")
        f.write("| Method | Architecture | Dataset | Manifold% | "
                "Coverage | Quality |\n")
        f.write("|--------|-------------|---------|-----------|"
                "----------|--------|\n")
        f.write("| MeshGPT (published) | VQ-VAE + GPT | ShapeNet chairs | "
                "~98% | 85.4% | 93.7% |\n")
        f.write(f"| **Ours (Phase B)** | DLFL + xAttn Transformer | "
                f"Thingi10K (g=0-2) | "
                f"**{manifold_rate:.1f}%** | "
                f"sil-IoU={mean_sil_iou:.4f} | — |\n\n")

        f.write("## Sample Images\n\n")
        f.write(f"Side-by-side visualisations (target | generated) saved to "
                f"`{args.samples_dir}/`.\n\n")
        f.write("Format: left 2×2 grid = 4 conditioning silhouette views of "
                "target; right 2×2 grid = same 4 views rendered from generated mesh.\n\n")

        f.write("## Notes\n\n")
        f.write("### Silhouette IoU\n\n")
        f.write("Rendered from 4 conditioning viewpoints (azimuths 0/90/180/270°, "
                "elevation 0°, radius=3.0, 128×128). Binary threshold 0.5.\n\n")
        f.write("### Genus accuracy\n\n")
        f.write("Compares `mesh.genus()` of generated mesh with ground-truth "
                "genus stored in val shard.\n\n")
        f.write("### Manifold validity\n\n")
        f.write("DLFL guarantees manifold property for valid structural sequences. "
                "Sub-100% indicates out-of-range ordinal references or "
                "truncated sequences.\n")

    print(f"\nResults written to: {results_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase B OpSeqModel")
    parser.add_argument(
        '--checkpoint',
        default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'),
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        '--data_dir',
        default=os.path.join(_SCRIPT_DIR, 'data'),
        help="Directory with val/ shards",
    )
    parser.add_argument(
        '--distill_log',
        default=os.path.join(_SCRIPT_DIR, 'data', 'distill_log.jsonl'),
        help="Path to distill_log.jsonl for pipeline stats",
    )
    parser.add_argument(
        '--n_eval',
        type=int,
        default=100,
        help="Number of val samples to evaluate",
    )
    parser.add_argument(
        '--samples_dir',
        default=os.path.join(_SCRIPT_DIR, 'samples'),
        help="Directory for side-by-side sample PNGs",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
