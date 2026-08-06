#!/usr/bin/env python3
"""
eval.py — Evaluate a trained OpSeqModel on 100 val conditions.

Metrics
-------
  (i)   Manifold validity rate of detokenised meshes.
        Hypothesis: 100 % — DLFL guarantees manifold property after any valid
        sequence of structural ops.  Non-100 % indicates out-of-bounds ordinals.
  (ii)  Token accuracy + exact-match rate (greedy vs ground truth).
  (iii) Silhouette IoU between rendered generated mesh and conditioning images.
  (iv)  Genus accuracy (mesh.genus() vs ground-truth genus).

Parse failures (malformed sequence: bad ordinal, wrong CV count, etc.) are
counted separately from manifold failures.

Results written to experiments/opseq/results.md

Usage:
    python eval.py [--checkpoint experiments/opseq/ckpt/best.pt]
                   [--data_dir   experiments/opseq/data]
                   [--n_eval 100]
"""

from __future__ import annotations

import argparse
import glob
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

import nvdiffrast.torch as dr

from topmod.validate import is_manifold
from topmod.tokenizer import build_vocabulary, decode_sequence, detokenize

from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette

from model import OpSeqModel, count_params, BOS_ID, PAD_ID, EOS_ID, VOCAB_SIZE

# ── Vocab ──────────────────────────────────────────────────────────────────────

N_BINS      = 128
MAX_ORDINAL = 128
VOCAB       = build_vocabulary(n_position_bins=N_BINS, max_ordinal=MAX_ORDINAL)
VOCAB_INV   = {v: k for k, v in VOCAB.items()}

AZIMUTHS    = [0.0, 90.0, 180.0, 270.0]
IMG_RES     = 128
CAM_RADIUS  = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Mesh utilities (duplicated from gen_data to avoid circular imports)
# ═════════════════════════════════════════════════════════════════════════════

def _mesh_to_tensors(mesh, device: str = 'cuda'):
    """DLFLMesh → (verts [V,3], faces [F,3]) for nvdiffrast."""
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
    ctx:        dr.RasterizeCudaContext,
    mesh,
    gt_images:  np.ndarray,   # [4, H, W] uint8
    device:     str = 'cuda',
) -> float:
    """
    Render mesh from 4 views and compute mean IoU vs GT silhouettes.
    Returns 0.0 on any render error.
    """
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
            pred = pred_sil[0, :, :, 0]   # [H, W]
            gt   = gt_f[i]                 # [H, W]

            pred_bin = (pred > 0.5).float()
            gt_bin   = (gt   > 0.5).float()
            inter    = (pred_bin * gt_bin).sum().item()
            union    = ((pred_bin + gt_bin) > 0).float().sum().item()
            ious.append(inter / max(union, 1.0))

        return float(np.mean(ious))
    except Exception:
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Val data loader (lightweight, no PyTorch Dataset overhead)
# ═════════════════════════════════════════════════════════════════════════════

def load_val_records(shard_paths: List[str]) -> List[dict]:
    """Load all val records into memory as dicts."""
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
                'image':  images[i],                          # [4, H, W] uint8
                'tokens': tokens[i, :L].astype(np.int64),    # [L] actual IDs
                'length': L,
                'genus':  int(genera[i]),
            })
    return records


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load model ─────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = OpSeqModel().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Parameters  : {count_params(model):,}")
    best_epoch = ckpt.get('epoch', '?')
    best_val   = ckpt.get('val_loss', float('nan'))
    print(f"Saved at epoch {best_epoch}, val_loss={best_val:.4f}")

    # ── Load val data ──────────────────────────────────────────────────
    val_shards = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.npz')))
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")
    records = load_val_records(val_shards)
    n_eval  = min(args.n_eval, len(records))
    print(f"Evaluating on {n_eval} / {len(records)} val samples…")

    ctx = dr.RasterizeCudaContext()

    # ── Metrics ────────────────────────────────────────────────────────
    n_manifold   = 0
    n_parse_fail = 0
    n_genus_ok   = 0
    token_accs   : List[float] = []
    exact_matches = 0
    sil_ious      : List[float] = []

    t0 = time.time()

    for idx in range(n_eval):
        r      = records[idx]
        img_np = r['image']                      # [4, H, W] uint8
        gt_ids = r['tokens']                     # [L] int64
        L      = r['length']
        genus  = r['genus']

        img_t = torch.from_numpy(img_np.astype(np.float32) / 255.0
                                 ).unsqueeze(0).to(device)   # [1, 4, H, W]

        # ── Greedy decode ───────────────────────────────────────────────
        generated_ids = model.sample_greedy(img_t, max_new_tokens=args.max_seq_len)

        # ── Attempt detokenise ──────────────────────────────────────────
        try:
            tokens = decode_sequence(generated_ids, VOCAB_INV)
            mesh   = detokenize(tokens)
        except Exception as exc:
            n_parse_fail += 1
            if n_parse_fail <= 5:
                print(f"  [parse fail {idx}] {exc!r}")
            sil_ious.append(0.0)
            token_accs.append(0.0)
            continue

        # Manifold check
        if is_manifold(mesh):
            n_manifold += 1

        # Genus check
        try:
            gen_genus = mesh.genus()
        except Exception:
            gen_genus = -999
        if gen_genus == genus:
            n_genus_ok += 1

        # ── Token accuracy ──────────────────────────────────────────────
        min_L    = min(len(generated_ids), L)
        gen_arr  = np.array(generated_ids[:min_L])
        gt_arr   = gt_ids[:min_L]
        n_match  = int((gen_arr == gt_arr).sum())
        token_accs.append(n_match / L)
        if len(generated_ids) == L and (gen_arr == gt_arr).all():
            exact_matches += 1

        # ── Silhouette IoU ──────────────────────────────────────────────
        iou = compute_silhouette_iou(ctx, mesh, img_np, device)
        sil_ious.append(iou)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n_eval}] manifold={n_manifold}  "
                  f"parse_fail={n_parse_fail}  "
                  f"iou={np.mean(sil_ious):.3f}  ({elapsed:.0f}s)")

    # ── Nucleus sampling — first 10 samples ───────────────────────────
    print("\nNucleus sampling (top_p=0.9) on first 10 samples…")
    nucleus_manifold, nucleus_parse_fail = 0, 0
    n_nucleus = min(10, n_eval)
    for idx in range(n_nucleus):
        r      = records[idx]
        img_t  = torch.from_numpy(
            r['image'].astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)
        gen_ids = model.sample_nucleus(img_t, max_new_tokens=args.max_seq_len)
        try:
            toks = decode_sequence(gen_ids, VOCAB_INV)
            mesh = detokenize(toks)
            if is_manifold(mesh):
                nucleus_manifold += 1
        except Exception:
            nucleus_parse_fail += 1

    nucleus_rate = nucleus_manifold / n_nucleus * 100

    # ── Aggregate ─────────────────────────────────────────────────────
    n_decoded = n_eval - n_parse_fail
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
    print(f"(i)  Manifold validity : {manifold_rate:.1f}%  "
          f"({n_manifold}/{n_decoded} decoded)")
    print(f"     Parse failures    : {n_parse_fail}/{n_eval} "
          f"({parse_fail_rate:.1f}%)")
    print(f"(ii) Token accuracy    : {mean_token_acc:.1f}%")
    print(f"     Exact match       : {exact_match_rate:.1f}%  "
          f"({exact_matches}/{n_eval})")
    print(f"(iii)Silhouette IoU    : {mean_sil_iou:.4f}")
    print(f"(iv) Genus accuracy    : {genus_acc_rate:.1f}%  "
          f"({n_genus_ok}/{n_decoded})")
    print(f"\nNucleus (top-p=0.9)  : {nucleus_rate:.1f}% manifold "
          f"({nucleus_manifold}/{n_nucleus}), "
          f"{nucleus_parse_fail} parse fails")
    print(f"{'='*60}")

    # ── Write results.md ──────────────────────────────────────────────
    results_path = os.path.join(_SCRIPT_DIR, 'results.md')
    with open(results_path, 'w') as f:
        f.write("# OpSeq Phase A — Evaluation Results\n\n")
        f.write(f"**Checkpoint**: `{args.checkpoint}`\n\n")
        f.write(f"**Epoch**: {best_epoch}  |  **Best val loss**: {best_val:.4f}\n\n")
        f.write(f"**Samples evaluated**: {n_eval}  |  "
                f"**Eval time**: {elapsed:.1f}s\n\n")

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
                f"**{genus_acc_rate:.1f}%** ({n_genus_ok}/{n_decoded}) |\n\n")

        f.write("## Nucleus Sampling (top-p=0.9, first 10 samples)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Manifold validity | {nucleus_rate:.1f}% "
                f"({nucleus_manifold}/{n_nucleus}) |\n")
        f.write(f"| Parse failures | {nucleus_parse_fail}/{n_nucleus} |\n\n")

        f.write("## Notes\n\n")
        f.write("### (i) Manifold validity\n\n")
        f.write("Hypothesis: **100%**.  The DLFL invariant ensures that any\n")
        f.write("valid sequence of structural operators (HDL, CC, IE, DE) produces\n")
        f.write("a closed orientable 2-manifold.  The only way to break this is to\n")
        f.write("emit an ordinal reference outside the current mesh's face/edge count.\n")
        f.write("A sub-100% rate here indicates the model generating out-of-range\n")
        f.write("ordinals, which is counted as a parse failure.\n\n")
        f.write("### (iii) Silhouette IoU\n\n")
        f.write("Rendered from the same 4 viewpoints (azimuths 0/90/180/270°,\n")
        f.write("elevation 0°, radius=3.0) used during data generation.\n")
        f.write("IoU is computed at binary threshold 0.5 and averaged over 4 views.\n\n")
        f.write("### (iv) Genus accuracy\n\n")
        f.write("Compares `mesh.genus()` of the generated mesh with the ground-truth\n")
        f.write("genus stored in the val shard.\n")

    print(f"\nResults written to: {results_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpSeqModel")
    parser.add_argument('--checkpoint',  default=os.path.join(_SCRIPT_DIR, 'ckpt', 'best.pt'))
    parser.add_argument('--data_dir',    default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--n_eval',      type=int,   default=100)
    parser.add_argument('--max_seq_len', type=int,   default=1200)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
