#!/usr/bin/env python3
"""
distill.py — Silhouette-conditioned distillation pipeline for Phase B.

For each Thingi10K mesh:
  1. Load with trimesh, normalize to [-1.6, 1.6].
  2. Estimate genus (0–2) from thingi10k manifest (pre-computed, reliable).
  3. Build DLFL seed with topology tokens.
  4. Optimize seed verts to match target silhouettes.
  5. Accept if mean binary IoU >= min_iou threshold.
  6. Tokenize fitted vertices.
  7. Render 4 conditioning views of original target.
  8. Write shard .npz files (train/val split).

Resumable: already-processed file_ids are skipped via distill_log.jsonl.

Usage:
    python3 distill.py [--manifest PATH] [--out_dir PATH] [--val_frac 0.1]
                       [--max_meshes N] [--min_iou 0.75] [--device cuda]
                       [--seed 42] [--poll_interval 30]

Background launch:
    setsid nohup python3 -u distill.py ... > logs/distill.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nvdiffrast.torch as dr
import trimesh

from topmod.primitives    import make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.subdivision   import catmull_clark
from topmod.io            import to_triangle_arrays
from topmod.validate      import is_manifold
from topmod.tokenizer     import (
    build_vocabulary,
    quantize_coord,
    decode_sequence,
    detokenize,
    encode_sequence,
    _find_compatible_face_pair,
    _face_ordinal,
)

from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import render_silhouette, optimize

# PAD_ID for npz padding
PAD_ID = 263

# Shard size (meshes per .npz file)
SHARD_SIZE = 250


# ═════════════════════════════════════════════════════════════════════════════
# Mesh building helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_seed_with_tokens(
    genus: int,
    subdivisions: int = 3,
    device: str = 'cuda',
) -> Tuple:
    """
    Build DLFL seed and collect structural token IDs simultaneously.

    Returns: (dlfl_mesh, verts_t, faces_t, struct_ids, vocab)
      struct_ids: HDL + CC integer token IDs (CV and EOS not yet added)
      verts_t, faces_t: float32 / int32 tensors on `device`
    """
    vocab = build_vocabulary(n_position_bins=128, max_ordinal=128)

    mesh = make_icosahedron()
    struct_ids: List[int] = []
    excluded_vids: set = set()

    for _ in range(genus):
        f1, f2 = _find_compatible_face_pair(mesh, excluded_vids)
        f1_ord = _face_ordinal(mesh, f1)
        f2_ord = _face_ordinal(mesh, f2)
        excluded_vids |= (
            {v.id for v in f1.vertices()} | {v.id for v in f2.vertices()}
        )
        struct_ids.extend([
            vocab['HDL'],
            vocab[f'REF_{f1_ord}'],
            vocab[f'REF_{f2_ord}'],
        ])
        add_handle(mesh, f1, f2)

    for _ in range(subdivisions):
        struct_ids.append(vocab['CC'])
        mesh = catmull_clark(mesh)

    positions, triangles = to_triangle_arrays(mesh)
    verts_np = np.array(positions, dtype=np.float32)
    faces_np = np.array(triangles, dtype=np.int32)

    verts_t = torch.tensor(verts_np, device=device, dtype=torch.float32)
    faces_t = torch.tensor(faces_np, device=device, dtype=torch.int32)

    return mesh, verts_t, faces_t, struct_ids, vocab


def normalize_mesh(verts_np: np.ndarray, scale: float = 1.6) -> np.ndarray:
    """
    Normalize vertex positions to [-scale, scale] by centering + uniform scale.
    Returns float32 array of same shape.
    """
    verts = verts_np.astype(np.float32)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts = verts - center
    max_extent = np.abs(verts).max()
    if max_extent > 1e-8:
        verts = verts / max_extent * scale
    return verts


# ═════════════════════════════════════════════════════════════════════════════
# Log helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    """ISO timestamp prefix for log lines."""
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def _load_processed_ids(log_path: str) -> set:
    """Load already-processed file_ids from distill_log.jsonl."""
    processed = set()
    if not os.path.exists(log_path):
        return processed
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                processed.add(entry['file_id'])
            except Exception:
                pass
    return processed


# ═════════════════════════════════════════════════════════════════════════════
# Shard writer
# ═════════════════════════════════════════════════════════════════════════════

class ShardWriter:
    """Accumulates records and flushes .npz shards."""

    def __init__(self, out_dir: str, split: str):
        self.out_dir   = os.path.join(out_dir, split)
        self.split     = split
        self.shard_idx = 0
        self.buffer: List[dict] = []
        os.makedirs(self.out_dir, exist_ok=True)

        # Find the next shard index to resume correctly
        existing = sorted(
            f for f in os.listdir(self.out_dir) if f.endswith('.npz')
        )
        if existing:
            last = existing[-1]
            try:
                self.shard_idx = int(last.split('_')[1].split('.')[0]) + 1
            except Exception:
                self.shard_idx = len(existing)

    def add(self, images: np.ndarray, token_ids: List[int], genus: int) -> None:
        """images: [4, 128, 128] uint8"""
        self.buffer.append({
            'images': images,
            'token_ids': token_ids,
            'genus': genus,
        })

    def flush(self) -> None:
        if not self.buffer:
            return
        N = len(self.buffer)
        max_len = max(len(r['token_ids']) for r in self.buffer)

        images_arr  = np.stack([r['images']   for r in self.buffer], axis=0)  # [N,4,128,128]
        tokens_arr  = np.full((N, max_len), PAD_ID, dtype=np.int32)
        lengths_arr = np.zeros(N, dtype=np.int32)
        genera_arr  = np.zeros(N, dtype=np.int32)

        for i, r in enumerate(self.buffer):
            ids = r['token_ids']
            L   = len(ids)
            tokens_arr[i, :L] = ids
            lengths_arr[i]    = L
            genera_arr[i]     = r['genus']

        shard_path = os.path.join(
            self.out_dir, f'shard_{self.shard_idx:04d}.npz'
        )
        np.savez_compressed(
            shard_path,
            images=images_arr,
            tokens=tokens_arr,
            lengths=lengths_arr,
            genera=genera_arr,
        )
        print(f"{_ts()} [shard] Wrote {N} records → {shard_path}")
        sys.stdout.flush()

        self.buffer.clear()
        self.shard_idx += 1


# ═════════════════════════════════════════════════════════════════════════════
# Main distillation loop
# ═════════════════════════════════════════════════════════════════════════════

def distill(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"{_ts()} [warn] CUDA not available, falling back to cpu")
        device = 'cpu'

    # ── Setup directories ──────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(_SCRIPT_DIR, 'logs'), exist_ok=True)

    log_path = os.path.join(args.out_dir, 'distill_log.jsonl')

    # ── Load manifest ──────────────────────────────────────────────────
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}\n"
            f"Run acquire_data.py first."
        )

    with open(args.manifest, 'r') as f:
        manifest = [json.loads(l) for l in f if l.strip()]

    print(f"{_ts()} [distill] Manifest: {len(manifest)} entries")
    sys.stdout.flush()

    # ── Resume: skip already processed ────────────────────────────────
    processed_ids = _load_processed_ids(log_path)
    print(f"{_ts()} [distill] Already processed: {len(processed_ids)}")
    sys.stdout.flush()

    manifest = [r for r in manifest if r['file_id'] not in processed_ids]
    if args.max_meshes > 0:
        manifest = manifest[:args.max_meshes - len(processed_ids)]

    print(f"{_ts()} [distill] To process this run: {len(manifest)}")
    sys.stdout.flush()

    if not manifest:
        print(f"{_ts()} [distill] Nothing to do.")
        return

    # ── Rasterizer ────────────────────────────────────────────────────
    ctx = dr.RasterizeCudaContext() if device == 'cuda' else dr.RasterizeGLContext()

    # ── Camera setups ─────────────────────────────────────────────────
    mvps_8, _ = orbit_cameras(
        8, elevation_deg=20.0, radius=3.0, fov_deg=40.0, device=device
    )
    mvps_cond, _ = orbit_cameras(
        4, elevation_deg=0.0, radius=3.0, device=device,
        azimuths_deg=[0.0, 90.0, 180.0, 270.0],
    )

    # ── Accumulation buffers (pre-split) ──────────────────────────────
    # We collect all accepted records, then split and save at the end.
    accepted_records: List[dict] = []

    n_total    = 0
    n_accepted = 0
    n_rejected = 0
    distill_ious: List[float] = []

    # Track first 20 accepted for sanity check
    n_sanity_checked = 0

    log_fh = open(log_path, 'a')

    def _log(entry: dict) -> None:
        log_fh.write(json.dumps(entry) + '\n')
        log_fh.flush()

    try:
        for mesh_idx, row in enumerate(manifest):
            file_id   = row['file_id']
            file_path = row['file_path']
            genus     = int(row['genus'])  # from manifest (reliable)
            genus     = max(0, min(2, genus))

            n_total += 1
            t_mesh = time.time()

            # ── Load mesh ──────────────────────────────────────────────
            try:
                tm = trimesh.load(file_path, force='mesh')
                target_verts_np = np.array(tm.vertices, dtype=np.float32)
                target_faces_np = np.array(tm.faces,    dtype=np.int32)
            except Exception as exc:
                print(f"{_ts()} [skip] file_id={file_id}: load error: {exc!r}")
                sys.stdout.flush()
                _log({'file_id': file_id, 'status': 'load_error', 'error': str(exc)})
                n_rejected += 1
                continue

            if target_faces_np.shape[0] == 0:
                _log({'file_id': file_id, 'status': 'rejected', 'reason': 'no_faces'})
                n_rejected += 1
                continue

            # ── Normalize target ───────────────────────────────────────
            target_verts_np = normalize_mesh(target_verts_np, scale=1.6)
            target_verts_t  = torch.tensor(
                target_verts_np, device=device, dtype=torch.float32
            )
            target_faces_t  = torch.tensor(
                target_faces_np, device=device, dtype=torch.int32
            )

            # ── Build seed ─────────────────────────────────────────────
            try:
                seed_mesh, seed_verts_t, seed_faces_t, struct_ids, vocab = \
                    build_seed_with_tokens(genus, subdivisions=3, device=device)
            except Exception as exc:
                print(f"{_ts()} [skip] file_id={file_id}: seed build error: {exc!r}")
                sys.stdout.flush()
                _log({'file_id': file_id, 'status': 'rejected',
                      'reason': 'seed_build_error', 'error': str(exc)})
                n_rejected += 1
                continue

            # ── Render 8 target silhouettes ────────────────────────────
            try:
                target_sils = []
                for i in range(8):
                    sil = render_silhouette(
                        ctx, target_verts_t, target_faces_t,
                        mvps_8[i], (256, 256),
                    )
                    target_sils.append(sil)   # [1, 256, 256, 1]
                target_images = torch.cat(target_sils, dim=0)  # [8, 256, 256, 1]
            except Exception as exc:
                print(f"{_ts()} [skip] file_id={file_id}: target render error: {exc!r}")
                sys.stdout.flush()
                _log({'file_id': file_id, 'status': 'rejected',
                      'reason': 'target_render_error', 'error': str(exc)})
                n_rejected += 1
                continue

            # ── Optimize seed verts ────────────────────────────────────
            try:
                verts_final, _ = optimize(
                    ctx, seed_verts_t, seed_faces_t,
                    target_images, mvps_8,
                    num_steps=200, lr=5e-3,
                    lambda_lap=0.05, lambda_edge=0.01,
                    lambda_vol=0.0, resolution=(256, 256),
                    log_every=9999,
                )
            except Exception as exc:
                print(f"{_ts()} [skip] file_id={file_id}: optimize error: {exc!r}")
                sys.stdout.flush()
                _log({'file_id': file_id, 'status': 'rejected',
                      'reason': 'optimize_error', 'error': str(exc)})
                n_rejected += 1
                continue

            # ── Compute mean binary IoU (8 views) ─────────────────────
            with torch.no_grad():
                ious = []
                for i in range(8):
                    pred = render_silhouette(
                        ctx, verts_final, seed_faces_t,
                        mvps_8[i], (256, 256),
                    )[0, :, :, 0]
                    gt = target_images[i, :, :, 0]
                    pred_bin = (pred > 0.5).float()
                    gt_bin   = (gt   > 0.5).float()
                    inter    = (pred_bin * gt_bin).sum().item()
                    union    = ((pred_bin + gt_bin) > 0).float().sum().item()
                    ious.append(inter / max(union, 1.0))
            mean_iou = float(np.mean(ious))

            # ── IoU threshold check ────────────────────────────────────
            if mean_iou < args.min_iou:
                _log({
                    'file_id': file_id, 'status': 'rejected',
                    'reason': 'low_iou', 'iou': mean_iou,
                })
                n_rejected += 1
                if (n_total) % 10 == 0:
                    print(f"{_ts()} [progress] total={n_total} "
                          f"accepted={n_accepted} rejected={n_rejected} "
                          f"last_iou={mean_iou:.3f}")
                    sys.stdout.flush()
                continue

            # ── Normalize fitted verts ─────────────────────────────────
            fitted_np = verts_final.detach().cpu().numpy()
            fitted_np = normalize_mesh(fitted_np, scale=1.6)

            # ── Build full token sequence ──────────────────────────────
            token_ids = list(struct_ids)   # HDL + CC ids
            cv_id = vocab['CV']
            for x, y, z in fitted_np:     # vertex insertion order
                qx = quantize_coord(float(x))
                qy = quantize_coord(float(y))
                qz = quantize_coord(float(z))
                token_ids.extend([
                    cv_id,
                    vocab[f'COORD_{qx}'],
                    vocab[f'COORD_{qy}'],
                    vocab[f'COORD_{qz}'],
                ])
            token_ids.append(vocab['EOS'])

            # ── Sanity check (first 20 accepted) ──────────────────────
            if n_sanity_checked < 20:
                try:
                    vocab_inv = {v: k for k, v in vocab.items()}
                    dec_toks  = decode_sequence(token_ids, vocab_inv)
                    check_mesh = detokenize(dec_toks)
                    if not is_manifold(check_mesh):
                        print(f"{_ts()} [sanity] file_id={file_id}: "
                              f"WARNING detokenized mesh is NOT manifold")
                    else:
                        print(f"{_ts()} [sanity] file_id={file_id}: OK manifold")
                    sys.stdout.flush()
                except Exception as exc:
                    print(f"{_ts()} [sanity] file_id={file_id}: "
                          f"detokenize error: {exc!r}")
                    sys.stdout.flush()
                n_sanity_checked += 1

            # ── Render 4 conditioning views (original target) ──────────
            try:
                cond_imgs = []
                for i in range(4):
                    sil = render_silhouette(
                        ctx, target_verts_t, target_faces_t,
                        mvps_cond[i], (128, 128),
                    )
                    arr = (
                        sil[0, :, :, 0].detach().cpu().numpy() * 255
                    ).astype(np.uint8)
                    cond_imgs.append(arr)
                cond_np = np.stack(cond_imgs, axis=0)   # [4, 128, 128]
            except Exception as exc:
                print(f"{_ts()} [skip] file_id={file_id}: cond render error: {exc!r}")
                sys.stdout.flush()
                _log({'file_id': file_id, 'status': 'rejected',
                      'reason': 'cond_render_error', 'error': str(exc)})
                n_rejected += 1
                continue

            # ── Accept ─────────────────────────────────────────────────
            accepted_records.append({
                'images':    cond_np,       # [4, 128, 128] uint8
                'token_ids': token_ids,
                'genus':     genus,
                'file_id':   file_id,
            })
            distill_ious.append(mean_iou)
            n_accepted += 1

            elapsed_mesh = time.time() - t_mesh
            _log({
                'file_id': file_id,
                'status':  'accepted',
                'genus':   genus,
                'iou':     mean_iou,
                'seq_len': len(token_ids),
                'elapsed_s': round(elapsed_mesh, 2),
            })

            if n_total % 10 == 0:
                print(f"{_ts()} [progress] total={n_total} "
                      f"accepted={n_accepted} rejected={n_rejected} "
                      f"mean_iou={np.mean(distill_ious):.3f}")
                sys.stdout.flush()

    finally:
        log_fh.close()

    # ── Split train / val ──────────────────────────────────────────────
    print(f"\n{_ts()} [distill] Splitting {n_accepted} accepted records "
          f"into train/val (frac={args.val_frac}) …")
    sys.stdout.flush()

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(accepted_records)).tolist()
    shuffled = [accepted_records[i] for i in perm]

    n_val   = max(1, int(len(shuffled) * args.val_frac))
    n_train = len(shuffled) - n_val
    train_records = shuffled[:n_train]
    val_records   = shuffled[n_train:]

    print(f"{_ts()} [distill] Train: {len(train_records)} | Val: {len(val_records)}")
    sys.stdout.flush()

    def _write_split(records: List[dict], split: str) -> None:
        writer = ShardWriter(args.out_dir, split)
        for r in records:
            writer.add(r['images'], r['token_ids'], r['genus'])
            if len(writer.buffer) >= SHARD_SIZE:
                writer.flush()
        writer.flush()   # final partial shard

    _write_split(train_records, 'train')
    _write_split(val_records,   'val')

    # ── Summary ────────────────────────────────────────────────────────
    n_processed    = n_accepted + n_rejected
    rejection_rate = n_rejected / max(n_processed, 1) * 100
    mean_iou_str   = (
        f"{np.mean(distill_ious):.4f}" if distill_ious else "N/A"
    )

    print(f"\n{_ts()} ── Distillation complete ──")
    print(f"  Total processed  : {n_processed}")
    print(f"  Accepted         : {n_accepted}")
    print(f"  Rejected         : {n_rejected}")
    print(f"  Rejection rate   : {rejection_rate:.1f}%")
    print(f"  Mean distill IoU : {mean_iou_str}")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B distillation pipeline")
    parser.add_argument(
        '--manifest',
        default=os.path.join(_SCRIPT_DIR, 'data', 'manifest.jsonl'),
        help="Input manifest JSONL from acquire_data.py",
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join(_SCRIPT_DIR, 'data'),
        help="Output directory for train/val shards and distill_log.jsonl",
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.1,
        help="Fraction of accepted meshes used for validation",
    )
    parser.add_argument(
        '--max_meshes',
        type=int,
        default=2000,
        help="Maximum number of meshes to attempt (0 = all in manifest)",
    )
    parser.add_argument(
        '--min_iou',
        type=float,
        default=0.75,
        help="Minimum mean binary IoU to accept a distilled sample",
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help="Torch device (cuda or cpu)",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--poll_interval',
        type=int,
        default=30,
        help="(Informational) Suggested polling interval in seconds for log tailing",
    )
    args = parser.parse_args()
    distill(args)


if __name__ == '__main__':
    main()
