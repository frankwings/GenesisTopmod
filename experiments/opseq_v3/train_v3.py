#!/usr/bin/env python3
"""
train_v3.py — Teacher-forced cross-entropy training for OpSeqModelV3 (Phase A'').

Key differences from train_v2.py:
  - Loads V3 shards (topology-only tokens, very short sequences ~5–15 tokens)
  - batch_size=128 (large batches fit due to short sequences)
  - lr=1e-4, cosine decay, 100 epochs

Usage:
    python train_v3.py [--data_dir experiments/opseq_v3/data]
                       [--ckpt_dir experiments/opseq_v3/ckpt]
                       [--n_epochs 100] [--batch_size 128] [--lr 1e-4]
                       [--seed 42]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model_v3 import OpSeqModelV3, count_params, BOS_ID, PAD_ID, EOS_ID, VOCAB_SIZE


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class OpSeqDatasetV3(Dataset):
    """
    Loads V3 .npz shards from gen_data_v3.py.

    Shard format (int16 tokens for compactness):
      images  : [N, 4, 128, 128] uint8   (white-bg silhouettes)
      tokens  : [N, max_len]     int16   (padded with PAD_ID=98)
      lengths : [N]              int16   (actual length including EOS)
      genera  : [N]              int8    (genus label)

    __getitem__ returns:
      img     : [4, 128, 128] float32 in [0, 1]
      inp     : [length]      int64   [BOS, t0, …, t_{L-2}]
      tgt     : [length]      int64   [t0,  …, t_{L-1}=EOS]
      length  : int
      genus   : int
    """

    def __init__(self, shard_paths: List[str]):
        self.records: List[dict] = []
        for p in shard_paths:
            data    = np.load(p, allow_pickle=False)
            N       = int(data['images'].shape[0])
            images  = data['images']
            tokens  = data['tokens'].astype(np.int32)
            lengths = data['lengths'].astype(np.int32)
            genera  = data['genera'].astype(np.int32)
            for i in range(N):
                L = int(lengths[i])
                self.records.append({
                    'image':  images[i],
                    'tokens': tokens[i, :L].astype(np.int64),
                    'length': L,
                    'genus':  int(genera[i]),
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r    = self.records[idx]
        L    = r['length']
        toks = r['tokens']    # [L] int64, last = EOS

        inp    = np.empty(L, dtype=np.int64)
        inp[0] = BOS_ID
        inp[1:] = toks[:L - 1]
        tgt    = toks

        # White-bg uint8 → float32 [0,1] (0=fg, 1=bg)
        img = r['image'].astype(np.float32) / 255.0

        return img, inp, tgt, L, r['genus']


def collate_fn(batch: list) -> Tuple:
    """Dynamic padding to longest sequence in batch."""
    imgs, inps, tgts, lengths, genera = zip(*batch)
    B, max_L = len(lengths), max(lengths)

    imgs_t   = torch.from_numpy(np.stack(imgs, axis=0))
    inp_t    = torch.full((B, max_L), PAD_ID, dtype=torch.long)
    tgt_t    = torch.full((B, max_L), PAD_ID, dtype=torch.long)
    pad_mask = torch.ones((B, max_L), dtype=torch.bool)

    for i, (inp, tgt, L) in enumerate(zip(inps, tgts, lengths)):
        inp_t   [i, :L] = torch.from_numpy(inp)
        tgt_t   [i, :L] = torch.from_numpy(tgt)
        pad_mask[i, :L] = False

    return (imgs_t,
            inp_t,
            tgt_t,
            pad_mask,
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(genera,  dtype=torch.long))


# ═════════════════════════════════════════════════════════════════════════════
# Validation helper
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_val(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for imgs, inps, tgts, masks, lengths, genera in loader:
        imgs  = imgs .to(device, non_blocking=True)
        inps  = inps .to(device, non_blocking=True)
        tgts  = tgts .to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(imgs, inps, masks)
            loss   = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgts  .reshape(-1),
                ignore_index=PAD_ID,
            )
        total += loss.item(); n += 1
    return total / max(n, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_shards = sorted(glob.glob(os.path.join(args.data_dir, 'train', '*.npz')))
    val_shards   = sorted(glob.glob(os.path.join(args.data_dir, 'val',   '*.npz')))

    if not train_shards:
        raise FileNotFoundError(f"No train shards in {args.data_dir}/train/")
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")

    train_ds = OpSeqDatasetV3(train_shards)
    val_ds   = OpSeqDatasetV3(val_shards)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    model = OpSeqModelV3().to(device)
    print(f"Parameters: {count_params(model):,}")

    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps  = len(train_loader) * args.n_epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float('inf')
    log: list = []
    t_start   = time.time()

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_loss, ep_steps = 0.0, 0

        for imgs, inps, tgts, masks, lengths, genera in train_loader:
            imgs  = imgs .to(device, non_blocking=True)
            inps  = inps .to(device, non_blocking=True)
            tgts  = tgts .to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs, inps, masks)
                loss   = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    tgts  .reshape(-1),
                    ignore_index=PAD_ID,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss  += loss.item()
            ep_steps += 1

        train_loss = ep_loss / max(ep_steps, 1)
        val_loss   = run_val(model, val_loader, device)
        lr_now     = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t_start

        print(f"Epoch {epoch:4d}/{args.n_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s")

        log.append({'epoch': epoch, 'train': train_loss, 'val': val_loss, 'lr': lr_now})

        if val_loss < best_val:
            best_val  = val_loss
            best_path = os.path.join(args.ckpt_dir, 'best.pt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':         val_loss,
                'args':             vars(args),
            }, best_path)
            print(f"  ↑ Best val {best_val:.4f} → {best_path}")

        if epoch % 10 == 0:
            ppath = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss}, ppath)

    log_path = os.path.join(args.ckpt_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Log → {log_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpSeqModelV3 (Phase A'')")
    parser.add_argument('--data_dir',   default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--ckpt_dir',   default=os.path.join(_SCRIPT_DIR, 'ckpt'))
    parser.add_argument('--n_epochs',   type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
