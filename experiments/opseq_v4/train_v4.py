#!/usr/bin/env python3
"""
train_v4.py — Classification training for TopoClassifier (Phase B).

Standard N-class classification:
  - CrossEntropyLoss on class labels (label_smoothing=0.1)
  - AdamW lr=1e-3, cosine decay, 200 epochs
  - Batch size 256 (tiny model, no sequence overhead)
  - bf16 mixed precision
  - Dropout 0.2 applied on top of the frozen encoder's pooled features via a
    thin wrapper (model_v4.TopoClassifier head has no dropout; we inject it
    at the training call-site to avoid modifying model_v4.py)
  - Logs train/val accuracy and loss
  - Saves best model by val accuracy

Usage:
    python train_v4.py [--data_dir experiments/opseq_v4/data]
                       [--ckpt_dir experiments/opseq_v4/ckpt]
                       [--n_epochs 200] [--batch_size 256] [--lr 1e-3]
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
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_v4 import TopoClassifier, count_params
from canonical_topos import N_CLASSES


# ═════════════════════════════════════════════════════════════════════════════
# Dropout wrapper
# ═════════════════════════════════════════════════════════════════════════════

class TopoClassifierWithDropout(nn.Module):
    """
    Thin wrapper around TopoClassifier that injects dropout=0.2 between the
    encoder's global-average-pool output and the linear classification head.

    We do NOT modify model_v4.py — instead we intercept the forward pass by
    hooking into the encoder and head separately.
    """

    def __init__(self, base: TopoClassifier, p: float = 0.2):
        super().__init__()
        self.encoder  = base.encoder
        self.head     = base.head
        self.dropout  = nn.Dropout(p=p)
        self.n_classes = base.n_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)        # [B, d_model, 8, 8]
        pooled   = features.mean(dim=[2, 3])   # [B, d_model]
        pooled   = self.dropout(pooled)        # dropout applied before head
        return self.head(pooled)               # [B, n_classes]


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class TopoDatasetV4(Dataset):
    """
    Loads v4 .npz shards from gen_data_v4.py.

    Shard format:
      images : [N, 4, 128, 128] uint8   (white-bg silhouettes)
      labels : [N]              int8    (class ID, 0..19)

    __getitem__ returns:
      img   : [4, 128, 128] float32 in [0, 1]
      label : int64
    """

    def __init__(self, shard_paths: List[str]):
        images_list = []
        labels_list = []
        for p in shard_paths:
            data   = np.load(p, allow_pickle=False)
            images_list.append(data['images'])
            labels_list.append(data['labels'].astype(np.int64))

        self.images = np.concatenate(images_list, axis=0)  # [N, 4, 128, 128]
        self.labels = np.concatenate(labels_list, axis=0)  # [N]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img   = self.images[idx].astype(np.float32) / 255.0  # [4, 128, 128]
        label = int(self.labels[idx])
        return img, label


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)
    imgs_t   = torch.from_numpy(np.stack(imgs, axis=0))       # [B, 4, 128, 128]
    labels_t = torch.tensor(labels, dtype=torch.long)          # [B]
    return imgs_t, labels_t


# ═════════════════════════════════════════════════════════════════════════════
# Validation helper
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_val(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    """Returns (val_loss, val_accuracy)."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels)
        total_loss    += loss.item() * labels.shape[0]
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count   += labels.shape[0]
    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


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

    train_ds = TopoDatasetV4(train_shards)
    val_ds   = TopoDatasetV4(val_shards)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {N_CLASSES}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    _base  = TopoClassifier(n_classes=N_CLASSES)
    model  = TopoClassifierWithDropout(_base, p=0.2).to(device)
    print(f"Parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
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
    best_val_acc = 0.0
    log: list = []
    t_start   = time.time()

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_count, ep_steps = 0.0, 0, 0, 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs)
                loss   = F.cross_entropy(logits, labels, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss    += loss.item() * labels.shape[0]
            ep_correct += (logits.argmax(dim=1) == labels).sum().item()
            ep_count   += labels.shape[0]
            ep_steps   += 1

        train_loss = ep_loss / max(ep_count, 1)
        train_acc  = ep_correct / max(ep_count, 1)
        val_loss, val_acc = run_val(model, val_loader, device)
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(f"Epoch {epoch:4d}/{args.n_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s")

        log.append({
            'epoch':     epoch,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'val_loss':   val_loss,
            'val_acc':    val_acc,
            'lr':         lr_now,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path    = os.path.join(args.ckpt_dir, 'best.pt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':          val_acc,
                'val_loss':         val_loss,
                'args':             vars(args),
            }, best_path)
            print(f"  ^ Best val_acc {best_val_acc:.4f} -> {best_path}")

        if epoch % 10 == 0:
            ppath = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'val_loss': val_loss}, ppath)

    log_path = os.path.join(args.ckpt_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Log -> {log_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopoClassifier (Phase B)")
    parser.add_argument('--data_dir',   default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--ckpt_dir',   default=os.path.join(_SCRIPT_DIR, 'ckpt'))
    parser.add_argument('--n_epochs',   type=int,   default=200)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
