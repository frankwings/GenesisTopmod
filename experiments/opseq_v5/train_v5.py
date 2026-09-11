#!/usr/bin/env python3
"""
train_v5.py — Multi-task training for TopoShapeNet (Plan C).

Dual loss:
  - Classification: CrossEntropyLoss (label_smoothing=0.1) on topology class
  - Regression: SmoothL1Loss on shape half-extents [hx, hy, hz]
  - Total = cls_loss + λ_shape * shape_loss  (λ_shape=1.0)

Training config:
  - AdamW lr=1e-3, cosine decay, 200 epochs
  - Batch size 256, bf16 mixed precision
  - Dropout 0.2 on pooled features (same wrapper approach as V4)

Usage:
    python train_v5.py [--data_dir experiments/opseq_v5/data]
                       [--ckpt_dir experiments/opseq_v5/ckpt]
                       [--n_epochs 200] [--batch_size 256] [--lr 1e-3]
                       [--lambda_shape 1.0] [--seed 42]
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
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_v5 import TopoShapeNet, count_params
from canonical_topos import N_CLASSES


# ═════════════════════════════════════════════════════════════════════════════
# Dropout wrapper for training
# ═════════════════════════════════════════════════════════════════════════════

class TopoShapeNetWithDropout(nn.Module):
    def __init__(self, base: TopoShapeNet, p: float = 0.2):
        super().__init__()
        self.encoder    = base.encoder
        self.cls_head   = base.cls_head
        self.shape_head = base.shape_head
        self.dropout    = nn.Dropout(p=p)
        self.n_classes  = base.n_classes

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        pooled   = features.mean(dim=[2, 3])
        pooled   = self.dropout(pooled)
        cls_logits   = self.cls_head(pooled)
        shape_params = self.shape_head(pooled)
        return cls_logits, shape_params


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class TopoDatasetV5(Dataset):
    """
    Loads v5 .npz shards.

    __getitem__ returns:
      img          : [4, 128, 128] float32 in [0, 1]
      label        : int64
      shape_params : [3] float32
    """

    def __init__(self, shard_paths: List[str]):
        images_list = []
        labels_list = []
        shapes_list = []
        for p in shard_paths:
            data = np.load(p, allow_pickle=False)
            images_list.append(data['images'])
            labels_list.append(data['labels'].astype(np.int64))
            shapes_list.append(data['shape_params'].astype(np.float32))

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
# Validation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_val(
    model: nn.Module, loader: DataLoader, device: str, lambda_shape: float,
) -> Tuple[float, float, float, float]:
    """Returns (val_loss, val_cls_acc, val_shape_mae, val_total_loss)."""
    model.eval()
    total_cls_loss = total_shape_loss = 0.0
    total_correct = total_count = 0
    total_shape_mae = 0.0

    for imgs, labels, shapes in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        shapes = shapes.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls_logits, shape_pred = model(imgs)
            cls_loss   = F.cross_entropy(cls_logits, labels)
            shape_loss = F.smooth_l1_loss(shape_pred, shapes)

        B = labels.shape[0]
        total_cls_loss   += cls_loss.item() * B
        total_shape_loss += shape_loss.item() * B
        total_correct    += (cls_logits.argmax(dim=1) == labels).sum().item()
        total_count      += B
        total_shape_mae  += (shape_pred - shapes).abs().mean().item() * B

    n = max(total_count, 1)
    avg_cls_loss   = total_cls_loss / n
    avg_shape_loss = total_shape_loss / n
    accuracy       = total_correct / n
    avg_shape_mae  = total_shape_mae / n
    total_loss     = avg_cls_loss + lambda_shape * avg_shape_loss
    return total_loss, accuracy, avg_shape_mae, avg_cls_loss


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

    train_ds = TopoDatasetV5(train_shards)
    val_ds   = TopoDatasetV5(val_shards)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {N_CLASSES}")
    print(f"Shape params range: min={train_ds.shapes.min(axis=0)}, "
          f"max={train_ds.shapes.max(axis=0)}, "
          f"mean={train_ds.shapes.mean(axis=0)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    _base = TopoShapeNet(n_classes=N_CLASSES)
    model = TopoShapeNetWithDropout(_base, p=0.2).to(device)
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
    t_start = time.time()

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_cls_loss = ep_shape_loss = 0.0
        ep_correct = ep_count = ep_steps = 0

        for imgs, labels, shapes in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            shapes = shapes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cls_logits, shape_pred = model(imgs)
                cls_loss   = F.cross_entropy(cls_logits, labels, label_smoothing=0.1)
                shape_loss = F.smooth_l1_loss(shape_pred, shapes)
                loss = cls_loss + args.lambda_shape * shape_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            B = labels.shape[0]
            ep_cls_loss   += cls_loss.item() * B
            ep_shape_loss += shape_loss.item() * B
            ep_correct    += (cls_logits.argmax(dim=1) == labels).sum().item()
            ep_count      += B
            ep_steps      += 1

        n = max(ep_count, 1)
        train_cls_loss   = ep_cls_loss / n
        train_shape_loss = ep_shape_loss / n
        train_acc        = ep_correct / n

        val_total, val_acc, val_shape_mae, val_cls_loss = run_val(
            model, val_loader, device, args.lambda_shape,
        )
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(f"Epoch {epoch:4d}/{args.n_epochs}  "
              f"cls={train_cls_loss:.4f}  shape={train_shape_loss:.4f}  "
              f"acc={train_acc:.4f}  "
              f"v_acc={val_acc:.4f}  v_mae={val_shape_mae:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s")

        log.append({
            'epoch':            epoch,
            'train_cls_loss':   train_cls_loss,
            'train_shape_loss': train_shape_loss,
            'train_acc':        train_acc,
            'val_acc':          val_acc,
            'val_shape_mae':    val_shape_mae,
            'val_cls_loss':     val_cls_loss,
            'lr':               lr_now,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path    = os.path.join(args.ckpt_dir, 'best.pt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':          val_acc,
                'val_shape_mae':    val_shape_mae,
                'args':             vars(args),
            }, best_path)
            print(f"  ^ Best val_acc {best_val_acc:.4f} -> {best_path}")

        if epoch % 10 == 0:
            ppath = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc}, ppath)

    log_path = os.path.join(args.ckpt_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Log -> {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TopoShapeNet (Plan C)")
    parser.add_argument('--data_dir',     default=os.path.join(_SCRIPT_DIR, 'data'))
    parser.add_argument('--ckpt_dir',     default=os.path.join(_SCRIPT_DIR, 'ckpt'))
    parser.add_argument('--n_epochs',     type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lambda_shape', type=float, default=1.0)
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
