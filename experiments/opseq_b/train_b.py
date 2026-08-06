#!/usr/bin/env python3
"""
train_b.py — Phase B training for OpSeqModel with max_seq_len=5000.

Runs two training passes:
  1. Fine-tune: load Phase A best.pt weights, extend positional embedding,
     train on Phase B data.
  2. Scratch:   train from random initialisation on Phase B data.

Both passes use identical hyperparameters. Results are compared at end.

Usage:
    python3 train_b.py [--data_dir PATH] [--ckpt_dir PATH]
                       [--phase_a_ckpt PATH] [--n_epochs 60]
                       [--batch_size 4] [--lr 3e-4] [--seed 42]
                       [--scratch_only] [--finetune_only]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Import Phase A components ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'opseq'))
from train import OpSeqDataset, collate_fn, run_val   # noqa: E402
from model import OpSeqModel, count_params, BOS_ID, PAD_ID, EOS_ID, VOCAB_SIZE  # noqa: E402

# Phase B model uses a much larger positional context
MAX_SEQ_LEN_B = 5000

# Phase A model's max_seq_len (needed for weight transfer)
MAX_SEQ_LEN_A = 1200


# ═════════════════════════════════════════════════════════════════════════════
# Weight loading
# ═════════════════════════════════════════════════════════════════════════════

def load_phase_a_weights(model: OpSeqModel, ckpt_path: str) -> None:
    """
    Load Phase A checkpoint into a Phase B model.

    All weights are copied exactly EXCEPT pos_embed:
      - Positions 0..MAX_SEQ_LEN_A+1 are copied from Phase A.
      - Positions MAX_SEQ_LEN_A+2.. are randomly initialised (trunc_normal, std=0.02).

    The pos_embed table has shape [max_seq_len+2, d_model].
    Phase A: [1202, 256]   Phase B: [5002, 256]
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    src_state = ckpt['model_state_dict']

    tgt_state = model.state_dict()

    # Copy everything except pos_embed
    for key, tgt_param in tgt_state.items():
        if key == 'pos_embed.weight':
            continue
        if key in src_state and src_state[key].shape == tgt_param.shape:
            tgt_state[key].copy_(src_state[key])
        elif key in src_state:
            print(f"[weight_load] Shape mismatch for {key}: "
                  f"src={src_state[key].shape} tgt={tgt_param.shape}, skipping")

    # pos_embed: partial copy + random init for new positions
    src_pos = src_state['pos_embed.weight']   # [1202, d_model]
    tgt_pos = tgt_state['pos_embed.weight']   # [5002, d_model]
    n_copy  = min(src_pos.shape[0], tgt_pos.shape[0])

    nn.init.trunc_normal_(tgt_pos, std=0.02)   # initialise all first
    tgt_pos[:n_copy].copy_(src_pos[:n_copy])   # overwrite with Phase A weights

    model.load_state_dict(tgt_state)
    print(f"[weight_load] Copied {n_copy} pos_embed rows from Phase A, "
          f"randomly initialised {tgt_pos.shape[0] - n_copy} new rows.")


# ═════════════════════════════════════════════════════════════════════════════
# Training loop (parameterised)
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    model:       OpSeqModel,
    ckpt_dir:    str,
    args:        argparse.Namespace,
    label:       str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:      str,
) -> float:
    """
    Run one full training pass. Returns best val loss achieved.

    `label` is used in print statements to distinguish 'finetune' vs 'scratch'.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Optimiser & LR schedule ────────────────────────────────────────
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

    best_val = float('inf')
    log: list = []
    t_start   = time.time()

    print(f"\n[{label}] Starting training: {args.n_epochs} epochs, "
          f"batch_size={args.batch_size}, lr={args.lr}")

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

        print(f"[{label}] Epoch {epoch:4d}/{args.n_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s")

        log.append({'epoch': epoch, 'train': train_loss,
                    'val': val_loss, 'lr': lr_now})

        if val_loss < best_val:
            best_val  = val_loss
            best_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':         val_loss,
                'args':             vars(args),
                'label':            label,
            }, best_path)
            print(f"[{label}]   Best val {best_val:.4f} → {best_path}")

        if epoch % 10 == 0:
            ppath = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'label': label,
            }, ppath)

    log_path = os.path.join(ckpt_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[{label}] Training complete. Best val loss: {best_val:.4f}")
    print(f"[{label}] Log → {log_path}")

    return best_val


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpSeqModel Phase B")
    parser.add_argument(
        '--data_dir',
        default=os.path.join(_SCRIPT_DIR, 'data'),
        help="Directory containing train/ and val/ shard subdirs",
    )
    parser.add_argument(
        '--ckpt_dir',
        default=os.path.join(_SCRIPT_DIR, 'ckpt'),
        help="Root checkpoint directory; subdirs finetune/ and scratch/ created here",
    )
    parser.add_argument(
        '--phase_a_ckpt',
        default=os.path.join(_SCRIPT_DIR, '..', 'opseq', 'ckpt', 'best.pt'),
        help="Path to Phase A best.pt (for fine-tune run)",
    )
    parser.add_argument('--n_epochs',   type=int,   default=60)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument(
        '--scratch_only',
        action='store_true',
        help="Skip fine-tune run; only train from scratch",
    )
    parser.add_argument(
        '--finetune_only',
        action='store_true',
        help="Skip scratch run; only fine-tune from Phase A",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────
    train_shards = sorted(glob.glob(os.path.join(args.data_dir, 'train', '*.npz')))
    val_shards   = sorted(glob.glob(os.path.join(args.data_dir, 'val',   '*.npz')))

    if not train_shards:
        raise FileNotFoundError(f"No train shards in {args.data_dir}/train/")
    if not val_shards:
        raise FileNotFoundError(f"No val shards in {args.data_dir}/val/")

    train_ds = OpSeqDataset(train_shards)
    val_ds   = OpSeqDataset(val_shards)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    results = {}

    # ── Fine-tune run ──────────────────────────────────────────────────
    if not args.scratch_only:
        phase_a_ckpt = os.path.normpath(
            os.path.join(_SCRIPT_DIR, args.phase_a_ckpt)
            if not os.path.isabs(args.phase_a_ckpt)
            else args.phase_a_ckpt
        )
        if not os.path.exists(phase_a_ckpt):
            print(f"[warn] Phase A checkpoint not found: {phase_a_ckpt}")
            print("[warn] Skipping fine-tune run.")
        else:
            print(f"\n{'='*60}")
            print(f"Phase B — Fine-tune from Phase A checkpoint")
            print(f"Phase A ckpt: {phase_a_ckpt}")
            print(f"{'='*60}")

            ft_model = OpSeqModel(max_seq_len=MAX_SEQ_LEN_B).to(device)
            load_phase_a_weights(ft_model, phase_a_ckpt)
            print(f"Parameters: {count_params(ft_model):,}")

            ft_ckpt_dir = os.path.join(args.ckpt_dir, 'finetune')
            best_ft = run_training(
                ft_model, ft_ckpt_dir, args,
                label='finetune',
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
            results['finetune'] = best_ft

    # ── Scratch run ────────────────────────────────────────────────────
    if not args.finetune_only:
        print(f"\n{'='*60}")
        print(f"Phase B — Train from scratch")
        print(f"{'='*60}")

        sc_model = OpSeqModel(max_seq_len=MAX_SEQ_LEN_B).to(device)
        print(f"Parameters: {count_params(sc_model):,}")

        sc_ckpt_dir = os.path.join(args.ckpt_dir, 'scratch')
        best_sc = run_training(
            sc_model, sc_ckpt_dir, args,
            label='scratch',
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        results['scratch'] = best_sc

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase B training complete — comparison:")
    print(f"{'='*60}")
    for label, val in results.items():
        print(f"  {label:12s} best val loss: {val:.4f}")

    if len(results) == 2:
        winner = min(results, key=results.get)
        print(f"\n  Winner: {winner}  (lower val loss = {results[winner]:.4f})")
        # Write symlink best.pt → winner/best.pt for eval_b.py default
        best_link = os.path.join(args.ckpt_dir, 'best.pt')
        winner_best = os.path.join(args.ckpt_dir, winner, 'best.pt')
        try:
            if os.path.islink(best_link) or os.path.exists(best_link):
                os.remove(best_link)
            os.symlink(winner_best, best_link)
            print(f"  Symlink: {best_link} → {winner_best}")
        except Exception as exc:
            print(f"  [warn] Could not create symlink: {exc}")


if __name__ == '__main__':
    main()
