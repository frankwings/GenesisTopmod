#!/usr/bin/env python3
"""
model.py — OpSeqModel for autoregressive operator sequence generation.

Architecture
------------
  ImageEncoder  : 4-channel silhouette CNN → 64 spatial memory tokens
                  ~390 K parameters
  OpSeqDecoder  : 6-layer causal transformer with cross-attention to image memory
                  ~6.8 M parameters
  Total         : ~7.2 M parameters  (<< 15 M limit)

Vocabulary conventions (must match gen_data.py)
----------------------------------------------
  IDs 0..261  = real token vocabulary (EOS=0, CC=1, CV=2, IE=3, DE=4, HDL=5,
                COORD_0..127, REF_0..127)
  ID 262      = BOS  (start of sequence, decoder input only)
  ID 263      = PAD  (padding, ignored in loss)
  TOTAL_EMBED = 264  (embedding table size)
  VOCAB_SIZE  = 262  (output logit dimension — we never predict BOS/PAD)

Usage:
    python model.py           # prints param count + sanity check
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary constants (shared with gen_data / train / eval)
VOCAB_SIZE  = 262   # 6 ops + 128 COORD + 128 REF
BOS_ID      = 262
PAD_ID      = 263
EOS_ID      = 0     # VOCAB['EOS']
TOTAL_EMBED = 264   # includes BOS and PAD


class ImageEncoder(nn.Module):
    """
    Small CNN: 4 silhouette views (stacked as channels) → spatial memory.

    Input : [B, 4, H, H]  float32 in [0, 1]   (H=128 by default)
    Output: [B, 64, d_model]   — 64 = 8×8 spatial positions

    Architecture
    ------------
    4  → 32  → 64  → 128  → d_model  (each block: Conv3×3 + BN + GELU + MaxPool2)
    128×128 → 64×64 → 32×32 → 16×16 → 8×8

    Parameters: ~390 K
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,       32,      3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(32,      64,      3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(64,      128,     3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(128,     d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(), nn.MaxPool2d(2),
        )
        # → [B, d_model, 8, 8]; reshape to [B, 64, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)                               # [B, d, 8, 8]
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 64, d]


class OpSeqModel(nn.Module):
    """
    Autoregressive model: silhouette images → TopMod operator token sequences.

    Inference
    ---------
    model.sample_greedy(image)   – greedy argmax decoding
    model.sample_nucleus(image)  – nucleus (top-p) sampling

    Training (see train.py)
    -----------------------
    Teacher-forced: input = [BOS, t0, …, t_{L-2}], target = [t0, …, t_{L-1}=EOS]
    """

    def __init__(
        self,
        d_model:        int   = 256,
        nhead:          int   = 8,
        n_layers:       int   = 6,
        dim_feedforward: int  = 1024,
        dropout:        float = 0.1,
        max_seq_len:    int   = 1200,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # Image encoder → cross-attention memory
        self.image_encoder = ImageEncoder(d_model)

        # Token + positional embeddings
        # padding_idx keeps PAD embedding as zero and skips its gradient
        self.token_embed = nn.Embedding(TOTAL_EMBED, d_model, padding_idx=PAD_ID)
        self.pos_embed   = nn.Embedding(max_seq_len + 2, d_model)

        # Transformer decoder (Pre-LN for stability)
        dec_layer = nn.TransformerDecoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = dim_feedforward,
            dropout          = dropout,
            batch_first      = True,
            norm_first       = True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # Output projection: d_model → VOCAB_SIZE (no BOS/PAD outputs)
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ── Forward pass ───────────────────────────────────────────────────────
    def forward(
        self,
        images:   torch.Tensor,                  # [B, 4, 128, 128] float32 [0,1]
        tokens:   torch.Tensor,                  # [B, T] int64
        pad_mask: Optional[torch.Tensor] = None, # [B, T] bool (True = ignored)
    ) -> torch.Tensor:
        """Return logits [B, T, VOCAB_SIZE]."""
        B, T    = tokens.shape
        device  = tokens.device

        memory  = self.image_encoder(images)           # [B, 64, d_model]

        pos     = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        x       = self.token_embed(tokens) + self.pos_embed(pos) # [B, T, d_model]

        # Upper-triangular boolean causal mask (True = blocked)
        # Using bool matches tgt_key_padding_mask dtype and avoids deprecation warning
        causal = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )

        x = self.transformer(
            tgt=x, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
        )
        return self.out_proj(x)   # [B, T, VOCAB_SIZE]

    # ── Shared: image memory pre-computation ──────────────────────────────
    @torch.no_grad()
    def _get_memory(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image once; reuse across all sampling steps."""
        return self.image_encoder(image)   # [1, 64, d_model]

    @torch.no_grad()
    def _step(
        self,
        gen:    list,
        memory: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        One decoder forward step.  Re-uses pre-computed image memory.
        Returns logits for the LAST position only: [VOCAB_SIZE].
        """
        T   = len(gen)
        inp = torch.tensor([gen], dtype=torch.long, device=device)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.token_embed(inp) + self.pos_embed(pos)
        causal = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
        x = self.transformer(x, memory, tgt_mask=causal)
        return self.out_proj(x[0, -1])   # [VOCAB_SIZE]

    # ── Greedy sampling ────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_greedy(
        self,
        image:          torch.Tensor,   # [1, 4, 128, 128] float32 [0,1]
        max_new_tokens: int = 1200,
    ) -> List[int]:
        """Return token IDs (BOS stripped, may end with EOS).

        Image is encoded *once*; only the decoder is re-run per step.
        """
        self.eval()
        device = image.device
        memory = self._get_memory(image)   # [1, 64, d_model]
        gen    = [BOS_ID]

        for _ in range(max_new_tokens):
            logits  = self._step(gen, memory, device)
            next_id = int(logits.argmax().item())
            gen.append(next_id)
            if next_id == EOS_ID:
                break

        return gen[1:]   # strip BOS

    # ── Nucleus (top-p) sampling ───────────────────────────────────────────
    @torch.no_grad()
    def sample_nucleus(
        self,
        image:          torch.Tensor,   # [1, 4, 128, 128] float32 [0,1]
        max_new_tokens: int   = 1200,
        top_p:          float = 0.9,
        temperature:    float = 1.0,
    ) -> List[int]:
        """Nucleus sampling; returns token IDs (BOS stripped).

        Image is encoded *once*; only the decoder is re-run per step.
        """
        self.eval()
        device = image.device
        memory = self._get_memory(image)   # [1, 64, d_model]
        gen    = [BOS_ID]

        for _ in range(max_new_tokens):
            logits = self._step(gen, memory, device) / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)

            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)

            # Zero out tokens beyond the nucleus
            mask = (cum_probs - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs /= sorted_probs.sum().clamp(min=1e-8)

            sampled_idx = torch.multinomial(sorted_probs, 1).item()
            next_id     = int(sorted_ids[sampled_idx].item())
            gen.append(next_id)
            if next_id == EOS_ID:
                break

        return gen[1:]   # strip BOS


# ── Utility ────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    model = OpSeqModel()
    n     = count_params(model)
    print(f"OpSeqModel total trainable parameters: {n:,}")
    assert n < 15_000_000, f"Exceeds 15 M limit: {n:,}"

    # Forward sanity check
    imgs   = torch.zeros(2, 4, 128, 128)
    tokens = torch.tensor([[BOS_ID, 1, 2], [BOS_ID, 3, PAD_ID]], dtype=torch.long)
    pad_m  = torch.tensor([[False, False, False], [False, False, True]])
    out    = model(imgs, tokens, pad_m)
    assert out.shape == (2, 3, VOCAB_SIZE), f"Bad shape: {out.shape}"
    print(f"Forward pass OK — output shape: {out.shape}")

    # Greedy sampling smoke test
    img1    = torch.zeros(1, 4, 128, 128)
    sampled = model.sample_greedy(img1, max_new_tokens=20)
    print(f"Greedy sample (20 steps max): {sampled}")
    print("All checks passed.")
