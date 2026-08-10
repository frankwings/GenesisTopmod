#!/usr/bin/env python3
"""
model_v3.py — OpSeqModelV3 for topology-only prediction (Phase A'').

Predicts ONLY the topology program: [BASE_x] [HDL REF REF]* [OP]* [EOS]
Geometry is handled entirely by direct vertex optimization at inference time.

Changes from model_v2.py:
  - VOCAB_SIZE  = 97  (IDs 0–96; excludes BOS=97 and PAD=98)
  - TOTAL_EMBED = 99
  - BOS_ID      = 97
  - PAD_ID      = 98
  - EOS_ID      = 0
  - MAX_SEQ_LEN = 30   (sequences are ~5–15 tokens)
  - n_layers    = 4    (reduced from 6; shorter sequences need less capacity)
  - All other architecture unchanged (same CNN encoder, d_model=256, nhead=8)

Vocabulary V3 layout (99 tokens):
  0       : EOS
  1–29    : Operators
  30–32   : BASE_CUBE, BASE_TETRAHEDRON, BASE_ICOSAHEDRON
  33–96   : REF_0 .. REF_63
  97      : BOS
  98      : PAD

Usage:
    python model_v3.py    # prints param count + sanity check
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary constants (must match build_vocabulary_v3 in tokenizer.py) ─────

VOCAB_SIZE  = 97    # IDs 0–96: all real tokens (never predict BOS/PAD)
BOS_ID      = 97
PAD_ID      = 98
EOS_ID      = 0
TOTAL_EMBED = 99    # embedding table size (includes BOS and PAD)

MAX_SEQ_LEN = 30    # maximum generated sequence length (excludes BOS)


# ── Image encoder (identical to V2) ───────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    4-channel silhouette CNN → 64 spatial memory tokens.

    Input : [B, 4, H, H]  float32 in [0, 1]   (H=128)
    Output: [B, 64, d_model]

    Architecture: 4→32→64→128→d_model  (Conv3×3 + BN + GELU + MaxPool2)
    128×128 → 64 → 32 → 16 → 8×8 = 64 spatial tokens
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,        32,      3, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(32,       64,      3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(64,       128,     3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(128,      d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(), nn.MaxPool2d(2),
        )  # → [B, d_model, 8, 8]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # [B, 64, d]


# ── Full model ─────────────────────────────────────────────────────────────────

class OpSeqModelV3(nn.Module):
    """
    Topology-only autoregressive model.

    Predicts: [BASE_x] [HDL REF REF]* [OP]* [EOS]
    No COORD tokens, no SEP. Sequences are ~5–15 tokens.

    Inference
    ---------
    model.sample_greedy(image)   – greedy argmax decoding
    model.sample_nucleus(image)  – nucleus (top-p) sampling
    """

    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        n_layers:        int   = 4,    # reduced from 6 (short sequences)
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        max_seq_len:     int   = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # Image encoder
        self.image_encoder = ImageEncoder(d_model)

        # Token + positional embeddings
        self.token_embed = nn.Embedding(TOTAL_EMBED, d_model, padding_idx=PAD_ID)
        self.pos_embed   = nn.Embedding(max_seq_len + 2, d_model)

        # 4-layer Pre-LN causal TransformerDecoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model          = d_model,
            nhead            = nhead,
            dim_feedforward  = dim_feedforward,
            dropout          = dropout,
            batch_first      = True,
            norm_first       = True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # Output projection: d_model → VOCAB_SIZE
        self.out_proj = nn.Linear(d_model, VOCAB_SIZE)

        self._init_weights()

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

    def forward(
        self,
        images:   torch.Tensor,                   # [B, 4, 128, 128] float32 [0,1]
        tokens:   torch.Tensor,                   # [B, T] int64
        pad_mask: Optional[torch.Tensor] = None,  # [B, T] bool (True = ignored)
    ) -> torch.Tensor:
        """Return logits [B, T, VOCAB_SIZE]."""
        B, T   = tokens.shape
        device = tokens.device

        memory = self.image_encoder(images)           # [B, 64, d_model]
        pos    = torch.arange(T, device=device).unsqueeze(0)
        x      = self.token_embed(tokens) + self.pos_embed(pos)

        causal = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )
        x = self.transformer(tgt=x, memory=memory,
                             tgt_mask=causal, tgt_key_padding_mask=pad_mask)
        return self.out_proj(x)   # [B, T, VOCAB_SIZE]

    @torch.no_grad()
    def _get_memory(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)   # [1, 64, d_model]

    @torch.no_grad()
    def _step(self, gen: list, memory: torch.Tensor, device) -> torch.Tensor:
        """One decoder step; returns logits for last position [VOCAB_SIZE]."""
        T   = len(gen)
        inp = torch.tensor([gen], dtype=torch.long, device=device)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.token_embed(inp) + self.pos_embed(pos)
        causal = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )
        x = self.transformer(x, memory, tgt_mask=causal)
        return self.out_proj(x[0, -1])   # [VOCAB_SIZE]

    @torch.no_grad()
    def sample_greedy(
        self,
        image:          torch.Tensor,   # [1, 4, 128, 128] float32 [0,1]
        max_new_tokens: int = MAX_SEQ_LEN,
    ) -> List[int]:
        """Return token IDs (BOS stripped; may end with EOS)."""
        self.eval()
        device = image.device
        memory = self._get_memory(image)
        gen    = [BOS_ID]
        for _ in range(max_new_tokens):
            next_id = int(self._step(gen, memory, device).argmax().item())
            gen.append(next_id)
            if next_id == EOS_ID:
                break
        return gen[1:]   # strip BOS

    @torch.no_grad()
    def sample_nucleus(
        self,
        image:          torch.Tensor,
        max_new_tokens: int   = MAX_SEQ_LEN,
        top_p:          float = 0.9,
        temperature:    float = 1.0,
    ) -> List[int]:
        """Nucleus sampling; returns token IDs (BOS stripped)."""
        self.eval()
        device = image.device
        memory = self._get_memory(image)
        gen    = [BOS_ID]
        for _ in range(max_new_tokens):
            logits = self._step(gen, memory, device) / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=0)
            mask = (cum - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs /= sorted_probs.sum().clamp(min=1e-8)
            next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1).item()].item())
            gen.append(next_id)
            if next_id == EOS_ID:
                break
        return gen[1:]


# ── Utility ────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    model = OpSeqModelV3()
    n     = count_params(model)
    print(f"OpSeqModelV3 total trainable parameters: {n:,}")
    assert n < 15_000_000, f"Exceeds 15M limit: {n:,}"

    imgs   = torch.zeros(2, 4, 128, 128)
    tokens = torch.tensor([[BOS_ID, 30, 1], [BOS_ID, 32, PAD_ID]], dtype=torch.long)
    pad_m  = torch.tensor([[False, False, False], [False, False, True]])
    out    = model(imgs, tokens, pad_m)
    assert out.shape == (2, 3, VOCAB_SIZE), f"Bad shape: {out.shape}"
    print(f"Forward pass OK: {out.shape}")

    img1    = torch.zeros(1, 4, 128, 128)
    sampled = model.sample_greedy(img1, max_new_tokens=15)
    print(f"Greedy sample (15 steps): {sampled}")
    print("All checks passed.")
