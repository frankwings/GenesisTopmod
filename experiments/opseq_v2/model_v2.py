#!/usr/bin/env python3
"""
model_v2.py — OpSeqModel V2 for the propose-and-optimize paradigm (Phase A').

Architecture is identical to Phase A (experiments/opseq/model.py) but with:
  - VOCAB_SIZE  = 354  (IDs 0–353; excludes BOS=354 and PAD=355)
  - TOTAL_EMBED = 356  (full embedding table including BOS and PAD)
  - BOS_ID      = 354
  - PAD_ID      = 355
  - EOS_ID      = 0
  - MAX_SEQ_LEN = 200  (short topology+cage programs, down from 5000)

Vocabulary V2 layout (356 tokens):
  0       : EOS
  1–29    : Operators
  30–32   : BASE_CUBE, BASE_TETRAHEDRON, BASE_ICOSAHEDRON
  33      : SEP
  34–289  : COORD_0 .. COORD_255
  290–353 : REF_0 .. REF_63
  354     : BOS
  355     : PAD

Usage:
    python model_v2.py    # prints param count + sanity check
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary constants (must match build_vocabulary_v2 in tokenizer.py) ─────

VOCAB_SIZE  = 354   # IDs 0–353: all real tokens (never predict BOS/PAD)
BOS_ID      = 354
PAD_ID      = 355
EOS_ID      = 0     # VOCAB_V2['EOS']
SEP_ID      = 33    # VOCAB_V2['SEP']
TOTAL_EMBED = 356   # embedding table size (includes BOS and PAD)

MAX_SEQ_LEN = 200   # maximum generated sequence length (excludes BOS)


# ── Image encoder ──────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    4-channel silhouette CNN → 64 spatial memory tokens.

    Input : [B, 4, H, H]  float32 in [0, 1]   (H=128)
    Output: [B, 64, d_model]   — 64 = 8×8 spatial positions

    Architecture (same as Phase A):
    4  → 32  → 64  → 128  → d_model  (Conv3×3 + BN + GELU + MaxPool2)
    128×128 → 64×64 → 32×32 → 16×16 → 8×8

    Parameters: ~390 K
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
        x = self.net(x)                              # [B, d, 8, 8]
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 64, d]


# ── Full model ─────────────────────────────────────────────────────────────────

class OpSeqModelV2(nn.Module):
    """
    Autoregressive model for propose-and-optimize sequences.

    Predicts: [BASE_x] [HDL REF REF]* [OP]* [SEP] [COORD]* [EOS]

    Inference
    ---------
    model.sample_greedy(image)   – greedy argmax decoding
    model.sample_nucleus(image)  – nucleus (top-p) sampling

    Training
    --------
    Teacher-forced cross-entropy; input=[BOS, t0, …, t_{L-2}], target=[t0, …, EOS].
    """

    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        n_layers:        int   = 6,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        max_seq_len:     int   = MAX_SEQ_LEN,
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

        # Output projection: d_model → VOCAB_SIZE (IDs 0–353, no BOS/PAD outputs)
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
        images:   torch.Tensor,                   # [B, 4, 128, 128] float32 [0,1]
        tokens:   torch.Tensor,                   # [B, T] int64
        pad_mask: Optional[torch.Tensor] = None,  # [B, T] bool (True = ignored)
    ) -> torch.Tensor:
        """Return logits [B, T, VOCAB_SIZE]."""
        B, T   = tokens.shape
        device = tokens.device

        memory = self.image_encoder(images)           # [B, 64, d_model]

        pos    = torch.arange(T, device=device).unsqueeze(0)    # [1, T]
        x      = self.token_embed(tokens) + self.pos_embed(pos)  # [B, T, d_model]

        # Upper-triangular causal mask (True = blocked)
        causal = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )

        x = self.transformer(
            tgt=x, memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
        )
        return self.out_proj(x)   # [B, T, VOCAB_SIZE]

    # ── Shared image memory pre-computation ───────────────────────────────
    @torch.no_grad()
    def _get_memory(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)   # [1, 64, d_model]

    @torch.no_grad()
    def _step(
        self,
        gen:    list,
        memory: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """One decoder forward step; returns logits for the last position."""
        T   = len(gen)
        inp = torch.tensor([gen], dtype=torch.long, device=device)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.token_embed(inp) + self.pos_embed(pos)
        causal = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
        )
        x = self.transformer(x, memory, tgt_mask=causal)
        return self.out_proj(x[0, -1])   # [VOCAB_SIZE]

    # ── Greedy sampling ────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_greedy(
        self,
        image:          torch.Tensor,   # [1, 4, 128, 128] float32 [0,1]
        max_new_tokens: int = MAX_SEQ_LEN,
    ) -> List[int]:
        """Return token IDs (BOS stripped; may end with EOS)."""
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
        max_new_tokens: int   = MAX_SEQ_LEN,
        top_p:          float = 0.9,
        temperature:    float = 1.0,
    ) -> List[int]:
        """Nucleus sampling; returns token IDs (BOS stripped)."""
        self.eval()
        device = image.device
        memory = self._get_memory(image)   # [1, 64, d_model]
        gen    = [BOS_ID]

        for _ in range(max_new_tokens):
            logits = self._step(gen, memory, device) / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)

            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)

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
    model = OpSeqModelV2()
    n     = count_params(model)
    print(f"OpSeqModelV2 total trainable parameters: {n:,}")
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
