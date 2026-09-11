#!/usr/bin/env python3
"""
model_v4.py — TopoClassifier: 4-channel silhouette CNN -> N_CLASSES logits.

Simple CNN classifier replacing the autoregressive transformer from v3.
Same encoder architecture (4->32->64->128->d_model, 4 blocks with BN+GELU+MaxPool2)
followed by global average pooling and a linear classification head.

Input : [B, 4, 128, 128] float32 in [0, 1]
Output: [B, N_CLASSES] logits

Usage:
    python model_v4.py    # prints param count + sanity check
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from canonical_topos import N_CLASSES


class TopoClassifier(nn.Module):
    """4-channel silhouette CNN -> N_CLASSES logits."""

    def __init__(self, n_classes: int = N_CLASSES, d_model: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.d_model   = d_model

        # CNN encoder: 4->32->64->128->d_model
        # 128x128 -> 64 -> 32 -> 16 -> 8x8 after 4 MaxPool2 stages
        self.encoder = nn.Sequential(
            nn.Conv2d(4,        32,      3, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(32,       64,      3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(64,       128,     3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),

            nn.Conv2d(128,      d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(), nn.MaxPool2d(2),
        )  # -> [B, d_model, 8, 8]

        # Classification head
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : [B, 4, 128, 128] float32 in [0, 1]

        Returns
        -------
        logits : [B, n_classes]
        """
        features = self.encoder(images)          # [B, d_model, 8, 8]
        pooled   = features.mean(dim=[2, 3])     # [B, d_model] global avg pool
        return self.head(pooled)                  # [B, n_classes]


# ── Utility ──────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(0)
    model = TopoClassifier()
    n     = count_params(model)
    print(f"TopoClassifier total trainable parameters: {n:,}")

    imgs = torch.zeros(2, 4, 128, 128)
    out  = model(imgs)
    assert out.shape == (2, N_CLASSES), f"Bad shape: {out.shape}"
    print(f"Forward pass OK: {out.shape}")
    print("All checks passed.")
