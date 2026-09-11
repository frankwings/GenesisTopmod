#!/usr/bin/env python3
"""
model_v5.py — TopoShapeNet: dual-head CNN for topology classification + shape regression.

Plan C: predict {topology_class, coarse_shape_params} from 4-view silhouettes.

Input : [B, 4, 128, 128] float32 in [0, 1]
Output: (class_logits [B, N_CLASSES], shape_params [B, 3])

Shape params = [hx, hy, hz] = axis-aligned half-extents of the normalized mesh.
These capture the rough proportions (elongation, flattening) without rotation.

Usage:
    python model_v5.py    # prints param count + sanity check
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_V4_DIR     = os.path.join(os.path.dirname(_SCRIPT_DIR), 'opseq_v4')
for _p in (_REPO_ROOT, _SCRIPT_DIR, _V4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from canonical_topos import N_CLASSES


class TopoShapeNet(nn.Module):
    """
    Dual-head CNN: shared encoder -> classification head + shape regression head.

    Classification: predict which of 20 canonical topologies.
    Regression: predict 3 axis-aligned half-extents [hx, hy, hz].
    """

    def __init__(self, n_classes: int = N_CLASSES, d_model: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.d_model   = d_model

        # Shared CNN encoder (same as V4): 4->32->64->128->d_model
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

        # Classification head: -> N_CLASSES logits
        self.cls_head = nn.Linear(d_model, n_classes)

        # Shape regression head: -> 3 half-extents (always positive)
        # Two-layer MLP for slightly more capacity on regression
        self.shape_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softplus(),  # enforce positive outputs
        )

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

    def forward(
        self, images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        images : [B, 4, 128, 128] float32 in [0, 1]

        Returns
        -------
        cls_logits   : [B, n_classes]
        shape_params : [B, 3]  positive half-extents
        """
        features = self.encoder(images)          # [B, d_model, 8, 8]
        pooled   = features.mean(dim=[2, 3])     # [B, d_model]
        cls_logits   = self.cls_head(pooled)      # [B, n_classes]
        shape_params = self.shape_head(pooled)    # [B, 3]
        return cls_logits, shape_params


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = TopoShapeNet()
    n     = count_params(model)
    print(f"TopoShapeNet total trainable parameters: {n:,}")

    imgs = torch.zeros(2, 4, 128, 128)
    cls_logits, shape_params = model(imgs)
    assert cls_logits.shape == (2, N_CLASSES), f"Bad cls shape: {cls_logits.shape}"
    assert shape_params.shape == (2, 3), f"Bad shape shape: {shape_params.shape}"
    assert (shape_params >= 0).all(), "Shape params should be positive"
    print(f"Forward pass OK: cls={cls_logits.shape}, shape={shape_params.shape}")
    print("All checks passed.")
