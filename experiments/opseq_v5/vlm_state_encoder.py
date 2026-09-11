"""
vlm_state_encoder.py — Render a 6-view triptych image for VLM consumption.

For each camera view produces three columns:
  [current render | GT silhouette | error heatmap + candidate overlays]

All 6 views are stacked vertically → final PNG is H=6*IMG_RES, W=3*IMG_RES.

Candidate regions are drawn as filled semitransparent circles + numbered labels
so the VLM can refer to them by number (e.g. "region 2").

Usage
-----
    from vlm_state_encoder import encode_state

    png_bytes = encode_state(
        pred_sils,       # [N_V, H, W] float32 (0..1)
        gt_uint8,        # [N_V, H, W] uint8 (0=fg, 255=bg)
        pred_depths,     # [N_V, H, W] float32 NDC-z (unused except for reference)
        candidates,      # list[dict]  — each has: {"label": int, "center_px": [(x,y)...]}
        step: int,
        iou: float,
    )  -> bytes
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import io
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


# ── palette for candidate regions ────────────────────────────────────────────
_CANDIDATE_COLORS = [
    (255,  80,  80),   # 1: red
    (80,  200,  80),   # 2: green
    (80,  120, 255),   # 3: blue
    (255, 200,  50),   # 4: yellow
    (200,  80, 255),   # 5: purple
    (80,  230, 230),   # 6: cyan
]

# Radius (px) of the region circle drawn on the error heatmap
_REGION_RADIUS = 20
_LABEL_OFFSET  = 4    # px offset for text from circle centre


def _sil_to_rgb(sil_hw: np.ndarray) -> np.ndarray:
    """Float [H,W] silhouette → [H,W,3] uint8 grayscale."""
    gray = np.clip(sil_hw * 255, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _gt_to_rgb(gt_hw: np.ndarray) -> np.ndarray:
    """uint8 [H,W] GT mask (0=fg) → [H,W,3] white-on-black RGB."""
    fg = (gt_hw < 128).astype(np.uint8) * 255
    return np.stack([fg, fg, fg], axis=-1)


def _error_to_heatmap(sil_hw: np.ndarray, gt_hw: np.ndarray) -> np.ndarray:
    """Compute per-pixel error and render as red-on-black heatmap [H,W,3]."""
    pred_fg = sil_hw > 0.5
    gt_fg   = gt_hw  < 128
    # Missing pixels (GT fg but not pred): bright red
    missing = gt_fg  & (~pred_fg)
    # Extra pixels (pred fg but not GT):  blue
    extra   = pred_fg & (~gt_fg)
    rgb = np.zeros((*sil_hw.shape, 3), dtype=np.uint8)
    rgb[missing, 0] = 220   # R channel for missing
    rgb[extra,   2] = 180   # B channel for extra
    return rgb


def _draw_candidate_overlays(
    rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    view_idx: int,
) -> np.ndarray:
    """Overlay candidate circles + numbers on an [H,W,3] uint8 array.

    Each candidate has:
      "label"     : int — 1-based region number
      "center_px" : list of (x, y) pixel coords for each view

    Only centres for *this* view are drawn.
    """
    if not _HAVE_PIL:
        return rgb   # No PIL → skip overlay (image still valid)

    img  = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for cand in candidates:
        label      = int(cand.get("label", 0))
        centres_px = cand.get("center_px", [])
        if view_idx >= len(centres_px):
            continue
        cx, cy = centres_px[view_idx]
        if cx < 0 or cy < 0:
            continue   # candidate not visible in this view

        color_idx = (label - 1) % len(_CANDIDATE_COLORS)
        r, g, b   = _CANDIDATE_COLORS[color_idx]

        # Semitransparent filled circle
        r0 = max(0, cx - _REGION_RADIUS)
        c0 = max(0, cy - _REGION_RADIUS)
        r1 = cx + _REGION_RADIUS
        c1 = cy + _REGION_RADIUS
        draw.ellipse(
            [(r0, c0), (r1, c1)],
            fill=(r, g, b, 80),
            outline=(r, g, b, 220),
            width=2,
        )

        # Number label
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:
            font = ImageFont.load_default()
        draw.text(
            (cx - _LABEL_OFFSET, cy - _LABEL_OFFSET),
            str(label),
            fill=(255, 255, 255, 255),
            font=font,
        )

    return np.array(img)


def encode_state(
    pred_sils:   np.ndarray,            # [N_V, H, W] float32
    gt_uint8:    np.ndarray,            # [N_V, H, W] uint8
    pred_depths: Optional[np.ndarray],  # [N_V, H, W] float32 or None
    candidates:  List[Dict[str, Any]],  # region descriptors
    step:        int,
    iou:         float,
) -> bytes:
    """Compose the 6-view triptych and return PNG bytes.

    Parameters
    ----------
    pred_sils : [N_V, H, W] float32
        Current rendered silhouettes (0..1).
    gt_uint8 : [N_V, H, W] uint8
        GT silhouette masks (0=fg, 255=bg).
    pred_depths : [N_V, H, W] float32 or None
        NDC-z depth renders — reserved for future columns, currently unused.
    candidates : list[dict]
        Each dict: {"label": int, "center_px": [(x,y) per view]}.
    step : int
        Current optimisation step (for header text).
    iou : float
        Current mean IoU (for header text).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    if not _HAVE_PIL:
        raise ImportError("Pillow is required for encode_state: pip install pillow")

    N_V, H, W = pred_sils.shape
    # 3 columns: render | GT | error+candidates
    col_w  = W
    canvas_h = N_V * H
    canvas_w = 3 * W
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for vi in range(N_V):
        row_y = vi * H

        # Column 0: current render
        canvas[row_y:row_y+H, 0:W, :] = _sil_to_rgb(pred_sils[vi])

        # Column 1: GT silhouette
        canvas[row_y:row_y+H, W:2*W, :] = _gt_to_rgb(gt_uint8[vi])

        # Column 2: error heatmap + candidate overlays
        err_rgb = _error_to_heatmap(pred_sils[vi], gt_uint8[vi])
        err_rgb = _draw_candidate_overlays(err_rgb, candidates, vi)
        canvas[row_y:row_y+H, 2*W:3*W, :] = err_rgb

    # Draw divider lines between views
    img  = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(img)
    for vi in range(1, N_V):
        y = vi * H
        draw.line([(0, y), (canvas_w - 1, y)], fill=(80, 80, 80), width=1)
    # Dividers between columns
    for col in [1, 2]:
        x = col * W
        draw.line([(x, 0), (x, canvas_h - 1)], fill=(80, 80, 80), width=1)

    # Header text (on top of view 0, col 0)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    labels = ["Render", "GT", "Error+Candidates"]
    for ci, lbl in enumerate(labels):
        draw.text((ci * W + 4, 2), lbl, fill=(255, 220, 0), font=font)
    draw.text((4, 14), f"step={step}  IoU={iou:.4f}", fill=(200, 200, 200), font=font)

    # Candidate legend (top-right of col 2)
    legend_x = 2 * W + 4
    legend_y = 30
    for cand in candidates[:6]:
        label = int(cand.get("label", 0))
        desc  = cand.get("description", f"region {label}")
        color_idx = (label - 1) % len(_CANDIDATE_COLORS)
        r, g, b   = _CANDIDATE_COLORS[color_idx]
        draw.text(
            (legend_x, legend_y),
            f"{label}: {desc[:24]}",
            fill=(r, g, b),
            font=font,
        )
        legend_y += 13

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
