#!/usr/bin/env python3
"""
probe_vlm_reading.py — Isolate whether the VLM actually READS the error heatmap
or just anchors to region "1".

Reuses the real `encode_state` encoder. Builds a synthetic 6-view triptych where
the red "missing geometry" blob sits UNDER a KNOWN candidate number, and varies
that number across trials.

  - If VLM answer follows the red blob's label  -> it reads the image.
  - If VLM answer stays "1" regardless          -> it anchors to the number.
"""
from __future__ import annotations
import sys, os, numpy as np
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from vlm_state_encoder import encode_state
from vlm_client import VLMClient
import re

IMG_RES = 256
N_VIEWS = 6

_VLM_SYSTEM_PROMPT = """You are a 3D mesh topology advisor.
You will see a 6-row image. Each row shows one camera view:
  Column 1 (left):   current rendered silhouette (white=mesh, black=background)
  Column 2 (centre): ground-truth silhouette
  Column 3 (right):  error heatmap — red pixels = geometry MISSING in current render;
                     numbered coloured circles = candidate regions to extrude

Your task: identify which numbered region (1, 2, or 3) most consistently appears
in the red-pixel areas across the most views.

Reply with EXACTLY one line in this format:
REGION: <number>
REASON: <one sentence>

Do not add anything else."""


def _disk(cx, cy, r):
    yy, xx = np.mgrid[0:IMG_RES, 0:IMG_RES]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def build_synthetic(bite_center, cand_centers, red_label_idx):
    """
    gt = big foreground disk. pred = same disk minus a 'bite' at bite_center.
    -> the bite becomes red (missing) in the error heatmap.
    3 candidate circles placed at cand_centers; the one at index red_label_idx
    is the one sitting on the bite.
    """
    cx0, cy0 = IMG_RES // 2, IMG_RES // 2
    gt_fg = _disk(cx0, cy0, 90)
    bite = _disk(bite_center[0], bite_center[1], 34)
    pred_fg = gt_fg & (~bite)

    pred_sils = np.stack([pred_fg.astype(np.float32)] * N_VIEWS, axis=0)
    gt_uint8 = np.stack([np.where(gt_fg, 0, 255).astype(np.uint8)] * N_VIEWS, axis=0)

    candidates = []
    for lbl in range(1, 4):
        cx, cy = cand_centers[lbl - 1]
        candidates.append({
            "label": lbl,
            "center_px": [(cx, cy)] * N_VIEWS,
            "description": f"region {lbl}",
        })
    return pred_sils, gt_uint8, candidates


def parse(reply):
    m = re.search(r"REGION\s*:\s*(\d+)", reply, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def main():
    vlm = VLMClient()
    # 3 fixed candidate slots (left, right, bottom)
    slots = [(90, 110), (170, 110), (128, 175)]
    # Trials: put the red bite under a different labelled slot each time
    trials = [
        ("bite under label 1", slots[0], 0),
        ("bite under label 2", slots[1], 1),
        ("bite under label 3", slots[2], 2),
        ("bite under label 2 (repeat)", slots[1], 1),
        ("bite under label 3 (repeat)", slots[2], 2),
    ]
    print("=== VLM reading probe ===")
    results = []
    for name, bite, red_idx in trials:
        pred, gt, cands = build_synthetic(bite, slots, red_idx)
        png = encode_state(pred, gt, None, cands, step=0, iou=0.5)
        correct_label = red_idx + 1
        try:
            reply = vlm.ask_with_image(_VLM_SYSTEM_PROMPT, png)
            got = parse(reply)
        except Exception as e:
            reply = f"ERR {e}"; got = -1
        ok = "✅" if got == correct_label else "❌"
        results.append((name, correct_label, got, ok))
        print(f"{ok} {name}: correct={correct_label} vlm_said={got}")
        print(f"     reply={reply[:120]!r}")
    n_ok = sum(1 for _, c, g, _ in results if c == g)
    print(f"\nSCORE: {n_ok}/{len(results)} correct")
    if n_ok <= 1:
        print("VERDICT: VLM is NOT reading the heatmap (anchoring to a number).")
    elif n_ok >= len(results) - 1:
        print("VERDICT: VLM IS reading the heatmap.")
    else:
        print("VERDICT: VLM partially reads — noisy / unreliable.")


if __name__ == "__main__":
    main()
