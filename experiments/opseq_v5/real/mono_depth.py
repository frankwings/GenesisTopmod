#!/usr/bin/env python3
"""Cache Depth Anything V2 inverse-depth maps for every image in <dataset_dir>/images/.

Usage:
    python3 real/mono_depth.py real/dino3

Output: float16 .npy files in <dataset_dir>/depth_da2/ (skips existing files).
Model:  depth-anything/Depth-Anything-V2-Large-hf  (loaded from local HF cache,
        HF_HUB_OFFLINE=1 to avoid network access).

Convention: the saved array is the RAW model output (relative INVERSE depth,
larger = nearer), same shape as the input image (H, W), float16.
No crop or flip is applied here — real_scene.py does that per-view.
"""
import sys, os, glob, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def run(dataset_dir: str) -> None:
    img_dir = os.path.join(dataset_dir, "images")
    out_dir = os.path.join(dataset_dir, "depth_da2")
    os.makedirs(out_dir, exist_ok=True)

    jpgs = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.JPG")) +
        glob.glob(os.path.join(img_dir, "*.png")) +
        glob.glob(os.path.join(img_dir, "*.PNG"))
    )
    if not jpgs:
        print(f"[mono_depth] No images found in {img_dir}", flush=True)
        return
    print(f"[mono_depth] {len(jpgs)} images found in {img_dir}", flush=True)

    # Count how many are already done
    existing = sum(
        1 for j in jpgs
        if os.path.exists(os.path.join(out_dir,
                                        os.path.splitext(os.path.basename(j))[0] + ".npy"))
    )
    if existing == len(jpgs):
        print(f"[mono_depth] All {len(jpgs)} maps already cached — nothing to do.", flush=True)
        return
    print(f"[mono_depth] {existing} already cached, {len(jpgs) - existing} to process", flush=True)

    print("[mono_depth] loading Depth-Anything-V2-Large model ...", flush=True)
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Use fp16 on GPU to save VRAM
    if device == "cuda":
        model = model.half()
    model.eval()
    print(f"[mono_depth] model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    done = existing
    t_start = time.time()
    with torch.no_grad():
        for jpg in jpgs:
            stem = os.path.splitext(os.path.basename(jpg))[0]
            out_path = os.path.join(out_dir, stem + ".npy")
            if os.path.exists(out_path):
                continue

            img = Image.open(jpg).convert("RGB")
            W_orig, H_orig = img.size  # PIL: (width, height)

            inputs = processor(images=img, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to(device).half() for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs)
            # predicted_depth: [1, H_model, W_model] — resize back to original
            pred = F.interpolate(
                out.predicted_depth.float().unsqueeze(0),   # ensure float32
                size=(H_orig, W_orig),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()                           # [H_orig, W_orig]

            np.save(out_path, pred.astype(np.float16))
            done += 1
            if done % 10 == 0 or done == len(jpgs):
                elapsed = time.time() - t_start
                rate = (done - existing) / max(elapsed, 1e-3)
                remaining = (len(jpgs) - done) / max(rate, 1e-3)
                print(f"[mono_depth] {done}/{len(jpgs)}  "
                      f"{rate:.1f} img/s  ETA {remaining:.0f}s", flush=True)

    print(f"[mono_depth] finished — {done} maps in {out_dir}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 real/mono_depth.py <dataset_dir>")
        sys.exit(1)
    run(sys.argv[1])
