#!/usr/bin/env python3
"""sam2_masks.py <dataset_dir> : refine <dir>/masks (rembg) into <dir>/masks_sam2 with SAM2.1-large.
Prompt per frame = bbox of the rembg mask (+8% margin) + positive points: centroid and 4 interior points
(distance-transform peaks) + negative points just outside the bbox corners. Largest component kept, holes filled."""
import sys, os, glob, time, numpy as np, cv2, torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
D = sys.argv[1]; out = f"{D}/masks_sam2"; os.makedirs(out, exist_ok=True)
pred = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", os.path.expanduser("~/.cache/sam2/sam2.1_hiera_large.pt"), device="cuda"))
t0 = time.time(); ious = []
for mp in sorted(glob.glob(f"{D}/masks/*.png")):
    nm = os.path.basename(mp)[:-4]; m0 = cv2.imread(mp, 0) > 127; img = cv2.cvtColor(cv2.imread(f"{D}/images/{nm}.jpg"), cv2.COLOR_BGR2RGB)
    ys, xs = np.nonzero(m0); h, w = m0.shape; mg = 0.08
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); bw, bh = x1 - x0, y1 - y0
    box = np.array([max(x0 - mg * bw, 0), max(y0 - mg * bh, 0), min(x1 + mg * bw, w - 1), min(y1 + mg * bh, h - 1)])
    pts = [[xs.mean(), ys.mean()]]
    sub = np.stack([xs, ys], 1)[::max(1, len(xs) // 4000)].astype(np.float32)          # k-means centroids spread over the object (head, body, base)
    _, _, cen = cv2.kmeans(sub, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    for c in cen:
        if m0[int(c[1]), int(c[0])]: pts.append([float(c[0]), float(c[1])])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(img)
        masks, scores, _ = pred.predict(point_coords=np.array(pts, np.float32), point_labels=np.ones(len(pts), int), box=box[None], multimask_output=True)
    iou3 = [((mm > 0.5) & m0).sum() / ((mm > 0.5) | m0).sum() for mm in masks]; masks = [masks[int(np.argmax(iou3))]]
    m = (masks[0] > 0.5) & (cv2.dilate(m0.astype(np.uint8), np.ones((int(0.04 * max(bw, bh)) | 1,) * 2, np.uint8)) > 0)   # SAM2 boundary, but never beyond a dilated rembg mask (kills background leaks)
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if nlab > 2: m = lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); 
    ff = m.copy(); cv2.floodFill(ff, None, (0, 0), 1); m = (m | (ff == 0)).astype(bool)       # fill holes
    cv2.imwrite(f"{out}/{nm}.png", m.astype(np.uint8) * 255); ious.append((m & m0).sum() / (m | m0).sum())
print(f"[sam2] {len(ious)} masks in {time.time()-t0:.0f}s | IoU vs rembg min/med {min(ious):.3f}/{np.median(ious):.3f} | most changed: {[os.path.basename(p)[:-4] for p,_ in sorted(zip(sorted(glob.glob(f'{D}/masks/*.png')), ious), key=lambda t: t[1])[:5]]}")
