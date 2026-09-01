"""Diagnose Phase1b-64v signal starvation: green (GT-gap) vs red (pred-excess)
composition per view, with and without the dilation used by find_candidates."""
import os, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label as cc_label
import nvdiffrast.torch as dr

import run_64v
from phase1b_pipeline import (load_obj, normalize_to_range, BUNNY_PATH,
                              render_sil_and_ids, DILATE, MIN_BLOB, DEVICE)

SHAPE = os.environ.get("SHAPE", "armadillo")
BASE_NPZ = "/tmp/liou_cow_viz/cow_armadillo_64v.npz"

ctx = dr.RasterizeCudaContext()
gv, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
mvps, views = run_64v.star_cameras(max_r)
gt, gtd, _, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
gt_fg = [(gt[i] < 128) for i in range(64)]

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)
ft = torch.tensor(np.asarray(Fa, np.int32), dtype=torch.int32, device=DEVICE)

rows = []
for i in range(64):
    sil, _ = render_sil_and_ids(ctx, vt, ft, mvps[i])
    pred = (sil > 0.5).cpu().numpy()
    green_raw = gt_fg[i] & ~pred
    red_raw = pred & ~gt_fg[i]
    green_dil = gt_fg[i] & ~binary_dilation(pred, iterations=DILATE)
    lab, nb = cc_label(green_dil)
    blobs = [int((lab == k).sum()) for k in range(1, nb + 1)]
    big = [b for b in blobs if b >= MIN_BLOB]
    rows.append((i, int(green_raw.sum()), int(red_raw.sum()),
                 int(green_dil.sum()), len(big), max(blobs) if blobs else 0,
                 pred, green_raw, red_raw))

g_tot = sum(r[1] for r in rows); r_tot = sum(r[2] for r in rows)
gd_tot = sum(r[3] for r in rows)
print(f"TOTAL over 64 views: green_raw={g_tot}  red_raw={r_tot}  "
      f"green/(green+red)={g_tot/(g_tot+r_tot):.2%}")
print(f"green after dilation(iter={DILATE}): {gd_tot}  "
      f"survival={gd_tot/max(g_tot,1):.2%}")
print(f"views with >=1 blob >= {MIN_BLOB}px: "
      f"{sum(1 for r in rows if r[4] > 0)}/64")
print("\nper-view (top 12 by green_raw):")
for i, gr, rr, gd, nbig, mx, *_ in sorted(rows, key=lambda r: -r[1])[:12]:
    print(f"  view {i:2d}: green={gr:5d} red={rr:5d} green_dil={gd:4d} "
          f"blobs>={MIN_BLOB}px: {nbig} maxblob={mx}")

# montage of 8 worst views: GT-gap green, excess red, overlap gray
worst = sorted(rows, key=lambda r: -(r[1] + r[2]))[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (i, gr, rr, gd, nbig, mx, pred, green, red) in zip(axes.flat, worst):
    img = np.ones((*pred.shape, 3))
    both = pred & gt_fg[i]
    img[both] = [0.6, 0.6, 0.6]
    img[green] = [0.0, 0.8, 0.0]
    img[red] = [0.9, 0.1, 0.1]
    ax.imshow(img); ax.set_title(f"v{i} g={gr} r={rr} gdil={gd}", fontsize=9)
    ax.axis("off")
plt.suptitle("Phase1b-64v error map: green=GT-gap (extrude can fix), "
             "red=pred-excess (extrude cannot)", fontsize=12)
plt.tight_layout()
plt.savefig("/tmp/liou_cow_viz/diag_p1b64_errmap.png", dpi=110)
print("\nsaved /tmp/liou_cow_viz/diag_p1b64_errmap.png")
