#!/usr/bin/env python3
"""hull_consistency.py <dataset_dir> [--masks masks_sam2] [--model sparse/0] [--views a-b] [--N 128] [--slack 2]
Pose/mask sanity check BEFORE reconstruction: carve a visual hull from the masks + COLMAP poses, project it back into
every view and report the fraction of each mask covered by the hull (1.0 = poses+masks consistent, rigid object).
Also writes <dir>/hull_overlay.jpg (red = mask not covered by hull, blue = hull outside mask) for the worst/best views."""
import sys, os, argparse, numpy as np, cv2, pycolmap
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--masks", default="masks_sam2"); ap.add_argument("--model", default="sparse/0")
ap.add_argument("--views", default=""); ap.add_argument("--N", type=int, default=128); ap.add_argument("--slack", type=int, default=2); a = ap.parse_args()
rec = pycolmap.Reconstruction(os.path.join(a.dir, a.model)); cam = rec.cameras[1]; f, cx, cy, k = cam.params
ims = sorted(rec.images.values(), key=lambda im: im.name); names = [im.name[:-4] for im in ims]
if a.views: lo_, hi_ = map(int, a.views.split("-")); keep = [i for i, n in enumerate(names) if lo_ <= int(n[2:5]) <= hi_]; ims = [ims[i] for i in keep]; names = [names[i] for i in keep]
n = len(ims); Rcw = [im.cam_from_world().rotation.matrix() for im in ims]; tcw = [im.cam_from_world().translation for im in ims]
C = np.array([-R.T @ t for R, t in zip(Rcw, tcw)]); masks = [cv2.imread(f"{a.dir}/{a.masks}/{nm}.png", 0) > 127 for nm in names]
D = []
for R, m in zip(Rcw, masks):
    ys, xs = np.nonzero(m); d = np.array([(xs.mean() - cx) / f, (ys.mean() - cy) / f, 1.0]); d /= np.linalg.norm(d); D.append(R.T @ d)
A = np.zeros((3, 3)); b = np.zeros(3)
for c, d in zip(C, D): P = np.eye(3) - np.outer(d, d); A += P; b += P @ c
X = np.linalg.solve(A, b); r = np.median(np.linalg.norm(C - X, axis=1)); half = 0.6 * r; N = a.N
g = np.linspace(-half, half, N); G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + X; votes = np.zeros(len(G), np.int32)
def pr(P, R, t):
    p = P @ R.T + t; zz = p[:, 2]; xn = p[:, 0] / zz; yn = p[:, 1] / zz; rr = xn * xn + yn * yn
    return zz, f * xn * (1 + k * rr) + cx, f * yn * (1 + k * rr) + cy
for R, t, m in zip(Rcw, tcw, masks):
    zz, u, vv = pr(G, R, t); ok = (zz > 0) & (u >= 0) & (u < m.shape[1] - 1) & (vv >= 0) & (vv < m.shape[0] - 1)
    inside = np.zeros(len(G), bool); inside[ok] = m[vv[ok].astype(int), u[ok].astype(int)]; votes += inside
H = G[votes >= n - a.slack]; cov = []
for R, t, m in zip(Rcw, tcw, masks):
    zz, u, vv = pr(H, R, t); u = u.astype(int); vv = vv.astype(int); img = np.zeros(m.shape, bool); ok = (u >= 0) & (u < m.shape[1]) & (vv >= 0) & (vv < m.shape[0]); img[vv[ok], u[ok]] = True
    img = cv2.dilate(img.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0; cov.append((img & m).sum() / m.sum())
cov = np.array(cov); order = np.argsort(cov)
print(f"[hull_consistency] views {n} hull {len(H)} vox (slack {a.slack}) | mask coverage min {cov.min():.3f} p10 {np.percentile(cov,10):.3f} median {np.median(cov):.3f} | worst {[names[i] for i in order[:5]]}")
tiles = []
for i in list(order[:3]) + [order[-1]]:
    m = masks[i]; img = cv2.imread(f"{a.dir}/images/{names[i]}.jpg"); zz, u, vv = pr(H, Rcw[i], tcw[i]); u = u.astype(int); vv = vv.astype(int)
    hp = np.zeros(m.shape, bool); ok = (u >= 0) & (u < m.shape[1]) & (vv >= 0) & (vv < m.shape[0]); hp[vv[ok], u[ok]] = True; hp = cv2.dilate(hp.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ov = img.copy(); ov[m & ~hp] = (0.3 * ov[m & ~hp] + 0.7 * np.array([0, 0, 255])).astype(np.uint8); ov[hp & ~m] = (0.3 * ov[hp & ~m] + 0.7 * np.array([255, 0, 0])).astype(np.uint8)
    ys, xs = np.nonzero(m | hp); y0, y1, x0, x1 = max(ys.min()-40, 0), ys.max()+40, max(xs.min()-40, 0), xs.max()+40
    crop = cv2.resize(ov[y0:y1, x0:x1], (400, 400)); cv2.putText(crop, f"{names[i]} cov={cov[i]:.2f}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); tiles.append(crop)
cv2.imwrite(f"{a.dir}/hull_overlay.jpg", np.concatenate(tiles, 1)); print(f"[hull_consistency] wrote {a.dir}/hull_overlay.jpg")
