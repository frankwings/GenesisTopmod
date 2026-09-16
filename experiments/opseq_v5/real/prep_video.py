#!/usr/bin/env python3
"""prep_video.py: orbit video -> sharp frames + rembg masks + pycolmap poses (Plan A step 1).
Usage: python3 prep_video.py <video.mp4> <out_dir> [--n 72] [--sharp-window 10]
Output: out_dir/images/f_XXX.jpg, out_dir/masks/f_XXX.png (255=object), out_dir/sparse/0 (COLMAP), out_dir/prep.json
Background features are kept for SfM (static desk helps pose); masks are for the reconstruction stage only."""
import sys, os, json, argparse, subprocess, time, shutil
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=72)
ap.add_argument("--sharp-window", type=int, default=10); ap.add_argument("--skip-colmap", action="store_true"); a = ap.parse_args()
os.makedirs(f"{a.out}/images", exist_ok=True); os.makedirs(f"{a.out}/masks", exist_ok=True)
t0 = time.time()
cap = cv2.VideoCapture(a.video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
stride = max(1, N // a.n); print(f"[prep] {N} frames @ {fps:.1f} fps -> stride {stride}, target {a.n}", flush=True)
# sharpness (Laplacian variance) per frame, pick sharpest within each window
sharp = np.zeros(N); i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(fr, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY); sharp[i] = cv2.Laplacian(g, cv2.CV_64F).var(); i += 1
N = i; picks = []
for s in range(0, N - stride + 1, stride):
    w = min(a.sharp_window, stride); c = s + (stride - w) // 2; picks.append(int(c + np.argmax(sharp[c:c + w])))
cap = cv2.VideoCapture(a.video); i = 0; k = 0; names = []
while True:
    ok, fr = cap.read()
    if not ok: break
    if i in picks: nm = f"f_{k:03d}.jpg"; cv2.imwrite(f"{a.out}/images/{nm}", fr, [cv2.IMWRITE_JPEG_QUALITY, 95]); names.append(nm); k += 1
    i += 1
print(f"[prep] wrote {len(names)} frames ({time.time()-t0:.0f}s), sharpness median {np.median(sharp[picks]):.0f} vs all {np.median(sharp):.0f}", flush=True)
# masks
from rembg import remove, new_session; from PIL import Image
sess = new_session("isnet-general-use"); fg = []
for nm in names:
    im = Image.open(f"{a.out}/images/{nm}"); al = np.array(remove(im, session=sess, only_mask=True))
    m = (al > 127).astype(np.uint8)
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(m)            # keep largest component only
    if nlab > 2: m = (lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cv2.imwrite(f"{a.out}/masks/{nm[:-4]}.png", m * 255); fg.append(float(m.mean()))
print(f"[prep] masks done ({time.time()-t0:.0f}s), fg fraction min/med/max {min(fg):.3f}/{np.median(fg):.3f}/{max(fg):.3f}", flush=True)
info = dict(video=a.video, n_frames_video=N, picks=picks, names=names, fg=fg)
if not a.skip_colmap:
    import pycolmap
    db = f"{a.out}/database.db"; sp = f"{a.out}/sparse"
    for p in (db,): 
        if os.path.exists(p): os.remove(p)
    shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "SIMPLE_RADIAL"
    eo = pycolmap.FeatureExtractionOptions()
    try:
        eo.sift.max_num_features = 16384
        if os.environ.get('SIFT_ROBUST'): eo.sift.estimate_affine_shape = True; eo.sift.domain_size_pooling = True; print('[colmap] affine-shape + DSP SIFT on')
    except Exception as e: print(f"[colmap] max_num_features not set ({e})")
    pycolmap.extract_features(db, f"{a.out}/images", camera_mode=getattr(pycolmap.CameraMode, os.environ.get('CAM_MODE', 'SINGLE')), reader_options=ro, extraction_options=eo)
    print(f"[colmap] features ({time.time()-t0:.0f}s)", flush=True)
    so = pycolmap.SequentialPairingOptions(); so.overlap = 25
    pycolmap.match_sequential(db, pairing_options=so); print(f"[colmap] sequential matching overlap 25 ({time.time()-t0:.0f}s)", flush=True)
    mo = pycolmap.FeatureMatchingOptions(); mo.guided_matching = bool(os.environ.get('SIFT_ROBUST'))
    pycolmap.match_exhaustive(db, matching_options=mo); print(f"[colmap] exhaustive matching ({time.time()-t0:.0f}s)", flush=True)
    maps = pycolmap.incremental_mapping(db, f"{a.out}/images", sp)
    best = max(maps.items(), key=lambda kv: kv[1].num_reg_images()) if maps else None
    if best is None: print("[colmap] FAILED: no model"); sys.exit(1)
    idx, rec = best
    print(f"[colmap] models={len(maps)} best={idx}: registered {rec.num_reg_images()}/{len(names)} images, {rec.num_points3D()} pts, mean reproj err {rec.compute_mean_reprojection_error():.2f}px, cam {rec.cameras[1].model.name} {rec.cameras[1].params}", flush=True)
    if idx != 0: shutil.rmtree(f"{sp}/0", ignore_errors=True); shutil.copytree(f"{sp}/{idx}", f"{sp}/0")
    info.update(colmap=dict(registered=rec.num_reg_images(), points=rec.num_points3D(), reproj=rec.compute_mean_reprojection_error(), camera=str(rec.cameras[1])))
json.dump(info, open(f"{a.out}/prep.json", "w"), indent=1); print(f"[prep] done {time.time()-t0:.0f}s", flush=True)
