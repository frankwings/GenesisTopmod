# REAL_DATA_SPEC — run the golden (topo-carving) chain on a real captured object

Goal: `REAL_DATA=<dir> SHAPE=dino TAGP=real1 bash golden_v3_chain.sh` runs the full chain
(Stage 1 sphere->cc3 | 2 clean | 3 genus discovery by hull | 4 cc4+despike | 5 Palfinger loop | 5b | 6 LAP x3 | 7 Taubin | exam)
on a real dataset produced by `real/prep_video.py` + `real/sam2_masks.py`, with **no GT mesh**.
Synthetic path (no REAL_DATA) must stay bit-identical.

## Input dataset layout (already exists: `experiments/opseq_v5/real/dino3/`)
- `images/f_XXX.jpg` 1080x1920 portrait (cv2 auto-rotated; COLMAP width=1080 height=1920)
- `masks_sam2/f_XXX.png` 255 = object (use these; `masks/` = rembg fallback)
- `sparse/0/` pycolmap model, one SIMPLE_RADIAL camera (f, cx, cy, k), 145 registered images
- `prep.json` (names, picks)

## New module `despike/real_scene.py`
`load_real_scene(dir, device, n_train=64, n_hold=16, res=TRAIN_RES, hull_hires=512)` -> `RealScene` with:
- `names_train`, `names_hold`: choose 16 held-out frames uniformly over the registered sequence (every ~9th),
  then 64 training frames uniformly from the remainder. Deterministic.
- **World normalisation**: object centre X = least-squares intersection of the rays through each mask centroid
  (code exists in `real/` analysis snippets: A += I - d d^T, b += (I - d d^T) c). Scale s = 0.8 / (radius of the
  visual-hull bbox around X at 128^3 with 0.6*median camera distance half-width; compute hull once, take the
  max |voxel - X| of hull voxels). Normalised world: v' = (v - X) / s. Cameras: R unchanged,
  t' = (t + R X) / s  (cam_from_world). After this the object fits in radius ~0.8, like `normalize_to_range` shapes.
- **Square crop per view**: crop side S = 1.25 * max over ALL registered views of max(bbox_w, bbox_h) of the mask
  (global constant, so scale is consistent), centred per view on the mask bbox centre (clamped inside the frame).
  Adjusted intrinsics for the res x res render: f' = f*res/S, cx' = (cx - x0)*res/S, cy' = (cy - y0)*res/S.
  Ignore radial k (|k| < 0.02 -> < 0.3 % at the frame edge).
- **Projection**: build a 4x4 OpenGL-style MVP per view from (f', cx', cy', res) and cam_from_world so that
  `transform_to_clip(v, mvp)` + `dr.rasterize(..., resolution=[res,res])` reproduces the *cropped, resized mask*.
  Conventions are unverified (row order / y flip): DETERMINE THEM EMPIRICALLY — rasterize the marching-cubes mesh of the
  visual hull (skimage.measure.marching_cubes on the mask-carved 128^3 hull) with the candidate mvp and pick the
  y-flip (and if needed x-flip) that maximises IoU with the mask. Required: median IoU(hull render, mask) >= 0.85 over
  the 64 training views; print it. near/far: 0.1*dist .. 10*dist in normalised units.
- `views [N,4,4]`: world->camera matrices in the OpenGL camera frame (COLMAP cam: x right, y down, z forward ->
  OpenGL: x right, y up, z backward: flip y and z rows). Used only for headlight shading; W_DIFF is 0 on real data.
- `gt [64,res,res] uint8` in the run_64v convention: **< 128 = foreground** (i.e. 0 inside object, 255 background).
- `gtd = zeros [64,res,res] float32`, `gtdiff = zeros`, `max_r = 0.8`.
- `masks_hires`: the same crops resized to hull_hires x hull_hires bool, for the hull.
- `ho_mvps [16,4,4]`, `ho_gt [16,res,res]` for the held-out exam.
- `hull(ctx, extra_pts, nres=256, hires=512, vote=2)`: refactor `hull_field.build_vote_hull` so the voxel-carving
  loop is a function `carve_hull(fgs, mvps, lo, hi, nres, hires, vote, device)`; synthetic path renders the GT to
  get `fgs` and calls it (unchanged result); real path passes `masks_hires` (dilated 1 px like the synthetic
  `dilate=True`). Returns a `HullField` with `.hull` set, bbox lo/hi = normalised hull bbox +-0.02 widened by extra_pts.
- `heldout_exam(ctx, v, t)`: same signature/return as `phase1b_pipeline.heldout_exam` (iou, hair, maxblob) but on
  `ho_mvps/ho_gt` (render with render_views_n and compare to ho_gt < 128).

## Wiring (env `REAL_DATA`, empty = synthetic, untouched)
- `run_64v.py main()`: if REAL_DATA: `scene = real_scene.load_real_scene(...)`; mvps, views, gt, gtd, gtdiff, max_r
  from scene instead of `load_obj`/`star_cameras`/`make_gt`; `W_DEPTH = W_DIFF = 0` (force); `heldout_exam` -> scene's.
  `HULL_INIT` path: `hull_init_project` must take the hull from `scene.hull(...)` instead of `build_vote_hull(GT)`.
- `phase4_inloop.py` MODE=64v block: same substitutions (mvps/views/gt/gtd/gtdiff, HF = scene.hull(ctx, V), W_DEPTH=W_DIFF=0,
  `gtn_t=None`). `p1b.heldout_exam` calls -> scene's exam (set `phase1b_pipeline.heldout_exam = scene.heldout_exam`
  after import so every caller in despike sees it; check `cow_v13` / `escape_util` only consume `gt`/`mvps`).
- `phase7_handle.py` setup: same substitutions; `G_TARGET` from hull genus as today (GENUS_TARGET=hull).
- `phase5_taubin.py` / any exam that loads the GT: if REAL_DATA, report scene.heldout_exam only.
- `golden_v3_chain.sh`: `GT[$S]` lookup must not fail for unknown shapes (default "?" and the `[RESULT]` line prints
  hull genus); exam stage: if REAL_DATA set, run `python3 despike/exam_real.py raw=... taubin=...` (new, prints
  `[exam_real] <tag> V= F= watertight genus ho16=<iou> hair= maxblob=`) instead of `eval_cd_iou.py`.
- `HULL_PLUGS_CACHE` default path includes SHAPE -> fine.

## Acceptance
1. `python3 despike/real_scene.py real/dino3` prints: 64/16 split, S (crop px), chosen flip, median/min IoU(hull render vs mask) >= 0.85 / >= 0.7, hull voxel count, hull genus (expect 0).
2. `REAL_DATA=real/dino3 SHAPE=dino TAGP=real1 HULL_INIT=1 S1_STEPS=400 bash golden_v3_chain.sh` completes; final
   mesh `results_genus/dino_real1_auto.npz` watertight, genus 0, `[exam_real]` ho16 printed. Target ho16 >= 0.95 (silhouette-only).
3. Synthetic regression: `MODE=64v SHAPE=rockerarm TAG=rockerarm_regress_cc3 STOP_AFTER=cc3 python3 despike/run_64v.py`
   gives ho16 0.9489 hair 19473 (identical to golden v6 log) with REAL_DATA unset.
4. `pytest despike/test_hull_locate.py` still 5/5 (uses coarse meshes in /tmp/liou_cow_viz).
Notes: GPU is shared; keep resolutions as specified. Do not touch the DLFL kernel or losses. Real losses = silhouette only for this
step (photometric/vertex colour is the next spec).
