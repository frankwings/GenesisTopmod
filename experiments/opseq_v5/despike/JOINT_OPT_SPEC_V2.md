# JOINT_OPT_SPEC_V2 — monocular depth prior + shaded photometric term (Boss approved 2026-09-18)

Supersedes the geometry part of JOINT_OPT_SPEC.md. Findings from v1 (despike/phase8_joint.py, dino_real7_auto input):
- Unshaded per-vertex colour gives geometry NO meaningful photometric gradient: vertices jitter to tune colour
  interpolation -> crumpled surface (real/dino3/cmp_joint_geo.png), not fixed by LR_V/6 + 10x Laplacian
  (mesh_jointreg_geo_grid.png). Eyes/teeth are painted, not carved; arms still missing.
- Cameras are not identifiable in the joint form: a known 0.26 deg perturbation is recovered 7 % with the prior and
  -45 % without it (colour + vertices absorb the misalignment first). => CAMERAS FROZEN in v2 (JOINT_CAM=0 default;
  keep the code path, do not run it).

## What to add (extend despike/phase8_joint.py; keep v1 behaviour reachable with W_DEPTH_MONO=0 SHADE=0)
### 1. Monocular depth prior (Depth Anything v2)
- Model: HF `depth-anything/Depth-Anything-V2-Large-hf`, already in the local HF cache; load with HF_HUB_OFFLINE=1 via
  transformers AutoImageProcessor / AutoModelForDepthEstimation (verified: 2.8 s incl. load, 2.25 GB, sensible output —
  real/dino3/da2_f066.png). Output is RELATIVE INVERSE depth (larger = nearer), unknown scale and shift per image.
- New script `real/mono_depth.py <dataset_dir>`: runs the model on every image in <dir>/images at full resolution and
  caches float16 maps to <dir>/depth_da2/f_XXX.npy (skip existing). ~155 frames, < 2 min.
- `real_scene.py`: expose `scene.mono [64,res,res] float32` (and `scene.ho_mono [16,...]`): the cached maps cropped /
  resized (BILINEAR) / vertically flipped EXACTLY like the masks and rgb. Load lazily only if depth_da2 exists.
- Rendered quantity: per-pixel camera-space depth z_cam > 0 of the mesh (interpolate the camera-space z of the vertices:
  z_cam = -(view_w2c @ v).z in the OpenGL frame), then inverse depth q = 1 / z_cam.
- Loss (scale-shift invariant, per view i, MiDaS/MonoSDF style): on pixels P_i = rendered fg AND mask fg AND valid AND
  >= 3 px inside the mask boundary, solve the closed-form least squares (s_i, b_i) = argmin sum_P (s * m + b - q)^2 with
  m = mono map and q = rendered inverse depth DETACHED for the solve (s, b are constants w.r.t. the graph), then
  L_depth = mean_i mean_P | s_i * m + b_i - q |  / mean_P(q)   (normalised so the weight is scene-scale free).
  Guard: skip a view if |P_i| < 500 px or s_i <= 0 (report how many are skipped). Weight W_DEPTH_MONO (default 1.0).
- Also add the gradient-matching term of MiDaS at 2 scales (|grad(s m + b - q)| L1 on P, weight 0.5 of L_depth):
  it is what sharpens the arm/torso and mouth discontinuities.

### 2. Shaded photometric term
- SHADE=1 (default): rendered colour = albedo(vertex RGB) * shade, shade = AMB + (1 - AMB) * max(0, n_cam . l),
  n_cam = interpolated vertex normal in the camera frame (recompute vertex normals every step, as render_sdd_batch does),
  l = (0, 0, 1) towards the camera (headlight; the capture is an orbit under roughly frontal/ambient light), AMB = 0.5
  (env). Normals must carry gradient to the vertices (do not detach).
- Photometric weight lowered: W_PHO default 0.3 (was 1.0); PHO_CLIP 0.3 kept (specular highlights stay outliers).
- Phase A (colour only, 300 steps) now optimises the ALBEDO under the shading of the input mesh.

### 3. Geometry regularisation against crumpling (because v1 showed it is needed)
- LR_V default 1e-4; W_LAP_J default 5 x cow_v13.W_LAP; add a normal-consistency term W_NC * mean(1 - n_f . n_g) over
  adjacent face pairs (W_NC default 0.05). Report the back-dihedral median / mean adjacent-face angle before and after
  as the "crumpling" metric (v1 result must be measured too for the table: load results_genus/dino_joint_cam1.npz).

## Runs (all on despike/results_genus/dino_real7_auto.npz, REAL_DATA=real/dino3 TRAIN_RES=512 HULL_VOTE=8 HULL_DEAD=3)
| tag | W_DEPTH_MONO | SHADE | W_PHO | purpose |
|---|---|---|---|---|
| d1s0 | 1.0 | 0 | 0.0 | depth prior alone (no colour gradient to geometry at all: W_PHO=0 in phase B; colours still fitted in phase A for the pictures) |
| d1s1 | 1.0 | 1 | 0.3 | full v2 |
| d0s1 | 0.0 | 1 | 0.3 | shading alone |

## Outputs per run
results_genus/dino_joint_{tag}.npz (verts, tris, colors); real/dino3/joint_{tag}_geo_grid.png = GEOMETRY-ONLY shaded
render (real/render_mesh.py has no flag to ignore colours: add `--no-colors`); real/dino3/joint_{tag}_views.png
(4 held-out views: photo | shaded-colour render | geometry-only render | mono depth vs rendered inverse depth after
alignment); turntable mp4 of the geometry-only render. One comparison sheet real/dino3/cmp_joint_v2.png with rows
real7 input | v1 cam1 | d1s0 | d1s1 | d0s1 (geometry only, same 4 views as cmp_joint_geo.png).

## Acceptance (report the table)
| metric | requirement |
|---|---|
| watertight, genus 0 | all runs |
| held-out silhouette IoU | >= 0.9669 (input 0.9696) |
| crumpling: mean adjacent-face angle | d1s0 and d1s1 <= 1.3 x the real7 input value (v1 cam1 is expected to be several x) |
| held-out depth agreement: mean_P |s m + b - q| / mean q on the 16 HELD-OUT views | lower than the input mesh's value in d1s0 and d1s1 |
| wall | < 10 min per run; peak GPU < 10 GB (GPU is shared, check /usr/lib/wsl/lib/nvidia-smi) |
Visual judgement (arms separated from the belly, mouth cavity, eye sockets) is the Boss's: produce the images.
Constraints: do not touch golden_chain.sh, phase4_inloop.py losses, the DLFL kernel. /tmp/liou_cow_viz must be a symlink to
experiments/opseq_v5/out_liou (ln -sfn if missing). Never use `pkill -f` with a pattern that occurs in your own command line.
