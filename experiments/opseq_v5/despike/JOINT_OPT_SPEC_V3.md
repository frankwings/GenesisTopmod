# JOINT_OPT_SPEC_V3 — real UV texture (coarse-to-fine) + depth-gradient prior + camera recovery test, then 3-way joint
Boss approved 2026-09-18. Builds on despike/phase8_joint.py (v1/v2) — keep v1/v2 reachable (TEX=0).

## Why (findings so far, all on real/dino3, input = silhouette-only mesh dino_real7_auto)
- v1 per-vertex colour, unshaded: surface crumpled (mean adjacent-face angle 47.9 deg vs 5.7 input); cameras moved 0.006 deg;
  a known 0.26 deg perturbation of the training poses was recovered 7 % (prior on) / -45 % (prior off).
- BUT that camera verdict was reached with a blurry appearance model: 52k vertex colours = one sample per 3.7 px at 512.
  Sharp features (belly stripes, eyes, teeth) are what pin a camera; vertex colour blurs them and absorbs the misalignment.
  => v3 re-tests camera refinement with a REAL texture before concluding anything (Boss's point).
- v2 mono depth (Depth Anything v2): gradient-dominant setting `dgrad` (W_DEPTH_L1=0.2 W_DEPTH_GRAD=1.0 W_SIL=6 SHADE=0
  W_PHO=0) is the best geometry so far: crumpling 2.7 deg, held-out sil IoU 0.9669 (input 0.9696), held-out depth err
  0.0133 (input 0.0163). Shading-only (d0s1) carved texture into geometry -> SHADE stays 0 in v3.

## 1. Texture appearance model (TEX=1)
- Input mesh: despike/results_genus/dino_joint_dgrad.npz (52,026 V / 104,048 F, genus 0). Topology fixed in v3.
- UV atlas: `xatlas.parametrize(V, F)` (installed; 21 s; 55,030 atlas verts). Cache to results_genus/<stem>_uv.npz
  (vmapping, indices, uvs). Positions keep the ORIGINAL index buffer F; UVs use the atlas index buffer `indices`
  (same face order) — nvdiffrast interpolate accepts a separate index buffer for attributes.
- Texture T: learnable [1, TEX_RES, TEX_RES, 3], TEX_RES=2048, init 0.5. Sample with
  `dr.texture(T, uv, uv_da, filter_mode="linear-mipmap-linear", max_mip_level=...)`, uv and uv_da from
  `dr.interpolate(uvs, rast, uv_idx, rast_db=rast_db, diff_attrs="all")` (rasterize must return rast_db).
- COARSE-TO-FINE (the part that matters for cameras): a mip bias schedule `mip_level_bias = B(step)` going linearly from
  B0=5 (texture effectively 64^2) to 0 over the first 60 % of the phase, then 0. Implement with the `mip_level_bias`
  argument of dr.texture (tensor [N,H,W]) — verify the argument exists in the installed nvdiffrast; if not, emulate by
  sampling an explicitly average-pooled copy of T.
- Photometric loss as v1 (rendered fg AND mask fg AND valid AND >= 3 px inside the mask, L1 clipped at PHO_CLIP=0.3),
  no shading. Texture TV regulariser W_TV=1e-3 on T.
- Pixels of T never seen by any training view stay 0.5: do not report them as error.

## 2. Camera recovery test (THE gate; geometry FROZEN, TEX=1)
Same perturbation as the v1 test: CAM_PERTURB_DEG=0.3, seed 0 (mean 0.256 deg, camera 0 untouched).
Variables: T, omega_i, t_i. Schedule: 2000 steps, coarse-to-fine as above; LR_T=1e-2, LR_CAM=5e-4 for the first 60 %,
then 2e-4; NO separate texture-only warm-up at full sharpness (that is what baked the misalignment in v1) — cameras
and texture start together at B0=5. Camera prior W_CAM=0.1 (weak). Losses: photometric + silhouette (silhouette
also carries camera gradient).
Report: mean/max residual rotation |omega_i + om_p_i| in deg, recovery % = 1 - residual/perturbation; also the same for a
control run WITHOUT perturbation (how far do cameras drift from COLMAP on their own: mean |omega|, mean |t|).
Ablation to isolate the texture effect: identical run with TEX=0 (vertex colour) -> expected ~7 % as before.
GATE: recovery >= 60 % with TEX=1. If it fails, STOP and report; do not run section 3.

## 3. Three-way joint (only if the gate passes): geometry + texture + cameras, unperturbed COLMAP poses
Geometry settings = dgrad (LR_V 1e-4, W_LAP_J 5x, W_NC 0.05, W_DEPTH_MONO=1 W_DEPTH_L1=0.2 W_DEPTH_GRAD=1.0, W_SIL=6,
hull field as before) + W_PHO=0.3 through the texture; cameras LR 2e-4, W_CAM=0.1; 2500 steps, mip bias 5 -> 0 over 60 %.
Two runs: `v3_cam0` (cameras frozen) and `v3_cam1` (cameras free) so the camera effect is measurable.
Metrics (table): held-out sil IoU, held-out PSNR-in-mask (held-out views use ORIGINAL poses; note this in the table),
held-out depth err, crumpling, mean |omega| / |t|, and TRAIN-view pose-consistency = median over the 64 training views of
"fraction of the mask covered by the hull carved from the 64 training masks with the given mvps" (use
hull_field.carve_hull at 256^3 with vote=2 in the normalised frame, project back, dilate 15 px like
real/hull_consistency.py). Baseline with COLMAP poses is ~0.87; if camera refinement is real this number goes UP.

## Outputs
results_genus/dino_joint_{tag}.npz (verts, tris, uvs, uv_idx) + real/dino3/tex_{tag}.png (the texture);
real/dino3/joint_{tag}_views.png (4 held-out views: photo | textured render | geometry-only SHADED render via
real/render_mesh.py --no-colors — NOT a flat silhouette); real/dino3/cams_joint_{tag}.json; one summary table
(markdown) covering: recovery test TEX=1, TEX=0, control drift, v3_cam0, v3_cam1.
## Constraints
GPU shared (check /usr/lib/wsl/lib/nvidia-smi; peak < 12 GB). No assert that aborts a run on a metric — print WARNING.
Do not modify golden_chain.sh, phase4_inloop.py losses, the DLFL kernel. /tmp/liou_cow_viz -> experiments/opseq_v5/out_liou
(ln -sfn if missing). Never `pkill -f` with a pattern that occurs in your own command line. State explicitly which
files you changed, and verify any pre-existing partial code against this spec instead of assuming it is complete.
