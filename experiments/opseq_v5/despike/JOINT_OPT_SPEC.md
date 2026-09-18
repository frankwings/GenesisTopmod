# JOINT_OPT_SPEC — joint DR of mesh + per-vertex colour + camera poses on real data (Boss, 2026-09-17)

Goal: on the dinosaur (`real/dino3`), starting from the silhouette-only golden result (`results_genus/dino_real7_auto.npz`,
52k V, genus 0, held-out sil IoU 0.9699), run a joint differentiable-rendering refinement that optimises
(a) vertex positions, (b) a per-vertex RGB colour, (c) a 6-DoF correction per training camera — simultaneously.
Motivation: silhouettes carry almost no information about the arms (they overlap the belly in most views) and none about
concavities (mouth, eyes); photometric consistency does, and it also pins each camera in-plane (belly stripes, eyes).
This is an experiment, NOT part of golden_chain.sh. New script `despike/phase8_joint.py`; real-data only (REAL_DATA set).

## Inputs (extend `real_scene.py`)
- `scene.rgb  [64,res,res,3] float32 in [0,1]`: the training photos cropped exactly like the masks (same S_px window,
  same resize, same vertical flip to nvdiffrast row order). `scene.ho_rgb [16,res,res,3]` for held-out views.
- Existing: `scene.gt` (0 = fg), `scene.valid` (inside original frame), `scene.mvps`, `scene.views`, `scene.hull(ctx)`.
- Camera decomposition: `real_scene` must also expose per-view intrinsics-projection `scene.proj [64,4,4]` and the
  world->camera view matrix `scene.view_w2c [64,4,4]` (OpenGL frame, already normalised world) so that
  `mvp_i = proj_i @ (delta_i @ view_w2c_i)` where `delta_i = [exp(omega_i) | t_i]` is the learnable correction
  applied in the CAMERA frame (small rotation about the camera centre + small translation). Verify at start:
  with omega = t = 0 the rebuilt mvps equal `scene.mvps` to 1e-5.

## Variables and initialisation
- V: from the input mesh (float32, requires_grad). F fixed (no remeshing in this experiment; DLFL flip_sweep every 50
  steps is optional via FLIP_EVERY, default off).
- C: per-vertex RGB, init = median photo colour sampled by projecting each vertex into all views where it is visible
  (rasterise vertex ids or use depth test) — or simply init 0.5 grey if the projection init is not ready in 1 h.
- omega_i, t_i: zeros, 64 x 3 each; camera 0 is frozen (reference frame); an L2 prior `W_CAM * (|omega|^2/sig_r^2 +
  |t|^2/sig_t^2)` with sig_r = 1 deg, sig_t = 0.01 (normalised units, object radius 0.8) keeps corrections small.
  JOINT_CAM=0 freezes all cameras (ablation).

## Rendering (nvdiffrast, batched over the 64 views like render_sdd_batch)
- rasterise V with the rebuilt mvps at res = TRAIN_RES (512); silhouette via antialias of a constant attribute (as now);
  colour image = antialias(interpolate(C, rast, F)); rendered-fg mask from rast[...,3] > 0.
- Headlight shading is NOT applied (per-vertex colour absorbs shading; the toy is glossy — see robust loss).

## Losses (per step, means over the 64 views)
1. silhouette: existing `sil_loss_batch(sil, soft_targets(gt, SIL_BLUR))` with `batch_losses.VALID` weighting.
2. photometric: `L_pho = mean over P of rho(|I_render - I_photo|_1)` where P = rendered fg AND mask fg AND valid AND
   NOT within 3 px of the mask boundary (erode the mask; segmentation edges are unreliable);
   rho(x) = min(x, PHO_CLIP=0.3) (specular highlights become outliers, not gradients). Weight W_PHO (default 1.0).
3. colour smoothness: `W_CLAP * mean |C - mean_neighbours(C)|^2` (W_CLAP default 0.05) — suppresses per-vertex noise.
4. geometry regularisers as in phase4_inloop: laplacian_loss (W_LAP * LAP_MULT), edge_length_loss (W_EDGE), fold_loss,
   spike_pen, hull-field loss with DEAD=HULL_DEAD voxels and vote HULL_VOTE (from scene.hull), MEMB_EXEMPT not needed.
5. camera prior (above).

## Schedule
- Phase A (STEPS_A=300): freeze V and cameras, optimise C only (lr 2e-2) -> meaningful albedo.
- Phase B (STEPS_B=1500): all variables. Adam: V lr = LR_V (default 3e-4 normalised units), C lr 1e-2, omega lr 2e-4,
  t lr 2e-4; cosine decay to 0.1x. Log every 100 steps: sil loss, pho loss, PSNR-in-mask (train), held-out sil IoU,
  held-out PSNR-in-mask (held-out cameras are NOT optimised: they use the original poses), mean |omega| (deg), mean |t|.
- Every 250 steps: check_watertight(F) (must hold — topology is fixed), self-intersection fraction (report only).

## Outputs
- `results_genus/dino_joint_{tag}.npz` with verts, tris, colors (float32 [V,3]); `real/dino3/cams_joint_{tag}.json`
  (omega, t per train view + summary stats).
- Renders: `real/dino3/joint_{tag}_views.png` = 4 held-out views x (photo | coloured render | silhouette-only render
  from the input mesh) ; turntable `real/dino3/joint_{tag}_turn.mp4` with vertex colours (extend real/render_mesh.py
  to accept `colors`, flat-shaded colour x soft key light).
- A markdown table in the log: input mesh vs phase A vs phase B: held-out sil IoU, held-out PSNR, camera stats.

## Acceptance
1. Sanity: omega=t=0 reproduces scene.mvps (max abs diff < 1e-5); phase A alone raises held-out PSNR-in-mask over the
   grey-init value; watertight after the run.
2. Ablation run JOINT_CAM=0 and JOINT_CAM=1 on dino_real7_auto; report both. Held-out silhouette IoU must not drop
   below 0.9699 - 0.003 in either.
3. Visual: `joint_{tag}_views.png` shows the mouth opening / eye sockets / arm-belly separation in the coloured render
   better than the silhouette-only mesh (Boss judges; produce the images).
4. Wall time < 20 min per run on the shared GPU (batched 64-view render at 512 was 2.9 GB peak; keep under 8 GB).
Do not modify golden_chain.sh, phase4_inloop.py losses, or the DLFL kernel. Reuse cow_v13/batch_losses functions.
