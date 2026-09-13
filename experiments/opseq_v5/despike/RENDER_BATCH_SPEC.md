# Batched multi-view rendering for the DR loop (spec, 2026-09-13)

## Why (measured, cProfile on Stage 6, armadillo 51k F, 150 steps = 115 s)
The DR loop renders the 64 training views ONE AT A TIME (`for i in range(NV)` in phase4_inloop.py ~L401 and
run_64v.py ~L191). Per step: 66 rasterize launches, 64 interpolate/antialias chains, `vertex_normals` recomputed
64x, and `depth_loss_masked` does `int(mask.sum().item())` per view = 64 GPU syncs per step (9.3 s of 115).
Backward is 41 % because it walks 64 separate graph branches. DLFL ops + self-intersection are already ~0 s.
Target: same optimisation, 2-3x faster loop, by rendering all 64 views in ONE batched nvdiffrast call.

## Non-goals
- No change to losses, weights, schedules, remeshing, seeds, or any file outside the two loop files + a helper.
- No change to the per-view loop code path: keep it, selectable with `RENDER_BATCH=0` (default `1`).
- Do NOT edit the main checkout while bg task `v4cpp_fert_seeds` runs (it imports these modules per stage).
  Work in a git worktree: `git worktree add /home/kingy/Projects/Genesis/GenesisTopmod-wt-batch -b render-batch`
  (from HEAD of main). All validation runs in the worktree. Data: `/tmp/liou_cow_viz` -> main repo `out_liou`
  (read the BASE_NPZ from there, absolute path); use unique TAGs `batchtest_*` for outputs so nothing collides;
  if any input data dir is untracked and missing in the worktree, symlink it from the main checkout.

## Deliverables
1. `run_64v.py`: `render_sdd_batch(ctx, verts_t, faces_t, mvps, views)` with `mvps [N,4,4]`, `views [N,4,4]`
   (they are already stacked tensors, see `star_cameras`). Returns `sil [N,H,W]`, `ndc_z [N,H,W]`, `fg [N,H,W]` bool,
   `diff [N,H,W]` — the same quantities `render_sdd` returns per view (drop the leading 1 of `sil[0]`).
   - clip positions: `pos = (mvps @ verts_h.T).transpose(1,2)` -> `[N,V,4]` contiguous; one `dr.rasterize`.
   - `ones` attr `[1,V,3]` broadcast is NOT supported by nvdiffrast for batch>1 — pass `[N,V,3]` (expand+contiguous) or
     use `dr.interpolate` with `rast` batch and attr `[N,V,C]`. Same for `clip_zw` (`pos[..., 2:4]`).
   - `vertex_normals` ONCE per step; camera-space z = `einsum('nij,vj->nvi', views[:, :3, :3], vn)[..., 2].abs()`
     -> attr `[N,V,1]`.
   - `dr.antialias(color, rast, pos, faces)` batched as-is.
2. Batched losses (in the loop file, or a small helper module `despike/batch_losses.py`):
   - silhouette: `F.l1_loss(sil, targets[..., 0])` — mean over N*H*W equals mean of per-view means (equal pixel
     counts), so identical to the loop up to float summation order.
   - depth: replicate `depth_loss_masked` per view WITHOUT `.item()`: `mask = fg & gtfg [N,H,W]`;
     `cnt = mask.sum((1,2))`; `err = (|ndc_z - gtd| * mask).sum((1,2))`; per-view `torch.where(cnt >= 4, err / cnt.clamp(min=1), 0)`;
     `dl = per_view.sum() / NV`. This equals the loop's `sum_i l1(pred[mask_i], gt[mask_i]) / NV` with views
     under 4 pixels contributing 0 (loop returns `pred.sum()*0`).
   - diffuse: `F.l1_loss(diff, gtdf_stack)` with `gtdf_t` stacked to `[N,H,W]` (same argument as silhouette).
   - `gtd_t`, `gtfg_t`, `gtdf_t` are Python lists of per-view tensors today: build stacked copies once at startup
     (keep the lists for the other callers: `face_residual`, exam, snapshots).
3. Wire into BOTH loops behind `RENDER_BATCH` env (default `1`): phase4_inloop.py main loop (~L399-409, MODE=="64v"
   branch; `W_NORMAL>0` branch may stay per-view, it is off in the golden chain) and run_64v.py training loop
   (~L189-195). The non-64v mode (`render_sil_and_depth`) stays per-view.
4. `despike/test_render_batch.py`: on `results_genus/armadillo_v4_auto.npz` (49.9k V) and the coarse
   `/tmp/liou_cow_viz/cow_armadillo_v4cpp2_early.npz` (or any *_early.npz) with the armadillo 64-view scene:
   (a) sil/ndc_z/fg/diff from `render_sdd_batch` vs stacked per-view `render_sdd`: `torch.equal` for fg and
   `allclose(atol=1e-6)` for the floats (rasterization is deterministic; same triangles, same barycentrics);
   (b) the three batched losses vs loop losses: `allclose(rtol=1e-5)`; (c) `loss.backward()` gradients on verts_t:
   `allclose(rtol=1e-4, atol=1e-7)`; (d) timing: forward+backward of one step, loop vs batch, print ms and speedup.
5. Equivalence run (in the worktree, GPU guard: read `/usr/lib/wsl/lib/nvidia-smi` used MiB, run only if < 28000;
   each run needs ~2 GB): Stage 6 config for 150 steps on armadillo from `/tmp/liou_cow_viz/cow_armadillo_v4cpp2_p4g.npz`,
   once with `RENDER_BATCH=0` and once with `RENDER_BATCH=1`, same SEED=0:
   ```
   cd <worktree>/experiments/opseq_v5 && env MODE=64v SHAPE=armadillo TAG=batchtest_s6 ADAM_BETAS=0.8,0.8 PALF_LAP=0.02 \
     PALF_CLIP=10 LR_EDGE=0.3 ADAPT_REMESH=1 ADAPT_MODE=velocity ADAPT_NU_GAIN=0.2 ADAPT_LMIN_PX=1.3 ADAPT_MAX_F=100000 \
     ADAPT_SI_GATE=0.3 FLIP_EVERY=25 COLLAPSE_EVERY=50 COLLAPSE_RATIO=0.4 SI_PUSH=0.15 STEPS=150 LAP_MULT=3 \
     BASE_NPZ=/tmp/liou_cow_viz/cow_armadillo_v4cpp2_p4g.npz RENDER_BATCH=1 python3 -u despike/phase4_inloop.py
   ```
   Reference (loop, measured 2026-09-13 with another job sharing the GPU): `[final] V=25627 F=51250 watertight=True |
   train=0.9944 ho16=0.9964 hair=0 | SI=0 (0.0%) folds=0.0%`, 115 s wall incl. ~10 s startup.
   Acceptance: watertight, |ho16 diff| <= 0.001, |V diff| <= 2 % (remesh decisions may drift by float order),
   wall time of the 150 steps >= 2x faster than the RENDER_BATCH=0 run made in the same session.
   Also run the Stage-1-style loop once (run_64v.py, STOP_AFTER=cc3 as in golden_v3_chain.sh Stage 1, any shape,
   RENDER_BATCH=1) to prove that path works; compare its `[final]`/cc3 IoU to the RENDER_BATCH=0 run.

## Notes
- nvdiffrast: `rasterize(ctx, pos [N,V,4], tri [F,3])` batched with shared topology; `interpolate(attr [N,V,C], rast, tri)`;
  `antialias(color [N,H,W,C], rast, pos, tri)`. Peak VRAM at N=64, 256^2, 100k F should stay < 3 GB — print it
  (`torch.cuda.max_memory_allocated`) in the equivalence run.
- Keep `[vram]` / `[final]` / `[adapt]` log lines byte-identical in format (the chain greps them).
- Commit on branch `render-batch` in the worktree; do not merge. Report: test output, both `[final]` lines, both wall
  times, peak VRAM.
