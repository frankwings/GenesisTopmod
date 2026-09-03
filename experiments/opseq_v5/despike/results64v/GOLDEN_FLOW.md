# Golden chain — step-by-step (armadillo, 64 training views, ho16 exam on 16 held-out views)

Everything below is pure GenesisTopmod: differentiable rendering (nvdiffrast) for vertex
positions + TopMod DLFL operators for every topology change (`check_watertight` asserted
after each). Supervision = 64 star-camera renders of the GT mesh (silhouette + depth +
Lambertian diffuse, 256²), identical to DMesh's mv-recon setup. No DMesh output is used.

```mermaid
flowchart TD
  S0([icosphere cc2, 320 f]) --> S1
  S1[1 run_64v.py<br/>C2F: cc2 800 → subdiv → cc3 800 → subdiv → cc4 800<br/>v22 escape-seeded surgery ×2 + settle 400<br/>7.0k f · ho16 0.9279]
  S1 --> S2[2 phase1c_pipeline.py ×2<br/>speculative GROW/CARVE via DLFL extrude_face<br/>K≤16 candidates per round, p-gated, 6 rounds each<br/>33+29 = 62 ops kept · 7.0k f · 0.9572]
  S2 --> S3[3 phase1f_hullpull.py HULL_MODE=vote<br/>voting visual hull (space carving) → EDT field<br/>800 steps, W_HULL 20, dead zone 1 voxel, anneal 200<br/>7.0k f · 0.9841]
  S3 --> S4[4 phase4_inloop.py<br/>same losses + hull field, 1200 steps<br/>every 25 steps: DLFL flip sweep + collapse_edge_tri + SI push + tangential smooth<br/>6.3k f · 0.9865]
  S4 --> S5[5 phase4_inloop.py SUBDIV_ALL=1 LAP_MULT=3<br/>global DLFL subdivide_edge + stellate → 36.5k f<br/>same 1200-step loop · 0.9914]
  S5 --> S6[6 phase5_taubin.py ITERS=5<br/>Taubin λ=0.5 μ=−0.53, positions only<br/>36.5k f · 0.9957 = golden v1]
  S6 --> S7[7 phase4_inloop.py SUBDIV_TOP=1500 LAP_MULT=3<br/>DLFL subdivide the 1500 largest faces (+1-ring) → 56.8k f<br/>same 1200-step loop · 0.9927]
  S7 --> S8[8 phase5_taubin.py ITERS=5<br/>56.8k f · **0.9972 = golden v2**]
```

## Stage details

### 0 · Scene (shared by all stages)
- GT `armadillo.obj` (100k f) normalized to our frame; 64 star cameras = 8 azimuths × 8 elevations
  (−70°..70°), radius 2× mesh max-radius, fov 53.13° — mirrors DMesh `make_star_cameras(8,8,distance=2)`.
- GT renders per view: silhouette (uint8), NDC depth (masked), diffuse (headlight). 256².
- Exam: 16 held-out views, IoU of silhouettes (`heldout_exam`) + hair_px (stray pixels) + maxblob.

### 1 · `run_64v.py` — coarse-to-fine from an icosphere → 0.9279
- Init: icosphere "cc2" (from `setup_scene`), seed 0.
- Loss: sil L1 + `W_DEPTH` masked depth L1 (warm-up) + diffuse L1 + Laplacian + edge-length +
  `L_qual` 0.01 (triangle quality) + spike + sliver + fold + tube penalties (cow_v13 recipe).
- Schedule: 800 steps @cc2 → `midpoint_subdivide` → 800 @cc3 → subdivide → 800 @cc4 (Adam, cosine lr).
- v22 despike surgery (needle removal, `NEEDLE_REMOVAL.md`): escape-seeded DLFL collapse, IoU budget
  3e-4/op, ≤8 rounds → settle 400 steps → second surgery (budget 2e-4, ≤4 rounds).
- Output `cow_armadillo_64v.npz`: 3,490 v / 6,976 f.

### 2 · `phase1c_pipeline.py` (run twice: p1d64b → p1d64c) — DLFL extrude carving → 0.9572
- Per round: rasterize error blobs vs GT silhouettes. GREEN blob (GT yes / pred no) → grow candidate,
  RED blob (pred yes / GT no) → carve candidate; blob ≥ 15 px.
- Candidate = real DLFL `extrude_face` on the face under the blob, distance ±3 × mean edge, applied via
  obj round-trip (formal manifold guarantee). Up to K=16 candidates per round.
- Selection: differentiable gating p_k (symmetric opacity compositing
  `S = S_base + Σ p_k (S_k − S_base)`) + depth term `W_DEPTH_SPEC · p_k (E_k − E_base)`; keep ops with
  p > 0.5, then settle vertices. 6 rounds per run.
- Kept program: 33 ops (run 1) + 29 ops (run 2) = 62, all carves/grows logged in
  `cow_armadillo_p1d64c_program.json` (triple, blob, sign, p, round).
- Output: 3,494 v / 6,984 f.

### 3 · `phase1f_hullpull.py HULL_MODE=vote` — space carving as a loss → 0.9841
- Hull (`hull_field.build_vote_hull`): 256³ voxels; a voxel is "outside" only if ≥ 2 of the 64
  training silhouettes (rasterized at 512², 1-px dilated) say so → robust to sub-pixel-thin fingers.
  Built ONLY from training views.
- Field: Euclidean distance transform to the hull (world units), trilinear `grid_sample`.
- Loss: `W_HULL=20 · mean relu(d − 1 voxel)` over 7 barycentric samples per face (area-weighted)
  + vertices; ramp-in 200 steps, anneal to 0 over the last 200 of 800 steps. Plus stage-1 losses.
- No topology change; the mid-slab webbing collapses onto the fingers by pure vertex motion.
- Output: 3,494 v / 6,984 f.

### 4 · `phase4_inloop.py` — DLFL cleanup inside the optimizer → 0.9865
- Losses = stage 3 (sil + depth + diffuse + hull field `W_T=20` + regularizers).
- Every `FLIP_EVERY=25` steps (no grad): `collapse_short_edges` every 100 steps (ratio 0.5, ≤300,
  DLFL `collapse_edge_tri`, link-condition guarded) → `flip_sweep` 3 passes (fold edges with
  n_a·n_b < 0 flipped via `delete_edge` + `insert_edge`, only if the new pair is less folded and
  non-degenerate) → SI push (Open3D self-intersecting pairs nudged ±0.15 × mean edge along the mean
  normal) → 1 tangential-smoothing iteration. Faces/adjacency rebuilt in place; if V changed, a fresh
  Adam at the current lr.
- 1200 steps. Output: 3,131 v / 6,258 f, SI 1.6 %, folds 0.1 %.

### 5 · `phase4_inloop.py SUBDIV_ALL=1 LAP_MULT=3` — global DLFL subdivision → 0.9914
- Pre-pass: DLFL `subdivide_edge` on all 9,387 edges, then `stellate` every non-triangle face
  (centroid split; fan triangulation created duplicate edges — rejected). 6,258 → 37,548 f.
- Same 1200-step loop as stage 4 with Laplacian ×3 (collapse ratio 0.4, ≤600).
- Output: 18,253 v / 36,502 f, SI 4.3 %.

### 6 · `phase5_taubin.py ITERS=5` — fairing → 0.9957 (golden v1)
- Open3D Taubin λ|μ (0.5 / −0.53), 5 iterations; positions only, connectivity untouched.
- Removes optimizer jitter (back dihedral 30.5° → 7.9°; GT-decimated reference 10.1°). IoU rises
  because the jitter was silhouette noise. SI 1.4 %, hair 8.

### 7 · `phase4_inloop.py SUBDIV_TOP=1500 LAP_MULT=3` — partial subdivision → 0.9927
- Pre-pass: DLFL subdivide the 1,500 largest faces (+1-ring): 36,502 → 58,122 f (resolution
  equalization, keeps mean edge ≥ 256² pixel size — global ×6 to 219k f tangled and was aborted).
- Same 1200-step loop (collapse ≤800). Output: 28,396 v / 56,788 f, SI 1.9 %.

### 8 · `phase5_taubin.py ITERS=5` — → **0.9972 (golden v2)**
- 28,396 v / 56,788 f, watertight genus 0, SI 0.5 %, folds 0.4 %, hair 1, back dihedral 7.3°.
- Ceiling at this face count (GT quadric-decimated) ≈ 0.999. DMesh best: 0.9894 (5.5k f, soup).

## DLFL operator inventory (all from `topmod/`)
| op | where | V/E/F effect |
|---|---|---|
| `extrude_face` | stage 2 | +n V, +2n E, +n F |
| `subdivide_edge` | stages 5, 7 | +1 V, +1 E, F unchanged |
| `stellate` | stages 5, 7 (after edge splits) | +1 V, +k E, +(k−1) F |
| `collapse_edge_tri` | stages 1 (surgery), 4, 5, 7 | −1 V, −3 E, −2 F |
| `delete_edge` + `insert_edge` (= flip) | stages 4, 5, 7 | unchanged |

## Reproduce
```
MODE=64v SHAPE=armadillo TAG=armadillo_64v python3 despike/run_64v.py
MODE=64v TAG=p1d64b BASE_NPZ=.../cow_armadillo_64v.npz   python3 despike/phase1c_pipeline.py
MODE=64v TAG=p1d64c BASE_NPZ=.../cow_armadillo_p1d64b.npz python3 despike/phase1c_pipeline.py
HULL_MODE=vote TAG=p1f64c STEPS=800 W_HULL=20 BASE_NPZ=.../p1d64c.npz python3 despike/phase1f_hullpull.py
MODE=64v TAG=p4c_64 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_MAX=300 SI_PUSH=0.15 BASE_NPZ=.../p1f64c(+3e).npz python3 despike/phase4_inloop.py
MODE=64v TAG=p5_64 SUBDIV_ALL=1 LAP_MULT=3 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.4 COLLAPSE_MAX=600 SI_PUSH=0.15 BASE_NPZ=.../p4c_64.npz python3 despike/phase4_inloop.py
MODE=64v ITERS=5 TAG=p5_64_taubin5 BASE_NPZ=.../p5_64.npz python3 despike/phase5_taubin.py
MODE=64v TAG=p6_50k SUBDIV_TOP=1500 LAP_MULT=3 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.4 COLLAPSE_MAX=800 SI_PUSH=0.15 BASE_NPZ=.../p5_64_taubin5.npz python3 despike/phase4_inloop.py
MODE=64v ITERS=5 TAG=p6_50k_taubin5 BASE_NPZ=.../p6_50k.npz python3 despike/phase5_taubin.py
```
