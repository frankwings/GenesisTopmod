# GOLDEN v6.1 (2026-09-17) — five shapes, tunnels located on the MESH (Boss's formulation) + propose-and-verify
**Chain**: `despike/golden_chain.sh` (renamed from golden_v3_chain.sh), run with `TAGP=v6n`. Same stages as v5/v6; Stage 3 / 5b now:
1. **Topology oracle** g* = persistent genus of the space-carved hull (unchanged, LESSONS 24).
2. **Membrane detection on the DR mesh** (`membrane_locate.py`, `DETECT=membrane`): the hull never intersects the object, so a
   mesh FACE whose interior samples lie > 2 voxels outside the hull spans air = a membrane / a mouth of an unopened chamber.
   Faces -> edge-connected patches (split by normal sign when both skins are pressed together). Patches are kept only if
   they sit on a voxel block of "mesh material in hull air" whose removal raises the mesh genus (k >= 1); two patches on the
   same block are paired iff the segment between their faces runs through hull air AND mesh interior -> `add_handle(fi, fj)`.
   No closing radius, no throat-vs-mouth ambiguity (closing seals at the narrowest cross-section, which is not the mouth of a
   multi-exit chamber - fertility's upper cavity has 3 exits + 1 opening, needing 3 handles).
3. **Verify** (`phase7_multi.sh`): one handle per round -> 400 DR steps -> accept iff held-out IoU +0.002 or hair -10 %; else
   revert and blacklist the pair. Stage 5b runs the same loop on the refined mesh (safety net).
4. Membranes are exempt from the hull-field loss in Stages 2/3/5 (`MEMB_EXEMPT=2`) so they stay visible until opened.
**Meshes**: `results_genus/<shape>_v6n_auto.npz` (fertility: `_v6n1/2/3`), handles in `handles_<shape>_v6n*.json`.

| shape | genus (GT) | handles (verified / rejected) | VolIoU | CD | V | wall (uncontended) |
|---|---|---|---|---|---|---|
| armadillo | 0 (0) | 0 / 0 | 0.9949 | 0.00620 | 50.1k | 4.4 min |
| kitten | 1 (1) | 1 / 0 | 0.9980 | 0.00664 | 57.3k | 4.4 min |
| fertility x3 | 4 (4) x3 | 5/1, 4/0, 4/0 | 0.9909 / 0.9909 / 0.9900 | 0.00640-0.00656 | 48-51k | 5.5-6.2 min |
| rocker-arm | 1 (1) | 1 / 0 | 0.9871 | 0.00613 | 50.1k | 4.4 min |
| threeholes | 3 (3) | 3 / 0 | 0.9914 | 0.00707 | 48.8k | 5.0 min |

Fertility was the open problem: v6 (closing-ladder plugs) succeeded 1 run in 4 (handles landed inside the chamber, genus 4 but
tunnels unopened, VolIoU 0.84); v6.1 3/3 with every accepted handle DR-verified. Supersedes tag `golden-v6`. Tag: `golden-v6.1`.
Details: LESSONS §33.

---
# GOLDEN v6 (2026-09-15) — all five shapes, handles LOCATED by the space-carved hull (no rays), same chain otherwise
**Chain**: `despike/golden_v3_chain.sh` run with `TAGP=v6c`; identical to v5 except Stage 3 / 5b tunnel location now uses
`despike/hull_locate.py` (`DETECT=hull`, default) instead of ray casting. Ray detector kept only as a fallback — never triggered on the five shapes.
`hull_locate.py` (Boss's formulation): closing-radius ladder R=4..40 on the cleaned 128^3 voting hull; a genus drop at radius R
yields a sealing sheet (= essential increment component: removing it re-opens the tunnel); accepted plugs are FROZEN into the
closing source so later radii cannot re-claim them (structural mutual exclusion, fixes the threeholes double-plug); tunnel axis
from sheet shape (cylinder long axis / disk normal / skeleton centreline for S-shaped tunnels); the face pair comes from an
occupancy walk along the axis/centreline (out->in and in->out crossings -> nearest triangles) with the crossings required to lie
in hull air. Every handle is then added with the TopMod DLFL `add_handle` operator as before.
**Meshes**: `results_genus/<shape>_v6c_auto.npz` (Taubin) / `<shape>_v6c_raw.npz`; handles in `handles_<shape>_v6c.json` (source `hull` for all 9 handles).

| shape | genus (GT) | handles by hull | ho16 | VolIoU | CD | V | wall |
|---|---|---|---|---|---|---|---|
| armadillo | 0 (0) | 0 / 0 | 0.9982 | 0.9949 | 0.00617 | 49.4k | 12.2 min* |
| kitten | 1 (1) | 1 / 1 | 0.9993 | 0.9977 | 0.00666 | 58.6k | 12.6 min* |
| fertility | 4 (4) | 4 / 4 | 0.9973 | 0.9908 | 0.00647 | 46.6k | 16.9 min* |
| rocker-arm | 1 (1) | 1 / 1 | 0.9984 | 0.9870 | 0.00615 | 49.5k | 4.5 min |
| threeholes | 3 (3) | 3 / 3 | 0.9981 | 0.9915 | 0.00707 | 49.5k | 4.5 min |

\* wall measured while another 20-23 GB job shared the GPU; uncontended chain time is the v5 figure (4.3-5.6 min) — the hull locator itself is < 60 s / shape.
Scores equal v5 within 0.001 on every metric; genus 9/9 plugs = g* on all shapes, zero ray fallbacks, zero Stage-5b contact joins
(v5 fertility needed one). All watertight, SI 0 (fertility 0.1 %), hair 0. Unit test `test_hull_locate.py` 5/5 (< 60 s per shape).
Visualisation: `results_genus/hull_plugs_viz.png`. Git tag: `golden-v6`. Details: LESSONS §30-31, `HULL_LOCATE_SPEC.md`.

---
# GOLDEN v5 (2026-09-14) — all five shapes, all-TopMod chain, genus discovered, C++ kernel + batched render
**Chain**: `despike/golden_v3_chain.sh` (defaults now `DLFL_BACKEND=cpp`, `RENDER_BATCH=1`, `C2F_SUBDIV=cc`), run with `TAGP=v5`.
Stages: sphere -> TopMod CC2/CC3 + DR | DLFL clean 400 | genus discovery (rays + space-carved-hull target g*) |
CC4 + despike | Palfinger-param loop 1200 | late genus pass (contact join) | LAP x3 loop 1200 | AUTO Taubin | exam.
Every topology change is a TopMod DLFL operator (C++ kernel, bit-identical to the Python reference).
**Meshes**: `results_genus/<shape>_v5_auto.npz` (Taubin) / `<shape>_v5_raw.npz`; handles in `handles_<shape>_v5.json`.

| shape | genus (GT) | ho16 | VolIoU | CD | V | wall |
|---|---|---|---|---|---|---|
| armadillo | 0 (0) | 0.9983 | 0.9949 | 0.00620 | 49.5k | 5.4 min |
| kitten | 1 (1) | 0.9993 | 0.9979 | 0.00665 | 49.9k | 4.5 min |
| fertility | 4 (4) | 0.9971 | 0.9908 | 0.00644 | 45.9k | 5.6 min |
| rocker-arm | 1 (1) | 0.9985 | 0.9869 | 0.00615 | 54.3k | 4.3 min |
| threeholes | 3 (3) | 0.9980 | 0.9915 | 0.00706 | 53.8k | 5.0 min |

All watertight, SI 0 (fertility 0.2 %), hair 0. Fertility genus stable over 3 independent runs (4/4/4).
Competitors (same GPU): Palfinger 3.6 min armadillo (genus fixed), Nicolet 4.7-15.4 (genus 0 always), DMesh 12-20 (soup).
History: v4 (2026-09-13) = same algorithm on the Python backend (120 min/shape), kept as the equivalence reference.
Git tag: `golden-v5`. Details: LESSONS §23-29.

---
# GOLDEN v3 (2026-09-08) — armadillo, same 64-view setup, Palfinger optimizer params on the DLFL loop
**Mesh**: `cow_armadillo_golden_v3.npz` (Taubin) / `cow_armadillo_golden_v3_raw.npz` (no Taubin) — V=49,795 F=99,586, watertight, genus 0, **SI 0 %**
**Exam**: ho16 **0.9983** (raw 0.9978), **VolIoU 0.9948** (raw 0.9943), CD 0.00620 — beats Palfinger 2022 original code (0.9965 / 0.9938 / 0.00678, 37.9k V) on all three; DMesh 0.9894 / 0.9634; Nicolet 0.9593 / 0.8985.
**Chain**: golden v2 pre-Taubin mesh (`cow_armadillo_p6_50k.npz`) + 1200 in-loop steps with
`ADAM_BETAS=0.8,0.8 PALF_LAP=0.02 PALF_CLIP=10 LR_EDGE=0.3 ADAPT_REMESH=1 ADAPT_MODE=velocity ADAPT_NU_GAIN=0.2 ADAPT_LMIN_PX=1.3 ADAPT_MAX_F=100000 ADAPT_SI_GATE=0.3 COLLAPSE_EVERY=50 FLIP_EVERY=25 COLLAPSE_RATIO=0.4 SI_PUSH=0.15` → AUTO Taubin (x2). Details: LESSONS 22.
**Caveat**: 49.8k V vs Palfinger 37.9k (+31 %); wall 71 min vs 3.6 min. Genus shapes not yet re-run with v3 params.

---
# GOLDEN v2 / v1 (2026-09-02..03) — kept below for the chain history

# GOLDEN — armadillo, 64 training views (DMesh mv-recon setup), pure GenesisTopmod

**Mesh**: `cow_armadillo_p5_64_taubin5.npz` (+ .obj) — V=18,253 F=36,502, watertight, genus 0
**Exam (16 held-out views)**: ho16 IoU **0.9957**, hair_px 8, maxblob ≤ 5
**Geometry quality**: self-intersecting faces 1.4%, fold edges 0.2%, back dihedral median 7.9° (GT decimated to same face count: 10.1°)
**Reference**: DMesh @64v (official code, same exam) 0.9891, 2,400 v / 4,799 f, non-manifold soup

No DMesh anywhere in this chain. Supervision identical to DMesh's: 64 star-camera renders (silhouette+depth+diffuse) of the GT mesh; hull evidence built only from the 64 training silhouettes.

## Reproduction chain (all commits in this repo)
1. `run_64v.py`                       icosphere → C2F growth + v22 despike            → 0.9279  (`cow_armadillo_64v.npz`)
2. `phase1c_pipeline.py` ×2           62 p-gated DLFL extrude carves                  → 0.9572  (`p1d64c` + `_program.json`)
3. `phase1f_hullpull.py` HULL_MODE=vote  voting visual hull field pull (dead zone, anneal) → 0.9841 (`p1f64c`)
4. `phase3d_flip.py` / `phase4_inloop.py`  in-loop DLFL flip + collapse + SI push       → 0.9865  (`p4c_64`)
5. `phase4_inloop.py` SUBDIV_ALL=1 LAP_MULT=3  global DLFL subdivision + same loop      → 0.9914  (`p5_64`)
6. `phase5_taubin.py` ITERS=5         Taubin λ|μ fairing (positions only)             → **0.9957** (`p5_64_taubin5`)

Topology ops are TopMod DLFL only (extrude_face, subdivide_edge, stellate, collapse_edge_tri, delete_edge+insert_edge flips); `check_watertight` asserted after every topology change. Taubin is a standard position-only smoother (Open3D), not part of the operator program.

## Resolution controls (2026-09-02)
| setting | V / F | ho16 | note |
|---|---|---|---|
| ours 64v golden | 18,253 / 36,502 | **0.9957** | watertight |
| ours 6v pure, same recipe (subdiv + 6-view hull + in-loop + Taubin) | 20,404 / 40,804 | 0.9555 | train 0.9970 -> overfits 6 views; information-limited, not resolution-limited |
| DMesh 64v default (1000/3000/10000 seeds) | 2,400 / 4,799 | 0.9891 | 75k internal points |
| DMesh 64v hi-res (3000/10000/25000 seeds) | 2,738 / 5,478 | 0.9894 | 190k internal points; output triangles barely grow (existence thresholding), score plateaus |

## Fair face-count comparison vs DMesh (2026-09-02)

DMesh prunes its own faces: epoch_3 starts at 45k (hires) / 188k (dense_b) faces and its
existence-probability optimization removes >85% within 500 steps, even with the real
regularizer set to 0 and reals frozen to 1 every step (`armadillo_64v_dense_{a,b}.yaml`).

| Method | faces | ho16 |
|---|---|---|
| DMesh 64v (default, 14k seeds) | 4.8k | 0.9891 |
| DMesh hires (38k seeds) | 5.5k | 0.9894 |
| DMesh dense_a (no real reg, frozen reals) | 8.1k | 0.9889 |
| DMesh dense_b (dense_a + 98k seeds + ud_thresh 1e-2) | 9.2k | 0.9885 |
| Ours p4c_64 (optimized directly at low res) | 6.3k | 0.9865 |
| Ours golden decimated (quadric) to 5.5k | 5.5k | 0.9921 |
| Ours golden decimated (quadric) to 9.2k | 9.2k | 0.9946 |
| Ours golden | 36.5k | 0.9957 |

Honest reading: at equal LOW resolution optimized directly, DMesh is slightly ahead
(0.9894 vs 0.9865). Our advantage comes from coarse-to-fine (DLFL global subdivision +
in-loop untangle + Taubin) which DMesh cannot do because its representation prunes
itself back to ~5-9k faces regardless of seed count. Figure: dmesh_dense_compare.png.

## Phase 6: resolution ladder (2026-09-02)
| step | V / F | ho16 | SI | back dihedral |
|---|---|---|---|---|
| golden (36.5k) | 18,253 / 36,502 | 0.9957 | 1.4 % | 7.9° |
| + partial DLFL subdiv of 1500 largest faces + in-loop 1200 + Taubin×5 (`p6_50k_taubin5`) | 28,396 / 56,788 | **0.9972** | 0.5 % | 7.3° |
| + global subdiv ×6 (219k f) — ABORTED | 109,508 / 219,012 | (0.9958 before optimizing) | 32 % after 100 steps | — |

Ceiling at 56.8k faces (GT decimated) ≈ 0.999. The 219k run tangled because the mean edge (0.015)
fell below the 256² supervision pixel size (~0.01): per-pixel losses carry no information at that
scale. Effective resolution limit for this recipe ≈ 50–60k faces at 256²; go to 512² images first
to push further. Command: `SUBDIV_TOP=1500 LAP_MULT=3 STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100
COLLAPSE_RATIO=0.4 COLLAPSE_MAX=800 SI_PUSH=0.15 BASE_NPZ=<golden> phase4_inloop.py` then Taubin×5.
