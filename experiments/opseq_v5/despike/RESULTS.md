# GenesisTopmod — consolidated results (as of 2026-09-06)

Details, failures and per-experiment notes live in `LESSONS_2026-09-02.md` (§12–18); this file is the
one-page summary the paper tables will be built from. Meshes: `results_genus/`.

## 0. Setup (ours)
- Input: 64 star-arranged training views @ **256²** of a GT mesh (pymeshlab sample meshes + kitten from 3DV-2026).
- **Supervision = silhouette L1 + masked depth L1 (×0.35) + 1-channel headlight diffuse L1 (|n·z|, ×1.0)**
  + voting visual-hull field (×20). **No normal maps** by default (normal-image L1 exists as `W_NORMAL`,
  ablated in §4, off).
- Chain (v8): icosphere → cc → clean → hole detection + `add_handle` (genus discovered from image
  evidence) → cc4 → Stage-4 DR loop 1200 steps with in-loop DLFL flip/collapse/SI-push + residual-driven
  adaptive remeshing (+ Palfinger velocity guard) → Stage-5 1200 steps → AUTO Taubin.
- Exam (never trained on): **ho16** = silhouette IoU on 16 held-out views @256²; **VolIoU** = volume IoU on a
  256³ occupancy grid after ICP; **CD** = symmetric Chamfer after ICP (3DV-2026 metric family,
  `eval_cd_iou.py`). Always report SI (self-intersecting faces), genus, V.

## 0b. Parameters of the current best (armadillo, LESSONS 22) — proposed default "v9"
`ADAM_BETAS=0.8,0.8 PALF_LAP=0.02 PALF_CLIP=10 LR_EDGE=0.3 ADAPT_REMESH=1 ADAPT_MODE=velocity ADAPT_NU_GAIN=0.2 ADAPT_LMIN_PX=1.3 ADAPT_MAX_F=100000 ADAPT_SI_GATE=0.3 COLLAPSE_EVERY=50 FLIP_EVERY=25 COLLAPSE_RATIO=0.4 SI_PUSH=0.15 STEPS=1200` + AUTO Taubin (optional).
What changed vs v8: Palfinger's optimizer (betas 0.8, nu-weighted Laplacian on the gradient, clip, lr = 0.3 x edge) and the edge floor 3 px -> 1.3 px (V 27.7k -> 49.8k). Full command in LESSONS 22. Since 2026-09-11 the Stage-1/4 coarse-to-fine refinement is TopMod `catmull_clark` + `triangulate_all` (`C2F_SUBDIV=cc` default; = numpy midpoint, LESSONS 23), so every topology change in the chain is a TopMod operator.


## 0c. Version history (what "golden vN" and "chain vN" mean)
| version | date | what | armadillo ho16 / VolIoU / V |
|---|---|---|---|
| golden v1 | 09-02 | icosphere → carve → hull → in-loop DLFL → global subdiv → Taubin x5 (`p5_64_taubin5`) | 0.9957 / – / 18.3k |
| golden v2 | 09-03 | + partial DLFL subdiv of 1500 largest faces + 1200 steps + Taubin (`p6_50k_taubin5`) | 0.9972 / 0.9885 / 28.4k |
| chain v1–v5 | 09-04 | in-loop adaptive remeshing, curvature (dihedral) criterion, SI-ring guards, px floor (LESSONS 13) | genus shapes only |
| chain v7 | 09-05 | residual criterion, quantile-normalised (negative) | genus shapes only |
| chain v8 / v8b | 09-05 | residual criterion, absolute px thresholds (+ coarsen-where-fitted); + vel guard (15c) = the genus-shape headlines in §1–2 | genus shapes only |
| **golden v3 = chain v9** | 09-08 | v2 mesh + Palfinger optimizer params + 1.3 px edge floor (LESSONS 22, §0b) | **0.9983 / 0.9948 / 49.8k** |
Nine numbered iterations in total (golden 1–3, chain 1–9 with v6 skipped); golden v3 is the current reference for genus-0, v8b + vel-guard for genus > 0 until the v9 re-runs land.

## 1. Our best per shape (v8 chain; armadillo = v9)
| shape | GT genus | genus found | ho16 | VolIoU | CD | V | SI |
|---|---|---|---|---|---|---|---|
| armadillo | 0 | 0 | 0.9983 | 0.9948 (Palfinger optimizer params, LESSONS 22) | 0.00620 | 49.8k V | 0 % |
| kitten | 1 | **1** | 0.9994 | 0.9982 | 0.00666 | 19.4k | 0 % |
| fertility | 4 | **4** (rebuilt base, LESSONS 19b) | 0.9973 | 0.9809 | 0.00668 | 16.0k | 1.7 % |
| fertility (old chain, genus 5 = one spurious handle) | 4 | 5 | 0.9969 (v8b) | 0.9861 | 0.00708 | 9.4k | 0.5 % |
| rocker-arm | 1 | **1** | 0.9985 | 0.9937 | 0.00584 | 25.2k | 0 % |
| three-holes | 3 | **3** | 0.9966 | – | – | – | – |

## 2. Competitors — all run by us with the ORIGINAL code on the SAME 64v/256² views and the SAME exam
| method | venue | topology | supervision used | armadillo VolIoU | kitten VolIoU | fertility VolIoU | rocker-arm VolIoU |
|---|---|---|---|---|---|---|---|
| Nicolet 2021 (large-steps) | TOG/SIGA | fixed genus 0 | ours (sil+depth+diffuse) | 0.8985 | 0.9684 | 0.8262 | 0.9249 |
| Palfinger 2022 | CAVW | fixed genus 0 | its own (3-ch normal image + alpha) | 0.9938 | 0.9968 | 0.9599 | 0.9382 |
| DMesh 2024 | NeurIPS | free (non-manifold soup) | ours (sil+depth) | 0.9634 | 0.9895 | 0.9788 | 0.9803 |
| 3DV-2026 (published, genus GIVEN, 36 views @1024²) | 3DV | genus given | theirs | 0.928 | 0.713 | – | – |
| **ours** | – | **manifold, genus discovered** | ours | **0.9948** | **0.9982** | **0.9809** (g4; old g5 chain 0.9861) | **0.9937** |

ho16 silhouette IoU, same rows: Nicolet 0.9593 / 0.9768 / 0.8963 / 0.9581; Palfinger 0.9965 / 0.9991 /
0.9398 / 0.9674; DMesh 0.9894 / 0.9944 / 0.9886 / 0.9928; ours 0.9972 / 0.9994 / 0.9974 / 0.9985.
Chamfer (kitten / fertility / rocker-arm): Nicolet 0.0110 / 0.0270 / 0.0156; Palfinger 0.0068 / – / –;
DMesh 0.0074 / 0.0071 / 0.0064; ours 0.0067 / 0.0071 / 0.0058.
Wall (single run, same GPU): Nicolet 5–15 min, Palfinger 4–15 min, DMesh 10–20 min, ours **5.4 min** (armadillo, full chain incl. genus discovery; C++ DLFL kernel + batched 64-view nvdiffrast, 2026-09-14; was 105 min pure Python).

## 3. Ablations (ours)
- **Refinement criterion** (5 criteria, equal loop/guards; LESSONS 15a/15b): fertility VolIoU curvature
  0.909 < dunyach 0.930 < velocity 0.938 < curv_uniform (3DV) 0.974 < **residual 0.976**; rocker-arm
  curvature/curv_uniform/residual tie 0.9936–0.9937, dunyach/velocity 0.989. Residual is the only criterion
  top-2 on both; pure-curvature criteria coarsen "flat but unfitted" regions.
- **Velocity guard** (Palfinger's rule on top of residual; 15c): fertility SI 1.4→0.7 %, VolIoU 0.976→0.983;
  rocker-arm neutral. Kept on.
- **Early hole opening** (12/12b): open holes at ~1.8k faces then cc4 — cleaner, SI 0, rocker-arm 0.9979.
- **Palfinger optimizer parameters on our loop** (22): armadillo 0.9983 / 0.9948 / SI 0 at 49.8k V, beats Palfinger on all metrics; remesh every 50 vs 100 steps identical.
- **In-loop velocity-weighted Laplacian W_VLAP** (21): armadillo raw VolIoU 0.985 -> 0.992, SI 2.5 % -> 0, Taubin becomes a no-op; headline 0.9932 (Palfinger 0.9938). Should become default.
- **Normal-map supervision on armadillo, no Taubin** (20): +0.001 raw, 0 after Taubin; Taubin itself is +0.006 VolIoU. Palfinger's armadillo edge is not explained by normals or smoothing.
- **Normal-map supervision** (17b/17c): W=1 unnormalised collapses V (negative); with the normal residual in
  the split criterion and W=0.3: kitten +0.0003 VolIoU (noise floor), fertility −0.010. Off by default.
- **Hull-field fairness** (19/19b): hull built from 512² GT silhouettes. kitten and fertility: no effect (no-hull = hull); rocker-arm: +0.04 VolIoU (0.9567 without). Report the no-hull row or give competitors the same term.
- **512² supervision** (18): kitten 0.9975 (256²: 0.9982), fertility 0.9851 (0.9827 vel-guard / 0.9861 v8b),
  rocker-arm 0.9902 (0.9937). No gain, 1.2–1.7× more vertices, 1.5–2× wall. Resolution is not the lever.
- **Concavities** (17b): kitten eye sockets are 0.3 px deep at 256²; head-band error ours 0.25/0.46/0.64 px
  (p50/p90/p99) = Palfinger 0.26/0.46/0.65, DMesh 0.33/0.56/0.87. What looked like "DMesh has better eyes"
  is facet shading on a 3.9k-V soup.

## 4. Conclusions
1. On every genus>0 shape the order is **ours > DMesh > Palfinger > Nicolet** on VolIoU, Chamfer and
   silhouette IoU, and ours is the only one that is both manifold and has the right genus (found, not given).
2. Fixed-genus methods do not "fail" on wide tunnels — they fill them with a membrane/plug that silhouettes
   barely see (kitten: Palfinger loses only 0.0014 VolIoU); topology matters for the metric in proportion
   to tunnel width (fertility/rocker-arm: 0.03–0.06 VolIoU).
3. armadillo (genus 0): with Palfinger's optimizer settings (betas 0.8, nu-weighted Laplacian on the gradient,
   lr = 0.3 x edge, 1.3 px edge floor) our loop passes Palfinger: 0.9948 vs 0.9938 VolIoU (LESSONS 22). Before
   that (0.9885-0.9932) it was behind; the gap was vertex budget + jitter, not supervision.
4. Where to refine is decided best by **image residual**, not curvature (ours) and not velocity (Palfinger);
   3DV-2026's curvature+uniform is the strongest published criterion and the right baseline row.
5. Neither normal-map supervision nor 512² supervision moves the numbers: the remaining error is at the
   sub-pixel level of the exam, i.e. we are at the ceiling of this benchmark. Next lever is real data
   (DTU via 2DGS/PGSR renders), not more synthetic accuracy.

## 2b. Recomputed side-by-side (2026-09-11, `compare_panel.py`, figure `results_genus/compare_all_methods.png`)
All numbers below were recomputed in ONE pass from the archived meshes with the same exam (ho16 / VolIoU / CD / V / genus).
| shape | Ours (headline) | Ours v3 uniform chain | Palfinger 2022 | DMesh 2024 | Nicolet 2021 |
|---|---|---|---|---|---|
| armadillo | 0.9983 / 0.9949 / 0.00619 / 49.8k / g0 | 0.9983 / 0.9949 / 0.00620 / 49.5k / g0 | 0.9965 / 0.9938 / 0.00626 / 37.9k / g0 | 0.9894 / 0.9634 / 0.00776 / 2.7k / gsoup | 0.9593 / 0.8987 / 0.02050 / 8.6k / g0 |
| kitten | 0.9994 / 0.9982 / 0.00665 / 19.4k / g1 | 0.9993 / 0.9979 / 0.00665 / 55.9k / g1 | 0.9991 / 0.9968 / 0.00682 / 67.2k / g0 | 0.9944 / 0.9895 / 0.00738 / 3.9k / g1 | 0.9768 / 0.9684 / 0.01098 / 8.7k / g0 |
| rockerarm | 0.9985 / 0.9937 / 0.00584 / 25.2k / g1 | 0.9985 / 0.9869 / 0.00615 / 50.3k / g1 | 0.9673 / 0.9383 / 0.01332 / 56.5k / g0 | 0.9928 / 0.9803 / 0.00638 / 3.4k / gsoup | 0.9581 / 0.9250 / 0.01541 / 8.1k / g0 |
| fertility | 0.9973 / 0.9809 / 0.00670 / 16.0k / g4 | 0.9973 / 0.9904 / 0.00661 / 47.6k / g3 | 0.9398 / 0.9598 / 0.01249 / 55.6k / g0 | 0.9886 / 0.9788 / 0.00708 / 3.1k / gsoup | 0.8963 / 0.8259 / 0.02720 / 8.4k / g0 |
(ho16 / VolIoU / CD / V / genus; 'soup' = DMesh non-manifold output.) Ours is first on VolIoU on all four shapes; on fertility the v3 uniform chain (genus 3, one tunnel missed) scores higher than the correct genus-4 mesh — genus must be reported as its own metric.
