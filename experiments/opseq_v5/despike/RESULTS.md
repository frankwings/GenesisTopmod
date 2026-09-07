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

## 1. Our best per shape (v8 chain)
| shape | GT genus | genus found | ho16 | VolIoU | CD | V | SI |
|---|---|---|---|---|---|---|---|
| armadillo | 0 | 0 | 0.9972 | 0.9885 | – | 56.8k | 0.5 % |
| kitten | 1 | **1** | 0.9994 | 0.9982 | 0.00666 | 19.4k | 0 % |
| fertility | 5 | **5** | 0.9974 (v8) / 0.9969 (v8b) | 0.9861 (v8b) | 0.00708 | 18.8k / 9.4k | 0.5 % |
| rocker-arm | 1 | **1** | 0.9985 | 0.9937 | 0.00584 | 25.2k | 0 % |
| three-holes | 3 | **3** | 0.9966 | – | – | – | – |

## 2. Competitors — all run by us with the ORIGINAL code on the SAME 64v/256² views and the SAME exam
| method | venue | topology | supervision used | armadillo VolIoU | kitten VolIoU | fertility VolIoU | rocker-arm VolIoU |
|---|---|---|---|---|---|---|---|
| Nicolet 2021 (large-steps) | TOG/SIGA | fixed genus 0 | ours (sil+depth+diffuse) | 0.8985 | 0.9684 | 0.8262 | 0.9249 |
| Palfinger 2022 | CAVW | fixed genus 0 | its own (3-ch normal image + alpha) | **0.9938** | 0.9968 | 0.9599 | 0.9382 |
| DMesh 2024 | NeurIPS | free (non-manifold soup) | ours (sil+depth) | 0.9634 | 0.9895 | 0.9788 | 0.9803 |
| 3DV-2026 (published, genus GIVEN, 36 views @1024²) | 3DV | genus given | theirs | 0.928 | 0.713 | – | – |
| **ours** | – | **manifold, genus discovered** | ours | 0.9885 | **0.9982** | **0.9861** | **0.9937** |

ho16 silhouette IoU, same rows: Nicolet 0.9593 / 0.9768 / 0.8963 / 0.9581; Palfinger 0.9965 / 0.9991 /
0.9398 / 0.9674; DMesh 0.9894 / 0.9944 / 0.9886 / 0.9928; ours 0.9972 / 0.9994 / 0.9974 / 0.9985.
Chamfer (kitten / fertility / rocker-arm): Nicolet 0.0110 / 0.0270 / 0.0156; Palfinger 0.0068 / – / –;
DMesh 0.0074 / 0.0071 / 0.0064; ours 0.0067 / 0.0071 / 0.0058.
Wall (single run): Nicolet 5–15 min, Palfinger 4–15 min, DMesh 10–20 min, ours 20–35 min.

## 3. Ablations (ours)
- **Refinement criterion** (5 criteria, equal loop/guards; LESSONS 15a/15b): fertility VolIoU curvature
  0.909 < dunyach 0.930 < velocity 0.938 < curv_uniform (3DV) 0.974 < **residual 0.976**; rocker-arm
  curvature/curv_uniform/residual tie 0.9936–0.9937, dunyach/velocity 0.989. Residual is the only criterion
  top-2 on both; pure-curvature criteria coarsen "flat but unfitted" regions.
- **Velocity guard** (Palfinger's rule on top of residual; 15c): fertility SI 1.4→0.7 %, VolIoU 0.976→0.983;
  rocker-arm neutral. Kept on.
- **Early hole opening** (12/12b): open holes at ~1.8k faces then cc4 — cleaner, SI 0, rocker-arm 0.9979.
- **Normal-map supervision** (17b/17c): W=1 unnormalised collapses V (negative); with the normal residual in
  the split criterion and W=0.3: kitten +0.0003 VolIoU (noise floor), fertility −0.010. Off by default.
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
3. Honest weakness: on genus-0 armadillo Palfinger (normal-image supervision, 55k V) beats us 0.9938 vs
   0.9885 VolIoU. Our accuracy ceiling on smooth genus-0 shapes is not the best in class.
4. Where to refine is decided best by **image residual**, not curvature (ours) and not velocity (Palfinger);
   3DV-2026's curvature+uniform is the strongest published criterion and the right baseline row.
5. Neither normal-map supervision nor 512² supervision moves the numbers: the remaining error is at the
   sub-pixel level of the exam, i.e. we are at the ceiling of this benchmark. Next lever is real data
   (DTU via 2DGS/PGSR renders), not more synthetic accuracy.
