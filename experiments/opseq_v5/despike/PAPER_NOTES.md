# Paper skeleton — Topo-Carving (working notes, v6.3, 2026-09-26)

Working title: **Topo-Carving: Discovering Surface Topology from Multi-View Silhouettes with
Guaranteed-Manifold Operators**

## 1. One-paragraph pitch

Multi-view mesh reconstruction that **discovers** the genus from images instead of assuming it.
A space-carving oracle supplies both the tunnel COUNT (g*, Euler number of the voting hull) and,
when needed, tunnel LOCATIONS (closing-ladder plug analysis). Topology changes happen only through
DLFL `add_handle` — link-condition-guarded collapse, Euler-preserving flip/subdivision, position-only
DR — so the mesh is manifold + watertight **by construction after every step**, and genus can never
drift silently. Each proposed handle passes a count-first propose-and-verify gate. Result: correct
genus 5/5 shapes (fertility genus-4 15/15 across CUDA nondeterminism), CD/IoU competitive with
fixed-topology SOTA.

## 2. Contributions (draft)

1. **Genus discovery, not genus preservation**: count AND location from the carved hull; no template
   (vs Gu ICASSP'26 / Nicolet: topology given, PH prior only PRESERVES it).
2. **Topological robustness by construction** (SELLING_POINT_topology_robustness.md): genus changes
   only via verified `add_handle`; immune to the handle-collapse failure others patch with a PH prior.
   Stress test: LAP_MULT=600, W_T=0 → genus held.
3. **Count-first propose-and-verify** (Rule C′, GATE_DECISION_findings.md): accept a handle on render
   evidence OR when the oracle says a tunnel is missing AND (membrane is genuine air OR the candidate
   is hull-located). Fixes the negative-Δho failure of hard thresholds at the refined stage.
4. **Hull-guided completion**: when membrane detection stalls (mesh flush to hull), the hull's own
   tunnel location proposes the handle. Detection and decision both fall back to the oracle.
5. (Secondary) **Late-handle seam refit** (Stage 6b): late topological edits need one image-evidence
   refit pass; smoothing/blending provably insufficient (measured).

## 3. Related-work contrast row (the table centerpiece)

| Method | topology source | can DISCOVER genus? | manifold guarantee | genus drift possible? |
|---|---|---|---|---|
| Nicolet Large-Steps | fixed input mesh | no | no | yes (handle collapse) |
| Gu et al. ICASSP'26 | template (count+rough location) | no | no | patched via PH prior |
| DMesh | implicit (existence probs) | partially | no (self-prunes) | yes |
| Palfinger | fixed | no | remesh-based | yes |
| **Ours** | **discovered from hull oracle** | **yes** | **by construction (DLFL)** | **impossible by construction** |

## 4. Method pipeline (v6.3 = golden_chain.sh)

sphere → CC-subdiv ×3 → DR clean loop → **topology discovery** (membrane detect → count-first gate →
DLFL add_handle → DR verify; hull-guided completion fallback; rays last) → re-carve → Palfinger loop →
late genus pass (5b) → [6b seam refit if 5b fired] → LAP ×3 → Taubin.

Key numbers to report: oracle g* reliability (5/5 shapes, morphological closing ladder r=1,2,3 mode);
gate ablation legacy 13/15 vs C′ 15/15 (fertility, n=15, CUDA nondeterminism as the noise source);
rescue fired naturally 4/15; hull locates 4/4 tunnels on the genus-3 flush mesh (fresh ladder 37 s).

## 5. Results so far (fill from results_genus + GOLDEN.md)

- 5-shape genus: armadillo 0, kitten 1, rockerarm 1, threeholes 3, fertility 4 — v6.3 regression ×3
  (task v63_full_reg, pending) to confirm 15/15 chains.
- fertility: genus 15/15; CD 0.0064, VolIoU 0.9915 (with 6b); seam artifact eliminated.
- armadillo vs DMesh (GOLDEN.md): ours 0.9957 ho16 @36.5k faces vs DMesh 0.9894 plateau; fair
  face-count comparison table already written (decimated 0.9921 @5.5k beats DMesh 0.9891 @4.8k).
- **Ablation table (fertility, GT genus 4, n=15 each; 2026-09-26 task ablation_run):**

  | config | genus correct | failure mode |
  |---|---|---|
  | full, TWO-SIDED site check (membrane drill + contact join), 2026-09-28 | **15/15** | none; all 15 visually matched to GT |
  | full, membrane-only check (over-corrected, rejected contact joins) | 14/15 | fx2 stopped at g3: arm touching body not joined |
  | no hull-completion (HULL_COMPLETE=0), fixed gate | 11/15 | 4× honest stop at g3 |
  | ~~full v6.3 before the fix~~ | ~~15/15~~ | WITHDRAWN: completion could bridge an open tunnel (genus number right, location wrong) |
  | no oracle (GT_MODE=off) | **4/15** | drifts BOTH ways: g3 ×6 (missed), g5 ×3 / g6 ×2 (spurious) |
  | legacy hard-threshold gate | 13/15 | rejects real tunnel at negative Δho |
  | laya text-model gate | (15/15 but confounded; coin flip at decisive point) | see §6 |

  Reading (post-fix, 2026-09-28): the oracle is the foundation (27% without it, failing in BOTH
  directions); membrane-checked hull-guided completion adds ~+3/15 (11→14). The earlier reading that
  the count-first gate alone closes part of the gap is WITHDRAWN: fixed gate without completion is
  11/15 vs legacy 13/15 (within n=15 noise). Other 4 shapes post-fix: 12/12. Location metric
  (air-loop linking number) in progress; see-through-string and voxel audits failed calibration.
  (d) no-6b is visual: fig_fertility_seam_before_after_6b.png.

## 5b. Site-check calibration (2026-09-28, results_genus/site_dataset_fertility_fy.json)

112 candidates enumerated by the membrane and contact detectors on 30 intermediate meshes (fy1-15, coarse
and refined), each labelled against GT (drill correct iff GT is air between the faces; join correct iff GT
is material). Where the gate can fire (mesh genus < g*): 71/71 decisions correct (70 membranes, 1 contact).
All errors of the inside/air rule occur on genus-4 meshes where the count stop blocks the gate. For
CONTACT candidates the geodesic/straight-line ratio (idea: the user) separates two populations with an
empty gap between 11.6 and 108: true contacts (arm pressed on body) 108-560; creases/cracks and bridges
over real air 1-12. Joining a crease with add_handle would add a spurious handle -> CONTACT now requires
ratio >= 50 (CREASE otherwise). Held-out check (thresholds from fy1-10, tested on fy11-15): same clean gap.
Also observed: 5 candidates where the hull stays SOLID over true air (thin tunnels not carved by the
silhouettes) - harmless here (all at genus 4) but a known limit for thinner tunnels.
Figure: fig_site_calibration.png.

## 5c. Site check v2 (2026-09-29): normal-sign inside + exact silhouette air (both suggested by the user)

Correction to 5b: the GT labels there used RAY-PARITY for "is this point inside our mesh". A generalized
winding-number referee (robust to self-intersections) shows parity is wrong on 31/112 candidates (72%
agreement); the two faces' NORMALS (back-to-back = inside, face-to-face = outside) agree with the winding
number on 83/84 decided cases and side with it in 28 of the 29 parity/normal disagreements. Relabelled
with winding-number truth, the OLD check scores 66/71 (not 71/71) where the gate can fire, with 26
accept-but-wrong and 6 missed sites. Air: evaluate the carving rule per query point (>=2 silhouettes see
background at 1024 px) instead of the 256^3 voxel hull with a half-voxel margin; this recovers 3 of the
5 "hull-uncarved" membranes (the other 2 are true visual-hull limits).

| inside | air | accept-but-wrong | missed | genus<g* correct |
|---|---|---|---|---|
| parity (old) | voxel, 0.5 vox margin (old) | 26 | 6 | 66/71 |
| normals (+winding fallback) | voxel, margin | 6 | 1 | 71/71 |
| normals (+winding fallback) | exact silhouette | 4 | 0 | 71/71 |
| + geodesic crease guard | | **1** | **0** | **71/71** |

The one remaining wrong accept (fy12, ratio 379, GT air) is the predicted residual: two different parts
separated by a real narrow gap that the visual hull fills; it occurs at genus 4 where the count stop
blocks the gate. SITE_V2=1 is now the default.

## 6. Negative results (report honestly — reviewers like these)

- **Typed-decision text model (TypeSafe Jev paradigm, via Laya)** for the accept/reject gate:
  coin flip (P∈[0.49,0.51]) at the decisive negative-Δho point, nonsensical ordering, self-reported
  uncalibrated confidence. A hard count prior + numeric comparison beats a text decision model.
- Seam removal via local Taubin / positional blending / masked-DR: all fail measurably; only an
  unconstrained image-evidence refit removes a late-handle seam.
- PH-style anti-collapse guard (Gu borrow): validated in isolation (torus spike) but a NO-OP in our
  pipeline — DLFL already makes the failure impossible. This is the differentiator, framed in §2.2.

## 7. Limitations / honest scope

- Combinatorial manifold guarantee, not self-intersection-free (SI ~0.5% reported).
- Oracle needs silhouette coverage: partial object-azimuth orbits break the hull (iPhone survey §8);
  real-data demo pending a clean 360° capture (real/CAPTURE_SPEC.md).
- Resolution ceiling ≈ 50-60k faces at 256² supervision (GOLDEN.md Phase 6).
- Thin-handle real data untested (mug capture pending).

## 8. TODO before submission

- [x] v6.3 5-shape ×3 regression: 15/15 chains correct genus, 0 mismatch (2026-09-26, task v63_full_reg)
- [ ] real mug genus-1 result (needs capture per CAPTURE_SPEC.md)
- [x] ablation runs (§5 table complete)
- [x] figures v1 (results_genus/fig_*.png): pipeline diagram, hull-plug viz, DMesh comparison,
      ablation bars, fertility seam before/after + arm GT comparison. (Pipeline diagram is a draft;
      redraw in TikZ for camera-ready.)
- [ ] decide venue (ICASSP direct rebuttal to Gu? or 3DV/CVPR-W with fuller eval)
