# Lessons 2026-09-26 → 09-29: tunnel-site validity, add_handle semantics, and how we measure "correct"

Written after a long debugging session with the Boss. Several of the fixes below were his ideas; they are
marked (Boss). Numbers are from fertility (GT genus 4) unless stated. Code: `phase7_handle.py`
(`membrane_check`, `site_kind`, `site_kind_full`, `winding_numbers`, `hull_air_exact`, `normal_side`,
`contact_geo_ratio`), `phase7_multi.sh`, `jev_handle_gate.py`, `golden_chain.sh`.

## 1. What `add_handle(f1, f2)` does (and why it can be wrong)

Deletes two faces and stitches their boundary loops with n side quads (a tube). On the same connected
surface this ALWAYS raises the genus by exactly 1. Geometrically it means two different things:

- the two faces are the two pages of a thin membrane (our material in between) → it **drills a hole**;
- the two faces face each other across air → it **builds a bridge** (a solid tube through air).

Both are valid DLFL operations and both give genus +1, so **the genus number cannot tell a correct
handle from a wrong one**. Only the location can. Explained with a real TopMod run on a membrane donut:
`results_genus/fig_add_handle_donut.png`, `demo_add_handle_donut.py`.

## 2. The bug (found by the Boss from a visualization that "looked wrong")

hull-guided completion took the first hull plug (`_hc[0]`) and the gate accepted any hull-located
candidate unconditionally; also `phase7_multi` labelled ordinary membrane candidates `prov=hull` by
default. A forced-stall test opened a handle **through the base plate** while the genus read 4 and the
real missing upper tunnel stayed closed. Our earlier "E2E verified" had only checked the genus number.

**Lesson: never accept a topology result on the genus count alone; check where the handles are.**

## 3. Fixes, in the order they were found

| step | change | why |
|---|---|---|
| 1 | membrane check before any count rescue | stop unconditional accepts (the bridge) |
| 2 | two-sided site kinds: MEMBRANE (drill) / CONTACT (join) / INVALID | step 1 over-corrected: it rejected *contacts* (arm pressed on body, LESSONS 25b), which is exactly how fertility's upper tunnels close. fx2/cp6 verified against GT close-ups |
| 3 | remember rejected positions (`rej_mid`), all detectors skip within R_REJ | after a rejection the mesh reverts, so detectors re-proposed the same spot under a new image/cluster key (Boss: "don't search near the rejected one again") |
| 4 | geodesic/straight ratio for CONTACT (Boss) | a true contact joins two DIFFERENT parts (far along the surface); a crease/crack is one sheet folded (near along the surface) and joining it adds a spurious handle |
| 5 | exact per-point silhouette air (Boss) | replace "256³ voxel hull + half-voxel margin" by the carving rule evaluated at each query point (≥2 silhouettes see background, 1024 px) = an infinitely refined octree |
| 6 | inside/outside: normals (Boss) → generalized winding number | ray parity was wrong on ~28% of candidates |
| 7 | winding ≈ 2 ⇒ CONTACT; winding < 0 ⇒ INVALID | interpenetrating parts; locally inverted mesh |

## 4. The mathematics of the site check

Sets: Ω = true solid (unknown), M = our mesh solid, H = carved visual hull. Visual hull property: Ω ⊆ H,
hence **x ∉ H ⇒ x ∉ Ω** (hull air is proof of air) but x ∈ H is ambiguous (H∖Ω = concavities and
uncarved thin tunnels). A drill is right iff the stretch between the faces is in M and outside Ω; a join
is right iff it is outside M (or in two overlapping parts of M) and inside Ω, between two different parts.
We substitute H for the unknown Ω — every error comes from that substitution:
- MEMBRANE false positive needs air where Ω is solid → impossible up to discretisation (73/73 correct).
- MEMBRANE false negative when the stretch lies in H∖Ω: a thin tunnel of width w, length L is visible
  only from a cone of solid angle ∝ (w/L)²; with 64 cameras and a 2-vote rule it may never be carved.
- CONTACT false positive when two different parts are separated by a real narrow gap the hull fills —
  geodesic cannot see it (observed once: fy12, ratio 379, at genus 4 where the gate cannot fire).
- Boundary errors from impure segments and 1/7 quantisation (0.43, 0.57 fall in the 0.4–0.6 dead zone).

## 5. Inside/outside: three methods compared

- **Ray parity** (open3d `compute_occupancy`): counts crossings of one ray. Breaks near self-intersections
  and between sheets 1–2 voxels apart. Wrong on 31/112 GT-labelled candidates.
- **Normal sign (Boss)**: back-to-back faces ⇒ segment inside (membrane pages); face-to-face ⇒ outside
  (contact gap). Agrees with the winding number on 83/84 decided cases, undecided on oblique pairs.
  Blind spot: it cannot COUNT layers, so interpenetration looks like a membrane.
- **Generalized winding number** (Jacobson 2013): Σ solid angles / 4π; 0 outside, 1 inside, **2 where
  two parts interpenetrate** (18/112 candidates: arms pushed into the body), **negative where the mesh is
  locally inverted**. Used as the referee and now as the primary measure. The old parity code "worked" on
  interpenetration by accident: parity counts two layers as outside → CONTACT → join.

## 6. How we measured, and the measurement mistakes we made

- Voxel audit of final meshes (128³): thin arms merge into fake loops — failed calibration.
- "See-through strings" through each tunnel: tunnels sealed by a pinch still look open — failed.
- Handle-midpoint coordinates: membranes land at different spots run to run — not discriminative.
- **GT-labelled candidate dataset** (112 candidates, fy1–15, coarse + refined meshes): the labels first
  used ray parity for "inside our mesh" — i.e. they contained the very error being measured. Re-labelled
  with the winding number, the old check was 66/71 (not 71/71) where the gate can fire.
- I once implemented the rule with a parity fallback while the offline evaluation used a winding
  fallback: **the evaluated configuration must be the implemented one.**
- **Per-handle GT check** (GT occupancy around each accepted handle's midpoint): catches the known bridge
  and keeps the known real tunnel (drills reliable); for JOINS it falsely flagged the GT-verified fx2
  contact (the midpoint sits in a thin gap at the GT surface) — join criterion still to be designed.
- Calibration data covered only two mesh stages; stage-3 mid-round meshes (more self-intersections)
  produced a true contact at geodesic ratio 20 (rejected by the 50 threshold) and a true membrane at
  winding −1 (rejected by the negative-winding rule). Both were rescued by render evidence. **The
  calibration set must include every stage the gate runs on.**

## 7. Results by version (fertility ×15 unless noted)

| version | genus correct | location evidence | note |
|---|---|---|---|
| legacy threshold gate (v6.1) | 13/15 | none | |
| v6.2/v6.3 (unconditional hull accept) | 15/15 | **withdrawn** | could bridge |
| membrane-only check | 14/15 | every handle a membrane | fx2: contact rejected |
| two-sided + rej memory | 15/15 | all 15 visually vs GT | fig_audit_fy_* |
| + geodesic (old inside/air) | 15/15; others 12/12 | — | baseline for the new site check |
| site v2 (normals, parity fallback) | 11/12 partial | 12 real membranes labelled INVALID | implementation ≠ evaluation; stopped |
| site v3 (winding primary) | in progress: 4/4, others 4/4 | per-handle GT 16/16 (drills reliable) | two rule false-rejections rescued by render |

Other findings: the no-oracle ablation (4/15, errors in both directions) and the gate-alone claim being
withdrawn are in PAPER_NOTES §5.

## 7b. 2026-09-30 addendum: negative winding = a crushed membrane (self-collision)

v66 regression (winding-primary): fertility 15/15, other shapes 12/12; per-handle GT audit 57/57 drills on
real air. But 14 accepted handles had been labelled INVALID by the rule "w < 0 -> inverted -> no action" and
were only rescued by render evidence; all were real tunnels. Cause: DR keeps squeezing a membrane (the
silhouette loss wants that material gone and cannot see interior crossings), the two pages pass THROUGH
each other, and the pocket between them is bounded by front faces: w = -1 ("negative thickness"). It is
one closed surface colliding with itself, not two meshes. Interpenetrating parts give w = 2 instead.
Fix: classify on |w| (|w|>=0.5 material, |w|>=1.5 overlap). Offline: 112-candidate set 0 wrong accepts,
71/71 where the gate fires; the 14 live cases become MEMBRANE. Remaining known misses: a zero-thickness
coarse membrane (w ~ 0, pages coincide; 1 case, strong render evidence) and a contact at geodesic ratio 20.
DR cannot avoid self-collision by itself; mitigation = open membranes earlier (several handles per
round), not a collision barrier. Every candidate's features + face centroids are now written to
results_genus/sitelog_<tag>.jsonl for calibration on mid-stage meshes.

## 8. Open problems

1. Recalibrate on stage-3 mid-round meshes: geodesic threshold (a true contact at 20) and the handling
   of negative winding (a true membrane at −1). Save those meshes and every candidate's features + face
   centroids.
2. Per-handle GT criterion for joins (e.g. GT connectivity between the regions of the two sheets).
3. Thin tunnels the visual hull cannot carve (the oracle itself would undercount) — second source of air
   evidence from image-space see-through, or a finer / adaptive hull (Boss: octree near the surface).
4. Seam cracks on the arms (folds = the "crease" population): zip them topology-preservingly.
5. hull-completion face-pair search is slow (~15 min per query); batch the rays.
