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

## 7c. 2026-10-01: "open every membrane at once" (BATCH_OPEN) is rejected - counterexample bt1

Idea (Boss): open all detected membranes in one go, no DR in between; expected to be faster and to avoid
DR crushing the membranes that wait (7b). Implemented as `BATCH_OPEN=1` (default OFF, experimental).

- bt1 (no guard): 4 site-valid MEMBRANE handles opened at once, final genus 4 = GT, per-handle GT audit
  4/4 "on real air" - and the result is WRONG: the base arch is still sealed by a membrane, one handle in
  the upper region is redundant (figures `fig_batch_open_counterexample.png`, `fig_batch_open_bt1_handles.png`).
  The redundant one is most likely H2, a 7-edge-long tube bored through the thick slab of wrong material
  that fills the whole upper concavity on the coarse mesh (inference from length/position, not proven).
- Why it cannot work in principle: without DR nothing distinguishes two membranes (or two places of one
  slab) of the SAME tunnel. In sequential mode DR widens the tube and eats the leftover membrane, so the
  next detection no longer proposes that tunnel; the count stop then protects the rest. In batch mode the
  spare handle steals the quota of a real tunnel and the genus count cannot tell.
- Guards that do NOT work: (a) per-handle GT audit - blind to redundancy (now proven, not only predicted);
  (b) hull plugs - the 5 fertility plugs overlap (plug0/1 share 11k voxels, plug2/3 6k; the three upper
  plug centres are 0.15 apart), one long tube touches all three upper plugs at 0.2 vox; (c) voxel-block
  capacity k - one block covers all 4 tunnels and k is unstable (4 -> 3 -> 6 -> 4, thin tubes vanish in
  the 128 voxelisation); (d) closing-ladder on the final mesh - voxel genus 9 (cracks count as holes).
- batch-lite (`BATCH_SEP_MAX=3`: only thin membranes are batched, thick ones go to the DR-verified
  rounds): bt2, bt3 correct layout by eye, n=2 only. bt3 opened nothing in stage 3 and all 4 in the late
  5b pass (rougher surface, slower). Known defects: a skipped THICK candidate is re-proposed 13-23 times;
  the sitelog audit cannot read mixed batch+sequential logs.
- Speed: there is little to gain. Timestamps: stage 3 (all handles) is 100-134 s of a ~400 s chain,
  ~30 s per round; the refinement stages (~270 s) are untouched. Ideal batch saves ~80 s (20%); batch-lite
  saved nothing (342 / 487 s vs 405 s). The real tail is a LATE handle: hull face-pair search, 15+ min
  (g67f2 1472 s, np4 1291 s).
- This also withdraws the mitigation proposed at the end of 7b ("several handles per round").

## 7d. 2026-10-01: near-pair detector ("close in space, far along the surface") - fast, not selective

Idea (Boss): two faces close in Euclidean distance but far geodesically are the two sides of a membrane.
- On our data this signature belongs to CONTACTS, not membranes: 138 real membranes have a
  geodesic/straight ratio with median 6.6 (115 below 10, only 2 above 50); contacts are 82-560. Coarse
  membranes are thick slabs (page distance 0.3-0.4, 10-15% of the object), crushed ones have overlapping
  1-rings (ratio ~0). And a real thin part (plate, ear) has the same signature - hull air is still needed.
- Where it helps: the refined stage. `find_near_pairs` (`NEAR_PAIRS=1`, default OFF): KD-tree face pairs
  within 6 vox, opposed normals, non-adjacent, clustered, each cluster classified by the site check,
  membranes before contacts. Runs in 1.4-2.8 s on 44k faces and reproduces the late contact joins that
  hull completion needs ~16 min for (g67f2; np4 only with `THIN_ALIGN=0.2`: interpenetrating sheets,
  w=2, are opposed but laterally offset, centre line vs normal alignment 0.24-0.28).
- Why it is not enabled: (1) 5 of 7 finished, correct genus-4 meshes also carry a site-valid CONTACT
  cluster that needs no join - only the count stop blocks it; a chain stuck at genus 3 for a missing
  MEMBRANE would get a wrong join (same class of error as bt1); (2) one CONTACT candidate (g67f5) lies on
  GT air (narrow real gap filled by the hull); (3) distance to the hull plugs does not separate needed
  joins (4.7, 5.0 vox) from unneeded ones (4.1-5.4), the closest (1.8) is the wrong one; (4) cluster
  choice is sensitive to the alignment threshold (bt3: 12 -> 5 membrane clusters).
- Regression with NEAR_PAIRS=1, alignment 0.7: fertility 6/6 correct genus (np1-6), the detector never
  fired; this also re-confirms the sequential mode after the 2026-09-30/10-01 edits.
- Needed before enabling: a criterion for "does this contact need a join" (= open problem 2).

## 7e. 2026-10-01: linking-number audit - genus right, shape right, surface topology WRONG in 15/21 runs

Question that started it: which near-pair CONTACT needs a join (7d)? Dump of all 19 site-valid contact clusters
on 21 late-stage fertility meshes: the two needed joins (g67f2, np4: w=2, overlap 1, geodesic ratio 78 / 81) are
numerically identical to 13 unneeded ones on finished genus-4 meshes (w=2, overlap 1, ratio 79-186), at the same
two places. No local feature can separate them, and the geodesic DISTANCE cannot either. What works is the
geodesic PATH: close it with the straight segment and ask whether a hull tunnel passes through that loop.

Tool (`linking_audit.py`):
- Air loops: union of the cached plug blocks = sealed air; a sealed block with m mouths to the outer air gives
  m-1 closed curves (mouth0 -> mouth j through the block, back through the outer air). Fertility: upper chamber
  4 mouths + base 2 mouths = 4 loops = g*. No hand-tuning.
- Gauss linking number of two closed polylines (exact solid-angle form per segment pair): clean integers.
- Candidate handle (fi, fj): linking vector of [segment + Dijkstra surface path]. Face-to-face gaps (w ~ 0,
  including the g67f5 candidate on GT air) get the zero vector -> correctly spurious. Every interpenetration
  (w = 2) links a tunnel, on genus-3 AND genus-4 meshes.
- Mesh audit: linking matrix of the 2g tree-cotree generator loops with the air loops; rank = number of hull
  tunnels the SURFACE realises. GT fertility 4/4. bt1 (7c) 2/4, base column zero: the counterexample that the
  per-handle GT audit passed is caught, without GT.

Result on v6.4 (`results_genus/linking_audit_fertility_v64.txt`): fertility 6/21 pass (g67f1, f2, f4, f13, np4,
np5); 15 have rank 3 or 2. 13 of the 15 contain a MICRO-HANDLE (generator loops of 3 vertices, length 0.02-0.04,
linking vector 0); none of the 6 passing meshes does. Most failing meshes also carry an unjoined interpenetration
whose loop links a tunnel; the two runs that did the late contact join (g67f2, np4) pass. So in ~70% of runs one
tunnel is only "open" because two parts interpenetrate (the union solid and the silhouettes are right) while the
4th surface handle is a hidden triangle-sized tube. Genus count 27/27 stands; "surface topology equals GT" does
not: fertility 6/21, other shapes not audited yet.

Not yet done: a picture of one micro-handle; which stage creates it (negative-winding handles do not explain it:
f4 has two and passes, f14 has none and fails); air-loop construction for other shapes (kitten: the sealed
block shows ONE interface component, so no loop is built - mouth labelling must be made robust).

Consequence for the gate (Boss, 2026-10-01: go for strict topology): accept a handle only if its linking
vector is linearly independent of the vectors already realised by the mesh. One rule covers redundancy (bt1),
"does this contact need a join", and micro-handles; it also makes the 2-second near-pair detector safe to use.

## 7f. 2026-10-01: strict repair on the refined mesh (STRICT=1) - 8/10 strict-correct end to end

- RANK_GATE in Stage 3 (accept a handle only if "tunnels realised" goes up by exactly 1) does not work: on the
  coarse mesh other tunnels are still blocked by membranes, the hull air loops pass THROUGH that material and
  the linking numbers are polluted (readings like 0 -> 2 for one handle). 6 chains: 2 reach 4/4, 4 stop honestly at
  genus 3, no fake handle anywhere, but 30-73 min per chain (everything is pushed into the slow late hull search).
  The measure is only valid once the mesh is flush with the hull.
- Where the fake handle comes from: 13 of 15 failing runs are already wrong at the end of Stage 3; 1 lost a tunnel
  in a pure DR stage (the surface passed through itself, the handle slipped off the tunnel); 1 got a redundant
  late handle.
- Every fake handle on the refined mesh is a MICRO-HANDLE = a non-face 3-cycle (three existing edges whose
  triangle is not a face: the neck of a triangle-sized tube), non-separating, linking vector 0.
  `strict_repair.py` cuts along it and caps both sides (genus -1, manifold + connected checked, rank unchanged):
  14/14 failing genus-4 meshes repaired to genus == rank; a correct mesh is left untouched.
- Late pass under STRICT: near-pair detector (2 s) -> linking-vector prefilter (the loop must enclose a tunnel
  not realised yet) -> add_handle + DR -> accept iff tunnels realised +1 (RANK_ONLY, replaces the render gate).
  No hull face-pair search (HULL_COMPLETE=0 in this pass). On the 15 failing v6.4 meshes: 15/15 end at
  genus 4, 4/4 tunnels, 52-178 s for the pass (was 11-55 min).
- End to end (st1-10, `results_genus/linking_audit_fertility_strict_st.txt`): strict-correct 8/10 (v6.4: 6/21),
  the other 2 stop at genus 3 with 3/3 (no fake handle, honest miss). Wall 334-497 s, CD 0.0063-0.0065,
  VolIoU 0.990-0.992 - same as v6.4. Later stages (refine, seam refit, Taubin) did not break the topology.
- Cause of the 2 misses (st1 reproduced): the interpenetration clusters are there but the ONE representative
  pair picked per cluster is a pair across a thin real part (inside 1, air 0, no overlap -> INVALID). With the
  relaxed alignment (0.2) clusters are large and mixed. Fix to try: choose the representative among the pairs
  whose midpoint has |w| >= 1.5.
- Limits: fertility only (air loops for other shapes need robust mouth labelling); the cut-and-cap runs on
  triangle arrays, not as a DLFL operator.

## 7g. 2026-10-02: late joins decided by the linking vector - fertility 10/10 strict-correct

The two misses of 7f (st1, st10) were candidate SELECTION, not the rule:
- st10: one representative pair per cluster was a pair across a thin real part. Now the pairs whose midpoint has
  |w| >= 1.5 (two sheets really interpenetrating) are tried first, up to NEAR_TRIES=5 distinct pairs per cluster.
- st1: the audit's null vector (-1, 1, 0, 0) says which tunnel is missing (the bar between upper mouths 1 and 2 is
  not closed). All six sampled pairs of the cluster at region A enclose exactly that tunnel and GT is solid there,
  but the midpoint winding is a mix of 0 / 1 / -1 (parts touching, locally inverted), which the coarse-stage site
  rule calls INVALID. New rule on the refined mesh (RANK_PREFILTER): a DRILL still needs a MEMBRANE site; a JOIN
  needs "no hull air between the faces" + "the loop encloses a hull tunnel the surface does not realise yet".
  The dangerous case (real narrow gap filled by the hull, g67f5) has linking vector 0 and stays excluded.
- Both meshes then end at genus 4, 4/4 (91 s and 246 s for the late pass).

End to end, sx1-10 (`results_genus/linking_audit_fertility_strict_sx.txt`): **10/10 genus 4 with 4/4 tunnels
realised**, no run with genus > rank. 5 runs were already right after refinement, 4 had micro-handles removed (one
or two) and re-joined, 1 arrived at genus 3 honestly and was completed. One rejection in total. CD 0.0063-0.0065,
VolIoU 0.989-0.992. Wall 777-1146 s, but the GPU was shared during this run (every stage 2-3x slower than in
st1-10, e.g. Stage 2 34 s vs 12 s); the strict steps themselves cost 3-6 s (repair) + 12 s (late pass with
nothing to do) to ~165 s (two late handles, under contention). Timing must be re-measured on a free GPU.

Still open: other shapes (air loops), the origin of the micro-handles in Stage 3, cut-and-cap as a DLFL operator.

## 7h. 2026-10-02/03: all five shapes under STRICT=1 - 15/15 strict-correct, no time tail

- Air loops are now shape-agnostic: tree-cotree generators of the hull's marching-cubes surface, lifted off the
  surface up the distance-to-solid field and routed through the air (cost 1/distance); the g threading,
  independent ones are kept by linking them with the hull's own loops (count must equal the hull genus). The
  plug-based construction failed on kitten (disk plug: one interface component, centreline not through the hole)
  and threeholes (duplicate plugs). GT meshes audit 4/4, 3/3, 1/1, 1/1. v6.4 finals of threeholes / kitten /
  rockerarm / armadillo are strict-correct 12/12: fake handles are a fertility problem (parts pressed together).
- s5 (first 5-shape run): 14/15. fertility s5B was 4/4 after the late pass and 2/4 after the Stage-6 refine: the
  refine collapsed two handles AND damaged the geometry (CD 0.0112, VolIoU 0.9856, held-out 0.93) while the genus
  count said OK. Stage 6s added: re-audit after the refine; on a mismatch repair + late pass + 6b refit. s5B
  resumed -> 4/4, CD 0.0064, VolIoU 0.9919 (one extra handle was not a 3-cycle and could not be removed; the 6b
  refit happened to re-inflate it onto its tunnel - not guaranteed by the rule, the final audit is the safety net).
- s6 (second run, with 6s, GPU free; `results_genus/linking_audit_strict_s6.txt`): **15/15 genus right and all
  tunnels realised**. Stage 6s did not fire. fertility: micro-handles removed in 2 of 3 runs. Wall: threeholes
  340-342 s, kitten 276-299, rockerarm 279-342, armadillo 254-290, fertility 482-523. No 20-minute tail (no hull
  face-pair search in the strict late pass). CD / VolIoU as v6.4.
- Counting everything run with the final rules: fertility 10 (sx) + 3 (s6) + s5A/C + s5B resumed, other shapes
  12 (s5) + 12 (s6): no strict failure after Stage 6s was added.

## 7i. 2026-10-03: where the fake handles come from (Stage-3 diagnostics) and what does NOT save time

Diagnostics: Stage 3 re-run on 8 chains with every round's mesh kept (29 accepted handles).
- Tried and failed as Stage-3 criteria: rim shape (3-gon rims are suspicious but not decisive); render gain
  (`GATE_DHO_ONLY=1 VERIFY_DHO=0.005`, t3 run: 7/7 strict but micro-handles still removed in 4 of 7 chains, time
  unchanged - the "low gain = fake" split seen on 8 chains did not hold); linking vector of the candidate loop (zero
  for 22 of 29, including every handle of fully correct chains: polluted by the unopened membranes); non-face
  3-cycles near the handle (thin real arms have them too); voxel genus tests (thin sheets make the voxel genus noisy).
- What works: **air-loop crossings** (`linking_audit.loop_crossings`). Cast the hull-tunnel air loops through the
  current mesh: a sealed tunnel's loop crosses the surface exactly twice, an open one zero times. Clean integers at
  the coarse stage, exact for thin membranes. Real drills lie 0.00-0.29 from a crossing point.
- Three kinds of fake handle: (1) a hole drilled in a free-standing FIN, >= 0.75 from any crossing; (2) a drill made
  when NO loop is crossed any more: all tunnels are already open (through interpenetrating parts), genus is short,
  and the pipeline keeps looking for something to drill - what is missing then is a JOIN; (3) a second drill in a
  tunnel that already has one: the strip between the two holes shrinks to a thread = the micro-handle.
- `AIR_GUARD=1` (default off): a drill needs a crossing within AIR_GUARD_R=0.5 and at least one sealed tunnel;
  joins exempt. It refuses kinds (1) and (2) (5 refusals in the 8-chain diagnostic, 3 in ag1-10) but not (3).
  ag1-10 (full chains): 10/10 strict, micro-handles still removed in 4/10, time unchanged.
- The time lesson (my earlier estimate was wrong): fake handles are NOT what costs fertility +2.5 min over v6.4.
  Chains without any fake handle (ag3, ag4, ag10) need the late pass too, because one tunnel is typically opened
  by interpenetration and needs a JOIN, which is only reliable on the refined mesh. Removing a micro-handle costs
  4 s; the late join (~95 s) + seam refit (~45 s) are the cost, in more than half of the fertility chains.
  Speed can only come from those two steps themselves (DR verification 400 steps, refit 500 steps).
- A stricter Stage-3 rule "a handle must reduce some loop's crossings" would reject real handles: big slab drills
  often need more than one 400-step loop before the loop is free (sx1 r1, sx3 r3, sx4 r1, sx6 r1, sx7 r2).

## 8. Open problems

1. Recalibrate on stage-3 mid-round meshes: geodesic threshold (a true contact at 20) and the handling
   of negative winding (a true membrane at −1). Save those meshes and every candidate's features + face
   centroids.
2. Per-handle GT criterion for joins (e.g. GT connectivity between the regions of the two sheets).
3. Thin tunnels the visual hull cannot carve (the oracle itself would undercount) — second source of air
   evidence from image-space see-through, or a finer / adaptive hull (Boss: octree near the surface).
4. Seam cracks on the arms (folds = the "crease" population): zip them topology-preservingly.
5. hull-completion face-pair search is slow (~15 min per query); batch the rays. The near-pair detector (7d)
   finds the same sites in 2 s but needs the join criterion of item 2 first.
6. Redundancy-aware layout audit: per-handle GT audit passes bt1 (7c); only the visual GT gallery catches it.
