# Topo Carving — the golden v6.1 algorithm (2026-09-17)

Golden tag: `golden-v6.1` (commit ad292a8). Chain: `despike/golden_chain.sh`. This document records the algorithm as run,
in the order the chain runs it, with the definitions used in code. Numbers are from the v6n runs (single RTX 5090, uncontended).

## 0. Problem and guarantee
Input: N=64 calibrated views of an object (synthetic: silhouette + depth + diffuse renders of a GT mesh; real: silhouettes
from SAM2 + COLMAP poses). Output: a single closed, combinatorially manifold triangle mesh whose genus is DISCOVERED from the
images. Every topology change is a TopMod DLFL operator (Catmull-Clark subdivision, edge split/collapse/flip, add_handle),
so manifoldness and watertightness hold by construction after every step (`check_watertight` asserted after each operator).
Self-intersection is not guaranteed, only measured (0 % on all five shapes).

## 1. Chain (stages)
| stage | what | script | knobs |
|---|---|---|---|
| 1 | icosphere -> TopMod CC2 -> 400/800 DR steps -> CC3 -> 400/800 DR steps | run_64v.py STOP_AFTER=cc3 | S1_STEPS (default 800), HULL_INIT (real data only) |
| 2 | DLFL clean loop, 400 steps, coarse (flip/collapse/SI push) | phase4_inloop.py | MEMB_EXEMPT=2 |
| 3 | **topology discovery**: propose one handle -> add_handle -> 400 DR steps -> verify/revert, until genus == g* | phase7_multi.sh -> phase7_handle.py -> membrane_locate.py | ROUNDS=8, DETECT=membrane, GENUS_TARGET=hull |
| 4 | CC4 + despike (needle removal v22) | run_64v.py RESUME_FROM | |
| 5 | Palfinger-parameter DLFL loop, 1200 steps, velocity-adaptive remeshing | phase4_inloop.py | $PALF, MEMB_EXEMPT=2 |
| 5b | late topology pass: same propose-and-verify loop as Stage 3 on the refined mesh (safety net) | phase7_multi.sh | ROUNDS=4 |
| 6 | LAP x3 refinement loop, 1200 steps (full hull-field loss) | phase4_inloop.py | LAP_MULT=3 |
| 7 | AUTO Taubin (position-only smoothing) | phase5_taubin.py | |
| exam | 16 held-out views (ho16 IoU), volumetric IoU, Chamfer vs GT | eval_cd_iou.py / exam_real.py | |
Wall: armadillo/kitten/rocker-arm 4.4 min, threeholes 5.0, fertility 5.5-6.2 (4-5 handle rounds).

## 2. Space carving (the hull) — used for TWO things only
`hull_field.build_vote_hull`: 256^3 voxel grid over the GT/object bbox; a voxel is carved if it projects to BACKGROUND in
>= 2 of the 64 training silhouettes (voting tolerates single-view mask noise; real data: HULL_VOTE=8). The hull is a
SUPERSET of the object (silhouette carving cannot see concavities).
(a) **Topology oracle g\***: `hull_genus` = 1 - Euler(filled solid) after morphological closing+opening at r = 1,2,3 voxels,
    largest component, cavities filled; g* = mode over r (LESSONS 24; equals GT genus on all five shapes).
(b) **Air classifier**: `HullField.dist(p)` = Euclidean distance from p to the hull (0 inside). Points with dist > margin
    are "air": the object cannot be there. Also drives the hull-field loss in DR (relu(dist - DEAD), DEAD = 1 voxel).
The hull is NOT used to locate tunnels any more (v6 did: closing-radius ladder; see §6 why it failed).

## 3. Membrane detection on the DR mesh (Boss's formulation; `membrane_locate.py`)
Premise: the DR mesh is a closed genus-g surface fitted to the silhouettes; where the true object has a tunnel the mesh has no
tunnel wall yet, so its surface SPANS the tunnel air: a closed surface cannot have a single-layer membrane, so a spanned
tunnel always shows as two skins (top disk / bottom disk of a slab hole), possibly pressed together, or as disks covering
the exits of an unopened chamber. Those spanning faces are exactly the mesh faces that lie in hull air.
1. **Membrane faces**: sample 10 interior barycentric points per face; a face is membrane if the MEDIAN sample has
   hull distance > MARGIN = 2 voxels. (Vertices are useless here: coarse meshes span a tunnel with a few large faces whose
   vertices sit on the rim, inside the margin.) Fitting slop near the hull is inside the margin.
2. **Patches**: edge-connected components of membrane faces. If a component holds both skins (normals pointing both ways,
   each side >= 30 % of the faces) it is split by normal sign. Representative face of a patch = its face deepest in air.
3. **Voxel-block ranking**: voxelise the mesh (surface rasterised at half-voxel spacing | occupancy at voxel centres, holes
   filled) on the hull grid; M = mesh_solid minus dilate(hull, 2) = "mesh material where the hull says air"; connected
   blocks of M; for each block C, k = genus(mesh_solid \ dilate(C,1)) - genus(mesh_solid). A block with k >= 1 is a
   tunnel blocker (a chamber with m exits fully solid in the mesh has k = m-1). A patch is kept only if its representative
   face is within 3 voxels of a k >= 1 block; two patches may pair only on the same block. (Removes fitting-slop patches.)
4. **Pairing**: patches (a, b) on the same block are one tunnel iff the segment between their representative faces has
   >= 80 % of 23 samples in hull air (dist > 0.5 voxel) AND >= 80 % inside the mesh (Open3D occupancy). Pinched skins
   (separation < 1.5 voxels, opposite normals) pair directly. Rank: larger min(patch size) first, then shorter separation.
5. Output: (face_i, face_j) of the best pair -> DLFL `add_handle`. Cost: 1-3 s per call (128^3 voxelisation dominates).

## 4. Propose-and-verify (`phase7_multi.sh`)
Per round: detect -> add ONE handle (membrane merge + tube refine + radial projection as before) -> 400-step DLFL/DR loop
(MEMB_EXEMPT=2) -> held-out exam. Accept iff ho16 >= previous + 0.002 or hair <= 0.9 x previous; else revert to the
previous mesh and blacklist the pair (handles json: `rejected`, `mid=None`). Stop when genus == g* or no candidate.
Fallbacks after the membrane detector returns nothing while genus < g*: ray detector (see-through pixels, relaxation ladder),
contact join. Stage 5b repeats the loop on the 20k-V mesh (v6m3 recovered 3 handles there).
Verification cost is ~free: the old chain already ran 400 DR steps after every handle; only rejected rounds cost (~25 s).

## 5. Membrane exemption from the hull-field loss (MEMB_EXEMPT)
In Stages 2/3/5 a face whose interior median lies > 2 voxels outside the hull, and its vertices, get zero hull-field
penalty. Without this the last small membrane (fertility: child's arm / mother's neck gap) was pressed flat onto the tunnel
wall by the hull pull and became invisible to every distance-based test (v6m2: 3/4). Stage 6 uses the full loss again.

## 6. Why the closing-radius ladder (golden v6) was replaced
Closing with radius R fills wherever the air passage is narrower than 2R = the narrowest cross-section, which is the
mouth only for passages that widen inward. Fertility's upper cavity is one chamber with 3 exits (+1 opening); an interior
wall splitting it also lowers the genus by exactly 1, so "dg = -1" never proved a sheet was at a mouth, and at 128^3 chamber
and exits sealed at the same radius (13) into one block whose watershed slices were not mouths. Handles then landed inside
the chamber: genus 4 reached, tunnels unopened, VolIoU 0.84, 1 success in 4 runs. Dead ends tried and recorded in LESSONS 33:
increment-sheet Euler tests, end caps of the sealed block, 0.5-voxel radius steps, deterministic hull grid. All kept as
`DETECT=hull` for comparison; none is on the golden path.

## 7. Results (v6n)
| shape | genus (GT) | handles verified/rejected | VolIoU | wall |
|---|---|---|---|---|
| armadillo | 0 (0) | 0/0 | 0.9949 | 4.4 min |
| kitten | 1 (1) | 1/0 | 0.9980 | 4.4 |
| fertility x3 | 4 (4) | 5/1, 4/0, 4/0 | 0.9909 / 0.9909 / 0.9900 | 5.5-6.2 |
| rocker-arm | 1 (1) | 1/0 | 0.9871 | 4.4 |
| threeholes | 3 (3) | 3/0 | 0.9914 | 5.0 |

## 8. Files
membrane_locate.py (detector), phase7_handle.py (add_handle + fallbacks), phase7_multi.sh (rounds + verify), hull_field.py
(hull, genus oracle, distance field), hull_locate.py (v6 ladder, optional), phase4_inloop.py (DR loops, MEMB_EXEMPT, THIN_GUARD),
golden_chain.sh (stages), viz_membranes.py / viz_mesh_membranes.py (figures), real_scene.py (real-data adapter, see real/README.md).
