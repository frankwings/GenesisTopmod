# Hull-located handles: find WHERE the tunnels are from the space-carved hull (spec, 2026-09-14)

## Why
Genus discovery (phase7_handle.py) already gets the COUNT g* from the space-carved hull (LESSONS 24). The LOCATION
still comes from image evidence (`find_tunnel_by_rays`: see-through blobs in training silhouettes, MIN_PX/OUT_VOX
ladder, one handle per round, DR verification between rounds). Boss wants the location from the hull too: a 3D
connected-component analysis of the tunnels. PoC `poc_hull_plugs.py` (fertility, coarse genus-0 mesh
`cow_fertility_fertility_v5_cc3p4.npz`) shows it works: closing-radius ladder on the clean hull, each newly filled
component that lowers the (cavity-corrected) hull genus by exactly 1 is one tunnel's throat "plug"; 4 plugs found
= g* in 48 s at 128^3; rays from the plug centroid along +-thin-axis hit two non-adjacent mesh faces (the two sides
of the blocking membrane) for plugs 1, 2, 4; plug 3 missed on one side (centroid not inside the membrane) and plug
4 was a 256k-voxel blob (throat fused with concavity fillets at R=26). Those two weaknesses are what to fix.

## Deliverable: `find_tunnel_by_hull(V, F, HF, prev_handles, g_target)` in `phase7_handle.py`
Returns a LIST of candidates `(fi, fj, ci, cj, key)` (same tuple contract as `find_tunnel_by_rays`, key =
["hull", [cx,cy,cz]] with the plug centroid in world units, rounded 3 dp), one per tunnel not yet handled, ordered
by closing radius (thinnest throat first). Empty list if none.

Algorithm (numpy/scipy/skimage only, CPU):
1. `hc = clean(HF.hull)` exactly as `hull_genus` does (closing+opening r=2, largest component, fill cavities).
   Downsample to 128^3 by 2x2x2 max-pool (PoC) — keep `HULL_LOC_RES` env (128 default, 256 allowed).
2. `genus_solid(vol) = 1 - euler_number(binary_fill_holes(vol), connectivity=1)` (cavity-corrected; LESSONS 24).
3. Closing ladder R = 4,6,...,40 via two EDTs (dilate: EDT(~h) <= R; erode: EDT(dil) > R). At each R:
   `D = closing_R(hc) & ~hc & ~claimed`; 26-connected components; for each component (size >= 50) compute
   `dg = genus_solid(cur | comp) - genus_solid(cur)` with `cur = hc | claimed`.
   - dg == -1: it is one throat -> plug.
   - dg <= -2: merged plugs -> split. Do NOT use the PoC's raw watershed-into-100-pieces; instead: watershed on
     -EDT(comp) with peak markers (min_distance 6), then GREEDILY MERGE adjacent pieces starting from the piece
     with the highest EDT peak until the union has dg == -1; that union is the plug; repeat on the remainder.
   - dg == 0: skip (fillet).
   Claim each plug (`claimed |= plug`), stop when genus_solid(hc | claimed) == 0 or R > 40.
   Also stop early if the number of plugs == g_target - current mesh genus (never propose more than needed).
4. THROAT ISOLATION for the last/large plugs: after a plug is accepted, shrink it: keep only voxels with
   EDT(comp) below the median of the plug's EDT along its thin axis? Simpler and sufficient: the plug's THIN AXIS
   and CENTER come from the sub-blob = plug voxels within 1.5x the throat radius of the plug's EDT maximum
   (throat radius = max EDT). Use that sub-blob's PCA smallest-variance direction as the tunnel axis and its
   centroid as the center. Assert extents: sqrt(eig) along the axis < along the other two (a disk), else fall back
   to rays for this tunnel.
5. FACE PAIR: cast rays with Open3D RaycastingScene from sample points inside the plug (centroid first, then up to
   8 voxels sampled along the axis line through the centroid within the plug) along +axis and -axis; the first hit
   in each direction gives (fi, fj). Accept when both hit, fi != fj, no shared vertex, both faces' membrane patches
   are disks (`membrane_patches` chi == 1, same rule as rays), and both face centroids are within 3 voxels of the
   plug (the ray must not have escaped to the outer skin). Otherwise try the next sample point; if none works,
   drop this tunnel (rays fallback will handle it).
6. Dedup against `prev_handles` (R_DEDUP rule already in phase7) so re-runs do not re-add.

## Integration
- New env `DETECT=hull` (make it the default; `DETECT=rays` = old behaviour). In the main loop, when DETECT=hull:
  call `find_tunnel_by_hull` once per round, add ALL returned handles in that round (each via the existing
  `add_handle`/merge path, re-checking watertight + genus after each), then the existing g* stop rule applies.
  If genus < g* after the hull candidates are exhausted, fall back to `find_tunnel_by_rays` with the RELAX ladder
  (existing code) — hull first, rays as fallback. Log `[p7] hull plugs: N found (R=..), M accepted, genus a -> b`.
- Do not change golden_v3_chain.sh; Stage 3 and 5b pick up DETECT=hull automatically via the default.

## Acceptance
- `test_hull_locate.py`: on the archived coarse meshes for all five shapes
  (`/tmp/liou_cow_viz/cow_<shape>_<shape>_v5_cc3p4.npz`, build HF exactly as phase7 does) the number of plugs found
  == g* (armadillo 0, kitten 1, rockerarm 1, threeholes 3, fertility 4) and every accepted candidate passes the
  face-pair rules. Print per-shape wall time (target < 60 s per shape at 128^3).
- Full chain: `TAGP=v6 SHAPES="fertility threeholes kitten rockerarm armadillo" bash golden_v3_chain.sh` reaches
  5/5 correct genus; scores within 0.002 VolIoU of GOLDEN v5; Stage 3 wall not slower than v5 (fertility 80 s).
  GPU guard as usual (/usr/lib/wsl/lib/nvidia-smi used < 28000 MiB before launching; ~2 GB per run).
- Report: per-shape plugs/genus/wall, the v6 table, and which tunnels needed the rays fallback (be explicit).
