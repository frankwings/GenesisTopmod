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
