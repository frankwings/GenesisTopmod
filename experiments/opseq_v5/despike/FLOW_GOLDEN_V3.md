# Golden v3 chain — every step and sub-step (as run in `gif_v9_chain.sh`, 2026-09-09)

Figure: `results_genus/flow_golden_v3.png`. Blue = nvdiffrast DR (every step renders all 64 views),
orange = TopMod DLFL operator (`check_watertight` asserted after each), green = Open3D Taubin, grey = numpy logic.

| Stage | Script | What happens every step | TopMod operators |
|---|---|---|---|
| 1 sphere → coarse | `run_64v.py STOP_AFTER=cc3` (C2F_SUBDIV=cc default) | icosphere = TopMod `make_icosahedron` + 2x `catmull_clark` (240 quads, fan-triangulated by `triangulate_all`); cc2 800 steps; TopMod `catmull_clark` on the quad mesh (1->4) + `triangulate_all`; cc3 800 steps. Each step: 64-view nvdiffrast (sil + depth + headlight diffuse) → L1 + Laplacian + edge + quality + spike/sliver/fold/tube regs → Adam | `catmull_clark`, `triangulate_all` |
| 2 coarse clean | `phase4_inloop.py` 400 | DR loss + voting-hull field | every 100: `collapse_edge_tri`; every 25: flip = `delete_edge` + `insert_edge`, then tangential smooth + SI push (geometric) |
| 3 genus discovery | `phase7_multi.sh` ≤ 8 rounds | detect see-through pixels (≥ 30 px) of training silhouettes where the mesh is solid → membrane patches outside the voting hull → disk (χ) test → disjoint-1-ring face pair; then a 400-step Stage-2 loop; stop when no evidence | `add_handle(face_i, face_j)`, `stellate` (side quads), `subdivide_edge` (tube refine) |
| 4 cc4 + despike | `run_64v.py RESUME_FROM` | TopMod `catmull_clark` on the triangle mesh (1 tri -> 3 quads -> 6 tris via `triangulate_all`) → cc4 800 steps → surgery 1 → settle 400 → surgery 2 | `catmull_clark`, `triangulate_all`, `collapse_edge_tri` (amputation) |
| 5 golden v3 loop | `phase4_inloop.py` 1200 | DR + Palfinger optimizer (β 0.8/0.8, ν-weighted Laplacian on the gradient, clip 10|m1|, lr 0.3 × edge); every 50 steps adaptive remesh (velocity-controlled target, floor 1.3 px, cap 100k f, SI gate 30 %) | split = `subdivide_edge` + `stellate`; collapse = `collapse_edge_tri`; flip = `delete_edge` + `insert_edge` |
| 6 golden v3 loop, LAP ×3 | `phase4_inloop.py` 1200 | identical | identical |
| 7 fairing | `phase5_taubin.py AUTO` | Open3D Taubin λ 0.5 / μ −0.53, iteration count = argmax training IoU (min 2); positions only | none |
| exam | `eval_cd_iou.py` | ho16 on 16 held-out silhouettes; ICP → Chamfer → 256³ VolIoU | — |

## Despike surgery (Stage 4, `surgery_lib.py`, v22 = seed + extent + propagation; see NEEDLE_REMOVAL.md)
1. **Extent** — geometric: a vertex is *thin* if its distance to the nearest non-adjacent surface is < 0.4 × mean edge (`tube_mask`); spike vertices are flagged too. Thin vertices are grouped into connected components (needle = whole thin shaft).
2. **Seed** — image evidence: `escape_mask` projects every vertex into the training views and flags those landing OUTSIDE the (1-px-dilated) GT silhouette: the needle tip creates area the target does not have.
3. **Propagation** — a thin component that contains ≥ 1 seed is condemned end-to-end (root and all); a thin component with no escaping tip (the ear) is spared.
4. **Amputation** — TopMod `collapse_edge_tri`: first collapse edges inside the condemned set, then collapse every remaining condemned vertex into a healthy neighbour, so the vertices truly disappear and the mesh stays a manifold.
5. **Acceptance** — the amputation is kept only if the training-view IoU drops ≤ 3e-4 relative to the current value and never below iou0 − 1.5e-3; otherwise reverted. ≤ 8 rounds, then 400 settle steps, then a second pass (budget 2e-4, ≤ 4 rounds).

## Which stages touch topology
- TopMod DLFL: stages 1, 2, 3, 4, 5, 6 — EVERY topology change in the chain. Operators: `catmull_clark`, `triangulate_all`, `collapse_edge_tri`, `delete_edge`, `insert_edge`, `subdivide_edge`, `stellate`, `add_handle` (`check_all` / `check_watertight` after each).
- Since 2026-09-11 (`C2F_SUBDIV=cc` default, LESSONS 23) the coarse-to-fine refinement is TopMod Catmull-Clark too; the legacy numpy midpoint split (`C2F_SUBDIV=midpoint`) gave identical final scores (0.9983/0.9949 vs 0.9982/0.9948 on armadillo).
- No topology change: Stage 7 Taubin, every DR step.
