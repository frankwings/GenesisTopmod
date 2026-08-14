# Operator Vocabulary Roadmap

*Created 2026-08-06 · Updated 2026-08-13 · Source: verified inventory of
`davyrisso/topmod3d` (`DLFLSubdiv.hh` 22 subdivision functions + `dlflaux`
connect/crust modules). All extensions are clean-room re-implementations
from documented semantics — no GPL code is copied.*

## Current Vocabulary (46 operators, oracle-validated)

### 1. Fundamental Operators (Akleman & Chen 2003)

| Token | Operator | Oracle | Diff | Status |
|---|---|---|---|---|
| `CV` | `create_vertex` | V+1, F+1 (point sphere) | ✅ param | ✅ |
| — | `delete_vertex` | V−1, F−1 | — | ✅ |
| `IE` | `insert_edge` | E+1; same face→F+1, cross→F−1 | ✅ identity | ✅ |
| `DE` | `delete_edge` | E−1; merge/split face | ✅ identity | ✅ |

### 2. High-Level Operators

| Token | Operator | Oracle | Diff | Status |
|---|---|---|---|---|
| — | `extrude_face` | V+n, E+2n, F+n | ✅ torch | ✅ |
| — | `stellate` | V+1, E+n, F+n−1 | ✅ torch | ✅ |
| — | `subdivide_edge` | V+1, E+1, F+0 | ✅ torch | ✅ |
| — | `subdivide_face` | V+1, E+n, F+n−1 | ✅ torch | ✅ |
| `HDL` | `add_handle` | E+n, F+n−2, genus+1 | ✅ identity | ✅ |
| `STA` | `stellate_all` | V'=V+F, E'=3E, F'=2E | ✅ linear | ✅ |
| — | `collapse_edge` | V−1, E−(d0+d1−3), F−2 | ❌ pending | ✅ |
| — | `trisect_edge` | V+2, E+2, F+0 | ❌ pending | ✅ |
| `SAE` | `subdivide_all_edges` | V'=V+E, E'=2E, F unchanged | ✅ linear | ✅ |
| — | `subdivide_all_faces` | V'=V+F, E'=3E, F'=2E | ✅ linear | ✅ |
| — | `triangulate_face` | V+0, E+(n−3), F+(n−3) | ✅ identity | ✅ |
| `TRI` | `triangulate_all` | V+0, E+Σ(d_i−3), F+Σ(d_i−3) | ✅ identity | ✅ |
| — | `double_stellate_face` | complex | ✅ torch | ✅ |
| `STSUB` | `stellate_subdivide` | V'=V+F, E'~2Σd_i, F'~Σd_i | ✅ linear | ✅ |
| — | `punch_hole` | = add_handle | ✅ identity | ✅ |
| — | `extrude_face_dome` | V+~5n, E+~10n, F+~5n | ✅ torch | ✅ |
| — | `make_wireframe` | complex (MCC→crust→punch) | ❌ pending | ✅ |

### 3. Classic Subdivision

| Token | Operator | Oracle | Diff | Status |
|---|---|---|---|---|
| `CC` | `catmull_clark` | V'=V+E+F, E'=4E, F'=2E (all-quad) | ✅ linear | ✅ |
| `DUAL` | `dual` | V'=F, E'=E, F'=V | ✅ linear | ✅ |
| `DS` | `doo_sabin` | V'=2E, E'=4E, F'=V+E+F | ✅ linear | ✅ |
| `SIMP` | `simplest_subdivide` | V'=E, E'=2E, F'=F+V | ✅ linear | ✅ |
| `VC` | `vertex_cutting` | V'=2E, E'=3E, F'=F+V | ✅ linear | ✅ |
| `LOOP` | `loop_subdivide` | V'=V+E, E'=4E, F'=4F (tri-only) | ✅ linear | ✅ |
| `SQRT3` | `sqrt3_subdivide` | V'=V+F, E'=3E, F'=3F (tri-only) | ✅ linear | ✅ |

### 4. TopMod Remeshing Schemes

| Token | Operator | Oracle | Diff | Status |
|---|---|---|---|---|
| `HONEY` | `honeycomb_subdivide` | V'=2E, E'=3E, F'=V+F | ✅ linear | ✅ |
| `STAR` | `star_subdivide` | V'=V+F+2E, E'=9E, F'=6E | ✅ torch | ✅ |
| `CCUT` | `corner_cutting` | V'=2E, E'=4E, F'=V+E+F | ✅ linear | ✅ |
| `LSTYLE` | `loop_style_subdivide` | V'=V+E, E'=4E, F'=F+2E | ✅ linear | ✅ |
| `FRAC` | `fractal_subdivide` | V'=V+E+F, E'=6E, F'=4E | ✅ torch | ✅ |
| `PENT` | `pentagonal_subdivide` | V'=V+2E+F, E'=5E, F'=2E | ✅ linear | ✅ |
| `PENT2` | `pentagonal2_subdivide` | V'=V+3E, E'=6E, F'=F+2E | ✅ linear | ✅ |
| `D1264` | `dual1264_subdivide` | V'=4E, E'=6E, F'=F+E+V | ✅ linear | ✅ |
| `ROOT4` | `root4_subdivide` | V'=V+2E, E'=4E, F'=F+E | ✅ linear | ✅ |
| `CHKB` | `checkerboard_remesh` | V'=V+4E, E'=9E, F'=F+4E | ✅ linear | ✅ |
| `DSBC` | `ds_bc_new_subdivide` | V'=V+4E, E'=7E, F'=F+2E | ✅ linear | ✅ |
| `DOME` | `dome_subdivide` | V'=V+59E, E'=116E, F'=F+56E | ✅ torch | ✅ |
| — | `doo_sabin_bc` | complex (SAE then DS) | ✅ torch | ✅ |
| — | `two_stellate_subdivide` | complex (2-pass stellate) | ✅ torch | ✅ |
| `MCC` | `modified_corner_cutting` | V'=2E, E'≈5E, F'=V+F | ✅ torch | ✅ |
| `MCC2` | `modified_corner_cutting2` | V'=2E, E'≈5E, F'=V+F | ✅ torch | ✅ |

### 5. Structural Operators

| Token | Operator | Oracle | Diff | Status |
|---|---|---|---|---|
| `CRUST` | `create_crust` | V'=2V, E'=2E, F'=2F | ✅ torch | ✅ |
| — | `create_crust_with_scaling` | V'=2V, E'=2E, F'=2F | ✅ torch | ✅ |

## Summary

| Metric | Count |
|---|---|
| Total operators | 46 |
| Tokenizer opcodes | 26 (others are local/composite ops without tokens) |
| Differentiable (PyTorch) | 43 / 46 (93%) |
| Not yet differentiable | 3 (`collapse_edge`, `trisect_edge`, `make_wireframe`) |
| Blender addon coverage | 46 / 46 (100%) |
| Before/after visualizations | 47 (46 + insert_edge_cross variant) |
| Tests passing | 471 |

## Differentiability Gap

Three operators remain non-differentiable:

| Operator | Reason | Path to differentiable |
|---|---|---|
| `collapse_edge` | Removes vertices — element count decreases, tracing impossible | Would need a soft relaxation (weight vertex to zero) |
| `trisect_edge` | Trivially linear (2 interpolation points) but not yet traced | Easy — add to `_trace_linear()` |
| `make_wireframe` | Compound op (MCC→crust→punch_hole) — each sub-op is differentiable individually | Chain the 3 DiffOps |

## Tier 3 — Potential Additions

From `dlflaux` standalone modules (verified present 2026-08-06):

| Token candidate | Reference | Value | Status |
|---|---|---|---|
| `multi_connect` | `DLFLMultiConnect` | Generalized add_handle across face sets | Not implemented — data-dependent matching unsuitable as deterministic token |
| `bezier_handle` | `DLFLCubicBezierConnect` | Curved handles — parametric geometry + topology | Not planned |
| `convex_hull` | `DLFLConvexHull` | Utility, not manifold-preserving | Not planned |

## Rules for Adding a Token

1. Write the **oracle first**: closed-form ΔV/ΔE/ΔF/Δχ prediction in
   `tests/test_semantic_oracle.py`, parametrized over all primitives.
2. Clean-room implement from paper/header semantics only (GPL hygiene).
3. `check_all()` must pass after the op on every primitive + genus-1/2 meshes.
4. Register in `tokenizer.py` vocabulary; add round-trip test.
5. Parametric ops (corner_cutting α, extrude dist) — parameters are
   continuous token attributes, not separate tokens.

## Timeline

| Date | Milestone |
|---|---|
| 2026-08-06 | Tier 1 complete: 4 fundamental + CC + 7 subdivision |
| 2026-08-07 | Tier 2 complete: all 13 remeshing schemes + crust |
| 2026-08-08 | Batch 5: MCC, MCC2, doo_sabin_bc, two_stellate, crust_scaling |
| 2026-08-09 | High-level ops: collapse/trisect/triangulate/dome/wireframe |
| 2026-08-10 | Diffgeo: 35/46 differentiable via sparse trace + torch |
| 2026-08-12 | Blender addon: 46/46 complete coverage |
| 2026-08-13 | Docs: 47 gallery entries, insert_edge 4-vertex selection |
