# Operator Vocabulary Roadmap

*Created 2026-08-06 · Source: verified inventory of `davyrisso/topmod3d`
(`DLFLSubdiv.hh` 22 subdivision functions + `dlflaux` connect/crust modules).
All extensions are clean-room re-implementations from documented semantics —
no GPL code is copied.*

## Current Vocabulary (implemented, oracle-validated)

| Token | Oracle (ΔV, ΔE, ΔF for n-gon) | Status |
|---|---|---|
| `extrude_face` | (+n, +2n, +n), Δg=0 | ✅ tests/test_semantic_oracle.py |
| `add_handle` | (0, +n, +n−2), Δg=+1 | ✅ |
| `stellate` / `subdivide_face` | (+1, +n, +n−1) | ✅ |
| `subdivide_edge` | (+1, +1, 0) | ✅ |
| `catmull_clark` | V'=V+E+F, E'=4E, F'=2E, all-quad | ✅ |
| `dual` (token `DUAL`) | V'=F, E'=E, F'=V; involution | ✅ 2026-08-06, topmod/remeshing.py |
| `doo_sabin` (token `DS`) | V'=2E, E'=4E, F'=V+E+F | ✅ 2026-08-06, topmod/remeshing.py |
| `stellate_all` (token `STA`) | V'=V+F, E'=3E, F'=2E, all-tri | ✅ 2026-08-06, topmod/high_level_ops.py |
| `simplest_subdivide` (token `SIMP`) | V'=E, E'=2E, F'=F+V (cube→cuboctahedron) | ✅ 2026-08-06, topmod/remeshing.py |
| `vertex_cutting` (token `VC`) | V'=2E, E'=3E, F'=F+V (cube→truncated cube) | ✅ 2026-08-06, topmod/remeshing.py |
| `loop_subdivide` (token `LOOP`) | tri-only: V'=V+E, E'=4E, F'=4F | ✅ 2026-08-06, topmod/remeshing.py |
| `sqrt3_subdivide` (token `SQRT3`) | tri-only: V'=V+F, E'=3E, F'=3F | ✅ 2026-08-06, topmod/remeshing.py |
| `honeycomb_subdivide` (token `HONEY`) | V'=2E, E'=3E, F'=V+F (= dual∘stellate_all) | ✅ 2026-08-07, topmod/remeshing.py |
| `star_subdivide` (token `STAR`) | V'=V+F+2E, E'=9E, F'=6E (= stellate_all², param offset) | ✅ 2026-08-07 |
| `corner_cutting` (token `CCUT`) | V'=2E, E'=4E, F'=V+E+F (DS topology, param α) | ✅ 2026-08-07 |
| `loop_style_subdivide` (token `LSTYLE`) | V'=V+E, E'=4E, F'=F+2E (polygonal Loop connectivity) | ✅ 2026-08-07 |
| `fractal_subdivide` (token `FRAC`) | V'=V+E+F, E'=6E, F'=4E (loop_style + apex fans, param offset) | ✅ 2026-08-07 |
| 4 fundamental ops (create/delete vertex, insert/delete edge) | per Akleman & Chen 2003 | ✅ |

Tokenizer note: `DUAL`/`DS` vocabulary IDs are appended AFTER the REF block,
so all legacy IDs (EOS/CC/CV/IE/DE/HDL, COORD_*, REF_*) are unchanged —
sequences and Phase A/B checkpoints encoded with the old vocabulary stay valid.

## Tier 1 — DONE (2026-08-06)

All Tier-1 candidates implemented and oracle-validated; see the table above.

## Tier 2 — status after batch 2 (2026-08-07)

Semantics for ALL TopMod-specific schemes extracted clean-room into
`docs/reference_semantics.md` (χ-verified closed-form oracles).

DONE (batch 2): honeycomb, star, corner_cutting, loop_style, fractal —
see the Current Vocabulary table.

Remaining:

| Token candidate | Oracle (V', E', F') | Notes |
|---|---|---|
| `pentagonal` | V+2E+F, 5E, 2E | trisect edges + centroid spokes (batch 3) |
| `pentagonal2` | V+3E, 6E, F+2E | batch 3 |
| `dual1264` | 4E, 6E, V+E+F | DS-like, 2d-gon inner faces (batch 3) |
| `root4` | V+2E, 4E, F+E | honeycomb-mask inner polygon + bridge (batch 3) |
| `checkerboard` | V+4E, 9E, F+4E | batch 4 |
| `ds_bc_new` | V+4E, 7E, F+2E | batch 4 |
| `dome` | V+59E, 116E, F+56E | needs subdivide_all_edges + DS-extrude (batch 4) |

## Tier 3 — Structural operators (beyond subdivision)

From `dlflaux` standalone modules (verified present 2026-08-06):

| Token candidate | Reference | Value |
|---|---|---|
| `multi_connect` | `DLFLMultiConnect` | Generalized add_handle across face sets |
| `bezier_handle` | `DLFLCubicBezierConnect` | Curved handles — parametric geometry + topology |
| `crust` / `shell` | `DLFLCrust` | Solidify: mesh → double-walled shell (Δg doubles+) |
| `convex_hull` | `DLFLConvexHull` | Utility, not manifold-preserving per se — gate carefully |

## Rules for Adding a Token

1. Write the **oracle first**: closed-form ΔV/ΔE/ΔF/Δχ prediction in
   `tests/test_semantic_oracle.py`, parametrized over all primitives.
2. Clean-room implement from paper/header semantics only (GPL hygiene).
3. `check_all()` must pass after the op on every primitive + genus-1/2 meshes.
4. Register in `tokenizer.py` vocabulary; add round-trip test.
5. Parametric ops (corner_cutting α, extrude dist) — parameters are
   continuous token attributes, not separate tokens.

## Ceiling

18 current + ~7 (T2 remaining) + 4 (T3) ≈ **29-op vocabulary**, matching the
original TopMod's expressive range while remaining differentiable-pipeline-
and Blender-embeddable.
