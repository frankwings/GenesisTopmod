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
| 4 fundamental ops (create/delete vertex, insert/delete edge) | per Akleman & Chen 2003 | ✅ |

Tokenizer note: `DUAL`/`DS` vocabulary IDs are appended AFTER the REF block,
so all legacy IDs (EOS/CC/CV/IE/DE/HDL, COORD_*, REF_*) are unchanged —
sequences and Phase A/B checkpoints encoded with the old vocabulary stay valid.

## Tier 1 — DONE (2026-08-06)

All Tier-1 candidates implemented and oracle-validated; see the table above.

## Tier 2 — Medium (distinct topology flavors)

| Token candidate | Reference | Notes |
|---|---|---|
| `honeycomb` | `honeycombSubdivide` | Hexagonal-dominant remeshing |
| `pentagonal` | `pentagonalSubdivide` ×2 | Pentagon-dominant |
| `corner_cutting` | `cornerCuttingSubdivide` ×3 (α variants) | Parametric — first token with a continuous parameter |
| `root4` | `root4Subdivide` | Triangle scheme (parametric: a, twist) |
| `star` / `fractal` / `dome` | `starSubdivide` etc. | Ornamental; showcase value for Blender demo |

Note: honeycomb / pentagonal / corner_cutting / star / fractal / dome /
dual1264 are TopMod-specific — their exact semantics must first be
extracted from the reference headers/papers (read for semantics only,
clean-room re-implementation) before oracles can be written.

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

13 current + ~7 (T2) + 4 (T3) ≈ **24-op vocabulary**, matching the
original TopMod's expressive range while remaining differentiable-pipeline-
and Blender-embeddable.
