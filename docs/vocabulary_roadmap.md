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
| 4 fundamental ops (create/delete vertex, insert/delete edge) | per Akleman & Chen 2003 | ✅ |

Tokenizer note: `DUAL`/`DS` vocabulary IDs are appended AFTER the REF block,
so all legacy IDs (EOS/CC/CV/IE/DE/HDL, COORD_*, REF_*) are unchanged —
sequences and Phase A/B checkpoints encoded with the old vocabulary stay valid.

## Tier 1 — High value, low risk (next)

Global schemes with clean closed-form oracles; each is one new token and
directly useful in both the generative pipeline and the Blender add-on.

| Token candidate | Reference | Oracle (closed form) | Why first |
|---|---|---|---|
| `simplest_subdivide` | `simplestSubdivide` | mid-edge scheme: V'=V+E, faces split | Trivial oracle, cheap win |
| `vertex_cutting` | `vertexCuttingSubdivide` | truncation: V'=Σ valence, per-vertex n-gon added | Dual flavor of CC |
| `stellate_all` | `stellateSubdivide` (global) | per face: (+1, +n, +n−1) summed | Already have per-face version |

## Tier 2 — Medium (distinct topology flavors)

| Token candidate | Reference | Notes |
|---|---|---|
| `honeycomb` | `honeycombSubdivide` | Hexagonal-dominant remeshing |
| `pentagonal` | `pentagonalSubdivide` ×2 | Pentagon-dominant |
| `corner_cutting` | `cornerCuttingSubdivide` ×3 (α variants) | Parametric — first token with a continuous parameter |
| `root4` / `sqrt3` | `root4Subdivide`, `sqrt3Subdivide` | Triangle schemes |
| `loop_subdivide` | `loopSubdivide` | Requires all-tri input — first token with a precondition |
| `star` / `fractal` / `dome` | `starSubdivide` etc. | Ornamental; showcase value for Blender demo |

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

8 current + 3 (T1) + ~9 (T2) + 4 (T3) ≈ **24-op vocabulary**, matching the
original TopMod's expressive range while remaining differentiable-pipeline-
and Blender-embeddable.
