# Review: davyrisso/topmod3d — Why We Don't Build on It

*Reviewed: 2026-08-06 · https://github.com/davyrisso/topmod3d*

## What It Is

This repository is the **original TopMod C++ codebase from Akleman's group
(Texas A&M)**, revived by davyrisso in 2023 to compile against modern Qt 5.13
(macOS/Linux/Windows). It is a maintenance fork, not a new implementation —
last meaningful activity is a Windows build fix merged 2023-08.

Structure (173 C++ files):

| Module | Contents |
|---|---|
| `include/dlflcore` (~11k lines) | The authentic DLFL data structure — corner-based (`DLFLFaceVertex`) face lists, faithful to the 2003 paper; edges, faces, file I/O |
| `include/dlflaux` | ~20 subdivision/remeshing schemes (Doo-Sabin, honeycomb, pentagonal, star, fractal, dual, corner-cutting, loop, root-3, …), crust/shell modeling, sculpting, multi-connect, Bezier handles |
| `include/pydlfl` | Python bindings — ancient C-API, Python-2 era, effectively unusable |
| `topmod/` | Qt widget GUI (modes, renderers, script editor) |

## Why It Is Unsuitable as Our Substrate

1. **GPL v2 contagion.** The entire codebase is GPL v2+ (verified
   2026-08-06: no root LICENSE file, but source files carry full
   "GPL v2 or later" header blocks — per-file headers are the only safe
   interpretation). Linking or porting
   its code would force GenesisTopmod (and anything shipping it) under GPL,
   blocking future commercial licensing. We may read it for *semantics*,
   never copy implementation.

2. **No differentiability / no ML interop.** Plain C++ with hand-rolled
   vector math, deeply coupled to Qt/OpenGL rendering. No autograd, no
   tensor interop. Our pipeline needs meshes that flow into
   PyTorch/nvdiffrast (Tab 2 fitting, Tab 4 differentiable manifold loss,
   Phase A training) — retrofitting that into this codebase costs more than
   our from-scratch Python library did.

3. **GUI coupling and build burden.** Core logic is entangled with a Qt 5.13
   desktop app (registration-gated Qt download, qmake build). As a library
   dependency this is dead weight; the Python bindings that could have
   rescued it are unmaintained.

4. **Wrong deployment shape for the Blender plugin goal.** Blender add-ons
   are pure-Python (`bpy`) packages. Our 2,600-line zero-dependency Python
   `topmod/` library can be vendored into an add-on as-is and runs inside
   Blender's bundled interpreter on every platform. The C++ codebase would
   require per-platform compiled binding wheels, a build matrix, and GPL
   licensing of the binding layer — all cost, no benefit.

## What It IS Useful For

1. **Semantic oracle.** Reference implementation to cross-validate our
   operators: same mesh, same operation, compare V/E/F/genus against ours.

2. **Vocabulary roadmap.** Its subdivision/remeshing schemes are candidate
   tokens for our tokenizer (currently 6 ops). Verified count (2026-08-06,
   shallow clone): `DLFLSubdiv.hh` exposes **22 subdivision functions**
   (loop, honeycomb, pentagonal ×2, doo-sabin BC ×2, corner-cutting ×3,
   root4, catmull-clark, star, sqrt3, fractal, stellate ×2, dome, dual1264,
   checkerboard, simplest, vertex-cutting, loop-style). It proves the
   operator algebra scales to 30+ ops — a much higher expressiveness
   ceiling for the generative model. When we extend the vocabulary
   (e.g. a Doo-Sabin token), we re-implement from its documented semantics
   in `DLFLSubdiv.hh` — clean-room, no code copied.

   Beyond subdivision, `dlflaux` also contains standalone high-value token
   candidates: `DLFLConvexHull`, `DLFLCrust` (crust/shell modeling),
   `DLFLMultiConnect`, and `DLFLCubicBezierConnect` (Bezier handles) —
   richer connect operators beyond our current `add_handle`.

3. **Positioning evidence.** Confirms the original is C++/GPL/GUI-bound,
   which is precisely why a from-scratch, ML-native Python reimplementation
   is a real contribution rather than duplication.

## Blender Plugin Direction (new goal, noted 2026-08-06)

Target: expose TopMod operators inside Blender as an add-on
(interactive topology modeling + a bridge to our generative pipeline).

Key facts for planning:
- Our pure-Python `topmod/` library is directly embeddable — no compiled
  deps, works in Blender's bundled Python (3.11+).
- Mesh exchange: `topmod.io.to_triangle_arrays()` ↔ `bpy` mesh
  (`from_pydata` / `bmesh`); DLFL remains the source of truth for topology,
  Blender is the viewport/editor.
- Licensing: Blender add-ons that import `bpy` are effectively
  GPL-compatible when distributed. We own our copyright, so we can
  dual-license: the add-on layer GPL, the core `topmod/` library under our
  own license. Keeping the original topmod3d's GPL code OUT of our tree is
  what preserves this freedom.
- Natural add-on features (v1): genus-N primitive creation, add-handle
  between two selected faces, CC/Doo-Sabin subdivision, manifold validity
  HUD (χ, genus, check_all), operator-history panel (= live token
  sequence view).
