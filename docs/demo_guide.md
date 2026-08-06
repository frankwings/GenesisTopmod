# GenesisTopmod Interactive Demo — Guide & Positioning

*Last updated: 2026-08-06*

The Gradio demo (`app.py`) is the public face of the project. Its four tabs are
not four independent toys — they are the four experiments of one thesis:

> **Shift 3D mesh generation from "triangle soup" to "topology operator
> programs". Manifold validity becomes a compile-time guarantee instead of a
> runtime repair.**

---

## Tab-by-Tab: What Each Tab Demonstrates

### Tab 1 — Topology Explorer (Phase 1: DLFL + operators)

**Claim: the operator system can construct arbitrary topology.**

Builds meshes of any genus (0–3 handles) with optional Catmull-Clark
subdivision, purely from TopMod/DLFL operators. Every intermediate and final
mesh is a valid orientable 2-manifold *by construction* — the stats panel
shows V/E/F, Euler characteristic χ = V−E+F, and a manifold validity check
that never fails.

Contrast: deformation-based pipelines (N3MR, SoftRas, DIB-R) start from a
sphere and can never change genus.

### Tab 2 — Geometry Optimizer (Phase 2: nvdiffrast silhouette fitting)

**Claim: TopMod meshes plug into modern differentiable-rendering pipelines.**

Fits vertex positions of a topology seed to target silhouettes via
differentiable rasterization (nvdiffrast) + Adam. This tab is deliberately
*commodity technology* — the same technique as N3MR/SoftRas — because the
point is compatibility, not novelty. The differentiator is the **genus
slider**: fitting a genus-1 target with a genus-1 seed is something a
deformed sphere mathematically cannot do.

Practical details (hard-won, see commit `6ea7887`):
- Upload 1..N silhouette images; each image = one camera view
  (1 → front; 2 → front + side at 90°, best depth constraint;
  N ≥ 3 → evenly spaced over 360°).
- Uploaded targets are normalized: binarized (object = dark minority
  region), bbox-cropped, square-padded, scaled to ~55% frame occupancy to
  match the seed sphere's projected extent.
- **Single-view depth degeneracy**: a silhouette loss cannot see depth — a
  paper-thin sheet has the same silhouette as a solid object, and smoothness
  regularizers actively push toward collapse. With < 3 views we enable two
  anti-flattening regularizers: a *volume hinge* (differentiable mesh volume
  must stay ≥ 40% of the seed volume) and *normal consistency loss*
  (folded thin sheets have adjacent faces with opposing normals).
  This is a fundamental ill-posedness, not a bug: the literature resolves it
  with symmetry priors (CMR), multi-view supervision, or learned priors
  (Zero123/TripoSR family).

### Tab 3 — Tokenizer (Phase 3: MeshGPT-style operator sequences)

**Claim: meshes compile to operator programs — the vocabulary for generative
models.**

Any mesh in the operator space decompiles into a token sequence
(`HDL`/`CC`/`CV`/`IE`/`DE`/`EOS`) and reconstructs from an icosahedron
template. The roundtrip test (`detokenize(tokenize(mesh))`) verifies genus
and manifold validity are preserved.

Why this matters for generation:
- Mainstream mesh generators (MeshGPT, PolyGen, MeshAnything) emit triangles
  token-by-token — like laying bricks one at a time; one wrong brick yields
  non-manifold output. An autoregressive model over *operator* tokens
  **cannot express an invalid mesh** — every prefix of every sequence is a
  valid manifold.
- Sequences are far more compact: one `CC` token quadruples face count;
  MeshGPT needs hundreds of tokens for the same resolution increase.
- The sequence *is* an editable construction history — parametric CAD
  semantics that triangle soup cannot offer.

### Tab 4 — Manifold Loss (Phase 4: differentiable topology constraints)

**Claim: topological validity can enter gradient-based training.**

"Is this mesh a valid manifold?" is naturally a discrete yes/no with no
gradient. This tab relaxes three topological criteria — Euler
characteristic, edge-manifoldness, orientability — into continuous,
differentiable losses over per-face existence probabilities (DMesh-style).
The probability sweep plot shows the losses varying smoothly with a rogue
face's existence probability, with usable gradients throughout.

This lets "topological validity" be written directly into a training loss,
pushing a generative model toward valid meshes *during learning* instead of
repairing its outputs afterwards.

---

## Honest Positioning (what is and is not ours)

**Not our invention:**
- TopMod/DLFL and the manifold-preserving operator set — Akleman & Chen
  (Texas A&M, 2003).
- "Generate shapes as operation programs" — validated in the CAD domain
  (DeepCAD, BrepGen).
- Differentiable rendering (N3MR/SoftRas/DIB-R/nvdiffrast) and mesh
  tokenization (PolyGen/MeshGPT).

**Our actual contribution (so far):** the *integration* — DLFL operators +
differentiable rendering + operator-sequence tokenizer + differentiable
manifold loss, wired into one pipeline. This combination is unexplored in
the subdivision-mesh domain. It is a promising research direction, **not yet
a proven breakthrough**: the demo is infrastructure; zero learning
experiments have been run.

**Evidence chain required before claiming a breakthrough:**
1. Train an operator-sequence generative model on a real dataset.
2. Head-to-head vs MeshGPT/MeshAnything: validity rate (ours should be 100%
   by construction), sequence length, generation quality.
3. Show the Tab 4 differentiable loss measurably improves training, not just
   that it is mathematically well-defined.
4. At least one killer demo competitors cannot do — e.g. genus-controlled
   generation.

Status analogy: the engine test bench is built and all four bench tests
pass; the aircraft has not flown yet.

---

## Known Gradio 6.12 Pitfalls (operational notes)

- **Never write outputs into a hidden tab from `demo.load`** — triggers a
  Svelte `effect_update_depth_exceeded` infinite loop that freezes the whole
  page on any tab switch (with ≥ 3 tabs). Tab 4 therefore has no auto-compute
  on load; the user clicks the button.
- **Model3D renders material-less OBJ as solid black** — export GLB with a
  PBR material instead (`_write_glb`).
- `--share` mode loads gradio.js from an AWS S3 CDN (blocked in some
  networks → blank page). Serve locally and expose via Tailscale Funnel with
  `--root-path /topmod`.

Serving: `python3 -u app.py --root-path /topmod` on port 7860, public at
`https://kingy.taile5f3af.ts.net/topmod/`.
