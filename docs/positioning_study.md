# Positioning Study — OpSeq vs. the Field

*Recorded 2026-08-08. Survey of where our LLM-token → operator approach ("OpSeq")
sits in the literature, what is borrowed, what is ours, and how it combines with
the differentiable operators (`topmod/diffgeo.py`).*

---

## 1. What the algorithm is called

Our approach is an instance of **neural shape program synthesis**: an
image-conditioned autoregressive transformer emits a *program* whose
instructions are DLFL manifold operators. Sibling work: CSGNet (CVPR'18),
Tian et al. *Learning to Infer and Execute 3D Shape Programs* (ICLR'19),
ShapeAssembly. Internal codename: **OpSeq** (`experiments/opseq`,
`experiments/opseq_b`).

With the differentiable executor added, the combined paradigm is
**propose-and-optimize** (neural-guided program synthesis): the LLM proposes
discrete structure, gradient descent solves continuous parameters.

## 2. Provenance of the Phase A/B/C pipeline

The scaffold is an established paradigm — deliberately so. The novelty is the
instruction set, not the training recipe.

| Phase | What we do | Prior art |
|---|---|---|
| A | Sample random programs → execute → render → train | Synthetic-program pretraining (CSGNet; Tian et al.) |
| B | Fit real meshes by optimization → recover program pseudo-labels → finetune | Pseudo-label bootstrapping (PLAD, CVPR'22); wake-sleep (Hinton '95); STaR in LLM land |
| C | Model proposes program, differentiable executor refines geometry at test time | Neural-guided synthesis + test-time optimization (common in CSG/CAD reconstruction) |

What is genuinely ours:

1. **The instruction set**: first use of DLFL manifold operators as a program
   vocabulary. Consequence: every executable program yields a 2-manifold; genus
   is an explicit, countable token (HDL).
2. **Exact tokenization loop**: Phase B fitting happens *inside* operator space
   (same-genus seed + vertex optimization), so recovered pseudo-labels are
   exact, not approximate as in CSGNet/PLAD.
3. **The differentiable-executor interface**: subdivision operators are linear
   in vertex positions, so the whole geometry chain is a sparse-matrix product —
   a clean differentiable structure CSG booleans do not have.

## 3. MeshGPT comparison (verified against CVPR 2024 paper)

MeshGPT: two stages. (1) Learn a face vocabulary — SAGEConv graph encoder over
faces (features: vertex coords, normals, edge angles, area), residual vector
quantization depth D=6 → 6 codebook tokens per triangle, 1D conv decoder
reconstructs the 9 face coordinates. (2) Decoder-only GPT autoregressively
predicts the token sequence (faces sorted spatially); decoded output is a
triangle set.

| | MeshGPT | OpSeq (ours) |
|---|---|---|
| Token meaning | learned latent code (opaque) | hand-defined operator (executable instruction) |
| Token granularity | 6 tokens/face → ~4800 for 800 faces | one op can rewrite the whole mesh (one CC = global subdivision) |
| Output | triangle soup, no manifold guarantee (~98% learned) | program execution, 100% manifold by construction |
| Genus | implicit, luck | explicit HDL tokens |
| Stage-1 training | required (codebook is learned) | none (vocabulary is defined) |
| Conditioning | none / class label | 4-view silhouettes |

Shared pain point: geometric detail is expensive in discrete tokens (their 6
tokens/face ↔ our ~4000 CV tokens). They mitigate with RQ; we eliminate it via
the differentiable control cage (geometry goes to gradients, tokens keep only
structure).

## 4. Manifold-guarantee landscape (we are NOT unique on this axis)

Web-verified survey, three families that already guarantee manifold output:

| Family | Representative | Guarantee mechanism | Limitation |
|---|---|---|---|
| Implicit + iso-surface | NeuManifold (WACV'25); SDF + Marching Cubes | watertight extraction from a field | dense unstructured output; no editability; genus implicit |
| Template deformation | Neural Mesh Flow (NeurIPS'20) | diffeomorphic flow preserves the sphere's manifoldness | locked to genus 0 |
| Continuous connectivity embedding | **SpaceMesh** (NVIDIA, SIGGRAPH Asia 2024) | per-vertex embeddings decode to a halfedge mesh, manifold **by construction** | output is a mesh with no *process* — not readable, not editable, not replayable |

**SpaceMesh is the closest competitor** and must be cited head-on: it makes the
same "halfedge → manifold by construction" claim. The difference is the *form*
of the representation: their manifoldness lives in an opaque embedding space;
ours lives in an interpretable operator program.

## 5. Honest positioning (supersedes earlier "fundamental breakthrough" phrasing)

> Our unique claim is **not** "the only method guaranteeing manifold output"
> (false — see §4). It is: **the only method combining the manifold guarantee
> with topology-as-program** — manifoldness, interpretability, editability,
> explicit genus control, and a differentiable-refinement interface in one
> representation. We occupy the "programmatic" quadrant of the converging
> manifold-aware-generation space.

## 6. Combining the LLM path with differentiable operators

The operator vocabulary splits naturally:

| Kind | Ops | Differentiable? |
|---|---|---|
| Discrete topology | HDL, IE, DE, choice of subdivision scheme | no (changes connectivity) |
| Continuous geometry | CV coords; all subdivision/remeshing schemes | yes — linear in vertex positions once topology is fixed |

This line is exactly where the two algorithms meet:

```
final_verts = S_k · … · S_1 · V_cage      ← differentiable chain (sparse mm)
              ↑ discrete (LLM)        ↑ continuous (gradient)
```

Integration at all three stages:

1. **Distillation**: search discrete skeletons; for each candidate, gradient-fit
   the control cage through the differentiable chain; keep the shortest
   best-fitting program. Sequences shrink ~4000 → ~100 tokens, killing the
   coordinate error accumulation observed in Phase B.
2. **Training**: soft-argmax dequantization of COORD logits → differentiable
   chain → render loss backpropagated into the token logits — a second,
   geometric supervision signal beyond cross-entropy. (The model finally "sees"
   its own mesh during training.)
3. **Inference (propose-and-optimize)**: LLM emits skeleton + coarse cage;
   executor refines the cage against the conditioning silhouettes for a few
   dozen steps. The LLM only needs to be roughly right.
4. **Bootstrap loop (optional)**: re-tokenize refined results into the training
   set and finetune again (wake-sleep / STaR). Phase B's finetune-beats-scratch
   result is the first turn of this loop.

**Status**: the differentiable executor is DONE — `topmod/diffgeo.py`
(commit 39e72b3): 17 linear ops via symbolic trace into sparse weight matrices
(float implementation remains the single source of truth; nonlinear use raises
during trace) + dedicated torch CRUST; `DiffSequence` API matches the interface
contract above; 64/64 tests pass incl. `gradcheck` on all ops. Remaining:
STAR/FRAC/DOME (need custom torch normal code), and per-parameter
differentiability (dual-number trace, planned). See `docs/diffgeo.md`.

## 7. Known evaluation caveat (must fix before quoting numbers)

Phase A/B silhouette-IoU numbers are **inflated by an image-polarity bug**:
dataset silhouettes are white-background (bg=255 ≈ 91% of pixels), and the
`>0.5` binarization measures *background* overlap (a degenerate tiny-triangle
sample scored "0.908" ≈ exactly the background fraction). Manifold-validity and
genus-accuracy metrics are unaffected. Foreground-IoU re-evaluation is required
before any external claim.

## References

- MeshGPT, CVPR 2024 — https://arxiv.org/abs/2311.15475
- SpaceMesh, SIGGRAPH Asia 2024 — https://arxiv.org/abs/2409.20562
- NeuManifold, WACV 2025 — https://arxiv.org/abs/2305.17134
- Neural Mesh Flow, NeurIPS 2020 — https://proceedings.neurips.cc/paper/2020/file/1349b36b01e0e804a6c2909a6d0ec72a-Paper.pdf
- CSGNet, CVPR 2018 — https://arxiv.org/abs/1712.08290
- Learning to Infer and Execute 3D Shape Programs, ICLR 2019 — https://arxiv.org/abs/1901.02875
- PLAD, CVPR 2022 — https://arxiv.org/abs/2011.13045
