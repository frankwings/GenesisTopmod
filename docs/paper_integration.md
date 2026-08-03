# Paper Integration Plan — TopMod × Existing Methods

## Overview

We have a working TopMod Python library (Phase 1 complete). This document plans
how to integrate with 5 key papers in the AI mesh generation space, ordered by
strategic priority.

---

## Integration Map

```
                        TopMod (Ours)
                    4 operators, manifold guarantee
                            |
            ┌───────────────┼───────────────┐
            |               |               |
     [Token Space]    [Constraint]    [Data Structure]
            |               |               |
      ┌─────┴─────┐    ┌───┴───┐      ┌────┴────┐
      |           |    |       |      |         |
   MeshGPT    LATO.2  DMesh  TSSR  SpaceMesh
   (CVPR24)  (2026)  (NeurIPS24)(2025)(SIGAsia24)
```

Three integration paradigms:
- **Token Space**: TopMod operators AS the token vocabulary
- **Constraint**: TopMod invariants as loss/regularization during training
- **Data Structure**: TopMod's DLFL as the native mesh representation

---

## Priority 1: MeshGPT Integration (Highest Impact)

### Why First
- Autoregressive mesh generation is the hottest direction (MeshGPT→MeshAnything→DeepMesh→Edgerunner)
- TopMod operators are a natural token vocabulary with mathematical completeness guarantee
- Directly addresses MeshGPT's biggest weakness: no topological guarantee

### MeshGPT's Current Approach
```
Triangle face → Graph Conv → Residual VQ (6 codebook indices per face) → Token
Transformer autoregressively predicts next token → Decode → Mesh
```
Problem: tokens encode geometry (vertex coordinates), topology is implicit.
Result: non-manifold artifacts, wrong genus, self-intersections.

### Our Integration: TopMod-Tokenized MeshGPT
```
Phase 1 — Data Preparation:
    Training meshes (ShapeNet/Objaverse)
        → Analyze each mesh: extract genus, boundaries, components
        → Decompose into TopMod operator sequence:
            [CV(x,y,z), CV(x,y,z), ..., IE(f1,f2), IE(f3,f4), ..., DE(e1), ...]
        → This is the "TopMod tokenization" (reverse engineering)

Phase 2 — Token Vocabulary:
    Token types:
        CV(x,y,z)     = CreateVertex with quantized coordinates
        DV(v_id)      = DeleteVertex
        IE(c1,c2)     = InsertEdge between two corners
        DE(e_id)      = DeleteEdge
        EXT(f,d)      = ExtrudeFace (high-level macro = multiple IE)
        HDL(f1,f2)    = AddHandle (high-level macro)
        CC()          = CatmullClark subdivision
        EOS           = End of sequence

    Residual VQ not needed — operators ARE the vocabulary.

Phase 3 — Transformer Training:
    Architecture: GPT-2 style decoder-only transformer
    Input: [condition_embedding, op_1, op_2, ..., op_t]
    Output: predict op_{t+1}
    Loss: cross-entropy on operator type + MSE on continuous params (coords)

Phase 4 — Inference:
    Condition (image/text/class) → Autoregressive generation of operator sequence
        → Execute each operator on DLFLMesh → Final mesh
    Guarantee: every intermediate state is valid 2-manifold
```

### Key Research Challenge
**Reverse decomposition**: Given an arbitrary mesh, find the optimal TopMod operator
sequence that reconstructs it. This is non-trivial:
- Not unique (many sequences produce the same mesh)
- Need shortest/most natural sequence for efficient training
- Possible approaches:
  (a) Greedy: start from empty, iteratively add vertices/edges to match target
  (b) Template-based: classify genus → start from template → refine
  (c) Learn the decomposition itself (meta-learning)

### Deliverables
- [ ] TopMod tokenizer: mesh → operator sequence
- [ ] TopMod detokenizer: operator sequence → mesh (already done — our library)
- [ ] Training pipeline on ShapeNet subset
- [ ] Comparison: TopMod tokens vs MeshGPT tokens (manifold rate, FID, coverage)

### Paper Angle
"Topology-Complete Mesh Tokenization for Autoregressive Generation"
- Contribution 1: TopMod operator vocabulary (first provably complete token set for meshes)
- Contribution 2: Reverse decomposition algorithm
- Contribution 3: Manifold guarantee by construction (0% non-manifold vs MeshGPT's ~X%)
- Target: SIGGRAPH / SIGGRAPH Asia / Eurographics SGP

---

## Priority 2: LATO.2 Integration (Complementary)

### Why Second
- LATO.2 already separates vertex flow (V-Flow) and topology flow (T-Flow)
- Our TopMod operators can directly replace or constrain T-Flow
- Most natural architectural fit

### LATO.2's Current Approach
```
Mesh → VAE encoder → latent (vertex latent + topology latent)
V-Flow: flow matching on vertex positions
T-Flow: flow matching on connectivity (conditioned on realized vertices)
```
Problem: T-Flow operates in continuous latent space — decoded connectivity may be invalid.

### Our Integration: TopMod-Constrained T-Flow
```
Option A — Replace T-Flow with TopMod sequence generation:
    V-Flow generates vertex positions (unchanged)
        → Given vertices, predict TopMod operator sequence to connect them
        → Every step maintains manifold

Option B — TopMod manifold loss for T-Flow:
    T-Flow generates connectivity in latent space (unchanged)
        → Decode to mesh
        → Compute manifold violation loss using our validate.py
        → L_total = L_flow + λ * L_manifold
        → Gradients push T-Flow toward manifold-valid outputs

Option C — TopMod latent space for T-Flow VAE:
    Instead of generic VAE for topology, train VAE where:
        → Encoder: mesh connectivity → TopMod operator sequence → latent
        → Decoder: latent → TopMod operator sequence → mesh connectivity
    This ensures the latent space only represents valid 2-manifolds.
```

### Deliverables
- [ ] Implement manifold violation loss (differentiable version of validate.py)
- [ ] Benchmark: LATO.2 baseline vs LATO.2 + TopMod constraint
- [ ] If Option A: TopMod sequence generator conditioned on vertex positions

### Paper Angle
"Manifold-Constrained Topology Flow for 3D Mesh Generation"
- Builds on LATO.2's vertex/topology separation
- Contribution: provably manifold topology flow via TopMod operators

---

## Priority 3: DMesh Integration (Theoretical)

### Why Third
- DMesh is the closest theoretically (both about differentiable topology)
- But DMesh is not mainstream — limited practical impact
- Good for a theory-focused paper

### DMesh's Current Approach
```
Points (position + weight + ψ) → WDT → Face existence probability Λ(F)
    → Gradient optimization of point attributes
    → Topology changes via probability shifts (Λ: 0→1 or 1→0)
```
Problem: faces are independently probabilistic → can produce invalid combinations
(face A exists but neighbor B doesn't → non-manifold edge).

### Our Integration: TopMod Invariant as DMesh Constraint
```
At each optimization step:
    1. DMesh computes Λ(F) for all faces
    2. Threshold to get binary face set
    3. Check manifold invariants (our validate.py)
    4. Compute L_manifold = count of violations (differentiable relaxation)
    5. L_total = L_chamfer + λ * L_manifold

Differentiable manifold violation:
    - Edge manifold: for each edge, penalize |sum(Λ(adjacent_faces)) - 2|
      (each edge should have exactly 2 adjacent faces with high probability)
    - Euler penalty: |V_eff - E_eff + F_eff - (2 - 2g_target)|²
      where V_eff, E_eff, F_eff are probability-weighted counts
```

### Deliverables
- [ ] Differentiable manifold violation loss
- [ ] Plug into DMesh codebase (PyTorch)
- [ ] Benchmark: DMesh vs DMesh + TopMod constraint (non-manifold rate)

### Paper Angle
"Manifold-Guaranteed Differentiable Mesh via Topological Invariants"

---

## Priority 4: SpaceMesh Integration (Data Structure Level)

### Why Fourth
- SpaceMesh already uses halfedge mesh (closely related to DLFL)
- Integration is at the data structure level — deep but less novel
- More of an engineering contribution than research contribution

### SpaceMesh's Current Approach
```
Per-vertex continuous embedding → Halfedge connectivity → Manifold mesh
Guarantee: halfedge construction ensures edge-manifoldness by design
```

### Our Integration: Extend SpaceMesh with TopMod High-Level Ops
```
SpaceMesh gives: base manifold mesh from learned embeddings
TopMod adds: structured topology editing on top
    → add_handle(f1, f2): genus modification
    → extrude_face(f, d): local geometry modification
    → catmull_clark(): resolution increase

Use case: SpaceMesh generates coarse mesh → TopMod refines topology
```

### Deliverables
- [ ] Adapter: SpaceMesh halfedge ↔ our DLFL conversion
- [ ] Pipeline: SpaceMesh generation → TopMod refinement
- [ ] Comparison: SpaceMesh alone vs SpaceMesh + TopMod refinement

---

## Priority 5: TSSR Integration (Diffusion)

### Why Last
- TSSR uses discrete diffusion — less natural fit for TopMod
- But TopMod operators could define valid transition kernels

### TSSR's Current Approach
```
Mesh tokens → Discrete diffusion (noising/denoising)
Two stages: topology sculpting → shape refinement
```

### Our Integration: TopMod Transition Kernel
```
Instead of generic discrete diffusion on mesh tokens:
    → Define diffusion transitions as TopMod operations
    → Forward process: random InsertEdge/DeleteEdge (controlled noise)
    → Reverse process: learned denoising predicts which TopMod op to undo
    → Every state in the diffusion chain is a valid 2-manifold
```

### Deliverables
- [ ] TopMod-based discrete diffusion kernel
- [ ] Comparison: TSSR baseline vs TopMod-transition TSSR

---

## Implementation Roadmap

```
Phase 1 (DONE): TopMod Python library
    ✅ DLFL + 4 operators + high-level ops + CC subdivision
    ✅ 129 tests, interactive UI

Phase 2 (NEXT): Plan A demo
    → Topology construction + nvdiffrast geometry optimization
    → Demo: genus-controlled mesh from target image
    → Timeline: 1-2 weeks

Phase 3: MeshGPT integration (Priority 1)
    → TopMod tokenizer (reverse decomposition)
    → Training pipeline
    → Timeline: 3-4 weeks

Phase 4: LATO.2 integration (Priority 2)
    → Manifold constraint loss
    → Timeline: 2-3 weeks

Phase 5: Paper writing
    → Experiments, ablations, comparisons
    → Timeline: 2-3 weeks
```

---

## Evaluation Metrics

| Metric | What it measures | How |
|--------|-----------------|-----|
| **Manifold Rate** | % of generated meshes that are valid 2-manifold | Our validate.py |
| **Genus Accuracy** | Does generated mesh have correct genus? | Euler formula check |
| **FID** | Visual quality of generated shapes | Render → FID score |
| **Coverage** | Diversity of generated shapes | Standard metric |
| **Chamfer Distance** | Geometric accuracy vs target | Point-to-point distance |
| **Token Efficiency** | Sequence length for same mesh | Fewer tokens = better |
| **Inference Speed** | Time to generate one mesh | Wall clock |

Key comparison axes:
- Ours (TopMod tokens) vs MeshGPT (coordinate tokens) vs LATO.2 (flow matching)
- Focus on manifold rate and genus accuracy — our unique advantage

---

## References

1. MeshGPT (CVPR 2024) — Siddiqui et al. "Generating Triangle Meshes with Decoder-Only Transformers"
2. LATO.2 (2026) — "Factorized 3D Mesh Generation with Vertex and Topology Flow"
3. DMesh (NeurIPS 2024) — Son et al. "A Differentiable Mesh Representation"
4. SpaceMesh (SIGGRAPH Asia 2024) — "A Continuous Representation for Learning Manifold Surface Meshes"
5. TSSR (2025) — "Topology Sculptor, Shape Refiner: Discrete Diffusion Model"
6. MeshAnything (ICLR 2025) — "Artist-Created Mesh Generation with Autoregressive Transformers"
7. DeepMesh (2025) — Encoder-Hourglass Transformer + RL(DPO) alignment
8. Akleman & Chen (2003) — "A minimal and complete set of operators for robust manifold mesh modelers"
