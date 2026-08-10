# GenesisTopmod — Summary Report: Neural Shape Program Synthesis with DLFL Operators

*August 2026*

---

## 1. What We Built

A **neural shape program synthesis** system ("OpSeq") that generates 3D meshes as executable programs of DLFL manifold operators — the first system to use Akleman & Chen's Doubly-Linked Face List (DLFL) operators as a machine learning vocabulary.

### Core Components

1. **TopMod Python Library** (`topmod/`, 2,600 lines, zero dependencies)
   - Pure-Python half-edge (DLFL) mesh data structure
   - 29 operators: 17 linear subdivision/remeshing, 7 nonlinear, 3 topology-only, 2 local
   - All 29 operators have PyTorch differentiable implementations (`diffgeo.py`, 84/84 tests passing)
   - 100% manifold guarantee by construction — any valid operator sequence produces a closed orientable 2-manifold

2. **OpSeq Model** (6.9M parameters)
   - 4-channel silhouette CNN encoder → 64 spatial memory tokens
   - 6-layer Pre-LN Transformer decoder (d=256, nhead=8)
   - Autoregressive token prediction over 356-token vocabulary
   - Conditioned on 4-view silhouette images

3. **Differentiable Rendering Pipeline**
   - nvdiffrast-based silhouette fitting
   - Propose-and-optimize: LLM predicts topology → differentiable rendering optimizes vertex positions
   - Supports silhouette, depth, and normal supervision signals

4. **Blender Add-on** (21 operators exposed in Blender's sidebar)
   - Pure-Python, directly embeds the topmod library
   - Genus-N primitives, add-handle, all subdivision schemes, manifold HUD

---

## 2. Experiments and Key Results

### Phase A: Synthetic Pretraining (6-operator vocabulary)
- Random DLFL programs → execute → render → train
- **Result**: Manifold validity 100%, Genus accuracy 59%, **Foreground IoU 0.013**
- **Diagnosis**: ~4000 CV coordinate tokens per sequence → accumulated quantization error → geometry is random

### Phase B: Thingi10K Distillation
- Real meshes → same-genus DLFL seed → 8-view differentiable fitting → exact tokenization → finetune
- **Result**: Manifold 100%, Genus 62%, **Foreground IoU 0.181**, Finetune beats scratch (1.2720 vs 1.2929)
- **Diagnosis**: Still too many coordinate tokens; distillation quality is good (IoU 0.936) but model can't reproduce

### Phase A': Propose-and-Optimize (29-operator vocabulary)
- Short topology programs (~50 tokens) + coarse cage coordinates
- Post-generation differentiable refinement via nvdiffrast
- **Result**: Manifold 100%, Genus 66.8%, **Pre-refine IoU 0.836**, Post-refine IoU 0.743

### Critical Ablation: What Actually Matters?

| Condition | Foreground IoU |
|-----------|---------------|
| A) Cage + differentiable subdivision chain | 0.962 |
| **B) Direct all-vertex optimization (GT topology)** | **0.974** |
| C) Plain cube, no topology | 0.947 |

**By subdivision depth:**

| Depth | A (cage+chain) | B (direct) | C (plain cube) |
|-------|---------------|------------|-----------------|
| 0 | 0.971 | 0.969 | 0.977 |
| 1 | 0.971 | 0.990 | 0.942 |
| 2 | 0.920 | 0.967 | 0.898 |
| 3 | 0.978 | 0.991 | 0.894 |

**Conclusions from ablation:**
1. Topology selection matters (B > C at depth >= 2, gap up to 10 points)
2. Differentiable subdivision chain provides no IoU benefit (B >= A everywhere)
3. Optimal pipeline: LLM selects topology → float-execute operators → directly optimize all vertices

---

## 3. Competitive Landscape

### Manifold-Guarantee Methods

| Method | Venue | Guarantee Mechanism | Limitations |
|--------|-------|-------------------|-------------|
| NeuManifold | WACV 2025 | SDF + Differentiable Marching Cubes | Dense unstructured output, no editability |
| Neural Mesh Flow | NeurIPS 2020 | Diffeomorphic flow from sphere | Locked to genus 0 |
| **SpaceMesh** | SIGGRAPH Asia 2024 | Per-vertex continuous embeddings → halfedge mesh | Manifold by construction but opaque, not editable |
| DMesh | NeurIPS 2024 | Weighted Delaunay Triangulation, face probability differentiable | Topology also gradient-optimized, no manifold guarantee |
| **Ours (OpSeq)** | — | DLFL operator execution | Manifold by construction + interpretable program |

### Mesh Generation Methods

| Method | Venue | Approach | Output |
|--------|-------|----------|--------|
| MeshGPT | CVPR 2024 | VQ-VAE face codebook + GPT | Triangle soup, ~98% manifold |
| MeshAnythingV2 | 2025 | Point cloud → autoregressive mesh | Artist-like mesh |
| VIGA | arXiv 2026 | VLM agent + Blender code-render-inspect loop | Editable Blender scene, zero-training |
| CSGNet | CVPR 2018 | Neural program synthesis with CSG primitives | CSG tree, no manifold guarantee |
| PLAD | CVPR 2022 | Pseudo-label bootstrapping for program synthesis | Program, approximate pseudo-labels |
| **Ours (OpSeq)** | — | Neural program synthesis with DLFL operators | Executable manifold program |

### Pipeline Lineage (What We Borrowed vs. Invented)

| Phase | Our Implementation | Prior Art |
|-------|-------------------|-----------|
| A: Synthetic pretraining | Random DLFL programs | CSGNet (CVPR'18), Tian et al. (ICLR'19) |
| B: Fitting distillation | Same-genus seed + diff fitting → exact pseudo-labels | PLAD (CVPR'22), wake-sleep (Hinton '95) |
| C: Propose-and-optimize | LLM topology + diff rendering vertex optimization | Neural-guided synthesis (common pattern) |

**What is genuinely ours:**
1. First use of DLFL manifold operators as a program vocabulary
2. Exact tokenization loop (fitting inside operator space → lossless pseudo-labels)
3. 29-operator differentiable library with symbolic trace + dedicated torch implementations

---

## 4. Honest Assessment of TopMod's Advantages

| Claimed Advantage | Status | Evidence |
|-------------------|--------|----------|
| 100% manifold by construction | True but not unique | SpaceMesh, NeuManifold also guarantee this |
| Explicit genus control (HDL tokens) | **Unique** | No other method has countable topology tokens |
| Interpretable/editable program output | **Unique** | Output is `CUBE→HDL→CC`, human-readable, modifiable |
| Differentiable subdivision chain improves optimization | **False** | Ablation: direct vertex optimization (0.974) > cage+chain (0.962) |
| Topology selection improves shape fitting | **True** | Ablation: GT topology (0.974) > plain cube (0.947) at depth >= 2 |
| Composable operator algebra | **Unique** | 29 ops combinatorially produce diverse topologies; template methods cannot |

**Core positioning:**
> Our unique contribution is not "the only method guaranteeing manifold output" (false).
> It is: **the only method that outputs an interpretable, editable, executable topology program
> with manifold guarantee and explicit genus control.**
> We occupy the "programmatic" quadrant of the manifold-aware generation space.

---

## 5. Future Directions

### Direction 1: Blender Plugin (Product)
- **Status**: v1 add-on implemented (21 operators in Blender sidebar)
- **Value**: Interactive topology modeling → bridge to generative pipeline
- **Next**: Operator-history panel (= live token sequence), manifold validity HUD, undo/redo support
- Pure-Python `topmod/` library vendors directly into `bpy` add-on, cross-platform

### Direction 2: LLM Training (Research)
- **Immediate**: Remove COORD tokens entirely — LLM predicts only topology (~5-10 tokens), differentiable rendering handles all geometry
- **Architecture**: VIGA-inspired closed-loop — LLM proposes topology → diff rendering fits → render result fed back → LLM revises topology
- **Data**: Extend to Thingi10K/ShapeNet with topology search (find shortest operator program per target)
- **Supervision**: Add depth maps and normal maps beyond silhouettes (nvdiffrast supports all natively)
- **Longer term**: Larger vocabulary exploitation (29 ops → combinatorial diversity), real-image conditioning (replace synthetic silhouettes)

---

## 6. References

1. Akleman, E. & Chen, J. (2003). *Guaranteeing the 2-manifold property for meshes with doubly linked face list.* International Journal of Shape Modeling.
2. MeshGPT — Siddiqui et al., CVPR 2024. https://arxiv.org/abs/2311.15475
3. SpaceMesh — Sharp et al., SIGGRAPH Asia 2024. https://arxiv.org/abs/2409.20562
4. DMesh — Son et al., NeurIPS 2024. https://arxiv.org/abs/2404.13445
5. NeuManifold — Wei et al., WACV 2025. https://arxiv.org/abs/2305.17134
6. Neural Mesh Flow — Gupta et al., NeurIPS 2020.
7. VIGA — Yang et al., arXiv 2026. https://arxiv.org/abs/2601.11109
8. TopoGen — Hu et al., CGF 2025.
9. CSGNet — Sharma et al., CVPR 2018. https://arxiv.org/abs/1712.08290
10. Learning to Infer and Execute 3D Shape Programs — Tian et al., ICLR 2019. https://arxiv.org/abs/1901.02875
11. PLAD — Jones et al., CVPR 2022. https://arxiv.org/abs/2011.13045
