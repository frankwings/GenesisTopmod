# Slide Deck: OpSeq — Neural Shape Program Synthesis with DLFL Operators

*Presentation outline. Each ## = one slide.*

---

## Slide 1: Title

**OpSeq: Neural Shape Program Synthesis with DLFL Manifold Operators**

- Image → Executable Topology Program → Manifold Mesh
- Key visual: 4 silhouettes → [CUBE → HDL → CC] → 3D mesh

---

## Slide 2: The Problem

**How do neural networks generate 3D meshes?**

| Approach | Output | Manifold? | Editable? | Topology Control? |
|----------|--------|-----------|-----------|-------------------|
| NeRF / SDF + Marching Cubes | Dense triangle soup | Yes (watertight) | No | No |
| Template deformation (NMF) | Deformed sphere | Yes | No | Genus 0 only |
| MeshGPT (CVPR'24) | Triangle sequence | ~98% | No | No |
| SpaceMesh (SIGGRAPH Asia'24) | Halfedge mesh | Yes | No | Implicit |
| **Ours: OpSeq** | **Executable program** | **Yes (100%)** | **Yes** | **Explicit (HDL)** |

---

## Slide 3: Key Idea — Mesh as Program

**Instead of generating vertices and faces, generate a program that builds the mesh.**

```
Input: 4 silhouette images
  ↓
LLM (6.9M params): [CUBE] [HDL] [CC] [EOS]
  ↓
Execute DLFL operators → manifold mesh (guaranteed)
  ↓
Differentiable rendering → optimize vertex positions
  ↓
Output: manifold mesh with correct topology
```

- Program is short (~5-10 tokens) vs MeshGPT (~4800 tokens)
- Program is human-readable, editable, replayable

---

## Slide 4: DLFL Operators — The Vocabulary

**29 operators from Akleman & Chen's DLFL formulation**

| Category | Operators | Function |
|----------|-----------|----------|
| Topology | HDL (add_handle) | Add a hole (genus +1) |
| Subdivision (17) | CC, DS, Loop, sqrt3, honeycomb, star, ... | Increase mesh resolution + define connectivity |
| Shell/Crust | CRUST | Create double-walled shell |
| Local | Extrude, Stellate, Subdivide_edge/face | Local modifications |

- Every valid operator sequence → guaranteed 2-manifold
- Operators define both topology AND vertex count

---

## Slide 5: What We Built

**Four components:**

1. **TopMod Python Library** — 2,600 lines, pure Python, 29 operators, zero deps
2. **Differentiable Operator Layer** — All 29 ops in PyTorch (symbolic trace → sparse matrices)
3. **OpSeq Model** — CNN + 6-layer Transformer, 356-token vocabulary, 6.9M params
4. **Blender Add-on** — 21 operators in Blender sidebar, manifold HUD

---

## Slide 6: Training Pipeline (3 Phases)

| Phase | Method | Prior Art |
|-------|--------|-----------|
| **A** | Random programs → render → train | CSGNet (CVPR'18) |
| **B** | Real mesh → fit DLFL seed → pseudo-label → finetune | PLAD (CVPR'22) |
| **A'** | Short programs (29 ops) + diff rendering refinement | Propose-and-optimize |

- Phase A/B: proved manifold guarantee holds (100%) and genus is learnable (59-62%)
- Phase A': reduced sequence from ~4000 to ~50 tokens, IoU from 0.013 to 0.836

---

## Slide 7: Phase A' Results

| Metric | Phase A (old) | Phase A' (new) |
|--------|--------------|----------------|
| Vocabulary | 6 ops, 264 tokens | 29 ops, 356 tokens |
| Sequence length | ~4000 | ~50 |
| Manifold validity | 100% | 100% |
| Foreground IoU | 0.013 | **0.836** |
| Genus accuracy | 59% | **66.8%** |

- 64× improvement in shape matching
- Short sequences eliminate coordinate error accumulation

---

## Slide 8: Critical Ablation — What Actually Matters?

**3-way comparison on 200 samples (GT topology given):**

| Condition | IoU | What it tests |
|-----------|-----|---------------|
| A) Cage + diff subdivision chain | 0.962 | Does diff chain help? |
| **B) Direct vertex optimization** | **0.974** | Just optimize all vertices |
| C) Plain cube (no topology) | 0.947 | Does topology matter? |

**Findings:**
- B > C: **Topology selection matters** (up to 10 points at depth 3)
- B > A: **Differentiable chain does NOT help** — direct optimization is better
- **Conclusion**: DLFL value = topology selection, not differentiable chain

---

## Slide 9: Depth Breakdown — Topology Matters More for Complex Shapes

| Subdivision Depth | With DLFL Topology | Plain Cube | Gap |
|-------------------|--------------------|------------|-----|
| 0 | 0.969 | 0.977 | -0.8% |
| 1 | 0.990 | 0.942 | +4.8% |
| 2 | 0.967 | 0.898 | **+6.9%** |
| 3 | 0.991 | 0.894 | **+9.7%** |

- Simple shapes: cube is enough
- Complex shapes: **DLFL topology provides increasingly larger advantage**

---

## Slide 10: Competitive Landscape

| Method | Manifold | Editable | Genus Control | Programmatic |
|--------|----------|----------|---------------|-------------|
| NeuManifold (WACV'25) | Yes | No | No | No |
| Neural Mesh Flow (NeurIPS'20) | Yes | No | Genus 0 only | No |
| SpaceMesh (SIGGRAPH Asia'24) | Yes | No | Implicit | No |
| DMesh (NeurIPS'24) | No | No | Gradient-optimized | No |
| MeshGPT (CVPR'24) | ~98% | No | No | No |
| VIGA (arXiv'26) | N/A | Yes | No | Yes (Blender Python) |
| **Ours** | **100%** | **Yes** | **Explicit (HDL)** | **Yes (DLFL ops)** |

---

## Slide 11: Our Unique Position

> **We are the only method combining manifold guarantee + interpretable program output + explicit genus control.**

Three things no competitor has:
1. **Countable topology**: HDL token = one hole. Period.
2. **Editable output**: Change `CC` to `DS` → different subdivision style. Add `HDL` → add a hole.
3. **Composable algebra**: 29 ops → combinatorial explosion of topologies

---

## Slide 12: Future Direction 1 — Blender Plugin (Product)

**Interactive DLFL topology modeling inside Blender**

- v1 implemented: 21 operators in sidebar panel
- Pure Python — vendors directly into Blender's bundled interpreter
- Natural features:
  - Genus-N primitive creation
  - Add handle between selected faces
  - All subdivision schemes
  - Manifold validity HUD (Euler characteristic, genus)
  - Operator history panel (= live token sequence)
- Bridge: Blender ↔ generative pipeline (export topology program, import generated mesh)

---

## Slide 13: Future Direction 2 — LLM Training (Research)

**Next-generation pipeline: topology-only prediction + closed-loop refinement**

```
Silhouettes → LLM → [CUBE HDL CC] (5-10 tokens, topology only)
                          ↓
              Execute operators → mesh topology
                          ↓
              Diff rendering → optimize all vertices (500 steps, 2s)
                          ↓
              Render result → compare with target
                          ↓
              Feed discrepancy back to LLM → revise topology
                          ↓
              (VIGA-inspired closed loop)
```

Research questions:
- Topology-only vocabulary (remove all COORD tokens)
- VIGA-style code-render-inspect loop for topology search
- Depth/normal supervision beyond silhouettes
- Scale to real images (not just synthetic silhouettes)

---

## Slide 14: Summary

| What We Achieved | |
|---|---|
| 29 DLFL operators in Python | Clean-room, ML-ready, fully tested |
| 100% manifold generation | Structural guarantee, not learned |
| 0.013 → 0.836 IoU improvement | Phase A → A' via propose-and-optimize |
| Topology matters (+9.7% at depth 3) | Ablation-verified |
| Blender add-on | 21 operators, ready to use |

| What We Learned | |
|---|---|
| Manifold guarantee is not unique | SpaceMesh, NeuManifold also guarantee it |
| Diff subdivision chain doesn't help IoU | Direct vertex optimization is better |
| Our unique value = programmatic output | Editable, interpretable, composable |

**The right framing: not "better 3D reconstruction" but "image → editable topology program"**

---
