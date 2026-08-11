# Adaptive Remeshing During Differentiable Rendering Optimization

*August 10, 2026. Verified on Stanford Bunny.*

## Problem

When optimizing mesh vertex positions via differentiable rendering (nvdiffrast),
**thin line artifacts** appear. Root cause:

1. nvdiffrast's `antialias` only provides gradients at detected silhouette edges
   (adjacent pixels with different triangle IDs).
2. When a triangle degenerates to sub-pixel size, antialias cannot detect it →
   no gradient → the triangle becomes a "gradient dead zone."
3. Silhouette loss only constrains boundary vertices; interior vertices have
   near-zero gradient and drift into degenerate configurations.
4. Regularization (Laplacian, edge length) fights this but cannot eliminate it
   without destroying shape fidelity — it's a fundamental tradeoff.

This is a **known limitation of nvdiffrast** acknowledged in the original paper
(Laine et al., 2020).

## Solution: Continuous Adaptive Remeshing

Inspired by [Palfinger 2022, "Continuous Remeshing for Inverse Rendering"](https://www.researchgate.net/publication/362099386_Continuous_remeshing_for_inverse_rendering):
interleave gradient descent steps with remeshing operations that **collapse
degenerate triangles as they form**, rather than trying to prevent them with
loss terms.

### Algorithm

```
for phase in range(N_PHASES):           # e.g. 16 phases
    # ── Gradient descent (GPU) ──
    for step in range(STEPS_PER_PHASE):  # e.g. 50 steps
        loss = silhouette_loss + λ_lap * laplacian_loss
        loss.backward()
        optimizer.step()
        verts.clamp_(-bound, bound)

    # ── Adaptive remesh (CPU) ──
    verts_np = verts.detach().cpu().numpy()
    tris_np  = tris.cpu().numpy()
    verts_np, tris_np = adaptive_remesh(verts_np, tris_np)
    # Reinitialize optimizer with new vertex set
    verts = tensor(verts_np, requires_grad=True)
    tris  = tensor(tris_np)
    optimizer = Adam([verts], lr=lr)
```

Total steps = N_PHASES × STEPS_PER_PHASE (e.g. 16 × 50 = 800).

### Remeshing Operations

Each `adaptive_remesh()` call performs two passes:

#### 1. Edge Collapse (remove degeneracies)

```python
def collapse_short_edges(verts, tris, min_len):
    """
    Collapse edges shorter than min_len.
    min_len = 0.3 × mean_edge_length (adaptive threshold).

    For each short edge (v0, v1):
      - Merge v1 into v0 at midpoint position
      - Remove triangles that become degenerate (two vertices coincide)
      - Compact vertex array (remove unreferenced vertices)

    Process shortest edges first; skip edges where either endpoint
    was already involved in a collapse this round (greedy, non-overlapping).
    """
```

This directly kills thin lines: the degenerate triangle's short edge gets
collapsed, merging its vertices → the triangle disappears.

#### 2. Edge Split (add resolution where needed)

```python
def split_long_edges(verts, tris, max_len):
    """
    Split edges longer than max_len by inserting midpoint vertices.
    max_len = 2.0 × mean_edge_length (adaptive threshold).

    For each long edge (v0, v1):
      - Create new vertex at (v0 + v1) / 2
      - Split each triangle containing this edge into 2 triangles

    Limit splits per round to V/4 (longest first) to prevent
    vertex count explosion.
    """
```

This adds vertices where the mesh needs more detail (stretched regions),
improving shape fidelity without global subdivision.

### Thresholds

Both thresholds are **relative to the current mean edge length**, making them
self-adapting:

| Operation | Threshold | Rationale |
|-----------|-----------|-----------|
| Collapse  | edge_len < 0.3 × mean | Triangles with edges 3× shorter than average are likely degenerate |
| Split     | edge_len > 2.0 × mean | Triangles with edges 2× longer than average lack resolution |

These ratios (0.3, 2.0) worked well on the Bunny test. More aggressive collapse
(0.5×) removes more artifacts but risks losing valid thin features.

## Results: Stanford Bunny

Topology: `tetrahedron → DSBC → PENT2 → CHKB` (initial: 1162 verts, 2320 tris).
8-view silhouette supervision, 800 total optimization steps, Laplacian λ=0.5.

| Method | IoU | Verts | Tris | Thin lines? |
|--------|-----|-------|------|-------------|
| No remeshing | 0.989 | 1162 | 2320 | Yes (multiple) |
| **Adaptive remeshing** | **0.991** | **719** | **1269** | **Nearly none** |

Key observations:
- Remeshing **improved** IoU (0.991 > 0.989) despite using fewer vertices
- Vertex count naturally decreased from 1162 → 719 as degenerate triangles
  were collapsed — the mesh "self-cleaned"
- The remaining vertices concentrated at geometrically important locations
- Regularization weight could be kept low (λ=0.5) since remeshing handles
  degeneracies directly

## Vertex count evolution

```
Phase  4: 1162 → 880 verts (collapse removed degeneracies)
Phase  8: 880 → 786 verts
Phase 12: 786 → 741 verts
Phase 16: 741 → 719 verts (stabilized)
```

## Integration with TopMod Pipeline

The adaptive remeshing modifies mesh topology (adds/removes vertices and faces),
so the final mesh is **no longer strictly the output of the TopMod operator
sequence**. However:

1. The initial topology is still determined by TopMod operators (LLM prediction)
2. Remeshing preserves manifold property (edge collapse and split on manifold
   meshes produce manifold meshes)
3. The operator sequence remains interpretable and editable — remeshing is a
   post-processing step during the optimization phase only

The full pipeline becomes:

```
4 silhouettes → LLM → [TET DSBC PENT2 CHKB] → execute operators
    → initial mesh (1162 verts, manifold)
    → 16 rounds of (50 gradient steps + adaptive remesh)
    → final mesh (719 verts, manifold, no artifacts)
```

## Implementation Notes

- Remeshing runs on CPU (numpy), optimization on GPU (torch). The CPU↔GPU
  transfer every 50 steps adds ~1ms overhead per phase — negligible.
- The optimizer must be re-created after each remesh because the parameter
  tensor shape changes. Adam momentum is lost, but 50-step phases are short
  enough that this doesn't hurt convergence.
- `verts.clamp_(-2, 2)` after each step prevents vertices from escaping the
  scene bounds (another source of artifacts).
- Edge split is capped at V/4 splits per round to prevent vertex count
  explosion. In practice, splits are rare after the first few phases because
  the mesh quickly reaches sufficient resolution.

## References

- Palfinger, 2022. "Continuous remeshing for inverse rendering." Computer
  Animation and Virtual Worlds.
  https://onlinelibrary.wiley.com/doi/10.1002/cav.2101
- DMesh++, NeurIPS 2024. Face existence probability as alternative to
  explicit remeshing. https://arxiv.org/abs/2412.16776
- Laine et al., 2020. "Modular Primitives for High-Performance Differentiable
  Rendering." (nvdiffrast) — documents the sub-pixel triangle limitation.
  https://nvlabs.github.io/nvdiffrast/
