# OpSeq Phase B — Evaluation Results

**Checkpoint**: `/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_b/ckpt/best.pt`

**Run type**: finetune

**Epoch**: 49  |  **Best val loss**: 1.2720

**Samples evaluated**: 100  |  **Eval time**: 5938.9s

**Max seq len**: 5000 (genus=0: ~3852, genus=1: ~4231, genus=2: ~4610 tokens)

## Greedy Sampling Metrics

| # | Metric | Value |
|---|--------|-------|
| (i) | **Manifold validity rate** | **100.0%** (100/100 decoded) |
|     | Parse failures | 0/100 (0.0%) |
| (ii) | **Token accuracy** | **22.5%** |
|      | Exact-match rate | 0.0% (0/100) |
| (iii) | **Silhouette IoU** | **0.7029** |
| (iv) | **Genus accuracy** | **62.0%** (62/100) |
| (v) | **Mean distill IoU** | **0.9363** |
| (vi) | **Rejection rate** | **13.7%** (274/2000) |

## Distillation Pipeline Stats

| Stat | Value |
|------|-------|
| Total attempted | 2000 |
| Accepted | 1726 |
| Rejected | 274 |
| Mean distill IoU | 0.9363 |

## Comparison with MeshGPT (Published Numbers)

> **Important caveat**: This comparison is provided for context only. Direct comparison is not meaningful due to fundamental differences:
> - MeshGPT uses a VQ-VAE + GPT architecture; we use DLFL topology tokens + silhouette conditioning.
> - MeshGPT is trained/evaluated on ShapeNet chairs; we use Thingi10K (multi-category, genus 0–2).
> - MeshGPT uses raw triangle mesh tokens; we use DLFL structural operators + quantized vertex coordinates.
> - The metrics (Coverage, Quality) are different from ours (Silhouette IoU, Manifold validity, Genus accuracy).

| Method | Architecture | Dataset | Manifold% | Coverage | Quality |
|--------|-------------|---------|-----------|----------|--------|
| MeshGPT (published) | VQ-VAE + GPT | ShapeNet chairs | ~98% | 85.4% | 93.7% |
| **Ours (Phase B)** | DLFL + xAttn Transformer | Thingi10K (g=0-2) | **100.0%** | sil-IoU=0.7029 | — |

## Sample Images

Side-by-side visualisations (target | generated) saved to `/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_b/samples/`.

Format: left 2×2 grid = 4 conditioning silhouette views of target; right 2×2 grid = same 4 views rendered from generated mesh.

## Notes

### Silhouette IoU

Rendered from 4 conditioning viewpoints (azimuths 0/90/180/270°, elevation 0°, radius=3.0, 128×128). Binary threshold 0.5.

### Genus accuracy

Compares `mesh.genus()` of generated mesh with ground-truth genus stored in val shard.

### Manifold validity

DLFL guarantees manifold property for valid structural sequences. Sub-100% indicates out-of-range ordinal references or truncated sequences.
