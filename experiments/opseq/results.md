# OpSeq Phase A — Evaluation Results

**Checkpoint**: `/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq/ckpt/best.pt`

**Epoch**: 110  |  **Best val loss**: 0.8159

**Samples evaluated**: 100  |  **Eval time**: 166.4s

## Greedy Sampling Metrics

| # | Metric | Value |
|---|--------|-------|
| (i) | **Manifold validity rate** | **100.0%** (100/100 decoded) |
|     | Parse failures | 0/100 (0.0%) |
| (ii) | **Token accuracy** | **16.8%** |
|      | Exact-match rate | 0.0% (0/100) |
| (iii) | **Silhouette IoU** | **0.0128** |
| (iv) | **Genus accuracy** | **59.0%** (59/100) |

## Nucleus Sampling (top-p=0.9, first 10 samples)

| Metric | Value |
|--------|-------|
| Manifold validity | 100.0% (10/10) |
| Parse failures | 0/10 |

## Notes

### (i) Manifold validity

Hypothesis: **100%**.  The DLFL invariant ensures that any
valid sequence of structural operators (HDL, CC, IE, DE) produces
a closed orientable 2-manifold.  The only way to break this is to
emit an ordinal reference outside the current mesh's face/edge count.
A sub-100% rate here indicates the model generating out-of-range
ordinals, which is counted as a parse failure.

### (iii) Silhouette IoU

Rendered from the same 4 viewpoints (azimuths 0/90/180/270°,
elevation 0°, radius=3.0) used during data generation.
IoU is computed at binary threshold 0.5 and averaged over 4 views.

### (iv) Genus accuracy

Compares `mesh.genus()` of the generated mesh with the ground-truth
genus stored in the val shard.
