# V5 TopoShapeNet — Evaluation Results (Plan C)

**Checkpoint:** `/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/ckpt/best.pt`
**Val samples (classification):** 5000
**Val samples (IoU eval):** 50

## Classification Accuracy

| Metric   | Value  |
|----------|--------|
| Top-1    | 42.30% |
| Top-3    | 72.20% |
| Top-5    | 85.12% |
| Shape MAE | 0.0787 |

## IoU Comparison: V5 (shape-aware) vs V4 (canonical)

| Metric               | Mean IoU |
|----------------------|----------|
| V5 Best-of-1 (shape) | 0.8968 |
| V5 Best-of-5 (shape) | 0.9257 |
| V4 Best-of-1 (plain) | 0.8404 |
| **V5 - V4 delta**    | +0.0564 |
