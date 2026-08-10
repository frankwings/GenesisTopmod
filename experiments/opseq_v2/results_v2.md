# Phase A' Evaluation Results

## Metrics

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| Pre-refine IoU  | 0.8361 | > 0.05 | ✓ |
| Post-refine IoU | 0.7431 | > 0.40 | ✓ |
| Manifold        | 1.0000 | 1.00   | ✓ |
| Op Token Acc    | 0.5284 | > 0.80 | ✗ |
| Genus Accuracy  | 0.6680 | –      | – |

## Settings

- Samples evaluated: 500
- Refinement steps: 200

## Comparison

| System | Post-refine IoU | Notes |
|--------|----------------|-------|
| **Phase A' (this)** | **0.7431** | Propose-and-optimize |
| Phase A (baseline) | 0.013 | Sequential CV token prediction |

*Phase A failed because it predicted ~4000 CV coordinate tokens sequentially.
Phase A' predicts a short topology program + coarse cage, then refines via
differentiable rasterization.*
