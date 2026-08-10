# Phase A'' Evaluation Results (Topology-Only + Direct Vertex Optimization)

## Key Insight

Ablation results show direct all-vertex optimization (IoU 0.974) outperforms
DiffSequence cage optimization (0.962). This pipeline separates concerns cleanly:
- **LLM**: predicts short topology program only (~5–15 tokens)
- **Optimizer**: refines all mesh vertices directly via nvdiffrast

## Metrics

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| Pre-refine IoU    | 0.7741 | –      | – |
| Post-refine IoU   | 0.8272 | > 0.40 | ✓ |
| Manifold          | 1.0000 | 1.00   | ✓ |
| Op Token Acc      | 0.5239 | > 0.80 | ✗ |
| Genus Accuracy    | 0.6600 | –      | – |
| Topology Match    | 0.3100 | –      | – |

## Ablation Comparison

| System | Post-refine IoU | Notes |
|--------|----------------|-------|
| **Phase A'' (this)** | **0.8272** | Topology-only LLM + direct vertex opt |
| Direct all-vertex opt (oracle topology) | ~0.974 | Upper bound |
| Phase A' (cage+chain) | ~0.962 | V2: topology + cage coords |
| Plain cube baseline | ~0.947 | No topology learning |
| Phase A (sequential CV) | 0.013 | Original failure |

## Settings

- Samples evaluated: 200
- Refinement steps: 200
- Model: OpSeqModelV3 (4-layer transformer, 99-token topology-only vocab)
