# Phase A' — Propose-and-Optimize with Full Differentiable Operator Set

*Approved 2026-08-09. Supersedes Phase A (IoU 0.013, 6-operator vocabulary).*

## Motivation

Phase A failed because the LLM was asked to predict ~4000 CV coordinate tokens
sequentially — accumulated quantization error produced random geometry (foreground
IoU 0.013). Now that all 29 DLFL operators have differentiable PyTorch support
(`topmod/diffgeo.py`, 84/84 tests passing), we can split the problem:

- **LLM**: predict a short topology program (~50-120 tokens) + coarse control cage
- **Differentiable executor**: refine the cage against conditioning silhouettes via gradient descent

This is the **propose-and-optimize** paradigm from neural program synthesis.

## Architecture Overview

```
4 silhouettes → CNN encoder → 64 spatial memory tokens
    → Transformer decoder (6-layer, d=256)
    → Token sequence: [BASE] [OP₁] ... [OPₖ] [SEP] [COORD₁] ... [COORDₘ] [EOS]
    → Parse: DiffSequence from topology section, cage coords from COORD section
    → Differentiable refinement: optimize cage via silhouette loss (500 steps Adam)
    → Final mesh: seq.forward(optimized_cage), seq.triangles()
```

## Vocabulary V2

| Range | Tokens | Count |
|-------|--------|-------|
| 0 | EOS | 1 |
| 1–29 | Operators (CC, DS, HDL, IE, DE, DUAL, STA, SIMP, VC, LOOP, SQRT3, HONEY, STAR, CCUT, LSTYLE, FRAC, PENT, PENT2, D1264, ROOT4, CHKB, DSBC, DOME, CRUST, EXTRUDE_FACE, STELLATE, SUBDIVIDE_EDGE, SUBDIVIDE_FACE, CV) | 29 |
| 30–32 | BASE primitives (CUBE, TETRAHEDRON, ICOSAHEDRON) | 3 |
| 33 | SEP (topology↔geometry boundary) | 1 |
| 34–289 | COORD bins (256 levels, maps to [-2, +2]) | 256 |
| 290–353 | REF ordinals (face/edge/vertex references for IE/DE/HDL) | 64 |
| 354 | BOS | 1 |
| 355 | PAD | 1 |
| **Total** | | **356** |

## Sequence Format

```
[BOS] [BASE_x] [OP₁] [REF...] [OP₂] [REF...] ... [SEP] [COORD₁] [COORD₂] ... [EOS]
```

- Before SEP: topology section — base primitive + operator chain with parameters
- After SEP: geometry section — flattened cage vertex coordinates (x,y,z,x,y,z,...)
- Cage vertex count is determined by executing the topology section, NOT by the LLM

## Implementation Steps

### Step 1: Vocabulary V2 (`topmod/tokenizer.py`) — 30min

Add `build_vocabulary_v2()` returning the 356-token mapping above.
Add `tokenize_v2()` and `detokenize_v2()` that produce/consume the new format.

### Step 2: Data Generation (`experiments/opseq_v2/gen_data_v2.py`) — 2hr

Grammar-based program sampler:
- Sample base primitive uniformly from {cube, tetrahedron, icosahedron}
- Sample operator depth 0–4 from geometric distribution (mean ~1.5)
- Sample operators from the 17 global linear ops (CC, DS, DUAL, STA, SIMP, VC,
  LOOP, SQRT3, HONEY, CCUT, LSTYLE, PENT, PENT2, D1264, ROOT4, CHKB, DSBC)
- Optionally prepend 0–2 HDL ops (genus 0–2)
- Execute via DiffSequence to get cage vertex count
- Apply deformations to cage vertices:
  - Global: random rotation, non-uniform scale (0.5–1.5 per axis), shear
  - Local: random subset (30–70%) of vertices get independent offset ±0.3
- Normalize to 80% of quantization range (reuse gen_data.py logic)
- Render 4-view silhouettes via nvdiffrast (azimuths 0/90/180/270°)
- Tokenize: topology section + SEP + COORD section + EOS

Target: 20,000 training + 2,000 validation samples.

### Step 3: Model V2 (`experiments/opseq_v2/model_v2.py`) — 30min

Copy OpSeqModel with changes:
- `VOCAB_SIZE = 354` (output logits, excludes BOS/PAD)
- `TOTAL_EMBED = 356`
- `MAX_SEQ_LEN = 200` (down from 5000)
- Same architecture: 4-channel CNN → 64 spatial tokens, 6-layer transformer decoder

### Step 4: Training (`experiments/opseq_v2/train_v2.py`) — 30min

- Pure cross-entropy on token sequence (teacher forcing)
- bf16 mixed precision
- 100 epochs, batch size 64 (short sequences → larger batches fit)
- AdamW, lr=1e-4, cosine decay
- Checkpoint best val loss

### Step 5: Differentiable Refinement (`pipeline/geometry_optimizer.py`) — 1hr

Add `optimize_through_chain()`:

```python
def optimize_through_chain(ctx, seq, targets, mvps, num_steps=500, lr=1e-2,
                           lambda_lap=0.05, lambda_edge=0.01):
    cage = seq.verts0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([cage], lr=lr)
    tris = seq.triangles(device=cage.device)

    for step in range(num_steps):
        optimizer.zero_grad()
        final_verts = seq.forward(cage)  # differentiable chain

        # Silhouette loss on final mesh
        sil_loss = multi_view_silhouette_loss(ctx, final_verts, tris, targets, mvps)

        # Regularization on FINAL mesh, not cage
        # Gradients backprop through S_k·...·S_1 transpose → cage-aware regularization
        reg = (lambda_lap * laplacian_loss(final_verts, tris) +
               lambda_edge * edge_length_loss(final_verts, tris))

        (sil_loss + reg).backward()
        optimizer.step()

    return cage.detach()
```

### Step 6: Inference Pipeline (`experiments/opseq_v2/infer_v2.py`) — 1.5hr

```python
def infer(model, images, ctx, device):
    # 1. LLM generates token sequence (greedy or nucleus)
    token_ids = model.sample_greedy(images, max_len=200)

    # 2. Parse with fault tolerance
    base, ops, raw_coords = parse_v2_sequence(token_ids)
    seq = DiffSequence(base)
    for op in ops:
        seq.append(op)
    n_cage = seq.verts0.shape[0]

    # 3. Fault-tolerant coordinate alignment
    if len(raw_coords) > n_cage * 3:
        raw_coords = raw_coords[:n_cage * 3]
    elif len(raw_coords) < n_cage * 3:
        raw_coords += [128] * (n_cage * 3 - len(raw_coords))  # center-fill

    cage_verts = dequantize(raw_coords).reshape(n_cage, 3)
    seq.verts0 = torch.tensor(cage_verts, dtype=torch.float32, device=device)

    # 4. Differentiable refinement
    mvps = orbit_cameras(4, azimuths=[0,90,180,270], ...)
    targets = preprocess_silhouettes(images)
    refined_cage = optimize_through_chain(ctx, seq, targets, mvps)

    # 5. Final mesh
    seq.verts0 = refined_cage
    final_verts = seq.forward()
    faces = seq.triangles()
    return final_verts, faces
```

### Step 7: Evaluation (`experiments/opseq_v2/eval_v2.py`) — 1hr

Metrics (all on foreground):
- **Pre-refine silhouette IoU**: how good is LLM's raw prediction
- **Post-refine silhouette IoU**: after diff optimization (primary metric)
- **Manifold validity**: should be 100% by construction
- **Operator accuracy**: % of topology tokens matching ground truth
- **Genus accuracy**: generated vs ground-truth genus

## Success Criteria

| Metric | Target | Phase A baseline |
|--------|--------|-----------------|
| Post-refine foreground IoU | **> 0.40** | 0.013 |
| Pre-refine foreground IoU | **> 0.05** | 0.013 |
| Manifold validity | **100%** | 100% |
| Op token accuracy | **> 80%** | N/A (different vocab) |

## Risk Mitigations

1. **LLM can't count cage vertices** → parser computes count, coords are
   truncated/padded; diff optimizer compensates
2. **Smooth "dough" shapes only** → local vertex perturbation in datagen;
   Phase A' validates architecture, not final quality
3. **Quantization overflow** → normalize to 80% range in datagen
4. **Operator compatibility** → restrict datagen to 17 global linear ops
   (all verified differentiable + gradcheck passing)

## What This Does NOT Cover (deferred)

- Real-mesh distillation (Phase B' — reuse `opseq_b/distill.py` infrastructure)
- Render loss as training signal (soft-argmax through COORD logits — Phase C)
- KV-cache for faster inference
- Nonlinear ops in programs (CRUST, STAR, FRAC, DOME — need parameter prediction)
- Closed-loop generation (model sees intermediate mesh)
