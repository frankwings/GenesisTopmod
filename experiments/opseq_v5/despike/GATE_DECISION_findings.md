# Handle accept/reject gate — decision study (痛点 4)

The add_handle propose-and-verify step decides whether a proposed genus-raising handle is a
real through-tunnel (keep) or spurious (revert). This note records what we tried and shipped.

## Baseline (legacy, `phase7_multi.sh`)
Hard threshold OR-rule:  `accept ⟺ Δho16 ≥ 0.002  OR  hair ≤ 0.9·ref`.
Failure mode on fertility: the 4th (thin) tunnel, when it first surfaces at the refined Stage-5b,
opens with a **negative** Δho (≈ −0.0014, render noise floor) and no hair drop → rejected → genus 3.
Measured: **13/15** runs correct (fertility GT genus 4), 2 misses to genus 3 (leg5, leg13).

## Tried: typed-decision models (Jev paradigm / Laya)
`jev_handle_gate.py` wires the TypeSafe **Jev** paradigm (typed discrete decision + calibrated
probability, non-autoregressive). Real Jev needs a paid key; **Laya** (`convaiinnovations/laya`,
ModernBERT/mmBERT encoder, free, local, `pip install laya`) is the same paradigm and was used.

Findings (rigorous):
- Laya reads **words, not numbers** — it will not compare 2.4 vs 1.1 voxels. Feature numbers must be
  pre-translated into qualitative statements ("open air" vs "inside solid"), which means the
  discrimination is in *our* thresholding, not the model.
- Live fertility ×15: Laya gate scored **15/15** — but this is **confounded**. In those runs the 4th
  tunnel happened to be found in Stage-3 with positive Δho (detection nondeterminism), never reaching
  the hard Stage-5b decision. It was luck, not gate intelligence.
- **Isolated test at the decisive point** (tunnel missing per oracle + NEGATIVE Δho + air): Laya is a
  **coin flip** — P(accept) ∈ [0.49, 0.51], and the ordering is nonsensical (more air → *less* likely
  to accept). Its confidence self-reports as UNCALIBRATED (temperature-clamp warning).
- Conclusion: a text decision model is the **wrong tool** for this decision. The decision is a hard
  count prior (g\*) + a numeric comparison, not a linguistic judgment. **Negative result — archived.**

## Shipped: Rule C — count-first deterministic gate (`GATE_BACKEND=count`, default)
The pipeline only proposes a handle when `g < g*` (the space-carving oracle guarantees a tunnel is
still missing). So:

    accept ⟺ legacy(Δho ≥ thr  OR  hair ≤ 0.9·ref)
             OR  rescue(g < g*  AND  out_vox ≥ 2  AND  blob ≥ 40)

The **rescue** clause keeps the handle regardless of Δho sign once the oracle says a tunnel is missing
there and the membrane is genuine open air — exactly the case legacy (needs +0.002) and the text model
(coin flip) both fail. Deterministic, reproducible, no external dependency, bounded by the g\* stop
(cannot overshoot genus).

Measured: **14/15** (rescue fired 8× in the real pipeline). Residual miss (c15): the 4th tunnel only
surfaced at Stage-5b with **out_vox = 0** — the refined mesh is flush to the hull, so the "air" signal
is gone even though the tunnel is real. Rule C's air-gate correctly-by-its-own-logic rejected it.

## Open: Rule C′ — trust the oracle's LOCATION
Root cause of the residual: at the refined stage the geometric air evidence vanishes; the only reliable
signal left is the hull oracle's *knowledge* that a tunnel must be there. Proposed C′: pass the real
candidate provenance (hull-located vs ray-fallback) to the gate and, when `g < g*` and the candidate is
**hull-located** (the oracle pointing at the known missing-tunnel location), accept unconditionally
regardless of out_vox/Δho. Ray-fallback candidates still require the air evidence. Expected → 15/15.
(This is the long-pending "hull-guided completion".)
