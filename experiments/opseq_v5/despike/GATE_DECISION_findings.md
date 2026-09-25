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

## Shipped: Rule C′ — trust the oracle's LOCATION (gate half)
C′ passes real candidate provenance (hull-located vs ray-fallback) to the gate and, when `g < g*` and
the candidate is **hull-located**, accepts unconditionally regardless of out_vox/Δho; ray-fallback
candidates still require air. `phase7_multi.sh` now derives `prov` from the phase7 log (`(ray` → ray).

Measured: **14/15** — identical to C, because **`rescue-hull` fired 0 times**. Diagnosis (cp6 stage-5b):
the 4th tunnel candidate arrives as **`prov=ray`, out_vox=0** — the refined mesh is flush to the hull, so
the outside-hull membrane detector finds nothing → phase7 falls through to the ray fallback, which
image-guesses at out_vox=0. C′ correctly rejects that unreliable guess. So the gate is no longer the
bottleneck: **the hull's known tunnel LOCATION is never used to propose the handle at the refined stage.**

## Shipped: hull-guided completion (upstream / detection half) — E2E VERIFIED
`phase7_handle.py`: when `g < g*` and outside-hull membrane detection is empty, `find_tunnel_by_hull`
(closing-ladder plug analysis on the carved hull, mesh-independent) proposes the handle with
`prov=hull` before falling back to rays (`HULL_COMPLETE=1`, default). C′ accepts it via `rescue-hull`.

Two bugs found and fixed on the way (`hull_locate.py`):
- **Stale-truncated plug cache**: `find_plugs` stopped early at `n_needed = g*−g_mesh`, so a cache written
  when only 1–3 plugs were needed poisoned later calls needing more (observed: a 3-plug cache made the
  locator report 3/4). Fix: the ladder now always locates ALL `g0` hull plugs (cache complete and
  mesh-independent; face-pair casting still runs only for non-deduped plugs), and a loaded cache with
  fewer plugs than the hull genus is discarded as stale.

End-to-end proof (forced Stage-5b scenario, `phase7_multi.sh` on a real genus-3 flush fertility mesh):
membrane detection stalls → hull-completion locates 4/4 plugs (fresh ladder 37 s) → the missing 4th
arrives as `prov=hull, Δho=−0.0014, out_vox=0.0` → gate `count:rescue-hull` accepts → handle verified →
**final genus 4 = GT**. This is exactly the case where legacy (13/15) and Rule C alone (14/15) fail.
