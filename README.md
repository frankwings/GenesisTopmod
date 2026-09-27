# Topo-Carving

**Discovering surface topology from multi-view silhouettes — with guaranteed-manifold operators.**

Most differentiable mesh reconstruction assumes the topology (a template with the right number of
holes) and only preserves it. **Topo-Carving discovers the genus from the images**: a space-carving
oracle tells the optimizer *how many* tunnels the shape has and *where* they are, and every topology
change is executed as a TopMod **DLFL `add_handle`** — so the mesh is a valid orientable 2-manifold,
watertight, after **every** step, and the genus can never drift silently.

## The whole algorithm in one strip

<p align="center">
<img src="experiments/opseq_v5/despike/results_genus/fig_evolution_strip.png" width="980" alt="evolution: sphere to genus-4"/>
</p>

<p align="center">
<img src="experiments/opseq_v5/despike/results_genus/gifs/fertility_evolution_hd.gif" width="680" alt="fertility reconstruction: genus discovered 0 -> 4"/>
<br/><i>Live: fertility from a genus-0 icosphere to the correct genus-4 statue. The title bar tracks
the stage and the genus — watch <code>add_handle</code> open each tunnel.</i>
</p>

## Highlights (golden v6.3)

- **Correct genus on 5/5 benchmark shapes** (armadillo 0, kitten 1, rockerarm 1, threeholes 3,
  fertility 4) — 15/15 chains in a ×3 regression, robust to CUDA nondeterminism.
- **Count-first propose-and-verify**: a handle is accepted on render evidence *or* when the oracle
  says a tunnel is missing and the membrane is genuine air / hull-located — fixing the classic
  failure where a real thin tunnel opens with a *negative* render gain and gets rejected.
- **Hull-guided completion**: when the refined mesh sits flush to the hull and image-space detection
  stalls, the hull's own tunnel location proposes the handle.
- Held-out silhouette IoU **0.9972** on armadillo (DMesh plateaus at 0.989), volume IoU **0.9915**
  on fertility with the seam-free finish.

## How it works

<p align="center">
<img src="experiments/opseq_v5/despike/results_genus/fig_pipeline.png" width="920" alt="pipeline"/>
</p>

**1 — Carve the oracle.** A voting visual hull is carved from the input silhouettes once. Its Euler
number gives the tunnel **count** g\* (morphological closing ladder, mode over radii); a
closing-ladder plug analysis gives each tunnel's **location** on demand.

**2 — See the membranes.** Faces whose interior samples sit in hull *air* are membranes blocking a
tunnel — the orange patches below are literally what the detector proposes to open:

<p align="center">
<img src="experiments/opseq_v5/despike/results_genus/fig_membrane_detect.png" width="440" alt="membrane detection"/>
<img src="experiments/opseq_v5/despike/results_genus/fig_hull_plugs.png" width="470" alt="hull plugs"/>
</p>

**3 — Propose-and-verify every handle.** Each membrane pair becomes a DLFL `add_handle` (manifold by
construction), kept only if a count-first gate and a differentiable-render check agree. The loop
stops exactly at g\*. When the refined mesh hugs the hull and membranes vanish, the oracle's plug
locations (right figure — the yellow plug is a missing 4th tunnel found on a flush mesh) take over.

**4 — Refine with guaranteed-manifold operators only.** Link-condition-guarded collapses,
Euler-preserving flips/subdivision, position-only DR: topological robustness holds **by
construction**, not by a regularizer.

## Results — genus discovered, never assumed

| | |
|:--:|:--:|
| <img src="experiments/opseq_v5/despike/results_genus/gifs/threeholes_evolution_hd.gif" width="380"/> | <img src="experiments/opseq_v5/despike/results_genus/gifs/kitten_evolution_hd.gif" width="380"/> |
| threeholes — discovered genus **3** | kitten — discovered genus **1** |
| <img src="experiments/opseq_v5/despike/results_genus/gifs/rockerarm_evolution_hd.gif" width="380"/> | <img src="experiments/opseq_v5/despike/results_genus/gifs/armadillo_evolution_hd.gif" width="380"/> |
| rockerarm — discovered genus **1** | armadillo — discovered genus **0** (no false tunnels) |

## Comparison with prior work

| Method | topology source | discovers genus? | manifold guarantee | genus drift possible? |
|---|---|---|---|---|
| Nicolet et al. (Large Steps) | fixed input mesh | no | no | yes (handle collapse) |
| Gu et al. (ICASSP 2026) | template (count + rough location) | no | no | patched via PH prior |
| DMesh | implicit (existence probs) | partially | no (self-prunes) | yes |
| Palfinger | fixed | no | remesh-based | yes |
| **Topo-Carving (ours)** | **discovered from the carved hull** | **yes** | **by construction (DLFL)** | **impossible by construction** |

<p align="center">
<img src="experiments/opseq_v5/despike/results_genus/fig_dmesh_compare.png" width="480" alt="DMesh comparison"/>
<img src="experiments/opseq_v5/despike/results_genus/fig_ablation.png" width="440" alt="ablation"/>
</p>

Every component earns its place (fertility, GT genus 4, n=15 per config): removing the oracle
collapses genus accuracy to **4/15** and fails in *both* directions (missed **and** spurious
tunnels); the count-first gate and hull-guided completion each close the remaining 13/15 → 15/15
gap from different sides (decision vs detection). Full study:
[`GATE_DECISION_findings.md`](experiments/opseq_v5/despike/GATE_DECISION_findings.md) — including
the honest negative result that a typed-decision language model is a coin flip at the decisive
geometric decision.

## Reproduce

```bash
cd experiments/opseq_v5
SHAPES="fertility threeholes kitten rockerarm armadillo" TAGP=repro \
USE_JEV=1 GATE_BACKEND=count HULL_COMPLETE=1 bash despike/golden_chain.sh
# prints [RESULT] <shape>: final genus G (GT g) OK per shape
# add SNAPSHOT_EVERY=2 SNAPSHOT_HERO=1 to record the evolution GIFs
```

Requires: PyTorch (cu-enabled), nvdiffrast, open3d, scipy/scikit-image. Tested on RTX 5090
(torch 2.14 + cu130, sm_120).

| tag | what |
|---|---|
| `golden-v6.1` | baseline chain (fertility 13/15) |
| `golden-v6.2` | count-first gate + hull-guided completion + plug-cache fix (15/15) |
| `golden-v6.3` | + late-handle seam refit (Stage 6b), seam artifact eliminated |

## Documentation

- [`PAPER_NOTES.md`](experiments/opseq_v5/despike/PAPER_NOTES.md) — paper skeleton: contributions,
  ablation table, negative results, limitations
- [`results64v/GOLDEN.md`](experiments/opseq_v5/despike/results64v/GOLDEN.md) — version log with all
  measured numbers (incl. the fair face-count comparison vs DMesh)
- [`SELLING_POINT_topology_robustness.md`](experiments/opseq_v5/despike/SELLING_POINT_topology_robustness.md)
  — why genus drift is impossible by construction
- [`real/CAPTURE_SPEC.md`](experiments/opseq_v5/real/CAPTURE_SPEC.md) — iPhone capture protocol for
  real-data topology discovery (ARKit poses + SAM2 masks, tooling included)

## The TopMod library underneath

The topology machinery is a pure-Python implementation of Dr. Ergun Akleman's **TopMod** DLFL mesh
system: **29 operators** (4 fundamental + 6 high-level + 7 classic subdivision + 12 remeshing
schemes) with closed-form oracle tests, **100% differentiable** position maps (PyTorch), a
**Blender addon** (21 operators in Edit Mode), and an autoregressive mesh tokenizer. Zero required
dependencies for the core.

→ Full library documentation, quick start, Blender install guide and the operator reference:
**[`LIBRARY.md`](LIBRARY.md)** · [`docs/operators.md`](docs/operators.md)

## Citation

Paper in preparation. For now:

```bibtex
@misc{topocarving2026,
  title  = {Topo-Carving: Discovering Surface Topology from Multi-View Silhouettes
            with Guaranteed-Manifold Operators},
  author = {GenesisTopmod team},
  year   = {2026},
  url    = {https://github.com/frankwings/GenesisTopmod}
}
```

Built on Dr. Ergun Akleman's TopMod topological mesh modeling theory.
