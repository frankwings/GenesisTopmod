# Related Work — positioning of Topo Carving (golden v6.1)

## KEY COMPARISON PAPER (closest competitor, 2026)
**Gao, Wang, Liu, Wang, Huang, Chen, Gu — "Inverse Rendering for High-Genus 3D Surface Meshes from Multi-view
Images with Persistent Homology Priors"** — ICASSP 2026 (accepted), arXiv:2601.12155 (2026-01-17, cs.CV).
Stony Brook Univ. (Xianfeng Gu group) + Capital Normal Univ. 4-page IEEE.

What it does:
- Built on Nicolet et al. "Large Steps in Inverse Rendering" (2021): fixed-connectivity mesh, large-steps preconditioner,
  Mitsuba3 path tracer, RGB + silhouette supervision.
- Core idea: a **Persistent Homology (PH) prior** — extracts H1 barcodes of the mesh and adds a loss that PENALISES
  collapsing tunnels/handle loops during optimisation. Prevents the well-known failure that "uniform camera sampling
  collapses high-genus structure."
- Topology is **GIVEN**: starts from a template / initial mesh that already has the target genus. PH only PRESERVES it.
- Reports lower Chamfer Distance and higher Volume IoU than Nicolet on high-genus shapes (Thingi10K, Google Scanned
  Objects; synthetic-primary).

How WE differ (the three lines to hammer in every draft):
1. **DISCOVERY vs GIVEN**: they need the genus specified up front (template); we DISCOVER it from the silhouettes
   (space-carved hull as the genus oracle g*). Their method physically cannot go genus-0 -> genus-1.
2. **CREATE vs PRESERVE**: they only stop a given handle from vanishing; we ADD handles with the TopMod DLFL
   `add_handle` operator, verified by DR (propose-and-verify) and reverted if ineffective.
3. **MANIFOLD BY CONSTRUCTION**: they deform a fixed-connectivity mesh (no remeshing, no manifold/watertight guarantee);
   every one of our topology changes is a DLFL operator with `check_watertight` asserted, combinatorially manifold by
   construction (SI 0 % measured).
Same metrics (Chamfer, Volume IoU) and same baseline (Nicolet) -> direct head-to-head is possible.

What to BORROW: their PH anti-collapse prior is complementary and fixes a real risk in OUR pipeline — after `add_handle`,
the subsequent DR loop can slowly close a thin handle (e.g. a mug handle). A PH / topology-preservation loss as a
"handle guard" during the post-add DR steps is a clean addition. We ADD (TopMod), they PRESERVE (PH); combine.
What to ADDRESS (reviewers will raise): (a) "PH is more principled than your closing-ladder/membrane heuristics" ->
different problem (preserve vs discover) + our detection is DR-verified; (b) "uniform-sampling assumption is biased"
(their own point) -> our eval must report camera-coverage sensitivity (partial-orbit real captures degrade the hull).

## Other baselines (already in our tables)
- **Nicolet et al. 2021, "Large Steps in Inverse Rendering"** — genus-0 only (fixed sphere topology); the base both the
  PH paper and Palfinger build on. Our runs: genus 0 always, fertility VolIoU 0.826.
- **Palfinger 2022 (Continuous Remeshing)** — adaptive velocity remeshing; genus is EXTERNALLY fixed, cannot discover;
  fails fertility. 3.6 min armadillo, 37.9k V (our vertex-budget comparison target).
- **DMesh (2404.13445) / DMesh++ (2412.16776)** — differentiable mesh via point existence probabilities; produces
  NON-MANIFOLD soup needing post-processing (ManifoldPlus). 12-20 min; VolIoU 0.826-0.963.
- **Neural-implicit + Marching Cubes (NeuS/2DGS/SuGaR/MILo etc.)** — topology "emerges" from the level set, uncontrolled
  (floaters, spurious handles), non-manifold before cleanup. Not topology-discovery, not manifold-guaranteed.

## Our one-sentence position
Topo Carving is the only method that DISCOVERS genus from images AND guarantees a combinatorially manifold, watertight
mesh by construction — via silhouette-carved topology detection + TopMod DLFL surgery + DR verification; the ICASSP 2026
PH paper preserves a GIVEN genus on a fixed-connectivity mesh, and every other baseline either fixes genus or lets it
emerge non-manifold.

## What to BORROW from Gu et al. (ICASSP 2026) — three concrete, actionable items (2026-09-18)
Their method = Nicolet "Large Steps" (bi-Laplacian preconditioned rendering loss, Eq.1) + persistent homology. Topology is
TEMPLATE-GIVEN; PH only PRESERVES it. Three pieces are borrowable into our discover+manifold pipeline (they are orthogonal
to our TopMod create step):

1. **Persistent-homology anti-collapse guard (highest value; fixes our thin-handle risk).**
   They build a filtration (Vietoris-Rips) on mesh vertices, read H1 as a persistence diagram; each tunnel loop has a
   birth-death interval = its robustness. A loss keeping that interval >= threshold stops optimisation from thinning a
   handle to nothing. OUR risk: after DLFL `add_handle`, the following 400-step DR loop (Laplacian + collapse remesh) can
   slowly pinch a thin handle shut (mug-handle failure mode). We ADD (TopMod), their PH says DON'T-COLLAPSE — orthogonal,
   stackable. Practical proxy (no full PH library): keep the tunnel throat (min cross-section of the hole) open; see
   `despike/handle_guard.py` + spike below.
2. **Spanning-tree loop extraction -> cheap H1 = 2g verification.**
   Their algorithm: spanning tree T of the mesh edge graph; non-tree edges G\T = {e1..ek} give k independent cycles;
   rank H1 = 2g for genus g. We can run this after every `add_handle` to confirm a real non-contractible loop was created
   (rank went up by 2), a topology-side second check alongside our DR-verified accept/reject.
3. **Topology-aware evaluation metric (persistence-diagram distance, their "Birthday" metric).**
   Beyond Chamfer / Volume IoU we report only binary "genus correct?". Adding a bottleneck / persistence-diagram distance
   to GT gives a CONTINUOUS topological-correctness score, more informative under noise and a direct answer to their
   strongest column.

NOT borrowable: their template-given topology (Eq.1 starts from a genus-g mesh) and their "collaborative rendering"
(Phong vertex+fragment shaders for Mitsuba path tracing; we use nvdiffrast rasterisation). Our discovery + manifold-by-
construction differentiators stay the headline.
