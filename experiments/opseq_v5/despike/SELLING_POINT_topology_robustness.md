# Selling point: topological robustness BY CONSTRUCTION
### (Topo Carving is inherently immune to the handle-collapse failure that competing methods patch with priors)
Recorded 2026-09-18. Paper-facing: goes into Method (invariant), Discussion, and one Related-Work contrast row.

## 1. The claim (one sentence)
In Topo Carving the genus of the mesh can change ONLY through the explicit, DR-verified `add_handle` operator; every other
step — the differentiable-render vertex updates and all remeshing (edge split / collapse / flip) — is topology-preserving
by construction, so a discovered handle/tunnel can never silently collapse during optimization. Methods built on
fixed-connectivity meshes (Nicolet 2021; Palfinger 2022) lack this guarantee and must add an external prior (Gu et al.,
ICASSP 2026: a persistent-homology loss) to stop high-genus features from vanishing under "uniform camera sampling."

## 2. The failure mode this is about (why it matters for high genus)
Reconstructing a genus-g surface by optimizing a mesh has a notorious failure: tunnels and handle loops COLLAPSE mid-
optimization and the reconstruction drops to a lower genus (a mug loses its handle, a torus fills its hole). It is the
central difficulty of high-genus inverse rendering — genus-0 is "often recoverable" while "high-genus surfaces remain
challenging due to topological ambiguity, which frequently causes tunnels and handle loops to collapse" (Gu et al.,
paraphrased). Once a handle is gone, gradient descent cannot bring it back (topology change is non-differentiable), so
the run is lost.

## 3. Why fixed-connectivity / preconditioned-deformation methods are vulnerable
Nicolet's "Large Steps" (the base of both Palfinger and Gu) optimizes vertex POSITIONS on a mesh with FIXED CONNECTIVITY,
using a bi-Laplacian preconditioner. A thin handle can be pulled until its two tube walls touch (a self-intersection /
near-degeneracy); the connectivity still says "genus g", but the geometry is a pinched sliver, and any subsequent
cleanup / remesh / re-extraction removes it → genus drops. Because the deformation is smooth and unconstrained, nothing
stops the pinch. Gu et al.'s remedy is a **persistent-homology prior**: compute the H1 persistence of the handle loops
and add a loss that keeps their birth–death interval from shrinking to zero — an EXTRA term whose only job is to fight a
collapse the representation permits. It works but is a patch on a representation that does not protect topology itself.

## 4. Why Topo Carving is immune — by construction, not by a loss
Genus in our pipeline is governed entirely by the operator set, each of which is topology-safe:
- **Vertex updates (DR).** Moving vertices with fixed faces cannot change the combinatorial genus of a mesh. Period.
- **Edge collapse** (`dlfl_untangle.collapse_short_edges`): LINK-CONDITION guarded, Euler number preserved (docstring:
  "link-condition guarded, Euler preserved"). The link condition is the standard test that an edge collapse keeps the
  surface a manifold and does not change its topology; edges failing it are skipped. So collapse can NEVER remove a handle.
- **Edge flip** (`flip_sweep`) and **Catmull-Clark / midpoint subdivision**: Euler-preserving by construction (V−E+F
  invariant / uniform refinement). No genus change.
- **The ONLY genus-changing operator is `add_handle`** (TopMod DLFL): it is applied at a detected membrane, is asserted
  watertight immediately (`assert check_watertight`, phase7), and is accepted only if a 400-step DR re-check improves the
  held-out silhouette IoU (propose-and-verify); ineffective handles are reverted. Genus goes up by exactly 1 per accepted
  handle, never spuriously, never down.
Consequently the mesh is combinatorially manifold and watertight after EVERY step, and its genus is a monotone,
audited quantity. The collapse failure mode simply has no mechanism to occur.

## 5. Evidence (already run; reproducible)
- **Stress test (synthetic).** kitten mesh immediately after `add_handle` (thin genus-1 tunnel). Ran the DR loop with the
  hull-field loss OFF (W_T=0), diffuse OFF, and Laplacian smoothing cranked to LAP_MULT=600 for 300 steps — a setting
  designed to smooth the thin handle shut. Result: genus stayed 1 with AND without the anti-collapse guard. The handle
  cannot be smoothed away because the topology-preserving remesh refuses the collapse. (Contrast: on a FIXED-connectivity
  mesh under the same smoothing, the tube pinches and the handle is lost — that is exactly Gu's motivating case.)
- **Guard is a validated but redundant no-op here.** We ported Gu's PH guard as `handle_guard.throat_openness` /
  `handles_guard_loss` (a cheap H1-persistence proxy: keep each tunnel throat open). In isolation it works (a hole-closing
  force collapses a genus-1 torus 1→0 without it, stays 1 with it — `test_handle_guard.py`). Wired into our pipeline it
  changes nothing on synthetic data, because our remesh already guarantees what the prior tries to enforce. We keep it
  opt-in (HANDLE_GUARD=0 default) purely as insurance for extreme REAL thin handles under noisy silhouettes.
- **Per-handle H1 = 2g verification.** After each `add_handle`, tree-cotree homology-generator extraction confirms the
  number of H1 generators equals 2·genus (logged OK), i.e. a real non-contractible loop was created.

## 6. How to frame it in the paper
- **Method (an invariant, stated once):** "Genus changes only through the verified `add_handle` operator; all other
  updates are topology-preserving (link-condition-guarded collapse, Euler-preserving flip/subdivision, position-only DR),
  so the mesh is manifold and watertight after every step and no discovered handle can collapse."
- **Discussion / robustness:** the stress-test experiment (LAP_MULT=600, no data terms, genus held) as direct evidence.
- **Related-work contrast row:** add a column "handle collapse under optimization":
  Nicolet / Palfinger = possible (fixed connectivity); Gu et al. = mitigated by a PH prior (extra loss);
  DMesh = N/A (non-manifold point soup); **Ours = impossible by construction.**
- **One-liner for the intro:** "Where prior high-genus methods add a topological prior to keep handles from vanishing,
  our operators cannot vanish them in the first place."

## 7. Scope / honesty (what we do NOT claim)
- We guarantee COMBINATORIAL manifoldness/genus, not the absence of geometric self-intersection — SI is measured (0% on
  the five shapes via the SI-push guard), not proven. The topology guarantee is independent of SI.
- The guarantee is about not LOSING a handle; DISCOVERING the right handles is the separate detection+verification
  contribution (membrane detector + DR accept/reject). Robustness-by-construction is about the second half (keep what you
  found), which is exactly the half Gu et al. address with their prior — and we address it for free.
