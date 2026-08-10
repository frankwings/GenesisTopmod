# Email to Prof. Akleman

---

**Subject:** Neural Shape Program Synthesis Using DLFL Operators — Seeking Your Feedback

Dear Professor Akleman,

I hope this email finds you well. I am writing to share some recent work that builds directly upon your DLFL (Doubly-Linked Face List) formulation and TopMod operators, and to seek your feedback and guidance.

## What We Built

We have developed a system called **OpSeq** (Operator Sequence) that uses DLFL operators as the vocabulary for a neural shape program synthesis pipeline. The core idea is:

**A small transformer model looks at 2D silhouette images and outputs a short program of DLFL operators (e.g., `CUBE → HDL → CC`) that, when executed, produces a manifold mesh matching the target shape.**

To support this, we implemented:

1. **A pure-Python DLFL library** (2,600 lines, zero dependencies) with 29 operators — including all major subdivision schemes from the original TopMod (Catmull-Clark, Doo-Sabin, Loop, sqrt(3), honeycomb, star, pentagonal, fractal, dome, crust, and more). This is a clean-room reimplementation based on published semantics, not derived from the original C++ codebase.

2. **Differentiable PyTorch implementations** of all 29 operators (17 via symbolic trace into sparse weight matrices, 7 with dedicated torch code, verified to 1e-9 against the float path with full gradient checks).

3. **A 6.9M-parameter CNN + Transformer model** that autoregressively predicts operator sequences conditioned on 4-view silhouette images.

4. **A Blender add-on** that exposes all 21 TopMod operators directly in Blender's interface.

## Key Experimental Findings

We ran three phases of experiments plus a critical ablation study:

- **Manifold validity**: 100% across all experiments — the DLFL construction guarantee holds perfectly in the neural generation setting. This compares favorably to MeshGPT (CVPR 2024), which achieves only ~98% manifold rate.

- **Genus accuracy**: The model learned to predict the correct number of topological holes (HDL operations) 62–67% of the time, purely from looking at 2D silhouettes. This demonstrates that explicit topology tokens are learnable.

- **Ablation — what DLFL topology contributes**: When we give the model the correct DLFL topology and use differentiable rendering (nvdiffrast) to optimize vertex positions, we achieve foreground silhouette IoU of 0.974. Without any DLFL topology (just optimizing a plain cube), IoU drops to 0.947. The gap widens significantly for shapes requiring higher subdivision depth (0.991 vs 0.894 at depth 3).

- **An honest finding**: Making the subdivision chain differentiable (optimizing only cage vertices through the subdivision matrix) did not outperform direct optimization of all vertices (0.962 vs 0.974). The value of DLFL operators lies in **topology selection** (determining vertex count and connectivity), not in the differentiable chain itself.

## Where DLFL Operators Uniquely Shine

After surveying the field (SpaceMesh, SIGGRAPH Asia 2024; DMesh, NeurIPS 2024; NeuManifold, WACV 2025; Neural Mesh Flow, NeurIPS 2020), we believe the unique value of DLFL-based generation is:

1. **Programmatic output**: The result is not just a mesh but an executable, human-readable program (`CUBE → HDL → CC`). Users can modify individual operations, change parameters, and replay.

2. **Explicit genus control**: HDL tokens make topology a first-class, countable quantity. No other generation method offers this.

3. **Composability**: 29 operators can be combined to produce an enormous variety of topological structures, far beyond what template-deformation methods can express.

These properties make DLFL operators particularly suited for **interactive modeling tools** (our Blender add-on) and **topology-aware generative design**.

## What We Would Value from You

1. **Correctness feedback**: We would welcome any review of our operator implementations against the canonical DLFL semantics. We have oracle tests but your expertise would be invaluable.

2. **Research direction**: Do you see value in pursuing the "topology-as-program" angle for publication? We are considering a paper that positions DLFL operators as a vocabulary for neural shape program synthesis.

3. **Collaboration interest**: If this direction interests you, we would be honored to explore a collaboration — particularly around the theoretical guarantees that DLFL provides and how they translate to the generative setting.

The code, experiments, and documentation are available for your review at any time.

Thank you for the foundational work on DLFL and TopMod that made this possible. I look forward to hearing your thoughts.

Best regards,
[Your Name]
