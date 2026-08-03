# GenesisTopmod

**Topology-Guaranteed Mesh Generation via TopMod Operators**

A research project combining Dr. Ergun Akleman's TopMod topological mesh theory with modern AI mesh generation, enabling topology-aware 3D mesh generation from images.

## Core Idea

Current AI mesh generation methods (MeshGPT, TRELLIS, etc.) have no topological guarantees — outputs often contain non-manifold edges, unwanted holes, and self-intersections. TopMod's mathematical framework (4 minimal & complete operators) guarantees that every intermediate and final result is a valid orientable 2-manifold.

This project implements **Plan A: Topology First, Geometry Later** — use TopMod operators to construct topologically correct mesh skeletons, then optimize vertex positions via differentiable rendering to match target images.

## Project Status

- [ ] Phase 1: TopMod Python library (DLFL + 4 operators + high-level ops)
- [ ] Phase 2: Plan A pipeline (topology construction + nvdiffrast geometry optimization)
- [ ] Phase 3: Demo for advisor review

## Architecture

See [docs/plan.md](docs/plan.md) for the full research plan.

## References

- Akleman & Chen 2003: "A minimal and complete set of operators for the development of robust manifold mesh modelers"
- DMesh (NeurIPS 2024): Differentiable face existence probability
- MeshGPT (CVPR 2024): Autoregressive mesh token generation
- SpaceMesh (SIGGRAPH Asia 2024): Continuous halfedge manifold representation
- LATO.2 (2026): Factorized vertex + topology flow matching

---
*Zengyn42 / Genesis Research*
