# GenesisTopmod Research Plan

## Background

### The Problem
AI-generated 3D meshes (from TRELLIS, MeshGPT, Marching Cubes, Dual Contouring, etc.) frequently exhibit:
- Non-manifold edges/vertices
- Unwanted topology (wrong genus, closed boundaries that should be open)
- Self-intersecting faces
- No explicit topological control

### The Insight
Dr. Akleman's TopMod theory proves that **4 operators are minimal and complete** for generating all orientable 2-manifold meshes. Any mesh operation can be decomposed into sequences of these operators, and every intermediate result is guaranteed to be a valid 2-manifold.

### Our Approach: Plan A — Topology First, Geometry Later

```
Input Image
    |
    v
[Topology Analysis] --> infer genus, boundary count, component count
    |
    v
[TopMod Operator Sequence] --> construct skeleton mesh with correct topology
    |                          (manifold guaranteed at every step)
    v
[Geometry Optimization] --> fix topology, optimize vertex positions
    |                       via differentiable rendering (nvdiffrast)
    |                       loss = L1(rendered_image, target_image)
    v
Output: topologically correct + geometrically faithful mesh
```

Key separation:
- **TopMod** controls connectivity (which vertices connect to which faces, genus, boundaries)
- **nvdiffrast** controls geometry (where vertices sit in 3D space)

---

## Phase 1: TopMod Python Library

### Goal
Pure Python implementation of TopMod's core: DLFL data structure + 4 fundamental operators + essential high-level operations.

### 1.1 DLFL Data Structure

The Doubly-Linked Face List is TopMod's mesh representation. It stores:

```
Vertex:  id, position (x,y,z), list of outgoing half-edges
HalfEdge (Corner): id, origin vertex, face, next, prev, twin
Face:    id, one half-edge (entry point), list of half-edges forming boundary
Edge:    id, two half-edges (twins)
```

Key properties:
- Every edge has exactly 2 half-edges (twins) → guarantees 2-manifold
- Every face is a closed loop of half-edges
- Traversal: face loop (next/prev), vertex fan (twin→next)

### 1.2 Four Fundamental Operators

| Operator | Input | Effect | Topology Change |
|----------|-------|--------|-----------------|
| `create_vertex(pos)` | 3D position | Creates isolated vertex + degenerate face | New component |
| `delete_vertex(v)` | vertex | Removes vertex and its star | Removes component |
| `insert_edge(corner1, corner2)` | 2 corners (vertex-face pairs) | Splits face or joins faces by inserting new edge | May change genus |
| `delete_edge(edge)` | 1 edge | Merges two adjacent faces into one | May change genus |

The `insert_edge` operator is the most complex — its topological effect depends on whether corner1 and corner2 are on the same face or different faces:
- **Same face**: splits the face into two (genus unchanged)
- **Different faces, same component**: merges faces, genus +1 (adds handle)
- **Different faces, different components**: merges components

### 1.3 High-Level Operations (built on 4 operators)

Priority for demo:

| Operation | Description | Composed of |
|-----------|-------------|-------------|
| `extrude_face(face, dist)` | Push face outward along normal | insert_edge × N |
| `add_handle(face1, face2)` | Connect two faces with a tube | insert_edge sequence |
| `stellate(face)` | Add center vertex, create pyramid | create_vertex + insert_edge × N |
| `subdivide_edge(edge)` | Split edge at midpoint | create_vertex + insert_edge |
| `subdivide_face(face)` | Split face by adding center vertex | create_vertex + insert_edge × N |

### 1.4 Utility Functions

- `genus(mesh)` — compute genus via Euler formula: V - E + F = 2 - 2g
- `boundary_count(mesh)` — count boundary loops
- `is_manifold(mesh)` — verify 2-manifold property (should always be True)
- `to_obj(mesh, path)` — export to OBJ format
- `from_obj(path)` — import from OBJ format
- `to_trimesh(mesh)` — convert to triangulated mesh (for rendering)
- Primitive generators: `make_cube()`, `make_tetrahedron()`, `make_icosahedron()`, etc.

### 1.5 Validation Suite

Every operator must pass:
1. Manifold invariant check (every edge has exactly 2 half-edges)
2. Euler relation check (V - E + F = 2(C - g) where C = components, g = total genus)
3. Face loop closure check (following next pointers returns to start)
4. Vertex fan consistency check

---

## Phase 2: Plan A Pipeline

### Goal
End-to-end pipeline: input image → topologically correct mesh matching the image.

### 2.1 Topology Analysis (simple version for demo)

For the demo, manually specify topology:
```python
target_topology = {
    "genus": 1,        # torus = 1 hole
    "boundaries": 0,   # closed surface
    "components": 1    # single object
}
```

Future: use vision model to infer topology from image automatically.

### 2.2 Topology Construction

Given target topology, generate a skeleton mesh using TopMod operators:

```python
def build_topology(genus=0, boundaries=0):
    mesh = make_icosahedron()  # start with genus-0 base

    # Add handles for each genus
    for i in range(genus):
        f1, f2 = select_opposite_faces(mesh)
        add_handle(mesh, f1, f2)

    # Add boundaries (open holes)
    for i in range(boundaries):
        f = select_face(mesh)
        delete_face(mesh, f)  # open a hole

    # Subdivide for resolution
    for _ in range(2):
        catmull_clark(mesh)

    return mesh  # guaranteed 2-manifold with correct topology
```

### 2.3 Geometry Optimization via Differentiable Rendering

```python
import nvdiffrast.torch as dr

# Freeze connectivity (faces), optimize vertex positions
vertices = torch.nn.Parameter(mesh.vertices)  # learnable
faces = torch.tensor(mesh.faces)              # fixed

optimizer = torch.optim.Adam([vertices], lr=1e-3)

for step in range(num_steps):
    # Render from multiple views
    rendered = nvdiffrast_render(vertices, faces, camera_poses)

    # Compare with target image(s)
    loss = F.l1_loss(rendered, target_images)

    # Optional regularizers
    loss += lambda_lap * laplacian_smoothness(vertices, faces)
    loss += lambda_edge * edge_length_regularization(vertices, faces)

    loss.backward()
    optimizer.step()
```

### 2.4 Demo Targets

| Demo | Input | Expected Output |
|------|-------|-----------------|
| Sphere | sphere image | genus-0 mesh matching shape |
| Torus | donut image | genus-1 mesh matching shape |
| Cup | mug image | genus-1 + 1 boundary mesh |
| Double torus | figure-8 image | genus-2 mesh |

Success criteria: correct topology (genus/boundary count) + visually recognizable shape.

---

## Phase 3: Advisor Demo

### Deliverables
1. Working TopMod Python library with tests
2. 3-4 demo results showing topology control + geometry fitting
3. Side-by-side comparison: our method (correct topology) vs. Marching Cubes (broken topology)
4. Short slide deck explaining the approach

### Paper Direction (post-demo, if advisor approves)
- Title candidates:
  - "Topology-Complete Mesh Generation via Manifold Operator Sequences"
  - "TopMod Meets Neural Rendering: Manifold-Guaranteed 3D Mesh Generation"
- Target venue: SIGGRAPH Asia / Eurographics / SGP
- Key contribution: first method to use provably complete topological operators for AI mesh generation

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mesh data structure | DLFL (half-edge) | TopMod's native representation; manifold by construction |
| Language | Python | Fast prototyping; easy integration with PyTorch |
| Differentiable renderer | nvdiffrast | Faster than PyTorch3D; more accurate gradients via hardware rasterization |
| Base shapes | Platonic solids | Clean starting topology; easy to subdivide |
| Subdivision | Catmull-Clark | Most widely used; smooth organic results |
| Coordinate quantization | Not needed (continuous optimization) | Unlike MeshGPT, we optimize coordinates continuously |

---

## File Structure

```
GenesisTopmod/
├── README.md
├── docs/
│   └── plan.md                 # this file
├── topmod/
│   ├── __init__.py
│   ├── dlfl.py                 # DLFL data structure
│   ├── operators.py            # 4 fundamental operators
│   ├── high_level_ops.py       # extrude, handle, stellate, etc.
│   ├── subdivision.py          # Catmull-Clark, Doo-Sabin
│   ├── primitives.py           # cube, tetrahedron, icosahedron generators
│   ├── io.py                   # OBJ import/export
│   └── validate.py             # manifold invariant checks
├── pipeline/
│   ├── __init__.py
│   ├── topology_builder.py     # construct mesh with target topology
│   ├── geometry_optimizer.py   # nvdiffrast vertex optimization
│   └── demo.py                 # end-to-end demo script
├── tests/
│   ├── test_dlfl.py
│   ├── test_operators.py
│   ├── test_high_level_ops.py
│   ├── test_subdivision.py
│   └── test_invariants.py
└── requirements.txt
```

---

## References

1. Akleman, E., & Chen, J. (2003). "A minimal and complete set of operators for the development of robust manifold mesh modelers." *Graphical Models*, 65(5), 286-304.
2. Akleman, E., et al. (2004). "TopMod: Interactive Topological Mesh Modeler." *Computer Graphics International*.
3. Akleman, E., et al. (2008). "TopMod3D." *Computer Graphics International*.
4. Srinivasan, V. (2005). "Modeling High-Genus Surfaces." PhD Dissertation, Texas A&M.
5. Siddiqui, Y., et al. (2024). "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers." *CVPR 2024*.
6. Son, S., et al. (2024). "DMesh: A Differentiable Mesh Representation." *NeurIPS 2024*.
7. Luo, H., et al. (2024). "SpaceMesh: A Continuous Representation for Learning Manifold Surface Meshes." *SIGGRAPH Asia 2024*.
8. Laine, S., et al. (2020). "Modular Primitives for High-Performance Differentiable Rendering." *ACM TOG*.
