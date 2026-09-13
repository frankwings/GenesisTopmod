# topmod_core — C++ DLFL kernel for the GenesisTopmod chain (spec, 2026-09-12)

## Why
Golden v3 chain wall time is ~2 h per shape; GPU rendering is ~4 min of it. The rest is the pure-Python DLFL
half-edge structure (`topmod/dlfl.py`, `operators.py`, `high_level_ops.py`): `collapse_edge_tri` costs 7.8 ms/op
(it scans ALL half-edges to re-origin `v1`), and every remesh pass round-trips arrays -> OBJ tempfile -> DLFLMesh ->
arrays. A fertility Stage-6 pass = 12k collapses + 7k splits = 156 s, x24 passes. Target: same results, >=100x faster
(collapse < 20 us, one 50k-V pass < 2 s), so the chain runs in ~15 min.

## Non-goals
- No change of rules, guards or results. The C++ kernel must reproduce the Python operators' topology exactly
  (same link condition, same vertex placement = midpoint, same centroid for stellate, same fan order).
- No GPU. No change to the optimisation loop. Python stays the orchestration layer.
- Do NOT edit any existing Python module while task `v4_all` is running (it imports them per stage). Integration is a
  separate step behind an env switch `DLFL_BACKEND=cpp` (default stays `py` until validated).

## Deliverables
1. `topmod/cpp/dlfl_core.cpp` (+ `.h`): half-edge mesh with integer handles (no pointers exposed), pybind11 module
   `topmod_core` built by `topmod/cpp/setup.py` (pybind11 3.1.0 is installed for user; g++ 13; python 3.12 headers at
   /usr/include/python3.12). `python3 setup.py build_ext --inplace` must produce `topmod/topmod_core*.so`.
2. Python wrapper `topmod/core_backend.py` exposing array-level batch functions with the SAME signatures and return
   values as the current callers (so integration is a one-line dispatch):
   - `flip_sweep(V, F, passes=4, fold_cos=0.0) -> (V, F2, n_flips)`            (dlfl_untangle.flip_sweep)
   - `collapse_short_edges(V, F, ratio=0.3, max_n=400, thr_abs=None, vthr=None) -> (V2, F2, n)`  (dlfl_untangle)
   - `subdivide_faces(V, F, fids, expand_ring=True) -> (V2, F2, n_split_edges)`  (phase1c.dlfl_subdivide_arrays)
   - `catmull_clark(V, polys) -> (V2, polys2)`, `triangulate_all(V, polys) -> tris`  (cc_subdiv.py)
   - `add_handle(V, F, fi, fj) -> (V2, F2)` (single op incl. stellating the side quads; used by phase7)
   - `check_watertight(F) -> (ok, n_bad_edges)` and `euler_genus(V, F)`.
   Vertex ORDER contract: output vertex i == input vertex i for all surviving input vertices; new vertices appended;
   collapse removes `v1` (the second endpoint) and keeps `v0` at the midpoint — exactly like the Python version;
   removed vertices are compacted out and F re-indexed (callers already handle changing V, see phase4_inloop).
3. `topmod/cpp/test_equivalence.py`: for each function, run Python (`topmod`/`dlfl_untangle`/`phase1c`) and C++ on
   the same inputs and assert identical results: same V (allclose 1e-12), same face set as unordered triangles
   (sorted index triples), same n. Inputs: icosphere cc2/cc3 from `topmod.primitives` + `catmull_clark`, the archived
   meshes `experiments/opseq_v5/despike/results_genus/armadillo_g3chain_raw.npz` (49k V) and
   `.../fertility_g3ccchain_raw.npz`, plus random-perturbed copies. Include `add_handle` on a known face pair
   (take one from `results_genus/handles_fertility_v4.json` round 1 or any two far-apart faces) and check
   watertight + genus+1.
4. `topmod/cpp/bench.py`: time Python vs C++ for one full pass of collapse_short_edges (ratio 0.4, cap 20% F),
   flip_sweep(3 passes), subdivide_faces(5% of faces) on the 49k-V armadillo mesh. Print ms and speedup.

## Operator semantics to reproduce (read the Python source; it is the reference)
- Fundamental (operators.py): `insert_edge(he1, he2)` (same face -> split face; different faces -> merge/handle),
  `delete_edge(edge)` (merge faces). Keep them as the primitives that `flip` and `add_handle` are built from.
- `collapse_edge_tri(edge)` (high_level_ops.py:930): closed triangle meshes only; guard = link condition (common
  neighbours of v0,v1 == {a,b}, a!=b, a!=v0, b!=v1, both faces triangles); v0 moves to midpoint; all half-edges with
  origin v1 re-origin to v0 (use the vertex fan, NOT a global scan); the two flanking triangles are removed and their
  duplicate edges merged (e_keep/e_del pairs as in the Python). Returns surviving vertex or None (guard reject).
- `subdivide_edge(edge)`: midpoint vertex, V+1 E+1, faces unchanged (n-gons grow by one vertex).
- `stellate(face)`: centroid vertex, face -> n triangles.
- `try_flip(edge, fold_cos)` (dlfl_untangle.py:20): triangles only; c,d = opposite corners; reject if c==d or c-d
  already adjacent; flip only if current normals' dot < fold_cos AND the new pair is less folded (n1.n2 > cur+1e-6)
  and non-degenerate (area > 1e-3*(la+lb)); implemented as delete_edge + insert_edge(hc, hd) with hc, hd = the
  half-edges of the merged face whose origins are c, d.
- `flip_sweep`: up to `passes` sweeps over a snapshot of the edge list, stop when a sweep makes no flip.
- `collapse_short_edges`: edges sorted by length ascending; stop when length > thr (thr = ratio*mean edge, or
  thr_abs, or per-vertex vthr: edge collapses only if shorter than min(vthr[a], vthr[b]) and vthr.max()); at most
  max_n successful collapses; skip edges already removed by earlier collapses.
- `subdivide_faces(fids, expand_ring)`: target set = fids (+ edge-adjacent faces if expand_ring); subdivide_edge on
  every edge of every target face (each edge once); then stellate every non-triangular face of the whole mesh.
  Returns number of split edges.
- `add_handle(face1, face2)` (high_level_ops.py:174): both faces removed, n side quads between corresponding
  vertices (n = min valence), then stellate the quads (caller does that today; do it inside the batch function).
- Catmull-Clark (topmod/subdivision.py:27) and triangulate_all (fan from first vertex).
- Validation: `check_all` = face-loop, vertex-fan, twin and Euler checks (validate.py). Expose `validate()` in C++ and
  call it in the tests after every batch.

## Performance notes
- Store vertices/half-edges/faces/edges in std::vector with free-lists; ids are indices; deleted flag; compaction on
  export. Vertex fan traversal via twin->next. No std::map in hot loops.
- Batch functions: build once from arrays, apply N ops, export once. Export must preserve input vertex order.
- Edge length sort: std::sort on (len2, edge id). Deterministic given the same input (tie-break by id).

## Acceptance
- test_equivalence.py passes on all listed meshes for all functions.
- bench.py: collapse pass on armadillo 49k V >= 100x faster than Python; flip_sweep >= 50x.
- No modification of files outside `topmod/cpp/`, `topmod/core_backend.py`, `topmod/__init__.py` (only to add a
  lazy `try: import topmod_core` flag) in this task.
