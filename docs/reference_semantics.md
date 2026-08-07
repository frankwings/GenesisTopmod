# TopMod Reference Scheme Semantics (clean-room extraction)

**Date:** 2026-08-06
**Source studied:** `davyrisso/topmod3d`, files `include/dlflaux/DLFLSubdiv.cc`, `DLFLCrust.cc`, `DLFLMultiConnect.cc`, `DLFLConnect.cc`, `DLFLExtrude.cc` (GPL, read-only reference).
**Clean-room policy:** this document records *topological semantics only*, described in our own words for independent reimplementation. No GPL code was copied, translated, or paraphrased line-by-line. Only observable behavior (what elements are created/destroyed and where new vertices are placed geometrically) is documented.

Notation: input mesh has `V` vertices, `E` edges, `F` faces, face degrees `d_f` with `Σ d_f = 2E`, vertex valences `n_v` with `Σ n_v = 2E`, Euler characteristic `χ = V − E + F`. All schemes assume a closed 2-manifold polygonal DLFL mesh (DLFL surfaces are always closed). Every subdivision oracle below preserves `χ` (verified per scheme); crust and multi-connect change it as stated.

## Oracle summary table

| Scheme | V′ | E′ | F′ | Composition / relation to existing ops | Cube (8,12,6) → |
|---|---|---|---|---|---|
| honeycombSubdivide | 2E | 3E | V + F | topologically = `dual ∘ stellate_all` | 24, 36, 14 |
| pentagonalSubdivide (offset) | V + 2E + F | 5E | 2E | trisect edges + centroid fan (all pentagons) | 38, 60, 24 |
| pentagonalSubdivide2 (sf) | V + 3E | 6E | F + 2E | mid-edge split + inset-at-midpoints | 44, 72, 30 |
| cornerCuttingSubdivide (alpha) | 2E | 4E | V + E + F | geometric variant of `doo_sabin` (identical topology) | 24, 48, 26 |
| modifiedCornerCuttingSubdivide(2) | 2E | 4E | V + E + F | geometric variant of `doo_sabin` | 24, 48, 26 |
| root4Subdivide (a, twist) | V + 2E | 4E | F + E | standalone (inset-and-bridge, then dissolve old edges) | 32, 48, 18 |
| starSubdivide (offset) | V + F + 2E | 9E | 6E | = `stellate_all ∘ stellate_all` | 38, 108, 72 |
| fractalSubdivide (offset) | V + E + F | 6E | 4E | = loopStyle split, then stellate each central face | 26, 72, 48 |
| domeSubdivide (length, sf) | V + 59E | 116E | F + 56E | = 4-sect all edges, then 7 stacked Doo-Sabin extrusions per face | 716, 1392, 678 |
| dual1264Subdivide (sf) | 4E | 6E | V + E + F | Doo-Sabin-like with 2d-gon inner faces (1/3–2/3 points) | 48, 72, 26 |
| checkerBoardRemeshing (t) | V + 4E | 9E | F + 4E | inset all faces + trisect edges + corner rewiring | 56, 108, 54 |
| loopStyleSubdivide (length) | V + E | 4E | F + 2E | = `loop` connectivity (polygonal generalization; geometry differs) | 20, 48, 30 |
| dooSabinSubdivideBCNew (sf, length) | V + 4E | 7E | F + 2E | mid-edge split + zero DS-extrude + midpoint elimination | 56, 84, 30 |

All rows satisfy `V′ − E′ + F′ = V − E + F` (checked symbolically and on the cube; pentagonalSubdivide on a tetrahedron gives 20/30/12 — the dodecahedron, a nice cross-check).

Crust and multi-connect (non-subdivision, χ-changing) are covered in their own sections at the end.

---

## Primitive operations referenced below (DLFL semantics)

These are the building blocks whose element-count effects everything else composes from:

- **insertEdge (cofacial)** — connect two corners of the *same* face: splits the face. ΔE=+1, ΔF=+1.
- **insertEdge (non-cofacial)** — connect corners of two *different* faces: merges them. ΔE=+1, ΔF=−1.
- **deleteEdge** — if its two sides are different faces, merges them: ΔE=−1, ΔF=−1. If both sides are the same face, splits it (or spawns a point-sphere): ΔE=−1, ΔF=+1.
- **subdivideEdge / subdivideAllEdges(k)** — split every edge into k segments: ΔV=+(k−1) per edge, ΔE=+(k−1) per edge, faces' degrees multiply accordingly.
- **trisectEdge(len)** — insert two vertices, each at distance `len` from one endpoint: ΔV=+2, ΔE=+2 per edge.
- **stellateFace(offset)** — apex at face centroid displaced `offset` along the normal, fanned to all corners: for a d-gon ΔV=+1, ΔE=+d, ΔF=+(d−1). Over all faces: V+F, 3E, 2E.
- **createFace(coords)** — creates a free-floating two-sided polygon: d new vertices, d edges, 2 faces (an outward and an inward orientation).
- **connectFaces(f1, f2)** — bridge two d-gon faces with a prism ring: both faces are consumed, d bridge edges and d quads created. ΔV=0, ΔE=+d, ΔF=+(d−2). Δχ=−2 (this is what punches holes / adds handles).
- **connectEdges(e1@f1, e2@f2)** — bridge two half-edges with a quad by two edge insertions: ΔE=+2, ΔF=0 net (one merge + one split). Δχ=−2 when the half-edges were on separate components/faces of a bridging arrangement; in Doo-Sabin-style usage the aggregate face count works out as derived per scheme.
- **extrudeFace / extrudeFacePlanarOffset (dist 0)** — inset: new d-gon parallel copy of the face connected by a ring of d quads; the original face region becomes the ring + top. ΔV=+d, ΔE=+2d, ΔF=+d. `thickness` sets the inset margin (planar offset inward within the face plane); "fractional" interprets it as a fraction of local edge length.
- **extrudeFaceDS(d, twist, sf)** — same element counts as a plain extrude (ΔV=+d, ΔE=+2d, ΔF=+d), but the new d-gon's vertices are computed with the Doo-Sabin averaging mask over the (optionally twist-interpolated) old boundary, then scaled by `sf` about their centroid. Returns the new top face (still a d-gon).

---

## 1. honeycombSubdivide

- **Precondition:** any closed polygonal mesh.
- **Construction:** for every face of degree d, build a brand-new d-gon whose vertices are weighted averages of that face's *edge midpoints* (mask: diagonal weight `α = 1/√2 − 1/4 + 5/(4d)`, off-diagonal weight `(1−α)(3 + 2cos(2π(i−j)/d)) / (3d−5)`). All original vertices, edges, and faces are then destroyed. Finally, for each original edge, a single edge is inserted joining the corresponding corners of the two new inner polygons that flank that edge.
- **Output faces:** one shrunken d-gon per old face, plus one 2n-gon per old vertex of valence n (the region enclosed by the inner-polygon corners and connecting edges around the vertex). Old edges yield *edges only*, not faces — this is the key difference from Doo-Sabin.
- **Counts:** `V′ = 2E` (one vertex per half-edge/corner-midpoint), `E′ = 2E (inner polygon edges) + E (connectors) = 3E`, `F′ = F + V`. χ preserved. Cube → 24/36/14 (6 quads + 8 hexagons). On triangle meshes the vertex faces are the "honeycomb" hexagons.
- **Composition:** topologically identical to `dual(stellate_all(M))` — stellation makes 2E triangles whose dual has 2E vertices, 3E edges, V+F faces. Only vertex positions differ.
- **Parameters:** none.

## 2a. pentagonalSubdivide (offset)

- **Precondition:** any closed polygonal mesh (faces of degree ≥ 2).
- **Construction:** (1) compute and cache each face's centroid; (2) split every edge into three equal parts (2 new vertices per edge, so each d-gon becomes a 3d-gon); (3) for each old face, create an isolated point-vertex at the cached centroid and insert spoke edges from it to *every third* corner of the 3d-gon — specifically to one trisection point per original edge, chosen consistently by rotation order so adjacent faces never pick the same trisection point. Each spoke insertion splits off a region bounded by the spoke vertex, three consecutive boundary corners, and the centroid — a pentagon. The `offset` parameter pulls the trisection vertex adjacent to each spoke toward the face centroid by fraction `offset`.
- **Output faces:** every old face of degree d becomes exactly d pentagons fanned around the centroid vertex. All output faces are pentagons.
- **Counts:** `V′ = V + 2E + F`, `E′ = 3E (trisected boundary) + 2E (spokes, one per corner) = 5E`, `F′ = Σ d_f = 2E`. χ preserved. Cube → 38/60/24. Tetrahedron → 20/30/12 = dodecahedron combinatorics.
- **Parameters:** `offset ∈ [0,1]` — geometric pull of spoke-adjacent points toward the centroid; topology unaffected.

## 2b. pentagonalSubdivide2 (scale_factor)

- **Precondition:** faces of original degree ≥ 3 (smaller faces are skipped).
- **Construction:** (1) split every edge at its midpoint (d-gon → 2d-gon); (2) per face, create a *new free-floating* d-gon whose vertices sit at the face's edge midpoints, scaled by `scale_factor` about their centroid (these are new vertices, not the midpoint vertices themselves); (3) insert d edges, one per midpoint, from each midpoint corner of the enlarged boundary to the matching corner of the inner polygon (traversed in reverse since the inward-facing copy is used). The first insertion merges the floating polygon into the face; the rest split off regions.
- **Output faces:** per old d-gon face: one inner d-gon plus d pentagons (each pentagon: midpoint, old vertex, next midpoint on the outer boundary + two inner-polygon corners). Old vertices survive with unchanged valence.
- **Counts:** `V′ = (V + E) + 2E = V + 3E`, `E′ = 2E (split boundary) + 2E (inner polygons) + 2E (connectors) = 6E`, `F′ = F (inner) + 2E (pentagons)`. χ preserved. Cube → 44/72/30 (6 quads + 24 pentagons).
- **Parameters:** `scale_factor` — inner polygon shrink about its centroid; topology unaffected.

## 3. cornerCuttingSubdivide (alpha), modifiedCornerCuttingSubdivide, modifiedCornerCuttingSubdivide2

- **Precondition:** any closed polygonal mesh.
- **Construction:** all three are *structurally identical to Doo-Sabin*: per face of degree d, create a new free-floating d-gon; destroy all old vertices/edges/faces; per old edge, bridge the corresponding half-edges of the two flanking inner polygons with a quad (two edge insertions each, via edge-connect). The three functions differ **only in where the inner polygon's vertices are placed**:
  - `cornerCuttingSubdivide(alpha)`: same trigonometric averaging mask as Doo-Sabin but with the diagonal weight exposed as the tension parameter `alpha` (off-diagonal `(1−alpha)(3+2cos(2π(i−j)/d))/(3d−5)`). Doo-Sabin's own mask corresponds to a fixed diagonal of `1/4 + 5/(4d)` (with a different off-diagonal normalization).
  - `modifiedCornerCuttingSubdivide(thickness)`: each new corner is the old corner `v0` displaced along the (un-normalized) angle bisector `n1 + n2` of its two incident face edges by magnitude `x = sqrt(t² / (1 − (n1·n2)²))` with `t = thickness/2`. This makes the *perpendicular distance* from each new edge to the old edge exactly `thickness/2`, giving uniform-width struts (used by the wireframe pipeline). Degenerate collinear corners (`n1·n2 = −1`) fall back to displacing toward the face centroid by `t`. It additionally tags each face's outward-facing duplicate as a "hole" face for later wireframe hole-punching (no topological effect here).
  - `modifiedCornerCuttingSubdivide2(scale)`: same, but the displacement magnitude is simply `scale` along `n1+n2` (no thickness compensation).
- **Counts (all three):** `V′ = 2E`, `E′ = 2E (inner polygons) + 2E (bridge quads: 2 per old edge) = 4E`, `F′ = F (inner faces) + E (edge quads) + V (vertex faces, valence-n-gons)`. χ preserved. Cube → 24/48/26.
- **Relation:** pure **geometric variants of `doo_sabin`** — reuse our Doo-Sabin topology with pluggable vertex placement.

## 4. root4Subdivide (a, twist)

- **Precondition:** any closed polygonal mesh (named for its √4 behavior on triangle meshes).
- **Construction:** (1) per face of degree d, compute "twisted" boundary samples `m_i = (1−twist)·p_i + twist·p_{i+1}` (twist 0 → the old corners themselves; twist 0.5 → edge midpoints), then apply the honeycomb averaging mask (diagonal `1/√2 − 1/4 + 5/(4d)`) to get a new inner d-gon, created as a free-floating polygon; (2) bridge each old face to its inner polygon with a full prism ring (d bridge edges + d quads, consuming the old face and the inward copy — i.e., a zero-height extrusion topology); (3) delete **all original edges** — each deletion merges the two side-quads flanking that edge into one hexagon; (4) reposition each old vertex `p` of valence n as `p′ = a·q + (1−a)·p`, where `q = (2·Σ_incident-edge-midpoints − n·p)/n` (i.e., the average of the opposite endpoints of its incident edges).
- **Output faces:** one inner d-gon per old face + one hexagon per old edge (two merged bridge quads; its corners are the old edge's two endpoints plus two inner-polygon corners on each side). Old vertices survive (valence unchanged: they lose n old edges but keep n bridge edges).
- **Counts:** `V′ = V + 2E`, `E′ = E + 4E − E = 4E` (per face: d polygon edges + d bridge edges; minus deleted old E), `F′ = F + Σd − Σ(over edges)1 = F + 2E − E = F + E`. χ preserved. Cube → 32/48/18 (6 quads + 12 hexagons).
- **Parameters:** `twist` — slides inner-ring sample points along the boundary edges (geometric); `a` — old-vertex smoothing blend toward the neighbor average (0 = keep, 1 = fully smoothed).

## 5. starSubdivide (offset)

- **Precondition:** any closed polygonal mesh.
- **Construction:** record each face's normal; stellate every face with a centroid apex (zero displacement); then stellate every resulting triangle again; finally displace each **first-round** apex by `offset` along its original face's normal.
- **Counts:** exactly two applications of `stellate_all`:
  - after round 1: `V+F, 3E, 2E`;
  - after round 2: `V′ = V + F + 2E`, `E′ = 9E`, `F′ = 6E`.
  χ preserved. Cube → 38/108/72 (all triangles).
- **Relation:** = `stellate_all ∘ stellate_all`; the `offset` only affects first-round apex geometry.

## 6. fractalSubdivide (offset)

- **Precondition:** any closed polygonal mesh (degree ≥ 3).
- **Construction:** (1) split every edge at its midpoint; (2) per old face of degree d: create an isolated apex vertex at `centroid + h·normal` where `h = offset · sqrt(L2² − L1²)` (`L2` = distance between two consecutive midpoints, `L1` = half the distance across the face between opposite corners — a heuristic height); then, walking the 2d-gon boundary, insert midpoint-to-midpoint chords that cut off each old-vertex corner as a triangle, interleaved with spokes from the apex to each midpoint. Result per face: d corner triangles around the rim and d apex triangles forming a pyramid over the midpoint polygon.
- **Counts:** `V′ = V + E + F`, `E′ = 2E (split boundary) + 2E (chords, d per face) + 2E (spokes, d per face) = 6E`, `F′ = 2E (corner triangles) + 2E (apex triangles) = 4E`. χ preserved. Cube → 26/72/48 (all triangles).
- **Composition:** topologically = **loopStyle split (below) followed by stellating each central midpoint-polygon** with an offset apex. Sanity: loopStyle gives (V+E, 4E, F+2E); stellating the F central d-gons adds (F, 2E, 2E−F) → (V+E+F, 6E, 4E). ✓
- **Parameters:** `offset` — apex height scale along the face normal (geometric only).

## 7. domeSubdivide (length, sf)

- **Precondition:** any closed polygonal mesh.
- **Construction:** (1) split every edge into 4 equal segments (3 new vertices per edge; d-gon → 4d-gon); (2) on every old face (now a 4d-gon), perform **seven successive Doo-Sabin-style extrusions** (each: new DS-averaged 4d-gon connected by a prism ring), with hard-coded profile — heights `0, 0.3, 0.18, 0.1, 0.05, 0.025, 0.01` × `length` and scale factors `1.6, 1.7, 1.6, 1.4, 1.2, 1.1, 0.01` × `sf` — producing a rounded dome capped by a tiny near-degenerate face on each old face.
- **Counts:** edge quadrisection: `(V+3E, 4E, F)`; each DS extrusion of a 4d-gon adds `(4d, 8d, 4d)`; 7 of them per face, `Σ 4d = 8E` per round: `V′ = V + 3E + 7·8E = V + 59E`, `E′ = 4E + 7·16E = 116E`, `F′ = F + 7·8E = F + 56E`. χ preserved. Cube → 716/1392/678.
- **Relation:** pure composition: `subdivide_all_edges(4)` + 7 × per-face DS-extrude (we need an `extrude_face_ds` op: element counts identical to plain extrude, vertex placement = DS mask + scale).
- **Parameters:** `length` scales the height profile, `sf` scales the shrink profile; both geometric only.

## 8. dual1264Subdivide (sf)

- **Precondition:** any closed polygonal mesh.
- **Construction:** Doo-Sabin-like, but per face of degree d the new free-floating inner polygon is a **2d-gon** whose vertices are the points at 1/3 and 2/3 along each boundary edge (per-face copies), optionally scaled by `sf` about their centroid. Old vertices/edges/faces are destroyed; per old edge, the two flanking inner polygons are bridged with a quad (two edge insertions), where the connected inner edge is the "middle-third" segment aligned with the old edge.
- **Output faces:** one 2d-gon per old face, one quad per old edge, one 2n-gon per old vertex of valence n. (On triangle meshes: hexagons/quads/2n-gons — the "12.6.4-like" pattern that names the scheme.)
- **Counts:** `V′ = Σ 2d = 4E`, `E′ = 4E (inner polygons) + 2E (bridges) = 6E`, `F′ = F + E + V`. χ preserved. Cube → 48/72/26 (6 octagons + 12 quads + 8 hexagons; degree sum 144 = 2·72 ✓).
- **Relation:** same face-classification as Doo-Sabin (`F+E+V` faces) but with doubled inner degree; implementable as our Doo-Sabin topology generator parameterized by "corners per face boundary edge = 2".
- **Parameters:** `sf` — inner-polygon scale (geometric only).

## 9. checkerBoardRemeshing (thickness)

- **Precondition:** any closed polygonal mesh; `thickness` clamped to (0, 0.5], default 0.25 if negative (0.5 and 0 give coincident geometry).
- **Construction:** (1) inset every face (zero-height planar-offset extrusion with margin = `thickness` × local edge length): adds an inner d-gon and a ring of d side quads per face; original boundary edges remain shared between the rings of adjacent faces; (2) trisect every *original* edge at `thickness`·length from each end (2 new "checker" vertices per edge); (3) for every original vertex v (current valence 2n: n boundary edges now ending at checker points + n inset spokes): insert a corner-cutting chord across each of its 2n corners (prev→next), then delete every incident edge whose far endpoint is *not* a checker point — i.e., the n inset spokes — merging faces.
- **Output:** the classic checkerboard: each face keeps its inner d-gon; around it, edge- and corner-regions alternate diagonally. For an all-quad input the output is all quads.
- **Counts:** inset: `(V+2E, 5E, F+2E)`; trisect: `(V+4E, 7E, F+2E)`; corner chords: +4E edges/+4E faces; spoke deletions: −2E edges/−2E faces. Total: `V′ = V + 4E`, `E′ = 9E`, `F′ = F + 4E`. χ preserved. Cube → 56/108/54 (all quads: 216 = 4·54 ✓).
- **Parameters:** `thickness` — fraction of edge length controlling both the inset margin and the trisection offsets (geometric only).

## 10. loopStyleSubdivide (length)

- **Precondition:** any closed polygonal mesh.
- **Construction:** (1) split every edge at its midpoint; (2) per face (now a 2d-gon), starting from a midpoint corner, insert chords between corners two apart, walking around the face — each chord cuts off one old-vertex corner as a triangle, leaving a central d-gon of midpoints; (3) move each **old** vertex to `length·p_old + (1−length)·(average of its adjacent midpoints)`.
- **Output faces:** per old d-gon: d corner triangles + one central midpoint d-gon. On a triangle mesh this is exactly the Loop / 1-to-4 connectivity.
- **Counts:** `V′ = V + E`, `E′ = 2E + 2E = 4E`, `F′ = F + 2E`. χ preserved. Cube → 20/48/30.
- **Relation:** same connectivity oracle as our `loop`; differs only in vertex placement (simple midpoint + `length`-blend instead of Loop's β-weights) and in generalizing to non-triangle faces (central face is a d-gon, not a triangle).
- **Parameters:** `length ∈ [0,1]` — 1 keeps old vertices fixed, 0 snaps them to the midpoint average.

## 11. dooSabinSubdivideBCNew (sf, length) — vs. plain Doo-Sabin

Plain Doo-Sabin (for contrast): per face one DS-mask inner d-gon, old elements destroyed, edge-connect bridges; oracle `(2E, 4E, V+E+F)`; original vertices do *not* survive.

**BCNew differences:**
- Pipeline: (1) mark old vertices; split every edge at its midpoint; mark midpoints; (2) per face perform a **zero-height DS extrusion** with scale `sf` (instead of the delete-and-bridge construction) — old faces and old vertices remain in the mesh; (3) delete all pre-extrusion edges (merging side-quads pairwise across each old edge); (4) eliminate every midpoint vertex: insert a bypass chord between its two neighboring corners, then delete the midpoint's two remaining edges (the midpoint degenerates to a point-sphere and is cleaned up); (5) reposition old vertices with the same `length` blend as loopStyle (toward the average of their adjacent new points).
- Net effect: like Doo-Sabin, each face gets a shrunk DS polygon — but computed from the **mid-edge-refined** boundary (a 2d-gon), and the **original vertices survive** as mesh vertices, so edge faces are octagon-like (containing the two old endpoints) instead of quads.
- **Counts:** `V′ = V + 4E` (old V + 2d DS corners per face = 4E; midpoints removed), `E′ = 7E`, `F′ = F + 2E` (per old face one 2d-gon; per old edge one merged face containing both old endpoints; per old vertex one valence-gon). χ preserved. Cube → 56/84/30 (6 octagons + 12 octagonal edge faces + 8 triangles + retained vertices; degree sum 168 = 2·84 ✓).
- **Parameters:** `sf` — DS extrusion scale; `length` — old-vertex blend (1 = keep).

(For completeness: `dooSabinSubdivideBC` = mid-edge split of all edges followed by plain Doo-Sabin, i.e., a straight composition of two ops we already have.)

---

## Crust / shell creation (DLFLCrust.cc)

### createCrust(thickness, uniform) / createCrustWithScaling(scale_factor)

- **Construction:** serialize the whole object with all faces **reversed** and re-read it appended to itself — producing a second, disjoint, orientation-reversed copy of the entire surface (the inner shell). Then offset one copy:
  - thickness > 0: move the *new* copy inward along per-vertex average normals by `thickness`; thickness < 0: move the *original* vertices outward instead.
  - `uniform=true`: per-vertex distance is corrected by averaging `thickness / (avg_normal · corner_normal)` over the vertex's corner normals so the wall thickness stays uniform at creases.
  - Scaling variant: instead of normal offset, scale one copy about the object centroid by `|scale_factor|` (clamped to [−1,1]; negative scales the original copy up by `1/|sf|`).
  - Bookkeeping: parallel arrays pair each outer face with its mirrored inner face (by creation order) for later hole punching.
- **Counts after createCrust alone:** `V′ = 2V, E′ = 2E, F′ = 2F` — two nested disjoint closed surfaces, `χ′ = 2χ`, 2 connected components. No holes yet.

### Hole punching (cmMakeHole / punchHoles) — the crustModeling pipeline

- For each user-tagged face, take the (outer face, mirror inner face) pair and bridge them with `connectFaces`: both d-gon faces are consumed and replaced by d side quads + d edges, creating a tunnel through the crust wall. Optional cleanup deletes infinitesimally thin faces that appear when two punched holes share an original edge (detected as two bridge edges flanked by the same two faces; the whole degenerate face's edge set is deleted).
- **Per hole (d-gon):** ΔV=0, ΔE=+d, ΔF=+(d−2), Δχ=−2.
- **After k ≥ 1 holes on a connected genus-g input:** one connected component, `χ′ = 2(2−2g) − 2k`, hence **genus g′ = 2g + k − 1**. Sanity: sphere + 1 hole → thickened disc ≅ sphere (g′=0); sphere + 2 holes → torus (g′=1); all 6 faces of a cube crust punched → g′=5.
- `createCrustForWireframe(2)` are the same duplication with specialized vertex-offset geometry (offset direction derived from the local hole-face frame; used by `makeWireframe` = modifiedCornerCutting → crust → punch all hole-tagged faces). Topology identical to createCrust + punchHoles.

## multiConnectFaces (DLFLMultiConnect.cc)

Three variants; all connect k ≥ 2 selected faces:

1. **Half-edge pairing variant** `multiConnectFaces(obj, faces)`: collect all boundary edges of the selected faces; enumerate candidate half-edge pairs lying on *different* selected faces, filtered by (a) near-coplanarity of the two half-edges (planarity above cos 5°) and (b) the pair's plane being roughly perpendicular to the plane spanned by the face centroids about the global centroid; sort by midpoint distance; greedily bridge the closest pairs via `connectEdges` (a quad per pair, built from 2 edge insertions), discarding any remaining candidate that reuses a connected edge; finally run edge-cleanup on the newly inserted edges to remove redundant/2-gon edges.
   - **Element effect:** per accepted pair: ΔV=0, ΔE=+2 (before cleanup), ΔF=0 net, **Δχ=−2** (each bridge is a handle/tunnel between two faces). The number of accepted pairs m is data-dependent (greedy geometric matching, at most ⌊(total candidate edges)/2⌋), so no input-only closed form exists; the oracle is `χ′ = χ − 2m` with V unchanged.
2. **Convex-hull variant** `(faces, scale_factor, extrude_dist, use_max_offsets)`: gather the distinct vertices of the selected faces (optionally offset each face along its normal by `extrude_dist` or by automatically computed maximal non-intersecting offsets, and/or scaled about their centroid by `scale_factor`); build their convex hull as a separate object; reverse it if it faces the wrong way; splice it in; then for each selected face find the hull face that is exactly antiparallel to it and bridge the two with `connectFaces` (prism ring).
   - **Element effect:** `V′ = V + V_hull`, `E′ = E + E_hull + Σ dᵢ`, `F′ = F + F_hull + Σ (dᵢ − 2)` over the matched pairs; χ: hull contributes +2, each of the k bridges −2 → for k matched faces on one component, k−1 handles are created.
3. **Adaptive-hull variant** `(faces, min_factor, make_connections)`: same as (2) but the hull vertex positions are found by a bounded binary search on a translation factor toward the global centroid, seeking the smallest hull still containing all original points; then identical matching + bridging.

Related in the same file (useful context, same counting rules): `multiConnectMidpoints` (mid-edge split, corner chords without deleting old edges, plus one edge per midpoint inside the middle faces) and `multiConnectCrust` / `modifiedMultiConnectCrust` / `createSponge` (scaled crust + per-face zero-length extrusion + `connectFaces` per face pair → Menger-sponge-style: k = F holes ⇒ genus 2g + F − 1, before optional edge collapsing of short/self-intersecting bridge edges, which is geometric cleanup that can further merge elements).

---

## Implementation notes for our oracle tests

- All 13 subdivision oracles above are linear in (V, E, F) and were verified symbolically against Euler's formula and numerically on the cube (and tetrahedron for pentagonal).
- Recommended oracle test style: assert (V′,E′,F′) exactly, plus χ invariance, plus the face-degree census where stated (e.g., pentagonal → all faces degree 5; star/fractal → all degree 3; checkerboard on quad input → all degree 4).
- Crust tests: assert element doubling, component count 2 (k=0), and genus `2g + k − 1` after punching k holes.
