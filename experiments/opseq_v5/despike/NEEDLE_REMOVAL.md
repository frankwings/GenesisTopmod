# Needle Removal — GT-Mismatch Seed + Thickness Propagation (v22)

Final algorithm for eliminating 毛刺 (needle/fin artifacts) when fitting a
subdivided icosphere to 6-view silhouette+depth targets, under the hard
constraint: **only the 6 training views may supervise removal** — no held-out
view, no "burn the artifact by adding more views". The 16 held-out views are
exam-only.

## The problem

The C2F fit (`cow_v13.py`: cc2→subdiv→cc3→subdiv→cc4) grows a straight, thin
**tube** (needle) at the shoulder. It is born during cc2 (steps ~100–300, no
prior there yet) and frozen by subsequent subdivision. Two facts make it hard
to kill without collateral damage:

1. **The needle is silhouette-supported.** It fits the 6 training views well,
   so an IoU-based gate (v18) exempts it — it comes right back.
2. **The mature needle is a straight tube, not a fold.** A dihedral/fold gate
   (v19) misses it — near-parallel adjacent faces, no antiparallel pair.
3. **The needle and the ear are BOTH thin.** A blanket thickness prior at
   `TUBE_THR=0.4` (v17) kills the needle but also blunts the ear. Lowering to
   `TUBE_THR=0.25` (v20) spares the ear but is a fragile hand-tuned proxy.

## Key empirical finding

Projected into the 6 training cameras, the needle **only pokes its TIP outside
the GT silhouette** (15–281 px/view, ~195 verts total). Its **shaft lies buried
inside the body silhouette** — that is exactly why it survived training. So a
per-vertex "outside GT" test (v21, pure B) only ever flags the tip; amputating
the tip leaves the shaft, which regrows on the next settle. v21 result:
hair_px 32956 (3.4× worse than v20), ho16 0.9091 — FAILED.

## The v22 algorithm

Separate the two roles:

- **SEED = GT mismatch (2D).** In each of the 6 training views, the pixels
  where `pred_silhouette AND NOT dilate(GT_silhouette)` are the "red" mismatch
  (extra area the target does not have). A vertex projecting into such a pixel
  is an **escape seed**. (Green mismatch — GT present, pred missing — is the
  opposite defect, under-fit; NOT used for deletion.) `escape_util.escape_mask`.

- **EXTENT = thickness (3D).** The whole needle is thin end-to-end
  (`detect()` / `tube_mask` with `TUBE_THR=0.4`, blanket). This is the same
  thickness rule as v20 — but here it defines *how far to remove*, not
  *whether to remove*.

- **PROPAGATION binds them.** Group the thin vertices into connected components
  (1-hop adjacency). **Condemn a whole thin component iff it contains at least
  one escape seed.** The needle: its tip escapes → the seed propagates over the
  thin shaft → the entire needle (tip + buried shaft) is condemned → root-and-
  all removal. The ear: thin, but its component has NO escaping tip (it stays
  inside GT) → spared. `surgery_lib._propagate_flag`.

```
condemn(component) = is_thin(component) AND (∃ vertex in component that escapes GT)
```

This is applied in TWO places, both fed only by the 6 training views:

1. **Training tube prior** (`cow_v13.tube_mask` monkeypatched in `cow_v22.py`):
   every `TUBE_EVERY` steps, pull toward the local centroid only those thin
   vertices whose thin-component contains an escape seed. Prevents the needle
   at/after birth without shrinking the ear.
2. **Surgery detector** (`surgery_lib.surgery(..., escape_fn=...)`): condemn
   `(thin|spike)` components that contain an escape seed, then run the
   χ-preserving TopMod `collapse_edge_tri` amputation with ring-growth and a
   train6-IoU budget guard.

Because escape+propagation supplies the discrimination, `TUBE_THR` returns to
0.4 (blanket thin) — the threshold no longer has to separate ear from needle.

## Results (cow, 256px, 6 train / 16 held-out)

| version | discriminator | train6 | ho16 | hair_px | maxblob |
|---------|---------------|--------|------|---------|---------|
| v17 | blanket thin thr0.4 | 0.9761 | 0.9477 | 11332 | 1156 |
| v18 | IoU gate | — | 0.9066 | (needle back) | — |
| v19 | fold gate | — | 0.8962 | (needle back) | — |
| v20 | thickness thr0.25 | 0.9789 | 0.9438 | 9544 | — |
| v21 | pure B (per-vertex escape) | 0.9790 | 0.9091 | 32956 | 2842 |
| **v22** | **seed+propagate** | **0.9813** | **0.9568** | **9100** | **721** |

v22 wins on every metric simultaneously: needle gone (like v20), ear preserved
(unlike v20), lowest hair and 3–4× smaller max blob.

## Files

- `escape_util.py` — `escape_mask` (project verts to 6 training views, flag
  those outside dilated GT). Pixel convention calibrated: `col=x, row=y`, no
  y-flip (94.9% of GT verts land inside their own silhouette).
- `surgery_lib.py` — `detect` (thin/spike), `_propagate_flag` (component seed
  propagation), `surgery` (greedy incremental amputation with escape gate).
- `cow_v13.py` — C2F base recipe + online spike/sliver/tube priors.
- `cow_v22.py` — winning pipeline: escape-seeded, thin-propagated tube prior in
  training + escape-gated surgery. Supervision = 6 training views only.
- `topmod/high_level_ops.py::collapse_edge_tri` — χ-preserving, link-condition
  guarded edge collapse with duplicate-edge merging (the amputation primitive).

## Why this obeys the constraint

Removal is decided by projecting the candidate geometry against the **6 training
GT silhouettes** and deleting only what is inconsistent with them ("project the
needle vs GT, then remove"), not by declaring any protrusion an artifact and not
by consulting any held-out view.
