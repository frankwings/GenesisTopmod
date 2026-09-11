"""Dissect WHY carve candidates get p->0: per-term loss delta at p=1 vs p=0.

For each of the top red candidates (K=1, no other ops):
  d_sil   = sum over views of region-restricted |S_mix-GT| at p=1 minus p=0
  d_depth = sum over views of (e_k - e_base)
Negative delta = term votes FOR the op; positive = votes AGAINST.
Also break d_sil into views to see where the damage happens.
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
import nvdiffrast.torch as dr
import phase1c_pipeline as P
from phase1b_pipeline import render_sil_and_ids
from eval_extrude_v3 import render_sil_and_depth
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
import run_64v

DEVICE = "cuda"
SHAPE = "armadillo"
BASE_NPZ = "/tmp/liou_cow_viz/cow_armadillo_64v.npz"

ctx = dr.RasterizeCudaContext()
gv, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
mvps, views = run_64v.star_cameras(max_r)
gt, gtd, _, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
gt_sils = [torch.from_numpy((gt[i] < 128).astype(np.float32)).to(DEVICE) for i in range(64)]
gt_deps = [torch.from_numpy(np.asarray(gtd[i], np.float32)).to(DEVICE) for i in range(64)]
gt_fgs = [(g > 0.5) for g in gt_sils]

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
es = set()
for a, b, c in Fa:
    for e in ((a, b), (b, c), (c, a)): es.add((min(e), max(e)))
src = np.array([e[0] for e in es]); dst = np.array([e[1] for e in es])
me = float(np.linalg.norm(V[src] - V[dst], axis=1).mean())

cands = P.find_candidates(ctx, V, Fa, [(g > 0.5).cpu().numpy() for g in gt_sils],
                          mvps, me)
print(f"{len(cands)} candidates; dissecting top 4 red + the near-winner:")
picks = [c for c in cands if c[2] < 0][:4]
nw = [c for c in cands if c[1] == 2770]
if nw and nw[0] not in picks: picks.append(nw[0])

for sz, fi, sign in picks:
    ln = P.OP_LEN if sign > 0 else P.CARVE_LEN
    V1, F1, vidx = P.dlfl_extrude_arrays(V, Fa, fi, sign * ln * me)
    vt = torch.tensor(V1, dtype=torch.float32, device=DEVICE)
    f_all = torch.tensor(F1.astype(np.int32), dtype=torch.int32, device=DEVICE)
    f_base = torch.tensor(Fa.astype(np.int32), dtype=torch.int32, device=DEVICE)
    d_sil_tot = d_dep_tot = 0.0
    worst = []
    with torch.no_grad():
        for i in range(64):
            s0, _ = render_sil_and_ids(ctx, vt, f_base, mvps[i])
            s1, _ = render_sil_and_ids(ctx, vt, f_all, mvps[i])
            gt_i = gt_sils[i]
            region = (((gt_i > 0.01) & (s0 < 0.99)) | ((gt_i < 0.99) & (s0 > 0.01))
                      | ((s1 - s0).abs() > 0.01))
            n = region.sum() + 1
            e0 = (s0 - gt_i).abs()[region].sum() / n
            e1 = (s1 - gt_i).abs()[region].sum() / n
            d_sil = float(e1 - e0)
            _, dz0, fg0 = render_sil_and_depth(ctx, vt, f_base, mvps[i])
            _, dz1, fg1 = render_sil_and_depth(ctx, vt, f_all, mvps[i])
            m0 = fg0 & gt_fgs[i]; m1 = fg1 & gt_fgs[i]
            eb = (dz0 - gt_deps[i]).abs()[m0].mean() if m0.any() else torch.tensor(0.)
            ek = (dz1 - gt_deps[i]).abs()[m1].sum() / (m1.sum() + 1)
            d_dep = float(ek - eb)
            d_sil_tot += d_sil; d_dep_tot += d_dep
            worst.append((d_sil + d_dep, i, d_sil, d_dep))
    worst.sort(reverse=True)
    tag = "grow" if sign > 0 else "carve"
    print(f"\nface {fi} ({tag}, blob {sz}px): "
          f"d_sil={d_sil_tot:+.5f}  d_depth={d_dep_tot:+.5f}  "
          f"total={'REJECT' if d_sil_tot + d_dep_tot > 0 else 'ACCEPT'}")
    for t, i, ds, dd in worst[:3]:
        print(f"   worst view {i:2d}: d_sil={ds:+.5f} d_depth={dd:+.5f}")
    for t, i, ds, dd in worst[-2:]:
        print(f"   best  view {i:2d}: d_sil={ds:+.5f} d_depth={dd:+.5f}")
