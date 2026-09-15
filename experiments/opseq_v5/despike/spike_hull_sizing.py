#!/usr/bin/env python3
"""THROWAWAY SPIKE: hull-thickness vs actual edge length on the golden v6 meshes.
Question: how many vertices would a thickness-driven sizing field L=clamp(c*T, 1.3px, Lmax) need vs the ~50k we use now?"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch, nvdiffrast.torch as dr
from scipy import ndimage
import run_64v
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
from hull_field import build_vote_hull
from hull_locate import clean_hull

SHAPES = os.environ.get("SHAPES", "armadillo kitten fertility rockerarm threeholes").split()
TAG = os.environ.get("TAGP", "v6c")
ctx = dr.RasterizeCudaContext()
out = {}
for S in SHAPES:
    t0 = time.time()
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{S}.obj")); gvn = normalize_to_range(gv)
    R = float(np.linalg.norm(gvn, axis=1).max()); mvps, views = run_64v.star_cameras(R); PX = 2.0 * R / run_64v.TRAIN_RES
    d = np.load(f"despike/results_genus/{S}_{TAG}_raw.npz"); V = d["verts"].astype(np.float64); F = d["tris"].astype(np.int64)
    HF = build_vote_hull(ctx, mvps, gvn, gf, V, "cuda", nres=256, hires=512, vote=2)
    hull = clean_hull(np.asarray(HF.hull).astype(bool)); lo = np.asarray(HF.lo, float); hi = np.asarray(HF.hi, float); sp = (hi - lo) / 255.0
    edt = ndimage.distance_transform_edt(hull, sampling=tuple(sp))          # inside distance, world units
    # outward vertex normals (area weighted)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); fa = 0.5 * np.linalg.norm(fn, axis=1)
    vn = np.zeros_like(V); Av = np.zeros(len(V))
    for k in range(3): np.add.at(vn, F[:, k], fn); np.add.at(Av, F[:, k], fa / 3.0)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    # thickness: walk inward along -n, sample EDT, first peak -> T = 2*peak
    step = float(sp.min()) * 0.5; nst = int(0.8 * R / step)
    ts = np.arange(1, nst + 1) * step
    P = V[:, None, :] - vn[:, None, :] * ts[None, :, None]                         # [Nv, nst, 3]
    g = (P - lo) / (hi - lo) * 255.0
    e = ndimage.map_coordinates(edt, g.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(V), nst)
    # first peak: running max stops growing for 4 samples
    rm = np.maximum.accumulate(e, axis=1)
    grow = np.diff(rm, axis=1) > 1e-9
    # index of first sample after which no growth for 4 consecutive samples
    win = 4
    stalled = np.ones((len(V), nst - 1 - win + 1), bool)
    for w in range(win): stalled &= ~grow[:, w:grow.shape[1] - win + 1 + w]
    first = np.where(stalled.any(1), stalled.argmax(1), nst - win)
    T = 2.0 * rm[np.arange(len(V)), first]
    # local edge length
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    le = np.zeros(len(V)); cnt = np.zeros(len(V)); np.add.at(le, E[:, 0], el); np.add.at(cnt, E[:, 0], 1); np.add.at(le, E[:, 1], el); np.add.at(cnt, E[:, 1], 1); le /= np.maximum(cnt, 1)
    out[S] = dict(T=T, le=le, Av=Av, PX=PX, R=R, nV=len(V), nF=len(F), hull_vox=int(hull.sum()), edt_max=float(edt.max()))
    print(f"[{S}] V={len(V)} F={len(F)} PX={PX:.4f} T px: min/med/max={T.min()/PX:.1f}/{np.median(T)/PX:.1f}/{T.max()/PX:.1f} | le px: min/med/max={le.min()/PX:.2f}/{np.median(le)/PX:.2f}/{le.max()/PX:.2f} | {time.time()-t0:.0f}s", flush=True)
np.savez_compressed("despike/results_genus/hull_sizing_spike.npz", **{f"{S}_{k}": v for S, d in out.items() for k, v in d.items()})

# ---- analysis ----
K = np.sqrt(3) / 2   # area per vertex for an equilateral mesh of edge L: 2 tri * sqrt(3)/4 L^2
print("\n==== calibration: V_pred(L=le) vs actual V ====")
for S, d in out.items():
    vp = (d["Av"] / (K * d["le"] ** 2)).sum(); print(f"[{S}] V_pred(le)={vp:.0f} actual={d['nV']} ratio={d['nV']/vp:.3f}")
print("\n==== thickness bins (px): vertex share, median le (px), median le/T ====")
bins = [0, 4, 8, 16, 32, 64, 1e9]
for S, d in out.items():
    Tp = d["T"] / d["PX"]; lp = d["le"] / d["PX"]; row = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (Tp >= a) & (Tp < b)
        row.append(f"[{a:.0f},{b if b<1e8 else 'inf'}): {m.mean()*100:4.1f}% le={np.median(lp[m]) if m.any() else 0:.2f} le/T={np.median(lp[m]/Tp[m]) if m.any() else 0:.3f}")
    print(f"[{S}] " + " | ".join(row))
print("\n==== predicted V for L=clamp(c*T, 1.3px, Lmax) (Lmax = 99th pct of current le); calibrated by ratio above ====")
cs = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
print("shape      actualV  " + "  ".join(f"c={c:<5}" for c in cs) + "  | floor-bound share @c=0.3")
for S, d in out.items():
    cal = d["nV"] / (d["Av"] / (K * d["le"] ** 2)).sum(); Lmax = np.quantile(d["le"], 0.99); fl = 1.3 * d["PX"]; row = []
    for c in cs:
        L = np.clip(c * d["T"], fl, Lmax); row.append(f"{cal*(d['Av']/(K*L**2)).sum():7.0f}")
    L3 = np.clip(0.3 * d["T"], fl, Lmax)
    print(f"{S:10s} {d['nV']:7d}  " + "  ".join(row) + f"  | {(L3<=fl+1e-12).mean()*100:.0f}% at floor, Lmax={Lmax/d['PX']:.1f}px")
