#!/usr/bin/env python3
"""strict_repair.py - remove handles that realise no hull tunnel (2026-10-01, strict topology).

A v6.4 chain can reach genus g* with a hidden MICRO-HANDLE: a triangle-sized tube whose loops link no hull tunnel
(LESSONS 7e). On the refined mesh such a tube shows up as a NON-FACE 3-CYCLE: three edges a-b, b-c, c-a that
exist in the mesh while (a, b, c) is not a face - the neck of the tube. If cutting along it does not disconnect
the surface (non-separating = it is a handle, not a pinch between two parts) and its linking vector with the hull
tunnel air loops is zero, the handle is spurious: cut along the cycle and cap both sides with one triangle each.
Closed, orientable, manifold again; genus - 1; no real tunnel touched.

Usage: SHAPE=fertility python3 despike/strict_repair.py in.npz out.npz      (out == in copy when nothing to do)
"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import linking_audit as la

def nonface_3cycles(F):
    nb = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)): nb.setdefault(int(u), set()).add(int(v)); nb.setdefault(int(v), set()).add(int(u))
    faces = {tuple(sorted(map(int, f))) for f in F}; out = set()
    for a in nb:
        for b in nb[a]:
            if b <= a: continue
            for c in nb[a] & nb[b]:
                if c > b and (a, b, c) not in faces: out.add((a, b, c))
    return sorted(out)

def cut_and_cap(V, F, cyc):
    """Cut the closed triangle mesh along the 3-cycle (a,b,c) and cap both sides. Returns (V, F) or None if the
    cycle is not a clean simple cut (non-manifold fan) or separates the surface."""
    a, b, c = cyc; F = np.asarray(F, np.int64)
    dire = {}
    for fi, f in enumerate(F):
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])): dire[(int(u), int(v))] = fi
    if not all((u, v) in dire and (v, u) in dire for u, v in ((a, b), (b, c), (c, a))): return None
    vf = {x: [fi for fi in range(len(F)) if x in F[fi]] for x in (a, b, c)} if len(F) < 2000 else None
    if vf is None:
        vf = {x: np.where((F == x).any(1))[0].tolist() for x in (a, b, c)}
    left = set()
    for x, nxt, prv in ((a, b, c), (b, c, a), (c, a, b)):          # loop direction a -> b -> c -> a; at x: in from prv, out to nxt
        # walk the fan of x from the left face of (x, nxt) until reaching the edge (x, prv)
        f = dire[(x, nxt)]; seen = 0
        while True:
            left.add((x, f)); tri = [int(t) for t in F[f]]; k = tri.index(x); w = tri[(k + 2) % 3]      # face (x, u, w): next edge around x is (x, w)
            if w == prv: break
            f = dire.get((x, w));  seen += 1
            if f is None or seen > 10000: return None
    V2 = np.vstack([V, V[[a, b, c]]]); n = len(V); new = {a: n, b: n + 1, c: n + 2}; F2 = F.copy()
    for x in (a, b, c):
        for f in vf[x]:
            if (x, f) not in left: F2[f][F2[f] == x] = new[x]      # right side gets the duplicated vertex
    F2 = np.vstack([F2, [[a, c, b]], [[new[a], new[b], new[c]]]])
    # manifold + still one component?
    E = np.sort(np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]), 1); _, cnt = np.unique(E, axis=0, return_counts=True)
    if not (cnt == 2).all(): return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    used = np.unique(F2); G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V2), len(V2)))
    ncomp = len(np.unique(connected_components(G, directed=False)[1][used]))
    if ncomp != 1: return None
    # pull the two caps slightly apart (towards their own 1-ring) so they are not exactly coincident
    for x in (a, b, c):
        for y in (x, new[x]):
            ring = np.unique(F2[(F2 == y).any(1)]); ring = ring[~np.isin(ring, [a, b, c, n, n + 1, n + 2])]
            if len(ring): V2[y] = V2[y] + 0.3 * (V2[ring].mean(0) - V2[y])
    return V2, F2

def repair(V, F, AIR, log=print):
    removed = 0
    for _ in range(8):
        a0 = la.audit(V, F, AIR)
        if a0["genus"] <= a0["rank"]: break
        done = False
        for cyc in nonface_3cycles(F):
            P = V[list(cyc)]; vec = np.round([la.linking(P, A) for A in AIR])
            if np.any(vec != 0): continue                                   # this neck encloses a real tunnel: keep
            res = cut_and_cap(V, F, cyc)
            if res is None: continue                                         # separating pinch or irregular: not a handle
            V2, F2 = res; a1 = la.audit(V2, F2, AIR)
            if a1["genus"] == a0["genus"] - 1 and a1["rank"] == a0["rank"]:
                log(f"[repair] removed micro-handle at {np.round(P.mean(0), 3)} (cycle {cyc}, perimeter {np.linalg.norm(P - np.roll(P, 1, 0), axis=1).sum():.3f}): genus {a0['genus']} -> {a1['genus']}, tunnels realised {a1['rank']}")
                V, F = V2, F2; removed += 1; done = True; break
        if not done: log(f"[repair] genus {a0['genus']} > tunnels realised {a0['rank']} but no removable micro-handle found"); break
    return V, F, removed

if __name__ == "__main__":
    shape = os.environ.get("SHAPE", "fertility"); src, dst = sys.argv[1], sys.argv[2]
    m = np.load(src); V, F = m["verts"].astype(float), m["tris"].astype(np.int64)
    try: AIR = la.air_loops(shape, log=lambda s: None)
    except Exception as e: AIR = []; print(f"[repair] no air loops for {shape} ({e}): nothing done")
    if AIR:
        a = la.audit(V, F, AIR); print(f"[repair] in: genus {a['genus']}, tunnels realised {a['rank']}/{len(AIR)}")
        V, F, k = repair(V, F, AIR); a = la.audit(V, F, AIR); print(f"[repair] out: genus {a['genus']}, tunnels realised {a['rank']}/{len(AIR)}, removed {k}")
    d = {k: m[k] for k in m.files}; d["verts"] = V.astype(m["verts"].dtype); d["tris"] = F.astype(m["tris"].dtype); np.savez(dst, **d)
