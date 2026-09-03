"""Generic (non-TopMod) equivalents of the DLFL operators used in the golden
chain, on plain index arrays. Same geometric rules and guards as the DLFL
versions so that the ablation isolates the *implementation* (DLFL half-edge
structure with manifold closure) from the *rule*:

  flip_sweep_np          == dlfl_untangle.flip_sweep    (delete_edge + insert_edge)
  collapse_short_edges_np== dlfl_untangle.collapse_short_edges (collapse_edge_tri, midpoint, link condition)
  subdivide_all_np       == phase1c.dlfl_subdivide_arrays(all faces) (subdivide_edge every edge + stellate)

Open3D has no edge-flip / single-edge-collapse primitives, so these are numpy;
Open3D is used only for the self-intersection audit elsewhere.
Enabled by env GENERIC_OPS=1 (dispatch inside the DLFL wrappers).
"""
import numpy as np
from collections import defaultdict


def _tri_n(p, q, r):
    n = np.cross(q - p, r - p); l = np.linalg.norm(n)
    return (n / l if l > 1e-14 else None), l


def _edge_map(F):
    em = defaultdict(list)
    for i, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            em[(min(u, v), max(u, v))].append(i)
    return em


def flip_sweep_np(V, F, passes=4, fold_cos=0.0):
    V = np.asarray(V, float); F = np.asarray(F, np.int64).copy()
    em = _edge_map(F)
    total = 0
    for _ in range(passes):
        n = 0
        for key in list(em.keys()):
            fs = em.get(key)
            if fs is None or len(fs) != 2: continue
            fa, fb = fs
            ta, tb = F[fa], F[fb]
            u, v = key
            c = [x for x in ta if x != u and x != v]; d = [x for x in tb if x != u and x != v]
            if len(c) != 1 or len(d) != 1: continue
            c, d = c[0], d[0]
            if c == d: continue
            if (min(c, d), max(c, d)) in em: continue           # would duplicate an edge
            # orient: fa must contain u->v directed (else swap u,v)
            ia = list(ta).index(u)
            if ta[(ia + 1) % 3] != v: u, v = v, u
            na, la = _tri_n(V[u], V[v], V[c]); nb, lb = _tri_n(V[v], V[u], V[d])
            if na is None or nb is None: continue
            cur = float(na @ nb)
            if cur >= fold_cos: continue
            n1, l1 = _tri_n(V[c], V[u], V[d]); n2, l2 = _tri_n(V[d], V[v], V[c])
            if n1 is None or n2 is None or l1 < 1e-3 * (la + lb) or l2 < 1e-3 * (la + lb): continue
            if float(n1 @ n2) <= cur + 1e-6: continue
            # apply: fa -> (c,u,d), fb -> (d,v,c)
            F[fa] = (c, u, d); F[fb] = (d, v, c)
            del em[key]; em[(min(c, d), max(c, d))] = [fa, fb]
            for e, old, new in (((min(v, c), max(v, c)), fa, fb), ((min(u, d), max(u, d)), fb, fa)):
                lst = em[e]; lst[lst.index(old)] = new
            n += 1
        total += n
        if n == 0: break
    return V, F, total


def collapse_short_edges_np(V, F, ratio=0.3, max_n=400):
    V = np.asarray(V, float).copy(); F = np.asarray(F, np.int64).copy()
    em = _edge_map(F)
    el = {k: np.linalg.norm(V[k[0]] - V[k[1]]) for k in em}
    thr = ratio * float(np.mean(list(el.values())))
    alive = np.ones(len(F), bool)
    nbr = defaultdict(set)
    for a, b, c in F:
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
    n = 0
    for key in sorted(em.keys(), key=lambda k: el[k]):
        if key not in em: continue
        u, v = key
        if np.linalg.norm(V[u] - V[v]) > thr: break
        fs = em[key]
        if len(fs) != 2: continue
        fa, fb = fs
        a = [x for x in F[fa] if x != u and x != v]; b = [x for x in F[fb] if x != u and x != v]
        if len(a) != 1 or len(b) != 1: continue
        a, b = a[0], b[0]
        if a == b: continue
        if (nbr[u] & nbr[v]) != {a, b}: continue                # link condition
        # collapse v -> u at midpoint
        V[u] = (V[u] + V[v]) / 2
        alive[fa] = alive[fb] = False
        for f in (fa, fb):
            for x, y in ((F[f][0], F[f][1]), (F[f][1], F[f][2]), (F[f][2], F[f][0])):
                k = (min(x, y), max(x, y))
                if k in em:
                    em[k] = [g for g in em[k] if g != f]
                    if not em[k]: del em[k]
        # repoint faces of v to u
        for k in [k for k in list(em.keys()) if v in k]:
            fl = em.pop(k)
            w = k[0] if k[1] == v else k[1]
            nk = (min(u, w), max(u, w))
            em[nk] = em.get(nk, []) + fl
        for f in np.where(alive)[0]:
            if v in F[f]: F[f][F[f] == v] = u
        nbr[u] = (nbr[u] | nbr[v]) - {u, v}
        for w in nbr[v]: nbr[w].discard(v); nbr[w].add(u)
        nbr[u].discard(u); nbr[v] = set()
        for w in list(nbr[u]): nbr[w].add(u)
        n += 1
        if n >= max_n: break
    if n == 0:
        return V, F, 0
    F = F[alive]
    used = np.unique(F); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[F], n


def subdivide_all_np(V, F):
    """1 -> 6: midpoint on every edge + centroid in every face (same connectivity
    as DLFL subdivide_edge-all + stellate)."""
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    em = {}
    mids = []
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (min(u, v), max(u, v))
            if k not in em:
                em[k] = len(V) + len(mids); mids.append((V[u] + V[v]) / 2)
    nm = len(mids)
    cents = V[F].mean(1)
    out = []
    for i, (a, b, c) in enumerate(F):
        ci = len(V) + nm + i
        m01, m12, m20 = em[(min(a, b), max(a, b))], em[(min(b, c), max(b, c))], em[(min(c, a), max(c, a))]
        out += [(ci, a, m01), (ci, m01, b), (ci, b, m12), (ci, m12, c), (ci, c, m20), (ci, m20, a)]
    V2 = np.concatenate([V, np.asarray(mids), cents])
    return V2, np.asarray(out, np.int64), len(em)
