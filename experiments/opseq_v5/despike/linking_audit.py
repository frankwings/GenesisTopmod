#!/usr/bin/env python3
"""linking_audit.py - GT-free layout audit by linking numbers (2026-10-01).

Genus is only a count. This checks WHERE the handles are: the mesh surface must realise every tunnel of the
carved hull. For each independent hull tunnel we build a closed curve in the hull AIR (through the tunnel, back
through the outer air). For each of the 2g generator loops of the mesh surface (tree-cotree) we compute the Gauss
linking number with every air curve. rank(linking matrix) = number of hull tunnels the surface realises.
  correct mesh: rank == g*;  redundant / hidden handle: rank < genus (e.g. bt1: genus 4, rank 2).
The same linking vector classifies a CANDIDATE handle (faces fi, fj): loop = straight segment + surface geodesic
path; vector 0 = no tunnel through it (spurious), vector in the span of the existing handles = redundant.

Usage:  SHAPE=fertility python3 despike/linking_audit.py results_genus/fertility_*_auto.npz
Needs the hull plug cache (out_liou/hull_plugs_<SHAPE>_128.json, written by hull_locate) for the sealed blocks.
"""
import sys, os, json, glob
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
S26 = np.ones((3, 3, 3), bool)
OUTD = os.environ.get("OUTD", "/tmp/liou_cow_viz")

def grid_frame(shape, res=128, pad=24):
    """voxel -> world map of the hull_locate grid (GT bbox +-0.02, same as build_vote_hull / find_tunnel_by_hull)."""
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    gv, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj")); g = normalize_to_range(gv)
    lo, hi = g.min(0) - 0.02, g.max(0) + 0.02
    return lambda v: lo + (np.asarray(v, float) - pad) / (res - 1) * (hi - lo)

def air_loops(shape, cache=None, min_mouth=20, log=print):
    """One closed curve per independent hull tunnel, entirely in hull air (world coordinates).
    Sealed block (union of plug blocks, air voxels) with m mouths to the outer air -> m-1 loops:
    mouth0 -> mouth j through the block, back through the outer air."""
    from skimage.graph import route_through_array
    cache = cache or f"{OUTD}/hull_plugs_{shape}_128.json"
    memo = cache + ".airloops.npz"
    if os.path.exists(memo) and os.path.getmtime(memo) >= os.path.getmtime(cache):
        z = np.load(memo); return [z[k] for k in sorted(z.files, key=lambda k: int(k[4:]))]
    plugs = json.load(open(cache)); bz = np.load(cache + ".blocks.npz"); hs = bz["hs"].astype(bool)
    P = np.zeros_like(hs)
    for i in range(len(plugs)): b = bz[f"plug{i}"]; P[b[:, 0], b[:, 1], b[:, 2]] = True
    P &= ~hs; W0 = ~hs & ~P
    lab, _ = ndimage.label(W0, structure=S26); sizes = np.bincount(lab.ravel()); sizes[0] = 0; W0 = lab == sizes.argmax()
    clab, cn = ndimage.label(P, structure=S26); v2w = grid_frame(shape); loops = []
    for c in range(1, cn + 1):
        C = clab == c
        if C.sum() < 50: continue
        iface = ndimage.binary_dilation(C, S26) & W0
        ml, _ = ndimage.label(iface, structure=S26); msz = np.bincount(ml.ravel())[1:]
        mouths = [k + 1 for k in np.argsort(-msz) if msz[k] >= min_mouth]
        log(f"[air] sealed block {c}: {int(C.sum())} vox, mouths {[int(msz[k - 1]) for k in mouths]}")
        if len(mouths) < 2: continue
        vox = [np.argwhere(ml == k) for k in mouths]
        rep = [tuple(a[np.argmin(np.linalg.norm(a - a.mean(0), axis=1))]) for a in vox]
        costC = np.where(C | iface, 1.0, -1.0); costW = np.where(W0, 1.0, -1.0)
        for j in range(1, len(mouths)):
            p1, _ = route_through_array(costC, rep[0], rep[j], fully_connected=True)
            p2, _ = route_through_array(costW, rep[j], rep[0], fully_connected=True)
            loops.append(v2w(np.array(p1 + p2[1:], float)))
    try: np.savez(memo, **{f"loop{i}": l for i, l in enumerate(loops)})
    except Exception: pass
    return loops

def linking(A, B):
    """Gauss linking number of two closed polylines (exact per segment pair, solid-angle form)."""
    a, b = A, np.roll(A, -1, 0); c, d = B, np.roll(B, -1, 0)
    r13 = c[None] - a[:, None]; r14 = d[None] - a[:, None]; r23 = c[None] - b[:, None]; r24 = d[None] - b[:, None]
    nrm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-300)
    n1 = nrm(np.cross(r13, r14)); n2 = nrm(np.cross(r14, r24)); n3 = nrm(np.cross(r24, r23)); n4 = nrm(np.cross(r23, r13))
    asn = lambda x: np.arcsin(np.clip(x, -1, 1))
    om = asn((n1 * n2).sum(-1)) + asn((n2 * n3).sum(-1)) + asn((n3 * n4).sum(-1)) + asn((n4 * n1).sum(-1))
    sgn = np.sign((np.cross((d - c)[None], (b - a)[:, None]) * r13).sum(-1))
    return float((om * sgn).sum() / (4 * np.pi))

def genus(V, F):
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

def surface_matrix(V, F, AIR):
    """(loops, lengths, integer linking matrix [2g x n_air], max rounding error) for the mesh's H1 generators."""
    import handle_guard as hg
    loops = hg.tree_cotree_loops(V, F); M = np.zeros((len(loops), len(AIR))); Ls = []
    for i, l in enumerate(loops):
        P = V[np.asarray(l)]; Ls.append(float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()))
        M[i] = [linking(P, A) for A in AIR]
    R = np.round(M); return loops, Ls, R, float(np.abs(M - R).max()) if M.size else 0.0

def candidate_vector(V, F, fi, fj, AIR):
    """Linking vector of the loop a handle between faces fi, fj would close (segment + surface geodesic path)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V), len(V))).tocsr()
    s = int(F[fi][0]); d, pred = dijkstra(G, indices=s, return_predecessors=True)
    t = int(F[fj][np.argmin(d[F[fj]])]); path = [t]
    while path[-1] != s: path.append(int(pred[path[-1]]))
    L = np.vstack([V[F[fi]].mean(0)[None], V[path[::-1]], V[F[fj]].mean(0)[None]])
    return np.round([linking(L, A) for A in AIR]).astype(int)

def audit(V, F, AIR):
    loops, Ls, R, err = surface_matrix(V, F, AIR)
    rank = int(np.linalg.matrix_rank(R)) if R.size else 0
    return dict(genus=int(genus(V, F)), rank=rank, per_tunnel=[int(np.any(R[:, k] != 0)) for k in range(len(AIR))] if R.size else [0] * len(AIR),
                tiny_loops=int(sum(x < 0.3 for x in Ls)), int_err=err)

if __name__ == "__main__":
    shape = os.environ.get("SHAPE", "fertility")
    if len(sys.argv) > 2 and sys.argv[1] == "--rank":     # machine-readable: "<rank> <genus> <n_air> <tiny>" (rank -1: no air loops)
        try: AIR = air_loops(shape, log=lambda m: None)
        except Exception: AIR = []
        m = np.load(sys.argv[2]); V, F = m["verts"].astype(float), m["tris"].astype(np.int64)
        if not AIR: print(-1, int(genus(V, F)), 0, 0); sys.exit(0)
        a = audit(V, F, AIR); print(a["rank"], a["genus"], len(AIR), a["tiny_loops"]); sys.exit(0)
    AIR = air_loops(shape, log=lambda m: print(m, file=sys.stderr))
    print(f"# {shape}: {len(AIR)} hull tunnel air loops"); ok = n = 0
    for pat in sys.argv[1:]:
        for f in sorted(glob.glob(pat)):
            m = np.load(f); a = audit(m["verts"].astype(float), m["tris"].astype(np.int64), AIR)
            good = a["rank"] == len(AIR) and a["genus"] == len(AIR); ok += good; n += 1
            print(f"{os.path.basename(f):44s} genus {a['genus']}  tunnels realised {a['rank']}/{len(AIR)}  per tunnel {a['per_tunnel']}  tiny loops {a['tiny_loops']}  {'OK' if good else 'MISMATCH'}", flush=True)
    print(f"# strict topology: {ok}/{n}")
