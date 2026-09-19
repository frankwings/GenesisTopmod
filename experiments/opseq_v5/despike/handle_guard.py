#!/usr/bin/env python3
"""handle_guard.py — port of Gu et al. (ICASSP 2026) topology tools into our discover+manifold pipeline.
No external PH library: homology generators via the tree-cotree algorithm (their Alg.: primal spanning tree T,
dual spanning tree C on edges not in T; the 2g edges in neither give the H1 generators). Used for:
  (1) anti-collapse handle GUARD: keep each generator loop from contracting (radius of gyration >= target); a loop that
      can shrink to a point has low H1 persistence = a dying handle. Cheap differentiable proxy for PH persistence.
  (2) H1 = 2g VERIFICATION after add_handle.
  (3) topology metric: the loop-length / loop-span spectrum, compared GT vs reconstruction.
"""
import numpy as np, torch

def _edges(F):
    E = np.sort(np.concatenate([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]], 0), axis=1)
    return np.unique(E, axis=0)

def tree_cotree_loops(V, F, max_loops=64):
    """Return H1 generators as vertex-index cycles (2g of them for a closed genus-g mesh)."""
    V = np.asarray(V); F = np.asarray(F, np.int64); n = len(V)
    E = _edges(F); ekey = {(int(a), int(b)): idx for idx, (a, b) in enumerate(E)}
    def ei(a, b): a, b = (a, b) if a < b else (b, a); return ekey[(int(a), int(b))]
    # adjacency
    adj = [[] for _ in range(n)]
    for idx, (a, b) in enumerate(E): adj[a].append((b, idx)); adj[b].append((a, idx))
    # PRIMAL spanning tree (BFS)
    in_tree = np.zeros(len(E), bool); par = -np.ones(n, np.int64); par_e = -np.ones(n, np.int64); seen = np.zeros(n, bool)
    from collections import deque
    q = deque([0]); seen[0] = True
    while q:
        u = q.popleft()
        for w, idx in adj[u]:
            if not seen[w]: seen[w] = True; in_tree[idx] = True; par[w] = u; par_e[w] = idx; q.append(w)
    # DUAL graph: faces as nodes; a mesh edge NOT in the primal tree connects its two faces
    nf = len(F); face_of_edge = {}
    for fi, f in enumerate(F):
        for a, b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])): face_of_edge.setdefault(ei(a,b), []).append(fi)
    dadj = [[] for _ in range(nf)]
    for idx in range(len(E)):
        if in_tree[idx]: continue
        fs = face_of_edge.get(idx, [])
        if len(fs) == 2: dadj[fs[0]].append((fs[1], idx)); dadj[fs[1]].append((fs[0], idx))
    in_cotree = np.zeros(len(E), bool); fseen = np.zeros(nf, bool); q = deque([0]); fseen[0] = True
    while q:
        u = q.popleft()
        for w, idx in dadj[u]:
            if not fseen[w]: fseen[w] = True; in_cotree[idx] = True; q.append(w)
    # generator edges = in neither tree nor cotree
    gen = [idx for idx in range(len(E)) if not in_tree[idx] and not in_cotree[idx]]
    def tree_path(a):
        p = []
        while a != -1 and par[a] != -1: p.append(a); a = par[a]
        p.append(a); return p
    loops = []
    for idx in gen[:max_loops]:
        a, b = E[idx]
        pa, pb = tree_path(a), tree_path(b)                    # both to root
        sa, sb = set(pa), set(pb)
        # lowest common ancestor: trim shared tail
        i = len(pa) - 1; j = len(pb) - 1
        while i >= 0 and j >= 0 and pa[i] == pb[j]: i -= 1; j -= 1
        loop = pa[:i+2] + pb[:j+1][::-1]
        loops.append(np.array(loop, np.int64))
    return loops

def loop_span(Vt, loop_idx):
    """differentiable radius of gyration of a loop's vertices (0 when the loop contracts to a point)."""
    p = Vt[loop_idx]; c = p.mean(0)
    return (p - c).pow(2).sum(-1).mean().clamp_min(1e-12).sqrt()

def _axis_radius(Vt, a0, u):
    d = Vt - a0; return (d - (d @ u).unsqueeze(-1) * u).norm(dim=-1)

def throat_openness(Vt, a0, u, target, soft=60.0):
    """soft-min perpendicular distance to the tunnel axis (a0,u) = the throat/hole radius; penalise below target.
    a0,u are the add_handle axis (the line between the two joined faces passes THROUGH the hole)."""
    r = _axis_radius(Vt, a0, u); throat = -torch.logsumexp(-soft * r, 0) / soft
    return torch.relu(target - throat), throat

def handles_guard_loss(Vt, axes, targets, w=1.0):
    """axes = list of (a0,u) torch tensors (from add_handle); keep each tunnel throat open. Validated: preserves a
    genus-1 handle under a hole-closing force (test_handle_guard.py)."""
    if not axes: return Vt.sum() * 0.0
    tot = 0.0
    for (a0, u), tg in zip(axes, targets):
        gl, _ = throat_openness(Vt, a0, u, tg); tot = tot + gl
    return w * tot / max(len(axes), 1)

def axis_throat_target(Vt, a0, u, keep=0.8, soft=60.0):
    r = _axis_radius(Vt, a0, u); return keep * float((-torch.logsumexp(-soft * r, 0) / soft))

def loop_length_spectrum(V, F, loops=None):
    """topology metric (borrow #3): sorted lengths (world units) of the H1 generator loops.
    Compare GT vs reconstruction spectra (e.g. sorted L1) as a continuous topological-fidelity score."""
    V = np.asarray(V)
    if loops is None: loops = tree_cotree_loops(V, F)
    lens = [float(np.linalg.norm(np.diff(V[np.r_[l, l[0]]], axis=0), axis=1).sum()) for l in loops]
    return sorted(lens)

def genus_from_counts(V, F):
    V = np.asarray(V); F = np.asarray(F); E = len(_edges(F)); return (2 - (len(V) - E + len(F))) // 2
