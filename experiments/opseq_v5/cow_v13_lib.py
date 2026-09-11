"""Shared geometry priors for the v13 despike recipe (see despike/cow_v13.py)."""
import numpy as np
import torch
import torch.nn.functional as F


def midpoint_subdivide(verts, tris):
    verts = list(map(tuple, verts))
    em = {}
    def mid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in em:
            va, vb = verts[a], verts[b]
            verts.append(((va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2))
            em[k] = len(verts) - 1
        return em[k]
    nt = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(nt, dtype=np.int32)


def build_pairs(tris_np):
    e2f = {}
    for fi, (a, b, c) in enumerate(tris_np):
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            e2f.setdefault(k, []).append(fi)
    return np.array([p for p in e2f.values() if len(p) == 2], dtype=np.int64)


def fold_loss(verts_t, faces_l, pairs_t):
    tri = verts_t[faces_l]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    d = (n[pairs_t[:, 0]] * n[pairs_t[:, 1]]).sum(-1)
    return F.relu(-d).pow(2).mean()


def build_adj(tris_np, nv, device):
    src, dst = [], []
    es = set()
    for a, b, c in tris_np:
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            if k in es: continue
            es.add(k)
            src += [e[0], e[1]]; dst += [e[1], e[0]]
    src = torch.tensor(src, dtype=torch.long, device=device)
    dst = torch.tensor(dst, dtype=torch.long, device=device)
    deg = torch.zeros(nv, device=device).index_add_(
        0, src, torch.ones_like(src, dtype=torch.float32)).clamp(min=1).unsqueeze(-1)
    A = torch.zeros(nv, nv, dtype=torch.bool, device=device)
    A[src, dst] = True
    A2 = (A.float() @ A.float()) > 0
    excl = A | A2 | torch.eye(nv, dtype=torch.bool, device=device)
    return src, dst, deg, excl


def centroid(v, src, dst, deg):
    cen = torch.zeros_like(v).index_add_(0, src, v[dst])
    return cen / deg


def spike_pen(v, src, dst, deg, mean_edge, thr):
    lap = (v - centroid(v, src, dst, deg)).norm(dim=-1)
    return F.relu(lap - thr * mean_edge).pow(2).mean()


def sliver_pen(v, faces_l, mean_e):
    tri = v[faces_l]
    e0 = (tri[:, 1] - tri[:, 0]).norm(dim=-1)
    e1 = (tri[:, 2] - tri[:, 1]).norm(dim=-1)
    e2 = (tri[:, 0] - tri[:, 2]).norm(dim=-1)
    lmax = torch.stack([e0, e1, e2], -1).max(-1).values
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                             dim=-1).norm(dim=-1)
    h = 2 * area / (lmax + 1e-12)
    p = F.relu(0.25 * mean_e - h).pow(2).mean()
    for e in (e0, e1, e2):
        p = p + F.relu(e - 3 * mean_e).pow(2).mean()
    return p


@torch.no_grad()
def tube_mask(v, excl, mean_edge, thr):
    D = torch.cdist(v, v)
    D[excl] = 1e9
    return D.min(1).values < thr * mean_edge


def mean_edge_of(v, src, dst):
    return (v[src] - v[dst]).norm(dim=-1).mean()
