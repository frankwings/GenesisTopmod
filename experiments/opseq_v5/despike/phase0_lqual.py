"""Phase 0 (B-minimal plan): DMesh-style L_qual triangle-quality regularizer.

q(tri) = 4*sqrt(3)*Area / (l1^2+l2^2+l3^2)  in (0,1], 1 = equilateral.
L_qual = mean(1 - q).  Injected via edge_length_loss wrapper (same trick as
cow_v23), effective weight W_QUAL independent of W_EDGE.

W_QUAL=0 -> pure v22 baseline reproduction (armadillo ref ho16=0.9149).
Usage:  SHAPE=armadillo W_QUAL=0.01 python3 phase0_lqual.py
"""
import os, sys
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")

W_QUAL = float(os.environ.get("W_QUAL", "0"))
SHAPE = os.environ.get("SHAPE", "armadillo")
os.environ["SHAPE"] = SHAPE
os.environ.setdefault("TAG", f"{SHAPE}_p0_q{W_QUAL:g}")

import torch
import cow_v13
from eval_local_refine import W_EDGE

_orig_edge = cow_v13.edge_length_loss
_SQRT3_4 = 4.0 * (3.0 ** 0.5)


def _qual_loss(verts_t, faces_t):
    tri = verts_t[faces_t.long()]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 1]
    e2 = tri[:, 0] - tri[:, 2]
    l2 = (e0 * e0).sum(-1) + (e1 * e1).sum(-1) + (e2 * e2).sum(-1)
    area = 0.5 * torch.cross(e0, -e2, dim=-1).norm(dim=-1)
    q = _SQRT3_4 * area / (l2 + 1e-12)
    return (1.0 - q).mean()


def _edge_with_qual(verts_t, faces_t):
    base = _orig_edge(verts_t, faces_t)
    if W_QUAL <= 0:
        return base
    return base + (W_QUAL / max(W_EDGE, 1e-9)) * _qual_loss(verts_t, faces_t)


cow_v13.edge_length_loss = _edge_with_qual
print(f"[phase0] SHAPE={SHAPE} W_QUAL={W_QUAL} (0 => baseline repro)", flush=True)

import run_v22
run_v22.main()
