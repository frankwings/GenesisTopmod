"""Phase 1a — dual-pass layered rendering + operator-existence probability p_k.

Toy gradient-path validation (Go/No-Go for Phase 1b):
  GT      = icosphere with ONE bump (extruded face, dist=0.35) rendered in 6 views.
  Base    = plain icosphere (no bump).
  Cand G  = speculative extrude on the face closest to the GT bump direction.
  Cand B  = speculative extrude on the OPPOSITE face (no GT support).

Dual-pass scheme (debate B-minimal, correction #1):
  S_base = render(base mesh)          — no p involved, existing geometry safe.
  S_k    = render(base + spec faces of operator k)
  incremental region D_k = pixels where S_k > S_base (soft: relu(S_k - S_base))
  L_k = p_k * BCE(D_k region silhouette vs GT)   restricted to D_k pixels.
  Total = sum_k L_k  (+ tiny prior pulling p to 0 = Occam).

Alternating optimization (correction #2):
  Step A: p frozen at 1, optimize spec vertex positions (3 iters)
  Step B: positions frozen, optimize theta_k (1 iter)
  Repeat R rounds; Step C: threshold 0.5.

PASS criteria: p_good -> >0.8, p_bad -> <0.2, and spec verts of G moved
toward GT bump (position gradient flows through the p-weighted loss).
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from eval_extrude_v3 import make_6_cameras, render_sil_and_depth
from pipeline.cameras import transform_to_clip

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
IMG = 256
N_VIEWS = 6
BUMP_DIR = np.array([1.0, 0.3, 0.2]); BUMP_DIR /= np.linalg.norm(BUMP_DIR)
BUMP_DIST = 0.35
SPEC_DIST0 = 0.15          # spec extrude initial length (deliberately wrong)
ROUNDS = 20
POS_ITERS = 3
LR_POS = 5e-3
LR_P = 0.5
W_OCCAM = 0.0005           # tiny pull toward p=0 (Occam prior)


def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    for _ in range(subdiv):
        vl = list(map(tuple, v)); em = {}; nf = []
        def mid(a, b):
            k = (a, b) if a < b else (b, a)
            if k not in em:
                m = (np.array(vl[a]) + np.array(vl[b])) / 2
                vl.append(tuple(m)); em[k] = len(vl) - 1
            return em[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        v = np.array(vl); f = np.array(nf, int)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * 0.8
    return v, f


def extrude_face(V, Fa, fi, dist):
    """Triangle extrude: remove face fi, add 3 offset verts, 6 side tris + cap.
    Returns newV, newF, spec_face_indices (indices into newF of the 7 new tris)."""
    V = np.asarray(V, float); Fa = np.asarray(Fa, int)
    a, b, c = Fa[fi]
    tri = V[[a, b, c]]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= (np.linalg.norm(n) + 1e-12)
    base = len(V)
    newv = tri + n * dist
    V2 = np.vstack([V, newv])
    a2, b2, c2 = base, base + 1, base + 2
    keep = np.delete(Fa, fi, axis=0)
    sides = [(a, b, b2), (a, b2, a2), (b, c, c2), (b, c2, b2),
             (c, a, a2), (c, a2, c2)]
    cap = [(a2, b2, c2)]
    F2 = np.vstack([keep, np.array(sides + cap, int)])
    spec_idx = np.arange(len(keep), len(F2))
    new_vidx = np.array([a2, b2, c2])
    return V2, F2, spec_idx, new_vidx


def render_sil(ctx, verts_t, faces_t, mvp):
    pos = transform_to_clip(verts_t, mvp)
    rast, _ = dr.rasterize(ctx, pos, faces_t, resolution=[IMG, IMG])
    ones = torch.ones(1, verts_t.shape[0], 1, dtype=torch.float32, device=DEVICE)
    col, _ = dr.interpolate(ones, rast, faces_t)
    return dr.antialias(col, rast, pos, faces_t)[0, :, :, 0]


def main():
    torch.manual_seed(0); np.random.seed(0)
    ctx = dr.RasterizeCudaContext()
    mvps, _ = make_6_cameras(radius=3.0, device=DEVICE)

    # ---- GT: icosphere + bump on face closest to BUMP_DIR
    v0, f0 = icosphere(2)
    ctrs = v0[f0].mean(1); ctrs_n = ctrs / np.linalg.norm(ctrs, axis=1, keepdims=True)
    fi_gt = int(np.argmax(ctrs_n @ BUMP_DIR))
    gt_v, gt_f, _, _ = extrude_face(v0, f0, fi_gt, BUMP_DIST)
    gvt = torch.tensor(gt_v, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gt_f, dtype=torch.int32, device=DEVICE)
    with torch.no_grad():
        gt_sils = torch.stack([render_sil(ctx, gvt, gft, mvps[i]) for i in range(N_VIEWS)])

    # ---- candidates: good (same face) + bad (opposite face)
    fi_bad = int(np.argmin(ctrs_n @ BUMP_DIR))
    cands = [("G", fi_gt), ("B", fi_bad)]

    # build ONE mesh containing base + both spec extrudes (they are disjoint faces)
    V, Fa = v0.copy(), f0.copy()
    spec_face_ids, spec_vert_ids, op_of_face = [], [], {}
    # NB: apply extrudes sequentially; face indices shift after deletion.
    # Track by extruding on the CURRENT face array each time.
    for name, fi in cands:
        # find current index of original face (identify by vertex triple)
        tgt = set(f0[fi])
        cur = next(i for i, ff in enumerate(Fa) if set(ff) == tgt)
        V, Fa, sidx, vidx = extrude_face(V, Fa, cur, SPEC_DIST0)
        # previous spec ids survive (deletion happens before appended block)
        spec_face_ids = [np.searchsorted(np.arange(len(Fa)), s) for s in spec_face_ids]
        spec_face_ids.append(sidx)
        spec_vert_ids.append(vidx)
        for s in sidx: op_of_face[int(s)] = name

    # re-derive spec ids robustly: last 14 faces = 7 per operator in order
    nb = len(Fa) - 14
    spec_face_ids = [np.arange(nb, nb + 7), np.arange(nb + 7, nb + 14)]
    base_face_ids = np.arange(nb)

    verts_t = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    spec_verts = sorted(set(int(x) for vv in spec_vert_ids for x in vv))
    spec_mask = torch.zeros(len(V), dtype=torch.bool, device=DEVICE)
    spec_mask[spec_verts] = True
    verts_t.requires_grad_(True)
    theta = torch.zeros(2, device=DEVICE, requires_grad=True)  # p = sigmoid(theta) = 0.5

    faces_all = torch.tensor(Fa, dtype=torch.int32, device=DEVICE)
    faces_base = faces_all[:nb]
    faces_ops = [faces_all[torch.tensor(s, device=DEVICE).long()] for s in spec_face_ids]

    opt_pos = torch.optim.Adam([verts_t], lr=LR_POS)
    opt_p = torch.optim.Adam([theta], lr=LR_P)

    def spec_loss(force_p=None):
        """Probability-as-opacity compositing (signed gradient for p):
        S_mix = S_base + sum_k p_k * relu(S_k - S_base), loss = L1(S_mix, GT).
        Covering a GT gap lowers loss -> p up; covering background -> p down."""
        total = torch.tensor(0.0, device=DEVICE)
        p = torch.sigmoid(theta) if force_p is None else force_p
        for i in range(N_VIEWS):
            s_base = render_sil(ctx, verts_t, faces_base, mvps[i])
            s_mix = s_base
            for k in range(2):
                fk = torch.cat([faces_base, faces_ops[k]], 0)
                s_k = render_sil(ctx, verts_t, fk, mvps[i])
                s_mix = s_mix + p[k] * (s_k - s_base.detach()).clamp(min=0.0)
            s_mix = s_mix.clamp(0, 1)
            # focus loss on active region (union of GT-gap and spec coverage),
            # otherwise the 256^2 mean dilutes the p gradient below the prior
            region = ((gt_sils[i] > 0.01) & (s_base.detach() < 0.99)) | (s_mix.detach() > s_base.detach() + 0.01)
            err = (s_mix - gt_sils[i]).abs()
            total = total + err[region].sum() / (region.sum() + 1)
        return total / N_VIEWS + W_OCCAM * torch.sigmoid(theta).sum()

    print(f"GT bump face={fi_gt} bad face={fi_bad}  p0={torch.sigmoid(theta).tolist()}")
    hist = []
    for r in range(ROUNDS):
        # Step A: positions (spec verts only), p frozen at 1
        for _ in range(POS_ITERS):
            opt_pos.zero_grad()
            loss = spec_loss(force_p=torch.ones(2, device=DEVICE))  # p frozen at 1
            loss.backward()
            verts_t.grad[~spec_mask] = 0             # only spec verts move
            opt_pos.step()
        # Step B: p, positions frozen
        opt_p.zero_grad()
        loss = spec_loss()
        loss.backward()
        verts_t.grad = None
        opt_p.step()
        p = torch.sigmoid(theta).tolist()
        hist.append(p)
        if r % 2 == 0 or r == ROUNDS - 1:
            print(f"round {r:2d}: p_good={p[0]:.3f} p_bad={p[1]:.3f} loss={float(loss):.4f}", flush=True)

    p_good, p_bad = torch.sigmoid(theta).tolist()
    # Step C: hard decision
    decision = [("G", p_good > 0.5), ("B", p_bad > 0.5)]
    print(f"\nStep C decisions: {decision}")
    # spec vertex displacement toward GT bump tip?
    tipG = V[spec_vert_ids[0]].mean(0)
    tipG_now = verts_t.detach().cpu().numpy()[spec_vert_ids[0]].mean(0)
    gt_tip = gt_v[-3:].mean(0)
    d_before = np.linalg.norm(tipG - gt_tip); d_after = np.linalg.norm(tipG_now - gt_tip)
    print(f"good-cand tip dist to GT tip: {d_before:.4f} -> {d_after:.4f}")
    ok = p_good > 0.8 and p_bad < 0.2 and d_after < d_before
    print(f"\nPHASE1A {'PASS' if ok else 'FAIL'}  (p_good>{0.8}? {p_good:.3f} | p_bad<{0.2}? {p_bad:.3f} | tip closer? {d_after < d_before})")

    # viz
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    h = np.array(hist)
    fig = plt.figure(figsize=(15, 4), dpi=110)
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(h[:, 0], "g-o", ms=3, label="p_good"); ax.plot(h[:, 1], "r-o", ms=3, label="p_bad")
    ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_ylim(0, 1)
    ax.set_xlabel("round"); ax.set_title("operator existence prob"); ax.legend()
    Vn = verts_t.detach().cpu().numpy()
    for j, (name, vv, ff) in enumerate([("GT", gt_v, gt_f),
                                        ("base+spec init", V, Fa),
                                        ("after optim", Vn, Fa)]):
        ax = fig.add_subplot(1, 4, j + 2, projection="3d")
        tri = np.asarray(vv)[np.asarray(ff)]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
        li = np.array([0.3, 0.5, 0.8]); li /= np.linalg.norm(li)
        sh = np.clip(np.abs(fn @ li), 0.1, 1)
        colarr = np.stack([sh * 0.5, sh * 0.7, sh * 0.9], 1)
        if j > 0:
            colarr[nb:nb + 7] = np.array([0.2, 0.9, 0.2]) * sh[nb:nb + 7, None]
            colarr[nb + 7:] = np.array([0.95, 0.3, 0.3]) * sh[nb + 7:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colarr, edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=15, azim=115); ax.axis("off")
        ax.set_title(name, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/phase1a_toy.png", bbox_inches="tight")
    print(f"saved {OUT}/phase1a_toy.png")


if __name__ == "__main__":
    main()
