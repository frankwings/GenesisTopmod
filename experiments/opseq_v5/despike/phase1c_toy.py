"""Phase 1c — carve operator (DLFL negative extrude) gradient validation.

Toy Go/No-Go before pipeline integration:
  GT      = icosphere with ONE dent (DLFL extrude_face, dist=-0.4) in 6 views.
  Base    = plain icosphere (no dent) -> RED excess at the dent notch.
  Cand G  = speculative carve on the GT dent face.
  Cand B  = speculative carve on the OPPOSITE face (GT has material there).

Changes vs phase1a:
  1. Operators applied via the REAL DLFL extrude_face (topmod/high_level_ops),
     not the NumPy replica. Arrays <-> DLFLMesh via obj roundtrip (~0.06s).
  2. Symmetric opacity compositing (carve makes S_k < S_base):
       S_mix = S_base + sum_k p_k * (S_k - S_base.detach())      (NO clamp)
  3. Active region includes RED side: (gt<0.99 & s_base>0.01).

PASS: p_good > 0.8, p_bad < 0.2, notch depth error shrinks.
"""
import sys, os, tempfile
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
import nvdiffrast.torch as dr
from eval_extrude_v3 import make_6_cameras, render_sil_and_depth
from pipeline.cameras import transform_to_clip
from topmod.io import from_obj, to_triangle_arrays
from topmod.high_level_ops import extrude_face as dlfl_extrude, triangulate_face
from phase1a_toy import icosphere, render_sil

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
IMG = 256
N_VIEWS = 6
DENT_DIR = np.array([1.0, 0.3, 0.2]); DENT_DIR /= np.linalg.norm(DENT_DIR)
DENT_DIST = -0.4
SPEC_DIST0 = -0.15         # spec carve initial depth (deliberately too shallow)
ROUNDS = 20
POS_ITERS = 3
LR_POS = 5e-3
LR_P = 0.5
W_OCCAM = 0.0005
W_DEPTH = 1.0              # depth term: dent DEPTH is invisible to silhouette
                           # (bottom occluded by the rim) -> depth carries it


def dlfl_extrude_arrays(V, Fa, fi, dist):
    """Apply the real DLFL extrude on (V,Fa) at face fi; triangulate; return
    (V2, F2, spec_fidx, new_vidx) with base faces first, spec faces last."""
    V = np.asarray(V, float); Fa = np.asarray(Fa, int)
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, z in V: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    faces = list(mesh.iter_faces())
    assert len(faces) == len(Fa)
    newf = dlfl_extrude(mesh, faces[fi], dist=dist)
    for f in list(newf):
        if len(f.vertices()) > 3:
            triangulate_face(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2 = np.asarray(vv, float); F2 = np.asarray(ff, int)
    # sanity: original verts preserved in order
    assert len(V2) == len(V) + 3 and np.allclose(V2[:len(V)], V, atol=1e-9), \
        "DLFL export reordered vertices"
    nb0 = len(V)
    is_spec = np.array([any(x >= nb0 for x in f) for f in F2])
    order = np.concatenate([np.where(~is_spec)[0], np.where(is_spec)[0]])
    F2 = F2[order]
    n_spec = int(is_spec.sum())
    spec_fidx = np.arange(len(F2) - n_spec, len(F2))
    new_vidx = np.arange(nb0, nb0 + 3)
    return V2, F2, spec_fidx, new_vidx


def main():
    torch.manual_seed(0); np.random.seed(0)
    ctx = dr.RasterizeCudaContext()
    mvps, _ = make_6_cameras(radius=3.0, device=DEVICE)

    # ---- GT: icosphere + dent via DLFL negative extrude
    v0, f0 = icosphere(2)
    ctrs = v0[f0].mean(1); ctrs_n = ctrs / np.linalg.norm(ctrs, axis=1, keepdims=True)
    fi_gt = int(np.argmax(ctrs_n @ DENT_DIR))
    fi_bad = int(np.argmin(ctrs_n @ DENT_DIR))
    gt_v, gt_f, _, gt_vidx = dlfl_extrude_arrays(v0, f0, fi_gt, DENT_DIST)
    gvt = torch.tensor(gt_v, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gt_f, dtype=torch.int32, device=DEVICE)
    with torch.no_grad():
        gt_sils = torch.stack([render_sil(ctx, gvt, gft, mvps[i]) for i in range(N_VIEWS)])
        gt_deps, gt_fgs = [], []
        for i in range(N_VIEWS):
            _, dz, fg = render_sil_and_depth(ctx, gvt, gft, mvps[i])
            gt_deps.append(dz); gt_fgs.append(fg)

    # ---- base + two speculative carves (DLFL, sequential)
    V, Fa = v0.copy(), f0.copy()
    spec_vert_ids = []
    for fi in (fi_gt, fi_bad):
        tgt = set(f0[fi])
        cur = next(i for i, ff in enumerate(Fa) if set(ff) == tgt)
        V, Fa, sidx, vidx = dlfl_extrude_arrays(V, Fa, cur, SPEC_DIST0)
        spec_vert_ids.append(vidx)
    n_per = 7                                   # 1 cap + 6 side tris per carve
    nb = len(Fa) - 2 * n_per
    # NOTE: after 2nd extrude the 1st op's spec faces contain verts < nb0 of 2nd
    # call, so the reorder above may interleave. Re-derive robustly by vertex id:
    nv0 = len(v0)
    owner = np.full(len(Fa), -1)
    for i, f in enumerate(Fa):
        hi = [x for x in f if x >= nv0]
        if hi:
            owner[i] = 0 if min(hi) < nv0 + 3 else 1
    order = np.concatenate([np.where(owner < 0)[0],
                            np.where(owner == 0)[0], np.where(owner == 1)[0]])
    Fa = Fa[order]
    assert (owner >= 0).sum() == 2 * n_per and len(Fa) - 2 * n_per == nb

    verts_t = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    spec_mask = torch.zeros(len(V), dtype=torch.bool, device=DEVICE)
    for vv in spec_vert_ids: spec_mask[torch.tensor(vv, device=DEVICE).long()] = True
    verts_t.requires_grad_(True)
    theta = torch.zeros(2, device=DEVICE, requires_grad=True)

    faces_all = torch.tensor(Fa, dtype=torch.int32, device=DEVICE)
    faces_base = faces_all[:nb]
    faces_ops = [faces_all[nb:nb + n_per], faces_all[nb + n_per:]]
    # base WITHOUT op-k's seed face is not needed: for carve, S_k comes from
    # rendering base-minus-original-face + carve walls; here base still has the
    # original faces of both carve sites REMOVED (extrude deleted them), so
    # "base" for compositing = faces_base + other op's faces at p... To keep the
    # toy honest we render: S_k = faces_base + op_k faces (site of other op
    # stays open -> tiny hole, invisible in silhouette since backfaces fill).
    # S_base = faces_base + BOTH ops' CAP+SIDES? No — use base = plain sphere:
    sphere_faces = torch.tensor(f0, dtype=torch.int32, device=DEVICE)

    opt_pos = torch.optim.Adam([verts_t], lr=LR_POS)
    opt_p = torch.optim.Adam([theta], lr=LR_P)

    def spec_loss(force_p=None):
        """Symmetric compositing: S_mix = S_base + sum p_k (S_k - S_base)."""
        total = torch.tensor(0.0, device=DEVICE)
        p = torch.sigmoid(theta) if force_p is None else force_p
        for i in range(N_VIEWS):
            s_base = render_sil(ctx, verts_t[:len(v0)], sphere_faces, mvps[i])
            s_mix = s_base
            for k in range(2):
                # mesh for op k alone: base faces + op k walls + the OTHER
                # site's original face restored (= plain sphere there)
                other = 1 - k
                restore = torch.tensor(f0[[fi_gt, fi_bad][other]][None, :],
                                       dtype=torch.int32, device=DEVICE)
                fk = torch.cat([faces_base, faces_ops[k], restore], 0)
                s_k = render_sil(ctx, verts_t, fk, mvps[i])
                s_mix = s_mix + p[k] * (s_k - s_base.detach())
                # depth term (signed coupling, same spirit as compositing):
                # E_k - E_base < 0 when the carve improves depth match -> p up.
                _, d_k, fg_k = render_sil_and_depth(ctx, verts_t, fk, mvps[i])
                with torch.no_grad():
                    _, d_b, fg_b = render_sil_and_depth(
                        ctx, verts_t[:len(v0)], sphere_faces, mvps[i])
                    m_k = fg_k.detach() & gt_fgs[i]
                    m_b = fg_b & gt_fgs[i]
                    e_b = (d_b - gt_deps[i]).abs()[m_b].mean()
                e_k = (d_k - gt_deps[i]).abs()[m_k].sum() / (m_k.sum() + 1)
                total = total + W_DEPTH * p[k] * (e_k - e_b)
            s_mix = s_mix.clamp(0, 1)
            gt_i = gt_sils[i]
            region = (((gt_i > 0.01) & (s_base.detach() < 0.99))
                      | ((gt_i < 0.99) & (s_base.detach() > 0.01))
                      | ((s_mix.detach() - s_base.detach()).abs() > 0.01))
            err = (s_mix - gt_i).abs()
            total = total + err[region].sum() / (region.sum() + 1)
        return total / N_VIEWS + W_OCCAM * torch.sigmoid(theta).sum()

    print(f"GT dent face={fi_gt} bad face={fi_bad}  p0={torch.sigmoid(theta).tolist()}")
    hist = []
    for r in range(ROUNDS):
        for _ in range(POS_ITERS):
            opt_pos.zero_grad()
            loss = spec_loss(force_p=torch.ones(2, device=DEVICE))
            loss.backward()
            verts_t.grad[~spec_mask] = 0
            opt_pos.step()
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
    tipG_now = verts_t.detach().cpu().numpy()[spec_vert_ids[0]].mean(0)
    gt_tip = gt_v[gt_vidx].mean(0)
    tipG_init = V[spec_vert_ids[0]].mean(0)
    d_before = np.linalg.norm(tipG_init - gt_tip)
    d_after = np.linalg.norm(tipG_now - gt_tip)
    print(f"\ngood-carve tip dist to GT dent bottom: {d_before:.4f} -> {d_after:.4f}")
    ok = p_good > 0.8 and p_bad < 0.2 and d_after < d_before
    print(f"PHASE1C {'PASS' if ok else 'FAIL'}  (p_good {p_good:.3f} | p_bad {p_bad:.3f} | deeper? {d_after < d_before})")

    # viz
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    h = np.array(hist)
    fig = plt.figure(figsize=(15, 4), dpi=110)
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(h[:, 0], "g-o", ms=3, label="p_good (dent)")
    ax.plot(h[:, 1], "r-o", ms=3, label="p_bad")
    ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_ylim(0, 1)
    ax.set_xlabel("round"); ax.set_title("carve operator existence prob"); ax.legend()
    Vn = verts_t.detach().cpu().numpy()
    for j, (name, vv, ff) in enumerate([("GT (dent)", gt_v, gt_f),
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
            colarr[nb:nb + n_per] = np.array([0.2, 0.9, 0.2]) * sh[nb:nb + n_per, None]
            colarr[nb + n_per:] = np.array([0.95, 0.3, 0.3]) * sh[nb + n_per:, None]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colarr, edgecolor="none"))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=15, azim=115); ax.axis("off")
        ax.set_title(name, fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/phase1c_toy.png", bbox_inches="tight")
    print(f"saved {OUT}/phase1c_toy.png")


if __name__ == "__main__":
    main()
