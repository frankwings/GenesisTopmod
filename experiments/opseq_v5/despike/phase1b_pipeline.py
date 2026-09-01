"""Phase 1b — speculative extrude with operator-existence probability, full pipeline.

Base = Phase-0 winner (v22 + L_qual 0.01) armadillo mesh (reloaded from npz for
identical baseline). Then R rounds of:
  1. GREEN mismatch heatmap (gt present, pred missing) in the 6 training views.
  2. Blob -> root face via rasterizer triangle_id at nearest pred pixel.
  3. Greedy candidate selection with 3D-distance isolation (proxy for geodesic
     d_min=3), K<=8 per round.
  4. Speculative extrude each candidate (along face normal, 3x mean edge).
  5. Alternating optimization: Step A (p=1, spec verts only, 3 iters) /
     Step B (positions frozen, theta step) x 20; probability-as-opacity
     compositing loss (Phase-1a validated form).
  6. Step C: p>0.5 -> commit into operator program; else rollback (exact DLFL
     undo: drop 7 spec faces + 3 verts, restore original face).
  7. Settle phase (300 steps, full loss incl. L_qual + escape-gated tube).
Exam: same 16 held-out views. Go/No-Go: ho16 vs same-batch baseline +1.5pts.
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import json, time, collections
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as cc_label
import nvdiffrast.torch as dr

import cow_v13
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               laplacian_loss, edge_length_loss,
                               depth_loss_masked, load_obj, normalize_to_range,
                               BUNNY_PATH, IMG_RES, LR, LR_MIN, W_DEPTH,
                               W_LAP, W_EDGE)
from eval_extrude_v3 import orbit_cameras
from pipeline.cameras import transform_to_clip
from surgery_lib import surgery, _propagate_flag
from escape_util import escape_mask

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", f"{SHAPE}_p1b")
BASE_NPZ = os.environ.get("BASE_NPZ", f"{OUT}/cow_{SHAPE}_p0_q0.01.npz")
W_QUAL = 0.01
ROUNDS = int(os.environ.get("ROUNDS", "3"))
K_MAX = 8
MIN_BLOB = 30          # px
AB_ROUNDS = 20
POS_ITERS = 3
LR_POS = 5e-3
LR_P = 0.5
W_OCCAM = 0.0005
EXTRUDE_LEN = 3.0      # x mean_edge
ISO_DIST = 3.0         # x mean_edge, 3D isolation between candidate roots
DILATE = 2
cow_v13.TUBE_THR = 0.4

_SQRT3_4 = 4.0 * (3.0 ** 0.5)


def _qual_loss(v, f):
    tri = v[f.long()]
    e0, e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]
    l2 = (e0 * e0).sum(-1) + (e1 * e1).sum(-1) + (e2 * e2).sum(-1)
    area = 0.5 * torch.cross(e0, -e2, dim=-1).norm(dim=-1)
    return (1.0 - _SQRT3_4 * area / (l2 + 1e-12)).mean()


def render_sil_and_ids(ctx, verts_t, faces_t, mvp):
    pos = transform_to_clip(verts_t, mvp)
    rast, _ = dr.rasterize(ctx, pos, faces_t, resolution=[IMG_RES, IMG_RES])
    ones = torch.ones(1, verts_t.shape[0], 1, dtype=torch.float32, device=DEVICE)
    col, _ = dr.interpolate(ones, rast, faces_t)
    sil = dr.antialias(col, rast, pos, faces_t)[0, :, :, 0]
    tid = rast[0, :, :, 3].long() - 1          # -1 = background
    return sil, tid


def extrude_face(V, Fa, fi, dist):
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    a, b, c = Fa[fi]
    tri = V[[a, b, c]]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= (np.linalg.norm(n) + 1e-12)
    base = len(V)
    V2 = np.vstack([V, tri + n * dist])
    a2, b2, c2 = base, base + 1, base + 2
    keep = np.delete(Fa, fi, axis=0)
    newf = np.array([(a, b, b2), (a, b2, a2), (b, c, c2), (b, c2, b2),
                     (c, a, a2), (c, a2, c2), (a2, b2, c2)], np.int64)
    return V2, np.vstack([keep, newf]), (int(a), int(b), int(c))


def check_watertight(Fa):
    cnt = collections.Counter()
    for a, b, c in Fa:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(min(e), max(e))] += 1
    bad = [k for k, v in cnt.items() if v != 2]
    return len(bad) == 0, len(bad)


def find_candidates(ctx, V, Fa, gt_fg, mvps, mean_edge):
    """GREEN blobs -> root faces, isolated in 3D."""
    vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(Fa, np.int32), dtype=torch.int32, device=DEVICE)
    ctrs = V[Fa].mean(1)
    cands = []                                  # (score, face_idx)
    for i in range(len(mvps)):
        sil, tid = render_sil_and_ids(ctx, vt, ft, mvps[i])
        pred = (sil > 0.5).cpu().numpy()
        tid = tid.cpu().numpy()
        green = gt_fg[i] & ~binary_dilation(pred, iterations=DILATE)
        lab, nb = cc_label(green)
        for k in range(1, nb + 1):
            m = lab == k
            sz = int(m.sum())
            if sz < MIN_BLOB: continue
            ys, xs = np.where(m)
            cy, cx = ys.mean(), xs.mean()
            pys, pxs = np.where(pred)
            if len(pys) == 0: continue
            d2 = (pys - cy) ** 2 + (pxs - cx) ** 2
            j = int(np.argmin(d2))
            f_root = int(tid[pys[j], pxs[j]])
            if f_root < 0 or f_root >= len(Fa): continue
            cands.append((sz, f_root))
    cands.sort(reverse=True)
    chosen = []
    for sz, fi in cands:
        c = ctrs[fi]
        if any(np.linalg.norm(c - ctrs[fj]) < ISO_DIST * mean_edge
               for _, fj in chosen):
            continue
        chosen.append((sz, fi))
        if len(chosen) >= K_MAX: break
    return chosen


def speculative_round(ctx, V, Fa, gt_sils_t, mvps, mean_edge, rnd):
    """One round: select K, extrude, A/B optimize, Step C commit/rollback.
    Returns new V, Fa, committed op records."""
    cands = find_candidates(ctx, V, Fa,
                            [(g > 0.5).cpu().numpy() for g in gt_sils_t],
                            mvps, mean_edge)
    if not cands:
        print(f"[round {rnd}] no candidates", flush=True)
        return V, Fa, []
    print(f"[round {rnd}] {len(cands)} candidates: "
          f"{[(s, f) for s, f in cands]}", flush=True)

    V0, Fa0 = V.copy(), np.asarray(Fa, np.int64).copy()
    Vc, Fc = V0.copy(), Fa0.copy()
    ops = []
    for sz, fi_orig in cands:
        # face indices shift as earlier extrudes delete faces; find by triple
        tgt = set(Fa0[fi_orig])
        cur = next((i for i, ff in enumerate(Fc) if set(ff) == tgt), None)
        if cur is None: continue                # face already consumed
        Vc, Fc, triple = extrude_face(Vc, Fc, cur, EXTRUDE_LEN * mean_edge)
        ops.append({"triple": triple, "blob": int(sz)})
    K = len(ops)
    if K == 0: return V, Fa, []
    nb = len(Fc) - 7 * K
    spec_slices = [np.arange(nb + 7 * k, nb + 7 * (k + 1)) for k in range(K)]
    nvb = len(Vc) - 3 * K
    spec_vidx = [np.arange(nvb + 3 * k, nvb + 3 * (k + 1)) for k in range(K)]

    verts_t = torch.tensor(Vc, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_all = torch.tensor(Fc, dtype=torch.int32, device=DEVICE)
    faces_base = faces_all[:nb]
    faces_ops = [faces_all[torch.tensor(s, device=DEVICE).long()] for s in spec_slices]
    theta = torch.zeros(K, device=DEVICE, requires_grad=True)
    smask = torch.zeros(len(Vc), dtype=torch.bool, device=DEVICE)
    for s in spec_vidx: smask[torch.tensor(s, device=DEVICE).long()] = True
    opt_pos = torch.optim.Adam([verts_t], lr=LR_POS)
    opt_p = torch.optim.Adam([theta], lr=LR_P)

    def spec_loss(force_p=None):
        p = torch.sigmoid(theta) if force_p is None else force_p
        total = torch.tensor(0.0, device=DEVICE)
        for i in range(len(mvps)):
            s_base, _ = render_sil_and_ids(ctx, verts_t, faces_base, mvps[i])
            s_mix = s_base
            for k in range(K):
                fk = torch.cat([faces_base, faces_ops[k]], 0)
                s_k, _ = render_sil_and_ids(ctx, verts_t, fk, mvps[i])
                s_mix = s_mix + p[k] * (s_k - s_base.detach()).clamp(min=0.0)
            s_mix = s_mix.clamp(0, 1)
            gt_i = gt_sils_t[i]
            region = (((gt_i > 0.01) & (s_base.detach() < 0.99))
                      | (s_mix.detach() > s_base.detach() + 0.01))
            err = (s_mix - gt_i).abs()
            total = total + err[region].sum() / (region.sum() + 1)
        return total / len(mvps) + W_OCCAM * torch.sigmoid(theta).sum()

    for r in range(AB_ROUNDS):
        for _ in range(POS_ITERS):
            opt_pos.zero_grad()
            loss = spec_loss(force_p=torch.ones(K, device=DEVICE))
            loss.backward()
            verts_t.grad[~smask] = 0
            opt_pos.step()
        opt_p.zero_grad()
        loss = spec_loss()
        loss.backward()
        verts_t.grad = None
        opt_p.step()
    p_fin = torch.sigmoid(theta).detach().cpu().numpy()
    print(f"[round {rnd}] p = {np.round(p_fin, 3).tolist()}", flush=True)

    # Step C: rebuild from V0/Fa0 applying only accepted ops (exact rollback)
    Vn, Fn = V0.copy(), Fa0.copy()
    committed = []
    Vopt = verts_t.detach().cpu().numpy()
    for k, op in enumerate(ops):
        if p_fin[k] <= 0.5: continue
        tgt = set(op["triple"])
        cur = next((i for i, ff in enumerate(Fn) if set(ff) == tgt), None)
        if cur is None: continue
        Vn, Fn, _ = extrude_face(Vn, Fn, cur, EXTRUDE_LEN * mean_edge)
        Vn[-3:] = Vopt[spec_vidx[k]]           # carry optimized positions
        committed.append({**op, "p": float(p_fin[k]), "round": rnd})
    ok, nbad = check_watertight(Fn)
    print(f"[round {rnd}] committed {len(committed)}/{K} watertight={ok}", flush=True)
    assert ok, f"non-manifold after round {rnd}: {nbad} bad edges"
    return Vn, Fn, committed


def settle(ctx, V, Fa, gt, gtd, mvps, steps, escape_fn):
    """Full-loss settle with L_qual; wraps cow_v13 machinery."""
    _orig_edge = cow_v13.edge_length_loss

    def _edge_q(v, f):
        return _orig_edge(v, f) + (W_QUAL / max(W_EDGE, 1e-9)) * _qual_loss(v, f)
    cow_v13.edge_length_loss = _edge_q
    try:
        v, iou = cow_v13.optimize_phase(ctx, V, np.asarray(Fa, np.int32), gt, gtd,
                                        mvps, steps, "settle", settle=True,
                                        use_fold=True, use_tube=True)
    finally:
        cow_v13.edge_length_loss = _orig_edge
    return v, iou


_CUR_ADJ = None
_MVPS = None
_GT = None
_orig_tube = cow_v13.tube_mask


def _set_faces(Fa):
    global _CUR_ADJ
    adj = collections.defaultdict(set)
    for a, b, c in np.asarray(Fa):
        a, b, c = int(a), int(b), int(c)
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    _CUR_ADJ = adj


@torch.no_grad()
def _tube_gated(v, excl, mean_edge):
    thin = _orig_tube(v, excl, mean_edge)
    if _MVPS is None or _CUR_ADJ is None: return thin
    esc = escape_mask(v, _MVPS, _GT, dilate=DILATE)
    keep = _propagate_flag(thin.cpu().numpy(), esc.cpu().numpy(), _CUR_ADJ)
    return torch.from_numpy(keep).to(v.device)


cow_v13.tube_mask = _tube_gated


def heldout_exam(ctx, v, t):
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gv = normalize_to_range(gv)
    gvt = torch.tensor(gv, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gf, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5, azimuths_deg=azs, device=DEVICE)
    if isinstance(mv, tuple): mv = mv[0]
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(np.asarray(t, np.int32), dtype=torch.int32, device=DEVICE)
    gs = render_views_n(ctx, gvt, gft, mv)
    ps = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; maxblob = 0
    for i in range(16):
        g = gs[i] > 0.5; p = ps[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2); hair += int(out.sum())
        lab, nbk = cc_label(out)
        for k in range(1, nbk + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return inter / max(un, 1), hair, maxblob


MODE = os.environ.get("MODE", "6v")


def main():
    global _MVPS, _GT
    torch.manual_seed(0); np.random.seed(0)
    if MODE == "64v":
        import run_64v                     # safe: main() guarded
        ctx = dr.RasterizeCudaContext()
        gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
        max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
        mvps, views = run_64v.star_cameras(max_r)
        gt, gtd, _gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
        run_64v._MVPS, run_64v._GT = mvps, gt   # its tube gate now active
        cow_v13.N_VIEWS = 64                    # settle loops over 64 views
    else:
        scene = setup_scene(SHAPE, DEVICE)
        ctx, mvps = scene["ctx"], scene["mvps"]
        gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    _MVPS, _GT = mvps, gt
    gt_sils_t = [torch.from_numpy((gt[i] < 128).astype(np.float32)).to(DEVICE)
                 for i in range(len(mvps))]

    z = np.load(BASE_NPZ)
    V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
    ok, _ = check_watertight(Fa)
    print(f"[base] {BASE_NPZ} V={len(V)} F={len(Fa)} watertight={ok}", flush=True)

    def iou_fn(vv, ff):
        vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(ff, dtype=np.int32), dtype=torch.int32, device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    def escape_fn(Vnp):
        vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
        return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

    base_ho = heldout_exam(ctx, V, Fa)
    print(f"[base] train6={iou_fn(V, Fa):.4f} ho16={base_ho[0]:.4f} "
          f"hair={base_ho[1]} maxblob={base_ho[2]}", flush=True)

    t0 = time.time()
    program = []
    for rnd in range(ROUNDS):
        src, dst = [], []
        es = set()
        for a, b, c in Fa:
            for e in ((a, b), (b, c), (c, a)):
                kk = (min(e), max(e))
                if kk in es: continue
                es.add(kk); src.append(e[0]); dst.append(e[1])
        me = float(np.linalg.norm(V[np.array(src)] - V[np.array(dst)], axis=1).mean())
        V, Fa, committed = speculative_round(ctx, V, Fa, gt_sils_t, mvps, me, rnd)
        program += committed
        if committed:
            _set_faces(Fa)
            V, iou_s = settle(ctx, V, Fa, gt, gtd, mvps, 300, escape_fn)
            print(f"[round {rnd}] settle iou={iou_s:.4f} V={len(V)}", flush=True)

    # final surgery pass (v22 escape-gated)
    _set_faces(Fa)
    V, Fa = surgery(np.asarray(V, np.float64), np.asarray(Fa, np.int64),
                    iou_fn=iou_fn, iou_budget=2e-4, global_cap=6e-4,
                    max_rounds=4, max_grow=4, escape_fn=escape_fn)
    iou_final = iou_fn(V, Fa)
    Fa = np.asarray(Fa, np.int32)
    with open(f"{OUT}/cow_{TAG}.obj", "w") as fh:
        for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
    np.savez(f"{OUT}/cow_{TAG}.npz", verts=V, tris=Fa)
    json.dump(program, open(f"{OUT}/cow_{TAG}_program.json", "w"), indent=1)
    ho, hair, mb = heldout_exam(ctx, V, Fa)
    print(f"\n=== {TAG.upper()} RESULT ===")
    print(f"operators committed: {len(program)}")
    print(f"train6 IoU={iou_final:.4f}")
    print(f"heldout16: IoU={ho:.4f} hair_px={hair} maxblob={mb}")
    print(f"baseline : IoU={base_ho[0]:.4f} hair_px={base_ho[1]} maxblob={base_ho[2]}")
    print(f"delta ho16 = {(ho - base_ho[0]) * 100:+.2f} pts")
    print(f"V={len(V)} time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
