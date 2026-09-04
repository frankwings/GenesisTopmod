"""Phase 1c — speculative GROW + CARVE via real DLFL extrude, seed-style sampling.

Changes vs phase1b:
  1. Operators applied by the real DLFL extrude_face (topmod/high_level_ops)
     through an obj roundtrip (~0.06s/round) — formal manifold guarantee,
     NumPy replica deleted.
  2. Seed-style candidate sampling (DMesh philosophy at operator level):
     over-provision K<=16 mixed candidates, let p_k gating select.
       GREEN blob (gt yes / pred no)  -> grow  (dist = +3 x mean_edge)
       RED   blob (pred yes / gt no)  -> carve (dist = -3 x mean_edge)
     No dilation filter (it killed 73% of signal), MIN_BLOB 15.
  3. Symmetric opacity compositing (phase1c_toy validated):
       S_mix = S_base + sum_k p_k (S_k - S_base.detach())     (no clamp)
     active region includes the RED side.
  4. Depth term in spec loss (dent depth invisible to silhouettes):
       total += W_DEPTH_SPEC * p_k * (E_k - E_base)   E = masked |ndc_z - gt|.

Run (64v): MODE=64v SHAPE=armadillo TAG=armadillo_p1c64 \
           BASE_NPZ=/tmp/liou_cow_viz/cow_armadillo_64v.npz python3 phase1c_pipeline.py
"""
import sys, os, json, time, collections, tempfile
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
from scipy.ndimage import label as cc_label
import nvdiffrast.torch as dr

import cow_v13
# importing phase1b_pipeline installs the escape-gated tube mask patch
import phase1b_pipeline as p1b
from phase1b_pipeline import (render_sil_and_ids, check_watertight, settle,
                              _set_faces, heldout_exam, _qual_loss)
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               load_obj, normalize_to_range, BUNNY_PATH, IMG_RES)
from eval_extrude_v3 import render_sil_and_depth
from surgery_lib import surgery
from escape_util import escape_mask
from topmod.io import from_obj, to_triangle_arrays
from topmod.high_level_ops import (extrude_face as dlfl_extrude,
                                   triangulate_face, stellate as dlfl_stellate,
                                   subdivide_edge as dlfl_subdivide_edge)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", f"{SHAPE}_p1c")
BASE_NPZ = os.environ.get("BASE_NPZ", f"{OUT}/cow_{SHAPE}_64v.npz")
MODE = os.environ.get("MODE", "64v")
W_QUAL = 0.01
ROUNDS = int(os.environ.get("ROUNDS", "6"))   # slot cutting eats in from the rim
K_MAX = 16
MIN_BLOB = 15
AB_ROUNDS = 20
POS_ITERS = 3
LR_POS = 5e-3
LR_P = 0.5
W_OCCAM = 0.0005
W_DEPTH_SPEC = 1.0
OP_LEN = 3.0           # grow dist = OP_LEN x mean_edge
CARVE_LEN = 1.5        # carve |dist| (deep narrow init self-intersects)
ISO_DIST = 3.0
DILATE = 2
cow_v13.TUBE_THR = 0.4
# Phase 1g knobs
GROW_ONLY = os.environ.get("GROW_ONLY", "0") == "1"
SUBDIV = os.environ.get("SUBDIV", "0") == "1"     # DLFL midpoint subdiv at grow sites
SUBDIV_MIN = int(os.environ.get("SUBDIV_MIN", "100"))  # px blob size threshold
FENCE = os.environ.get("FENCE", "0") == "1"       # hull fence on spec verts
W_FENCE = float(os.environ.get("W_FENCE", "50.0"))
_HULL = None                                       # HullField, set in main()


def dlfl_extrude_arrays(V, Fa, fi, dist):
    """Real DLFL extrude at face fi; triangulated; verts appended at end.
    Returns V2, F2 (face order NOT normalized), new_vidx (3 new verts)."""
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
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
    newf = dlfl_extrude(mesh, faces[fi], dist=float(dist))
    for f in list(newf):
        if len(f.vertices()) > 3:
            triangulate_face(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2 = np.asarray(vv, float); F2 = np.asarray(ff, np.int64)
    assert len(V2) == len(V) + 3 and np.allclose(V2[:len(V)], V, atol=1e-9), \
        "DLFL export reordered vertices"
    return V2, F2, np.arange(len(V), len(V) + 3)


DENSIFY_MIN = 150      # px: blobs this big get their root 1-ring stellated


def densify_faces(V, Fa, fids):
    """Stellate faces fids + their edge-neighbors (flat centroid vertex =
    geometry-neutral densify, 1 tri -> 3). Returns V2, F2, n_stellated."""
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    # expand to 1-ring (edge-adjacent faces share 2 verts)
    tgt = set(int(f) for f in fids)
    vsets = [set(map(int, f)) for f in Fa]
    for fi in list(tgt):
        for j, vs in enumerate(vsets):
            if j != fi and len(vsets[fi] & vs) == 2:
                tgt.add(j)
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, z in V: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    faces = list(mesh.iter_faces())
    for fi in sorted(tgt):
        dlfl_stellate(mesh, faces[fi])          # Face objects stay valid
    vv, ff = to_triangle_arrays(mesh)
    V2 = np.asarray(vv, float); F2 = np.asarray(ff, np.int64)
    assert len(V2) == len(V) + len(tgt) and np.allclose(V2[:len(V)], V, atol=1e-9)
    return V2, F2, len(tgt)


def dlfl_subdivide_arrays(V, Fa, fids, expand_ring=True):
    """Phase 1g: DLFL midpoint subdivision of faces fids + 1-ring.
    subdivide_edge every edge of the region, then triangulate all non-tri
    faces (fan from a midpoint would be degenerate; triangulate_face is not).
    Real resolution increase: long edges actually get split, unlike stellate.
    """
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    if os.environ.get("GENERIC_OPS") == "1":
        assert len(fids) == len(Fa), "generic subdivision supports all-faces only"
        from generic_ops import subdivide_all_np
        return subdivide_all_np(V, Fa)
    tgt = set(int(f) for f in fids)
    vsets = [set(map(int, f)) for f in Fa]
    for fi in (list(tgt) if expand_ring else []):   # expand to edge-adjacent ring
        for j, vs in enumerate(vsets):
            if j != fi and len(vsets[fi] & vs) == 2:
                tgt.add(j)
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, z in V: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    faces = list(mesh.iter_faces())
    edges = {}
    for fi in tgt:
        for he in faces[fi].halfedges():
            edges[id(he.edge)] = he.edge
    for e in edges.values():
        dlfl_subdivide_edge(mesh, e)
    # stellate (centroid) rather than fan-triangulate: fan diagonals can
    # duplicate existing boundary edges when adjacent n-gons share two edges
    # (produced doubled faces in testing); centroid split cannot collide.
    for f in list(mesh.faces.values()):
        if len(f.vertices()) > 3:
            dlfl_stellate(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2 = np.asarray(vv, float); F2 = np.asarray(ff, np.int64)
    assert np.allclose(V2[:len(V)], V, atol=1e-9), "subdiv reordered verts"
    return V2, F2, len(edges)


def find_candidates(ctx, V, Fa, gt_fg, mvps, mean_edge):
    """Seed-style: green AND red blobs, no dilation, mixed grow/carve."""
    vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(Fa, np.int32), dtype=torch.int32, device=DEVICE)
    ctrs = V[np.asarray(Fa)].mean(1)
    cands = []                                  # (score, face_idx, sign)
    for i in range(len(mvps)):
        sil, tid = render_sil_and_ids(ctx, vt, ft, mvps[i])
        pred = (sil > 0.5).cpu().numpy()
        tid = tid.cpu().numpy()
        pairs = ((gt_fg[i] & ~pred, +1),) if GROW_ONLY else \
                ((gt_fg[i] & ~pred, +1), (pred & ~gt_fg[i], -1))
        for mask, sign in pairs:
            lab, nb = cc_label(mask)
            for k in range(1, nb + 1):
                m = lab == k
                sz = int(m.sum())
                if sz < MIN_BLOB: continue
                ys, xs = np.where(m)
                cy, cx = ys.mean(), xs.mean()
                if sign < 0:
                    # red: seed at the blob's SILHOUETTE RIM (slot mouth), not
                    # centroid — a pit mid-webbing is invisible in projection;
                    # a notch at the rim shrinks the silhouette immediately.
                    from scipy.ndimage import binary_dilation as _bd
                    rim = m & _bd(~pred, iterations=1)
                    if rim.any():
                        rys, rxs = np.where(rim)
                    else:                       # fully interior blob: skip
                        continue
                    j = int(np.argmin((rys - cy) ** 2 + (rxs - cx) ** 2))
                    f_root = int(tid[rys[j], rxs[j]])
                else:
                    # green: nearest pred pixel to blob centroid
                    pys, pxs = np.where(pred)
                    if len(pys) == 0: continue
                    j = int(np.argmin((pys - cy) ** 2 + (pxs - cx) ** 2))
                    f_root = int(tid[pys[j], pxs[j]])
                if f_root < 0 or f_root >= len(Fa): continue
                cands.append((sz, f_root, sign))
    cands.sort(reverse=True)
    chosen = []
    for sz, fi, sign in cands:
        c = ctrs[fi]
        if any(np.linalg.norm(c - ctrs[fj]) < ISO_DIST * mean_edge
               for _, fj, _ in chosen):
            continue
        chosen.append((sz, fi, sign))
        if len(chosen) >= K_MAX: break
    return chosen


def speculative_round(ctx, V, Fa, gt_sils_t, gt_deps_t, gt_fgs_t, mvps,
                      mean_edge, rnd):
    cands = find_candidates(ctx, V, Fa,
                            [(g > 0.5).cpu().numpy() for g in gt_sils_t],
                            mvps, mean_edge)
    if not cands:
        print(f"[round {rnd}] no candidates", flush=True)
        return V, Fa, []
    print(f"[round {rnd}] {len(cands)} candidates: "
          f"{[(s, f, '+' if sg > 0 else '-') for s, f, sg in cands]}", flush=True)

    V0, Fa0 = V.copy(), np.asarray(Fa, np.int64).copy()
    nv0 = len(V0)
    Vc, Fc = V0.copy(), Fa0.copy()
    ops = []
    for sz, fi_orig, sign in cands:
        tgt = set(Fa0[fi_orig])
        cur = next((i for i, ff in enumerate(Fc) if set(ff) == tgt), None)
        if cur is None: continue                # face consumed by earlier op
        ln = OP_LEN if sign > 0 else CARVE_LEN
        Vc, Fc, vidx = dlfl_extrude_arrays(Vc, Fc, cur, sign * ln * mean_edge)
        ops.append({"triple": tuple(int(x) for x in Fa0[fi_orig]),
                    "blob": int(sz), "sign": int(sign)})
    K = len(ops)
    if K == 0: return V, Fa, []

    # ownership by new-vertex id range: op k owns verts [nv0+3k, nv0+3k+3)
    owner = np.full(len(Fc), -1)
    for i, f in enumerate(Fc):
        hi = [x for x in f if x >= nv0]
        if hi: owner[i] = (min(hi) - nv0) // 3
    order = np.concatenate([np.where(owner < 0)[0]] +
                           [np.where(owner == k)[0] for k in range(K)])
    Fc = Fc[order]
    blocks = [int((owner == k).sum()) for k in range(K)]
    nb = int((owner < 0).sum())
    off = np.cumsum([nb] + blocks)
    spec_slices = [np.arange(off[k], off[k + 1]) for k in range(K)]
    spec_vidx = [np.arange(nv0 + 3 * k, nv0 + 3 * (k + 1)) for k in range(K)]

    verts_t = torch.tensor(Vc, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_all = torch.tensor(Fc, dtype=torch.int32, device=DEVICE)
    faces_base = faces_all[:nb]
    faces_ops = [faces_all[torch.tensor(s, device=DEVICE).long()] for s in spec_slices]
    # original faces (for restoring other ops' sites in per-op renders)
    orig_faces = [torch.tensor(np.array(op["triple"], np.int32)[None, :],
                               dtype=torch.int32, device=DEVICE) for op in ops]
    base_faces_t = torch.tensor(Fa0.astype(np.int32), dtype=torch.int32,
                                device=DEVICE)     # pristine base mesh
    theta = torch.zeros(K, device=DEVICE, requires_grad=True)
    smask = torch.zeros(len(Vc), dtype=torch.bool, device=DEVICE)
    for s in spec_vidx: smask[torch.tensor(s, device=DEVICE).long()] = True
    opt_pos = torch.optim.Adam([verts_t], lr=LR_POS)
    opt_p = torch.optim.Adam([theta], lr=LR_P)

    def spec_loss(force_p=None):
        """Per-op LOCALIZED signed contrast (fix for signal dilution):
        L_k = p_k * (err_k - err_base) computed ONLY on pixels the op changes.
        Local signal ~1e-2..1e-1 >> Occam 5e-4; a shared global region divided
        the per-op benefit down to ~1e-5 and the prior killed every candidate."""
        p = torch.sigmoid(theta) if force_p is None else force_p
        total = torch.tensor(0.0, device=DEVICE)
        for i in range(len(mvps)):
            gt_i = gt_sils_t[i]
            with torch.no_grad():
                s_base, _ = render_sil_and_ids(ctx, verts_t, base_faces_t, mvps[i])
                _, d_b, fg_b = render_sil_and_depth(ctx, verts_t, base_faces_t,
                                                    mvps[i])
            for k in range(K):
                restore = [orig_faces[j] for j in range(K) if j != k]
                fk = torch.cat([faces_base, faces_ops[k]] + restore, 0)
                s_k, _ = render_sil_and_ids(ctx, verts_t, fk, mvps[i])
                reg = (s_k.detach() - s_base).abs() > 0.005
                if not reg.any():
                    continue
                n = reg.sum() + 1
                e1 = (s_k - gt_i).abs()[reg].sum() / n
                e0 = (s_base - gt_i).abs()[reg].sum() / n
                total = total + p[k] * (e1 - e0)
                _, d_k, fg_k = render_sil_and_depth(ctx, verts_t, fk, mvps[i])
                m1 = reg & fg_k.detach() & gt_fgs_t[i]
                m0 = reg & fg_b & gt_fgs_t[i]
                if m0.any() or m1.any():
                    e1d = (d_k - gt_deps_t[i]).abs()[m1].sum() / (m1.sum() + 1)
                    with torch.no_grad():
                        e0d = (d_b - gt_deps_t[i]).abs()[m0].sum() / (m0.sum() + 1)
                    total = total + W_DEPTH_SPEC * p[k] * (e1d - e0d)
        return total / len(mvps) + W_OCCAM * torch.sigmoid(theta).sum()

    def fence_pen():
        """Phase 1g hull fence: spec verts must not grow outside the voting
        hull (3D evidence boundary); free within one voxel dead zone."""
        if _HULL is None: return torch.tensor(0.0, device=DEVICE)
        d = _HULL.dist(verts_t[smask])
        return W_FENCE * torch.relu(d - _HULL.pitch).mean()

    for r in range(AB_ROUNDS):
        for _ in range(POS_ITERS):
            opt_pos.zero_grad()
            loss = spec_loss(force_p=torch.ones(K, device=DEVICE)) + fence_pen()
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

    # Step C: rebuild from pristine V0/Fa0, re-apply accepted DLFL ops only
    Vn, Fn = V0.copy(), Fa0.copy()
    committed = []
    Vopt = verts_t.detach().cpu().numpy()
    for k, op in enumerate(ops):
        if p_fin[k] <= 0.5: continue
        tgt = set(op["triple"])
        cur = next((i for i, ff in enumerate(Fn) if set(ff) == tgt), None)
        if cur is None: continue
        ln = OP_LEN if op["sign"] > 0 else CARVE_LEN
        Vn, Fn, vidx = dlfl_extrude_arrays(Vn, Fn, cur, op["sign"] * ln * mean_edge)
        Vn[vidx] = Vopt[spec_vidx[k]]          # carry optimized positions
        committed.append({**op, "p": float(p_fin[k]), "round": rnd})
    ok, nbad = check_watertight(Fn)
    print(f"[round {rnd}] committed {len(committed)}/{K} watertight={ok}", flush=True)
    assert ok, f"non-manifold after round {rnd}: {nbad} bad edges"
    return Vn, Fn, committed


def main():
    torch.manual_seed(0); np.random.seed(0)
    if MODE == "64v":
        import run_64v                          # main() is __main__-guarded
        ctx = dr.RasterizeCudaContext()
        gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
        max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
        mvps, views = run_64v.star_cameras(max_r)
        gt, gtd, _gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
        run_64v._MVPS, run_64v._GT = mvps, gt   # its tube gate needs these
        cow_v13.N_VIEWS = 64
    else:
        scene = setup_scene(SHAPE, DEVICE)
        ctx, mvps = scene["ctx"], scene["mvps"]
        gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    p1b._MVPS, p1b._GT = mvps, gt
    p1b.SHAPE = SHAPE                           # heldout_exam reads it
    gt_sils_t = [torch.from_numpy((gt[i] < 128).astype(np.float32)).to(DEVICE)
                 for i in range(len(mvps))]
    gt_deps_t = [torch.from_numpy(np.asarray(gtd[i], np.float32)).to(DEVICE)
                 for i in range(len(mvps))]
    gt_fgs_t = [(g > 0.5) for g in gt_sils_t]

    z = np.load(BASE_NPZ)
    V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
    ok, _ = check_watertight(Fa)
    print(f"[base] {BASE_NPZ} V={len(V)} F={len(Fa)} watertight={ok}", flush=True)

    if FENCE and MODE == "64v":
        global _HULL
        from hull_field import build_vote_hull
        _HULL = build_vote_hull(ctx, mvps, normalize_to_range(gv), _gf, V, DEVICE)
        print(f"[base] hull fence ready pitch={_HULL.pitch:.4f} "
              f"W_FENCE={W_FENCE}", flush=True)

    def iou_fn(vv, ff):
        vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(ff, dtype=np.int32), dtype=torch.int32,
                          device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    def escape_fn(Vnp):
        vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
        return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

    base_ho = heldout_exam(ctx, V, Fa)
    print(f"[base] train={iou_fn(V, Fa):.4f} ho16={base_ho[0]:.4f} "
          f"hair={base_ho[1]} maxblob={base_ho[2]}", flush=True)

    t0 = time.time()
    program = []
    for rnd in range(ROUNDS):
        es = set()
        for a, b, c in Fa:
            for e in ((a, b), (b, c), (c, a)):
                es.add((min(e), max(e)))
        src = np.array([e[0] for e in es]); dst = np.array([e[1] for e in es])
        me = float(np.linalg.norm(V[src] - V[dst], axis=1).mean())
        # densify pre-pass: stubborn big blobs get finer knives (blade width =
        # face size; a 1-face notch is too small a fraction of a 500px blob)
        pre = find_candidates(ctx, V, Fa,
                              [(g > 0.5).cpu().numpy() for g in gt_sils_t],
                              mvps, me)
        big = [fi for sz, fi, sg in pre if sg < 0 and sz >= DENSIFY_MIN]
        if big:
            V, Fa, nst = densify_faces(V, Fa, big)
            ok, _ = check_watertight(Fa)
            print(f"[round {rnd}] densified {nst} faces around "
                  f"{len(big)} big blobs watertight={ok}", flush=True)
            assert ok
        if SUBDIV:
            # Phase 1g: real resolution at big GREEN sites -- a finger needs
            # edges shorter than the finger; stellate can't split long edges.
            big_g = [fi for sz, fi, sg in pre if sg > 0 and sz >= SUBDIV_MIN]
            if big_g:
                V, Fa, ne = dlfl_subdivide_arrays(V, Fa, big_g)
                ok, _ = check_watertight(Fa)
                print(f"[round {rnd}] subdivided {ne} edges at {len(big_g)} "
                      f"grow sites V={len(V)} watertight={ok}", flush=True)
                assert ok
        V, Fa, committed = speculative_round(ctx, V, Fa, gt_sils_t, gt_deps_t,
                                             gt_fgs_t, mvps, me, rnd)
        program += committed
        if committed:
            _set_faces(Fa)
            V, iou_s = settle(ctx, V, Fa, gt, gtd, mvps, 300, escape_fn)
            print(f"[round {rnd}] settle iou={iou_s:.4f} V={len(V)}", flush=True)

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
    print(f"operators committed: {len(program)} "
          f"(grow {sum(1 for o in program if o['sign'] > 0)}, "
          f"carve {sum(1 for o in program if o['sign'] < 0)})")
    print(f"train IoU={iou_final:.4f}")
    print(f"heldout16: IoU={ho:.4f} hair_px={hair} maxblob={mb}")
    print(f"baseline : IoU={base_ho[0]:.4f} hair_px={base_ho[1]} maxblob={base_ho[2]}")
    print(f"delta ho16 = {(ho - base_ho[0]) * 100:+.2f} pts")
    print(f"V={len(V)} time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
