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

def ensure_plugs(shape, cache, res=128, pad=24, log=print):
    """Build the plug cache (same format as hull_locate.find_tunnel_by_hull) when no chain has written it yet:
    carve the voting hull from the GT silhouettes (GPU), downsample, run the closing ladder."""
    if os.path.exists(cache): return
    import torch, nvdiffrast.torch as dr
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    from hull_field import build_vote_hull
    from hull_locate import clean_hull, downsample, genus_solid, find_plugs
    import run_64v
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj")); gvn = normalize_to_range(gv)
    mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
    ctx = dr.RasterizeCudaContext()
    HF = build_vote_hull(ctx, mvps, gvn, gf, None, "cuda", nres=256, hires=int(os.environ.get("HULL_HIRES", "512")), vote=2)
    hs = np.pad(downsample(clean_hull(np.asarray(HF.hull).astype(bool)), res), pad); g0 = genus_solid(hs)
    log(f"[air] {shape}: hull genus {g0}, locating plugs")
    plugs = find_plugs(hs, g0, log=lambda m: None) if g0 > 0 else []
    v2w = grid_frame(shape, res, pad)
    for p in plugs: p["throat_w"] = v2w(p["throat_vox"]).tolist(); p["cen_w"] = v2w(p["cen_vox"]).tolist(); p["key"] = ["hull", [round(float(x), 3) for x in p["throat_w"]]]
    np.savez_compressed(cache + ".blocks.npz", hs=hs, **{f"plug{i}": p["block_vox"] for i, p in enumerate(plugs)})
    json.dump([{k: v for k, v in p.items() if k != "block_vox"} for p in plugs], open(cache, "w"))

def air_loops(shape, cache=None, min_mouth=20, log=print):
    """One closed curve per independent hull tunnel, entirely in hull air (world coordinates).
    Sealed block (union of plug blocks, air voxels) with m mouths to the outer air -> m-1 loops:
    mouth0 -> mouth j through the block, back through the outer air."""
    from skimage.graph import route_through_array
    cache = cache or f"{OUTD}/hull_plugs_{shape}_128.json"
    ensure_plugs(shape, cache, log=log)
    memo = cache + ".airloops.npz"
    if os.path.exists(memo) and os.path.getmtime(memo) >= os.path.getmtime(cache):
        z = np.load(memo); return [z[k] for k in sorted(z.files, key=lambda k: int(k[4:]))]
    plugs = json.load(open(cache)); bz = np.load(cache + ".blocks.npz"); hs = bz["hs"].astype(bool)
    P = np.zeros_like(hs)
    for i in range(len(plugs)): b = bz[f"plug{i}"]; P[b[:, 0], b[:, 1], b[:, 2]] = True
    P &= ~hs; W0 = ~hs & ~P
    lab, _ = ndimage.label(W0, structure=S26); sizes = np.bincount(lab.ravel()); sizes[0] = 0; W0 = lab == sizes.argmax()
    v2w = grid_frame(shape); loops = []
    # Construction (2026-10-02, shape-agnostic): H1 generators of the hull SURFACE (tree-cotree on the marching-cubes
    # surface of hs), pushed into the air: anchor points are moved off the surface up the distance-to-solid gradient
    # (towards the middle of the air channel) and joined by shortest routes in the air weighted by 1/distance. Of the
    # 2g surface loops the g "meridians" become air loops threading a tunnel, the g "longitudes" become contractible;
    # the validation below keeps the g independent threading ones. Plug blocks are only used to confirm the count.
    try:
        from skimage.measure import marching_cubes
        import handle_guard as hg
        from scipy.ndimage import distance_transform_edt
        vv, ff, _, _ = marching_cubes(hs.astype(np.float32), 0.5); ff = ff.astype(np.int64)
        hl = [l for l in hg.tree_cotree_loops(vv, ff, max_loops=96) if len(l) >= 10]
        Wair = ~hs; dair = distance_transform_edt(Wair); _, near = distance_transform_edt(hs, return_indices=True)
        cost = np.where(Wair, 1.0 / (dair + 0.5), -1.0); N = hs.shape[0]
        nbr = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
        def lift(q):
            q = np.clip(np.round(q).astype(int), 0, N - 1)
            if not Wair[tuple(q)]: q = near[:, q[0], q[1], q[2]]
            for _ in range(8):                                   # climb the distance field away from the solid
                cand = np.clip(q + nbr, 0, N - 1); dv = dair[cand[:, 0], cand[:, 1], cand[:, 2]]
                if dv.max() <= dair[tuple(q)]: break
                q = cand[int(dv.argmax())]
            return tuple(int(x) for x in q)
        for l in hl:
            P = vv[np.asarray(l)]; step = max(1, len(P) // 24); anchors = [lift(P[k]) for k in range(0, len(P), step)]
            anchors = [a for i, a in enumerate(anchors) if i == 0 or a != anchors[i - 1]]
            pts = []
            okloop = True
            for i in range(len(anchors)):
                a0, a1 = anchors[i], anchors[(i + 1) % len(anchors)]
                if a0 == a1: continue
                try: seg, _ = route_through_array(cost, a0, a1, fully_connected=True)
                except Exception: okloop = False; break
                pts += seg[:-1]
            if okloop and len(pts) >= 4: loops.append(v2w(np.array(pts, float)))
        log(f"[air] hull surface: {len(ff)} faces, {len(hl)} generator loops -> {len(loops)} air loops before validation")
    except Exception as e: log(f"[air] surface-based construction failed ({e}); falling back to plug centrelines")
    if not loops:
        _, idx = ndimage.distance_transform_edt(~W0, return_indices=True); costW = np.where(W0, 1.0, -1.0)
        for i, p in enumerate(plugs):
            pv = np.asarray(p.get("path_vox", []), float)
            if len(pv) < 2: continue
            q = np.clip(np.round(pv).astype(int), 0, hs.shape[0] - 1)
            e0 = tuple(int(x) for x in idx[:, q[0, 0], q[0, 1], q[0, 2]]); e1 = tuple(int(x) for x in idx[:, q[-1, 0], q[-1, 1], q[-1, 2]])
            if e0 == e1: continue
            try: back, _ = route_through_array(costW, e1, e0, fully_connected=True)
            except Exception: continue
            loops.append(v2w(np.vstack([np.array(e0, float)[None], pv, np.array(e1, float)[None], np.array(back[1:-1], float)])))
    # Validate against the HULL itself (GT-free): the hull surface's own H1 generators must link the air loops with
    # rank == hull genus. Drop loops that no hull loop links (construction failed) and duplicates of the same
    # tunnel (two plug blocks on one tunnel) by keeping a maximal independent column set.
    if int(os.environ.get("AIR_NO_VALIDATE", "0")): return loops
    try:
        from skimage.measure import marching_cubes
        import handle_guard as hg
        vv, ff, _, _ = marching_cubes(hs.astype(np.float32), 0.5); vv = v2w(vv); ff = ff.astype(np.int64)   # full res: half res has spurious handles
        hl = [l for l in hg.tree_cotree_loops(vv, ff, max_loops=256) if len(l) >= 10]
        Mh = np.round(np.array([[linking(vv[np.asarray(l)], A) for A in loops] for l in hl])) if hl and loops else np.zeros((0, len(loops)))
        from hull_locate import genus_solid
        g0 = int(genus_solid(hs)); keep = []
        for k in range(len(loops)):
            if Mh.size and np.any(Mh[:, k] != 0) and np.linalg.matrix_rank(Mh[:, keep + [k]]) > len(keep): keep.append(k)
        log(f"[air] hull genus {g0}: {len(loops)} loops built, {len(keep)} independent and threaded -> {'OK' if len(keep) == g0 else 'INCOMPLETE'}")
        loops = [loops[k] for k in keep]
    except Exception as e: log(f"[air] hull validation skipped ({e})")
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

def loop_crossings(V, F, AIR, step=0.02):
    """Where do the hull-tunnel air loops pass THROUGH the mesh surface? A tunnel the mesh still seals is crossed
    (twice: into the membrane and out); an open one is not. Exact for thin membranes (ray casting along the loop).
    Returns (per-loop crossing counts, all crossing points [k,3])."""
    import open3d as o3d
    rs = o3d.t.geometry.RaycastingScene(); rs.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32))))
    counts, pts = [], []
    for A in AIR:
        P = np.vstack([A, A[:1]]); a = []
        for u, v in zip(P[:-1], P[1:]):
            n = max(1, int(np.linalg.norm(v - u) / step)); a += [u + (v - u) * t / n for t in range(n)]
        a = np.array(a); b = np.roll(a, -1, 0); d = b - a; L = np.linalg.norm(d, axis=1); ok = L > 1e-9; dn = d[ok] / L[ok][:, None]
        t = rs.cast_rays(o3d.core.Tensor(np.hstack([a[ok], dn]).astype(np.float32)))["t_hit"].numpy(); hit = t <= L[ok]
        counts.append(int(hit.sum())); pts.append((a[ok] + dn * np.where(np.isfinite(t), t, 0)[:, None])[hit])
    return counts, (np.vstack(pts) if any(len(x) for x in pts) else np.zeros((0, 3)))

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
        if not AIR: print(0 if os.path.exists(f"{OUTD}/hull_plugs_{shape}_128.json") else -1, int(genus(V, F)), 0, 0); sys.exit(0)   # 0 loops = genus-0 hull (valid), -1 = unavailable
        a = audit(V, F, AIR); print(a["rank"], a["genus"], len(AIR), a["tiny_loops"]); sys.exit(0)
    AIR = air_loops(shape, log=lambda m: print(m, file=sys.stderr))
    print(f"# {shape}: {len(AIR)} hull tunnel air loops"); ok = n = 0
    for pat in sys.argv[1:]:
        for f in sorted(glob.glob(pat)):
            m = np.load(f); a = audit(m["verts"].astype(float), m["tris"].astype(np.int64), AIR)
            good = a["rank"] == len(AIR) and a["genus"] == len(AIR); ok += good; n += 1
            print(f"{os.path.basename(f):44s} genus {a['genus']}  tunnels realised {a['rank']}/{len(AIR)}  per tunnel {a['per_tunnel']}  tiny loops {a['tiny_loops']}  {'OK' if good else 'MISMATCH'}", flush=True)
    print(f"# strict topology: {ok}/{n}")
