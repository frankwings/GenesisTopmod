"""Hull-located tunnel plugs (2026-09-15, Boss's formulation).

Closing-radius ladder on the cleaned space-carved hull. At radius R, the voxels added by the closing
at THIS radius (increment) that lower the hull genus by exactly 1 are one tunnel's sealing sheet
("plug"). Claimed plugs are frozen into the solid (they become hull for all later radii), so a sealed
tunnel cannot yield a second plug. For each plug the tunnel centreline is the 3D skeleton of the plug
block (longest path, spurs dropped); the throat is the centreline point closest to the hull wall; thin
sheet-like plugs use the sheet normal through the centroid instead. The add_handle face pair is found
by walking the centreline through the coarse mesh occupancy: the out->in and in->out transitions
around the throat are the two membrane crossings; nearest triangles to them are the pair. No rays.
"""
import os, json, time
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation, feature

S26 = np.ones((3, 3, 3), dtype=bool)
NB26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def genus_solid(vol):
    return 1 - measure.euler_number(ndimage.binary_fill_holes(vol), connectivity=1)


def clean_hull(hull_full):
    S3 = ndimage.generate_binary_structure(3, 1)
    hc = ndimage.binary_opening(ndimage.binary_closing(hull_full, S3, iterations=2), S3, iterations=2)
    lab, n = ndimage.label(hc)
    if n > 1: hc = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return ndimage.binary_fill_holes(hc)


def downsample(hc, res):
    N = hc.shape[0]
    if res == N: return hc.copy()
    if N % res == 0:
        f = N // res
        return hc.reshape(res, f, res, f, res, f).max(axis=(1, 3, 5))
    return ndimage.zoom(hc.astype(float), res / N, order=0).astype(bool)


def _closing(edt_bg, R):
    dil = edt_bg <= R
    return ndimage.distance_transform_edt(dil) > R


def _split_merged(comp, base, g_base, want):
    """comp lowers genus by >=2: watershed on its EDT, greedily merge pieces (highest peak first)
    until a union lowers genus by exactly 1; repeat on the remainder."""
    edt_c = ndimage.distance_transform_edt(comp).astype(np.float32)
    pk = feature.peak_local_max(edt_c, min_distance=6, labels=comp.astype(int), exclude_border=False)
    if len(pk) < 2: return [comp]
    mk = np.zeros(comp.shape, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    ws = segmentation.watershed(-edt_c, mk, mask=comp)
    peak_h = {i + 1: float(edt_c[tuple(p)]) for i, p in enumerate(pk)}
    remaining = set(peak_h); out = []; S6 = ndimage.generate_binary_structure(3, 1)
    while remaining and len(out) < want:
        seed = max(remaining, key=lambda i: peak_h[i]); union = ws == seed; remaining.discard(seed)
        for _ in range(len(remaining) + 1):
            dg = genus_solid(base | union) - g_base
            if dg == -1: out.append(union.copy()); break
            if dg < -1 or not remaining: break
            udil = ndimage.binary_dilation(union, S6)
            adj = [i for i in remaining if (ws == i)[udil].any()]
            if not adj: break
            nxt = max(adj, key=lambda i: peak_h[i]); union |= ws == nxt; remaining.discard(nxt)
    return out


def _skeleton_graph(block):
    sk = morphology.skeletonize(block)
    pts = np.argwhere(sk)
    if len(pts) < 3: return None, None
    idx = {tuple(p): i for i, p in enumerate(pts)}
    adj = [[] for _ in pts]
    for i, p in enumerate(pts):
        for d in NB26:
            j = idx.get(tuple(p + d))
            if j is not None: adj[i].append(j)
    return pts, adj


def _bfs(adj, s, blocked=()):
    dist = np.full(len(adj), -1); par = np.full(len(adj), -1); dist[s] = 0; q = [s]; bl = set(blocked)
    for u in q:
        for v in adj[u]:
            if dist[v] < 0 and v not in bl: dist[v] = dist[u] + 1; par[v] = u; q.append(v)
    return dist, par


def _path_to(par, s, t):
    path = [t]
    while path[-1] != s: path.append(int(par[path[-1]]))
    return path[::-1]


def centreline_through(block, seed_pt):
    """Longest skeleton path of `block` that passes through the skeleton voxel nearest `seed_pt`
    (the sealing sheet centre). Returns (path Nx3 float voxel coords, index of the seed on it)."""
    pts, adj = _skeleton_graph(block)
    if pts is None: return None, None
    s = int(np.argmin(np.linalg.norm(pts - seed_pt, axis=1)))
    d1, p1 = _bfs(adj, s); a = int(np.argmax(d1)); pa = _path_to(p1, s, a)          # s -> a
    d2, p2 = _bfs(adj, s, blocked=pa[1:]); b = int(np.argmax(d2)); pb = _path_to(p2, s, b)   # s -> b avoiding pa
    if len(pa) < 3 or len(pb) < 3: return None, None
    path = pa[::-1] + pb[1:]
    return pts[path].astype(float), len(pa) - 1


def _thicken(m, inside):
    return ndimage.binary_dilation(m, S26) & inside


def analyse_plug(loc, sheet, seed, R):
    """loc: local tunnel fill around the throat; sheet: the sealing increment (essential component).
    The sealing increment is the tunnel segment whose radius <= R: a CYLINDER along the tunnel for
    thick walls (long axis = tunnel axis) or a wide DISK for thin membranes (normal = tunnel axis).
    Ambiguous shapes fall back to the skeleton centreline of the local fill through the seed."""
    pts = np.argwhere(sheet).astype(float) if sheet.sum() >= 10 else np.argwhere(loc).astype(float)
    cen = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov(pts.T)) if len(pts) > 3 else (np.ones(3), np.eye(3))
    sd = np.sqrt(np.maximum(ev, 0))                         # ascending
    path, ti = centreline_through(loc, np.asarray(seed, float))
    if sd[2] > 2.0 * sd[1]:   axis = evec[:, 2]; mode = "cylinder"
    elif sd[0] < 0.5 * sd[1]: axis = evec[:, 0]; mode = "disk"
    elif path is not None:    axis = path[min(ti + 3, len(path) - 1)] - path[max(ti - 3, 0)]; mode = "skeleton"
    else:                     axis = evec[:, 0]; mode = "disk-fallback"
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    throat = np.asarray(seed, float)
    if path is None: path = np.stack([throat - 3 * R * axis, throat, throat + 3 * R * axis]); ti = 1
    return {"cen_vox": cen.tolist(), "throat_vox": throat.tolist(), "tangent_vox": axis.tolist(),
            "path_vox": np.asarray(path, float).tolist(), "throat_idx": int(ti), "mode": mode,
            "extent_sd": sd.tolist(), "size": int(loc.sum()), "R": int(R)}


def find_plugs(hs, n_needed, viz_dir=None, log=print, r_max=40):
    """Closing-radius ladder on the cleaned hull.
    Detection uses CUMULATIVE fill regions (close_R minus solid), which are flush with the hull, so
    genus tests are reliable; a region that lowers genus(solid) by exactly 1 is one tunnel's fill.
    Accepted fills are frozen into the solid (Boss's rule) so a sealed tunnel is solid for all later
    radii. The sealing sheet (fill ∩ increment at this radius) seeds the centreline through the fill."""
    t0 = time.time(); solid = hs.copy(); edt_bg = ndimage.distance_transform_edt(~hs).astype(np.float32)
    prev_close = hs.copy(); plugs = []
    if viz_dir: np.save(os.path.join(viz_dir, "hull_ds.npy"), hs)
    for R in range(4, r_max + 1, 2):
        g_solid = genus_solid(solid)
        if len(plugs) >= n_needed or g_solid == 0: break
        close_R = _closing(edt_bg, R)
        D = close_R & ~solid
        incr = D & ~prev_close
        lab, n = ndimage.label(D, structure=S26)
        sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
        claimed_now = False
        for c in np.argsort(-sizes):
            if sizes[c] < 50 or len(plugs) >= n_needed: break
            comp = lab == (c + 1)
            dg = genus_solid(solid | comp) - g_solid
            if dg == 0: continue
            pieces = [comp] if dg == -1 else _split_merged(comp, solid, g_solid, n_needed - len(plugs))
            if dg <= -2: log(f"[p7] hull R={R}: merged dg={dg} -> {len(pieces)} pieces")
            for piece in pieces:
                if len(plugs) >= n_needed: break
                if genus_solid(solid | piece) - g_solid != -1: continue
                sheet = piece & incr
                if sheet.sum() >= 10:
                    # ESSENTIAL sheet component: the one whose removal re-opens the tunnel
                    # (skin increments of the same radius are attached to the fill but do not seal it)
                    lk, nk = ndimage.label(sheet, structure=S26)
                    if nk > 1:
                        g_sealed = genus_solid(solid | piece); chosen = None
                        for kk in np.argsort(-np.bincount(lk.ravel())[1:]):
                            ck = lk == (kk + 1)
                            if ck.sum() < 10: break
                            if genus_solid(solid | (piece & ~ck)) > g_sealed: chosen = ck; break
                        sheet = chosen if chosen is not None else (lk == (np.bincount(lk.ravel())[1:].argmax() + 1))
                    seed = np.argwhere(sheet).mean(0)
                else:
                    edt_p = ndimage.distance_transform_edt(piece); seed = np.array(np.unravel_index(edt_p.argmax(), edt_p.shape), float)
                # local tunnel segment around the throat (skin fillets attached to the fill are excluded):
                # fill voxels within 2R+8 of the sheet centre, connected to the sheet
                rad = 2.0 * R + 8.0; c0 = np.round(seed).astype(int); r0 = int(np.ceil(rad))
                sl = tuple(slice(max(c0[k] - r0, 0), min(c0[k] + r0 + 1, piece.shape[k])) for k in range(3))
                sub = piece[sl].copy(); gi = np.indices(sub.shape).reshape(3, -1).T + np.array([sl[k].start for k in range(3)])
                sub &= (np.linalg.norm(gi - seed, axis=1) <= rad).reshape(sub.shape)
                ls, ns = ndimage.label(sub, structure=S26)
                if ns > 1:
                    sv = np.argwhere(sheet[sl])
                    lab_sheet = ls[tuple(sv.T)] if len(sv) else np.zeros(0, int)
                    lab_sheet = lab_sheet[lab_sheet > 0]
                    sub = ls == (np.bincount(lab_sheet).argmax() if len(lab_sheet) else np.bincount(ls.ravel())[1:].argmax() + 1)
                loc = np.zeros_like(piece); loc[sl] = sub
                info = analyse_plug(loc, sheet, seed, R)
                info["block_vox"] = np.argwhere(loc).astype(np.int16); info["fill"] = int(piece.sum())
                plugs.append(info); solid |= piece; g_solid -= 1; claimed_now = True
                if viz_dir:
                    k = len(plugs)
                    np.save(os.path.join(viz_dir, f"plug{k}_vox.npy"), np.argwhere(sheet if sheet.sum() >= 10 else piece))
                    np.save(os.path.join(viz_dir, f"plug{k}_block.npy"), np.argwhere(loc))
                    np.save(os.path.join(viz_dir, f"plug{k}_path.npy"), np.asarray(info["path_vox"]))
                log(f"[p7] hull R={R:2d} PLUG fill={info['fill']:6d} local={info['size']:6d} sheet={int(sheet.sum()):6d} throat={np.round(info['throat_vox'],1)} "
                    f"mode={info['mode']} tangent={np.round(info['tangent_vox'],2)}")
        if claimed_now:                                   # freeze: sealed tunnels are solid from now on
            edt_bg = ndimage.distance_transform_edt(~solid).astype(np.float32)
        prev_close = close_R | solid
        log(f"[p7] hull R={R}: genus_remain={genus_solid(solid)} plugs={len(plugs)}/{n_needed} ({time.time()-t0:.0f}s)")
    return plugs


def _walk_line(L, scene, F, tri_cen, ti, air=None):
    """Occupancy walk along sampled line L (world). Returns pair tuple or None.
    air(pts)->bool array: hull-air test; a valid membrane crossing lies in hull air (the mesh material
    blocking a tunnel sits where the hull says air), runs through the shape's body are skipped."""
    import open3d as o3d
    occ = scene.compute_occupancy(o3d.core.Tensor(L.astype(np.float32))).numpy() > 0.5
    if not occ.any(): return None, "line never enters the mesh"
    runs = []; i = 0
    while i < len(occ):
        if occ[i]:
            j = i
            while j + 1 < len(occ) and occ[j + 1]: j += 1
            runs.append((i, j)); i = j + 1
        else: i += 1
    if air is not None:
        ok_runs = [r for r in runs if r[0] > 0 and r[1] < len(occ) - 1 and air(L[[r[0] - 1, r[1] + 1]]).all()]
        if not ok_runs:
            if any(r[0] == 0 or r[1] == len(occ) - 1 for r in runs): return None, "line ends inside the mesh"
            return None, "no inside run with both crossings in hull air"
        runs = ok_runs
    a, b = min(runs, key=lambda r: 0 if r[0] <= ti <= r[1] else min(abs(r[0] - ti), abs(r[1] - ti)))
    if a == 0 or b == len(occ) - 1: return None, "line ends inside the mesh"
    pa = 0.5 * (L[a - 1] + L[a]); pb = 0.5 * (L[b] + L[b + 1])
    q = scene.compute_closest_points(o3d.core.Tensor(np.stack([pa, pb]).astype(np.float32)))
    fi, fj = [int(x) for x in q["primitive_ids"].numpy()]
    if fi == fj or (set(F[fi]) & set(F[fj])): return None, f"crossings map to the same/adjacent faces {fi},{fj}"
    return (fi, fj, tri_cen[fi].copy(), tri_cen[fj].copy(), pa, pb), "ok"


def face_pair_for_plug(p, v2w, w2v, V, F, tri_cen, scene, step_w, log=print, hs=None):
    """The plug says WHERE the tunnel is (fill block + centreline through the sealing sheet).
    (a) Walk the centreline (extended straight at both ends if needed) through the coarse mesh:
        the inside run nearest the throat gives the out->in / in->out crossings -> two faces.
    (b) Fallback: mesh material inside the tunnel fill NEAR the throat (<= 2R+4 vox) = the blocking
        membrane; walk a line through its centroid along the local centreline tangent.
    A pair whose separation exceeds 3R+10 voxels is rejected (that is not a membrane crossing)."""
    import open3d as o3d
    path = np.asarray(p["path_vox"], float); ti = int(p["throat_idx"]); R = float(p["R"])
    throat = path[ti]; throat_w = v2w(throat)
    def air(pw):
        if hs is None: return np.ones(len(pw), bool)
        q = np.round(w2v(pw)).astype(int); q = np.clip(q, 0, hs.shape[0] - 1)
        return ~hs[q[:, 0], q[:, 1], q[:, 2]]
    pmax = float(np.max(v2w([1, 1, 1]) - v2w([0, 0, 0])))      # largest voxel pitch (grid is anisotropic)
    cap = (3.0 * R + 10.0) * pmax; near_w = (2.0 * R + 6.0) * pmax
    def _ok(res_pair):
        if res_pair is None: return False
        return float(np.linalg.norm(res_pair[2] - res_pair[3])) <= cap
    why_all = []
    # (0) straight walk through the throat along the tunnel axis (from the sealing increment's shape)
    axis = np.asarray(p["tangent_vox"], float); axis /= np.linalg.norm(axis) + 1e-12
    half = 2.0 * R + 6.0
    for _ in range(4):
        tv = np.linspace(-half, half, int(4 * half) + 1)
        Lv = throat[None, :] + tv[:, None] * axis[None, :]
        res_pair, why = _walk_line(v2w(Lv), scene, F, tri_cen, int(np.argmin(np.abs(tv))), air=air)
        if _ok(res_pair): return res_pair, f"axis walk ({p['mode']})"
        why_all.append(f"axis walk half={half:.0f}: {why if res_pair is None else 'pair too far'}")
        if res_pair is None and "ends inside" not in why: break
        half *= 1.6
    for ext in (0.0, 10.0, 25.0, 50.0):
        t0v = path[min(3, len(path) - 1)] - path[0]; t1v = path[-1] - path[max(len(path) - 4, 0)]
        t0v /= np.linalg.norm(t0v) + 1e-12; t1v /= np.linalg.norm(t1v) + 1e-12
        Q = np.concatenate([[path[0] - ext * t0v], path, [path[-1] + ext * t1v]]) if ext > 0 else path
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1); sacc = np.concatenate([[0], np.cumsum(seg)])
        u = np.arange(0, sacc[-1], 0.5); Lv = np.stack([np.interp(u, sacc, Q[:, k]) for k in range(3)], 1)
        tix = int(np.argmin(np.linalg.norm(Lv - throat, axis=1)))
        res_pair, why = _walk_line(v2w(Lv), scene, F, tri_cen, tix, air=air)
        if _ok(res_pair): return res_pair, f"centreline ext={ext:.0f}"
        why_all.append(f"centreline ext={ext:.0f}: {why if res_pair is None else 'pair too far'}")
        if res_pair is None and "never enters" not in why and "ends inside" not in why: break
    blk = np.asarray(p["block_vox"], float)
    near = blk[np.linalg.norm(v2w(blk) - throat_w, axis=1) <= near_w]
    if len(near):
        occ_b = scene.compute_occupancy(o3d.core.Tensor(v2w(near).astype(np.float32))).numpy() > 0.5
        if occ_b.any():
            m = near[occ_b].mean(0)
            k = int(np.argmin(np.linalg.norm(path - m, axis=1)))
            tang = path[min(k + 3, len(path) - 1)] - path[max(k - 3, 0)]
            if np.linalg.norm(tang) < 1e-6: tang = np.asarray(p["tangent_vox"], float)
            tang = tang / np.linalg.norm(tang)
            p["memb_vox"] = m.tolist(); p["memb_tangent_vox"] = tang.tolist(); p["memb_size"] = int(occ_b.sum())
            half = 2.0 * R + 6.0
            for _ in range(3):
                tv = np.linspace(-half, half, int(4 * half) + 1)
                Lv = m[None, :] + tv[:, None] * tang[None, :]
                res_pair, why = _walk_line(v2w(Lv), scene, F, tri_cen, int(np.argmin(np.abs(tv))), air=air)
                if _ok(res_pair): return res_pair, "membrane"
                if res_pair is None and "ends inside" not in why: break
                half *= 1.6
            why_all.append(f"membrane: {why if res_pair is None else 'pair too far'}")
        else: why_all.append("membrane: no mesh material near the throat")
    # (c) contact tunnel: two mesh sheets pressed together (no volume to walk through). Take the two
    #     nearest faces to the throat with opposed normals on distinct sheets.
    Vf = np.asarray(V); Ff = np.asarray(F)
    cand = np.where(np.linalg.norm(tri_cen - throat_w, axis=1) <= near_w)[0]
    if len(cand) >= 2:
        n = np.cross(Vf[Ff[cand, 1]] - Vf[Ff[cand, 0]], Vf[Ff[cand, 2]] - Vf[Ff[cand, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        best = None
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                fa, fb = int(cand[a]), int(cand[b])
                if set(Ff[fa]) & set(Ff[fb]): continue
                cosn = float(n[a] @ n[b])
                if cosn > -0.5: continue
                d = float(np.linalg.norm(tri_cen[fa] - tri_cen[fb]))
                if d > cap: continue
                score = d + 5.0 * (1.0 + cosn)          # close and opposed
                if best is None or score < best[0]: best = (score, fa, fb)
        if best is not None:
            _, fa, fb = best
            return (fa, fb, tri_cen[fa].copy(), tri_cen[fb].copy(), v2w(throat), v2w(throat)), "contact"
        why_all.append(f"contact: no opposed pair among {len(cand)} faces near the throat")
    else: why_all.append("contact: no faces near the throat")
    return None, "; ".join(why_all)


def find_tunnel_by_hull(V, F, HF_in, prev_handles=(), g_target=None, res=128, cache=None,
                        genus_fn=None, r_dedup=0.3, viz_dir=None, log=print):
    import open3d as o3d
    t0 = time.time()
    lo = np.asarray(HF_in.lo, float); hi = np.asarray(HF_in.hi, float)
    def v2w(v): return lo + np.asarray(v, float) / (res - 1) * (hi - lo)
    def w2v(w): return (np.asarray(w, float) - lo) / (hi - lo) * (res - 1)
    plugs = None
    if cache and os.path.exists(cache):
        try:
            plugs = json.load(open(cache)); bz = np.load(cache + ".blocks.npz")
            for i, p in enumerate(plugs): p["block_vox"] = bz[f"plug{i}"]
            hs = bz["hs"]
            log(f"[p7] hull: loaded {len(plugs)} plug(s) from cache {cache}")
        except Exception: plugs = None
    if plugs is None:
        hs = downsample(clean_hull(np.asarray(HF_in.hull).astype(bool)), res)
        g0 = genus_solid(hs); g_mesh = genus_fn(V, F) if genus_fn else 0
        n_needed = (g_target - g_mesh) if g_target is not None else g0
        log(f"[p7] hull: {res}^3 genus_solid={g0} mesh_genus={g_mesh} need={n_needed}")
        plugs = find_plugs(hs, n_needed, viz_dir=viz_dir, log=log) if (n_needed > 0 and g0 > 0) else []
        for p in plugs:
            p["throat_w"] = v2w(p["throat_vox"]).tolist(); p["cen_w"] = v2w(p["cen_vox"]).tolist()
            p["key"] = ["hull", [round(float(x), 3) for x in p["throat_w"]]]
        if cache:
            try:
                np.savez_compressed(cache + ".blocks.npz", hs=hs, **{f"plug{i}": p["block_vox"] for i, p in enumerate(plugs)})
                json.dump([{k: v for k, v in p.items() if k != "block_vox"} for p in plugs], open(cache, "w")); log(f"[p7] hull: cached to {cache}")
            except Exception as e: log(f"[p7] hull: cache write failed ({e})")
        log(f"[p7] hull plugs: {len(plugs)} found (R ladder) target={n_needed} in {time.time()-t0:.0f}s")
    # face pairs
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.int32))))
    tri_cen = np.asarray(V)[np.asarray(F)].mean(1)
    step_w = 0.5 * float(((hi - lo) / (res - 1)).min())
    out = []; n_dedup = n_fail = 0
    for p in plugs:
        thr = np.asarray(p["throat_w"])
        dup = False
        for h in prev_handles:
            if not isinstance(h, dict): continue
            if h.get("blob") == p["key"] or ("mid" in h and np.linalg.norm(np.asarray(h["mid"]) - thr) < r_dedup): dup = True; break
        if dup: n_dedup += 1; log(f"[p7] hull plug R={p['R']} at {np.round(thr,3)}: dedup skip"); continue
        res_pair, why = face_pair_for_plug(p, v2w, w2v, V, F, tri_cen, scene, step_w, log=log, hs=hs)
        if res_pair is None:
            n_fail += 1; log(f"[p7] hull plug R={p['R']} at {np.round(thr,3)}: no face pair ({why})"); continue
        fi, fj, ci, cj, pa, pb = res_pair
        p["cross_w"] = [pa.tolist(), pb.tolist()]
        if viz_dir: json.dump([{k: v for k, v in q.items() if k != "block_vox"} for q in plugs], open(os.path.join(viz_dir, "plugs.json"), "w"))
        log(f"[p7] hull plug R={p['R']} ({p['mode']}/{why}): accepted faces {fi},{fj} sep={np.linalg.norm(cj-ci):.3f}")
        out.append((fi, fj, ci, cj, p["key"]))
    log(f"[p7] hull plugs: {len(plugs)} found, {len(out)} accepted (skip: {n_dedup} dedup, {n_fail} !face)")
    return out
