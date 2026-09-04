"""Phase 7: genus change from space-carving evidence + DLFL add_handle.

A hole the sphere-init chain cannot make shows up as a dimple: two patches of
our surface pushed together inside the GT tunnel. Both patches lie OUTSIDE the
voting visual hull (the hull has a tunnel there), their outward normals face
each other, and the segment between them stays outside the hull. That is
proof of a tunnel. The two faces are back-to-back (normals away from each
other, our slab between them). DLFL add_handle(face_i, face_j): both faces removed, a
tube of side quads inserted through the slab (genus +1, manifold preserved), quads stellated.
Then the normal loop (phase4) pulls the tube walls to the hole wall.

Run: MODE=64v SHAPE=rockerarm BASE_NPZ=... TAG=... [MAX_HANDLES=1] python3 despike/phase7_handle.py
"""
import sys, os, tempfile
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
os.environ.setdefault("MODE", "64v")
import numpy as np, torch
import nvdiffrast.torch as dr
import cow_v13
from cow_v13 import DEVICE
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from hull_field import build_vote_hull
from topmod.io import from_obj, to_triangle_arrays
from topmod.high_level_ops import add_handle, stellate as dlfl_stellate
import run_64v

SHAPE = os.environ.get("SHAPE", "rockerarm")
TAG = os.environ.get("TAG", f"{SHAPE}_p7")
BASE_NPZ = os.environ["BASE_NPZ"]
MAX_HANDLES = int(os.environ.get("MAX_HANDLES", "1"))
TUBE_SUBDIV = int(os.environ.get("TUBE_SUBDIV", "0"))
PROJECT = int(os.environ.get("PROJECT", "0"))
EXT_VOX = float(os.environ.get("EXT_VOX", "8.0"))
MIN_SEP = float(os.environ.get("MIN_SEP", "1.5"))
PROJ_RADIUS = float(os.environ.get("PROJ_RADIUS", "0.6"))  # also radially project the membrane around the tube (within this radius of the axis) so the mouth eats it; 0 = tube only    # faces closer than this (mean-edge units) are fold/sliver remnants, not a slab   # tunnel must continue hull-free this far beyond BOTH faces (bay vs through-hole)
DRY = int(os.environ.get("DRY", "0"))
DETECT = os.environ.get("DETECT", "rays")
ABSORB = int(os.environ.get("ABSORB", "0"))
OPEN = os.environ.get("OPEN", "merge")   # merge = collapse membrane interior verts to the rim, delete interior edges -> rim polygon, add_handle(rim1, rim2)   # after add_handle, eat the blocking membrane into the mouth by DLFL collapses (mouth ring grows to the membrane rim)
HANDLES_JSON = os.environ.get("HANDLES_JSON", "")     # persisted list of handle midpoints across rounds (one handle per tunnel)
R_DEDUP = float(os.environ.get("R_DEDUP", "0.3"))      # a new candidate closer than this to an existing handle is the SAME tunnel -> skip   # rays = see-through pixels of the training silhouettes (default) | hull = back-to-back faces outside the hull             # 1 = detect and report only       # after refinement, project every vertex that is outside the hull onto the hull boundary (deterministic inflate)  # DLFL-subdivide the new tube faces N times so the hull field can inflate a long tube
OUT_VOX = float(os.environ.get("OUT_VOX", "4.0"))     # both faces must be > this many voxels outside the hull
FACE_COS = float(os.environ.get("FACE_COS", "-0.5"))  # n_i . n_j below this (facing each other)
MAX_SEP = float(os.environ.get("MAX_SEP", "100.0"))   # max centroid separation (mean-edge units); thick slabs need long tubes (3holes: 12 edges)
OUTD = "/tmp/liou_cow_viz"

z = np.load(BASE_NPZ); V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ctx = dr.RasterizeCudaContext()
gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj")); gvn = normalize_to_range(gv)
mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE); cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=512, vote=2)
pitch = HF.pitch

def genus(V, F):
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

def hdist(P):
    return HF.dist(torch.tensor(np.asarray(P, np.float32), device=DEVICE)).detach().cpu().numpy()

def report(tag, V, F):
    ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F)
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} genus={genus(V, F)} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]}", flush=True)

def find_tunnel_pairs(V, F):
    tri = V[F]; cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    d = hdist(cen)
    cand = np.where(d > OUT_VOX * pitch)[0]
    print(f"[p7] faces outside hull by >{OUT_VOX} voxels: {len(cand)} / {len(F)} (pitch {pitch:.4f}, mean edge {me:.4f})", flush=True)
    import open3d as o3d
    _scene = o3d.t.geometry.RaycastingScene()
    _scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    pairs = []
    for a in range(len(cand)):
        i = cand[a]
        for j in cand[a + 1:]:
            if len(set(F[i]) & set(F[j])): continue                   # must not share vertices
            v = cen[j] - cen[i]; L = np.linalg.norm(v)
            if L > MAX_SEP * me or L < MIN_SEP * me: continue
            if n[i] @ n[j] > FACE_COS: continue                          # facing each other
            # membrane = thin slab of OUR volume inside the GT tunnel: the two faces are
            # back-to-back, outward normals point AWAY from each other
            if n[i] @ v >= 0 or n[j] @ (-v) >= 0: continue
            seg = cen[i] + np.linspace(0.05, 0.95, max(9, int(L / (0.5 * pitch))))[:, None] * v   # sample every half voxel
            if (hdist(seg) < 0.5 * pitch).any(): continue               # whole segment outside hull
            # the segment must not cross OUR mesh either: if it does, it runs through an
            # existing tube/hole (fertility: 5th handle in an already-opened hole -> genus 5 vs GT 4)
            if _scene is not None:
                r = o3d.core.Tensor(np.concatenate([(cen[i] + 0.02 * L * v / L)[None], (v / L)[None]], 1).astype(np.float32))
                hit = _scene.cast_rays(r)["t_hit"].numpy()[0]
                if np.isfinite(hit) and hit < L * 0.98: continue
            # bay vs tunnel: a real through-hole keeps going (hull-free) beyond both faces;
            # an unfilled bay has hull material right behind its bottom face (fertility: 7 handles vs GT 4)
            ext = np.linspace(0.0, EXT_VOX * pitch, 9)[1:]
            back = cen[i] - ext[:, None] * (v / L); fwd = cen[j] + ext[:, None] * (v / L)
            if (hdist(back) < 0.5 * pitch).any() or (hdist(fwd) < 0.5 * pitch).any(): continue
            pairs.append((min(d[i], d[j]) / pitch, -L / me, int(i), int(j)))
    pairs.sort(reverse=True)
    return pairs, cen, d


def project_to_hull(V, iters=3):
    """Hard space-carving step: a vertex outside the voting hull is provably misplaced;
    move it along -grad(dist) onto the hull boundary (trilinear field, few iterations)."""
    P = torch.tensor(V, dtype=torch.float32, device=DEVICE); moved = 0
    for it in range(iters):
        P = P.detach().requires_grad_(True)
        dd = HF.dist(P); m = dd > 0.5 * pitch
        if not m.any(): break
        g, = torch.autograd.grad(dd.sum(), P); g = g / (g.norm(dim=1, keepdim=True) + 1e-9)
        P = P.detach(); P[m] = P[m] - dd[m, None] * g[m]; moved = max(moved, int(m.sum()))
    return P.detach().cpu().numpy().astype(np.float64), moved


def radial_project(V, a0, u, L, tube_verts):
    """Vertices in the tunnel near its medial axis have an ill-defined EDT gradient (it flips
    from vertex to vertex -> faces cross the tunnel and cap it). Instead march each outside
    vertex RADIALLY away from the tunnel axis (a0 + t*u) until it reaches the hull boundary."""
    V = V.copy(); rel = V - a0; t = rel @ u
    radial = rel - t[:, None] * u; r = np.linalg.norm(radial, axis=1)
    d = hdist(V)
    is_tube = np.isin(np.arange(len(V)), list(tube_verts))
    # membrane remnants left around the mouth re-trigger the tunnel test (fertility: 7 handles vs 4);
    # eat them into the mouth by projecting everything outside the hull near the axis radially too
    near = (d > 0.5 * pitch) & (is_tube | ((r < PROJ_RADIUS) & (t > -0.2 * L) & (t < 1.2 * L)))
    idx = np.where(near)[0]
    if len(idx) == 0: return V, 0
    perp1 = np.cross(u, [1.0, 0, 0]); perp1 = perp1 if np.linalg.norm(perp1) > 0.1 else np.cross(u, [0, 1.0, 0]); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(u, perp1)
    dirs = radial[idx] / (r[idx, None] + 1e-12)
    deg = r[idx] < 0.5 * pitch
    ang = 2 * np.pi * (idx[deg] % 7) / 7.0
    dirs[deg] = np.cos(ang)[:, None] * perp1 + np.sin(ang)[:, None] * perp2
    steps = np.arange(0.0, 0.5, 0.5 * pitch)
    P = V[idx][:, None, :] + steps[None, :, None] * dirs[:, None, :]
    dd = hdist(P.reshape(-1, 3)).reshape(len(idx), len(steps))
    inside = dd <= 0.5 * pitch
    first = np.where(inside.any(1), inside.argmax(1), 0)   # no hull boundary found along the ray (e.g. through the mouth) -> leave the vertex
    V[idx] = V[idx] + steps[first][:, None] * dirs
    return V, len(idx)



def membrane_patches(V, F):
    """Connected components (edge adjacency) of faces lying outside the voting hull, with the
    Euler characteristic of each patch. A membrane that blocks a tunnel is a topological DISK
    (chi = 1). Once a handle pierces it, the patch becomes an annulus (chi = 0): tunnel already open."""
    from collections import defaultdict
    tri = V[F]; cen = tri.mean(1); d = hdist(cen)
    out = np.where(d > OUT_VOX * pitch)[0]
    if len(out) == 0: return {}, {}
    em = defaultdict(list)
    for fi in out:
        a, b, c = F[fi]
        for e in ((a, b), (b, c), (c, a)): em[(min(e), max(e))].append(fi)
    parent = {int(f): int(f) for f in out}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for fs in em.values():
        for f2 in fs[1:]: parent[find(int(fs[0]))] = find(int(f2))
    comp = {int(f): find(int(f)) for f in out}
    chi = {}
    for root in set(comp.values()):
        fs = [f for f, r in comp.items() if r == root]
        verts = set(); edges = set()
        for f in fs:
            a, b, c = map(int, F[f]); verts |= {a, b, c}
            for e in ((a, b), (b, c), (c, a)): edges.add((min(e), max(e)))
        chi[root] = len(verts) - len(edges) + len(fs)
    return comp, chi


def absorb_membrane(mesh, ring_ids, membrane_ids, max_iter=100000):
    """Cut the membrane from the mouth outward: repeatedly collapse an edge (ring vertex, membrane
    vertex) with DLFL collapse_edge_tri, survivor placed at the membrane vertex -> the mouth ring
    advances one vertex per collapse until it reaches the membrane rim (which lies on the hull =
    tunnel wall). Euler characteristic preserved (genus fixed by the handle), manifold preserved."""
    from topmod.high_level_ops import collapse_edge_tri
    vlist = list(mesh.vertices.values())
    ring = set(id(vlist[i]) for i in ring_ids)
    memb = set(id(vlist[i]) for i in membrane_ids) - ring
    n = 0; stuck = set(); progress = True
    while progress and n < max_iter:
        progress = False
        for v in list(mesh.vertices.values()):
            if id(v) not in ring or v.id not in mesh.vertices: continue
            for h in list(v.outgoing_halfedges()):
                u = h.twin.origin if h.twin else None
                if u is None or id(u) not in memb or h.edge is None or h.edge.id in stuck: continue
                ux, uy, uz = u.x, u.y, u.z
                sv = collapse_edge_tri(mesh, h.edge)
                if sv is None: stuck.add(h.edge.id); continue
                sv.x, sv.y, sv.z = ux, uy, uz
                memb.discard(id(u)); memb.discard(id(sv)); ring.add(id(sv))
                n += 1; progress = True
                break
    return n, len(memb)


def _is_boundary(v, faces):
    for h in v.outgoing_halfedges():
        if h.face is None or id(h.face) not in faces: return True
    return False

def membrane_to_polygon(mesh, flist, patch_face_idx):
    """Turn a membrane patch (disk of triangles) into ONE polygon face whose boundary is the
    membrane rim: (1) DLFL-collapse every interior vertex into a rim neighbour (survivor placed
    at the rim vertex), (2) delete_edge on every edge shared by two patch faces. Manifold and
    Euler characteristic preserved throughout."""
    from topmod.high_level_ops import collapse_edge_tri
    from topmod.operators import delete_edge
    faces = {id(flist[f]): flist[f] for f in patch_face_idx}
    for _ in range(50):
        interior = [v for v in {id(v): v for f in faces.values() for v in f.vertices()}.values()
                    if v.id in mesh.vertices and not _is_boundary(v, faces)]
        if not interior: break
        moved = False
        for v in interior:
            if v.id not in mesh.vertices: continue
            for pref in (True, False):
                ok = False
                for h in list(v.outgoing_halfedges()):
                    u = h.twin.origin if h.twin else None
                    if u is None or h.edge is None: continue
                    if pref and not _is_boundary(u, faces): continue
                    ux, uy, uz = u.x, u.y, u.z
                    fa, fb = h.face, h.twin.face
                    sv = collapse_edge_tri(mesh, h.edge)
                    if sv is None: continue
                    sv.x, sv.y, sv.z = ux, uy, uz; ok = True; moved = True
                    for f in (fa, fb):
                        if f is not None and f.id not in mesh.faces: faces.pop(id(f), None)
                    break
                if ok: break
        if not moved: break
    faces = {k: f for k, f in faces.items() if f.id in mesh.faces}
    progress = True
    while progress and len(faces) > 1:
        progress = False
        for e in list(mesh.edges.values()):
            if e.id not in mesh.edges: continue
            fa, fb = e.he0.face, e.he1.face
            if fa is None or fb is None or fa is fb: continue
            if id(fa) in faces and id(fb) in faces:
                nf = delete_edge(mesh, e)
                faces.pop(id(fa), None); faces.pop(id(fb), None); faces[id(nf)] = nf
                progress = True
    assert len(faces) == 1, f"membrane did not merge into one polygon ({len(faces)} left)"
    return next(iter(faces.values()))

def open_tunnel_merge(mesh, flist, patch_i, patch_j):
    """membrane_i -> rim polygon, membrane_j -> rim polygon, equalize vertex counts by DLFL
    subdivide_edge on the smaller rim, align the start vertices, add_handle(rim_i, rim_j)."""
    from topmod.high_level_ops import subdivide_edge as dlfl_subdivide_edge
    f1 = membrane_to_polygon(mesh, flist, patch_i)
    f2 = membrane_to_polygon(mesh, flist, patch_j)
    def nverts(f): return len(list(f.halfedges()))
    while nverts(f1) != nverts(f2):
        small = f1 if nverts(f1) < nverts(f2) else f2
        hes = list(small.halfedges())
        longest = max(hes, key=lambda h: (h.origin.x - h.twin.origin.x)**2 + (h.origin.y - h.twin.origin.y)**2 + (h.origin.z - h.twin.origin.z)**2)
        dlfl_subdivide_edge(mesh, longest.edge)
    n = nverts(f1)
    # align: pair verts1[0] with the rim-2 vertex nearest to it (add_handle pairs verts1[t] with reversed verts2[t])
    hes1 = list(f1.halfedges()); v0 = hes1[0].origin
    hes2 = list(f2.halfedges())
    k = min(range(n), key=lambda t: (hes2[t].origin.x - v0.x)**2 + (hes2[t].origin.y - v0.y)**2 + (hes2[t].origin.z - v0.z)**2)
    f2.he = hes2[(k + 1) % n]          # reversed list index n-1 -> hes2[k].origin pairs with v0
    add_handle(mesh, f1, f2)
    return n

def find_tunnel_by_rays(V, F, min_px=30, prev_handles=()):
    """Image-domain space-carving evidence. In a TRAINING view, a background pixel enclosed by
    foreground (a 2D hole in the GT silhouette) proves free space along its whole ray. If our
    mesh is hit by that ray, the entry and exit faces are the two sides of the membrane that
    blocks the tunnel -> add_handle(entry, exit). Holes are ranked by pixel area; the ray is
    taken at the hole's centroid (plus a few fallbacks inside the blob)."""
    import open3d as o3d
    from scipy import ndimage
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    gtn = np.asarray(gt); H, W = gtn.shape[1:]
    cands = []
    for k in range(len(gtn)):
        fg = gtn[k] < 128
        lab, n = ndimage.label(~fg)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        for c in range(1, n + 1):
            if c in border: continue
            m = lab == c; area = int(m.sum())
            if area < min_px: continue
            rr, cc = np.where(m); order = np.argsort((rr - rr.mean())**2 + (cc - cc.mean())**2)
            cands.append((area, k, rr[order[:5]], cc[order[:5]]))
    cands.sort(key=lambda x: -x[0])
    comp, chi = membrane_patches(V, F)
    print(f"[p7] outside-hull membrane patches: {len(chi)} (disks: {sum(1 for c in chi.values() if c == 1)})", flush=True)
    print(f"[p7] see-through blobs >= {min_px}px across training views: {len(cands)}", flush=True)
    inv = [np.linalg.inv(np.asarray(torch.as_tensor(m).cpu().numpy(), np.float64)) for m in mvps]
    for area, k, rr, cc in cands:
        key = [int(k), int(rr[0]), int(cc[0])]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)):
            continue   # this see-through blob already got its handle (cascade guard: flaps / narrow tubes)
        for r_, c_ in zip(rr, cc):
            x = (c_ + 0.5) / W * 2 - 1; y = (r_ + 0.5) / H * 2 - 1
            p0 = inv[k] @ np.array([x, y, -1.0, 1.0]); p1 = inv[k] @ np.array([x, y, 1.0, 1.0])
            p0 = p0[:3] / p0[3]; p1 = p1[:3] / p1[3]; dvec = p1 - p0; dvec /= np.linalg.norm(dvec)
            ray = o3d.core.Tensor(np.concatenate([p0, dvec])[None].astype(np.float32))
            h1 = sc.cast_rays(ray); t1 = float(h1["t_hit"].numpy()[0])
            if not np.isfinite(t1): continue
            fi = int(h1["primitive_ids"].numpy()[0])
            ray2 = o3d.core.Tensor(np.concatenate([p1, -dvec])[None].astype(np.float32))
            h2 = sc.cast_rays(ray2); t2 = float(h2["t_hit"].numpy()[0])
            if not np.isfinite(t2): continue
            fj = int(h2["primitive_ids"].numpy()[0])
            if fi == fj or (set(F[fi]) & set(F[fj])): continue
            ci = V[F[fi]].mean(0); cj = V[F[fj]].mean(0)
            # topological one-handle-per-tunnel rule: both membrane patches must still be disks
            ki, kj = comp.get(fi), comp.get(fj)
            if ki is None or kj is None or chi[ki] != 1 or chi[kj] != 1:
                print(f"[p7]   skip view {k} hole {area}px: membrane patches chi={None if ki is None else chi[ki]},{None if kj is None else chi[kj]} (not disks -> tunnel already pierced / not a membrane)", flush=True)
                continue
            print(f"[p7] ray evidence: view {k}, hole {area}px, entry face {fi} exit face {fj}, sep {np.linalg.norm(cj-ci):.3f}", flush=True)
            return fi, fj, ci, cj, key
    return None

report("base", V, Fa)
import json
prev_handles = json.load(open(HANDLES_JSON)) if HANDLES_JSON and os.path.exists(HANDLES_JSON) else []
n_added = 0
for k in range(MAX_HANDLES):
    if DETECT == "rays":
        hit = find_tunnel_by_rays(V, Fa, prev_handles=prev_handles)
        if hit is None: print("[p7] tunnel-evidence pairs: 0", flush=True); break
        i, j, _ci, _cj, _blob = hit
        tri = V[Fa]; cen = tri.mean(1); d = hdist(cen); negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
        print(f"[p7] tunnel-evidence pairs: 1 (ray)", flush=True)
    else:
        pairs, cen, d = find_tunnel_pairs(V, Fa)
        print(f"[p7] tunnel-evidence pairs: {len(pairs)}", flush=True)
        if not pairs: break
        score, negL, i, j = pairs[0]
    if DRY:
        print(f"[p7] DRY: best pair {i},{j} out {d[i]/pitch:.1f}/{d[j]/pitch:.1f} vox, sep {-negL:.2f} edges, centroids {np.round(cen[i],2)} {np.round(cen[j],2)}", flush=True); break
    print(f"[p7] add_handle between faces {i},{j}: out {d[i]/pitch:.1f}/{d[j]/pitch:.1f} vox, sep {-negL:.2f} edges, centroids {np.round(cen[i],3)} {np.round(cen[j],3)}", flush=True)
    def _load_mesh():
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
            for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
            for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
            path = fh.name
        m = from_obj(path); os.unlink(path)
        return m, list(m.iter_faces())
    mesh, faces = _load_mesh()
    # membrane vertex sets (front patch of face i, back patch of face j) BEFORE the handle
    if ABSORB:
        comp_, chi_ = membrane_patches(V, Fa)
        memb_faces = [f for f, r in comp_.items() if r in (comp_.get(i), comp_.get(j))]
        memb_ids = set(int(x) for f in memb_faces for x in Fa[f])
        ring_ids = set(int(x) for x in Fa[i]) | set(int(x) for x in Fa[j])
    if OPEN == "merge":
        comp_, chi_ = membrane_patches(V, Fa)
        pi = [f for f, r in comp_.items() if r == comp_.get(i)]; pj = [f for f, r in comp_.items() if r == comp_.get(j)]
        if comp_.get(i) is None: pi = [i]
        if comp_.get(j) is None: pj = [j]
        if comp_.get(i) is not None and comp_.get(i) == comp_.get(j):
            print("[p7] entry and exit faces lie in the SAME membrane patch (thin sheet): using single faces", flush=True); pi, pj = [i], [j]
        try:
            nrim = open_tunnel_merge(mesh, faces, pi, pj)
            for f in list(mesh.faces.values()):
                if len(f.vertices()) > 3: dlfl_stellate(mesh, f)
            _vv, _ff = to_triangle_arrays(mesh)
            _wt, _nb = check_watertight(np.asarray(_ff, np.int64))
            if not _wt: raise RuntimeError(f"merge left {_nb} bad edges")
            print(f"[p7] membranes merged to rim polygons ({len(pi)}+{len(pj)} faces) -> handle with {nrim}-gon rims", flush=True)
        except Exception as ex:
            # thin sheets: front/back rims can share vertices -> non-manifold. Fall back to the plain
            # single-face handle on a fresh mesh (never leave a broken mesh behind).
            print(f"[p7] merge failed ({ex}); falling back to single-face add_handle", flush=True)
            mesh, faces = _load_mesh()
            add_handle(mesh, faces[i], faces[j])
    else:
        add_handle(mesh, faces[i], faces[j])
    if ABSORB:
        n_abs, left = absorb_membrane(mesh, ring_ids, memb_ids)
        print(f"[p7] membrane absorbed into the mouth: {n_abs} DLFL collapses ({len(memb_ids)} membrane verts, {left} left)", flush=True)
    for f in list(mesh.faces.values()):
        if len(f.vertices()) > 3: dlfl_stellate(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2, F2 = np.asarray(vv, float), np.asarray(ff, np.int64)
    wt, nbad = check_watertight(F2)
    if not wt:
        print(f"[p7] handle broke watertightness ({nbad} bad edges): blacklisting this blob and continuing", flush=True)
        prev_handles.append({"mid": None, "blob": _blob if DETECT == "rays" else None})
        if HANDLES_JSON: json.dump(prev_handles, open(HANDLES_JSON, "w"))
        continue
    n_before = len(V) if not (ABSORB or OPEN == 'merge') else -1
    if ABSORB or OPEN == 'merge':
        # after collapses vertex order changed: tube verts = those outside the hull among the new mesh
        d2 = hdist(V2); tube_verts = set(np.where(d2 > 0.5 * pitch)[0].tolist())
    else:
        tube_verts = set(map(int, Fa[i])) | set(map(int, Fa[j])) | set(range(n_before, len(V2)))
    V, Fa = V2, F2
    a0 = cen[i]; u = cen[j] - cen[i]; L_ = float(np.linalg.norm(u)); u = u / L_   # tunnel axis from the two face centroids
    if PROJECT:
        # project FIRST (thin tube -> tunnel wall), then refine on the wall, re-project each level.
        # Refining the thin tube before projecting produced sliver fans that crossed when inflated.
        V, mv = radial_project(V, a0, u, L_, tube_verts); print(f"[p7] radial projection (pre-refine): moved {mv} verts", flush=True)
    for _ in range(TUBE_SUBDIV):
        # a 3-edge-wide tube spanning a thick slab has no interior vertices for the hull
        # field to act on (3holes: 10 edges long, hole never opened). Refine the tube.
        from phase1c_pipeline import dlfl_subdivide_arrays
        fids = [k for k, f in enumerate(Fa) if all(int(x) in tube_verts for x in f)]
        nb = len(V)
        V, Fa, ne = dlfl_subdivide_arrays(V, Fa, fids, expand_ring=False)
        tube_verts |= set(range(nb, len(V)))
        wt, nbad = check_watertight(Fa); assert wt, nbad
        print(f"[p7] tube refine: {len(fids)} faces, split {ne} edges -> V={len(V)} F={len(Fa)}", flush=True)
        if PROJECT:
            V, mv = radial_project(V, a0, u, L_, tube_verts); print(f"[p7] radial projection: moved {mv} verts", flush=True)
    n_added += 1
    prev_handles.append({"mid": ((cen[i] + cen[j]) / 2).tolist(), "blob": _blob if DETECT == "rays" else None})
    if HANDLES_JSON: json.dump(prev_handles, open(HANDLES_JSON, "w"))
    report(f"after handle {n_added}", V, Fa)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[p7] handles added: {n_added}; saved cow_{SHAPE}_{TAG}.npz", flush=True)
