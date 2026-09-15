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
PRE_SUBDIV = int(os.environ.get("PRE_SUBDIV", "0"))   # 1 = always pre-subdivide entry/exit faces before add_handle (auto when their 1-rings overlap)
DETECT = os.environ.get("DETECT", "hull")
ABSORB = int(os.environ.get("ABSORB", "0"))
OPEN = os.environ.get("OPEN", "merge")   # merge = collapse membrane interior verts to the rim, delete interior edges -> rim polygon, add_handle(rim1, rim2)   # after add_handle, eat the blocking membrane into the mouth by DLFL collapses (mouth ring grows to the membrane rim)
HANDLES_JSON = os.environ.get("HANDLES_JSON", "")     # persisted list of handle midpoints across rounds (one handle per tunnel)
R_DEDUP = float(os.environ.get("R_DEDUP", "0.3"))      # a new candidate closer than this to an existing handle is the SAME tunnel -> skip   # rays = see-through pixels of the training silhouettes (default) | hull = back-to-back faces outside the hull             # 1 = detect and report only       # after refinement, project every vertex that is outside the hull onto the hull boundary (deterministic inflate)  # DLFL-subdivide the new tube faces N times so the hull field can inflate a long tube
OUT_VOX = float(os.environ.get("OUT_VOX", "4.0"))     # both faces must be > this many voxels outside the hull
FACE_COS = float(os.environ.get("FACE_COS", "-0.5"))  # n_i . n_j below this (facing each other)
MAX_SEP = float(os.environ.get("MAX_SEP", "100.0"))   # max centroid separation (mean-edge units); thick slabs need long tubes (3holes: 12 edges)
OUTD = "/tmp/liou_cow_viz"
HULL_LOC_RES = int(os.environ.get("HULL_LOC_RES", "128"))
HULL_FACE_DIST_VOX = float(os.environ.get("HULL_FACE_DIST_VOX", "6"))
HULL_PLUGS_CACHE = os.environ.get("HULL_PLUGS_CACHE", f"{OUTD}/hull_plugs_{SHAPE}_{HULL_LOC_RES}.json")

z = np.load(BASE_NPZ); V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ctx = dr.RasterizeCudaContext()
gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj")); gvn = normalize_to_range(gv)
mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE); cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=int(os.environ.get("HULL_HIRES", "512")), vote=2)
pitch = HF.pitch
from hull_field import hull_genus
_GT_ENV = os.environ.get("GENUS_TARGET", "hull")   # hull = persistent genus of the space-carved hull (LESSONS 24) | off | integer
if _GT_ENV == "off": G_TARGET = None
elif _GT_ENV == "hull": G_TARGET, _gr = hull_genus(HF.hull); print(f"[p7] genus target from space-carved hull: g*={G_TARGET} (per radius {_gr})", flush=True)
else: G_TARGET = int(_GT_ENV)
RELAX = [(float(os.environ.get("MIN_PX", "30")), OUT_VOX), (15.0, 2.0), (8.0, 1.0)]   # (min_px, out_vox) ladder while genus < g*

def genus(V, F):
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

def hdist(P):
    return HF.dist(torch.tensor(np.asarray(P, np.float32), device=DEVICE)).detach().cpu().numpy()

def report(tag, V, F):
    ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F)
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} genus={genus(V, F)} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]}", flush=True)

def _snap(title, V, F, hold=30):
    if os.environ.get("SNAPSHOT_DIR"):
        import viz_snap; viz_snap.snap(ctx, mvps, V, F, title, hold=hold)

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



def membrane_patches(V, F, out_vox_override=None):
    """Connected components (edge adjacency) of faces lying outside the voting hull, with the
    Euler characteristic of each patch. A membrane that blocks a tunnel is a topological DISK
    (chi = 1). Once a handle pierces it, the patch becomes an annulus (chi = 0): tunnel already open."""
    from collections import defaultdict
    ov = out_vox_override if out_vox_override is not None else OUT_VOX
    tri = V[F]; cen = tri.mean(1); d = hdist(cen)
    out = np.where(d > ov * pitch)[0]
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

def find_tunnel_by_rays(V, F, min_px=int(os.environ.get("MIN_PX", "30")), prev_handles=()):
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

def find_bridge(V, F, min_vox=int(os.environ.get("BRIDGE_MIN_VOX", "60")), max_try=6):
    """Dual of find_tunnel_by_rays (LESSONS 25): hull-SOLID voxels that the mesh leaves EMPTY, slab-shaped
    (a wall between two tunnels the mesh merged into one opening). add_handle between the two mesh faces at the
    ends of the slab's long axis inserts a bar through the opening; the DR loop inflates it into the wall.
    Returns (fi, fj, ci, cj, key) or None."""
    import open3d as o3d
    from scipy import ndimage as ndi
    S3 = ndi.generate_binary_structure(3, 1)
    Hc = np.asarray(HF.hull).astype(bool)
    Hc = ndi.binary_opening(ndi.binary_closing(Hc, S3, iterations=2), S3, iterations=2)
    lab, n = ndi.label(Hc); Hc = ndi.binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
    N = Hc.shape[0]; lo, hi = np.asarray(HF.lo), np.asarray(HF.hi)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    ax = [np.linspace(lo[i], hi[i], N, dtype=np.float32) for i in range(3)]; M = np.zeros((N, N, N), bool)
    yy, zz = np.meshgrid(ax[1], ax[2], indexing="ij")
    for xi in range(N):
        P = np.stack([np.full_like(yy, ax[0][xi]), yy, zz], -1).reshape(-1, 3)
        M[xi] = (sc.compute_occupancy(o3d.core.Tensor(P)).numpy() > 0.5).reshape(N, N)
    gap = Hc & ~M                                                   # hull material the mesh does not cover
    gl, gn = ndi.label(gap); sz = np.bincount(gl.ravel())[1:]; order = [int(c) for c in np.argsort(-sz) if sz[c] >= min_vox][:40]
    world = lambda v: lo + np.asarray(v, float) / (N - 1) * (hi - lo)
    def free_both_sides(cen_v, nrm, t):
        """A WALL between two tunnels has hull-FREE space on both broad sides within a few voxels; hull slack
        (a concavity the hull cannot carve) has free space on one side only."""
        ok = []
        for sgn in (+1.0, -1.0):
            hit = False
            for d in np.arange(t / 2 + 1, t / 2 + 12, 1.0):
                q = np.round(cen_v + sgn * d * nrm).astype(int)
                if (q < 0).any() or (q >= N).any(): hit = True; break
                if M[q[0], q[1], q[2]]: break                       # ran into mesh material first -> not free on this side
                if not Hc[q[0], q[1], q[2]]: hit = True; break     # outside the hull -> free space
            ok.append(hit)
        return all(ok)
    walls = []
    for c in order:
        pts = np.argwhere(gl == c + 1).astype(float); cen_v = pts.mean(0); u, s, vt = np.linalg.svd(pts - cen_v, full_matrices=False)
        ext = 2 * s / np.sqrt(len(pts))
        if ext[2] > 12 or ext[0] < 6: continue                     # a wall is thin (<12 vox) and has some extent
        if free_both_sides(cen_v, vt[2], ext[2]): walls.append((c, pts, cen_v, vt, ext))
    print(f"[p7] bridge evidence: hull-solid/mesh-empty components >= {min_vox} vox: {len(order)}; thin + free on both sides (walls): {len(walls)} (sizes {[int(sz[w[0]]) for w in walls]})", flush=True)
    for c, pts, cen_v, vt, ext in walls[:max_try]:
        c0 = world(cen_v); a = vt[0] * np.sign(vt[0] @ (hi - lo))   # long axis of the slab, world frame (grid is axis-aligned)
        a = a / (np.linalg.norm(a) + 1e-12); L = float(ext[0]) * float(HF.pitch)
        key = ["bridge", int(c + 1), [round(float(x), 3) for x in c0]]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)): continue
        hits = []
        for sgn in (+1.0, -1.0):
            ray = o3d.core.Tensor(np.concatenate([c0, sgn * a])[None].astype(np.float32)); h = sc.cast_rays(ray)
            t = float(h["t_hit"].numpy()[0]); hits.append((t, int(h["primitive_ids"].numpy()[0])) if np.isfinite(t) else None)
        if any(h is None for h in hits): print(f"[p7]   slab {c+1} ({int(sz[c])} vox, extent {np.round(ext,1)} vox): long-axis ray misses the mesh on one side -> skip", flush=True); continue
        (ti, fi), (tj, fj) = hits
        if ti + tj > 3.0 * L + 6 * HF.pitch: print(f"[p7]   slab {c+1}: end faces too far apart ({(ti+tj)/HF.pitch:.0f} vox vs slab {ext[0]:.0f}) -> skip", flush=True); continue
        if fi == fj or (set(F[fi]) & set(F[fj])): continue
        ci, cj = V[F[fi]].mean(0), V[F[fj]].mean(0)
        print(f"[p7] bridge evidence: slab {c+1} {int(sz[c])} vox at {np.round(c0,3)}, extent {np.round(ext,1)} vox, end faces {fi},{fj} (gaps {ti/HF.pitch:.0f}/{tj/HF.pitch:.0f} vox)", flush=True)
        return fi, fj, ci, cj, key
    return None

def find_contact_join(V, F, prev_handles=(), r_vox=float(os.environ.get("CONTACT_R_VOX", "1.5")), cos_max=-0.7):
    """LESSONS 25b: a handle that no image can see = two surface sheets pressed together but not joined
    (fertility 4th tunnel wall). Find non-adjacent face pairs with opposed normals within r_vox voxels,
    cluster them, and return the best pair of the largest cluster for a zero-length add_handle (= join)."""
    from scipy.spatial import cKDTree
    tri = V[F]; cen = tri.mean(1); nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    vf = {}
    for k, f in enumerate(F):
        for x in f: vf.setdefault(int(x), set()).add(k)
    ring = [set().union(*[vf[int(x)] for x in f]) for f in F]                       # faces sharing a vertex (1-ring)
    ringv = lambda k: set(int(x) for q in ring[k] for x in F[q])                        # vertices of the 1-ring
    disjoint = lambda a, b: not (ringv(a) & set(map(int, F[b]))) and not (ringv(b) & set(map(int, F[a])))
    pairs = cKDTree(cen).query_pairs(r=r_vox * pitch, output_type="ndarray")
    ok = [(int(i), int(j)) for i, j in pairs if j not in ring[i] and nrm[i] @ nrm[j] < cos_max
          and abs((cen[j] - cen[i]) @ nrm[i]) > 0.3 * np.linalg.norm(cen[j] - cen[i])]
    print(f"[p7] contact search: {len(pairs)} face pairs within {r_vox} vox, {len(ok)} opposed-normal non-adjacent contacts", flush=True)
    if not ok: return None
    P = np.array([(cen[i] + cen[j]) / 2 for i, j in ok]); parent = list(range(len(P)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(P).query_pairs(r=4 * pitch, output_type="ndarray"): parent[find(int(a))] = find(int(b))
    roots = np.array([find(i) for i in range(len(P))])
    for r in sorted(set(roots.tolist()), key=lambda r: -(roots == r).sum()):
        m = np.where(roots == r)[0]; mid = P[m].mean(0); key = ["contact", [round(float(x), 2) for x in mid]]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)): continue
        best = sorted(m, key=lambda q: nrm[ok[q][0]] @ nrm[ok[q][1]])                   # most opposed first
        for q in best:
            i, j = ok[q]
            if disjoint(i, j):
                print(f"[p7] contact evidence: cluster of {len(m)} pairs at {np.round(mid, 3)}, join faces {i},{j} (dist {np.linalg.norm(cen[j]-cen[i])/pitch:.2f} vox, n.n {nrm[i]@nrm[j]:.2f})", flush=True)
                return i, j, cen[i], cen[j], key
        print(f"[p7]   contact cluster at {np.round(mid, 3)}: no pair with disjoint 1-rings -> skip", flush=True)
    return None


def find_tunnel_by_hull(V, F, HF_in, prev_handles=(), g_target=None):
    """Locate tunnel face pairs from space-carved hull closing-radius analysis.

    Closing ladder R=4,6,...,40 on a downsampled (HULL_LOC_RES^3) clean hull.  Each newly
    filled 26-connected component that lowers the cavity-corrected hull genus by exactly -1 is
    one tunnel throat ("plug").  Merged plugs (dg<=-2) are split via greedy-merge watershed.
    Throat isolation via EDT sub-blob for axis/centroid.  Face pairs found with Open3D rays
    along +-axis; accepted when both hit, non-adjacent, membrane patches are disks, face
    centroids within HULL_FACE_DIST_VOX voxels of the plug.

    Returns list of (fi, fj, ci, cj, key) ordered by closing radius (thinnest throat first).
    key = ["hull", [cx,cy,cz]] with plug centroid in world units (3 dp).  Empty list if none.
    Plug centroids cached to HULL_PLUGS_CACHE JSON so subsequent rounds skip the ladder.
    """
    from skimage import measure, morphology, segmentation, feature
    from scipy import ndimage
    import open3d as o3d, time as _time, json as _json

    t_start = _time.time()

    def _genus_solid(vol):
        return 1 - measure.euler_number(ndimage.binary_fill_holes(vol), connectivity=1)

    lo_w = np.asarray(HF_in.lo, float); hi_w = np.asarray(HF_in.hi, float)
    res = HULL_LOC_RES
    def vox2world(v): return lo_w + np.asarray(v, float) / (res - 1) * (hi_w - lo_w)
    def world2vox(w): return (np.asarray(w, float) - lo_w) / (hi_w - lo_w) * (res - 1)

    # ── Load or compute plug centroids ─────────────────────────────────────────
    plugs_raw = None
    if HULL_PLUGS_CACHE and os.path.exists(HULL_PLUGS_CACHE):
        try:
            plugs_raw = _json.load(open(HULL_PLUGS_CACHE))
            print(f"[p7] hull: loaded {len(plugs_raw)} plug(s) from cache {HULL_PLUGS_CACHE}", flush=True)
        except Exception as _e:
            print(f"[p7] hull: cache load failed ({_e}), recomputing", flush=True)
            plugs_raw = None

    if plugs_raw is None:
        hull_full = np.asarray(HF_in.hull).astype(bool)
        N_full = hull_full.shape[0]

        # Clean hull: closing+opening r=2 (6-connected S3, iterations=2) exactly as hull_genus
        S3_c = ndimage.generate_binary_structure(3, 1)
        hc = ndimage.binary_opening(ndimage.binary_closing(hull_full, S3_c, iterations=2), S3_c, iterations=2)
        lab_hc, n_hc = ndimage.label(hc)
        if n_hc > 1: hc = lab_hc == (np.bincount(lab_hc.ravel())[1:].argmax() + 1)
        hc = ndimage.binary_fill_holes(hc)
        print(f"[p7] hull: cleaned 256^3 hull in {_time.time()-t_start:.0f}s", flush=True)

        # Downsample to res^3
        if res < N_full and N_full % res == 0:
            f_ds = N_full // res
            hs = hc.reshape(res, f_ds, res, f_ds, res, f_ds).max(axis=(1, 3, 5))
        elif res == N_full:
            hs = hc.copy()
        else:
            from scipy.ndimage import zoom as _zoom
            hs = _zoom(hc.astype(float), res / N_full, order=0).astype(bool)

        g0 = _genus_solid(hs)
        g_mesh = genus(V, F)
        n_needed = (g_target - g_mesh) if g_target is not None else g0
        print(f"[p7] hull: {res}^3 genus_solid={g0}, mesh_genus={g_mesh}, need={n_needed}", flush=True)

        plugs_raw = []
        if n_needed > 0 and g0 > 0:
            pitch_xyz = (hi_w - lo_w) / (res - 1)
            edt_bg = ndimage.distance_transform_edt(~hs).astype(np.float32)
            S26 = np.ones((3, 3, 3), dtype=bool)
            claimed = np.zeros_like(hs, dtype=bool)
            prev_close = hs.copy()            # closing at the previous radius (increment base)
            if os.environ.get("HULL_VIZ_DIR"): np.save(os.path.join(os.environ["HULL_VIZ_DIR"], "hull_ds.npy"), hs)

            for R in range(4, 41, 2):
                cur = hs | claimed
                gcur = _genus_solid(cur)
                if gcur == 0: break
                if len(plugs_raw) >= n_needed: break

                dil = edt_bg <= R
                edt_dil = ndimage.distance_transform_edt(dil).astype(np.float32)
                close_R = dil & (edt_dil > R)
                incr = close_R & ~prev_close        # voxels added at THIS radius only (fillets from lower R excluded)
                prev_close = close_R
                D = close_R & ~hs & ~claimed   # cumulative region from morphological closing (acceptance test)

                lab_d, n_lab = ndimage.label(D, structure=S26)
                if n_lab == 0:
                    print(f"[p7] hull R={R}: no new region gcur={gcur} ({_time.time()-t_start:.0f}s)", flush=True)
                    continue
                sizes = np.bincount(lab_d.ravel())[1:]

                for c_idx in np.argsort(-sizes):
                    if sizes[c_idx] < 50: break
                    if len(plugs_raw) >= n_needed: break
                    comp = lab_d == (c_idx + 1)
                    dg = _genus_solid(cur | comp) - gcur

                    if dg == 0:
                        continue
                    elif dg == -1:
                        pieces_list = [comp]
                    elif dg <= -2:
                        # Greedy-merge watershed for merged plugs
                        edt_c = ndimage.distance_transform_edt(comp).astype(np.float32)
                        pk = feature.peak_local_max(edt_c, min_distance=6,
                                                    labels=comp.astype(int), exclude_border=False)
                        if len(pk) == 0:
                            pieces_list = [comp]
                        else:
                            mk = np.zeros(comp.shape, int)
                            mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
                            ws_labels = segmentation.watershed(-edt_c, mk, mask=comp)

                            remaining = set(range(1, len(pk) + 1))
                            pieces_list = []
                            S6 = ndimage.generate_binary_structure(3, 1)

                            _g_base = _genus_solid(hs | claimed)
                            while remaining:
                                best_start = max(remaining,
                                    key=lambda pi: float(edt_c[ws_labels == pi].max()) if (ws_labels == pi).any() else 0)
                                union = ws_labels == best_start
                                remaining.discard(best_start)

                                for _iter in range(len(remaining) + 2):
                                    dg_u = _genus_solid(hs | claimed | union) - _g_base
                                    if dg_u == -1:
                                        pieces_list.append(union.copy()); break
                                    elif dg_u < -1:
                                        break  # over-merged: this seed bridges too many tunnels
                                    elif remaining:
                                        # dg_u == 0: not yet bridging a tunnel, merge more adjacent
                                        udil = ndimage.binary_dilation(union, S6, iterations=1)
                                        adj = [pi for pi in remaining
                                               if (ws_labels == pi)[udil].any()]
                                        if not adj: break
                                        best_adj = max(adj,
                                            key=lambda pi: float(edt_c[ws_labels == pi].max()) if (ws_labels == pi).any() else 0)
                                        union = union | (ws_labels == best_adj)
                                        remaining.discard(best_adj)
                                    else:
                                        break

                            print(f"[p7] hull R={R} merged dg={dg}: "
                                  f"watershed+greedy -> {len(pieces_list)} plugs "
                                  f"({len(pk)} watershed pieces)", flush=True)
                    else:
                        continue

                    for piece in pieces_list:
                        if len(plugs_raw) >= n_needed: break
                        dgp = (_genus_solid(hs | claimed | piece)
                               - _genus_solid(hs | claimed))
                        if dgp != -1: continue

                        # Throat isolation: sub-blob = plug voxels within 1.5x
                        # throat_r of the EDT maximum (spec)
                        edt_p = ndimage.distance_transform_edt(piece).astype(np.float32)
                        throat_r = float(edt_p.max())
                        if throat_r < 1.5: continue

                        # EDT peak = center of the throat
                        peak_idx = np.unravel_index(edt_p.argmax(), edt_p.shape)
                        peak_vox = np.array(peak_idx, dtype=float)

                        # Throat core = the part of this plug that was added at THIS closing radius
                        # (the last sheet that sealed the tunnel): a thin disk across the throat.
                        # Fillets filled at lower radii belong to the plug for the genus bookkeeping
                        # but must not bias the centre/axis. Largest 26-component of the increment.
                        core = piece & incr
                        if core.sum() >= 10:
                            lab_k, nk = ndimage.label(core, structure=S26)
                            if nk > 1: core = lab_k == (np.bincount(lab_k.ravel())[1:].argmax() + 1)
                        if core.sum() >= 10:
                            pts = np.array(np.nonzero(core)).T.astype(float)
                            peak_vox = pts.mean(0)              # disk centroid = tunnel centre
                            core_mask = core
                        else:                                   # fallback: EDT-peak sub-blob (old rule)
                            pts_all = np.array(np.nonzero(piece)).T.astype(float)
                            dists_to_peak = np.linalg.norm(pts_all - peak_vox[None, :], axis=1)
                            sub_mask = dists_to_peak <= 1.5 * throat_r
                            pts = pts_all[sub_mask] if sub_mask.sum() >= 10 else pts_all
                            core_mask = piece

                        cen_vox = pts.mean(0)
                        if len(pts) > 3:
                            ev, evec = np.linalg.eigh(np.cov(pts.T))
                        else:
                            ev, evec = np.ones(3), np.eye(3)
                        axis_vox = evec[:, 0]  # smallest variance = tunnel axis
                        is_disk = bool(ev[1] > 0 and ev[0] < ev[1])

                        cen_w = vox2world(cen_vox)
                        peak_w = vox2world(peak_vox)
                        axis_w = axis_vox * pitch_xyz
                        aw_n = float(np.linalg.norm(axis_w))
                        axis_w = axis_w / (aw_n + 1e-12)

                        claimed |= piece
                        if os.environ.get("HULL_VIZ_DIR"):
                            np.save(os.path.join(os.environ["HULL_VIZ_DIR"], f"plug{len(plugs_raw)}_vox.npy"), np.argwhere(core_mask))
                            np.save(os.path.join(os.environ["HULL_VIZ_DIR"], f"plug{len(plugs_raw)}_full.npy"), np.argwhere(piece))
                        key = ["hull", [round(float(x), 3) for x in cen_w]]
                        plugs_raw.append({
                            "R": int(R), "key": key,
                            "cen_vox": cen_vox.tolist(), "cen_w": cen_w.tolist(),
                            "peak_w": peak_w.tolist(),
                            "axis_w": axis_w.tolist(), "is_disk": is_disk,
                            "size": int(piece.sum()), "throat_r": float(throat_r),
                        })
                        print(f"[p7] hull R={R:2d} PLUG size={int(piece.sum()):6d} "
                              f"cen_vox={np.round(cen_vox,1)} is_disk={is_disk} "
                              f"throat_r={throat_r:.1f} axis={np.round(axis_w,2)}", flush=True)

                print(f"[p7] hull R={R}: genus_remain={_genus_solid(hs|claimed)} "
                      f"plugs={len(plugs_raw)}/{n_needed} ({_time.time()-t_start:.0f}s)", flush=True)

        print(f"[p7] hull plugs: {len(plugs_raw)} found (R ladder) "
              f"target={n_needed} in {_time.time()-t_start:.0f}s", flush=True)

        if HULL_PLUGS_CACHE:
            try:
                os.makedirs(os.path.dirname(HULL_PLUGS_CACHE) or ".", exist_ok=True)
                _json.dump(plugs_raw, open(HULL_PLUGS_CACHE, "w"))
                print(f"[p7] hull: cached to {HULL_PLUGS_CACHE}", flush=True)
            except Exception as _e:
                print(f"[p7] hull: cache write failed ({_e})", flush=True)

    # ── Ray-cast face pairs for each unclaimed plug ─────────────────────────────
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(V.astype(np.float32)),
        o3d.core.Tensor(F.astype(np.int32))))
    # Use a low OUT_VOX threshold for hull mode: coarse meshes have membrane faces
    # barely outside the hull (< 1 voxel), while the default OUT_VOX=4 is tuned for
    # the rays detector on refined meshes.  0.5 voxels catches them reliably.
    comp_mp, chi_mp = membrane_patches(V, F, out_vox_override=0.5)

    E_all = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    me_hull = np.linalg.norm(V[E_all[:, 0]] - V[E_all[:, 1]], axis=1).mean()

    def _is_dedup(cen_w_chk, key_chk=None):
        for h in prev_handles:
            if not isinstance(h, dict): continue
            if key_chk is not None and h.get("blob") == key_chk:
                return True
            mid = h.get("mid")
            if mid is not None and np.linalg.norm(np.asarray(mid) - np.asarray(cen_w_chk)) < R_DEDUP * me_hull:
                return True
        return False

    result_list = []
    n_dedup = n_no_face = 0

    tri_cen = V[F].mean(1)   # [F, 3]  face centroids

    for p in plugs_raw:
        cen_w = np.asarray(p["cen_w"])
        axis_w = np.asarray(p["axis_w"])
        key = p["key"]
        R = p["R"]
        is_disk = p.get("is_disk", True)
        throat_r = p.get("throat_r", 2.0)
        peak_w = np.asarray(p.get("peak_w", p["cen_w"]))

        if _is_dedup(cen_w, key):
            n_dedup += 1
            print(f"[p7] hull plug R={R} at {np.round(cen_w,3)}: dedup skip", flush=True)
            continue

        # ── Strategy 1: proximity-based pair from membrane patches ────────────
        # Find membrane disk faces near the plug, grouped by patch (connected
        # component). Pick the closest face from each of the two nearest patches.
        # Grouping by patch instead of axis side handles boundary plugs where
        # both membrane sheets project onto the same axis side.
        cen_vox_ref = world2vox(cen_w)
        peak_vox_ref = world2vox(peak_w)
        from collections import defaultdict as _ddict
        patch_faces_near = _ddict(list)   # root -> [(fi, dist)]
        for fi, root in comp_mp.items():
            if chi_mp.get(root) != 1: continue   # only disk patches
            fc = tri_cen[fi]
            fc_vox = world2vox(fc)
            d_cen = float(np.linalg.norm(fc_vox - cen_vox_ref))
            d_peak = float(np.linalg.norm(fc_vox - peak_vox_ref))
            d_min = min(d_cen, d_peak)
            if d_min > 50: continue  # sanity limit
            patch_faces_near[root].append((fi, d_min))

        # Sort patches by distance of their closest face to the plug
        patch_order = sorted(patch_faces_near.keys(),
            key=lambda r: min(d for _, d in patch_faces_near[r]))

        fi_fj = None
        # Try pairs from distinct patches (closest two patches first)
        for pi_idx in range(len(patch_order)):
            if fi_fj is not None: break
            for pj_idx in range(pi_idx + 1, len(patch_order)):
                if fi_fj is not None: break
                ri = patch_order[pi_idx]; rj = patch_order[pj_idx]
                for fi_c, _ in sorted(patch_faces_near[ri], key=lambda x: x[1]):
                    if fi_fj is not None: break
                    for fj_c, _ in sorted(patch_faces_near[rj], key=lambda x: x[1]):
                        if fi_c == fj_c: continue
                        if set(F[fi_c]) & set(F[fj_c]): continue
                        fi_fj = (fi_c, fj_c, tri_cen[fi_c].copy(), tri_cen[fj_c].copy())
                        break

        # ── Strategy 2: ray-cast fallback ─────────────────────────────────────
        if fi_fj is None:
            sample_pts = [peak_w, cen_w]
            for origin in [peak_w, cen_w]:
                for t_frac in np.linspace(-1.0, 1.0, 12):
                    sample_pts.append(origin + t_frac * float(throat_r) * float(HF_in.pitch) * axis_w)

            for pt in sample_pts:
                if fi_fj is not None: break
                hits_pair = []
                for sgn in (+1, -1):
                    dvec = (sgn * axis_w).astype(np.float32)
                    ray = o3d.core.Tensor(np.concatenate([pt.astype(np.float32), dvec])[None])
                    h_res = sc.cast_rays(ray)
                    t_hit = float(h_res["t_hit"].numpy()[0])
                    fi_hit = int(h_res["primitive_ids"].numpy()[0]) if np.isfinite(t_hit) else -1
                    hits_pair.append(fi_hit)

                fi_h, fj_h = hits_pair
                if fi_h < 0 or fj_h < 0 or fi_h == fj_h: continue
                if set(F[fi_h]) & set(F[fj_h]): continue
                ki = comp_mp.get(fi_h); kj = comp_mp.get(fj_h)
                if ki is None or kj is None: continue
                if chi_mp.get(ki) != 1 or chi_mp.get(kj) != 1: continue
                ci_w = V[F[fi_h]].mean(0); cj_w = V[F[fj_h]].mean(0)
                if np.linalg.norm(world2vox(ci_w) - cen_vox_ref) > 50: continue
                if np.linalg.norm(world2vox(cj_w) - cen_vox_ref) > 50: continue
                fi_fj = (fi_h, fj_h, ci_w, cj_w)

        if fi_fj is None:
            n_no_face += 1
            print(f"[p7] hull plug R={R} at {np.round(cen_w,3)}: no valid face pair "
                  f"({len(patch_faces_near)} patches near), rays fallback", flush=True)
            continue

        fi_r, fj_r, ci_r, cj_r = fi_fj
        print(f"[p7] hull plug R={R}: accepted faces {fi_r},{fj_r} sep={np.linalg.norm(cj_r-ci_r):.3f}", flush=True)
        result_list.append((fi_r, fj_r, ci_r, cj_r, key))

    g_mesh_cur = genus(V, F)
    print(f"[p7] hull plugs: {len(plugs_raw)} found (R=..), {len(result_list)} accepted "
          f"(skip: {n_dedup} dedup, {n_no_face} !face), "
          f"genus {g_mesh_cur} -> {g_mesh_cur + len(result_list)}", flush=True)
    return result_list


report("base", V, Fa); _snap(f"Stage 7 [genus discovery] base genus={genus(V, Fa)}", V, Fa)
import json
prev_handles = json.load(open(HANDLES_JSON)) if HANDLES_JSON and os.path.exists(HANDLES_JSON) else []
n_added = 0
MODE_BRIDGE = False
# When DETECT=hull, override MAX_HANDLES so phase7_multi.sh's MAX_HANDLES=1
# doesn't prevent adding multiple hull handles in one invocation (spec: "add ALL
# returned handles in that round").  find_tunnel_by_hull is called fresh each
# iteration because add_handle modifies V/Fa and invalidates face indices.
_hull_max = MAX_HANDLES if DETECT != "hull" else max(MAX_HANDLES, G_TARGET if G_TARGET else MAX_HANDLES)
for k in range(_hull_max):
    MODE_BRIDGE = False
    if G_TARGET is not None and genus(V, Fa) >= G_TARGET:
        print(f"[p7] genus {genus(V, Fa)} == target g*={G_TARGET}: no more handles", flush=True); break
    if DETECT == "hull":
        _hull_cands = find_tunnel_by_hull(V, Fa, HF, prev_handles, G_TARGET)
        if _hull_cands:
            hit = _hull_cands[0]
            i, j, _ci, _cj, _blob = hit
            tri = V[Fa]; cen = tri.mean(1); d = hdist(cen)
            negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
            print(f"[p7] tunnel-evidence pairs: 1 (hull)", flush=True)
        elif G_TARGET is not None and genus(V, Fa) < G_TARGET:
            # Hull candidates exhausted or no valid face pair: fall back to rays
            print(f"[p7] hull: 0 valid candidates, genus {genus(V, Fa)} < g*={G_TARGET}: rays fallback", flush=True)
            hit = None
            for _lv, (_mp, _ov) in enumerate(RELAX):
                OUT_VOX = _ov
                hit = find_tunnel_by_rays(V, Fa, min_px=int(_mp), prev_handles=prev_handles)
                if hit is not None:
                    if _lv > 0: print(f"[p7] rays fallback at relaxation level {_lv} (min_px {_mp}, out_vox {_ov})", flush=True)
                    break
            if hit is None and int(os.environ.get("BRIDGE", "0")):
                hit = find_bridge(V, Fa)
                if hit is not None: MODE_BRIDGE = True
            if hit is None and int(os.environ.get("CONTACT", "1")):
                hit = find_contact_join(V, Fa, prev_handles=prev_handles)
                if hit is not None: MODE_BRIDGE = True
            if hit is None:
                print(f"[p7] tunnel-evidence pairs: 0 (genus {genus(V, Fa)} < g*={G_TARGET}: UNREACHED)", flush=True); break
            i, j, _ci, _cj, _blob = hit
            tri = V[Fa]; cen = tri.mean(1); d = hdist(cen)
            negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
            print(f"[p7] tunnel-evidence pairs: 1 (ray fallback)", flush=True)
        else:
            print(f"[p7] tunnel-evidence pairs: 0", flush=True); break
    elif DETECT == "rays":
        hit = None
        for _lv, (_mp, _ov) in enumerate(RELAX if G_TARGET is not None else RELAX[:1]):
            OUT_VOX = _ov
            hit = find_tunnel_by_rays(V, Fa, min_px=int(_mp), prev_handles=prev_handles)
            if hit is not None:
                if _lv > 0: print(f"[p7] evidence found at relaxation level {_lv} (min_px {_mp}, out_vox {_ov}) because genus {genus(V, Fa)} < g*={G_TARGET}", flush=True)
                break
        if hit is None and G_TARGET is not None and genus(V, Fa) < G_TARGET and int(os.environ.get("BRIDGE", "0")):
            hit = find_bridge(V, Fa)
            if hit is not None: MODE_BRIDGE = True
        if hit is None and G_TARGET is not None and genus(V, Fa) < G_TARGET and int(os.environ.get("CONTACT", "1")):
            hit = find_contact_join(V, Fa, prev_handles=prev_handles)
            if hit is not None: MODE_BRIDGE = True                                   # plain single-face add_handle (no membrane merge)
        if hit is None: print(f"[p7] tunnel-evidence pairs: 0" + (f" (genus {genus(V, Fa)} < g*={G_TARGET}: UNREACHED)" if G_TARGET is not None else ""), flush=True); break
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
    # Thin/pinched membrane: entry and exit faces are on different sheets but their 1-rings overlap
    # (sheets touch at 1-ring distance) -> the tube's side quads would duplicate existing edges and break
    # watertightness (fertility 4th tunnel: sep 0.9 edges, ring overlap 10). Step each face away from the
    # other sheet: pick the nearest face (by centroid) on the same side whose vertex 1-ring is disjoint
    # from the other face's vertex 1-ring.
    _vf = {}
    for _k, _f in enumerate(Fa):
        for _x in _f: _vf.setdefault(int(_x), []).append(_k)
    def _ringverts(f):                       # vertices of all faces sharing a vertex with face f
        return set(int(x) for v in Fa[f] for k in _vf[int(v)] for x in Fa[k])
    def _disjoint(a, b): return not (_ringverts(a) & set(int(x) for x in Fa[b])) and not (_ringverts(b) & set(int(x) for x in Fa[a]))
    if not _disjoint(i, j):
        _i0, _j0 = i, j
        _ni = np.argsort(np.linalg.norm(cen - cen[i], axis=1))[:80]; _nj = np.argsort(np.linalg.norm(cen - cen[j], axis=1))[:80]
        _ni = [int(k) for k in _ni if d[k] > 0]; _nj = [int(k) for k in _nj if d[k] > 0]     # stay outside the hull (on the membrane)
        _pairs = [(a_, b_) for a_ in _ni for b_ in _nj if _disjoint(a_, b_)]
        if _pairs:
            i, j = min(_pairs, key=lambda ab: np.linalg.norm(cen[ab[0]] - cen[_i0]) + np.linalg.norm(cen[ab[1]] - cen[_j0]))
        print(f"[p7] pinched membrane (1-rings overlapped): stepped faces {_i0},{_j0} -> {i},{j}; candidates={len(_pairs)}, new sep {np.linalg.norm(cen[j]-cen[i])/np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean():.2f} edges", flush=True)
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
    if OPEN == "merge" and not MODE_BRIDGE:
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
    prev_handles.append({"mid": ((cen[i] + cen[j]) / 2).tolist(), "blob": _blob if DETECT in ("rays", "hull") else None})
    if HANDLES_JSON: json.dump(prev_handles, open(HANDLES_JSON, "w"))
    report(f"after handle {n_added}", V, Fa); _snap(f"Stage 7 [DLFL add_handle #{n_added}] genus={genus(V, Fa)}", V, Fa, hold=45)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[p7] handles added: {n_added}; saved cow_{SHAPE}_{TAG}.npz | genus {genus(V, Fa)}" + (f" / target {G_TARGET}" if G_TARGET is not None else ""), flush=True)
