"""Iterative TopMod amputation surgery: tube-collapse + snap + spike-snap,
repeated until detectors are clean. Judged only by 6-view training data."""
import sys
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
import numpy as np, torch, collections
from topmod.primitives import _build_mesh
from topmod.high_level_ops import collapse_edge_tri
from topmod.validate import check_all
from topmod.io import to_triangle_arrays


def _adj_torch(F, nv, dev):
    adj = collections.defaultdict(set)
    for a, b, c in F:
        adj[int(a)] |= {int(b), int(c)}
        adj[int(b)] |= {int(a), int(c)}
        adj[int(c)] |= {int(a), int(b)}
    src, dst = [], []
    for i, ns in adj.items():
        for j in ns:
            src.append(i); dst.append(j)
    src = torch.tensor(src, device=dev); dst = torch.tensor(dst, device=dev)
    return adj, src, dst


def detect(V, F, dev="cuda", tube_thr=0.4, spike_thr=0.85):
    nv = len(V)
    v = torch.tensor(V, dtype=torch.float32, device=dev)
    adj, src, dst = _adj_torch(F, nv, dev)
    me = (v[src] - v[dst]).norm(dim=-1).mean()
    A = torch.zeros(nv, nv, dtype=torch.bool, device=dev)
    A[src, dst] = True
    excl = A | ((A.float() @ A.float()) > 0) | torch.eye(nv, dtype=torch.bool, device=dev)
    D = torch.cdist(v, v); D[excl] = 1e9
    thin = D.min(1).values < tube_thr * me
    deg = torch.zeros(nv, device=dev).index_add_(
        0, src, torch.ones_like(src, dtype=torch.float32)).clamp(min=1).unsqueeze(-1)
    cen = torch.zeros_like(v).index_add_(0, src, v[dst]) / deg
    lap = (v - cen).norm(dim=-1)
    spike = lap > spike_thr * me
    return thin.cpu().numpy(), spike.cpu().numpy(), adj, float(me)


def _propagate_flag(flagged, seed, adj):
    """Component-propagation gate (v22): keep a flagged vertex ONLY if its
    flagged-connected-component contains at least one `seed` vertex.
    The needle's whole thin shaft is `flagged` (thin end-to-end); only its TIP
    is a `seed` (escapes GT).  Propagating the tip seed over the thin component
    condemns the WHOLE shaft -> root-and-all removal.  The ear is a flagged
    (thin) component with NO seed (stays inside GT) -> spared."""
    n = len(flagged)
    seen = np.zeros(n, bool)
    out = np.zeros(n, bool)
    for i in np.where(flagged)[0]:
        if seen[i]:
            continue
        stack = [i]; seen[i] = True; comp = [i]; has_seed = bool(seed[i])
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if flagged[w] and not seen[w]:
                    seen[w] = True; comp.append(w); stack.append(w)
                    if seed[w]:
                        has_seed = True
        if has_seed:
            for c in comp:
                out[c] = True
    return out


def surgery(V, F, iou_fn=None, iou_budget=8e-4, global_cap=3e-3,
            max_rounds=6, max_grow=4, verbose=True, escape_fn=None):
    """Amputate flagged (thin|spike) components with train6-IoU verification.
    Per-comp acceptance: real amputation must cost <= iou_budget RELATIVE TO
    CURRENT iou, and never sink below iou0 - global_cap.  A comp whose bare
    amputation is too costly (dangling-spike distortion) is retried with
    1..max_grow rings of neighbors added — whole-fin deletion of a zombie
    structure is nearly silhouette-neutral while base-only cuts are not."""
    iou0 = iou_fn(V, F) if iou_fn else None
    for rnd in range(max_rounds):
        nv_before = len(V)
        thin, spike, adj, me = detect(V, F)
        flagged = thin | spike
        # v22: escape is the GT-mismatch SEED (needle tip pokes outside GT),
        # thickness is the REMOVAL EXTENT.  Propagate the tip seed over the
        # thin-connected component so the whole buried shaft is condemned;
        # a thin component with no escaping tip (the ear) is spared.
        if escape_fn is not None:
            esc = escape_fn(V)
            flagged = _propagate_flag(flagged, esc, adj)
            if verbose:
                print(f"round {rnd}: geom={int((thin|spike).sum())} "
                      f"seed={int(esc.sum())} condemned={int(flagged.sum())}")
        if not flagged.any():
            if verbose: print(f"round {rnd}: clean, stop")
            break
        # group flagged verts into connected components
        seen = np.zeros(len(V), bool)
        comps = []
        for i in np.where(flagged)[0]:
            if seen[i]: continue
            stack = [i]; seen[i] = True; comp = [i]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if flagged[w] and not seen[w]:
                        seen[w] = True; comp.append(w); stack.append(w)
            comps.append([V[c].copy() for c in comp])
        # greedy incremental: real surgery per comp (with ring growth on
        # failure), verify, rollback if bad.  Vertex indices shift after each
        # accepted surgery; remap comps by exact position matching.
        n_acc = n_rej = 0
        for cpos in comps:
            posmap = {tuple(np.round(V[i], 9)): i for i in range(len(V))}
            comp2 = [posmap.get(tuple(np.round(p, 9))) for p in cpos]
            if any(c is None for c in comp2):
                n_rej += 1; continue  # verts vanished in earlier surgery
            iou_cur = iou_fn(V, F) if iou_fn else 0.0
            # rebuild adjacency lazily only when growing (indices shift)
            adj_cur = None
            grown = list(comp2)
            accepted = False
            for g in range(max_grow + 1):
                if g > 0:
                    if adj_cur is None:
                        adj_cur = collections.defaultdict(set)
                        for a, b, c in F:
                            adj_cur[int(a)] |= {int(b), int(c)}
                            adj_cur[int(b)] |= {int(a), int(c)}
                            adj_cur[int(c)] |= {int(a), int(b)}
                    ring = set(grown)
                    for i in list(ring):
                        ring |= adj_cur[i]
                    if len(ring) == len(grown):
                        break
                    grown = list(ring)
                V2, F2, ok = _amputate_comp(V, F, grown)
                if not ok:
                    continue
                iou = iou_fn(V2, F2) if iou_fn else iou0
                if iou_fn and (iou < iou_cur - iou_budget
                               or iou < iou0 - global_cap):
                    continue
                V, F = V2, F2
                n_acc += 1; accepted = True
                break
            if not accepted:
                n_rej += 1
        if verbose:
            print(f"round {rnd}: comps={len(comps)} accepted={n_acc} "
                  f"rejected={n_rej} iou={iou_fn(V, F):.4f} V={len(V)}"
                  if iou_fn else f"round {rnd}: accepted={n_acc}", flush=True)
        if n_acc == 0 or len(V) == nv_before:
            if verbose: print("no effective surgery left, stop")
            break
    return V, F


def _amputate_comp(V, F, comp):
    """Topological amputation of one flagged component. Returns (V2,F2,ok)."""
    mesh = _build_mesh([tuple(x) for x in V], [list(map(int, f)) for f in F])
    vlist = list(mesh.vertices.values())
    condemned = set(id(vlist[i]) for i in comp)
    skipped = set(); progress = True
    while progress:
        progress = False
        for e in list(mesh.edges.values()):
            if e.id in skipped or e.id not in mesh.edges: continue
            va, vb = e.vertices()
            if id(va) in condemned and id(vb) in condemned:
                s = collapse_edge_tri(mesh, e)
                if s is None: skipped.add(e.id); continue
                condemned.add(id(s)); progress = True
    # phase 2: surviving condemned verts have no condemned partner left —
    # collapse each INTO a healthy neighbor so the vertex truly disappears.
    # Survivor takes the healthy neighbor's position and leaves the condemned
    # set (guaranteed termination, cannot eat into the healthy mesh).
    progress = True
    while progress:
        progress = False
        for v_ in list(mesh.vertices.values()):
            if id(v_) not in condemned or v_.id not in mesh.vertices: continue
            for h in v_.outgoing_halfedges():
                u = h.twin.origin if h.twin else None
                if u is None or id(u) in condemned: continue
                if h.edge is None or h.edge.id in skipped: continue
                hx, hy, hz = u.x, u.y, u.z
                s = collapse_edge_tri(mesh, h.edge)
                if s is None:
                    skipped.add(h.edge.id); continue
                s.x, s.y, s.z = hx, hy, hz
                condemned.discard(id(s))
                progress = True
                break
    for _ in range(4):
        for v_ in mesh.vertices.values():
            if id(v_) not in condemned: continue
            nb = [h.twin.origin for h in v_.outgoing_halfedges() if h.twin]
            healthy = [u for u in nb if id(u) not in condemned] or nb
            if not healthy: continue
            v_.x = sum(u.x for u in healthy) / len(healthy)
            v_.y = sum(u.y for u in healthy) / len(healthy)
            v_.z = sum(u.z for u in healthy) / len(healthy)
    ok, _ = check_all(mesh)
    pos, tris = to_triangle_arrays(mesh)
    return np.array(pos), np.array(tris), ok
