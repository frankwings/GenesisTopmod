"""C2F v2: coarse-to-fine + late-phase fold penalty + IoU-guarded surgical postfix.

Stage 1: cc2 800 -> cc3 800 -> cc4 800, where cc4 phase adds fold penalty
         ramping in over steps 400-800 (low-LR tail; crumples no longer needed).
Stage 2: surgical postfix — only smooth verts of faces that are BOTH hard-fold
         AND rasterize outside the 2-dilated GT (actual hair faces), small
         steps, revert any step that costs > 0.0005 IoU cumulative.
Report: IoU / hard folds / hair pixels at every stage.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from scipy.ndimage import binary_dilation
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
)

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
W_FOLD = 0.02          # late-phase fold penalty peak weight
FOLD_START = 400        # ramp in from this step of cc4 phase
IOU_BUDGET = 0.0015     # max cumulative IoU we may spend in postfix


def midpoint_subdivide(verts, tris):
    verts = list(map(tuple, verts))
    edge_mid = {}
    def mid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in edge_mid:
            va, vb = verts[a], verts[b]
            verts.append(((va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2))
            edge_mid[k] = len(verts) - 1
        return edge_mid[k]
    nt = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(nt, dtype=np.int32)


def build_pairs(tris_np):
    edge2f = {}
    for fi, (a, b, c) in enumerate(tris_np):
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            edge2f.setdefault(k, []).append(fi)
    return np.array([p for p in edge2f.values() if len(p) == 2], dtype=np.int64)


def face_normals(verts_t, faces_l):
    tri = verts_t[faces_l]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-12)


def fold_loss(verts_t, faces_l, pairs_t):
    n = face_normals(verts_t, faces_l)
    d = (n[pairs_t[:, 0]] * n[pairs_t[:, 1]]).sum(-1)
    return F.relu(-d).pow(2).mean()


def census(verts_np, tris_np, pairs_np):
    v = torch.tensor(verts_np, dtype=torch.float32)
    f = torch.tensor(tris_np, dtype=torch.int64)
    n = face_normals(v, f)
    pt = torch.tensor(pairs_np)
    d = (n[pt[:, 0]] * n[pt[:, 1]]).sum(-1)
    return int((d < 0).sum()), int((d < -0.5).sum())


def hair_pixels(ctx, verts_t, faces_t, mvps, gt_uint8):
    sils = render_views_n(ctx, verts_t, faces_t, mvps)
    total = 0
    for i in range(N_VIEWS):
        pred = sils[i] > 0.5
        gt = gt_uint8[i] < 128
        total += int((pred & ~binary_dilation(gt, iterations=2)).sum())
    return total


def hair_faces(ctx, verts_t, faces_t, mvps, gt_uint8):
    """Face IDs whose pixels rasterize outside the 2-dilated GT (any view)."""
    ids = set()
    vh = torch.cat([verts_t, torch.ones_like(verts_t[:, :1])], dim=-1)
    for i in range(N_VIEWS):
        clip = (vh @ mvps[i].T).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(ctx, clip, faces_t, (IMG_RES, IMG_RES))
        fid = rast[0, :, :, 3].long().cpu().numpy()  # 0 = bg, else face_id+1
        gt = gt_uint8[i] < 128
        out = (fid > 0) & ~binary_dilation(gt, iterations=2)
        ids.update((fid[out] - 1).tolist())
    return np.array(sorted(ids), dtype=np.int64)


def optimize_phase(ctx, verts_np, tris_np, gt_uint8, gt_depths, mvps,
                   steps, label, pairs_np=None, w_fold=0.0, fold_start=0):
    targets = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(DEVICE)
                  for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt_uint8[i] < 128).to(DEVICE)
               for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    pairs_t = (torch.tensor(pairs_np, device=DEVICE)
               if pairs_np is not None else None)
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=LR_MIN)
    for step in range(steps):
        opt.zero_grad()
        sil_loss = torch.tensor(0.0, device=DEVICE)
        d_loss = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            sil_loss += F.l1_loss(sil[0], targets[i])
            d_loss += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sil_loss /= N_VIEWS
        d_loss /= N_VIEWS
        loss = (sil_loss + W_DEPTH * d_loss
                + W_LAP * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t))
        if pairs_t is not None and w_fold > 0 and step >= fold_start:
            ramp = min(1.0, (step - fold_start) / max(1, (steps - fold_start) * 0.5))
            loss = loss + w_fold * ramp * fold_loss(verts_t, faces_l, pairs_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps),
                                gt_uint8)
            print(f"  [{label}] step {step:4d}/{steps}  sil={sil_loss.item():.4f}"
                  f"  iou={iou:.4f}", flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt_uint8)
    print(f"  [{label}] PHASE END  IoU={iou:.4f}", flush=True)
    return verts_t.detach().cpu().numpy().astype(np.float64), iou


def surgical_postfix(ctx, verts_np, tris_np, pairs_np, mvps, gt_uint8,
                     max_iters=60):
    """Smooth only verts of faces that are hard-fold AND render outside GT.
    Each step: 30% blend toward 1-ring centroid; revert if IoU drops below
    (start_iou - IOU_BUDGET)."""
    v = verts_np.copy()
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    # vertex -> neighbor adjacency
    V = len(v)
    nbr = [set() for _ in range(V)]
    for a, b, c in tris_np:
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
    nbr = [np.array(sorted(s)) for s in nbr]

    def iou_of(vv):
        vt = torch.tensor(vv, dtype=torch.float32, device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, faces_t, mvps), gt_uint8)

    start_iou = iou_of(v)
    floor = start_iou - IOU_BUDGET
    for it in range(max_iters):
        vt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
        hf = hair_faces(ctx, vt, faces_t, mvps, gt_uint8)
        # hard folds
        n = face_normals(vt.cpu(), torch.tensor(tris_np, dtype=torch.int64))
        pt = torch.tensor(pairs_np)
        d = (n[pt[:, 0]] * n[pt[:, 1]]).sum(-1)
        fold_faces = set()
        for (fa, fb), dd in zip(pairs_np, d.numpy()):
            if dd < -0.3:
                fold_faces.add(int(fa)); fold_faces.add(int(fb))
        targets_f = [f for f in hf if f in fold_faces] or list(hf)
        if not len(targets_f):
            print(f"  [postfix] iter {it}: no hair faces left", flush=True)
            break
        vids = np.unique(tris_np[np.array(targets_f, dtype=np.int64)].ravel())
        prop = v.copy()
        for vid in vids:
            if len(nbr[vid]):
                prop[vid] = 0.7 * v[vid] + 0.3 * v[nbr[vid]].mean(axis=0)
        new_iou = iou_of(prop)
        if new_iou >= floor:
            v = prop
            if it % 10 == 0:
                print(f"  [postfix] iter {it}: faces={len(targets_f)} "
                      f"iou={new_iou:.4f} (accepted)", flush=True)
        else:
            print(f"  [postfix] iter {it}: iou {new_iou:.4f} < floor "
                  f"{floor:.4f} — stop", flush=True)
            break
    return v, iou_of(v), start_iou


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt_uint8, gt_depths = scene["gt_uint8"], scene["gt_depths"]

    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _ = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc2")
    v, t = midpoint_subdivide(v, t)
    v, _ = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc3")
    v, t = midpoint_subdivide(v, t)
    pairs = build_pairs(t)
    v, iou4 = optimize_phase(ctx, v, t, gt_uint8, gt_depths, mvps, 800, "cc4+fold",
                             pairs_np=pairs, w_fold=W_FOLD, fold_start=FOLD_START)

    neg, hard = census(v, t, pairs)
    vt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    hp = hair_pixels(ctx, vt, ft, mvps, gt_uint8)
    print(f"\n[stage1 c2f+foldpen] IoU={iou4:.4f} neg={neg} hard={hard} hair={hp}",
          flush=True)

    v2, iou_pf, iou_pre = surgical_postfix(ctx, v, t, pairs, mvps, gt_uint8)
    neg2, hard2 = census(v2, t, pairs)
    vt2 = torch.tensor(v2, dtype=torch.float32, device=DEVICE)
    hp2 = hair_pixels(ctx, vt2, ft, mvps, gt_uint8)

    print(f"\n=== V2 RESULT ===")
    print(f"c2f v1 ref:        IoU=0.9838 hard=914  hair=447")
    print(f"stage1 (+foldpen): IoU={iou4:.4f} hard={hard} hair={hp}")
    print(f"stage2 (+postfix): IoU={iou_pf:.4f} hard={hard2} hair={hp2}")
    print(f"cc4 direct ref:    IoU=0.9804 hard=2123 hair~1690")
    print(f"time: {time.time()-t0:.0f}s", flush=True)

    with open(f"{OUT}/cow_c2f_v2.obj", "w") as fh:
        for x, y, z in v2:
            fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t:
            fh.write(f"f {a+1} {b+1} {c+1}\n")

    import imageio.v2 as imageio
    sils = render_views_n(ctx, vt2, ft, mvps)
    rows = []
    for i in range(N_VIEWS):
        pred = sils[i] > 0.5
        gt = gt_uint8[i] < 128
        img = np.full((IMG_RES, IMG_RES, 3), 255, np.uint8)
        img[gt & ~pred] = (220, 60, 60)
        img[pred & ~gt] = (60, 200, 60)
        img[gt & pred] = (200, 200, 80)
        rows.append(img)
    grid = np.concatenate([np.concatenate(rows[:3], 1),
                           np.concatenate(rows[3:], 1)], 0)
    imageio.imwrite(f"{OUT}/cow_c2f_v2.png", grid)
    print(f"saved {OUT}/cow_c2f_v2.obj / .png", flush=True)


if __name__ == "__main__":
    main()
