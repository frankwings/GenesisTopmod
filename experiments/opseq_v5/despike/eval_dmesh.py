"""Evaluate DMesh reconstruction outputs with OUR exam (16 held-out views).

DMesh normalizes meshes into its own domain, so we first similarity-align the
prediction to our normalized GT frame using the bounding boxes of DMesh's own
saved gt_mesh.obj vs our normalize_to_range(armadillo).

Usage:
  python3 eval_dmesh.py --pred <mesh.obj> --dmesh-gt <gt_mesh.obj> \
      --shape armadillo --tag dmesh64
Outputs: metrics printed + aligned obj saved to /tmp/liou_cow_viz/<tag>_aligned.obj
"""
import os, sys, argparse
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
from scipy.ndimage import binary_dilation, label as cc_label
import nvdiffrast.torch as dr
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               load_obj, normalize_to_range, BUNNY_PATH)
from eval_extrude_v3 import orbit_cameras

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"


def load_any_mesh(path):
    if path.endswith(".obj"):
        try:
            return load_obj(path)
        except Exception:
            pass
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def similarity_from_bbox(src_v, dst_v):
    """Uniform scale + translation mapping src bbox -> dst bbox."""
    s_min, s_max = src_v.min(0), src_v.max(0)
    d_min, d_max = dst_v.min(0), dst_v.max(0)
    scale = ((d_max - d_min) / np.maximum(s_max - s_min, 1e-12)).mean()
    s_c, d_c = (s_min + s_max) / 2, (d_min + d_max) / 2
    return scale, d_c - scale * s_c


def exam16(ctx, v, t, shape):
    gt_v, gt_f = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj"))
    gt_v = normalize_to_range(gt_v)
    gvt = torch.tensor(gt_v, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gt_f, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5, azimuths_deg=azs, device=DEVICE)
    if isinstance(mv, tuple): mv = mv[0]
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(np.asarray(t, np.int32), dtype=torch.int32, device=DEVICE)
    gt_sils = render_views_n(ctx, gvt, gft, mv)
    pr_sils = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; maxblob = 0
    for i in range(16):
        g = gt_sils[i] > 0.5; p = pr_sils[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2); hair += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return inter / max(un, 1), hair, maxblob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--dmesh-gt", required=True)
    ap.add_argument("--shape", default="armadillo")
    ap.add_argument("--tag", default="dmesh")
    a = ap.parse_args()

    scene = setup_scene(a.shape, DEVICE)
    ctx, mvps, gt = scene["ctx"], scene["mvps"], scene["gt_uint8"]

    our_gt_v, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{a.shape}.obj"))
    our_gt_v = normalize_to_range(our_gt_v)
    dgt_v, _ = load_any_mesh(a.dmesh_gt)
    scale, off = similarity_from_bbox(dgt_v, our_gt_v)
    print(f"[align] scale={scale:.5f} offset={off}")

    pv, pf = load_any_mesh(a.pred)
    pv = pv * scale + off

    vt = torch.tensor(pv, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(pf, np.int32), dtype=torch.int32, device=DEVICE)
    train6 = compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)
    ho, hair, mb = exam16(ctx, pv, pf, a.shape)
    print(f"\n=== {a.tag.upper()} (aligned to our frame) ===")
    print(f"V={len(pv)} F={len(pf)}")
    print(f"train6 IoU={train6:.4f}")
    print(f"heldout16: IoU={ho:.4f} hair_px={hair} maxblob={mb}")

    outp = f"{OUT}/{a.tag}_aligned.obj"
    with open(outp, "w") as fh:
        for x, y, z in pv: fh.write(f"v {x} {y} {z}\n")
        for f3 in np.asarray(pf, np.int64): fh.write(f"f {f3[0]+1} {f3[1]+1} {f3[2]+1}\n")
    print(f"saved {outp}")


if __name__ == "__main__":
    main()
