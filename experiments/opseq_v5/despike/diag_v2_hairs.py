"""Attribute remaining hair pixels of cow_c2f_v2.obj: real protrusions vs boundary noise."""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import nvdiffrast.torch as dr
from scipy.ndimage import binary_dilation, label
from eval_local_refine import (setup_scene, render_views_n, IMG_RES, N_VIEWS)

DEVICE = "cuda"
scene = setup_scene("cow", DEVICE)
ctx, mvps, gt_uint8 = scene["ctx"], scene["mvps"], scene["gt_uint8"]

# load mesh
vs, fs = [], []
for line in open("/tmp/liou_cow_viz/cow_c2f_v2.obj"):
    p = line.split()
    if not p: continue
    if p[0] == "v": vs.append([float(x) for x in p[1:4]])
    elif p[0] == "f": fs.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
v = torch.tensor(np.array(vs), dtype=torch.float32, device=DEVICE)
f = torch.tensor(np.array(fs), dtype=torch.int32, device=DEVICE)

vh = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
face_hits = {}
per_view = []
blob_sizes_all = []
for i in range(N_VIEWS):
    clip = (vh @ mvps[i].T).unsqueeze(0).contiguous()
    rast, _ = dr.rasterize(ctx, clip, f, (IMG_RES, IMG_RES))
    fid = rast[0, :, :, 3].long().cpu().numpy()
    gt = gt_uint8[i] < 128
    for dil in (2, 3, 4, 6):
        n_out = int(((fid > 0) & ~binary_dilation(gt, iterations=dil)).sum())
        if dil == 2: n2 = n_out
        if dil == 4: n4 = n_out
        if dil == 6: n6 = n_out
    out2 = (fid > 0) & ~binary_dilation(gt, iterations=2)
    lab, nblob = label(out2)
    sizes = [int((lab == k).sum()) for k in range(1, nblob + 1)]
    blob_sizes_all += sizes
    for fi in np.unique(fid[out2]):
        if fi > 0: face_hits[int(fi - 1)] = face_hits.get(int(fi - 1), 0) + 1
    per_view.append((n2, n4, n6, nblob, max(sizes) if sizes else 0))

print("view | out@dil2 out@dil4 out@dil6 | blobs maxblob")
for i, (a, b4, b6, nb, mx) in enumerate(per_view):
    print(f"  {i}  |   {a:4d}    {b4:4d}     {b6:4d}   |  {nb:3d}   {mx:4d}")
print(f"\ntotal out@2={sum(p[0] for p in per_view)} @4={sum(p[1] for p in per_view)} @6={sum(p[2] for p in per_view)}")
bs = sorted(blob_sizes_all, reverse=True)
print(f"blob size dist (top15): {bs[:15]}")
print(f"blobs<=3px: {sum(1 for s in bs if s<=3)}/{len(bs)}  px in blobs<=3: {sum(s for s in bs if s<=3)}")
print(f"distinct hair faces: {len(face_hits)}")
hf = sorted(face_hits.items(), key=lambda x: -x[1])[:10]
print(f"top faces: {hf}")
