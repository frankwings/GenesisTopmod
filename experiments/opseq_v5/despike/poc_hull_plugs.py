import os, sys, numpy as np, time
os.environ.setdefault("SHAPE","fertility"); os.environ.setdefault("BASE_NPZ","/tmp/liou_cow_viz/cow_fertility_fertility_v5_cc3p4.npz")
src=open("phase7_handle.py").read(); cut=src.index("pitch = HF.pitch"); exec(compile(src[:cut+len("pitch = HF.pitch")],"p7","exec"))
from skimage import measure, morphology, segmentation, feature
from scipy import ndimage
def genus_solid(vol):  # cavity-corrected: fill cavities first, then chi = 1 - g for one component
    vol = ndimage.binary_fill_holes(vol); return 1 - measure.euler_number(vol, connectivity=1)
def clean(h, r=2):
    b = morphology.ball(r); h = morphology.binary_opening(morphology.binary_closing(h, b), b)
    lab, n = ndimage.label(h)
    if n > 1: h = lab == (np.bincount(lab.ravel())[1:].argmax()+1)
    return ndimage.binary_fill_holes(h)
t=time.time(); hc = clean(HF.hull); N=hc.shape[0]
hs = hc.reshape(128,2,128,2,128,2).max(axis=(1,3,5)); g0=genus_solid(hs); print("128^3 genus", g0)
edt_bg = ndimage.distance_transform_edt(~hs)
def plugs_at(R, base):
    dil = edt_bg <= R; cl = ndimage.distance_transform_edt(dil) > R
    return cl & ~base
found=[]; claimed=np.zeros_like(hs)
for R in range(6, 40, 2):
    cur = hs | claimed; gcur = genus_solid(cur)
    if gcur == 0: break
    D = plugs_at(R, hs) & ~claimed
    lab, n = ndimage.label(D, structure=np.ones((3,3,3))); sizes=np.bincount(lab.ravel())[1:]
    for c in np.argsort(-sizes)[:8]:
        if sizes[c] < 50: break
        comp = lab==(c+1); dg = genus_solid(cur | comp) - gcur
        if dg == 0: continue
        if dg == -1:
            pieces=[comp]
        else:
            # split merged plug: watershed on the distance-to-boundary inside comp, markers = peaks
            edt = ndimage.distance_transform_edt(comp)
            pk = feature.peak_local_max(edt, min_distance=6, labels=comp.astype(int), exclude_border=False)
            mk = np.zeros(comp.shape, int); mk[tuple(pk.T)] = np.arange(1, len(pk)+1)
            ws = segmentation.watershed(-edt, mk, mask=comp)
            pieces=[ws==k for k in range(1, len(pk)+1)]
            print(f"  R={R} merged comp {sizes[c]}vox dg={dg}: watershed -> {len(pieces)} pieces")
        for p in pieces:
            dgp = genus_solid(cur | claimed | p) - genus_solid(cur | claimed)
            if dgp == -1:
                pts=np.array(np.nonzero(p)).T; ev,evec=np.linalg.eigh(np.cov(pts.T))
                found.append((R, int(p.sum()), pts.mean(0), evec[:,0])); claimed |= p
                print(f"  R={R:2d} PLUG size={int(p.sum()):6d} centroid={np.round(pts.mean(0),1)} extents={np.round(np.sqrt(np.maximum(ev,0)),1)} axis={np.round(evec[:,0],2)}  genus now {genus_solid(hs|claimed)}")
    print(f"R={R}: genus remaining {genus_solid(hs|claimed)} ({time.time()-t:.0f}s)")
print("plugs", len(found), "target", g0, f"{time.time()-t:.0f}s")
print("=== finer ladder for the last tunnel (after claiming first 3) ===")
claimed3 = np.zeros_like(hs)
for R,s,c,a in found[:3]:
    pass
# rebuild claimed from the first three plugs by re-running the same logic is messy; instead test rays for all 4
print("=== rays from plug centroid along +-axis onto the coarse mesh ===")
import open3d as o3d
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(Fa.astype(np.int32))))
comp_, chi_ = {}, {}
for R,s,cen_vox,axis in found:
    cen_w = HF.lo + cen_vox/(128-1)*(HF.hi-HF.lo)
    hits=[]
    for sgn in (+1,-1):
        d = sgn*axis/np.linalg.norm(axis)
        # start far outside along -d so the ray crosses the whole membrane region: entry = first hit going +d
        p0 = cen_w
        ray = o3d.core.Tensor(np.concatenate([p0, d])[None].astype(np.float32)); h = sc.cast_rays(ray)
        t_ = float(h["t_hit"].numpy()[0]); fi = int(h["primitive_ids"].numpy()[0]) if np.isfinite(t_) else -1
        hits.append((fi, t_))
    fi, fj = hits[0][0], hits[1][0]
    ok = fi>=0 and fj>=0 and fi!=fj and not (set(Fa[fi]) & set(Fa[fj]))
    disks = "n/a"
    sep = np.linalg.norm(V[Fa[fi]].mean(0)-V[Fa[fj]].mean(0)) if ok else -1
    print(f"  plug R={R} size={s}: faces {fi},{fj} nonadjacent={ok} membrane_disks={disks} sep={sep:.3f}")
