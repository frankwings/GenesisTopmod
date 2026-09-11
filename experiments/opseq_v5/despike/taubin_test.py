import sys, os
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from eval_local_refine import setup_scene, render_views_n, compute_iou_n

z=np.load("/tmp/liou_cow_viz/cow_v22.npz"); V=z["verts"].astype(np.float64); F=z["tris"].astype(np.int64)
scene=setup_scene("cow","cuda"); ctx=scene["ctx"]; gt=scene["gt_uint8"]; mvps=scene["mvps"]
def iou(v):
    vt=torch.tensor(v,dtype=torch.float32,device="cuda"); ft=torch.tensor(F,dtype=torch.int32,device="cuda")
    return compute_iou_n(render_views_n(ctx,vt,ft,mvps),gt)

# build 1-hop neighbor lists
nb=collections.defaultdict(set)
for a,b,c in F:
    a,b,c=int(a),int(b),int(c)
    nb[a]|={b,c}; nb[b]|={a,c}; nb[c]|={a,b}
n=len(V); idx=[np.array(sorted(nb[i])) for i in range(n)]
def umbrella(v):
    L=np.zeros_like(v)
    for i in range(n):
        if len(idx[i]): L[i]=v[idx[i]].mean(0)-v[i]
    return L
def taubin(v,it,lam=0.5,mu=-0.53):
    v=v.copy()
    for _ in range(it):
        v=v+lam*umbrella(v)
        v=v+mu*umbrella(v)
    return v

variants={"v22 raw":V}
for it in (5,10,20):
    variants[f"Taubin-{it}"]=taubin(V,it)
for k,v in variants.items():
    print(f"{k:12s} train6_IoU={iou(v):.4f}")

# Lambertian shaded turntable
def face_normals(v,f):
    tri=v[f]; nrm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    ln=np.linalg.norm(nrm,axis=1,keepdims=True); return nrm/np.clip(ln,1e-9,None)
light=np.array([0.3,0.5,0.8]); light=light/np.linalg.norm(light)
angles=[0,60,120,220]
cols=list(variants.items())
fig=plt.figure(figsize=(4*len(cols),4*len(angles)),dpi=100)
for r,az in enumerate(angles):
    for c,(name,v) in enumerate(cols):
        ax=fig.add_subplot(len(angles),len(cols),r*len(cols)+c+1,projection="3d")
        fn=face_normals(v,F); sh=np.clip(fn@light,0.05,1.0)
        colarr=np.stack([sh*0.55,sh*0.78,sh*0.42],1)  # shaded green
        pc=Poly3DCollection(v[F],facecolors=colarr,edgecolor="none")
        ax.add_collection3d(pc)
        ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1);ax.set_box_aspect((2,2,2))
        ax.view_init(elev=15,azim=az);ax.axis("off")
        if r==0: ax.set_title(name,fontsize=11)
plt.tight_layout(); plt.savefig("/tmp/liou_cow_viz/cow_v22_smooth.png",bbox_inches="tight")
print("saved cow_v22_smooth.png")
