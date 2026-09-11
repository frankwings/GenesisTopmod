import sys, os, glob, subprocess
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.cm as cm
from eval_local_refine import setup_scene, render_sil_and_depth, load_obj, normalize_to_range, BUNNY_PATH
from eval_extrude_v3 import orbit_cameras

SHAPE=os.environ.get("SHAPE","cow")
NPZ=os.environ.get("NPZ","/tmp/liou_cow_viz/cow_v22.npz")
RES=256; NF=48; DEV="cuda"
OUTDIR=f"/tmp/depth360_{SHAPE}"; os.makedirs(OUTDIR,exist_ok=True)
os.system(f"rm -f {OUTDIR}/f_*.png")

scene=setup_scene(SHAPE,DEV); ctx=scene["ctx"]
gv,gf=load_obj(os.path.join(os.path.dirname(BUNNY_PATH),f"{SHAPE}.obj")); gv=normalize_to_range(gv)
gvt=torch.tensor(gv,dtype=torch.float32,device=DEV); gft=torch.tensor(gf,dtype=torch.int32,device=DEV)
z=np.load(NPZ); pvt=torch.tensor(z["verts"],dtype=torch.float32,device=DEV); pft=torch.tensor(z["tris"],dtype=torch.int32,device=DEV)

azs=[360.0*i/NF for i in range(NF)]
mv=orbit_cameras(n=NF,elevation_deg=15.0,radius=2.5,azimuths_deg=azs,device=DEV)
if isinstance(mv,tuple): mv=mv[0]

@torch.no_grad()
def depth(v,f,m):
    sil,z,fg=render_sil_and_depth(ctx,v,f,m,(RES,RES))
    d=z.detach().cpu().numpy(); fgm=fg.detach().cpu().numpy().astype(bool)
    if fgm.ndim==3: fgm=fgm[0]
    if d.ndim==3: d=d[0]
    return d,fgm

# global depth range from GT across frames for consistent colormap
lo,hi=1e9,-1e9
gts=[]
for i in range(NF):
    d,m=depth(gvt,gft,mv[i]); gts.append((d,m))
    if m.any(): lo=min(lo,d[m].min()); hi=max(hi,d[m].max())
rng=max(hi-lo,1e-6)
import matplotlib.pyplot as _plt2; cmap=_plt2.colormaps["turbo"]
def colorize(d,m):
    img=np.ones((RES,RES,3))
    if m.any():
        nd=((d-lo)/rng).clip(0,1)
        img[m]=cmap(1.0-nd[m])[:,:3]  # near=warm
    return img

for i in range(NF):
    gd,gm=gts[i]; pd,pm=depth(pvt,pft,mv[i])
    fig,ax=plt.subplots(1,2,figsize=(8,4),dpi=100)
    ax[0].imshow(colorize(gd,gm)); ax[0].set_title(f"GT {SHAPE} depth",fontsize=11); ax[0].axis("off")
    ax[1].imshow(colorize(pd,pm)); ax[1].set_title("v22 result depth",fontsize=11); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/f_{i:03d}.png",bbox_inches="tight"); plt.close(fig)
print("frames done",flush=True)
mp4=f"{OUTDIR}/depth360_{SHAPE}.mp4"
subprocess.run(["ffmpeg","-y","-framerate","12","-i",f"{OUTDIR}/f_%03d.png",
    "-c:v","libx264","-pix_fmt","yuv420p","-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",mp4],
    check=True,capture_output=True)
print("MP4:",mp4,os.path.getsize(mp4)//1024,"KB")
