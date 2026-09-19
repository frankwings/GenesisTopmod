import sys, numpy as np, torch
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
from handle_guard import throat_openness_loss, axis_radius
from scipy import ndimage; from skimage import measure
dev="cuda" if torch.cuda.is_available() else "cpu"
nu,nv=96,48; R,r=1.0,0.35    # hole radius R-r=0.65
th=np.linspace(0,2*np.pi,nu,endpoint=False); ph=np.linspace(0,2*np.pi,nv,endpoint=False)
T,P=np.meshgrid(th,ph,indexing="ij")
V0=np.stack([(R+r*np.cos(P))*np.cos(T),(R+r*np.cos(P))*np.sin(T),r*np.sin(P)],-1).reshape(-1,3)
def vid(i,j): return (i%nu)*nv+(j%nv)
F=np.array([t for i in range(nu) for j in range(nv) for t in ([vid(i,j),vid(i+1,j),vid(i+1,j+1)],[vid(i,j),vid(i+1,j+1),vid(i,j+1)])])
def genus_vox(Vnp,Fnp,N=110):
    lo=Vnp.min(0)-0.12; hi=Vnp.max(0)+0.12; sp=(hi-lo); g=np.zeros((N,N,N),bool)
    for tri in Fnp:
        p=Vnp[tri]; n=8
        for a in range(n+1):
            for b in range(n+1-a):
                q=p[0]+a/n*(p[1]-p[0])+b/n*(p[2]-p[0]); ii=np.clip(((q-lo)/sp*(N-1)).astype(int),0,N-1); g[ii[0],ii[1],ii[2]]=True
    g=ndimage.binary_closing(g,iterations=2); fill=ndimage.binary_fill_holes(g)
    return int(1-measure.euler_number(fill,connectivity=1))
print("initial torus voxel genus:",genus_vox(V0,F),"| hole radius",round(R-r,3))
a0=torch.zeros(3,device=dev); u=torch.tensor([0,0,1.0],device=dev); tube_idx=torch.arange(len(V0),device=dev)
cen=torch.tensor(V0.mean(0),dtype=torch.float32,device=dev)
def run(guard, steps=500, lr=5e-3, target=0.45):
    V=torch.tensor(V0,dtype=torch.float32,device=dev,requires_grad=True); opt=torch.optim.Adam([V],lr=lr)
    for s in range(steps):
        opt.zero_grad()
        shrink=((V-cen)**2).sum(-1).mean()             # collapse force: pull to centroid -> closes the hole
        loss=shrink
        if guard:
            gl,_=throat_openness_loss(V,a0,u,tube_idx,target); loss=loss+50.0*gl
        loss.backward(); opt.step()
    hole=float(axis_radius(V.detach(),a0,u).min()); return hole,genus_vox(V.detach().cpu().numpy(),F)
hn,gn=run(False); hy,gy=run(True)
print(f"NO guard : hole {hn:.3f} genus {gn}")
print(f"WITH guard: hole {hy:.3f} genus {gy}")
print(">>> GUARD WORKS: handle collapsed without it, survived with it" if gn<1<=gy else ">>> inconclusive")
