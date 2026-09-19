"""Validation spike for handle_guard: a hole-closing force collapses a genus-1 torus (genus 1->0); the loop-span guard
(keep each tree-cotree H1 generator from contracting) preserves it. Run: python3 despike/test_handle_guard.py"""
import sys, numpy as np, torch
sys.path.insert(0,"/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
from handle_guard import handles_guard_loss, axis_throat_target, genus_from_counts
from scipy import ndimage; from skimage import measure
dev="cuda" if torch.cuda.is_available() else "cpu"
nu,nv=96,48; R,r=1.0,0.35
th=np.linspace(0,2*np.pi,nu,endpoint=False); ph=np.linspace(0,2*np.pi,nv,endpoint=False); T,P=np.meshgrid(th,ph,indexing="ij")
V0=np.stack([(R+r*np.cos(P))*np.cos(T),(R+r*np.cos(P))*np.sin(T),r*np.sin(P)],-1).reshape(-1,3)
vid=lambda i,j:(i%nu)*nv+(j%nv); F=np.array([t for i in range(nu) for j in range(nv) for t in ([vid(i,j),vid(i+1,j),vid(i+1,j+1)],[vid(i,j),vid(i+1,j+1),vid(i,j+1)])])
def genus_vox(Vnp,N=110):
    lo=Vnp.min(0)-0.12; hi=Vnp.max(0)+0.12; sp=hi-lo; g=np.zeros((N,N,N),bool)
    for tri in F:
        p=Vnp[tri]
        for a in range(9):
            for b in range(9-a):
                q=p[0]+a/8*(p[1]-p[0])+b/8*(p[2]-p[0]); ii=np.clip(((q-lo)/sp*(N-1)).astype(int),0,N-1); g[ii[0],ii[1],ii[2]]=True
    return int(1-measure.euler_number(ndimage.binary_fill_holes(ndimage.binary_closing(g,iterations=2)),connectivity=1))
cen=torch.tensor(V0.mean(0),dtype=torch.float32,device=dev)
def run(guard, steps=500, lr=5e-3, keep=0.8):
    V=torch.tensor(V0,dtype=torch.float32,device=dev,requires_grad=True); opt=torch.optim.Adam([V],lr=lr)
    a0=torch.zeros(3,device=dev); u=torch.tensor([0,0,1.0],device=dev); axes=[(a0,u)]; targets=[axis_throat_target(V.detach(),a0,u,keep)]
    for s in range(steps):
        opt.zero_grad(); loss=((V-cen)**2).sum(-1).mean()          # collapse force: pull to centroid
        if guard: loss=loss+50.0*handles_guard_loss(V,axes,targets)
        loss.backward(); opt.step()
    return genus_vox(V.detach().cpu().numpy())
print("initial torus voxel genus:",genus_vox(V0))
gn=run(False); gy=run(True)
print(f"NO guard : genus {gn}\nWITH guard: genus {gy}")
print(">>> GUARD WORKS" if gn<1<=gy else ">>> inconclusive")
