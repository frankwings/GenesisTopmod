"""What TopMod add_handle does, shown on a donut whose hole is covered by a thin membrane.
Builds the genus-0 membrane donut with topmod.primitives._build_mesh, calls the REAL
topmod.high_level_ops.add_handle on the two membrane-centre faces (genus 0 -> 1), and renders
before / cutaway / after / cutaway-zoom / final-torus panels. Run from despike/: python3 demo_add_handle_donut.py"""
import sys; import os; sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))
import numpy as np, torch
import nvdiffrast.torch as dr
from topmod.primitives import _build_mesh
from topmod.high_level_ops import add_handle
from topmod.diffgeo import mesh_to_arrays, _fan_triangulate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- a donut whose hole is covered by a thin membrane (surface of revolution, genus 0) ----
N=20; Rc,a,tm=1.0,0.36,0.085
def half_t(r):
    tor=np.sqrt(np.clip(a*a-(r-Rc)**2,0,None))
    k=40.0; return np.log(np.exp(k*tor)+np.exp(k*tm))/k          # smooth max(torus, membrane)
rs=np.linspace(0.16,Rc+a-0.02,22)
prof=[(r,half_t(r)) for r in rs]+[(Rc+a,0.0)]+[(r,-half_t(r)) for r in rs[::-1]]
P=[];F=[]
for (r,z) in prof:
    for j in range(N):
        th=2*np.pi*j/N; P.append((r*np.cos(th),r*np.sin(th),z))
nr=len(prof)
for i in range(nr-1):
    for j in range(N):
        a0=i*N+j; a1=i*N+(j+1)%N; b0=(i+1)*N+j; b1=(i+1)*N+(j+1)%N
        F.append([a0,b0,b1,a1])
F.append([j for j in range(N)])                       # top membrane centre N-gon  (f1)
F.append([(nr-1)*N+j for j in range(N)][::-1])       # bottom membrane centre N-gon (f2)
mesh=_build_mesh([tuple(map(float,p)) for p in P],F)
g0=mesh.genus()
def arrays(m):
    V,Fp=mesh_to_arrays(m); return np.asarray(V,float),[list(f) for f in Fp]
V0,F0=arrays(mesh)
fcs=list(mesh.iter_faces())
def fc(f): return np.mean([(h.origin.x,h.origin.y,h.origin.z) for h in f.halfedges()],0)
f1=max([f for f in fcs if sum(1 for _ in f.halfedges())==N],key=lambda f:fc(f)[2])
f2=min([f for f in fcs if sum(1 for _ in f.halfedges())==N],key=lambda f:fc(f)[2])
f1v=np.array([(h.origin.x,h.origin.y,h.origin.z) for h in f1.halfedges()]); f2v=np.array([(h.origin.x,h.origin.y,h.origin.z) for h in f2.halfedges()])
add_handle(mesh,f1,f2); g1=mesh.genus()
V1,F1=arrays(mesh)
old={tuple(sorted(f)) for f in F0}; tube={tuple(sorted(f)) for f in F1 if tuple(sorted(f)) not in old}
print(f"genus {g0} -> {g1}; tube quads {len(tube)}")

# ---- pipeline result: membrane absorbed -> a plain torus (illustration of what DLFL collapse achieves) ----
M=48; K=24; VT=[];FT=[]
for i in range(M):
    for j in range(K):
        u=2*np.pi*i/M; v=2*np.pi*j/K
        VT.append(((Rc+a*np.cos(v))*np.cos(u),(Rc+a*np.cos(v))*np.sin(u),a*np.sin(v)))
for i in range(M):
    for j in range(K):
        FT.append([i*K+j,((i+1)%M)*K+j,((i+1)%M)*K+(j+1)%K,i*K+(j+1)%K])
VT=np.array(VT)

# ---- renderer (nvdiffrast, per-face colour, optional cutaway) ----
ctx=dr.RasterizeCudaContext(); RES=900
GREY=np.array([0.90,0.89,0.87]); RED=np.array([0.89,0.29,0.28]); BLUE=np.array([0.16,0.47,0.84]); ORANGE=np.array([0.92,0.41,0.20])
def render(V,Fp,color_of,cut=False,az=-35,el=32,edges=True,zoom=1.55):
    keep=[f for f in Fp if not (cut and V[f].mean(0)[1]>1e-6)]
    tris=[];tcol=[]
    for f in keep:
        c=color_of(f)
        for t in _fan_triangulate([f]): tris.append(t); tcol.append(c)
    T=np.array(tris); C=np.array(tcol)
    a_,e_=np.radians(az),np.radians(-el)
    Rz=np.array([[np.cos(a_),-np.sin(a_),0],[np.sin(a_),np.cos(a_),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(e_),-np.sin(e_)],[0,np.sin(e_),np.cos(e_)]])
    W=(V@Rz.T)                                   # spin about z
    W=np.stack([W[:,0],W[:,2],-W[:,1]],1)        # z-up -> screen-y up, depth = -y
    W=W@Rx.T; s=1.0/zoom
    pos=np.concatenate([W[:,:2]*s,(W[:,2:3]*s*0.4+0.5),np.ones((len(V),1))],1).astype(np.float32)
    rast,_=dr.rasterize(ctx,torch.tensor(pos[None],device="cuda"),torch.tensor(T.astype(np.int32),device="cuda"),resolution=[RES,RES])
    tri=W[T]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-12
    L=np.array([0.35,0.6,0.72]); L/=np.linalg.norm(L)
    sh=(np.abs(n@L)*0.7+0.3)[:,None]*C
    fid=rast[0,...,3].long().cpu().numpy(); img=np.ones((RES,RES,3)); m=fid>0; img[m]=sh[fid[m]-1]
    return np.flipud(img)
def col0(f):
    s=tuple(sorted(f))
    if len(f)==N and V0[f].mean(0)[2]>0: return RED
    if len(f)==N and V0[f].mean(0)[2]<0: return BLUE
    return GREY
col1=lambda f: ORANGE if tuple(sorted(f)) in tube else GREY
panels=[(render(V0,F0,col0),f"1) BEFORE (genus {g0}): a donut whose hole\nis covered by a THIN MEMBRANE\nred = f1 (top centre face)"),
        (render(V0,F0,col0,cut=True,az=0,el=22),"2) cut in half: thick rim + thin membrane\nred f1 = top page, blue f2 = bottom page\n(the membrane = material inside the hole)"),
        (render(V1,F1,col1),f"3) AFTER add_handle(f1,f2) (genus {g1})\nf1, f2 deleted -> a small hole\nthrough the membrane"),
        (render(V1,F1,col1,cut=True,az=0,el=22,zoom=0.55),"4) cut in half, ZOOMED on the centre:\norange = the N new side quads = TUBE WALL\njoining rim(f1) to rim(f2); inside = AIR"),
        (render(VT,FT,lambda f:GREY),"5) pipeline only: DLFL collapse absorbs\nthe leftover membrane -> hole grows to the\ndonut's inner rim = the real tunnel")]
fig,axs=plt.subplots(1,5,figsize=(25,6.4))
for ax,(img,t) in zip(axs,panels):
    ax.imshow(img); ax.axis("off"); ax.set_title(t,fontsize=12.5,loc="left")
plt.tight_layout(); plt.savefig("results_genus/fig_add_handle_donut.png",dpi=100,bbox_inches="tight"); print("saved")
