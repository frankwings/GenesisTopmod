import sys, numpy as np, pycolmap, cv2
from skimage import measure; from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
O=sys.argv[1]; rec=pycolmap.Reconstruction(f"{O}/sparse/0"); cam=rec.cameras[1]; f,cx,cy,k=cam.params
ims=sorted(rec.images.values(),key=lambda im:im.name); names=[im.name for im in ims]; n=len(ims)
Rcw=[im.cam_from_world().rotation.matrix() for im in ims]; tcw=[im.cam_from_world().translation for im in ims]
C=np.array([-R.T@t for R,t in zip(Rcw,tcw)]); masks=[cv2.imread(f"{O}/masks_sam2/{nm[:-4]}.png",0)>127 for nm in names]
val=[i for i in range(n) if masks[i].sum()>=200]
D=[]
for i in val: ys,xs=np.nonzero(masks[i]); d=np.array([(xs.mean()-cx)/f,(ys.mean()-cy)/f,1.0]);d/=np.linalg.norm(d);D.append(Rcw[i].T@d)
Cv=C[val]; A=np.zeros((3,3));b=np.zeros(3)
for c,d in zip(Cv,D):P=np.eye(3)-np.outer(d,d);A+=P;b+=P@c
X=np.linalg.solve(A,b); rmed=np.median(np.linalg.norm(Cv-X,axis=1))
miss=np.array([np.linalg.norm(np.cross(c-X,d)) for c,d in zip(Cv,D)])
# camera azimuth coverage
v=Cv-X; u,s,vt=np.linalg.svd(v-v.mean(0)); az=np.sort(np.degrees(np.arctan2(v@vt[1],v@vt[0]))%360); gaps=np.diff(np.r_[az,az[0]+360])
print(f"[{O}] {len(val)}/{n} valid masks | ray-miss med {np.median(miss):.4f}m p90 {np.percentile(miss,90):.4f}m | orbit {360-gaps.max():.0f}deg")
N=192; half=0.5*rmed; g=np.linspace(-half,half,N); G=np.stack(np.meshgrid(g,g,g,indexing='ij'),-1).reshape(-1,3)+X
votes=np.zeros(len(G),np.int32)
for i in val:
    m=masks[i]; p=G@Rcw[i].T+tcw[i]; z=p[:,2]; u2=f*p[:,0]/z+cx; v2=f*p[:,1]/z+cy; ok=(z>0)&(u2>=0)&(u2<m.shape[1]-1)&(v2>=0)&(v2<m.shape[0]-1)
    ins=np.zeros(len(G),bool); ins[ok]=m[v2[ok].astype(int),u2[ok].astype(int)]; votes+=ins
nv=len(val)
for slk in (2,4,8):
    h=(votes>=nv-slk).reshape(N,N,N); lab,nc=ndimage.label(h)
    if h.sum()>0: h=lab==(1+np.argmax(np.bincount(lab.ravel())[1:]))
    gh=1-measure.euler_number(ndimage.binary_fill_holes(h),connectivity=1); print(f"  slack {slk}: {int(h.sum())} vox genus {gh} comps {nc}")
h=(votes>=nv-3).reshape(N,N,N); lab,nc=ndimage.label(h)
if h.sum()>50: h=lab==(1+np.argmax(np.bincount(lab.ravel())[1:]))
gh=1-measure.euler_number(ndimage.binary_fill_holes(h),connectivity=1)
V,F,_,_=measure.marching_cubes(h.astype(np.float32),0.5,spacing=(2*half/(N-1),)*3)
fig=plt.figure(figsize=(20,5))
for i,az_ in enumerate([0,90,180,270]):
    ax=fig.add_subplot(1,4,i+1,projection='3d'); Vc=V-V.mean(0)
    nrm=np.cross(Vc[F[:,1]]-Vc[F[:,0]],Vc[F[:,2]]-Vc[F[:,0]]); nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)+1e-9
    sh=0.4+0.6*np.clip(nrm@[0.3,-0.7,0.6],0,1)
    pc=Poly3DCollection(Vc[F]); pc.set_facecolor(np.stack([0.55*sh,0.62*sh,0.72*sh,np.ones_like(sh)],1)); ax.add_collection3d(pc)
    R_=np.abs(Vc).max(); ax.set_xlim(-R_,R_);ax.set_ylim(-R_,R_);ax.set_zlim(-R_,R_); ax.view_init(12,az_); ax.set_axis_off()
fig.suptitle(f"{O} hull (ARKit+SAM2video) | genus {gh} | ray-miss {np.median(miss)*1000:.0f}mm | orbit {360-gaps.max():.0f}deg")
fig.savefig(f"{O}/hull.png",dpi=95,bbox_inches='tight'); print(f"  saved {O}/hull.png genus {gh}")
