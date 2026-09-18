#!/usr/bin/env python3
"""sam2_video_masks.py <dataset_dir> [--click-frame 0]: temporally-consistent masks via SAM2 video predictor.
Anchor = rembg-mask centroid of the click frame (+ a handle click) as positive points; propagate through all frames.
Writes <dir>/masks_sam2/f_XXX.png (overwrites the per-frame ones)."""
import sys, os, glob, shutil, numpy as np, cv2, torch
sys.argv0=sys.argv
D=sys.argv[1]; CLICK=int(sys.argv[sys.argv.index("--click-frame")+1]) if "--click-frame" in sys.argv else 0
from sam2.build_sam import build_sam2_video_predictor
cfg="configs/sam2.1/sam2.1_hiera_l.yaml"; ckpt=os.path.expanduser("~/.cache/sam2/sam2.1_hiera_large.pt")
pred=build_sam2_video_predictor(cfg, ckpt, device="cuda")
# SAM2 video wants a dir of frames named <int>.jpg
names=sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{D}/images/f_*.jpg"))
tmp=f"{D}/_sam2frames"; shutil.rmtree(tmp,ignore_errors=True); os.makedirs(tmp)
for i,nm in enumerate(names): shutil.copy(f"{D}/images/{nm}.jpg", f"{tmp}/{i}.jpg")
# anchor points from rembg mask of the click frame
m0=cv2.imread(f"{D}/masks/{names[CLICK]}.png",0)>127; ys,xs=np.nonzero(m0)
# positive: centroid + 3 distance-transform peaks (spread over body+handle); negative: 4 points just outside bbox
dt=cv2.distanceTransform(m0.astype(np.uint8),cv2.DIST_L2,5); pts=[[xs.mean(),ys.mean()]]
for _ in range(4):
    iy,ix=np.unravel_index(np.argmax(dt),dt.shape); pts.append([float(ix),float(iy)]); cv2.circle(dt,(int(ix),int(iy)),int(max(np.ptp(xs),np.ptp(ys))*0.12),0,-1)
x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max(); neg=[[x0-15,y0-15],[x1+15,y0-15],[x0-15,y1+15],[x1+15,y1+15]]
P=np.array(pts+neg,np.float32); L=np.array([1]*len(pts)+[0]*len(neg),np.int32)
with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
    state=pred.init_state(video_path=tmp)
    pred.add_new_points_or_box(state,frame_idx=CLICK,obj_id=1,points=P,labels=L)
    out={}
    for fi,oids,logits in pred.propagate_in_video(state): out[fi]=(logits[0]>0).cpu().numpy()[0]
    if CLICK>0:
        for fi,oids,logits in pred.propagate_in_video(state,reverse=True): out[fi]=(logits[0]>0).cpu().numpy()[0]
os.makedirs(f"{D}/masks_sam2",exist_ok=True); ious=[]
for i,nm in enumerate(names):
    m=out.get(i,np.zeros_like(m0))
    nl,lab,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if nl>2: m=lab==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))
    ff=m.astype(np.uint8).copy(); cv2.floodFill(ff,None,(0,0),1); m=(m|(ff==0))
    cv2.imwrite(f"{D}/masks_sam2/{nm}.png",m.astype(np.uint8)*255)
    r=cv2.imread(f"{D}/masks/{nm}.png",0)>127; ious.append((m&r).sum()/max((m|r).sum(),1))
shutil.rmtree(tmp,ignore_errors=True)
print(f"[sam2-video] {len(names)} masks | IoU vs rembg min/med {min(ious):.3f}/{np.median(ious):.3f} | fg frac med {np.median([ (cv2.imread(f'{D}/masks_sam2/{nm}.png',0)>127).mean() for nm in names]):.3f}")
