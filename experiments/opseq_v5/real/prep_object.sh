#!/bin/bash
# prep_object.sh <arkit_id> <out> : frames + ARKit poses + rembg + SAM2-video masks
set -e; SRC=/home/kingy/Projects/Genesis/benchmarks/Iphone16Data/$1; O=$2
mkdir -p $O/images
python3 - "$SRC" "$O" <<'EOF'
import cv2,numpy as np,sys
src,out=sys.argv[1],sys.argv[2]; cap=cv2.VideoCapture(f"{src}/rgb.mp4"); n=int(cap.get(7)); stride=max(1,n//140)
sh=[]
while True:
    ok,fr=cap.read()
    if not ok:break
    g=cv2.cvtColor(cv2.resize(fr,None,fx=.25,fy=.25),cv2.COLOR_BGR2GRAY); sh.append(cv2.Laplacian(g,cv2.CV_64F).var())
sh=np.array(sh);N=len(sh);picks=[int(s+np.argmax(sh[s:s+min(8,stride)])) for s in range(0,N-stride+1,stride)]
cap=cv2.VideoCapture(f"{src}/rgb.mp4");i=0;k=0;keep=set(picks);m={}
while True:
    ok,fr=cap.read()
    if not ok:break
    if i in keep: cv2.imwrite(f"{out}/images/f_{k:03d}.jpg",fr,[cv2.IMWRITE_JPEG_QUALITY,95]);m[k]=i;k+=1
    i+=1
np.save(f"{out}/frame_map.npy",np.array([[a,b] for a,b in m.items()])); print(f"{N}f->{k}frames")
EOF
python3 prep_arkit.py $SRC $O 2>&1 | tail -1
python3 - "$O" <<'EOF'
from rembg import remove,new_session; from PIL import Image; import numpy as np,cv2,glob,os,sys
O=sys.argv[1]; os.makedirs(f"{O}/masks",exist_ok=True); s=new_session("isnet-general-use")
for f in sorted(glob.glob(f"{O}/images/f_*.jpg")):
    nm=os.path.basename(f)[:-4]; a=np.array(remove(Image.open(f),session=s,only_mask=True)); m=(a>127).astype(np.uint8)
    nl,lab,st,_=cv2.connectedComponentsWithStats(m)
    if nl>2: m=(lab==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    cv2.imwrite(f"{O}/masks/{nm}.png",m*255)
print("rembg",len(glob.glob(f"{O}/masks/*.png")))
EOF
python3 sam2_video_masks.py $O --click-frame 0 2>&1 | grep "\[sam2-video\]"
