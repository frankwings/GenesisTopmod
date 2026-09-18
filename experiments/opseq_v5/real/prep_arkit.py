#!/usr/bin/env python3
"""prep_arkit.py <arkit_src_dir> <dataset_dir> : write a COLMAP text model from ARKit (StrayScanner) poses so the
existing real_scene.py (COLMAP + masks) reads iPhone data unchanged. Poses are metric and reliable — no SfM needed.
odometry.csv rows are per-frame (row i = video frame i), c2w in OpenCV convention (verified in convert_stray_to_ns.py)."""
import sys, os, numpy as np, csv
src, out = sys.argv[1], sys.argv[2]
fm = np.load(f"{out}/frame_map.npy")            # [[our_k, orig_frame_idx], ...]
K = np.loadtxt(f"{src}/camera_matrix.csv", delimiter=","); fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
import cv2; im0 = cv2.imread(f"{out}/images/f_000.jpg"); H, W = im0.shape[:2]
# odometry rows
odom = {}
with open(f"{src}/odometry.csv") as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        i = int(float(row[1])); x,y,z = map(float,row[2:5]); qx,qy,qz,qw = map(float,row[5:9]); odom[i]=(x,y,z,qx,qy,qz,qw)
def quat_to_R(qx,qy,qz,qw):
    n=(qx*qx+qy*qy+qz*qz+qw*qw)**.5; qx,qy,qz,qw=qx/n,qy/n,qz/n,qw/n
    return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
def R_to_quat(R):  # returns qw,qx,qy,qz (COLMAP order)
    t=np.trace(R)
    if t>0: s=.5/np.sqrt(t+1); qw=.25/s; qx=(R[2,1]-R[1,2])*s; qy=(R[0,2]-R[2,0])*s; qz=(R[1,0]-R[0,1])*s
    else:
        i=np.argmax([R[0,0],R[1,1],R[2,2]])
        if i==0: s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); qw=(R[2,1]-R[1,2])/s; qx=.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
        elif i==1: s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); qw=(R[0,2]-R[2,0])/s; qx=(R[0,1]+R[1,0])/s; qy=.25*s; qz=(R[1,2]+R[2,1])/s
        else: s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); qw=(R[1,0]-R[0,1])/s; qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=.25*s
    return qw,qx,qy,qz
os.makedirs(f"{out}/sparse/0", exist_ok=True)
with open(f"{out}/sparse/0/cameras.txt","w") as fh:
    fh.write(f"1 SIMPLE_RADIAL {W} {H} {(fx+fy)/2} {cx} {cy} 0.0\n")
lines=[]; nok=0
for k, orig in fm:
    if orig not in odom: continue
    x,y,z,qx,qy,qz,qw = odom[orig]; Rc2w = quat_to_R(qx,qy,qz,qw); tc2w=np.array([x,y,z])
    Rw2c = Rc2w.T; tw2c = -Rw2c@tc2w; qwv,qxv,qyv,qzv = R_to_quat(Rw2c)
    lines.append(f"{int(k)+1} {qwv} {qxv} {qyv} {qzv} {tw2c[0]} {tw2c[1]} {tw2c[2]} 1 f_{int(k):03d}.jpg\n\n"); nok+=1
with open(f"{out}/sparse/0/images.txt","w") as fh: fh.writelines(lines)
open(f"{out}/sparse/0/points3D.txt","w").close()
print(f"[prep_arkit] wrote COLMAP text model: {nok} images, {W}x{H}, f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f})")
