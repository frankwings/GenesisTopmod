import os,sys,re,json,glob,numpy as np
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5"); sys.path.insert(0,"despike"); sys.path.insert(0,"."); os.environ["SHAPE"]="fertility"
from viz_render import load
import open3d as o3d
R="despike/results_genus"; LOG="/home/kingy/Foundation/EdenGateway/agents/hani/bg_tasks/logs/v66_fert.log"
Vg,Fg=load("GT"); Vg=np.asarray(Vg,float); Fg=np.asarray(Fg,np.int64)
ref=np.load(f"{R}/fertility_fy1_auto.npz")["verts"].astype(float); T0=np.eye(4); T0[:3,3]=ref.mean(0)-Vg.mean(0)
reg=o3d.pipelines.registration.registration_icp(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vg)),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref)),0.05,T0,o3d.pipelines.registration.TransformationEstimationPointToPoint())
Vg=(np.c_[Vg,np.ones(len(Vg))]@reg.transformation.T)[:,:3]
sc=o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(Vg.astype(np.float32)),o3d.core.Tensor(Fg.astype(np.uint32)))
def gt_occ(p,r=0.012):
    P=p[None]+np.array([[0,0,0],[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[0,0,r],[0,0,-r]]); return float((sc.compute_occupancy(o3d.core.Tensor(P.astype(np.float32))).numpy()>0.5).mean())
log=open(LOG).read().splitlines()
tags=sorted({m.group(1) for l in log for m in [re.search(r"\[V66F (g66f\d+)\] \[RESULT\]",l)] if m},key=lambda t:int(t[4:]))
tot=bad=0
for t in tags:
    res=[l for l in log if f"[V66F {t}] [RESULT]" in l][0]; g=re.search(r"final genus (\d)",res).group(1)
    gates=[(("3" if "[fertility 3]" in l else "5b"),re.search(r"site=([A-Z]+)",l).group(1),'"gated_accept": true' in l) for l in log if l.startswith(f"[V66F {t}] ") and "JEV gate" in l]
    ents=[]
    for st,f in (("3",f"{R}/handles_fertility_{t}.json"),("5b",f"{R}/handles_fertility_{t}_5b.json")):
        if os.path.exists(f):
            E=json.load(open(f))
            if st=="5b": E=E[len([x for x in ents if x[0]=="3"]):] if len(E)>len([x for x in ents if x[0]=="3"]) else []
            ents+= [(st,e) for e in E]
    line=f"{t:7s} genus {g}:"
    for (st,site,acc),(_,e) in zip(gates,ents):
        if not acc: continue
        mid=np.array(e["mid"]); o=gt_occ(mid)
        kind="join" if site in("CONTACT","CREASE") else "drill"
        ok=(o<=0.3) if kind=="drill" else (o>=0.7)
        tot+=1; bad+= (not ok)
        line+=f"  [{st} {site[:4]} {kind} GT={'air' if o<=0.3 else 'mat' if o>=0.7 else 'mix'} {'OK' if ok else 'CHECK'}]"
    print(line)
print(f"\naccepted handles checked: {tot}, consistent with GT: {tot-bad}, to check: {bad}")
