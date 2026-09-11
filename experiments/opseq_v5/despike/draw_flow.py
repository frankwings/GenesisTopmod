"""Draw results_genus/flow_golden_v3.png (step / sub-step flowchart of the golden v3 chain)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
DR="#1f77b4"; TM="#ff7f0e"; GE="#7f7f7f"; TB="#2ca02c"
stages=[
("Stage 1  run_64v.py  — icosphere → coarse shape (STOP_AFTER=cc3)", DR,[
 ("icosphere = TopMod make_icosahedron + 2× catmull_clark → 240 quads, rendered via  [TopMod triangulate_all]",TM),
 ("cc2: 800 steps — EACH step: nvdiffrast renders 64 views (sil + depth + headlight diffuse) → L1 vs GT",DR),
 ("   + Laplacian + edge + tri-quality + spike / sliver / fold / tube regularisers → backward → Adam",DR),
 ("refine: [TopMod catmull_clark] on the quad mesh (1 quad → 4) + [TopMod triangulate_all]   (C2F_SUBDIV=cc, default)",TM),
 ("cc3: 800 steps, same DR loop (+settle)  → 1.9k faces",DR)]),
("Stage 2  phase4_inloop.py 400 steps — DLFL clean loop on the coarse mesh", DR,[
 ("every step: same 64-view DR loss (+ voting-hull field W_T=20) → Adam",DR),
 ("every 100 steps: collapse short edges  [TopMod collapse_edge_tri]",TM),
 ("every 25 steps: flip folded edges  [TopMod delete_edge + insert_edge]  → tangential smooth → SI push (geometric)",TM)]),
("Stage 3  phase7_multi.sh — GENUS DISCOVERY (≤8 rounds, 1 handle per round)", TM,[
 ("detect: see-through pixels (≥30 px) in TRAINING silhouettes where our mesh is solid → membrane patches outside",GE),
 ("   the voting hull → disk (χ) test → face pair on the two sheets with disjoint 1-rings",GE),
 ("open tunnel  [TopMod add_handle(face_i, face_j)]  → side quads  [TopMod stellate]  → tube refine  [TopMod subdivide_edge]",TM),
 ("400-step Stage-2 loop pulls the tube walls to the hole; repeat until no evidence   (armadillo 0 · kitten/rocker 1 · fertility 3–4)",DR)]),
("Stage 4  run_64v.py RESUME_FROM — cc4 + DESPIKE surgery", DR,[
 ("refine: [TopMod catmull_clark] on the triangle mesh (1 tri → 3 quads → 6 tris via triangulate_all) → cc4: 800 steps DR",TM),
 ("DESPIKE surgery 1 (surgery_lib.py, v22 'seed + extent + propagation'):",GE),
 ("   detect thin: vertex-to-non-adjacent-surface distance < 0.4 × mean edge  |  spike vertices",GE),
 ("   seed: vertices whose projection lands OUTSIDE the GT silhouette in the training views (escape_mask, 1-px dilate)",DR),
 ("   propagate the seed over the thin-connected component (whole needle condemned; a thin ear with no escaping tip is spared)",GE),
 ("   amputate component  [TopMod collapse_edge_tri: condemned edges first, then each condemned vertex into a healthy neighbour]",TM),
 ("   accept only if train-IoU drop ≤ 3e-4 (global cap 1.5e-3) else revert; ≤8 rounds",DR),
 ("settle 400 steps DR → surgery 2 (budget 2e-4, ≤4 rounds)",DR)]),
("Stage 5  phase4_inloop.py 1200 steps — GOLDEN v3 loop (Palfinger optimizer params)", DR,[
 ("every step: 64-view DR loss + hull field; Adam β=(0.8,0.8); grad += 0.02·ν·(v − nbr mean); clip 10|m1|; lr = 0.3 × mean edge",DR),
 ("every 50 steps ADAPTIVE REMESH: target edge = velocity-controlled (ν gain 0.2), floor 1.3 px, cap 100k faces, no splits while SI > 30 %",GE),
 ("   split  [TopMod subdivide_edge + stellate]      collapse  [TopMod collapse_edge_tri]",TM),
 ("every 25 steps: flip  [TopMod delete_edge + insert_edge]  → tangential smooth → SI push",TM)]),
("Stage 6  phase4_inloop.py 1200 steps — same golden v3 loop, Laplacian ×3", DR,[
 ("identical ops; V grows to ~49k (armadillo)",TM)]),
("Stage 7  phase5_taubin.py AUTO — fairing (NO topology change: Open3D Taubin λ=0.5 μ=−0.53, positions only)", TB,[
 ("iteration count = argmax training-view IoU (min 2); connectivity untouched",TB)]),
("Exam  ho16 (16 held-out silhouettes) + VolIoU / CD (eval_cd_iou.py: ICP → 256³ occupancy)", GE,[]),
]
W=15.5; fig_h=0.55*sum(1+len(s) for _,_,s in stages)+2.2
fig,ax=plt.subplots(figsize=(W,fig_h)); ax.set_xlim(0,W); ax.set_ylim(0,fig_h); ax.axis("off")
y=fig_h-0.3
ax.text(0.3,y,"GenesisTopmod golden v3 chain — every step & sub-step (gif_v9_chain.sh; ALL topology changes = TopMod operators, 2026-09-11)",fontsize=14,weight="bold",va="top"); y-=0.45
for i,(t,c,subs) in enumerate(stages):
    h=0.5+0.42*len(subs)+0.15
    ax.add_patch(FancyBboxPatch((0.3,y-h),W-0.6,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc="#f7f7f7",ec=c,lw=2.5))
    ax.text(0.5,y-0.12,t,fontsize=11.5,weight="bold",color=c,va="top")
    yy=y-0.55
    for s,sc in subs:
        ax.text(0.7,yy,("● " if not s.startswith("   ") else "")+s,fontsize=9.1,color=sc if sc!=GE else "#333333",va="top",weight="bold" if sc==TM else "normal"); yy-=0.42
    y-=h+0.12
    if i<len(stages)-1: ax.annotate("",xy=(W/2,y+0.02),xytext=(W/2,y+0.12),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
ly=y-0.2
for lab,c in [("blue  = nvdiffrast differentiable rendering (every step, 64 views)",DR),("orange = TopMod DLFL operator (manifold guaranteed; check_all / check_watertight asserted after each)",TM),("green = Open3D Taubin (positions only)",TB),("grey  = numpy geometry / detection logic (no topology change)",GE)]:
    ax.text(0.5,ly,"■ "+lab,fontsize=9.8,color=c,va="top",weight="bold"); ly-=0.32
plt.savefig("results_genus/flow_golden_v3.png",dpi=120,bbox_inches="tight"); print("flow saved")
