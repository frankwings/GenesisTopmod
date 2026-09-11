"""Locate the residual needle on cow_v13t.obj, trace it back through the
cc3/cc4 snapshots, and render a step-by-step timeline (full + zoom)."""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np
from collections import deque
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = "/tmp/liou_cow_viz"

# --- load final mesh + tube mask ---
vs, fs = [], []
for line in open(f"{OUT}/cow_v13t.obj"):
    p = line.split()
    if not p: continue
    if p[0] == "v": vs.append([float(x) for x in p[1:4]])
    elif p[0] == "f": fs.append([int(x)-1 for x in p[1:4]])
V = np.array(vs); Fc = np.array(fs)
tm = np.load(f"{OUT}/trace_tubemask.npy")
print("verts", len(V), "tube verts", tm.sum())

# adjacency
adj = [[] for _ in range(len(V))]
for a, b, c in Fc:
    for x, y in ((a,b),(b,c),(c,a)):
        adj[x].append(y); adj[y].append(x)
adj = [list(set(n)) for n in adj]

# connected components among tube verts
seen = np.zeros(len(V), bool)
ccs = []
for i in np.where(tm)[0]:
    if seen[i]: continue
    q = deque([i]); seen[i] = True; comp = [i]
    while q:
        u = q.popleft()
        for w in adj[u]:
            if tm[w] and not seen[w]:
                seen[w] = True; comp.append(w); q.append(w)
    ccs.append(np.array(comp))

# protrusion score per CC: max distance from CC vert to nearest non-tube vert
nontube = V[~tm]
scores = []
for comp in ccs:
    d = np.sqrt(((V[comp][:, None] - nontube[None]) ** 2).sum(-1)).min(1)
    scores.append(d.max())
order = np.argsort(scores)[::-1]
for k in order[:5]:
    print(f"CC size={len(ccs[k])} protrusion={scores[k]:.4f} "
          f"center={V[ccs[k]].mean(0).round(3)}")
needle = ccs[order[0]]
ncenter = V[needle].mean(0)
print("NEEDLE:", len(needle), "verts, center", ncenter.round(3))

# --- load snapshots ---
z4 = np.load(f"{OUT}/trace_cc4.npz")
t4 = z4["tris"]
steps4 = sorted(int(k[1:]) for k in z4.files if k.startswith("s"))
z3 = np.load(f"{OUT}/trace_cc3.npz")
t3 = z3["tris"]
steps3 = sorted(int(k[1:]) for k in z3.files if k.startswith("s"))
n3 = z3["s0"].shape[0]

# cc3 ancestry: needle verts with index < n3 existed in cc3 directly;
# midpoints map to their position at cc4 step0
p0 = z4["s0"]
seed_pos = p0[needle]

# growth curve: protrusion of needle verts over cc4 steps
curve = []
for s in steps4:
    P = z4[f"s{s}"]
    nt_ = P[~tm] if len(P) == len(tm) else None
    d = np.sqrt(((P[needle][:, None] - P[~tm][None]) ** 2).sum(-1)).min(1)
    curve.append(d.max())

def render(ax, P, T, red_idx, elev, azim, zoom=None):
    pc = Poly3DCollection(P[T], facecolor="#5cb85c", edgecolor="none", alpha=1.0)
    ax.add_collection3d(pc)
    r = P[red_idx]
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], c="red", s=6, depthshade=False)
    if zoom is None:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    else:
        c, h = zoom
        ax.set_xlim(c[0]-h, c[0]+h); ax.set_ylim(c[1]-h, c[1]+h)
        ax.set_zlim(c[2]-h, c[2]+h)
    ax.set_box_aspect((2, 2, 2)); ax.view_init(elev=elev, azim=azim); ax.axis("off")

# pick frames every 100 steps for cc4
frames = [s for s in steps4 if s % 100 == 0]
ncol = len(frames)
fig = plt.figure(figsize=(2.2 * ncol, 7), dpi=110)
for j, s in enumerate(frames):
    P = z4[f"s{s}"]
    ax = fig.add_subplot(3, ncol, j + 1, projection="3d")
    render(ax, P, t4, needle, 0, 0)
    ax.set_title(f"cc4 s{s}", fontsize=8)
    ax = fig.add_subplot(3, ncol, ncol + j + 1, projection="3d")
    render(ax, P, t4, needle, 0, 0, zoom=(P[needle].mean(0), 0.28))
    ax = fig.add_subplot(3, ncol, 2 * ncol + j + 1, projection="3d")
    render(ax, P, t4, needle, 30, -60, zoom=(P[needle].mean(0), 0.28))
plt.suptitle("needle CC (red) through cc4 | row1 full az0 | row2 zoom az0 | row3 zoom az-60",
             fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/needle_timeline_cc4.png", bbox_inches="tight")
print("saved needle_timeline_cc4.png")

# cc3 timeline: needle ancestors = cc3 verts near seed positions
P30 = z3[f"s{steps3[-1]}"]  # cc3 final == positions that generated cc4 step0
d = np.sqrt(((P30[:, None] - seed_pos[None]) ** 2).sum(-1)).min(1)
anc = np.where(d < 1e-6)[0]
if len(anc) == 0:
    anc = np.argsort(d)[:6]
print("cc3 ancestors:", len(anc))
frames3 = [s for s in steps3 if s % 200 == 0]
fig = plt.figure(figsize=(2.2 * len(frames3), 5), dpi=110)
for j, s in enumerate(frames3):
    P = z3[f"s{s}"]
    ax = fig.add_subplot(2, len(frames3), j + 1, projection="3d")
    render(ax, P, t3, anc, 0, 0)
    ax.set_title(f"cc3 s{s}", fontsize=8)
    ax = fig.add_subplot(2, len(frames3), len(frames3) + j + 1, projection="3d")
    render(ax, P, t3, anc, 0, 0, zoom=(P[anc].mean(0), 0.35))
plt.suptitle("needle ancestors (red) through cc3", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/needle_timeline_cc3.png", bbox_inches="tight")
print("saved needle_timeline_cc3.png")

# growth curve
fig, ax = plt.subplots(figsize=(6, 3), dpi=110)
ax.plot(steps4, curve, "o-")
ax.set_xlabel("cc4 step"); ax.set_ylabel("needle protrusion")
ax.axvline(400, ls="--", c="gray", label="FOLD_START")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/needle_growth_curve.png")
print("saved needle_growth_curve.png")
print("curve:", [round(c, 4) for c in curve])
