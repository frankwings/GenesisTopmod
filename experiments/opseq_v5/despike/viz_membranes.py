#!/usr/bin/env python3
"""viz_membranes.py <shape> <cc3p4.npz> <out.png>: DR mesh (light), faces spanning hull air coloured by patch, chosen pairs as lines."""
import sys, os, numpy as np
sys.path[:0] = ["/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5", "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike", "/home/kingy/Projects/Genesis/GenesisTopmod"]
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
shape, npz, out = sys.argv[1:4]
src = open("despike/phase7_handle.py").read(); SETUP = src[:src.index('report("base", V, Fa)')]
os.environ.update(SHAPE=shape, BASE_NPZ=npz, GENUS_TARGET="hull", MODE="64v", DETECT="membrane"); os.environ.pop("REAL_DATA", None)
ns = {}; exec(compile(SETUP, "p7", "exec"), ns); V, F, HF = ns["V"], ns["Fa"], ns["HF"]
import membrane_locate as ml
res = ml.find_tunnel_by_membranes(V, F, HF, [], ns["G_TARGET"]); st = ml.find_tunnel_by_membranes.last
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
patches, lab, memb_f, fdist = st["patches"], st["lab"], st["memb_f"], st["fdist"]
cols = plt.cm.tab20(np.linspace(0, 1, 20)); cen = V[F].mean(1)
fig = plt.figure(figsize=(21, 7)); fig.patch.set_facecolor("white"); ext = V.max(0) - V.min(0); c0 = (V.max(0) + V.min(0)) / 2
for vi, (el, az) in enumerate([(25, -60), (25, 30), (65, 120)]):
    ax = fig.add_subplot(1, 3, vi + 1, projection="3d")
    fc = np.tile([0.85, 0.85, 0.85, 0.25], (len(F), 1))
    for pi, (ids, f0, c, nm) in enumerate(patches): fc[ids] = [*cols[pi % 20][:3], 0.95]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors=[0.6, 0.6, 0.6, 0.15], linewidths=0.2))
    for pi, (ids, f0, c, nm) in enumerate(patches): ax.text(c[0], c[1], c[2], f"p{pi}", fontsize=8, weight="bold")
    for (fi, fj, ci, cj, key) in res[:6]: ax.plot([ci[0], cj[0]], [ci[1], cj[1]], [ci[2], cj[2]], color="k", lw=2)
    ax.set_xlim(c0[0]-ext[0]/2, c0[0]+ext[0]/2); ax.set_ylim(c0[1]-ext[1]/2, c0[1]+ext[1]/2); ax.set_zlim(c0[2]-ext[2]/2, c0[2]+ext[2]/2)
    ax.view_init(el, az); ax.set_axis_off(); ax.set_box_aspect(tuple(ext / ext.max()))
fig.suptitle(f"{shape} {os.path.basename(npz)}: faces whose interior is > {ml.MARGIN:g} vox outside the hull, coloured by patch ({len(patches)} patches); black = proposed pairs (top 6 of {len(res)})", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=100); print("saved", out)
