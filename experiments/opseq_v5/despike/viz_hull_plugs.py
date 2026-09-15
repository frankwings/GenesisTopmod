"""Visualise hull-located tunnel plugs: hull (grey), plugs (colour), axis, chosen add_handle face pair (red/blue)."""
import os, sys, glob, subprocess, numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "results_genus")
SHAPES = os.environ.get("SHAPES", "kitten rockerarm threeholes fertility").split()
GT = {"armadillo": 0, "kitten": 1, "rockerarm": 1, "threeholes": 3, "fertility": 4}
COLS = ["tab:orange", "tab:green", "tab:purple", "tab:cyan", "gold", "tab:pink"]
rows = []
for S in SHAPES:
    vd = f"/tmp/hullviz_{S}"; os.makedirs(vd, exist_ok=True)
    for f in glob.glob(f"{vd}/*.npy"): os.remove(f)
    cache = f"/tmp/liou_cow_viz/hull_plugs_{S}_128.json"
    if os.path.exists(cache): os.remove(cache)
    # run the real detector once (fresh mesh, fresh ladder) and dump plugs + accepted pairs
    code = f'''
import os, json, numpy as np
os.environ["SHAPE"]="{S}"; os.environ["BASE_NPZ"]="/tmp/liou_cow_viz/cow_{S}_{S}_v6_cc3p4.npz"; os.environ["HULL_VIZ_DIR"]="{vd}"
src=open("{D}/phase7_handle.py").read(); cut=src.index("_hull_max = "); exec(compile(src[:cut],"p7","exec"))
res = find_tunnel_by_hull(V, Fa, HF, (), G_TARGET)
np.savez("{vd}/mesh.npz", V=V, F=Fa, lo=HF.lo, hi=HF.hi, pitch=HF.pitch)
json.dump([{{"fi":int(a),"fj":int(b),"ci":list(map(float,c)),"cj":list(map(float,d)),"key":k}} for a,b,c,d,k in res], open("{vd}/pairs.json","w"))
print("VIZ", "{S}", "gstar", G_TARGET, "pairs", len(res))
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(D, ".."), capture_output=True, text=True)
    print([l for l in r.stdout.splitlines() if l.startswith("VIZ") or "hull plugs:" in l or "no valid" in l][-4:]); 
    if r.returncode: print(r.stderr[-1500:])
    m = np.load(f"{vd}/mesh.npz"); hs = np.load(f"{vd}/hull_ds.npy"); N = hs.shape[0]
    lo, hi = m["lo"], m["hi"]; w = lambda v: lo + np.asarray(v, float) / (N - 1) * (hi - lo)
    surf = hs & ~ndimage.binary_erosion(hs); sp = w(np.argwhere(surf)); sp = sp[::max(1, len(sp)//6000)]
    plugs = [np.load(f) for f in sorted(glob.glob(f"{vd}/plug*_vox.npy"))]
    cache_d = json.load(open(cache)) if os.path.exists(cache) else []
    pairs = json.load(open(f"{vd}/pairs.json"))
    rows.append((S, sp, plugs, cache_d, pairs, m))
fig = plt.figure(figsize=(6 * 3, 5.2 * len(rows)))
for r, (S, sp, plugs, cache_d, pairs, m) in enumerate(rows):
    for c, (el, az) in enumerate([(20, -60), (20, 30), (85, -90)]):
        ax = fig.add_subplot(len(rows), 3, r * 3 + c + 1, projection="3d")
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1, c="lightgrey", alpha=0.25, depthshade=False)
        N = np.load(f"/tmp/hullviz_{S}/hull_ds.npy").shape[0]; lo, hi = m["lo"], m["hi"]
        for i, pv in enumerate(plugs):
            pw = lo + pv / (N - 1) * (hi - lo); pw = pw[::max(1, len(pw)//1500)]
            ax.scatter(pw[:, 0], pw[:, 1], pw[:, 2], s=4, c=COLS[i % len(COLS)], alpha=0.9, depthshade=False, label=f"plug {i+1} (R={cache_d[i]['R']})" if i < len(cache_d) else f"plug {i+1}")
            if i < len(cache_d):
                cen = np.asarray(cache_d[i]["cen_w"]); axw = np.asarray(cache_d[i]["axis_w"]); L = 0.25 * (hi - lo).max()
                ax.plot(*zip(cen - L * axw, cen + L * axw), c="k", lw=1.2)
        for pr in pairs:
            ci, cj = np.asarray(pr["ci"]), np.asarray(pr["cj"])
            ax.scatter(*ci, s=60, c="red", marker="^", depthshade=False); ax.scatter(*cj, s=60, c="blue", marker="v", depthshade=False)
            ax.plot(*zip(ci, cj), c="red", lw=2)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        lim = np.stack([lo, hi]); ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1]); ax.set_zlim(lim[:, 2]); ax.set_box_aspect(hi - lo)
        if c == 0: ax.set_title(f"{S}: GT genus {GT[S]} | plugs {len(plugs)} | face pairs {len(pairs)}/{len(plugs)}", fontsize=13, loc="left")
        if c == 1 and plugs: ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Hull-located tunnel plugs on the coarse Stage-2 mesh: grey = space-carved hull surface, colour = plug voxels (closing radius R),\nblack = tunnel axis, red/blue = chosen add_handle face pair (entry/exit)", fontsize=13)
plt.tight_layout(); out = os.path.join(OUT, "hull_plugs_viz.png"); plt.savefig(out, dpi=110); print("saved", out)
