"""Visualise hull-located tunnel plugs: hull (grey), tunnel fill block (light colour), sealing sheet (colour),
skeleton centreline (black), throat (star), chosen add_handle face pair (red/blue) on the coarse mesh."""
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
    for f in glob.glob(f"{vd}/*.npy") + glob.glob(f"{vd}/*.json") + glob.glob(f"{vd}/*.npz"): os.remove(f)
    for f in glob.glob(f"/tmp/liou_cow_viz/hull_plugs_{S}_128.json*"): os.remove(f)
    code = f'''
import os, json, numpy as np
os.environ["SHAPE"]="{S}"; os.environ["BASE_NPZ"]="/tmp/liou_cow_viz/cow_{S}_{S}_v6_cc3p4.npz"; os.environ["HULL_VIZ_DIR"]="{vd}"
src=open("{D}/phase7_handle.py").read(); cut=src.index("_hull_max = "); exec(compile(src[:cut],"p7","exec"))
res = find_tunnel_by_hull(V, Fa, HF, (), G_TARGET)
np.savez("{vd}/mesh.npz", V=V, F=Fa, lo=HF.lo, hi=HF.hi)
json.dump([{{"fi":int(a),"fj":int(b),"ci":list(map(float,c)),"cj":list(map(float,d)),"key":k}} for a,b,c,d,k in res], open("{vd}/pairs.json","w"))
print("VIZ", "{S}", "gstar", G_TARGET, "pairs", len(res))
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(D, ".."), capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("VIZ") or "accepted faces" in l or "no face pair" in l or "PLUG" in l))
    if r.returncode: print(r.stderr[-1500:])
    m = np.load(f"{vd}/mesh.npz"); hs = np.load(f"{vd}/hull_ds.npy"); N = hs.shape[0]
    lo, hi = m["lo"], m["hi"]; w = lambda v: lo + np.asarray(v, float) / (N - 1) * (hi - lo)
    surf = hs & ~ndimage.binary_erosion(hs); sp = w(np.argwhere(surf)); sp = sp[::max(1, len(sp)//6000)]
    k = len(glob.glob(f"{vd}/plug*_path.npy"))
    plugs = [dict(sheet=w(np.load(f"{vd}/plug{i}_vox.npy")), block=w(np.load(f"{vd}/plug{i}_block.npy")), path=w(np.load(f"{vd}/plug{i}_path.npy"))) for i in range(1, k + 1)]
    info = json.load(open(f"{vd}/plugs.json")) if os.path.exists(f"{vd}/plugs.json") else []
    pairs = json.load(open(f"{vd}/pairs.json"))
    rows.append((S, sp, plugs, info, pairs, m))
fig = plt.figure(figsize=(6 * 3, 5.4 * len(rows)))
for r, (S, sp, plugs, info, pairs, m) in enumerate(rows):
    lo, hi = m["lo"], m["hi"]; V = m["V"]
    for c, (el, az) in enumerate([(20, -60), (20, 30), (85, -90)]):
        ax = fig.add_subplot(len(rows), 3, r * 3 + c + 1, projection="3d")
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1, c="lightgrey", alpha=0.2, depthshade=False)
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=2, c="dimgray", alpha=0.35, depthshade=False)
        for i, pg in enumerate(plugs):
            col = COLS[i % len(COLS)]
            b = pg["block"][::max(1, len(pg["block"])//2500)]; ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=2, c=col, alpha=0.12, depthshade=False)
            sh = pg["sheet"][::max(1, len(pg["sheet"])//1200)]; ax.scatter(sh[:, 0], sh[:, 1], sh[:, 2], s=5, c=col, alpha=0.9, depthshade=False, label=f"plug {i+1} R={info[i]['R']} ({info[i]['mode']})" if i < len(info) else f"plug {i+1}")
            P = pg["path"]; ax.plot(P[:, 0], P[:, 1], P[:, 2], c="k", lw=1.6)
            if i < len(info):
                t = np.asarray(info[i]["throat_w"]); ax.scatter(*t, s=90, c=col, marker="*", edgecolors="k", depthshade=False)
        for pr in pairs:
            ci, cj = np.asarray(pr["ci"]), np.asarray(pr["cj"])
            ax.scatter(*ci, s=70, c="red", marker="^", depthshade=False); ax.scatter(*cj, s=70, c="blue", marker="v", depthshade=False)
            ax.plot(*zip(ci, cj), c="red", lw=2.5)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(hi - lo)
        if c == 0: ax.set_title(f"{S}: GT genus {GT[S]} | plugs {len(plugs)} | face pairs {len(pairs)}/{len(plugs)}", fontsize=13, loc="left")
        if c == 1 and plugs: ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Hull-located tunnels on the coarse Stage-2 mesh (dark dots = mesh vertices). light colour = tunnel fill at sealing radius, "
             "solid colour = sealing sheet (throat),\nblack = skeleton centreline through the sheet, star = throat, red/blue = add_handle face pair from the occupancy walk", fontsize=12)
plt.tight_layout(); out = os.path.join(OUT, "hull_plugs_viz.png"); plt.savefig(out, dpi=110); print("saved", out)
