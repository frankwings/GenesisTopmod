"""Side-by-side comparison figure: 4 shapes x {GT, ours headline, ours golden-v3 uniform chain, Palfinger 2022,
DMesh 2024, Nicolet 2021}. Every number is recomputed here from the archived meshes with the SAME exam:
ho16 (silhouette IoU on 16 held-out views), VolIoU / CD (eval_cd_iou: ICP -> Chamfer -> 256^3 occupancy),
V, F, genus (Euler, only for watertight meshes; DMesh output is a triangle soup)."""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
from PIL import Image, ImageDraw, ImageFont
D = "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike"; os.chdir(D); R = f"{D}/results_genus"
SHAPES = [  # (shape, az, el, [(label, path), ...])
 ("armadillo", 180, 10, [("Ours golden v3", f"{D}/results64v/cow_armadillo_golden_v3.npz"), ("Ours v3 uniform chain", f"{R}/armadillo_g3chain_auto.npz"),
                         ("Palfinger 2022", f"{R}/armadillo_palfinger1200.npz"), ("DMesh 2024", f"{R}/armadillo_dmesh_last.obj"), ("Nicolet 2021", f"{R}/armadillo_nicolet1200.npz")]),
 ("kitten", 200, 15, [("Ours v8b (headline)", f"{R}/kitten_v8_auto.npz"), ("Ours v3 uniform chain", f"{R}/kitten_g3chain_auto.npz"),
                      ("Palfinger 2022", f"{R}/kitten_palfinger1200.npz"), ("DMesh 2024", f"{R}/kitten_dmesh_last.obj"), ("Nicolet 2021", f"{R}/kitten_nicolet1200.npz")]),
 ("rockerarm", 90, 20, [("Ours v8b (headline)", f"{R}/rockerarm_adapt_v8res_auto.npz"), ("Ours v3 uniform chain", f"{R}/rockerarm_g3chain_auto.npz"),
                        ("Palfinger 2022", f"{R}/rockerarm_palfinger1200.npz"), ("DMesh 2024", f"{R}/rockerarm_dmesh_last.obj"), ("Nicolet 2021", f"{R}/rockerarm_nicolet1200.npz")]),
 ("fertility", 180, 20, [("Ours v8b genus-4 (headline)", f"{R}/fertility_g4hull256_auto.npz"), ("Ours v3 uniform chain", f"{R}/fertility_g3chain_auto.npz"),
                         ("Palfinger 2022", f"{R}/fertility_palfinger1200.npz"), ("DMesh 2024", f"{R}/fertility_dmesh_last.obj"), ("Nicolet 2021", f"{R}/fertility_nicolet1200.npz")]),
]
RES = int(os.environ.get("TILE", "1000")); SC = RES / 420; PAD = int(78 * SC)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(15 * SC))
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(13 * SC))
FONT_T = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(22 * SC))

def genus_of(V, F):
    from phase1b_pipeline import check_watertight
    wt, _ = check_watertight(np.asarray(F, np.int64))
    if not wt: return None
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

rows = []; table = []
CACHE = json.load(open(f"{R}/compare_all_methods.json")) if os.environ.get("REUSE") else None
def cached(shape, lab):
    for sh, lb, m in (CACHE or []):
        if sh == shape and lb == lab and isinstance(m, dict): return m
    return None
for shape, az, el, entries in SHAPES:
    os.environ["SHAPE"] = shape
    import importlib, viz_render, eval_cd_iou, phase1b_pipeline as p1b
    importlib.reload(viz_render); p1b.SHAPE = shape
    from viz_render import load, render, ctx
    Vg, Fg = load("GT"); Vg = np.asarray(Vg, float); Fg = np.asarray(Fg, np.int64)
    ctr = (Vg.min(0) + Vg.max(0)) / 2; dist = 3.1 * float(np.linalg.norm(Vg - ctr, axis=1).max())
    lo, hi = Vg.min(0) - 0.05, Vg.max(0) + 0.05; og = None if CACHE else eval_cd_iou.occupancy(Vg, Fg, lo, hi)
    tiles = []
    for lab, p in [("GT", "GT")] + entries:
        if p != "GT" and not os.path.exists(p):
            tiles.append((lab, None, f"(mesh missing: {os.path.basename(p)})", "")); table.append((shape, lab, "missing")); continue
        V, F = load(p); V = np.asarray(V, float); F = np.asarray(F, np.int64)
        img = render(V, F, az, el, ctr, dist, 35, res=RES)
        if p == "GT":
            g = genus_of(V, F); l1 = f"V={len(V):,}  F={len(F):,}  genus={g}"; l2 = "reference (100k-face source)"; table.append((shape, lab, dict(V=len(V), F=len(F), genus=g)))
        else:
            m = cached(shape, lab)
            if m: ho, iou, cd, gs = m["ho16"], m["VolIoU"], m["CD"], m["genus"]
            else:
                ho, hair, mb = p1b.heldout_exam(ctx, V, F)
                Va, fit = eval_cd_iou.icp_align(V, F, Vg, Fg); cd = eval_cd_iou.chamfer(Va, F, Vg, Fg)
                o = eval_cd_iou.occupancy(Va, F, lo, hi); iou = float((o & og).sum() / max((o | og).sum(), 1))
                g = genus_of(V, F); gs = "soup" if g is None else str(g)
            l1 = f"ho16 {ho:.4f}   VolIoU {iou:.4f}   CD {cd:.5f}"; l2 = f"V={len(V):,}  F={len(F):,}  genus={gs}"
            table.append((shape, lab, dict(ho16=round(float(ho), 4), VolIoU=round(iou, 4), CD=round(float(cd), 5), V=len(V), F=len(F), genus=gs)))
            print(f"[{shape}] {lab:28s} {l1} | {l2}", flush=True)
        tiles.append((lab, img, l1, l2))
    rows.append((shape, tiles))

ncol = 6; W = ncol * RES; TOP = int(40 * SC); H = len(rows) * (RES + PAD) + TOP
canvas = Image.new("L", (W, H), 255); d = ImageDraw.Draw(canvas)
d.text((10, int(8 * SC)), "GenesisTopmod vs competitors — same 64 views @256², same held-out exam, all meshes as produced by each method (recomputed 2026-09-11)", fill=0, font=FONT_T)
for r, (shape, tiles) in enumerate(rows):
    y0 = TOP + r * (RES + PAD)
    for c, (lab, img, l1, l2) in enumerate(tiles):
        x0 = c * RES
        if img is not None: canvas.paste(Image.fromarray(img), (x0, y0))
        else: d.rectangle([x0 + 5, y0 + 5, x0 + RES - 5, y0 + RES - 5], outline=128); d.text((x0 + 20, y0 + RES // 2), l1, fill=100, font=FONT_S)
        d.text((x0 + 8, y0 + RES + int(4 * SC)), (f"{shape} · " if c == 0 else "") + lab, fill=0, font=FONT)
        d.text((x0 + 8, y0 + RES + int(26 * SC)), l1, fill=0, font=FONT_S); d.text((x0 + 8, y0 + RES + int(44 * SC)), l2, fill=0, font=FONT_S)
canvas.save(f"{R}/compare_all_methods.png"); json.dump(table, open(f"{R}/compare_all_methods.json", "w"), indent=1)
print(f"saved {R}/compare_all_methods.png ({W}x{H})")
