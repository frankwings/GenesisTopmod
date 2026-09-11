"""Render an N-column x 4-row visual comparison of meshes given as label=path
(npz or obj, all in our normalized frame). Helpers live in viz_render.py.
Usage: [ROWS=front,back] python3 despike/render_compare.py OUT.png "GT=GT" "label=path" ...
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
from PIL import Image, ImageDraw, ImageFont
from viz_render import load, render, ROWS
_rows = os.environ.get("ROWS")           # optional comma list of row names to keep
if _rows: ROWS = [r for r in ROWS if r[0] in _rows.split(",")]
out = sys.argv[1]; items = [a.split("=", 1) for a in sys.argv[2:]]
res = 700; W = res * len(items); H = res * len(ROWS) + 60
canvas = Image.new("L", (W, H), 255); d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except Exception: font = ImageFont.load_default()
meshes = [(lab, load(p)) for lab, p in items]
for j, (lab, (V, F)) in enumerate(meshes):
    d.text((j * res + 10, 12), f"{lab}  ({len(F)/1000:.1f}k f)", fill=0, font=font)
    for i, (rn, kw) in enumerate(ROWS):
        canvas.paste(Image.fromarray(render(V, F, **kw)), (j * res, 60 + i * res))
for i, (rn, _) in enumerate(ROWS): d.text((10, 60 + i * res + 8), rn, fill=90, font=font)
canvas.save(out); print("saved", out, canvas.size)
