#!/usr/bin/env python3
"""
Generate before/after visualizations for every TopMod operator, plus the
docs/operators.md reference that links them.

Usage:
    python3 scripts/generate_op_gallery.py            # images + markdown
    python3 scripts/generate_op_gallery.py --md-only  # regenerate markdown only

Output:
    docs/assets/ops/<name>.png   — side-by-side before/after render
    docs/operators.md            — full operator reference (generated)

The registry below is the single source of truth: adding an operator here
updates both the gallery and the markdown.
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from topmod import (
    DLFLMesh,
    make_cube, make_tetrahedron, make_icosahedron,
    insert_edge, delete_edge,
    extrude_face, add_handle, stellate, stellate_all,
    subdivide_edge, subdivide_face,
    catmull_clark,
    dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide,
    honeycomb_subdivide, star_subdivide, corner_cutting,
    loop_style_subdivide, fractal_subdivide,
    pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide,
    checkerboard_remesh, ds_bc_new_subdivide, dome_subdivide,
    create_crust,
)

ASSET_DIR = os.path.join(ROOT, "docs", "assets", "ops")
MD_PATH   = os.path.join(ROOT, "docs", "operators.md")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render(ax, mesh: DLFLMesh, alpha: float = 0.92) -> None:
    polys = []
    for f in mesh.iter_faces():
        polys.append([(v.x, v.y, v.z) for v in f.vertices()])
    coll = Poly3DCollection(polys, alpha=alpha, linewidths=0.6)
    coll.set_facecolor("#a8c4e0")
    coll.set_edgecolor("#1a2a3a")
    ax.add_collection3d(coll)

    xs = [v.x for v in mesh.iter_vertices()]
    ys = [v.y for v in mesh.iter_vertices()]
    zs = [v.z for v in mesh.iter_vertices()]
    lo = min(min(xs), min(ys), min(zs))
    hi = max(max(xs), max(ys), max(zs))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_zlim(lo - pad, hi + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_pair(name: str, before: DLFLMesh, after: DLFLMesh,
                alpha_after: float = 0.92) -> str:
    fig = plt.figure(figsize=(8, 4.2))
    for i, (mesh, title, alpha) in enumerate(
            [(before, "before", 0.92), (after, "after", alpha_after)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _render(ax, mesh, alpha=alpha)
        ax.set_title(f"{title}  V={mesh.V()} E={mesh.E()} F={mesh.F()}",
                     fontsize=9)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    out = os.path.join(ASSET_DIR, f"{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Operator registry — single source of truth for gallery + markdown
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpEntry:
    name: str                     # image/file name
    category: str
    signature: str
    token: str                    # tokenizer opcode or '—'
    oracle: str                   # closed-form element-count effect
    params: str                   # parameter meaning ('—' if none)
    desc: str                     # what it does (Chinese)
    example: str                  # usage snippet
    base: Callable[[], DLFLMesh] = make_cube
    base_name: str = "cube"
    # apply(mesh) -> mesh_after (may mutate in place and return same mesh)
    apply: Optional[Callable[[DLFLMesh], DLFLMesh]] = None
    alpha_after: float = 0.92
    no_image: bool = False
    no_image_reason: str = ""


def _first_face(m): return next(iter(m.faces.values()))
def _first_edge(m): return next(iter(m.edges.values()))


def _apply_insert_edge(m):
    f = _first_face(m)
    hes = f.halfedges()
    insert_edge(m, hes[0], hes[2])   # diagonal chord across the quad
    return m


def _apply_delete_edge(m):
    delete_edge(m, _first_edge(m))   # merge the two flanking faces
    return m


def _apply_extrude(m):
    extrude_face(m, _first_face(m), dist=0.6)
    return m


def _apply_stellate(m):
    stellate(m, _first_face(m))
    return m


def _apply_subdivide_edge(m):
    subdivide_edge(m, _first_edge(m))
    return m


def _apply_subdivide_face(m):
    subdivide_face(m, _first_face(m))
    return m


def _apply_add_handle(m):
    faces = list(m.faces.values())
    add_handle(m, faces[0], faces[1])    # tunnel between two OPPOSITE faces
    return m


def _apply_stellate_all(m):
    stellate_all(m)
    return m


def _apply_star(m):
    star_subdivide(m, offset=0.15)
    return m


def _apply_dome(m):
    dome_subdivide(m)
    return m


def _apply_crust_hole(m):
    out, pairs = create_crust(m, thickness=0.25)
    add_handle(out, pairs[0][0], pairs[0][1])   # punch one hole to reveal shell
    return out


OPS: List[OpEntry] = [
    # ── 1. 基础算子（Akleman & Chen 2003 极小完备集）────────────────────
    OpEntry("create_vertex", "1. 基础算子",
            "create_vertex(mesh, x, y, z) -> Vertex", "CV",
            "V+1, E+0, F+1（点球面）",
            "x,y,z — 坐标",
            "创建一个孤立点球面（point sphere）：单顶点自成一个组件，是所有构造的起点。",
            "v = create_vertex(mesh, 0.0, 0.0, 0.0)",
            no_image=True, no_image_reason="单个点，无可视化意义"),
    OpEntry("delete_vertex", "1. 基础算子",
            "delete_vertex(mesh, vertex)", "—",
            "V−1, E+0, F−1",
            "—",
            "删除一个孤立点球面。仅对无边顶点合法。",
            "delete_vertex(mesh, v)",
            no_image=True, no_image_reason="单个点，无可视化意义"),
    OpEntry("insert_edge", "1. 基础算子",
            "insert_edge(mesh, he1, he2) -> Edge", "IE",
            "E+1；同面 → F+1（分面），异面 → F−1（并组件/开柄）",
            "he1, he2 — 两个角（半边）",
            "在两个角之间插入一条边。同一面内插入把面一分为二；跨面插入把两个面合并。这是 DLFL 的两大核心算子之一，任何时刻保持 2-流形。",
            "hes = face.halfedges()\ninsert_edge(mesh, hes[0], hes[2])  # 四边形对角线",
            apply=_apply_insert_edge),
    OpEntry("delete_edge", "1. 基础算子",
            "delete_edge(mesh, edge)", "DE",
            "E−1；两侧异面 → F−1（并面），两侧同面 → F+1",
            "—",
            "删除一条边。两侧是不同面时合并为一个面（图示：立方体一条边删除后两个方形并成六边形）。insert_edge 的逆算子。",
            "delete_edge(mesh, edge)",
            apply=_apply_delete_edge),

    # ── 2. 高层算子 ────────────────────────────────────────────────────
    OpEntry("extrude_face", "2. 高层算子",
            "extrude_face(mesh, face, dist=1.0) -> List[Face]", "—",
            "V+n, E+2n, F+n（n = 面的度）",
            "dist — 沿法线挤出距离",
            "沿法线挤出一个面：顶面 + n 个侧面四边形。返回 [顶面] + 侧面列表。",
            "new_faces = extrude_face(mesh, face, dist=0.6)\ntop = new_faces[0]",
            apply=_apply_extrude),
    OpEntry("stellate", "2. 高层算子",
            "stellate(mesh, face, dist=0.0) -> Vertex", "—",
            "V+1, E+n, F+n−1",
            "dist — 顶点沿法线位移",
            "星化一个面：质心加顶点，与各角连边，n 边形变 n 个三角形。",
            "apex = stellate(mesh, face)",
            apply=_apply_stellate),
    OpEntry("subdivide_edge", "2. 高层算子",
            "subdivide_edge(mesh, edge) -> Vertex", "—",
            "V+1, E+1, F+0",
            "—",
            "在中点劈开一条边，返回中点顶点。",
            "mid = subdivide_edge(mesh, edge)",
            apply=_apply_subdivide_edge),
    OpEntry("subdivide_face", "2. 高层算子",
            "subdivide_face(mesh, face) -> Vertex", "—",
            "V+1, E+n, F+n−1",
            "—",
            "面细分：质心顶点向各角连边（拓扑同 stellate，位置在面内）。",
            "c = subdivide_face(mesh, face)",
            apply=_apply_subdivide_face),
    OpEntry("add_handle", "2. 高层算子",
            "add_handle(mesh, face1, face2) -> List[Edge]", "HDL",
            "V+0, E+n, F+n−2, χ−2, genus+1（同组件）",
            "—",
            "在两个同度面之间加柄（隧道）：两面消耗，生成 n 个侧面四边形。唯一改变 genus 的算子；也用于 crust 打孔。图示：立方体顶底面打通成方环。",
            "add_handle(mesh, top_face, bottom_face)",
            apply=_apply_add_handle, alpha_after=0.55),
    OpEntry("stellate_all", "2. 高层算子",
            "stellate_all(mesh) -> List[Vertex]  # 原地", "STA",
            "V'=V+F, E'=3E, F'=2E",
            "—",
            "对所有面星化，输出全三角形网格。是 honeycomb / star 的组成模块。",
            "apexes = stellate_all(mesh)",
            apply=_apply_stellate_all),

    # ── 3. 经典细分 ────────────────────────────────────────────────────
    OpEntry("catmull_clark", "3. 经典细分",
            "catmull_clark(mesh) -> DLFLMesh", "CC",
            "V'=V+E+F, E'=4E, F'=2E（全四边形）",
            "—",
            "Catmull-Clark 细分：面点/边点/顶点平滑，输出全四边形，是工业标准光滑细分。",
            "out = catmull_clark(mesh)",
            apply=catmull_clark),
    OpEntry("dual", "3. 经典细分",
            "dual(mesh) -> DLFLMesh", "DUAL",
            "V'=F, E'=E, F'=V",
            "—",
            "组合对偶：面变顶点（质心）、顶点变面。dual(dual(M)) ≅ M。立方体 ↔ 八面体。",
            "out = dual(mesh)",
            apply=dual),
    OpEntry("doo_sabin", "3. 经典细分",
            "doo_sabin(mesh) -> DLFLMesh", "DS",
            "V'=2E, E'=4E, F'=V+E+F",
            "—",
            "Doo-Sabin 细分（切角类）：每角一个新顶点，生成面面/边面/顶点面三类面。",
            "out = doo_sabin(mesh)",
            apply=doo_sabin),
    OpEntry("simplest_subdivide", "3. 经典细分",
            "simplest_subdivide(mesh) -> DLFLMesh", "SIMP",
            "V'=E, E'=2E, F'=F+V",
            "—",
            "中边（simplest / Peters-Reif）细分：边中点为新顶点。立方体 → 立方八面体。",
            "out = simplest_subdivide(mesh)",
            apply=simplest_subdivide),
    OpEntry("vertex_cutting", "3. 经典细分",
            "vertex_cutting(mesh, offset=0.25) -> DLFLMesh", "VC",
            "V'=2E, E'=3E, F'=F+V",
            "offset ∈ (0,0.5) — 切角深度",
            "顶点截断：每个顶点被切掉，n 边形变 2n 边形。立方体 → 截角立方体。",
            "out = vertex_cutting(mesh, offset=0.25)",
            apply=vertex_cutting),
    OpEntry("loop_subdivide", "3. 经典细分",
            "loop_subdivide(mesh) -> DLFLMesh  # 仅三角网格", "LOOP",
            "V'=V+E, E'=4E, F'=4F",
            "—",
            "Loop 细分：1 分 4 + β 权重平滑。非三角输入抛 ValueError。",
            "out = loop_subdivide(make_icosahedron())",
            base=make_icosahedron, base_name="icosahedron",
            apply=loop_subdivide),
    OpEntry("sqrt3_subdivide", "3. 经典细分",
            "sqrt3_subdivide(mesh) -> DLFLMesh  # 仅三角网格", "SQRT3",
            "V'=V+F, E'=3E, F'=3F",
            "—",
            "√3 细分（Kobbelt）：质心插点 + 全边翻转。非三角输入抛 ValueError。",
            "out = sqrt3_subdivide(make_icosahedron())",
            base=make_icosahedron, base_name="icosahedron",
            apply=sqrt3_subdivide),

    # ── 4. TopMod 特色细分（clean-room 自参考语义实现）──────────────────
    OpEntry("honeycomb_subdivide", "4. TopMod 特色细分",
            "honeycomb_subdivide(mesh) -> DLFLMesh", "HONEY",
            "V'=2E, E'=3E, F'=F+V",
            "—",
            "蜂窝细分 = dual ∘ stellate_all。三角输入产生六边形主导网格。",
            "out = honeycomb_subdivide(mesh)",
            apply=honeycomb_subdivide),
    OpEntry("star_subdivide", "4. TopMod 特色细分",
            "star_subdivide(mesh, offset=0.0)  # 原地", "STAR",
            "V'=V+F+2E, E'=9E, F'=6E（全三角）",
            "offset — 一轮顶点沿原面法线位移",
            "星形细分 = stellate_all 两次；offset 把第一轮顶点抬起形成星刺。",
            "star_subdivide(mesh, offset=0.3)",
            apply=_apply_star),
    OpEntry("corner_cutting", "4. TopMod 特色细分",
            "corner_cutting(mesh, alpha=0.5) -> DLFLMesh", "CCUT",
            "V'=2E, E'=4E, F'=V+E+F（同 Doo-Sabin 拓扑）",
            "alpha ∈ (0,1) — 张力（对角权重）",
            "切角细分：Doo-Sabin 的参数化几何变体，alpha 控制新角靠近原角的程度。",
            "out = corner_cutting(mesh, alpha=0.7)",
            apply=corner_cutting),
    OpEntry("loop_style_subdivide", "4. TopMod 特色细分",
            "loop_style_subdivide(mesh, length=1.0) -> DLFLMesh", "LSTYLE",
            "V'=V+E, E'=4E, F'=F+2E",
            "length ∈ [0,1] — 原顶点混合（1=保持）",
            "Loop 连通性的多边形推广：每面切下角三角形，留中点 d 边形。三角输入时连通性 = Loop。",
            "out = loop_style_subdivide(mesh)",
            apply=loop_style_subdivide),
    OpEntry("fractal_subdivide", "4. TopMod 特色细分",
            "fractal_subdivide(mesh, offset=1.0) -> DLFLMesh", "FRAC",
            "V'=V+E+F, E'=6E, F'=4E（全三角）",
            "offset — 尖顶高度系数",
            "分形细分 = loop_style + 中央多边形星化（尖顶沿法线抬升），产生分形尖刺外观。",
            "out = fractal_subdivide(mesh, offset=1.0)",
            apply=fractal_subdivide),
    OpEntry("pentagonal_subdivide", "4. TopMod 特色细分",
            "pentagonal_subdivide(mesh, offset=0.0) -> DLFLMesh", "PENT",
            "V'=V+2E+F, E'=5E, F'=2E（全五边形）",
            "offset ∈ [0,1] — 辐条邻点向质心收拢",
            "五边形细分：三等分每条边 + 质心辐条，每个 d 边形变 d 个五边形。四面体 → 正十二面体组合结构。",
            "out = pentagonal_subdivide(mesh)",
            apply=pentagonal_subdivide),
    OpEntry("pentagonal2_subdivide", "4. TopMod 特色细分",
            "pentagonal2_subdivide(mesh, scale_factor=0.75) -> DLFLMesh", "PENT2",
            "V'=V+3E, E'=6E, F'=F+2E",
            "scale_factor — 内多边形收缩",
            "五边形细分变体 2：中点劈边 + 缩放内 d 边形 + 连接边；每面 = 内 d 边形 + d 个五边形。",
            "out = pentagonal2_subdivide(mesh, scale_factor=0.7)",
            apply=pentagonal2_subdivide),
    OpEntry("dual1264_subdivide", "4. TopMod 特色细分",
            "dual1264_subdivide(mesh, sf=1.0) -> DLFLMesh", "D1264",
            "V'=4E, E'=6E, F'=F+E+V",
            "sf — 内多边形缩放",
            "12.6.4 对偶细分：类 Doo-Sabin，但每面的内多边形是 2d 边形（边上 1/3、2/3 点），三角输入产生 12.6.4 式镶嵌。",
            "out = dual1264_subdivide(mesh)",
            apply=dual1264_subdivide),
    OpEntry("root4_subdivide", "4. TopMod 特色细分",
            "root4_subdivide(mesh, a=0.0, twist=0.0) -> DLFLMesh", "ROOT4",
            "V'=V+2E, E'=4E, F'=F+E",
            "a — 原顶点平滑混合；twist — 内环采样滑移",
            "Root-4 细分：蜂窝掩码内多边形 + 棱柱桥接 + 删除全部原边；原顶点保留（每边一个六边形）。",
            "out = root4_subdivide(mesh, a=0.3, twist=0.2)",
            apply=root4_subdivide),
    OpEntry("checkerboard_remesh", "4. TopMod 特色细分",
            "checkerboard_remesh(mesh, thickness=0.25) -> DLFLMesh", "CHKB",
            "V'=V+4E, E'=9E, F'=F+4E",
            "thickness ∈ (0,0.5) — 内缩/三等分比例",
            "棋盘重网格化：内缩面 + 边三等分 + 角切弦；四边形输入输出仍全四边形，呈棋盘交错。",
            "out = checkerboard_remesh(mesh, thickness=0.25)",
            apply=checkerboard_remesh),
    OpEntry("ds_bc_new_subdivide", "4. TopMod 特色细分",
            "ds_bc_new_subdivide(mesh, sf=1.0, length=1.0) -> DLFLMesh", "DSBC",
            "V'=V+4E, E'=7E, F'=F+2E",
            "sf — DS 缩放；length — 原顶点混合",
            "Doo-Sabin BC-new：对中点加密后的 2d 边形边界做 DS，原顶点存活；每面一个 2d 边形 + 每边两个五边形。",
            "out = ds_bc_new_subdivide(mesh, sf=0.9)",
            apply=ds_bc_new_subdivide),
    OpEntry("dome_subdivide", "4. TopMod 特色细分",
            "dome_subdivide(mesh, length=1.0, sf=1.0)  # 原地", "DOME",
            "V'=V+59E, E'=116E, F'=F+56E",
            "length — 高度轮廓；sf — 缩放轮廓",
            "穹顶细分：每边四等分 + 每原面 7 层 DS 式挤出（内置高度/缩放轮廓），每面隆起一个圆顶。",
            "dome_subdivide(mesh)",
            apply=_apply_dome),

    # ── 5. 结构算子 ────────────────────────────────────────────────────
    OpEntry("create_crust", "5. 结构算子",
            "create_crust(mesh, thickness=0.1) -> (DLFLMesh, pairs)", "CRUST",
            "V'=2V, E'=2E, F'=2F，2 个组件；打 k 孔后 genus'=2g+k−1",
            "thickness — 壳厚（负值向外偏移）",
            "壳体化：整个曲面反向复制并沿顶点平均法线内偏，形成内外双层壳。返回镜像面对列表，用 add_handle 逐对打孔形成隧道。图示：壳体 + 打穿一个孔。",
            "out, pairs = create_crust(mesh, thickness=0.25)\nfor outer, inner in pairs[:2]:\n    add_handle(out, outer, inner)   # 每孔 genus +1（首孔连通两壳）",
            apply=_apply_crust_hole, alpha_after=0.45),
]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown generation
# ─────────────────────────────────────────────────────────────────────────────

MD_HEADER = """# TopMod 算子完全参考

> 由 `scripts/generate_op_gallery.py` 生成 — 手改会被覆盖，改注册表后重跑脚本。

GenesisTopmod 的全部网格算子：签名、参数、闭式 oracle（元素数量效果）、
tokenizer 词元与 before/after 可视化。所有算子在每一步都保持 2-流形
（Akleman & Chen 2003 的 DLFL 构造性保证）。

- **oracle 列**给出算子对 (V, E, F) 的精确闭式效果，是
  `tests/test_semantic_oracle.py` 的断言依据；除 `add_handle`/crust 打孔外
  全部保持 χ 与 genus。
- **token 列**是 `topmod/tokenizer.py` 词汇表中的操作码；带 token 的算子可被
  序列化为整数 ID 序列并由 `detokenize` 回放（自回归生成的基础）。
- 可视化图由本脚本渲染，before/after 均标注 V/E/F。

## 速查表

| 算子 | token | oracle (V', E', F') | 可视化 |
|---|---|---|---|
"""

MD_USAGE_FOOTER = """
## Tokenizer 用法

```python
from topmod import tokenize, detokenize, build_vocabulary, encode_sequence
from topmod.tokenizer import TopModToken

# 任何 token 序列从确定性的二十面体基元开始执行
tokens = [TopModToken(op='PENT'), TopModToken(op='DUAL'), TopModToken(op='EOS')]
mesh = detokenize(tokens)            # 保证 is_manifold(mesh) == True

vocab = build_vocabulary()           # 词元 → 整数 ID（append-only，向后兼容）
ids = encode_sequence(tokens, vocab)
```

带孔壳体（生成模型可学的 genus 构造）：

```python
# CRUST 后镜像面对的序号确定：外层面 i ↔ 内层面 F+i，
# 因此打孔可以直接用现有 HDL(face1, face2) 词元表达。
tokens = [TopModToken(op='CRUST'),
          TopModToken(op='HDL', refs=(0, 20)),   # 二十面体：F=20
          TopModToken(op='EOS')]
```

## 测试

```bash
python3 -m pytest tests/ -q --ignore=tests/test_manifold_loss.py --ignore=tests/test_pipeline.py
```

每个算子的 oracle 测试在 `tests/test_semantic_oracle.py`（四种基元 ×
精确 ΔV/ΔE/ΔF/χ/genus + 面度数普查），token 测试在 `tests/test_tokenizer.py`。

## 参考

- Akleman & Chen 2003 — DLFL 极小完备算子集
- `docs/reference_semantics.md` — 参考库（davyrisso/topmod3d, GPL）语义的
  clean-room 提取与 χ 验证
- `docs/vocabulary_roadmap.md` — 词汇表演进路线
"""


def gen_markdown() -> None:
    lines = [MD_HEADER]
    for op in OPS:
        img = (f"[图](assets/ops/{op.name}.png)"
               if not op.no_image else "—")
        lines.append(f"| [`{op.name}`](#{op.name.lower()}) | {op.token} "
                     f"| {op.oracle} | {img} |\n")

    cat = None
    for op in OPS:
        if op.category != cat:
            cat = op.category
            lines.append(f"\n---\n\n## {cat}\n")
        lines.append(f"\n### {op.name}\n\n")
        lines.append(f"{op.desc}\n\n")
        lines.append(f"- **签名**: `{op.signature}`\n")
        lines.append(f"- **token**: `{op.token}`\n")
        lines.append(f"- **oracle**: {op.oracle}\n")
        lines.append(f"- **参数**: {op.params}\n")
        lines.append(f"- **示例基元**: {op.base_name}\n\n")
        lines.append("```python\n" + op.example + "\n```\n")
        if op.no_image:
            lines.append(f"\n*（无可视化：{op.no_image_reason}）*\n")
        else:
            lines.append(f"\n![{op.name}](assets/ops/{op.name}.png)\n")

    lines.append(MD_USAGE_FOOTER)
    with open(MD_PATH, "w") as fh:
        fh.write("".join(lines))
    print(f"wrote {MD_PATH}")


def gen_images() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    for op in OPS:
        if op.no_image or op.apply is None:
            continue
        before = op.base()
        after = op.apply(op.base())
        out = render_pair(op.name, before, after, alpha_after=op.alpha_after)
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-only", action="store_true")
    args = ap.parse_args()
    if not args.md_only:
        gen_images()
    gen_markdown()
