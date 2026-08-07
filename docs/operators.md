# TopMod 算子完全参考

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
| [`create_vertex`](#create_vertex) | CV | V+1, E+0, F+1（点球面） | — |
| [`delete_vertex`](#delete_vertex) | — | V−1, E+0, F−1 | — |
| [`insert_edge`](#insert_edge) | IE | E+1；同面 → F+1（分面），异面 → F−1（并组件/开柄） | [图](assets/ops/insert_edge.png) |
| [`delete_edge`](#delete_edge) | DE | E−1；两侧异面 → F−1（并面），两侧同面 → F+1 | [图](assets/ops/delete_edge.png) |
| [`extrude_face`](#extrude_face) | — | V+n, E+2n, F+n（n = 面的度） | [图](assets/ops/extrude_face.png) |
| [`stellate`](#stellate) | — | V+1, E+n, F+n−1 | [图](assets/ops/stellate.png) |
| [`subdivide_edge`](#subdivide_edge) | — | V+1, E+1, F+0 | [图](assets/ops/subdivide_edge.png) |
| [`subdivide_face`](#subdivide_face) | — | V+1, E+n, F+n−1 | [图](assets/ops/subdivide_face.png) |
| [`add_handle`](#add_handle) | HDL | V+0, E+n, F+n−2, χ−2, genus+1（同组件） | [图](assets/ops/add_handle.png) |
| [`stellate_all`](#stellate_all) | STA | V'=V+F, E'=3E, F'=2E | [图](assets/ops/stellate_all.png) |
| [`catmull_clark`](#catmull_clark) | CC | V'=V+E+F, E'=4E, F'=2E（全四边形） | [图](assets/ops/catmull_clark.png) |
| [`dual`](#dual) | DUAL | V'=F, E'=E, F'=V | [图](assets/ops/dual.png) |
| [`doo_sabin`](#doo_sabin) | DS | V'=2E, E'=4E, F'=V+E+F | [图](assets/ops/doo_sabin.png) |
| [`simplest_subdivide`](#simplest_subdivide) | SIMP | V'=E, E'=2E, F'=F+V | [图](assets/ops/simplest_subdivide.png) |
| [`vertex_cutting`](#vertex_cutting) | VC | V'=2E, E'=3E, F'=F+V | [图](assets/ops/vertex_cutting.png) |
| [`loop_subdivide`](#loop_subdivide) | LOOP | V'=V+E, E'=4E, F'=4F | [图](assets/ops/loop_subdivide.png) |
| [`sqrt3_subdivide`](#sqrt3_subdivide) | SQRT3 | V'=V+F, E'=3E, F'=3F | [图](assets/ops/sqrt3_subdivide.png) |
| [`honeycomb_subdivide`](#honeycomb_subdivide) | HONEY | V'=2E, E'=3E, F'=F+V | [图](assets/ops/honeycomb_subdivide.png) |
| [`star_subdivide`](#star_subdivide) | STAR | V'=V+F+2E, E'=9E, F'=6E（全三角） | [图](assets/ops/star_subdivide.png) |
| [`corner_cutting`](#corner_cutting) | CCUT | V'=2E, E'=4E, F'=V+E+F（同 Doo-Sabin 拓扑） | [图](assets/ops/corner_cutting.png) |
| [`loop_style_subdivide`](#loop_style_subdivide) | LSTYLE | V'=V+E, E'=4E, F'=F+2E | [图](assets/ops/loop_style_subdivide.png) |
| [`fractal_subdivide`](#fractal_subdivide) | FRAC | V'=V+E+F, E'=6E, F'=4E（全三角） | [图](assets/ops/fractal_subdivide.png) |
| [`pentagonal_subdivide`](#pentagonal_subdivide) | PENT | V'=V+2E+F, E'=5E, F'=2E（全五边形） | [图](assets/ops/pentagonal_subdivide.png) |
| [`pentagonal2_subdivide`](#pentagonal2_subdivide) | PENT2 | V'=V+3E, E'=6E, F'=F+2E | [图](assets/ops/pentagonal2_subdivide.png) |
| [`dual1264_subdivide`](#dual1264_subdivide) | D1264 | V'=4E, E'=6E, F'=F+E+V | [图](assets/ops/dual1264_subdivide.png) |
| [`root4_subdivide`](#root4_subdivide) | ROOT4 | V'=V+2E, E'=4E, F'=F+E | [图](assets/ops/root4_subdivide.png) |
| [`checkerboard_remesh`](#checkerboard_remesh) | CHKB | V'=V+4E, E'=9E, F'=F+4E | [图](assets/ops/checkerboard_remesh.png) |
| [`ds_bc_new_subdivide`](#ds_bc_new_subdivide) | DSBC | V'=V+4E, E'=7E, F'=F+2E | [图](assets/ops/ds_bc_new_subdivide.png) |
| [`dome_subdivide`](#dome_subdivide) | DOME | V'=V+59E, E'=116E, F'=F+56E | [图](assets/ops/dome_subdivide.png) |
| [`create_crust`](#create_crust) | CRUST | V'=2V, E'=2E, F'=2F，2 个组件；打 k 孔后 genus'=2g+k−1 | [图](assets/ops/create_crust.png) |

---

## 1. 基础算子

### create_vertex

创建一个孤立点球面（point sphere）：单顶点自成一个组件，是所有构造的起点。

- **签名**: `create_vertex(mesh, x, y, z) -> Vertex`
- **token**: `CV`
- **oracle**: V+1, E+0, F+1（点球面）
- **参数**: x,y,z — 坐标
- **示例基元**: cube

```python
v = create_vertex(mesh, 0.0, 0.0, 0.0)
```

*（无可视化：单个点，无可视化意义）*

### delete_vertex

删除一个孤立点球面。仅对无边顶点合法。

- **签名**: `delete_vertex(mesh, vertex)`
- **token**: `—`
- **oracle**: V−1, E+0, F−1
- **参数**: —
- **示例基元**: cube

```python
delete_vertex(mesh, v)
```

*（无可视化：单个点，无可视化意义）*

### insert_edge

在两个角之间插入一条边。同一面内插入把面一分为二；跨面插入把两个面合并。这是 DLFL 的两大核心算子之一，任何时刻保持 2-流形。

- **签名**: `insert_edge(mesh, he1, he2) -> Edge`
- **token**: `IE`
- **oracle**: E+1；同面 → F+1（分面），异面 → F−1（并组件/开柄）
- **参数**: he1, he2 — 两个角（半边）
- **示例基元**: cube

```python
hes = face.halfedges()
insert_edge(mesh, hes[0], hes[2])  # 四边形对角线
```

![insert_edge](assets/ops/insert_edge.png)

### delete_edge

删除一条边。两侧是不同面时合并为一个面（图示：立方体一条边删除后两个方形并成六边形）。insert_edge 的逆算子。

- **签名**: `delete_edge(mesh, edge)`
- **token**: `DE`
- **oracle**: E−1；两侧异面 → F−1（并面），两侧同面 → F+1
- **参数**: —
- **示例基元**: cube

```python
delete_edge(mesh, edge)
```

![delete_edge](assets/ops/delete_edge.png)

---

## 2. 高层算子

### extrude_face

沿法线挤出一个面：顶面 + n 个侧面四边形。返回 [顶面] + 侧面列表。

- **签名**: `extrude_face(mesh, face, dist=1.0) -> List[Face]`
- **token**: `—`
- **oracle**: V+n, E+2n, F+n（n = 面的度）
- **参数**: dist — 沿法线挤出距离
- **示例基元**: cube

```python
new_faces = extrude_face(mesh, face, dist=0.6)
top = new_faces[0]
```

![extrude_face](assets/ops/extrude_face.png)

### stellate

星化一个面：质心加顶点，与各角连边，n 边形变 n 个三角形。

- **签名**: `stellate(mesh, face, dist=0.0) -> Vertex`
- **token**: `—`
- **oracle**: V+1, E+n, F+n−1
- **参数**: dist — 顶点沿法线位移
- **示例基元**: cube

```python
apex = stellate(mesh, face)
```

![stellate](assets/ops/stellate.png)

### subdivide_edge

在中点劈开一条边，返回中点顶点。

- **签名**: `subdivide_edge(mesh, edge) -> Vertex`
- **token**: `—`
- **oracle**: V+1, E+1, F+0
- **参数**: —
- **示例基元**: cube

```python
mid = subdivide_edge(mesh, edge)
```

![subdivide_edge](assets/ops/subdivide_edge.png)

### subdivide_face

面细分：质心顶点向各角连边（拓扑同 stellate，位置在面内）。

- **签名**: `subdivide_face(mesh, face) -> Vertex`
- **token**: `—`
- **oracle**: V+1, E+n, F+n−1
- **参数**: —
- **示例基元**: cube

```python
c = subdivide_face(mesh, face)
```

![subdivide_face](assets/ops/subdivide_face.png)

### add_handle

在两个同度面之间加柄（隧道）：两面消耗，生成 n 个侧面四边形。唯一改变 genus 的算子；也用于 crust 打孔。图示：立方体顶底面打通成方环。

- **签名**: `add_handle(mesh, face1, face2) -> List[Edge]`
- **token**: `HDL`
- **oracle**: V+0, E+n, F+n−2, χ−2, genus+1（同组件）
- **参数**: —
- **示例基元**: cube

```python
add_handle(mesh, top_face, bottom_face)
```

![add_handle](assets/ops/add_handle.png)

### stellate_all

对所有面星化，输出全三角形网格。是 honeycomb / star 的组成模块。

- **签名**: `stellate_all(mesh) -> List[Vertex]  # 原地`
- **token**: `STA`
- **oracle**: V'=V+F, E'=3E, F'=2E
- **参数**: —
- **示例基元**: cube

```python
apexes = stellate_all(mesh)
```

![stellate_all](assets/ops/stellate_all.png)

---

## 3. 经典细分

### catmull_clark

Catmull-Clark 细分：面点/边点/顶点平滑，输出全四边形，是工业标准光滑细分。

- **签名**: `catmull_clark(mesh) -> DLFLMesh`
- **token**: `CC`
- **oracle**: V'=V+E+F, E'=4E, F'=2E（全四边形）
- **参数**: —
- **示例基元**: cube

```python
out = catmull_clark(mesh)
```

![catmull_clark](assets/ops/catmull_clark.png)

### dual

组合对偶：面变顶点（质心）、顶点变面。dual(dual(M)) ≅ M。立方体 ↔ 八面体。

- **签名**: `dual(mesh) -> DLFLMesh`
- **token**: `DUAL`
- **oracle**: V'=F, E'=E, F'=V
- **参数**: —
- **示例基元**: cube

```python
out = dual(mesh)
```

![dual](assets/ops/dual.png)

### doo_sabin

Doo-Sabin 细分（切角类）：每角一个新顶点，生成面面/边面/顶点面三类面。

- **签名**: `doo_sabin(mesh) -> DLFLMesh`
- **token**: `DS`
- **oracle**: V'=2E, E'=4E, F'=V+E+F
- **参数**: —
- **示例基元**: cube

```python
out = doo_sabin(mesh)
```

![doo_sabin](assets/ops/doo_sabin.png)

### simplest_subdivide

中边（simplest / Peters-Reif）细分：边中点为新顶点。立方体 → 立方八面体。

- **签名**: `simplest_subdivide(mesh) -> DLFLMesh`
- **token**: `SIMP`
- **oracle**: V'=E, E'=2E, F'=F+V
- **参数**: —
- **示例基元**: cube

```python
out = simplest_subdivide(mesh)
```

![simplest_subdivide](assets/ops/simplest_subdivide.png)

### vertex_cutting

顶点截断：每个顶点被切掉，n 边形变 2n 边形。立方体 → 截角立方体。

- **签名**: `vertex_cutting(mesh, offset=0.25) -> DLFLMesh`
- **token**: `VC`
- **oracle**: V'=2E, E'=3E, F'=F+V
- **参数**: offset ∈ (0,0.5) — 切角深度
- **示例基元**: cube

```python
out = vertex_cutting(mesh, offset=0.25)
```

![vertex_cutting](assets/ops/vertex_cutting.png)

### loop_subdivide

Loop 细分：1 分 4 + β 权重平滑。非三角输入抛 ValueError。

- **签名**: `loop_subdivide(mesh) -> DLFLMesh  # 仅三角网格`
- **token**: `LOOP`
- **oracle**: V'=V+E, E'=4E, F'=4F
- **参数**: —
- **示例基元**: icosahedron

```python
out = loop_subdivide(make_icosahedron())
```

![loop_subdivide](assets/ops/loop_subdivide.png)

### sqrt3_subdivide

√3 细分（Kobbelt）：质心插点 + 全边翻转。非三角输入抛 ValueError。

- **签名**: `sqrt3_subdivide(mesh) -> DLFLMesh  # 仅三角网格`
- **token**: `SQRT3`
- **oracle**: V'=V+F, E'=3E, F'=3F
- **参数**: —
- **示例基元**: icosahedron

```python
out = sqrt3_subdivide(make_icosahedron())
```

![sqrt3_subdivide](assets/ops/sqrt3_subdivide.png)

---

## 4. TopMod 特色细分

### honeycomb_subdivide

蜂窝细分 = dual ∘ stellate_all。三角输入产生六边形主导网格。

- **签名**: `honeycomb_subdivide(mesh) -> DLFLMesh`
- **token**: `HONEY`
- **oracle**: V'=2E, E'=3E, F'=F+V
- **参数**: —
- **示例基元**: cube

```python
out = honeycomb_subdivide(mesh)
```

![honeycomb_subdivide](assets/ops/honeycomb_subdivide.png)

### star_subdivide

星形细分 = stellate_all 两次；offset 把第一轮顶点抬起形成星刺。

- **签名**: `star_subdivide(mesh, offset=0.0)  # 原地`
- **token**: `STAR`
- **oracle**: V'=V+F+2E, E'=9E, F'=6E（全三角）
- **参数**: offset — 一轮顶点沿原面法线位移
- **示例基元**: cube

```python
star_subdivide(mesh, offset=0.3)
```

![star_subdivide](assets/ops/star_subdivide.png)

### corner_cutting

切角细分：Doo-Sabin 的参数化几何变体，alpha 控制新角靠近原角的程度。

- **签名**: `corner_cutting(mesh, alpha=0.5) -> DLFLMesh`
- **token**: `CCUT`
- **oracle**: V'=2E, E'=4E, F'=V+E+F（同 Doo-Sabin 拓扑）
- **参数**: alpha ∈ (0,1) — 张力（对角权重）
- **示例基元**: cube

```python
out = corner_cutting(mesh, alpha=0.7)
```

![corner_cutting](assets/ops/corner_cutting.png)

### loop_style_subdivide

Loop 连通性的多边形推广：每面切下角三角形，留中点 d 边形。三角输入时连通性 = Loop。

- **签名**: `loop_style_subdivide(mesh, length=1.0) -> DLFLMesh`
- **token**: `LSTYLE`
- **oracle**: V'=V+E, E'=4E, F'=F+2E
- **参数**: length ∈ [0,1] — 原顶点混合（1=保持）
- **示例基元**: cube

```python
out = loop_style_subdivide(mesh)
```

![loop_style_subdivide](assets/ops/loop_style_subdivide.png)

### fractal_subdivide

分形细分 = loop_style + 中央多边形星化（尖顶沿法线抬升），产生分形尖刺外观。

- **签名**: `fractal_subdivide(mesh, offset=1.0) -> DLFLMesh`
- **token**: `FRAC`
- **oracle**: V'=V+E+F, E'=6E, F'=4E（全三角）
- **参数**: offset — 尖顶高度系数
- **示例基元**: cube

```python
out = fractal_subdivide(mesh, offset=1.0)
```

![fractal_subdivide](assets/ops/fractal_subdivide.png)

### pentagonal_subdivide

五边形细分：三等分每条边 + 质心辐条，每个 d 边形变 d 个五边形。四面体 → 正十二面体组合结构。

- **签名**: `pentagonal_subdivide(mesh, offset=0.0) -> DLFLMesh`
- **token**: `PENT`
- **oracle**: V'=V+2E+F, E'=5E, F'=2E（全五边形）
- **参数**: offset ∈ [0,1] — 辐条邻点向质心收拢
- **示例基元**: cube

```python
out = pentagonal_subdivide(mesh)
```

![pentagonal_subdivide](assets/ops/pentagonal_subdivide.png)

### pentagonal2_subdivide

五边形细分变体 2：中点劈边 + 缩放内 d 边形 + 连接边；每面 = 内 d 边形 + d 个五边形。

- **签名**: `pentagonal2_subdivide(mesh, scale_factor=0.75) -> DLFLMesh`
- **token**: `PENT2`
- **oracle**: V'=V+3E, E'=6E, F'=F+2E
- **参数**: scale_factor — 内多边形收缩
- **示例基元**: cube

```python
out = pentagonal2_subdivide(mesh, scale_factor=0.7)
```

![pentagonal2_subdivide](assets/ops/pentagonal2_subdivide.png)

### dual1264_subdivide

12.6.4 对偶细分：类 Doo-Sabin，但每面的内多边形是 2d 边形（边上 1/3、2/3 点），三角输入产生 12.6.4 式镶嵌。

- **签名**: `dual1264_subdivide(mesh, sf=1.0) -> DLFLMesh`
- **token**: `D1264`
- **oracle**: V'=4E, E'=6E, F'=F+E+V
- **参数**: sf — 内多边形缩放
- **示例基元**: cube

```python
out = dual1264_subdivide(mesh)
```

![dual1264_subdivide](assets/ops/dual1264_subdivide.png)

### root4_subdivide

Root-4 细分：蜂窝掩码内多边形 + 棱柱桥接 + 删除全部原边；原顶点保留（每边一个六边形）。

- **签名**: `root4_subdivide(mesh, a=0.0, twist=0.0) -> DLFLMesh`
- **token**: `ROOT4`
- **oracle**: V'=V+2E, E'=4E, F'=F+E
- **参数**: a — 原顶点平滑混合；twist — 内环采样滑移
- **示例基元**: cube

```python
out = root4_subdivide(mesh, a=0.3, twist=0.2)
```

![root4_subdivide](assets/ops/root4_subdivide.png)

### checkerboard_remesh

棋盘重网格化：内缩面 + 边三等分 + 角切弦；四边形输入输出仍全四边形，呈棋盘交错。

- **签名**: `checkerboard_remesh(mesh, thickness=0.25) -> DLFLMesh`
- **token**: `CHKB`
- **oracle**: V'=V+4E, E'=9E, F'=F+4E
- **参数**: thickness ∈ (0,0.5) — 内缩/三等分比例
- **示例基元**: cube

```python
out = checkerboard_remesh(mesh, thickness=0.25)
```

![checkerboard_remesh](assets/ops/checkerboard_remesh.png)

### ds_bc_new_subdivide

Doo-Sabin BC-new：对中点加密后的 2d 边形边界做 DS，原顶点存活；每面一个 2d 边形 + 每边两个五边形。

- **签名**: `ds_bc_new_subdivide(mesh, sf=1.0, length=1.0) -> DLFLMesh`
- **token**: `DSBC`
- **oracle**: V'=V+4E, E'=7E, F'=F+2E
- **参数**: sf — DS 缩放；length — 原顶点混合
- **示例基元**: cube

```python
out = ds_bc_new_subdivide(mesh, sf=0.9)
```

![ds_bc_new_subdivide](assets/ops/ds_bc_new_subdivide.png)

### dome_subdivide

穹顶细分：每边四等分 + 每原面 7 层 DS 式挤出（内置高度/缩放轮廓），每面隆起一个圆顶。

- **签名**: `dome_subdivide(mesh, length=1.0, sf=1.0)  # 原地`
- **token**: `DOME`
- **oracle**: V'=V+59E, E'=116E, F'=F+56E
- **参数**: length — 高度轮廓；sf — 缩放轮廓
- **示例基元**: cube

```python
dome_subdivide(mesh)
```

![dome_subdivide](assets/ops/dome_subdivide.png)

---

## 5. 结构算子

### create_crust

壳体化：整个曲面反向复制并沿顶点平均法线内偏，形成内外双层壳。返回镜像面对列表，用 add_handle 逐对打孔形成隧道。图示：壳体 + 打穿一个孔。

- **签名**: `create_crust(mesh, thickness=0.1) -> (DLFLMesh, pairs)`
- **token**: `CRUST`
- **oracle**: V'=2V, E'=2E, F'=2F，2 个组件；打 k 孔后 genus'=2g+k−1
- **参数**: thickness — 壳厚（负值向外偏移）
- **示例基元**: cube

```python
out, pairs = create_crust(mesh, thickness=0.25)
for outer, inner in pairs[:2]:
    add_handle(out, outer, inner)   # 每孔 genus +1（首孔连通两壳）
```

![create_crust](assets/ops/create_crust.png)

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
