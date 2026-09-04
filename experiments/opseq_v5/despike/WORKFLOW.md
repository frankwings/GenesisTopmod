# GenesisTopmod 多视角重建 Workflow（2026-09-03 定稿版）

目标：从一个球出发，用 64 张训练视角的照片（剪影 + 深度 + 漫反射，256²），重建出一张
**封闭、无自交、genus 正确的流形网格**。考试 = 16 张从未参与训练的视角上的剪影 IoU（ho16）。
所有拓扑变化都是 TopMod DLFL 算子；顶点位置由 nvdiffrast 可微渲染 + Adam 优化。

## 主链（6 个阶段 + 1 个可选的拓扑阶段）

| # | 阶段 | 脚本 | 做什么 | 拓扑 |
|---|---|---|---|---|
| 1 | 球 → 大形状 | `run_64v.py` | cc2 球 800 步 → 中点细分 → 800 → 细分 → 800 → 去毛刺手术 ×2 + settle 400 | 中点细分 (numpy)，`collapse_edge_tri` |
| 4 | 循环清理 | `phase4_inloop.py` | 1200 步：渲染损失 + 投票视觉壳距离场；每 25 步 DLFL 翻边 / 短边塌缩 / 自交推开 / 切向平滑 | flip = `delete_edge`+`insert_edge`，`collapse_edge_tri` |
| **7** | **加把手（有洞物体）** | `phase7_handle.py` | 隧道证据：两个背对背的面都在壳外 > 2 体素、连线全程在壳外 → `add_handle` → 3 个侧面四边形 stellate → 回到阶段 4 | `add_handle`（genus +1），`stellate` |
| 5 | 全局细分 + 循环 | `phase4_inloop.py SUBDIV_ALL=1 LAP_MULT=3` | 每条边切中点 + 每个面 stellate（×6 面数），再跑 1200 步循环 | `subdivide_edge`，`stellate` |
| 6 | 抛光 | `phase5_taubin.py ITERS=5` | Taubin λ=0.5 μ=−0.53 ×5，只动位置 | — |
| (7b) | 二次加密（可选） | `phase4_inloop.py SUBDIV_TOP=1500` → Taubin | 只细分最大的 1500 个面 → 56.8k 面 | `subdivide_edge`，`stellate` |

阶段 2（挤出雕刻）和 3（独立壳拉拽）已被消融证明多余（壳损失内置于阶段 4/5）；阶段 4 不能省
（"先清理再细分"，否则细分放大自交）。

## 结果

| 形状 | genus | 链 | ho16 | 自交 | hair | 面数 | DMesh |
|---|---|---|---|---|---|---|---|
| armadillo | 0 | 1-4-5-6 | 0.9959 | 1.5 % | 0 | 36.9k | 0.9894 |
| armadillo | 0 | + 7b 二次加密 | **0.9972** | 0.5 % | 1 | 56.8k | |
| rocker-arm | 1 | 1-4-**7**-4-5-6 | **0.9975** | 0.1 % | 0 | 29.5k | — |
| rocker-arm 不加把手 | 0 (错) | 1-4 | 0.9608 | | 17,617 | | |

## 复现命令（rocker-arm）
```
export MODE=64v SHAPE=rockerarm
python3 despike/run_64v.py                                     # TAG=rockerarm_64v
TAG=rockerarm_p4  STEPS=1200 FLIP_EVERY=25 COLLAPSE_EVERY=100 COLLAPSE_RATIO=0.5 COLLAPSE_MAX=300 SI_PUSH=0.15 \
  BASE_NPZ=/tmp/liou_cow_viz/cow_rockerarm_64v.npz            python3 despike/phase4_inloop.py
TAG=rockerarm_p7  MAX_HANDLES=1 BASE_NPZ=.../cow_rockerarm_rockerarm_p4.npz   python3 despike/phase7_handle.py
TAG=rockerarm_p7b STEPS=1200 (同上参数) BASE_NPZ=.../cow_rockerarm_rockerarm_p7.npz  python3 despike/phase4_inloop.py
TAG=rockerarm_p5  SUBDIV_ALL=1 LAP_MULT=3 STEPS=1200 COLLAPSE_RATIO=0.4 COLLAPSE_MAX=600 (其余同上) \
  BASE_NPZ=.../cow_rockerarm_rockerarm_p7b.npz                 python3 despike/phase4_inloop.py
ITERS=5 TAG=rockerarm_taubin5 BASE_NPZ=.../cow_rockerarm_rockerarm_p5.npz  python3 despike/phase5_taubin.py
```
每步后 `check_watertight` 断言；genus 用欧拉数 V−E+F 验证；通孔用沿轴射线穿透计数验证。

## 关键结论（详见 LESSONS_2026-09-02.md）
- 精度来自：可微渲染 + 壳距离场 + 先清理再细分 + Taubin；与算子库无关（numpy 等价物 0.9957 vs 0.9959）。
- TopMod 的价值：构造性流形保证 + 算子程序 + **改 genus 的流形算子**（`add_handle`），后者是 Open3D/trimesh 没有、DMesh 只能以非流形换取的能力。
- 分辨率上限由监督图像分辨率决定：256² 下 ~50–60k 面，219k 面会在像素以下自我纠缠。
- 稀疏视角（6v）下流形约束是正则化器：DMesh 随机起步 0.53，我们 0.95。
- 已知 TODO：多把手互斥（同一轮第二个把手会撞上新管壁）；多洞形状（3holes g3, fertility g4）。
