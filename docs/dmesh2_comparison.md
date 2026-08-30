# DMesh++ (DMesh2) 对比备忘 — 可借鉴 vs 不可移植

> 记录日期：2026-08-26
> 论文：DMesh++: An Efficient Differentiable Mesh for Complex Shapes
> arXiv 2412.16776（ICCV 2025，Son et al.）；code: github.com/SonSang/dmesh2
> 前作：DMesh（arXiv 2404.13445）

## 一句话定位

DMesh++ 用**点集 + 局部 Minimum-Ball 测试**推导可微连通性，天生**不保证流形**（流形化只是可选后处理，高分辨率配置直接跳过）。
我们是**固定半边拓扑 + 流形安全算子注入**，输出**保证流形**——这是我们真正的差异化，不放弃。
可借鉴的只有一样东西：**"哪里细化能降低渲染损失"的信号**，把它接到我们的流形算子上。

## DMesh++ 真实机制（去营销版）

### 1. 网格表示 & 可微面存在性
- 每个点是 `(d+1)` 维向量 `(x₁…x_d, ψ)`：位置 + real 值 `ψ∈[0,1]`（是否在表面）。
- 连通性**不存储**，由点集 tessellation 推导。面存在概率分解：
  `Λ(F) = Λ_min(F) · Λ_real(F)`
  - `Λ_real(F)`：所有顶点 real（`min_{p∈F} Ψ(p) > 0.5`）
  - `Λ_min(F)`：Minimum-Ball 项
- **real 面**（ψ=1 且过 Minimum-Ball）= 最终网格；**imaginary 面**补全 tessellation 但丢弃。
- 这取代了 DMesh 的全局 WDT / 幂图 + 每点权重公式，换成纯**局部**球测试。

### 2. Minimum-Ball（Def 3.1）
- `F ∈ 𝔽_min` iff 没有 ℙ 中的点严格落在 `B_F`（过 F 顶点的最小外接球）内。
- 可微化用有符号距离：
  `d(B_F,ℙ) = B_F^r − min_{p∈ℙ−F} ‖p − B_F^c‖`，`Λ_min(F) = σ(d·α_min)`
- 复杂度 `O(|F|·log|ℙ|)`（GPU KNN，缓存 K=10 邻居，每 n₁=50 步刷新），对比旧 WDT 的 O(|ℙ|)。

### 3. Reinforce-Ball（自适应分辨率核心）
- 真正的 **REINFORCE / policy-gradient** 估计器，不是 loss 梯度，也不是启发式。
- 每点一个 Bernoulli 存在概率 `ϕ(p)`；采样配置 `P(𝐏|Φ)=Π ϕ·Π(1−ϕ)`。
- 目标 `E_{𝐏∼Φ}[L_rl]`，`L_rl = L_recon + ε_card·L_card`，`L_card=|𝐏|`（点数）。
- 用 log-derivative trick 在 B 个采样批上估梯度。
- **信号 = "去/留这个点，对 recon loss 的影响 vs 基数惩罚"**：不帮重建的点被剪，帮的地方保留（等效致密）。
- 存在原因：DMesh++ 丢了 DMesh 的每点权重，需要新杠杆控局部密度。**论文主要在 2D 演示，标注为 experimental。**

### 4. 3D 重建管线
- **64 视角**，渲染 512×512 diffuse color + depth。
- `L = L_recon + α_qual·L_qual`，L_recon = L1 像素（color+depth），L_qual = 三角形长宽比正则（α_qual=1e-3）。
- 初始化：**body-centered cubic 晶格**（不是点云）。
- 每 epoch 两阶段：位置优化（n₀=2000 步）→ real 值优化（α_real=1e-4）。
- **epoch 之间做 face-splitting 细分**——这才是 3D 里真正用的自适应分辨率机制（不是 Reinforce-Ball）。
- benchmark：Thingi10K 10 闭 + 10 开曲面。Adam（代码里）。

### 5. 流形性 — 不保证
- 输出一般**非流形**。流形化是可选后处理，高分辨率配置**跳过**（太贵）。
- 显式支持开曲面 → 可能有非流形边、洞、非水密。
- **我们的保证流形输出仍是真差异化。**

## 可移植 vs 不可移植

### ✅ 可移植
1. **每区域"细化换 loss 下降"的信号**。Reinforce-Ball 核心（Bernoulli 采样 keep/refine，用 REINFORCE 量 ΔL_recon vs 基数惩罚）与表示无关。可用同一信号决定**在哪触发流形安全的 extrude/stellate/subdivide**。直击我们的 IoU 天花板（本质是 where-to-refine 问题）。
2. **epoch 间 face-splitting 细分**——已经流形安全，是 DMesh++ 3D 自适应分辨率的实际来源。
3. **每面渲染 depth/silhouette 误差**作为细化启发式（比完整 REINFORCE 便宜的近似）。

### ❌ 不可移植（绑死在点集 + tessellation）
- Minimum-Ball / WDT / 幂图面存在性——前提是连通性由点集推导，与我们固定半边拓扑相反。
- real/imaginary 面 + ψ gating——假设面可自由生灭，破坏流形保证。
- Reinforce-Ball 的点**删除**——删任意点导致非流形；只有**决策信号**可迁移，落到我们的算子上。

## 与我们已确认结论的交叉验证

- **分辨率假设已被我们独立证实**（2026-08-26）：cow 256px 下 cc2/C=0.9477±0.0051 → **cc3/C=0.9692±0.0052**（+0.0258，决定性越过 0.94 天花板）。cc3 baseline 单独（0.9588）就打过 cc2 全套 extrude 机制（0.9477）。
- 这与 DMesh++ 的核心教训一致：**3D 里真正提升来自 face-count / 分辨率预算**（他们用 face-splitting 细分），而不是纠结单个算子类型。
- 我们的方向 = 用**局部误差驱动**的流形算子细化，作为均匀 3×CC 的高效替代（只在误差高的区域花 face 预算），而非全局堆分辨率。

## 结论：下一步方向

1. 把均匀 cc3 作为**已验证的强 baseline**（0.9692）。
2. 实现**局部误差驱动细化**：per-face 渲染损失下降信号 → 流形 stellate/subdivide 高误差面，剪掉平坦区（不细化）。目标：用远少于 cc3 的 face 数达到相近 IoU。
3. 提高视角数（6 → 16/32），对齐 DMesh++ 的 64 视角杠杆（便宜且方向一致）。

---

## 实验落档：REINFORCE 负结果 & 「预算饱和」洞见（2026-08-27）

> 数据文件：`experiments/opseq_v5/eval_out/reinforce_h2h_cow.json`
> 前置：`local_refine_cow_final.json`（960F 紧预算结果）

### 背景
老板判定「便宜误差启发式只是『当前误差』代理，不是真『边际收益』」，明确要求效果优先、成本不限，选择忠实移植 DMesh++ 的 Reinforce-Ball（REINFORCE 策略梯度）。为避免自欺，做 **iso-face-budget（~1920F）head-to-head**：REINFORCE vs probe-and-keep 贪心 vs 均匀 cc3 vs 便宜启发式。

### 结果（cow, 800 steps, 4 trials, iso-budget ~1920F）

| 变体 | IoU (mean±std) | Faces | 时间/trial |
|------|----------------|-------|-----------|
| Biso（便宜误差启发式） | 0.9588±0.0030 | 1920 | 11s |
| P（probe-and-keep 贪心） | 0.9586±0.0055 | 1590* | 14s |
| cc3（均匀 3×CC） | 0.9577±0.0011 | 1920 | 12s |
| **R（REINFORCE, Eq.8）** | 0.9577±0.0028 | 1920 | 21s |

\* P 更挑剔，拒绝了部分 stellate，未填满预算。

**全部两两差异 < 0.02 噪声地板**（最大差 Biso−cc3=+0.0011，是地板的 1/18）→ 四者统计上不可区分。用 `welch_significant(noise_floor=0.02)` 判定全为 NOISE。

### REINFORCE 负结果（确认）
- ϕ 在 5 步 RL 后基本不动（0.498–0.502），Bernoulli 采样退化成均匀——**没学到策略**。720 候选面 × 每触发仅 5 步 RL，信噪比太低。
- 2× 成本换零增益。**与 DMesh++ 论文自身在 3D 弃用 Reinforce-Ball（仅 2D 演示、标 experimental）的结局一致。** 之前写在本文档「不可移植」一节的警示应验。
- 结论：**REINFORCE where-to-refine 在本设定下砍掉**。便宜误差启发式（Biso）是务实赢家：同 IoU、最快、最简单。

### 「预算饱和」洞见（关键，防误读）
这**不是**「自适应密度失败」，而是「cow@1920F 预算已饱和」：

- 紧预算 960F 下（`local_refine_cow_final.json`）：自适应 B=0.9462 用 **1/3 新增面补回 53% 的 cc2→cc3 差距** —— 自适应此时**有**价值。
- 饱和预算 1920F 下：所有策略都到 ~0.958 顶点，face 放哪都一样 —— where-to-refine 变 moot。

**规律：face 往哪放，只在预算稀缺时才有意义。** cow 在 1920 面下每区域都够用，放置策略自然失效。

### 尚未回答的问题
自适应密度的真正价值需在「**紧 face 预算 + 细节集中型形状**」上验证——光滑主体 + 局部高频特征（尖角/凹陷/薄结构）。cow 太均匀圆润，不是好靶子。均匀细分在细节集中型形状上浪费预算，自适应应能把面砸在特征上。**这是下一个判决实验，尚未做。**

---

## 实验落档：Phantom 梯度 & 局部-IoU-亏损 放置 双双 FAIL（2026-08-28）

在 REINFORCE 之后，又用两个更强的 where-to-refine 信号做了 H1 审判（"自适应放置 > 随机放置" 同 face 数），全部失败。

### 方法 3 — Phantom 顶点梯度（FAIL）
每个面插一个 ε-抬升幽灵顶点（`a_f=c_f+ε·n_f`），单次 backward，打分 `|∇_apex·n_f|/(A_f+eps)`，top-K + 1-ring NMS。意图：直接问「这里给一个 DOF，loss 梯度想不想用它」。
- cow@800F：PH 0.9451±0.0035 vs RND 0.9417±0.0021（名义微赢，但 < 地板）
- armadillo@800F：PH 0.9552±0.0025 vs **RND 0.9577±0.0045（随机名义赢）**
- **根因（读日志发现）**：plateau 从未真触发，每轮 refine 都是 80 步安全阀强制 → phantom 梯度测的是「优化半途」而非「收敛点」，信号被污染。判 confound，不算数。

### plateau 信号诊断（关键副产品）
`diag_plateau.py`：位移是**坏**的收敛信号——它被 cosine LR schedule 主导（一路降到 step 1700），而 loss 在 ~step 800 就压平、IoU 在 ~1200 见顶。位移到 5e-5 阈值要等到 step 1393，且 natural floor 才 4.4e-6。**正确的收敛信号是 loss-EMA stall，不是位移。**

### 方法 4 — 局部-IoU-亏损 @ 收敛（LIOU，FAIL，但这是公平审判）
Boss 的想法：不用位移、用**每面最差视角的 (1−local_IoU)+λ·depth_err** 当「哪里加」信号；触发改为 **loss-EMA 相对 stall**（α=0.95，50 步窗，<1% 变化）；stellate 后 **partial warm-start Adam**（存活顶点保动量，仅新 apex 清零）。修掉了 plateau confound。
- armadillo@800F：**LIOU 0.9568±0.0108 vs RND 0.9630±0.0054** → Δ=−0.0062 → NOISE → **H1 FAIL**
- **validity gate 全过**：8/8 trial 均真 stall 触发（`triggers=[stall×8]`），零安全阀。这次 confound 全移除、给足 2400 步——**是最公平的一次审判**。随机不仅名义赢，方差还更小。

### 累计结论：4 个方法全败
REINFORCE / 便宜启发式 / Phantom 梯度 / 局部-IoU-亏损 —— **没有任何放置策略在同 face 数下统计显著打赢随机/均匀**。前三次有 confound（预算饱和 / plateau 坏），第四次 confound 全清仍败。

> **在光滑/圆润形状（cow、armadillo）上：face 放哪不重要，只有放多少重要。**

### 最后一枪：cow@800F（进行中）
armadillo/cow 均为均匀团块。探针（600 步 depth baseline）显示唯一「面数真有用」的形状是 **cow**：cc2(480F)=0.9240 → cc3(1920F)=0.9506，增益 **+0.0266 > 地板**；而 airplane 4× 面数仅 +0.0036（拓扑接不住薄翼，淘汰）、bunny 480F 已饱和。故用 **cow@800F（面饿判别区，细节集中于角/腿/尾）** 做最后判决实验：LIOU vs 随机，6 trials，2400 步。`liou_h1_cow.json` 待回填。

---

## 实验落档：放置问题终审 + 毛刺歼灭战（2026-08-29）

### 最后一枪结果：cow@800F H1 FAIL + LIOU_EX（extrude）也 FAIL
- cow@800F：LIOU 0.9464 vs RND 0.9494±0.0053 → NOISE → **H1 FAIL（第 5 败）**
- LIOU_EX（stall 时 ray-voting 判拓扑缺口 → `topmod_extrude_cluster`，MAX=3）：extrude 确实长到了牛腿，但增益 ≈ face 预算成本，净零 → **FAIL（第 6 败）**
- **放置问题正式关案**：6 种方法（REINFORCE/启发式/Phantom/LIOU×2/LIOU_EX）无一在同 face 数下打赢随机。**只有总 face 数有意义。**

### 收敛 face 阶梯（cow，2400 步收敛值，非 600 步探针）
| 网格 | F | IoU |
|------|---|-----|
| cc2 | 480 | 0.9389±0.0029 |
| cc3 | 1920 | 0.9662±0.0007 |
| cc4 | 7680 | 0.9788–0.9804 |

每 4× face 砍掉 ~40% 的 gap-to-1.0；256px 下天花板 ≈0.985。
（注意：600 步探针曾误导出「800F≈1920F」的效率结论，收敛后被推翻——**探针只能用于筛形状，不能用于下效率结论**。）

### 毛刺（针刺/皮瓣）歼灭战 — 8 连试，最终配方 4 段
**症状**：cc4 直训 IoU 0.9804 但满身发丝状毛刺。诊断（face-ID 光栅化归因）：
毛刺 = **fold-over 皮瓣**，全网格 23%（2636/11520）邻面对 normal dot<0。

失败路径（全部落档防重蹈）：
1. L4 laplacian W=200 — 毛刺仍在
2. long-edge 惩罚 — 长边清零但毛刺仍在
3. sliver 高度惩罚（皱褶期）— 无效
4. fold penalty 全程 W=0.05/0.5 — **更糟**（0.9017），皱褶在承重
5. 事后全局抹平 hard folds — IoU 0.9804→0.8742，皮瓣带着真覆盖
6. settle 配方单独用（LR warmup+lap boost+depth phase-in 每次细分后）— 不够
7. 轮廓外「焚烧」损失单独用 — 只清轮廓外，轮廓内几何针测不到
8. **四段组合 = 成功**（见下）

**最终配方（v8，总耗时 <3 min）**：
1. **C2F 渐进细分** cc2(800步)→中点细分→cc3(800)→细分→cc4(800)：皱褶从源头不产生（hard folds 2123→914）
2. **cc4 末段 fold penalty** W=0.02 从 step400 ramp-in（低 LR 段皱褶不再承重）：hard folds→73，IoU 无损
3. **焚烧损失**：W=20~60 重罚 3-dilated-GT 外的预测覆盖，续训 1200 步：轮廓外毛刺 1690→~120px
4. **针刺钳制 + sliver 抛光**：`relu(lap−0.85·mean_edge)²`×400 + sliver 高度/长边惩罚×2000，600–800 步低 LR：几何针(lap/edge>1.0) 182→**0**，sliver(aspect>30) 859→**0**，IoU 回满 **0.9803**

**根因与 2026-08-22 depth phase-in 修毛刺同源**：depth L1 压在未安顿几何上。
**验收教训**：轮廓外像素指标看不见轮廓内几何针——毛刺验收必须做
(a) lap/edge census (b) sliver aspect census (c) 3D 渲染目检。剩余细突起须对照 GT 确认是否真实解剖（牛角/耳/尾）。

### 工程注意
- `pipeline/geometry_optimizer.py:129` `normal_consistency_loss` 是 **O(V²) 内存**（`torch.zeros(V*V,3)`），cc4 级别禁用。
- 复现脚本（/tmp，未入库）：`cow_c2f.py`、`cow_c2f_v2.py`（+fold pen+外科后修）、`cow_v3_burn.py`（焚烧）、v7/v8 内联（针刺+sliver）；产物 `/tmp/liou_cow_viz/cow_v8.obj`。

### 毛刺终章：盲区视角审计 + 22 视角焚烧（v9，2026-08-29）
v8 在 6 个训练视角下干净，但 **16 个 held-out 视角审计暴露 18331 毛刺像素**（maxblob 1571）——毛刺全藏在训练视角之间的盲区，6 视角 loss 对它们零约束。这就是「指标干净但渲染有刺」的最终答案。

**v9 = v8 + 22 视角焚烧**（6 训练 + 16 held-out，GT 网格渲染盲区监督，W_BURN=40，1000 步）：

| 网格 | IoU22（真 3D 保真度） | train6 | heldout16 | ho 毛刺px |
|------|------|--------|-----------|-----------|
| cc4 直训 | 0.8874 | 0.9802 | 0.8526 | 41777 |
| v8 | 0.9208 | 0.9802 | 0.8985 | 18331 |
| **v9** | **0.9492** | 0.9688 | **0.9418** | **1489** |

train6 掉 0.011 是假损失——那部分覆盖本来就是盲区里的鼓包/毛刺。**评估教训（最重要）：多视角拟合的验收必须用 held-out 视角 IoU，训练视角 IoU 会系统性高估并藏住盲区伪影。**
产物：`despike/cow_v9.obj`、`cow_v9_final.png`（GT/旧/新 四角对照）。

### 正解：v10 — 纯 6 视角训练中杜绝毛刺（2026-08-29，Boss 裁定方案）
Boss 否决 v9 的多视角焚烧（用了训练外 GT 信息 = 作弊）。正解是**训练中几何先验**：
盲区毛刺在 6 视角下与合法几何信息不可分，唯一正当防御是先验禁止针状几何。

**v10 配方（从零训练，只用 6 视角，2400 步，41s）**：
- C2F 渐进细分（cc2→cc3→cc4，各 800 步）
- 细分后 settle（LR warmup 20 + lap boost 0.40×50 步 + depth phase-in 80 步）
- **spike 钳制（`relu(lap−0.85·mean_edge)²`×400）+ sliver 惩罚（高度+长边 ×2000）全程在线**
- cc4 末段 fold penalty W=0.02

**结果**：train6 IoU=**0.9827**（超过直训 0.9804）；spikes(lap/edge>1.0)=**0**、
slivers(aspect>30)=**0**（直训分别为 205/859）；四任意角度渲染目检无一根毛刺。
held-out16 覆盖 IoU=0.8714——差距是**盲区鼓包**（形状信息极限，6 视角原理上不可解），
不是毛刺。毛刺（针状伪影）与鼓包（形状误差）必须区分：前者可由先验杜绝，后者需要更多视角信息。
产物：`despike/cow_v10.obj`、`cow_v10_check.png`。**v10 是默认配方；v9 仅当允许用全视角 GT 时使用。**
