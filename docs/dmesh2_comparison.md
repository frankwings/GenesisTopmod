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
