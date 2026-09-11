# nvdiffrast Backward 非确定性问题记录

> 记录日期：2026-08-25
> 状态：**已确认**（源码 + 官方 GitHub Issue 双重验证）

## 问题现象

在 `experiments/opseq_v5` 的 error-driven extrude 优化中，**同一份代码、同一输入、无任何随机种子改动，多次运行 IoU 结果不同**：

- cow eval 连跑 3 次，champion C = 0.9334 / 0.9405 / 0.9517
- run-to-run IoU 方差 ~1.8%
- 曾经"消失的 0.9519 冠军" 实为方差（Run3=0.9517 已复现），**不是丢失的特性**

这个方差会被离散注入决策（vote argmax、plateau 触发）进一步放大，导致单次运行的变体对比不可靠（方差 ~1.8% > 许多声称的差异）。

## 根本原因：nvdiffrast antialias 反向 kernel 的浮点 atomicAdd

**不是我们的 bug，不是 RNG，是 CUDA 实现的固有特性。**

### 调用链
1. **调用点**：`pipeline/geometry_optimizer.py:60` — `render_silhouette` 里 `dr.antialias(color, rast, pos_clip, faces)`，为轮廓边界提供可微性。
2. **Python 包**：nvdiffrast 0.4.0，`/home/kingy/.local/lib/python3.12/site-packages/nvdiffrast/`；包装器 `torch/ops.py` 的 `_antialias_func.backward` → 编译好的 `_nvdiffrast_c.antialias_grad`（预编译二进制）。
3. **CUDA 源码**：`/home/kingy/Projects/TextureMap/nvdiffrast/csrc/common/antialias.cu`

### 确切出处：`AntialiasGradKernel`（antialias.cu:387-556）
两处 float atomicAdd 散射：
```cuda
// 颜色梯度 (line 459-460)
atomicAdd(&pGrad0[i], -v);
atomicAdd(&pGrad1[i],  v);

// 顶点位置梯度 (line 553-554) —— 打到我们优化变量上的那个
caAtomicAdd3_xyw(p.gradPos + 4 * vi1, gp1x, gp1y, gp1w);
caAtomicAdd3_xyw(p.gradPos + 4 * vi2, gp2x, gp2y, gp2w);
```

### 为什么必然不确定（三条源码证据）
1. **持久线程 + 动态取工作项**（line 395-404）：`for(;;)` 循环用 `atomicAdd(&workBuffer[0].y, ...)` 动态领取工作项，哪个 block 先处理哪条边完全取决于运行时调度，无固定顺序。
2. **多条边写同一顶点**：一个顶点被多条轮廓边共享，每条边算出 `gp*` 后累加到同一 `gradPos` 槽。浮点加法非结合律 → 累加顺序变则 bit 级结果变。
3. **`caAtomicAdd3_xyw` 只在 warp 内 coalesce**（line 548-550）：跨 warp/block 仍是裸 atomicAdd 竞争，coalescing 只减少冲突数量，消不掉跨 block 顺序不确定性。

### 对照 forward（bit-identical）
forward kernel 的 atomicAdd（line 201/205/228）只做**整数** work-buffer 计数分配（顺序无关），line 371 输出写不重叠像素。实测 forward Δgrad=0，backward max|Δgrad|=288.6（`/tmp/nondet_probe.py`）。

## 官方确认：GitHub Issue #13

nvdiffrast 作者 **Samuli Laine (s-laine, NVIDIA)** 亲口证实：

- **atomic 是根源**："nvdiffrast uses atomic operations and they almost certainly lead to some discrepancies between runs."
- **无法关闭 / 无 deterministic 选项**："Being fully deterministic would in practice require removing atomics... So the answer is no."
- **依赖优化稳定性放大**："Even small differences can amplify if the system is dynamically unstable."
- **他实测数字与我们一致**：res=256 时 run-to-run 差异 "at worst 3.5% but only about 1.7% on average"（我们测到 ~1.8%）。
- **另一非确定源 = topology hash**：仅当非流形（>2 三角形共边）时。我们是流形 icosphere，不受影响。

链接：https://github.com/NVlabs/nvdiffrast/issues/13

## 应对措施

**seed 无效**（是原子累加顺序，不是 RNG）。正确应对：

1. **评测侧（必须）**：多试统计——跑 N 次，报 mean±std，把 <0.02 的差异当噪声，不作为结论依据。这是 Phase 2 step 3（加 operator menu）的前置条件。
2. **优化侧（可选）**：作者建议
   - 增加迭代数 / 更激进的 lr rampdown，让优化真正收敛（收敛后绝对差异趋近 0）
   - mesh regularization（我们已有 W_LAP=0.10）
   - 直接优化顶点位置本就是病态的，需要好的参数化/正则

## 关联结论

extrude-only 天花板 ~0.94（无论 VLM 还是手写 C 选择），在方差范围内区分不出选择器优劣。**要突破必须加新 operator（add_handle / stellate / subdivide）**，而不是纠结当前变体的小数点差异——这些差异大多在噪声内。
