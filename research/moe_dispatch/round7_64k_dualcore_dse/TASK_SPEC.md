# 任务：64K 乘法器下单核 / 同构双核 / 一大一小双核的 DSE，以及访存需求反推

## 0. 背景与目标

我们研究 MoE 解码时计算核形状对性能的影响。负载是 DeepSeek-V2-Lite 一层 MoE。总乘法器固定为 PLENA 原版的 65,536 个，比较三种组织：

1. **单核**：一个核，65,536 个乘法器
2. **同构双核**：两个形状完全相同的核，各 32,768 个
3. **一大一小双核**：两个形状不同的核，乘法器分配比例也作为搜索变量

做两件事：

- **第一步**：在理想访存（HBM 和 SRAM 带宽不受限）下，为每种组织搜出最优 shape
- **第二步**：对搜出的 shape，反推要跑到这个速度需要多少 HBM 带宽和 SRAM 带宽，并给出参考带宽下的取数时间

**重要规则**

- 不要编造任何数据。
- 找不到路由 trace，或 trace 格式无法得到第 2 节要求的信息时，停下来报告，不要用随机数或假设分布代替。
- 所有假设写进结果文档。

---

## 1. 固定条件

| 项 | 值 |
|---|---|
| 精度 | BF16，每元素 2 字节 |
| 主频 | 1 GHz |
| 总乘法器 | 65,536 |
| 数据流 | Output Stationary |
| Batch | B ∈ {2, 4, 8, 16, 64, 96, 128} |
| 模型 | DeepSeek-V2-Lite（请从仓库内的 HF config 核对以下数值） |
| hidden_size | 2048 |
| moe_intermediate_size | 1408 |
| 路由专家数 / top-k | 64 / 6 |
| 共享专家数 | 2（每个中间维度 1408，合起来 2816） |

---

## 2. 输入数据：路由 trace

在仓库中找到之前实验使用的 **135 个留出窗口**的路由 trace。

DSE 需要的信息是：对每个窗口 w、每个 batch B，64 个路由专家各分到多少个 token，记为 M_e(w, B, e)。

- 先在结果文档里说明：trace 文件路径、原始格式、每个窗口代表什么（哪一层、哪个时刻），以及每个 batch 的 B 个 token 是怎么从窗口中取出的。
- 沿用之前实验的取法，不要自己改。
- 检查：对每个 (w, B)，Σ_e M_e 应等于 B × 6。不满足时报告。

---

## 3. 任务定义

一个窗口、一个 batch 下，这一层的任务集合为：

- **2 个共享专家任务**：每个 M = B
- **每个 M_e > 0 的路由专家 1 个任务**：M = M_e

每个任务是一个完整专家的三个矩阵乘：

| 矩阵 | 输入 [M, K] | 权重 [K, N] | K | N |
|---|---|---|---|---|
| Gate | [M, 2048] | [2048, 1408] | 2048 | 1408 |
| Up | [M, 2048] | [2048, 1408] | 2048 | 1408 |
| Down | [M, 1408] | [1408, 2048] | 1408 | 2048 |

**粒度规则**：一个任务（一个专家的 Gate、Up、Down）必须整体放在同一个核上，不跨核拆分。SiLU 和逐元素乘由向量单元完成，不计入耗时。

---

## 4. 计算核与单任务耗时

计算核形状记为 P_M × P_N × G：

- **P_M**：每周期同时处理的行数（token 数，对应矩阵的 M 维）
- **P_N**：每周期同时处理的输出列数（对应 N 维）
- **G**：每个输出每周期累加的乘积个数（对应 K 维，加法树深度）

乘法器数 = P_M × P_N × G。

一个矩阵乘 [M, K] × [K, N] 在该核上的周期数：

```
cycles = ceil(M / P_M) × ceil(N / P_N) × ceil(K / G) + overhead
```

一个任务的周期数 = Gate + Up + Down 三者之和。

- `overhead` 是每个矩阵乘的固定流水线开销，参数化，默认 0。另跑一次 overhead = 16 做敏感性检查。
- **如果仓库里已有之前实验使用的周期级模拟器或耗时函数，优先复用它**，并在文档中说明用的是哪一个。

---

## 5. 搜索空间

**统一约束（三种组织完全相同）**

- G 固定为 64
- P_N 必须是 16 的倍数，且 P_N ≤ P_N_cap
- P_M 为正整数，P_M × P_N × 64 必须恰好等于该核的乘法器数
- P_N_cap 跑两组：**256（主实验）** 和 **512（敏感性分析）**

**三种组织**

1. **单核**：一个核，65,536 个乘法器，枚举所有合法形状。
2. **同构双核**：两个核各 32,768 个，形状相同，枚举所有合法形状。
3. **一大一小**：
   - 小核乘法器 S ∈ {8,192, 16,384, 24,576, 32,768}，大核 = 65,536 − S
   - 两个核各自枚举合法形状
   - S = 32,768 时，排除两核形状相同的组合，因为那就是同构双核

**参照点（不参与搜索，只列入结果表）**

- PLENA 原版单核 32 × 32 × 64
- 旧实验的四个形状：单核 4×256×64、同构 2×256×64 ×2、异构 4×128×64 + 1×512×64、纯流式同构 1×512×64 ×2
- 旧异构和纯流式同构在 cap = 256 下不合法，只作参照，表中注明

候选总数只有几百个，全部穷举，不用启发式搜索。

---

## 6. 调度与一层的延迟

- **单核**：延迟 = 所有任务周期数之和。
- **双核**：任务之间没有依赖，每个任务分给一个核。一层的延迟 = 两个核中较晚完成的那个（makespan）。

**DSE 阶段使用最优调度**：精确求解使 makespan 最小的任务分配。这是两台不同机器的最小化完工时间问题，任务数 ≤ 66。可以用按核 0 负载做的伪多项式动态规划，或用 OR-Tools CP-SAT 精确求解。用小规模暴力枚举验证正确性，写成单元测试。

**贪心调度（只在选出最优 shape 之后评估）**：

1. 任务按"在两个核上耗时的较小值"降序排序
2. 依次把每个任务分给能最早完成它的核（当前负载 + 该任务在此核上的耗时，取较小者）

报告贪心相对最优的延迟差距。

---

## 7. DSE 目标

1. 对每个候选、每个 batch：对 135 个窗口的延迟取**算术平均**，得到该 batch 的延迟 T(B)。
2. **目标函数**：7 个 batch 的 T(B) 的**几何平均**，越小越好。
3. 每种组织选目标函数最小的候选作为最优 shape。
4. **参照**：另外对每个 batch 单独选出最优 shape，用来说明"一个设计兼顾所有 batch"相对"每个 batch 用各自最优"的损失。

---

## 8. 输出指标定义

以下指标对每个窗口计算，再对 135 个窗口取算术平均。时间统一用 µs，带宽用 TB/s（1 TB/s = 10¹² B/s）。

**计算相关**

| 指标 | 定义 |
|---|---|
| **Compute Time (µs)** | 理想访存、最优调度下这一层的延迟 |
| **Useful MACs** | Σ_任务 Σ_三个矩阵 M × K × N（真实维度，不含填充） |
| **Equivalent FLOPs (GFLOP)** | 2 × Useful MACs ÷ 10⁹ |
| **Equivalent Throughput (TFLOPS)** | Equivalent FLOPs ÷ Compute Time |
| **MAC Utilization** | Equivalent Throughput ÷ 峰值 131.072 TFLOPS |

**HBM → SRAM（权重不常驻 SRAM，每层权重每次都从 HBM 读）**

| 指标 | 定义 |
|---|---|
| **Active Experts** | M_e > 0 的路由专家个数 |
| **Weight Bytes (MB)** | (2 + Active Experts) × 3 × 2048 × 1408 × 2 B，即每个专家约 17.30 MB。激活和输出的 HBM 流量很小，忽略，并在文档中注明 |
| **HBM Equivalent Fetch Bandwidth (TB/s)** | Weight Bytes ÷ Compute Time。含义：要让计算核不等数据所需的 HBM 带宽 |
| **HBM Fetch Time (µs)** | Weight Bytes ÷ B_ref_HBM。B_ref_HBM 默认 4 TB/s（MemExplorer 基线配置 HBM3E × 4），参数化 |
| **Roofline Latency (µs)** | max(Compute Time, HBM Fetch Time)。对应 MemExplorer 中双缓冲时计算与取数重叠的模型 |
| **Bound** | Compute Time ≥ Fetch Time 记为 compute-bound，否则为 memory-bound |

**SRAM → 计算核**

| 指标 | 定义 |
|---|---|
| **SRAM Peak Feed (TB/s)** | 每个核每周期读 P_N × G 个权重 + P_M × G 个激活。对所有核求和后 × 2 B × 1 GHz。只由 shape 决定 |
| **SRAM Bytes per Layer (MB)** | Σ_任务 Σ_矩阵 [cycles × (P_N × G + P_M × G) × 2 B]，按硬件实际读宽计，包括填充的槽位。overhead 周期不计读取 |
| **SRAM Equivalent Fetch Bandwidth (TB/s)** | SRAM Bytes per Layer ÷ Compute Time |
| **SRAM Fetch Time (µs)** | SRAM Bytes per Layer ÷ B_ref_SRAM。B_ref_SRAM 默认 4 TB/s（MemExplorer 表 II 单个 SRAM die），参数化 |
| **SRAM Die Equivalent** | SRAM Equivalent Fetch Bandwidth ÷ 4 TB/s，即需要几个 MemExplorer 标准 SRAM die 的带宽 |

---

## 9. 结果表格（每张表同时输出 Markdown 和 CSV）

主实验 P_N_cap = 256。另外为 cap = 512 输出表 1、表 2 和表 7。

**表 1：硬件配置汇总**

列：组织 | 核 1 形状 | 核 2 形状 | 乘法器分配 | 峰值 TFLOPS | SRAM Peak Feed (TB/s) | 7-batch GM Compute Time (µs)

行：单核最优、同构最优、一大一小最优，以及第 5 节的所有参照点。

**表 2：各 batch 计算性能（三种最优组织）**

列：Batch | 组织 | Compute Time (µs) | Equivalent FLOPs (GFLOP) | Equivalent Throughput (TFLOPS) | MAC Utilization | 相对单核加速比 | 相对同构加速比

末尾加 GM 行。

**表 3：HBM 取数（三种最优组织）**

列：Batch | 组织 | Active Experts | Weight Bytes (MB) | Compute Time (µs) | HBM Equivalent Fetch Bandwidth (TB/s) | HBM Fetch Time (µs) | Roofline Latency (µs) | Bound

**表 4：SRAM 取数（三种最优组织）**

列：Batch | 组织 | SRAM Peak Feed (TB/s) | SRAM Bytes per Layer (MB) | SRAM Equivalent Fetch Bandwidth (TB/s) | SRAM Fetch Time (µs) | SRAM Die Equivalent

**表 5：每个 batch 单独最优的 shape**

列：Batch | 组织 | 该 batch 最优形状 | 该形状 Compute Time | 7-batch 最优形状在此 batch 的 Compute Time | 损失 %

**表 6：贪心调度与最优调度对比（双核两种组织）**

列：Batch | 组织 | 最优调度 Compute Time | 贪心 Compute Time | 差距 %

**表 7：Top-10 候选**

每种组织按目标函数列出前 10 个候选，用于判断最优解是否明显领先。

**表 8：参考带宽（直接引用 MemExplorer arXiv:2604.16007 表 II，不计算）**

| 存储 | 每单元容量 | 每单元带宽 |
|---|---|---|
| SRAM（1 die） | 256 MB | 4 TB/s |
| 3D 堆叠 SRAM（1 层） | 1 GB | 8 TB/s |
| HBM3E 8H（1 stack） | 24 GB | 1 TB/s |
| HBM4 12H（1 stack） | 36 GB | 2 TB/s |

**表 9：与旧结果的一致性检查**

用旧实验的四个形状重新跑 cap 无关的评估，与旧数据对比，逐 batch 列出新旧值和偏差 %。

旧值（单位 ms）：

| Batch | 单核 4×256×64 | 同构 2×256×64×2 | 异构 4×128×64+1×512×64 |
|---|---|---|---|
| B2 | 0.007606 | 0.003834 | 0.003819 |
| B4 | 0.012895 | 0.007198 | 0.006108 |
| B8 | 0.017922 | 0.012020 | 0.010827 |
| B16 | 0.029931 | 0.021830 | 0.020516 |
| B64 | 0.088412 | 0.077105 | 0.075633 |
| B96 | 0.124548 | 0.112607 | 0.111226 |
| B128 | 0.160710 | 0.148366 | 0.146526 |

偏差超过 5% 时，分析原因（耗时函数、调度规则、trace 取法），写进文档，不要强行调参去拟合。

---

## 10. 交付物

1. **代码**：模块化，分为 trace 读取、耗时模型、调度器、DSE、指标计算、出表。所有参数集中在一个配置文件里，包括 P_N_cap、overhead、B_ref_HBM、B_ref_SRAM、batch 列表。
2. **单元测试**：
   - 耗时公式的手算例子
   - 最优调度与暴力枚举在小规模上一致
   - Σ M_e = 6B 检查
3. **`results/` 目录**：所有表的 CSV 文件，以及 `results.md`，汇总全部表格。
4. **`results.md` 开头写清楚**：
   - trace 来源与取法
   - 使用的耗时函数
   - 所有假设
   - 一段不超过 10 句话的结论，按以下顺序：
     1. 理想访存下哪种组织最快、快多少
     2. 它需要的 HBM 和 SRAM 带宽分别是多少，是另外两种的几倍
     3. 在参考带宽下各 batch 是 compute-bound 还是 memory-bound
     4. cap = 512 时结论是否改变