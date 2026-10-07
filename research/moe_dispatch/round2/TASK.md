# 任务说明：MoE 大小核 NPU 第二轮完整评估（单分支，全部跑完）

你在 PLENA 研究仓库工作，代码在 `research/moe_dispatch/` 下。本文件替代之前所有的 round2 说明。
从第 1 节到第 10 节按顺序执行，全部跑完，并交付第 11 节列出的所有结果。
RTL 本轮不做。

---

## 0. 全局规则（每一节都必须遵守）

### 0.1 精度与 HBM
- 只用 BF16。不跑 W4 / W8，任何表里都不出现。
- HBM 用 HBM2 配置：标称 256 GB/s，延迟 64 周期，请求 32 B。
  - 主设定为 256 信用，推得供数上限约 126 GB/s。
  - 512 信用（约 252 GB/s）只用于标明的敏感性分析。

### 0.2 输入
- DeepSeek-V2-Lite 已捕获的真实路由（BFCL / GPQA / SWE），H=2048，routed F=1408，Shared F=2816，top-6。
- 沿用已有的 18 个开发窗口和 135 个留出窗口，不改划分。
- batch 分组：B2、B4、B8、B16、混合窗口 64 / 96 / 128。
- 合成路由只允许在第 6 节（负载区域图）使用，且必须单独标注。

### 0.3 等资源约束
- 主乘法器严格等于 12,288。
- 存储 2,158,592 B。
- W / X / 累加 bank 总数分别为 64 / 24 / 12，每个 bank 16 B/cycle。
- 共享同一个 HBM。
- 两核之间，乘法器、私有缓冲（W / X / 累加 / Z）、各类 bank 三者各自独立切分，只约束总量。

### 0.4 基线
每张对比表都必须同时出现以下三个：
- **B0**：PLENA 原形状单核 `6x4x512`，仅作参考。
- **B1**：形状调优的单核，这是主基线。
- **B2**：调优的同构 3+3（两核形状相同）。

### 0.5 指标
- 主指标：留出集上"每个窗口配对延迟比"的几何平均，按 batch 分组报告，同时给出全部窗口的几何平均。
- 不用总 ms 之和选设计，也不用它下结论。

### 0.6 判定门槛
- **进入校准**：相对 B1 和 B2 都快至少 5%。
- **宣布胜出**：校准后快至少 10%，且配对 bootstrap 的 95% 置信区间下界不低于 5%。
- 本轮没有 RTL 校准，所以只能写"达到 / 未达到进入校准门槛"，不得宣布任何架构胜出。
- 面积 / 能耗档本轮关闭，不得用乘法器数代替面积。

### 0.7 片上时序模式（RTL 未做，用以下三种模式代替）
用全局开关 `--onchip-mode` 切换：

| 模式 | 含义 |
|---|---|
| `pipelined` | 每拍可以发射一次，速度只受 bank 端口带宽限制。这是当前 3D 模型的做法。 |
| `port_tight` | W 端口总带宽约 134.7 B/cycle，等效于 Rust 的 30.4 ns / 4 KiB tile，在各核之间切分，总量不变。 |
| `fixed_issue` | 每次发射固定 30.4 拍，每个核各有一条流水线。这种模式不满足等资源，只作参考，表里必须注明。 |

- 主表使用 `pipelined`。
- 另外两种模式另附表。
- 第 8 节把"每 tile 片上时间"作为连续参数做全局分析。

### 0.8 可复现
- 每个数字都能用一条命令重新生成。
- 每个结果目录都附 `README.md`（写明命令、提交号、输入哈希）和 `PROVENANCE.csv`。
- 每个配置跑两遍，结果必须一致。
- 旧结果不得修改。

### 0.9 诚实
- 负结果照实写。
- 不得事后更换基线、指标、窗口或门槛。
- 不得把不同 batch 里"赢的格子"拼成一个结论。
- 各项消融的收益不得相加。

---

## 1. 合并到单一分支（最先做，做完才进入第 2 节）

目标：把我（作者 mcl123）在这个仓库里创建的所有研究工作，合并到同一个分支 `research/moe-supply-first-v3`。之后的全部工作都只在这个分支上进行。

1. **列清单**
   - 列出所有本地和远端分支、所有 git worktree（包括 `/scratch/shared/mcl123/plena/worktrees/` 下的），以及每个分支相对 `main` 的提交数和最后提交时间。
   - 只挑出作者是我的分支。
   - 写入 `research/moe_dispatch/round2/BRANCHES_BEFORE.md`。
2. **逐个合并**
   - 把这些分支合并进 `research/moe-supply-first-v3`。
   - 用 `git merge --no-ff`，保留历史，不用 rebase，不用 squash。
   - 有冲突时，保留双方的结果文件，代码按功能取较新的版本。
   - 每个冲突的处理方式写进 `round2/MERGE_LOG.md`。
3. **整理目录**
   - 把散落在各个 worktree 里、未提交但有用的结果目录，复制进 `research/moe_dispatch/archive/<原分支名>/` 后提交。
   - 超过 50 MB 的原始数据不进 git：在 `archive/LARGE_FILES.md` 里记下路径和 sha256。
4. **合并后检查**
   - 全部单元测试通过。
   - 上一轮 BF16/256 的三个选定设计（单核 `6x16x128`、同构 `6x16x64+6x16x64`、异构 `1x2x64+5x19x128`）在新分支上复现，误差为 0。
   - 历史上能精确复现的 945 个窗口依然精确复现。
5. **不删除任何分支或 worktree**
   - 只在 `round2/BRANCHES_AFTER.md` 里列出"已合并、可以删除"的分支，由我手动删。
6. **之后**
   - 不再新建分支。
   - 本轮所有新代码放在 `research/moe_dispatch/round2/` 下（库代码可以放在 `research/moe_dispatch/` 已有模块里，但改动必须向后兼容）。
   - 本轮所有结果放在 `research/moe_dispatch/round2/results/` 下。

**输出**
- `round2/BRANCHES_BEFORE.md`
- `round2/MERGE_LOG.md`
- `round2/BRANCHES_AFTER.md`
- `round2/results/E0/reproduce_check.csv`
  - 列：`design, window_id, batch, prev_ms, now_ms, abs_diff`

---

## 2. E0 统一基准与旧数据来源说明

1. 冻结输入窗口列表、资源账本、HBM 模型，写入 `round2/results/E0/frozen_inputs.json`。
2. 对旧数据源各写一段话：它是什么、可以用于什么、不能用于什么。旧数据源包括：
   - Rust 720 行表；
   - 组会截图表；
   - v6 报告；
   - validation_regime 报告。

   只写说明，不做交叉加速比。

**输出**
- `round2/results/E0/frozen_inputs.json`
- `round2/results/E0/SOURCES.md`

---

## 3. E1 下限与余量（逐窗口）

对每个留出窗口、每个设计（B0、B1、B2、固定 3+3、固定 4+2、上一轮异构）、每种 onchip 模式，计算以下各项：

| 字段 | 定义 |
|---|---|
| `hbm_floor_unique` | 唯一权重字节 ÷ 供数上限 |
| `hbm_floor_actual` | 实际读取字节 ÷ 供数上限 |
| `mac_floor` | 有效 MAC ÷ 12,288 |
| `port_floor` | 必要的最小 W / X / 累加搬运量 ÷ 对应端口的总带宽，取三者最大值 |
| `task_floor` | 单个最大任务在全部资源上的最短时间 |
| `bound` | 以上五项的最大值 |
| `headroom` | 实测 ÷ bound − 1 |

**输出**
- `round2/results/E1/bounds_per_window.csv`
  - 列：`window_id, batch, design, onchip_mode, latency_ms, hbm_floor_unique, hbm_floor_actual, mac_floor, port_floor, task_floor, binding_term, headroom_pct`
- `round2/results/E1/headroom_by_batch.csv`
  - B1 在每个 batch × 模式下余量的中位数和几何平均。
- `round2/results/E1/SUMMARY.md`
  - 列出 B1 余量至少 5% 的格子。异构只可能在这些格子里赢。

---

## 4. E2 数据流对比：OS / WS / IS

### 4.1 三种循环顺序的定义

在模型里按以下定义实现，并写进 README：

- **OS**：部分和留在阵列里，沿 K 方向推进；每个 M 波次都要从 SRAM 重读 W 块。
- **WS**：W 块留在阵列里，依次处理各个 M 波次；部分和按 K 段在累加 SRAM 里读改写。
- **IS**：X 块留在阵列里，沿 N 方向推进；部分和要读改写。组会上说的 RS 在 GEMM 里没有卷积的行复用，所以按 input-stationary 实现，报告中注明"RS = IS"。
- 工作集放不下缓冲时，按实际重读量计入 HBM 和 SRAM 流量。

### 4.2 E2a 单专家实验

- 形状：`6x16x128`、`6x4x512`、上一轮异构的两个核形状、`4x4x512`、`2x4x512`。
- Me ∈ {1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128}。
- 专家类型：routed 和 Shared。
- 每个组合 × 三种数据流 × 三种 onchip 模式都要跑。

### 4.3 E2b 整层 3×3 网格

- 固定形状、资源切分和分派策略，只换数据流（大核 3 种 × 小核 3 种）。
- 对以下设计分别做：固定 4+2、上一轮异构、第 5 节选出的最优异构。
- B1 只做 1×3。

### 4.4 输出
- `round2/results/E2/micro.csv`
  - 列：`shape, dataflow, expert_type, Me, onchip_mode, cycles, issues, w_sram_bytes, x_sram_bytes, acc_sram_bytes, hbm_bytes, refetch_factor, spatial_util`
- `round2/results/E2/layer_grid.csv`
  - 列：`design, df_big, df_small, onchip_mode, batch, geomean_ms, ratio_vs_OS_OS, w_sram_GiB, acc_sram_GiB, hbm_GiB`
- `round2/results/E2/SUMMARY.md`，回答以下问题，每条注明成立于哪几种模式：
  1. Me ≤ 行数时，小核的 OS 是否等于 WS；
  2. 在 Shared 和热专家上，大核用 WS 是否优于 OS，优多少；
  3. "两个核都用 OS"是否在最优的 1% 以内。

---

## 5. E3 证明式 DSE（替代穷举）

本节由五层组成，按 5.1 → 5.5 的顺序完成。每层的产物供后面的层使用。

### 5.1 第 1 层：带下界剪枝的分支定界（给出最优性证明）

**设计空间**
- 组织：单核，或任意两核，两核乘法器之和等于 12,288。
- 每核形状 PM×PN×PK：PM 取 1–16，PN 取 1–192，PK ∈ {32, 64, 128, 256, 512, 1024}，乘积等于该核分到的乘法器数。
- 每核数据流：OS / WS / IS。
- 资源切分（连续区间）：私有 W / X / 累加 / Z 容量，W / X / 累加 bank 数（取整数），vector 宽度。
- 分派：由第 2 层给出（硬件以最优调度下的延迟来评价）。

**分支顺序**
1. 组织（每核乘法器数）
2. 每核 PM
3. PN×PK
4. 数据流
5. 资源切分区间，对半二分，最细到 bank 为 1 个、容量为 1 KiB

**区域下界 LB(R)**

LB(R) 必须是区域 R 内所有设计在每个窗口上延迟的合法下界，取以下各项的最大值：
- `hbm_floor_unique`（与设计无关）；
- 有效 MAC ÷ 区域内乘法器总数；
- 区域内最大单任务在其最快可能核上的最短时间；
- 双核时还要加上：对每个核，"必须分给它的任务"（其他核放不下或明显更慢的任务）的最短时间之和；
- 必要搬运量 ÷ 区域内各端口能拿到的最大带宽；
- 由区域内最小缓冲容量导出的重读下界。

**下界合法性检查（必须做）**
- 随机抽 2,000 个具体设计 × 全部开发窗口，验证 LB 不超过仿真结果。
- 只要有一次违反就停下修正，并在报告中说明。

**目标与剪枝**
- 目标：开发窗口上的配对几何平均。
- 区域下界：对每个窗口的 LB 取几何平均（几何平均是单调的，所以下界依然合法）。
- 当前最优解初始化为 B1。只要叶子上出现更好的设计，就立即更新当前最优。
- 跑两遍：
  - **证明 A**：δ = 5%。若 LB(R) ≥ 当前最优 ÷ 1.05，剪掉 R。这一遍回答"有没有设计能比当前最优快 5%"。
  - **证明 B**：δ = 0。找出模型内的全局最优（允许设置运行时间上限；若到时未完成，报告剩余未剪区域的下界作为最优性差距）。
- 叶子上使用解析模型，配合第 2 层的调度评估。

**输出**
- `round2/results/E3/bnb_certificate.csv`
  - 每个被剪区域一行，列：`proof, region_id, region_desc, lb_geomean_ms, incumbent_ms, pruned_reason`
- `round2/results/E3/bnb_leaves.csv`
  - 所有被完整评估的设计。
- `round2/results/E3/bnb_summary.json`
  - 字段：区域总数、被剪数、覆盖率（必须为 100%，否则写明未完成的部分）、最终最优解、证明 A 的结论、证明 B 的最优性差距。
- `round2/results/E3/lb_validity.csv`
  - 列：`design, window_id, lb_ms, sim_ms, ok`

### 5.2 第 2 层：调度与硬件分离，内层求精确解

对每个硬件设计、每个窗口：

1. **最优分配**
   - 用 MILP（OR-Tools CP-SAT 或 Gurobi）求专家到核的分配。
   - 变量 x_ic ∈ {0,1}，约束 Σ_c x_ic = 1。
   - 每核负载：Σ_i x_ic · d_ic ≤ T。
   - HBM：Σ_i bytes_i ≤ BW · T。
   - 端口：各类端口流量 ≤ 端口带宽 · T。
   - 目标为 min T。
   - d_ic 取任务 i 在核 c 上独占资源时的时间。
   - 解出的 T* 是该硬件在这个窗口上的调度下界。
2. **可执行调度**
   - 把 MILP 给出的分配，按 LPT 顺序放进完整的流式仿真，得到 `T_milp_sched`。
3. **实际 runtime**
   - 用第 7 节的分派策略和预测器跑出 `T_runtime`。
4. **报告两个差距**
   - `T_milp_sched / T*`：衡量模型中带宽耦合与顺序的影响。
   - `T_runtime / T_milp_sched`：衡量 runtime 最多还剩多少可提升空间。

第 1 层的叶子评估使用 `T_milp_sched`。最终表里，`T_milp_sched` 和 `T_runtime` 两种都要报告。

**输出**
- `round2/results/E3/schedule_gaps.csv`
  - 列：`design, window_id, batch, onchip_mode, T_lb, T_milp_sched, T_runtime_<policy>, gap_sched_pct, gap_runtime_pct`

### 5.3 第 3 层：反向搜索负载（异构收益区域图）

问题：在什么负载下，异构才会比调优单核更好？

**负载参数**
合成负载必须与真实 trace 校准：在真实参数点上，distinct expert 数和 Me 直方图的 KL 散度都要报告。

| 参数 | 取值 |
|---|---|
| batch | 2, 4, 8, 16, 32, 64, 128, 256 |
| 路由集中度 | 从均匀到 SWE 级别集中，取 5 档（Dirichlet 浓度，用真实 trace 拟合两端） |
| Shared 规模 | 0、1、2、4 个 expert 当量 |
| 专家数 / top-k | (64, 6)、(128, 8)、(256, 8) |
| 专家宽度 F | 512、1408、2048 |
| 带宽与算力之比 | HBM 126 / 252 / 504 GB/s，或乘法器预算 × {1/4, 1/2, 1, 2}（只缩放算力时，等资源约束随之缩放） |

**做法**
1. 在全网格的每个负载点上，用第 1 层（证明 A，δ 放宽到 2%，以加快速度）加第 2 层，求出 Δ = 最优异构延迟 ÷ 最优单核延迟 − 1。
2. 在网格之外，用 CMA-ES 在参数的连续范围内最小化 Δ，找出异构收益最大的负载，最多 500 次评估。
3. 对找到的最优负载点，用完整精度（δ = 0）重新验证一次。

**输出**
- `round2/results/E3/workload_map.csv`
  - 列：`batch, concentration, shared_units, E, topk, F, bw_or_mac_scale, best_single_ms, best_hetero_ms, best_homo_ms, delta_vs_single_pct, delta_vs_homo_pct, hetero_design`
- `round2/results/E3/workload_extreme.json`
  - 内容：异构收益最大的负载、对应设计、Δ，以及该负载离真实负载有多远。
- `round2/figures/fig_workload_map.pdf`
  - Δ 的热力图（两两参数切片），并标出真实负载所在位置。

### 5.4 第 4 层：稳健目标

对第 1 层的最终候选（各组织族的最优解及其 1% 以内的近优集合），在留出集上计算三种目标：
- 几何平均；
- CVaR10，即最差 10% 窗口的平均比值；
- 按 batch 分组后取最差一组（minimax）。

如果三种目标选出的设计相同，就说明结论稳健。如果不同，报告每种目标选出的设计，以及结论依赖于哪些 batch。

另外，在开发窗口上做 200 次 bootstrap，报告最常被选中的设计的占比。

**输出**
- `round2/results/E3/robust_objectives.csv`
- `round2/results/E3/selection_stability.csv`

### 5.5 第 5 层：未校准参数的全局敏感性

**不确定参数及范围**

| 参数 | 范围 |
|---|---|
| 每 tile 片上时间 | 1–30.4 拍，连续 |
| bank 带宽 | 8–32 B/cycle |
| 点积延迟 | 每级 1–4 拍 |
| 信用数 | 256–512 |
| vector 吞吐 | 0.5–2 倍 |

**做法**
- 输出量取 Δ = 最优异构 ÷ 最优单核 − 1。每个采样点都重新做第 1 层（证明 A，δ = 2%）和第 2 层。
- 用 Saltelli 采样，至少 N = 256 个基础样本。
- 报告每个参数的一阶 Sobol 指数 S1 和总效应指数 ST。
- 再报告翻转边界：Δ = 0 和 Δ = −5% 所对应的参数取值，用一维或二维切片表示。

这一层的结论直接用来指导之后的 RTL 测量：RTL 只需要精确测出 ST 最大的一两个参数。

**输出**
- `round2/results/E3/sobol.csv`
  - 列：`param, S1, S1_ci, ST, ST_ci`
- `round2/results/E3/flip_boundary.csv`
- `round2/figures/fig_sobol.pdf`
- `round2/figures/fig_flip_boundary.pdf`

---

## 6. E4 组会第一张表与时间分解

使用第 5 节得到的设计，在留出集上出表。

**表中的条目**
- B0、B1、B2
- 固定 3+3（`3x4x512` 两核）
- 固定 4+2（`4x4x512 + 2x4x512`）
- 最优 5+1、最优 4+2、最优 2+4
- 最优异构（任意组织）
- U1：同一个 12,288 乘法器阵列，每个专家可选自己最优的形状，切换零代价
- U2：B1 形状，权重在所有 M 波次中只从 SRAM 读一次

**输出**
- `round2/results/E4/heldout_main_table.csv`
  - 每个条目 × 每种 onchip 模式一行。
  - 列：`entry, onchip_mode, design, B2, B4, B8, B16, B64, B96, B128, all_geomean, ratio_vs_B1, ratio_vs_B2, ci95_low_vs_B1, ci95_low_vs_B2, gate_5pct_pass, sched_type`
  - `sched_type` 取 `milp` 或 `runtime`，两种都要报告。
- `round2/results/E4/breakdown.csv`
  - 列：`entry, onchip_mode, batch, core0_compute_busy, core1_compute_busy, w_port_busy, x_port_busy, acc_port_busy, hbm_busy_frac, core_finish_gap, idle_frac, binding_term`
  - 重叠的各项不得相加成墙钟时间。
- `round2/results/E4/hbm512_sensitivity.csv`
  - 只对各族最优重新评估，不重新搜索。
- `round2/results/E4/SUMMARY.md`，回答：
  1. 每种模式下，是否有异构设计相对 B1 和 B2 都过了 5% 门槛；
  2. 最优异构的小核有多大，若不超过 5% 乘法器要明确写出；
  3. U1、U2 相对 B1 的余量；
  4. 同构拆分的代价如何随 batch 和模式变化。

---

## 7. E5 分派与预测器

硬件冻结为第 6 节在每种模式下的"最优异构"和"固定 4+2"。

### 7.1 分派对比
- 纯阈值 T=2（组会规则）
- 阈值 + 负载回退（T 取开发集上的最优值）
- 每层自适应 T
- EFT
- 随机
- MILP 最优分配（作为上限，来自第 5.2 节）

### 7.2 预测器对比
预测器同时用于分派中的时间估计，以及 Next 任务的预取和绑定时机。

| 预测器 | 做法 |
|---|---|
| `random` | 在 [0, 2 × 该核平均任务时间] 内均匀取值 |
| `static` | 每核一个运行平均值 |
| `btb` | 表项索引为（核, 是否 Shared, min(Me, 9)），存上次实测时间；未命中时退回 static |
| `ema` | 与 btb 同一张表，用 EMA 更新，α = 1/4 |
| `ours` | 由成本表算 max(计算时间, 字节数 ÷ 带宽份额)，乘以按（核, Me 档）学到的修正系数（EMA），并在任务进度 1/4、1/2、3/4 时按实测速率修正 |
| `oracle` | 同一调度下的真实时间，跑两遍仿真得到 |

**协议**
- 先用开发窗口预热，再在留出窗口上统计。
- 所有预测器使用同一个窗口序列。

### 7.3 指标定义（写进 README）
- `mae_pct`：|预测时长 − 实际时长| ÷ 实际时长，取平均。
- `success_pct`：Next 的第一块权重落在 [Current 结束 − W, Current 结束] 区间内的比例，W 取两个块的计算时间。
- `late_pct`：Next 的第一块权重在 Current 结束之后才到的比例。
- `stall_cycles`：PE 因等待 Next 权重而停顿的周期数。
- `e2e_ratio_vs_oracle`：端到端延迟除以 oracle 的端到端延迟。

### 7.4 输出
- `round2/results/E5/dispatch_table.csv`
- `round2/results/E5/predictor_table.csv`
  - 列：`design, onchip_mode, predictor, mae_pct, success_pct, late_pct, stall_cycles, e2e_ratio_vs_oracle, e2e_ratio_vs_ours`
- `round2/results/E5/dispatcher_state_bits.csv`
- `round2/results/E5/SUMMARY.md`，回答：
  1. 纯阈值与"阈值 + 回退"相差多少；
  2. 各预测器的端到端差多少；
  3. runtime 与 MILP 最优分配之间还差多少。

---

## 8. E6 端到端

1. **MoE 层**：B0、B1、B2、最优异构，按 batch 给出延迟的几何平均和相对 B1 的比值。直接取自第 6 节，不重算。
2. **整模型每 token**
   - 非 MoE 部分（attention、router、norm）对所有设计相同，从 PLENA 已有 trace 中取每层的时间，按层数折算到每 token。
   - 如果没有可用数据，写明缺失，不得估造。
3. GPU 对比本轮不做。

**输出**
- `round2/results/E6/moe_layer_e2e.csv`
- `round2/results/E6/model_token_e2e.csv`
  - 列：`design, batch, moe_ms_per_layer, non_moe_ms_per_layer, layers, token_ms, ratio_vs_B1`
- `round2/results/E6/SUMMARY.md`

---

## 9. 总报告

文件：`round2/REPORT_ZH.md`，固定按以下结构写。

1. **结论**（不超过 10 行）：
   - 证明 A 的结论；
   - 哪些门槛过了、哪些没过，在哪种模式下；
   - 异构收益区域在哪里；
   - 结论最敏感的参数是哪个。
2. **设定与等资源账本**：一张表。
3. **下限与余量**：来自 E1。
4. **组会第一张表与时间分解**：来自 E4。主表用 `pipelined`，另外两种模式附后。
5. **证明式 DSE**：剪枝覆盖率、证明 A / B、调度差距。
6. **负载区域图**：真实负载位于图中何处，异构收益最大的负载离真实负载有多远。
7. **稳健性**：三种目标是否一致，以及 bootstrap 选型的稳定性。
8. **敏感性**：Sobol 指数、翻转边界、需要 RTL 测量的参数。
9. **数据流 3×3 表**：来自 E2。
10. **分派与预测器**：来自 E5。
11. **端到端**：来自 E6。
12. **局限**：
    - 没有做 RTL；
    - 片上时序未校准；
    - 面积 / 能耗档关闭；
    - 留出集过去曾被访问过；
    - 合成负载只用于区域图。
13. **下一步**：只写由数据直接推出的事项。

---

## 10. 图

每张图都放在 `round2/figures/` 下，PDF 和 PNG 各一份：

- `fig_headroom`：B1 在各 batch 下的实测延迟与五种下限。
- `fig_main_bars`：各设计相对 B1 的加速比，按 batch 分组，每种模式一个子图。
- `fig_breakdown`：B1 / B2 / 最优异构的时间分解。
- `fig_bnb_coverage`：剪枝过程中，被剪区域的比例随层级的变化，以及当前最优解随迭代的变化。
- `fig_workload_map`
- `fig_sobol`
- `fig_flip_boundary`
- `fig_dataflow_grid`：3×3 热力表。
- `fig_me_crossover`：每个 Me 下，各形状完成一个 routed 专家所需的周期，并标出交叉点。
- `fig_predictor`：各预测器的 MAE 与端到端比值。

---

## 11. 交付清单与完成标准

全部完成后，回复我以下内容：

1. 分支名（应为 `research/moe-supply-first-v3`）和最终提交号。
2. `REPORT_ZH.md` 和各节 `SUMMARY.md` 的路径。
3. 下表每一行的状态：完成、部分完成或失败，失败的写明原因。

| 节 | 必须存在的文件 |
|---|---|
| 1 | BRANCHES_BEFORE.md、MERGE_LOG.md、BRANCHES_AFTER.md、E0/reproduce_check.csv（误差全为 0） |
| 2 | E0/frozen_inputs.json、E0/SOURCES.md |
| 3 | E1/bounds_per_window.csv、E1/headroom_by_batch.csv、E1/SUMMARY.md |
| 4 | E2/micro.csv、E2/layer_grid.csv、E2/SUMMARY.md |
| 5.1 | E3/bnb_certificate.csv、E3/bnb_leaves.csv、E3/bnb_summary.json（覆盖率 100%）、E3/lb_validity.csv（全部 ok） |
| 5.2 | E3/schedule_gaps.csv |
| 5.3 | E3/workload_map.csv、E3/workload_extreme.json |
| 5.4 | E3/robust_objectives.csv、E3/selection_stability.csv |
| 5.5 | E3/sobol.csv、E3/flip_boundary.csv |
| 6 | E4/heldout_main_table.csv、E4/breakdown.csv、E4/hbm512_sensitivity.csv、E4/SUMMARY.md |
| 7 | E5/dispatch_table.csv、E5/predictor_table.csv、E5/dispatcher_state_bits.csv、E5/SUMMARY.md |
| 8 | E6/moe_layer_e2e.csv、E6/model_token_e2e.csv、E6/SUMMARY.md |
| 9–10 | REPORT_ZH.md、figures/ 下全部 10 张图 |

**完成标准**
- 上表每一行都"完成"。
- 每个配置跑两遍结果一致。
- 全部测试通过。
- 没有新建任何分支。

如果某项因算力或运行时间限制无法完整完成，要报告：
- 已完成的部分；
- 未完成部分的下界或差距；
- 继续运行所需的命令。

不得降低要求后报告为"完成"。