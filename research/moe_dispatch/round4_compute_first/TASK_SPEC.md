# 任务说明：在 PLENA 原乘法器数量下，先做纯计算 DSE，再反推需要多少 HBM 带宽（第四轮）

分支：继续在 `research/moe-supply-first-v3` 上做，不新建分支。代码和结果放在 `research/moe_dispatch/round4_compute_first/` 下。第二、三轮已提交的结果一律不改。

不考虑运行时间，所有步骤都要完整跑完。

## 0. 目的

1. 先假设 HBM 带宽无限大，搜索出计算最快、且物理上合理的单核、同构、异构设计。异构不限定 1:1。
2. 再反推：要让每个设计跑到它的纯计算速度，HBM 至少需要多大带宽？
3. 在几档真实带宽下，填出四项指标（Equivalent Fetch Bandwidth、Fetch time、Equivalent Flops、Compute Time），并判断这样的带宽能不能做到。

## 1. 全局规则

- **乘法器总数**：12,288，即 PLENA 原形状 `6x4x512`。所有组织相同。
  - 如果 PLENA 论文里的乘法器数量与此不同，在 `round4_compute_first/README.md` 里写明出处，并把这个全局参数改成论文中的数值。
- **精度**：只用 BF16，不做 W4/W8。
- **其他资源**：与第三轮主设计相同，三种组织一视同仁。
  - 片上存储总量 2,158,592 B，可分配的私有与落地容量合计 532 KiB；
  - W/X/累加 bank 总量 64/24/12；
  - 向量吞吐 64 元素/周期；
  - 1 GHz。
- **物理约束**：一套约束，对每一个核都一样。不允许只限制单核、不限制异构的小核。
  - PM ∈ 1–16；
  - PN ∈ 1–256；
  - PK ∈ {32, 64, 128, 256, 512}；
  - 乘积等于该核分到的乘法器数。
  - 敏感性分析：把 PN 上限放宽到 512，对所有核同时放宽，重跑一遍。
  - 每个上限都要在 README 里写明依据：综合或时序数据、已发表芯片，或者注明"假设"。
- **组织**：
  - 单核；
  - 同构 2 核；
  - 同构 4 核；
  - 异构 2 核：乘法器比例取 1:1、2:1、3:1、5:1、7:1，每核自由选形状。1:1 的意思是两核乘法器相等、形状不同。
- **数据流**：
  - 每核可选 OS、WS、IS；
  - 另加一个选项"按任务切换数据流"：每个专家任务按自己的 Me 选择数据流。这个选项对单核也开放。
- **窗口和统计口径**：
  - 18 个开发窗口用于选择设计，选定后冻结，再评估 135 个留出窗口；
  - 主口径：按 batch（B2/4/8/16/64/96/128）分组取几何平均，另给全部 135 个窗口的几何平均；
  - 不用"7 个 batch 等权几何平均"作主口径，如需要可以放附录。
- **门槛**（同前几轮）：比单核和同构都快至少 5% 才进入校准；快至少 10%、且 95% 置信区间下界至少 5%，才能写"胜出"。
- **可复现**：每个配置跑两遍，结果逐位一致；所有测试通过。

## 2. E1：纯计算 DSE（HBM 带宽无限大）

分两种模式跑：

| 模式 | HBM 带宽 | SRAM 容量和 SRAM 端口带宽 | 用途 |
|---|---|---|---|
| B（主） | 无限大 | 无限大 | 主结果：纯计算极限，搜索在这个模式下进行 |
| A（对照） | 无限大 | 按第三轮的真实值（2,158,592 B；W/X/累加 bank 64/24/12 × 16 B/周期） | 看真实 SRAM 会让计算慢多少 |

模式 B 里 SRAM 是无限大的，所以每个权重只需要从 HBM 读一次，其余的复用都发生在 SRAM 内部。代价不能就此消失，要和 HBM 带宽一样反推出来（见 E2）。

每个设计、每个窗口，都要记录以下数据：

1. **计算时间 T_c**（ms），以及每核的计算服务时间。
2. **HBM 字节**：
   - 模式 B：等于唯一权重字节，加上激活和输出的读写；
   - 模式 A：该数据流实际需要读的字节，包括 PM 不够时的重复读取和 SRAM 放不下时的重复读取。例如 PM=1 的 OS，在 SRAM 放不下整个专家时，每个 token 都要把权重从 HBM 重读一遍。
3. **需要的 SRAM 容量**：每核在整个窗口里同时存活的最大字节数，按 W、X、累加、Z 分开记录。
4. **需要的 SRAM 端口带宽**：每核 W、X、累加三类端口，各自的平均和峰值 B/周期。
5. **唯一权重字节**：每个被激活的专家只读一次时的字节数。
6. **有用运算量**：2 × 有用乘加。
7. **两核各自的任务数和工作量占比，以及空闲比例**。

输出放在 `round4_compute_first/E1/`：

- `selected_designs.json`：每个组织族的最优设计，字段包括：
  - 每核形状、每核数据流（或"按任务切换"）、乘法器分配；
  - 每核 W/X/累加/Z 容量、bank 份额。
- `compute_by_window.csv`，列：
  `mode, design, window_id, batch, Tc_ms, core0_service_ms, core1_service_ms, hbm_bytes, unique_bytes, flops, core0_work_pct, core1_work_pct, sram_peak_W, sram_peak_X, sram_peak_acc, sram_peak_Z, port_avg_W, port_peak_W, port_avg_X, port_peak_X, port_avg_acc, port_peak_acc`
  （存储与端口各列都按每核分别给出）
- `compute_main_table.csv`：设计 × batch 的 T_c 几何平均、有效 GFLOP/s、相对单核和相对同构的比值。

## 3. E2：反推需要多少 HBM 带宽和片上存储

### 3.1 片上存储（只对模式 B 选出的设计）

对每个设计、每个 batch，取窗口最大值和几何平均，给出：

- 需要的 SRAM 总容量，以及 W、X、累加、Z 各自的容量，并与第三轮的 2,158,592 B 对比，算出倍数；
- 需要的 W、X、累加端口带宽，并与第三轮的 bank 总量（64/24/12 × 16 B/周期）对比，算出倍数；
- 再把这个设计放回真实 SRAM（模式 A），给出 T_c 变慢多少、HBM 字节多了多少。

输出：`round4_compute_first/E2/sram_requirement.csv`，列：

`design, batch, sram_total_peak_KiB, W_KiB, X_KiB, acc_KiB, Z_KiB, sram_over_round3, port_W_peak_Bpc, port_X_peak_Bpc, port_acc_peak_Bpc, port_over_round3, Tc_modeA_over_modeB, hbm_bytes_modeA_over_modeB`

### 3.2 HBM 带宽

对模式 B 和模式 A 分别计算。每个设计、每个窗口：

```
平衡带宽（GB/s）          = HBM 字节 ÷ T_c
平衡带宽下限（GB/s）       = 唯一字节 ÷ T_c
```

按 batch 取几何平均。

再把结果换算成需要多少个存储堆栈：

| 类型 | 每堆栈带宽 |
|---|---|
| 现在的 HBM2 模型 | 256 GB/s |
| HBM3 | 819 GB/s |
| HBM3E | 1,229 GB/s |

每个设计、每个 batch，都给出"需要几个 HBM2 堆栈 / 几个 HBM3E 堆栈"，并标注"现实可行"或"现实不可行"。判断标准：不超过 8 个 HBM3E 堆栈，约 9.8 TB/s。

输出：`round4_compute_first/E2/balance_bandwidth.csv`，列：

`mode, design, batch, Tc_ms, hbm_MiB, unique_MiB, balance_GBps, balance_unique_GBps, HBM2_stacks, HBM3E_stacks, feasible`

## 4. E3：在几档真实带宽下填四项指标

带宽档位：256、512、1,229、2,458、9,830 GB/s（分别对应 1 个 HBM2、2 个 HBM2、1 个 HBM3E、2 个 HBM3E、8 个 HBM3E）。

模式 B 和模式 A 各填一套。每档带宽下，每个设计、每个 batch，填以下各项：

| 指标 | 定义 |
|---|---|
| Compute Time | T_c（来自 E1） |
| Fetch time | HBM 字节 ÷ 该档带宽 |
| 估计整层时间 | max(T_c, Fetch time)，这是下限 |
| Equivalent Fetch Bandwidth | HBM 字节 ÷ 估计整层时间 |
| Equivalent Flops | 有用运算量 ÷ 估计整层时间 |
| 受限于 | 计算 或 取数 |

**验证**：在 256 和 1,229 GB/s 两档，用完整的系统仿真实际跑一遍 E1 选出的设计：HBM、真实 SRAM 容量和端口、派工全部打开，用 fixed 在线派工。如果设计需要的 SRAM 超过真实值，按真实值截断后再跑，并在表里注明。

- 报告"实际整层时间 ÷ 估计整层时间"。
- 偏差超过 5% 的地方要列出来，并说明原因。

输出放在 `round4_compute_first/E3/`：

- `four_metrics.csv`，列：
  `mode, bw_GBps, design, batch, Tc_ms, fetch_ms, est_layer_ms, eq_fetch_GBps, eq_GFLOPs, bound_by, sim_layer_ms, sim_over_est`
- `four_metrics_ZH.md`：每档带宽一张表，格式与原表"大小核Table"的 expectation 一致：
  - 行为 batch，列为四项指标的 a／b／c 数值；
  - 原预期一列；
  - 实际关系一列。

## 5. E4：异构在什么带宽下才有意义

1. 对每个异构族，相对最优单核和最优同构，求出两个最小带宽：
   - 让"估计整层时间"快至少 5% 所需的最小带宽；
   - 快至少 10% 所需的最小带宽。
   - 按 batch 和全部窗口分别给出；如果在 9,830 GB/s 以内都达不到，写"不存在"。
2. 画一张图：横轴带宽（对数坐标），纵轴异构相对单核的加速比，每个 batch 一条线，标出 5% 和 10% 两条线、各档存储堆栈的位置。
3. 单独比较"异构"和"单核按任务切换数据流"：如果单核按任务切换就能拿到异构的大部分收益，要明确写出来。

输出放在 `round4_compute_first/E4/`：

- `crossover.csv`；
- `fig_speedup_vs_bw.pdf`；
- `HETERO_VS_PERTASK_DATAFLOW.md`。

## 6. 报告

`round4_compute_first/REPORT_ZH.md`，第一页用一张表回答：

1. 在 12,288 乘法器、HBM 和 SRAM 都无限大时，单核、同构、异构各自最快是什么设计，异构快多少？
2. 要跑到这个速度，各需要多大 HBM 带宽、多大 SRAM 容量、多大 SRAM 端口带宽？分别是现在的几倍？是否现实可行？
3. 在 256 GB/s 和 1,229 GB/s 下，四项指标分别是多少？整层受计算还是取数限制？
4. 带宽到多少，异构才能比单核和同构快 5% 和 10%？
5. 异构的收益里，有多少是单核按任务切换数据流也能拿到的？

## 7. 写结论时的要求

- 纯计算结果必须标明"HBM 带宽和 SRAM 都假设为无限大"，不能直接当成系统加速。
- 估计整层时间是下限；凡是引用系统时间的地方，以 E3 的完整仿真结果为准。
- 不得为了让某个设计胜出而只给它放宽约束。约束放宽时必须对所有核同时放宽。
- 没有过门槛，不得写"异构胜出"。

## 8. 提交与完成标准

- E1～E4 各提交一次，提交信息以 `round4:` 开头，推送到 `research/moe-supply-first-v3`。
- `round4_compute_first/README.md` 列出：约束依据、乘法器数量的出处、每步的提交号和运行命令。
- 完成标准：
  - 每个配置跑两遍结果一致；
  - E3 的完整仿真偏差已报告；
  - 第 2～6 节列出的文件全部存在；
  - 报告第一页五个问题都有数值答案。