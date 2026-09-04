# Nemotron Agentic Workload：Matrix L-Compute DSE

## 1. 这轮解决了什么

这轮没有增加 ISA、cache 或硬件结构。它修复的是实验可信度：把真实 B200
Agentic 数据严格接入 Nemotron 52 层时间线，并同时报告会影响论文结论的模型边界。

- 48 条 workload：BFCL v3、GPQA-Diamond、SWE-bench Verified 各 16 条；
- 真实官方 NVFP4 checkpoint，revision
  `ce1b118ae66ec705d02c241525192832eb045fd3`；
- `B1/B2/B4/B8/B16` 的真实 length-sorted timing 分组；
- 23 个 MoE 层、Top-6、连续 32 个 decode step 的 eager route；
- 严格串行和理想资源重叠两个时间线端点；
- checkpoint mixed NVFP4、统一 MX8/MXFP8、统一 BF16 三种权重流量；
- 公平固定接线 D′ 的 packet-level bank 对照；
- 每个汇总行的样本数 `N` 和统计口径。

外部归档 SHA256 为
`18ff9fa81a8993cc9d5406e1f1db0d3d73357bcfc935751f160ed5a636827435`。
`SHA256SUMS` 有 38 项；本模型实际消费的 8 个文件逐个校验，其中
`campaign_summary.json` 的 SHA256 是
`bc2c8230c1766d2d66f67672b42b6d975449b95f9d9b871e4d8379986d9dcc0b`。

## 2. 路由数据现在如何使用

路由导入采用 strict mode。模型、batch、context、step 或 token 数任何一项不匹配，
DSE 直接失败，不再静默退回 `min(128, 6B)` 理论上界。

| 事件范围 | 数量 | 校验与用途 |
|---|---:|---|
| 原始事件 | 140,921 | 全部通过 metadata 和 conservation 校验 |
| Prefill | 1,104 | 验证 48 条输入均覆盖 23 个 MoE 层 |
| Decode | 139,817 | 全部逐行检查 Top-K ID、权重归一化和 expert counts |
| 进入 32-step DSE | 35,328 | `48 × 32 × 23`，用于本轮结果 |
| 晚于 replay window | 104,489 | 已完整校验，但明确不进入本轮 DSE |

优化 timing 与 eager routing 的完整生成序列只有 `3/48` 条相同；前 32 token
有 `20/48` 条相同。因此 eager trace 只提供自洽 route，优化 trace 只提供 GPU
baseline。B2 以上的 route 是同一 timing group 中多个独立真实 B1 eager trace 的
expert 并集，不是直接采集的 batched route。

原始 artifact 含 prompt token ID，可以恢复题面，不能随论文公开。本仓只保存 hash、
聚合测量和派生 route 统计。

## 3. 三类数字不能混用

| 数据 | 含义 |
|---|---|
| GPU timing/energy | 真实 B200 NVFP4 测量 |
| GPU eager routing | 真实 checkpoint route，决定每步读取哪些专家权重 |
| PLENA cycles | 官方尺寸、Compiler recurrence、symbolic weights 的 pre-RTL 公式时间线 |

GPU 毫秒没有进入 PLENA 周期公式。PLENA 的 1 GHz 代理也不能除以 B200 ITL，称为
芯片加速比。

## 4. GPU baseline 的可追溯来源

下表逐项来自外部 `campaign_summary.json -> timing.aggregate.all`，而不是从本仓
`summary.csv` 二次推断。每个 batch 都有 960 个 request measurement；trial 数依次为
960、480、240、120、60。

| Batch | TTFT median | ITL median | 吞吐 median | Batch 能耗 median |
|---:|---:|---:|---:|---:|
| 1 | 81.79 ms | 4.454 ms | 144.80 tok/s | 65.30 J |
| 2 | 80.82 ms | 5.086 ms | 267.54 tok/s | 75.78 J |
| 4 | 83.73 ms | 5.614 ms | 494.06 tok/s | 97.90 J |
| 8 | 86.44 ms | 5.766 ms | 969.99 tok/s | 129.02 J |
| 16 | 122.05 ms | 6.738 ms | 1569.63 tok/s | 186.69 J |

本仓 `summary.csv` 的 GPU 列是“workload-group 中位数的中位数”，用于和同一组
PLENA DSE 对齐；它与上面的“全部 request/trial 全局聚合”是两个统计量。

## 5. 路由负载与样本数

旧边界假定每一步每层触碰 `min(128, 6B)` 个专家。真实 route union 的中位数更低：

| Batch | 理论最大 distinct | 实测中位数 | 汇总 group 数 N | P95 口径 |
|---:|---:|---:|---:|---|
| 1 | 6 | 6 | 48 | descriptive |
| 2 | 12 | 11 | 24 | descriptive |
| 4 | 24 | 19 | 12 | exploratory |
| 8 | 48 | 32 | 6 | exploratory |
| 16 | 96 | 49 | 3 | exploratory |

`N` 是三个 benchmark 的 length-sorted、互不重叠 workload group 数。`N < 20`
时 P95 仍写入 CSV 方便复现，但只作为描述性探索值，不能当稳定尾延迟结论。

## 6. 两个时间线端点

固定硬件假设：`MLEN=2048`、`BLEN=32`、64 banks、1 MiB BF16 Matrix SRAM、
1 GHz、1560 B/cycle HBM。每个 group 连续模拟 32 个 decode step。

- **严格串行**：`HBM + Matrix + Vector + L-Compute`。这是当前依赖安全的结果。
- **理想资源重叠下界**：`max(HBM, Matrix, Vector + L-Compute)`。它假设三个资源
  完全重叠，不考虑依赖、容量和仲裁；不是 Compiler 已发射的调度。

| B | N | D/A 串行 | D/B 串行 | D/C 串行 | D/B 理想重叠端点 | D TPOT 串行 | D TPOT 理想下界 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 2.016x | 1.545x | 1.098x | 1.000x | 2.009 ms | 1.872 ms |
| 2 | 24 | 2.580x | 1.848x | 1.152x | 1.024x | 2.583 ms | 2.309 ms |
| 4 | 12 | 3.286x | 2.227x | 1.220x | 1.564x | 3.571 ms | 3.022 ms |
| 8 | 6 | 4.100x | 2.664x | 1.298x | 2.270x | 5.267 ms | 4.168 ms |
| 16 | 3 | 5.082x | 3.191x | 1.393x | 3.274x | 7.999 ms | 5.780 ms |

低 batch 时 HBM 是 B 和 D 的共同下界：理想重叠会把 B1 的 D/B 收益完全隐藏。
batch 增大后 Arlo B 转为 Vector/issue 主导，而 D 仍接近 HBM 主导，所以 B4 以后即使
在乐观重叠端点，L-Compute 的多行递推仍有明显收益。

`D/A` 和 `D/B` 是多行 `L_TILE`、紧凑 descriptor、减少 chunk/issue 的组合收益，
不是“可编程 skew 单项收益”。`D/C` 也混有 descriptor/chunk/issue 差异。

## 7. 权重精度敏感性

三种策略使用完全相同的 93 个 route group、模型尺寸和 recurrence，只改变权重
logical storage：

- checkpoint mixed NVFP4：默认 NVFP4，按 B200 checkpoint 记录保留 BF16 exclusion；
- MX8/MXFP8：每元素 1 byte，每 128 个元素 1 byte scale；
- BF16：每元素 2 bytes。

NVFP4 计数包括每 16 个值一个 FP8 block scale，但三种策略都不计 tensor-global
scale、物理 padding/alignment 和反量化计算。因此这是流量敏感性，不是完整量化执行。

| B | Mixed NVFP4 D/B（串行 / 理想） | MXFP8 D/B（串行 / 理想） | BF16 D/B（串行 / 理想） |
|---:|---:|---:|---:|
| 1 | 1.545x / 1.000x | 1.484x / 1.000x | 1.254x / 1.000x |
| 2 | 1.848x / 1.024x | 1.696x / 1.000x | 1.371x / 1.000x |
| 4 | 2.227x / 1.564x | 1.945x / 1.154x | 1.514x / 1.000x |
| 8 | 2.664x / 2.270x | 2.235x / 1.577x | 1.692x / 1.000x |
| 16 | 3.191x / 3.274x | 2.626x / 2.211x | 1.949x / 1.165x |

以 workload-group 中位数计，B1 的 32-step logical weight read 为 85.20 GiB
(mixed NVFP4)、96.94 GiB (MXFP8)、192.38 GiB (BF16)；B16 分别为 234.96、
365.26、724.85 GiB。精度越高，HBM 越容易掩盖片上多行执行收益。

## 8. Bank conflict 的准确结论

在 93 个 group 中：

- C 的累计 bank stall 为 `188,416 × B` cycles，占 C 整模周期中位数约
  `0.27%` (B1) 到 `0.85%` (B16)；
- D 的 bank stall 全部为 0；
- 本轮每个 group 都显式携带重新计算的 D′ packet evidence；
- D′ 使用原固定对角接线和合法 per-tile base phase，同样 0 stall；
- D/D′ 的纯 bank-service speedup 是 `1.00x`。

因此当前证据支持 **Matrix SRAM 上的多行递推执行**，但不支持“新增任意可编程
row-skew 加法器带来整模加速”。最终 pre-RTL 候选已经删除该系数，只保留紧凑
编码多 tile base phase 的 6-bit phase accumulator。若论文要把任意可编程斜率作为主贡献，必须找到
D′ 无法用普通 base placement 表达的真实多对象 packet；不能再拿被限制的 C 当
最强固定对照。

## 9. 当前能和不能声称的结论

可以声称：

1. 真实 Agentic route 改变 MoE 权重流量，不能假设专家均匀或最大分散；
2. 在严格串行模型中，多行 L-Compute 相对 Arlo 的收益随 batch 增大；
3. B4 以上，这项收益在理想资源重叠端点仍存在；
4. 结论对权重精度敏感，BF16 会显著放大 HBM 下界；
5. 公平固定 D′ 已达到 phased D 的 bank floor，当前任意 programmable row skew 没有独立收益。

不能声称：

- PLENA 相对 B200 的实测加速或 Token/J；
- 理想重叠端点已经由 Compiler 发射并在 Rust 执行；
- 真实 30B 权重已在 Rust 从第一层运行到最后一层；
- B2/B4/B8/B16 route 是直接 batched hook 测量；
- D/B 或 D/C 全部来自 bank-conflict 消除。

## 10. 复现

```bash
SIMULATOR_ROOT=/absolute/path/to/PLENA_Simulator
COMPILER_ROOT=/absolute/path/to/PLENA_Compiler
CAMPAIGN_ROOT=/absolute/path/to/AGENTIC_NEMOTRON_B200_20260903

cd "$SIMULATOR_ROOT"
nix develop --no-write-lock-file --command \
  just matrix-lcompute-agentic "$CAMPAIGN_ROOT" "$COMPILER_ROOT"
```

输出：

- `campaign.json`：证据来源、严格 route 计数、两个时间线端点、三精度和 D′；
- `group_results.csv`：93 个 group 的逐组结果；
- `summary.csv`：按 benchmark/batch 汇总的 `N`、中位数和描述性 P95；
- `README.md`：由 campaign 数据动态生成的使用边界。
