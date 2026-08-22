# PLENA Prefill DSE 项目完整技术历史与 Thesis 写作交接

> 状态日期：2026-08-22
> 工作分支：`PLENA_Simulator: yx/prefill-DSE`、`PLENA_Compiler: yx-optimized-gqa-attention`、`PLENA_RTL: yx/prefill-chip`
> 用途：供 final thesis 写作和另一 AI 建立完整上下文。本文既是技术总结，也是“哪些结论可以安全写入论文”的可信度审计。
> 排除项：`PLENA_Tools` 从本项目提交与整理中明确排除；两篇本地论文 PDF 只是参考资料，不应提交到仓库。

## 0. 如何使用本文

本文同时提供四类信息：

1. **项目历史**：从最早的 Qwen 支持、旧解析模型，到 CostEmitter、HBM V4、RTL-v6、DSE 和 A100 serving 测量。
2. **当前 canonical stack**：最终代码实际使用哪套 compiler schedule、latency、memory、area、power 和 multi-chip 模型。
3. **实验结果与证据等级**：哪些是实测，哪些是 RTL/DC 校准，哪些是结构外推，哪些只是 ideal architectural result。
4. **过时或无效结论**：防止后续写作把旧 schema 的数字误当成当前结果。

建议另一 AI 在写 thesis 时遵循以下规则：

- 先引用本文第 2、11、15、18 节确定当前口径。
- 实现章节可使用第 5 至 14 节。
- 实验章节优先使用第 16 至 18 节中的当前数据。
- 所有速度、能耗和面积 claim 必须附带第 3 节定义的 fidelity 标签。
- 不要把 `ideal-II1` 写成 cycle-exact RTL performance。
- 不要把 `R16` softmax row lanes 写成已经 ISA 编码或 RTL 验证的设计。
- 不要把 analytical disaggregated pipeline 写成真实 PLENA-to-A100 serving deployment。
- 不要把 vLLM batch first-token barrier 写成标准 request-visible TTFT。

## 1. 项目最终研究问题

本项目逐步从“为 PLENA 增加 Qwen prefill 支持”演化为以下完整研究问题：

> 在现有 PLENA ISA/RTL 基础上，通过 compiler lowering、有限硬件增强和经过校准的解析模型，能否为 Qwen3 dense 与 MoE 的长上下文 prefill 搜索出面积和内存预算可行的架构；再将其作为 disaggregated prefill engine，与 A100 decode backend 组合，在可接受 E2E latency 下提高 output throughput energy efficiency？

项目最后不是单一优化，而是一个贯通栈：

```text
Qwen3 workload / precision profile
    -> PLENA compiler schedule
    -> structured CostTrace / DMA / EnergyAction lineage
    -> compute CostEmitter + HBM V4
    -> area + power + tile-aware DP/TP/EP model
    -> latency/energy prefill DSE
    -> measured A100 decode/runtime data
    -> analytical disaggregated system composition
    -> output TPS / output tokens per joule Pareto
```

最终主结论应更接近：

> PLENA 的潜在优势主要是满足 latency/resource 约束下的 system energy efficiency，而不是无条件获得比 GPU 更低的单批 TTFT。

## 2. 当前 Canonical Stack

截至本文日期，正式或候选组件如下。

| 层级 | 当前模式 | 作用 | 可信度边界 |
|---|---|---|---|
| Workload | Qwen3-32B dense、Qwen3-235B-A22B MoE | short/long/ultra-long prefill | 235B multi-chip routing 仅支持 `fixed-balanced` |
| Compiler profile | `current-dse-v1` | 当前正式 DSE schedule 组合 | historical schedules 不进入普通搜索 |
| Attention layout | compact packed GQA | batch/head packing、logical KV groups | transactional correctness 覆盖主要配置 |
| Attention pipeline | `row-interleaved-v1` | K block 与 query-row group 调度 | CostEmitter/ASM 共用 planner |
| Softmax | RTL-v6 multi-row candidate | packed state、direct PV、R-way row parallel | R1/2/4/8 实现域；R16 仅结构外推 |
| Vector scalar baseline | RTL-v5 | auto-tier compact-stat SIMD | production RTL integrated |
| Address generation | `loop-agu-v1` + `live-stride-v1` | loop stream 与残余 stride | AGU-v2 已退役 |
| FFN | `affine-loop-v2` | 全 SRAM policy 共享 affine IR | legacy template 仅 oracle/fallback |
| MoE | `compact-route-v2` | compact router、dispatch/combine | fixed-balanced summary 是解析 routing 假设 |
| Compute latency | CostEmitter `ideal-II1` | compiler-derived dynamic work | architectural ideal，不是时序闭合结果 |
| Memory latency | HBM V4 | production DMA physical-line service | 单芯片校准；多芯片为 rank-local analytical use |
| Area | precision-aware structural v4 + RTL-v6 delta | logic regression + SRAM macro tiling | ASAP7 到目标工艺仍有映射不确定性 |
| Power | action-based v2 | on-chip dynamic、SRAM、HBM、leakage | ideal gating 为主，ungated 为 shadow |
| Multi-chip | tile-aware DP x TP x EP v4 | rank-local tile reconstruction | stage-level analytical，不是 distributed simulation |
| DSE | latency-energy Pareto | 16,384 COMPLETE / campaign | system selector 只重评分 prefill Pareto |
| Serving | fixed-batch serving metrics v3 | output TPS、tokens/J、TTFT/TPOT | 无真实 KV import、无 queueing simulation |

当前最重要的 profile 组合：

```text
compute timing               = ideal-II1
compiler trace               = affine-block-summary-v1
vector/scalar schedule       = rtl-v6 candidate
softmax row issue            = wavefront-v1 / structured summary
softmax row lanes            = 1,2,4,8,16
softmax state                = row-bank-simd-v3
PV accumulation              = direct-packed-rmw-v1
FFN projection               = affine-loop-v2
FFN address                  = live-stride-v1
address generation           = loop-agu-v1
HBM                          = V4 sufficient statistics
multi-chip                   = tile-aware-dp-tp-ep-v4
clock gating                 = ideal-hierarchical nominal
artifact retention           = compact
```

## 3. Fidelity 词典

后续所有结果应使用以下标签之一。

| 标签 | 含义 | 可以声称什么 | 不可以声称什么 |
|---|---|---|---|
| `measured` | A100/NVML 或 RTL/DC 工具直接产生 | 在指定 setup 下的实测值 | 外推到其他 topology/width 后仍是实测 |
| `rtl-functional` | cocotb/RTL module 测试通过 | 指定模块功能被实现和测试 | full-chip 时序或完整 workload 已验证 |
| `fitted` | 对 RTL/DC/Ramulator 数据拟合 | holdout 误差范围内的 proxy | 任意域外配置同样准确 |
| `structurally extrapolated` | 用 tile/macro/结构公式外推 | 指定结构假设下的趋势 | ISA 可编码、RTL 已实现或 1 GHz 闭合 |
| `architectural ideal` | ideal-II1、ideal dual-port、ideal gating | 理想架构性能上界或研究候选 | production timing/power 实测 |
| `scheduler-visible` | vLLM 前端观察到的请求行为 | 用户可见 TTFT/E2E/输出时间 | GPU 内部 kernel admission/KV-ready 精确时刻 |
| `analytical system composition` | PLENA 与 A100 分项解析组合 | 固定批次流水 envelope | 真实排队、continuous batching 或真实 KV import |
| `historical but valid` | 旧 profile 的可复现实验 | 解释优化演进 | 代表当前默认 |
| `superseded` | 已由更完整模型替代 | 作为消融/历史 | 当前正式结果 |
| `invalidated` | 后续审计证明语义错误 | 说明发现了什么 bug | 任何当前性能结论 |

特别重要的边界：

- `ideal-II1` 对每条 Vector/Scalar/Control dynamic instruction 收取 1 cycle，并保留 Matrix structural timing；它是 architectural scheduling assumption。
- 1 GHz 是 latency 换算假设。RTL-v6 的部分 leaf synthesis 有负 WNS，项目后来明确停止以 timing closure 作为正式 gate。
- RTL-v6 当前最准确描述是：`production_datapaths_integrated_focused_tests_passed_full_core_top_level_not_run`。
- R16 进入 DSE 是有意的结构研究点，标签为 `structural_extrapolation_not_isa_encodable`。
- PLENA serving 结果没有真实地把 PLENA 生成的 KV 导入 A100；A100 decode 是 `imported_kv_decode_proxy`。

## 4. 仓库、分支与所有权边界

### 4.1 Root Simulator

```text
repository: /home/yh3525/FYP/PLENA_Simulator
main base:  dfccdab
branch:     yx/prefill-DSE
HEAD:       e6cb85f  (2026-08-15 committed state)
tag:        research/prefill-dse-20260802 -> d1dfe24
```

该分支相对 `main` 有 87 个项目提交，覆盖 simulator、analytical models、DSE、RunPod harness 和报告。

另有三个从相同主线抽取的可上游化分支：

| 分支 | HEAD | 内容 |
|---|---|---|
| `yx/upstream-costemitter-v1` | `6a82866` | generic compiler-trace latency schemas、symbolic schedule evaluator、semantic cache |
| `yx/upstream-hbm-v4-v1` | `2ccb517` | 上述内容加 production-DMA HBM V4 与 exact sufficient statistics |
| `yx/upstream-power-v1` | `e170fb5` | 上述内容再加 action energy、clock/HBM power bounds |

这些 upstream 分支包含 Qichao/Arlo 的 routed-MoE emulator 与 timing prerequisites；不能把上游已有工作全部归为本项目贡献。

### 4.2 Compiler Submodule

```text
repository: /home/yh3525/FYP/PLENA_Simulator/PLENA_Compiler
branch:     yx-optimized-gqa-attention
HEAD:       0ba3b65
tag:        research/prefill-dse-20260802 -> a384108
```

历史分支：

- `yx-optimized-gqa-attention-before-main-rebase-20260625`：最早 5 个 packed-GQA 改动的 rebase 前备份。
- `yx/upstream-cost-trace-v1`：通用 symbolic final-schedule CostTrace 抽取分支，HEAD `64a6118`。

Compiler 分支还包含 Kevin、George 等人在 `main` 上提供的短序列 attention、batch>1、O(n^2) 到 O(n) codegen、FFN hardware loop、多 tile causal mask、HBM region size 等前置修复。这些是本项目能继续优化的基础，但不是本项目独立完成的贡献。

### 4.3 RTL Repository

```text
repository: /home/yh3525/FYP/PLENA_RTL
branch:     yx/prefill-chip
committed:  364a1e6
tag:        research/prefill-dse-20260802 -> c77b5c5
```

已提交部分包括 precision-safe profiles、DC workflow、pipelined Scalar、segment-parallel Vector、auto compact-stat SIMD，以及 RTL-v6 opcode decode/leaf datapaths。

2026-08-18 的未提交 RTL 工作树进一步包含：

- production `fp_vector_sram` row banking；
- `softmax_row_engine`、state/stat/factor banks；
- `VectorMachine` row-mode 接入；
- pipeline/control/hazard 修改；
- `packed_pv_writeback` 与 Matrix writeback integration；
- VectorMachine integration wrappers 和 cocotb tests；
- top-level wires 已开始接入，但按用户决定没有运行完整 top-level workload。

因此 thesis 必须把“已提交 leaf RTL”和“当前 working-tree production integration”区分开。

### 4.4 明确排除

- `PLENA_Tools`：本项目整理、提交、同步时均排除；root 可能显示 submodule dirty，但不属于本任务。
- `2509.09505v3.pdf`、`2604.16007v1.pdf`：只做文献和数据口径检查，不进入 commit。
- DSE worker logs、SQLite WAL、pickle/cache、完整 trial 目录：不作为论文 artifact。

## 5. 时间线总览

### Phase A：模型接入与基础可运行性，2026-06-05 至 2026-06-11

目标是让 Qwen 模型和 prefill 路径在 simulator/analytic path 中可运行。

完成：

- 修复 emulator output comparison 兼容性。
- Simulator 和 analytic Llama/Qwen 模型尊重显式 `head_dim`。
- 建立 Qwen3-8B sliced prefill probe 与 native full-size preset。
- 增加 opcode memory profiling，定位大 trace/大 context 的内存问题。
- 增加 MoE analytic prefill mode，为后续 Qwen3-235B-A22B 奠定基础。

这一阶段还不是最终性能模型，只解决 workload 表达和运行入口。

### Phase B：第一代 latency/area/DSE，2026-06-22 至 2026-07-13

完成：

- 第一版 analytic area proxy。
- Qwen3 dense closed-form analytic latency。
- HBM channel 参数化。
- precision-aware area proxy，使 MXINT/MXFP quantization 能影响 PE 面积和搜索结果。
- Compiler 端开始 packed GQA sequence tiling、correctness 修复、hardware KV loop 和 head chunk generalization。

该阶段暴露的主要问题：

- closed-form 模型只按 tensor shape 估 operation count，无法观察 compiler 实际生成的 padding、state maintenance、address/control 与 DMA。
- 量化最初作为 categorical profile 采样，硬件面积和 PE 端口宽度影响不够直接。
- 大 trace 与 Ramulator 无法支撑大规模 DSE。

### Phase C：可信度基础栈，2026-07-17 至 2026-07-19

这是项目从“估算器”转为“compiler-derived model”的关键阶段。

完成：

- 修复 Ramulator completion 与 production DMA semantics。
- 建立 RTL-calibrated transactional timing scheduler 与 calibration harness。
- Compiler 增加 symbolic CostTrace 和 compressed schedule traces。
- Root 增加 CostEmitter calibration/regression fixtures。
- 实现 HBM V4 calibration。
- 建立 Qwen3-32B precision-aware Optuna DSE。
- 将 MatrixMachine area 改为结构模型，并让 compact traces 可同时驱动 timing/HBM。

这阶段形成 thesis 的第一条核心主线：

> 不能只估 tensor operation；必须从最终 compiler schedule 重建实际动态工作。

### Phase D：Compiler/RTL 全栈优化，2026-07-19 至 2026-07-27

完成：

- native Qwen3 MoE 与 compact prefill lowering。
- RTL-v4 ISA、loop AGU-v1、packed-GQA optimization。
- unified affine FFN 与 partial KV residency。
- MoE compact-route lowering 与 CostTrace lineage 重建。
- versioned timing artifacts、area overlays、action power、compressed HBM V4。
- tile-aware TP/CP/EP 第一版以及 modular canonical DSE。
- 隔离 historical models/generated artifacts。
- causal-prefix DMA folding 和 long-context compact lowering。

这一阶段修复了多个“性能反直觉点”，包括 FFN 地址立即数爆炸、projection-full 走旧模板、多芯片直接 `/N` 的虚假线性加速等。

### Phase E：RTL-v5 与自动扩展 SIMD，2026-08-02

完成：

- compiler/emulator/RTL 同步到 rtl-v5。
- compact-stat SIMD lane tier 随 VLEN/HLEN/head count 自动推导。
- area 与 power 对 4/8/16/32/64 lane tier 建模。
- MoE SRAM demand、decode KV handoff endpoint。
- latency-energy DSE 以及 report audit freeze tag。

RTL-v5 是当前已明确 production integrated 的 Vector/Scalar baseline。

### Phase F：RTL-v6、多芯片 v4 与性能基础设施，2026-08-06 至 2026-08-08

完成：

- multi-row softmax、packed state、direct packed-PV 的 compiler/emulator/area/power 初版。
- HBM V4 exact write aggregation vectorization 与 semantic cache。
- multi-chip 正式从 CP 路线转为 request-level DP x TP x EP。
- DSE worker 扩展、serialized Optuna ask、cross-study cache、compact artifacts。
- RunPod A100 resumable benchmark harness、NVML energy、RoPE/YaRN、topology 与 runtime lock。

这一阶段将研究从单芯片 DSE 延伸到真实 GPU baseline 和 disaggregated system composition。

### Phase G：RTL-v6 物理校准、最终 DSE 与 serving 语义，2026-08-15 至今

完成或当前工作树已完成：

- banked RTL-v6 VectorMachine DC area calibration。
- bank-aware RTL-v6 power calibration。
- softmax pipeline/reduction II 审计。
- W4-only precision domain、R16 structural domain、0.9 A100 area budget。
- 五组 16,384 COMPLETE DSE campaign。
- A100 TTFT phase semantics 修正。
- system selector 从 requests/s 与模糊 throughput/W 改为 output TPS 与 output tokens/J。
- production Vector SRAM/VectorMachine/packed-PV RTL integration 工作树。

尚未完成的最高层证据：

- 没有运行完整 PLENA top-level packed-GQA workload；用户明确要求先完成 VectorMachine integration，不因其他 top-level 模块问题阻塞。
- 当前 serving v4 metric 源码与 RTL production integration 仍未提交。
- banking 校准后新的五组 DSE 已完成，但 system result 仍包含 R16 structural candidates，必须在报告中显式披露。

## 6. Compiler 路径的完整演进

### 6.1 Packed GQA 与 compact layout

最初 Qwen GQA prefill 面临的问题不是单一 opcode 慢，而是物理布局和 code generation 没有与 Qwen 的 head grouping、batch packing 和长序列 tiling 对齐。

主要改动：

- 按显式 `head_dim` 计算 Q/K/V 和 Matrix SRAM shape。
- 支持 sequence-tiled packed GQA，而不是要求整段 sequence 一次驻留。
- 使用 hardware KV loop，避免 compiler 展开每个 K/V tile。
- 将 Q heads 按 logical KV group 组织，复用同一 K/V head。
- compact batch packing：把不同 batch 的有效 rows 填入同一 physical MLEN block。
- compact Q/O storage：只保存真实 packed heads，不为 unused physical lanes 复制空间。
- packed-batch mask/segment selector，保持 causal 和 batch boundary correctness。

代表性 compact-layout 数据：

```text
Qwen3-32B, seq=482, batch=16, MLEN=VLEN=2048
physical sequence rows: 32768 -> 8192
Q/O logical width:      16384 -> 8192
controlled 64-layer latency: 54.373 s -> 21.706 s
```

该结果是历史 compiler A/B，用于证明 layout 的重要性；不是当前 rtl-v6 最终性能。

### 6.2 Online softmax 的第一批 compiler 优化

必须使用准确描述：旧实现并非逐元素 Scalar softmax。旧 VectorMachine 已经一次处理一个 `VLEN`-wide score row：

```text
Vector: MAX -> SUB -> EXP -> SUM -> scale over one VLEN row
Scalar: maintain m/l/m_res for that row, state loads/stores, selectors/control
```

早期 compiler 改动包括：

- `direct-first-block-v1`：第一个 K block 不需要读旧 `m/l/O` 状态，直接初始化。
- active physical rows：不对 padding/dummy rows 做完整 softmax state work。
- selector hoisting：不在最内层重复构造同一个 segment selector。
- overwrite reduction：first reduction 直接覆盖结果，避免冗余 accumulation/reset。
- streamed softmax：K block 到达后在线更新，而不是 materialize 全部 score matrix。

历史数据：

```text
M2048 controlled 64-layer:
compact baseline       21.706 s
direct first block     18.503 s
reduction              14.75%
QK/PV/HBM work         invariant
```

### 6.3 Segment-parallel normalization 与 compact statistics

Q/K RMSNorm 和 online-softmax statistics 具有很多长度为 `HLEN` 的独立 segment。旧路径虽有 VLEN-wide VectorMachine，但一次只激活一个 segment，导致 `VLEN=2048, HLEN=128` 时只有 128/2048 lanes 做有用缩放。

解决方案分两步：

1. **Multi-segment reduction/broadcast**：一次 Vector operation 为所有 packed head segments 产生 reduction roots，并将每个 segment 的 scalar factor 广播回对应 lanes。
2. **Compact-stat SIMD**：为各 segment 并行执行 scalar-like `MUL -> ADD -> RSQRT`，避免把 16 个统计值逐个送入 ScalarMachine。

RTL-v5 进一步自动推导 compact SIMD tier：

```text
required_segments = min(num_attention_heads, VLEN // HLEN)
configured tier   = smallest in [4,8,16,32,64] >= required_segments
```

Qwen3-32B 的映射：

| VLEN | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---:|---:|---:|---:|---:|---:|---:|
| Compact lanes | 4 | 4 | 8 | 16 | 32 | 64 |

当前 stack 的历史消融，`seq=482,B16,M/V/B=2048/2048/1024`：

| 配置 | One-layer roofline latency | 相对完整当前栈 |
|---|---:|---:|
| 当前 rtl-v5 compiler stack | 18.258 ms/layer | baseline |
| 无 compact statistics | 21.906 ms/layer | +16.65% |
| 无 selector hoist | 19.244 ms/layer | +5.12% |
| 无 overwrite reduction | 19.245 ms/layer | +5.13% |
| 无 AGU | 26.201 ms/layer | +30.32% |
| 无 affine FFN | 18.376 ms/layer | +0.64% |
| head-major + SRAM fallback | 21.426 ms/layer | +14.78% |
| 全 compatibility stack | 32.136 ms/layer | +43.18% |

这些数字包含多项 compiler/RTL schedule 影响，不能单独归因于某一个硬件 block。

### 6.4 Streamed K-major GQA 与 broadcast

为避免 Q heads 重复执行相同 K/V work，compiler 采用 K-major logical-KV-group schedule：

```text
for each K/V head group:
    load/cache K/V tile
    process all associated Q heads
    reuse K/V and broadcast relevant state
```

代表性历史变化：

```text
QK Matrix operations/layer: 256 -> 32
PV Matrix operations/layer: unchanged at 256
pre-rtl-v4 latency:         26.537 -> 25.342 ms/layer
```

注意：报告长期保留了 `broadcast_rtl_unvalidated` 声明。Compiler 和 analytical latency 能表达该 schedule，但不能将 broadcast 的全部时序视为已经 full-RTL 验证。

### 6.5 Loop AGU-v1

问题：即使地址完全确定，compiler 仍需要在运行时沿 hardware loops 递增 base/stride。完全 compile-time 常量并不能消除 loop-carried address state，因为单一静态 program 仍需访问不同 SRAM/HBM 地址；但这些更新可以由小型 AGU stream 完成，而不是 Scalar ISA 指令。

实现：

- 在 hardware loop 入口绑定最多 6 个 affine address streams。
- loop iteration 自动更新 base/stride。
- `live-stride-v1` 处理少量不能绑定到 loop 的 block correction。
- 不增加 instruction opcode；通过 compiler rewrite 和 frontend sidecar 控制。

历史 A/B：

```text
64-layer old trace compute: 2.209B -> 1.729B cycles
latency:                    2.20935 -> 1.72927 s
total reduction:            21.73%
Scalar cycles:              -38.61%
Control cycles:             -88.29%
loop-end execution:         -99.945%
Matrix/Vector/HBM:          unchanged
```

RTL sidecar 映射面积约 `1.891 mm2`；1 ns 结果接近零 WNS，但这不是完整 physical signoff。

负面结果：`loop-agu-v2` 尝试 post-increment/更复杂 stream inference，但固定 workload 中没有找到合法新增 stream，也没有性能收益，因此实现和 runtime 选择已退役，只保留历史报告。

### 6.6 Partial K/V residency

这里的 `KV residency` 不是 decode 时跨层长期保存的 KV cache。它指当前 decoder layer 的 attention K/V tiles 在 Matrix SRAM 中的临时驻留和重用。

策略：

```text
streaming
projection-full
KV-25
KV-50
KV-75
KV-100
```

SRAM 容量同时影响：

- attention K/V prefix 能否驻留；
- projection/FFN 权重或 activation 的 K chunk 数；
- DMA 次数、HBM V4 row/channel pattern；
- Matrix SRAM area、dynamic energy 和 leakage。

一个容易误读的历史例子：

```text
M512/B64 short context
streaming, 2 tiles: 20.418 s, 404.652 J, SRAM 16.874 mm2
KV-50, 50 tiles:   11.795 s, 347.423 J, SRAM 24.717 mm2
```

该大收益主要来自 projection/FFN chunking 改善，不应全部归因于“一次少读 K/V”。长序列 `seq=65536,M512` 下，K/V tile load reduction 随 residency 为：

```text
KV-25: 43.2%
KV-50: 74.0%
KV-75: 92.4%
KV-100: 98.4%
```

### 6.7 FFN 地址 lowering 修复

#### 6.7.1 大立即数与 dead update

异常点 `M8192/B32/N4` 的旧结果：

```text
aggregate compute = 259.310B cycles
S_ADDI_INT         = 173.947B cycles
latency            = 64.855 s
```

根因：单 K-tile 或大 stride 路径仍生成无效 loop-carried pointer updates；18-bit immediate legalization 又把一个大步长拆成最多 1025 条 `S_ADDI_INT`。

修复：

- 删除 dead address updates。
- 大 stride 使用现有 `S_ADD_INT` 和空闲 GP registers。
- `live-stride-v1` 结构化表达，保持 Matrix、HBM 和 numerical order 不变。

结果：

```text
latency:             64.855 -> 21.796 s
FFN S_ADDI_INT:      -99.11%
Matrix/HBM/numerical invariant
```

#### 6.7.2 Unified affine-loop-v2

进一步发现 `projection-full` 会从新 structured schedule 切回旧 `_ffn_asm_with_loops`，导致 SRAM 更大反而更慢。

长期修复：

- 所有 SRAM policy 共享 `FfnProjectionPlan`。
- 统一表示 K chunk、output MLEN block、BLEN tile、activation column、K accumulation tile。
- SRAM 只决定原有 `k_chunks()`，不决定模板家族。
- ASM、CostTrace、DMA、metadata 从同一 IR 派生。
- guard 比较 K boundaries、Matrix/V_ADD 顺序、DMA manifest、address sequence 与 ideal work；失败则 per-projection fallback。

代表点：

```text
M4096/B64/N4: 75.835B -> 44.419B cycles, FFN S_ADDI -99.76%
M8192/B32/N4: total reduction 43.09%
```

2048-point no-regression audit：

```text
failed/fallback trials: 0
adjacent matched SRAM pairs: 123
performance regressions: 0
>2% improvements: 108
```

### 6.8 MoE compact-route-v2

旧 MoE path 中 router、route probability extraction、dispatch 和 combine 使用了非常宽的 vector identity、full-row resets 和重复 route scan。

改动：

- Router 使用 single-block softmax，只处理 active rows 和 `valid_cols=num_experts`。
- 删除 `MLEN x MLEN` route identity。
- 使用 `S_LD_VLANE_FP/S_ST_VLANE_FP` 直接读写 expert lane。
- top-k probability 放在 rotating FP registers，按 rank 做 normalization。
- route-weight row 只写前 top-k lanes，不清零完整 FPRAM row。
- combine 每条 route 只加载一次 weight，并跨 hidden blocks 复用。
- `MoeExpertRoutePlan` 按 expert/rank 一次组织 routes，避免 `O(active_experts x routes)` 扫描。
- fixed-balanced 模式直接生成 affine runs，不 materialize route objects。
- 相同 padded bucket rows 的 experts 共享一次 FFN template，再按 expert count replay。
- Expert FFN 复用 `affine-loop-v2 + live-stride-v1 + AGU-v1`。

代表点 `Qwen3-235B-A22B, seq=482,B16,M/V=512,BLEN=64,fixed-balanced`：

```text
one-layer compute:                120.673M -> 111.253M cycles (-7.81%)
router+dispatch+combine:           15.518M ->   6.098M cycles (-60.7%)
roofline latency:                 120.983 -> 111.577 ms
HBM traffic:                      -5.014 MB
cold CostEmitter+V4:              9.13 s
warm evaluation:                  15.34 ms
peak RSS:                         ~0.6 GiB
fixed-balanced route objects:     0
```

限制：runtime arg-topk 仍由 host 选择；multi-chip MoE 正式只支持 fixed-balanced routing，不能把 static-index distributed path 当成已验证功能。

## 7. RTL 架构增强

### 7.1 ScalarMachine pipeline 与 ROB

RTL-v4 阶段为 ScalarMachine 增加：

- scalar ROB/scoreboard；
- forwarding 和 dependency handling；
- loop AGU-v1 frontend integration；
- 与 transactional scheduler 对齐的 ready/done 行为。

其目的不是让 address computation 变成“真实数学运算”，而是在地址/状态不可完全静态消除时，避免 ScalarMachine 因 dependency 和 loop control 串行阻塞。

### 7.2 Segment-parallel VectorMachine

新增能力：

- multi-segment `SUM/MAX` reduction；
- segment factor broadcast；
- lane access 与 overwrite reduction；
- compact statistics SIMD。

在 `VLEN=2048, HLEN=128` 时，可同时覆盖 16 个 packed segments；原有 reduction tree 的 intermediate roots 被复用，不是复制 16 棵完整 reduction tree。

### 7.3 RTL-v6：Multi-row online softmax

#### 正确的问题定义

旧路径已经沿 key-column/VLEN 方向 vectorized；不足之处是独立 query rows 仍串行经过 row softmax 和 scalar state maintenance。

Score tile：

```text
MLEN query rows x VLEN key columns
old issue: 1 x VLEN stripe
v6 issue:  R x VLEN stripes
```

v6 不展开整个 `MLEN x VLEN` tile，也不是通用 VLEN x VLEN SRAM read。

#### 三项可分离优化

1. **Query-row parallelism**：复制 `R-1` 条 attention-only row slices，使 R 个独立 rows 同时执行 MAX/SUB/EXP/SUM/scale。
2. **Packed state SIMD/cache**：banked state/stat/factor storage 并行维护 `m/l/m_res`，删除 attention Scalar FP SRAM 往返。
3. **Direct packed-PV accumulation**：Matrix PV 结果直接 overwrite/accumulate 到 packed-O lane，删除 PV scratch、`V_SHIFT_V` 和 post-shift `V_ADD_VV`。

#### SRAM banking

生产方案按 physical row 静态拆 bank：

```text
bank     = row % R
bank_row = row // R
```

性质：

- logical SRAM bits 不随 R 增加；不是复制 R 份数据。
- 每个 bank 保存完整 VLEN-wide rows。
- R 个对齐 rows 可同时由不同 banks 读取。
- 普通 Vector、Matrix、HBM 仍访问同一逻辑地址，由 bank router 仲裁。
- group base 不对齐或同 bank conflict 时 fallback/stall。
- tail group 只启用有效 banks/slices。

面积增大来自：

- 每个浅 bank 独立支付 SRAM macro 最小深度、外围和 rounding；
- `R-1` auxiliary row slices；
- state/stat/factor banks；
- cross-bank control、scoreboard、routing；
- packed-PV accumulator。

这解释了为什么 logical capacity 不变但 R16/R32 banking area 仍会出现 macro-granularity cliff。

#### Reduction pipeline audit

导师反馈要求区分 latency 和 throughput：

```text
R1 temporal pipeline: one independent VLEN row may be accepted each cycle if II=1
R-way spatial path:   R independent VLEN rows per accepted group
```

因此 v6 不应描述为“消除了 reduction latency”；收益来自 R 倍 row throughput capacity、state work 并行和 PV data movement 删除。真正收益还受 stage Amdahl fraction、tail、bank conflicts、Matrix/FFN work 约束。

#### 当前 production integration 状态

已存在并测试的模块/路径：

- banked `fp_vector_sram`；
- `softmax_row_engine`；
- state/stat/factor banks；
- `VectorMachine` row transaction path；
- pipeline/control scoreboard changes；
- packed-PV accumulator/writeback；
- VectorMachine integration wrapper/cocotb。

未完成或未运行：

- 完整 PLENA top-level packed-GQA workload；
- 所有 MXFP/Matrix burst 在 full core 中的端到端验证；
- 1 GHz timing closure。

### 7.4 RTL-v6 latency 消融

Canonical historical setup `M/V/B=2048/2048/1024`：

| Workload | rtl-v5 | Combined R4 | Reduction |
|---|---:|---:|---:|
| Qwen3-32B `482 x 16` | 18.258 ms/layer | 13.228 ms/layer | 27.55% |
| Qwen3-32B `32768 x 1` | 451.295 ms/layer | 136.772 ms/layer | 69.69% |
| Qwen3-32B `90000 x 8` | 25.459 s/layer | 6.552 s/layer | 74.26% |

长上下文 `32768 x 1` 三项消融：

| Arm | Layer compute latency |
|---|---:|
| rtl-v5 baseline | 451.295 ms |
| Packed state only, R1 | 292.162 ms |
| Direct PV only | 404.432 ms |
| State + direct PV, R1 | 245.300 ms |
| Query-row R4 + state, legacy PV | 183.634 ms |
| Combined R4 | 136.772 ms |

这组数据说明大收益并非仅来自“更多 SRAM ports”：packed state 单独贡献很大，direct PV 贡献较小但稳定，R-way row parallel 再缩短剩余 row work。

### 7.5 单芯片单层 90k/B8 A/B

为避免虚构 235B 可完整驻留单芯片，后续实验统一为 one-layer microbenchmark：

```text
seq=90000, batch=8
M/V=2048, BLEN=128
W4/A4/KV4, FP E6M5
KV-25
one chip, 80 GB HBM
```

Qwen3-32B dense layer：

| Arm | Layer latency | Energy | Core area |
|---|---:|---:|---:|
| rtl-v5 | 30,086.52 ms | 0.623 kJ | 83.48 mm2 |
| Packed state R1 | 20,166.97 ms | 0.548 kJ | 83.59 mm2 |
| Direct PV | 27,364.10 ms | 0.594 kJ | 83.50 mm2 |
| Combined R1 | 17,444.55 ms | 0.520 kJ | 83.61 mm2 |
| Combined R2 | 13,279.24 ms | 0.491 kJ | 85.14 mm2 |
| Combined R4 | 11,196.59 ms | 0.477 kJ | 88.83 mm2 |
| Combined R8 | 10,155.26 ms | 0.470 kJ | 96.20 mm2 |
| R16 projection | 9,634.60 ms | 0.467 kJ | 110.94 mm2 |
| R32 projection | 9,374.27 ms | 0.467 kJ | 140.44 mm2 |

Qwen3-235B-A22B representative MoE layer：

```text
rtl-v5:      27,405.57 ms/layer, 0.465 kJ, 83.51 mm2
Combined R4:  8,513.12 ms/layer, 0.319 kJ, 88.87 mm2
Combined R8:  7,471.25 ms/layer, 0.313 kJ, 96.24 mm2
```

235B 数值只代表一个 fixed-balanced MoE decoder layer，不代表完整模型可放入 80 GB。

R4 以后收益递减的原因：

- softmax row work 已不再占全部 layer latency；
- Matrix QK/PV、projection、FFN/MoE、HBM 和全局控制不随 R 缩放；
- R 只并行 query rows，不减少一行跨多个 VLEN key tiles 的 K-block 数；
- tail/group utilization 与 bank granularity 下降；
- packed state 和 direct PV 的收益在 R1 已经兑现。

### 7.6 RTL-v6 Area/Power 校准

Area campaign：64 component/integration points，加 8 个 matched rtl-v5 baselines。当前回归验证：

```text
state holdout max error              0.24%
packed-PV holdout max                4.36%
wrapper width holdout max            0.84%
wrapper precision holdout max        2.34%
paired VectorMachine holdout max     3.59%
all coefficients nonnegative         true
area monotonic in R/VLEN              true
```

E5M6 state leaf DC area：

```text
R1  4,647.81 um2
R2  9,109.83 um2
R4 17,926.71 um2
R8 35,317.25 um2
```

这些是 state leaf，不是完整 Vector SRAM 或 full VectorMachine area。

Power campaign：51 train/replay jobs + 12 holdouts：

```text
row action holdout max error     7.35%
packed-PV holdout max           11.56%
mixed workload max              12.52%
```

当前模型对大 VLEN 的逻辑使用小宽度 RTL structural extrapolation；SRAM macro tiling按目标逻辑 shape 精确计算。不能把它写成 VLEN=2048 full VectorMachine 已综合。

VLEN=2048 one-layer microbenchmark 的 SRAM macro 粒度：

| R | Logical Mibit | Covered Mibit | Covered/logical | Banked SRAM | Core area |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.18 | 20.16 | 1.98x | 0.785 mm2 | 83.61 mm2 |
| 2 | 10.18 | 20.16 | 1.98x | 0.943 mm2 | 85.14 mm2 |
| 4 | 10.18 | 40.31 | 3.96x | 1.886 mm2 | 88.83 mm2 |
| 8 | 10.18 | 80.62 | 7.92x | 3.771 mm2 | 96.20 mm2 |
| 16 | 10.18 | 161.25 | 15.84x | 7.542 mm2 | 110.94 mm2 |
| 32 | 10.18 | 322.50 | 31.68x | 15.084 mm2 | 140.44 mm2 |

首次明显 macro cliff 在 R4：bank depth 降到 65 rows 后，每个 bank 的最小 macro/外围使 covered capacity 跳到 logical 的 3.96x。R4 是 area-efficient default，R8 是低 latency implemented candidate；R16/R32 的 marginal latency saved/mm2 已明显恶化。

### 7.7 RTL-v6 Numerical 与 Work Invariants

Canonical `482 x 16,R4`：

```text
row groups                 123,904
row-lane utilization        99.59%
Scalar state loads removed 493,568
Scalar state stores removed493,568
PV shifts removed          493,568
PV vector adds removed     493,568
```

`32768 x 1,R4`：

```text
full row groups                  4,456,448
row utilization                 100%
Scalar state loads/stores       -33,554,432 each
PV shift/add pairs              -17,825,792
bank-conflict fallbacks         0 in tested trace
```

短/长 Matrix work 保持不变；physical HBM read/write bytes 也逐项不变：

```text
short read/write = 1,737,228,288 / 272,629,760 bytes
long  read/write = 9,896,198,144 / 1,090,519,040 bytes
```

Transactional numerical evidence：

```text
allclose match rate      100%
bitwise match rate        53.255%
mean absolute error        2.594e-4
maximum absolute error     0.00390625
P99 absolute error         0.001953125
P99 relative error         0.17360
relative-match rate       99.21875%
```

Bitwise 不一致来自改变合法 accumulation/state execution placement 后的浮点舍入；验收沿用原 correctness gate，没有宣称 bitwise equivalence。

`seq=4096/4097` 的 latency cliff 仍被保留，因为任何非空 tail 都执行 full-width BMM；本项目没有增加 active-row BMM ISA。

最终相关回归记录为 193 passes；Rust suite 94 passes、1 ignored。它们仍不替代 full top-level rtl-v6 workload test。

## 8. Compute Latency：从 Closed-form 到 CostEmitter

### 8.1 旧模型为何不足

旧解析模型的流程近似：

```text
model tensor shapes
    -> hand-written operation counts
    -> fixed latency constants
```

它可以估计主要 GEMM/BMM 数学量，但看不到：

- compiler 选择的 packed/unpacked layout；
- tile padding、tail 与 dummy rows；
- Q/K RMSNorm 和 online-softmax state maintenance；
- projection/RoPE/state movement；
- address generation 与 hardware loop；
- FFN 大立即数 legalization；
- route dispatch/combine；
- 最终 DMA occurrence、shape 和 physical role；
- schedule fallback 和容量导致的 K chunk change。

因此旧模型的主要误差不是“某条 opcode 常数差一点”，而是没有测量最终 compiled program。

### 8.2 CostEmitter 原理

CostEmitter 直接消费 compiler 最终 schedule 生成的 structured `CostTrace`。每个 kernel/action 至少记录：

```text
decoder stage ownership
opcode and operand/timing variant
dynamic multiplicity or RepeatAxis
logical and physical tile geometry
active rows/columns and padding
DMA opcode, address family, bytes and memory role
EnergyAction and ClockWork lineage
parallelization semantics
```

两个 backend：

- `detailed`：materialize 动态 occurrence，用于 parity/debug/transactional comparison。
- `affine-block-summary-v1`：保存 template + repeat axes/sufficient statistics，用于 DSE。

二者必须共享 schedule、DMA、EnergyAction 与 lineage IR；summary 不能在事后按公式随意减 cycles。

### 8.3 Timing modes

```text
ideal-II1:
    Matrix uses structural timing variant
    every Vector/Scalar/Control issue contributes 1 cycle
    assumes ideal initiation/overlap within model semantics

rtl-v1 / scheduled shadow:
    dependency-aware ready/done and resource scheduling
    used for sensitivity/validation
```

当前正式 DSE 使用 `ideal-II1`。它代表“如果各非 Matrix pipeline 可以 II=1 发射”的 architectural work，不代表真实 full RTL cycle trace。

### 8.4 旧模型与 compiled program 的公平对比

受控 setup：

```text
Qwen3-32B prefill, one decoder layer
seq=482, batch=16
MLEN/VLEN/BLEN=2048/2048/1024
memory excluded
CostEmitter ideal-II1
```

| Stage | Legacy closed-form | CostEmitter compiled trace | Ratio |
|---|---:|---:|---:|
| Normalization | 4.690 ms | 5.816 ms | 1.24x |
| Projection + RoPE | 0.140 ms | 5.155 ms | 36.9x |
| Attention core | 7.597 ms | 16.856 ms | 2.22x |
| Residual | 0.062 ms | 0.246 ms | 3.98x |
| FFN | 2.363 ms | 6.441 ms | 2.73x |
| Total | 14.851 ms | 34.521 ms | 2.32x |

这里 Legacy bar 是 closed-form term 的 semantic attribution；不是同一 opcode stream 的另一种 timing。主结论：missing compiler-generated work dominates discrepancy。

更早的 64-layer historical comparison：

```text
legacy analytic          0.950 s
real-trace ideal-II1     2.209 s
ordered one-cycle        2.374 s
rtl-v1 scheduled         7.517 s
```

这些是旧 compiler stack 的模型演进证据，不用于当前 rtl-v6 性能 claim。

### 8.5 当前受控 baseline

RTL-v5 compiler stack 下：

```text
Qwen3-32B seq482 B16, M2048/B1024:
    18.258 ms/layer
    1.061 s/64 layers

seq32768 B1:
    451.295 ms/layer
    28.357 s/64 layers

seq90000 B8 single-chip:
    25.459 s/layer
    1606.780 s/64 layers
```

最后一项是单芯片结构 baseline，不是 DSE 多芯片最终点。

### 8.6 CostEmitter 性能工程

长上下文最初会生成海量 per-row/per-tile Python objects，冷评估可达数十秒到分钟、RSS 数 GiB。

优化：

- symbolic repeats / `RepeatAxis`；
- causal-prefix DMA family folding；
- full/tail kernel template 只 materialize 一次；
- fixed-balanced MoE 不生成 route objects；
- final schedule semantic cache；
- one-layer result cache 后按 layer count 精确 scaling；
- algebraic HBM V4 grouping；
- process/global cross-study cache；
- AGU refolding 避免重新扫描展开 trace。

代表 long-context summary backend：

```text
worst cold path: ~27.03 s
warm path:       ~45.8 ms
peak RSS:        ~1.275 GiB
Q/K objects:     0 dynamic materialization
```

后续 semantic cache 和 trial claim 优化进一步支撑 288-worker remote DSE。

## 9. HBM V4 Memory Service Model

### 9.1 动机

Ramulator2 可作为 golden memory simulator，但逐 DSE trial 运行太慢，且早期 PLENA DMA emission 与真实 production scatter/gather/RMW semantics 不一致。旧 bandwidth constraint 直接 prune memory-bound points，也无法表达 channel/bank/row locality。

HBM V4 的目标：

> 用 production compiler DMA 的 physical 64-byte line census 和访问 pattern，快速预测 Ramulator2-equivalent service latency，让 memory-bound candidate 留在 DSE 中由 objective 自然淘汰。

### 9.2 输入与特征

每个 DMA family 重建：

- physical 64-B line addresses；
- read/write/RMW/gather/scatter semantics；
- channel/bank distribution；
- row hits、row switches、tails；
- request count、bytes、burst/stride geometry；
- role：weight、activation、matrix_kv、vector_kv、integer 等。

预测结构：

```text
latency = nonnegative physical floor
        + fitted nonnegative residual(features)
```

summary backend 对相同 full-address/feature signature 精确分组，而不是只按 bytes 比例缩放完整 latency。

### 9.3 Calibration dataset 与误差

```text
generic points:       2,592
row-anchor points:       36
training:             2,043
holdout:                585
```

Holdout 结果：

```text
median error      3.48%
P95 error        18.10%
max error        42.18%
weighted MAPE     2.80%
H_STORE P95      19.92%
manifest mismatch     0
```

Qwen trace 上 HBM work component 的误差可达 `0.79%-11.70%`，但 compute-dominated stage makespan 误差仅 `0.086%-0.573%`。后者不能用来掩盖 HBM 本身的 P95 error。

### 9.4 关键 correctness 修复

- 修复 Ramulator completion 计时。
- 使用 production DMA target，而不是旧 synthetic store target。
- 正确处理 scatter/gather 和 read-modify-write physical lines。
- KV overlay 同时匹配 `matrix_kv` 和 `vector_kv`，不只匹配 `role=kv`。
- multi-chip 按 rank-local DMA census 重算 V4，不按总 bytes 简单缩放。
- write aggregation vectorization，避免 90k context exact path 成为 bottleneck。

一个历史 bug 例子：旧 V3 synthetic store target 预测约 `12,289 ns`，production DMA 实际语义约 `52 ns`。因此 V3 只保留历史 artifact loader/comparison，不是当前 memory model。

### 9.5 边界

- 单芯片 calibration 是 Ramulator2/production DMA 对齐证据。
- N>1 的 HBM V4 是 rank-local analytical reconstruction，不是多 GPU/HBM distributed simulation。
- aggregate HBM capacity/bandwidth 预算由 DSE 资源模型提供，HBM V4 不证明现实系统一定达到 peak bandwidth。

## 10. Area Model

### 10.1 总体结构

当前 area proxy 由三部分组成：

```text
logic area:
    precision-aware structural models / nonnegative regressions

SRAM area:
    ASAP7 macro tiling with width/depth rounding

integration residual:
    full-chip paired points and conservative reserve
```

所有主要点使用 ASAP7 TT、0.7 V、25 C 的 Synopsys DC/macro 数据。Thesis 应把它称为 7 nm-class ASAP7 proxy，而不是 TSMC N7 实测。

### 10.2 MatrixMachine structural v4

Matrix area 不再只按 `MLEN x BLEN` 拟合高阶多项式，而是显式 census：

```text
PE count                       = MLEN x BLEN
cross-K reduction nodes        = BLEN^2 x (MLEN/BLEN - 1)
result/output/control structures
precision-dependent datapath width
```

验证结果：

```text
MXINT median/P95 error = 2.92% / 9.78%
MXFP  median/P95 error = 0.79% / 2.36%
```

原 presentation 中“MatrixMachine 71 train / 71 test”是将更广泛的 synthesis families 合并统计；当前 structural report 的直接 paired split 为：

```text
MXINT: 48 train / 48 holdout
MXFP:  23 train / 23 holdout
```

写 thesis 时应明确使用哪一种 dataset 计数，不能混在同一表中。

### 10.3 其他组件与 full-chip residual

历史/当前 holdout 指标：

```text
Matrix module        4.81% representative holdout
Vector module        4.19%
Scalar module        0.20%
HBM controller       5.54%
full-chip corrected composite MAPE 1.77%
logic-only MAPE                    2.15%
full-chip residual train/test      12 / 5
```

模型额外保留约 10% top-level integration residual/预算余量。

### 10.4 SRAM port semantics

DC 逻辑综合会 blackbox/idealize SRAM，因此 SRAM area 单独使用 macro table。

正式模式：

```text
ideal-dual-port
```

保守敏感性：

```text
replicated-single-port
```

RTL-v6 row banking 的 logical bits 不随 R 增加，但 shallow banks 分别支付 macro rounding/periphery。这部分现在使用 exact macro tiling，而不是简单乘一个 port factor。

### 10.5 DSE area budget

早期预算按 1 x A100 silicon area `826 mm2`。最终修正为：

```text
per-A100-equivalent PLENA area budget
    = 0.9 x 826
    = 743.4 mm2
```

10% reserve 用于：

- 未建模的 integration/routing/PHY/clock/power delivery；
- ASAP7 与实际 TSMC 7 nm iso-area 比较余量。

多预算 campaign 的总面积上限为 `P x 743.4 mm2`，不是每个 physical PLENA chip 都必须接近 743.4 mm2。

## 11. Power/Energy Model

### 11.1 v1 失败与 v2 重建

旧 power v1 的 holdout 很差：

```text
median error 38.08%
P95 error  1109.45%
```

因此不能使用。v2 改为 compiler action-based model：

- 每个 final-schedule action 有 EnergyAction lineage；
- ClockWork 表示实际 clocked instances；
- component idle/clock energy 按 mapped area 和 action state计算；
- HBM/SRAM/interconnect 单独计费。

Calibration：

```text
31 mapped configurations
395 activity replays
16 reused + 15 newly synthesized configurations
```

Grouped holdout：

| Component | Median | P95 |
|---|---:|---:|
| Matrix | 6.10% | 11.69% |
| Vector | 7.89% | 20.93% |
| Scalar | 0.90% | 8.69% |
| HBM controller | 7.82% | 21.65% |
| Overall | 7.47% | 20.88% |

```text
overall max error       28.29%
idle median/P95          3.83% / 8.34%
worst Qwen-like mix     11.41%
cache-hit evaluation     3.21 ms
```

### 11.2 Clock gating semantics

Nominal DSE 使用 `ideal-hierarchical`：

- inactive functional blocks 的 dynamic clock work 为零；
- active lanes/rows 按真实 mask 收费；
- leakage 始终存在；
- 不计 wakeup、gating controller 或 clock-tree residual。

因此它是理想下界，不是已经在 RTL 实现的 clock gating。`ungated` shadow 对所有 configured instances 收费，用于 sensitivity，不参与正式排序。

过去使用过 `CG50 = 0.5 x (ideal + ungated)` 的展示值，但它只是中间敏感性，不是测量 P50，也不应作为当前 canonical selector。

### 11.3 SRAM leakage

ASAP7 public SRAM Liberty 提供了 dynamic access energy，但没有可直接用于本模型的完整 leakage 数据。项目从公开 SRAM study 引入 `10 W/GB` 作为 MemExplorer 报告 `10-50 W/GB` 范围的最低端 proxy。

必须写清楚：

- 这是文献 lower endpoint，不是 ASAP7 实测；
- dynamic 仍来自 ASAP7 access characterization；
- SRAM leakage 不因 bank count 简单复制 logical capacity，但 periphery/integration leakage 通过结构 proxy 处理；
- 选择最低端会使总 energy 偏乐观。

### 11.4 HBM 与 interconnect energy

- External HBM 使用独立 HBM3E energy artifact，按 physical traffic 和 background capacity 计费。
- NVLink/interconnect nominal `8 pJ/bit`，保留 `1.3/70.9 pJ/bit` sensitivity。
- NVLink endpoint static/leakage 因缺少可信数据而排除。
- package、cooling、VRM、fabric/NVSwitch static 也未完整计入。

### 11.5 Cross-K reduction energy bug

早期 `matrix.cross_k_reduction` 缺 coefficient/fallback，可能让扁平 Matrix shape 的 energy 偏低。修复后：

- 单 K split 时 cross-K work 明确为 structural zero；
- 多 split 按真实 reduction nodes/action 计费；
- coverage 区分“合法零”与“遗漏 action”。

## 12. Multi-Chip Analytical Model 的四代演进

### 12.1 ideal-linear-lower-bound-v1

最早模型将几乎所有 work 和 traffic 直接除以 chip count `N`。在 aggregate HBM 固定时，小芯片还可能有更高 tile/area efficiency，因此 objective 必然偏向最大 N。

用途：只保留理论下界。不能作为正式结果。

### 12.2 factorized TP x CP v2

改进：搜索合法 `TP x CP`，而不是固定 TP。

```text
chip_count = TP x CP
TP divides Q heads and KV heads
CP uses causal zigzag two-chunk partition
```

加入：

- TP ring all-reduce；
- CP KV ring；
- CP weight replication；
- NVLink 1/2/4 ports；
- per-port 450 GB/s one-way peak；
- startup alpha 2.5 us；
- endpoint area 24.7 mm2/port nominal。

问题：compute/HBM 仍主要按 fraction 缩放，没有重新计算 local tile padding，导致大 N 仍过于理想。

### 12.3 tile-aware TP x CP x EP v3

核心修正：

```text
global shape
 -> rank-local logical shape
 -> MLEN/BLEN ceil/padding
 -> shared compiler planners
 -> local opcode/DMA/energy census
 -> slowest-rank stage latency
```

重新构建：

- SequencePackingPlan；
- AttentionHeadPacking/PackedGQA；
- FfnProjectionPlan；
- KVResidencyPlan；
- causal full/tail Q/K blocks；
- MoE expert bucket padding。

但初版 lineage audit 发现 final transformed schedule 的 kernel ownership/DMA/energy mapping 不完整，导致一些 CP/N16 结果虚假乐观。

明确 invalidated 的历史数字：

```text
TP4 x CP4 = 5.491 s
TP1 x CP16 = 2.413 s
```

这些只能用于说明 bug，不得出现在当前 performance conclusion。

### 12.4 tile-aware DP x TP x EP v4

根据目标 workload `batch>1` 和导师反馈，正式模型移除 CP，采用 request/data parallelism。

Dense：

```text
N = DP x TP
DP <= batch
EP = 1
```

MoE：

```text
N = DP x TP x EP
EP divides num_experts
rank = (dp_rank, ep_rank, tp_rank)
```

语义：

- DP 将完整 requests 分到 model replicas，不切一条 sequence，不通信。
- TP 切 heads/projection dimensions，并在 attention O 和 FFN down 后 all-reduce。
- EP 在 MoE experts 间切分，并做 dispatch/return all-to-all。
- shared dense/attention weights 在 EP ranks 复制，expert weights 在 EP 内分片、在 DP replicas 复制。
- 每个 rank 根据 local batch/head/expert bucket 重新调用 compiler planners。
- stage latency 取 slowest rank。

Latency：

```text
local phase = max(local compute, local HBM V4)

dependent TP/EP communication is added serially at dependency points
DP replicas execute in parallel
```

TP striped ring：

```text
active_rings = min(nvlink_ports, TP - 1)
T_tp = 2(TP-1)alpha
     + [2(TP-1)/TP x bytes]
       / [active_rings x 450 GB/s]
```

EP：从 fixed-balanced route census 生成 source-destination bytes，以 port-aware rounds 计算 dispatch/return；不与 expert compute 错误取 max。

### 12.5 Multi-chip 模型边界

- 这是 compiler-derived stage-level analytical model，不是 distributed cycle-exact simulation。
- NVLink 使用 100% architectural peak；没有经验 efficiency penalty。
- 假设 non-blocking fabric；未建模跨 group contention/NVSwitch area。
- 纯 DP 内部通信为零，但每芯片至少计一个 endpoint 供最终 KV handoff。
- 当前正式 workload 面向多 request；`batch=1` 时强制 `DP=1`，不会从 DP 得到虚假加速。
- CP 仍可作为 historical/sensitivity model，但不进入正式 DSE。

## 13. DSE Infrastructure 与 Search Domain

### 13.1 从单脚本到模块化 runner

原始 DSE 脚本逐步拆分为：

- CLI/domain generation；
- objective evaluator；
- worker pool/resource monitor；
- artifact retention；
- result/Pareto selection；
- cross-study compiler/area caches；
- multi-chip rank-local scorer。

支持：

- conditional categorical domain；
- Optuna TPE startup/random samples；
- serialized `ask()` 减少 SQLite lock contention；
- worker heartbeat、RSS recycling、emergency stop/recovery；
- complete-count target，而不是 attempt-count target；
- failed/pruned trace preservation；
- tmux 后台和 remote 288-core server。

### 13.2 Artifact compaction

最初 2048-point studies 的产物过大：

```text
short campaign: about 12 GiB
long campaign:   about 2.7 GiB
```

引入：

```text
--artifact-retention compact   # default
--artifact-retention full      # compatibility
```

Compact 保留：

- SQLite compressed checkpoint；
- all trials 的 compact summary；
- Pareto 和 selector 关键 trial 的完整详情；
- run summary/fingerprint/hash；
- failure reasons；
- 可恢复 state。

删除/压缩：

- worker JSONL/log/heartbeat；
- WAL/SHM checkpoint 后压缩；
- 非关键 per-trial full traces；
- 可重建 cache。

旧 2048 点瘦身结果：

```text
short: 247.7 MiB
long:  169.2 MiB
```

当前 16,384 COMPLETE campaign 的例子：

```text
before cleanup: ~3.95 GB
after cleanup:   275.1 MB
compression ratio: 6.97%
```

### 13.3 Remote execution

大规模 DSE 最终迁移到 `ee-tromokratis.ee.ic.ac.uk` 的用户空间：

- 288 logical CPUs；
- 默认 worker 数使用全部可见 quota，不预留线程；
- serialized Optuna ask + full-worker evaluation；
- shared global compiler/area cache；
- compact artifacts 回传本机。

过程中修复：

- workers 只有约 13 个实际运行：startup/TPE 串行 claim、cache miss 和内存 recycle 共同导致，不是 CPU domain 本身太小。
- unlimited quota 被误解析：增加 explicit unlimited worker handling。
- prune 过多导致 complete target 达不到：增加 attempt headroom，按 COMPLETE 而非 total trial 停止。
- 235B 错误 model/profile path：停止、清理错误 campaign 后重跑。
- long-context AGU refolding/cache 重复工作：引入 semantic cached summaries。

### 13.4 当前 precision domain

正式只保留 4-bit weight profiles：

```text
--allowed-weight-element-bits 4
```

过滤掉真实存储为 8-bit 的 weight formats，包括 MXINT8、MXFP E4M3/E5M2。Activation/KV 仍允许 4/8 bit。

当前 profiles：

```text
Qwen3-32B:          56 profiles, 24 Matrix signatures
Qwen3-235B-A22B:    51 profiles, 24 Matrix signatures
accuracy levels:    0.92, 0.94, 0.96, 0.98
constraint:         accuracy > 0.9
```

所有超过 0.9 的 profile 都可参与 selector，不要求比较候选 accuracy 相同。最终报告必须同时披露 accuracy 和 precision profile。

### 13.5 当前 hardware/search domain

```text
MLEN = VLEN       256,512,1024,2048,4096,8192
BLEN              conditional 32..1024
INT data width     16,32,64
Matrix SRAM        streaming, projection-full, KV25/50/75/100
softmax R          1,2,4,8,16
NVLink ports       1,2,4
chip multiplier    1,2,4,8,16 relative to P budget
TP/DP/EP           legal tuple generated conditionally
```

R support分层：

```text
ISA/RTL tiers:      R1,R2,R4,R8
model tiers:        R1,R2,R4,R8,R16
```

R16 只允许 ideal-II1 + affine summary，不生成 assembler binary；`--require-rtl-validated` 会排除。

### 13.6 Objectives

Prefill DSE 仍使用两目标最小化：

```text
prefill_latency_ms
prefill_system_energy_mj_ideal
```

旧字段：

```text
normalized_latency = prefill_latency_ms
normalized_energy  = prefill_system_energy_mj_ideal
objective_normalization = identity
```

它们没有除以 A100，只是兼容 alias。

为何这仍适合最终 system objective：对固定 decode/handoff 数据，任何同时降低 prefill latency 和 prefill energy 的设计都不会降低 pipeline output TPS 或 output tokens/J。因此 system selector 可以只重评分 prefill Pareto。但这有一个 accepted limitation：不同 ports/跨预算 mapping 可能使某些 prefill non-Pareto candidate 在 system 层翻盘。

### 13.7 Search budget

每组：

```text
16,384 COMPLETE trials
2,048 startup trials
128 EI candidates
artifact retention = compact
```

去除 W8 后没有继续增加预算，因为有效 precision x R 域已充分覆盖，而且每个 R tier 有 startup anchors。

### 13.8 五组最终 campaign

Workload：`input=90,000, output=8,000, global batch=8`。Area budget 为 `P x 743.4 mm2`。

| Model | P budget | COMPLETE | PRUNED | FAIL | Attempts | Unique prefill Pareto | Compact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32B | 2 | 16,384 | 22,712 | 0 | 39,096 | 5 | 275 MB |
| 32B | 4 | 16,384 | 27,661 | 0 | 44,045 | 7 | 292 MB |
| 235B | 4 | 16,384 | 17,116 | 0 | 33,500 | 10 | 347 MB |
| 235B | 8 | 16,384 | 22,000 | 0 | 38,384 | 20 | 406 MB |
| 235B | 12 | 16,384 | 20,477 | 0 | 36,861 | 16 | 361 MB |

PRUNED 主要来自：

- area/HBM capacity infeasible；
- conditional BLEN/MLEN/TP/EP illegal tuple；
- DP/EP weight replication 后 per-rank capacity 不足；
- R16/detailed fidelity gate incompatibility；
- duplicate physical candidates。

不是 compiler crash；五组最终 `FAIL=0`。

`fidelity_qualified_completed=0` 不能解释为“所有 trial 错误”。该字段要求严格的 full RTL/timing qualification；当前 formal search 有 ideal-II1、大 VLEN extrapolation 和 R16 structural points，因此没有 candidate 满足最严格全链 RTL gate。

### 13.9 DSE 中出现过的设计趋势

#### 为什么 prefill 仍选择扁平 Matrix

PLENA 原论文强调 flattened systolic array 对 decode GEMV 有利，但当前 prefill DSE 也常选择 `MLEN/BLEN=1024/128` 或 `2048/128`。原因不是 prefill 变成 GEMV，而是当前联合设计中：

- 长 sequence/global batch 提供大量 row parallelism，较大 MLEN 能装更多 token/query rows。
- `MLEN=VLEN` 绑定使增大 MLEN 同时增大 Vector row width，影响 attention tile 数和 softmax packing。
- 固定 area 下减小 BLEN 可放更多 MLEN rows；代价是 K 方向 split/cross-K reduction。
- ideal-II1 和 affine loops 使某些 K split/control overhead 较低。
- BLEN=128 与 HLEN=128 对 QK/PV 和 packed head lane 较自然。
- 修正后的 cross-K energy 已计入，但 timing/clock-gating/large-width extrapolation 仍可能偏好扁平 shape。

因此这个趋势是“当前 workload + tied M/V + ideal model”的结果，不能泛化为所有 prefill accelerator 都应扁平。

#### 为什么 P8/P12 有时 latency 接近

- TP 受 Q/KV heads divisibility 和 tile floor 限制；增加 TP 后 local dimension 可能仍执行同样数量 physical tiles。
- DP 不能超过 batch，B8 到一定程度出现 local batch=1/2 的 packing under-utilization。
- EP 只缩 expert work，但 shared attention/dense work 在 EP ranks 复制。
- 更多 P budget 不要求设计使用更多 area/chips；若 prefill 已不是 system bottleneck，selector 不会为无 TPS 收益支付能耗。

#### 为什么有些点选择 4 ports

- 4 ports 只在 TP/EP communication 位于 critical path 且 endpoint area 仍可接受时被选。
- 纯 DP 没有内部 communication，额外 ports 通常无 latency 收益。
- 当前 bandwidth 是 100% peak 假设，因此 4-port benefit 是 optimistic architectural bound。

#### 为什么没有 W2

当前 accuracy/calibration profile 只覆盖正式 W4/W8 families；最终又过滤为 W4-only。没有经过 accuracy、Matrix timing、area、power 和 correctness gate 的 W2 profile，因此不能把 W2 随意加入 DSE。

## 14. A100 RunPod Measurement Campaign

### 14.1 目的

PLENA 是 disaggregated prefill candidate，最终 baseline 应是 aggregated A100 serving，而不是只比较 paper 中一个不明确口径的 TTFT 数字。由于没有真实 PLENA hardware/KV import，A100 campaign 提供：

- aggregated runtime E2E/TTFT/TPOT；
- decode phase proxy；
- topology/local-batch scaling；
- GPU energy/power/capacity；
- system composition 所需 decode curves。

### 14.2 Environment

```text
RunPod Secure Cloud
8 x A100 SXM 80 GB
vLLM 0.19.0
PyTorch 2.10.0
32B:  Qwen/Qwen3-32B-AWQ
235B: QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ
backend: AWQ-Marlin where supported, otherwise AWQ
KV: FP16
```

32B long context 使用 YaRN factor 4、original max position 32768、`max_model_len=131072`；235B 使用 checkpoint native 262144 context。

运行设置：deterministic token IDs、greedy、ignore EOS、prefix cache/speculative/CPU offload disabled。

### 14.3 Workloads 与覆盖

```text
short:      1.4k input / 0.2k output
primary:   90k input / 8k output
holdout:  114k input / 5k output
global batch: 8 for system comparison
```

正式/补充 topology 包括：

32B：TP1/TP2/TP4/TP8 与可组合 DP replicas。
235B：TP4/TP8；超过 8 GPU 的 12/16 GPU baseline 使用 measured-replica extrapolation，并用双 TP4 concurrent run 检查复制误差。

原始 AWQ campaign audit：

```text
42 formal topology/workload points
70 measurement repetitions/samples
858 files, ~29.6 MB original campaign archive
```

随后增加了 replica concurrency、low-batch、full 114k、100 Hz short-energy、static-batch 和 topology follow-up。2026-08-18 本地完整树为：

```text
1,715 files
48,876,667 bytes
formal_v2 subtree: 1,497 files / 44,108,815 bytes
```

主要 `formal_v2` summary 数：

```text
screening 32B/235B              8 / 6
short 32B/235B                  8 / 6
full 114k 32B/235B              8 / 6
fixed-batch prefill audit          33
replica/low-batch follow-up         13
short-energy 100 Hz v2              14
static-batch probe                    3
topology follow-up                   15
```

这些 campaign 有重用/重测 topology，不能将 summary 数简单相加称为独立 formal points。

长 workload 全部完成且没有 observed preemption/prefix-cache hit，output token count 守恒。部分 short repeated runs 有 greedy token nondeterminism，因此这些 run 只能作为性能/energy measurement，不是 numerical correctness evidence。

另行下载了官方 Qwen3-32B BF16 source checkpoint，并写了 `FP16 runtime (BF16 source checkpoint)` sensitivity plan；当前本地 archive 只有 model revision/path 和计划，没有完成的 FP16 measurement summaries。因此正式 baseline 仍是 AWQ W4A16，不能引用未跑出的 FP16 性能。

### 14.4 Energy measurement

- 每张 GPU 读取 NVML total-energy counter。
- 同时 20 Hz 采样 power、clock、temperature、memory。
- counter 为正式 energy，积分采样用于 cross-check。
- 只统计实际分配 topology 的 GPUs。
- 每次 enqueue 前测 idle power，修复了最初把加载/排队阶段混入 decode proxy 的问题。

短窗口 energy 曾出现 NVML sample/counter warning；后续补跑用于修正短点。长点持续时间足够，energy 更稳健。

### 14.5 vLLM batch 与 TTFT 语义

长请求下 first-token completions 形成近似串行 staircase。不能仅凭这一现象断言 GPU 内部 mini-batch 恒等于 1；vLLM 会根据 `max_num_batched_tokens`、chunked prefill、KV capacity 和 scheduler budget 对 prompt chunks 调度。

必须区分：

```text
request-visible TTFT
    = first token - frontend/request arrival

scheduler-admitted TTFT
    = first token - first_scheduled_time

batch first-token barrier
    = all requests have emitted first token

throughput-equivalent interval
    = batch barrier / batch
    # service interval proxy, not TTFT

full-batch E2E
    = last output completion - campaign start
```

旧 raw data 没保存 `first_scheduled_time` 时，可在严格 staircase 条件下推断 admitted proxy：

```text
sample[0] = first completion - request start
sample[i] = completion[i] - completion[i-1]
```

只有 identical prompt、无 preemption、single completion timestamps、interval CV<=5%、barrier consistency<=5% 时才输出。

32B 90k TP4/B8 one-token audit：

```text
mean request-visible TTFT         94.834 s
inferred scheduler-admitted TTFT  21.0628345 s
batch first-token barrier        168.503 s
throughput-equivalent interval    21.063 s/request
staircase CV                       0.317%
```

关闭 chunked prefill 的 static probe：

```text
batch barrier                    159.030 s
interval                          19.879 s/request
```

由于 first tokens 成组同 timestamp 完成，不允许生成 admitted-TTFT proxy。

### 14.6 Representative A100 B8 data

32B 90k/8k：

| Topology | Full-batch latency | Energy |
|---|---:|---:|
| TP1 x DP8 | 73.10 s | 288.44 kJ |
| TP2 x DP4 | 77.56 s | 290.56 kJ |
| TP4 x DP2 | 84.10 s | 307.84 kJ |
| TP8 x DP1 | 105.26 s | 341.92 kJ |

235B 90k/8k：

| Topology | Full-batch latency | Energy | Fidelity |
|---|---:|---:|---|
| TP4 x DP2 on 8 GPUs | 111.86 s | 420.31 kJ | measured concurrent replicas |
| TP8 x DP1 | 208.79 s | 629.84 kJ | measured |

16-GPU aggregated comparison用 TP4 x DP4 measured-replica extrapolation：

```text
full-batch E2E       217.098 s
output TPS           294.7975
energy               1,143.832 kJ
output tokens/J      0.0559523
median request TTFT   44.836 s
P95 request TTFT      61.597 s
mean request TPOT      21.487 ms
P95 request TPOT       23.533 ms
```

### 14.7 与 paper TTFT 的关系

本地 `2509.09505v3.pdf` Table XII 的 A100/PLENA TTFT 口径在论文文本和作者沟通中仍存在歧义。安全写法：

- 并列 paper TTFT、earliest/median/latest request TTFT、batch barrier 和 admitted proxy。
- 不声称 paper 一定测错。
- 不把 PLENA fixed-B8 prefill makespan 除以 8 后称为 TTFT。
- 如果对比“scheduler-admitted service”，A100 可使用严格推断的 admitted proxy；PLENA 对应 fixed-batch admitted makespan，但两者 scheduler 语义仍不完全相同。

## 15. Disaggregated Serving Model 与最终指标

### 15.1 Resource budget

系统总资源：

```text
R_total = P + D
P = PLENA prefill A100-equivalent budget
D = A100 decode GPUs
```

正式 splits：

| Model | Aggregated baseline | Disaggregated splits |
|---|---|---|
| Qwen3-32B | 8 x A100 | P2:D6、P4:D4 |
| Qwen3-235B-A22B | 16 x A100 | P4:D12、P8:D8、P12:D4 |

PLENA resource constraints：

```text
area <= P x 743.4 mm2
HBM capacity <= P x 80 GB
HBM bandwidth <= P x 2039 GB/s
```

`reference_a100_count=P`，不能设为总资源后再额外加 decode GPUs。

### 15.2 Pipeline composition

单 batch E2E：

```text
PLENA fixed-batch prefill
 -> FP16 input-token KV handoff
 -> A100 imported-KV decode proxy
```

跨 batch steady-state：

```text
service interval = max(prefill interval,
                       handoff interval,
                       decode interval)
```

同一 batch 内三个阶段串行；不同 batches 允许 pipeline overlap。因此 throughput 可能提升，即使单 batch E2E 变长。

KV handoff：

- bytes 只包括 input tokens 的 FP16 KV；
- decode capacity 包括 input+output tokens 的 FP16 KV；
- latency 按 PLENA endpoint ports、decode replica mapping、slowest link scheduling；
- RunPod 内部 NVLink 不能冒充 PLENA-to-A100 measured link。

### 15.3 当前 canonical serving metrics

Phase schema：`request-visible-v4`。
System metric schema：`fixed-batch-serving-metrics-v3`。
Selector schema：`output-tps-energy-efficiency-v2`。

Latency：

```text
request-visible median/P95 TTFT
scheduler-admitted TTFT exact/proxy
mean/P95 TPOT
P95 TBT where token timestamps exist
full-batch E2E
```

Throughput：

```text
Aggregated output TPS = batch x output_tokens / measured full-batch E2E

Disaggregated projected output TPS
    = batch x output_tokens / pipeline service interval
```

Energy efficiency：

```text
output tokens/J = total output tokens / system energy
energy/output token = reciprocal
energy/request = system energy / batch
```

Requests/s 保留为辅助字段。没有显式 TTFT/TPOT SLO 时不使用 `goodput` 一词。

### 15.4 Selector

```text
maximize projected output TPS
maximize projected output tokens/J

subject to:
    E2E <= 1.25 x matched aggregated A100 E2E
    accuracy > 0.9
    area/HBM feasible
```

同时报告 1.0x/1.25x/1.5x sensitivity。1.25x 是主结果；1.0x 当前无可行 disaggregated endpoint。

主 energy 使用 ideal hierarchical gating；ungated 只作为 shadow，不改变 selector。

## 16. 当前 Primary 90k/8k System Results

以下结果来自 `primary_90k8_system_search_v3`，是当前最完整的 metric schema。它们仍是 analytical system composition，不是实机 PLENA deployment。

### 16.1 Qwen3-32B：Maximum Output TPS

System：

```text
split                  P2:D6
decode topology        TP2 x DP3, local batches 3/3/2
PLENA physical chips   16
PLENA parallelism      DP4 x TP4
MLEN/VLEN/BLEN         1024/1024/128
softmax R              8
Matrix SRAM            projection-full, 7 tiles
NVLink ports           1/chip
precision              W4/A4/KV4, FP E6M5
accuracy               0.92
PLENA aggregate area   ~924.6 mm2
```

Comparison：

```text
Aggregated A100 output TPS      224.66
Disaggregated output TPS        236.55   (1.053x)

Aggregated output tokens/J       0.066616
Disaggregated output tokens/J     0.098245 (1.475x)

Aggregated E2E                  284.869 s
Disaggregated E2E               340.399 s (1.195x)
```

Pipeline bottleneck 为 decode，因此进一步加速 PLENA prefill 不再提升 TPS，除非 decode 资源/拓扑同步变化。

### 16.2 Qwen3-32B：Maximum Output Tokens/J

```text
split                  P4:D4
decode topology        TP4 x DP1, B8
PLENA physical chips   32
PLENA parallelism      DP8 x TP4
MLEN/VLEN/BLEN         1024/1024/128
softmax R              8
Matrix SRAM            KV-25, 46 tiles
NVLink ports           1/chip
precision              W4/A4/KV4, FP E6M5
accuracy               0.92
PLENA aggregate area   ~2403.5 mm2
```

```text
Aggregated A100 output TPS      222.59
Disaggregated output TPS        205.52   (0.923x)

Aggregated output tokens/J       0.070466
Disaggregated output tokens/J     0.118939 (1.688x)

Aggregated E2E                  287.522 s
Disaggregated E2E               346.353 s (1.205x)
```

该点牺牲约 7.7% output TPS，换取约 1.69x output-token energy efficiency，仍满足 1.25x E2E constraint。

### 16.3 Qwen3-235B：Maximum Output TPS

```text
split                  P4:D12
decode topology        TP4 x DP3, local batches 3/3/2
PLENA physical chips   16
PLENA parallelism      DP2 x TP2 x EP4
MLEN/VLEN/BLEN         1024/1024/128
softmax R              16
Matrix SRAM            KV-25, 46 tiles
NVLink ports           4/chip
precision              W4/A4/KV4, FP E6M5
accuracy               0.96
PLENA aggregate area   2505.67 mm2
```

```text
Aggregated A100 output TPS      294.797
Disaggregated output TPS        333.87   (1.133x)

Aggregated output tokens/J       0.055952
Disaggregated output tokens/J     0.093741 (1.675x)

Aggregated E2E                  217.098 s
Disaggregated E2E               269.112 s (1.240x)
```

重要：R16 是 structural extrapolation，`rtl_validation_available=false`。该 endpoint 不能称为完整 RTL-validated design。

### 16.4 Qwen3-235B：Maximum Output Tokens/J

```text
split                  P8:D8
decode topology        TP4 x DP2, local batch 4
PLENA physical chips   16
PLENA parallelism      DP4 x TP2 x EP2
MLEN/VLEN/BLEN         2048/2048/128
softmax R              16
Matrix SRAM            streaming, 2 tiles
NVLink ports           2/chip
precision              W4/A4/KV4, FP E5M6
accuracy               0.92
PLENA aggregate area   1944.03 mm2
```

```text
Aggregated A100 output TPS      294.797
Disaggregated output TPS        313.487  (1.063x)

Aggregated output tokens/J       0.055952
Disaggregated output tokens/J     0.109104 (1.950x)

Aggregated E2E                  217.098 s
Disaggregated E2E               257.161 s (1.185x)
```

同样包含 R16 structural extrapolation。Energy result 还依赖 ideal clock gating 和 optimistic SRAM leakage endpoint。

### 16.5 如何解释这些结果

安全结论：

> Under a 1.25x full-batch E2E constraint, the analytical PLENA+A100 pipeline improves projected output TPS by up to 1.05x for Qwen3-32B and 1.13x for Qwen3-235B, while the energy-efficiency endpoints improve projected output tokens/J by 1.69x and 1.95x, respectively.

必须紧跟的限定：

- 32B endpoints 使用 R8，属于 RTL-v6 implemented tier，但 full top-level 未跑。
- 235B endpoints 使用 R16，只是 structural extrapolation。
- PLENA energy 使用 ideal gating 与低 SRAM leakage proxy。
- A100 12/16 GPU 数据含 measured-replica extrapolation。
- KV handoff 和系统流水是 analytical。
- 没有 queueing、continuous batching 或真实 TTFT/TPOT SLO goodput simulation。

不安全结论：

- “PLENA 实测比 A100 快 1.13x。”
- “PLENA 已证明节能 1.95x。”
- “R16 已在 RTL 1 GHz 工作。”
- “disaggregation 将 user TTFT 降低。”

## 17. 发现过的异常、根因与修复

| 现象 | 根因 | 修复 | 当前状态 |
|---|---|---|---|
| 旧 analytic 与 emulator/CostEmitter 差异大 | closed-form 漏 compiler-generated work | final schedule CostTrace/CostEmitter | fixed |
| Ramulator store 延迟异常高 | synthetic DMA target 与 production emitter 不一致 | production DMA physical-line semantics | fixed |
| 2048 trial 产物 2.7-12 GiB | 每 trial full trace/cache/worker logs | compact retention + summary + gzip | fixed |
| M8192/B32 FFN 64.855 s | dead pointer update + 18-bit immediate expansion | live-stride + register add | fixed |
| SRAM 更大反而 FFN 更慢 | projection-full 自动切 legacy template | unified affine-loop-v2 + guard | fixed |
| AGU-v2 没收益 | postincrement detector 找不到新增合法 streams | retire AGU-v2 | closed negative result |
| MoE CostEmitter ~42.6 s | route objects、重复 expert FFN lower、identity vectors | compact-route-v2/template replay | fixed |
| 多芯片总偏 N16 | all work `/N` 且无 tile/replication/comm | factorized then tile-aware models | superseded |
| CP v3 长 context 异常快 | final schedule lineage/ownership 错误 | fail-closed lineage audit | invalidated old results |
| TP 增加仍虚假加速 | fractional dimensions 不保留 tile floor | local shape planner reconstruction | fixed in v3/v4 |
| RMSNorm/residual 被 TP 错除 | kernel classification 粗糙 | ParallelKernelCensus semantic classes | fixed |
| `matrix.cross_k_reduce` energy 缺失 | coefficient/structural zero 未区分 | explicit cross-K action/zero | fixed |
| P8 235B energy 异常强 | Pareto CSV/energy field audit | corrected export and lineage checks | fixed; current ideal-gating caveat remains |
| Long DSE 只有 13 workers active | serialized startup claims、cold cache、RSS recycling | serialized ask + full workers + cache | fixed/improved |
| 大量 PRUNED | 80 GB/P capacity、replication、conditional legality | corrected KV handoff/capacity semantics | expected constraints |
| 235B campaign model错误 | wrong campaign profile/path | kill, clean and rerun | final campaigns clean |
| vLLM B8 first tokens 近似串行 | scheduler/chunked prompt budget，而非简单“batch=1”证明 | phase audit and source-level semantics | interpretation fixed |
| `prefill_latency_s` 被当 TTFT | 实际是 batch first-token barrier | schema rename/compatibility aliases | fixed in working tree |
| `mean_tpot_s` 实为最慢 generation span | 旧 phase tracker 没逐 request TPOT | request-visible-v4 token timestamps | fixed in working tree |
| throughput/W 单位含糊 | requests/s 与 output tokens/s 混用 | output TPS + output tokens/J | fixed in working tree |
| rtl-v6 初期像“免费 60%” | 未计真实 banks/row slices/area | production banking + DC/power calibration | model corrected |
| Vector SRAM banking area 大 | shallow macro rounding/per-bank peripheral | exact macro tiling + R sweep | expected feature |
| R4 以后加速小 | Amdahl limit、state/PV 已优化、tail/bank/other stages | report marginal speedup/mm2 | expected feature |

## 18. 负面结果、退役路径与已失效结果

### 18.1 明确退役

- **AGU-v2**：没有发现新增合法 streams，实测无收益。删除 runtime/compiler path，报告保留。
- **legacy-auto-v1 FFN automatic selection**：只作为 no-regression oracle，不由 SRAM 容量自动触发。
- **HBM V3**：只保留 artifact loader/历史复现。
- **legacy closed-form latency**：移入 comparison/legacy namespace，不进入正式 objective。
- **factorized TP/CP v2**：保留 analytical baseline，不是正式 multi-chip 模型。
- **CP v3 formal DSE**：目标 workload 改为 batch-oriented DP；CP 只作历史/长单 request sensitivity。
- **power v1**：holdout 误差不可接受，不能再引用。

### 18.2 已失效数字

以下不能作为当前结论：

```text
TP4 x CP4 = 5.491 s
TP1 x CP16 = 2.413 s
```

原因：pre-lineage-audit multi-chip scaling。

旧 `all work/N` 下的 N16 best points 只代表 ideal lower bound。

早期 DSE 面积预算 `1.10 x 826 mm2` 已由 `0.9 x 826 mm2` 替代。

2026-08-02 freeze 前的五组/2048-point Pareto 属于 pre-banking、pre-DP-v4 或 old precision domain，不进入当前 selector。

### 18.3 仍有价值的 historical A/B

- closed-form vs CostEmitter：证明为何需要 compiler-derived model。
- compact layout/direct-first-block/AGU/FFN/MoE：证明每层优化来源。
- factorized/tile-aware CP：解释 multi-chip model 如何被审计和改进。
- rtl-v5 vs rtl-v6：定义硬件增量。
- old artifact sizes：证明 DSE engineering scalability。

## 19. Validation Matrix

### 19.1 Compiler/Functional

覆盖：

- short/long sequence、batch packing、dummy rows；
- packed GQA 与 multi-tile causal mask；
- FFN K tiles 1/2/4/7/multi-chunk；
- streaming/projection-full/KV policies；
- MoE top-k 1/2/8、balanced/skewed/static tiny cases；
- R1/2/4/8 full/tail row groups；
- MXINT/MXFP precision variants。

要求/已采用：

- transactional correctness gate；
- Matrix QK/PV work invariant；
- HBM opcode/bytes/manifest invariant where optimization should be compute-only；
- ASM/CostTrace/emulator opcode histogram parity；
- resolved address sequence parity for AGU/FFN changes。

### 19.2 RTL

已运行证据包括：

- decoder/assembler opcode tests；
- Scalar ROB/AGU focused tests；
- Vector multi-segment reduction/compact SIMD tests；
- softmax row engine R1/2/4/8 module cases；
- banked SRAM address/conflict tests；
- packed-PV overwrite/accumulate module tests；
- VectorMachine integration wrappers。

未完成的 gate：full PLENA top-level packed-GQA execution。

### 19.3 Model

- HBM V4 train/holdout 与 manifest parity。
- Area nonnegative coefficients、holdout errors、monotonicity。
- Power action/replay holdouts、structural zero coverage。
- N=1 multi-chip identity。
- rank-local kernel/DMA/EnergyAction/ClockWork coverage 100%。
- token/request/expert/route conservation。
- lower bound <= nominal <= upper bound。

### 19.4 DSE

- 64 COMPLETE smoke before full campaign。
- each campaign exactly 16,384 COMPLETE。
- zero FAIL in current five campaigns。
- study fingerprint includes compiler schedules、R domain、weight domain、area/power calibration、multi-chip schema。
- compact artifact retains Pareto full details and all-trial resume summary。

### 19.5 Serving

- output token count/global batch conserved。
- per-request TPOT identity：`TPOT x (tokens-1) = last-first token`。
- output tokens/J 与 energy/output token reciprocal。
- output TPS = requests/s x output tokens/request。
- no duplicate A100 prefill in disaggregated composition。
- staircase proxy rejects grouped completions/preemption/nonidentical prompts。
- replica reconstruction includes slowest makespan and idle tail energy。

## 20. 当前未完成、风险与 Thesis 必须披露的限制

### 20.1 RTL-v6

- Production VectorMachine/SRAM integration 是当前未提交 working tree。
- Full-core top-level 没有跑通/没有作为本轮目标。
- R16 不可 ISA 编码；R32 只出现在 microbenchmark 趋势分析。
- 大 VLEN 逻辑由小宽度 DC 点外推。
- 1 GHz 是假设，不是 timing closure。

### 20.2 CostEmitter/Latency

- Ideal-II1 不模拟所有 dependency、backpressure、bank conflict 和 pipeline latency。
- rtl-v1 shadow 对最新 trace 的部分 reduction lineage fail closed，尚不能为所有 current points 提供完整 scheduled ratio。
- Full-batch PLENA prefill 是静态编译 makespan，没有 per-request streaming completion model。

### 20.3 Area/Power

- ASAP7 proxy 与目标 foundry/process 有偏差。
- SRAM dual-port 是 formal ideal；replicated shadow 更保守。
- SRAM leakage 使用文献 lower endpoint。
- ideal clock gating 没有 RTL implementation/overhead。
- NVLink endpoint static、fabric、package/cooling 未完整计入。

### 20.4 HBM/Multi-chip

- Multi-chip HBM 是 analytical rank-local scaling。
- NVLink bandwidth 使用 100% peak。
- no fabric contention/NVSwitch model。
- fixed-balanced MoE routing，不模拟实际 expert skew dynamics。
- system selector 只使用 prefill Pareto，可能漏掉少量 system-level non-Pareto reversal。

### 20.5 A100/Serving

- vLLM 在线 scheduler 与 PLENA fixed batch 口径不同。
- 旧 runs 无 exact `first_scheduled_time`，admitted TTFT 是严格条件下的 proxy。
- 235B 12/16 GPU 是 measured-replica extrapolation。
- 无真实 PLENA KV import；decode phase 是 proxy。
- 无 queueing/continuous batching/arrival trace，因此 output TPS 是 fixed-batch pipeline envelope，不是 production goodput。

## 21. Thesis 推荐故事线

### Chapter 1：Problem and Motivation

- PLENA 原工作偏向 Matrix-dominated analytic view。
- Qwen3 prefill 的 compiled execution 暴露大量 normalization、softmax state、layout、address 和 data movement。
- Ramulator/transactional simulation 太慢，无法直接进入大规模 DSE。
- Disaggregated deployment 的价值主要是 energy efficiency，而不一定是 raw TTFT。

### Chapter 2：Compiler-Derived Modeling Stack

- CostEmitter 和 Structured CostTrace。
- HBM V4 sufficient-statistics model。
- Precision-aware area 和 action-based power。
- Fidelity taxonomy 与 validation methodology。

### Chapter 3：Compiler Optimizations

- compact packed GQA/layout。
- segment-parallel normalization。
- loop AGU-v1。
- affine-loop-v2 FFN。
- compact-route-v2 MoE。
- partial KV residency。

### Chapter 4：Hardware Enhancements

- Scalar pipeline/ROB/AGU。
- multi-segment reduction + compact-stat SIMD rtl-v5。
- rtl-v6 query-row parallel softmax、packed state、direct PV。
- R-way SRAM banking area/power trade-off。

### Chapter 5：Tile-Aware Multi-Chip and DSE

- 为什么 `/N` 模型错误。
- factorized CP 的历史与为何改为 DP。
- DP/TP/EP rank-local reconstruction。
- conditional DSE、artifact/cache/worker engineering。
- 五组 budget-conditioned Pareto。

### Chapter 6：A100 Measurement and System Composition

- RunPod measurement harness、phase/energy semantics。
- TTFT/TPOT/output TPS 定义。
- aggregated A100 curves。
- PLENA prefill + A100 decode analytical pipeline。
- maximum TPS 与 maximum tokens/J endpoints。

### Chapter 7：Limitations and Future Work

- full RTL-v6 top-level validation。
- R16 implementation/timing。
- calibrated clock gating/SRAM leakage。
- real PLENA-to-GPU KV transfer。
- queueing/continuous batching/system scheduler。
- non-fixed-balanced MoE routing。

## 22. 可安全使用的核心 Contributions

建议论文明确列为贡献：

1. **Compiler-derived CostEmitter**：从 final lowering 生成 stage/opcode/DMA/energy lineage，替代 closed-form tensor estimate。
2. **HBM V4**：从 production DMA physical-line pattern 快速预测 memory service，支持 memory-bound DSE points。
3. **Qwen3 dense/MoE compiler stack**：packed GQA、AGU-v1、unified affine FFN、compact MoE route。
4. **Segment-parallel Vector/Scalar path**：multi-reduction、compact-stat SIMD、auto lane tiers。
5. **RTL-v6 online-softmax candidate**：query-row spatial parallelism、packed state、direct PV，以及真实 SRAM banking/area/power calibration。
6. **Tile-aware DP/TP/EP model**：按 rank-local shape 重建 compiler work，而非 fractional `/N`。
7. **Scalable DSE infrastructure**：conditional search、semantic cache、compact artifacts、remote full-worker execution。
8. **Serving evaluation methodology**：真实 A100 measurements 与明确 phase semantics，使用 output TPS/tokens-J 组合 analytical disaggregation。

不要把以下写成已证明 contribution：

- full-chip RTL-v6 at 1 GHz；
- R16 hardware implementation；
- measured PLENA system energy；
- real disaggregated serving goodput；
- production-grade clock gating。

## 23. 关键报告与 Artifact 索引

### Canonical/current

| Topic | Path |
|---|---|
| Area model | `Workspace/reports/area/precision_aware_area_model_v4.md` |
| HBM V4 | `Workspace/reports/hbm_v4/hbm_dma_service_v4_full_report.md` |
| Power v2 | `Workspace/reports/power/rtl_activity_power_candidate_v2.md` |
| Ideal gating | `Workspace/reports/power/ideal_hierarchical_clock_gating_v1.md` |
| Current compiler stack | `Workspace/reports/compiler/current_stack_implementation_comparison_20260727.md` |
| FFN affine v2 | `Workspace/reports/compiler/unified_affine_ffn_loop_lowering_v2.md` |
| MoE compact route | `Workspace/reports/compiler/moe_compiler_optimization_v2.md` |
| Partial KV | `Workspace/reports/compiler/partial_resident_kv_ideal_dual_port_dse.md` |
| RTL-v6 | `Workspace/reports/rtl/multirow_softmax_state_pv_v1.md` |
| RTL-v6 one-layer A/B | `Workspace/reports/rtl/rtl_v6_long_context_single_layer_ab_v2.md` |
| DP/TP/EP v4 | `Workspace/reports/dse/dp_tp_ep_multichip_model_v4.md` |
| A100 audit | `Workspace/reports/runpod/a100_awq_campaign_audit_v1.md` |
| Fixed-batch phase audit | `Workspace/reports/serving/fixed_batch_prefill_audit_v1.md` |
| Current system search | `Workspace/reports/serving/primary_90k8_system_search_v3.md` |
| Experiment plan | `Workspace/disaggregate_serving_DSE_plan.md` |

### Historical but useful

- `legacy_analytic_vs_costemitter_m2048_b1024.md`
- `native_decoder_compact_layout.md`
- `native_packed_gqa_optimization_v2.md`
- `packed_gqa_pipeline_optimization_v1.md`
- `loop_agu_v1.md`
- `vector_scalar_rtl_v2_optimization.md`
- `vector_scalar_rtl_v3_segment_parallel.md`
- `factorized_multichip_model_v2.md`
- `tile_aware_multichip_model_v3.md`
- `dse_artifact_compaction_ffn_address_lowering_v1.md`

### 审计提醒

- `Workspace/reports/system_validation_status.md` 最后更新时间早于当前 rtl-v6/serving schema，不能作为最终总状态。
- `Workspace/reports/README.md` 的分类部分已过时，应以本文和 git history 为准。
- `primary_90k8_system_search_v2` 已由 v3 指标语义替代。
- 报告中的 1 GHz、ideal gating、broadcast unvalidated、R16 extrapolation 声明不能删。

## Appendix A：Root Simulator 完整提交账本

以下为 `main..yx/prefill-DSE` 的全部 87 个项目提交，按时间正序。它是最可靠的“我们先后做了什么”索引。

```text
2026-06-05 3f19bf2 Fix emulator output comparison compatibility
2026-06-05 5223777 Respect explicit head_dim in simulator configs
2026-06-05 cbdc8c6 Respect explicit head_dim in analytic llama model
2026-06-05 f96c881 Add Qwen3-8B sliced prefill probe
2026-06-05 2c49dec Add Qwen3-8B native full-size preset
2026-06-06 41f93db Add opcode memory profiling for emulator
2026-06-11 8d43b2a Add analytic prefill mode for MoE models

2026-06-22 c98aaa5 Add analytic area proxy model
2026-06-22 7c89c37 Add Qwen3 dense analytic latency model
2026-06-26 9a92a9e Parameterize transactional HBM channels

2026-07-13 879b2cf Add precision-aware area proxy models
2026-07-17 3d578bf Fix Ramulator completion and production DMA semantics
2026-07-17 b399bab Add RTL-calibrated transactional timing scheduler
2026-07-17 ec3b66c Add RTL timing calibration and validation harnesses
2026-07-17 24a9688 Update PLENA Compiler for scheduled cost traces
2026-07-17 6e370c1 Add CostEmitter calibration and regression fixtures
2026-07-17 4cdd3aa Add RTL-calibrated CostEmitter scheduling models
2026-07-17 531f750 Add accepted production-DMA HBM service V4 calibration
2026-07-17 b1d47c5 Update development environment for parallel Optuna DSE
2026-07-17 fa6f8a2 Add Qwen3-32B precision-aware Optuna DSE
2026-07-17 23e3e72 Document transactional rtl-v1 and HBM V4 validation
2026-07-17 8c1b746 Ignore generated HBM calibration run artifacts
2026-07-19 8656edd Add structural MatrixMachine area model v4
2026-07-19 112d6f5 Integrate compact compiler traces with rtl-v1 and HBM V4
2026-07-19 b976fa1 Scale Qwen3 precision DSE with compact CostEmitter models
2026-07-19 f98b009 Document the calibrated PLENA prefill modeling stack

2026-07-26 af4737e Update PLENA compiler dependency for optimized prefill lowering
2026-07-26 df104a0 Synchronize the transactional emulator with RTL-v4
2026-07-26 e107098 Add versioned RTL timing calibration artifacts
2026-07-26 ad40b49 Calibrate structural area overlays for prefill hardware
2026-07-26 9ccca28 Add action-based on-chip and external-memory power models
2026-07-26 9db3f15 Integrate HBM V4 with compressed compiler cost traces
2026-07-26 5c78b2b Add tile-aware TP-CP-EP analytical scaling
2026-07-26 53e1126 Modularize canonical four-objective DSE execution
2026-07-26 4ad7df8 Quarantine historical analytic models and generated artifacts
2026-07-26 4329bf5 Restore scalar ROB scheduling with RTL-v4 timing
2026-07-26 4c4a845 Audit and document the finalized PLENA prefill modeling stack
2026-07-27 6025187 Update compiler dependency for compressed long-context lowering
2026-07-27 fe9d9e2 Fold causal-prefix DMA families in HBM V4
2026-07-27 e9d07d1 Fix streamed KV capacity and persistent DSE worker budgets
2026-07-27 0410d28 Distinguish structural zeroes in matrix energy coverage

2026-08-02 aed9ab2 Update compiler dependency for RTL-v5 normalization
2026-08-02 fc35600 Synchronize emulator and timing models with RTL-v5
2026-08-02 82d4551 Calibrate auto-tiered compact SIMD area scaling
2026-08-02 a1650f3 Calibrate RTL-v5 compact SIMD action energy
2026-08-02 5b97f82 Model MoE SRAM demand and decode KV handoff endpoints
2026-08-02 eaaad63 Refine latency-energy DSE for dense and MoE prefill
2026-08-02 d1dfe24 Audit reports for the finalized RTL-v5 modeling stack

2026-08-06 22de506 Update the compiler for RTL-v6 packed attention
2026-08-06 d87467a Synchronize the emulator with RTL-v6 attention
2026-08-06 79dfede Model RTL-v6 softmax and packed-PV area
2026-08-06 922ad07 Model RTL-v6 actions and SRAM background power
2026-08-06 14f47d3 Vectorize exact HBM V4 write aggregation
2026-08-06 58841fe Add tile-aware DP-TP-EP analytical scaling
2026-08-06 de86d08 Integrate RTL-v6 costs and semantic V4 caching
2026-08-06 2ac3c4a Adopt RTL-v6 DP-aware four-objective DSE

2026-08-07 50a92b8 Scale DSE workers across large shared servers
2026-08-07 ee59c72 Report actual DSE worker pool utilization
2026-08-07 9f642e7 Condition disaggregated DSE on deployable capacity
2026-08-07 cac6845 Accelerate serialized DSE trial claims
2026-08-07 5097862 Compact finalized DSE studies and expand sample budgets
2026-08-07 2e46c21 Stream compact DSE trial artifacts
2026-08-07 934c604 Retain only essential DSE trial summaries
2026-08-07 692f05b Compress finalized DSE summaries

2026-08-08 4b5d13c Add resumable A100 serving benchmark campaign
2026-08-08 db88046 Fix constrained DSE Pareto result export
2026-08-08 c9ac56d Support ephemeral RunPod model caches
2026-08-08 3d9b7a4 Handle styled NVIDIA topology output
2026-08-08 6524ce6 Support current vLLM RoPE overrides
2026-08-08 69064d5 Map YaRN overrides to current Transformers config
2026-08-08 d173644 Capture NVLink traffic without DCGM
2026-08-08 1e17b00 Allow audited runtime locks without image metadata
2026-08-08 b5bc438 Fix RunPod decode proxy and idle power measurement
2026-08-08 4d3eb65 Measure GPU idle before enqueueing requests
2026-08-08 b215624 Report nondeterministic greedy benchmark outputs

2026-08-15 b039a82 Update compiler dependency for multi-row softmax summaries
2026-08-15 b2580f5 Calibrate banked RTL-v6 VectorMachine area
2026-08-15 354e576 Audit RTL-v6 softmax scheduling and pipeline timing
2026-08-15 b7810a7 Calibrate bank-aware RTL-v6 power deltas
2026-08-15 e11f902 Align DSE objectives resources and calibrated profiles
2026-08-15 f1d1484 Expand prefill DSE campaigns for W4 and R16
2026-08-15 0a9abb9 Correct A100 phase semantics and serving selectors
2026-08-15 7b8e136 Audit disaggregated serving evaluation semantics
2026-08-15 e35be97 Preserve compact DSE failure tracebacks
2026-08-15 c3765c3 Stabilize TPE startup under full-worker concurrency
2026-08-15 19464cf Handle unlimited worker quotas in TPE startup
2026-08-15 e6cb85f Increase DSE attempt headroom for pruned domains
```

## Appendix B：Compiler 完整项目提交账本

### B.1 本项目 Yuxuan 提交

```text
2026-06-24 20e9b69 Fix Qwen head_dim and matrix SRAM sizing
2026-06-24 4111950 Support sequence-tiled packed GQA attention
2026-06-24 829d517 Fix sequence-tiled packed GQA correctness
2026-06-24 60f8147 Use hardware KV loop for tiled packed GQA
2026-06-24 b9a3b3f Generalize packed GQA head chunking
2026-06-25 9cd46db Alias vector shift opcode in assembler
2026-07-11 0345156 Optimize packed GQA around logical KV groups
2026-07-12 f26581f Add symbolic cost emission for native Qwen3 prefill
2026-07-17 0cdb87f Add compressed schedule traces for CostEmitter
2026-07-19 1c1d77a Add native Qwen3 MoE and compact prefill lowering
2026-07-26 7e29ab5 Extend PLENA ISA assembly for RTL-v4 and loop AGU v1
2026-07-26 6d100dc Optimize native packed-GQA lowering for prefill
2026-07-26 01762d8 Unify affine FFN lowering and partial K/V residency
2026-07-26 bd7b656 Optimize MoE lowering and rebuild CostTrace lineage
2026-07-26 1470478 Consolidate canonical compiler schedule profiles
2026-07-27 d3318c1 Compress long-context packed-attention cost lowering
2026-08-02 1440916 Auto-scale compact statistics lowering for wide vectors
2026-08-06 4708891 Extend the assembler for RTL-v6 softmax operations
2026-08-06 ce6c2b2 Add multi-row softmax and direct packed-PV lowering
2026-08-15 0ba3b65 Integrate multi-row softmax lowering and R16 summaries
```

### B.2 关键 inherited prerequisites

这些提交在 compiler branch 历史中，但作者不是本项目，thesis acknowledgements/attribution 应保持准确：

```text
7539bdd Kevin   native decoder attention for seq_len < MLEN
a4c80f8 Kevin   batch_size > 1 native MHA and true sub-64
74ff7a5 Kevin   O(n^2) -> O(n) ISA emission and parser speedups
e6f4955 Kevin   FFN inner K accumulation hardware C_LOOP
ebdba9e Kevin   multi-tile causal attention mask correctness
bd91d46 George  per-region hbm_sizes frontend metadata
17b2bd0 George  normalization instruction order for RTL simulation
```

### B.3 Compiler branch topology

```text
yx-optimized-gqa-attention-before-main-rebase-20260625
    -> preserves original five packed-GQA commits

yx-optimized-gqa-attention
    -> rebased/current compiler research branch

yx/upstream-cost-trace-v1
    -> generic final-schedule symbolic CostTrace extraction
       HEAD 64a6118 Normalize summary templates inside symbolic repeats
```

## Appendix C：RTL 完整项目提交账本

```text
2026-07-19 3e69165 Make PLENA RTL precision profiles synthesis-safe
2026-07-19 0beb43f Add reproducible RTL checking and DC area workflows
2026-07-26 b5feafa Add pipelined scalar and segment-parallel vector datapaths
2026-08-02 c77b5c5 Scale compact statistics SIMD with vector width
2026-08-06 dea4b6d Decode RTL-v6 multi-row softmax operations
2026-08-06 364a1e6 Add banked softmax and packed-PV RTL-v6 datapaths
```

当前未提交 integration 涉及：

```text
src/control/rtl/data_flow_control.sv
src/control/rtl/pipeline_control.sv
src/core/rtl/plena.sv
src/core/rtl/plena_artix7_top.sv
src/matrix_machine/rtl/matrix_machine.sv
src/matrix_machine/rtl/packed_pv_accumulator.sv
src/matrix_machine/rtl/packed_pv_writeback.sv
src/memory/vector_sram/rtl/fp_vector_sram.sv
src/vector_machine/rtl/softmax_row_engine.sv
src/vector_machine/rtl/softmax_state_bank.sv
src/vector_machine/rtl/softmax_state_simd.sv
src/vector_machine/rtl/softmax_value_bank.sv
src/vector_machine/rtl/vector_machine.sv
integration wrappers and cocotb tests
```

## Appendix D：Upstream Extraction Branches

### D.1 CostEmitter branch

在 Qichao/Arlo routed-MoE emulator 与 deterministic timing 基础上，本项目抽取：

```text
8f1941a Add compiler-trace latency schemas and timing providers
80dfa7a Evaluate stage compute work from symbolic schedules
3496ca4 Integrate compiler-derived latency with the existing CLI
c1c0164 Validate Qwen dense and MoE cost traces against the emulator
523434d Cache compiler-derived latency by semantic inputs
6a82866 Modernize compiler-trace latency typing
```

### D.2 HBM V4 branch

在 CostEmitter branch 上增加：

```text
70eff6f Add a generic production-DMA HBM V4 backend
5db336e Compress affine DMA service with exact feature statistics
03a4d28 Recalibrate HBM V4 against main Ramulator semantics
2ccb517 Polish the standalone HBM V4 package
```

### D.3 Power branch

在 HBM branch 上增加：

```text
089eef9 Add compiler-action on-chip energy accounting
1e33f53 Add ideal clock-gating and external HBM power bounds
6985d0d Calibrate main-compatible action energy and Qwen holdouts
e170fb5 Validate detailed and compressed analytical traces end to end
```

## Appendix E：当前代码状态（2026-08-22）

Root 已分批提交：

- `f88888b`：serving phase schema `request-visible-v4`、request TPOT/TBT、scheduler-admitted TTFT exact/proxy、output TPS/output tokens/J 以及 system report regression；
- `9be3e1b`：按 model 解析 weight parameter count，并为 completed DSE campaign 增加 resume/skip 逻辑。

RTL `1e0eb06` 已提交 production Vector SRAM banking、VectorMachine row engine、tagged softmax state pipeline、packed-PV Matrix writeback、control/hazard 和 focused tests。新增 focused cocotb tests 通过；full-core top-level 未运行，不作 full-chip RTL performance claim。

Compiler 当前 clean，HEAD `0ba3b65`。

`PLENA_Tools` dirty 由外部/本地状态造成，本项目不得修改或提交。

## Appendix F：缩写表

| 缩写 | 含义 |
|---|---|
| MLEN | MatrixMachine 长维/row dimension |
| BLEN | MatrixMachine block/inner array dimension |
| VLEN | VectorMachine row width |
| HLEN | Attention head dimension |
| GQA | Grouped-Query Attention |
| AGU | Address Generation Unit |
| QK/PV | Attention 的 QK transpose 与 probability-value Matrix operations |
| R | softmax query-row spatial lanes |
| TP | Tensor Parallelism |
| DP | Request/Data Parallelism |
| CP | Context Parallelism，当前正式模型已移除 |
| EP | Expert Parallelism |
| TTFT | Time To First Token；必须注明 request-visible 或 admitted semantics |
| TPOT | Time Per Output Token |
| TBT | Inter-token Time / Time Between Tokens |
| TPS | 本文 canonical 指 output tokens per second |
| E2E | Full-batch end-to-end makespan |
| CostTrace | Compiler final schedule 的结构化动态工作表示 |
| EnergyAction | 与 final kernel/opcode lineage 绑定的动态能耗 action |
| ClockWork | 实际被 clocked 的 instances/work accounting |
| HBM V4 | production-DMA physical-pattern memory service proxy |

## Appendix G：交给另一 AI 的最短上下文

如果上下文预算有限，可只提供下面这段：

> 本项目为 PLENA 构建了 Qwen3 dense/MoE prefill 的 compiler-derived DSE stack。核心软件贡献是 packed GQA、loop AGU-v1、unified affine FFN、compact MoE routing，以及从 final compiler schedule 生成 opcode/DMA/EnergyAction lineage 的 CostEmitter。HBM V4 用 production DMA 的 64-B physical-line/channel/bank/row sufficient statistics 拟合 Ramulator2，holdout median/P95 error 为 3.48%/18.10%。Area 使用 ASAP7 precision-aware structural regression + SRAM macro tiling，power 使用 RTL activity action model，overall power holdout median/P95 为 7.47%/20.88%。硬件方面，rtl-v5 增加 segment-parallel reduction 与 auto-tier compact-stat SIMD；rtl-v6 候选进一步增加 R-way query-row online-softmax、packed m/l state 和 direct PV accumulation。R1/2/4/8 是实现域，R16 只为结构外推。正式 compute latency 用 ideal-II1 architectural model，不是 full-RTL cycle exact；RTL-v6 production datapaths 已接入 Vector SRAM、VectorMachine、Matrix writeback 和 core control，focused module tests 通过，但 full-core top-level 未跑。Multi-chip 模型从错误的 work/N 和 TP/CP 演化到 tile-aware DP/TP/EP，按 rank-local shape 重新生成 tile/opcode/DMA，并串行计 dependent TP/EP communication。最终五组 90k/8k DSE 每组完成 16,384 trials，weight 固定 W4，accuracy>0.9，area budget 为每个 A100-equivalent 743.4 mm2。RunPod 测量了 8xA100 上 32B/235B 的 1.4k/0.2k、90k/8k、114k/5k，当前 serving schema 使用 request TTFT、TPOT、output TPS 与 output tokens/J。解析组合的 1.25x E2E constrained endpoints：32B 最大 TPS 为 A100 的 1.053x、tokens/J 1.475x；32B 最大 efficiency 为 TPS 0.923x、tokens/J 1.688x；235B 最大 TPS 为 1.133x、tokens/J 1.675x；235B 最大 efficiency 为 1.063x、tokens/J 1.950x。所有 system results 都是 analytical fixed-batch pipeline envelope，无真实 PLENA-to-A100 KV import；235B endpoints 使用 R16 structural extrapolation，能耗还依赖 ideal clock gating 和低 SRAM leakage proxy。
