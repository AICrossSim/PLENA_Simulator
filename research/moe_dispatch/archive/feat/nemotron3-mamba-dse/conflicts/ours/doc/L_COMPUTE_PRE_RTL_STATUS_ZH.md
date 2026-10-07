# L-Compute RTL 前冻结状态

## 目标

Compiler 告诉硬件下一阶段会同时读取哪些 projection 值；`L_SCATTER_M`
在片上按 consumer packet 把值写进 banked layout buffer，然后 `X_STATE`
直接消费该布局。逻辑 tensor 不变，也不经过 HBM transpose/repack。

## 已完成

| 项目 | 结果 |
|---|---|
| ISA | `L_SCATTER_M=0x3F`，256-byte、64-byte aligned descriptor |
| 模式 | `ROW_MAJOR`、`TRANSPOSE`、`MAMBA_SKEW`、`KDA_SKEW`、`CUSTOM` |
| Compiler | L 指令位于 matching `X_STATE` 之前；descriptor 与 HBM image 同时生成 |
| Rust | 真实 `banks x rows` 数据存储；duplicate write、unwritten read、identity mismatch 直接失败 |
| Flow | 64-value FIFO、spill、backpressure、write/read bank service 全部计入 counter |
| 数据正确性 | Mamba/KDA 全值 roundtrip；alias mapping 负向测试会失败 |
| 连续数值链 | `Mamba -> MoE` 与 `AttnRes -> KDA -> AttnRes -> MoE` 在 Rust 对拍通过 |
| 完整结构 | Nemotron 52 层为 23 Mamba + 23 MoE + 6 Attention；Kimi 93 层为 69 KDA + 24 MLA |
| Transactional prefill | S16/S128 的 state chunk、GQA/MLA cache 与 multi-token MoE 全部在 Rust 对拍 |
| Compact 整模 | Nemotron 52 层与 Kimi 93 层在一次 Rust 运行中完成 S16 prefill + decode 4 token |
| Mixed precision | state 支持 FP32/BF16/FP16/MX8-B128；第一版 activation 冻结为 BF16 |

## 冻结的第一版参数

```text
layout banks              16
ports per bank            1
producer burst            64 BF16 values
projection FIFO           64 values
Mamba mapping             group-major + local-row/group skew
KDA mapping               group-major + k rotation 8 + beta group rotation 1
descriptor                256 B, 64-B aligned
activation                BF16
```

冻结依据不是单个 toy counter：analytic `ProjectionFifoSpillModel` 的逐拍 FIFO
sweep 中 64 与 256 的最大占用和周期相同；
Mamba/KDA 的 full-shape decode trace、物理数据往返和 compact Rust 数值链全部通过。

## Bank 结果

| Workload | Row-major read | Co-layout read | Read speedup | Read+write reduction |
|---|---:|---:|---:|---:|
| Nemotron 23 Mamba 层 | 53,176 | 14,904 | 3.568x | 56.16% |
| Kimi 69 KDA 层 | 536,544 | 430,560 | 1.246x | 14.15% |

Mamba co-layout 的 read stall 为 0，但 full trace 写入增加 92 cycles；KDA 的
`k=8` 与 beta group rotation 后 read/write stall 都为 0。表中是 layout-buffer
service，不是整层或整芯片 speedup。

纯 transpose 也不是结论。对 `16 x 128` tile，row-major 与 transpose 的读写总和
都是 2,176 cycles；只有同时考虑 producer 写入和 consumer 读取的 diagonal
co-layout 才降到 256 cycles。因此论文点应写成 compiler-guided producer-consumer
co-layout，而不是“增加 transpose 指令”。

## Rust 连续执行证据

| Path | Cycles | Max abs error | 结果 |
|---|---:|---:|---|
| Nemotron real-state Mamba | 1,710,884 | 0.015625 | BF16 tolerance 内通过 |
| Nemotron Mamba -> MoE | 1,725,597 | 0.03125；handoff=0 | 通过 |
| Kimi KDA | 72,348 | 0 | 通过 |
| Kimi KDA -> MoE | 94,523 | 0 | 通过 |
| Kimi AttnRes -> KDA -> AttnRes -> MoE | 96,980 | 0 | 通过 |

Mamba 的 64-entry 与旧 256-entry 版本周期相同——但这一条**不能引用 Rust 的相等作为
证据**：Rust 的 `fifo_capacity_values` 只进 `fifo_peak_values = min(burst, capacity)`，
`fifo_backpressure_cycles` 是 spill 宽度与 write packet 数的闭式，与容量无关，所以两个
容量给出相同周期是模型的恒真式。真实证据来自 analytic 侧逐拍步进的
`ProjectionFifoSpillModel`（有真实 occupancy 与 high-watermark）。KDA 和 Mamba
assembly 都明确是：

```text
Matrix projection
L_SCATTER_M
X_STATE
Vector/output/next block
```

## Mixed precision 结论

CPU reference 已覆盖 state storage 的 FP32、BF16、FP16、MX8-B128。Mamba 128-token
实验中 MX8 output relative-L2 为 1.18%，KDA 64-token 为 0.286%；这说明 MX8 值得
进入 DSE，但不足以证明真实模型质量不掉。RTL 前只冻结格式支持，不冻结默认 state
precision；默认仍使用官方 runtime 的 FP32 state，BF16/FP16/MX8 必须经过长序列与
任务精度实验。

## Compact 整模证据

| Model | Instructions | Cycles | 检查 |
|---|---:|---:|---|
| Nemotron 52 层 | 426,814 | 13,660,404 | 1,040 checkpoints 100%；23 state、6 GQA cache、920 route entries |
| Kimi 93 层 | 4,646,741 | 80,526,139 | 3,740 checkpoints 100%；69 state、24 compressed cache、3,680 route decisions |

这两条是完整层顺序和数据生命周期的执行证据，不是把单层周期相加。它们使用合成权重
和缩小的 hidden/head/expert 外围维度，因此不能作为真实 checkpoint 的 TTFT/TPOT。
Kimi 的 3,680 个 route decisions 都进入对应 MoE 后的 checkpoint；其中最终一层 40 个
expert IDs 还会从 Int SRAM dump 直接对拍，前面各层不声称保留了全部 route-ID dump。

## 不能声称的内容

1. 当前 Rust 的 architectural L pass 从已经完成的 Vector-SRAM Matrix writeback 读取；
   RTL 中计划做的 `M_MM_WO -> L_SCATTER_M` stream tap 尚未实现。
2. Compact 52/93 层已在 Rust 从第一层运行到最后一层；但真实 checkpoint 权重和真实
   外围维度尚未整模执行，不能给出真实模型 PLENA latency。
3. Rust bank cycle 未包含 mux、地址生成器、布线、SRAM macro timing 和 PPA。
4. `TRANSPOSE` 已有 descriptor、物理 roundtrip 和 dense microbenchmark，但还没有通用
   Matrix consumer 的端到端 Rust 数值链；它不能替代 Mamba/KDA co-layout 证据。
5. 当前数字不能作为 FPGA 相对 GPU 的端到端加速比。

## 进入 RTL 的门槛

Compiler/Simulator 侧已经满足开始 RTL 的功能门槛。RTL 第一阶段只实现冻结配置：
16 single-port banks、64-value FIFO、五种 mode 的地址生成、L/X scoreboard 和 counters。
综合后必须重新报告频率、面积、写/读 stall 与端到端周期；如果 mux 代价抵消收益，
应回到 DSE 调整 bank 数或 mode，而不是保留无法证明的复杂度。
