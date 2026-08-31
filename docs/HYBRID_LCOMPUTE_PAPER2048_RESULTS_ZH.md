# Hybrid L-Compute：PLENA 2048-wide 验证结果

## 1. 测了什么

本轮把 Compiler/Simulator 重新对齐到 PLENA 论文最终系统点：

```text
BLEN=32, MLEN=2048, VLEN=2048, 1 GHz
HBM=1560 B/cycle/device（由论文的等带宽 16-device 对比推导）
```

论文没有给出 output SRAM 的 bank 数。本文候选使用 `32 banks x 64 FP16
elements`、每 bank `2R1W`，保持一个 2048-element 普通 Vector row 一拍可读。
Nemotron Mamba 保留 64-element 状态行；Kimi KDA 使用自然的 128-element
状态行。后者很重要：若继续沿用旧 64-wide 编译结果，会把 KDA 普通基线多拆一倍，
虚高 L-Compute 收益。

没有加入 cache、`X_STATE`、Mamba/KDA 专用指令或隐藏 state SRAM。新增语义仍只有
通用 `L_CFG`；数学由现有 `V_MUL_VF/V_FMA_VF` 执行，消费者通过指令内的
`funct1[2:0]` 明确选择三个 consumer view；第四个 slot 固定服务 Matrix final
writeback。`V_FMA_VF` 是 `V_MUL_VF` 的 accumulate 模式，不占独立 opcode。

## 2. 2048 packet 如何工作

一个 packet 含 32 个 64-element bank word：

- Mamba：一次合并 32 个 64-element 状态行，每段可有不同 scalar。
- KDA：一个 packet 读取 32 个 128-element 状态行的同一个 64-element atom；
  连续两个 packet 分别处理前/后半行，并复用同一组 32 个 scalar。
- row-major packet 把 32 个 word 挤到同一 bank；`alpha=1` 令第 `r` 段旋转到
  第 `r mod 32` 个 bank，读出后做逆旋转恢复 lane 顺序。

Affine packet 同时做紧凑存储：row-major 的 32 个短行分散在 32 个宽物理行，
affine 将同一 packet 的 32 个 bank word 放进一个宽物理行。这是同一批数据的
置换，不是额外 SRAM。Rust 在真实 `L_CFG -> 显式 lmask 的 V_FMA_VF` dispatch 上验证了：

| 布局 | 物理行/packet | 每个 FMA packet 的额外 bank stall | 数值 |
|---|---:|---:|---|
| row-major | 32 | 46 cycles | 与 golden 一致 |
| affine-skewed | 1 | 0 | 与 row-major 逐元素一致 |

另一个官方 KDA-head 数测试使用 96 行、每行两个 atom、共 6 个 packet，验证
跨 32-row block 后 scalar 会正确前进；这条专门防止只测首个 block 而漏掉错配。

普通 2048-element Attention/MoE row 不绑定 packet view；1-bank 与 32-bank
backing 的结果、周期和 packet counters 完全相同。

## 3. Compiler 结果

以下是官方尺寸一个递推层的动态发射数，不是整层硬件周期：

| Workload | 静态基线 | Arlo 后自增 | 普通 stream | 2048 affine packet |
|---|---:|---:|---:|---:|
| Nemotron Mamba recurrence | 92,399 | 51,311 | 33,257 | 19,049 |
| Kimi K3 KDA mixer | 215,387 | 116,219 | 81,659 | 61,115 |

元素数严格守恒：

- Nemotron：每层 `256 V_MUL + 256 V_FMA`，即每种操作覆盖
  `256 x 2048 = 64 x 128 x 64` 个 state elements。
- Kimi：每层 `768 V_MUL + 768 V_FMA`，即每种操作覆盖
  `768 x 2048 = 96 x 128 x 128` 个 state elements。

两份 assembly 均已生成合法 32-bit machine words。

## 4. 完整 52/93 层 Decode

统一时间线包含共享 Matrix、Vector、HBM 和 banked output SRAM。权重精度按实际
checkpoint policy，activation 为 BF16，recurrent state 为 FP32。表中是 B1、连续
4 token、context 2048：

| 模型 | B: Arlo | E: stream | H: row packet | I: affine packet | J: I+overlap |
|---|---:|---:|---:|---:|---:|
| Nemotron 52 层 | 12,570,486 | 10,909,518 | 11,745,614 | 9,614,158 | 9,602,382 |
| Kimi 93 层 | 393,100,112 | 383,561,552 | 397,286,480 | 377,904,656 | 377,891,408 |

| 对比 | Nemotron | Kimi | 回答的问题 |
|---|---:|---:|---|
| H -> I | 1.22170x | 1.05129x | 只改变物理排布，斜存是否消冲突 |
| E -> I | 1.13473x | 1.01497x | packet 是否超过最佳普通 stream |
| B -> J | **1.30910x** | **1.04025x** | 完整机制相对 Arlo 基线 |

H 的完整时间线 bank stalls 分别为 2,166,784 和 19,501,056 cycles；I/J 均为 0。
32-token decode 得到几乎相同的稳态加速：Nemotron 1.30909x，Kimi 1.04025x。

## 5. Lane DSE

每个点都由 Compiler 重新生成 packet，不是把 64-wide 周期除以 lane 数：

| Lanes | Nemotron I/E | Nemotron J/B | Kimi I/E | Kimi J/B |
|---:|---:|---:|---:|---:|
| 64 | 0.95287x | 1.09795x | 0.99566x | 1.04026x |
| 128 | 1.03647x | 1.19671x | 0.99768x | 1.02282x |
| 256 | 1.08500x | 1.25407x | 1.00679x | 1.03200x |
| 512 | 1.11128x | 1.28514x | 1.01144x | 1.03669x |
| 1024 | 1.12497x | 1.29926x | 1.01379x | 1.03906x |
| 2048 | **1.13473x** | **1.30910x** | **1.01497x** | **1.04025x** |

因此 affine packet 相对普通 stream 的 crossover 是：Mamba 约 128 lanes，KDA
约 256 lanes。64-wide 点上它只消冲突但不值得；2048-wide 点上才同时通过冲突和
整模性能 gate。

## 6. HBM 带宽边界

| HBM B/cycle | Nemotron J/B | Kimi J/B |
|---:|---:|---:|
| 64 | 1.01578x | 1.00176x |
| 256 | 1.06121x | 1.00698x |
| 512 | 1.11764x | 1.01380x |
| 1024 | 1.21827x | 1.02701x |
| 1560（论文点） | **1.30910x** | **1.04025x** |
| 2048 | 1.38139x | 1.05180x |
| 4096 | 1.60884x | 1.09571x |

低带宽时权重/HBM 淹没片上优化；带宽升高后递推发射和多行取数才进入关键路径。

## 7. Prefill 与资源边界

S16/S128 的 52/93 层 prefill 时间线均通过，但 chunked Mamba/KDA 尚未 lowered
到 decode packet 路径，所以 J/B 为 1.00x。当前结论只主张 decode 加速。

不做 RTL 时只能报告结构代理：

```text
额外 SRAM payload: 0（同容量 SRAM 分 bank）
短行 footprint: row-major 32 physical rows/packet -> affine 1 row/packet
bank geometry: 32 x 1024-bit, 2R1W
FIFO: 2048 x FP16 = 32,768 bits = 4 KiB
affine address adders: 32
cyclic lane restore: 2048 lanes
layout registers upper bound: 1,920 bits
```

这些不是面积、功耗或频率数据。`32 x 64` bank 几何是否值得，只能在后续 RTL/PPA
阶段回答。

## 8. 完成边界

已完成：官方层数/尺寸、Compiler machine code、Rust 2048 packet 数值与 bank 服务、
52/93 层 symbolic-weight prefill/decode 时间线、A-J 消融、lane/bank/FIFO/HBM/precision
DSE，以及 GPU shape/bottleneck 交叉检查。

未完成：真实 Nemotron/Kimi checkpoint 从第一层数值执行到 logits、chunked prefill
packet、RTL/PPA 和实测能耗。因此本文数据是 Compiler/Simulator 架构估计，不能写成
真实芯片或完整 checkpoint 的 measured speedup。

复现命令和 canonical hash 见
[`artifacts/hybrid_lcompute_paper2048_v1/README.md`](../artifacts/hybrid_lcompute_paper2048_v1/README.md)。
