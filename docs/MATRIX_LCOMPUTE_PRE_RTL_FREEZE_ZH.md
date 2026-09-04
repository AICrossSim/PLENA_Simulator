# Matrix SRAM L-Compute：Pre-RTL 冻结版本

本文是 Compiler/Simulator 阶段的唯一交接说明。它只冻结已经由代码和测试
支持的行为；不包含 RTL、综合、频率、面积、功耗或 Token/J 结论。

## 1. 最终架构边界

```text
HBM -- 显式 viewed DMA --> 现有 banked Matrix SRAM
                                |
                         L_TILE 多行遍历
                                |
                  固定对角映射 + 编译器 tile phase
                                |
                         cyclic lane restore
                                |
                    现有 Vector 乘加与归约
                                |
Matrix SRAM -- 显式 viewed DMA --> HBM
```

Compiler 决定全部地址、生命周期、搬入和写回。最终候选明确不包含：

- cache、tag、命中率、替换策略或隐式一致性；
- 私有 recurrent-state SRAM；
- `X_STATE`、模型专用 STEP 指令或 256-byte HBM descriptor；
- 运行时队列、调度器、事件或完成记录；
- 新 MAC lane 或额外 Matrix SRAM 端口。

Matrix SRAM、prepared fields、state 和 output 统一存 BF16；现有算术通路内部
使用 FP32 累加。官方 GPU 的 FP32 state 只作为 baseline 和精度参考。

## 2. 冻结 ISA

Matrix-view descriptor 的冻结版本是 v3。v3 删除了可编程 row coefficient 和
可选存储精度；解码器必须拒绝原先占用这些位置的 bit，不能静默解释成新语义。

Matrix L-Compute 只占一个物理 opcode：

```text
L_TILE_CFG   slot, shape_reg, map_reg
L_TILE_EXEC  dst_base, src_base, scale_base, primitive[, axis_mask]
```

- `L_TILE=0x3F`
- `funct1=1`：原子配置一个 Matrix view
- `funct1=3`：遍历 view 并执行一个通用代数原语
- slot 数量固定为 4
- operand base 显式写在 `L_TILE_EXEC` 中，不使用隐式 `SELECT`
- 配置先于使用由 Compiler 的控制流支配检查保证

两个配置字冻结为：

```text
shape   = rows_minus_one[11:0] | cols_minus_one[23:12]
        | tile_count_minus_one[31:24]
mapping = tile_pitch_rows[15:0] | reserved_zero[21:16]
        | tile_phase_stride[27:22] | flags[31:28]
```

`mapping[21:16]` 和 flag bit 0/1/2 必须为零。flag bit 3 只表示短字段广播；
边界检查始终开启。固定对角 row 系数恒为 1，不在
descriptor 中重复编码。`tile_phase_stride=0` 是普通固定布局；非零值只负责
紧凑表达多个 tile 的编译期基址相位。Matrix-view 存储格式固定为 BF16。

三个原语是：

| 原语 | 数学语义 |
|---|---|
| `SCALE_ACCUM` | 分段执行 `dst = a*dst + b*src` |
| `DOT_REDUCE` | 分段乘加归约 |
| `OUTER_UPDATE` | rank-1 destination update |

它们不包含 Mamba、KDA、模型名、固定 head 数或 cache 行为。Mamba-2 和 KDA
由 Compiler 组合同一组原语。viewed load/store 复用现有 `H_PREFETCH_V` 和
`H_STORE_V` 的地址形式，不新增物理 opcode。

完整静态 Mamba/KDA 路径还使用两个与 Matrix layout 正交的算术 opcode：

| Opcode | 用途 |
|---|---|
| `V_SOFTPLUS_V=0x3D` | 生成 Mamba/KDA 系数 |
| `S_MAP_FP_V=0x3E` | 整行 Vector 到 FP register file 的反向映射 |

因此完整功能相对共同 main 的物理 opcode 增量是 3 个；其中 Matrix
L-Compute 本身只占 `0x3F`。`V_FMA_VF` 是现有 `V_MUL_VF` 的编码模式，不另占
opcode。

旧 `L_CFG` 位于 `0x3F/funct1=0`，仅用于复现历史 Vector-stream 实验。官方
52/93 层 schedule 有测试保证不会发出它；它不属于交给 RTL 的冻结接口。

## 3. 冻结物理映射

每个逻辑 bank word 使用：

```text
bank_row = base_row
         + tile * tile_pitch_rows
         + logical_row * row_groups
         + word / bank_count

bank = (base_bank
      + bank_row
      + tile_phase_stride * tile
      + fixed_group_phase * (bank_row / bank_count)
      + word) mod bank_count
```

读写使用同一公式，读出后做循环 lane 恢复。Compiler 在生成程序时拒绝
alias、越界和超容量 view。

PLENA 原有固定对角 row term 保持不变。任意可编程 `row_skew` 已删除：公平
对照 D′ 证明固定对角接线加合法的 per-tile base phase，能在 Nemotron 和 Kimi
官方 packet 上达到与旧可编程版本完全相同的物理坐标和零 stall。

保留 `tile_phase_stride` 的理由不是宣称额外 bank 加速，而是用一个 descriptor
紧凑表达多个 Compiler 已知的 tile base phase，避免每个 head 单独配置。

## 4. Compiler 策略

Arlo 的逐行静态 lowering 保留为功能 fallback 和 B baseline。它与 L-Compute
不是同一 tensor 上的前后两次搬运；Compiler 对每个 recurrent region 二选一：

```text
不支持、小尺寸或尾块 -> Arlo 逐行 Vector lowering
规则的大型 view      -> L_TILE 多行 lowering
```

官方 schedule 中，23 个 Nemotron Mamba layer 和 69 个 Kimi KDA layer 都发出
合法 `L_TILE`。Attention、MLA、MoE 继续走 PLENA 原有路径，不伪装成 L-Tile。

## 5. 当前证据

已通过的主要证据：

- Compiler、assembler 和 Rust 对同一 32-bit contract 解码一致；
- checked connected artifact 显式记录 schema v2 / Matrix-view contract v3，
  与当前 Compiler 不一致时测试直接失败；
- Rust 使用真实 `banks x rows x bank_width` 数据，而不是只统计公式；
- row/column access、viewed DMA、lane restore、output writeback 和负向 alias
  检查通过；
- Nemotron `64 heads x 128 state rows x 64 head-dim` 与 Kimi
  `96 heads x 128 key-dim x 128 value-dim` 的 recurrence geometry 连续执行
  4 token，state/output 对拍通过；
- synthetic Mamba/KDA S128 chunked prefill 对拍通过，但不是整模 TTFT；
- 公开 Mamba2-130M 的 24 层真实权重链中，每层 recurrent core 由 Rust
  `L_TILE` 执行；外围算子仍由 host BF16 reference 执行；
- 普通 Attention/MLA/MoE 的 row/column service 在 64 个 allocation phase 下
  不退化；
- D′ 与 D 的纯 bank-service 比值为 `1.00x`，因此没有把不可证明的
  programmable-skew 收益写进结论。

公式时间线在 `MLEN=2048`、`BLEN=32`、64 banks、1 MiB BF16 Matrix SRAM、
1560 HBM B/cycle 下给出 B1 decode：

| Model | Original A | Arlo B | L-Tile D | D/A | D/B |
|---|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,014,094 | 2.0134x | 1.5442x |
| Kimi K3 | 103,816,704 | 97,013,856 | 91,173,903 | 1.1387x | 1.0641x |

这些是带显式 SRAM/HBM/算术项的 pre-RTL timeline，不是硅上周期，也不是相对
GPU 的加速。D/B 主要来自多行执行、紧凑 descriptor、减少发射和 spill；纯
bank mapping 收益必须单独写成 D/D′=`1.00x`。

## 6. 资源代理

当前只能给结构上界：

- 额外 SRAM payload：0
- cache metadata：0
- 新 MAC lane：0
- 额外 SRAM read/write port：0
- 4 个 view record：256 bits
- sequencer：不超过 256 bits 加 3 个 loop counter
- programmable row-bank coefficient：已删除，新增加法器 0
- tile phase：每个活动 view 一个 6-bit accumulator，4 slot 上界 24 bits
- cyclic lane restore：64 个 512-bit bank word、6 级循环 mux

没有 RTL 综合前，不得把这些代理改写成面积、频率、功耗或 PPA 数字。

## 7. 明确未完成

- Nemotron/Kimi 真实完整 checkpoint 的全算子 Rust 执行；
- 完整 transactional prefill 的 A/B 加速与整模 TTFT；
- 精确 scoreboard bank-word overlap；当前 E 不获得虚构 overlap 收益；
- RTL、PPA、Token/J 或相对 B200/5090 的硬件加速。

这些不影响 ISA/Compiler/Simulator 的 pre-RTL 冻结，但在论文中必须明确区分。

## 8. 交接门禁

```bash
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute /absolute/path/to/PLENA_Compiler
```

只有该命令退出码为 0，且官方 schedule 不含 `L_CFG`、所有数值对拍和 D′ 公平
检查通过，才允许进入 RTL 阶段。

本次冻结使用 Compiler commit
`c2e7d03e14b4c43350fd3d232cb2ee6058a494c4`，门禁退出码为 0：

- Simulator Python：105 passed；
- Compiler：188 passed；
- Rust workspace：180 passed；
- Matrix projection：65,664 个 BF16 值逐项一致，0 bank stall；
- official-geometry recurrence：Nemotron/Kimi fixed/phased 四组全部通过。

关键机器可读证据的 SHA256：

```text
01a8965c58c9203c05272edab50459b64fe66fb5f4340166d57218c6d5b180c6  artifacts/matrix_lcompute_connected_bf16/summary.json
d775693d8284e4ebf9454e7fd753f56cf05b3ef73ecdd5ef79b8ed9fafae2e06  artifacts/matrix_lcompute_e2e_v5/campaign.json
9ca3e834f5968cefae47272991d87af86bd33109b19062d05d9ba3c44c1b02fc  artifacts/matrix_lcompute_e2e_v5/headline.csv
3f0f015c2dc420b3ee13c827a61b086ec78904a5005efa72e6a303864af7534b  artifacts/matrix_lcompute_agentic_v1/campaign.json
11c549ad31da440fe8973af98eca5e2234b4d99bdb4a061cd27a019e5bab41c5  artifacts/matrix_lcompute_agentic_v1/summary.csv
```
