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

因此，窄口径的 Mamba/KDA 功能路径相对共同 main 需要 3 个物理 opcode；
其中 Matrix L-Compute 本身只占 `0x3F`。`V_FMA_VF` 是现有 `V_MUL_VF`
的编码模式，不另占 opcode。

旧 `L_CFG` 位于 `0x3F/funct1=0`，仅用于复现历史 Vector-stream 实验。官方
52/93 层 schedule 有测试保证不会发出它；它不属于交给 RTL 的冻结接口。

### 2.1 相对 main 的完整 opcode 增量

上面的 3 个是功能路径口径，不是整条分支的合并口径。相对 Compiler main
`d89ad59`，本分支占用了原先剩余的全部 7 个 6-bit opcode：

| Opcode | 名称 | 引入 commit | 递推 lowering 是否使用 |
|---|---|---|---|
| `0x39` | `C_ROUTE_BEGIN` | `aec6dcb` | 否 |
| `0x3A` | `C_ROUTE_LOOP_START` | `aec6dcb` | 否 |
| `0x3B` | `C_ROUTE_LOOP_END` | `aec6dcb` | 否 |
| `0x3C` | `V_ROUTE_MUL` | `aec6dcb` | 否 |
| `0x3D` | `V_SOFTPLUS_V` | `56fd25a` | 是 |
| `0x3E` | `S_MAP_FP_V` | `56fd25a` | 是 |
| `0x3F` | `L_TILE` | `01f37ea` | 是 |

Matrix L-Compute 自身增加 1 个 opcode，完整静态递推路径需要 3 个，而原样合并
本分支会增加 7 个。`0x39..0x3C` 属于与 Matrix L-Compute 正交的 MoE routing
分析路径，`matrix_recurrence_lowering.py` 不会发射它们。**本次 Matrix
L-Compute 交接明确不包含这四条指令**；它们在实现前必须拆到 routed-MoE
分支，或作为另一个阶段单独评审。

若七条全部合并，main 的 6-bit opcode 空槽会从 7 个降为 0；后续扩展只能复用
`funct1` 子形式或使用扩展编码。交接评审必须附带
`git diff main..HEAD -- doc/operation.svh` 的完整输出，不能用“功能路径 3 条”
替代真实分支差异。

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
1560 HBM B/cycle 下给出如下 B1 decode 敏感性：

| 模型 | 权重密度 | 端点 | A | B | C | D | D/A | D/B | D/C |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | mixed NVFP4 | 严格串行 | 4,055,091 | 3,110,067 | 2,192,850 | 2,014,094 | 2.0134x | 1.5442x | 1.0888x |
| Nemotron 3 | mixed NVFP4 | 理想重叠 | 2,127,686 | 1,876,583 | 1,876,583 | 1,876,583 | 1.1338x | 1.0000x | 1.0000x |
| Nemotron 3 | uniform BF16 | 严格串行 | 6,360,486 | 5,415,462 | 4,498,245 | 4,319,489 | 1.4725x | 1.2537x | 1.0414x |
| Nemotron 3 | uniform BF16 | 理想重叠 | 4,181,978 | 4,181,978 | 4,181,978 | 4,181,978 | 1.0000x | 1.0000x | 1.0000x |
| Kimi K3 | mixed NVFP4 | 严格串行 | 103,816,704 | 97,013,856 | 93,124,740 | 91,173,903 | 1.1387x | 1.0641x | 1.0214x |
| Kimi K3 | mixed NVFP4 | 理想重叠 | 88,142,659 | 88,142,659 | 88,420,867 | 88,142,590 | 1.0000x | 1.0000x | 1.0032x |
| Kimi K3 | uniform BF16 | 严格串行 | 149,593,151 | 142,790,303 | 138,901,187 | 136,950,350 | 1.0923x | 1.0426x | 1.0142x |
| Kimi K3 | uniform BF16 | 理想重叠 | 133,919,106 | 133,919,106 | 134,197,314 | 133,919,037 | 1.0000x | 1.0000x | 1.0021x |

严格串行端点等于 `HBM + Matrix + Vector + L-Compute`，是当前依赖安全的
结果。理想重叠端点等于 `max(HBM, Matrix, Vector + L-Compute)`，假设三类
资源完全重叠，不考虑依赖、SRAM 容量和仲裁；它只是下界，不是 Compiler 已经
发出的 schedule。

五个变体的 Matrix 周期相同。上表 Nemotron 的 HBM 周期也相同，但这个结论
**不能推广到 Kimi**：Kimi C 有固定布局的中间 spill，C/D 还使用精确 state-DMA
计数。例如 mixed Kimi B1 的 C 为 88,420,867 HBM 周期，D 为 88,142,590，
所以理想端点的 D/C 是 1.0032x，而不是 1.0000x。这是数据对“所有变体的 HBM
周期完全相同”这一更强假设的直接反例。

A/B 的递推成本采用“每条动态指令 1 周期”的发行代理，并把 Matrix service 与
Vector arithmetic 成本置零；C/D/E 来自真实 lowering 的 service 模型。因此
D/A 和 D/B 的分子、分母属于两类不同证据。表中收益合并了多行执行、紧凑
descriptor、减少发射以及可能的 spill 变化；它们不是硅上加速，也不是可编程
skew 的加速。

### 5.1 Agentic 工作负载包络

下表对 93 个按长度排序且互不重叠的 workload group 取中位数。每组运行 32 个
decode step，并严格重放实测 Nemotron eager-routing 专家并集；`N` 是 group
数量。B4、B8、B16 的 P95 因 `N < 20` 只能作为探索性结果。

| B | N | D/A 理想 | D/C 理想 | D/B 理想 | D/C 串行 | D/B 串行 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 1.1361x | 1.0000x | 1.0000x | 1.0890x | 1.5456x |
| 2 | 24 | 1.8423x | 1.0000x | 1.0239x | 1.1385x | 1.8490x |
| 4 | 12 | 2.8151x | 1.0000x | 1.5642x | 1.2003x | 2.2282x |
| 8 | 6 | 4.0843x | 1.0000x | 2.2699x | 1.2717x | 2.6661x |
| 16 | 3 | 5.8904x | 1.0000x | 3.2743x | 1.3579x | 3.1943x |

| B | N | mixed NVFP4 D/B 串行 / 理想 | MXFP8 D/B 串行 / 理想 | BF16 D/B 串行 / 理想 |
|---:|---:|---:|---:|---:|
| 1 | 48 | 1.546 / 1.000 | 1.485 / 1.000 | 1.254 / 1.000 |
| 2 | 24 | 1.849 / 1.024 | 1.697 / 1.000 | 1.371 / 1.000 |
| 4 | 12 | 2.228 / 1.564 | 1.945 / 1.154 | 1.515 / 1.000 |
| 8 | 6 | 2.666 / 2.270 | 2.236 / 1.576 | 1.692 / 1.000 |
| 16 | 3 | 3.194 / 3.274 | 2.628 / 2.211 | 1.950 / 1.165 |

Agentic 中 D/C 理想端点恒为 1.0000x 是 Nemotron 结论：routing 会改变 MoE
时间线，但不改变递推 state packet 的 bank 坐标。它不能推广到有固定布局 spill
的 Kimi C。

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
- RTL、PPA、Token/J 或相对 B200/5090 的硬件加速；
- 能达到理想资源重叠下界的可执行 schedule；Agentic 包络中所有 batch 的
  D/C 理想端点均为 1.0000x，B1 的 D/B 理想端点为 1.000x，因此不能把严格
  串行下的发行节省写成在完全重叠机器上的必然收益；
- 超出当前流量公式的权重与反量化行为：mixed NVFP4 按约 0.5625 byte/元素
  计数（每 16 个值一个 FP8 block scale），不含反量化计算、tensor-global
  scale 和物理 padding；Matrix SRAM 元素仍为 BF16。Agentic B16 中，权重从
  mixed NVFP4 改成 uniform BF16 后，D/B 从 3.194/3.274（串行/理想）降为
  1.950/1.165。

这些不影响 ISA/Compiler/Simulator 的 pre-RTL 冻结，但在论文中必须明确区分。

## 8. 交接门禁

```bash
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute /absolute/path/to/PLENA_Compiler
```

只有该命令退出码为 0，且官方 schedule 不含 `L_CFG`、所有数值对拍和 D′ 公平
检查通过，才允许进入 RTL 阶段。交接材料还必须附带
`git diff main..HEAD -- doc/operation.svh` 的完整输出。

本次冻结的 Simulator gitlink 使用 Compiler tip
`330e93da425eee107a0f3299f5f039fad1d74cd4`；其中机制实现 commit 是
`c2e7d03e14b4c43350fd3d232cb2ee6058a494c4`。门禁退出码为 0：

- Simulator Python：108 passed；
- Compiler：188 passed；
- Rust workspace：298 passed（13 个 test binary 合计；其中
  `transactional_emulator` 单体 180）；
- Matrix projection：65,664 个 BF16 值逐项一致，0 bank stall；
- official-geometry recurrence：Nemotron/Kimi fixed/phased 四组全部通过。

关键机器可读证据的 SHA256：

```text
01a8965c58c9203c05272edab50459b64fe66fb5f4340166d57218c6d5b180c6  artifacts/matrix_lcompute_connected_bf16/summary.json
cfd26f07ce7c81b36f11532c31bd6435f8e8d24a138029fba7ab467bd60dd6c1  artifacts/matrix_lcompute_e2e_v5/campaign.json
2ee3fa0d15f65e276f71b6763c5b55de078efef816be81ddc7f143242f135aed  artifacts/matrix_lcompute_e2e_v5/headline.csv
3f0f015c2dc420b3ee13c827a61b086ec78904a5005efa72e6a303864af7534b  artifacts/matrix_lcompute_agentic_v1/campaign.json
11c549ad31da440fe8973af98eca5e2234b4d99bdb4a061cd27a019e5bab41c5  artifacts/matrix_lcompute_agentic_v1/summary.csv
```
