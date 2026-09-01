# Matrix SRAM L-Compute：可复现实验结果

## 结论

真实 Compiler lowering 对应的是 **Outcome 1**：Kimi K3 需要每个 tensor
可配置的 `alpha` 才能达到 bank floor；Nemotron 3 的最佳固定映射已经等于
可配置映射。旧的 `32x/16x` 和整模 `1.174x/1.032x` 结论来自错误的
tile-local row 公式，已经撤回。

现在能够成立的结论是：

- Kimi 官方尺寸的 Matrix-SRAM 局部服务，`D` 相对最强固定映射 `D'` 为
  `2.0x`，stall 从 3072 降到 0；Nemotron 为 `1.0x`。
- 官方 FP32 state 下，B1 整模 decode 的纯 layout 收益为 Nemotron
  `1.00000x`、Kimi `1.00216x`。HBM、MoE 和其他层掩盖了局部收益。
- KDA prefill 的 identity-GEMM 转置是更大的独立机会：BF16/MX8 候选下，
  Kimi S16/S128 串行整模时间线为 `3.387x/1.713x`。
- 没有增加 cache、私有 state SRAM、MAC、SRAM 端口或运行时调度器。

## 设计是什么

PLENA 原有 Matrix SRAM 已支持固定斜存，使一行或一列可以并行读取。那是
原论文 Figure 9 的已有能力，不是本工作的 novelty。本工作把固定映射扩展为
Compiler 按 tensor 选择的仿射映射：

```text
bank = (alpha * physical_row
        + gamma * floor(physical_row / banks)
        + bank_word) mod banks
```

`physical_row` 包括 allocation base、tile pitch、逻辑 row 和宽 row 的 word
group。之前按 tile-local row 计算会丢掉这些信息，并制造一个不存在的
per-tile phase 收益。

当前测量中：

- `gamma` 是机器常数；
- `alpha` 由 Compiler 根据 tensor 的逻辑 row 宽度选择；
- 没有真实 lowering 需要 `beta` 或 per-tile phase；
- 读取后只做 cyclic lane restore，算术看到的顺序不变；
- Matrix accumulator 使用同一 view 直接斜着写回，动态结果不会退回 row-major。

## ISA

```text
L_MVIEW.FULL   slot, shape_reg, map_reg
L_MVIEW.FIELD  slot, field, value_reg
<consumer>     ..., view=slot
```

`FULL/FIELD` 共用一个 opcode，以 `funct` 区分。`FULL` 是正常热路径，
`FIELD` 只用于冷门的部分修改。consumer 在自己的编码中显式点名 slot，
因此 Compiler 可以静态检查“先配置、后使用”；没有隐式 `SELECT`。

公开语义只包含 rows、columns、tile pitch 和 `alpha`。它不包含 Mamba、KDA、
head 数、bank 数或递推公式。行/列方向仍由已有 `M_MM/M_TMM` 表达。Decoder
固定展开为：

```text
AFFINE_ADDRESS -> BANK_READ -> LANE_RESTORE
               -> EXISTING_OPERATION -> BANK_WRITE
```

这是一种 Matrix operand addressing mode，不是模型专用 fused instruction。

## 公平对比

| 版本 | 唯一变化 | Credit |
|---|---|---|
| A | 原始固定布局和原始 lowering | baseline |
| B | A + Arlo 的循环、地址和指令压缩 | Compiler |
| C | 多行 packet，原始固定映射 | 并行访问基线 |
| D' | 多行 packet，4096 个全局固定 `(alpha,gamma)` 中最优者 | 最强固定控制组 |
| D | 与 D' 相同计算和 issue stream，只允许每个 view 配置 `alpha` | 纯 L-Compute |
| E | D + Compiler 静态安排 Matrix/consumer overlap | overlap |

论文中只能把 `B/A` 记给 Arlo Compiler，把 `D/D'` 记给 L-Compute，把
`E/D` 记给 overlap。

## 物理 Bank 结果

Rust Matrix SRAM 真实保存 `banks x rows x bank_width` 数据。测试把每个 source
index 作为不同数值写入，读回后恢复 lane，并检查丢值、重值、错 lane、别名和
未写先读。下面不是公式预测，而是对真实 Compiler 动态地址的逐值 replay。

配置：`MLEN=2048`、`BLEN=32`、64 banks、每 bank word 32 个 BF16。

| 官方尺寸 decode traffic | C service / stall | D' service / stall | D service / stall | 检查值数 |
|---|---:|---:|---:|---:|
| Nemotron Mamba | 1536 / 768 | 768 / 0 | 768 / 0 | 1,572,864 |
| Kimi KDA | 12,288 / 9,216 | 6,144 / 3,072 | 3,072 / 0 | 6,291,456 |

因此 `C/D` 的局部服务是 Nemotron `2x`、Kimi `4x`；但严格的 novelty 对照
必须使用 `D/D'`，即 Nemotron `1x`、Kimi `2x`。

同一物理 cells 的 row/column 读取也已验证：固定对角布局下两者都达到
`ceil(values/banks)`；row-major 的 column read 需要多拍。普通 GQA、MLA、
MoE gate 的 row/column service 在 C/D'/D 中逐值和周期完全一致，没有退化。

## 52/93 层 Decode 时间线

模型结构为：

- Nemotron 3 Nano：52 层 = 23 Mamba + 23 MoE + 6 GQA；
- Kimi K3：93 层 = 69 KDA + 24 MLA，另有 92 层 latent MoE + 1 dense FFN。

主点使用 1 GHz 周期换算假设、1560 B/cycle HBM、官方 FP32 recurrent state。
它是官方 shape、GPU 校准、真实 Nemotron routing 和 symbolic PLENA weights 的
解析时间线，不是真 checkpoint 从第一层到最后一层的 Rust 数值执行。

### B1、decode 1 token、官方 FP32 state

| 模型 | A | B | C | D' | D | E |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron | 4,087,452 | 3,142,428 | 3,160,138 | 3,142,474 | 3,142,474 | 3,132,745 |
| Kimi | 105,011,094 | 98,168,502 | 98,804,544 | 98,380,608 | 98,168,640 | 97,890,432 |

| 模型 | `A/B` Compiler | `D'/D` 纯 layout | `B/E` 合并 |
|---|---:|---:|---:|
| Nemotron | 1.30073x | 1.00000x | 1.00309x |
| Kimi | 1.06970x | 1.00216x | 1.00284x |

### Batch sweep：官方 FP32，`D'/D`

| Batch | Nemotron | Kimi |
|---:|---:|---:|
| 1 | 1.00000x | 1.00216x |
| 2 | 1.00000x | 1.00339x |
| 4 | 1.00000x | 1.00474x |
| 8 | 1.00000x | 1.00592x |
| 16 | 1.00000x | 1.00676x |

Kimi 的 packet stall 随 batch 放大，因此 layout 收益逐步增加；Nemotron 的
最佳固定映射已经无冲突，所以保持 `1x`。

### HBM sweep：官方 FP32

| HBM B/cycle | Nem `D'/D` | Nem `B/E` | Kimi `D'/D` | Kimi `B/E` |
|---:|---:|---:|---:|---:|
| 64 | 1.00000x | 1.00020x | 1.00010x | 1.00013x |
| 256 | 1.00000x | 1.00075x | 1.00039x | 1.00051x |
| 512 | 1.00000x | 1.00138x | 1.00076x | 1.00100x |
| 1024 | 1.00000x | 1.00234x | 1.00147x | 1.00193x |
| 1560 | 1.00000x | 1.00309x | 1.00216x | 1.00284x |
| 8192 | 1.00000x | 1.00610x | 1.00797x | 1.01057x |

这说明 bank conflict 可以被真正消除，但若整模主要在等 HBM/MoE，局部 SRAM
提升不会自动变成大的端到端提升。

## KDA Prefill：消除 Identity Transpose

旧 Compiler 的 `kda_prefill_state_to_decode_layout_v0` 真正发出每 head 4096 个
`M_TMM` 和 4096 个 `M_MM_WO`。在 96 heads、69 KDA layers 下：

- 数学工作量：13,891,534,848 MAC；
- 当前 MLEN padding 后的发射量：56,899,726,737,408 MAC；
- Matrix 周期：868,220,928；
- Matrix view：0 transpose MAC，保守按每层重新配置共 345 issue cycles。

16,384 个非对称数值证明 column view 与真实 transpose 相同；错误 row view
会得到 finite 但错误的结果，并在 Compiler 发射前被拒绝。

| BF16/MX8 候选 | 旧 identity-GEMM 总周期 | Matrix-view 总周期 | 加速 |
|---|---:|---:|---:|
| Kimi prefill S16 | 1,231,961,177 | 363,740,594 | 3.38692x |
| Kimi prefill S128 | 2,086,343,447 | 1,218,122,864 | 1.71275x |

这是独立 prefill 边界结果，不算进 `D/D'`。官方 Kimi state 是 FP32，而 Matrix
SRAM 是 BF16，所以官方 FP32 时间线不领取这项收益。prefill 的 per-head
`alpha=1` column view 与 decode 的 16-head compact `alpha=4` packet 是两种
不同 view，中间仍有显式 streamed handoff；没有声称同一 resident allocation
直接跨越两阶段。

## Precision 与容量

| 项目 | Nemotron | Kimi |
|---|---:|---:|
| FP32 recurrent state / layer | 2 MiB | 6 MiB |
| 全 recurrent layers / request | 46 MiB | 414 MiB |
| 全层 state read+write / token | 92 MiB | 828 MiB |
| Analytic Matrix SRAM | 1 MiB BF16 | 1 MiB BF16 |
| Transactional Matrix SRAM | 512 KiB BF16 | 512 KiB BF16 |

没有 cache 或隐式 residence。官方 FP32 state 按显式 tile traffic 计费。低精度
state 只是 DSE：

| 实验 | output relative L2 | state relative L2 |
|---|---:|---:|
| Nemotron BF16 chunk128, S32768 | 0.0312% | 0.1668% |
| Kimi BF16 per-token, S2048 | 1.706% | 1.781% |
| Kimi FP16 per-token, S2048 | 0.209% | 0.217% |
| Kimi MX8 per-token, S2048 | 26.28% | 27.57% |

在 checkpoint quality gate 前，BF16/FP16/MX8 都不能替代官方 FP32结果。
BF16 候选的 B1 `B/E` 上界是 Nemotron `1.566x`、Kimi `1.082x`。

## Layout DSE

每个点都写入不同编号值，再读回并恢复 lane；不是只套周期公式。

| Banks x values/bank | Mamba `C/D` local | KDA `C/D` local |
|---|---:|---:|
| 256 x 8 | 8x | 16x |
| 128 x 16 | 4x | 8x |
| 64 x 32 | 2x | 4x |
| 32 x 64 | 1x | 2x |

逻辑 row 宽度 32/64/128/256 对应的 `C/D` 为 1/2/4/8x。packet width
512/1024/2048 不改变该局部比例；它改变总服务次数而不是每个 row 的 bank-word
数量。

## 结构开销代理

这是 pre-RTL 结构计数，不是面积或 PPA：

| 项目 | 数量 |
|---|---:|
| 新 SRAM payload | 0 bytes |
| cache tag / replacement bits | 0 |
| 新 MAC lanes | 0 |
| 4 个 view records | 256 bits |
| bank-select adders | 64 x 6-bit |
| cyclic lane restore | 64 个 512-bit words，6 stages |
| 新每-bank读/写端口 | 0 / 0 |
| 新 operand SRAM | 0 bytes；复用已有 Vector operand buffer |
| Matrix-to-Vector payload | 32,768 bits/cycle |
| Vector-to-Matrix writeback payload | 32,768 bits/cycle |

没有 RTL 综合，因此不能给 LUT、门数、面积、频率、功耗、Token/J 或 PPA。

## GPU 数据的用途

GPU 只用于锁定真实 shape、state dtype、瓶颈和 baseline，不用于把 GPU kernel
时间直接换成 PLENA cycle。

- 完整 Nemotron NVFP4 / B200：context 2048、decode 128 token 的 ITL median
  `4.0476 ms`，约 `247.08 token/s`。
- Nemotron decode NCU：Mamba `3.8657 ms`、Attention `0.4231 ms`、MoE
  `6.5309 ms`；该 profile 中 MoE 是主要部分。
- KDA B1 / B200：单层 kernel 总时间 `0.35995 ms`；Matrix 路径占 `74.45%`，
  recurrent core 占 `5.02%`。B8 总时间 `0.41188 ms`，core 占 `11.65%`。
- 官方 KDA recurrent state 已确认是 FP32、6 MiB/layer。

这些数字是外部 baseline 和模型校准，不是当前 PLENA 相对 B200 的加速比。

## 当前证明到哪里

已证明：

1. 通用 `L_MVIEW` 编码、canonical 检查和 dominance；
2. 真实 banked Matrix SRAM、row/column read、skew writeback 和 lane restore；
3. 官方尺寸 Mamba/KDA packet 全值往返；
4. reduced-shape Mamba/KDA 连续多 token 数值递推；
5. Nemotron 52 层和 Kimi 93 层的完整 analytic prefill/decode 时间线；
6. 普通 Attention/MLA/MoE 不退化；
7. Nemotron B200 完整 checkpoint baseline 和 KDA B200 单层 calibration。

未证明：

1. 真实权重的 52/93 层 first-to-last Rust 数值执行；
2. RTL timing、面积、功耗、PPA 或 Token/J；
3. Kimi 的真实 batch routing trace；
4. 低精度 state 的完整 checkpoint quality。

## 验证状态

| Gate | 结果 | 说明 |
|---|---:|---|
| Compiler L-Compute gate | 113 passed | ISA、dominance、packet extraction、prefill handoff、Mamba/KDA fallback |
| Compiler 最后定向复验 | 39 passed | `alpha` 编码、prefill capacity 字段和报告重命名后的复验 |
| Simulator performance 全套 | 123 passed, 3 skipped | skip 仅为未挂载的可选原始 GPU 归档交叉检查；导入后的正式 GPU summary 已测试 |
| Matrix campaign / state / runner | 88 passed | 最终 artifact 重生成后复验 |
| Rust workspace release | 272 passed | 物理 bank、row/column、写回、lane restore、递推和既有 emulator 回归 |
| Compiler -> Rust connected | pass | 1,041 个 Matrix packet、65,664 个值、0 bank stall；输出 max error = 0 |
| Ruff / Rust fmt / `git diff --check` | clean | 两仓相关文件 |

直接运行 Compiler 全仓 `pytest` 仍在 collection 阶段报告 26 个环境错误：
`generator` 缺 `numpy/transformers`，`tilelang_tvm_compiler` 缺 `tvm`。
这些模块尚未开始执行，不是 L-Compute 测试失败；因此这里不声称 Compiler
全仓 green，也不把它们记成算法回归。

## 复现

```bash
cd /scratch/shared/mcl123/plena/review_20260828/simulator-static-kda-latest
nix develop --no-write-lock-file --command just test-matrix-lcompute

nix develop --no-write-lock-file --command \
  just matrix-lcompute-campaign
```

结果位于 `artifacts/matrix_lcompute_e2e_v1/`：

- `campaign.json`：完整证据和边界；
- `ablation.csv`：所有模型、batch、token、precision 和 variant；
- `headline.csv`：`D'` 与 `D` 主结果；
- `state_residency.json`：容量、流量和精度。
