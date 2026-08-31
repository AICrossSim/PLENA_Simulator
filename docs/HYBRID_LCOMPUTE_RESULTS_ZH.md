# Hybrid L-Compute：Compiler/Simulator 结果

## 1. 先说结论

这轮完成了可执行的多行 L-Compute 数据通路，而不再只是独立布局测试：

- Compiler 用通用 `L_CFG` 把 Nemotron Mamba-2 和 Kimi K3 KDA 的状态衰减、秩一更新降低为多行 packet。
- Rust 让现有 `V_MUL_VF`、`V_FMA_VF` 从多个 SRAM bank 同时取数，按逻辑顺序恢复 lane，再执行原有算术。
- row-major 与 affine 使用完全相同的数值运算。row-major packet 有冲突，`alpha=1` 的斜存 packet 达到 bank 带宽下限，冲突周期为 0。
- 普通 Attention/MoE 的整行 Vector 路径不进入 packet 逻辑，数值、周期和计数器均不变。
- 官方真实尺寸的 Nemotron 52 层和 Kimi 93 层已完成 A-J 共享资源时间线比较。

但实验也否定了一个过强结论：**斜存解决了 packet 自身的 bank conflict，却没有超过 Arlo 的最佳普通整行 stream 路径。** 原因是两条路径每拍都只计算 64 个元素；当前 packet 改变取数拓扑，没有增加算力，且多了少量配置指令。

所以准确结论是：

> `L_CFG` 的规则流寻址有整模收益；affine multi-row packet 已被完整实现并证明能消除冲突，但在当前 64-lane 数据通路上尚未挣到相对最佳整行路径的性能优势。

## 2. 架构边界

PLENA 的 projection 结果来自 Matrix 单元，但最终写入 Vector/output SRAM。这里“Matrix 斜存”准确指：**Matrix 结果在 final writeback 时直接斜着落入 banked Vector/output SRAM**，而不是把 Matrix SRAM 的权重区拿来保存状态。

```text
Matrix SRAM -> existing Matrix compute -> final writeback stream
                                          |
                                  affine placement
                                          |
                         16-bank Vector/output SRAM
                                          |
                        multi-row read + inverse rotation
                                          |
                           existing Vector arithmetic
```

状态仍是普通 tensor，由 Compiler 显式分配、搬入和写回。没有 cache、tag、命中、替换、私有 state SRAM、命令队列或 `X_STATE`。

物理映射为：

```text
bank     = (stripe + alpha*major + beta*field + gamma*group) mod banks
bank_row = physical_base + outer*pitch + floor(stripe/banks)
sublane  = minor mod bank_width
```

`alpha/beta/gamma` 只描述 row、field、group 的物理旋转，不包含 Mamba、KDA 或 head 数。

## 3. ISA 与 packet 语义

唯一新增 opcode：

```text
L_CFG value_reg, target_reg, slot, field
```

它配置 base、extent、advance、packet stride、storage atom 和 affine 系数。算术仍由已有指令完成：

```text
L_CFG ...              # configure 16 logical rows x 4 elements
C_LOOP_START ...
V_MUL_VF ..., lmask    # existing opcode: explicit stream slots
V_FMA_VF ..., lmask    # existing opcode: explicit stream slots
C_LOOP_END ...
```

`L_CFG` 本身没有隐藏效果。消费者指令的 `lmask` 明确选择本拍使用的 slot；未选择
时机器码和原 PLENA 完全一致，普通 GP register 也不会被自动修改。
`funct1[2:0]` 只选择 consumer slot 0--2；slot 3 固定服务 Matrix final writeback。
`V_FMA_VF` 复用 `V_MUL_VF` 的 `funct1[3]` accumulate 模式，因此并未再占一个 opcode。

`packet_elements=64`、`storage_atom=4` 时，一条 Vector 指令读取 16 个逻辑 row，每个 row 取 4 个连续元素。读出后按 segment 顺序恢复成 64 lanes。

Compiler 只给实际斜存的移动 state operand 配置 `alpha=1`。固定 source row 和 FPRAM 标量流保持 `alpha=0`；本轮专门修复并测试了这一点，否则 recurrence 会静默读错值。

field 15 是 `PACKET_STRIDE`。Compiler 与 Rust 共同固定 golden machine word：

```text
L_CFG gp1, gp2, 3, 15 -> 0x003CC87F
```

## 4. 实现和验证证据

### Compiler

- Nemotron：52 层，23 Mamba、23 MoE、6 GQA，projection width 10,304。
- Kimi：93 层，69 KDA、24 MLA、92 latent MoE、1 dense FFN；KDA 为 96 heads x 128。
- Mamba 每层实际 packet 运算：8,192 次 `V_MUL_VF` + 8,192 次 `V_FMA_VF`。
- KDA 每层实际 packet 运算：24,576 次 `V_MUL_VF` + 24,576 次 `V_FMA_VF`。
- 真实尺寸 packet assembly 均可汇编成合法 32-bit machine words；没有 `MAMBA_STEP`、`KDA_STEP` 或 `X_STATE`。
- prediction/readout 是跨 row reduction，继续走普通整行路径，避免把重复 destination 错当成独立 lanes。

### Rust transactional emulator

- 真实模拟 `banks x bank_rows x sublanes`，不是只计算冲突公式。
- 两读一写端口模型：每 bank 2R1W。
- 多 bank packet 读、重复物理 word 去重、lane 恢复、分段标量广播和 packet 写回全部进入现有 Vector dispatch。
- 非法 slot/field/flag、越界 packet、重复目标、物理 alias 都 fail closed。
- runtime counters：packet 数、bank words、service/floor/stall cycles、lane restore values。

### 数值连接测试

Rust 通过同一条 `L_CFG -> 显式 lmask 的 V_MUL_VF/V_FMA_VF` 路径执行两种递推：

- Mamba：一个 head 的 decay 在多个 state row 共享，B/update scalar 按 row 变化。
- KDA：decay 与 k/update scalar 都按 key row 变化。

两者都使用非零、非恒等随机式数据；row-major 与 affine 输出逐元素相同，并与逐步 BF16 CPU 公式完全一致。

Matrix 侧另有 `K=128` connected test：分块 GEMM final writeback 经过 affine placement 和 lane restore，256 个 BF16 值逐元素一致。

## 5. A-J 公平消融

| 版本 | 含义 |
|---|---|
| A | row-major + 显式 gather + 普通静态循环 |
| B | Arlo stride/post-increment，论文主软件基线 |
| C | consumer-major direct write |
| D | affine placement，不启用 stream |
| E | 普通整行 stream addressing，当前最佳执行基线 |
| F | E + projection affine placement |
| G | F + Matrix/writeback overlap |
| H | 实际递推改用 row-major multi-row packet |
| I | 与 H 相同的递推，只把 state 改成 affine packet |
| J | I + producer/writeback overlap |

基础点：`MLEN=VLEN=64`、`BLEN=4`、16 banks x 4 BF16、2R1W、64-value FIFO、
3 个 consumer slots + 1 个 Matrix producer slot、HBM 512 B/cycle、1 GHz 时间换算假设。

H→I 只回答“斜存是否解决 packet 冲突”；E→I 回答“packet 是否比最佳普通整行路径更快”。这两个问题不能混为一谈。

## 6. 完整 52/93 层结果

### 连续 4-token decode

| 模型 | B cycles | E cycles | H row packet | I affine packet | J overlap |
|---|---:|---:|---:|---:|---:|
| Nemotron 3 | 80,275,692 | 78,614,724 | 111,929,028 | 78,979,780 | 78,791,364 |
| Kimi K3 | 2,856,881,544 | 2,838,814,584 | 3,138,431,352 | 2,840,404,344 | 2,840,192,376 |

| 对比 | Nemotron | Kimi K3 | 含义 |
|---|---:|---:|---|
| B→E | 1.02113x | 1.00636x | 通用 stream addressing 的真实整模收益 |
| H→I | 1.41719x | 1.10492x | affine 消除 row-packet conflict |
| E→I | 0.99538x | 0.99944x | packet 仍略慢于最佳普通整行路径 |
| B→J | 1.01884x | 1.00588x | 整套 packet+overlap 相对 Arlo 基线 |

### 连续 32-token decode

| 模型 | E cycles | H row packet | I affine packet | H→I | E→I |
|---|---:|---:|---:|---:|---:|
| Nemotron 3 | 629,015,904 | 895,530,336 | 631,936,352 | 1.417x | 0.9954x |
| Kimi K3 | 22,714,959,936 | 25,111,894,080 | 22,727,678,016 | 1.105x | 0.9994x |

完整时间线实际消费的 packet 数：

- Nemotron 4 token：1,507,328；32 token：12,058,624。
- Kimi 4 token：13,565,952；32 token：108,527,616。

这些不是独立 microbenchmark 数，而是 23/69 个真实 recurrence layer 在所有 decode token 上的实际计数。

### Prefill

S16/S128 的完整 52/93 层时间线已运行。当前 chunked Mamba/KDA prefill 没有复用 decode packet lowering，因此 packet 数为 0；J 的 writeback overlap 回到 E 的周期，不能声称 prefill packet 加速。

## 7. Bank conflict 证据

一个 16-row state step 包含 16 次 decay packet 和 16 次 update packet：

| 布局 | Read packets | Write packets | Service | Floor | Conflict stall |
|---|---:|---:|---:|---:|---:|
| Row-major | 48 | 32 | 784 | 80 | 704 |
| Affine `alpha=1` | 48 | 32 | 80 | 80 | 0 |

原因是 row-major 的 16 个 row words 同时落入一个 bank：读需要 `ceil(16/2)=8` 拍，写需要 16 拍。斜存后每个 bank 正好一个 word，读写都达到一拍带宽下限。

整模 4-token 中：

- Nemotron：33,161,216 packet conflict cycles -> 0。
- Kimi：298,450,944 packet conflict cycles -> 0。

因此“是否解决 bank conflict”的答案是明确的 **是**；“是否比普通整行执行更快”的答案目前是 **否**。

## 8. Attention/MoE 不退化

普通整行路径不绑定 packet view，Rust packet counters 全部为 0。测试分别在 1-bank、4-bank 和 16-bank backing 上执行普通 scalar FMA 与二元 Vector add：

- 输出逐元素一致。
- 执行周期完全一致。
- packet read/write/service/stall/lane counters 全部为 0。

结构约束 `banks x bank_width = VLEN` 保证一个普通 64-element row 仍可一拍读完；2R1W 保证二元 Vector op 不因 banking 退化。DSE 会拒绝 1R 配置。

## 9. GPU evidence 的用途

GPU 数据只固定真实 shape、dtype、流量和瓶颈，不直接替代 PLENA cycles：

- KDA：96 heads x 128，FP32 recurrent state 6 MiB/request/layer。
- B200 KDA decode B1：Matrix path 74.45%，state core 5.02%；B8 state core 11.6%。
- Nemotron B200 完整模型 decode median ITL 4.047566 ms。
- Nemotron prefill 的 MoE DRAM read 是 Mamba 的 8.919x；最热 expert 达平均负载 20.98x。
- RTX 5090 Mamba decode B1 state core 占 2.847%。

这解释了为什么局部 bank stall 很大但整模收益有限：MoE、Matrix weight streaming 和普通整行 recurrence 仍是更大的组成部分。

## 10. 资源与精度边界

当前只有结构代理，没有 RTL PPA：

- 不新增 state SRAM payload或 cache。
- 16-bank 地址映射、64-lane cyclic restore、64-value FIFO、3 个 consumer slots
  加 1 个 Matrix producer slot。
- 普通路径最低需要 2R1W 才不退化。

Nemotron S32768 state 误差已测：FP32 为基线；BF16 block128 output relative L2 为 0.0003124；FP16 为 0.0001418；MX8 为 0.0008061。KDA state/activation/weight 的同政策长序列精度尚未完成，不能冻结。

没有 RTL 综合，因此不能报告面积、功耗、频率、Token/J 或相对 A100/H100/H200/B200 的 PLENA 硬件加速比。

## 11. “跑通”的准确等级

| 项目 | 状态 |
|---|---|
| Mamba/KDA 实际 decay + rank-update packet 编译、解码、Rust 数值执行 | 完成 |
| Row-major/affine 物理 bank service 与 lane restore | 完成 |
| Nemotron 52 层、Kimi 93 层真实尺寸 decode 时间线 A/B | 完成 |
| S16/S128 prefill、4/32-token decode 共享资源模型 | 完成 |
| 普通 Attention/MoE row path 不退化 | 完成 |
| 真实 Nemotron/Kimi checkpoint 从第一层数值执行到 logits | 未完成 |
| transactional chunked prefill packet | 未完成 |
| RTL/PPA/Token-J/相对 GPU 加速比 | 不在本阶段 |

“完整 52/93 层”在这里指官方结构、真实尺寸、真实精度政策的性能时间线；权重是 symbolic manifest，不等于真实 checkpoint 数值整模执行。

## 12. 复现

```bash
git submodule update --init --recursive

PYTHONPATH="$PWD:$PWD/PLENA_Compiler" \
  .venv/bin/python -m pytest -q \
  analytic_models/performance/test_hybrid_lcompute_campaign.py

PYTHONPATH="$PWD:$PWD/PLENA_Compiler" \
  .venv/bin/python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --json-out /tmp/hybrid-lcompute.json \
  --csv-dir /tmp/hybrid-lcompute-csv \
  --long

nix develop --no-write-lock-file --command bash -lc \
  'cd transactional_emulator && cargo test --workspace'
```

本轮 report hash：

```text
report_sha256 = ee61d07b5c503a93711b1cdb6cd921a232e9ac5f474653ed4cb350945bb1ceb8
```

已检入的完整结果：

- [campaign.json](../artifacts/hybrid_lcompute_packet_v2/campaign.json)
- [A-J ablation CSV](../artifacts/hybrid_lcompute_packet_v2/tables/ablation.csv)
- [DSE CSV](../artifacts/hybrid_lcompute_packet_v2/tables/dse.csv)
- [precision CSV](../artifacts/hybrid_lcompute_packet_v2/tables/precision.csv)
- [schedule validation CSV](../artifacts/hybrid_lcompute_packet_v2/tables/schedule_validation.csv)
