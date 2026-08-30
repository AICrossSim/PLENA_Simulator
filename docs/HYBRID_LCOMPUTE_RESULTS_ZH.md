# Hybrid L-Compute：Compiler/Simulator 结果

## 1. 结论

这条分支完成了一套不依赖模型名字的静态 L-Compute 原型：

- 没有 `X_STATE`、Mamba/KDA 协处理器、私有 state cache、命令队列或运行时替换。
- `L_STREAM_CFG` 只配置规则地址流和仿射物理布局；数学仍由现有 Matrix/Vector 指令完成，循环仍由 `C_LOOP_START/END` 完成。
- Matrix 最后一轮累加结束时，可以直接按 consumer 需要的 affine-skewed 布局写入 banked Vector/output SRAM；Vector 读取时做逆向 cyclic rotation。
- 官方 52 层 Nemotron 3 和 93 层 Kimi K3 已进入同一套 Matrix/Vector/HBM/banked-output 共享资源时间线。

当前数据支持冻结“通用流寻址”，但不支持把 affine 多行读取冻结成最终硬件模式：物理斜存和数值往返已经实现，局部 packet 的 bank conflict 也被消除；然而当前可执行消费者仍然逐行读取，因此整模没有可消除的多行 bank stall。

## 2. 正确的架构边界

原始说法“在 Matrix SRAM 里斜存”不够准确。PLENA 的 Matrix SRAM 保存 Matrix 单元的输入/权重，projection 结果写到 Vector/output SRAM。因此本实现的边界是：

```text
existing Matrix SRAM -> existing Matrix compute
                              |
                         final writeback
                              |
                    affine placement mapper
                              |
                 banked Vector/output SRAM
                              |
                    inverse lane rotation
                              |
                  existing Vector arithmetic
```

这避免了 Matrix SRAM 同时供 systolic array 读权重、又被 projection 结果占用而产生新的端口竞争。状态仍是普通张量，由 Compiler 显式 preload/store；报告中没有 cache hit rate。

仿射映射为：

```text
bank = (stripe + alpha*major + beta*field + gamma*group) mod banks
bank_row = base + outer*pitch + floor(stripe/banks)
sublane = minor mod bank_width
```

`alpha/beta/gamma` 分别把 row、field 和 group 错开。布局描述符不包含 Mamba、KDA、head 数或递推公式。

## 3. ISA 为什么是通用的

唯一新增的 architectural opcode 是：

```text
L_STREAM_CFG value_reg, target_reg, slot, field
```

它不是 `MAMBA_STEP` 或 `KDA_STEP`，也不是把一整层融合为黑盒。它只做两件普通 ISA 难以紧凑表达的事：

1. 把编译期已知的地址推进和标量流绑定到现有 Matrix/Vector 操作数；
2. 携带 producer-consumer affine layout 元数据。

同一语义已经用于 Nemotron Mamba、Kimi KDA 和不属于任何模型的 SAXPY。短循环、逆向循环、不可证明为仿射或无收益的循环自动保留 Arlo 静态路径。

配置采用 fail-closed 语义：非法 field/flag/slot/register、零 extent、地址溢出、bank-row alias、重复 live target 和越界 packet 都会失败；失败更新是原子的，不会污染已有 slot。Compiler 总是最后写 `ENABLE`，重用前先 `RESET`。

## 4. 实际实现了什么

### Compiler

- 固定并校验 Nemotron 3：52 层，23 Mamba、23 MoE、6 GQA。
- 固定并校验 Kimi K3：93 层，69 KDA、24 MLA、1 dense FFN、92 latent MoE。
- 从规则循环提取 stream，降低地址推进和 scalar load。
- 枚举 row-major、transpose、consumer-major、affine-skewed，验证一一映射并计算生产者写、消费者读、lane restore 和冲突成本。
- 只在 GEMM 的最后一个 K tile 做 affine writeback，避免未完成的 FP32 partial sum 被提前重排。
- Kimi MLA cache 始终保持 512 latent + 64 RoPE = 576 个 BF16 元素/token，没有展开成 96-head K/V cache。

### Rust transactional emulator

- `L_STREAM_CFG` 解码、配置寄存器、严格异常语义和自动推进。
- 真实 `banks x bank_rows x sublanes` 存储，不是只算公式。
- Matrix affine final writeback 和 Vector inverse lane restore。
- 普通整行访问在 16 bank x 4 BF16 elements、2R1W 下仍为 1 cycle；二元 Vector 操作也保持 1 cycle。
- 连接测试执行 `K=128` 分块 GEMM，经过 affine 写入和 lane restore 后，256 个 BF16 值逐元素一致；结果为 6,790 cycles、263,168 HBM read bytes。

### 统一性能模型

- 独立的 prefill 和 decode lowering，不使用 `decode x 常数`。
- S16/S128 prefill 和连续 4/32-token decode。
- Matrix、Vector、HBM、banked output SRAM 各自只有一个共享服务资源，事件按依赖排队。
- 显式 MoE combine、HBM burst rounding、FIFO/backpressure、bank port、state/KV 生命周期和 routing 压力。

## 5. A-G 消融

| 版本 | 含义 |
|---|---|
| A | row-major + gather + 普通静态循环 |
| B | Arlo stride/post-increment；主要基线 |
| C | consumer-major direct write |
| D | affine layout，不做 stream addressing |
| E | stream addressing，不做 affine layout |
| F | affine layout + stream addressing |
| G | F + producer/consumer overlap |

基础评估点：`MLEN=VLEN=64`、`BLEN=4`、16 banks、每 bank 4 个 BF16 元素、2R1W、64-value FIFO、4 stream slots、HBM 512 B/cycle、1 GHz proxy；默认没有额外常驻 state tile。

### Decode 整模结果

| 模型/场景 | B cycles | E cycles | G cycles | E/G 相对 B |
|---|---:|---:|---:|---:|
| Nemotron，4 token | 80,275,692 | 78,556,396 | 78,556,396 | 1.021886x |
| Nemotron，32 token | 642,303,648 | 628,549,280 | 628,549,280 | 1.021883x |
| Kimi K3，4 token | 2,856,881,544 | 2,838,072,696 | 2,838,072,696 | 1.006627x |
| Kimi K3，32 token | 22,859,495,616 | 22,709,024,832 | 22,709,024,832 | 1.006626x |

在 1 GHz proxy 下，4-token 结果对应：

- Nemotron：20.068923M -> 19.639099M cycles/token。
- Kimi K3：714.220386M -> 709.518174M cycles/token。

这些是 Compiler/Simulator 周期，不是 RTL 频率、真实 TPOT 或相对 GPU 加速比。

### 各环节贡献

| 环节 | Nemotron | Kimi K3 | 解释 |
|---|---:|---:|---|
| A -> B，Arlo gather/地址优化 | 1.0494x 整模 | 1.0195x 整模 | 已保留为基线 |
| B -> E，通用 stream addressing | 1.0219x 整模 | 1.00663x 整模 | 当前真正可执行的新收益 |
| Mamba/KDA 对应层 B -> E | 1.07064x | 1.01864x | 只统计目标层 |
| E -> G，affine incremental | 1.0000x | 1.0000x | 当前 serial consumer 没有多行冲突 |

Prefill S16/S128 已完成整模共享时间线，但 E 相对 B 为 1.0x，因为当前 prefill lowering 没有错误地复用 decode 的 stream issue reduction。D/F 会支付 lane restore 而没有 packetized read 收益，因此略慢；G 的 overlap 只能隐藏这部分开销，回到 E，不能制造额外加速。

## 6. Bank conflict 到底解决了没有

答案分两层：

1. **物理机制层：解决了。** 映射是双射，数据能完整往返，故意 alias 的映射会失败；对候选多行 packet，affine 映射达到带宽下限，`conflict_stall=0`。
2. **当前整模执行层：没有可计入的收益。** 当前 Vector consumer 每条指令只读一行，B/E 本来就没有 multirow bank conflict，所以 G 不能比 E 更快。

| 局部 packet | Row-major total | Affine total | 局部上限 |
|---|---:|---:|---:|
| Nemotron projection | 34,976 | 4,256 | 8.218x |
| Kimi KDA projection | 8,162 | 2,786 | 2.930x |
| Nemotron state | 73,728 | 40,960 | 1.800x |
| Kimi KDA state | 221,184 | 221,184 | 1.000x |

这张表是 packet service 上限，不是单层或整模加速。要把它变成可执行收益，下一步必须加入一个仍由现有算术指令驱动的 packetized consumer lowering；在它通过数值对拍前，不冻结 affine ISA mode。

## 7. Compiler 指令发射结果

| Workload | 普通静态 | Arlo post-increment | Stream | 普通 / Stream |
|---|---:|---:|---:|---:|
| Nemotron Mamba recurrence | 92,399 | 51,311 | 32,623 | 2.832x |
| Kimi K3 KDA mixer | 428,238 | 226,242 | 158,094 | 2.709x |
| Generic SAXPY | 1,284 | 516 | 299 | 4.294x |

这解释了为什么 stream addressing 有资格成为通用 ISA，但不能把 issued-instruction reduction 直接写成周期加速。

## 8. GPU 数据如何使用

GPU 数据只用于固定 shape、dtype、真实瓶颈和 baseline，不直接替代 PLENA cycles。

- KDA 官方真实 shape：96 heads x 128，FP32 recurrent state 为 6 MiB/request/layer，三个 BF16 conv state 共 0.28125 MiB。
- B200 KDA decode B1：Matrix 路径占 74.45%，state core 占 5.02%；B8 state core 占 11.6%。
- Nemotron B200 完整模型 decode median ITL 为 4.047566 ms。
- Nemotron prefill MoE DRAM read 是 Mamba 的 8.919x；真实 routing 最热 expert 达平均负载 20.98x。
- RTX 5090 Mamba decode B1 state core 只占 2.847%。

这些数据解释了整模收益为何小于局部递推/布局收益：Nemotron 还受 MoE 和 Matrix 权重流量支配，Kimi K3 更受 92 层 latent MoE 与 MLA 支配。

## 9. Precision 和资源代理

Nemotron Mamba 的 S32768 实验：

| State | Bytes/layer | Output relative L2 | State relative L2 |
|---|---:|---:|---:|
| FP32 | 2,097,152 | 0 | 0 |
| BF16 block128 | 1,048,576 | 0.0003124 | 0.0016679 |
| FP16 block128 | 1,048,576 | 0.0001418 | 0.0002069 |
| MX8 block128 | 528,384 | 0.0008061 | 0.0268663 |

KDA state、activation MX8 和 weight precision 尚无同政策长序列数值实验，因此不能冻结。

当前结构代理，不是 PPA：

- 不增加 state SRAM payload；state 是普通 tensor。
- 16 个 affine address adders。
- 64 个 cyclic restore lanes。
- 64-value FIFO = 1,024 bits。
- 4 slots，配置位上限 1,920 bits；实测最宽 lowering 同时需要 3 slots。
- 2R1W 是普通二元 Vector row operation 不退化的最低点；1R 会变成 2 cycles，因此 DSE 判为不合格。

没有 RTL 综合，所以没有面积、功耗、频率、Token/J 或相对 A100/H100/H200/B200 的 PLENA 加速比。

## 10. “跑通”的准确等级

| 项目 | 状态 |
|---|---|
| Nemotron 52 层官方结构/真实尺寸性能时间线 | 完成 |
| Kimi K3 93 层官方结构/真实尺寸性能时间线 | 完成 |
| S16/S128 prefill、4/32-token decode | 完成 |
| Matrix -> affine banks -> Vector 数值连接测试 | 完成 |
| Mamba/KDA 静态数值核心与 synthetic connected tests | 完成 |
| 真实 Nemotron/Kimi 权重从第一层到 logits 的 Rust 执行 | 未完成 |
| packetized multirow consumer 的整模执行 | 未完成 |
| KDA mixed-precision 长序列数值验证 | 未完成 |
| RTL/PPA/Token-J/相对 GPU 加速比 | 不在本阶段 |

因此正确表述是：**官方真实结构和真实尺寸的完整性能执行已完成；完整 checkpoint 数值执行未完成。**

## 11. 复现

从 Simulator 根目录：

```bash
git submodule update --init --recursive

uv run pytest -q \
  analytic_models/performance/test_lcompute_layout.py \
  analytic_models/performance/test_hybrid_lcompute_campaign.py

PYTHONPATH="$PWD:$PWD/PLENA_Compiler:$PWD/PLENA_Tools:$PWD/transactional_emulator/testbench" \
  uv run python transactional_emulator/testbench/aten/affine_projection_test.py

uv run python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --json-out /tmp/hybrid-lcompute.json \
  --csv-dir /tmp/hybrid-lcompute-csv \
  --long

nix develop --no-write-lock-file --command bash -lc \
  'cd transactional_emulator && cargo test --workspace'
```

本轮正式报告：

```text
report_sha256 = 0d4c4fd117d505a90fc713e463ed7711e057d9a201c1a719ece44d34d077f00c
JSON file SHA256 = 17264268a4fac19920df6885e013c67b757bd3023ad5d0aae9c5b1c1b354af9b
```

报告中的哈希不包含输出路径和生成时间；相同代码、配置和 evidence manifest 应得到相同 `report_sha256`。
