# Matrix L-Compute：精度契约与 Rust 执行增量

本轮以 Nemotron NVFP4 checkpoint 的混合权重策略作为主实验；BF16 递推 state 和 prepared coefficients 独立冻结。MX8 仅作为权重流量敏感性实验，不复现原论文精度点。

## 精度和产物

`PrecisionContract` 分开配置 W、A、KV、state；每项记录元素格式、block 元素数、scale 格式/位宽、尾部 block padding 和字节对齐。全局 `storage_bytes(..., block_size=128)` 及 `plena_settings.toml` 默认值保持原样。

| 用途 | 元素 | block | scale | 本轮口径 |
|---|---|---:|---|---|
| Nemotron 主实验大部分线性权重 | NVFP4 E2M1 | 16 | E4M3 / 8 bit | 保留 checkpoint 的 BF16 exclusions |
| MX8 权重敏感性 | E4M3 | **8** | E8M0 / 8 bit | 128 元素为 144 B，旧 block128 为 129 B |
| A、KV、recurrent state、prepared coefficients | BF16 E8M7 | 无 | 无 | 2 B/元素 |

逻辑权重计数包含 block scales，不包含未知的 tensor-global FP32 scale 数量，也不假装包含物理 checkpoint padding。物理 HBM burst rounding 由独立硬件流量模型处理。NVFP4、MX8 敏感性改变权重流量，不声称执行了对应权重量化/反量化数值。

- 正式新表：`../artifacts/matrix_lcompute_agentic_v2/`，93 个真实路由重放分组。
- 新的完整解析产物：`../artifacts/matrix_lcompute_e2e_v6/`；`headline.csv` 与 v5 字节一致。
- 旧 block128：`../artifacts/matrix_lcompute_agentic_v1/`，数据原样保留并加历史标签。
- 修正的 Nemotron B16 MX8 D/B：**2.5001389746× 串行，1.9996014476× 理想重叠**。
- NVFP4 B16 解析 D/B 仍为 **3.1918283433× / 3.2757861907×**。这些旧 A/B 时间线仍是发行代理，不能当作新 Rust 执行计时。

## 修复了普通 DMA 的 state 精度错误

Compiler 一直用 selector 2 表示 BF16 state；旧 Rust 普通 `H_PREFETCH_V/H_STORE_V` 解码把所有非零 selector 都当 KV。KV 仍为 MX8 时，真实普通 Vector 递推会读错格式并写错范围。新增地址保护在首个普通基线运行中发现了这个问题。

现在普通和 viewed DMA 都把 selector 2 解码为 State；Activation=0、KV=1 不变。原始二进制如果故意使用 2 作为 KV 别名，需要重新汇编成 1。未增加 opcode，也没有修改全局 activation/KV/state 默认精度。旧官方 L_TILE 数值测试使用 viewed DMA，原本已正确解码 State，因此不受这个潜伏 bug 影响。

## 数值回归

`matrix_lcompute_numeric_v2` 对 Nemotron/Kimi、phased/fixed 各测试 3 个 seed、连续 32 token。`matrix_lcompute_numeric_long_v2` 另测 Nemotron 连续 128 token。

每个 token 的更新后 state 通过 Compiler 发出的额外 HBM snapshot store 读回并比较，不仅检查最后一个 state。phased 的这些确定性用例要求精确位比较；fixed 保留 relative-L2 ≤ 1%，逐元素 atol/rtol 均为 0.01，近零 RMS floor 为 1e-7。结果记录每个 token 的 output/state 哈希及误差。额外 snapshot DMA 只用于数值诊断，不进入性能表。

这些用例覆盖更多输入和时间长度，仍不是对所有输入的误差证明。D 的 state 存储为 BF16，局部归约使用 FP32 累加后写回 BF16；不能将“BF16 state”写成“所有中间算术都是 BF16”。

实际完成 14 组扩展用例，结果如下（max 为该模型/layout 所有 seed 和 token 的最大值）：

| 模型 | Layout | 用例数 | 最长 token | 最大 output rel-L2 | 最大中间 state rel-L2 |
|---|---|---:|---:|---:|---:|
| nemotron3_mamba2 | affine | 4 | 128 | 0.00000000 | 0.00000000 |
| nemotron3_mamba2 | fixed | 4 | 128 | 0.00569348 | 0.00000000 |
| kimi_k3_kda | affine | 3 | 32 | 0.00000000 | 0.00000000 |
| kimi_k3_kda | fixed | 3 | 32 | 0.00731770 | 0.00142622 |

Kimi fixed 的最差输出 rel-L2 为 0.00731770，距 1% 门限约剩 1.37 倍余量。门限未放宽。

## A/B/D 真实执行对照

`matrix_lcompute_execution_v1` 对 Nemotron 的 B1/B2/B4/B8/B16 各执行 2 token，15 次实际 Rust 运行。每个 batch 在一次 Rust 调用中按 token、request 顺序执行；request 拥有私有持久 HBM state，主机不在 token 间更新 state。输入/state/output/机器码哈希、写保护范围和逐请求参考结果均在产物中。

这里的 A/B 是新实现的 packed ordinary-VV 控制程序：A 显式装入地址，B 静态复用已知地址。**它们不是旧解析表中原版/Arlo 指令流的逐指令重放，不能直接替换旧表中的 A/B 标签。** 两者只执行普通 Vector 指令，D 执行 L_TILE。三者使用相同逻辑 BF16 输入和相同 SRAM 拓扑。

A/B 显式将系数扩展到 HBM lanes，D 使用 compact fields，所以 DMA 差异属于这个完整数据流对照，不能单独归因为 bank 改进。A/B 每条 VV 后舍入 BF16，D 使用局部 FP32 归约；各自必须精确匹配自己的操作序列参考，另外必须通过共同递推参考的 1% 误差预算，才允许发布加速比。

| Batch | D / 新 B 执行控制的串行加速 |
|---:|---:|
| 1 | 1.110520× |
| 2 | 1.117167× |
| 4 | 1.126951× |
| 8 | 1.133460× |
| 16 | 1.149947× |

全部 Nemotron batch 通过共同数值预算和私有状态检查。最大 output relative-L2 为 0.00821693，D 精确为零。

Kimi 的 B1 诊断保存在 `matrix_lcompute_execution_kda_diagnostic_v1`。普通 VV A/B 精确匹配各自舍入参考，但对共同参考的 output relative-L2 达 **0.01348517 > 0.01**；`qualified_comparison.csv` 因此将加速比留空。这是尚未解决的数值契约差异，不能用放宽门限或称作纯硬件加速掩盖。

## 计时契约和边界

`PLENA_UNIFIED_SERIAL_TIMING=1` 启用受限的递推计时契约：每条指令 1 个发行周期；普通全 VLEN VV 操作在数据就绪后计两次单端口 bank read 和一次 write；普通 DMA 的 bank 写入周期在传输完成后计费，不能与 DMA 等待重叠；Matrix view 按实际 packet bank service 计费；两者都从实际算术执行代码计费。还记录 scalar/control 和 DMA/memory wait，所有分量精确相加为总虚拟时间。不支持的 opcode 会被拒绝，避免误用于完整模型并漏计。

DMA/memory wait 是串行 dispatch 等待时间，不是独立 HBM 引擎 busy 时间。当前 Rust 使用 `Ramulator::hbm2_preset(8)`、假设 1 GHz，与整模解析表的 1560 B/cycle 流量模型不是同一套时间模型。新表也没有理想重叠加速列。详细分量和实际 HBM 字节见 `timing_components.csv`。

新的结果验证递推核心，不含真实 checkpoint 的 projection、Attention/MLA、MoE 和 residual，不是整模型 Rust 执行，也没有 RTL/PPA。历史 52/93 层时间线的成本校准仍需另外完成。下一步应先解决 Kimi 普通 Vector 与 L_TILE 的舍入契约，再明确论文使用新执行控制还是历史 Arlo 基线；不能直接把新递推比值乘到旧整模表上。

## 复现

在 Simulator 工作树内运行；Compiler submodule 固定到本轮提交：

```bash
nix develop --no-write-lock-file --command just test-matrix-lcompute PLENA_Compiler
```

生成实际执行对照（含 B1–B16，每个 batch 两个 token）：

```bash
nix develop --no-write-lock-file --command bash -c \
  'PLENA_COMPILER_ROOT=PLENA_Compiler python3 -m transactional_emulator.testbench.aten.matrix_lcompute_execution_compare --output-dir artifacts/matrix_lcompute_execution_v1'
```

生成三个 seed 的 32-token 数值回归，以及独立的 128-token 长链：

```bash
nix develop --no-write-lock-file --command just matrix-lcompute-numeric-sweep PLENA_Compiler
nix develop --no-write-lock-file --command bash -c \
  'PLENA_COMPILER_ROOT=PLENA_Compiler python3 -m transactional_emulator.testbench.aten.matrix_lcompute_numeric_sweep --model mamba --tokens 128 --seeds 20260903 --output-dir artifacts/matrix_lcompute_numeric_long_v2'
```

生成正式解析表：

```bash
nix develop --no-write-lock-file --command just matrix-lcompute-campaign PLENA_Compiler
nix develop --no-write-lock-file --command just matrix-lcompute-agentic \
  /scratch/shared/mcl123/plena/artifacts/gpu/AGENTIC_NEMOTRON_B200_20260903 PLENA_Compiler
```

## 验证状态

Python **156 passed**，Compiler **216 passed**，Rust workspace **304 passed**；投影、四组原始联通用例和新的 B1 执行门禁通过。扩展数值为 14 组，Nemotron 执行对照为 15 组，Kimi B1 诊断为 3 组。共同误差预算失败的 Kimi 普通 VV 行保留为明确的不合格诊断，不计作加速结论。
