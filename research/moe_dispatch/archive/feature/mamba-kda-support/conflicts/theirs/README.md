# PLENA Simulator：Mamba / KDA 支持

这个分支用来回答两个问题：

1. PLENA 能不能执行 Nemotron 3 的 Mamba-2 和 Kimi K3 的 KDA？
2. 把 projection 数据提前排到合适的 SRAM bank，能减少多少读取等待？

配套 Compiler 已固定在 `PLENA_Compiler/` 子目录。clone 本分支时使用
`--recurse-submodules`，即可取得匹配版本。

## 整体流程

```text
Compiler 生成 Matrix / L_SCATTER_M / X_STATE / MoE 指令
                         |
                         v
Rust Simulator 读取指令和内存镜像
                         |
                         v
执行 bank 排布、状态更新、Attention/MLA、MoE 和 residual
                         |
                         v
与独立 CPU 公式比较数值，并统计周期、访存、bank stall 和 cache 命中
```

### L-Compute 在做什么

Matrix SRAM 分成 16 个 bank，每个 bank 每拍只能服务一次读取。如果下一步同时
需要的数据都落在同一个 bank，就必须排队。`L_SCATTER_M` 根据下一位消费者的
读取方式，在 Matrix 写回时把数据放到不同 bank；`X_STATE` 随后直接读取这个布局。

它不改变 tensor 数值，也不把数据搬回 HBM 做 transpose。

## 当前进度

| 内容 | 状态 | 说明 |
|---|:---:|---|
| Nemotron 3 / Kimi K3 workload model | 完成 | 使用真实层数、shape 和 state；Nemotron 使用实测 routing，Kimi 可接入实测 trace，默认明确标为 sensitivity |
| GPU baseline 接入 | 完成 | B200/RTX 5090 数据用于检查工作量和瓶颈，不直接换算成 PLENA 周期 |
| Mamba-2 / KDA CPU reference | 完成 | 支持 FP32、BF16、FP16 和 MX8 state 实验 |
| Rust `X_STATE` | 完成 | 执行 Mamba-2/KDA 状态更新、cache、queue、fence 和 counters |
| Rust `L_SCATTER_M` | 完成 | 真实 banks x rows 存储，检查重复写、漏读、地址混淆和 spill |
| Compact Matrix hardware loops | 完成 | N/K tile 遍历由 `C_LOOP` 执行，MXFP8 与 BF16 stream-K 均已在 Rust 数值对拍 |
| 相邻层数值链 | 完成 | Mamba/KDA、MLA、Attention residual 和 MoE 已连接对拍 |
| 52/93 层 symbolic decode 机器码 | 完成 | Compiler 生成 52 层 23.663 MiB 和 93 层 44.490 MiB 合法机器码；权重尚未绑定 |
| 4-token GQA / compressed-MLA cache | 完成 | Nemotron 32Q/2KV GQA 与 Kimi 96-head MLA 均在 Rust 中逐 token 执行和对拍 |
| S16/S128 transactional prefill | 完成 | Mamba/KDA chunk、GQA/compressed-MLA cache 初始化与两种 MoE 均执行真实指令并对拍 |
| Compact synthetic 整模 Rust | 完成 | Nemotron 52 层和 Kimi 93 层均在一次 Rust 运行中执行；已覆盖 S128 prefill 和最长 128-token decode |
| 整模资源时间轴 | 完成 | 52/93 层共享 HBM/SRAM/Matrix/Vector/State，报告排队、traffic、TTFT 和 TPOT |
| Prefill 和整模多 token decode 模型 | 完成 | Mamba chunked SSD、KDA chunk-16、GQA/MLA cache 和逐 token context 均进入时间轴 |
| 统一硬件 DSE / 消融 | 完成 | 同一套 4096-MAC、16-bank、64-FIFO、32-MiB state-cache 候选同时评估两种模型 |
| Long-sequence state precision | 完成 | Mamba/KDA 的 BF16、FP16、MX8-B128 已跑 2K/8K/32K CPU recurrence |
| Kimi routing 数据接口 | 完成 | 校验 896 experts/top-16、92 个 MoE 层、连续 step、revision 和 prompt SHA；实测 trace 数据仍待采集 |
| 真实 checkpoint 整模 Rust | 未完成 | Symbolic HBM span 为 Nemotron 66.17 GB、Kimi 3.21 TB；当前 dense Rust 镜像后端不适用 |
| RTL、PPA 和相对 GPU 加速比 | 未开始 | 当前周期不能当成最终硬件性能 |

## 已验证结果

### SRAM bank 排布

| Workload | 普通排布读取周期 | 新排布读取周期 | 局部提升 |
|---|---:|---:|---:|
| Nemotron 的 23 个 Mamba 层 | 53,176 | 14,904 | 3.568x |
| Kimi 的 69 个 KDA 层 | 536,544 | 430,560 | 1.246x |

这里比较的是 **projection buffer 的读取服务时间**，不是整层、整模型或整颗芯片
的加速比。

### 整模 DSE 的结论

默认 DSE 是 B1 的两个独立校准点：prefill S128，以及在 context 2048 后连续 decode
4 token。同一硬件候选下：

| 模型 | 模拟 TTFT | 模拟 TPOT | Decode HBM utilization |
|---|---:|---:|---:|
| Nemotron 3 | 338.839 ms | 46.265 ms | 97.75% |
| Kimi K3 | 26.620 s | 907.500 ms | 96.75% |

这些是 `1 GHz / 64 B-cycle` 的 **pre-RTL 参数化结果**，不是实测硬件性能，也不是
相对 GPU 的加速比。它们的作用是指出当前整模主要受 weight/MoE HBM traffic 限制：
L-Compute 虽然清除了局部 bank stall，但整模收益会被 HBM 遮住。

S128 prefill 与长 decode 的 all-on/all-off 整模结果如下。这里的 off 同时关闭
L-Compute layout、projection bypass、state cache、fused activation flow 和 Mamba
B/C broadcast；其他硬件参数不变。

| 模型 | Decode steps | All on cycles | All off cycles | 整段提升 |
|---|---:|---:|---:|---:|
| Nemotron 3 | 32 | 1,817,709,624 | 1,869,299,644 | 1.02838x |
| Nemotron 3 | 127 | 6,208,266,140 | 6,359,219,795 | 1.02431x |
| Kimi K3 | 32 | 55,613,534,647 | 55,664,522,231 | 1.00092x |
| Kimi K3 | 128 | 142,599,838,039 | 142,749,520,535 | 1.00105x |

Nemotron 的 GPU trace 只有 127 个 recurrent decode steps；第一个输出 token 已包含在
TTFT 中，因此没有复制最后一步来伪造 D128。Kimi 仍使用 deterministic routing
sensitivity，尚不是实测完整 Kimi routing。详细边界和单项消融见
[`doc/HYBRID_FULL_SYSTEM_DSE_ZH.md`](doc/HYBRID_FULL_SYSTEM_DSE_ZH.md)。

### 连续数值执行

| 路径 | Rust 周期 | 最大绝对误差 | 结果 |
|---|---:|---:|---|
| Kimi MLA -> MoE | 58,782 | 0 | 通过 |
| Kimi AttnRes -> KDA -> AttnRes -> MoE | 96,980 | 0 | 通过 |
| Nemotron Mamba -> MoE | 1,725,597 | 0.03125 | BF16 容差内通过 |

这些测试使用确定性的合成权重。Mamba/KDA 保留真实状态维度，但外围 hidden size
会缩小，以便做快速、可重复的正确性测试。

### Transactional prefill

每一项都实际生成机器码、运行 Rust、更新持久状态/cache，并与独立 CPU 公式比较。
S128 的 Mamba/KDA 以 8 个 S16 chunk 执行。

| 模块 | S16 cycles | S128 cycles | 数值与持久数据 |
|---|---:|---:|---|
| Nemotron Mamba-2 | 87,803 | 661,103 | 100% allclose；state/conv state 检查通过 |
| Nemotron GQA | 2,179,833 | 14,259,499 | 100% allclose；4 份 K/V cache exact |
| Nemotron routed + shared MoE | 169,840 | 1,343,713 | 100% allclose；全部 Top-2 route 对拍 |
| Kimi KDA | 204,683 | 1,569,683 | output/state exact |
| Kimi compressed MLA (4 heads) | 1,129,127 | 8,341,875 | 100% allclose；compressed cache exact |
| Kimi LatentMoE | 250,520 | 1,988,266 | 100% allclose；全部 Top-2 route 对拍 |

### Compact synthetic 整模

这不是按层结果相加，而是一次 Rust invocation 中连续执行所有层；前一层输出就是
下一层输入，state、cache、residual 和 routing 都跨层/跨 token 保留。

| 模型 | 实际层结构 | 指令数 | Rust cycles | 检查结果 |
|---|---|---:|---:|---|
| Nemotron 3 | 23 Mamba + 23 MoE + 6 GQA | 426,814 | 13,660,404 | 1,040 checkpoints 100%；23 state 与 6 cache 生命周期通过 |
| Kimi K3 | 69 KDA + 24 MLA + 92 LatentMoE + 1 dense FFN | 4,646,465 | 80,522,239 | 3,740 checkpoints 100%；69 state、24 compressed cache 与 routing 生命周期通过 |

Kimi 的 24 个持久 MLA 对象只保存 compressed latent；展开的 all-head K/V HBM
对象数量为 0。上述周期来自 compact synthetic correctness fixture，不能当作真实尺寸
Nemotron/Kimi 的 TPOT 或与 GPU 的加速比。

下表是同一套 compact 整模在更长请求上的 Rust 数值门禁。指令数、周期和 HBM
traffic 都由一次完整 invocation 统计，不是把单层数据相加。

| 模型 | Prefill | Decode | 指令数 | Rust cycles | HBM read / write | 数值结果 |
|---|---:|---:|---:|---:|---:|---|
| Nemotron 3 | 128 | 4 | 2,958,660 | 64,403,989 | 165.20 / 12.14 MB | 99.9898% allclose |
| Nemotron 3 | 16 | 32 | 1,224,835 | 59,060,304 | 260.62 / 29.92 MB | 99.9674% allclose |
| Nemotron 3 | 16 | 128 | 4,165,000 | 219,857,989 | 1,031.49 / 114.17 MB | 99.8591% allclose |
| Kimi K3 | 128 | 4 | 29,761,510 | 420,873,526 | 986.18 / 105.26 MB | 100% allclose |
| Kimi K3 | 16 | 32 | 15,274,956 | 348,537,791 | 1,383.57 / 201.06 MB | 100% allclose |
| Kimi K3 | 16 | 128 | 66,016,808 | 1,492,322,041 | 5,666.65 / 999.39 MB | 100% allclose |

Nemotron D128 最终 hidden 的全局 relative-L2 为 1.9609%，低于 3% 长序列门限；
但最后一层、最后一个 token 的 allclose 分别降到 99.3598% 和 98.9483%。这是
BF16 跨 52 层迭代的误差积累，不能只报全局通过率。
Kimi 三个长测试都保持 24 个 compressed MLA cache，expanded persistent K/V
对象仍为 0。Kimi D128 的整模 hidden relative-L2 为 0.2163%；使用 Rust 实际的
producer hidden 重新计算每层 cache projection 后，24 个 cache 的最大绝对误差和
relative-L2 均为 0。这把上游 BF16 误差积累与 cache 地址/append/writeback 正确性
分开检查，避免把前者误报成 cache bug。

长测试为了在一次 Rust invocation 中检查每个 token 的 hidden/state/cache，会把
多个 decode step 展开到测试机器码中。这不是部署方案：真实设备应循环复用单
token decode program，不应把 D128 的整个验证 trace 放入 instruction memory。

基础 S16+D4 的 3,680 个 Kimi route decisions 都由每个 MoE 后的 hidden checkpoint
覆盖；D128 运行共检查 26,496 个 route decisions。受
1,024-entry Int SRAM 容量限制，最终一层的 40 个 expert IDs 还会直接从 dump 与 CPU
结果比较。参考机器上完整 Nemotron/Kimi fixture 约需 34 秒/198 秒，并分别产生约
557 MiB/1.2 GiB 的可重建 build artifacts。

### 4-token decode cache

| 路径 | Rust 周期 | 输出误差 | Cache 检查 |
|---|---:|---:|---|
| Nemotron GQA，32Q/2KV，head dim 128 | 2,655,669 | 0.0008544921875，100% allclose | 4 个 K/V tensor 全部 exact |
| Kimi compressed MLA，96 heads、4 个不同 RoPE 位置 | 30,021,830 | 0，100% exact | compressed history 与重建 K/V 全部 exact |

Kimi 每个 token 只把 576-wide compressed latent/shared-RoPE history 持久化到 HBM，
然后逐 head 重建 192-wide K 和 128-wide V，并复用同一对 single-head scratch。
HBM manifest 审计结果中 expanded all-head cache 对象为 0；4-token persistent payload
是 4,608 B，而展开 96-head K/V 需要 245,760 B，即 53.33x 更大。

### Compact Matrix 循环

| 测试 | 指令数 | Rust 周期 | 数值结果 |
|---|---:|---:|---|
| MXFP8 `1x320 @ 320x384`，2 个 K chunk、6 个 N tile | 93 | 38,215 | 384/384 exact |
| BF16 stream-K `1x320 @ 320x128`，5 个 K tile | 71 | 37,596 | 128/128 exact |

这两项证明 compact lowering 生成的嵌套 `C_LOOP` 在 Rust 中会访问正确的 HBM、
MRAM 和 VRAM 地址；它们不是完整模型的周期结果。

## 快速开始

```bash
git clone --branch feature/mamba-kda-support --recurse-submodules \
  https://github.com/AICrossSim/PLENA_Simulator.git
cd PLENA_Simulator
uv sync --frozen
```

先运行不需要 Rust 编译的 CPU 测试：

```bash
nix develop --no-write-lock-file --command just test-common-state-python
```

再运行三条连续数值链：

```bash
nix develop --no-write-lock-file --command just test-kimi3-connected --stage all
nix develop --no-write-lock-file --command just test-kimi3-kda-connected --stage all
nix develop --no-write-lock-file --command just test-nemotron3-mamba-connected --stage all
nix develop --no-write-lock-file --command just test-kimi3-compact-matrix
nix develop --no-write-lock-file --command just test-kimi3-compact-stream-k
nix develop --no-write-lock-file --command just test-nemotron3-gqa-cache
nix develop --no-write-lock-file --command just test-kimi3-mla-cache
nix develop --no-write-lock-file --command just test-state-prefill --model all --tokens 16
nix develop --no-write-lock-file --command just test-state-prefill --model all --tokens 128
nix develop --no-write-lock-file --command just test-moe-prefill --model all --tokens 16
nix develop --no-write-lock-file --command just test-moe-prefill --model all --tokens 128
nix develop --no-write-lock-file --command just test-nemotron3-full-synthetic
nix develop --no-write-lock-file --command just test-kimi3-full-synthetic
nix develop --no-write-lock-file --command just test-nemotron3-full-synthetic \
  --prefill-tokens 128 --decode-tokens 4 --build-dir build/nemotron-s128-d4
nix develop --no-write-lock-file --command just test-nemotron3-full-synthetic \
  --prefill-tokens 16 --decode-tokens 128 --build-dir build/nemotron-s16-d128
nix develop --no-write-lock-file --command just test-kimi3-full-synthetic \
  --prefill-tokens 128 --decode-tokens 4 --build-dir build/kimi-s128-d4
nix develop --no-write-lock-file --command just test-kimi3-full-synthetic \
  --prefill-tokens 16 --decode-tokens 128 --build-dir build/kimi-s16-d128
```

Python 必须在 `uv` 环境中启动；上面的 `just` recipe 会自动选择 `uv run python`，同时
使用 Nix 提供的 `just`、Rust 和 Ramulator。不要在 Nix 自带的 Python 中直接运行脚本，
因为它不包含 PyTorch。
第一次会编译 Rust emulator，需要几分钟；之后会复用构建结果。这些测试不下载模型
权重，也不需要 GPU。Docker 说明见 [`docker/README.md`](docker/README.md)。

## 运行 DSE

Nemotron 3 工作量和布局/cache 对比：

```bash
just nemotron3-workload --phase decode --decode-tokens 4 --body-only
just nemotron3-dse --decode-tokens 4 --weight-precision nvfp4 \
  --json-out build/nemotron3-dse.json
```

Kimi K3 层结构和 state-cache 容量：

```bash
just kimi-k3-full-workload --phase decode --batch-size 1 \
  --context-length 2048 --json-out build/kimi-k3-workload.json
just kimi-k3-cache-dse --json-out build/kimi-k3-cache.json
```

运行两种模型共用资源的整模 DSE：

```bash
just hybrid-system-dse --model all --grid quick \
  --context-length 2048 --decode-tokens 4 --prefill-tokens 128 \
  --json-out build/hybrid-system-dse.json
```

拿到 Kimi 实测 routing 后，可按
`analytic_models/performance/profiles/kimi_k3_routing_v1.schema.json` 整理，再运行：

```bash
just hybrid-system-dse --model kimi_k3 --grid quick \
  --kimi-routing-trace /path/to/kimi-k3-routing.json \
  --context-length 2048 --decode-tokens 4 --prefill-tokens 128 \
  --json-out build/kimi-k3-empirical-routing-dse.json
```

DSE 可以直接使用仓库中的标准化 profiling 摘要；不需要重新运行 B200/RTX 5090。
原始 GPU 时间只作为 baseline 和模型校验，不会被当成 PLENA 指令延迟。

## 代码位置

| 目录 | 内容 |
|---|---|
| `analytic_models/reference/` | Mamba-2 和 KDA 的 CPU 正确答案 |
| `analytic_models/performance/` | workload、GPU profile、cache、bank 和 mixed-precision DSE |
| `transactional_emulator/src/state_engine/` | Rust 状态引擎、布局、cache、精度和周期模型 |
| `transactional_emulator/testbench/models/` | Kimi/Nemotron 连续数值测试 |
| `doc/` | 详细设计、实验结果和当前限制 |

重点文档：

- [RTL 前完成情况](doc/L_COMPUTE_PRE_RTL_STATUS_ZH.md)
- [连续数值验证](doc/connected_hybrid_validation.md)
- [Projection bank/FIFO DSE](doc/PROJECTION_SCATTER_DSE_ZH.md)
- [整模联合模拟、DSE、消融和 mixed precision](doc/HYBRID_FULL_SYSTEM_DSE_ZH.md)

## 还不能怎么表述

- 可以说 compact synthetic 52/93 层已从第一层到最后一层在 Rust 跑通；不能说真实
  checkpoint 权重和真实外围宽度也已经整模执行。
- 可以说完整 52/93 层已经进入参数化资源时间轴；不能把 analytic timeline 写成
  “完整真实权重机器码已在 Rust 数值执行”。
- 不能把上面的 bank 局部提升写成 PLENA 相对 B200/RTX 5090 的整模加速。
- 不能把 Simulator 的地址生成和 bank 周期当成 RTL 已经满足频率与面积要求。
- MX8 state 的短序列误差实验只能说明值得继续测试，不能证明真实任务精度不掉。

PLENA 原始 Simulator 的通用说明和论文信息见
[`main` 分支 README](https://github.com/AICrossSim/PLENA_Simulator/blob/main/README.md)。
