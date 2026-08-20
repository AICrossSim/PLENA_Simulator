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
| Nemotron 3 / Kimi K3 workload model | 完成 | 使用真实层数、shape、state 大小和 routing 数据 |
| GPU baseline 接入 | 完成 | B200/RTX 5090 数据用于检查工作量和瓶颈，不直接换算成 PLENA 周期 |
| Mamba-2 / KDA CPU reference | 完成 | 支持 FP32、BF16、FP16 和 MX8 state 实验 |
| Rust `X_STATE` | 完成 | 执行 Mamba-2/KDA 状态更新、cache、queue、fence 和 counters |
| Rust `L_SCATTER_M` | 完成 | 真实 banks x rows 存储，检查重复写、漏读、地址混淆和 spill |
| Compact Matrix hardware loops | 完成 | N/K tile 遍历由 `C_LOOP` 执行，MXFP8 与 BF16 stream-K 均已在 Rust 数值对拍 |
| 相邻层数值链 | 完成 | Mamba/KDA、MLA、Attention residual 和 MoE 已连接对拍 |
| 52/93 层 symbolic decode 机器码 | 完成 | Compiler 生成 52 层 23.66 MiB 和 93 层 43.88 MiB 合法机器码；权重尚未绑定 |
| 4-token GQA / compressed-MLA cache | 完成 | Nemotron 32Q/2KV GQA 与 Kimi 96-head MLA 均在 Rust 中逐 token 执行和对拍 |
| Prefill 和整模多 token decode | 未完成 | 4-token 测试覆盖独立 block；52/93 层 symbolic builder 仍是单 token |
| 真实权重整模 Rust 执行 | 未完成 | 还不能给出 PLENA 整模 latency |
| state lane 宽度 sweep | 未完成 | 决定 L-Compute 是否值得占 RTL 面积的关键下一步 |
| RTL、PPA 和相对 GPU 加速比 | 未开始 | 当前周期不能当成最终硬件性能 |

## 已验证结果

### SRAM bank 排布

| Workload | 普通排布读取周期 | 新排布读取周期 | 局部提升 |
|---|---:|---:|---:|
| Nemotron 的 23 个 Mamba 层 | 53,176 | 14,904 | 3.568x |
| Kimi 的 69 个 KDA 层 | 536,544 | 430,560 | 1.246x |

这里比较的是 **projection buffer 的读取服务时间**，不是整层、整模型或整颗芯片
的加速比。

### 连续数值执行

| 路径 | Rust 周期 | 最大绝对误差 | 结果 |
|---|---:|---:|---|
| Kimi MLA -> MoE | 71,097 | 0.001953125 | 通过 |
| Kimi AttnRes -> KDA -> AttnRes -> MoE | 96,980 | 0 | 通过 |
| Nemotron Mamba -> MoE | 1,725,603 | 0.046875 | BF16 容差内通过 |

这些测试使用确定性的合成权重。Mamba/KDA 保留真实状态维度，但外围 hidden size
会缩小，以便做快速、可重复的正确性测试。

### 4-token decode cache

| 路径 | Rust 周期 | 输出误差 | Cache 检查 |
|---|---:|---:|---|
| Nemotron GQA，32Q/2KV，head dim 128 | 2,615,503 | 0.0008544921875，100% allclose | 4 个 K/V tensor 全部 exact |
| Kimi compressed MLA，96 heads、4 个不同 RoPE 位置 | 37,246,986 | 0，100% exact | compressed history 与重建 K/V 全部 exact |

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

## 还不能怎么表述

- 可以说 Compiler 已生成完整 52/93 层的 symbolic-weight decode 机器码；不能说它们
  已绑定真实权重或从第一层到最后一层在 Rust 跑通。
- 不能把上面的 bank 局部提升写成 PLENA 相对 B200/RTX 5090 的整模加速。
- 不能把 Simulator 的地址生成和 bank 周期当成 RTL 已经满足频率与面积要求。
- MX8 state 的短序列误差实验只能说明值得继续测试，不能证明真实任务精度不掉。

PLENA 原始 Simulator 的通用说明和论文信息见
[`main` 分支 README](https://github.com/AICrossSim/PLENA_Simulator/blob/main/README.md)。
