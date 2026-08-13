# Nemotron 3 Analytic Baseline

本文记录第一版、尚未经过 GPU 或 RTL 校准的 Nemotron 3 Nano 30B-A3B
workload 与硬件 DSE 结果。它的用途是检查公式、比较设计选项和确定 profiling
需求，不能作为论文中的最终 latency 或 speedup。

## 固定模型参数

- 52 层：23 Mamba、23 MoE、6 Attention
- Mamba：64 heads、head dimension 64、state dimension 128、8 groups
- 一个请求的 23 层 persistent state 加 conv state 约 48.16 MiB
- BF16 activation/weight，FP32 recurrent state

## Decode baseline

场景：batch 1、context 2048、连续 decode 16 tokens、排除 embedding/lm head。
workload 数值均为一个 decode step 的逻辑工作量。

```text
FLOPs/token                 6.028 GFLOPs
weight read/token           5,484.16 MiB
KV read/token                  12.00 MiB
state read/token               48.16 MiB
state write/token              48.16 MiB
total logical HBM read       5,546.21 MiB
total logical HBM write         50.12 MiB
```

在当前假设的 1 GHz、64 B/cycle HBM 和 256/512 state MAC/cycle 下：

| 设计 | us/decode step | Mamba us | HBM MiB/step | state hit |
|---|---:|---:|---:|---:|
| 无 state cache、skewed、B/C broadcast | 91,717.6 | 29,469.5 | 5,596.3 | 0% |
| 16 MiB LRU | 91,717.6 | 29,469.5 | 5,596.3 | 0% |
| 16 MiB pinned | 91,252.4 | 29,004.3 | 5,567.9 | 30.4% |
| 64 MiB state cache | 90,259.6 | 28,011.5 | 5,503.0 | 100% |

16 MiB LRU 为 0% hit 不是程序错误。decode 按 23 个 Mamba 层循环访问，cache
装不下全部 layer state 时，下一 token 到来前旧 entry 已被逐个替换；容量感知的
pinned policy 至少可以保留固定子集。

当前全模型只改善约 1.6%，Mamba 部分改善约 5%。原因是模型假设每个 step
都从 HBM 读取 active weights，约 5.36 GiB 权重流量盖过了 96 MiB state
读写。论文必须同时报告 full-model 和 Mamba-only breakdown。

## Batch 4 decode

64 MiB 只能容纳约三分之一的四请求 state。LRU 仍然 thrash，pinned policy
达到 32.6% hit；容纳全部 state 需要约 256 MiB candidate cache。

| 设计 | us/batch step | Mamba us | HBM MiB/step | state hit |
|---|---:|---:|---:|---:|
| 无 cache | 226,410.1 | 34,356.8 | 13,811.8 | 0% |
| 64 MiB LRU | 226,410.1 | 34,356.8 | 13,811.8 | 0% |
| 64 MiB pinned | 224,416.2 | 32,362.9 | 13,690.1 | 32.6% |
| 256 MiB cache | 220,578.0 | 28,524.8 | 13,438.6 | 100% |

一个 batch step 在这里生成 4 个 token，不能把表中的数直接称为 per-token
latency。

## Prefill 2048 ablation

场景：batch 1、sequence 2048、chunk size 128、chunked affine scan。

```text
workload FLOPs             12.190 TFLOPs
logical HBM read           62,760.35 MiB
logical HBM write           4,072.91 MiB
```

| Projection layout | B/C broadcast | Total ms | Mamba ms | Bank stall cycles |
|---|---:|---:|---:|---:|
| row-major | off | 2,176.75 | 826.52 | 120,586,240 |
| row-major | on | 2,171.44 | 821.20 | 78,381,056 |
| skewed | off | 2,140.58 | 790.34 | 0 |
| skewed | on | 2,135.26 | 785.03 | 0 |

在当前 packet 和 bank 假设下，skewed layout 消除了 projection-buffer bank
stall；B/C broadcast 进一步减少 group-shared B/C 的重复读取和 prefill C-B
计算。但这些周期还是 uncalibrated，不能直接当 FPGA latency。

## 可复现命令

环境没有 `just` 时，可以直接运行底层模块：

```bash
uv run python -m analytic_models.performance.nemotron3_model \
  --mode workload --phase decode --decode-tokens 16 --body-only \
  --json-out build/nemotron3_decode_workload.json

uv run python -m analytic_models.performance.nemotron3_model \
  --mode sweep --phase decode --decode-tokens 16 --body-only \
  --sweep-layouts row_major,group_major_skewed \
  --sweep-broadcasts 0,1 --sweep-cache-mib 0,16,64 \
  --sweep-cache-policies none,lru,pinned \
  --sweep-state-dim-lanes 8,16 \
  --json-out build/nemotron3_decode_dse.json
```

## 当前模型还不能证明什么

1. `matrix_macs_per_cycle`、state lanes、HBM bandwidth 是 candidate 参数，不是
   已综合 RTL 的实测值。
2. logical HBM activation traffic 还没有根据真实 SRAM residency 和 fusion
   消除中间结果 spill。
3. MoE unique-expert 数量使用场景假设；必须用真实 router trace 验证。
4. GPU 时间只能作为 baseline 和 workload validation，不能代替 PLENA RTL
   cycle calibration。
5. skewed SRAM 的面积、频率和布线代价必须等 RTL synthesis 后才能评价。
