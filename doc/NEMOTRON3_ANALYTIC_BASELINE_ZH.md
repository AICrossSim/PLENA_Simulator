# Nemotron 3 Simulator Calibration and DSE

本文记录进入 RTL 前，Nemotron 3、KDA、L-Compute 和系统 DSE 已经完成到哪里。
最重要的边界是：**真实 workload 和 GPU baseline 已校准，PLENA cycle 与 PPA 尚未校准。**

## 输入证据

完整 B200 campaign 已本地解包并重新校验：

- archive：`kda_nemotron_nvfp4_campaign_gpu3_attempt2_COMPLETE_20260819T043954Z.tar.gz`
- size：`123,292,703 bytes`
- SHA256：`eac1d2637ff82286365070a40e21e260d222b53b0c3b3b28172f5ef925ec15c9`
- 18 个 KDA NCU、6 个 Nemotron NCU、4 个 Nemotron NSYS
- 80 条完整模型 latency 原始记录
- 3,013 个 MoE layer-step routing event
- 顶层 43 个 checksum 和 193 个 collection artifact 已验证

归一化的 checked-in contract：

- `analytic_models/performance/profiles/b200_kda_nemotron_campaign_complete.json`
- `analytic_models/performance/profiles/nemotron3_decode_routing_trace.json`

`b200_campaign_raw.py` 可以从 raw campaign 重新生成这两个文件，避免手工抄表。

## 已校准的模型合同

| 项目 | 真实配置 |
|---|---|
| 层结构 | 23 Mamba + 23 MoE + 6 Attention = 52 层 |
| Mamba | 64 heads，head dim 64，state dim 128，8 groups |
| MoE | 128 routed experts，top-6，1 shared expert |
| activation | BF16 |
| recurrent state | FP32 |
| KV cache | FP8（GPU checkpoint） |
| 默认 linear weight | NVFP4 group size 16 |

checkpoint 不是“所有权重都 NVFP4”。真实 exclusion 为：

- Mamba `in_proj/out_proj` 的第 4/11/18/25/32/41 层保持 BF16；
- 6 个 Attention 层的 Q/K/V/O projection 全部 BF16；
- 23 个 Mamba conv 全部 BF16；
- embedding、norm 和 lm_head 按 model dtype/BF16；
- 其他 eligible linear，包括 MoE，使用 NVFP4 logical storage。

`WeightPrecisionPolicy` 已把这个 layer/stage map 接入 workload 和 DSE。uniform BF16
和 uniform NVFP4 仍可做 ablation，但不会再被标成“与正式 checkpoint 一致”。

## GPU baseline

### 完整 Nemotron

| Case | Median | P95 |
|---|---:|---:|
| Prefill S128 TTFT | 59.713778 ms | 62.203214 ms |
| Prefill S2048 TTFT | 57.098733 ms | 58.073123 ms |
| Prefill S8192 TTFT | 64.431871 ms | 64.592696 ms |
| Decode ITL，context 2048 | 4.047566 ms | 4.083077 ms |

Decode NCU layer-type 结果：

| Layer type | Kernel duration | DRAM read | 时间占比 |
|---|---:|---:|---:|
| Mamba | 3.865670 ms | 873,460,520 B | 35.73% |
| Attention | 0.423070 ms | 294,475,450 B | 3.91% |
| MoE | 6.530860 ms | 1,114,920,410 B | 60.36% |

Prefill S128 中 MoE 读取 12.538 GiB，约为 Mamba 的 8.9 倍。系统优化不能只做
state engine；MoE weight placement、reuse 和调度必须一起研究。

### KDA 单层

| Case | Kernel time | Matrix path | state core |
|---|---:|---:|---:|
| Prefill B1/S2048 | 3.01731 ms | 74.33% | 15.34% |
| Decode B1 | 0.35995 ms | 74.45% | 5.02% |
| Decode B8 | 0.41188 ms | 62.25% | 11.65% |

KDA recurrent state 是 `[B,96,128,128]` FP32，即 6 MiB/request/layer；三个 BF16
conv state 合计 0.28125 MiB。官方 wrapper 对拍 S=1/16/256/2048 全部逐元素一致。

即使把 KDA state core 变成无限快，完整单层上限也只有：Decode B1 1.053x、Decode
B8 1.132x、Prefill 1.181x。它证明 state engine 必须做，但不能把它包装成唯一瓶颈。

## Logical traffic 校准结果

使用真实 mixed checkpoint map 后，一个 Decode step 的 Simulator logical read 与
B200 physical DRAM read 为：

| Layer type | Logical | B200 physical | physical/logical |
|---|---:|---:|---:|
| Mamba | 0.8261 GiB | 0.8135 GiB | 0.9847 |
| Attention | 0.2733 GiB | 0.2743 GiB | 1.0033 |
| MoE | 0.9673 GiB | 1.0384 GiB | 1.0734 |

这说明 Decode shape、混合权重字节和主要 traffic accounting 已基本对齐。这个比例
不能直接乘到 PLENA HBM 上，因为它包含 B200 的 cache、kernel replay 和 runtime 行为。

Prefill S128 的 Attention/MoE 比例为 1.058/1.060，但 Mamba 是 1.678。说明当前
chunked scan 的中间张量、读写和 cache residency 仍过于理想，是 analytic model
下一项需要细化的内容。

## L-Compute DSE

冻结配置：16 single-port banks、64-value producer burst、64-value FIFO、BF16
activation。完整 23 层 Mamba decode packet 的结果：

| Layout | Read service | Read+write service | Stall |
|---|---:|---:|---:|
| row-major | 53,176 cycles | 67,988 cycles | 38,272 |
| Mamba skew | 14,904 cycles | 29,808 cycles | 92 |

- consumer read 局部提升 3.568x；
- read+write buffer service 局部提升 2.281x；
- 236,992 个值全部完成物理 banks-by-rows roundtrip；
- alias、重复写、未写读取的负向测试会失败；
- HBM repack bytes 为 0。

按 B200 Decode 中 Mamba 35.73% 占比做极度乐观的 Amdahl 上限：假设整个 Mamba
类别都获得 2.281x，整机也只到 1.251x。真实收益必然更低，因为 L-Compute 只优化
projection handoff，不减少 Mamba projection GEMM 权重与 MAC。

## Mamba state-cache DSE

| State | 每层 | 23 层准确容量 | 整数 MiB 配置 |
|---|---:|---:|---:|
| FP32 | 2.09375 MiB | 48.15625 MiB | 至少 49 MiB |
| BF16 | 1.046875 MiB | 24.078125 MiB | 至少 25 MiB |

因此 48 MiB FP32 cache 并不能完整驻留一个请求。正式 GPU baseline 必须使用 FP32；
BF16/MX8 只能作为 numerical candidate。部分容量下 layer-major LRU 会 thrash，pinned
policy 才能稳定保留固定层。

## 真实 routing 驱动的 MoE weight-cache DSE

compact trace 保存 context 2048 prefill active set，以及之后 127 个 decode step、每步
23 层、每层 6 个真实 expert ID。一个 routed expert 的两个 NVFP4 矩阵约 5.3525 MiB。

| Routed slots | SRAM | LRU hit（expert-ID 调度） |
|---:|---:|---:|
| 92 | 492.4 MiB | 0.00% |
| 137 | 733.3 MiB | 9.16% |
| 138 | 738.7 MiB | 68.11% |
| 256 | 1,370.2 MiB | 92.42% |
| 512 | 2,740.5 MiB | 94.73% |
| 1,024 | 5,481.0 MiB | 95.90% |

138 附近的突变来自一个 token window 有 `23 x 6 = 138` 个 routed access。若保留
top-k rank 访问顺序，138 slots 为 72.11%；若 scheduler 按 expert ID 分组则为
68.11%。所以最终硬件调度顺序必须作为 DSE 参数，不能只报一个命中率。

23 层 shared expert 全驻留还需 246.22 MiB。138 个 routed slots 本身约 738.65 MiB，
两者合计约 984.87 MiB，不能默认当作 FPGA 片上 SRAM；它只是容量上界。

## 真实 routing 驱动的 MoE 事件级 DSE

每一个 routed cache miss 现在都已接入共享时间线。时间线包含：固定 4096 PE、64B
burst、有限 weight-buffer slots、一个共享 HBM server、异步 Matrix 完成和一个共享
reduction/postprocess resource。比较的映射包括 Expert split、M split、K split、
stage-aware N-to-K、M-by-N、M-by-K 和一个非 oracle dynamic heuristic。

为了不把旧数据冒充新标定，报告并列保留两套周期假设：

| 假设 | 含义 | 138-slot + shared resident 下最佳候选 | Expert-body 局部加速 |
|---|---|---|---:|
| ideal geometry + 64 B/cycle | 只按 array occupancy 和 HBM 算的理想下限 | 4 x 1x1024 K-split | 1.136x |
| transferred Shared-MoE | 旧 Qwen/DeepSeek PLENA Matrix/HBM 常数按 MAC 比例转移 | 4 x 1x1024 K-split | 3.999x |

所有候选还固定使用同一个 10.705 MiB byte-accurate staging buffer，可放两份 routed
权重或一份 shared 权重；split 不会得到额外 buffer。两套假设都把 K-split 排第一，
但收益差了三倍以上，因此还不能冻结 RTL 数字。B1 下
每个 routed expert 只有一个 token，M-split 会退化为一个有效 core；M-by-N/M-by-K
的 M 维也同样塌缩。无 routed cache 且 shared streaming 时，模型读取约 0.961
GiB/token 的 expert weights，与独立 logical workload accounting 对齐。

## 当前 PLENA system sweep 的含义

formal report 固定以下未校准候选：1 GHz、4096 Matrix MAC/cycle、64 B/cycle HBM、
16 banks、1 port、64-value FIFO、256 state MAC/cycle。127-step decode 的结果约为：

| Candidate | Analytic us/token | HBM GiB/token | State hit |
|---|---:|---:|---:|
| row buffered + state stream | 35,564.9 | 2.116 | 0% |
| skew buffered + state stream | 35,547.2 | 2.116 | 0% |
| skew bypass + state stream | 35,547.2 | 2.116 | 0% |
| skew bypass + FP32 resident | 34,110.8 | 2.022 | 100% |

这些时间不能与 B200 的 4.047566 ms 做 speedup。它们只说明在当前单核和每步重读
MoE weights 的保守模型中，MoE/weight traffic 完全盖过 L-Compute 的局部收益。

## 已完成和未完成

已完成：

1. raw campaign 导入、hash/shape/counter/routing 一致性校验；
2. 真实 mixed NVFP4/BF16 checkpoint policy；
3. Decode/Prefill logical-vs-physical traffic cross-check；
4. L-Compute row/transpose/skew/custom 的真实 banked roundtrip 和 service cycles；
5. FP32/BF16 state capacity、LRU/pinned DSE；
6. 真实 127-step routing 的 routed/shared expert weight-cache DSE；
7. cache miss 驱动的 Expert/M/K 多核事件调度、HBM starvation、有限 buffer 和 reduction；
8. formal JSON/Markdown 报告和回归测试。

进入 RTL 前仍要做：

1. 用直接 Nemotron expert 的 Rust/RTL 周期替换转移来的 Matrix、dequant、reduction
   和 overlap 常数；
2. 细化 Prefill Mamba chunked-scan 中间 traffic，使 1.678x 差距可解释；
3. 用长序列和任务精度验证 FP32/BF16/MX8 state，才能冻结默认 state dtype；
4. 定义并验证 NVIDIA NVFP4 dequant/scale 的真实 Matrix 数据通路；
5. KDA 目前是完整真实单层校准；在声称 Kimi 93 层系统结果前，还需 MLA/LatentMoE
   代表层或完整模型 GPU 数据；
6. RTL/FPGA 测 Matrix、Vector、HBM、state engine、L-Compute 的周期、频率、面积和能耗。

## 复现

```bash
uv run python -m analytic_models.performance.b200_formal_campaign

uv run python -m analytic_models.performance.nemotron3_routing_dse \
  --json-out artifacts/dse/nemotron3_profile_driven_routing_dse.json

uv run python -m analytic_models.performance.nemotron3_moe_event_dse \
  --json-out artifacts/dse/nemotron3_moe_event_dse.json \
  --markdown-out artifacts/dse/nemotron3_moe_event_dse.md

uv run python -m analytic_models.performance.nemotron3_formal_dse \
  --json-out artifacts/dse/nemotron3_formal_calibrated_dse.json \
  --markdown-out artifacts/dse/nemotron3_formal_calibrated_dse.md
```

权威生成结果是 `artifacts/dse/nemotron3_formal_calibrated_dse.{json,md}`。
