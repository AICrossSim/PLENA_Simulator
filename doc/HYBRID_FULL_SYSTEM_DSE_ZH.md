# Nemotron 3 / Kimi K3 整模联合模拟与 RTL 前 DSE

## 这轮解决了什么

Simulator 现在不再分别报告一个 Mamba/KDA 小算子，而是把完整文本骨干按真实顺序
放到同一套 PLENA 资源上：

- Nemotron 3：23 Mamba + 23 MoE + 6 GQA；
- Kimi K3：69 KDA + 24 compressed MLA + 92 LatentMoE + 1 dense FFN + AttnRes；
- 每个 stage 竞争同一个 HBM server、片上 SRAM 带宽以及有限的 Matrix、Vector、
  State、Conv、Exp 和 L-Compute 资源；
- prefill 使用 Mamba chunked SSD 与 KDA chunk-16 prepare/recurrence；
- decode 会连续执行多个 token，context 逐 token 增长，state、GQA KV 和 compressed
  MLA history 持续更新；
- exact Nemotron 127-step top-6 routing 会驱动 expert cache；Kimi top-16 仍明确标成
  synthetic sensitivity traffic，不能冒充实测 routing；拿到实测数据后可通过严格校验的
  JSON trace 接口直接替换，不需要修改 DSE 代码。

入口是 `analytic_models/performance/hybrid_system_timeline.py`，DSE 驱动器是
`hybrid_system_dse.py`。

## 时间轴怎样计算

每个 stage 先按 workload contract 得到 MAC、elementwise/exp、state/KV/weight bytes，
再按以下顺序进入资源时间轴：

```text
shared HBM/SRAM read
        -> Matrix / Vector / State / Conv / Exp service
        -> shared SRAM write
        -> asynchronous shared-HBM writeback
```

前一 stage 的结果必须 ready，后一 stage 才能开始；state/KV writeback 可以与后续
compute 重叠，但会和下一次 HBM read 排队。所有 HBM transaction 按 64-byte burst
rounding。层内 hidden/residual tile 留在 4 MiB activation SRAM，避免把独立 stage
workload 中的 activation 读写重复计算成整模 HBM traffic。

这里的 cycles 来自显式硬件参数，不是把 B200/5090 kernel 时间换算成 PLENA 周期。
GPU 数据只校验 shape、dtype、routing、DRAM 方向和瓶颈。

## 同一设备的 RTL 前候选

Nemotron 与 Kimi 使用完全相同的硬件容量，而不是分别给两套 SRAM：

| 参数 | 候选值 |
|---|---:|
| Matrix throughput | 4096 MAC/cycle |
| State throughput | 8 head × 4 head-dim × 8 state-dim = 256 MAC/cycle |
| Projection layout SRAM | 16 single-port banks |
| Matrix result burst / FIFO | 64 / 64 values |
| Activation SRAM | 4 MiB |
| Persistent state cache | 32 MiB |
| KV read cache | 64 MiB |
| MoE weight cache | 256 MiB |
| HBM service | 64 B/cycle, 64-byte burst |

这只是 Compiler/Simulator 的 pre-RTL knee，不是最终 PPA freeze。频率、SRAM macro、
端口 mux 和 HBM controller efficiency 仍需 RTL 综合校准。

## 默认整模结果

固定场景包含 B1 的两个独立校准点：prefill S128，以及在 context 2048 后连续 decode
4 token。它们不是同一个连续请求；连续 S128 请求使用下一节的 context 128 报告。完整 JSON 位于
`analytic_models/performance/profiles/hybrid_system_dse_quick_v1.json`。

| 模型 | Prefill TTFT | Decode TPOT | Decode HBM traffic/4 token | Decode HBM utilization |
|---|---:|---:|---:|---:|
| Nemotron 3 | 338.839 ms | 46.265 ms | 10.78 GiB | 97.75% |
| Kimi K3 | 26.620 s | 907.500 ms | 209.34 GiB | 96.75% |

这些不是 PLENA 对 GPU 的加速比。它们说明在当前单设备带宽假设下，整模首先是
weight/MoE streaming 问题；Kimi 2.8T 的单设备结果尤其不适合被当成最终系统方案。
真正论文比较需要 RTL 频率/PPA、HBM 配置和多 device partition 后重新计算。

Nemotron 的旧 profile 曾错误地用 S2048 的活跃专家集合估算 S128 prefill，因此得到
393.122 ms。当前 loader 会按 sequence length 选择实测集合：S128 为 2,185 个活跃
layer/expert slots，S2048 为 2,807 个；上表 338.839 ms 是修正后的结果。

## S128 + 长 decode 的整模 A/B

下面每一行使用真实模型层数、真实外围 shape 和同一套共享资源。`all off` 同时关闭
L-Compute consumer layout、projection bypass、32 MiB state cache、fused activation
flow 和 Mamba B/C broadcast；其他吞吐、容量、routing 和 HBM 参数不变。Speedup 定义
为 `all-off cycles / all-on cycles`。

| 模型 | Decode steps | All on cycles | All off cycles | 整段 speedup | All-on TTFT | All-on TPOT |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 32 | 1,817,709,624 | 1,869,299,644 | 1.02838x | 338.839 ms | 46.215 ms |
| Nemotron 3 | 127 | 6,208,266,140 | 6,359,219,795 | 1.02431x | 338.839 ms | 46.216 ms |
| Kimi K3 | 32 | 55,613,534,647 | 55,664,522,231 | 1.00092x | 26.620 s | 906.059 ms |
| Kimi K3 | 128 | 142,599,838,039 | 142,749,520,535 | 1.00105x | 26.620 s | 906.095 ms |

Nemotron 只有 127 个实测 recurrent routing steps：GPU campaign 的第一个输出 token 属于
TTFT，随后才有 127 次 decode update。因此这里不复制最后一步来伪造 D128。S128
prefill 使用对应的实测活跃专家集合；decode top-6 顺序来自 S2048 campaign，并在报告
中明确标为跨 context 重用。Kimi 仍是 deterministic top-16 sensitivity，不能引用为
实测 Kimi routing 结果。

在 Nemotron D127 中，单独的 prefill A/B 为 1.05348x，decode A/B 为 1.02263x；组合后
是 1.02431x。Kimi D128 的对应结果只有 1.00105x，因为 2.8T 模型的 expert weight
traffic 几乎完全遮住片上优化。以上均为 `1 GHz / 64 B-cycle` 的 pre-RTL 参数化周期，
不是 RTL 测量值。

## DSE 得出的硬件 knee

下表给出只改一个参数时的 decode cycles 相对 4096/256/16-bank/64-FIFO/32-MiB
候选值；小于 1 表示更快。

| 改动 | Nemotron | Kimi | 判断 |
|---|---:|---:|---|
| Matrix 2048 MAC/cycle | 1.0176 | 1.0294 | 太窄会损失 1.8–2.9% |
| Matrix 8192 MAC/cycle | 0.9912 | 0.9853 | 资源翻倍只改善 0.9–1.5% |
| State lanes 128 MAC/cycle | 1.0017 | 1.0014 | 256 已接近 knee |
| State lanes 512 MAC/cycle | 0.9992 | 0.9993 | 加倍只改善约 0.1% |
| 32 banks | 0.99992 | 0.99999 | 16 banks 足够 |
| FIFO 128/256 | 1.0000 | 1.0000 | 64 values 足够 |
| 无 state cache | 1.0213 | 1.0011 | 32 MiB 对 Nemotron 更有价值 |
| state cache 64 MiB | 0.9886 | 0.9989 | Kimi state 太大，继续堆 SRAM 收益小 |

32 MiB 在 FP32 下固定驻留 Nemotron 15/23 个 state layer、Kimi 5/69 个 KDA layer；
其余 layer capacity-aware streaming。完整 48.16 MiB 与 433.41 MiB 是模型 state
总量，不是建议直接做成片上 SRAM。

## 消融：局部优化和整模收益必须分开

| 关闭的机制 | Nemotron slowdown | Kimi slowdown | 解释 |
|---|---:|---:|---|
| L-Compute consumer layout | 1.00054 | 1.000008 | 局部 bank stall 回来，但大部分被 HBM 隐藏 |
| State cache | 1.02132 | 1.00110 | Kimi 32 MiB 只能驻留少量 KDA layer |
| Fused layer activation flow | 1.00129 | 1.00004 | hidden bytes 远小于 expert weight bytes |
| Mamba B/C broadcast | 1.00029 | N/A | 只适用于 Mamba group-shared B/C |
| Projection direct bypass | 1.00000 | 1.00000 | 当前 64-value FIFO 没有 producer stall |

L-Compute 的局部结果仍是 Mamba `53,176 -> 14,904 cycles`、KDA
`536,544 -> 430,560 cycles`，且 bank stall 清零。整模收益很小并不推翻 mapping；
它说明论文必须同时解决 weight/MoE traffic，不能把局部 3.568x/1.246x 写成系统
加速比。

**上表只在 64 B/cycle 下成立，必须与下一节一起读。**

## 消融随 HBM 带宽反转

HBM 带宽是这套 DSE 里唯一决定大部分 decode 时间、却曾经被固定住的参数：64
B/cycle 时内存服务器 97% 忙，几乎吸收掉全部片上收益。只报一列会让读者得出
关于片上面积该投在哪里的相反结论，所以现在把两个承重消融在整个带宽范围重跑。

Nemotron 3：

| B/cycle | HBM 占用 | TPOT | 关闭 L-Compute | 关闭 state cache |
|---:|---:|---:|---:|---:|
| 64 | 97.75% | 46.265 ms | 0.0540% | 2.1320% |
| 256 | 91.56% | 12.350 ms | 0.2021% | 1.7254% |
| 512 | 84.44% | 6.698 ms | 0.3727% | 1.2573% |
| 1024 | 72.98% | 3.876 ms | 0.8026% | 0.6595% |
| 4096 | 39.99% | 1.770 ms | 2.1081% | 0.1017% |
| 8192 | 24.94% | 1.420 ms | 2.6950% | 0.0000% |

Kimi K3：

| B/cycle | HBM 占用 | TPOT | 关闭 L-Compute | 关闭 state cache |
|---:|---:|---:|---:|---:|
| 64 | 96.75% | 907.500 ms | 0.0008% | 0.1097% |
| 512 | 78.81% | 139.262 ms | 0.0055% | 0.0680% |
| 1024 | 65.03% | 84.395 ms | 0.1256% | 0.0362% |
| 8192 | 18.66% | 36.757 ms | 0.2883% | 0.0091% |

三条结论：

1. L-Compute 的整模收益随带宽单调上升，Nemotron 从 0.054% 到 2.695%（约 50x）。
   所以 `0.054%` 不能脱离 `64 B/cycle` 单独引用。
2. 但即使到 8 TB/s 量级，它也只有 2.695%（Nemotron）和 0.288%（Kimi）。
   "L-Compute 不是系统级胜负手"这个结论在整个带宽范围内成立，不是低带宽假设
   的产物。
3. 两个机制在 512 到 1024 B/cycle 之间换位：低带宽时省下 state 的 HBM 字节最
   值钱，高带宽时片上 bank 冲突才浮出水面。片上面积该投在哪里取决于带宽区间，
   这比任何单一 speedup 数字更适合作为论文结论。

Kimi 在整个范围内都受权重流量支配：8192 B/cycle 下单设备 TPOT 仍是 36.757 ms，
HBM 仍占 18.66%。单设备承载 2.8T 的结论与带宽假设无关。

完整数据在 `hybrid_system_dse_quick_v1.json` 的
`models.*.ablation.ablation_vs_hbm_bandwidth`。所有行只改内存服务器，compute、
SRAM、cache 容量和 routing 全部固定。这些仍是 analytic 消融比，不是 PLENA 对
GPU 的加速比；64 B/cycle 本身也是 RTL 前占位值，不是实测内存控制器。

Compact Matrix loop 的 ablation 单独报告 code size，不伪造 compute speedup。它减少
静态机器码与 host compilation cost，但不减少 GEMM MAC 或 weight bytes。

## Mixed precision

同一固定 32 MiB state cache 下，formal storage policy 是：

- Nemotron：BF16 activation、mixed NVFP4/BF16 weight、FP32 recurrent+conv state；
- Kimi：MXFP8 activation、MXFP4 weight、FP32 recurrent state、BF16 conv state。

降低 state 精度会让同一 SRAM 驻留更多 layer。CPU recurrence 在 2K/8K/32K token
下的输出 relative-L2 如下；state dimension 128 和真实公式保留，head/value parallel
维缩小，因为各 head 独立。

| 模型 / state storage | 2K | 8K | 32K | 容量压缩 |
|---|---:|---:|---:|---:|
| Mamba BF16 | 0.1019% | 0.1036% | 0.1043% | 2.00x |
| Mamba FP16 | 0.0126% | 0.0129% | 0.0130% | 2.00x |
| Mamba MX8-B128 | 2.0699% | 2.1016% | 2.1238% | 3.97x |
| KDA BF16 | 0.0218% | 0.0216% | 0.0218% | 2.00x |
| KDA FP16 | 0.0027% | 0.0027% | 0.0027% | 2.00x |
| KDA MX8-B128 | 0.3380% | 0.3461% | 0.3474% | 3.97x |

结果说明 Mamba MX8 风险明显高于 KDA，不能给两种 recurrence 强行用同一精度。
它仍不是 language-task accuracy；冻结 weight/state precision 前还要用真实 checkpoint
跑任务指标。

## ISA freeze

- `X_STATE=0x3D`：PRELOAD/RESET/PREFILL/STEP/COMMIT/EVICT/FENCE；
- `L_SCATTER_M=0x3F`：ROW/TRANSPOSE/MAMBA/KDA/CUSTOM；
- 两种 descriptor 都是 256 bytes、64-byte aligned，reserved bits 必须为 0；
- `0x39-0x3C` 只为 route extension 保留，当前没有实现；`0x3E` 空闲；
- malformed descriptor、address overflow、state hazard 和 unsupported precision 都有
  固定 completion status。

Compiler 的 `spec/hybrid_isa_freeze_v1.json` 固定两份子协议 SHA；Simulator 跨仓
测试会检查 opcode、golden bytes 与生成的 Rust 常量。

## 复现

```bash
just hybrid-system-dse --model all --grid quick \
  --context-length 2048 --decode-tokens 4 --prefill-tokens 128 \
  --json-out build/hybrid-system-dse.json

just hybrid-system-dse --model all --grid full \
  --context-length 2048 --decode-tokens 4 --prefill-tokens 128 \
  --json-out build/hybrid-system-dse-full.json

just hybrid-system-dse --model all --grid quick \
  --context-length 128 --decode-tokens 32 --prefill-tokens 128 \
  --json-out build/hybrid-s128-d32.json

just hybrid-system-dse --model nemotron3 --grid quick \
  --context-length 128 --decode-tokens 127 --prefill-tokens 128 \
  --json-out build/nemotron-s128-d127.json

just hybrid-system-dse --model kimi_k3 --grid quick \
  --context-length 128 --decode-tokens 128 --prefill-tokens 128 \
  --json-out build/kimi-s128-d128.json

uv run python -m analytic_models.performance.hybrid_system_dse \
  --model all --precision-error-tokens 2048 8192 32768 \
  --json-out build/hybrid-precision.json
```

Kimi routing trace 必须符合
`analytic_models/performance/profiles/kimi_k3_routing_v1.schema.json`，并覆盖每个连续
step 的全部 92 个 MoE 层。Loader 会拒绝错误 revision 元数据、错误 top-k、重复/越界
expert、缺层和不连续 step：

```bash
just hybrid-system-dse --model kimi_k3 --grid quick \
  --kimi-routing-trace /path/to/kimi-k3-routing.json \
  --json-out build/kimi-k3-empirical-routing-dse.json
```

`full` grid 扫描 3 种 Matrix、3 种 state lanes、2 种 bank、3 种 FIFO 和 6 种
state-cache 容量。报告用多目标 Pareto，不使用任意的“面积加权分数”。

## 仍需外部数据或 RTL 的部分

1. Kimi 完整模型的真实 top-16 routing trace；当前 deterministic routing 只用于敏感性。
2. 真实 checkpoint 的 mixed-weight/state benchmark accuracy；随机权重不能回答任务精度。
3. symbolic HBM manifest 绑定 checkpoint 后的整模 Rust 数值 replay。
4. RTL 的 SRAM macro、mux、频率、面积、功耗与真实 HBM efficiency。
5. A100/H100/H200/B200 的统一 TTFT/TPOT/token-J baseline；现有 B200 数据用于模型校验，
   不能和未综合的 PLENA cycles 直接比较。
