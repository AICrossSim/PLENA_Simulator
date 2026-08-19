# Projection Scatter / FIFO / Spill DSE

## 先说人话

Matrix 做完 Nemotron 3 的输入投影后，每个 token 产生 10,304 个 BF16 数：

```text
gate 4096 | x 4096 | B 1024 | C 1024 | dt 64
```

`x/B/C/dt` 马上由 X_STATE 使用，可以走小 FIFO 直接送过去。`gate` 要等
X_STATE 输出后由 Vector 做 gated RMSNorm，所以不能丢，必须写入 Vector SRAM。
如果 X_STATE 还在等 state，FIFO 中来不及消费的 state 字段也写入同一个
fallback record。这样快路径省 SRAM，慢路径仍然正确。

现在这条路径有可执行的 `L_SCATTER_M=0x3F`。Compiler 同时生成 256-byte layout
descriptor 和 `plena.projection_scatter/v1` debug view；Rust 先执行 L 指令完成物理
banked 写入，再允许匹配的 X_STATE 消费。X_STATE descriptor 不因 bank 数、FIFO
深度或 layout 改变，但完整 binary 中的 L mode 和 layout descriptor 会改变。

## Compiler 现在物理化了什么

每个 `PROJECTION_SCATTER` 包含：

- request/layer/token 范围和 ping-pong buffer index；
- fallback Vector-SRAM base 和 token stride；
- 每个字段的 source offset、producer、consumer 和 group shape；
- physical base row、bank 数、field offset 和 skew 公式；
- Matrix burst、FIFO 容量、spill bandwidth/policy；
- 对全部 logical -> `(row, bank)` 映射计算的 SHA-256。

Mamba 默认是 group-major/skewed。B200 Stage 2 又确认官方 KDA 的
`q/k/v/decay/beta` 来自独立 projection tensor，不存在官方 packed-QKV offset。
PLENA sidecar 因此显式把五份输出合成 per-head record。KDA 默认采用
group-major、`k` 固定旋转 8 banks；单值 `beta` 按 head/group 轮转 bank，避免
96 个 beta 在 producer burst 中全部写入 bank 0。row-major 仍保留作 ablation。
这些规则是 PLENA 每拍 `q[8]/k[8]/decay[8]` consumer packet 和 64-value producer
burst 的硬件选择，不是照抄 GPU thread order；它们由同一个 L opcode 的 mode 和
descriptor 表达，不为 Mamba/KDA 分别增加 opcode。

KDA 默认关闭直接 bypass：五个 projection 分时产生，而 recurrence packet 同时需要
q/k/decay，单一 source-order FIFO 不能在后两个字段尚未产生时正确直通。当前保守实现
使用恰好容纳一个 Matrix burst 的 64-value FIFO，并把结果 materialize 到 banked buffer；
未来若实现 field-aware reorder buffer，可以重新打开 bypass 做独立 ablation。

## Simulator 现在执行什么

Analytic model 按 Matrix source order 重放 burst，维护有限 FIFO，并把每段标为：

- `direct`：consumer ready，state 字段直接进入 X_STATE；
- `spill`：Vector 字段、consumer 未 ready，或关闭 bypass，写入 fallback SRAM。

它验证 source coverage、物理映射双射和 mapping hash，并输出：

- `direct_values`、`spill_values/bytes`；
- `fifo_high_watermark`、`fifo_stall_cycles`；
- scatter 写 bank stall；
- state/gate 读 bank stall；
- B/C SRAM reads 和 broadcast saved reads。

Rust transactional emulator 则执行真实 `banks × rows` 存储：重复写同一 cell、读取
未写 cell、source 遗漏/重复、descriptor identity 不匹配都会失败。它从 feature-tile-
major Vector SRAM 读取刚完成的 Matrix writeback，执行 `L_SCATTER_M`，然后 X_STATE
按同一份 consumer packet 读回。当前是显式 L pass，不是已经完成 RTL 的
`M_MM_WO` stream tap；但整个过程不访问 HBM 做 transpose/repack。

## 真实 Nemotron decode trace 的当前结果

条件：23 个真实 Mamba 层、B1、一个 decode token、BF16 activation、16 banks、
1 port/bank、64-value Matrix burst、64-value FIFO、16 values/cycle consumer。
这些是未做 RTL 校准的 transaction counters，不是最终 latency/PPA。

| Layout / flow | Spill B/layer/token | State read stall | Gate read stall | Scatter write stall |
|---|---:|---:|---:|---:|
| row-major / buffered | 20,608 | 896 | 768 | 0 |
| row-major / bypass | 8,192 | 0 | 768 | 0 |
| skewed / buffered | 20,608 | 0 | 0 | 4 |
| skewed / bypass | 8,192 | 0 | 0 | 0 |

因此目前能支持的结论是：

1. bypass 减少 60.3% projection SRAM materialization，但 gate 的 8 KiB 不能省；
2. skewed layout 消除这个 packet shape 的读冲突；
3. skewed buffered 写入有 4 cycles/layer 的小代价，论文必须一起报告；
4. 默认 Matrix 速率下 FIFO high-watermark 是 64；64 和 256 的周期相同，因此
   pre-RTL 默认值已冻结为 64 entries；
5. consumer 延迟超过单层 projection 时，bypass 退化为全 spill，但结果仍正确。

在当前 full-model analytic timing 里，buffered 和 bypass 的总 latency 相同：SRAM
写入被更慢的 projection compute/weight traffic 覆盖。因此现在能主张的是 traffic
和 conflict reduction，不能主张已得到端到端 speedup。全驻留 23 层 state 的容量
需求是精确值，不是扫描采样点：BF16 为 `23 × 1,097,728 B = 24.08 MiB`，FP32 为
`23 × 2,195,456 B = 48.16 MiB`。早前写的 32 / 64 MiB 是 `cache_mib` 网格
（0/16/24/32/48/64）跳过真实临界点后的最小全命中采样点，二者都高估约 33%，
不能作为 SRAM 容量结论。

## 排布是否"真的读对了"

`mapping_sha256` 只证明 Compiler 和 Simulator 对同一个映射达成一致，bank stall
只证明一个 packet **能**在一拍内服务完。两者都不能证明 consumer 拿到的那个数就是
projection 写进去的那个数。

`--verify-roundtrip` 补上这一级：producer value 与 consumer expected value 都从
独立 logical coordinate 生成；direct 值走 FIFO store，spill 值走 banked buffer，最后
按 packet lane 重组。重复/遗漏 source、重复写、读未写单元、direct/spill coverage 缺口
或 lane 串位都会直接失败。expected 不再由待测 mapping 自己生成。

```bash
just projection-scatter-verify build/nemotron3.lowered.json --roundtrip-tokens 2
```

23 层真实 decode trace，每层一个 token：

| Workload | Layout | 往返值数 | 最坏 bank 载荷 | service | ideal | 无冲突 |
|---|---|---:|---:|---:|---:|:--:|
| Nemotron Mamba-2 | row-major | 236,992 | 8 | 53,176 | 14,904 | ✗ |
| Nemotron Mamba-2 | group-major/skewed | 236,992 | 2 | 14,904 | 14,904 | ✓ |
| Kimi K3 KDA | row-major | 3,398,112 | 3 | 536,544 | 430,560 | ✗ |
| Kimi K3 KDA | group-major、无 rotation | 3,398,112 | 3 | 536,544 | 430,560 | ✗ |
| Kimi K3 KDA | group-major、k rotation=8 | 3,398,112 | **2** | **430,560** | 430,560 | **✓** |

五种配置都完整读回了写进去的全部值，因此这些 layout 都是正确的置换，差别只在服务
时间。两条新结论：

1. Mamba 的 skewed layout 在数据层面达到理想服务时间（`service == ideal`），不只是
   stall 计数为 0；每个 32 值 packet 恰好每 bank 2 个。
2. KDA 仅做 group-major 对齐没有收益；冲突来自 `q/k/decay` 三个 8-value tile 同落
   bank 0–7。对 256 个 `k/decay` rotation 组合做 packet sweep 后，87 个达到理论下限。
   当前选 `k=8, decay=0`，因为只旋转一个完整 8-value packet，地址生成最简单；它把
   consumer-read service `536,544 → 430,560` cycles（**-19.75%**），并通过全值
   roundtrip。完整 69 层重放还发现未轮转 beta 会增加 6,210 write-stall cycles；加入
   `beta group_stride=1` 后，默认 layout 的 read/write bank stall 都是 0。按单层
   read+write service 合计，`10,854 → 9,318` cycles（**-14.15%**）。这两个比例都不是
   KDA 整层 speedup。

正式 B200 六阶段 NCU 进一步给出了系统暴露比例。KDA 的
`qkv + gate/RMSNorm + out projection` 占 prefill、decode B1、decode B8 kernel time
的 74.33%、74.45%、62.25%，而 state core 只占 15.34%、5.02%、11.65%。把
14.15% buffer service reduction 极度乐观地套到**整个** Matrix 路径，Amdahl 上限也
只有 1.118x、1.118x、1.097x。真实值应更低，因为 rotation 不减少 GEMM MAC 和
weight read。这个上限已写入 `kimi_k3_kda_projection_rotation_dse.json`，用途是防止
把无冲突 buffer 的局部收益误报成整层加速，不是用 GPU 时间校准 PLENA。

## DSE 文件

- `artifacts/dse/l_scatter_m_v1_layout_ablation.json`
- `artifacts/dse/nemotron3_decode_projection_scatter_ablation.json`
- `artifacts/dse/nemotron3_prefill_s128_projection_scatter_ablation.json`
- `artifacts/dse/nemotron3_decode_state_stream_delay_ablation.json`
- `artifacts/dse/nemotron3_decode_bf16_gpu_validated_projection_scatter_ablation.json`
- `artifacts/dse/kimi_k3_kda_projection_rotation_dse.json`
- `analytic_models/performance/profiles/b200_kda_nemotron_campaign_complete.json`
- `artifacts/dse/nemotron3_formal_calibrated_dse.json`
- `artifacts/projection_scatter/nemotron3_decode_*_ready.report.json`
- `artifacts/projection_scatter/nemotron3_decode_*_state_stream.report.json`
- `artifacts/projection_scatter/*_roundtrip.report.json`（物理映射数据往返）

## 仍未证明的内容

- cycle model 未经 RTL counter 校准，不能把当前 us 当作芯片性能；
- FIFO/bank mux、地址生成、布线和 SRAM 端口的面积/频率代价尚无 PPA；
- GPU profile 说明 projection 占主要 kernel 时间，但不等于 PLENA 一定同样受限；
- 正式 B200 campaign 的 18 个 KDA NCU、6 个 Nemotron NCU、4 个 NSYS、latency、
  routing 和最终 raw archive 已完成哈希校验；它校准 workload 与 GPU baseline，仍不
  校准 PLENA cycle；
- 本机已有的 `KDA_B200_STAGE2_ingested_6fada27a` 是更早的 KDA core 原始子集；当前
  cross-check 已验证源码 revision、manifest 哈希、call count 和 core DRAM read 与正式
  汇总一致，但它只是早期 KDA core 子集，不替代完整 campaign；
- KDA rotation 已在 packet 计数和数据往返层验证，但仍缺 RTL mux/address-generation
  的频率与面积；19.75% 是 consumer-read reduction，14.15% 是 read+write buffer
  service reduction，都不是端到端加速；
- 数据往返证明的是映射正确和服务时间，不是 SRAM 端口的时序收敛；bank mux 和地址
  生成的频率代价仍要 FPGA 综合给出；
- MX8 activation 的 Vector-SRAM scale record 尚未定义，不能只把 byte/value 改成 1。

## Row / transpose / diagonal 的关键区别

对一个 `16 × 128` BF16 dense tile 做列读取：row-major 写入只要 128 cycles，但列读
要 2,048 cycles；纯 transpose 把数字反过来，写入 2,048、列读 128，总数仍是
2,176。`CUSTOM` diagonal mapping 用 `bank=(row+column)%16`，写和列读都是 128，
合计 256 cycles。也就是说论文 novelty 不能写成“transpose 本身提速”，正确说法是
Compiler 根据 producer 和 consumer 两边的 packet 共同选择 co-layout。
