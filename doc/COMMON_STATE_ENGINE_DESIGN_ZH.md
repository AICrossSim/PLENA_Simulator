# PLENA Common State Engine: Mamba-2 + KDA

## 当前范围

这一版已经在 Compiler 和 Transactional Simulator 中实现并验证一个问题：
Nemotron 3 Mamba-2 和 Kimi K3 KDA 能否共用同一个 `X_STATE` contract、
recurrent-state datapath 和 banked head-tile SRAM。它仍然不是 RTL。Compiler 已能
生成完整 single-token decode 的 symbolic-weight 机器码：Nemotron 52 层为
6,202,993 条指令（23.663 MiB），Kimi 93 层真实 96-head 配置为 11,662,716 条
指令（44.490 MiB）。Top-K expert body 和 Matrix N/K tile 已使用硬件 loop；MLA 的
96 个 head body 仍静态发射。小规模 connected program 已在 Rust 数值对拍，但两份
整模产物尚未绑定真实 checkpoint，也没有从第一层到最后一层在 Rust 执行。
vision tower 不在本阶段范围。

## 为什么可以共用

| Workload | 每个 head 的 state | 默认存储 | 每 head 容量 |
|---|---:|---:|---:|
| Nemotron 3 Mamba-2 | 64 x 128 | config FP32 | 32 KiB |
| RTX 5090 standalone mixer path | 64 x 128 | BF16 | 16 KiB |
| B200 full-model vLLM path | 64 x 128 | FP32 | 32 KiB |
| Kimi K3 KDA | 128 x 128 | FP32 | 64 KiB |

两种算法的公式不同，但都需要：state decay、state-vector reduction、rank-1
outer-product update 和高精度累加。因此候选硬件每条 head lane 使用两个
head buffer ping-pong：计算当前 head 时预取下一个 head。第一版有两条 head
lane；按真实 KDA FP32 state 定容后，recurrent head-tile SRAM 合计 256 KiB。
KDA 的短卷积历史仍按实测 BF16 另行存储。

Mamba 可以在一遍 state traversal 中完成 update 和 C reduction。KDA 必须先
完成 `prediction = S @ k` 才知道 error，再执行 update/output，所以在片上
head tile 上读两遍；state 从 HBM 只读一次、最终只写一次。

## SRAM 排布修正

简单的 `bank=(row+column)%banks` 只能保证一条 row 或 column 分散，不能保证
一次读取完整 `4 x 8` tile 时没有冲突。当前候选采用二维 cyclic mapping：

```text
local_bank = (row % 4) * 8 + (column % 8)
```

32 个元素恰好进入 32 个 bank。粗粒度 row/column tile 决定每个 bank 内的
offset，因此没有复制 state。Compiler 仍看到正常二维 state；mapping 对 ISA
透明。

## RTX 5090 profile 对设计的修正

官方 fast path 的 decode B1 mixer median 是 234.5 us，其中 selective scan
只有 1.792 us；B8 分别为 245.0/6.272 us。NCU 又确认 decode 大约读取完整的
`1 MiB x batch` BF16 recurrent state。因此 GPU 上单独加速 scan 的 Amdahl
上限很小，论文不能只卖一个 state-update 单元。

完整 Nemotron NVTX stage export 进一步确认两个 projection 占 active GPU kernel time：
Prefill B1/S2048 为 84.91%，Decode B1 为 88.41%，Decode B8 为 74.25%。所以
`X_STATE` 不能成为 Matrix 前后的孤立加速器；Matrix projection、projection
packet、State Engine 和 out projection 必须形成连续 producer-consumer pipeline。

Kimi K3 单层 profile 又修正了一个旧假设：官方 recurrent state 是
`[B,96,128,128]` FP32，即 6 MiB/layer；三个 BF16 conv state 合计
0.28125 MiB/layer。69 层合计约 433.4 MiB/request。B200 Stage 2 已补齐完整 wrapper
对拍和 directional NCU counters：Decode B1/B8 recurrence 分别实测约
`6.46/51.20 MB` DRAM read，和一个/八个 FP32 state 的 logical read 一致量级。
B1 的 DRAM write 为 0，但 L2 write 为 9.53 MB，表示 dirty state 尚未逐出，不表示
logical state write 为 0。GPU physical traffic、logical tensor bytes 与 PLENA timing
始终分列，不能用 GPU 时间拟合 State Engine 周期。

数值证据分成两层：自定义 wrapper 与官方 `KimiDeltaAttention` 在同一个 FLA core 下
bit-exact，证明 projection、dtype、layout 和 cache glue 一致；FlashKDA 与 FLA 在
S=16/256/2048 的独立比较中最小 cosine 为 0.999982，最坏 output/state max-abs 分别为
0.000977/0.010416，只能主张通过预设误差阈值，不能称为逐元素相同。B200 FlashKDA
prefill 还使用了 `state_v_first=True` 和 BF16 beta 兼容路径，不是原封不动的 HF wrapper；
GPU baseline 必须披露这项适配。该 backend comparison 在已校验 tar 包内，但不属于
Stage 2 manifest 单独哈希的 18 个正式输出，因此在报告中标为补充证据。

官方 runtime hook 还确认 q/k/v、decay low-rank、decay、beta、output gate 和 output
projection 都是独立 tensor。Compiler 不再假设官方 packed QKV；它在可执行
`L_SCATTER_M` descriptor 中
构造 per-head record。KDA 的 q/k/decay packet 对 k 做 8-bank rotation 后达到理论
bank service 下限，单值 beta 再按 head 轮转 bank 以消除写冲突；数据往返 49,248
values 无遗漏或串位。由于这些字段分时产生，第一版默认全部写入 64-value FIFO 后的
banked buffer，不假设不成立的全直通。

这不等于 PLENA 上 state 不重要：当前 PLENA 候选 HBM 只有 64 B/cycle，远低于
RTX 5090。BF16 Mamba 在该假设下每层 state+conv logical read/write 为 2.09 MiB，
streaming 下限是 34.3 us。1 head lane 加两个 slots 已经到达这个下限；2 lanes
主要为 KDA 和 resident-state 加速保留，而不是为了 Mamba streaming。

更直接的硬件动作是加入 `Matrix-result -> L_SCATTER_M -> X_STATE` projection FIFO：
Matrix 仍计算 `gate/x/B/C/dt`，State Engine 仍只做 conv/recurrent，但能消费的
结果不再先完整写入 Vector SRAM 再读回；FIFO 满或消费者未就绪时才 spill。
当前 Rust 用显式 Vector-SRAM source pass 验证功能和 bank 周期；未来 RTL 可把同一
architectural command 融进 Matrix writeback。Ablation 保持 Matrix/Vector/X_STATE
不变，只替换 L mode 和 descriptor。

## Layout 隔离实验

下面的旧 FP32、1-head-lane 实验只用于隔离 row-major 与 dual-axis cyclic 的差异，
不是最终候选参数。两组实验使用 1 GHz、64 B/cycle HBM、32 FMA lanes、
4 x 8 state tile、32 个 single-port banks 和两个 head-tile slots。

| Workload | State source | Layout | us/layer | Bank stall |
|---|---|---|---:|---:|
| Mamba-2 | HBM stream | row-major | 68.6 | 49,152 |
| Mamba-2 | HBM stream | dual-axis cyclic | 68.6 | 0 |
| Mamba-2 | resident | row-major | 65.5 | 49,152 |
| Mamba-2 | resident | dual-axis cyclic | 32.8 | 0 |
| KDA FP32 | HBM stream | row-major | 393.2 | 294,912 |
| KDA FP32 | HBM stream | dual-axis cyclic | 205.8 | 0 |

Mamba streaming 时 layout 没改变总时间，因为 2 MiB read + 2 MiB write 已经
成为瓶颈；表中 4.2 MiB HBM traffic 还包含 conv state。KDA 每层读写合计约
12.56 MiB，且两遍片上 traversal 使 bank conflict 更严重，所以 layout 直接影响总时间。

## Provisional RTL candidate

新增的 common-state sweep 扫描了 1/2/4 head lanes、16/32/64 banks per lane、
1/2 head-tile slots。当前折中点是：

```text
2 head lanes
32 FMA lanes/head
32 single-port banks/head
4 x 8 dual-axis cyclic tile
2 head-tile slots/head
256 KiB recurrent head-tile SRAM total (sized for profiled FP32 KDA)
```

在未校准模型里，该点的 FP32 Mamba streaming latency 是 68.6 us/layer，
profile-observed BF16 Mamba 是 34.3 us/layer，FP32 KDA 是 205.8 us/layer。若 state
完全 resident，2-lane Mamba/KDA 分别为 16.4 和 73.7 us/layer。4 head lanes
对 streaming 没有继续改善，却把 FMA、
bank 和 head-tile SRAM 再翻倍；16 banks 在 streaming 时够用，但 KDA resident
会从 73.7 退化到 98.3 us/layer。因此先保留 32 banks。

这只是 provisional candidate，不是最终 RTL 参数。256 KiB 只包含 recurrent
ping-pong head tiles，不等于整个 persistent cache；Nemotron config-declared
FP32 请求约 48.16 MiB，本次 observed BF16 runtime 约 24.08 MiB，
完整 Kimi KDA state 约 433.4 MiB/request，必须把 cache 容量作为独立层级设计。最终选择仍需
FPGA synthesis 给出 bank mux、地址生成、URAM/BRAM 和频率代价。

## 目前不能得出的结论

1. 2 head lanes、32 banks/head 和 544 KiB transient SRAM 是第一版 RTL 输入，
   不是经过 FPGA PPA 证明后的最终最优点。
2. DSE 没有包含 bank mux、地址生成、布线和跨时钟 SRAM 的面积/频率代价。
3. Mamba B/C group broadcast 属于 projection/input path，不等同于 state SRAM
   mapping。
4. KDA chunked prefill 的 tile-16 matrix path尚未映射到 PLENA Matrix Engine。
5. 完整模型仍可能被 projection/MoE 权重流量限制，不能把 state-core speedup
   当成端到端 speedup。

## 已完成的软件验证

| 项目 | 当前结果 |
|---|---|
| ISA/descriptor | `X_STATE=0x3D`，common header + Mamba/KDA payload，Compiler/Rust 共用生成常量和 golden bytes |
| 数值边界 | State Engine 只做 conv + recurrent core；Matrix 做 projection，Vector 做 gate/norm |
| Mamba | tiny `STEP` 对齐 Python reference；`PREFILL` 最终 output/state 等于连续 `STEP` |
| KDA | prediction 后 error/update 的两遍顺序对齐 Python reference；`PREFILL` 等价性通过 |
| 状态生命周期 | streaming、preload/reset/commit/clean-evict、slot alias 和跨 context hazard 均有测试 |
| Mixed precision | FP32/BF16/FP16/MX8-B128 state 路径已建模；E4M3FN 进位边界有测试 |
| Timing | logical HBM bytes、cache hit/miss、SRAM values、bank stall、recurrent/estimated cycles 可逐命令输出 |
| Queue | 默认 blocking；`--state-async-queues` 支持 16 个 in-order queues、event dependency 和真实 `FENCE` |
| 真实 trace | Compiler 已生成 Nemotron 52 层和 Kimi 93 层、96-head symbolic decode 机器码；Matrix N/K 与 Top-K 已循环化，MLA head body 仍静态发射 |
| Projection path | Compiler 定义 blocked Matrix layout、bank mapping 和 FIFO/spill；Rust 用真实 banked buffer 完成数据往返并输出 service/stall counters |

当前 MX8 只完整支持 state/parameter storage。Vector SRAM 还没有 scale-aware
activation 接口，所以 descriptor 选择 MX8 activation 时 Simulator 会明确报
`UNSUPPORTED_PRECISION`，不会把缺失 scale 的字节当普通 FP8 使用。

## RTL 实现前后的剩余工作

1. Mamba 单 mixer profile 和 KDA B200 单 mixer 的数值、layout、DRAM/L2 traffic 已完成；
   完整 Kimi 端到端 GPU baseline 仍未完成，但不阻塞 pre-RTL 合约和 DSE。
2. Compiler 已生成完整 93 层 KDA/MLA/LatentMoE/AttnRes 的 96-head 合法机器码；
   compact MLA、MoE、AttnRes 和 KDA 已降低为现有 Matrix/Vector/State ISA 并在 Rust
   对拍。Matrix output-column/K-tile 和 Top-16 routed expert 已循环化；MLA wide-head
   loop 仍是减少 44.490 MiB 指令 footprint 的后续优化，不再是生成 artifact 的阻塞项。
3. decode descriptor 的物理行容量已冻结为 BLEN=4，prefill 为 chunk=16；这项是
   Matrix feature-tile blocked layout 与 Rust gather 共用的 ABI。禁止拿 prefill chunk
   给 decode 留行距，否则会制造 4--16 倍补零计算和地址空洞。
4. Rust transactional State Engine 已执行真实 `Matrix blocked layout -> 16-bank
   projection buffer -> X_STATE -> blocked Vector SRAM` 数据往返，并输出 packet、ideal、
   service 和 stall counters。Python debug view 继续负责跨 token FIFO/spill/bypass DSE。
5. 32 MiB KDA cache 只能固定 5/69 个状态；完整一请求需要 433.40625 MiB。因此
   FIFO=64 values、16 banks x 1 port 已有数据证据，KDA persistent-cache 容量还不能
   简单冻结成 32 MiB，必须按 FPGA 容量做 pinned/streaming 分区。
6. 第一版 RTL 的 activation 已明确只允许 BF16；MX8 activation 必须等
   scale-aware Vector SRAM 接口完成后才能加入。
7. RTL/PPA 尚未开始；bank mux、地址生成、布线、URAM/BRAM 和频率代价必须综合后
   回填 analytic model，当前任何 PLENA latency 都仍标成 uncalibrated。
