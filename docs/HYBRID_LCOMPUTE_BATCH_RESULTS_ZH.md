# Hybrid L-Compute 批量 Decode 结果

> **历史结果，已停止用于当前结论。** 本文对应早期 Vector/output-SRAM
> `L_CFG` 实验，不是当前 Matrix-SRAM `L_TILE` 路径。统一 BF16、公平 D' 对照、
> Compiler-to-Rust 数值证据和最新周期请只看
> [MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md](MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md)。

## 1. 这次实际跑了什么

这次没有使用 GPU，也没有把 B1 周期简单乘以 batch。Simulator 对每个点重新构建并调度：

- Nemotron 3：完整 52 层，包含 23 Mamba、23 MoE、6 GQA；
- Kimi K3：完整 93 层，包含 69 KDA、24 MLA、92 latent-MoE 和 1 dense FFN；
- `B=1/2/4/8/16`，context 2048，连续 decode 32 token；
- PLENA paper 候选点：`BLEN=32, MLEN=VLEN=2048`、1 GHz、1560 B/cycle HBM；
- 同一个 Matrix、Vector、HBM 和 banked output SRAM 时间轴。

Batch 的处理规则是：Matrix projection 对一批输入做 GEMM，因此每个被用到的权重张量只读取一次，MAC 数随 B 增长；每个请求拥有独立 state 和 KV，因此这些字节随 B 增长；Compiler 的单请求递推指令体执行 B 次。本轮没有假设尚未实现的跨请求 state packet 合并。

## 2. 为什么有两个结果边界

MoE 延迟取决于同一批请求选中了多少个不同专家。现有数据没有 Kimi 的真实 batch routing，因此报告两个可复现边界：

- `full_overlap`：所有请求选择相同 top-k 专家，权重复用最多；
- `maximum_distinct`：请求尽量选择不同专家，权重流量最大。

B1 时两者相同。B>1 的真实结果应落在两者之间；这不是误差条，而是 routing 未知造成的系统边界。

## 3. 最终路径相对 Arlo

下面比较同一 batch 下的 `J: affine packet + overlap` 和 `B: Arlo post-increment`。区间左端是专家最大分散，右端是专家完全重叠。

| Batch | Nemotron 3 | Kimi K3 |
|---:|---:|---:|
| 1 | 1.309x | 1.040x |
| 2 | 1.428–1.500x | 1.065–1.075x |
| 4 | 1.531–1.723x | 1.093–1.133x |
| 8 | 1.603–1.932x | 1.118–1.215x |
| 16 | **1.647–2.088x** | **1.137–1.313x** |

这组数字包含两部分：Arlo 普通地址优化到 stream addressing 的 Compiler 收益，以及 stream 到 affine multi-row packet 的架构收益，不能全部归因于斜存。

## 4. 只看 L-Compute 的净收益

更公平的架构比较是 `I: affine packet` 对 `E: best ordinary-row stream`。两者都已有 stream addressing，只改变是否把多个短 row 合成无冲突 packet。

| Batch | Nemotron 3 | Kimi K3 |
|---:|---:|---:|
| 1 | 1.135x | 1.015x |
| 2 | 1.187–1.218x | 1.024–1.028x |
| 4 | 1.231–1.315x | 1.034–1.049x |
| 8 | 1.262–1.405x | 1.044–1.080x |
| 16 | **1.281–1.473x** | **1.051–1.116x** |

物理 bank 排布的隔离实验是 `H: row-major packet` 对 `I: affine packet`。B16 时：

| Routing | Nemotron 3 | Kimi K3 |
|---|---:|---:|
| 最大分散 | 1.463x | 1.175x |
| 完全重叠 | 1.778x | 1.398x |

H 在 B16 分别产生 277,348,352 和 2,496,135,168 个 bank-conflict stall cycles；I/J 均为 0。H 是故意保留冲突的物理对照，不应作为实际软件基线。

## 5. TPOT 和吞吐代理

以下是 J 路径在 1 GHz 假设下的 Simulator 代理值，不是 RTL 或芯片实测。区间按数值最小到最大排列。

| 模型 | Batch | 每请求 TPOT | 聚合吞吐 |
|---|---:|---:|---:|
| Nemotron 3 | 1 | 2.401 ms | 416.6 tok/s |
| Nemotron 3 | 16 | 10.912–18.360 ms | 871.5–1466.2 tok/s |
| Kimi K3 | 1 | 94.474 ms | 10.58 tok/s |
| Kimi K3 | 16 | 194.571–442.932 ms | 36.1–82.2 tok/s |

Batch 提高聚合吞吐，但不会保持线性效率：B16 的 J 路径相对 B1 吞吐，Nemotron 为 2.09–3.52x，Kimi 为 3.41–7.77x。原因是所有请求仍共享一个 Vector/HBM 时间轴；最大分散 routing 还会增加 MoE 权重读取。

## 6. 结论和边界

Batch 让 Matrix 权重成本被更多 token 分摊，因此递推在整模中的占比上升，L-Compute 的收益也随 B 增长。Nemotron 的效果明显；Kimi 的收益较小且更依赖 routing，因为 92 个 latent-MoE 层会重新成为 HBM 主瓶颈。

本实验使用官方尺寸、真实层结构、现有 GPU 数据校验过的 FP32 recurrent-state 容量，以及 Compiler 发出的真实单请求递推指令统计；完整权重仍是 symbolic performance execution。绝对 TPOT 不能直接和 B200/5090 比，也不能写成硬件实测。

完整记录见 [`artifacts/hybrid_lcompute_paper2048_batch_v1`](../artifacts/hybrid_lcompute_paper2048_batch_v1/)。
