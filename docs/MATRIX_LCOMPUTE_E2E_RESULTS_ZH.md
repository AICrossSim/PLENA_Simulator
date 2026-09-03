# Matrix SRAM L-Compute：修复后的结果与边界

## 1. 当前结论

这轮已经把 PLENA 路径统一为 BF16，并修复了上一轮评审指出的输出丢失、
Compiler/Rust 未联通、对照不公平、端口暗中加宽和测试假红等问题。

现在能够证明的是：

> Compiler 可以把 Nemotron Mamba-2 和 Kimi KDA 的递推核心编译成通用
> `L_TILE` 指令；这些机器码可由 Rust 通过真实 banked Matrix SRAM 连续执行
> 四个 token，并正确写回全部 state 和每个 head group 的 output。

现在不能证明的是“可编程 skew 比现有固定斜存更快”。最强固定对照 `D'`
已经能用固定对角接线和普通 tile 基址达到零冲突，因此 `D/D'` 的纯 bank
收益在两个模型上都是 `1.00x`。这一点是本轮最重要的纠正。

## 2. 数据路径

```text
prepared x/B/C/dt or q/k/v/decay/beta
                    +
              BF16 state in HBM
                    |
          viewed H_PREFETCH_V (0x29)
                    |
                    v
       existing 1 MiB BF16 Matrix SRAM
        fixed diagonal / affine view
                    |
       bank packet read + lane restore
                    |
                    v
       existing Vector mul/add/reduce
                    |
          viewed H_STORE_V (0x2a)
                    |
                    v
          BF16 state and output in HBM
```

Matrix SRAM 仍是 Compiler 显式管理的 scratchpad。没有 tag、命中、替换、
隐式一致性或运行时驻留决策，因此没有新增 cache。每个 head group 的 output
有独立 HBM 区间，后一组不会再覆盖前一组。

Mamba 的 `softplus`、`exp(dt*A)`，以及 KDA 的 decay/beta 系数准备仍在
上游 Vector 阶段；当前 `L_TILE` 只消费准备好的系数。KDA 使用官方公式
`decay = exp(lower_bound * sigmoid(rate * (gate + dt_bias)))` 和
`beta = sigmoid(beta_logit)`。Kimi B1 的 69 层合计 5,107,104 次普通逐元素
操作和 1,702,368 次指数操作，在 A/B/C/D/E 中都计为 4,485 个 Vector
周期；这些工作没有被偷偷算成零。Mamba 的 dt/exp 本来就是独立 stage，
同样没有被替换。

## 3. 精度和容量

本轮可执行 PLENA 配置统一为：

| 数据 | PLENA 格式 | B1 每层大小 |
|---|---:|---:|
| Nemotron Mamba state | BF16 | 1 MiB |
| Kimi KDA state | BF16 | 3 MiB |
| Matrix SRAM | BF16 | 1 MiB |
| Matrix bank word | 32 个 BF16 = 512 bit | - |

官方 GPU 路径使用 FP32 state，分别是 2 MiB 和 6 MiB/层。它只作为 GPU
baseline 与精度参考，不是当前 PLENA 执行格式。这样避免了上一轮用 BF16
SRAM 却按 FP32 一拍读取、等价于把端口暗中扩大到 1024 bit 的错误。

1 MiB SRAM 每次流式处理一个 head group：Nemotron 32 heads，Kimi 16 heads。
完整 state 不要求同时常驻；Compiler 用现有 HBM load/store 显式换组。

已有的独立长序列存储实验给出：Nemotron 在 S=32,768 时，BF16 state 相对
FP32 的 output/state 平均 relative-L2 分别是 `0.000312/0.001668`；Kimi 在
S=2,048 时分别是 `0.017061/0.017812`。这些是合成递推数值误差，不是完整
checkpoint 的语言质量或 benchmark accuracy；因此 BF16 是本轮可执行设计点，
不是已经完成的最终精度结论。

## 4. ISA

```text
L_TILE_CFG   slot, shape_reg, map_reg
L_TILE_EXEC  dst, src, scale, primitive[, axis_mask]
```

两种形式共用 opcode `0x3f`，并保留 `funct1=0` 的旧 `L_CFG` 编码。配置是
原子写入，不存在部分更新的隐藏状态。三个通用代数原语为：

| 原语 | 语义 |
|---|---|
| `SCALE_ACCUM` | `dst = a*dst + b*src` |
| `DOT_REDUCE` | 分段乘加归约 |
| `OUTER_UPDATE` | rank-1 外积更新 |

Mamba 使用前两个，KDA 使用三个。opcode、funct 和 decoder 中没有模型名、
固定 head 数或 cache 行为。

Matrix-view DMA 没有再占 opcode：它复用 `H_PREFETCH_V`/`H_STORE_V` 的
`0x29/0x2a`，bit 31 表示 viewed form，bits 30:29 选择 slot。旧机器码不变，
旧 DMA 的任何非零 precision selector 仍表示 KV；只有 viewed form 新增显式
BF16 state selector。

## 5. 公平的消融实验

| 版本 | 含义 | 可以归因给什么 |
|---|---|---|
| A | 原 PLENA 逐行 Vector 递推 | 基线 |
| B | Arlo 的地址/循环压缩 | Compiler 优化 |
| C | 单一 base phase 的固定 view | 可执行但受限的 descriptor 对照 |
| D' | 固定对角接线 + 每 tile 普通 base phase | 最强、零新 skew 硬件的 bank 对照 |
| D | compact affine descriptor + `L_TILE` | descriptor/分块/发射/溢出优化 |
| E | D + 容量合法的静态 overlap | overlap；当前为 0 |

公平原则是：对照组和处理组必须拥有相同容量、bank、端口、算术和合法物理
摆放自由度。`D'` 对 Nemotron 使用 `base_phase = 2*head`，对 Kimi 使用
`base_phase = 4*head`。它与 D 逐 cell 落在相同位置：

| 官方 BF16 state packet | D' service | D' stall | D/D' 纯 bank 加速 |
|---|---:|---:|---:|
| Nemotron，32 heads x 64 | 1 cycle | 0 | 1.00x |
| Kimi，16 heads x 128 | 1 cycle | 0 | 1.00x |

所以当前数据不支持“新增可编程 row/tile skew 是 bank-conflict novelty”。
真正成立的贡献候选是：通用 Matrix view + 多行 packet + `L_TILE` sequencer
把逐行递推压缩成紧凑执行。若论文一定要主张可编程 skew，必须先找到固定
对角加普通 base phase 无法实现的真实访问模式。

## 6. Compiler 到 Rust 的数值证据

四组测试都经过：

```text
Compiler lowering -> assembler -> canonical 32-bit words -> Rust decoder
-> BF16 banked Matrix SRAM -> explicit HBM state/output readback
```

| 模型 | layout | 机器码 | `L_TILE_EXEC` | state 比较值 | output 比较值 | state max abs | output rel-L2 | Rust cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nemotron | fixed | 33,360 | 1,040 | 524,288 | 16,384 | 0 | 0.00538 | 2,345,850 |
| Nemotron | affine | 1,200 | 32 | 524,288 | 16,384 | 0 | 0 | 2,307,301 |
| Kimi | fixed | 101,064 | 3,096 | 1,572,864 | 49,152 | 0.000977 | 0.00708 | 13,922,560 |
| Kimi | affine | 4,104 | 120 | 1,572,864 | 49,152 | 0 | 0 | 6,945,885 |

几何是官方递推尺寸，连续执行四个 token：Nemotron `64x128x64`，Kimi
`96x128x128`。输入不是全零路径，state 和 output 都检查。这里的 fixed/affine
周期差主要来自 descriptor 展开、指令数量和 Kimi 中间搬运；不能归因给
可编程 skew，因为公平的 bank-only D' 已经达到同一 bank 下界。

## 7. 官方尺寸单层解析回放

下面使用 Compiler 发出的完整地址和 service group，在 Python 中按 Rust
`L_TILE` 阶段顺序回放物理 bank。它不是 Rust 周期实测：

| 模型/版本 | 非 `L_TILE` issue | ideal service | service | stall | Vector 算术 | 局部周期 |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron C | 8,080 | 2,588 | 2,844 | 256 | 1,292 | 12,216 |
| Nemotron D | 292 | 2,096 | 2,096 | 0 | 1,292 | 3,680 |
| Kimi C | 24,492 | 13,878 | 13,878 | 0 | 6,930 | 45,300 |
| Kimi D | 996 | 10,854 | 10,854 | 0 | 6,930 | 18,780 |

这张表说明 compact view 可以显著减少发行和分块，但并不改变上一节的
`D/D'=1.00x` bank 结论。

## 8. 52/93 层 decode 时间线

Headline point：`MLEN=2048`、`BLEN=32`、64 banks、1 MiB BF16 Matrix SRAM、
HBM 1560 B/cycle。B1 结果：

| 模型 | A | B | C | D | D/A | D/B | D/C |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,210,882 | 2,014,554 | 2.0129x | 1.5438x | 1.0975x |
| Kimi K3 | 103,816,704 | 97,013,856 | 93,286,200 | 91,178,043 | 1.1386x | 1.0640x | 1.0231x |

这些是公式时间线，不是整模 Rust 执行。它使用官方 52/93 层结构、官方维度、
GPU calibration、Nemotron 实测 routing（可用处）和 symbolic PLENA weights。
只有 23 个 Mamba / 69 个 KDA 层在 schedule 中执行真实 `L_TILE` lowering；
Attention、MLA、MoE、dense 等仍是解析模型 stage。

其中 A/B 是“每发一条动态指令计一拍”的发行周期代理，并没有在 Rust 中执行；
C/D 才另外加入 Matrix service、算术和 HBM 项。因此 `D/A`、`D/B` 的主要来源是
多行利用率和指令压缩，不能叫可编程斜存本身的加速。

### C 到 D 的归因（B1）

| 模型 | 总节省 | HBM | issue | ideal service | bank stall | 只消 stall 的整模加速 |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron | 196,328 | 0 | 179,124 | 11,316 | 5,888 | 1.00267x |
| Kimi | 2,108,157 | 278,277 | 1,621,224 | 208,656 | 0 | 1.00000x |

`D/C` 不能叫“斜存硬件加速”：C 是受限的 single-base descriptor，D 同时减少
chunk、issue、ideal service 和 Kimi spill。纯 bank 比较必须看 D'，结果 1.00x。

## 9. Batch DSE

| Batch | Nem D/A | Nem D/B | Nem D/C | Kimi D/A | Kimi D/B | Kimi D/C |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.013x | 1.544x | 1.097x | 1.139x | 1.064x | 1.023x |
| 2 | 2.516x | 1.814x | 1.146x | 1.228x | 1.105x | 1.038x |
| 4 | 3.016x | 2.082x | 1.194x | 1.335x | 1.155x | 1.056x |
| 8 | 3.415x | 2.296x | 1.232x | 1.438x | 1.202x | 1.073x |
| 16 | 3.679x | 2.439x | 1.258x | 1.518x | 1.239x | 1.086x |
| 32 | 4.681x | 2.976x | 1.354x | 1.570x | 1.263x | 1.095x |
| 64 | 6.546x | 3.978x | 1.534x | 1.666x | 1.308x | 1.111x |

Nemotron B1 使用已有真实 routing。更大 batch 和 Kimi routing 仍使用 DSE
边界，不是 GPU 实测 throughput，不能直接写成真实批量性能。

## 10. HBM 敏感性（D 相对 B，B1）

| HBM B/cycle | Nemotron | Kimi |
|---:|---:|---:|
| 64 | 1.0239x | 1.0027x |
| 256 | 1.0947x | 1.0108x |
| 512 | 1.1871x | 1.0215x |
| 1024 | 1.3656x | 1.0425x |
| 1560 | 1.5438x | 1.0640x |
| 8192 | 3.2115x | 1.2945x |

Kimi 的完整时间线被巨大的 projection/MoE 权重流量支配，所以单独缩短 KDA
递推无法变成很大的整模收益。这个带宽边界必须和 headline 一起报告。

## 11. Overlap、Prefill 和资源边界

1 MiB 点无法同时容纳当前 working set 和下一组 state：Nemotron 至少还需要
45,312 bytes，Kimi 至少还需要 28,736 bytes。因此 E=D，没有虚构 overlap。

现在已有两条完整的 transactional S128 功能路径：Mamba-2 连续执行两个
64-token SSD chunk，KDA 连续执行八个 16-token chunk。两者都从 BF16 HBM
输入开始，在 Rust 中执行全部 chunk，读回 128 个输出和最终 state；共比较
12,288 个值，最大绝对误差均为 `0.0009765625`，allclose 通过率 100%，
Matrix-view bank stall 均为 0。对应周期分别为 `188,638` 和 `1,346,121`。

这些测试采用一头、64-wide 的缩小外围几何，证明 chunk 计算与 state 生命周期，
但没有 row-major/affine A/B 对照，因此仍不报告 transactional prefill 加速或
整模 TTFT。旧的 `3.387x/1.713x` 继续撤回。

当前只有 pre-RTL 结构代理：额外 SRAM payload/cache/new MAC/额外 SRAM 端口
均为 0；4 个 view record 共 256 bits；sequencer 上界 256 bits 加 3 个计数器；
最坏情况的 segment broadcast 是 `32x16` bits（Nemotron 为 32 个 64-value
segment，Kimi 为 16 个 128-value segment）；cyclic restore 是 64 个 512-bit bank word、
6 级 mux。可编程 bank select 上界是 128 个 6-bit 加法器，但当前 bank 数据
并未证明这些加法器值得保留。

没有 RTL 与综合，所以不能给 LUT、面积、频率、功耗、PPA、Token/J 或相对 GPU
的硬件加速比。

## 12. 已证明与未证明

已证明：

- Compiler、assembler 和 Rust decoder 的 `L_TILE` 契约一致；
- Rust 物理 bank、行/列读取、lane restore、viewed DMA 和 output 写回正确；
- 官方递推几何下四 token 的 Mamba/KDA 机器码数值对拍；
- 缩小外围几何下 Mamba/KDA 的完整 S128 transactional prefill 数值与最终 state；
- 公开 Mamba2-130M checkpoint 的 24 层真实权重连续 hidden 链，其中每层递推
  核心由 Rust `L_TILE` 执行；
- 52/93 层中全部 recurrent layer 都发出合法 `L_TILE`；
- D' 与 D 的物理 state 坐标相同，纯 bank speedup 是 1.00x；
- 普通 Attention/MLA/MoE row/column service 在全部 64 个 base phase 不退化；
- ANSI 日志不会再让复合门禁出现“计算通过但解析失败”。

未证明：

- Nemotron/Kimi 真实 checkpoint 的全算子、第一层到最后一层 Rust 数值执行；
- transactional prefill 的 A/B 加速和完整模型 TTFT；
- 1 MiB 下 producer/consumer overlap；
- view DMA/L_TILE 在 scoreboard 中按精确物理 bank-word 足迹进行的重叠时序；
  当前使用保守逻辑区间，物理 `Cell::Pending` 保证数值正确，本轮也未给 E
  任何 overlap 加速；
- 可编程 skew 相对最强固定斜存的性能收益；
- RTL timing、PPA、能耗或相对 GPU 的最终硬件性能。

## 13. 复现

```bash
cd /scratch/shared/mcl123/plena/review_20260828/simulator-static-kda-latest

nix develop --no-write-lock-file --command \
  just test-matrix-lcompute \
  /scratch/shared/mcl123/plena/review_20260828/compiler-static-kda-latest
```

机器可读结果：

- `artifacts/matrix_lcompute_connected_bf16/summary.json`
- `artifacts/matrix_lcompute_e2e_v5/campaign.json`
- `artifacts/matrix_lcompute_e2e_v5/headline.csv`
- `artifacts/matrix_lcompute_e2e_v5/c_to_d_attribution.csv`
- `artifacts/matrix_lcompute_e2e_v5/ablation.csv`
