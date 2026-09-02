# Matrix SRAM L-Compute：Round-3 可复现实验结果

**FIX 4 结论：Outcome 2。** 公平地把 `tile_pitch_rows` 也开放给控制组后，
固定 `alpha=1, gamma=0` 配合 Compiler pitch 已同时达到 Nemotron 和 Kimi 的
bank floor；可编程 alpha 没有额外收益，所以它已经从公开 ISA 删除。
Nemotron/Kimi 的 pitch 分别为 2/4，完整 recurrence 均为 0 alias、0 容量开销。
旧的 `3.387x/1.713x` prefill 加速结论也因对照不公平而正式撤回。

## 1. 这轮实际修了什么

| 修复 | 问题 | 结果 |
|---|---|---|
| FIX 1 | Matrix column read 总是取 bank word 的 lane 0 | 现在按 `column % bank_width` 恢复 lane；宽度 1/4/32 都测 |
| FIX 2 | 32-bit 元素在 `u32 >> 32` 时静默损坏 | 32-bit 编解码使用独立四字节路径；`[0,1,2,3]` 原样往返 |
| FIX 3 | 旧测试只用 `bank_width=1`，隐藏 lane bug | 新增宽 bank column 测试和数值 `M_TMM/M_TMV` 测试 |
| FIX 4 | 处理组可选 pitch，对照组被钉死在 pitch 1 | 两边自由度对齐后重跑全部结果，并删除不必要的 alpha ISA 字段 |

这轮没有新增 cache、私有 state SRAM、MAC、SRAM 端口、队列或运行时调度器。

## 2. 最终机制

PLENA 保留固定的对角 bank 接线：

```text
physical_row = base_row + tile * tile_pitch_rows
             + logical_row * row_groups + word_group

bank = (physical_row + bank_word) mod bank_count
```

Compiler 只配置 view 的 shape 和 `tile_pitch_rows`。Matrix writeback 和后续
consumer 使用同一个 view；读取后恢复逻辑 lane 顺序。Mamba 与 KDA 的区别只在
shape 和 pitch，不在 opcode 或 decoder 中。

pitch 大于 1 并不浪费容量。递推 row 填入相邻 head tile 的“空位”：

| 模型 | pitch | physical rows | compact rows | 检查值数 | alias | 容量开销 |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron Mamba | 2 | 4096 | 4096 | 262,144 | 0 | 0% |
| Kimi KDA | 4 | 2048 | 2048 | 262,144 | 0 | 0% |

这些是把编号 bank word 真正放入物理坐标、再按同一 mapping 读回得到的数据，
不是只计算地址范围。

## 3. ISA 决定

```text
L_MVIEW.FULL   slot, shape_reg, map_reg
L_MVIEW.FIELD  slot, field, value_reg
<Matrix op>    ..., view=slot
<Vector op>.MV ..., operand_view_mask
```

`FULL/FIELD` 共用 opcode `0x3F`，用 `funct1=1/2` 区分。公开 descriptor：

```text
shape   = rows_minus_one[11:0]
        | cols_minus_one[23:12]
        | tile_count_minus_one[31:24]

mapping = tile_pitch_rows[15:0]
        | reserved_zero[27:16]
        | flags[31:28]
```

原来的 alpha 位 `[21:16]` 连同 `[27:22]` 一起保留为 0；Compiler 和 Rust
都会拒绝非零编码。ISA 不包含 Mamba/KDA 名称、递推公式、head 数、bank 数或
遍历程序。已有算术 opcode 不变，`.MV` 只是 operand 的 Matrix-view 寻址模式。

Assembler 使用 loop-aware must-dataflow 检查，view 配置必须支配每次动态使用；
view-qualified `M_MM_WO` 也不会再继承旧 `L_CFG` 的地址自增状态。

## 4. 公平实验定义

| 版本 | 唯一变化 | 可以记给谁 |
|---|---|---|
| A | 原始 lowering + 原始访问 | baseline |
| B | A + Arlo 地址/循环/指令压缩 | Compiler |
| C | 与 D 相同 multi-row 计算，固定 pitch 1 | L-Compute 控制组 |
| D-impl | 固定对角接线 + Compiler pitch | Matrix L-Compute |
| D-cf | pitch 与 per-view alpha/gamma 都可选 | 非架构化性能上界 |
| E | D-impl + 解析模型中的 producer/consumer overlap | overlap，单独记账 |

处理组自由度：shape、base、row、pitch、alpha、gamma。公平控制中，D-impl 与
D-cf 都能选 shape/base/row/pitch；D-cf 额外拥有 alpha/gamma。结果两者相同，
所以这两个额外自由度不应进入 ISA。

## 5. 真实 Compiler 地址的局部 bank 结果

配置为 `MLEN=2048, BLEN=32, 64 banks`。每个 source index 都作为不同数值
写入 Python 物理 bank cells，再用真实 Compiler 动态地址读回。这里“service
cycles”是 one-port-per-bank replay counter；不是整层公式，也不是 Rust cycle。

### 单 packet

| 模型 packet | C pitch 1 | D-impl | D-cf | C/D-impl |
|---|---:|---:|---:|---:|
| Nemotron：32 heads x 64 values | 2 cycles / 1 stall | 1 / 0 | 1 / 0 | 2.0x |
| Kimi：16 heads x 128 values | 4 cycles / 3 stalls | 1 / 0 | 1 / 0 | 4.0x |

### 单个官方尺寸 recurrence lowering

| 模型 | 检查值数 | C service/stall | D-impl service/stall | D-cf service/stall |
|---|---:|---:|---:|---:|
| Nemotron Mamba | 1,572,864 | 1,536 / 768 | 768 / 0 | 768 / 0 |
| Kimi KDA | 6,291,456 | 12,288 / 9,216 | 3,072 / 0 | 3,072 / 0 |

`D-impl == D-cf` 是删除 alpha 的直接证据。普通 GQA、MLA、MoE gate 的 row
和 column 访问在全部 64 个 allocation-base phase 上都逐值和逐周期一致：
每种 row case 检查 131,072 个值，每种 column case 检查 8,192 个值。

## 6. 52/93 层 Decode 时间线

模型结构：

- Nemotron 3 Nano：52 层 = 23 Mamba + 23 MoE + 6 GQA；
- Kimi K3：93 层 = 69 KDA + 24 MLA，另有 92 层 latent MoE + 1 dense FFN。

下面是**公式型串行解析时间线**：官方 shape、真实 GPU calibration、真实
Nemotron B1 routing 和 symbolic PLENA weights。它不是 Rust 用真实 checkpoint
从第一层跑到最后一层。

### B1、decode 1 token、官方 FP32 state

| 模型 | A | B | C pitch 1 | D-impl | D-cf | E |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron | 4,087,452 | 3,142,428 | 3,160,138 | 3,142,474 | 3,142,474 | 3,132,745 |
| Kimi | 105,011,094 | 98,168,502 | 98,804,544 | 98,168,640 | 98,168,640 | 97,890,432 |

| 模型 | Arlo `A/B` | 纯 L-Compute `C/D-impl` | alpha 上界 `D-impl/D-cf` | overlap `D-impl/E` |
|---|---:|---:|---:|---:|
| Nemotron | 1.30073x | 1.00562x | 1.00000x | 1.00311x |
| Kimi | 1.06970x | 1.00648x | 1.00000x | 1.00284x |

不能把 `A/B` 算给 L-Compute，也不能把 `D-impl/E` 算给 bank mapping。

### Batch sweep：官方 FP32，纯 `C/D-impl`

| Batch | Nemotron | Kimi |
|---:|---:|---:|
| 1 | 1.00562x | 1.00648x |
| 2 | 1.00714x | 1.01017x |
| 4 | 1.00825x | 1.01421x |
| 8 | 1.00895x | 1.01775x |
| 16 | 1.00935x | 1.02027x |

### HBM bandwidth sweep：官方 FP32，纯 `C/D-impl`

| HBM B/cycle | Nemotron | Kimi |
|---:|---:|---:|
| 64 | 1.00037x | 1.00029x |
| 256 | 1.00137x | 1.00116x |
| 512 | 1.00251x | 1.00228x |
| 1024 | 1.00427x | 1.00440x |
| 1560 | 1.00562x | 1.00648x |
| 8192 | 1.01106x | 1.02392x |

结论是：bank conflict 确实被消除，但整模主要在等 HBM、MoE 和其他层，因此
局部 2x/4x 不会自动变成大的整模提升。

BF16 state 仍只是精度候选，不是官方路径。其 B1 `C/D-impl` 解析上界为
Nemotron `1.01771x`、Kimi `1.01402x`；alpha 上界仍为 `1.0x`。

## 7. Prefill 结论已撤回

仍然成立的两条独立事实：

1. 旧 Compiler 的 KDA identity-GEMM census 为 13,891,534,848 logical MAC，
   当前 padding 后为 56,899,726,737,408 emitted MAC；
2. BF16/MX8 column view 用 0 transpose MAC 对拍了 16,384 个非对称编号值。

它们不是同一 prefill workload 的两次完整测量，因此不能拼成 speedup。旧的
S16 `3.387x` 和 S128 `1.713x` 已从 artifact 标记为 withdrawn；官方 FP32
不领取这项收益。

## 8. 结构开销边界

以下是 pre-RTL 结构代理，不是 PPA：

| 项目 | 结果 |
|---|---:|
| 新 SRAM payload | 0 bytes |
| cache/tag/replacement | 0 |
| 新 MAC lanes | 0 |
| 新 SRAM read/write ports per bank | 0 / 0 |
| 4 个 view records | 256 bits |
| 可编程 skew adders | 0 |
| 新 operand staging SRAM | 0 bytes；复用已有 Vector operand buffer |
| lane-restore payload proxy | 64 x 512-bit words, 6 stages |

没有 RTL 和综合，所以不能报告 LUT、门数、面积、频率、功耗、Token/J 或 PPA。

## 9. Round-3 triage

| 发现 | 处理 |
|---|---|
| Matrix view region 可重叠 | 延后：这是显式地址 SRAM，Compiler allocator 必须证明；本轮不声称全 state 同驻留 |
| `mark_pending_tiles(cells>1)` | 已修并加 two-tile test |
| legacy `write_delayed` 变 blocking | 延后：当前 Matrix-view 生产路径不用它；异步 DMA 走 pending/fill |
| legacy write 丢 dtype assertion | 已恢复并加负向测试 |
| `MatrixSram::new` 不能建 MLEN>64 | 已修：最多 64 banks，增宽 bank word；MLEN=2048 已测 |
| public alpha 无独立价值 | 已删除；mapping `[27:16]` 保留为 0 |
| 手写 assembly 可错配读写 descriptor | 延后：generated path 同 descriptor 已测；typed ownership pass 为后续工作 |
| dominance 只看文本顺序 | 已修为 loop-aware must-dataflow；`C_BREAK` 同时建 fallthrough 和 loop-exit 边 |
| view `M_MM_WO` 继承旧 auto-advance | 已修并加集成测试 |
| `.MV` 无法选 slot 3 | 有意限制：三位分别限定 dst/src1/src2；slot 3 留给显式 Matrix 操作 |
| assembler 绕过 canonical encoder | 已修为直接调用 contract helper |
| `0x3F/funct1=0` 仍是旧 `L_CFG` | 兼容保留，但不属于冻结的 Matrix-view contract |
| geometry test 只保持乘积 2048 | 已补 32/64/128/2048 四种总宽度 |
| prefill `3.387x/1.713x` | 已撤回 |
| `211,968 -> 0` 被误称数据证据 | 已改为每个真实动态地址的 numbered Python cell replay |
| ordinary no-regression 只测 base 0 | 已补全部 64 个 base phase |
| whole-model 默认串行 | 已明确标为 formula-based serial analytic timeline |
| gamma 让搜索看似 4096 点 | alpha/gamma 均不进 ISA；4096 只保留作固定接线审计 |

## 10. 证明边界

已经证明：

- Compiler 编码、canonical 校验、loop dominance 和真实 pitch lowering；
- Rust 物理 bank、row/column read、lane restore、view writeback；
- reduced-shape 四 token Mamba/KDA 数值递推；
- 官方 shape packet 的完整 numbered-value replay；
- 官方 52/93 层结构的解析时间线；
- ordinary Attention/MLA/MoE 不退化。

没有证明：

- Nemotron/Kimi 真实权重 first-to-last Rust 执行；
- RTL timing、综合、面积、功耗、PPA、Token/J；
- Kimi 真实 batch routing；
- BF16/FP16/MX8 state 的完整 checkpoint quality。

## 11. 验证结果

2026-09-02 在 Nix dev shell 中执行完整 pre-RTL 门禁，终端最终输出：

```text
Simulator Matrix/layout/campaign Python: 83 passed
Compiler Matrix L-Compute targeted:      119 passed, 1 warning
Rust workspace:                          279 passed, 0 failed
Compiler -> binary -> Rust connected:    max_abs_error=0, allclose=100%
Matrix-view connected counters:          1,041 packets, 65,664 values,
                                         service=ideal=1,041, bank_stall=0
JUST_TEST_MATRIX_LCOMPUTE_EXIT=0
```

两组 mutation test 也实际执行过：恢复错误的 column lane 索引会让 wide-bank
列读和数值转置测试失败；删除 FP32 直接打包路径会重新产生
`[0, 1, inf, NaN]`。恢复修复后上述门禁通过，因此测试不只是“没有报错”，
也能抓住本轮修复的两个原始错误。

仓库根目录的无选择 `pytest` 不是本门禁：Compiler 会因未安装的
`numpy/transformers/tvm` 在收集期产生 26 个错误；Simulator 根目录还会递归
收集旧 Compiler submodule、缺少 `aria_lm_ops`，并遇到历史同名 test module。
这些用例没有执行到本轮代码。可复现结论以项目显式维护的下列门禁为准。

## 12. 复现

```bash
cd /scratch/shared/mcl123/plena/review_20260828/simulator-static-kda-latest
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute \
  /scratch/shared/mcl123/plena/review_20260828/compiler-static-kda-latest

PLENA_COMPILER_ROOT=/scratch/shared/mcl123/plena/review_20260828/compiler-static-kda-latest \
UV_CACHE_DIR=/scratch/shared/mcl123/plena/cache/uv \
uv run --offline python -m analytic_models.performance.matrix_lcompute_campaign \
  --compiler-root /scratch/shared/mcl123/plena/review_20260828/compiler-static-kda-latest \
  --output-dir artifacts/matrix_lcompute_e2e_v1
```

机器可读结果：

- `campaign.json`：完整证据、triage 和边界；
- `ablation.csv`：全部 state mode、batch、token、bandwidth 与 variant；
- `headline.csv`：pitch-1、实现 co-layout 和 alpha upper bound；
- `state_residency.json`：状态容量、流量和精度边界。
