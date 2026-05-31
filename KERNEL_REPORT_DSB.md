# PLENA Kernel Report — double_stream_block (MLEN=1024, 2026-05-29)

## 1. Manager 端到端整 block (MLEN=1024, 2026-05-29, `gp_only_spill` allocator)

通过 `tools/manager` (图驱动 + HBM-bin 接力, 不拼 asm) 把 MMDiT
double_stream_block 的 **39 个 kernel 串成一条真实链**跑通: image (I) 与
text (T) 两条流各跑 9 个 pre-attention kernel → 3 个 concat 把 Q/K/V 沿
sequence 拼成 joint 序列 → 1 个 flash_attention 跑 joint → split_attn 拆回
两条流 → 每条流 8 个 post-attention kernel。每个 kernel 独立一次 emulator
调用, 上一个的 `hbm_dump.bin` 作为下一个 `--hbm` 输入; weight 与 fp_sram
在每个 kernel 跑前 just-in-time 写入; 几何全部从 `plena_settings.toml` 派生。
默认 allocator: **`gp_only_spill`** (pre-pass 决定每个 i32 value 终身住址
GP 或 IntRAM, 3 个 GP 专用作 scratch)。

跑法: `PLENA_ALLOC_MODE=gp_only_spill bash tools/manager/run.sh`
(run.sh 默认设 `gp_only_spill`, 默认入口 `_validate_double_block`)。原始
报告 [`MANAGER_BLOCK_REPORT_DSB.md`](MANAGER_BLOCK_REPORT_DSB.md); ISA 产物在
`managerbuild_DSB/ir/<kernel>/<kernel>.isa`。

**硬件配置 (`plena_settings.toml`, `MODE.active = "analytic"`, `DC_EN=1`):**

| 项 | 值 |
|---|---|
| MLEN | 1024 |
| HLEN | 128 |
| BLEN | 128 |
| VLEN | 1024 |
| BROADCAST_AMOUNT (= MLEN/HLEN) | 8 |
| HBM_WIDTH | 1024 bytes |
| MX 元素 / scale | E4M3 / E8M0 (各 8 bit) |
| block_size | 8 |

**模型维度 (派生):** HEAD = 16, HLEN = 128, HD = HEAD*HLEN = 2048,
NUM_S_BLOCKS = 2, S/stream = 2048, joint sequence S_joint = 2*S = 4096。

### 1.1 三种延迟数字 — 各代表什么

| 层 | 含义 | 工具 | 回答 |
|---|---|---|---|
| **Emulator HW latency** | 真实硬件总耗时, 模拟器报告 `Simulation completed. Latency ...` | `_validate_double_block.py` | "**真硅片上这段花多少 ns**" |
| **ISA 静态 cycle** | 把 .isa 每条被执行的指令的固定 cycle 加起来, 循环按 trip count 嵌套相乘 | `_cycle_analyze.py` | "**时间花在哪类指令上**" |
| **HBM transmission** | emulator − ISA 静态 = 真实搬运耗时 | 算 | "**搬运占多少**" — HBM DMA 在 ISA 里 cost=0 (异步), 但真硬件要花时间, emulator 把它建模进去 |

公式: `emulator HW latency = ISA 静态 cycle + HBM transmission`

ISA 静态层 opcode → cycle 表直接 transcribe 自
[`transactional_emulator/src/main.rs`](transactional_emulator/src/main.rs)
里所有 `cycle!()` 点 (analytic profile, dc_lib_en, 1 cycle = 1 ns):

- matmul M_MM/TMM/BMM/BTMM/MV/TMV = MLEN = 1024; drain M_*_WO & M_BMV/BTMV = 1
- V_ADD/SUB/MUL/EXP = 1, V_RECI = 2, V_RED_MAX = 4, V_RED_SUM = 8
- S_MAP_* (v↔fpram) = VLEN = 1024
- scalar fp/int (含 IntRAM LD/ST) = 1
- C_LOOP_START/END / C_SET_*_REG = 1
- **HBM DMA (H_PREFETCH/STORE) = 0** — 这就是为什么 transmission 在 ISA 层看不见, 必须由 emulator 实测暴露

### 1.2 精度结果 (GLOBAL/累积 cosine)

**⚠️ 下表 cosine 和 NRMSE 是 GLOBAL(累积)口径, 不是 local 单 kernel 误差。**
DSB 不为每个 kernel 算 golden, 而是只为 4 个 compare 节点算一条端到端的
fp32 golden 链 (`stream_pre → concat → flash → split → stream_post`,
见 [_validate_double_block.py:171-190](tools/manager/_validate_double_block.py#L171-L190));
这条 golden 链在每个 HBM 跳点都做一次 `mxr` (MX-E4M3 round-trip), 量化
误差沿 chain 累积。compare 时用同一节点 emulator 的 HBM 字节 vs golden 的
量化后值, 所以 cosine 反映的是**从输入到该节点累积的总误差**, 节点越靠后
误差越大: 0.9969 → 0.9894 → 0.9865 / 0.9858。IMG_OUT/TXT_OUT = 整个
double_stream_block 的端到端误差。

| kernel:tensor | global cosine | global NRMSE | status |
|---|---|---|---|
| concat_q:QJ | 0.996943 | 0.5707% | OK |
| flash:ATTNJ | 0.989356 | 1.0149% | OK |
| I_res2:**IMG_OUT** | **0.986494** | 0.7455% | OK |
| T_res2:**TXT_OUT** | **0.985772** | 0.5868% | OK |

**Verdict: ALL PASS** (阈值 cosine ≥ 0.8; 端到端 IMG_OUT 0.9865 /
TXT_OUT 0.9858, 误差纯 MX 量化逐 hop 累积)。Manager 实际只在这 4 个
关键节点 (joint 序列拼接前后 + 两条流端到端) 上跑 cosine 检验, 不在每个
kernel 都比对, 详见 `_validate_double_block.py` 的 `compare=` 配置。

### 1.3 逐 kernel 三层延迟

emulator HW latency 来自
[`MANAGER_BLOCK_REPORT_DSB.md`](MANAGER_BLOCK_REPORT_DSB.md); ISA 静态 cycle
由 [`tools/manager/_cycle_analyze.py`](tools/manager/_cycle_analyze.py) 计算;
transmission = emu − static。

| kernel | emulator (ns) | ISA 静态 (ns) | transmission (ns) | trans% |
|---|---|---|---|---|
| I_norm1 | 866,438 | 414,749 | 451,689 | 52.1% |
| I_mod1 | 489,136 | 37,409 | 451,727 | 92.4% |
| I_linq | 735,758 | 135,919 | 599,839 | 81.5% |
| I_link | 735,758 | 135,919 | 599,839 | 81.5% |
| I_linv | 735,758 | 135,919 | 599,839 | 81.5% |
| I_qknq | 2,074,234 | 1,696,941 | 377,293 | 18.2% |
| I_qknk | 2,074,234 | 1,696,941 | 377,293 | 18.2% |
| I_ropeq | 674,186 | 148,248 | 525,938 | 78.0% |
| I_ropek | 674,186 | 148,248 | 525,938 | 78.0% |
| T_norm1 | 866,430 | 414,761 | 451,669 | 52.1% |
| T_mod1 | 489,136 | 37,409 | 451,727 | 92.4% |
| T_linq | 735,758 | 135,919 | 599,839 | 81.5% |
| T_link | 735,758 | 135,919 | 599,839 | 81.5% |
| T_linv | 735,758 | 135,919 | 599,839 | 81.5% |
| T_qknq | 2,074,234 | 1,696,941 | 377,293 | 18.2% |
| T_qknk | 2,074,234 | 1,696,941 | 377,293 | 18.2% |
| T_ropeq | 674,186 | 148,248 | 525,938 | 78.0% |
| T_ropek | 674,186 | 148,248 | 525,938 | 78.0% |
| concat_q | 2,425,040 | 2,000 | 2,423,040 | 99.9% |
| concat_k | 2,425,040 | 2,000 | 2,423,040 | 99.9% |
| concat_v | 2,425,040 | 2,000 | 2,423,040 | 99.9% |
| **flash** | **21,945,445** | **20,155,344** | **1,790,101** | **8.2%** |
| split_attn | 2,424,955 | 2,000 | 2,422,955 | 99.9% |
| I_proj | 735,758 | 135,919 | 599,839 | 81.5% |
| I_res1 | 489,135 | 37,408 | 451,727 | 92.4% |
| I_norm2 | 866,430 | 414,761 | 451,669 | 52.1% |
| I_mod2 | 489,136 | 37,409 | 451,727 | 92.4% |
| I_mlpin | 735,758 | 135,919 | 599,839 | 81.5% |
| I_gelu | 3,574,881 | 3,272,032 | 302,849 | 8.5% |
| I_mlpout | 735,758 | 135,919 | 599,839 | 81.5% |
| I_res2 | 489,136 | 37,409 | 451,727 | 92.4% |
| T_proj | 735,758 | 135,919 | 599,839 | 81.5% |
| T_res1 | 489,136 | 37,409 | 451,727 | 92.4% |
| T_norm2 | 866,430 | 414,761 | 451,669 | 52.1% |
| T_mod2 | 489,136 | 37,409 | 451,727 | 92.4% |
| T_mlpin | 735,758 | 135,919 | 599,839 | 81.5% |
| T_gelu | 3,574,881 | 3,272,032 | 302,849 | 8.5% |
| T_mlpout | 735,758 | 135,919 | 599,839 | 81.5% |
| T_res2 | 489,136 | 37,409 | 451,727 | 92.4% |
| **整链** | **65,996,873** | **37,677,495** | **28,319,378** | **42.9%** |

> 每个 kernel 的 trans% 是其瓶颈类型指示: trans% 高 (concat/split_attn 99.9%,
> mod/res 92%, lin/rope 80%) = **HBM 搬运 bound**, allocator/ISA 改不动总耗时;
> trans% 低 (flash 8.2%, gelu 8.5%, qknorm 18.2%) = **指令执行 bound**, ISA 层
> 节省直接反映到总耗时。

### 1.4 逐 kernel ISA 指令类占比

每行的百分比是**该 kernel 内**各类占该 kernel 的 ISA 静态 cycle 的比例
(不是整链占比)。"–" = 该 kernel 不发该类指令。表按
`_validate_double_block.py` 实际调度顺序排列 (I pre-attn → T pre-attn →
concat ×3 → flash → split → I post-attn → T post-attn)。

| # | kernel | matmul | vector | scalar_fp | scalar_int/地址 | control | total (ns) |
|---|---|---|---|---|---|---|---|
| 1 | I_norm1 | – | 24.7% | 12.8% | 49.6% | 12.9% | 414,749 |
| 2 | I_mod1 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 3 | I_linq | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 4 | I_link | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 5 | I_linv | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 6 | I_qknq | – | 18.6% | 34.8% | 33.8% | 12.8% | 1,696,941 |
| 7 | I_qknk | – | 18.6% | 34.8% | 33.8% | 12.8% | 1,696,941 |
| 8 | I_ropeq | 2.8% | 30.4% | – | 58.4% | 8.4% | 148,248 |
| 9 | I_ropek | 2.8% | 30.4% | – | 58.4% | 8.4% | 148,248 |
| 10 | T_norm1 | – | 24.7% | 12.8% | 49.6% | 12.9% | 414,761 |
| 11 | T_mod1 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 12 | T_linq | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 13 | T_link | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 14 | T_linv | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 15 | T_qknq | – | 18.6% | 34.8% | 33.8% | 12.8% | 1,696,941 |
| 16 | T_qknk | – | 18.6% | 34.8% | 33.8% | 12.8% | 1,696,941 |
| 17 | T_ropeq | 2.8% | 30.4% | – | 58.4% | 8.4% | 148,248 |
| 18 | T_ropek | 2.8% | 30.4% | – | 58.4% | 8.4% | 148,248 |
| 19 | concat_q | – | – | – | 72.3% | 27.6% | 2,000 |
| 20 | concat_k | – | – | – | 72.3% | 27.6% | 2,000 |
| 21 | concat_v | – | – | – | 72.3% | 27.6% | 2,000 |
| 22 | flash | 10.6% | 21.3% | 40.0% | 22.7% | 5.4% | 20,155,344 |
| 23 | split_attn | – | – | – | 72.3% | 27.6% | 2,000 |
| 24 | I_proj | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 25 | I_res1 | – | 21.9% | – | 66.8% | 11.3% | 37,408 |
| 26 | I_norm2 | – | 24.7% | 12.8% | 49.6% | 12.9% | 414,761 |
| 27 | I_mod2 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 28 | I_mlpin | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 29 | I_gelu | – | 13.8% | 9.0% | 43.8% | 33.4% | 3,272,032 |
| 30 | I_mlpout | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 31 | I_res2 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 32 | T_proj | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 33 | T_res1 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 34 | T_norm2 | – | 24.7% | 12.8% | 49.6% | 12.9% | 414,761 |
| 35 | T_mod2 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |
| 36 | T_mlpin | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 37 | T_gelu | – | 13.8% | 9.0% | 43.8% | 33.4% | 3,272,032 |
| 38 | T_mlpout | 6.0% | 30.1% | – | 54.6% | 9.2% | 135,919 |
| 39 | T_res2 | – | 21.9% | – | 66.8% | 11.3% | 37,409 |

**I/T 对称性 (静态层):** 除了 norm 系列因 RMS-eps 等少量 immediate 不同
导致 cycle 差 ≤ 12 ns 以外 (I_norm1=414,749 vs T_norm1=414,761; I_res1=
37,408 vs T_res1=37,409), 其余所有 I/T 同名 kernel 的 ISA 静态 cycle 与
分类占比**完全 bit-identical** — gp_only_spill 在两条流上行为一致, 没有
非对称 bug。详见 [MANAGER_BLOCK_REPORT_DSB.md](MANAGER_BLOCK_REPORT_DSB.md)
里 emulator HW latency 列的同样对称模式。

### 1.5 整链汇总

**逐 kernel emulator HW latency 占整链 (按真实硬件耗时排序):**

| kernel | emulator (ns) | 占比 |
|---|---|---|
| **flash** | 21,945,445 | **33.3%** |
| **I_gelu** | 3,574,881 | **5.4%** |
| **T_gelu** | 3,574,881 | **5.4%** |
| concat_q / concat_k / concat_v | 2,425,040 (×3) | 3.7% each |
| split_attn | 2,424,955 | 3.7% |
| I_qknq / I_qknk / T_qknq / T_qknk | 2,074,234 (×4) | 3.1% each |
| I/T_norm1, I/T_norm2 (×4) | 866,430–866,438 | 1.3% each |
| I/T_linq/link/linv/proj/mlpin/mlpout (×12) | 735,758 | 1.1% each |
| I/T_ropeq/ropek (×4) | 674,186 | 1.0% each |
| I/T_mod1/mod2/res1/res2 (×8) | 489,135–489,136 | 0.7% each |
| **整链 emulator** | **65,996,873 (≈66.0 ms)** | **100%** |

**逐 kernel 静态 cycle 占整链 (按 ISA 静态总 cycle 排序):**

| kernel | static (ns) | 占整链 |
|---|---|---|
| **flash** | 20,155,344 | **53.5%** |
| **I_gelu** | 3,272,032 | **8.7%** |
| **T_gelu** | 3,272,032 | **8.7%** |
| I_qknq | 1,696,941 | 4.5% |
| I_qknk | 1,696,941 | 4.5% |
| T_qknq | 1,696,941 | 4.5% |
| T_qknk | 1,696,941 | 4.5% |
| I_norm2 | 414,761 | 1.1% |
| T_norm1 | 414,761 | 1.1% |
| T_norm2 | 414,761 | 1.1% |
| I_norm1 | 414,749 | 1.1% |
| I_ropek / I_ropeq / T_ropek / T_ropeq | 148,248 (×4) | 0.4% each |
| I/T_link/linq/linv/mlpin/mlpout/proj (×12) | 135,919 each | 0.4% each |
| I/T_mod1/mod2/res1/res2 (×8) | 37,408–37,409 each | 0.1% each |
| concat_q / concat_k / concat_v / split_attn | 2,000 (×4) | 0.0% each |
| **整链 ISA 静态** | **37,677,495 (≈37.7 ms)** | **100%** |

**整链 ISA 静态层各类指令占比:**

| 类 | cycles (ns) | 占 ISA 静态层 |
|---|---|---|
| scalar_int / 地址 (S_*_INT) | 11,996,384 | 31.8% |
| scalar_fp (S_*_FP) | 11,223,040 | 29.8% |
| vector (V_*) | 7,610,368 | 20.2% |
| control (C_*/loop) | 4,600,951 | 12.2% |
| matmul (M_*) | 2,246,752 | 6.0% |
| HBM DMA (H_*) / v↔fpram (S_MAP_*) | 0 | 0% (在 transmission 层) |

**整链三层总账:**

| 层 | ns | ms | 占总耗时 |
|---|---|---|---|
| ISA 静态 cycle | 37,677,495 | 37.7 | 57.1% |
| HBM transmission (emu − isa) | 28,319,378 | 28.3 | 42.9% |
| **Emulator HW latency (合计)** | **65,996,873** | **66.0** | **100%** |

**关键观察:**

- **flash_attention 单 kernel 占整链 ISA 静态 53.5%** (vs SSB 的 41.4%) —
  因为 DSB 里 joint 序列 S_joint=4096 是 SSB 序列长度的 1 倍, 但 attention
  cost 在 S_joint 上是二次的, flash 这一步比 SSB 时贵约 4 倍; 而 DSB 大部分
  非-attn kernel 是 SSB 同款几何下 ×2 (一份 I 一份 T), 增长只是 ×2 线性,
  所以 flash 的占比反而上升。
- **gelu ×2 合计 17.4%, qknorm ×4 合计 17.8%** — 这俩家族跟 flash 一起占
  整链 ISA 静态约 89%, 是后续 ISA 层优化的重点。其余 31 个 kernel 合计
  ~11%, 单个都 ≤ 1.1%, 不是热点。
- **scalar_int (31.8%) + scalar_fp (29.8%)** 各占静态层三成 — 地址循环
  (S_ADDI/S_ADD/S_SLLI) 在 MLEN=1024 的内循环被放大, scalar_fp (S_LD_FP/
  S_ST_FP) 主要来自 flash + qknorm 的 online-softmax / reduce 链。
  matmul 只占 6.0%, **整链不是 matmul-bound**, 跟 SSB 结论一致。
- **DSB 整链是"指令执行 略 > 搬运"**: 静态 57.1% vs transmission 42.9%。
  对比 SSB (静态 68.8% / transmission 31.2%), DSB 的 transmission 占比更高
  ——因为 concat ×3 + split_attn 这 4 个 kernel 几乎全是搬运 (各 trans% 99.9%),
  把整链 transmission 拉上来了, 这是 DSB 把两条流拼成 joint 序列再拆回去
  的固有代价。
- **每个 kernel 的 trans% 揭示瓶颈类型** (见 §1.3): trans% 高
  (concat/split/mod/res/lin) 是**搬运为主**, allocator/ISA 改不动总耗时;
  trans% 低 (flash 8.2%/gelu 8.5%/qknorm 18.2%) 是**指令执行 bound**, ISA
  层节省直接反映到总耗时。

### 1.6 墙钟 (工程参考, 与硬件 cycle 无关)

> **这层不是硬件 latency**, 是你在终端等了多久。主要被 Python 工具链和
> MX-quantize 拖累, 跟真硅片速度无关, 仅供工程进度参考。

整 block build+run 墙钟 ≈ **6296.2 s (104.9 min)** (报告头部)。逐 step
累加 (compile + write + asm + emu) ≈ **6214.4 s (103.6 min)**, 其中绝大
部分是 MX-quantize 写 weight/fp_sram (`write` 列合计 4682 s ≈ 75%), emulator
实跑 1247 s ≈ 20%, compile 286 s ≈ 5%, assemble 0 s。两者差值 (~82 s) 是
manager 自身的 cosine compare / golden 生成 / 启动等开销。详见
[`MANAGER_BLOCK_REPORT_DSB.md`](MANAGER_BLOCK_REPORT_DSB.md) 的逐 kernel
墙钟表。

DSB 比 SSB 墙钟贵 ~54% (6296 vs 4087 s), 主要在 write quant (4682 vs 2720 s),
反映 DSB 39 个 kernel 比 SSB 15 个多写 ~2.4× 次 weight; emulator 实跑
(1247 vs 782 s) 只多 60%, 因为 flash 那一步 emu 时间在两边等长 (joint
S=4096 在 emulator 里跑约 180-182 s)。

### 1.7 复现

```bash
# 在 PLENA_Simulator/ 下
PYTHONPATH=tools ./.venv/bin/python3 tools/manager/_cycle_analyze.py managerbuild_DSB/ir
```

输出包含每个 kernel 的 per-category cycles + top opcodes + 末尾的 PER-KERNEL
TOTAL 排序表与 CHAIN TOTAL。

## 2. 验证方法 (HBM-direct compare)

DSB 在 4 个关键节点跑 cosine 检验 (`concat_q:QJ`, `flash:ATTNJ`,
`I_res2:IMG_OUT`, `T_res2:TXT_OUT`), 都走 **HBM 直接比对** (读
`hbm_dump.bin` 的 MX-E4M3 字节, **不重排** —— 物理字节顺序即 golden flatten
顺序), 绕开 VRAM staging 的 stride 重排。Golden 两侧都做 MX-E4M3 round-trip
(输入侧 + 输出侧), 与 `create_mem_for_sim` / 模拟器写回的量化一致, 故残差
只剩 MX 量化噪声。

比对工具 `transactional_emulator/tools/check_mem.py` 现输出两段:
- **Correctness (scale-invariant)**: NRMSE / SNR / global+per-row cosine
- **Basic error reference**: MSE / MAE / max abs / allclose / relative

**为何只在 4 个节点 compare**: SSB 每个 kernel 都接一次 cosine, DSB 39 个
kernel 全比代价高且许多中间 tensor 是 stream-internal (per-stream Q/K/V/
LN/MLP 等), 真正端到端意义的节点只有: (1) joint 拼接成功 (`QJ`),
(2) joint flash 输出正确 (`ATTNJ`), (3) 两条流各自的 block 端到端
(`IMG_OUT`/`TXT_OUT`)。其余 kernel 的正确性已被这 4 个累积 cosine 间接覆盖
(任一中间 kernel 出错都会传到 `IMG_OUT`/`TXT_OUT`)。详见
[`_validate_double_block.py`](tools/manager/_validate_double_block.py) 的
`compare=` 配置。

---
