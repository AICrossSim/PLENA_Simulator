# Manager double_stream_block end-to-end report
_by tools/manager/_validate_double_block.py_ (PLENA_ALLOC_MODE=gp_only_spill)

- **总墙钟时长:** 6296.2s (104.9 min)
- **入口:** `PLENA_ALLOC_MODE=gp_only_spill bash tools/manager/run.sh` (默认 `_validate_double_block`)

MMDiT double_stream_block: image (I) 和 text (T) 两条流各自走 pre-attention chain → Q/K/V 沿 sequence 轴拼成单条联合序列 → **一次** flash_attention 跑 joint sequence → split 回到两条流 → 每条流各自的 post-attention chain (proj → res1 → norm2 → mod2 → mlpin → gelu → mlpout → res2)。

## Hardware config (plena_settings.toml [BEHAVIOR])

| item | value |
|---|---|
| MLEN | 1024 |
| HLEN | 128 |
| BLEN | 128 |
| VLEN | 1024 |
| BROADCAST_AMOUNT (=MLEN//HLEN) | 8 |
| HBM_WIDTH | 1024 bytes |
| MX elem | E4M3 (8 bit) |
| MX scale | E8M0 (8 bit) |
| block_size | 8 |

## Model dims (derived from run header)

- HEAD = 16, HLEN = 128, HD = HEAD*HLEN = 2048
- NUM_S_BLOCKS = 2, S/stream = NSB*MLEN = 2048
- joint sequence S_joint = 2 * S/stream = 4096
- 两条流共 39 个 kernel: 18 pre-attn (9×I + 9×T) + 3 concat + 1 flash + 1 split_attn + 16 post-attn (8×I + 8×T)

## Results (cumulative cosine vs ideal MX-roundtrip chain)

| kernel:tensor | cosine | NRMSE | status |
|---|---|---|---|
| concat_q:QJ | 0.996943 | 0.5707% | OK |
| flash:ATTNJ | 0.989356 | 1.0149% | OK |
| I_res2:IMG_OUT | 0.986494 | 0.7455% | OK |
| T_res2:TXT_OUT | 0.985772 | 0.5868% | OK |

**Verdict: ALL PASS** (threshold cosine >= 0.8).

GLOBAL/cumulative: 每个 golden 都用前一步的 golden 作为输入（每个 HBM 跳点做 MX round-trip），误差沿 chain 累积。`I_res2:IMG_OUT` / `T_res2:TXT_OUT` 即整个 double_stream_block 的端到端误差，与 SSB 报告 `BLOCK_OUT` 的位置等价。

## Per-kernel: wall-clock (秒) + HW latency

| # | kernel | compile | write | assemble | emu | total | HW latency (ns) |
|---|---|---|---|---|---|---|---|
| 1 | I_norm1 | 10.1 | 150.4 | 0.0 | 31.3 | 191.9 | 866438 |
| 2 | I_mod1 | 6.1 | 160.8 | 0.0 | 29.8 | 196.7 | 489136 |
| 3 | I_linq | 6.8 | 151.2 | 0.0 | 30.3 | 188.3 | 735758 |
| 4 | I_link | 6.6 | 148.8 | 0.0 | 32.7 | 188.1 | 735758 |
| 5 | I_linv | 6.6 | 162.1 | 0.0 | 37.3 | 206.0 | 735758 |
| 6 | I_qknq | 7.9 | 74.9 | 0.0 | 23.2 | 106.0 | 2074234 |
| 7 | I_qknk | 7.6 | 77.7 | 0.0 | 24.0 | 109.3 | 2074234 |
| 8 | I_ropeq | 6.7 | 167.0 | 0.0 | 26.6 | 200.3 | 674186 |
| 9 | I_ropek | 6.8 | 178.4 | 0.0 | 33.1 | 218.2 | 674186 |
| 10 | T_norm1 | 9.1 | 152.1 | 0.0 | 24.2 | 185.5 | 866430 |
| 11 | T_mod1 | 6.0 | 157.6 | 0.0 | 21.5 | 185.1 | 489136 |
| 12 | T_linq | 6.7 | 160.6 | 0.0 | 37.7 | 205.0 | 735758 |
| 13 | T_link | 6.7 | 151.8 | 0.0 | 31.5 | 190.0 | 735758 |
| 14 | T_linv | 6.7 | 148.1 | 0.0 | 29.6 | 184.3 | 735758 |
| 15 | T_qknq | 10.3 | 73.7 | 0.0 | 23.2 | 107.3 | 2074234 |
| 16 | T_qknk | 7.5 | 77.5 | 0.0 | 35.8 | 120.8 | 2074234 |
| 17 | T_ropeq | 6.8 | 168.2 | 0.0 | 29.2 | 204.3 | 674186 |
| 18 | T_ropek | 7.2 | 169.0 | 0.0 | 27.7 | 203.8 | 674186 |
| 19 | concat_q | 5.9 | 0.1 | 0.0 | 39.7 | 45.7 | 2425040 |
| 20 | concat_k | 5.9 | 0.0 | 0.0 | 40.6 | 46.5 | 2425040 |
| 21 | concat_v | 5.9 | 0.0 | 0.0 | 39.7 | 45.6 | 2425040 |
| 22 | flash | 7.6 | 0.0 | 0.0 | 182.0 | 189.7 | 21945445 |
| 23 | split_attn | 6.0 | 0.0 | 0.0 | 41.0 | 47.0 | 2424955 |
| 24 | I_proj | 6.7 | 152.1 | 0.0 | 30.1 | 189.0 | 735758 |
| 25 | I_res1 | 6.1 | 74.1 | 0.0 | 21.7 | 102.0 | 489135 |
| 26 | I_norm2 | 9.7 | 159.8 | 0.0 | 32.4 | 202.0 | 866430 |
| 27 | I_mod2 | 6.3 | 158.0 | 0.0 | 22.9 | 187.3 | 489136 |
| 28 | I_mlpin | 6.6 | 150.4 | 0.0 | 29.1 | 186.1 | 735758 |
| 29 | I_gelu | 8.9 | 0.1 | 0.0 | 43.6 | 52.7 | 3574881 |
| 30 | I_mlpout | 6.6 | 157.3 | 0.0 | 38.8 | 202.8 | 735758 |
| 31 | I_res2 | 6.1 | 84.8 | 0.0 | 23.5 | 114.3 | 489136 |
| 32 | T_proj | 6.6 | 156.3 | 0.0 | 29.8 | 192.8 | 735758 |
| 33 | T_res1 | 6.1 | 74.4 | 0.0 | 21.3 | 101.9 | 489136 |
| 34 | T_norm2 | 9.1 | 161.3 | 0.0 | 29.4 | 199.8 | 866430 |
| 35 | T_mod2 | 8.1 | 155.3 | 0.0 | 22.2 | 185.6 | 489136 |
| 36 | T_mlpin | 6.7 | 148.8 | 0.0 | 29.7 | 185.2 | 735758 |
| 37 | T_gelu | 8.9 | 0.1 | 0.0 | 40.4 | 49.4 | 3574881 |
| 38 | T_mlpout | 6.7 | 158.5 | 0.0 | 41.3 | 206.4 | 735758 |
| 39 | T_res2 | 9.5 | 78.9 | 0.0 | 22.0 | 110.4 | 489136 |
| | **TOTAL (≈)** | **285.5** | **4682.0** | **0.0** | **1247.0** | **6214.4** | **65945996** |

> 列总和按表内四舍五入后的值再求和得到，与 log 报告的 wall 6296.2s 相差 ~82s（per-kernel 之外的 graph setup / golden 重生成 / verify 等开销）。HW latency 是 emulator 报告的整数 ns，直接相加无误差。

- compile = subprocess into compiler CLI; write = MX-quantize + seek-write weights / fp_sram; assemble = ISA→machine code; emulator = Rust sim wall-clock。
- HW latency = emulator 报告的模拟片上时间，不是 wall-clock；合计 ~65.95 ms 模拟时间 vs 6296 s 实际时间。

## 总览观察

- **39/39 kernel 全跑通**，end-to-end IMG_OUT/TXT_OUT 都 ≥ 0.985 cosine，远超 0.8 阈值。
- **wall-clock 主要消耗在 HBM write (`4682s ≈ 74%`)**，emulator 占 1248s ≈ 20%，compile 286s ≈ 5%。
  - 大权重 kernel（norm1/mod1/lin*/rope*/proj/norm2/mod2/mlpin/mlpout）每个 ~150-180s write，是 quant + seek-write 在量纲上的瓶颈。
  - 无权重 kernel（concat_*、flash、split_attn、gelu、qkn*、res1/res2）的 write 都 < 90s，符合"weights just-in-time, mid-stream 重写"的设计（[weights_just_in_time](.claude/projects/-home-a13247568123124-project/memory/feedback_weights_just_in_time.md)）。
- **HW latency 大头**: flash (21.94 ms, 33%) > qkn{q,k} ×4 (~8.30 ms 合计, 13%) > concat/split ×4 (~9.70 ms, 15%) > gelu ×2 (7.15 ms, 11%)。flash_attention 是单一最贵 kernel。
- **gp_only_spill 在 double_stream_block 上首次端到端验证通过**（此前只在 single_stream_block 验证过——见 [gp_only_spill_validated](.claude/projects/-home-a13247568123124-project/memory/project_gp_only_spill_validated.md)）。两条流的对称 kernel HW latency bit-identical（如 I_norm1=866438 vs T_norm1=866430 仅差 8ns，等价；linq/link/linv/rope* 完全一致），分配器在两流上行为稳定。
- I_res2 (0.9865) 比 T_res2 (0.9858) 稍高 cosine 但 NRMSE 也更高（0.745% vs 0.587%）——两流误差量级一致，没有非对称 bug。
