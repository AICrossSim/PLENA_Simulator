# Manager single_stream_block end-to-end report
_by tools/manager/_validate_block.py_

- **开始:** 2026-05-25 04:21:16
- **结束:** 2026-05-25 05:30:30
- **总墙钟时长:** 4153.6s (69.2 min)

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

## Model dims (derived)

- HEAD_COUNT = MLEN//HLEN*2 = 16
- H*D = HEAD*HLEN = 2048
- NUM_S_BLOCKS = 4, S = NSB*MLEN = 4096
- CONCAT_DIM = 2*H*D = 4096

## Results (GLOBAL/cumulative cosine vs ideal MX-roundtrip chain)

| step | kernel:tensor | cosine | NRMSE | status |
|---|---|---|---|---|
| 1 | layernorm:LN_Y | 0.999375 | 0.220% | OK |
| 2 | modulate:MOD_Y | 0.998851 | 0.261% | OK |
| 3 | linear_q:Q | 0.998006 | 0.763% | OK |
| 4 | linear_k:Kk | 0.998007 | 0.789% | OK |
| 5 | linear_v:V | 0.998006 | 0.763% | OK |
| 6 | linear_mlp:MLP | 0.998008 | 0.763% | OK |
| 7 | qknorm_q:QKN_Q | 0.997198 | 0.518% | OK |
| 8 | qknorm_k:QKN_K | 0.997201 | 0.501% | OK |
| 9 | rope_q:ROPE_Q | 0.996483 | 0.601% | OK |
| 10 | rope_k:ROPE_K | 0.996485 | 0.622% | OK |
| 11 | gelu:GELU | 0.997987 | 1.083% | OK |
| 12 | flash_attention:ATTN | 0.997645 | 1.014% | OK |
| 13 | concat:CONCAT | 0.997978 | 0.717% | OK |
| 14 | linear2:LIN2 | 0.997158 | 0.944% | OK |
| 15 | residual_gate:BLOCK_OUT | 0.996595 | 0.375% | OK |

**Verdict: ALL PASS** (threshold cosine >= 0.8).

## Wall-clock per kernel (seconds)

| step | kernel | compile | write(quant) | assemble | emulator | total |
|---|---|---|---|---|---|---|
| 1 | layernorm | 8.6 | 287.0 | 0.0 | 32.1 | 327.8 |
| 2 | modulate | 10.7 | 290.0 | 0.0 | 31.0 | 331.7 |
| 3 | linear_q | 6.9 | 212.2 | 0.0 | 43.4 | 262.5 |
| 4 | linear_k | 6.3 | 219.7 | 0.0 | 45.6 | 271.6 |
| 5 | linear_v | 6.4 | 214.3 | 0.0 | 43.8 | 264.5 |
| 6 | linear_mlp | 6.2 | 220.2 | 0.0 | 45.5 | 271.9 |
| 7 | qknorm_q | 7.1 | 144.0 | 0.0 | 34.2 | 185.3 |
| 8 | qknorm_k | 7.1 | 144.1 | 0.0 | 33.7 | 184.9 |
| 9 | rope_q | 6.3 | 314.1 | 0.0 | 41.2 | 361.7 |
| 10 | rope_k | 6.4 | 305.4 | 0.0 | 44.9 | 356.7 |
| 11 | gelu | 8.7 | 0.0 | 0.0 | 71.3 | 80.0 |
| 12 | flash_attention | 7.1 | 0.1 | 0.0 | 168.9 | 176.1 |
| 13 | concat | 5.7 | 0.0 | 0.0 | 25.9 | 31.6 |
| 14 | linear2 | 6.4 | 294.2 | 0.0 | 62.6 | 363.2 |
| 15 | residual_gate | 5.8 | 143.2 | 0.0 | 29.3 | 178.3 |
| | **TOTAL** | **105.8** | **2788.6** | **0.1** | **753.3** | **3647.8** |

- compile = subprocess into the compiler CLI;  write(quant) = MX-quantize + seek-write weights/fp_sram;  assemble = ISA→machine code;  emulator = the Rust sim run.

GLOBAL error: each golden uses the previous step's golden (ideal chain, MX-roundtrip per HBM hop); error accumulates kernel by kernel. BLOCK_OUT = end-to-end block error. Not local/per-kernel error.
