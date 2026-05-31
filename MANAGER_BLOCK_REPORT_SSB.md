# Manager single_stream_block end-to-end report
_by tools/manager/_validate_block.py_

- **开始:** 2026-05-28 10:44:12
- **结束:** 2026-05-28 11:52:18
- **总墙钟时长:** 4086.6s (68.1 min)

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

## Per kernel: wall-clock (seconds) + hardware latency

| step | kernel | compile | write(quant) | assemble | emulator | total | HW latency |
|---|---|---|---|---|---|---|---|
| 1 | layernorm | 9.1 | 292.4 | 0.0 | 40.3 | 341.8 | 1733034.000ns |
| 2 | modulate | 6.9 | 281.0 | 0.0 | 33.8 | 321.7 | 978251.000ns |
| 3 | linear_q | 7.0 | 207.2 | 0.0 | 46.5 | 260.6 | 1471326.000ns |
| 4 | linear_k | 11.2 | 207.2 | 0.0 | 47.1 | 265.5 | 1471326.000ns |
| 5 | linear_v | 7.8 | 212.3 | 0.0 | 45.2 | 265.3 | 1471326.000ns |
| 6 | linear_mlp | 7.1 | 213.3 | 0.0 | 47.7 | 268.1 | 1471326.000ns |
| 7 | qknorm_q | 7.9 | 137.6 | 0.0 | 32.5 | 178.0 | 4148409.000ns |
| 8 | qknorm_k | 8.4 | 138.7 | 0.0 | 40.0 | 187.1 | 4148409.000ns |
| 9 | rope_q | 12.7 | 300.3 | 0.0 | 40.7 | 353.7 | 1348457.000ns |
| 10 | rope_k | 9.9 | 302.0 | 0.0 | 42.1 | 354.0 | 1348457.000ns |
| 11 | gelu | 10.9 | 0.0 | 0.0 | 67.9 | 78.9 | 7149755.000ns |
| 12 | flash_attention | 7.8 | 0.0 | 0.0 | 174.5 | 182.3 | 21945445.000ns |
| 13 | concat | 6.2 | 0.0 | 0.0 | 27.9 | 34.1 | 1164820.000ns |
| 14 | linear2 | 7.2 | 289.9 | 0.0 | 64.9 | 362.0 | 2130983.000ns |
| 15 | residual_gate | 8.7 | 138.7 | 0.0 | 30.5 | 177.9 | 978374.000ns |
| | **TOTAL** | **129.0** | **2720.4** | **0.0** | **781.6** | **3631.0** | |

- compile = subprocess into the compiler CLI;  write(quant) = MX-quantize + seek-write weights/fp_sram;  assemble = ISA→machine code;  emulator = the Rust sim run (wall-clock).
- **HW latency** = the modeled hardware latency the emulator reports (`Simulation completed. Latency ...`) — i.e. the simulated on-chip cycle/time cost of running that kernel, NOT wall-clock.

GLOBAL error: each golden uses the previous step's golden (ideal chain, MX-roundtrip per HBM hop); error accumulates kernel by kernel. BLOCK_OUT = end-to-end block error. Not local/per-kernel error.
