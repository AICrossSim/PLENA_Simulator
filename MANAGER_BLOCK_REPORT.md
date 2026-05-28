# Manager single_stream_block end-to-end report
_by tools/manager/_validate_block.py_

- **开始:** 2026-05-27 11:00:32
- **结束:** 2026-05-27 12:10:07
- **总墙钟时长:** 4174.2s (69.6 min)

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
| 1 | layernorm | 8.7 | 299.4 | 0.0 | 34.9 | 343.0 | 2910304.000ns |
| 2 | modulate | 6.2 | 289.1 | 0.0 | 32.0 | 327.3 | 1160366.000ns |
| 3 | linear_q | 8.6 | 223.0 | 0.0 | 48.3 | 279.8 | 2030584.000ns |
| 4 | linear_k | 9.1 | 218.8 | 0.0 | 46.9 | 274.9 | 2030584.000ns |
| 5 | linear_v | 7.1 | 219.6 | 0.0 | 49.1 | 275.8 | 2030584.000ns |
| 6 | linear_mlp | 6.8 | 218.2 | 0.0 | 45.7 | 270.7 | 2030584.000ns |
| 7 | qknorm_q | 8.3 | 146.1 | 0.0 | 37.8 | 192.2 | 7994092.000ns |
| 8 | qknorm_k | 8.6 | 145.4 | 0.0 | 35.7 | 189.7 | 7994092.000ns |
| 9 | rope_q | 7.0 | 313.5 | 0.0 | 43.5 | 364.0 | 2021822.000ns |
| 10 | rope_k | 6.9 | 306.8 | 0.0 | 42.2 | 355.9 | 2021822.000ns |
| 11 | gelu | 8.9 | 0.0 | 0.0 | 73.4 | 82.3 | 16531160.000ns |
| 12 | flash_attention | 7.8 | 0.0 | 0.0 | 180.2 | 188.0 | 40857266.000ns |
| 13 | concat | 11.4 | 0.0 | 0.0 | 28.9 | 40.3 | 1166936.000ns |
| 14 | linear2 | 7.5 | 290.8 | 0.0 | 62.8 | 361.1 | 2691430.000ns |
| 15 | residual_gate | 6.6 | 145.8 | 0.0 | 32.6 | 185.0 | 1160425.000ns |
| | **TOTAL** | **119.5** | **2816.2** | **0.1** | **794.1** | **3729.9** | |

- compile = subprocess into the compiler CLI;  write(quant) = MX-quantize + seek-write weights/fp_sram;  assemble = ISA→machine code;  emulator = the Rust sim run (wall-clock).
- **HW latency** = the modeled hardware latency the emulator reports (`Simulation completed. Latency ...`) — i.e. the simulated on-chip cycle/time cost of running that kernel, NOT wall-clock.

GLOBAL error: each golden uses the previous step's golden (ideal chain, MX-roundtrip per HBM hop); error accumulates kernel by kernel. BLOCK_OUT = end-to-end block error. Not local/per-kernel error.
