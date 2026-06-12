# PLENA Analytic Report — final_layer (MMDiT LastLayer)

_Geometry: MLEN=1024, HLEN=128, **BLEN=8**, VLEN=1024 (MAC array = 8×1024 = 8,192); freq=1.0 GHz_

_Power model: bottom-up 7nm (BF16→FP32): MAC=1 mW, vector lane=0.5 mW, SRAM=0.055 pJ/bit_

_Cycle formula: George performance customISA (pipelined; M_BMM=131072, M_MM=8)_

The MMDiT `final_layer` (`LastLayer(hidden=3072, patch_size=1, out_channels=64)`):

```
shift, scale = adaLN_modulation(vec).chunk(2)   # SiLU + Linear(HD -> 2·HD), on vec only
x = (1 + scale) · norm_final(x) + shift          # LayerNorm(no affine) + modulate
x = linear(x)                                     # Linear(HD=3072 -> patch²·out_ch = 64)
```

Compiled as 5 PLENA kernels: `adaln_shift` + `adaln_scale` (modulation_gen, silu(vec)@W), `norm_final` (LayerNorm with scale=1/bias=0), `modulate` ((1+scale)·x+shift), `linear_out` (HD→64, output padded to MLEN). The output dimension (64) is padded up to one MLEN block, so the final linear lowers to M_BTMM over a 1024-wide tile of which only 64 columns are live — this padding is why the final layer is matmul-dominated despite its tiny true FLOP count.

This block runs **once per MMDiT pass**, versus 38 SSB + 19 DSB; it is ~0.03% of the pass latency.

## 1. Unit power (W, steady-state @100% utilisation)

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **8.2 W** | 1 mW/MAC × 8,192 MAC |
| Vector unit (1 thread) | 0.512 W | 0.5 mW/lane × 1024 lanes |
| Matrix SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |
| Vector SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |

## 2. Chain latency & energy vs SIMT (batch = 1)

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 4.45 | 0.0368 | 8.3 |
| 2 | 4.01 | 0.0364 | 9.1 |
| 4 | 3.78 | 0.0362 | 9.6 |
| 8 | 3.67 | 0.0361 | 9.8 |
| 16 | 3.62 | 0.0361 | 10.0 |
| 32 | 3.59 | 0.0361 | 10.0 |

### 2.1 Batch = 4 (full-GPU shape)

Latency and energy scale linearly with batch; average power unchanged.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 17.81 | 0.1472 | 8.3 |
| 2 | 16.05 | 0.1456 | 9.1 |
| 4 | 15.12 | 0.1448 | 9.6 |
| 8 | 14.69 | 0.1444 | 9.8 |
| 16 | 14.47 | 0.1442 | 10.0 |
| 32 | 14.36 | 0.1442 | 10.0 |

### 2.2 Batch = 4, device count = 12 (full deployment)

Twelve devices run the batch-4 workload in parallel (ideal linear scaling): latency ÷12, total energy unchanged, average power ×12.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 1.48 | 0.1472 | 93.6 |
| 2 | 1.34 | 0.1456 | 105.6 |
| 4 | 1.26 | 0.1448 | 112.8 |
| 8 | 1.22 | 0.1444 | 117.6 |
| 16 | 1.21 | 0.1442 | 120.0 |
| 32 | 1.20 | 0.1442 | 120.0 |

## 3. Per-kernel ISA cycles (raw, pre-SIMT, batch=1)

| Kernel | matmul | vector | scalarFP | #H instr | compute µs | mem µs | bound |
|---|---|---|---|---|---|---|---|
| linear_out | 3,538,953 | 92,160 | 0 | 207 | 3631.1 | 1.53 | comp |
| norm_final | 0 | 691,200 | 46,080 | 432 | 737.3 | 1.08 | comp |
| modulate | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| adaln_scale | 11,520 | 3,165 | 0 | 153 | 14.7 | 0.72 | comp |
| adaln_shift | 11,520 | 3,165 | 0 | 153 | 14.7 | 0.72 | comp |

The `linear_out` matmul (HD→64, padded to 1024) dominates at 82%; `norm_final` is the next largest. The two `adaln_*` modulation kernels act on a single vec row and are negligible (~0.015 ms each).

## 4.1 Per-ISA-opcode cycle share — SIMT-1

_Total effective compute = 4,453,059 cycles (4.45 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 3,538,944 | 79.5% |
| V_RED_SUM | vector | 442,368 | 9.9% |
| V_ADD_VF | vector | 135,222 | 3.0% |
| V_ADD_VV | vector | 129,042 | 2.9% |
| V_MUL_VV | vector | 82,962 | 1.9% |
| V_MUL_VF | vector | 27,672 | 0.6% |
| V_SUB_VF | vector | 27,648 | 0.6% |
| M_MV | matmul | 20,736 | 0.5% |
| S_MUL_FP | scalar | 18,432 | 0.4% |
| S_ADD_FP / S_SQRT_FP / S_RECI_FP | scalar | 9,216 each | 0.2% each |
| M_MV_WO | matmul | 2,304 | 0.1% |

## 4.2 Per-ISA-opcode cycle share — SIMT-16

_Total effective compute = 3,617,685 cycles (3.62 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 3,538,944 | 97.8% |
| V_RED_SUM | vector | 27,648 | 0.8% |
| M_MV | matmul | 20,736 | 0.6% |
| V_ADD_VF | vector | 8,451 | 0.2% |
| V_ADD_VV | vector | 8,065 | 0.2% |
| (remaining vector/scalar) | | | < 0.2% each |

## 4.3 Per-ISA-opcode cycle share — SIMT-32

_Total effective compute = 3,589,839 cycles (3.59 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 3,538,944 | 98.6% |
| M_MV | matmul | 20,736 | 0.6% |
| V_RED_SUM | vector | 13,824 | 0.4% |
| (remaining vector/scalar) | | | < 0.2% each |

## 5. Memory (HBM DMA, analytic bytes/bandwidth, batch=1)

_M port = 1024 B/cyc, V port = 1024 B/cyc; MX factor = 1.125_

| | H_PREFETCH_M | H_PREFETCH_V | H_STORE_V | total H | memory µs |
|---|---|---|---|---|---|
| total | 45 | 864 | 468 | 1,377 | 5.13 |

> Chain memory time (0.009 ms) ≪ compute time, so the final layer is compute-bound like the SSB/DSB blocks.
