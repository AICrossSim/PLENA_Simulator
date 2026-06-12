# PLENA Analytic Report — SingleStreamBlock (SSB)

_Geometry: MLEN=1024, HLEN=128, BLEN=128, VLEN=1024 (MAC array = 128×1024 = 131,072); freq=1.0 GHz_

_Power model: bottom-up 7nm (BF16→FP32): MAC=1 mW, vector lane=0.5 mW, SRAM=0.055 pJ/bit_

_Cycle formula: George performance customISA (pipelined column; M_BMM=8192, M_MM=128)_

## 1. Unit power (W, steady-state @100% utilisation)

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **131.1 W** | 1 mW/MAC × 131,072 MAC |
| Vector unit (1 thread) | 0.512 W | 0.5 mW/lane × 1024 lanes |
| Matrix SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |
| Vector SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |

> Under SIMT-N the vector unit packs N threads: vector cycles ÷N, vector-unit power ×N (energy conserved). Matrix core is unaffected by SIMT.

## 2. Chain latency & energy vs SIMT

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 79.66 | 1.738 | 21.8 |
| 2 | 46.03 | 1.708 | 37.1 |
| 4 | 29.22 | 1.692 | 57.9 |
| 8 | 20.81 | 1.685 | 81.0 |
| 16 | 16.61 | 1.681 | 101.2 |
| 32 | 14.51 | 1.679 | 115.7 |

### 2.1 Batch = 4 (full-GPU shape)

All comparisons use batch = 4 (the shape that saturates the GPU's SMs). Latency and energy scale linearly with batch; average power is unchanged.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 318.65 | 6.952 | 21.8 |
| 2 | 184.13 | 6.831 | 37.1 |
| 4 | 116.88 | 6.770 | 57.9 |
| 8 | 83.22 | 6.739 | 81.0 |
| 16 | 66.42 | 6.724 | 101.2 |
| 32 | 58.03 | 6.717 | 115.7 |

## 3. Per-kernel ISA cycles (raw, pre-SIMT)

| Kernel | matmul | vector | scalarFP | #H instr | compute µs | mem µs | bound |
|---|---|---|---|---|---|---|---|
| flash_attention | 3,997,107 | 36,329,472 | 8,183,808 | 702 | 48510.4 | 19.98 | comp |
| gelu | 0 | 13,049,856 | 0 | 864 | 13049.9 | 2.16 | comp |
| linear_mlp | 2,654,316 | 1,105,920 | 0 | 2,484 | 3760.2 | 18.36 | comp |
| linear2 | 3,317,787 | 276,480 | 0 | 2,241 | 3594.3 | 20.79 | comp |
| qknorm_k | 0 | 2,128,896 | 884,736 | 324 | 3013.6 | 0.81 | comp |
| qknorm_q | 0 | 2,128,896 | 884,736 | 324 | 3013.6 | 0.81 | comp |
| linear_k | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| linear_q | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| linear_v | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| layernorm | 0 | 691,200 | 46,080 | 432 | 737.3 | 1.08 | comp |
| rope_k | 221,211 | 304,128 | 0 | 459 | 525.3 | 2.16 | comp |
| rope_q | 221,211 | 304,128 | 0 | 459 | 525.3 | 2.16 | comp |
| modulate | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| residual_gate | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| concat | 0 | 0 | 0 | 1,080 | 0.0 | 2.70 | mem |

## 4.1 Per-ISA-opcode cycle share — SIMT-1

_Total effective compute = 79,660,737 cycles (79.66 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| V_RED_SUM | vector | 19,906,560 | 25.0% |
| M_BTMM | matmul | 10,395,648 | 13.0% |
| V_MUL_VF | vector | 10,008,576 | 12.6% |
| V_RED_MAX | vector | 9,953,280 | 12.5% |
| V_EXP_V | vector | 5,750,784 | 7.2% |
| V_ADD_VF | vector | 4,534,272 | 5.7% |
| V_ADD_VV | vector | 2,654,208 | 3.3% |
| S_MUL_FP | scalar | 2,451,456 | 3.1% |
| S_ADD_FP | scalar | 2,442,240 | 3.1% |
| V_SUB_VF | vector | 2,018,304 | 2.5% |
| S_SUB_FP | scalar | 1,990,656 | 2.5% |
| S_EXP_FP | scalar | 1,990,656 | 2.5% |
| M_MM | matmul | 1,990,656 | 2.5% |
| V_RECI_V | vector | 1,769,472 | 2.2% |
| S_RECI_FP | scalar | 672,768 | 0.8% |
| V_MUL_VV | vector | 663,552 | 0.8% |
| S_SQRT_FP | scalar | 451,584 | 0.6% |
| M_MM_WO | matmul | 15,552 | 0.0% |
| M_BMM_WO | matmul | 513 | 0.0% |

## 4.2 Per-ISA-opcode cycle share — SIMT-16

_Total effective compute = 16,606,017 cycles (16.61 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 10,395,648 | 62.6% |
| M_MM | matmul | 1,990,656 | 12.0% |
| V_RED_SUM | vector | 1,244,160 | 7.5% |
| V_MUL_VF | vector | 625,536 | 3.8% |
| V_RED_MAX | vector | 622,080 | 3.7% |
| V_EXP_V | vector | 359,424 | 2.2% |
| V_ADD_VF | vector | 283,392 | 1.7% |
| V_ADD_VV | vector | 165,888 | 1.0% |
| S_MUL_FP | scalar | 153,216 | 0.9% |
| S_ADD_FP | scalar | 152,640 | 0.9% |
| V_SUB_VF | vector | 126,144 | 0.8% |
| S_SUB_FP | scalar | 124,416 | 0.7% |
| S_EXP_FP | scalar | 124,416 | 0.7% |
| V_RECI_V | vector | 110,592 | 0.7% |
| S_RECI_FP | scalar | 42,048 | 0.3% |
| V_MUL_VV | vector | 41,472 | 0.2% |
| S_SQRT_FP | scalar | 28,224 | 0.2% |
| M_MM_WO | matmul | 15,552 | 0.1% |
| M_BMM_WO | matmul | 513 | 0.0% |

## 4.3 Per-ISA-opcode cycle share — SIMT-32

_Total effective compute = 14,504,193 cycles (14.50 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 10,395,648 | 71.7% |
| M_MM | matmul | 1,990,656 | 13.7% |
| V_RED_SUM | vector | 622,080 | 4.3% |
| V_MUL_VF | vector | 312,768 | 2.2% |
| V_RED_MAX | vector | 311,040 | 2.1% |
| V_EXP_V | vector | 179,712 | 1.2% |
| V_ADD_VF | vector | 141,696 | 1.0% |
| V_ADD_VV | vector | 82,944 | 0.6% |
| S_MUL_FP | scalar | 76,608 | 0.5% |
| S_ADD_FP | scalar | 76,320 | 0.5% |
| V_SUB_VF | vector | 63,072 | 0.4% |
| S_SUB_FP | scalar | 62,208 | 0.4% |
| S_EXP_FP | scalar | 62,208 | 0.4% |
| V_RECI_V | vector | 55,296 | 0.4% |
| S_RECI_FP | scalar | 21,024 | 0.1% |
| V_MUL_VV | vector | 20,736 | 0.1% |
| M_MM_WO | matmul | 15,552 | 0.1% |
| S_SQRT_FP | scalar | 14,112 | 0.1% |
| M_BMM_WO | matmul | 513 | 0.0% |

## 5. Memory (HBM DMA, analytic bytes/bandwidth)

_Port bandwidth: M port = 1024 B/cyc, V port = 1024 B/cyc; MX factor = 1.125 (E4M3 = 8-bit elem + E8M0 8-bit scale, block=8)_

| Kernel | H_PREFETCH_M | H_PREFETCH_V | H_STORE_V | total H | memory µs |
|---|---|---|---|---|---|
| flash_attention | 486 | 108 | 108 | 702 | 19.98 |
| gelu | 0 | 432 | 432 | 864 | 2.16 |
| linear_mlp | 324 | 1,728 | 432 | 2,484 | 18.36 |
| linear2 | 405 | 1,728 | 108 | 2,241 | 20.79 |
| qknorm_k | 0 | 216 | 108 | 324 | 0.81 |
| qknorm_q | 0 | 216 | 108 | 324 | 0.81 |
| linear_k | 81 | 432 | 108 | 621 | 4.59 |
| linear_q | 81 | 432 | 108 | 621 | 4.59 |
| linear_v | 81 | 432 | 108 | 621 | 4.59 |
| layernorm | 0 | 324 | 108 | 432 | 1.08 |
| rope_k | 27 | 324 | 108 | 459 | 2.16 |
| rope_q | 27 | 324 | 108 | 459 | 2.16 |
| modulate | 0 | 324 | 108 | 432 | 1.08 |
| residual_gate | 0 | 324 | 108 | 432 | 1.08 |
| concat | 0 | 540 | 540 | 1,080 | 2.70 |
| **total** | 1,512 | 7,884 | 2,700 | 12,096 | 86.94 |

> Chain memory time (0.157 ms) ≪ compute time, so every kernel is compute-bound and memory fully overlaps under the max(compute,memory) model.

---

# Appendix A — BLEN = 8 configuration

The same SSB compiled and analyzed at **BLEN = 8** (MAC array = 8 × 1024 = 8,192). Everything else (MLEN, HLEN, VLEN, power model, cycle formula) is unchanged. Smaller BLEN means the matrix core does many more tiles per batched matmul (M_BMM = (1024/8)² × 8 = 131,072 cycles vs 8,192 at BLEN=128), so latency rises and matrix power falls.

_Geometry: MLEN=1024, HLEN=128, **BLEN=8**, VLEN=1024 (MAC array = 8×1024 = **8,192**); freq=1.0 GHz_

_Cycle formula: George performance customISA (pipelined; M_BMM=131072, M_MM=8)_

## A.1 Unit power

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **8.2 W** | 1 mW/MAC × 8,192 MAC |
| Vector unit (1 thread) | 0.512 W | 0.5 mW/lane × 1024 lanes |
| Matrix SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |
| Vector SRAM | 0.901 W | 0.055 pJ/bit × 1024×16b × 1 GHz |

## A.2 Chain latency & energy vs SIMT (batch = 1)

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 269.42 | 2.110 | 7.8 |
| 2 | 235.79 | 2.080 | 8.8 |
| 4 | 218.98 | 2.065 | 9.4 |
| 8 | 210.57 | 2.057 | 9.8 |
| 16 | 206.37 | 2.054 | 10.0 |
| 32 | 204.27 | 2.052 | 10.0 |

### A.2.1 Batch = 4 (full-GPU shape)

Latency and energy scale linearly with batch; average power unchanged.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 1077.68 | 8.442 | 7.8 |
| 2 | 943.15 | 8.320 | 8.8 |
| 4 | 875.92 | 8.260 | 9.4 |
| 8 | 842.30 | 8.230 | 9.8 |
| 16 | 825.46 | 8.214 | 10.0 |
| 32 | 817.07 | 8.207 | 10.0 |

### A.2.2 Batch = 4, device count = 12 (full deployment)

Twelve PLENA devices run the batch-4 workload in parallel (ideal linear scaling): latency ÷12, total energy unchanged, average power ×12.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 89.81 | 8.442 | 93.6 |
| 2 | 78.60 | 8.320 | 105.6 |
| 4 | 72.99 | 8.260 | 112.8 |
| 8 | 70.19 | 8.230 | 117.6 |
| 16 | 68.79 | 8.214 | 120.0 |
| 32 | 68.09 | 8.207 | 120.0 |

## A.3 Per-kernel matmul cycles (raw, pre-SIMT, batch=1)

Only the matmul column changes vs BLEN=128 (vector/scalar are independent of BLEN). M_BMM/M_MM are now 16× larger per instruction.

| Kernel | matmul | vector | scalarFP | compute µs |
|---|---|---|---|---|
| flash_attention | 67,682,547 | 36,329,472 | 8,183,808 | 112195.8 |
| linear2 | 53,084,187 | 276,480 | 0 | 53360.7 |
| linear_mlp | 42,467,436 | 1,105,920 | 0 | 43573.4 |
| linear_k | 10,616,859 | 276,480 | 0 | 10893.3 |
| linear_q | 10,616,859 | 276,480 | 0 | 10893.3 |
| linear_v | 10,616,859 | 276,480 | 0 | 10893.3 |
| rope_k | 3,538,971 | 304,128 | 0 | 3843.1 |
| rope_q | 3,538,971 | 304,128 | 0 | 3843.1 |
| gelu | 0 | 13,049,856 | 0 | 13049.9 |
| qknorm_k | 0 | 2,128,896 | 884,736 | 3013.6 |
| qknorm_q | 0 | 2,128,896 | 884,736 | 3013.6 |
| layernorm | 0 | 691,200 | 46,080 | 737.3 |
| modulate | 0 | 55,296 | 0 | 55.3 |
| residual_gate | 0 | 55,296 | 0 | 55.3 |
| concat | 0 | 0 | 0 | 0.0 |

## A.4 Per-ISA-opcode cycle share

**SIMT-1** (total eff compute = 269,421,057 cyc = 269.42 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 166,330,368 | 61.7% |
| M_MM | matmul | 31,850,496 | 11.8% |
| V_RED_SUM | vector | 19,906,560 | 7.4% |
| V_MUL_VF | vector | 10,008,576 | 3.7% |
| V_RED_MAX | vector | 9,953,280 | 3.7% |
| V_EXP_V | vector | 5,750,784 | 2.1% |
| V_ADD_VF | vector | 4,534,272 | 1.7% |
| M_MM_WO | matmul | 3,981,312 | 1.5% |
| V_ADD_VV | vector | 2,654,208 | 1.0% |
| S_MUL_FP | scalar | 2,451,456 | 0.9% |
| S_ADD_FP | scalar | 2,442,240 | 0.9% |
| V_SUB_VF | vector | 2,018,304 | 0.7% |
| S_SUB_FP | scalar | 1,990,656 | 0.7% |
| S_EXP_FP | scalar | 1,990,656 | 0.7% |
| V_RECI_V | vector | 1,769,472 | 0.7% |
| S_RECI_FP | scalar | 672,768 | 0.2% |
| V_MUL_VV | vector | 663,552 | 0.2% |
| S_SQRT_FP | scalar | 451,584 | 0.2% |
| M_BMM_WO | matmul | 513 | 0.0% |

**SIMT-16** (total eff compute = 206,366,337 cyc = 206.37 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 166,330,368 | 80.6% |
| M_MM | matmul | 31,850,496 | 15.4% |
| M_MM_WO | matmul | 3,981,312 | 1.9% |
| V_RED_SUM | vector | 1,244,160 | 0.6% |
| V_MUL_VF | vector | 625,536 | 0.3% |
| V_RED_MAX | vector | 622,080 | 0.3% |
| V_EXP_V | vector | 359,424 | 0.2% |
| (remaining vector/scalar) | | | < 0.2% each |
| M_BMM_WO | matmul | 513 | 0.0% |

**SIMT-32** (total eff compute = 204,264,513 cyc = 204.26 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 166,330,368 | 81.4% |
| M_MM | matmul | 31,850,496 | 15.6% |
| M_MM_WO | matmul | 3,981,312 | 1.9% |
| V_RED_SUM | vector | 622,080 | 0.3% |
| (remaining vector/scalar) | | | < 0.2% each |
| M_BMM_WO | matmul | 513 | 0.0% |

## A.5 Memory (HBM DMA, batch=1)

| | H_PREFETCH_M | H_PREFETCH_V | H_STORE_V | total H | memory µs |
|---|---|---|---|---|---|
| total | 1,512 | 7,884 | 2,700 | 12,096 | 86.94 |

> Memory counts are identical to BLEN=128 (HBM traffic does not depend on BLEN); chain memory time (0.157 ms) ≪ compute, so every kernel is compute-bound.
