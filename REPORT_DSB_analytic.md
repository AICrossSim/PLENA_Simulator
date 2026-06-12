# PLENA Analytic Report — DoubleStreamBlock (DSB)

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
| 1 | 95.72 | 2.005 | 21.0 |
| 2 | 55.01 | 1.969 | 35.8 |
| 4 | 34.65 | 1.950 | 56.3 |
| 8 | 24.48 | 1.941 | 79.3 |
| 16 | 19.39 | 1.937 | 99.9 |
| 32 | 16.84 | 1.934 | 114.9 |

### 2.1 Batch = 4 (full-GPU shape)

All comparisons use batch = 4 (the shape that saturates the GPU's SMs). Latency and energy scale linearly with batch; average power is unchanged.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 382.86 | 8.022 | 21.0 |
| 2 | 220.06 | 7.876 | 35.8 |
| 4 | 138.61 | 7.802 | 56.3 |
| 8 | 97.91 | 7.764 | 79.3 |
| 16 | 77.55 | 7.746 | 99.9 |
| 32 | 67.37 | 7.737 | 114.9 |

## 3. Per-kernel ISA cycles (raw, pre-SIMT)

| Kernel | matmul | vector | scalarFP | #H instr | compute µs | mem µs | bound |
|---|---|---|---|---|---|---|---|
| flash | 4,934,700 | 44,820,480 | 10,076,160 | 840 | 59831.3 | 24.60 | comp |
| I_gelu | 0 | 13,049,856 | 0 | 864 | 13049.9 | 2.16 | comp |
| I_mlpin | 2,654,316 | 1,105,920 | 0 | 2,484 | 3760.2 | 18.36 | comp |
| I_qknk | 0 | 2,128,896 | 884,736 | 324 | 3013.6 | 0.81 | comp |
| I_qknq | 0 | 2,128,896 | 884,736 | 324 | 3013.6 | 0.81 | comp |
| I_mlpout | 2,654,235 | 276,480 | 0 | 1,836 | 2930.7 | 16.74 | comp |
| T_gelu | 0 | 1,449,984 | 0 | 96 | 1450.0 | 0.24 | comp |
| I_link | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| I_linq | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| I_linv | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| I_proj | 663,579 | 276,480 | 0 | 621 | 940.1 | 4.59 | comp |
| I_norm1 | 0 | 691,200 | 46,080 | 432 | 737.3 | 1.08 | comp |
| I_norm2 | 0 | 691,200 | 46,080 | 432 | 737.3 | 1.08 | comp |
| I_ropek | 221,211 | 304,128 | 0 | 459 | 525.3 | 2.16 | comp |
| I_ropeq | 221,211 | 304,128 | 0 | 459 | 525.3 | 2.16 | comp |
| T_mlpin | 294,924 | 122,880 | 0 | 276 | 417.8 | 2.04 | comp |
| T_qknk | 0 | 236,544 | 98,304 | 36 | 334.8 | 0.09 | comp |
| T_qknq | 0 | 236,544 | 98,304 | 36 | 334.8 | 0.09 | comp |
| T_mlpout | 294,915 | 30,720 | 0 | 204 | 325.6 | 1.86 | comp |
| T_link | 73,731 | 30,720 | 0 | 69 | 104.5 | 0.51 | comp |
| T_linq | 73,731 | 30,720 | 0 | 69 | 104.5 | 0.51 | comp |
| T_linv | 73,731 | 30,720 | 0 | 69 | 104.5 | 0.51 | comp |
| T_proj | 73,731 | 30,720 | 0 | 69 | 104.5 | 0.51 | comp |
| T_norm1 | 0 | 76,800 | 5,120 | 48 | 81.9 | 0.12 | comp |
| T_norm2 | 0 | 76,800 | 5,120 | 48 | 81.9 | 0.12 | comp |
| T_ropek | 24,579 | 33,792 | 0 | 51 | 58.4 | 0.24 | comp |
| T_ropeq | 24,579 | 33,792 | 0 | 51 | 58.4 | 0.24 | comp |
| I_mod1 | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| I_mod2 | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| I_res1 | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| I_res2 | 0 | 55,296 | 0 | 432 | 55.3 | 1.08 | comp |
| T_mod1 | 0 | 6,144 | 0 | 48 | 6.1 | 0.12 | comp |
| T_mod2 | 0 | 6,144 | 0 | 48 | 6.1 | 0.12 | comp |
| T_res1 | 0 | 6,144 | 0 | 48 | 6.1 | 0.12 | comp |
| T_res2 | 0 | 6,144 | 0 | 48 | 6.1 | 0.12 | comp |
| concat_k | 0 | 0 | 0 | 2,400 | 0.0 | 6.00 | mem |
| concat_q | 0 | 0 | 0 | 2,400 | 0.0 | 6.00 | mem |
| concat_v | 0 | 0 | 0 | 2,400 | 0.0 | 6.00 | mem |
| split_attn | 0 | 0 | 0 | 2,400 | 0.0 | 6.00 | mem |

## 4.1 Per-ISA-opcode cycle share — SIMT-1

_Total effective compute = 95,692,150 cycles (95.69 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| V_RED_SUM | vector | 24,576,000 | 25.7% |
| V_RED_MAX | vector | 12,288,000 | 12.8% |
| M_BTMM | matmul | 11,796,480 | 12.3% |
| V_MUL_VF | vector | 11,642,880 | 12.2% |
| V_EXP_V | vector | 6,881,280 | 7.2% |
| V_ADD_VF | vector | 5,222,400 | 5.5% |
| V_ADD_VV | vector | 3,317,760 | 3.5% |
| S_MUL_FP | scalar | 2,990,080 | 3.1% |
| S_ADD_FP | scalar | 2,969,600 | 3.1% |
| V_SUB_VF | vector | 2,519,040 | 2.6% |
| S_SUB_FP | scalar | 2,457,600 | 2.6% |
| S_EXP_FP | scalar | 2,457,600 | 2.6% |
| M_MM | matmul | 2,457,600 | 2.6% |
| V_RECI_V | vector | 1,966,080 | 2.1% |
| V_MUL_VV | vector | 860,160 | 0.9% |
| S_RECI_FP | scalar | 757,760 | 0.8% |
| S_SQRT_FP | scalar | 512,000 | 0.5% |
| M_MM_WO | matmul | 19,200 | 0.0% |
| M_BMM_WO | matmul | 630 | 0.0% |

## 4.2 Per-ISA-opcode cycle share — SIMT-16

_Total effective compute = 19,362,550 cycles (19.36 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 11,796,480 | 60.9% |
| M_MM | matmul | 2,457,600 | 12.7% |
| V_RED_SUM | vector | 1,536,000 | 7.9% |
| V_RED_MAX | vector | 768,000 | 4.0% |
| V_MUL_VF | vector | 727,680 | 3.8% |
| V_EXP_V | vector | 430,080 | 2.2% |
| V_ADD_VF | vector | 326,400 | 1.7% |
| V_ADD_VV | vector | 207,360 | 1.1% |
| S_MUL_FP | scalar | 186,880 | 1.0% |
| S_ADD_FP | scalar | 185,600 | 1.0% |
| V_SUB_VF | vector | 157,440 | 0.8% |
| S_SUB_FP | scalar | 153,600 | 0.8% |
| S_EXP_FP | scalar | 153,600 | 0.8% |
| V_RECI_V | vector | 122,880 | 0.6% |
| V_MUL_VV | vector | 53,760 | 0.3% |
| S_RECI_FP | scalar | 47,360 | 0.2% |
| S_SQRT_FP | scalar | 32,000 | 0.2% |
| M_MM_WO | matmul | 19,200 | 0.1% |
| M_BMM_WO | matmul | 630 | 0.0% |

## 4.3 Per-ISA-opcode cycle share — SIMT-32

_Total effective compute = 16,818,230 cycles (16.82 ms)_

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 11,796,480 | 70.1% |
| M_MM | matmul | 2,457,600 | 14.6% |
| V_RED_SUM | vector | 768,000 | 4.6% |
| V_RED_MAX | vector | 384,000 | 2.3% |
| V_MUL_VF | vector | 363,840 | 2.2% |
| V_EXP_V | vector | 215,040 | 1.3% |
| V_ADD_VF | vector | 163,200 | 1.0% |
| V_ADD_VV | vector | 103,680 | 0.6% |
| S_MUL_FP | scalar | 93,440 | 0.6% |
| S_ADD_FP | scalar | 92,800 | 0.6% |
| V_SUB_VF | vector | 78,720 | 0.5% |
| S_SUB_FP | scalar | 76,800 | 0.5% |
| S_EXP_FP | scalar | 76,800 | 0.5% |
| V_RECI_V | vector | 61,440 | 0.4% |
| V_MUL_VV | vector | 26,880 | 0.2% |
| S_RECI_FP | scalar | 23,680 | 0.1% |
| M_MM_WO | matmul | 19,200 | 0.1% |
| S_SQRT_FP | scalar | 16,000 | 0.1% |
| M_BMM_WO | matmul | 630 | 0.0% |

## 5. Memory (HBM DMA, analytic bytes/bandwidth)

_Port bandwidth: M port = 1024 B/cyc, V port = 1024 B/cyc; MX factor = 1.125 (E4M3 = 8-bit elem + E8M0 8-bit scale, block=8)_

| Kernel | H_PREFETCH_M | H_PREFETCH_V | H_STORE_V | total H | memory µs |
|---|---|---|---|---|---|
| flash | 600 | 120 | 120 | 840 | 24.60 |
| I_gelu | 0 | 432 | 432 | 864 | 2.16 |
| I_mlpin | 324 | 1,728 | 432 | 2,484 | 18.36 |
| I_qknk | 0 | 216 | 108 | 324 | 0.81 |
| I_qknq | 0 | 216 | 108 | 324 | 0.81 |
| I_mlpout | 324 | 1,404 | 108 | 1,836 | 16.74 |
| T_gelu | 0 | 48 | 48 | 96 | 0.24 |
| I_link | 81 | 432 | 108 | 621 | 4.59 |
| I_linq | 81 | 432 | 108 | 621 | 4.59 |
| I_linv | 81 | 432 | 108 | 621 | 4.59 |
| I_proj | 81 | 432 | 108 | 621 | 4.59 |
| I_norm1 | 0 | 324 | 108 | 432 | 1.08 |
| I_norm2 | 0 | 324 | 108 | 432 | 1.08 |
| I_ropek | 27 | 324 | 108 | 459 | 2.16 |
| I_ropeq | 27 | 324 | 108 | 459 | 2.16 |
| T_mlpin | 36 | 192 | 48 | 276 | 2.04 |
| T_qknk | 0 | 24 | 12 | 36 | 0.09 |
| T_qknq | 0 | 24 | 12 | 36 | 0.09 |
| T_mlpout | 36 | 156 | 12 | 204 | 1.86 |
| T_link | 9 | 48 | 12 | 69 | 0.51 |
| T_linq | 9 | 48 | 12 | 69 | 0.51 |
| T_linv | 9 | 48 | 12 | 69 | 0.51 |
| T_proj | 9 | 48 | 12 | 69 | 0.51 |
| T_norm1 | 0 | 36 | 12 | 48 | 0.12 |
| T_norm2 | 0 | 36 | 12 | 48 | 0.12 |
| T_ropek | 3 | 36 | 12 | 51 | 0.24 |
| T_ropeq | 3 | 36 | 12 | 51 | 0.24 |
| I_mod1 | 0 | 324 | 108 | 432 | 1.08 |
| I_mod2 | 0 | 324 | 108 | 432 | 1.08 |
| I_res1 | 0 | 324 | 108 | 432 | 1.08 |
| I_res2 | 0 | 324 | 108 | 432 | 1.08 |
| T_mod1 | 0 | 36 | 12 | 48 | 0.12 |
| T_mod2 | 0 | 36 | 12 | 48 | 0.12 |
| T_res1 | 0 | 36 | 12 | 48 | 0.12 |
| T_res2 | 0 | 36 | 12 | 48 | 0.12 |
| concat_k | 0 | 1,200 | 1,200 | 2,400 | 6.00 |
| concat_q | 0 | 1,200 | 1,200 | 2,400 | 6.00 |
| concat_v | 0 | 1,200 | 1,200 | 2,400 | 6.00 |
| split_attn | 0 | 1,200 | 1,200 | 2,400 | 6.00 |
| **total** | 1,740 | 14,160 | 7,680 | 23,580 | 124.20 |

> Chain memory time (0.124 ms) ≪ compute time, so every kernel is compute-bound and memory fully overlaps under the max(compute,memory) model.

---

# Appendix A — BLEN = 8 configuration

The same DSB compiled and analyzed at **BLEN = 8** (MAC array = 8 × 1024 = 8,192). Everything else unchanged. M_BMM = (1024/8)² × 8 = 131,072 cycles (16× the BLEN=128 value), so latency rises and matrix power falls.

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
| 1 | 314.42 | 2.437 | 7.8 |
| 2 | 273.71 | 2.401 | 8.8 |
| 4 | 253.36 | 2.382 | 9.4 |
| 8 | 243.18 | 2.373 | 9.8 |
| 16 | 238.09 | 2.369 | 9.9 |
| 32 | 235.55 | 2.366 | 10.0 |

### A.2.1 Batch = 4 (full-GPU shape)

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 1257.68 | 9.749 | 7.8 |
| 2 | 1094.84 | 9.602 | 8.8 |
| 4 | 1013.44 | 9.529 | 9.4 |
| 8 | 972.72 | 9.492 | 9.8 |
| 16 | 952.36 | 9.474 | 9.9 |
| 32 | 942.20 | 9.465 | 10.0 |

### A.2.2 Batch = 4, device count = 12 (full deployment)

Twelve PLENA devices run the batch-4 workload in parallel (ideal linear scaling): latency ÷12, total energy unchanged, average power ×12.

| SIMT | Latency (ms) | Energy (J) | Avg power (W) |
|---|---|---|---|
| 1 | 104.81 | 9.749 | 93.6 |
| 2 | 91.24 | 9.602 | 105.6 |
| 4 | 84.45 | 9.529 | 112.8 |
| 8 | 81.06 | 9.492 | 117.6 |
| 16 | 79.36 | 9.474 | 118.8 |
| 32 | 78.52 | 9.465 | 120.0 |

## A.3 Per-ISA-opcode cycle share

**SIMT-1** (total eff compute = 314,399,350 cyc = 314.40 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 188,743,680 | 60.0% |
| M_MM | matmul | 39,321,600 | 12.5% |
| V_RED_SUM | vector | 24,576,000 | 7.8% |
| V_RED_MAX | vector | 12,288,000 | 3.9% |
| V_MUL_VF | vector | 11,642,880 | 3.7% |
| V_EXP_V | vector | 6,881,280 | 2.2% |
| V_ADD_VF | vector | 5,222,400 | 1.7% |
| M_MM_WO | matmul | 4,915,200 | 1.6% |
| (remaining vector/scalar) | | | < 1.5% each |

**SIMT-16** (total eff compute = 238,069,750 cyc = 238.07 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 188,743,680 | 79.3% |
| M_MM | matmul | 39,321,600 | 16.5% |
| M_MM_WO | matmul | 4,915,200 | 2.1% |
| V_RED_SUM | vector | 1,536,000 | 0.6% |
| (remaining vector/scalar) | | | < 0.4% each |

**SIMT-32** (total eff compute = 235,525,430 cyc = 235.53 ms)

| opcode | bucket | eff cycles | share |
|---|---|---|---|
| M_BTMM | matmul | 188,743,680 | 80.1% |
| M_MM | matmul | 39,321,600 | 16.7% |
| M_MM_WO | matmul | 4,915,200 | 2.1% |
| V_RED_SUM | vector | 768,000 | 0.3% |
| (remaining vector/scalar) | | | < 0.2% each |

## A.4 Memory (HBM DMA, batch=1)

| | H_PREFETCH_M | H_PREFETCH_V | H_STORE_V | total H | memory µs |
|---|---|---|---|---|---|
| total | 1,740 | 14,160 | 7,680 | 23,580 | 124.20 |

> Memory counts are identical to BLEN=128; chain memory time (0.223 ms) ≪ compute, so every kernel is compute-bound.
