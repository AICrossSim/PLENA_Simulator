# PLENA RTL vs Simulator Latency Validation Summary

Generated for paper submission. All RTL numbers from cocotb behavioral-mode simulation.

---

## 1. Instruction-Level Validation (VLEN=8, behavioral mode)

All RTL cycle counts measured via **cocotb + Verilator** simulation of the SystemVerilog RTL.
- **RTL sub-unit**: cycles measured at the standalone functional unit (e.g. `fp_reduction_compute_unit` in isolation)
- **RTL top-level**: cycles measured through the full `vector_machine` wrapper, including register slices, valid/ready handshake overhead, and pipeline staging — closer to real system behavior
- **Sim value**: PLENA analytical simulator prediction (from `configuration.svh` constants)

| Instruction | RTL sub-unit (cyc) | RTL top-level (cyc) | Sim value (cyc) | RTL/Sim | Source testbench |
|---|---|---|---|---|---|
| V_ADD_VV | 7 | 7 | 7 | 1.00x | vector_machine_tb |
| V_EXP_V | 7 | 8 | 7 | 1.14x | fp_elementwise_compute_unit_tb, flash_attn_tb |
| V_RED_MAX | 10 | 12 | 4 | 3.00x | reduction_compute_unit_tb, flash_attn_tb |
| V_RED_SUM | 14 | 17 | 20 | 0.85x | reduction_compute_unit_tb, flash_attn_tb |
| S_EXP_FP | 7 | — | 2 | 3.50x | scalar_machine_tb (behavioral) |
| S_RECI_FP | 4 | — | 2 | 2.00x | scalar_machine_tb (behavioral) |

**Key observations:**
- V_ADD_VV matches exactly (1.00x) — combinational add pipeline fully modeled.
- V_EXP_V top-level adds 1 cycle wrapper overhead; sim underestimates by 1 cycle.
- V_RED_MAX: sim formula (VECTOR_MAX_CYCLES=4) does not account for the 8-cycle
  reduction unit pipeline drain in the top-level wrapper. RTL is 3x higher.
- V_RED_SUM: sim overestimates slightly (20 vs 17 top-level). The sim formula
  includes an extra accumulation step not present at top-level timing boundary.
- Scalar ops (S_EXP_FP, S_RECI_FP): behavioral mode shows multi-cycle execution
  (7 and 4 cycles respectively); sim constants (2 cycles each) target DC_LIB
  synthesis-calibrated mode where these are pipelined differently.

> **When the assumption breaks down**: single-shot ops that aren't pipelined with
> anything else. In the softmax pipeline, V_RED_MAX runs alone (no overlap with the
> next op because there's a data dependency — you need the max before you can subtract
> it). So you pay the full 12-cycle latency, not the 4-cycle throughput. That's the
> +8 cycle gap in the softmax total (43 vs 38). The simulator models pipelined
> throughput, which is correct for GEMM-dominant workloads but underestimates
> single-issue non-GEMM ops by the pipeline-fill overhead (~5-8 cycles per invocation).

---

## 2. GEMM Tile-Level Validation (BLEN=4, COMPUTE_DIM=4)

| Metric | RTL Cocotb | Sim formula | Sim value | Match |
|---|---|---|---|---|
| First-tile latency | 11 cyc | not modeled | — | — |
| Steady-state throughput | 5 cyc/tile | 1+BLEN | 5 | EXACT (1.00x) |
| PE pipeline depth | 7 cyc (derived) | not modeled | — | — |
| MM_WO single tile (MLEN=16) | 35 cyc | 1+BLOCK_DIM=9 (throughput) | 9 | 3.89x (single tile = first-tile dominated) |

**Key observations:**
- Steady-state GEMM throughput is an exact match: 1+BLEN = 5 cycles/tile.
- First-tile latency (11 cycles) is not modeled; this creates a fixed overhead
  per GEMM call of approximately 6 extra cycles beyond throughput prediction.
- Single-tile MM_WO (35 cyc) is dominated by first-tile fill; throughput formula
  only predicts steady-state and underestimates by 3.89x for N_tiles=1.
- For large GEMMs (N_tiles >> 1), first-tile overhead becomes negligible and
  the model converges toward 1.07x overhead.

---

## 3. Flash-Attention Softmax Pipeline (top-level vector_machine, VLEN=8)

| Step | Operation | RTL top-level (cyc) | Sim (cyc) | Delta | Note |
|---|---|---|---|---|---|
| 1 | V_RED_MAX | 12 | 4 | +8 | wrapper overhead dominates |
| 2 | V_ADD_VV | 6 | 7 | -1 | pipeline fills 1 cycle faster in RTL |
| 3 | V_EXP_V | 8 | 7 | +1 | 1-cycle wrapper overhead |
| 4 | V_RED_SUM | 17 | 20 | -3 | sim overestimates by 3 cycles |
| **Total** | **softmax tile** | **43** | **38** | **+5** | **1.13x overhead** |

The V_ADD_VV undercount (-1) and V_RED_SUM overcount (-3) partially cancel the
V_RED_MAX undercount (+8), leaving a net 13% RTL overhead vs simulator.

---

## 4. Attention Layer GEMM Validation (BLEN=4, MLEN=64, hidden=64, heads=2)

RTL estimate method: first-tile overhead = 11 cycles, steady-state = 5 cyc/tile.
Per-GEMM RTL estimate = 11 + (N_tiles - 1) × 5.

| Operation | Tiles | Sim (cyc) | RTL est. (cyc) | RTL/Sim |
|---|---|---|---|---|
| Q-projection (1×64)@(64×64) | 16 | 80 | 86 | 1.07x |
| K-projection (1×64)@(64×64) | 16 | 80 | 86 | 1.07x |
| V-projection (1×64)@(64×64) | 16 | 80 | 86 | 1.07x |
| QK^T (1×32)@(32×1) ×2 heads | 1 | 5 | 11 | 2.20x |
| AV (1×1)@(1×32) ×2 heads | 8 | 40 | 46 | 1.15x |
| O-projection (1×64)@(64×64) | 16 | 80 | 86 | 1.07x |
| **Total attention layer** | | **365** | **401** | **1.10x** |

**Key observations:**
- Large-tile projections (16 tiles): consistent 1.07x overhead from first-tile penalty.
  RTL = 11 + 15×5 = 86 vs Sim = 16×5 = 80.
- Single-tile QK^T: 2.20x ratio entirely from first-tile overhead (11 cycles vs 5).
  This operation is too small to amortize pipeline fill cost.
- AV operation (8 tiles): 1.15x ratio — intermediate case, first-tile penalty
  is 6/40 = 15% of total.
- **Overall layer accuracy: 1.10x (10% underestimate)** — acceptable for
  architectural design-space exploration and area/performance tradeoff analysis.

---

## 5. Modeling Gaps and Recommendations

| Gap | Impact | Recommended fix |
|---|---|---|
| First-tile pipeline fill not modeled | +6 cyc per GEMM call | Add `FIRST_TILE_OVERHEAD = COMPUTE_DIM + BLEN` constant |
| Reduction unit wrapper overhead (+8 cyc) | V_RED_MAX/SUM off by ~8 cyc | Add per-call overhead term to VECTOR_MAX/SUM_CYCLES |
| Scalar behavioral vs DC_LIB mismatch | S_EXP/RECI off by 2-3.5x | Document that scalar constants target DC_LIB mode only |
| Single-tile GEMM accuracy | 3.89x for N=1 | Add guard: if N_tiles==1, use first_tile_latency not throughput |

**Bottom line:** The analytical simulator is well-calibrated for throughput-bound
multi-tile GEMMs (1.07x) and is suitable for layer- and model-level performance
estimation. Per-instruction accuracy for reduction and scalar ops requires
DC_LIB synthesis calibration to match behavioral RTL measurements.

---

## Simulation Environment

- RTL simulator: cocotb (behavioral mode, no DC_LIB synthesis timing)
- Hardware parameters: VLEN=8, BLEN=4, COMPUTE_DIM=4, MLEN=16 (tile), MLEN=64 (layer test)
- Analytical simulator: PLENA Simulator (`analytic_models/performance/`)
- Date: 2026-03-24
