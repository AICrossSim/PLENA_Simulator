# RTL Validation Plan: PLENA Simulator Accuracy for MICRO 59

## Goal

Prove that the PLENA_Simulator produces accurate cycle-level estimates by comparing
cocotb RTL measurements to simulator formula predictions for key instructions and layers.

This evidence supports the claim in the d-PLENA paper (MICRO 59) that the simulator
faithfully models the hardware pipeline.

---

## What's Already Done ✓

| Op        | RTL Module              | Cocotb Measured | Sim Formula          | Match? |
|-----------|------------------------|-----------------|----------------------|--------|
| M_MM tile | mx_systolic_array       | 11 cyc (fill), +5/tile | 1+BLEN=5 cyc/tile | ✓ CONSISTENT |
| V_ADD_VV  | vector_machine          | 7 cycles        | VECTOR_ADD_CYCLES=7  | ✓ EXACT |
| V_EXP_V   | fp_elementwise_compute_unit | 7 cycles   | 1+VECTOR_EXP_CYCLES=7 | ✓ EXACT |
| V_RED_MAX | fp_reduction_compute_unit | 10 cyc (VLEN=8) | VECTOR_MAX_CYCLES=4 | latency>throughput (expected) |
| V_RED_SUM | fp_reduction_compute_unit | 14 cyc (VLEN=8) | VECTOR_SUM_CYCLES=20 | latency<throughput (VLEN=8 vs VLEN=64 model) |
| S_EXP_FP  | fp_sfu (scalar_machine_tb) | **7 cyc** (E5M10) | SCALAR_FP_EXP_CYCLES=2 | RTL behavioral pipeline deeper than dc_lib sim constant |
| S_RECI_FP | fp_sfu (scalar_machine_tb) | **4 cyc** (E5M10) | SCALAR_FP_RECI_CYCLES=2 | RTL behavioral pipeline deeper than dc_lib sim constant |
| MM_WO tile | matrix_machine_v2 (matrix_machine_tb) | **35 cyc** (1 tile, MLEN=16) | 1+BLOCK_DIM=9 (thput) | Phase1=11 (load), Phase2=22 (drain); amortizes in steady state |

### Flash-Attention Softmax (top-level vector_machine, VLEN=8) — MEASURED ✓

`fp_vector_machine_flash_attn_tb.py` — 2026-03-14

| Step | Op        | RTL cycles | Sim formula | Note |
|------|-----------|-----------|-------------|------|
| 1    | V_RED_MAX | **12**    | 4           | top-level register_slice overhead |
| 2    | V_ADD_VV  | **6**     | 7           | 1 cycle under sim (pipeline fills faster) |
| 3    | V_EXP_V   | **8**     | 7           | 1 cycle over sim |
| 4    | V_RED_SUM | **17**    | 20          | top-level, faster than sub-module (14 cyc) |
| **Total** | **softmax tile** | **43** | **38** | +5 cyc overhead from vector_machine wrapper |

---

## What We Need (3 new instruction-level measurements)

From Table III of the paper, the dominant non-GEMM ops are:
1. **V_RED_SUM** — 25.7% of sampling cycles
2. **V_RED_MAX / V_RED_MAX_IDX** — 12.85% of sampling cycles (paper claims 6 cyc, sim=VECTOR_MAX_CYCLES=4)
3. **V_EXP_V** — 3.21% of sampling cycles (sim=VECTOR_EXP_CYCLES=6)

---

## Implementation Tasks

### Task 1: V_RED_SUM and V_RED_MAX latency test
**File**: `PLENA_RTL/src/vector_machine/test/reduction_compute_unit_tb.py`
**RTL module**: `fp_reduction_compute_unit` (VLEN=8, EXP=4, MANT=3)
**Interface**: `v_in_valid` → `s_out_valid` (reduction to scalar)
**What to add**: `@cocotb.test() async def reduction_latency_test(dut):`
  - Drive `v_in_valid=1`, pack VLEN FP values into `v_in`
  - Set `operation` to RED_SUM (enum value from operation.svh)
  - Measure cycles from `v_in_valid` → `s_out_valid` rising edge
  - Repeat for RED_MAX
  - Log: `[LATENCY] V_RED_SUM: N cycles vs sim VECTOR_SUM_CYCLES=20`
  - Log: `[LATENCY] V_RED_MAX: N cycles vs sim VECTOR_MAX_CYCLES=4`

**Enum values** (from operation.svh):
```
V_REDUCT_OP: need to check exact values
```

### Task 2: V_EXP_V latency test
**File**: `PLENA_RTL/src/vector_machine/test/fp_elementwise_compute_unit_tb.py`
**RTL module**: `fp_elementwise_compute_unit` (VLEN=8)
**Interface**: `v_in_a_valid + v_in_b_valid` → `v_out_valid`
**Operation**: `EXP_V_ELEMENT = 4'h4`
**What to add**: `@cocotb.test() async def exp_latency_test(dut):`
  - Drive both `v_in_a_valid` and `v_in_b_valid` (join2 requires both)
  - Set `operation = 4` (EXP_V_ELEMENT)
  - Measure cycles from assertion → `v_out_valid`
  - Log: `[LATENCY] V_EXP_V: N cycles vs sim VECTOR_EXP_CYCLES=6`

### Task 3: Update gemm_latency_comparison.py
**File**: `PLENA_Simulator/analytic_models/performance/gemm_latency_comparison.py`
**What to add**: Table 8 — Instruction-Level RTL vs Simulator Summary
  - Fill in measured values from Tasks 1+2
  - Include V_ADD_VV (already known=7), V_RED_MAX, V_RED_SUM, V_EXP_V
  - Show: RTL measured | Sim constant | sim value (VLEN=8) | match?

---

## Target Output Table (Table 8 in gemm_latency_comparison.py)

```
================================================================================
Table 8: Instruction-Level RTL vs Simulator Validation (non-GEMM ops)
  Config: VLEN=8, EXP_WIDTH=4, MANT_WIDTH=3, behavioral mode, 100MHz
================================================================================
  Instruction   RTL (cocotb)   Sim Constant          Sim Value   Match?
--------------------------------------------------------------------------------
  V_ADD_VV      7 cycles       VECTOR_ADD_CYCLES      7           EXACT ✓
  V_RED_MAX     ? cycles       VECTOR_MAX_CYCLES      4           ?
  V_RED_SUM     ? cycles       VECTOR_SUM_CYCLES      20          ?
  V_EXP_V       ? cycles       VECTOR_EXP_CYCLES      6           ?
================================================================================
```

---

## Layer-Level Evidence (for paper Table)

Using instruction measurements above, derive layer-level:

| Layer (seq=1, hidden=64, VLEN=64) | Sim cycles | RTL est | Source |
|-----------------------------------|------------|---------|--------|
| Linear (Q-proj, 1×64 @ 64×64)    | 80 cyc     | 86 cyc  | Table 6 ✓ |
| FFN SiLU (V_EXP_V × inter_dim/VLEN) | computed | computed | Task 2 |
| Flash Attn softmax (V_RED_MAX + V_EXP_V + V_RED_SUM) | computed | computed | Tasks 1+2 |

---

## Simulator Config Reference

From `configuration.svh` (behavioral mode, the default):
```
VECTOR_ADD_CYCLES  = 7   ← verified ✓
VECTOR_MUL_CYCLES  = 5
VECTOR_EXP_CYCLES  = 6   ← need verification
VECTOR_MAX_CYCLES  = 4   ← need verification
VECTOR_SUM_CYCLES  = 20  ← need verification
SCALAR_FP_EXP_CYCLES  = 2
SCALAR_FP_RECI_CYCLES = 2
```

Paper Table I uses (likely dc_lib mode, 1GHz):
```
VECTOR_ADD_CYCLES  = 2
VECTOR_EXP_CYCLES  = 1
VECTOR_MAX_CYCLES  = 1 → paper says V_RED_MAX_IDX = 6 total (= 6+3+1+1 alone formula?)
```

The comparison should show BOTH modes if possible.

---

## Files to Modify

1. `PLENA_RTL/src/vector_machine/test/reduction_compute_unit_tb.py` — add latency test
2. `PLENA_RTL/src/vector_machine/test/fp_elementwise_compute_unit_tb.py` — add latency test
3. `PLENA_Simulator/analytic_models/performance/gemm_latency_comparison.py` — add Table 8

## Files NOT to modify
- RTL source files (*.sv) — read-only
- ISA JSON — read-only
- configuration.svh — read-only
