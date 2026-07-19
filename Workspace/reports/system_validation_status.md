# PLENA Prefill Modeling: Integrated Validation Status

## Document Status

This document records the integrated implementation as of 2026-07-19. It is
the entry point for interpreting the detailed area, latency, memory, compiler,
and DSE reports. Earlier reports remain useful A/B evidence, but their
intermediate best-latency values are not the current end-to-end result.

## Current Analysis Path

The production analysis path is:

```text
Qwen3 model/config
  -> native compiler compact batch/head layout
  -> direct-first-block-v1 packed attention
  -> compiler-v1 vector/scalar lowering
  -> CostEmitter dynamic opcode and DMA trace
  -> rtl-v1 calibrated compute resource work
  -> production-DMA V4 per-stage memory work
  -> sum(stage max(compute, memory)) latency estimate
  -> precision-aware area_new proxy
```

The corresponding DSE settings are:

```text
seq_len=482, batch_size=16, decoder layers=64
VLEN=MLEN
MLEN=[128,256,512,1024,2048]
BLEN divides MLEN, up to MLEN
103 measured software precision profiles with accuracy > 0.9
INT_DATA_WIDTH=[16,32,64]
A100 reference=826 mm2; feasibility budget=908.6 mm2
```

The complete sharded grid contains 13,905 unique points: 12,051 complete,
1,854 constraint-pruned, and zero failed. The final merge found no duplicate
or missing design keys.

## Reproducible Source State

The implementation summarized here is split into reviewable commits:

```text
PLENA_Compiler
  1c1d77a  Add native Qwen3 MoE and compact prefill lowering

PLENA_Simulator
  8656edd  Add structural MatrixMachine area model v4
  112d6f5  Integrate compact compiler traces with rtl-v1 and HBM V4
  b976fa1  Scale Qwen3 precision DSE with compact CostEmitter models

PLENA_RTL
  3e69165  Make PLENA RTL precision profiles synthesis-safe
  0beb43f  Add reproducible RTL checking and DC area workflows
```

The complete Compiler suite passed 98 tests. The Simulator-focused area,
performance, DSE, runner-contract, packed-GQA and MoE suite passed 84 tests.
The complete `SimTop` Verilator build passed, as did all three directly
modified `fp_adder`/`fp_mult` parameter configurations. Two legacy cp-unit FP
tests have stale ready-signal expectations, and the data-flow-control unit
test references a wrapper source absent from the RTL repository; those test
asset gaps remain disclosed rather than counted as RTL failures or successes.

## Current Qwen3-32B Result

The minimum modeled latency is 16,405.355 ms. It is shared by 72 precision and
integer-width configurations at `MLEN=VLEN=2048, BLEN=1024`. Selecting the
highest-accuracy tied profile gives:

```text
Weight / ACT / KV / FP = MXFP_E2M1 / MXFP_E1M2 / MXFP_E4M3 / FP_E6M5
INT_DATA_WIDTH          = 16
accuracy                = 0.98
nominal area            = 846.531 mm2
P90 area                = 854.467 mm2
latency                 = 16.405355 s
```

The fastest all-MXINT configuration uses MXINT4 weight/ACT/KV, FP_E6M5 and
the same `2048/1024` hardware. It predicts 16.477759 s and 646.277 mm2. This
supports the expected area-density advantage for MXINT in the current RTL,
while showing that the latency difference between arithmetic families is small
under the Stage-1 opcode timing used here.

The point closest to 826 mm2 is 834.651 mm2 with 16.405365 s latency. The area
budget is a comparison constraint, not an instruction to maximize die area;
the report therefore distinguishes fastest-under-budget from closest-to-A100.

## Compiler Improvement Chain

All values use the same Qwen3-32B workload and area/memory calibration unless
otherwise stated:

| Compiler state | Best modeled latency | Main change |
|---|---:|---|
| Pre-compaction | 36.678 s | One MLEN-padded slab per batch and sparse GQA columns |
| Compact batch/head layout | 21.706 s | Physical rows fixed at 8,192 and Q/O width fixed at 4,096 |
| Direct first-block packed attention | 18.503 s | First softmax block specialization and direct packed-O accumulation |
| Vector/scalar compiler-v1 | 16.405 s | Grouped segmented norm, active rows, and redundant-mask removal |

For the final `2048/1024` point, compiler-v1 reduces one-layer calibrated
resource work from 294,530,913 to 259,968,410 cycles. Matrix and HBM operation
counts are unchanged; Vector work falls by 10.17% and Scalar work by 19.37%.

## Latency Evidence and Limits

`rtl-v1` replaced MLEN-based Matrix opcode delays with measured full-Machine
timing, added completion-aware DMA, and introduced a hazard scoreboard. Tested
RTL microbenchmarks match ready/done/II boundaries within one cycle. An
observed-DMA CostEmitter replay matches the transactional timeline exactly at
25,452,302 cycles when both consume the same measured DMA completion sequence.

The formal fast compute quantity is nevertheless **serial resource work**, not
a cycle-exact program makespan. At the current fastest DSE point, only 17.50%
of resource cycles are from directly measured full-Machine opcode timing;
79.90% are structural extrapolations and 2.60% are unsupported-RTL timing
fallbacks. Production sizes are outside the measured opcode domain. These
fractions must accompany any absolute latency claim.

V4 memory prediction has a 585-sample generic holdout median error of 3.48%,
P95 of 18.10%, weighted MAPE of 2.80%, and store P95 of 19.92%. It is post-hoc:
it predicts production DMA service occurrences but does not preserve a single
online Ramulator state across all overlapping queues. The stage-wise roofline
is fast enough for DSE but was about 18% conservative relative to compressed
Stage 2 in the tested 32B/235B cases. That difference is scheduler omission,
not a V4 memory-error measurement or an absolute RTL-error measurement.

## Area Evidence and Limits

The nominal area is:

```text
structural MatrixMachine logic
+ fitted Vector/Scalar/HBM logic
+ proportional top-level logic residual
+ tiled ASAP7 SRAM macro area
```

Matrix structural-v4 grouped holdout errors are 2.92% median / 9.78% P95 for
MXINT and 0.79% median / 2.36% P95 for MXFP. Five held-out small full-chip
anchors give 2.15% corrected-logic MAPE and 1.77% corrected composite MAPE.
The top residual is fit on 12 separate anchors.

The DSE optimum is a structural extrapolation: DC MatrixMachine anchors cover
only MLEN 16-64 and BLEN 4-16. Large results follow exact RTL instance-count
growth and non-negative leaf fits, but no `2048/1024` design was synthesized.
Area excludes placement/routing overhead, clock/power grids, pads, HBM PHY,
HBM stacks, interposer and package. The 826 mm2 comparison is therefore an
early architectural reference, not a signoff die-area equivalence.

## Numerical Correctness Boundary

Compiler scheduling optimizations retain the existing comparison gate. Tiny
packed-GQA transactional cases cover batch co-packing, shared GQA storage,
dummy tails, partial K tiles and cross-batch isolation. The final vector/scalar
implementation was selected only after stage checkpoints rejected a
non-bitwise square-sharing variant.

Timing validation and numerical correctness are separate. Unsupported ISA/RTL
operations, mixed-precision hardware capability assumptions and production
shape extrapolation prevent describing the full Qwen3-32B estimate as
cycle-exact silicon validation.

## Canonical Artifacts

- DSE: `Workspace/qwen3_32b_dense_analytic/runs/roofline_v4_grid_vector_scalar_v1_sharded_20260719/`
- Area: `analytic_models/area_new/calibration/` and `Workspace/area_new_validation/matrix_structural_v4_20260718/`
- RTL timing: `Workspace/reports/transactional_emulator/rtl_v1_latency_validation_full.md`
- HBM V4: `Workspace/reports/hbm_v4/hbm_dma_service_v4_full_report.md`
- Compiler optimization: `Workspace/reports/compiler/`
- Fixed-balanced MoE: `Workspace/reports/model_latency/qwen3_fixed_balanced_latency_report.md`
