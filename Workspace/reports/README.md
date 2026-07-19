# PLENA Modeling and Validation Reports

This directory is the canonical location for human-readable modeling and
validation reports produced in the PLENA Simulator workspace. Machine-readable
calibration artifacts remain next to the implementation that consumes them.

## Reports

| Topic | Main report | Supporting evidence |
|---|---|---|
| Transactional emulator `rtl-v1` timing | [`transactional_emulator/rtl_v1_latency_validation_full.md`](transactional_emulator/rtl_v1_latency_validation_full.md) | Summary Markdown and JSON in the same directory |
| Production-DMA HBM service model V4 | [`hbm_v4/hbm_dma_service_v4_full_report.md`](hbm_v4/hbm_dma_service_v4_full_report.md) | Calibration artifacts under `analytic_models/performance/calibration/` |
| Current integrated validation status | [`system_validation_status.md`](system_validation_status.md) | Canonical configuration, latest DSE result, and cross-model claim boundary |
| Precision-aware area model v4 | [`area/precision_aware_area_model_v4.md`](area/precision_aware_area_model_v4.md) | Structural MatrixMachine fit, SRAM macros, top residual, and full-chip holdout |
| Qwen3-32B/235B fixed-balanced latency | [`model_latency/qwen3_fixed_balanced_latency_report.md`](model_latency/qwen3_fixed_balanced_latency_report.md) | Compact benchmark JSON/CSV under `Workspace/qwen3_fixed_balanced_latency/results/` |
| Native decoder compact batch/head layout | [`compiler/native_decoder_compact_layout.md`](compiler/native_decoder_compact_layout.md) | CostEmitter A/B JSON and transactional packed-GQA tests |
| Native packed-GQA schedule optimization v2 | [`compiler/native_packed_gqa_optimization_v2.md`](compiler/native_packed_gqa_optimization_v2.md) | Schedule A/B JSON/CSV and transactional bitwise parity tests |
| Native Vector/Scalar compiler optimization v1 | [`compiler/native_vector_scalar_optimization_v1.md`](compiler/native_vector_scalar_optimization_v1.md) | Bitwise/transactional validation, one-layer resource A/B, and complete 13,905-point DSE |
| Published PLENA prefill result audit | [`paper/plena_prefill_result_audit.md`](paper/plena_prefill_result_audit.md) | Reproducibility questions and conservative throughput-floor checks |

## Claim Policy

The reports distinguish four different validation levels:

1. Unit and request-geometry parity tests.
2. Held-out microbenchmark prediction error.
3. Production Qwen trace validation.
4. Full-system or DSE-level validation.

A result at one level must not be presented as evidence for a stronger level.
In particular:

- The 1 ns clock conversion is a reporting assumption, not an RTL timing
  closure result.
- The HBM V4 model is a post-hoc production-DMA service surrogate, not online
  cross-queue Ramulator co-simulation.
- Scheduled makespan error can be much smaller than HBM work error when a
  workload is compute-bound.
- A CostEmitter stage is a compiler semantic region containing one or more
  kernels. `sum(max(stage compute, stage memory))` can overestimate a scheduled
  timeline when work overlaps across stage boundaries; it is not a universal
  roofline lower bound.
- Numerical execution and the existing correctness gate were not changed by
  either latency-modeling effort.
