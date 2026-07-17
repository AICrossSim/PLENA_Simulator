# PLENA Modeling and Validation Reports

This directory is the canonical location for human-readable modeling and
validation reports produced in the PLENA Simulator workspace. Machine-readable
calibration artifacts remain next to the implementation that consumes them.

## Reports

| Topic | Main report | Supporting evidence |
|---|---|---|
| Transactional emulator `rtl-v1` timing | [`transactional_emulator/rtl_v1_latency_validation_full.md`](transactional_emulator/rtl_v1_latency_validation_full.md) | Summary Markdown and JSON in the same directory |
| Production-DMA HBM service model V4 | [`hbm_v4/hbm_dma_service_v4_full_report.md`](hbm_v4/hbm_dma_service_v4_full_report.md) | Calibration artifacts under `analytic_models/performance/calibration/` |

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
- Numerical execution and the existing correctness gate were not changed by
  either latency-modeling effort.

