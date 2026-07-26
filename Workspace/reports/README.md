# PLENA Modeling and Validation Reports

This directory is the canonical location for human-readable modeling and
validation reports produced in the PLENA Simulator workspace. Machine-readable
calibration artifacts remain next to the implementation that consumes them.

## Status Map

Interpret reports using these categories:

### Current canonical

- `system_validation_status.md`: authoritative integrated stack and claim
  boundary.
- `hbm_v4/hbm_dma_service_v4_full_report.md`: current single-chip
  production-DMA memory surrogate.
- `area/precision_aware_area_model_v4.md`: current pre-layout area proxy.
- `power/rtl_activity_power_candidate_v2.md` and
  `power/ideal_hierarchical_clock_gating_v1.md`: current architecture-level
  system-energy components and ideal/ungated clock bounds.
- `compiler/compact_stats_selector_overwrite_v1.md`,
  `compiler/partial_resident_kv_ideal_dual_port_dse.md`,
  `compiler/unified_affine_ffn_loop_lowering_v2.md`, and
  `compiler/moe_compiler_optimization_v2.md`: current compiler/hardware
  architecture path.
- `dse/tile_aware_multichip_lineage_audit_v4.md`: current corrected
  multi-chip controlled results. For N>1, HBM V4 is analytical traffic
  scaling rather than distributed request-manifest replay.

### Historical but valid

Staged compiler reports, `rtl-v1` timing reports, RTL-v2/v3 reports,
fixed-balanced MoE latency, DSE search/compression reports, and the legacy
analytic comparison remain valid for their stated source state and A/B
question. Their headline latency values are not current end-to-end results.

### Superseded

- `dse/factorized_multichip_model_v2.md` and the earlier ideal-linear
  multi-chip reports are reproducibility baselines under
  `analytic_models.legacy`; they are not selectable by the formal DSE CLI.
- `dse/tile_aware_multichip_model_v3.md` documents the initial implementation,
  but its original result tables are superseded by the v4 lineage audit.
- `compiler/packed_gqa_streaming_kmajor_agu_v2.md` is mixed-status:
  streaming/K-major remain current, while AGU-v2 is retired from the code.
- `power/rtl_activity_power_candidate_v1.md` is a rejected calibration
  candidate retained for failure analysis.

### Invalidated numeric claims

The old long-context values `TP4 x CP4 = 5.491 s` and
`TP1 x CP16 = 2.413 s` came from incomplete final-schedule lineage and must
not be reused. The corrected controlled values and explanation are in
`dse/tile_aware_multichip_lineage_audit_v4.md`.

Across every category, `CLOCK_PERIOD_PS=1000` is only a cycle-to-time
conversion assumption. Broadcast QK timing is nonzero but remains
`broadcast_rtl_unvalidated`; ideal-II1 and ideal hierarchical clock gating are
architectural assumptions, not cycle-exact or timing-closed RTL claims.

## Report Catalog

| Topic | Main report | Supporting evidence |
|---|---|---|
| Ideal II=1 compute timing rollout | [`model_latency/ideal_ii1_compute_timing_rollout.md`](model_latency/ideal_ii1_compute_timing_rollout.md) | Exact 2048/1024 reference under `analytic_models/performance/calibration/ideal_ii1_qwen3_32b_reference.json` |
| Transactional emulator `rtl-v1` timing | [`transactional_emulator/rtl_v1_latency_validation_full.md`](transactional_emulator/rtl_v1_latency_validation_full.md) | Summary Markdown and JSON in the same directory |
| Production-DMA HBM service model V4 | [`hbm_v4/hbm_dma_service_v4_full_report.md`](hbm_v4/hbm_dma_service_v4_full_report.md) | Calibration artifacts under `analytic_models/performance/calibration/` |
| Current integrated validation status | [`system_validation_status.md`](system_validation_status.md) | Canonical configuration, latest DSE result, and cross-model claim boundary |
| Final report/source audit | [`REPORT_AUDIT_20260726.md`](REPORT_AUDIT_20260726.md) | Report classification, implementation commit inventory, retained evidence, and validation summary |
| Multi-chip and Matrix-SRAM DSE v1 | [`dse/multi_chip_matrix_sram_dse_v1.md`](dse/multi_chip_matrix_sram_dse_v1.md) | TP+SP/TP-only assumptions, R-aware V4, NVLink lower bound, aggregate area, and end-to-end smoke evidence |
| Four-objective multi-chip energy DSE v1 | [`dse/four_objective_multichip_energy_dse_v1.md`](dse/four_objective_multichip_energy_dse_v1.md) | Latency/area/energy/accuracy objectives, aggregate energy partition, MO-TPE search, selectors, and validation |
| Tile-aware multi-chip lineage audit v4 | [`dse/tile_aware_multichip_lineage_audit_v4.md`](dse/tile_aware_multichip_lineage_audit_v4.md) | Final-schedule compute/action/DMA lineage repair, per-rank power aggregation, corrected short/long A/B, and distributed-HBM claim boundary |
| Tile-aware TP x CP x EP multi-chip model v3 | [`dse/tile_aware_multichip_model_v3.md`](dse/tile_aware_multichip_model_v3.md) | Rank-local compiler tile reconstruction, causal CP slabs, fixed-balanced MoE EP, all-rank power, v2/v3 A/B, and default DSE rollout |
| Searchable TP x CP multi-chip model v2 | [`dse/factorized_multichip_model_v2.md`](dse/factorized_multichip_model_v2.md) | Conditional TP/CP and NVLink-port search, causal zigzag partition, role-aware HBM, communication/area/energy assumptions, 64-point DSE smoke, and controlled lower-bound ablation |
| Canonical conditional DSE strategy v1 | [`dse/canonical_conditional_search_v1.md`](dse/canonical_conditional_search_v1.md) | Legal conditional BLEN/SRAM encoding, COMPLETE-trial budgets, shared caches, aggressive memory-aware workers, and short/long launcher |
| Short-context results and long CostEmitter analysis | [`dse/short_context_482x16_results_and_long_costemitter_analysis.md`](dse/short_context_482x16_results_and_long_costemitter_analysis.md) | Completed 2,048-point short DSE, selected designs, long-run stop diagnosis, and algebraic CostEmitter optimization requirements |
| Long-context CostEmitter algebraic compression v1 | [`dse/long_context_costemitter_algebraic_compression_v1.md`](dse/long_context_costemitter_algebraic_compression_v1.md) | Exact ideal-II1 trace/V4 compression, parity evidence, worst-point cold/warm/RSS acceptance, and parallel DSE smoke results |
| Long-context DSE domain, V4, and resource scheduling v2 | [`dse/long_context_dse_domain_v4_resource_v2.md`](dse/long_context_dse_domain_v4_resource_v2.md) | 26-shape M256-M4096 domain, exact V4 sufficient-statistics aggregation, progress-aware timeout, completed 2,048-point study, and 64-worker persistent-pool rollout |
| DSE artifact compaction and FFN address lowering v1 | [`dse/dse_artifact_compaction_ffn_address_lowering_v1.md`](dse/dse_artifact_compaction_ffn_address_lowering_v1.md) | Resume-safe compact artifacts, canonical report deduplication, large-MLEN FFN pointer-liveness fix, fixed-point A/B, and 64-worker smoke |
| Long-context DSE anomaly audit v1 | [`dse/long_context_dse_anomaly_audit_v1.md`](dse/long_context_dse_anomaly_audit_v1.md) | Old/new 2,048-point monotonicity audit, resolved M8192 anomaly, remaining projection-full SRAM reversal, and selector claim boundary |
| Precision-aware area model v4 | [`area/precision_aware_area_model_v4.md`](area/precision_aware_area_model_v4.md) | Structural MatrixMachine fit, SRAM macros, top residual, and full-chip holdout |
| RTL-activity on-chip power candidate v2 | [`power/rtl_activity_power_candidate_v2.md`](power/rtl_activity_power_candidate_v2.md) | Accepted Qwen-like action-energy model, grouped holdouts, activity envelope, and validation boundaries |
| Ideal hierarchical clock-gating power v1 | [`power/ideal_hierarchical_clock_gating_v1.md`](power/ideal_hierarchical_clock_gating_v1.md) | ClockWork accounting, ungated upper bound, ideal architectural lower bound, and three production DSE smokes |
| Rejected on-chip power candidate v1 | [`power/rtl_activity_power_candidate_v1.md`](power/rtl_activity_power_candidate_v1.md) | First replay experiment, failure analysis, and preserved bootstrap path |
| Qwen3-32B/235B fixed-balanced latency | [`model_latency/qwen3_fixed_balanced_latency_report.md`](model_latency/qwen3_fixed_balanced_latency_report.md) | Compact benchmark JSON/CSV under `Workspace/qwen3_fixed_balanced_latency/results/` |
| Legacy analytic model vs CostEmitter | [`model_latency/legacy_analytic_vs_costemitter_m2048_b1024.md`](model_latency/legacy_analytic_vs_costemitter_m2048_b1024.md) | Controlled 2048/1024/2048 comparison, one-cycle Vector/Scalar counterfactual, and restored-model assessment |
| Native decoder compact batch/head layout | [`compiler/native_decoder_compact_layout.md`](compiler/native_decoder_compact_layout.md) | CostEmitter A/B JSON and transactional packed-GQA tests |
| Native packed-GQA schedule optimization v2 | [`compiler/native_packed_gqa_optimization_v2.md`](compiler/native_packed_gqa_optimization_v2.md) | Schedule A/B JSON/CSV and transactional bitwise parity tests |
| Packed-GQA pipeline-aware compiler optimization v1 | [`compiler/packed_gqa_pipeline_optimization_v1.md`](compiler/packed_gqa_pipeline_optimization_v1.md) | Row-interleaved RTL-v3 scheduling, short/long-context A/B, transactional bitwise parity, and work invariants |
| Packed-GQA streaming, K-major broadcast, and AGU v2 | [`compiler/packed_gqa_streaming_kmajor_agu_v2.md`](compiler/packed_gqa_streaming_kmajor_agu_v2.md) | Short/long-context incremental A/B, bitwise transactional evidence, enlarged Scalar FP SRAM, broadcast validation boundary, and paired AGU-v2 DC result |
| Partial-resident K/V and ideal dual-port SRAM DSE | [`compiler/partial_resident_kv_ideal_dual_port_dse.md`](compiler/partial_resident_kv_ideal_dual_port_dse.md) | Shared prefix-cache planner, policy search, exact traffic sweep, CostEmitter anchors, and ideal SRAM-port assumption |
| Compact statistics SIMD, selector hoisting, and reduction overwrite v1 | [`compiler/compact_stats_selector_overwrite_v1.md`](compiler/compact_stats_selector_overwrite_v1.md) | Eight-arm short-context A/B, long-context results, opcode/HBM invariants, RTL-v4 timing, paired DC, and power delta |
| Unified affine FFN loop lowering v2 | [`compiler/unified_affine_ffn_loop_lowering_v2.md`](compiler/unified_affine_ffn_loop_lowering_v2.md) | Capacity-independent FFN IR, controlled A/B, bitwise parity, complete 2,048-point DSE, zero SRAM regressions in 123 matched comparisons, plots, and Matrix/DMA invariants |
| Flattened MatrixMachine shape sweep v1 | [`compiler/flatten_shape_sweep_v1/flatten_shape_sweep_v1.md`](compiler/flatten_shape_sweep_v1/flatten_shape_sweep_v1.md) | Latest combined compiler path across 28 equal-PE/equal-Matrix-SRAM short/long-context shape points, with area, tail, HBM, and component attribution |
| Non-Matrix optimization opportunities | [`compiler/non_matrix_optimization_opportunities.md`](compiler/non_matrix_optimization_opportunities.md) | Current stage/opcode evidence and ranked compiler/small-hardware opportunities after restoring AGU v1 as the default |
| Native Vector/Scalar compiler optimization v1 | [`compiler/native_vector_scalar_optimization_v1.md`](compiler/native_vector_scalar_optimization_v1.md) | Bitwise/transactional validation, one-layer resource A/B, and complete 13,905-point DSE |
| Vector/Scalar RTL optimization v2 | [`rtl/vector_scalar_rtl_v2_optimization.md`](rtl/vector_scalar_rtl_v2_optimization.md) | RTL architecture, 130-point timing artifact, transactional bitwise A/B, and target-point latency attribution |
| Vector/Scalar RTL-v3 segment parallelism | [`rtl/vector_scalar_rtl_v3_segment_parallel.md`](rtl/vector_scalar_rtl_v3_segment_parallel.md) | Multi-segment reduction, compact stats, Scalar ROB, 247 RTL measurements, and target-point A/B evidence |
| Six-stream loop AGU v1 | [`rtl/loop_agu_v1.md`](rtl/loop_agu_v1.md) | ISA/RTL/compiler integration, 21.73% ideal-II1 latency reduction, paired DC area, invariants, and limitations |
| RTL-v3 working-tree summary (Chinese) | [`rtl/vector_scalar_rtl_v3_working_tree_summary_cn.md`](rtl/vector_scalar_rtl_v3_working_tree_summary_cn.md) | Concise optimization list, measured benefits, validation status, and claim boundaries since the latest commits |
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
