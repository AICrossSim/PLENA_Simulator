# Final PLENA Prefill Report and Source Audit

## Scope

This audit closes the 2026-07-26 workspace consolidation. It classifies the
retained reports, records the implementation commits, and states which
evidence may support current PLENA prefill claims. `PLENA_Tools` is explicitly
outside this work and its local submodule state was not modified or committed.

The authoritative integrated description is
[`system_validation_status.md`](system_validation_status.md). This file is the
source/report inventory, not a replacement performance report.

## Current Canonical Stack

```text
compiler profile          current-dse-v1
compute timing            ideal-ii1
RTL sensitivity           rtl-v1 ordered hazard-aware timing
Vector/Scalar lowering    RTL-v4
attention                 streamed-v2 + broadcast-k-major-v1
address generation        loop-agu-v1
FFN                       affine-loop-v2 + live-stride-v1
MoE                       compact-route-v2
memory                    production-DMA HBM V4
CostTrace                 detailed validation / affine-summary DSE
multi-chip                tile-aware-tp-cp-ep-v3 with final lineage audit
SRAM                      partial K/V residency; ideal dual-port DSE default
power                     action-energy v2 + HBM3E + interconnect proxy
clock power               ideal hierarchical nominal / ungated upper bound
DSE objectives            min latency, min area, min energy, max accuracy
```

The following claim boundaries apply to every current result:

- `CLOCK_PERIOD_PS=1000` is a conversion assumption, not timing closure.
- Vector/Scalar/control II=1 and ideal clock gating are architectural
  assumptions.
- K-major Matrix broadcast has nonzero ordinary-Matrix structural timing but
  remains `broadcast_rtl_unvalidated`.
- N>1 HBM V4 is rank-local analytical traffic/service scaling, not
  distributed online Ramulator replay.
- Area is a pre-layout structural proxy. Ideal dual-port SRAM excludes
  physical dual-port overhead.
- Power uses RTL activity on mapped DC netlists plus literature HBM/interconnect
  parameters; it excludes CTS, routing, package, SRAM leakage, and PHY
  uncertainty.

## Report Classification

### Current Canonical

- `area/precision_aware_area_model_v4.md`
- `compiler/compact_stats_selector_overwrite_v1.md`
- `compiler/moe_compiler_optimization_v2.md`
- `compiler/partial_resident_kv_ideal_dual_port_dse.md`
- `compiler/unified_affine_ffn_loop_lowering_v2.md`
- `dse/long_context_costemitter_algebraic_compression_v1.md`
- `dse/tile_aware_multichip_lineage_audit_v4.md`
- `hbm_v4/hbm_dma_service_v4_full_report.md`
- `model_latency/ideal_ii1_compute_timing_rollout.md`
- `power/ideal_hierarchical_clock_gating_v1.md`
- `power/rtl_activity_power_candidate_v2.md`
- `rtl/loop_agu_v1.md`
- `system_validation_status.md`

### Historical but Valid

- Compact layout, packed-attention, compiler-v1 Vector/Scalar, flattened-shape,
  pipeline-scheduling, RTL-v2/v3, and fixed-balanced reports retain staged A/B
  evidence for their stated source state.
- `rtl-v1` transactional reports retain measured hazard-aware and
  observed-DMA validation evidence. They are sensitivity/validation reports,
  not the formal DSE timing policy.
- Canonical-search, artifact-compaction, resource-scheduling, and old
  short/long DSE reports retain implementation and performance-engineering
  evidence. Their old Pareto points are not current selectors.
- `paper/plena_prefill_result_audit.md` remains a methodology and
  reproducibility note rather than a simulator result.

### Superseded or Mixed Status

- `compiler/packed_gqa_streaming_kmajor_agu_v2.md`: streamed softmax and
  K-major scheduling remain current; AGU-v2 is retired.
- `dse/dse_artifact_compaction_ffn_address_lowering_v1.md`: artifact
  compaction remains current; its first FFN fix is superseded by
  `affine-loop-v2`.
- `dse/factorized_multichip_model_v2.md` and
  `dse/multi_chip_matrix_sram_dse_v1.md`: historical analytical baselines now
  isolated under `analytic_models.legacy`.
- `dse/tile_aware_multichip_model_v3.md`: implementation rationale remains
  useful; its original tables are superseded by the v4 lineage audit.
- `power/rtl_activity_power_candidate_v1.md`: rejected calibration candidate
  retained for failure analysis.

### Invalidated Results

The old long-context values `TP4 x CP4 = 5.491 s` and
`TP1 x CP16 = 2.413 s` were produced before final transformed-schedule lineage
was complete. They must not be used as current evidence. The corrected
controlled values and cause are recorded in
`dse/tile_aware_multichip_lineage_audit_v4.md`.

## Implementation Commits

```text
PLENA_RTL
  b5feafa  Add pipelined scalar and segment-parallel vector datapaths

PLENA_Compiler
  7e29ab5  Extend PLENA ISA assembly for RTL-v4 and loop AGU v1
  6d100dc  Optimize native packed-GQA lowering for prefill
  01762d8  Unify affine FFN lowering and partial K/V residency
  bd7b656  Optimize MoE lowering and rebuild CostTrace lineage
  1470478  Consolidate canonical compiler schedule profiles

PLENA_Simulator
  af4737e  Update PLENA compiler dependency for optimized prefill lowering
  df104a0  Synchronize the transactional emulator with RTL-v4
  e107098  Add versioned RTL timing calibration artifacts
  ad40b49  Calibrate structural area overlays for prefill hardware
  9ccca28  Add action-based on-chip and external-memory power models
  9db3f15  Integrate HBM V4 with compressed compiler cost traces
  5c78b2b  Add tile-aware TP-CP-EP analytical scaling
  53e1126  Modularize canonical four-objective DSE execution
  4ad7df8  Quarantine historical analytic models and generated artifacts
  4329bf5  Restore scalar ROB scheduling with RTL-v4 timing
```

## Validation Summary

- RTL syntax/elaboration check: 130 modules passed.
- Compiler focused assembler, layout, GQA, FFN, MoE, and CostTrace suites
  passed at their corresponding staged commits.
- Transactional emulator: 152 Rust tests passed, one evidence-only test
  ignored.
- Timing artifact parity: 43 tests passed.
- Area model and overlay tests: 35 passed.
- Power model tests: 57 passed.
- HBM V4 and compressed-trace tests: 43 passed.
- Tile-aware multi-chip tests: 17 passed.
- Modular DSE and legacy isolation tests: 37 passed.
- Final compatibility repair: 9 Scalar-pipeline tests plus the historical V3
  current-trace boundary test passed.
- DSE smoke: `3 COMPLETE / 1 worker` and `12 COMPLETE / 4 workers`, with zero
  prune/fail in the final canonical schema.

These checks establish implementation consistency and model invariants. They
do not convert the architecture-level assumptions above into signoff evidence.

## Retained Machine-Readable Evidence

Only compact evidence referenced by reports is intended for version control:

- `compiler/compact_stats_selector_overwrite_v1_results.json`
- `compiler/compact_stats_selector_overwrite_v1_{opcode,dc,power}_delta.csv`
- `compiler/flatten_shape_sweep_v1/flatten_shape_sweep_v1.csv`
- `compiler/moe_compiler_optimization_v2_results.json`
- `compiler/moe_compiler_optimization_v2_opcode_delta.csv`
- `compiler/partial_resident_kv_policy_sweep.csv`
- `compiler/unified_affine_ffn_loop_v2_no_regression_audit.json`
- the three small controlled `unified_affine_ffn_loop_v2_*.json` anchors
- `dse/tile_aware_multichip_lineage_audit_v4_results.json`
- `dse/tile_aware_multichip_lineage_audit_v4_ablation.csv`
- `power/ideal_hierarchical_clock_gating_v1_validation.json`

The retained figures are the partial-residency policy sweep and the three
unified-FFN latency/area/energy views. They are final report illustrations,
not serialized trial databases.

Trial directories, worker journals, caches, pickles, VCDs, mapped builds,
temporary settings, local PDFs, and failed candidates are excluded.
