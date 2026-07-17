# PLENA RTL opcode timing calibration

This directory contains cycle-count harnesses for the transactional emulator's
`rtl-v1` timing model.  The harnesses run against a temporary copy of
`PLENA_RTL`; they never patch the source checkout in place.

The committed timing artifact is:

```text
transactional_emulator/calibration/rtl_opcode_timing_v1.json
```

Despite the historical filename, schema version 3 separates backend resource
occupancy from result-ready latency and records whether a path is measured,
structural, or unsupported by current RTL. All cycle measurements use the
complete production Machine boundary rather than a leaf arithmetic primitive.

`rtl-v1` is the default transactional timing mode after the full-Machine and
scheduler acceptance suite below passes. Use `--timing-mode legacy` only for
an explicit historical baseline. Numerical comparison and its PASS thresholds
are identical in both modes.

Run a representative smoke calibration:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode smoke \
  --out-dir /tmp/plena_rtl_timing_smoke_results
```

Run the full-Machine MatrixMachine, VectorMachine, and ScalarMachine sweeps:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode full \
  --harness matrix_machine_full_timing.py \
  --harness vector_machine_full_timing.py \
  --harness scalar_machine_full_timing.py \
  --out-dir Workspace/rtl_v1_latency_validation/full_machine \
  --resume
```

Run one harness while developing or extending a formula:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode full \
  --harness mxfp_mcu_timing.py \
  --out-dir /tmp/plena_rtl_timing_mxfp_full
```

Outputs include one log per harness and `raw_measurements.json`.  Behavioral
RTL is used by default. `DC_LIB_EN` primitive timing is deliberately marked
unverified until a gate-level/DC-library timing harness is available.

Run the production `pipeline_control.sv` hazard/recovery harness separately:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode smoke \
  --harness pipeline_control_timing.py \
  --out-dir Workspace/rtl_v1_latency_validation/pipeline_control
```

The scheduler tests live in the emulator binary target, not the lightweight
DMA library target. Use this command so the tests are actually executed:

```bash
nix develop --command bash -lc \
  'cd transactional_emulator && cargo test --bin transactional_emulator'
```

Every `rtl-v1` testbench run writes a constant-size
`rtl_validation_summary.json`. To turn unsupported or out-of-domain timing into
a post-run failure while still producing state and comparison artifacts, pass:

```bash
python transactional_emulator/testbench/run_model.py <nickname> \
  --timing-mode rtl-v1 \
  --require-rtl-validated
```

This policy does not implicitly retain a full event trace. Add
`--event-trace` only when per-instruction debugging is required.

Generate the six scheduler differential traces:

```bash
nix develop --command bash -lc '
  cd transactional_emulator
  RTL_SCHEDULER_TRACE_OUT=../Workspace/rtl_v1_latency_validation/scheduler_differential_traces.json \
    cargo test --bin transactional_emulator \
      scheduler::tests::emit_rtl_scheduler_validation_trace -- \
      --ignored --exact
'
```

Finally, combine RTL microbenchmarks, scheduler traces, and optional system-run
artifacts into the auditable JSON/Markdown report:

```bash
python transactional_emulator/testbench/rtl_timing/run_qwen_rtl_v1_validation.py

python transactional_emulator/testbench/rtl_timing/report_rtl_v1_latency_validation.py \
  --current-stats Workspace/qwen3_32b_transactional_prefetch_sweep/runs/rtl_v1_validation_20260714/single_point/rust_emulator_run_stats.json \
  --fixed-stats Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/rust_emulator_run_stats.json \
  --current-profile Workspace/qwen3_32b_transactional_prefetch_sweep/runs/rtl_v1_validation_20260714/single_point/memory_profile.json \
  --fixed-profile Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/memory_profile.json \
  --current-comparison Workspace/qwen3_32b_transactional_prefetch_sweep/runs/rtl_v1_validation_20260714/single_point/comparison_results.json \
  --fixed-comparison Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/comparison_results.json \
  --functional-regression Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/functional_regression.json \
  --strict \
  --out-dir Workspace/reports/transactional_emulator
```

Current RTL limitations are intentionally visible rather than assigned made-up
validated timings. In particular, `S_MAX_FP` is absent and MXFP `M_MM_WO`
emits only one architectural write-valid pulse although `data_flow_control`
expects a `BLEN`-beat Vector SRAM write burst. The emulator can execute these
paths for functional debugging, but reports the run as `unsupported_opcodes`;
all `M_MM_WO` rows conservatively become readable at backend idle. This avoids
turning an incomplete RTL write protocol into a fabricated row cadence.
