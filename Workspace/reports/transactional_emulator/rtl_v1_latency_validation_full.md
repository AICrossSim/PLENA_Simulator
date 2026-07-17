# Transactional Emulator `rtl-v1` Latency Fidelity: Implementation and Validation Report

## 1. Document Information

| Item | Value |
|---|---|
| Report date | 2026-07-15 |
| Scope | Transactional emulator latency fidelity |
| Explicitly out of scope | Correctness gate, golden output, tolerances, and PASS thresholds |
| RTL repository | `/home/yh3525/FYP/PLENA_RTL` |
| RTL HEAD | `823d25adf46fc9c44e5b3f807371904743c473fe` |
| RTL state | Dirty working tree |
| RTL diff SHA-256 | `54eb1c7ff5e21e58a8dd879efa65beca6bf4592f97614dfb67cd50159d677eab` |
| Timing calibration artifact | `transactional_emulator/calibration/rtl_opcode_timing_v1.json` |
| Clock conversion | `CLOCK_PERIOD_PS=1000`, reported as a 1 GHz assumption |
| Fmax status | Not validated by synthesis or static timing analysis |

Cycle counts are the primary results in this report. Values in ns or ms are obtained by multiplying cycles by 1 ns. They do not demonstrate that the current RTL closes timing at 1 GHz.

## 2. Executive Summary

The original transactional emulator used functional execution time as hardware latency. Every opcode was executed and awaited before the next opcode could proceed. HBM DMA, MatrixMachine, VectorMachine, and ScalarMachine operations were therefore serialized even when the RTL permits them to overlap. Several opcode delays were also inconsistent with the current RTL. In particular, `M_MM` was charged primarily as a function of `MLEN`, Matrix writeout was fixed at one cycle, and a Ramulator write completed when the request was accepted rather than when its completion callback fired.

`rtl-v1` preserves the original numerical execution path and adds a separate RTL-oriented timing timeline:

1. Matrix, Vector, and Scalar latencies are calibrated using behavioral RTL microbenchmarks around production Machine top modules.
2. The timing model uses in-order issue, asynchronous completion events, operand readiness, and a resource scoreboard.
3. Stall, hazard, and one-cycle post-stall recovery behavior are derived from `pipeline_control.sv`.
4. Ramulator reads and writes both wait for completion callbacks, and all pending transactions are drained at program end.
5. DMA accounting uses actual MXINT/MXFP bit widths and coalesces accesses to 64-byte physical lines.
6. Profiling separates resource work from mutually exclusive critical-path contributions.
7. Unsupported and out-of-calibration-domain opcodes remain executable for functional debugging but are not reported as RTL validated.

For the tested Qwen3-32B decoder layer, the legacy functional executor reports 71,503,360 cycles and the current `rtl-v1` timeline reports 25,452,002 cycles. The 64.40% difference is primarily caused by correcting the MLEN-based `M_MM` timing and representing legal resource concurrency. It must not be presented as a demonstrated 2.81x hardware speedup. Only 19.18% of the production trace satisfies the current validation policy, and Ramulator timing is still placed on the scheduler timeline post hoc. The full-layer result is therefore not a full-system cycle-exact claim.

CostEmitter now consumes the same timing artifact and emits the same ordered
schedule semantics. For a second Qwen run with a compact observed-DMA trace,
Python and Rust match all opcode/resource work, all stall categories, all
critical-path categories, and the 25,452,302-cycle makespan exactly. The fast
Stage-1 DSE objective is calibrated serial resource work (0.364 ms per cached
evaluation); the 112.855-second exact ordered replay remains an opt-in
validation path.

## 3. Scope and Claim Boundary

### 3.1 Included changes

- Opcode latency.
- Memory request completion timing.
- Pipeline and resource hazards.
- Memory/compute overlap.
- Timeline profiling and timing provenance.

### 3.2 Unchanged behavior

- Functional tensor operations.
- Quantization numerical behavior.
- Golden tensors.
- Comparison `atol` and `rtol`.
- Numerical PASS thresholds.
- RTL implementation of unsupported opcodes.

### 3.3 Measurement boundary

Machine microbenchmarks use the following definitions:

```text
start = opcode/control and valid operands are accepted at the production Machine top
ready = the result is safe for a dependent consumer
done  = the backend resource is completely idle
II    = interval before the next independent instruction can be accepted
```

The Matrix harness instantiates the production `matrix_machine`, including its alignment buffers, MCU, reduction path, and result buffer. It supplies operand streams directly, so it does not include frontend decode, `data_flow_control`, or physical SRAM read latency. Those higher-level stages must not be folded into the Machine microbenchmark formula without a separate full-core measurement.

## 4. Problems in the Legacy Model

### 4.1 Opcode timing did not match the RTL

The legacy `MatrixMachine::mm()` delay was:

```text
M_MM cycles = SYSTOLIC_PROCESSING_OVERHEAD + MLEN
M_MM_WO cycles = 1
```

For the Qwen configuration in this report, `MLEN=512` and `BLEN=64`, so every `M_MM` was charged 512 cycles. The structural form supported by the tested MXFP RTL points is `BLEN+4`, which extrapolates to 68 cycles at this production point.

### 4.2 Dispatch forced complete serialization

The legacy dispatch loop awaited the functional execution of every opcode inside its match arm. HBM prefetch, Matrix compute, Vector operations, and Scalar operations could not overlap even when they used independent RTL resources.

### 4.3 Ramulator writes completed too early

The legacy wrapper waited for completion callbacks on reads, but a write returned when Ramulator accepted the request. This undercounted store completion latency and allowed program termination while writes could still be pending.

### 4.4 DMA element and byte units were inconsistent

Logical element offsets, packed byte offsets, 64-byte lines, and physical bursts were not represented as separate quantities. Low-bit data such as MXINT2 and MXINT4 could consequently generate traffic closer to one byte per element rather than its packed representation.

### 4.5 Profiling assumed serial execution

The legacy profiler accumulated functional elapsed time per opcode. Once overlap is permitted, this approach double-counts overlapping work and cannot identify a critical path.

## 5. Change-to-Evidence Matrix

| Change | Implementation | Validation evidence | Result |
|---|---|---|---|
| Explicit `legacy` and `rtl-v1` modes | `src/cli.rs`, `src/timing.rs` | Same functional build executed under both modes; output hash regression | PASS |
| Configurable clock period | `plena_settings.toml`, `src/runtime_config.rs` | Config parsing tests; all reports use 1000 ps consistently | PASS |
| Reject invalid effective configs | `src/load_config.rs` | `validation_rejects_prefetch_values_that_would_have_been_clamped` | PASS |
| Ramulator write callback | `lib/ramulator/src/raw.rs`, `model.rs` | Raw read/write callback test | PASS |
| Pending read/write drain | `lib/ramulator/src/model.rs`, `src/runner.rs` | Runner explicitly drains; wrapper test verifies zero pending transactions after drain | PASS |
| Event wrapper matches raw Ramulator | `lib/ramulator/src/model.rs` | `event_wrapper_matches_raw_read_completion_cycle` | PASS |
| 64-byte gather/scatter coalescing | `lib/memory/src/chunked.rs` | Duplicate-line gather and same-line scatter tests | PASS |
| Precision-aware packed bytes | `src/dma.rs` | MXINT4 offset/payload and MXFP scale payload tests | PASS |
| Full-Machine opcode timing | `src/opcode_timing.rs` | Matrix/Vector/Scalar RTL artifacts | Tested points PASS |
| Hazard-aware scheduler | `src/scheduler.rs` | Production `pipeline_control` harness and six differential traces | PASS |
| Mixed vector latency ordering | `src/scheduler.rs` | Full VectorMachine mixed-latency sequence | PASS |
| Timeline profiler | `src/timing.rs`, `src/profiler.rs` | Overlap and critical-path partition tests | PASS |
| Unsupported timing policy | `src/timing.rs`, `src/runner.rs` | Coverage summary and `--require-rtl-validated` | PASS |
| Timing-only functional regression | Qwen validation runner | BF16 SHA-256 and comparison metrics | Identical |

## 6. Ramulator and DMA Corrections

### 6.1 Completion semantics

Reads and writes now use the same completion sequence:

```text
request accepted -> increment pending counter
Ramulator advances -> completion callback fires
decrement pending counter -> operation future resolves
program end -> drain until pending reads + pending writes == 0
```

Validation results:

- `raw_read_and_write_callbacks_complete`: PASS.
- `read_and_write_resolve_on_completion_and_drain`: PASS.
- `event_wrapper_matches_raw_read_completion_cycle`: PASS.
- Ramulator crate: 3 passed, 0 failed.

### 6.2 DMA accounting

The runtime now distinguishes:

```text
logical_elements
useful_payload_bytes
packed_transfer_bytes
coalesced_64B_line_requests
physical_burst_bytes
physical_bursts
```

Validation covers:

- MXINT4 packed element offsets.
- Rounding 65 four-bit elements up to 33 bytes.
- MXINT shared-scale payloads.
- MXFP element plus shared-scale payloads.
- Fragment coalescing within one physical line.
- Cross-line scatter splitting and unaligned read-modify-write.

The memory crate passes 17 tests with no failures. DMA tests are included in the 77 passing transactional emulator binary tests.

## 7. RTL Opcode Timing Experiments

### 7.1 MatrixMachine

Measured points:

```text
(MLEN, BLEN) =
(16,4), (32,4), (64,4),
(32,8), (64,8), (64,16)
```

The MXFP operand format is E4M3 and the Machine internal FP format is E8M7.

| MLEN | BLEN | `M_MM` ready/done/II | `M_MM_WO` first pulse | `M_MM_WO` backend idle |
|---:|---:|---:|---:|---:|
| 16 | 4 | 8 | 7 | 21 |
| 32 | 4 | 8 | 7 | 21 |
| 64 | 4 | 8 | 7 | 21 |
| 32 | 8 | 12 | 7 | 29 |
| 64 | 8 | 12 | 7 | 29 |
| 64 | 16 | 20 | 7 | 45 |

Within the tested domain:

```text
M_MM ready/done/II       = BLEN + 4
M_MM_WO backend idle     = 2*BLEN + 13
```

Changing MLEN from 16 to 64 while holding BLEN constant does not change the measured `M_MM` latency. This directly rejects the legacy assumption that `M_MM` latency should scale with MLEN for the tested MXFP path.

Twenty-four Matrix ready/done/II/busy metrics were checked. Maximum absolute error is 0 cycles.

Limitation: the current `M_MM_WO` path emits one observed write-valid pulse rather than BLEN row-valid pulses. Consumer-visible row readiness is therefore not cycle-exact. `rtl-v1` conservatively makes all output rows ready at backend idle and marks this opcode `unsupported_rtl`.

### 7.2 VectorMachine

Measured configurations:

```text
VLEN = 8, 16, 32, 64 with FP E8M7
VLEN = 32 with FP E6M5 holdout
```

Principal E8M7 timing:

| Opcode | Ready/done cycles |
|---|---:|
| `V_ADD_VV`, `V_SUB_VV` | 12 |
| `V_ADD_VF`, `V_SUB_VF` | 13 |
| `V_MUL_VV` | 10 |
| `V_MUL_VF` | 11 |
| `V_EXP_V` | 21 |
| `V_RECI_V` | 11 |
| `V_RED_SUM` | `5 + 7*ceil(log2(VLEN+1))` |
| `V_RED_MAX` | `5 + 2*ceil(log2(VLEN+1))` |
| Independent element-op II | 1 |

Vector validation summary:

- Total metrics: 115.
- Training metrics: 88.
- Holdout metrics: 27.
- Holdout maximum absolute error: 0 cycles.
- All metrics within one cycle: PASS.

### 7.3 ScalarMachine

Scalar tests use the same VLEN and FP-format points as VectorMachine.

| Opcode | Ready | Backend idle / II |
|---|---:|---:|
| `S_ADD_FP`, `S_SUB_FP` | 9 | 10 |
| `S_MUL_FP` | 7 | 8 |
| `S_EXP_FP` | 19 | 20 |
| `S_RECI_FP` | 9 | 10 |
| `S_SQRT_FP` | 5 | 6 |
| `S_MAP_V_FP` | `VLEN+3` | `VLEN+4` |

Scalar validation summary:

- Total metrics: 70.
- Training metrics: 56.
- Holdout metrics: 14.
- Holdout maximum absolute error: 0 cycles.
- All metrics within one cycle: PASS.

`S_MAX_FP` is not implemented in the current RTL. It remains usable only for functional debugging and cannot be reported as RTL validated.

## 8. Hazard-Aware Scheduler Validation

### 8.1 Direct production `pipeline_control.sv` measurements

| Hazard | Raw stall | Recovery | Result |
|---|---:|---:|---|
| HBM vector transfer to vector operation | 3 | 1 | PASS |
| Vector reduction to scalar FP operation | 5 | 1 | PASS |
| Scalar SFU to vector scalar-broadcast operation | 4 | 1 | PASS |
| Vector SRAM write conflict | 2 | 1 | PASS |

Every observed scenario confirms one registered `b1_pipeline_stall` recovery cycle after the raw hazard is released.

### 8.2 Differential scheduler traces

The following six traces were generated and checked:

1. `M_MM -> M_MM_WO -> output consumer`.
2. `V_RED_SUM -> S_LD_FP`.
3. `S_EXP_FP -> V_MUL_VF`.
4. `H_PREFETCH_V -> V_ADD_VV`.
5. Stall -> one recovery cycle -> reissue.
6. Mixed-latency `V_ADD_VV -> V_MUL_VV` with in-order result retirement.

Result: 6/6 cases PASS. Monotonicity, dependency, stall reason, recovery, and result-order checks all pass.

Additional scheduler unit tests cover:

- Unrelated Matrix prefetch and Matrix compute overlap.
- Matrix operand consumers waiting for overlapping prefetches.
- HBM vector DMA blocking conflicting Vector operations.
- Unrelated Vector DMA and Matrix writeout overlap.
- Scalar stores waiting for the specific FP register they consume.
- Scalar SRAM operations on unrelated registers overlapping FP compute.
- Matrix writeout being accepted at the frontend and queued behind Matrix compute.

## 9. Qwen3-32B One-Layer System Experiment

### 9.1 Configuration

| Parameter | Value |
|---|---:|
| Sequence length | 482 |
| MLEN / VLEN | 512 / 512 |
| BLEN | 64 |
| HLEN | 128 |
| Broadcast amount | 4 |
| M_LOAD / V_LOAD / V_WRITE | 512 / 64 / 64 |
| Weight / activation / KV | MXFP E4M3 |
| Matrix/Vector/Scalar internal FP | E8M7 |
| Modeled HBM channels | 128 |
| HBM preset | HBM2_2Gbps |
| Theoretical modeled peak | 2048 GB/s |
| Host wall time | 489.36 s |

### 9.2 Latency evolution

| Mode | Cycles | Time under 1 GHz assumption | Interpretation |
|---|---:|---:|---|
| Legacy functional executor | 71,503,360 | 71.503 ms | Original serial and hardcoded timing |
| Initial rtl-v1 | 21,893,358 | 21.893 ms | Before full-Machine and hazard corrections |
| Fixed rtl-v1 | 25,452,002 | 25.452 ms | Current result |

Current `rtl-v1` relative to legacy:

```text
latency reduction = 64.404%
ratio             = 2.809x
```

Fixed `rtl-v1` relative to initial `rtl-v1`:

```text
+3,558,644 cycles
+16.254%
```

The increase from initial to fixed `rtl-v1` shows that the initial model was optimistic. Full-Machine fixed latency, resource occupancy, and RTL hazards increased the predicted makespan.

### 9.3 Main source of the legacy-to-rtl-v1 difference

For this configuration:

```text
Legacy M_MM = MLEN = 512 cycles
rtl-v1 M_MM = BLEN + 4 = 68 cycles
M_MM count  = 129,280
```

As serial resource work, the `M_MM` correction removes:

```text
129,280 * (512 - 68) = 57,400,320 cycles
```

Conversely:

```text
Legacy M_MM_WO = 1 cycle
rtl-v1 conservative backend occupancy = 2*64+13 = 141 cycles
M_MM_WO count = 14,208
additional work = 14,208 * 140 = 1,989,120 cycles
```

These are resource-work differences, not direct makespan contributions. The hazard-aware scheduler overlaps legal work and introduces stalls, so the resource deltas cannot simply be summed to obtain total latency.

### 9.4 Largest opcode work increases from initial to fixed rtl-v1

| Opcode | Count | Total delta | Delta/op |
|---|---:|---:|---:|
| `V_MUL_VF` | 157,184 | +628,736 | +4 |
| `V_ADD_VV` | 176,160 | +528,480 | +3 |
| `M_MM` | 129,280 | +517,120 | +4 |
| `V_MUL_VV` | 90,176 | +270,528 | +3 |
| `V_SUB_VF` | 56,448 | +225,792 | +4 |
| `S_ADD_FP` | 95,619 | +191,238 | +2 |
| `V_EXP_V` | 56,448 | +169,344 | +3 |
| `V_RED_SUM` | 46,208 | +138,624 | +3 |

### 9.5 Critical-path breakdown

The critical-path contributions are mutually exclusive and sum exactly to 25,452,002 cycles:

| Resource | Critical-path cycles | Makespan share |
|---|---:|---:|
| Vector pipeline | 8,389,905 | 32.96% |
| Matrix compute | 7,247,898 | 28.48% |
| Control/frontend | 3,473,181 | 13.65% |
| Matrix writeout | 2,778,984 | 10.92% |
| Scalar pipeline | 2,653,704 | 10.43% |
| Total HBM contribution | 908,330 | 3.57% |

Resource work can overlap and therefore need not sum to 100%. Vector resource work is 11,545,984 cycles, or 38.39% utilization. Matrix compute work is 8,795,392 cycles, or 34.56% utilization.

### 9.6 Stall breakdown

| Stall reason | Cycles |
|---|---:|
| `matrix_mcu_active` | 7,132,852 |
| `vector_sram_port_a_write` | 3,344,736 |
| `vector_reduction_result_not_ready` | 3,129,280 |
| `matrix_writeout_active` | 2,751,926 |
| `scalar_fp_compute_in_progress` | 1,366,400 |
| `vector_pipeline_busy` | 981,504 |
| `hbm_request_port_busy` | 868,877 |
| `scalar_fp_operand_not_ready` | 802,048 |
| `pipeline_recovery` | 785,033 |
| `vector_mixed_latency_in_order` | 503,145 |

### 9.7 HBM and overlap data

| Metric | Value |
|---|---:|
| HBM bytes read | 598,081,536 |
| HBM bytes written | 4,718,592 |
| Read DMA operations | 2,189 |
| Write DMA operations | 128 |
| 64-byte read-line requests | 9,345,024 |
| 64-byte write-line requests | 73,728 |
| 16-byte physical read bursts | 37,380,096 |
| 16-byte physical write bursts | 294,912 |
| HBM request/completion wait | 972,173 cycles |
| Memory/compute overlap | 96,855 cycles |
| Memory/compute overlap ratio | 0.381% |
| Reported bandwidth utilization | 8.43 GB/s |

Memory/compute overlap is small in this workload. The large legacy-to-`rtl-v1` change is driven mainly by opcode timing corrections, especially changing `M_MM` from MLEN-based to BLEN-based timing, rather than by a large amount of modeled overlap.

## 10. CostEmitter Synchronization Validation

### 10.1 Shared timing source and semantics

CostEmitter now reads the transactional artifact directly:

```text
transactional_emulator/calibration/rtl_opcode_timing_v1.json
SHA-256: 3948c720ab9270c99d0d0f5720b6a069ace12abba5dec4ade75eb978b387db63
```

There is no separately copied Python coefficient table. The formal DSE compute
objective is:

```text
serial_resource_work_cycles = sum(opcode_count * resource_cycles)
compute_latency_ns = serial_resource_work_cycles * CLOCK_PERIOD_PS / 1000
```

It is reported as `serial_resource_work`; it is not labeled as a scheduled
makespan. The legacy timing remains available only through explicit
`compute_timing_mode="legacy"`.

Python/Rust formula parity was tested for ten MXFP/MXINT hardware and precision
configurations, including the production `MLEN=VLEN=512, BLEN=64` shape. Every
timing field for every opcode matched exactly.

### 10.2 Qwen resource-work parity

CostEmitter rebuilt the one-layer Qwen schedule as schema-v4 compressed IR.
Against the transactional reference:

| Metric | Transactional | CostEmitter | Difference |
|---|---:|---:|---:|
| Dynamic instructions | 3,473,181 | 3,473,181 | 0 |
| Distinct non-HBM opcode counts compared | 35 | 35 | 0 mismatches |
| Control work | 440,364 | 440,364 | 0 cycles |
| Matrix compute work | 8,795,392 | 8,795,392 | 0 cycles |
| Matrix writeout work | 2,012,352 | 2,012,352 | 0 cycles |
| Scalar work | 4,982,815 | 4,982,815 | 0 cycles |
| Vector work | 11,545,984 | 11,545,984 | 0 cycles |
| Total non-HBM work | 27,776,907 | 27,776,907 | 0 cycles |

The compact regression source is
`analytic_models/performance/calibration/rtl_v1_qwen3_32b_resource_work_reference.json`.
A cached Stage-1 benchmark evaluated the opcode-count work 10,000 times in
3.64359 seconds, or 0.364359 ms per evaluation. This is the path used by the
DSE objective.

### 10.3 Exact observed-DMA schedule replay

The transactional emulator emitted a compact trace containing only 2,317 HBM
events: 2,005 matrix prefetches, 184 vector prefetches, and 128 vector stores.
CostEmitter replayed the complete ordered schedule using each event's observed
Ramulator completion interval. It did not substitute V3 service estimates.

| Metric | Transactional Rust | CostEmitter Python | Difference |
|---|---:|---:|---:|
| Makespan | 25,452,302 | 25,452,302 | 0 cycles |
| Resource-work categories | 8 | 8 | all exact |
| Stall categories | 15 | 15 | all exact |
| Critical-path categories | 8 | 8 | all exact |
| Critical-path sum | 25,452,302 | 25,452,302 | 0 cycles |
| Unresolved dependency owners | 0 | 0 | 0 |

The mutually exclusive critical-path values are also identical:

| Resource | Cycles |
|---|---:|
| Vector pipeline | 8,389,905 |
| Matrix compute | 7,247,898 |
| Control/frontend | 3,473,181 |
| Matrix writeout | 2,778,984 |
| Scalar pipeline | 2,653,704 |
| HBM matrix DMA | 894,780 |
| HBM vector DMA | 10,426 |
| HBM vector store | 3,424 |

This exact replay took 112.855 host seconds. It is therefore a validation
path, not a per-trial DSE path. Users must explicitly request
`scheduled_shadow`; the formal objective remains the sub-millisecond Stage-1
resource-work estimate.

The earlier emulator run in Section 9 measured 25,452,002 cycles. Re-running
Ramulator for the compact DMA trace measured 25,452,302 cycles, a 300-cycle
(0.00118%) difference. Opcode counts and non-HBM work were unchanged.
CostEmitter is compared with the events from the same second run, avoiding a
false cross-run comparison.

### 10.4 Compressed post-hoc V3 scaling

The post-hoc V3 shadow was also exercised without observed DMA events. This is
not a Rust parity test: V3 supplies surrogate service intervals after trace
construction. It verifies that schema-v4 repeats preserve exact scheduler
state and can represent the production workload without expanding billions of
dynamic instructions.

| Batch | Layers | Dynamic instructions | Expanded instructions | Makespan | Wall time | Max RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3,473,181 | 140,817 | 189,304,412 cycles | 9.72 s | 419 MiB |
| 16 | 1 | 89,609,871 | 441,049 | 7,463,624,189 cycles | 37.19 s | 619 MiB |
| 16 | 64 | 5,583,601,542 | 559,209 | 484,033,902,268 cycles | 47.01 s | 631 MiB |

The 64-layer result fast-forwards 5,583,042,333 instructions and its mutually
exclusive critical-path categories sum exactly to the reported makespan. A
regression also compares literal and compressed execution of the production
8192-row, 512-byte-stride affine address pattern and obtains identical
makespan, resource work, and stall counters.

The requested target of less than one second per scheduled-shadow trial is
not met. Consequently, post-hoc V3 scheduling remains opt-in and is not used
as the formal DSE objective. The Stage-1 cached resource-work path remains
0.364359 ms/evaluation. All three post-hoc cases are additionally labeled
`unsupported_opcodes` and outside production-shape calibration coverage.

### 10.5 DSE integration

- A one-worker smoke completed 3/3 trials.
- A four-worker JournalStorage smoke completed 12/12 trials with unique trial
  numbers and complete merged records.
- Trial-specific Weight, ACT, KV, internal FP, integer width, and clock settings
  are written to the transactional TOML and checked against the explicit
  precision configuration. A mismatch is a hard error.
- `--compiler-compute-timing rtl-v1` is the default; legacy compute latency is
  retained in each record for comparison.
- Unsupported and out-of-domain opcode counts remain visible and do not become
  silent validation claims.

The exact parity above proves that CostEmitter implements the same calibrated
timing and scoreboard semantics as the transactional model for this trace. It
does not prove that unsupported opcodes are faithful to RTL, that production
shapes are within the measured calibration domain, or that the design closes
timing at 1 GHz.

## 11. Functional Regression and Numerical Status

Before and after the timing-only correction:

```text
BF16 output SHA-256:
c8d05cfd817c498cc49344df0900216741ab54a4ad61f19b53383664a0943d59
```

- Decoded BF16 output is bitwise identical: PASS.
- Comparison metrics are identical: PASS.
- The scheduler and timing changes therefore did not alter the functional data path.

However, the underlying golden comparison does not pass:

| Metric | Value |
|---|---:|
| MSE | 0.0390625 |
| MAE | 0.142578125 |
| Maximum error | 3.0 |
| Relative match rate | 40.806% |
| Allclose/match rate | 87.638% |
| `atol` / `rtol` | 0.2 / 0.2 |
| `allclose_pass` | false |

This experiment demonstrates that timing changes preserve the functional result. It does not demonstrate numerical accuracy PASS for this Qwen case.

## 12. RTL Validation Coverage

The Qwen trace contains 3,473,181 opcodes:

| Classification | Count | Share |
|---|---:|---:|
| Validated by the current policy | 666,300 | 19.18% |
| Structural or out of calibration domain | 2,730,849 | 78.63% |
| Unsupported by current RTL | 76,032 | 2.19% |

Unsupported opcodes:

| Opcode | Count |
|---|---:|
| `S_MAX_FP` | 30,848 |
| `V_SHIFT_V` | 30,848 |
| `M_MM_WO` | 14,208 |
| `M_BMM_WO` | 64 |
| `M_BTMM` | 64 |

The main out-of-domain source is production shape extrapolation. Matrix timing is directly measured only through `MLEN<=64, BLEN<=16`, while the Qwen case uses `MLEN=512, BLEN=64`. Vector and Scalar tests directly cover `VLEN<=64`, while the Qwen case uses `VLEN=512`.

`--require-rtl-validated` writes all requested artifacts and then returns a non-zero status for unsupported or out-of-domain runs. This prevents such results from being silently presented as cycle-exact.

## 13. Test Summary

### 13.1 Transactional emulator binary

Command:

```bash
nix develop --command bash -lc \
  'cd transactional_emulator && cargo test --bin transactional_emulator'
```

Result:

```text
77 passed
0 failed
1 ignored evidence-emission test
```

### 13.2 Memory and Ramulator crates

Command:

```bash
nix develop --command bash -lc \
  'cd transactional_emulator && cargo test -p ramulator -p memory'
```

Result:

```text
memory:    17 passed, 0 failed
ramulator:  3 passed, 0 failed
```

### 13.3 RTL acceptance summary

| Check | Result |
|---|---|
| Matrix measured metrics within one cycle | PASS, maximum error 0 |
| Vector training metrics within one cycle | PASS |
| Vector holdout metrics within one cycle | PASS, 27/27 error 0 |
| Scalar training metrics within one cycle | PASS |
| Scalar holdout metrics within one cycle | PASS, 14/14 error 0 |
| Pipeline recovery cycle | PASS, 4/4 equal one cycle |
| Scheduler differential traces | PASS, 6/6 |
| Mixed vector result ordering | PASS |
| Timing-only functional output hash | PASS, bitwise identical |

### 13.4 CostEmitter and DSE Python regression

The CostEmitter timing/parity/scheduler tests, compiler schedule-emission
tests, and Optuna DSE integration tests completed with:

```text
64 passed, 0 failed
```

This includes the literal-versus-compressed 8192-row affine regression and
the schema-v4 Qwen kernel schedule equivalence test.

## 14. Claims Supported by Current Evidence

1. Legacy MLEN-based `M_MM` timing does not match the tested current MXFP MatrixMachine RTL; latency scales with BLEN over the measured domain.
2. A fixed one-cycle Matrix writeout substantially underestimates backend occupancy.
3. Serial opcode elapsed-time accumulation cannot represent RTL-permitted overlap and actual hazards.
4. Ramulator write completion must be tied to the completion callback rather than request acceptance.
5. Low-bit DMA accounting must distinguish logical elements, packed bytes, physical lines, and physical bursts.
6. `rtl-v1` reproduces all tested Matrix, Vector, and Scalar microbenchmark metrics with zero-cycle error.
7. The hazard scheduler passes direct production pipeline-control observations and six differential traces.
8. The timing-only changes do not alter the tested Qwen functional output.

## 15. Claims Not Supported Yet

1. The Qwen3-32B layer is not full-system cycle-exact.
2. The RTL is not demonstrated to close timing at 1 GHz.
3. The 71.50 ms to 25.45 ms difference is not a demonstrated 2.81x hardware speedup.
4. The Qwen numerical comparison does not pass.
5. `M_MM_WO` does not have cycle-exact per-row consumer readiness.
6. The MXFP calibration cannot be directly generalized to MXINT or mixed precision.
7. Strong memory-bound claims are not supported because Ramulator service is currently placed on a post-hoc scheduler timeline rather than being coupled online cycle by cycle.

## 16. Recommended Next Steps

1. Correct the RTL `M_MM_WO` protocol so that BLEN output rows have a defined valid cadence, then remeasure ready/done/II.
2. Add Matrix holdouts at `BLEN=32/64` and `MLEN=128/256/512` to reduce Qwen-shape extrapolation.
3. Add Vector reduction and Scalar map holdouts at `VLEN=128/256/512`.
4. Implement or remove `S_MAX_FP`, `V_SHIFT_V`, and unsupported Matrix broadcast/writeout ISA paths.
5. Move Ramulator to an online scheduler-coupled model and validate a memory-bound trace.
6. Regenerate all artifacts from a clean RTL commit and preserve waveform provenance.
7. Enable `--require-rtl-validated` as a formal DSE gate only after production shapes and opcodes are covered.

## 17. Artifact Index

| Artifact | Path |
|---|---|
| Machine timing calibration | `transactional_emulator/calibration/rtl_opcode_timing_v1.json` |
| Matrix raw measurements | `Workspace/rtl_v1_latency_validation/matrix_full_machine/raw_measurements.json` |
| Vector raw measurements | `Workspace/rtl_v1_latency_validation/vector_full_machine_v2/raw_measurements.json` |
| Scalar raw measurements | `Workspace/rtl_v1_latency_validation/scalar_full_machine/raw_measurements.json` |
| Pipeline-control measurements | `Workspace/rtl_v1_latency_validation/pipeline_control/raw_measurements.json` |
| Scheduler differential traces | `Workspace/rtl_v1_latency_validation/scheduler_differential_traces.json` |
| Qwen run statistics | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/rust_emulator_run_stats.json` |
| Qwen timeline profile | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/memory_profile.json` |
| Qwen comparison | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/comparison_results.json` |
| Functional regression | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/functional_regression.json` |
| Machine-readable report | `Workspace/reports/transactional_emulator/rtl_v1_latency_validation.json` |
| Generated short report | `Workspace/reports/transactional_emulator/rtl_v1_latency_validation.md` |
| CostEmitter resource-work reference | `analytic_models/performance/calibration/rtl_v1_qwen3_32b_resource_work_reference.json` |
| Compact observed DMA trace | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/dma_event_trace.json` |
| CostEmitter observed replay | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/costemitter_replay_report.json` |
| Rust/Python parity summary | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/costemitter_transactional_parity.json` |
| Stage-1 evaluator benchmark | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/costemitter_stage1_benchmark.json` |
| Stage-2 post-hoc benchmark | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/costemitter_stage2_posthoc_benchmark.json` |
| Stage-2 full-workload output | `Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/costemitter_stage2_posthoc_batch16_64layers.json` |

## 18. Reproduction Commands

Run the complete Machine-level microbenchmarks:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode full \
  --harness matrix_machine_full_timing.py \
  --harness vector_machine_full_timing.py \
  --harness scalar_machine_full_timing.py \
  --out-dir Workspace/rtl_v1_latency_validation/full_machine \
  --resume
```

Run the production pipeline-control harness:

```bash
python transactional_emulator/testbench/rtl_timing/run_rtl_timing_calibration.py \
  --mode smoke \
  --harness pipeline_control_timing.py \
  --out-dir Workspace/rtl_v1_latency_validation/pipeline_control
```

Regenerate the Qwen validation run and auditable report:

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

Rebuild the CostEmitter trace and replay the exact observed DMA completions:

```bash
python -m analytic_models.performance.compiler_cost_model evaluate \
  --model-config Workspace/qwen3_32b_dense_analytic/qwen3-32b.json \
  --settings Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/plena_settings.toml \
  --calibration analytic_models/performance/calibration/hbm_service_global_v3.json \
  --seq-len 482 \
  --batch-size 1 \
  --num-layers 1 \
  --compute-timing rtl-v1 \
  --scheduled-shadow \
  --scheduled-dma-completion-trace \
    Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/costemitter_dma_replay/dma_event_trace.json \
  --output /tmp/qwen3_32b_costemitter_observed_replay.json
```
