# PLENA Vector/Scalar RTL-v3 Segment-Parallel Optimization

> **Timing-policy update (2026-07-24):** the rtl-v3 lowering and hardware
> remain active, but its hazard-aware compute scoreboard is now an explicit
> `rtl-v1` sensitivity/validation mode. Formal DSE timing defaults to
> `ideal-ii1`; the instruction-count reductions documented here still apply.
> RTL-v4 adds compact-statistics SIMD and overwrite control on top of this
> architecture; see
> [`../compiler/compact_stats_selector_overwrite_v1.md`](../compiler/compact_stats_selector_overwrite_v1.md).

**Report date:** 2026-07-19

## Executive Summary

RTL-v3 implements three coordinated architecture changes and synchronizes
them through the assembler, native compiler, transactional emulator,
full-Machine timing calibration, CostEmitter, and DSE interface:

1. one Vector instruction reduces every aligned segment in a Vector SRAM word;
2. one Vector instruction broadcasts compact per-segment statistics back to
   their corresponding segments; and
3. ScalarMachine uses in-order issue with an eight-entry reorder buffer (ROB),
   dependency tracking, forwarding, and in-order retirement.

No fused mathematical operator was added. The existing correctness gate,
Matrix workload, HBM workload, V4 memory coefficients, and area coefficients
were not changed. RTL-v2 remains available for explicit A/B regression.

At the fixed Qwen3-32B target (`seq=482`, `batch=16`, one decoder layer,
`MLEN=VLEN=2048`, `BLEN=1024`, `HLEN=128`):

| Metric | rtl-v2 | rtl-v3 | Change |
|---|---:|---:|---:|
| Q/K scalar segment-sum reductions | 555,264 | 0 | removed |
| Q/K multi-segment reductions | 0 | 38,560 | replacement |
| Softmax scalar segment-sum reductions | 493,568 | 493,568 | unchanged |
| Compute resource work | 216,137,296 cycles | 183,723,640 cycles | -15.00% |
| Compute timing quantity | 216,137,296 serial cycles | 151,691,907 pipeline cycles | -29.82% |
| Stage-roofline one-layer latency | 218.551 ms | 154.111 ms | -29.49% |

The 29.82% number must not be interpreted as instruction deletion alone. About
15.00% comes from reduced work; the remaining improvement comes from modeling
the measured scalar/vector initiation intervals and ROB overlap instead of
serially adding every opcode latency.

## Architectural Motivation

Qwen3 packed Q/K RMSNorm operates on many independent `HLEN=128` head
segments inside a `VLEN=2048` Vector SRAM word. RTL-v2 exposed one reduction
per segment. For each active row, the compiler therefore issued 64 Q-head and
8 K-head reductions even though the existing balanced tree computes all
aligned subtrees in parallel.

RTL-v3 exposes that latent tree parallelism. A Q storage block containing 16
heads now produces 16 compact statistics with one reduction. Four Q blocks and
one K block reduce the per-row instruction count from 72 to 5:

```text
7712 active rows * (64 Q + 8 K) = 555,264 rtl-v2 reductions
7712 active rows * (4 Q blocks + 1 K block) = 38,560 rtl-v3 reductions
```

The compact values are normalized through ScalarMachine and written back to a
Vector SRAM stats word. A segment-broadcast Vector operation then applies all
statistics to their source segments in parallel.

## ISA and Datapath Changes

### New opcodes

```text
0x39  V_RED_SUM_SEGS
0x3A  V_RED_MAX_SEGS
0x3B  V_ALU_VSEG       (assembler: V_ADD_VSEG/V_SUB_VSEG/V_MUL_VSEG)
0x3C  S_LD_VLANE_FP
0x3D  S_ST_VLANE_FP
```

`V_RED_*_SEGS` partitions the input word into power-of-two segments. The first
`VLEN / segment_width` output lanes contain one result per segment; remaining
lanes are zero. The implementation selects an intermediate level of the
existing balanced reduction tree rather than duplicating the tree. Up to 16
segments are supported by the current compiler path.

`V_*_VSEG` reads compact statistics from Vector SRAM and broadcasts statistic
`i` to segment `i`. ADD, SUB, and MUL reuse the existing element ALUs and obey
the existing Vector write mask. `S_LD_VLANE_FP` and `S_ST_VLANE_FP` provide
explicit scalar access to compact Vector SRAM lanes. Their Vector SRAM port,
RAW, and masked-write dependencies are represented in the RTL hazard logic and
software scoreboards.

### Scalar pipeline

The scalar FP register encoding expands from three to four bits, exposing
`f0-f15`; `f0` remains constant zero. The new pipeline has:

```text
in-order issue
8-entry ROB
one pending-producer tag per FP register
RAW forwarding from completed ROB entries
WAW stall until the older producer retires
per-unit II tracking
in-order, one-result-per-cycle retirement
```

ADD/SUB, MUL, MAX/MOVE, SQRT, reciprocal, EXP, lane load, SRAM load, and
single-result reductions participate in the common dependency and writeback
rules. This is not out-of-order execution: there is no register renaming or
dynamic wakeup/select.

During integration, two pre-existing `V_SHIFT_V` bugs were exposed and fixed.
The Vector datapath used an invalid accumulator signal as its shift amount,
and the decoder routed the shift operand through the scalar-FP path rather
than the integer GP operand. A focused decoder regression now checks the
operand route directly.

## Compiler Lowering

The native compiler option is:

```python
vector_scalar_schedule="rtl-v3"
```

It is the default after validation. `rtl-v2`, `compiler-v1`, and `legacy`
remain explicit compatibility modes and are included in cache keys.

For each packed Q/K `(storage block, row)`, the compiler emits:

1. one square Vector word;
2. one `V_RED_SUM_SEGS` into compact Vector SRAM;
3. lane loads and independent scalar normalization chains, modulo-scheduled
   across up to eight FP registers;
4. lane stores of the normalized factors; and
5. one masked `V_MUL_VSEG` to update all active head lanes.

Softmax rows have one logical active segment and intentionally remain on the
rtl-v2 single-segment SUM/MAX path. Matrix QK/PV order and all DMA occurrences
are unchanged.

ASM and CostEmitter consume the same structured normalization plan. The
CostEmitter does not estimate these opcode counts from a separate analytical
formula.

## Full-Machine Timing Calibration

The behavioral harness measures from execute acceptance to consumer-ready and
backend-idle boundaries. It covered:

```text
VLEN = 8, 16, 32, 64
segment width = 4, 8, 16
segment count = 1, 2, 4, 8, 16 where legal
FP = E8M7 plus E6M5 holdout at VLEN=32
```

The run produced 247 measurements. Every numerical check passed. The raw
measurement SHA-256 is:

```text
3b8a26910dfbb29382fddb655292db62c6580709349dd203266c514641f10035
```

Measured Vector timings are:

| Operation | Width/shape | Ready/done cycles | II |
|---|---:|---:|---:|
| Multi SUM | 4 / 8 / 16 | 20 / 27 / 34 | 1 |
| Multi MAX | 4 / 8 / 16 | 10 / 12 / 14 | 1 |
| VSEG ADD/SUB | width 4 | 13 | 1 |
| VSEG MUL | width 4 | 11 | 1 |
| Vector-lane load | all tested VLEN | 6 | 1 |
| Vector-lane store | all tested VLEN | 7 | 1 |
| `V_SHIFT_V` | VLEN 8/16/32/64 | 10/11/12/13 | 1 |

The exact structural equations selected by the artifact are:

```text
multi_sum_cycles = 5 + 7*log2(segment_width) + 1 writeback
multi_max_cycles = 5 + 2*log2(segment_width) + 1 writeback
shift_cycles     = 7 + ceil(log2(VLEN))
```

The measured scalar boundaries are:

| Operation | Ready | Retired/done | Independent II |
|---|---:|---:|---:|
| ADD/SUB | 9 | 10 | 1 |
| MUL | 7 | 8 | 1 |
| EXP | 18 | 19 | 1 |
| Reciprocal | 8 | 9 | 1 |
| SQRT | 4 | 5 | 1 |
| MAX | 4 | 5 | 1 |
| Move | 3 | 4 | 1 |
| RSQRT | 9 | 10 | 1 |

Additional harnesses verify eight consecutive independent operations,
forwarded RAW chains, WAW retirement constraints, ROB-full behavior, mixed
latency in-order retirement, and one-result-per-cycle arbitration.

Canonical evidence:

```text
Workspace/rtl_vector_scalar_v3_calibration/full_machine/raw_measurements.json
transactional_emulator/calibration/rtl_opcode_timing_v3.json
Workspace/rtl_vector_scalar_v3_validation/target_2048_1024/ab_summary.json
Workspace/rtl_vector_scalar_v3_validation/target_2048_1024/opcode_delta.csv
```

The artifact records RTL HEAD, dirty-state diff hash, raw measurement hash,
measurement boundary, and support domain. The 1 ns conversion is a reporting
assumption, not timing closure.

## Emulator and CostEmitter Synchronization

The transactional emulator implements the new functional semantics, compact
stats layout, balanced-tree ordering, masks, lane accesses, ROB tags,
forwarding, and resource conflicts. Event traces include issue/start/ready/
retire information and explicit stall reasons.

Python and Rust load the same `rtl_opcode_timing_v3.json` artifact. Timing
parity tests compare every supported estimate field. CostEmitter evaluates the
compressed compiler schedule with the same dependency model and exact repeat
fast-forward. It expands only the instructions needed to prove repeat-state
transitions; the target `512/64` one-layer validation expanded 2,508,996
instructions and algebraically skipped 103,596,777 dynamic instructions.

For rtl-v3, formal compute latency is the compute pipeline makespan. Serial
resource work remains in the report as a diagnostic. DSE stage latency remains:

```text
sum(stage max(compute_pipeline_makespan, V4_memory_work))
```

When the ordered schedule is unavailable, rtl-v3 fails rather than silently
falling back to serial resource work.

For a multi-layer DSE trace, CostEmitter first performs an exact compressed
scoreboard replay of one decoder layer. It then repeats only `layer/*` stage
critical-path cycles and retains `global/*` setup/final stages once. The
reported fidelity is therefore:

```text
one layer: exact_compressed_scoreboard
full model: repeated_layer_stage_scaling
cross-layer overlap: excluded by the stage-serial roofline model
```

This replaces an impractical full 64-layer replay, but it is not described as
cycle-exact full-program scheduling. Source-, hardware-, precision-, timing-
artifact-, and scheduler-version-qualified results are stored in an immutable
cross-process cache. Writers use a per-key advisory lock and atomic rename, so
parallel DSE workers cannot consume partial results.

## DSE Runtime Evidence

The fixed Qwen3-32B smoke used `seq=482`, `batch=16`, 64 layers,
`MLEN=VLEN=512`, `BLEN=64`, and the production V4 memory path. Its exact
one-layer scoreboard result was:

```text
one-layer pipeline makespan =    523,713,105 cycles
64-layer scaled makespan    = 33,001,011,126 cycles
stage-roofline latency      =        33.001020 s
```

The one-layer replay itself took 101.31 seconds and 714,660 KiB peak RSS in an
earlier isolated measurement. A two-trial DSE cache smoke completed both
trials with identical latency and area; the second trial reused the in-process
complete report. A separate Python process then loaded the persisted trace,
pipeline, and V4 work and reproduced exactly the same cycles/latency in 2.90
seconds at 305,232 KiB peak RSS. A final fresh-process Optuna trial exercised
the same persistent path end to end in 3.19 seconds at 318,408 KiB peak RSS;
its trial JSON and CSV both record the cache key and `cache_hit=true`. The
pipeline cache entry is about 20 KiB.

The cache is an acceleration only. The key includes the compiler trace
identity, hardware, compute precision, clock period, RTL timing artifact hash,
schedule expansion policy, and the source hash of the timing/scheduler
implementation. Any semantic source or artifact change invalidates it.

## Target A/B Attribution

The one-layer target result is:

```text
rtl-v2 serial resource work        216,137,296 cycles
rtl-v3 serial resource work        183,723,640 cycles  (-14.997%)
rtl-v3 compute pipeline makespan    151,691,907 cycles  (-29.817% vs v2 serial)
rtl-v2 stage-roofline latency           218.551 ms
rtl-v3 stage-roofline latency           154.111 ms      (-29.485%)
```

Category work changes from rtl-v2 to rtl-v3:

| Category | rtl-v2 cycles | rtl-v3 cycles | Interpretation |
|---|---:|---:|---|
| Vector | 165,196,096 | 128,332,736 | multi-reduction and VSEG savings |
| Scalar | 39,864,461 | 44,923,521 | compact lane work is explicit; pipeline overlaps it |
| Matrix | 8,085,648 | 8,085,648 | unchanged |
| Control | 2,991,091 | 2,381,735 | fewer per-segment loops/masks |

Scalar work increases because 555,264 compact lane loads and stores replace
implicit per-segment scalar handoff. The architecture still improves makespan:
independent scalar chains issue at II=1 and overlap under the eight-entry ROB.
The target trace reports zero ROB-full stall cycles.

The workload invariants checked by the evidence generator are exact:

```text
Matrix: M_BMM_WO=256, M_BTMM=256, M_MM=3728, M_MM_WO=1680
HBM:    H_PREFETCH_M=502, H_PREFETCH_V=234, H_STORE_V=128
reads:  2,920,808,448 bytes / 45,637,632 requests
writes:   301,989,888 bytes /  4,718,592 requests
```

The two V4 latency totals differ by about 0.23% despite identical traffic
counts because compact stats change Vector SRAM target addresses used by the
per-occurrence geometry model. This is not a transfer-volume change.

## Numerical and Regression Evidence

A compact packed-GQA transactional test exercises batch packing, shared GQA
storage, segment-parallel Q/K norm, lane load/store, VSEG broadcast, and the
new scheduler:

```text
batch=4, seq=7, MLEN=VLEN=16, BLEN=4, HLEN=4, GQA ratio=2
relative match rate = 99.609375%
allclose match rate = 100%
maximum absolute error = 0.00390625
```

It passes the unchanged correctness gate. Balanced multi-reduction is allowed
to change FP association, so bitwise equivalence with rtl-v2 is not claimed.

Automated status at the measurement source state:

```text
full RTL static/Verilator check: PASS, 128 modules / 635 generated C++ files
focused decoder RTL tests:      3 passed
final combined Python regression: 95 passed
Rust all-target tests:          141 passed, 1 evidence-emission test ignored
```

The Rust scheduler tests include independent issue, ROB full, RAW forwarding,
WAW, compact stats hazards, repeat fast-forward, and mixed resource timing.

## Acceptance Status

| Requirement | Result |
|---|---|
| Q/K reductions `555,264 -> 38,560` | PASS |
| One-layer compute pipeline reduction at least 15% | PASS, 29.82% |
| Matrix opcode workload unchanged | PASS |
| HBM opcode/byte/request workload unchanged | PASS |
| Existing numerical correctness gate unchanged and passed | PASS |
| Python/Rust timing artifact parity | PASS |
| Full RTL check | PASS |

## Claim Boundary and Remaining Work

Supported claims:

- the new opcodes and Scalar ROB are implemented in RTL and pass full RTL
  checking;
- full-Machine microbenchmarks establish the tested ready/done/II values;
- compiler, emulator, and CostEmitter agree on the new ISA and timing artifact;
- the target reduction count and workload invariants are exact;
- the target model predicts a 29.49% one-layer stage-roofline improvement.

Not yet supported:

- paired DC area deltas are now available, but there is no post-synthesis Fmax
  or signoff power result for rtl-v3;
- production segment width 128 is a structural tree-depth extrapolation beyond
  measured widths 4/8/16;
- the target pipeline makespan is a compressed software timing model, not a
  cycle-by-cycle full RTL decoder simulation;
- multi-layer DSE latency repeats the exact one-layer stage critical path and
  does not model cross-layer overlap;
- unsupported Matrix opcodes in the tiny transactional run remain disclosed;
- a complete 13,905-point DSE rerun with rtl-v3 has not yet been published.

This report originally recorded:

```text
vector_scalar_area_calibration_status=recalibration_pending_rtl_v3
```

That status is superseded by the paired-delta artifacts:

```text
vector_rtl_v3_delta_coefficients.json
scalar_rtl_v3_delta_coefficients.json
```

The Vector total-area holdout error is 3.40%; the Scalar total-area holdout
error is 0.005%. Latency and calibrated mapped-area comparisons may use
rtl-v3. Final PPA claims still require timing closure and physical power
validation.
