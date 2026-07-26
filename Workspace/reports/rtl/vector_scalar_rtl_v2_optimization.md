# PLENA Vector/Scalar RTL Optimization v2

> **Superseded as the default execution path.** This remains the rtl-v2 A/B
> baseline. The current segment-parallel Scalar-ROB implementation is
> documented in
> [`vector_scalar_rtl_v3_segment_parallel.md`](vector_scalar_rtl_v3_segment_parallel.md).

**Report date:** 2026-07-19

## Executive Summary

This work adds segment-aware vector reductions and three missing scalar
operations to PLENA RTL, then synchronizes the native compiler, transactional
emulator, RTL timing calibration, CostEmitter, and DSE interface. The area
model is deliberately unchanged and must be recalibrated separately.

At the previous Qwen3-32B optimum (`MLEN=VLEN=2048`, `BLEN=1024`,
`seq=482`, `batch=16`), changing only the Vector/Scalar schedule from
`compiler-v1` to `rtl-v2` gives:

| Metric | compiler-v1 | rtl-v2 | Change |
|---|---:|---:|---:|
| One-layer compute resource work | 268,429,442 cycles | 219,033,936 cycles | -18.40% |
| Vector work | 197,310,528 cycles | 163,244,352 cycles | -17.27% |
| Scalar work | 57,850,003 cycles | 42,550,625 cycles | -26.45% |
| Matrix work | 9,998,256 cycles | 9,998,256 cycles | 0% |
| Control work | 3,270,655 cycles | 3,240,703 cycles | -0.92% |
| 64-layer stage-roofline latency | 16.946860 s | 13.790892 s | -18.62% |
| V4 memory work | 70.060376 ms | 70.060376 ms | 0% |

The 13.791 s value is an A/B prediction at one existing DSE point, not the
result of a new exhaustive DSE. It uses serial calibrated resource work inside
the stage roofline and inherits its previously documented extrapolation
limits.

## Why RTL Changes Were Required

The compiler-v1 schedule had already removed avoidable full-width copies,
dummy-row normalization, and repeated address setup. The remaining dominant
operations could not be removed by scheduling alone:

1. Q/K normalization and packed softmax needed a reduction over one logical
   segment, but RTL only exposed a full-`VLEN` reduction.
2. A scalar register copy used `S_ADD_FP x, zero`, although the internal ALU
   already contained an incomplete move operation.
3. inverse square root required separate `S_SQRT_FP` and `S_RECI_FP`
   instructions plus a register-file round trip.
4. `S_MAX_FP` had an ISA definition but no complete scalar datapath.

At `VLEN=2048`, reducing a 128-lane head or 512-row packed batch slot through
the entire tree wastes levels even when all other lanes are masked.

## RTL Architecture

### ISA additions

```text
0x35  V_RED_SUM_SEG  fD, gpVectorAddr, gpSegmentIndex, segment_log2
0x36  V_RED_MAX_SEG  fD, gpVectorAddr, gpSegmentIndex, segment_log2
0x37  S_MV_FP        fD, fA
0x38  S_RSQRT_FP     fD, fA
```

The existing `S_MAX_FP` opcode is now implemented. Segment widths are powers
of two and are encoded as `segment_log2`; the segment base is
`segment_index << segment_log2`.

### Segment reduction

The design does **not** instantiate a second reduction tree. It selects the
requested vector segment, prepends the existing scalar accumulator, and taps
the existing tree after `segment_log2 + 1` levels. Consequently:

```text
sum cycles = 5 + 7 * (segment_log2 + 1)
max cycles = 5 + 2 * (segment_log2 + 1)
```

The result depends on segment width rather than physical `VLEN`. Full-vector
reductions retain the existing path. During this work an existing odd-input
tree bug was also fixed: when a reduction level had an odd number of inputs,
the unpaired value was previously allocated but not forwarded.

### Scalar operations

- `S_MV_FP` forwards the latched source operand through the scalar ALU.
- `S_MAX_FP` uses the existing FP maximum primitive through a completed ALU
  path.
- `S_RSQRT_FP` chains the existing square-root output into reciprocal. The
  intermediate square root is rounded in the configured scalar format, so it
  preserves the numerical semantics of the prior two-op sequence while
  removing frontend and register-file work.

The frontend decoder, data-flow control, pipeline hazards, vector write mask,
and top-level wiring were updated for the new operations.

## RTL Timing Calibration

The full-machine behavioral harness was run at:

```text
VLEN/FP = 8/E8M7, 16/E8M7, 32/E8M7, 64/E8M7, 32/E6M5
measurement boundary = execute accept -> consumer ready / backend idle
```

It produced 130 measurements. All segment reduction value checks passed.
The directly measured extension timings are:

| Operation | Ready cycles | Done cycles |
|---|---:|---:|
| Segment sum, width 4 | 26 | 26 |
| Segment sum, width 8 | 33 | 33 |
| Segment max, width 4 | 11 | 11 |
| Segment max, width 8 | 13 | 13 |
| `S_MAX_FP` | 4 | 5 |
| `S_MV_FP` | 3 | 4 |
| `S_RSQRT_FP` | 10 | 11 |

The segment results are invariant across the measured `VLEN` values and both
measured FP formats. Production segment widths 128 and 512 use structural
tree-depth extrapolation. This extrapolation follows the selected RTL tree tap
exactly, but those large widths were not directly simulated by the timing
harness.

Provenance is preserved in:

```text
Workspace/rtl_vector_scalar_v2_calibration/full_machine/raw_measurements.json
transactional_emulator/calibration/rtl_opcode_timing_v2.json
RTL HEAD: 0beb43f703f8d2a225f036da1764928563f3fb98 (dirty measurement state)
RTL diff SHA-256: 21501ec61b33650da134083f6cd3acf94e671e71dd640e9bc6e26a505c38d759
```

The artifact and current RTL diff hashes match. The 1 ns period is a reporting
assumption, not synthesis timing closure.

## Compiler and CostEmitter Integration

`vector_scalar_schedule="rtl-v2"` is now the default in the native compiler,
CostEmitter API/CLI, and Qwen3-32B DSE. `compiler-v1` and `legacy` remain
available for explicit A/B regression.

The compiler uses segment reductions in two places:

- Q/K segmented RMSNorm: one reduction per active HLEN lane.
- packed short-sequence softmax: max/sum over one aligned batch slot.

Ordinary RMSNorm uses `S_MV_FP` and `S_RSQRT_FP`. CostEmitter does not use an
independent opcode-count formula: it consumes the same compressed schedule
tree as ASM generation. Because segment latency depends on an operand,
CostEmitter now counts `(opcode, operands)` variants and requires exact
coverage of every parameterized reduction.

The compact row layout aligns short sequences to a power-of-two batch slot.
For the target workload, `seq=482` uses a 512-row slot. Physical rows remain
8,192 at MLEN 512/1024/2048, while segment reductions can terminate at the
512-row tree level.

The DSE persistent trace cache required one additional compatibility change.
It intentionally discards the large ordered schedule to bound worker RSS, but
an opcode-only cache loses the `segment_log2` needed by operand-sensitive
timing. Before stripping the schedule, the cache now stores compact total,
one-layer, and per-stage `(opcode, operands, count)` summaries. The cache key
schema was bumped, and timing evaluation fails closed if summary coverage does
not exactly equal the parameterized opcode count.

## Opcode-Level Attribution

The principal one-layer changes at the target point are:

| Opcode change | Dynamic count delta | Resource-work effect |
|---|---:|---:|
| `V_RED_SUM` removed | -1,079,552 | -96,080,128 cycles |
| `V_RED_SUM_SEG` added | +1,048,832 | +70,888,704 cycles |
| `V_RED_MAX` removed | -524,288 | -15,204,352 cycles |
| `V_RED_MAX_SEG` added | +493,568 | +12,339,200 cycles |
| `S_ADD_FP` removed | -2,151,267 | -21,512,670 cycles |
| `S_MV_FP` added | +2,059,107 | +8,236,428 cycles |
| `S_SQRT_FP` removed | -578,400 | -3,470,400 cycles |
| `S_RECI_FP` removed | -578,400 | -5,784,000 cycles |
| `S_RSQRT_FP` added | +578,400 | +6,362,400 cycles |

Matrix opcode counts are identical:

```text
M_BMM_WO=256, M_BTMM=256, M_MM=3728, M_MM_WO=2608
```

HBM opcode counts and V4 work are also identical:

```text
H_PREFETCH_M=502, H_PREFETCH_V=234, H_STORE_V=128
V4 memory work=70,060,375.822 ns for 64 layers
```

This establishes that the measured latency delta is attributable to the new
Vector/Scalar execution path rather than a changed matmul or memory workload.

## Numerical and Cross-Layer Validation

### RTL checks

`just rtl-check -j16` passes on the modified RTL. The timing harness verifies
the segment sum/max result and all new scalar operation boundaries.

### Transactional native decoder A/B

A tiny one-layer Qwen3-MoE run covers decoder RMSNorm, Q/K norm, packed
attention, expert FFN, and route combine. Both schedules use the same seed,
weights, input and unchanged comparison gate:

| Result | compiler-v1 | rtl-v2 |
|---|---:|---:|
| Relative match rate | 90.5273% | 90.5273% |
| Allclose match rate | 100% | 100% |
| Maximum absolute error | 0.140625 | 0.140625 |
| rtl-v1 scheduled latency | 199,900 cycles | 194,812 cycles |

The complete 16 KiB output windows are byte-identical (`cmp` exit status 0),
not merely both within tolerance. The golden files are also identical.

### Automated regression

- 66 focused Python compiler/timing parity tests passed before the default
  switch.
- 112 compiler, CostEmitter, scheduler-shadow, DSE, and runner-contract tests
  passed after updating stale default/schema expectations.
- 141 Rust tests passed across all targets; one evidence-emission test remains
  intentionally ignored.
- RTL static checking passed.

A real one-trial DSE smoke also completed after exercising the persistent
counts/DMA cache. Its trial record contains:

```text
state=complete
vector_scalar_schedule=rtl-v2
vector_scalar_area_calibration_status=recalibration_pending
```

The cached trace is 5.2 MiB and retains operand-qualified timing coverage
without retaining ordered schedule replay state.

## Performance and Runtime

The full 64-layer CostEmitter A/B took 26.06 s for compiler-v1 and 30.08 s for
rtl-v2 on this machine. Peak RSS was approximately 491 MiB and 523 MiB. The
extra host time comes from tracking operand-qualified schedule variants; it is
an offline modeling overhead, not modeled accelerator latency.

## Claim Boundary and Remaining Work

What is supported:

- new RTL operations are wired, statically checked, behaviorally measured,
  and numerically exercised end to end;
- compiler and CostEmitter emit/count the same new opcodes;
- the target-point compute reduction is fully attributable at opcode level;
- transactional output is bitwise unchanged in the tested native decoder.

What is not yet supported:

- no DC area or power calibration has been run for the modified Vector/Scalar
  RTL;
- no post-synthesis Fmax or physical-design timing result exists;
- segment widths 128/512 are structurally extrapolated from the RTL tree;
- the complete 13,905-point DSE has not yet been rerun with rtl-v2;
- existing unsupported Matrix writeout and vector-shift timing fallbacks are
  unchanged.

Until area is recalibrated, the old area proxy may undercount the additional
decode, scalar, and segment-select logic. Latency comparisons may use the new
path, but a final PPA claim must wait for the area pass and a fresh exhaustive
DSE.
