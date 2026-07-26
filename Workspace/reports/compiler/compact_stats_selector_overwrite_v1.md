# Compact Statistics SIMD, Selector Hoisting, and Reduction Overwrite v1

**Report date:** 2026-07-25
**Status:** Default DSE architecture path; physical RTL qualification remains
experimental because the mapped 1 GHz WNS gate does not pass.

## Purpose

This work reduces non-Matrix instructions in native Qwen prefill without
changing Matrix operations, HBM transfers, or the numerical comparison gate.
It combines:

1. A fixed 16-lane VectorMachine unit for compact segment statistics.
2. Loop-invariant selector hoisting in packed softmax.
3. An overwrite mode for full and single-segment reductions.

The compiler and CostEmitter consume the same structured lowering. All
reported savings therefore come from emitted opcode changes, not a post-hoc
analytical subtraction.

## Architecture and ISA

### Compact statistics SIMD

`V_RED_SUM_SEGS` writes one statistic per segment into the first 1-16 lanes of
a compact Vector SRAM word. RTL-v4 processes those lanes with:

```text
V_STAT_MUL_F
V_STAT_ADD_F
V_STAT_RSQRT
```

The instructions reuse opcode `0x3B`; `funct1[3]` selects compact-stat mode and
`rstride+1` encodes the active lane count. Inactive output lanes are cleared.
The unit contains at most 16 narrow FP lanes and does not replicate an entire
production-width VectorMachine.

For each grouped Q/K RMSNorm block, the old data path was:

```text
lane select -> lane load -> scalar MUL -> scalar ADD -> scalar RSQRT
            -> lane select -> lane store -> V_MUL_VSEG
```

The new path is:

```text
V_STAT_MUL_F -> V_STAT_ADD_F -> V_STAT_RSQRT -> V_MUL_VSEG
```

The FP operations and quantization boundaries are preserved lane by lane.

### Selector hoisting

Packed softmax repeatedly selects the same segment in a row loop. The
`hoisted-v1` pass proves that the selector is constant, is not written in the
loop, and is consumed with the same value. It moves one setup before
`C_LOOP_START`. Dynamic masks and output-lane selectors remain in place.

### Reduction overwrite

`funct1[0]` selects overwrite mode for full and single-segment reductions:

```text
V_RED_SUM_OVR
V_RED_MAX_OVR
V_RED_SUM_SEG_OVR
V_RED_MAX_SEG_OVR
```

SUM starts from exact FP zero. MAX starts from the same quantized negative
identity as compiler constant slot 2. The reduction tree and writeback path are
unchanged, but the old Scalar FP destination is not read. This removes explicit
neutral-value loads/moves and their RAW dependency.

## Implementation Integration

The following public modes were added end to end:

```text
vector_scalar_schedule = rtl-v4 | rtl-v3 | ...
selector_schedule      = hoisted-v1 | legacy
reduction_output_mode  = overwrite-v1 | accumulate-v1
```

They are carried through the native compiler, assembler, transactional
emulator, CostEmitter, DSE CLI, persistent cache schema, timing model, area
proxy, and power model. Compatibility modes remain available. As of the
partial-K/V DSE integration, the DSE defaults are:

```text
vector_scalar_schedule = rtl-v4
selector_schedule      = hoisted-v1
reduction_output_mode  = overwrite-v1
address_generation_mode = loop-agu-v1
```

This is an architecture-search default under ideal-II1 timing. It is not a
claim that the current RTL-v4 netlist closes timing at 1 GHz.

## Short-Context A/B

Configuration:

```text
Qwen3-32B, seq_len=482, batch_size=16
MLEN=VLEN=2048, BLEN=1024
ideal-II1 compute semantics
```

| Arm | Cycles/layer | Saved | Reduction |
|---|---:|---:|---:|
| Baseline | 25,320,254 | 0 | 0.00% |
| Compact only | 21,672,478 | 3,647,776 | 14.41% |
| Hoist only | 24,334,142 | 986,112 | 3.89% |
| Overwrite only | 24,333,118 | 987,136 | 3.90% |
| Compact + hoist | 20,686,366 | 4,633,888 | 18.30% |
| Compact + overwrite | 20,685,342 | 4,634,912 | 18.31% |
| Hoist + overwrite | 23,347,006 | 1,973,248 | 7.79% |
| Combined | **19,699,230** | **5,621,024** | **22.20%** |

The combined result passes the `>=22%` reduction gate. It misses the original
`<=19.60M` absolute target by 99,230 cycles.

The baseline and combined category breakdown is:

| Category | Baseline | Combined | Delta |
|---|---:|---:|---:|
| Matrix | 9,306,320 | 9,306,320 | 0 |
| Vector | 6,903,104 | 7,018,784 | +115,680 |
| Scalar | 8,838,861 | 3,102,157 | -5,736,704 |
| Control | 271,969 | 271,969 | 0 |

The Vector increase is exactly the 115,680 new compact SIMD instructions.
It replaces substantially more Scalar work.

For a simple 64-layer scaling, compute falls from 1,601,559,905 to
1,241,814,369 cycles. The corresponding stage-roofline estimate falls from
about 1.601621 s to 1.241876 s.

### DSE default integration replay

A fresh fixed-point run through the complete DSE entry point reproduced the
combined implementation result exactly:

```text
MLEN=VLEN=2048, BLEN=1024
Matrix SRAM=2 tiles, chip_count=1
precision=w_mxfp_e1m2__act_mxfp_e5m2__kv_mxfp_e5m2__fp_e5m6

one-layer compute       19,699,230 cycles
64-layer compute     1,241,814,369 cycles
stage-roofline latency     1.241876 s
```

The trial also loaded the paired RTL-v4 area overlay, the RTL-v4 power-action
overlay, ideal dual-port SRAM, HBM V4, and ideal hierarchical clock gating.
Its aggregate area was 926.937 mm2, so this fixed integration point is outside
the formal 908.6 mm2 comparison budget; it is not presented as a selected DSE
candidate.

An `8 trials / 4 workers` SQLite-WAL smoke then completed with zero failed or
pruned trials. All eight fixed inputs produced identical latency, area, energy,
and mode metadata, confirming deterministic parallel cache/study integration.

## Opcode Attribution

At the reference point:

```text
compact stat operations emitted                         115,680
compact scalar-chain operations removed               2,714,624
compact lane-selector instructions removed            1,048,832
softmax selector loads hoisted                           986,112
neutral accumulator setups removed                      987,136
```

The planned full compact-chain count was 555,264 statistics. The implemented
path leaves 61,696 Scalar lane loads and selectors in split-head K
normalization. That path stores the K lanes in separate Vector words and needs
the selected statistic as a scalar for each word. Eliminating it requires a
new broadcast-from-stat-lane operation or a K storage-layout change; neither is
part of this implementation. The report does not subtract this residual
analytically.

## Workload Invariants

Baseline and combined traces have identical:

```text
Matrix opcodes
HBM opcodes
QK/PV work
physical DMA request counts
physical read/write bytes
DMA manifest semantics
```

The one-layer physical HBM totals are:

```text
read bytes      1,737,228,288
write bytes       272,629,760
read requests      27,144,192
write requests      4,259,840
```

The optimization changes compute instructions only.

## Long-Context Results

All rows use `batch_size=1`, `MLEN=VLEN=2048`, and `BLEN=1024`.

| Matrix SRAM tiles | Sequence | Baseline cycles | Combined cycles | Reduction |
|---:|---:|---:|---:|---:|
| 2 | 4096 | 15,909,592 | 13,185,752 | 17.12% |
| 2 | 4097 | 19,064,009 | 16,339,312 | 14.29% |
| 2 | 8192 | 45,382,554 | 38,886,298 | 14.31% |
| 8 | 4096 | 14,571,349 | 11,847,509 | 18.69% |
| 8 | 4097 | 17,051,297 | 14,326,600 | 15.98% |
| 8 | 8192 | 42,691,533 | 36,195,277 | 15.22% |

HBM traffic is unchanged in every pair. The 4096-to-4097 discontinuity remains
because the ISA lacks active-row BMM and the tail incurs full-width Matrix
work. This optimization does not claim to remove that limitation.

## Functional and Model Validation

The focused validation covers:

- Compact lane counts 1, 4, 8, and 16.
- MUL, ADD, and RSQRT representative values in every production DSE FP
  format; overwrite MAX additionally exercises a negative identity/input.
- Full and segment SUM/MAX overwrite equivalence.
- Tiny packed GQA, dummy-tail, multi-block attention, long context, and MoE
  regressions under the unchanged correctness gate.
- ASM and CostEmitter dynamic opcode parity.
- Rust/Python timing-estimator parity.
- Matrix and HBM invariants described above.
- A three-trial single-worker Optuna smoke using all three experimental modes.

End-to-end numerical A/B results are stronger than the unchanged tolerance
gate alone:

- The short packed-GQA output (`batch=3`, `seq=7`) is bitwise identical
  between compatibility and combined modes.
- The recurrent multi-block output (`batch=2`, `seq=39`) is bitwise identical
  after the overwrite encoding fix. Its unchanged gate reports MSE
  `4.976988e-06`, maximum absolute error `0.053711`, and 100% allclose.
- The tiny Qwen3-MoE regression completes with MSE `8.392334e-04`, maximum
  absolute error `0.140625`, and 100% allclose.

This validation exposed two implementation bugs that are now covered by
regression tests:

1. Full `V_RED_SUM/MAX` instructions passed through assembler tables that
   discarded the overwrite bit. Machine-code tests now verify `funct1[0]` for
   both full reductions.
2. Multi-segment reduction quantized a two-dimensional intermediate tensor
   through a one-dimensional quantizer. Each tree level is now flattened for
   quantization and restored to its segment shape afterward.

The RTL timing harness measures execute-acceptance to ready/done and initiation
interval. The generated timing artifact is
`transactional_emulator/calibration/rtl_opcode_timing_v4.json`.

The final artifact was rebuilt from 320 raw measurements. It includes all
three production DSE FP formats (`E5M6`, `E6M5`, and `E8M5`) plus the E8M7
calibration family:

| Compact operation | Ready | Done | Measured II |
|---|---:|---:|---:|
| MUL scalar | 9 cycles | 9 cycles | 1 |
| ADD scalar | 11 cycles | 11 cycles | 1 |
| RSQRT | 11 cycles | 11 cycles | 1 |

Overwrite SUM/MAX was measured for full reductions and segment widths 4, 8,
and 16. Its ready/done behavior is structurally equivalent to the corresponding
accumulate reduction. Python timing parity passes 13 tests; Rust timing tests
pass 20 executions across the library and binary targets.

The first DSE smoke correctly failed because ideal clock-work accounting did
not recognize `exact_compact_lanes`. CostEmitter already preserved the 8/16
lane count; the power layer treated it as an unresolved ordinary Vector mask.
The fix adds a fixed 16-lane clock domain and a dedicated
`compact_stats_simd` area domain. Its focused regression passes, and the
repeated three-trial smoke completes with:

```text
latency      1,241.836 ms
area           931.724 mm2
accuracy         0.98
```

The fixed point exceeds the 908.6 mm2 comparison budget, which is expected for
this integration smoke and is not a new DSE optimum. This successful smoke
predates installation of the final RTL-v4 area and power artifacts. A fresh
post-install Optuna smoke could not be run because the repository's Python
3.12 environment does not currently contain Optuna; the only local Optuna
environment uses Python 3.11 and cannot parse the compiler's Python 3.12 type
aliases. Area and power artifact loading are instead covered by their focused
model tests. No dependency or study state was modified to hide this
environment limitation.

## Physical Evidence

The final physical promotion uses paired 1 ns `normal` DC synthesis, not the
100 ns area-only mode:

```text
VectorMachine RTL-v3 vs RTL-v4 at VLEN=16,32,64
standalone compact-stat SIMD at lanes=4,8,16
ASAP7 mapped area and WNS
```

The corresponding RTL-activity replay measures compact MUL/ADD/RSQRT and
overwrite SUM/MAX energy. The machine-readable outputs are:

```text
compact_stats_selector_overwrite_v1_dc_delta.csv
compact_stats_selector_overwrite_v1_power_delta.csv
```

These tables are generated only from completed runs. A failed promotion gate
does not create a zero-cost area or power coefficient.

All nine paired/leaf DC points completed:

| Point | RTL-v3 area | RTL-v4 area | Delta | RTL-v4 WNS |
|---|---:|---:|---:|---:|
| Vector VLEN=16 | 11,677.268 um2 | 16,198.715 um2 | 4,521.448 um2 | -0.547 ns |
| Vector VLEN=32 | 22,607.806 um2 | 27,008.852 um2 | 4,401.046 um2 | -0.539 ns |
| Vector VLEN=64 | 45,759.432 um2 | 50,187.246 um2 | 4,427.815 um2 | -0.536 ns |

Standalone compact-SIMD areas are 1,204.118, 2,417.524, and 4,848.550 um2
for 4, 8, and 16 lanes. The paired delta is approximately fixed across VLEN;
the structural overlay therefore uses the 16-lane leaf area and a zero
separately identifiable overwrite-control constant. Its maximum paired
relative residual is 10.11%. The installed area artifact is
`analytic_models/area_new/calibration/vector_rtl_v4_delta_coefficients.json`.

Every mapped point has negative WNS at the requested 1 ns clock. The paired DC
results are valid pre-layout area evidence, but they do not demonstrate 1 GHz
timing closure. This is a physical promotion failure, not a reason to replace
the measured area delta with zero.

The replay completed all 26 requested jobs with no failures. Dynamic action
energy is fitted from matched-idle **non-clock** energy; using total dynamic
energy would let small clock-network estimation drift dominate the much
smaller datapath increment. Clock energy remains accounted for separately by
the ClockWork model.

| Action | Nominal energy | Slope R2 |
|---|---:|---:|
| Compact MUL | 15.738 pJ | 0.999997 |
| Compact ADD | 15.824 pJ | 0.999999 |
| Compact RSQRT | 15.738 pJ | 0.999997 |

Measured overwrite-control deltas range from 7.759 pJ for segment MAX to
32.769 pJ for full SUM. The accepted artifact is
`analytic_models/power/calibration/vector_rtl_v4_power_delta.json`.

## Acceptance Status

| Gate | Status |
|---|---|
| Combined compute reduction >=22% | PASS, 22.20% |
| Combined compute <=19.60M cycles/layer | FAIL, 19.699M |
| Matrix opcode delta = 0 | PASS |
| HBM traffic delta = 0 | PASS |
| Selector reduction >=99% | PASS, 99.90% |
| Neutral setup reduction =100% | PASS |
| Compact scalar/lane chain reduction =100% | PARTIAL; split-K residual remains |
| RTL timing and numerical checks | PASS, 320 records, bitwise A/B, and MoE gate |
| Paired DC area extraction | PASS, 9/9 points and 10.11% maximum pair residual |
| 1 ns WNS gate | FAIL, all mapped configurations have negative WNS |
| RTL-activity power gate | PASS, 26/26 jobs and R2 >0.999997 |

Because not all physical gates pass, `rtl-v4 + hoisted-v1 + overwrite-v1`
remains experimental as an RTL implementation. It is nevertheless the DSE
default requested for architectural exploration. RTL-v3 remains the
compatibility and more conservative physical-reference path.

## Claim Boundary

The latency numbers use the project's ideal-II1 DSE semantics: Matrix
instructions retain structural timing while Vector, Scalar, and Control
instructions cost one cycle each. They measure the benefit of reducing emitted
instructions under that architectural assumption; they are not a cycle-exact
claim for the current RTL.

The area evidence is pre-layout mapped DC and is installed as an experimental
overlay despite the failed 1 GHz timing gate. The power evidence is
RTL-activity replay on mapped DC and remains a shadow model. Neither includes
CTS, routed parasitics, package power, or SRAM leakage.

## Regression Summary

The final focused Python suite passes 112 tests and 2 unittest subtests. The
post-DSE-integration affected suite passes 143 tests across compiler lowering,
K/V residency, area, power, multi-chip modeling, and timing parity. The
complete Rust workspace passes 153 tests with one ignored test.
`just rtl-check -j 16` passes in the PLENA RTL environment.

## Reproduction

```bash
python analytic_models/performance/benchmark_compact_stats_v1.py \
  --output-dir Workspace/reports/compiler
```

Machine-readable short/long results and opcode deltas are stored next to this
report.
