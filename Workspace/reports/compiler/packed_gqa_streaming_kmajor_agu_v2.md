# Packed-GQA Streaming, K-Major Broadcast, and AGU v2

> **Historical mixed-status report, audited 2026-07-26.** Streamed softmax and
> K-major broadcast remain in `current-dse-v1`. AGU-v2 produced zero benefit
> on the evaluated Qwen traces and has now been removed from RTL, compiler,
> emulator, CostEmitter, area, power, and DSE runtime selection. Its numbers
> below remain valid as a negative-result experiment; only loop-AGU-v1 is
> selectable in the finalized implementation.

## Scope

The streaming and K-major changes remain compiler defaults. Post-increment AGU
v2 was evaluated but did not select any legal streams in the fixed Qwen
workloads, so the production address-generation default was restored to v1:

```text
softmax_state_schedule  = streamed-v2
packed_qk_schedule      = broadcast-k-major-v1
address_generation_mode = loop-agu-v1
```

It keeps the existing correctness gate, HBM V4 coefficients, ideal-II1
semantics, and Matrix structural timing unchanged. Compatibility modes remain
available as `sram-v1` and `head-major-v1`; AGU-v2 is report-only.

The principal changes are:

1. First/last-block softmax state liveness elimination.
2. Register forwarding of recurrent `m_res`, eliminating its Scalar FP SRAM
   store and reload.
3. K-major GQA traversal: one broadcast QK operation supplies all active query
   heads for a KV group.
4. Post-first-read loop AGU semantics for statically proven affine streams.

## Hardware Configuration

The broadcast schedule allocates separate `m/l` state for each active GQA
head:

```text
FP_SRAM_DEPTH = FP_CONSTANT_NUM + 2 * MLEN * physical_broadcast
```

For Qwen3-32B at `MLEN=2048`, `HLEN=128`, and GQA ratio 8:

```text
compatibility depth = 6,154 entries
new depth           = 32,778 entries
Scalar FP SRAM area = 5,391.36 -> 25,958.47 um2
```

The shared native-layout planner derives this value for the compiler,
CostEmitter, DSE constraints, area, and power. A DSE smoke exposed and fixed a
configuration propagation bug where the trial record contained 32,778 but the
temporary CostEmitter TOML retained 6,154.

## Short-Context Result

Fixed setup:

```text
Qwen3-32B, one decoder layer
seq=482, batch=16
MLEN=VLEN=2048, BLEN=1024, HLEN=128
E4M3 operands, FP_E8M7 internal
Matrix SRAM tiles=2, ideal-II1, HBM V4
```

| Schedule | Compute cycles | Stage roofline | QK | PV | HBM read |
|---|---:|---:|---:|---:|---:|
| Compatibility | 26,514,622 | 26.537 ms | 256 | 256 | 1.492 GB |
| Streamed softmax | 26,012,862 | 26.035 ms | 256 | 256 | 1.492 GB |
| K-major broadcast | 25,320,254 | 25.342 ms | 32 | 256 | 1.492 GB |
| Combined AGU v2 | 25,320,254 | 25.342 ms | 32 | 256 | 1.492 GB |

Incremental attribution:

| Change | Cycle delta | Reduction |
|---|---:|---:|
| Softmax state streaming | -501,760 | 1.89% |
| K-major broadcast | -692,608 | 2.66% |
| Post-increment AGU v2 | 0 | 0.00% |
| Combined | -1,194,368 | 4.50% |

A simple 64-layer repetition changes the E4M3 stage roofline from 1.698 s to
1.622 s. This is a repeated-layer estimate, not a scheduled full-chip replay.

The short case has one K block, so it has no recurrent `m_res` traffic to
remove. Its streaming gain comes from first/last-block state liveness.

## Long-Context Result

The following are one-layer results at the same compute shape with batch 1 and
two Matrix SRAM tiles.

| Sequence | Schedule | Compute cycles | Roofline | QK | PV | KV reload | HBM read |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | Compatibility | 18,521,784 | 18.533 ms | 192 | 192 | 8x | 2.378 GB |
| 4096 | Streamed | 17,619,064 | 17.631 ms | 192 | 192 | 8x | 2.378 GB |
| 4096 | K-major | 15,909,592 | 15.921 ms | 24 | 192 | 1x | 0.947 GB |
| 4097 | Compatibility | 22,792,969 | 22.809 ms | 384 | 384 | 8x | 4.252 GB |
| 4097 | Streamed | 21,889,865 | 21.906 ms | 384 | 384 | 8x | 4.252 GB |
| 4097 | K-major | 19,064,009 | 19.080 ms | 48 | 384 | 1x | 1.390 GB |
| 8192 | Compatibility | 53,347,802 | 53.370 ms | 640 | 640 | 8x | 6.672 GB |
| 8192 | Streamed | 51,080,794 | 51.103 ms | 640 | 640 | 8x | 6.672 GB |
| 8192 | K-major | 45,382,554 | 45.405 ms | 80 | 640 | 1x | 1.901 GB |

Combined compute reductions are 14.10%, 16.36%, and 14.93% for sequence
lengths 4096, 4097, and 8192. The K-major part is dominant because it changes
both QK compute and non-resident K/V reloads. Streamed softmax eliminates
`131,072`, `131,200`, and `786,432` recurrent `m_res` loads and the same number
of stores in the three cases.

Increasing Matrix SRAM from two to eight tiles changes the combined roofline:

| Sequence | 2 tiles | 8 tiles | Reduction |
|---:|---:|---:|---:|
| 4096 | 15.921 ms | 14.583 ms | 8.41% |
| 4097 | 19.080 ms | 17.068 ms | 10.55% |
| 8192 | 45.405 ms | 42.714 ms | 5.93% |

The `4096 -> 4097` cliff remains. One tail row creates another full-width Q
tile: QK rises from 24 to 48 and PV from 192 to 384. K-major reduces tail QK
from 192 to 24 occurrences, but cannot remove the full-width BMM work because
the ISA has no active-row BMM operation.

## Numerical and Structural Validation

Transactional validation used the existing, unchanged comparison gate:

| Case | Result |
|---|---|
| Compact packed-GQA: `batch=4, seq=7, M=V=16, B=4` | PASS |
| Multi-block: `batch=2, seq=39, M=V=16, B=4` | PASS |
| Multi-block optimized vs compatibility output | Bitwise equal |
| Multi-block values compared | 1,536 |
| Differing values | 0 |
| Maximum optimized-vs-compatibility absolute delta | 0 |

The multi-block test exercises recurrent softmax state, K-major traversal, and
Matrix K/V double buffering. A discovered V-buffer address bug was fixed by
using the Matrix tile size (`MLEN * MLEN`) rather than address 1 for slot 1.

CostEmitter tests verify:

- QK count decreases by the broadcast factor.
- PV count is unchanged.
- Short resident HBM traffic is unchanged.
- Non-resident K/V traffic is reused across heads.
- Every Scalar FP SRAM address is below the derived depth.
- ASM and CostTrace use the same transformed schedule.

Regression results:

```text
Compiler focused tests:             51 passed
Area/power/performance tests:       108 passed
DSE tests:                           20 passed
Area tests after AGU-v2 mapping:     32 passed
Transactional Rust tests:           150 passed, 1 ignored
Python compile checks:              passed
```

## Broadcast Timing Status

`M_BTMM` and `M_BMM_WO` are never assigned zero latency. At `BLEN=1024`,
the MXFP structural equivalents are:

```text
M_BTMM    = BLEN + 4       = 1,028 cycles
M_BMM_WO  = 2*BLEN + 13   = 2,061 cycles
```

MXINT uses `2*BLEN+8` and `BLEN+2`, respectively. Rust and Python consume the
same timing artifact.

The current RTL does not implement broadcast:

```text
broadcast_implemented=false
broadcast_timing_model=ordinary_matrix_structural_equivalent
broadcast_rtl_validated=false
rtl_validation_status=broadcast_rtl_unvalidated
```

Therefore K-major results are valid as an ideal architectural DSE scenario,
but are not cycle-exact evidence for the current RTL. `--require-rtl-validated`
rejects this path.

## AGU-v2 Hardware Result

ASAP7 TT 0.7 V, 25 C mapped DC at a 1 ns constraint:

| Block | v1 area | v2 area | v2 slack |
|---|---:|---:|---:|
| Loop AGU state | 1,722.569 um2 | 4,568.818 um2 | +0.01 ps |
| Loop-controller delta | 168.691 um2 | 168.691 um2 | +1.39 ps |
| Total AGU delta | 1,891.259 um2 | 4,737.509 um2 | - |

The v2 increase is 2,846.249 um2. It comes from matching both accepted GP read
ports against descriptors in up to four active frames and tracking post-stream
armed state. The result was previously used by `area_new` for
`loop-agu-v2`. The runtime overlay and candidate artifact have since been
retired; this table is preserved mapped-DC evidence for that decision.

The fixed Qwen traces selected no legal post-increment candidates. Thus AGU-v2
adds no latency benefit for these workloads and should not be presented as a
performance optimization result. Synthetic compiler and emulator tests still
validate post-first-read semantics, rearming, dual-source single-step, normal
GP-write reset, and nested binding rejection.

Because v2 has the same modeled latency as v1 while adding 2,846.249 um2 and
leaving negligible mapped timing margin, the v2 implementation, artifact,
CLI option, and runtime tests were removed. The historical synthetic tests
remain documented here, but the finalized RTL no longer contains a hidden
post-increment enable.

Power reuses the calibrated `agu_stream_step` action and reports:

```text
post_trigger_control_delta=not_separately_calibrated
```

Post-trigger power is not treated as zero.

## DSE Integration

A historical three-trial, one-worker Optuna smoke using AGU v2 completed all
trials with:

```text
FP_SRAM_DEPTH=32778
qk_recompute_factor=1
power_status=complete
broadcast_rtl_validated=false
rtl_validation_status=broadcast_rtl_unvalidated
```

The tested fixed `2048/1024` single-chip points have aggregate physical area
around 931-933 mm2 and therefore exceed the 908.6 mm2 nominal budget. This
smoke validates data flow and metadata, not feasibility of that fixed shape.

Artifacts:

- `Workspace/reports/compiler/packed_gqa_streamed_kmajor_agu_v2_results.json`
- `Workspace/qwen3_32b_dense_analytic/runs/smoke_streamed_kmajor_agu_v2_fixed_20260725/`

## Claim Boundary

- Softmax streaming is compiler-only and numerically bitwise validated on the
  exercised multi-block case.
- K-major QK reuse depends on a broadcast Matrix operation absent from current
  RTL. Its timing is nonzero but structurally extrapolated.
- Ideal-II1 remains an architectural assumption, not measured current-RTL
  hazard timing.
- HBM V4 remains a post-hoc production-DMA service model.
- The area values are pre-layout mapped/proxy values and do not include routed
  interconnect, CTS, or signoff margin.
- The 4096/4097 tail cliff remains until the ISA/MatrixMachine supports
  active-row BMM.
