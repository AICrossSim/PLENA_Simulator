# PLENA Prefill Modeling: Integrated Validation Status

## Document Status

This document records the integrated implementation as of 2026-07-26. It is
the entry point for interpreting the detailed area, latency, memory, compiler,
and DSE reports. Earlier reports remain useful A/B evidence, but their
intermediate best-latency values are not the current end-to-end result.

The current Matrix-SRAM search uses partial-resident K/V prefix policies and
ideal dual-port SRAM area. Single-chip K/V occurrences come from real compiler
lowering; the tile-aware multi-chip model reconstructs every TP/CP rank's
padded rows, packed heads, projection tiles, and local cache geometry. Tail
tiles still lack active-row BMM. See
`compiler/partial_resident_kv_ideal_dual_port_dse.md`.

## Current Analysis Path

The production analysis path is:

```text
Qwen3 model/config
  -> native compiler compact batch/head layout
  -> direct-first-block-v1 packed attention
  -> rtl-v4 compact-statistics vector/scalar lowering
  -> streamed-v2 softmax state
  -> broadcast-k-major-v1 QK reuse
  -> affine-loop-v2 unified FFN projection lowering
  -> loop-agu-v1 address generation
  -> CostEmitter dynamic opcode and DMA trace
  -> ideal-ii1 compute: structural Matrix timing plus one-cycle V/S/control
  -> production-DMA V4 per-stage memory work
  -> searchable tile-aware TP x CP x EP partition and NVLink communication
  -> per-stage compute/HBM/CP overlap plus serial TP collective
  -> precision-aware area_new proxy
  -> RTL-activity action-energy v2 plus HBM3E/interconnect system energy
```

`rtl-v1` ordered hazard-aware timing remains available as a conservative A/B
and RTL validation path. It is no longer the default DSE compute objective.
The post-increment AGU-v2 experiment has been retired from RTL, compiler,
emulator, CostEmitter, area, power, and DSE runtime selection because fixed
Qwen traces selected no legal post-increment streams. `loop-agu-v1` is the
only optimized production address-generation path; the v2 report and numbers
remain as historical negative-result evidence.

The corresponding DSE settings are:

```text
short: seq_len=482, batch_size=16, decoder layers=64
long:  seq_len=32768, batch_size=1, decoder layers=64
VLEN=MLEN
MLEN=[256,512,1024,2048,4096,8192]
BLEN divides MLEN, lower bound 32, area-feasible conditional upper bound
chip_count=[1,2,4,8,16]
TP=legal divisors of Q/KV heads and chip_count; CP=chip_count/TP
NVLink logical ports=[1,2,4], 450 GB/s one-way peak per port
Matrix SRAM policy=[streaming,projection-full,kv-25,kv-50,kv-75,kv-100]
103 measured software precision profiles with accuracy > 0.9
INT_DATA_WIDTH=[16,32,64]
A100 reference=826 mm2; feasibility budget=908.6 mm2
```

The formal multi-chip model is now `tile-aware-tp-cp-ep-v3`. It classifies
100% of CostEmitter stage/opcode work and reconstructs rank-local
MLEN/BLEN padding, compiler row/head packing, FFN K chunks, causal attention
tiles, and fixed-balanced MoE expert buckets. The previous fractional model
and `all work / chip_count` lower bound now live only under
`analytic_models.legacy`; neither appears in the formal DSE CLI.
Communication still assumes
100% peak link bandwidth and optimistic CP overlap, so v3 is a distributed
analytical model rather than cycle-exact multi-chip validation. See
`dse/tile_aware_multichip_model_v3.md`.

The 2026-07-26 lineage audit corrected the initial v3 controlled results.
Kernel ownership is now rebuilt from the final transformed schedule after
summary replay and AGU rewriting; structural EnergyAction/SRAM families and
layer DMA lineage are fail-closed. Tile-aware power is evaluated per rank
before clock-work caps are aggregated. The corrected long-context N16 optimum
for the controlled `M2048/B1024` trace is `TP2 x CP8, 4 ports = 1.839 s`;
the previously reported `TP4 x CP4 = 5.491 s` was invalid. Multi-rank HBM
remains role/opcode traffic rescaling of the V4 floor and residual, not exact
distributed request-manifest replay. See
`dse/tile_aware_multichip_lineage_audit_v4.md`.

The RTL-v4 path adds a 16-lane compact-statistics SIMD,
packed-softmax selector hoisting, and reduction overwrite. It is implemented
across RTL, compiler, transactional emulator, CostEmitter, area, and power and
is the current ideal-II1 DSE compiler default. Its physical validation limits
remain distinct from the ideal-II1 timing assumption. See
`compiler/compact_stats_selector_overwrite_v1.md`.

The FFN default is now `affine-loop-v2`. It removes the capacity-triggered
switch to the legacy loop template while preserving Matrix/HBM work and
bitwise numerical output. On the controlled `M4096/B64/N4` long-context point,
aggregate compute falls from `75.835B` to `44.419B cycles`; on short-context
`M2048/B1024/N1`, compute improves by `0.71%`. See
`compiler/unified_affine_ffn_loop_lowering_v2.md`.

The corresponding 2,048-point long-context DSE completed with zero pruned or
failed trials. All projections passed the structural guard with no fallback.
Across 123 exact adjacent-SRAM comparisons there were zero latency regressions
above 2% (or at any measured magnitude) and 108 improvements above 2%. This
replaces the old live-stride study's 59 regressions among 121 matched pairs.
The result validates the compiler no-regression invariant under ideal-II1; it
does not remove the existing broadcast, extrapolation, or multi-chip claim
boundaries.

The latest complete sharded grid contains 13,905 unique compiler-v1 points:
12,051 complete, 1,854 constraint-pruned, and zero failed. The final merge
found no duplicate or missing design keys. RTL-v3 is validated at the previous
optimum, but the complete grid has not yet been rerun. The global optimum
quoted from that grid is therefore historical; the rtl-v3 result below is a
controlled one-point A/B, not a new exhaustive optimum.

## Reproducible Source State

The finalized implementation is split into reviewable commits:

```text
PLENA_Compiler
  7e29ab5  Extend PLENA ISA assembly for RTL-v4 and loop AGU v1
  6d100dc  Optimize native packed-GQA lowering for prefill
  01762d8  Unify affine FFN lowering and partial K/V residency
  bd7b656  Optimize MoE lowering and rebuild CostTrace lineage
  1470478  Consolidate canonical compiler schedule profiles

PLENA_Simulator
  df104a0  Synchronize the transactional emulator with RTL-v4
  e107098  Add versioned RTL timing calibration artifacts
  ad40b49  Calibrate structural area overlays for prefill hardware
  9ccca28  Add action-based on-chip and external-memory power models
  9db3f15  Integrate HBM V4 with compressed compiler cost traces
  5c78b2b  Add tile-aware TP-CP-EP analytical scaling
  53e1126  Modularize canonical four-objective DSE execution
  4ad7df8  Quarantine historical analytic models and generated artifacts
  4329bf5  Restore scalar ROB scheduling with RTL-v4 timing

PLENA_RTL
  b5feafa  Add pipelined scalar and segment-parallel vector datapaths
```

`PLENA_Tools` is deliberately excluded from this source-state list and its
local submodule state is not part of the finalized simulator changes.

For rtl-v3, the final combined Python compiler, CostEmitter, timing-parity,
scheduler, and cache regression passes 95 tests; 141 Rust tests passed across
all targets, with one evidence-emission test intentionally ignored. The full
RTL check passed for 128 modules and 635 generated C++ files, and the focused
decoder integration suite passed all three tests. The final default-switch
regression and DSE smoke status are recorded in the rtl-v3 report.

## Qwen3-32B Results

### Streaming/K-major controlled reference before RTL-v4

At `MLEN=VLEN=2048`, `BLEN=1024`, `seq_len=482`, and `batch_size=16`,
the streamed/K-major E4M3 path before the compact-statistics RTL-v4 pass
predicts:

```text
compatibility one-layer roofline       26.537 ms
streamed-v2 one-layer roofline         26.035 ms
K-major with AGU-v1 one-layer roofline 25.342 ms
simple 64-layer repeated estimate       1.622 s
QK operations per layer               256 -> 32
PV operations per layer               256 -> 256
```

The combined reduction is 4.50%. The new Scalar FP SRAM depth is 32,778
entries. The current RTL does not implement Matrix broadcast, so the K-major
result is explicitly marked `broadcast_rtl_unvalidated` and uses nonzero
ordinary-Matrix structural timing. See
`compiler/packed_gqa_streaming_kmajor_agu_v2.md`.

### Current RTL-v4 non-Matrix A/B

At the same `2048/1024`, `seq=482`, `batch=16` reference, changing only the
Vector/Scalar schedule, selector handling, and reduction destination mode
gives:

```text
rtl-v3 compatibility compute       25,320,254 cycles/layer
rtl-v4 combined compute            19,699,230 cycles/layer
reduction                               22.20%

Matrix work delta                            0
HBM bytes/request delta                      0
```

The combined path removes 5,736,704 Scalar cycles and adds 115,680 compact
Vector SIMD operations. The final result passes the percentage target but
misses the original 19.60M absolute target by 99,230 cycles because split-head
K normalization retains 61,696 scalar lane accesses. The timing artifact was
rebuilt from 320 RTL records and records compact MUL/ADD/RSQRT at II=1.
Nine paired/leaf DC points and 26 RTL-activity power jobs have completed.
The area overlay observes a roughly fixed 4.40-4.52k um2 VectorMachine delta,
and compact action slopes have R2 above 0.999997. However, every 1 ns mapped
Vector point has negative WNS, the compact scalar/lane-chain elimination is
incomplete, and the absolute compute target is missed. RTL-v4 is the formal
DSE architecture/lowering default under ideal-II1. Its mapped 1 GHz physical
qualification remains experimental: the Vector points have negative WNS, so
this default must not be described as a timing-closed RTL implementation.

### Pre-streaming ideal-II=1 reference

At `MLEN=VLEN=2048`, `BLEN=1024`, `seq_len=482`, and `batch_size=16`,
the earlier pre-streaming controlled trace computed:

```text
Matrix       639,888,384 cycles
Vector       434,329,376 cycles
Scalar     1,050,974,721 cycles
Control       84,146,694 cycles
Total      2,209,339,175 cycles
V4 roofline      2.209350 s
```

This remains a useful historical A/B with ideal one-cycle Vector/Scalar/control
timing. It is an architectural assumption rather than a current-RTL timing
claim. The explicit hazard-aware sensitivity remains approximately 7.517 s.
See `model_latency/ideal_ii1_compute_timing_rollout.md`.

### Latest exhaustive grid before rtl-v3

The minimum modeled latency is 16,405.355 ms. It is shared by 72 precision and
integer-width configurations at `MLEN=VLEN=2048, BLEN=1024`. Selecting the
highest-accuracy tied profile gives:

```text
Weight / ACT / KV / FP = MXFP_E2M1 / MXFP_E1M2 / MXFP_E4M3 / FP_E6M5
INT_DATA_WIDTH          = 16
accuracy                = 0.98
nominal area            = 846.531 mm2
P90 area                = 854.467 mm2
latency                 = 16.405355 s
```

The fastest all-MXINT configuration uses MXINT4 weight/ACT/KV, FP_E6M5 and
the same `2048/1024` hardware. It predicts 16.477759 s and 646.277 mm2. This
supports the expected area-density advantage for MXINT in the current RTL,
while showing that the latency difference between arithmetic families is small
under the Stage-1 opcode timing used here.

The point closest to 826 mm2 is 834.651 mm2 with 16.405365 s latency. The area
budget is a comparison constraint, not an instruction to maximize die area;
the report therefore distinguishes fastest-under-budget from closest-to-A100.

### Current rtl-v3 controlled A/B

At the same prior optimum (`MLEN=VLEN=2048`, `BLEN=1024`) and one decoder
layer, rtl-v3 predicts:

```text
rtl-v2 stage-roofline latency     218.551 ms
rtl-v3 stage-roofline latency     154.111 ms
reduction                          29.485%

rtl-v2 serial resource work   216,137,296 cycles
rtl-v3 serial resource work   183,723,640 cycles (-14.997%)
rtl-v3 pipeline makespan       151,691,907 cycles
```

Q/K reductions fall from 555,264 scalar segment reductions to 38,560
multi-segment reductions. Matrix opcode counts, HBM opcode counts, transfer
bytes, and request counts are exactly unchanged. This is a one-layer,
one-point A/B result. A full 64-layer value and a new global optimum must come
from the pending exhaustive rtl-v3 DSE rerun.

The original rtl-v3 report predates the paired Vector/Scalar DC calibration.
The current area proxy now applies
`vector_rtl_v3_delta_coefficients.json` and
`scalar_rtl_v3_delta_coefficients.json`. Vector total-area holdout error is
3.40%; Scalar total-area holdout error is 0.005%. These mapped-area overlays
remove the earlier `recalibration_pending_rtl_v3` area limitation, but they do
not establish timing closure or signoff PPA.

## Compiler Improvement Chain

All values use the same Qwen3-32B workload and area/memory calibration unless
otherwise stated:

| Compiler state | Best modeled latency | Main change |
|---|---:|---|
| Pre-compaction | 36.678 s | One MLEN-padded slab per batch and sparse GQA columns |
| Compact batch/head layout | 21.706 s | Physical rows fixed at 8,192 and Q/O width fixed at 4,096 |
| Direct first-block packed attention | 18.503 s | First softmax block specialization and direct packed-O accumulation |
| Vector/scalar compiler-v1 | 16.405 s | Grouped segmented norm, active rows, and redundant-mask removal |
| Vector/scalar RTL-v2 at prior optimum | 13.791 s | Segment-aware reductions, scalar move/max/RSQRT |
| Vector/scalar RTL-v3 target, one layer | 154.111 ms | Multi-segment reductions, compact stats, and pipelined Scalar ROB |
| Ideal-II1 plus six-stream AGU v1 | 1.698 s | Zero-overhead loops and affine address sidecar |
| Streamed softmax plus K-major QK reuse | 1.622 s | State liveness, `m_res` forwarding, QK broadcast reuse, and AGU-v1 |

The last two values use ideal-II1 and are not directly comparable with the
hazard-aware rtl-v2/rtl-v3 rows. They are included to identify the currently
selected formal DSE semantics, not to imply an RTL speedup over those rows.

For the final `2048/1024` point, compiler-v1 reduces one-layer calibrated
resource work from 294,530,913 to 259,968,410 cycles. Matrix and HBM operation
counts are unchanged; Vector work falls by 10.17% and Scalar work by 19.37%.

Re-evaluating the same current-code point gives 268,429,442 compiler-v1 cycles
and 219,033,936 rtl-v2 cycles. The difference from the historical 259,968,410
compiler-v1 artifact reflects the subsequently finalized layout/reference
integration; the direct A/B comparison uses one source state and changes only
the Vector/Scalar schedule.

## Latency Evidence and Limits

`rtl-v1` replaced MLEN-based Matrix opcode delays with measured full-Machine
timing, added completion-aware DMA, and introduced a hazard scoreboard. Tested
RTL microbenchmarks match ready/done/II boundaries within one cycle. An
observed-DMA CostEmitter replay matches the transactional timeline exactly at
25,452,302 cycles when both consume the same measured DMA completion sequence.

For rtl-v2 and older paths, the formal fast compute quantity is **serial
resource work**, not a cycle-exact program makespan. RTL-v3 instead executes
one decoder layer of the compressed ordered schedule with the measured
ROB/resource scoreboard. Multi-layer DSE repeats the resulting `layer/*` stage
critical path while retaining `global/*` stages once. This improves overlap
fidelity and avoids serial resource-work fallback, but it is not a cycle-exact
full-program replay and excludes cross-layer overlap. Production Vector sizes
and segment width 128 remain structural extrapolations beyond the measured
VLEN 8-64 and segment widths 4/8/16.

At `MLEN=VLEN=512, BLEN=64`, the exact one-layer replay measured 523,713,105
cycles and required 101.31 seconds in isolation. A semantic cross-process
cache reproduces the same 64-layer 33,001,011,126-cycle result in a 2.90-second
direct fresh process and a 3.19-second fresh Optuna trial. DSE records both
pipeline fidelity and cache provenance in trial JSON and CSV outputs.

V4 memory prediction has a 585-sample generic holdout median error of 3.48%,
P95 of 18.10%, weighted MAPE of 2.80%, and store P95 of 19.92%. It is post-hoc:
it predicts production DMA service occurrences but does not preserve a single
online Ramulator state across all overlapping queues. The stage-wise roofline
is fast enough for DSE but was about 18% conservative relative to compressed
Stage 2 in the tested 32B/235B cases. That difference is scheduler omission,
not a V4 memory-error measurement or an absolute RTL-error measurement.

## Area Evidence and Limits

The nominal area is:

```text
structural MatrixMachine logic
+ fitted Vector/Scalar/HBM logic
+ proportional top-level logic residual
+ tiled ASAP7 SRAM macro area
```

Matrix structural-v4 grouped holdout errors are 2.92% median / 9.78% P95 for
MXINT and 0.79% median / 2.36% P95 for MXFP. Five held-out small full-chip
anchors give 2.15% corrected-logic MAPE and 1.77% corrected composite MAPE.
The top residual is fit on 12 separate anchors.

The DSE optimum is a structural extrapolation: DC MatrixMachine anchors cover
only MLEN 16-64 and BLEN 4-16. Large results follow exact RTL instance-count
growth and non-negative leaf fits, but no `2048/1024` design was synthesized.
Area excludes placement/routing overhead, clock/power grids, pads, HBM PHY,
HBM stacks, interposer and package. The 826 mm2 comparison is therefore an
early architectural reference, not a signoff die-area equivalence.

## Power Evidence and Limits

The default on-chip power component is `onchip_action_energy_v2`. It combines
compiler-emitted hardware actions, Qwen-like per-family dynamic-energy slopes,
an empirical low/random activity envelope, mapped-area idle clock scaling, and
ASAP7 SRAM Liberty read/write energy. The corrected calibration run contains
31 mapped configurations and 395 successful RTL-activity replays.

Grouped action holdouts give 7.47% median and 20.88% P95 error. Component
median/P95 errors are 6.10%/11.69% for Matrix, 7.89%/20.93% for Vector,
0.90%/8.69% for Scalar, and 7.82%/21.65% for the on-chip HBM controller. Idle
clock holdouts give 3.83%/8.34%; the worst Qwen-mix error is 11.41%. All action
slopes have R2 at least 0.999526 and cached evaluation takes 3.21 ms median.

This evidence is RTL VCD activity replayed on mapped DC netlists, not
gate-level simulation. The unified system estimator separately adds the
literature-parameterized external HBM3E model and nominal 8 pJ/bit
multi-chip communication proxy. It still excludes CTS, routing parasitics,
HBM PHY boundary uncertainty, package, final KV handoff, and SRAM leakage.
Fixed-amount HBM calibration points
identify per-opcode energy per accepted lane but not separate physical-line and
byte terms. Nominal system energy is the third formal DSE objective. The
evidence level remains architecture-model rather than signoff power, and no
hard power constraint is applied.

## Numerical Correctness Boundary

Compiler scheduling optimizations retain the existing comparison gate. Tiny
packed-GQA transactional cases cover batch co-packing, shared GQA storage,
dummy tails, partial K tiles and cross-batch isolation. The rtl-v3 compact test
passes with 99.609375% relative match, 100% allclose, and maximum absolute
error 0.00390625. Multi-segment reduction uses a balanced tree and may change
FP association, so rtl-v2 bitwise identity is not claimed.

Timing validation and numerical correctness are separate. Unsupported ISA/RTL
operations, mixed-precision hardware capability assumptions and production
shape extrapolation prevent describing the full Qwen3-32B estimate as
cycle-exact silicon validation.

## Canonical Artifacts

- DSE: `Workspace/qwen3_32b_dense_analytic/runs/roofline_v4_grid_vector_scalar_v1_sharded_20260719/`
- Area: `analytic_models/area_new/calibration/` and `Workspace/area_new_validation/matrix_structural_v4_20260718/`
- Power: `analytic_models/power/calibration/logic_energy_v2.json` and `Workspace/reports/power/rtl_activity_power_candidate_v2.md`
- RTL timing: `Workspace/reports/transactional_emulator/rtl_v1_latency_validation_full.md`
- HBM V4: `Workspace/reports/hbm_v4/hbm_dma_service_v4_full_report.md`
- Compiler optimization: `Workspace/reports/compiler/`
- Corrected multi-chip TP x CP x EP:
  `Workspace/reports/dse/tile_aware_multichip_lineage_audit_v4.md`
- Vector/Scalar RTL v2: `Workspace/reports/rtl/vector_scalar_rtl_v2_optimization.md`
- Vector/Scalar RTL v3: `Workspace/reports/rtl/vector_scalar_rtl_v3_segment_parallel.md`
- Fixed-balanced MoE: `Workspace/reports/model_latency/qwen3_fixed_balanced_latency_report.md`
