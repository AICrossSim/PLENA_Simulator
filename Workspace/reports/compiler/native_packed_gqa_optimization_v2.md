# Native Packed-GQA Compiler Optimization v2

**Status (2026-07-19):** Integrated as the default packed-attention schedule.
The 18.503 s result below intentionally isolates this pass. The later
vector/scalar compiler-v1 pass reduces the current exhaustive-DSE result to
16.405 s without changing Matrix or HBM operation counts.

## Scope

This optimization reduces compiler-generated work inside native packed-GQA
attention. It does not add an ISA opcode, change RTL, modify the numerical
comparison gate, or alter the `rtl-v1` opcode timing and V4 HBM coefficients.
It is independent of the earlier compact batch/head layout work.

The optimized schedule is selected with:

```text
packed_attention_schedule = "direct-first-block-v1"
```

The previous lowering remains available as `"legacy"` for A/B experiments.
The native compiler and CostEmitter use the same lowering and schedule option,
and the option is included in the CostEmitter/DSE cache key.

## Optimization 1: Specialized First K Block

The general online-softmax recurrence maintains a previous row maximum
`m_old`, normalization sum `l_old`, and accumulated output `O_old`:

```text
m_new = max(m_old, row_max(S))
r     = exp(m_old - m_new)
l_new = l_old * r + sum(exp(S - m_new))
O_new = O_old * r + exp(S - m_new) * V
```

For the first K block, the compiler already knows that
`m_old = -infinity`, `l_old = 0`, and `O_old = 0`. The specialized lowering
therefore emits the equivalent recurrence:

```text
row_max = reduce_max(scale(S))
m_new   = row_max
P       = exp(scale(S) - m_new)
l_new   = reduce_sum(P)
O_new   = P * V
```

It still writes `m_new` and `l_new` to their original FP SRAM addresses, so
all later K blocks execute the unchanged general recurrence. QK, causal-mask,
PV, and quantization order within the surviving operations are unchanged.

This removes first-block loads of known state, `max(x, -infinity)`, the
`exp(-infinity - m_new)` rescale path, multiplication by zero, and the
associated loop/address bookkeeping.

## Optimization 2: Direct Packed-O Accumulation

Legacy lowering allocated and cleared an MLEN-wide temporary output matrix for
each Q head, updated it across K blocks, normalized it, then copied the useful
HLEN lane into packed `O_full`.

The optimized lowering clears `O_full` once and accumulates directly into the
target packed lane:

1. `V_SHIFT_V` places the PV result at the target head offset.
2. `C_SET_V_MASK_REG` enables only the target HLEN lane.
3. The first K block adds PV directly, with no zero-output rescale.
4. Later blocks rescale only the target lane by the online-softmax factor.
5. Final `1/l` normalization is also restricted to the target lane.

The explicit lane mask prevents one logical GQA group from overwriting other
groups sharing the same compact Q/O storage block. This uses existing ISA and
does not assume that RTL can broadcast several different KV heads in one
operation.

## Metadata and Auditability

The compiler and CostEmitter now report:

```text
packed_attention_schedule
softmax_first_block_specialized_count
softmax_state_initializations_elided
temporary_o_matrices_elided
direct_o_lane_updates
qk_compute_count
pv_compute_count
qk_recompute_factor
kv_reload_factor
```

These fields are also flattened into each DSE trial record. They make it
possible to distinguish useful QK/PV work from repeated work and to verify
that a modeled latency change came from the selected compiler schedule.

## Numerical Validation

The existing comparison tolerance, PASS ratio, golden generation, and output
format were not changed.

| Test | Configuration | Result |
|---|---|---|
| Single K block | `batch=4, seq=7, MLEN=VLEN=16, BLEN=4, HLEN=4, Hq/Hkv=4/2` | PASS, 100% all-close, max absolute error 0.003906 |
| Multiple K blocks | `batch=2, seq=39, MLEN=VLEN=16, BLEN=4, HLEN=4, Hq/Hkv=4/2` | PASS, 100% all-close |
| Multi-block A/B | Same multi-block input, legacy versus optimized | Output-region SHA256 identical: `9887d46bc62a6035ee140c370cb4b3ab27f5f8cadd9b7e307a7d82c91af39e55` |

The multi-block case is the stronger recurrence test because it exercises the
special first block followed by unchanged general online-softmax blocks. Its
transactional latency changed from 145,719 to 124,789 cycles while the final
output remained bitwise identical.

Focused regression results after integration:

```text
12 compiler/CostEmitter frontend tests passed
35 rtl-v1/V4 performance-model tests passed
17 Qwen3-32B DSE integration tests passed
```

The full PLENA compiler test directory was also started, but was not used as an
acceptance claim because that broad run was interrupted during environment
cleanup. The focused suites above cover every modified API and lowering path.

## Qwen3-32B A/B Method

The target experiment fixes all inputs except the attention schedule:

```text
model: Qwen3-32B dense, 64 decoder layers
sequence: 482
batch: 16
hardware: HLEN=128, VLEN=MLEN
precision: W=MXFP_E2M1, ACT=MXFP_E1M2, KV=MXFP_E4M3,
           FP=E6M5, INT=16
timing: rtl-v1 resource work, stage-wise roofline
memory: production-DMA HBM service model V4
clock conversion: 1000 ps/cycle
layout: compact batch/head layout in both arms
```

Source artifacts are in
`Workspace/qwen3_32b_dense_analytic/runs/packed_attention_v2_ab_20260718/`.
They can be regenerated with:

```bash
python Workspace/qwen3_32b_dense_analytic/benchmark_packed_attention_schedule.py \
  --output-dir Workspace/qwen3_32b_dense_analytic/runs/packed_attention_v2_ab
```

## Qwen3-32B Results

| MLEN/BLEN | One-layer legacy | One-layer optimized | Compute-work reduction | 64-layer legacy | 64-layer optimized | Roofline reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 497,517,280 cycles | 447,105,760 cycles | 10.13% | 31.062 s | 27.836 s | 10.39% |
| 1,024/1,024 | 371,552,462 cycles | 321,266,894 cycles | 13.53% | 23.318 s | 20.100 s | 13.80% |
| 2,048/1,024 | 344,567,649 cycles | 294,530,913 cycles | 14.52% | 21.706 s | 18.503 s | 14.75% |

The optimized result is slightly smaller than the initial static estimate at
the 512 point and reaches the expected 13-15% range for the larger machines.
No timing coefficient was fitted to these measurements.

### Work-Invariance Checks

| MLEN/BLEN | QK count | PV count | `H_PREFETCH_M` | `H_PREFETCH_V` | `H_STORE_V` | V4 memory work |
|---:|---:|---:|---:|---:|---:|---:|
| 512/512 | 1,024 / 1,024 | 1,024 / 1,024 | 9,565 / 9,565 | 1,217 / 1,217 | 256 / 256 | identical |
| 1,024/1,024 | 512 / 512 | 512 / 512 | 1,848 / 1,848 | 345 / 345 | 128 / 128 | identical |
| 2,048/1,024 | 256 / 256 | 256 / 256 | 502 / 502 | 234 / 234 | 128 / 128 | identical |

Each pair is `legacy / optimized`. These equalities are important: the
latency reduction is not caused by dropping attention products or memory
transfers. The one-layer V4 memory predictions are respectively 1.910 ms,
1.146 ms, and 1.150 ms in both arms.

The dominant optimized-minus-legacy dynamic opcode changes are:

```text
S_ADDI_INT    -3.5M to -3.9M per layer
C_LOOP_END    -2.036M per layer
S_ST_FP       -1.542M per layer
S_LD_FP       -1.481M to -1.483M per layer
S_LUI_INT     about -0.99M per layer
V_MUL_VF      -0.987M per layer
S_SUB/MAX/EXP/MUL_FP and V_ADD_VV:
               each -0.494M per layer
```

The optimized schedule adds only the explicit packed-lane masks: 2,048,
1,024, and 512 `C_SET_V_MASK_REG` operations per layer for the three shapes.
This opcode signature matches the intended implementation: large reductions
in softmax state traffic, zero-output work, and loop bookkeeping, with a small
new masking cost.

## Exhaustive DSE After Integration

After the controlled A/B validation, the optimized schedule was used for an
exhaustive Qwen3-32B grid search. The run fixed the compact native layout and
varied the hardware shape, scalar integer width, and all 103 software
precision profiles with accuracy greater than 0.9:

```text
MLEN = VLEN = [128, 256, 512, 1024, 2048]
BLEN         = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
INT width    = [16, 32, 64]
profiles     = 103
grid points  = 13,905
```

The objective was the 64-layer `rtl-v1` plus production-DMA V4 stage-wise
roofline latency. Area used the precision-aware structural proxy, with
908.6 mm2 as the feasibility ceiling and 826 mm2 as the A100 reference.

Final accounting was:

```text
12,051 complete designs
 1,854 constraint-pruned designs
    30 historical failed attempts, all requeued to a settled grid point
13,905 unique settled grid points
```

Every completed design used `direct-first-block-v1`, compact native layout,
and exactly 8,192 physical token rows for the 7,712 logical tokens. All
latency and area values were finite and non-negative.

### Best Latency by Matrix Width

| MLEN | Best BLEN | Full-decoder latency | Area | Relative to previous MLEN |
|---:|---:|---:|---:|---:|
| 128 | 128 | 183.494 s | 6.09 mm2 | - |
| 256 | 256 | 61.522 s | 23.70 mm2 | -66.47% |
| 512 | 512 | 27.836 s | 93.54 mm2 | -54.75% |
| 1,024 | 1,024 | 20.100 s | 371.69 mm2 | -27.79% |
| 2,048 | 1,024 | 18.503 s | 846.53 mm2 | -7.94% |

This is the expected direction after removing compiler-created padding and
softmax/output bookkeeping: increasing MLEN no longer produces the previous
spurious latency regression. The benefit still diminishes at large widths,
because the remaining attention reductions and scalar/vector work do not
scale at the same rate as matrix throughput.

The deterministic lowest-area tie-break among the fastest points is trial
402:

```text
MLEN/VLEN/BLEN = 2048/2048/1024
precision      = W:E2M1, ACT:E1M2, KV:E4M3, FP:E6M5, INT=16
accuracy       = 0.98
latency        = 18.503496 s
nominal area   = 846.53 mm2
P90 area       = 854.47 mm2
```

The closest nominal-area design to the 826 mm2 A100 reference is trial 11742:

```text
MLEN/VLEN/BLEN = 2048/2048/1024
precision      = W:E1M2, ACT:E5M2, KV:E1M2, FP:E6M5, INT=16
accuracy       = 0.92
latency        = 18.503506 s
nominal area   = 834.65 mm2
P90 area       = 842.50 mm2
```

The fastest MXINT design is `2048/2048/1024`, W/ACT/KV=MXINT4 and
FP=E5M6. It reaches 18.575900 s at 646.48 mm2: 0.39% slower than the fastest
MXFP design but 23.6% smaller. This confirms that the structural area model
retains the expected MXINT density advantage; the absolute fastest latency is
selected by the opcode timing family rather than by area.

The final figure is available at
[qwen3_32b_latency_area_scatter_packed_attention_v2.png](../../qwen3_32b_dense_analytic/runs/roofline_v4_grid_packed_attention_v2_20260718/qwen3_32b_latency_area_scatter_packed_attention_v2.png).
The complete machine-readable results are in
`Workspace/qwen3_32b_dense_analytic/runs/roofline_v4_grid_packed_attention_v2_20260718/`.

### DSE Runtime Engineering

The exhaustive run uses two content-addressed persistent caches. The compiler
produced 39 distinct shape traces, with 12,012 completed trials hitting an
existing trace. V4 produced 1,755 distinct memory-work entries; after the
cache was introduced, 9,468 trials hit a persisted V4 result and 1,755
created one. The first 828 completed trials predate the persistent V4 cache
metadata but use the same V4 equations. Cache locks and atomic replacement
make both caches safe across the 64 worker processes.

The CSV intentionally preserves the 30 interrupted attempts for auditability.
They are not extra design points and do not appear as unresolved failures in
the final grid.

## Interpretation and Limits

The A/B evidence supports three claims:

1. The optimized equations are numerically equivalent on single- and
   multi-block transactional tests.
2. The useful QK, PV, and HBM work is preserved exactly in the realistic cost
   traces.
3. The measured model reduction is fully attributable to deleted emitted
   instructions; `rtl-v1` and V4 coefficients are unchanged.

The reported 64-layer values are stage-wise roofline estimates, not
cycle-exact scheduled RTL makespans. The 1 GHz conversion is an assumption and
does not imply timing closure. Memory work is unchanged, so this optimization
primarily benefits the compute-bound objective.

All production-size DSE candidates are currently marked `exploratory`: the
`rtl-v1` report contains unsupported opcodes and structural timing/area
extrapolation beyond the directly calibrated RTL shapes. Consequently, the
exhaustive sweep establishes internally consistent ranking and compiler-work
trends, but the 18.50 s result must not be presented as cycle-exact silicon
latency.

The remaining `qk_recompute_factor` is 4 at MLEN=512 and 8 at MLEN=1,024 and
2,048; `kv_reload_factor` is 1 in all three experiments. This identifies QK
reuse, rather than KV reloading, as the next packed-GQA scheduling target.
Long-context work must separately account for the capacity needed to preserve
per-head softmax state across K windows; it cannot be presented as a free
compiler-only optimization when it requires larger FP SRAM.

## Exhaustive DSE Validation

The optimized schedule was subsequently used for an exhaustive grid over all
103 accepted software precision profiles and all hardware tuples in the
Qwen3-32B search space. This is a fresh run: no legacy-schedule trial was
reused as an optimized result.

```text
run directory: Workspace/qwen3_32b_dense_analytic/runs/
               roofline_v4_grid_packed_attention_v2_20260718
requested unique designs: 13,905
complete designs:         12,051
constraint-pruned:         1,854
missing unique designs:        0
historical failed attempts:   30
Pareto designs:               42
workers:                      64
```

The 30 failed rows are attempts interrupted during a deliberate pause and are
all classified as `interrupted_worker_requeued`. They were requeued and
settled successfully; the deduplicated `grid_trials.csv` therefore contains
only 12,051 complete and 1,854 valid constraint-pruned rows. The unique key
`(precision profile, MLEN, VLEN, BLEN, INT_DATA_WIDTH)` occurs exactly once in
that file.

The DSE used:

```text
latency objective: stage-wise roofline, rtl-v1 compute + production-DMA V4
area: structural precision-aware area v4 + ASAP7 SRAM macro table
schedule: direct-first-block-v1
sequence/batch: 482 / 16
VLEN = MLEN
A100 reference: 826 mm2
feasibility budget: 908.6 mm2 (110% of A100)
```

### Scaling Trend

The table selects the highest-accuracy, smallest-area trial among exact
latency ties at each MLEN.

| MLEN | Best BLEN | 64-layer latency | Area | Accuracy |
|---:|---:|---:|---:|---:|
| 128 | 128 | 183.494 s | 6.078 mm2 | 0.98 |
| 256 | 256 | 61.522 s | 23.676 mm2 | 0.98 |
| 512 | 512 | 27.836 s | 93.489 mm2 | 0.98 |
| 1,024 | 1,024 | 20.100 s | 371.591 mm2 | 0.98 |
| 2,048 | 1,024 | 18.503 s | 846.531 mm2 | 0.98 |

The final optimum no longer occurs at MLEN=512. Increasing MLEN to 1,024 and
2,048 now reduces the emitted work as expected; the remaining improvement from
1,024 to 2,048 is smaller because BLEN is capped at 1,024 and attention/vector
work remains. This result is consistent with the fixed-work A/B experiment and
is evidence that the earlier 512 optimum was caused by compiler schedule
overhead, not by an inherent benefit of the smaller MatrixMachine.

The minimum modeled latency is shared by 72 precision/INT configurations
because the calibrated opcode timing is identical for those equal-width
compute paths. Applying the deterministic tie-break (highest accuracy, then
smallest area) selects:

```text
W=MXFP_E2M1, ACT=MXFP_E1M2, KV=MXFP_E4M3, FP=E6M5, INT=16
MLEN=VLEN=2048, BLEN=1024
latency=18.503496 s
area P50/P90=846.531/854.467 mm2
accuracy=0.98
```

The exact closest-to-826-mm2 point is:

```text
W=MXFP_E1M2, ACT=MXFP_E5M2, KV=MXFP_E1M2, FP=E6M5, INT=16
MLEN=VLEN=2048, BLEN=1024
latency=18.503506 s
area P50/P90=834.651/842.496 mm2
accuracy=0.92
```

Both are structural extrapolations beyond the DC MatrixMachine calibration
domain (`MLEN<=64`, `BLEN<=16`). They are useful architecture-ranking results,
not signoff area predictions. A diagnostic view restricted to `BLEN<=16`
finds a much slower minimum of 121.654 s at `MLEN=1024, BLEN=16`, illustrating
the cost of avoiding BLEN extrapolation rather than an alternative production
recommendation.

### Runtime and Cache Audit

The run directory was created at 21:58 on 18 July and finalized at 02:13 on
19 July. This 4 h 15 min wall interval includes a manual pause and resume. The
resume path recovered all 30 interrupted jobs. Cross-process caches contain 39
compiler traces and 1,755 V4 aggregate memory-work entries; lock files are
separate from data artifacts. A two-process cache smoke verified that a V4
miss followed by a hit produced exactly equal compute, memory, roofline, and
serial latency values.

### Figures

- `qwen3_32b_latency_area_packed_attention_v2.png`: all 12,051 complete designs.
- `qwen3_32b_latency_area_packed_attention_v2_blen16.png`: `BLEN<=16`
  calibration-domain diagnostic.

Both figures use color for MXINT/MXFP software families and marker shape for
GPTQ weight precision. Only the lowest-latency and closest-A100 points are
annotated.
