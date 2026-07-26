# Short-Context DSE Results and Long-Context CostEmitter Analysis

> **Historical search result, audited 2026-07-26.** The CostEmitter
> performance diagnosis and algebraic-compression motivation remain valid.
> The selected DSE points predate RTL-v4, affine-loop-v2, and final-lineage
> tile-aware multi-chip modeling and are not current Pareto claims.

## Scope

This report records the completed Qwen3-32B short-context DSE and the diagnosis
of the stopped long-context run.

> Update: the algebraic CostTrace/V4 work proposed below has now been
> implemented and accepted. See
> [`long_context_costemitter_algebraic_compression_v1.md`](long_context_costemitter_algebraic_compression_v1.md)
> for the implementation, parity evidence, 27.03-second cold result,
> 45.8-millisecond warm result, and parallel smoke data.

```text
Short: seq_len=482, batch_size=16
Long:  seq_len=32768, batch_size=1
Model: Qwen3-32B, 64 decoder layers
```

The search uses the current four objectives:

```text
minimize latency
minimize aggregate physical silicon area
minimize system energy
maximize accuracy
```

The latency model is RTL-v4 compiler lowering with ideal-II1 compute timing,
HBM V4, partial K/V residency, ideal dual-port SRAM, loop-AGU-v1, and
optimistic TP+SP multi-chip scaling.

## Short-Context Completion

The short run completed successfully:

| Metric | Result |
|---|---:|
| COMPLETE trials | 2,048 |
| PRUNED trials | 0 |
| FAIL trials | 0 |
| Effective completion rate | 100% |
| Pareto trials | 245 |
| Nominal-area feasible trials (`<=908.6 mm2`) | 1,989 |
| Trials within 5% of 826 mm2 | 21 |
| Trial execution interval | 26 min 50 s |
| Median / P95 evaluation time | 14.8 s / 142.0 s |
| Peak worker RSS | 3.16 GiB |
| Minimum observed `MemAvailable` | 59.0 GiB |

The canonical conditional encoding eliminated the previous structural pruning.
The worker pool launched 327 processes and recycled 267 of them; no recycle was
triggered by system-memory pressure.

## Selected Short-Context Designs

All energy values cover the fixed 7,712-input-token workload.

| Selector | Trial | Chips | MLEN/BLEN | SRAM policy/tiles | Latency | Area | Energy | Energy/token | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Fastest under 908.6 mm2 | 1737 | 8 | 512/512 | projection-full / 50 | 296.67 ms | 821.33 mm2 | 217.56 J | 28.21 mJ | 0.94 |
| Lowest energy under budget | 1588 | 8 | 256/256 | projection-full / 100 | 1303.78 ms | 248.22 mm2 | 58.81 J | 7.63 mJ | 0.92 |
| Highest accuracy under budget | 3 | 8 | 512/512 | streaming / 2 | 465.78 ms | 824.27 mm2 | 249.66 J | 32.37 mJ | 0.98 |
| Closest below 826 mm2 | 99 | 8 | 512/512 | streaming / 2 | 465.80 ms | 825.89 mm2 | 270.45 J | 35.07 mJ | 0.92 |
| Best energy-delay product | 1739 | 4 | 512/512 | projection-full / 50 | 825.91 ms | 512.42 mm2 | 69.97 J | 9.07 mJ | 0.92 |

The fastest feasible design uses:

```text
weight = MXFP E2M1
activation = MXFP E1M2
KV = MXFP E1M2
internal FP = E6M5
INT_DATA_WIDTH = 16
```

The lowest-energy feasible design uses MXINT4 for weight, activation, and KV,
with internal FP E6M5 and `INT_DATA_WIDTH=16`.

### Chip-count sensitivity

The fastest feasible point for each chip count is:

| Chips | Latency | Area | Energy | MLEN/BLEN | SRAM tiles |
|---:|---:|---:|---:|---:|---:|
| 1 | 1439.77 ms | 520.40 mm2 | 465.92 J | 2048/512 | 13 |
| 2 | 1162.02 ms | 231.82 mm2 | 266.52 J | 512/512 | 50 |
| 4 | 529.03 ms | 742.32 mm2 | 211.50 J | 1024/512 | 25 |
| 8 | 296.67 ms | 821.33 mm2 | 217.56 J | 512/512 | 50 |

The optimistic eight-chip estimate is 4.85x faster than the fastest sampled
single-chip point under the same aggregate area budget. This is not an
eight-chip RTL result: it depends on the stage-level optimistic TP+SP scaling
and the assumed NVLink-class communication model.

## Claim Boundary

The search itself completed, but no completed point is currently classified as
RTL-fidelity-qualified. The main reason is the unimplemented
`M_BTMM/M_BMM_WO` broadcast datapath:

```text
broadcast_rtl_validated = false
broadcast timing = ordinary Matrix structural equivalent
```

The reported latency is therefore an architecture estimate, not cycle-exact
RTL latency. It also assumes ideal-II1 Vector/Scalar/Control timing, ideal
dual-port SRAM, ideal hierarchical clock gating, and optimistic TP+SP
partitioning. The power result is a calibrated/literature shadow model rather
than signoff power.

## Long-Context Run Diagnosis

The long run was stopped after approximately 53 minutes:

```text
COMPLETE = 0
RUNNING  = 60
WAITING  = 68
```

One process was building the shared compiler report at approximately 7.5 GiB
RSS and one fully occupied CPU. Nearly all other workers were sleeping in
`locks_lock_inode_wait` on the same report-cache key. The shared lock correctly
prevented duplicate multi-GiB trace construction, but it also exposed the real
bottleneck: one long-context CostEmitter trace is too expensive to construct.

For `seq=32768, MLEN=512`:

```text
Q blocks = K blocks = 64
causal Q/K block pairs = 64*65/2 = 2,080
```

The packed-GQA lowering enumerates these block pairs in Python before the final
CostTrace compression. Each pair constructs QK, mask, softmax, PV, DMA,
EnergyAction, and schedule objects. `ScheduleRepeat` compresses row loops, but
it does not remove the 2,080 block-pair objects. The AGU optimization and final
trace serialization happen only after this construction.

HBM V4 has the same scaling issue. Formal DSE uses cached cold-occurrence
semantics, but `V4DmaServiceProvider.aggregate()` still loops over every
occurrence and builds a cycle sequence even when many occurrences map to the
same cached geometry.

There is also substantial output duplication. The completed short run occupies
approximately 12 GiB:

```text
shared compiler trace cache   1.2 GiB
shared compiler report cache  2.1 GiB
per-trial report copies       approximately 8 GiB
```

On a cache hit, the DSE still writes the complete compiler report into each
trial directory. This is unnecessary for analysis and would be much worse for
long-context reports.

## Recommended CostEmitter Optimization

### 1. Exact algebraic attention summary

Add a DSE-only `affine-block-summary-v1` trace granularity. It should retain the
same compiler kernel templates but represent block classes algebraically:

```text
first K block
recurrent K block
causal diagonal block
full off-diagonal block
tail block
resident K/V block
streamed K/V block
```

For ideal-II1 timing, opcode counts and stage work can be multiplied exactly.
No row or Q/K-pair schedule object is needed. This is exact relative to the
current formal DSE semantics; it is not a new latency approximation.

The detailed ordered trace remains available for transactional and RTL-v1
validation. It should not be used for the 32K formal sampler.

### 2. Algebraic V4 occurrence aggregation

For the current `one-layer-cached-occurrence-scaled` mode, group affine DMA
occurrences by the existing normalized V4 geometry key and multiply the
prediction and traffic counters by group size. Do not construct per-occurrence
cycle arrays when scheduled replay is disabled.

This preserves the current cold-geometry V4 semantics. Stateful observed-DMA
validation and scheduled replay must continue to use the ordered path.

### 3. Summary-only DSE artifacts

Store one complete report per shared cache key. A trial should contain:

```text
compiler report cache key/path/hash
objective values
stage/component summaries
fidelity and extrapolation metadata
```

It should not copy the full report. Use an LRU of extracted summaries in each
worker rather than retaining many full reports. Slim the selector JSON files so
they do not embed every large nested report field.

### 4. Avoid cold-cache worker herds

The current 103 precision anchors all begin at `MLEN=512, BLEN=512`. Interleave
hardware anchors across MLEN, BLEN, chip count, and SRAM configurations so the
first worker wave requests distinct compiler trace keys. Canonicalize the trace
cache key independently of accuracy/profile name; only precision fields that
change Matrix timing, V4, or power should remain in the evaluated-report key.

### 5. Mid-trial safeguards and profiling

Record phase timings and peak RSS for:

```text
layout/lowering
CostTrace finalization
AGU rewrite
ideal-II1 work
V4 aggregation
power
JSON serialization
```

The parent should monitor live worker RSS and elapsed time. The current recycle
check runs only after a trial and cannot protect against one 7.5-GiB build.

## Acceptance Criteria

Before restarting the long run:

1. Exact-summary and detailed paths must match at `seq=482` and `seq=4096` for
   opcode histograms, stage work, DMA occurrences/bytes, V4 work, power actions,
   and final roofline latency.
2. `seq=32768, MLEN=512` cold evaluation should complete in under 30 seconds
   with peak RSS below 2 GiB.
3. A warm evaluation should complete in under 1 second.
4. A `64 COMPLETE / 32 workers` long-context smoke must have zero structural
   prune/fail, no cache-lock herd, and no per-trial full-report copies.

These checks are complete. The replacement long-context run uses the clean
`affine_summary_v1` schema and does not resume the interrupted diagnostic
study.

## Artifacts

```text
/tmp/qwen3_32b_dse_short_482x16_canonical_v1/
Workspace/reports/dse/figures/short_482x16_latency_vs_aggregate_area.png
Workspace/reports/dse/figures/short_482x16_energy_vs_aggregate_area.png
Workspace/reports/dse/figures/short_482x16_latency_vs_energy.png
```
