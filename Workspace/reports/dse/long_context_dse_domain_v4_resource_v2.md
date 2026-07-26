# Long-Context DSE Domain, V4, and Resource Scheduling v2

> **Historical domain and execution milestone, audited 2026-07-26.** The V4
> sufficient-statistics backend and resource-management lessons remain
> current. The exact search domain and the completed-study selectors below
> predate final-lineage tile-aware TP x CP x EP modeling and must not be used
> as current Pareto evidence.

## Scope

This revision changes the formal Qwen3-32B long-context DSE representation and
execution infrastructure. It does not change HBM V4 coefficients, ideal-II1
compute semantics, area coefficients, power coefficients, or the four
objectives.

The formal workload remains:

```text
seq_len=32768, batch_size=1, decoder_layers=64
```

## Canonical Hardware Domain

The current `MLEN=VLEN` and conditional BLEN domain contains exactly 29
shapes:

```text
M256:  B32, B64, B128, B256
M512:  B32, B64, B128, B256, B512
M1024: B32, B64, B128, B256, B512, B1024
M2048: B32, B64, B128, B256, B512, B1024
M4096: B32, B64, B128, B256, B512
M8192: B32, B64, B128
```

`MLEN=128` and `BLEN=4/8/16` were removed. `M4096/B1024` is excluded because
the minimum-precision physical design already exceeds the 908.6 mm2 aggregate
area budget. M8192 is deliberately limited to B32/B64/B128 and is an
architectural sensitivity range rather than a calibrated RTL/DC domain.
M4096 and M8192 points carry:

```text
large_mlen_structural_extrapolation=true
```

M8192 additionally carries:

```text
very_large_mlen_structural_extrapolation=true
```

The chip-count domain is now `[1,2,4,8,16]`. Canonical sampling avoids
structurally dominated large-array/many-chip cross-products:

```text
N=1/2/4: MLEN <= 8192
N=8:     MLEN <= 4096
N=16:    MLEN <= 2048
```

MLEN uses a chip-count-specific Optuna parameter, so these limits do not
create structural PRUNED trials.

RTL-v4 compact statistics support at most 16 segments. M4096 has 32
`HLEN=128` segments per vector word, so Q normalization uses the existing
single-segment reduction/RSQRT/masked-scale instructions and reports
`segment_parallel_fallback`. This preserves real ISA costs instead of
inventing a 32-lane compact unit.

## Exact HBM V4 Aggregation

`sufficient-statistics-v2` keeps the same V4 equation and calibration
coefficients. It replaces physical manifest materialization with exact
statistics sufficient for the model:

- physical and payload read/write bytes;
- physical request counts;
- critical channel, pseudochannel, bank-group, and bank loads;
- cold row misses and within-bank row conflicts;
- theoretical phase floor and calibrated residual features.

Read streams use batched NumPy MOP4CLXOR mapping. Overlapping MX element and
scale line-runs are unioned at 64-B granularity before mapping. Write and RMW
geometries retain the scalar planner because the read phase changes write
open-row state. Equal feature signatures are accumulated with `math.fsum`.

The implementation reports:

```text
v4_aggregation=affine_feature_grouped_v2
exact_feature_equivalence=true
unique_address_geometry_count
unique_feature_signature_count
scalar_fallback_count
```

The persistent cache schema is
`v4_work_v5_affine_feature_grouped_stage_scaled`.

## Exactness Evidence

The complete HBM V4 test suite passes:

```text
15 passed in 94.30 s
```

It compares grouped and literal occurrence paths for latency, floor, physical
traffic, payload traffic, requests, stage/opcode attribution, row-state
regime, and extrapolation metadata. Focused overlapping-line and plain-format
vectorized tests are exactly equal to the scalar request planner.

The DSE domain/schema regression suite passes:

```text
25 passed in 1.60 s
```

The shared normalization suite, including the M4096 fallback, passes:

```text
9 passed in 0.82 s
```

## Performance Evidence

Worst retained small-Matrix shape:

```text
M256/B32, streaming SRAM, representative MXFP profile
```

| Metric | Before v2 | Final v2 |
|---|---:|---:|
| V4 aggregation | 44.59 s | 24.05 s |
| Full zero-cache CostEmitter | 64.19 s | about 43.5 s |
| Peak RSS | about 1.10 GiB | about 1.10 GiB |

The V4 phase meets the 30-second target. A completely empty compiler trace
cache still adds about 19.5 seconds of lowering; this is reported separately
and is not attributed to V4.

M4096/B32 completes successfully:

| Metric | Result |
|---|---:|
| Trace lowering | 49.46 s |
| V4 aggregation | 10.34 s |
| Full CostEmitter | 59.81 s |
| Peak RSS | 0.57 GiB |

Its slower lowering is caused by the honest 32-segment Q-normalization
fallback, not V4.

## Progress and Resource Control

Workers publish phase, progress, stream, current RSS, peak RSS, and available
memory. A worker is considered stalled only after:

```text
15 minutes without progress
AND 120 seconds below 5% process-tree CPU
```

There is no hard wall-time limit for a progressing, CPU-active trial.

The parent:

- launches up to eight workers every two seconds;
- supports up to 64 workers;
- uses phase-specific P90 RSS predictions;
- stops launching below 22 GiB available memory;
- resumes above 26 GiB;
- kills only the largest worker below the 18 GiB emergency floor;
- recycles a worker after its current trial above 2.5 GiB peak RSS;
- enforces a 6 GiB process-tree hard limit;
- forces BLAS, OpenMP, Torch, and NumExpr to one thread per worker.

## Parallel Smoke

```text
64 COMPLETE / requested 60 workers
0 PRUNED / 0 FAIL
wall time = 115.50 s
maximum concurrency = 60
steady-state system CPU = 98.96%
peak active process-tree RSS = 28.63 GiB
peak individual worker RSS = 1.00 GiB
minimum MemAvailable = 91.26 GiB
parent terminations = 0
output size = 100 MiB
```

The 8 COMPLETE / 4-worker precursor also completed with zero PRUNED/FAIL and
0.76 GiB peak worker RSS.

## Formal Run

The 2,048-COMPLETE long-context study below predates the M8192/N16 expansion
and remains the formal baseline for the previous M256-M4096/N1-N8 domain. It
was started in:

```text
tmux session: qwen3_dse_long_grouped_v2
run directory: /tmp/qwen3_32b_dse_long_32768x1_grouped_v2
launcher log: /tmp/qwen3_dse_long_grouped_v2.log
```

The study completed cleanly:

```text
2,048 COMPLETE / 0 PRUNED / 0 FAIL
wall time: 20 min 24 s
Pareto trials: 114
maximum concurrency: 60
mean system CPU utilization: 87.60%
peak active process-tree RSS: 30.80 GiB
peak individual worker RSS: 1.13 GiB
minimum MemAvailable: 89.13 GiB
parent terminations: 0
```

Workers in V4 aggregation reported increasing geometry counters in their
heartbeats. No active high-CPU trial was treated as stalled. The run used the
original eight-trial process cap and therefore launched 256 workers, including
196 count-based replacements. No replacement was requested by RSS or system
memory.

After this run, the default was changed to:

```text
workers = 64
worker_max_trials_per_process = 0  # no count-based recycling
```

RSS, process-tree, progress, and system-memory protections remain enabled.

### Selected Designs Under 908.6 mm2

| Selection | Latency | Area | Energy | Accuracy | Chips | M/B | SRAM |
|---|---:|---:|---:|---:|---:|---:|---|
| Fastest nominal-feasible | 4.051 s | 879.55 mm2 | 1714.2 J | 0.92 | 8 | 4096/128 | 2-tile streaming |
| Lowest energy | 15.710 s | 194.18 mm2 | 786.6 J | 0.92 | 8 | 1024/64 | 25-tile projection-full |
| Best energy-delay product | 4.615 s | 863.54 mm2 | 1244.3 J | 0.92 | 8 | 2048/256 | 2-tile streaming |
| Fastest at accuracy 0.98 | 5.992 s | 450.90 mm2 | 1126.3 J | 0.98 | 8 | 2048/128 | 2-tile streaming |
| Closest below 826 mm2 | 14.646 s | 822.93 mm2 | 1977.8 J | 0.98 | 4 | 1024/512 | 25-tile projection-full |

The fastest nominal-feasible point also has a P90 area estimate of
907.34 mm2, just below the 908.6 mm2 constraint.

Optimizing each chip count independently gives minimum feasible latencies of:

```text
1 chip: 19.093 s
2 chips: 12.453 s
4 chips:  5.849 s
8 chips:  4.051 s
```

The 8-chip result is 4.71x faster than the separately optimized single-chip
candidate, not an 8x linear speedup.

The completed study contains no M8192 or N16 samples. A new study directory is
required because the current search schema is
`canonical_conditional_hardware_v4_m8192_n16`.

End-to-end fixed-point smoke tests confirm that both additions traverse the
complete CostEmitter, HBM V4, area, power, and four-objective path:

| Smoke | Latency | Aggregate area | Result |
|---|---:|---:|---|
| M8192/B32, N=4 | 64.855 s | 909.08 mm2 | COMPLETE, nominally 0.48 mm2 over budget |
| M2048/B128, N=16 | 2.882 s | 2136.21 mm2 | COMPLETE, infeasible under area constraint |

> **Erratum:** the 64.855 s M8192 result above was generated before
> `ffn_address_schedule=live-stride-v1`. It contained 173.947B cycles of dead
> or repeatedly legalized FFN address updates. The corrected controlled result
> is 21.796 s with identical Matrix and HBM work. See
> [`long_context_dse_anomaly_audit_v1.md`](long_context_dse_anomaly_audit_v1.md).

The second point is intentionally not representative of the useful N16
region. It verifies plumbing; feasible N16 candidates must use substantially
smaller per-chip arrays.

### SRAM Observation

Only 18 of 114 Pareto trials use `projection-full`; the remaining 96 use
two-tile streaming. No `kv-50/75/100` candidate is Pareto-optimal in this
sample. Partial residency can materially improve isolated small-MLEN points,
but at the globally competitive large-MLEN/multi-chip points, compute remains
dominant and the extra SRAM area usually provides less value than more compute
or additional chips.

## Claim Boundary

The grouped V4 result is exact for the current cold,
cached-occurrence-equivalent V4 semantics. It is not online Ramulator
co-simulation. M4096 and M8192 are structural area/timing extrapolations.
Their normalization paths use existing-ISA fallbacks rather than validated
32/64-segment compact-stat RTL implementations.
