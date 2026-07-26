# Canonical Conditional DSE Strategy v1

> **Current mechanism, historical domain snapshot.** Conditional BLEN/SRAM
> encoding, COMPLETE-trial budgeting, and dynamic workers remain in the
> formal DSE. The exact MLEN/chip-count domain and model schema in this report
> have since expanded; use `../system_validation_status.md` for current
> settings.

## Purpose

The previous sampled DSE encoded legal constraints as post-sampling pruning.
In the interrupted short/long run, only 1,105 of 2,048 attempts completed;
943 attempts (46.0%) were pruned. The main causes were illegal `BLEN > MLEN`
topologies and Matrix-SRAM policies that mapped to an already sampled physical
cache configuration.

This revision moves those constraints into the search representation. The
formal budget is now 2,048 **COMPLETE** trials per context, rather than 2,048
attempts.

## Canonical Search Encoding

`canonical-conditional-v1` samples `MLEN` first and then exposes only legal
BLEN values through a conditional parameter:

```text
BLEN_LOG2_MLEN_<M>
```

For every `(MLEN, chip_count, parallel_model)`, the six user-facing SRAM
policies are evaluated before sampling. Policies with the same:

```text
(physical_tiles, resident_prefix_blocks)
```

are collapsed into one ordered integer choice:

```text
SRAM_CONFIG_INDEX_M<M>_N<N>_<parallel>
```

Each trial retains the canonical policy and all equivalent policy aliases.
The legacy encoding remains available as `legacy-policy-v1`.

The original v1 physical domains were:

| Context | Canonical hardware configurations | With 103 precision profiles |
|---|---:|---:|
| `seq=482, batch=16` | 936 | 96,408 |
| `seq=32768, batch=1` | 1,458 | 150,174 |

The current v2 long-context domain supersedes these counts: it removes M128 and
B4/B8/B16, adds M4096, and contains 972 canonical hardware configurations
(100,116 including 103 precision profiles). All BLEN values are legal and all
SRAM entries within each conditional domain
are physically unique. TPE may revisit an already evaluated complete
configuration, but it can no longer create a policy alias or illegal topology.

## Budget and Resume Semantics

The formal launcher uses:

```text
--target-complete-trials 2048
--max-total-attempts 2560
```

PRUNED and FAIL trials are automatically replaced until 2,048 COMPLETE trials
exist. Resume uses an absolute target and does not add another 2,048 trials.
Study metadata includes context, batch size, search encoding, hardware-domain
fingerprint, model hash, and precision artifact hash. Incompatible studies
are rejected.

## Aggressive Worker Policy

The 64-logical-core host now uses an aggressive but memory-bounded policy:

```text
worker limit                 60
hard trials per process       8
per-worker RSS recycle      2.5 GiB
initial memory token        1.5 GiB
stop spawning below        22.0 GiB MemAvailable
resume spawning above      26.0 GiB MemAvailable
emergency kill floor       18.0 GiB MemAvailable
known early-OOM threshold  15.0 GiB MemAvailable
```

Four logical CPUs and at least 5 GiB above the observed early-OOM threshold
remain reserved. Newly launched workers consume provisional memory tokens so
the parent cannot start a large burst before `/proc/meminfo` reflects Python
and native allocations. A worker also exits after its current trial if either
its RSS threshold is reached or system memory falls below the reserve.

Workers are reused for up to eight trials. This preserves process-local caches
without allowing the unbounded RSS growth seen in the previous long-lived
workers. The parent immediately fills vacated slots while memory permits.

Compiler reports are cached across processes by semantic cache key. A
per-key file lock prevents multiple workers from simultaneously constructing
the same expensive CostTrace; later trials read the shared report.

## Validation

### Static and unit tests

The focused suite contains 31 passing tests. It verifies:

- legal conditional BLEN domains;
- preservation and deduplication of all six SRAM policies;
- stable conditional parameter names;
- worker quota and resource accounting;
- accuracy/profile, area, HBM V4, multi-chip, and four-objective regressions.

### Integration results

| Test | Result | Efficiency | Peak worker RSS | Minimum available memory |
|---|---:|---:|---:|---:|
| 8 COMPLETE / 4 workers | 8 COMPLETE, 0 PRUNED, 0 FAIL | 100% | 1.31 GiB | 91.0 GiB |
| Resume same target | 0 new attempts | no overshoot | n/a | n/a |
| 64 COMPLETE / 8 workers | 64 COMPLETE, 0 PRUNED, 0 FAIL | 100% | 1.30 GiB | 88.6 GiB |
| 128 COMPLETE / 60-worker limit | 128 COMPLETE, 0 PRUNED, 0 FAIL | 100% | 1.24 GiB | 78.1 GiB |

The final 128-point smoke completed in approximately 136 seconds. It launched
64 short-lived workers in total, including four normal replacement launches.
No RSS- or memory-triggered recycle was required.

The broad hardware smoke exposed very expensive low-BLEN traces and concurrent
duplicate construction. This motivated the shared locked compiler-report
cache. The current domain removes M128 and BLEN below 32; see
`long_context_dse_domain_v4_resource_v2.md` for current evidence.

## Formal Runs

The launcher is:

```text
Workspace/qwen3_32b_dense_analytic/run_short_long_dse_aggressive.sh
```

It runs the contexts serially:

```text
short: seq=482,   batch=16
long:  seq=32768, batch=1
```

Both runs use RTL-v4 lowering, ideal-II1 compute timing, HBM V4,
partial-resident K/V, ideal dual-port SRAM, loop-AGU-v1, TP+SP multi-chip
scaling, ideal hierarchical clock gating, and the four objectives:

```text
min latency
min aggregate silicon area
min system energy
max accuracy
```

Plots and selectors are generated automatically after each run. The long run
starts only after the short run has produced 2,048 COMPLETE trials.

The formal tmux session was launched on 2026-07-25 as:

```text
qwen3_dse_canonical_short_long
```

The short-context run completed all 2,048 requested trials with zero PRUNED or
FAIL trials. Trial execution took 26 minutes 50 seconds. Peak worker RSS was
3.16 GiB and minimum observed `MemAvailable` was 59.0 GiB. The final Pareto set
contains 245 trials.

The subsequent long-context run was stopped after approximately 53 minutes
with zero completed trials. One `seq=32768, MLEN=512` CostEmitter build had
grown to approximately 7.5 GiB RSS while the other workers waited on its shared
cache lock. This is a CostEmitter long-context trace-construction bottleneck,
not a canonical-search or worker-memory-admission failure. The detailed
diagnosis and required algebraic trace optimization are recorded in:

```text
Workspace/reports/dse/short_context_482x16_results_and_long_costemitter_analysis.md
```

## Claim Boundary

This change improves search efficiency and resource utilization; it does not
change latency, area, energy, or accuracy semantics. Zero structural pruning
does not imply that every point is feasible under the A100 area constraint:
Optuna records that constraint for multi-objective sampling, while the physical
evaluation still completes and remains available for analysis.
