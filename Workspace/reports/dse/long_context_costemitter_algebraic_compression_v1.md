# Long-Context CostEmitter Algebraic Compression v1

## Purpose

The production DSE path uses ideal-II1 compute work, stage-wise roofline
composition, one-layer cached-occurrence HBM V4, and action-based system
energy. It does
not require an ordered instruction schedule. Before this change, CostEmitter
still materialized repeated packed-GQA work at Q/K-block granularity. A
`seq_len=32768, MLEN=512` layer has 2,080 causal Q/K block pairs, and old DSE
workers reached about 7.5 GiB RSS before completing a trial.

This implementation replaces that object expansion with exact algebraic
summaries. It does not change the compiler schedule, opcode multiplicities,
HBM V4 coefficients, area model, power coefficients, or DSE objectives.

## Implementation

### Algebraic CostTrace

`affine-block-summary-v1` classifies packed-GQA work by:

- first, recurrent, and last K block;
- causal diagonal and off-diagonal block;
- full and tail Q/K dimensions;
- resident and streamed K/V;
- full and tail broadcast group.

Each distinct kernel template is lowered once. Dynamic opcode counts,
parameterized timing variants, `EnergyAction`, stage work, QK/PV counts, and
cache statistics are multiplied by exact occurrence counts. The trace records
`materialized_block_pair_count=0` and does not provide an ordered schedule.

The public CostEmitter API remains `detailed` by default. Formal DSE defaults
to `affine-block-summary-v1`. Summary mode rejects RTL-v1 scheduling,
scheduled shadow, and observed-DMA replay because those modes need instruction
order.

### Affine HBM V4 Aggregation

Partial-resident K/V streams retain affine address axes. The cold V4 path
groups occurrences by exact normalized physical geometry and predicts each
unique key once. One exact layer is then scaled by stage to 64 decoder layers.
Traffic, stage/opcode/role breakdowns, floor, residual, occurrence count, and
domain metadata are all scaled from the same grouped census.

The current persistent cache schema is
`v4_work_v5_affine_feature_grouped_stage_scaled`; older V4 pickles are rejected.
Stateful V4 and observed-DMA validation continue to use literal occurrences.

### DSE Storage and Workers

- Compiler reports are stored once per semantic hardware/precision key.
- Trial directories contain a reference and compact objective/stage summary.
- Precision profile name and accuracy are excluded from the compiler cache key
  when the actual precision tuple is identical.
- Each worker retains at most four summary reports.
- Startup anchors are distributed across hardware, chip count, SRAM policy,
  and precision. Larger BLEN values are evaluated first to avoid an initial
  wave dominated by pathological cold V4 geometry counts.
- The parent stops and requeues a worker if process-tree RSS exceeds 6 GiB, or
  if a phase has no progress for 15 minutes while process-tree CPU remains
  below 5% for 120 seconds. There is no wall-time limit for progressing work.
  The 18 GiB emergency floor is separate from the 22/26 GiB launch
  pause/resume thresholds.

## Exactness Evidence

The following are compared between detailed and summary traces:

- static and dynamic opcode histograms;
- per-stage opcode histograms;
- parameterized Matrix and reduction variants;
- total DMA occurrences;
- structurally keyed `EnergyAction` count, busy cycles, and bytes.

Grouped V4 is also compared with literal cold-occurrence replay. Stage-scaled
one-layer V4 is compared with direct grouped evaluation using the same stage
multiplier. Integer traffic/count fields are identical; floating latency
fields agree within `1e-9 ns`.

The focused and regression suite completed:

```text
73 passed in 156.01 s
```

It covers HBM request semantics, grouped V4 parity, stage scaling,
CostEmitter detailed/summary parity, compiler-cost regressions, and canonical
DSE configuration logic.

## Worst-Point Acceptance

Configuration:

```text
Qwen3-32B, 64 layers
seq_len=32768, batch=1
MLEN=VLEN=512, BLEN=512
Matrix SRAM policy=streaming
ideal-II1 + one-layer-cached-occurrence-scaled V4
```

Results:

| Metric | Result |
|---|---:|
| Cold API evaluation | 27.03 s |
| Warm API evaluation | 45.8 ms |
| Peak RSS | 1,336,532 KiB (1.275 GiB) |
| Materialized Q/K pair objects | 0 |
| Logical V4 occurrences | 6,602,561 |
| Unique V4 geometries | 7,934 |
| Occurrences elided | 6,594,627 |
| Compute latency | 118,285,177,803 ns |
| HBM V4 work | 1,236,950,433.385 ns |
| Stage roofline latency | 118,285,192,132.139 ns |

Cold phase time:

| Phase | Time |
|---|---:|
| Trace/layout/census/lowering/finalization | 7.84 s |
| V4 aggregation | 18.84 s |
| ideal-II1 compute evaluation | 1.35 ms |
| stage roofline | 0.016 ms |

This meets the target of cold <=30 s, warm <=1 s, and RSS <=2 GiB.

## DSE Smoke Evidence

The fixed worst-shape parallel smoke produced:

```text
64 COMPLETE / 32 requested workers
0 PRUNED
0 FAIL
peak worker RSS = 1.617 GiB
wall time = 85.22 s
output size = 81 MiB
```

An additional stratified long-context smoke retained the complete hardware
domain:

```text
8 COMPLETE / 4 workers
0 PRUNED
0 FAIL
MLEN = 128, 256, 512, 1024, 2048 represented
```

The current v2 domain removes MLEN=128 and BLEN below 32, adds MLEN=4096, and
uses exact sufficient-statistics V4 aggregation. Current performance and
60-worker evidence are in `long_context_dse_domain_v4_resource_v2.md`.

## Remaining Performance Limitation

The following values describe the superseded v1 domain and motivated removal
of BLEN below 32:

| Shape | Unique V4 geometries | Cold V4 time |
|---|---:|---:|
| M512/B4 | 633,917 | 233.46 s |
| M1024/B4 | 295,832 | 110.51 s |
| M2048/B4 | 180,958 | 83.88 s |

These points remain valid and are not pruned. The cost is now V4 physical
geometry planning, not expanded softmax/PV/schedule objects. Further
compression would require a proven algebraic histogram of MOP4CLXOR features;
coarsening addresses at 64-byte granularity was tested and rejected because
it merged geometries with different V4 features.

## Interpretation

For the formal ideal-II1 DSE, this is an exact representation change rather
than a new latency approximation. RTL-v1 and observed-DMA validation still
use detailed ordered traces. Results from summary mode must therefore be
described as exact for the current ideal-II1 plus cold cached-occurrence V4
semantics, not as cycle-exact RTL scheduling.
