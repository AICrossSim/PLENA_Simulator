# Tile-Aware TP x CP x EP Multi-Chip Analytical Model v3

> **Correction notice (2026-07-26):** The controlled multi-chip tables in this
> implementation report were generated before final-schedule kernel lineage
> was preserved through summary replay and AGU rewriting. They are retained
> below as the pre-fix baseline, but are not valid DSE evidence. Corrected
> results and the complete chain audit are in
> [tile_aware_multichip_lineage_audit_v4.md](tile_aware_multichip_lineage_audit_v4.md).
> In particular, the old long-context `TP4 x CP4 = 5.491 s` result is corrected
> to `1.918 s` with one port and `1.897 s` with four ports.

## Executive Summary

The previous `factorized-tp-cp-v2` model scaled a completed single-chip trace
with fractional TP and CP factors. That model captured replicated work and
communication, but it could still assign a fraction of one physical Matrix
tile to a rank. The effect was close to ideal linear speedup even after a
rank-local tensor dimension became smaller than `MLEN`.

`tile-aware-tp-cp-ep-v3` reconstructs each rank's local tensor shapes before
estimating work:

```text
global logical shape
-> balanced TP dimensions and zigzag CP slabs
-> compiler SequencePackingPlan and AttentionHeadPacking
-> shared affine FFN projection plan
-> MLEN/BLEN padding and K-chunk boundaries
-> rank-local opcode, traffic, and energy census
-> slowest-rank stage latency
```

The model does not modify the compiler, ISA, RTL, or transactional emulator.
It is an analytical distributed mapping over a real single-chip CostEmitter
trace. It is now the default Qwen3 DSE multi-chip model. The following are
retained only under `analytic_models.legacy` for controlled A/B:

```text
factorized-tp-cp-v2
ideal-linear-lower-bound-v1
```

## Why Fractional Scaling Was Optimistic

The v2 approximation effectively allowed:

```text
0.25 of an MLEN-wide projection
0.50 of a full-width tail BMM
fractional Vector and Scalar setup
```

The current hardware cannot execute these fractions. A nonempty local tensor
still occupies complete `MLEN` column tiles and `BLEN` row tiles. A K dimension
crossing the Matrix-SRAM chunk boundary also causes another writeout and
partial-sum sequence. This creates three saturation effects:

1. **TP tile floor.** Sharding a dimension already below `MLEN` does not reduce
   the number of physical Matrix tiles.
2. **CP row padding.** Each nonempty local slab is padded by the same compiler
   layout rules as a single-chip sequence.
3. **Tail BMM floor.** A partial Q or K block still executes a full Matrix BMM
   because active-row BMM is not available in the current ISA.

The v3 result therefore need not scale monotonically with TP, CP, or chip
count. This is intentional.

## Semantic Kernel Census

CostTrace schema v7 contains a compressed `ParallelKernelCensus`. Every
non-HBM dynamic instruction belongs to one semantic kernel:

```text
token_replicated_hidden
token_tensor_sharded
column_parallel_projection
row_parallel_projection
attention_head_pair_sharded
expert_partitioned
replicated_setup
```

The compiler records the stage, kernel, opcode, logical shape, hardware tile
shape, multiplicity, and TP/CP/EP semantics. Census totals are reconciled after
AGU and zero-overhead-loop rewriting, so the census exactly covers the final
CostTrace opcode histogram. Unclassified work is a hard error.

Important consequences:

- RMSNorm, residual, and router post-processing scale with local tokens, not
  with `1/TP`.
- QKV and FFN up/gate use local column-parallel output shapes.
- O projection and FFN down use local row-parallel K shapes and retain the TP
  collective.
- attention uses local Q slabs, their global causal positions, and local GQA
  heads;
- setup that is genuinely replicated remains replicated.

## Rank-Local Reconstruction

### Tensor Parallelism

Logical dimensions use a balanced contiguous partition:

```text
local_dim(rank) =
    floor(global_dim / TP)
  + (rank < global_dim % TP)
```

The local dimensions are then passed through the same layout rules used by the
compiler. `SequencePackingPlan` and `AttentionHeadPacking` are imported
directly from the compiler. FFN projection Matrix counts use the shared
`FfnProjectionPlan`, including:

```text
MLEN output blocks
MLEN/BLEN output tiles per block
BLEN activation columns
K tiles per Matrix-SRAM chunk
writeout per K chunk
```

Utilization is reported only as a diagnostic. It is not multiplied into the
latency after the fact.

### Context Parallelism

CP retains the two-chunk zigzag mapping. Rank `r` owns chunks:

```text
r
2*CP - 1 - r
```

Token counts and global causal Q/K pairs are conserved exactly. Each chunk
keeps its global token start, so the two noncontiguous chunks of one sequence
are not treated as independent causal sequences. Slabs with equal shape may
reuse compiler templates, but their causal visibility remains distinct.

This design follows the load-balancing motivation of
[Striped Attention](https://arxiv.org/abs/2311.09431), while preserving
PLENA's full-tile tail behavior.

### Expert Parallelism

Dense Qwen3 fixes `EP=1`. Fixed-balanced Qwen3-MoE allows:

```text
N = TP * CP
EP divides CP
EP divides num_experts
```

EP reuses CP ranks rather than adding chips. Experts are assigned in balanced
contiguous ID ranges within each EP group. Expert FFNs are tensor parallel with
the same TP degree. Route buckets are padded independently to `BLEN`, and the
most heavily loaded expert rank controls expert-stage latency.

The model includes dependency-bound communication:

```text
router
-> TP router-logit reduction
-> EP dispatch all-to-all
-> expert compute
-> TP expert-output reduction
-> EP return all-to-all
-> combine
```

This separation follows the concepts in the
[Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
and [Megatron MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html).
It is a PLENA analytical mapping, not a Megatron implementation.

## Memory Model

Each rank gets a stage/opcode/precision-role traffic census based on its local
tile counts:

```text
weight
activation
matrix_kv
vector_kv
integer
```

Projection and expert weight traffic follows TP/EP ownership. CP ranks own
their local K/V and exchange remote K/V through the CP ring. The partial
resident K/V policy is applied to the local sequence.

The V4 bandwidth floor is recomputed from rank-local physical traffic and the
fixed per-chip share of aggregate HBM bandwidth. V4 startup/tail/drain
residuals are reweighted at stage-and-opcode granularity using the local
request census.

Claim boundary:

```text
N=1:
    exact CostEmitter and V4 identity

N>1:
    rank-local tile/role/opcode sufficient-statistics reconstruction
    not a distributed 64-B request-manifest replay
    not online multi-controller Ramulator co-simulation
```

The V4 coefficients and single-chip request semantics are unchanged.

## Communication, Area, and Energy

TP uses ring all-reduce after the attention output projection and FFN down
projection. CP uses a K/V ring. MoE adds router/expert TP reductions and two EP
all-to-all phases.

The link assumption remains deliberately optimistic:

```text
900 GB/s bidirectional per logical NVLink port
450 GB/s one-way per port
1, 2, or 4 ports
100% peak bandwidth
2.5 us nominal startup
```

The model reports full-overlap, nominal, and no-overlap bounds. The nominal
stage model is:

```text
Dense/FFN:
    max(local compute, local HBM) + required TP collective

Attention:
    max(local compute, local HBM, CP K/V ring)
  + TP output collective

MoE:
    router roofline
  + TP router reduction
  + EP dispatch
  + expert roofline
  + TP expert reduction
  + EP return
  + combine
```

Endpoint area remains `24.7 mm2/port/chip` nominal. Power aggregates actual
rank work, not slowest-rank power multiplied by `N`. Dynamic compute, SRAM,
controller, HBM, and interconnect energy are summed over ranks. Leakage uses
all physical chips and the global makespan.

## Controlled Results (Superseded Pre-Lineage-Fix Baseline)

The following tables document the original v3 rollout and must not be used as
current results. The corrected N16 long-context optimum is `TP2 x CP8` with
four ports at `1.839 s`; see the correction report linked above.

The dense A/B uses current Qwen3-32B CostEmitter traces:

```text
Short: seq=482, batch=16, MLEN=512, BLEN=64
Long:  seq=32768, batch=1, MLEN=2048, BLEN=1024
```

### Short Context

| Chips | Fastest v3 decomposition | v3 latency |
|---:|---|---:|
| 1 | TP1 x CP1, 1 port | 20.320 s |
| 2 | TP1 x CP2, 1 port | 10.160 s |
| 4 | TP1 x CP4, 1 port | 5.081 s |
| 8 | TP1 x CP8, 1 port | 2.541 s |
| 16 | TP4 x CP4, 4 ports | 1.401 s |

The previous headline `N16, TP4 x CP4, 1 port` changes from:

```text
v2 fractional: 1.283 s
v3 tile-aware: 1.405 s
delta:          +9.5%
communication:   9.17 ms
```

Most of the correction is tile work, not link latency. The same point has
121.7M cycles of analytical padding overhead, attention Matrix utilization
93.75%, and FFN Matrix utilization 90.14%.

### Long Context

| Chips | Fastest v3 decomposition | v3 latency |
|---:|---|---:|
| 1 | TP1 x CP1, 1 port | 29.066 s |
| 2 | TP1 x CP2, 1 port | 14.533 s |
| 4 | TP1 x CP4, 1 port | 7.267 s |
| 8 | TP1 x CP8, 1 port | 3.633 s |
| 16 | TP1 x CP16, 1 port | 2.413 s |

For `N16, TP4 x CP4, 1 port`:

```text
v2 fractional: 1.846 s
v3 tile-aware: 5.491 s
ratio:            2.98x
communication:   31.06 ms
```

At `MLEN=2048`, TP creates local hidden/head dimensions that still consume
full tiles. CP-only is therefore faster for this long-context point. This is
the expected saturation that v2 could not represent.

### Qwen3-235B-A22B MoE

The validation trace uses:

```text
seq=482, batch=16
MLEN=512, BLEN=64
128 experts, top_k=8
fixed-balanced routing
94 layers
```

Trace construction took 22.46 s cold and produced:

```text
61,696 routes per layer
128 active experts
0 materialized route objects
297 ParallelKernelCensus entries
100% census coverage
```

The full-decoder single-chip roofline is 10.385 s. Across 93 legal
TP x CP x EP x port combinations, the fastest analyzed point is:

```text
N16, TP4 x CP4, EP1, 4 ports
v3 latency = 1.251 s
v2 latency = 0.829 s
```

EP is not forced to improve latency. At this shape, all-to-all cost and
BLEN-padded expert buckets offset weight sharding; the best EP2 result is
1.256 s and higher EP degrees are slower. This is a model result for
fixed-balanced routing, not evidence about input-dependent production routing.

## Validation Evidence

Completed checks:

- `N=1` reproduces single-chip opcode work, V4 traffic/latency, roofline, and
  power inputs exactly.
- compiler census coverage is 100% for every tested trace;
- shared compiler planners are used for sequence rows, head storage, and FFN
  Matrix/K-chunk counts;
- zigzag token and causal-pair counts are conserved;
- dimensions below `MLEN` retain a one-tile floor;
- `seq=4097` retains full-width tail work and reports
  `active_row_bmm_unavailable`;
- fixed-balanced MoE route counts, contiguous expert ownership, bucket
  padding, and `CP/EP` weight replication are conserved;
- pure MoE communication contains router/expert TP reductions and no phantom
  dense FFN collective;
- EP dispatch/return bytes appear in both total and per-stage communication;
- multi-chip power uses all-rank action and traffic sums;
- explicit v2 mode remains available.

Regression results:

```text
123 combined DSE/multi-chip/power/compiler-census tests passed
53 HBM V4/RTL timing parity tests passed
64 COMPLETE concurrent DSE smoke: see run summary in the results JSON
```

The smoke's HBM-capacity prunes are valid constraints, not malformed TP/CP/SRAM
samples.

## Outputs

```text
tile_aware_multichip_model_v3.md
tile_aware_multichip_v2_v3_ablation.csv
tile_aware_multichip_v3_results.json
tile_aware_multichip_v3_short_results.json
tile_aware_multichip_v3_long_results.json
tile_aware_multichip_v3_moe_ablation.csv
tile_aware_multichip_v3_moe_results.json
```

## Remaining Limitations

- No distributed compiler or multi-chip transactional execution is generated.
- Multi-rank HBM is a tile/opcode sufficient-statistics reconstruction, not a
  distributed request-manifest replay.
- Partial K/V residency uses the current analytical local-cache overlay; it
  does not simulate a shared coherent cache.
- CP assumes direct logical rings and 100% peak link bandwidth.
- Active-row BMM is unavailable, so tail tiles retain full Matrix work.
- Endpoint static power, switches, package routing, link contention, and
  synchronization jitter remain excluded.
- MoE multi-chip supports fixed-balanced routing only.

These limitations make v3 a substantially more physical DSE model than v2,
but not a cycle-exact distributed implementation.
