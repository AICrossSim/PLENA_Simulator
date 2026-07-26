# Searchable TP x CP Multi-Chip Analytical Model v2

> **Superseded analytical baseline, audited 2026-07-26.** This fractional
> model is no longer selectable from the formal DSE CLI and now lives under
> `analytic_models.legacy`. It allowed fractional local tiles and must not be
> used for current performance claims. The implementation rationale remains
> useful; current results use the final-lineage tile-aware model documented in
> [`tile_aware_multichip_lineage_audit_v4.md`](tile_aware_multichip_lineage_audit_v4.md).

> **Status:** retained for historical A/B only. The formal DSE default is now
> `tile-aware-tp-cp-ep-v3`; see
> [`tile_aware_multichip_model_v3.md`](tile_aware_multichip_model_v3.md).
> v2 scales completed work fractionally and can therefore overstate speedup
> when a rank-local dimension crosses an MLEN/BLEN tile boundary.

## Executive Summary

The previous multi-chip model divided nearly all work by the chip count. That
model was useful only as an ideal linear-scaling lower bound: it had no
replicated work, no causal-attention imbalance, no tensor-parallel collective,
and no context-parallel K/V exchange. It therefore had a built-in preference
for the largest available chip count.

This revision keeps the compiler single-chip and introduces a searchable,
stage-level analytical decomposition:

```text
chip_count = TP * CP
```

For every chip count, DSE now searches every natural head-sharding TP degree
and derives CP. It also searches one, two, or four logical NVLink endpoints.
The old model remains available as:

```text
ideal-linear-lower-bound-v1
```

The new formal model is:

```text
factorized-tp-cp-v2
```

It accounts for:

- different TP/CP scaling of projection, attention, setup, and memory work;
- exact two-chunk zigzag causal load balance;
- weight replication across CP ranks;
- TP ring all-reduce after attention output and FFN down projection;
- CP point-to-point K/V ring exchange;
- local-sequence Matrix SRAM residency;
- fixed aggregate A100-aligned HBM capacity and bandwidth;
- per-port endpoint area and interconnect energy.

This is an analytical distributed execution model. It is not a distributed
compiler, multi-chip RTL simulation, or cycle-exact collective implementation.

## Why the Previous Model Preferred Maximum Chip Count

The old optimistic `tp-sp` path approximately applied:

```text
compute_N = compute_1 / N
traffic_N = traffic_1 / N
area_N    = N * per_chip_area
```

Its latency model contained almost no term that grew with `N`. Consequently,
an area-feasible `N=16` design was expected to dominate an otherwise similar
`N=8` design. This was a property of the equation, not evidence that a real
16-chip system would scale perfectly.

The corrected model introduces three effects that the old model omitted:

1. CP does not shard weights. Each CP group needs its own TP-sharded model
   weights, so aggregate weight traffic grows with CP.
2. TP requires two forward collectives per decoder layer.
3. CP attention requires K/V exchange and has a causal load-balance limit.

The old result remains useful as a lower bound, but is no longer the formal DSE
latency.

## Legal Parallel Domain

TP is constrained by the physical GQA head decomposition:

```text
TP <= num_kv_heads
num_attention_heads % TP == 0
num_kv_heads % TP == 0
chip_count % TP == 0
CP = chip_count / TP
```

For Qwen3-32B (`64` Q heads, `8` K/V heads), the implemented domain is:

| Chips | Legal TP x CP |
|---:|---|
| 1 | 1 x 1 |
| 2 | 1 x 2, 2 x 1 |
| 4 | 1 x 4, 2 x 2, 4 x 1 |
| 8 | 1 x 8, 2 x 4, 4 x 2, 8 x 1 |
| 16 | 1 x 16, 2 x 8, 4 x 4, 8 x 2 |

Pipeline parallelism remains fixed at one.

## Parallel Work Census

CostEmitter now emits per-stage, per-opcode compute work. The multi-chip model
classifies every cycle into one of four axes:

```text
token_hidden_sharded
attention_pair_head_sharded
tensor_only
replicated_setup
```

The per-rank scaling is:

```text
token_hidden_sharded:
    max_token_fraction / TP

attention_pair_head_sharded:
    max_causal_pair_fraction / TP

tensor_only:
    1 / TP

replicated_setup:
    1
```

An unknown stage/opcode raises an error. It is not silently assigned `1/N`.
The implementation also verifies that classified cycles equal CostEmitter
stage cycles. The measured census coverage in all validation runs is `100%`.

The classification is count-exact but semantically analytical. It assumes the
single-chip opcode work can be partitioned without changing the lowering.

## Zigzag Context Partition

The sequence is split into `2*CP` contiguous chunks. Rank `r` owns:

```text
chunk r
chunk (2*CP - 1 - r)
```

For each rank, the implementation exactly sums:

```text
local tokens
sum(query_position + 1)
```

The latter is the number of causal Q/K pairs owned by that rank. Both token and
causal-pair totals are checked for conservation, and the slowest rank supplies
the stage scaling factor.

This two-chunk assignment pairs an early, light causal chunk with a late, heavy
chunk. It follows the load-balancing motivation of Striped Attention while
remaining simple enough for an analytical model. It is not a claim that the
current compiler emits that distributed schedule.

## HBM and Matrix SRAM

HBM traffic is partitioned by the role already attached to each physical DMA:

```text
weight       -> 1 / TP
activation   -> max_token_fraction / TP
matrix_kv    -> max_token_fraction / TP
vector_kv    -> max_token_fraction / TP
integer      -> max_token_fraction
```

This fixes the earlier overlay bug that recognized only a generic `kv` role.
Both actual `matrix_kv` and `vector_kv` traffic now participate.

As a result:

```text
aggregate weight traffic = CP * single-chip weight traffic
```

Matrix SRAM residency is replanned from the slowest rank's local token extent.
The existing partial-resident K/V planner supplies exact local cache
occurrences under that sequence partition.

Aggregate memory resources remain fixed for the A100 comparison:

```text
capacity  = 80 GB decimal
bandwidth = 2039 GB/s

per-chip capacity  = 80 GB / N
per-chip bandwidth = 2039 GB/s / N
```

The model recomputes the bandwidth floor from the new per-rank physical bytes.
The calibrated V4 non-bandwidth residual is scaled by physical request count,
not by bytes. This is stronger than scaling complete V4 latency by a byte
ratio, but it is still an analytical V4 transformation rather than an online,
cross-rank Ramulator replay.

## Communication Model

Each logical endpoint supplies:

```text
900 GB/s bidirectional
450 GB/s one-way
```

The searched one-way bandwidth is therefore `450/900/1800 GB/s` for
`1/2/4` ports. No utilization discount is applied:

```text
bandwidth_efficiency = 1.0
```

Four ports correspond to the current NVIDIA headline total of `3.6 TB/s`
bidirectional bandwidth per GPU. The model's "port" is a 900 GB/s logical
endpoint, not one physical serdes lane.

TP uses the standard ring all-reduce model:

```text
T_tp =
    2*(TP-1)*alpha
  + 2*(TP-1)/TP * local_activation_bytes / B_oneway
```

It is charged twice per layer:

```text
after attention output projection
after FFN down projection
```

CP uses a point-to-point K/V ring:

```text
local_kv_bytes =
    2 * local_tokens * batch
      * (num_kv_heads / TP) * head_dim * kv_bits / 8

T_cp =
    (CP-1)*alpha
  + (CP-1)*local_kv_bytes / B_oneway
```

Nominal startup is `2.5 us`; `1 us` and `4 us` are reported as sensitivities.

The formal stage equation is:

```text
attention =
    max(local_compute, local_HBM_V4, CP_KV_ring)
  + TP_collective

FFN =
    max(local_compute, local_HBM_V4)
  + TP_collective
```

The report also records:

```text
full_overlap_lower_bound
nominal_stage_model
no_overlap_upper_bound
```

Peak link bandwidth and CP overlap are optimistic architectural assumptions.

## Endpoint Area

The nominal endpoint proxy is:

```text
24.7 mm2 per logical port
```

with:

```text
15.7 mm2 optimistic
38.4 mm2 conservative
```

The ISSCC NVLink-C2C result reports a 5 nm, 900 GB/s bidirectional interface
and approximately `552 Gbit/s/mm2` density. A raw bandwidth/density division is
about `13.0 mm2`. The three model values add progressively stronger process
translation and integration margins for the project's 7 nm comparison. The
nominal value is close to a simple `(7/5)^2` area scaling.

This is an engineering proxy. It is not a measured NVLink 6 macro, and endpoint
static/leakage power is not available.

Formal aggregate area is:

```text
total_silicon_area =
    chip_count * (core_area + ports*24.7 mm2)
```

and remains constrained to `908.6 mm2`.

## Energy

The multi-chip energy estimator consumes the same work census:

- compute, SRAM, controller actions, and ClockWork follow TP/CP axis scaling;
- logic leakage uses per-chip area and the multi-chip makespan;
- external HBM dynamic energy uses aggregate physical traffic;
- 80 GB HBM background is counted once for the aggregate system;
- interconnect dynamic energy uses `8 pJ/bit` nominal;
- `1.3` and `70.9 pJ/bit` remain explicit non-statistical sensitivities.

Power remains a DSE shadow model with the existing ideal hierarchical
clock-gating assumption.

## DSE Integration

The formal defaults are now:

```text
--multi-chip-model factorized-tp-cp-v2
--tp-degrees auto
--nvlink-port-counts 1,2,4
--nvlink-bandwidth-semantics peak
--nvlink-startup-us 2.5
```

TP is conditionally sampled after chip count and CP is derived. Port count is
also searched. The study schema and hardware fingerprint include:

```text
multi-chip model
TP domain
NVLink port domain
bandwidth semantics
startup latency
```

This prevents incompatible old studies from being resumed.

Trial output includes the TP/CP split, causal fractions, communication
bytes/latency, endpoint area, weight replication, overlap bounds, and census
coverage. Four objectives remain unchanged:

```text
minimize latency
minimize aggregate silicon area
minimize system energy
maximize accuracy
```

## Validation Evidence

### Unit and Integration Tests

The focused suite reports:

```text
92 passed
```

It covers:

- legal TP enumeration and invalid TP rejection;
- exact zigzag token and causal-pair conservation;
- `N=1, TP=1, CP=1` parity;
- non-1 GHz cycle conversion;
- CP weight replication;
- local-token TP collective size;
- 450/900/1800 GB/s one-way bandwidth;
- startup sensitivity ordering;
- lower/nominal/upper latency ordering;
- endpoint area scaling;
- multi-chip power and HBM traffic conservation;
- DSE conditional-domain and schema behavior.

During the 64-point smoke, analytical fractional per-rank byte counts exposed a
one-to-two-byte audit failure caused by independent integer rounding. The
external-memory and multi-chip power audits now preserve floating-point
analytical shares and use strict `fsum/isclose` conservation. Physical
single-chip DMA traffic remains integer-valued.

### DSE Smoke

The final cached smoke used 32 requested workers:

```text
64 COMPLETE
11 PRUNED
0 FAIL
75 attempts
42 canonical TP x CP x port hardware combinations
```

The 11 prunes were real capacity violations: per-chip weight plus K/V capacity
exceeded the fixed aggregate 80 GB allocation. They were not illegal TP/CP or
duplicate-encoding prunes.

Resource evidence:

```text
maximum dynamic concurrency       16
peak active process-tree RSS      2.05 GiB
peak worker RSS                   0.265 GiB
minimum MemAvailable              101.9 GiB
parent terminations               0
```

## Controlled Short-Context Ablation

The checked-in ablation uses one current CostEmitter report:

```text
Qwen3-32B
seq_len=482
batch=16
MLEN=VLEN=512
BLEN=64
Matrix SRAM=2 tiles, streaming
64 decoder layers
```

It evaluates all 42 legal `TP x CP x ports` combinations.

| Chips | Fastest split | Ports | New latency | Ideal 1/N lower bound | Gap |
|---:|---:|---:|---:|---:|---:|
| 1 | TP1 x CP1 | 1 | 20.3201 s | 20.3201 s | 0.000% |
| 2 | TP1 x CP2 | 1 | 10.1611 s | 10.1601 s | 0.010% |
| 4 | TP2 x CP2 | 4 | 5.0833 s | 5.0801 s | 0.064% |
| 8 | TP4 x CP2 | 4 | 2.5454 s | 2.5401 s | 0.209% |
| 16 | TP8 x CP2 | 4 | 1.2784 s | 1.2701 s | 0.652% |

For the fastest 16-chip point:

```text
TP collective latency = 8.165 ms
CP K/V ring latency    = 0.196 ms
no-overlap upper bound = 1.4686 s
```

The new nominal result remains close to ideal `1/N` in this controlled point
because:

- the trace is strongly compute-bound;
- CP communication is assumed to overlap perfectly;
- link bandwidth is the requested 100% architectural peak;
- no switch contention, endpoint queueing, or distributed-compiler overhead is
  modeled;
- replicated setup work is small relative to projection and FFN work.

Therefore the model can still select the largest area-feasible chip count. The
important correction is that this is now an explicit outcome of optimistic
communication assumptions and workload composition, rather than a consequence
of dividing every term by `N`.

CP's cost is visible in aggregate HBM traffic:

```text
CP1 aggregate physical traffic: 113.0 GB
CP2 aggregate physical traffic: 218.8 GB
```

The increase is principally replicated weight traffic.

## Legacy 2048-Point Post-Hoc Limitation

The existing short and long 2048-point artifacts predate the required
per-stage opcode census:

```text
stage_compute_opcode_work_cycles
```

Their compact reports contain only aggregate opcode counts. An exact TP/CP
classification cannot be reconstructed from those fields without guessing
which stage emitted each opcode. The post-hoc tool therefore rejects these old
reports instead of silently applying `1/N`.

New-schema compiler reports can be re-scored across TP, CP, and port count
without recompiling. A formal 2048-point rerun is still required to replace the
old Pareto study under the new semantics.

## Claim Boundary

Safe claims:

- all formal DSE multi-chip points use legal Q/KV head sharding;
- the single-chip case is exactly preserved;
- work-census and traffic accounting are complete and audited;
- the model exposes weight replication, causal imbalance, communication, and
  endpoint area that the prior lower bound omitted;
- no sustained-bandwidth discount is hidden in the result.

Unsafe claims:

- cycle-exact distributed execution;
- measured NVLink 6 latency, bandwidth, or endpoint area;
- guaranteed full communication overlap;
- a compiler-validated TP/CP schedule;
- routed, package-level, or switch-level area/power accuracy.

## Evidence Artifacts

- `factorized_multichip_model_v2_results.json`
- `factorized_multichip_model_v2_ablation.csv`
- `analytic_models/performance/test_multi_chip_model.py`
- `analytic_models/performance/benchmark_factorized_multichip_v2.py`

## References

- [NVIDIA NVLink specifications](https://www.nvidia.com/en-gb/data-center/nvlink/)
- [Megatron Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [Megatron-LM tensor parallelism](https://arxiv.org/abs/1909.08053)
- [Striped Attention](https://arxiv.org/abs/2311.09431)
- [ISSCC 2023 NVLink-C2C press material](https://www.isscc.org/s/ISSCC2023-PressKit.pdf)
- [NVIDIA Grace and NVLink-C2C technical description](https://developer.nvidia.com/blog/inside-nvidia-grace-cpu-nvidia-amps-up-superchip-engineering-for-hpc-and-ai/)
