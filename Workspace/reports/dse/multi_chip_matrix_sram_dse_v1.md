# Multi-Chip and Matrix-SRAM DSE v1

> **Historical lower-bound model, audited 2026-07-26.** The TP+SP/TP-only
> scaling in this report predates searchable TP x CP and rank-local tile
> reconstruction. Its single-chip SRAM observations remain useful, but N>1
> latency and energy numbers are not current evidence. Use
> [`tile_aware_multichip_lineage_audit_v4.md`](tile_aware_multichip_lineage_audit_v4.md)
> for corrected multi-chip results.

> **SRAM-search update (2026-07-25):** raw tile search remains for historical
> reproduction. The default now uses partial-resident K/V policies and ideal
> dual-port SRAM area. See
> `../compiler/partial_resident_kv_ideal_dual_port_dse.md`.
>
> **Compute-timing update (2026-07-24):** current runs use `ideal-ii1` by
> default. Historical values in this report that used rtl-v3 hazard-aware
> timing remain A/B evidence and should not be mixed into the new study.

## Scope

This extension adds two architectural search dimensions to the Qwen3-32B
prefill DSE:

```text
PLENA chip count    N = 1, 2, 4, 8
Matrix SRAM tiles     = 2, 4, 8, 16, 32, 64
```

Useful non-power-of-two Matrix SRAM saturation points are added automatically.
The default comparison fixes aggregate resources to one A100 reference:

```text
Area budget       = 1 x 826 x 1.10 mm2
HBM capacity      = 1 x 80 GB
HBM bandwidth     = 1 x 2039 GB/s
NVLink 6          = 3.6 TB/s bidirectional
One-way estimate  = 1.8 TB/s
```

## Matrix SRAM

`MATRIX_SRAM_SIZE = MATRIX_SRAM_TILES * MLEN` is passed directly to the native
compiler and CostEmitter. It therefore changes real projection K-chunking,
partial-result writeout, attention K/V residency, DMA occurrences, and the
ASAP7 SRAM macro area. Vector, scalar-FP, and scalar-INT SRAM capacities remain
at their minimum legal settings because their allocators do not yet expose a
credible capacity/performance policy.

The search adds a useful saturation point:

```text
projection = max(ceil(hidden / MLEN), ceil(intermediate / MLEN))
attention  = 2 * ceil(local_attention_seq_len / MLEN)
saturation = max(2, projection, attention)
```

Points above saturation are recorded as `capacity_dominated` and pruned before
CostEmitter evaluation.

## Multi-Chip Semantics

The compiler still emits one aggregate single-chip trace. Multi-chip behavior
is a labelled stage-level analytical post-process.

- `tp-sp`: all compute and physical DMA traffic are divided by `N`.
- `tp-only`: Matrix work and weight traffic are divided by `N`; Vector,
  Scalar, control, activation, and KV work are conservatively replicated.
- Both modes add two ring-collective activation transfers per decoder layer.
- Communication uses peak one-way NVLink bandwidth, with no startup, topology,
  congestion, or protocol efficiency loss.

The model reports both modes when requested. `tp-only` must never be faster
than optimistic `tp-sp`.

## R-Aware HBM V4

The default legacy bandwidth expression no longer prunes trials. It is retained
as `legacy_bandwidth_would_prune` and `required_feed_ratio`; strict reproduction
is available through `--legacy-bandwidth-policy strict`.

V4 latency is not divided by `N`. It is decomposed as:

```text
V4 latency = theoretical bandwidth floor + calibrated service residual
```

Physical 64-B traffic is partitioned by exact `stage x precision-role`
manifests. The floor is recomputed using `R * 2039 / N GB/s` per chip. Only the
service residual is scaled with the retained per-chip traffic. Consequently,
dividing both traffic and bandwidth by `N` does not create a false bandwidth
speedup.

The equivalent per-chip channel count is reported as `128 * R / N`. Counts at
8, 32, or 128 are calibration anchors; intermediate counts are explicitly
labelled as residual-scaled interpolation, and values outside this range carry
an extrapolation ratio.

## Area and KV Handoff

`area_new` remains a per-chip core proxy. The DSE objective uses:

```text
endpoint_area       = 10% * core_area
physical_chip_area  = core_area + endpoint_area
total_silicon_area  = N * physical_chip_area
```

P10/P50/P90 area bounds are aggregated in the same way. FP16 K/V handoff to a
decode chip is reported as a one-time shadow transfer over the same 1.8 TB/s
link and is not included in prefill latency.

## Validation

Focused unit and regression results:

```text
Multi-chip/Matrix-SRAM unit tests       5 passed
Existing DSE + HBM V4 regression       30 passed
N=1 end-to-end CostEmitter smoke       passed
N=2 end-to-end CostEmitter smoke       passed
Updated plot rendering                 passed
```

The `N=1, SRAM=2` real smoke reproduced the original CostEmitter roofline
exactly:

```text
original compiler roofline  7517.291016 ms
new DSE latency             7517.291016 ms
difference                     0.000000 ns
```

The same hardware with `N=2`, fixed aggregate HBM, produced:

```text
latency                         3762.862427 ms
per-chip compute scale                   0.5
V4 bandwidth floor, N=1          56.310656 ms
V4 bandwidth floor, N=2          56.310656 ms
V4 residual, N=1                 13.266702 ms
V4 residual, N=2                  6.633351 ms
NVLink communication, N=2         4.211780 ms
```

This confirms that the observed speedup comes from optimistic compute and
service-work partitioning, not from incorrectly increasing aggregate HBM
bandwidth.

For the same real CostEmitter report at `N=4`:

```text
TP+SP optimistic latency  1885.648 ms
TP-only latency           7223.077 ms
```

The endpoint-area check also behaved as intended:

```text
per-chip core area       847.000 mm2
per-chip endpoint         84.700 mm2
N=1 aggregate            931.700 mm2
N=2 aggregate           1863.400 mm2
```

### Matrix SRAM A/B smoke

A fixed single-chip Qwen3-32B run isolated the effect of Matrix SRAM capacity:

```text
seq=482, batch=16
MLEN=VLEN=2048, BLEN=1024
same precision, compiler schedule, HBM resources, and chip count
```

| Matrix SRAM | Logical capacity | K-split chunks (QKV/O/GU/D) | Latency | Physical area |
|---:|---:|---:|---:|---:|
| 2 tiles | 16.78 MB | 2 / 2 / 2 / 7 | 7517.291 ms | 931.700 mm2 |
| 13 tiles | 109.05 MB | 1 / 1 / 1 / 1 | 7252.611 ms | 988.936 mm2 |

The 13-tile saturation point reduced latency by `264.680 ms` (`3.52%`) while
increasing physical area by `57.236 mm2` (`6.14%`). Compute resource work fell
by `5.45%`. The material opcode-work changes were:

```text
M_MM                       unchanged
M_MM_WO          -141.401 million cycles
V_ADD_VV          -421.528 million cycles
remaining delta   control/address operations
```

HBM physical traffic, V4 floor/residual, attention residency, and K/V tile
loads were unchanged. The short-context benefit therefore comes specifically
from removing K-split partial writeout and vector accumulation, not from extra
MAC throughput or less off-chip traffic.

The ideal-gated power shadow predicted `1.66%` lower system energy but `1.93%`
higher average power because the larger SRAM shortens runtime by more than it
reduces total energy. This remains a power-model shadow, not measured power.

Both points exceed the `R=1` aggregate budget of `908.6 mm2`; this A/B isolates
the SRAM mechanism but is not evidence of an area-feasible improvement. A fair
frontier must exchange compute area for SRAM area under the aggregate budget.

## Limitations

- Multi-chip scheduling is not emitted as ISA and is not validated by the
  transactional emulator.
- NVLink latency is an optimistic peak-bandwidth lower bound.
- TP+SP assumes all work can be partitioned evenly.
- At this historical stage power was only a shadow and multi-chip power
  partitioning was not modeled. Current DSE uses per-rank tile-aware system
  energy; this sentence is retained only as a limitation of the old result.
- Matrix SRAM uses the existing local scratchpad, not a large 3D-stacked SRAM
  hierarchy.
- SRAM leakage remains unavailable in the ASAP7 macro data.
