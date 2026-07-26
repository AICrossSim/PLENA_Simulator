# Partial-Resident K/V Cache and Ideal Dual-Port SRAM DSE

## Scope

This change replaces the binary streaming/full-resident K/V decision with a
deterministic resident-prefix cache and makes SRAM policy, rather than raw tile
count, the default DSE variable. It also adds an explicitly ideal dual-port SRAM
area model.

The implementation does **not** add active-row BMM. Tail tiles still execute as
full `MLEN x MLEN` operations and retain:

```text
tail_isa_limitation = active_row_bmm_unavailable
```

## K/V Residency Semantics

The shared `KVResidencyPlan` is used by the compiler, CostEmitter, and DSE. For
`n = ceil(local_seq_len / MLEN)` K blocks and `C` resident prefix blocks:

```text
full resident tiles = 2*n
partial tiles       = 2*C + 2

MRAM:
  resident K[0:C]
  resident V[0:C]
  streaming K slot
  streaming V slot
```

The earliest causal K/V blocks are retained because they are reused by the
largest number of later Q blocks. The exact causal K+V load count is:

```text
loads = 2*C + 2*sum_q max(0, min(q+1,n)-C)
```

Resident blocks are prefetched once per batch/KV head. Streamed blocks are
loaded in their original K-block order. QK, PV, online-softmax, and packed-O
accumulation order are unchanged.

The compiler now receives `kv_residency_policy` directly. Physical capacity
alone cannot distinguish explicit `streaming` from opportunistic use of spare
tiles.

## DSE Policies

The default categorical variable is:

```text
streaming
projection-full
kv-25
kv-50
kv-75
kv-100
```

`projection-full` allocates enough Matrix SRAM for the largest projection K
dimension. Every non-streaming policy uses spare physical capacity as a K/V
prefix cache. Policies mapping to the same `(physical_tiles, resident_prefix)`
are pruned before CostEmitter evaluation.

For optimistic TP+SP multi-chip analysis, residency is planned from:

```text
local_seq_len = ceil(global_seq_len / chip_count)
```

The attention K/V occurrence count is replaced by the exact local-cache count.
The remaining stage-level compute partition is still the existing optimistic
analytical model. Fidelity is therefore reported as:

```text
single chip:
  exact_compiler_schedule_single_chip

multi-chip:
  exact_local_cache_occurrences_under_optimistic_tp_sp
```

## Ideal Dual-Port SRAM

The public area API remains backward compatible:

```text
replicated-single-port  # historical default
ideal-dual-port         # DSE default
```

In ideal-dual-port mode, logical ports and capacity are unchanged, but a
multi-port SRAM is represented by one macro area rather than replicated
single-port macros. No decoder, bitline, arbitration, routing, timing, or
leakage overhead is added. This is an architectural assumption, not an ASAP7
macro result.

Power remains access-based:

- Two same-cycle accesses consume two access energies.
- Removing macro replication does not change dynamic access energy for an
  identical trace.
- Reduced K/V loads and Matrix SRAM writes reduce energy through the real
  CostTrace.
- SRAM leakage remains unavailable.

## Validation Evidence

### Unit and integration tests

The implemented tests cover:

- Prefix/stream-slot address disjointness and bounds.
- Exact causal load formula and monotonic traffic reduction.
- Streaming and full-resident endpoints.
- Opportunistic use of projection capacity.
- Explicit streaming-policy propagation into real CostEmitter lowering.
- Exact partial-resident CostEmitter trace for three K blocks.
- Multi-chip local-cache occurrence replacement.
- Ideal dual-port area monotonicity and approximately 2x removal of replicated
  dual-port macro area.
- Unchanged SRAM dynamic energy for an unchanged access trace.

Completed regression groups:

```text
CostEmitter + residency planner        31 passed
PLENA compiler                         34 passed
Qwen3 DSE                              20 passed
Area + multi-chip + power              94 passed
```

Transactional numerical A/B:

```text
batch=2, seq=39, MLEN=VLEN=16, BLEN=4, K blocks=3
streaming:     resident prefix=0, physical HBM reads=18,944 B
partial cache: resident prefix=1, physical HBM reads=12,800 B

VRAM SHA256 for both runs:
4ffea147d8277da75b56d291d2f419ce0aa92e2135f850bcfec88538fd5bf447

existing correctness gate: PASS for both
allclose match rate: 100% for both
```

The MRAM dump is expected to differ because cached and streaming layouts use
different physical addresses. The architecturally visible VRAM output and
Scalar FP state are bitwise identical.

The required parallel smoke completed:

```text
12 trials / 4 workers
12 complete, 0 failed, 0 pruned
all latency values identical for the fixed configuration
all compiler records report resident_kv_blocks=0 for streaming
```

### Real CostEmitter/DSE anchor

Fixed configuration:

```text
Qwen3-32B, seq=482, batch=16
MLEN=VLEN=512, BLEN=64, chip_count=1
ideal-II1 + HBM V4 + ideal hierarchical clock gating
```

| Policy | Tiles | Resident K blocks | Latency | V4 memory work | Area | System energy |
|---|---:|---:|---:|---:|---:|---:|
| streaming | 2 | 0/1 | 20.757 s | 149.109 ms | 16.807 mm2 | 388.844 J |
| kv-50 | 50 | 1/1 | 12.134 s | 145.898 ms | 24.650 mm2 | 331.657 J |

The 41.5% latency reduction is **not** mainly a one-block K/V cache benefit.
At this short sequence, memory work changes by only 3.21 ms. Most of the
reduction comes from the same 50 physical tiles eliminating projection/FFN
K-split work. This distinction is preserved in the trial metadata.

### Latest-infrastructure DSE replay

After making RTL-v4 compact statistics, selector hoisting, and reduction
overwrite the DSE defaults, the same two SRAM policies were replayed through
the full four-objective DSE:

| Policy | Tiles | Resident K blocks | Compute/roofline latency | HBM read requests | Aggregate area | System energy |
|---|---:|---:|---:|---:|---:|---:|
| streaming | 2 | 0/1 | 20.418 s | 1,790,699,584 | 16.874 mm2 | 404.652 J |
| kv-50 | 50 | 1/1 | 11.795 s | 1,722,542,144 | 24.717 mm2 | 347.423 J |

This confirms that SRAM is connected to all relevant DSE paths:

- The policy changes the native compiler's Matrix SRAM depth and projection
  chunking.
- The resident-prefix plan changes actual K/V DMA occurrences consumed by
  HBM V4.
- `area_new` uses the selected depth with ideal dual-port macro semantics.
- The power model consumes the changed SRAM and HBM actions.

For this short-context point, `kv-50` reduces latency by 42.23% and energy by
14.14% while increasing aggregate area by 46.48%. As in the original anchor,
the latency gain is dominated by projection/FFN chunking rather than caching a
single attention K block.

The latest defaults also passed an `8 trials / 4 workers` fixed-policy smoke:
8 complete, 0 failed, and 0 pruned. Every trial reported the same SRAM policy,
RTL-v4 modes, latency, area, and energy, while using the new objective and
search schemas.

For these points, ideal dual-port SRAM removes 49.8-50.0% of the SRAM area
that the historical replicated-single-port model attributed to logical
dual-port memories:

| Policy | Ideal SRAM | Replicated SRAM | Saving |
|---|---:|---:|---:|
| streaming | 0.538 mm2 | 1.073 mm2 | 0.534 mm2 |
| kv-50 | 7.668 mm2 | 15.333 mm2 | 7.664 mm2 |

### Long-context planner sweep

The 360-point geometry sweep covers:

```text
seq = 4096, 4097, 8192, 32768, 65536
MLEN = 512, 1024, 2048
chip_count = 1, 2, 4, 8
all six policies
```

At `seq=65536, MLEN=512, N=1`:

| Policy | Tiles | Resident blocks | K/V tile-load reduction |
|---|---:|---:|---:|
| streaming | 2 | 0/128 | 0.0% |
| kv-25 | 66 | 32/128 | 43.2% |
| kv-50 | 130 | 64/128 | 74.0% |
| kv-75 | 194 | 96/128 | 92.4% |
| kv-100 | 256 | 128/128 | 98.4% |

The non-linear benefit is expected for causal attention: early resident blocks
avoid more reloads than late blocks. The sweep also retains the 4096/4097
increase in full-width K-block count; no active-row benefit is claimed.

Artifacts:

- `partial_resident_kv_policy_sweep.csv`
- `partial_resident_kv_policy_sweep.png`
- `Workspace/qwen3_32b_dense_analytic/runs/partial_kv_ideal_dp_smoke_3x1_v3`
- `Workspace/qwen3_32b_dense_analytic/runs/partial_kv_ideal_dp_smoke_12x4`

## Remaining Limitations

- Multi-chip compute partition is stage-level analytical scaling, not a
  multi-chip compiler or simulator.
- HBM V4 remains an occurrence-level post-hoc service model.
- Tail tiles remain full-width BMM operations.
- The ideal dual-port model excludes physical dual-port overhead and SRAM
  leakage.
- Planner-only traffic curves are not presented as latency or system-energy
  curves; those require CostEmitter/DSE evaluation.
