# Tile-Aware Multi-Chip Lineage Audit v4

## Scope

This audit was triggered by the implausible long-context result reported by
the first `tile-aware-tp-cp-ep-v3` implementation:

```text
Qwen3-32B, seq=32768, batch=1
MLEN=VLEN=2048, BLEN=1024
N16, TP4 x CP4, one NVLink port

old v3 latency = 5.491 s
```

The audit follows one final lowered instruction through every formal DSE
consumer:

```text
final transformed schedule
-> ParallelKernelCensus
-> rank-local tile reconstruction
-> ideal-II1 latency
-> EnergyAction and ClockWork
-> rank-local DMA traffic
-> HBM V4 analytical partition
-> aggregate system energy
```

## Root Cause

The original census was captured before summary-template replay and AGU loop
rewriting. Instructions materialized or transformed later had no semantic
kernel owner. The frontend then assigned their residual stage/opcode counts to
a broad fallback kernel.

This was particularly destructive for long-context packed softmax. In the
faulty trace, 89.76% of attention work was classified as
`token_replicated_hidden`. Representative fallback counts included:

```text
S_LD_FP       2.137B fallback vs 10.49M exact Q/K-state work
S_ST_FP       2.135B fallback vs 12.58M exact Q/K-state work
V_MUL_VF      2.267B fallback
V_RED_*       1.132B per reduction family
S_EXP_FP      1.002B fallback
```

TP therefore failed to shard work that belonged to attention-pair/head
kernels. The same fallback could also undercount another decomposition, so the
error was not a conservative bound.

## Corrections

### Final-Schedule Lineage

Every final schedule leaf now carries a `ParallelKernelTag`. Tags survive:

- affine summary capture and replay;
- hardware-loop compression;
- AGU-v1 rewriting;
- deferred template replay;
- dense FFN, packed-GQA, static MoE, and fixed-balanced MoE lowering.

`ParallelKernelCensus` is rebuilt from the final transformed schedule. Any
unclassified instruction inside a `layer/*` stage is now a hard error. Global
setup may remain explicitly unclassified because it is replicated and is not
silently assigned to a layer kernel.

### Energy and Clock Lineage

`EnergyAction` now retains the same stable kernel-lineage identity. Coverage is
verified exactly for every:

```text
stage x lineage x opcode x component x hardware action family
```

This is stricter than opcode-only coverage. For example, `M_MM` must account
for both array-compute and cross-K-reduction actions, as well as its Matrix
SRAM reads and write.

Tile-aware power uses per-rank kernel/opcode scales. It no longer applies a
stage-average scale to all Vector, Scalar, Matrix, SRAM, and control work.
Ideal clock caps are evaluated separately on every rank before energies are
summed:

```text
correct: sum_r min(rank_work_r, area x makespan)
invalid: N x min(average_work, area x makespan)
```

### DMA Power Lineage

HBM-controller and DMA-induced SRAM actions now use the exact:

```text
stage x HBM opcode x precision role
```

traffic scale. Weight, activation, Matrix-KV, and Vector-KV transfers in the
same stage no longer share a stage-average power scale.

### Fail-Closed Semantics

Formal tile-aware evaluation now requires:

```text
CostTrace schema                     = 7
ParallelKernelCensus schema          = final schedule lineage v2
EnergyAction lineage coverage        = 100%
compute timing                       = ideal-ii1
```

Hazard-aware timing is not additive and therefore cannot be repartitioned by
this analytical model. Old studies cannot resume because the energy objective
schema is now:

```text
latency_area_energy_accuracy_tile_aware_tp_cp_ep_v5_rank_power_lineage
```

## Corrected Dense Results

Both scenarios use fresh CostEmitter traces after all current compiler
lowering, AGU-v1, affine FFN, compact-statistics, selector, overwrite, and
partial-resident SRAM changes.

### Short Context

```text
seq=482, batch=16, MLEN=VLEN=512, BLEN=64
```

| Chips | Fastest corrected decomposition | Latency |
|---:|---|---:|
| 1 | TP1 x CP1, 1 port | 20.320 s |
| 2 | TP1 x CP2, 1 port | 10.160 s |
| 4 | TP1 x CP4, 1 port | 5.080 s |
| 8 | TP1 x CP8, 1 port | 2.541 s |
| 16 | TP4 x CP4, 4 ports | 1.349 s |

The previous short-context headline was 1.401 s. The corrected result is 3.7%
lower; the error was modest because short attention had far fewer deferred
softmax repetitions.

### Long Context

```text
seq=32768, batch=1, MLEN=VLEN=2048, BLEN=1024
```

| Chips | Fastest corrected decomposition | Latency |
|---:|---|---:|
| 1 | TP1 x CP1, 1 port | 29.066 s |
| 2 | TP1 x CP2, 1 port | 14.533 s |
| 4 | TP1 x CP4, 1 port | 7.267 s |
| 8 | TP1 x CP8, 1 port | 3.633 s |
| 16 | TP2 x CP8, 4 ports | 1.839 s |

The two decompositions that exposed the bug now behave as follows:

| N16 decomposition | Old v3 | Corrected, 1 port | Corrected, 4 ports |
|---|---:|---:|---:|
| TP1 x CP16 | 2.413 s | 3.604 s | 3.604 s |
| TP4 x CP4 | 5.491 s | 1.918 s | 1.897 s |

For corrected `TP4 x CP4`, attention compute is 1.761 s, FFN compute is
0.127 s, and the attention HBM estimate is 0.776 s. The old 5.491 s result was
therefore a lineage bug, not a genuine long-context tile penalty.

## Why Four Ports Appear

Four ports do not create a large compute improvement. They only reduce the
analytical communication term. At the corrected long-context N16 optimum:

```text
TP2 x CP8, 1 port: 1.8454 s
TP2 x CP8, 2 ports: 1.8409 s
TP2 x CP8, 4 ports: 1.8387 s
```

The total one-to-four-port gain is 6.71 ms, or 0.36%. Four-port points appear
in latency-oriented selectors because the objective treats even a small
improvement as real. They pay `24.7 mm2 x 3` additional endpoint area per chip
relative to one port, so one-port alternatives can remain Pareto-optimal in
area and energy.

For `TP1 x CP16`, the CP ring remains below the compute/HBM roofline and port
count does not change formal latency. For TP2/TP4, TP collectives are
dependency-bound and lie outside the stage `max`, so port bandwidth has a
small visible effect.

## HBM V4 Claim Boundary

The audit found no coefficient or single-chip V4 regression, but it corrected
an overstatement in the distributed model:

```text
N=1: exact CostEmitter DMA manifest and V4 identity
N>1: rank-local role/opcode physical-traffic rescaling of V4 floor/residual
```

N>1 does **not** regenerate and replay a complete distributed 64-B request
manifest. Reports now state:

```text
v4_local_geometry_reconstruction = false
v4_rank_latency_exact            = false
```

This remains an explicit analytical-model limitation, not a hidden zero-cost
fallback. A distributed manifest replay or online multi-controller Ramulator
model is future work.

## Validation Evidence

Completed checks include:

- detailed and affine-summary CostEmitter parity;
- final schedule/census opcode equality;
- exact structural EnergyAction and SRAM-family coverage;
- layer DMA lineage coverage with global setup explicitly separated;
- dense and fixed-balanced/static MoE compiler regression;
- per-kernel power scaling counterexample;
- per-opcode/role DMA-power scaling counterexample;
- per-rank clock-cap aggregation counterexample;
- single-chip identity and multi-chip rank-traffic conservation;
- fresh short/long decomposition enumeration.

The final automated regression record is:

```text
CostFrontend current-process regression                 27 passed
Dense/MoE/CostEmitter/multi-chip/power integration     115 passed
DSE schema/objective plus fresh Qwen trace              30 passed
Formal DSE smoke                                         3 COMPLETE
Formal DSE smoke PRUNED/FAIL                              0 / 0
```

The regression suite includes explicit fail-closed counterexamples for:

- a final `layer/*` instruction with no kernel lineage;
- a layer DMA occurrence with no kernel lineage;
- an `EnergyAction` missing one structural Matrix/SRAM family;
- two kernels sharing one stage/opcode but requiring different rank scales;
- weight and activation DMA sharing one stage but requiring different scales;
- aggregate clock capping performed before, rather than after, rank summation;
- non-additive hazard-aware timing passed to the tile-aware model.

The formal DSE smoke used the complete compiler, tile-aware TP/CP
reconstruction, per-rank power aggregation, area model, and four-objective
return path. Its compiler cache reported:

```text
energy_action_lineage_schema =
    energy_action_kernel_lineage_v3_structural_families
energy_action_lineage_coverage = 1.0
multi_chip_onchip_aggregation =
    sum_rank_energy_after_per_rank_clock_cap_v2
```

The controlled source hashes and corrected numeric records are stored in:

```text
tile_aware_multichip_lineage_audit_v4_results.json
tile_aware_multichip_lineage_audit_v4_ablation.csv
```

## Remaining Limitations

- Multi-chip execution remains an analytical mapping, not distributed ISA.
- N>1 HBM V4 is not distributed request-manifest replay.
- NVLink uses 100% peak bandwidth and fixed startup.
- Active-row BMM is unavailable; nonempty tails still use full Matrix work.
- Endpoint static power, package effects, synchronization jitter, SRAM
  leakage, and link contention remain excluded.
