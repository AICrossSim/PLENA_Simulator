# Disaggregated Serving DSE Experiment Plan

## Status And Research Question

This document defines the experiment used to compare an aggregated A100
serving system against a disaggregated system in which PLENA performs prefill
and A100 GPUs perform decode.

The primary research question is:

> Under the same total A100-equivalent resource budget and the same workload,
> can PLENA prefill plus A100 decode reduce total system energy and improve
> SLO-constrained goodput per watt while retaining acceptable latency?

PLENA is not assumed to beat A100 in raw TTFT or end-to-end latency. Its main
expected advantage is energy efficiency:

```text
Primary metrics:
  system energy/request
  system energy/token
  SLO-constrained goodput/W

Secondary metrics:
  matched-batch E2E latency
  TTFT
  TPOT/TBT
```

The primary selector is the minimum-system-energy configuration satisfying:

```text
E2E latency <= 1.25 x aggregated-A100 E2E latency
```

The same selection is also reported at `1.00x` and `1.50x` latency limits.
Average power alone is not evidence of an energy advantage.

## Resource Budgets

The resource split is defined as:

```text
R_total = P + D

P = A100-equivalent resource budget assigned to PLENA prefill
D = number of A100 GPUs assigned to decode
```

The formal systems are:

| Model | Aggregated baseline | Disaggregated splits |
|---|---:|---|
| Qwen3-32B | 8 x A100 | P2:D6, P4:D4 |
| Qwen3-235B-A22B | 16 x A100 | P4:D12, P8:D8, P12:D4 |

Each comparison is iso-resource within a model. The 32B and 235B systems have
different deployment budgets and must not be compared as if they used the
same amount of silicon.

The PLENA constraints for a prefill budget `P` are:

```text
aggregate PLENA area          <= P x 826 x 1.10 mm2
aggregate PLENA HBM capacity  <= P x 80 GB
aggregate PLENA HBM bandwidth <= P x 2039 GB/s
decode HBM capacity            = D x 80 GB
```

The current DSE arguments must therefore be interpreted as:

```text
reference_a100_count = P
decode_chip_count    = D
```

`reference_a100_count` must not be set to `R_total` and then combined with an
additional `D` decode GPUs. That would exceed the declared total budget.

## Workloads

### Primary design workload

```text
input tokens       = 90,000
output tokens      = 8,000
global batch       = 8
decode compute     = W4A16
decode KV cache    = FP16
```

The PLENA DSE models the 90,000-token prefill. The 8,000-token output length
is retained as system metadata for decode capacity, decode execution, and the
final system comparison.

### Fixed-hardware holdouts

The final hardware selected for each model is re-evaluated without redesign:

```text
1,400 input /   200 output
90,000 input / 8,000 output
114,000 input / 5,000 output

runtime batch = 1, 2, 4, 8
```

These points test whether the long-context design remains useful outside its
primary workload and provide the batch-dependent curves required by the
serving model.

## PLENA Candidate DSE

Five budget-conditioned prefill searches generate the candidate pool:

```text
Qwen3-32B:
  P=2, D=6
  P=4, D=4

Qwen3-235B-A22B:
  P=4,  D=12
  P=8,  D=8
  P=12, D=4
```

Each search uses:

```text
4096 COMPLETE trials
512 TPE startup trials
accuracy > 0.9

objectives:
  minimize prefill latency
  minimize prefill system energy

model stack:
  current-dse-v1
  tile-aware DP x TP x EP v4
  RTL-v6 compiler lowering
  ideal-II1 compute timing
  HBM V4
  power shadow enabled
```

The five searches are candidate-generation runs, not five final chip designs.
For each model, the union of their Pareto candidates is deduplicated by the
complete per-chip architecture signature. The final system study selects one
fixed PLENA microarchitecture per model, including:

```text
precision/datapath configuration
MLEN and VLEN
BLEN
Matrix SRAM organization
softmax row-lane count
```

The number of identical PLENA chips may change with deployment. Hardware
parameters must not change across P:D splits, runtime batches, or holdout
workloads after final selection.

Compiler and model caches should be shared across the budget-conditioned runs
when their semantic keys match. Search studies and result directories remain
separate because their area, HBM, and topology constraints differ.

## A100 Measurements

Measurements use one 8 x A100 80 GB RunPod node. Both models use W4A16 compute
and FP16 decode KV unless a failure is explicitly reported.

Every formal point records:

```text
TTFT
TPOT/TBT
full-request E2E latency
output throughput
peak GPU memory
available KV capacity
integrated energy
average and peak power
NVLink traffic
runtime and topology metadata
```

The formal 90k/8k points generate all 8,000 output tokens. Each point uses one
warm-up run followed by three measured runs. Raw samples are retained; the
median is used in the comparison and the spread is reported.

### Qwen3-32B topology candidates

```text
D4:
  TP4 x DP1
  TP2 x DP2
  TP1 x DP4

D6:
  TP2 x DP3
  TP1 x DP6
```

The aggregated 8-GPU baseline evaluates legal TP/DP decompositions on the
complete node. Decode capacity is checked independently for each topology.

### Qwen3-235B-A22B topology candidates

The primary homogeneous-replica mapping is:

```text
aggregated R16: TP4 x DP4
D12 decode:     TP4 x DP3
D8 decode:      TP4 x DP2
D4 decode:      TP4 x DP1
```

A single TP4 replica is measured directly. Two TP4 replicas are also run
concurrently on the 8-GPU node to check that replica scaling does not expose a
host, PCIe, or power-management bottleneck.

The 12-GPU and 16-GPU results are then reconstructed as independent replicas:

```text
global latency = max(replica latency)
global energy  = sum(replica energy)
```

These results must be labelled `measured-replica extrapolation`; they are not
direct 12-GPU or 16-GPU measurements.

## Capacity And KV Semantics

Prompt handoff and decode residency are different quantities:

```text
FP16 KV handoff bytes:
  KV generated for input tokens by PLENA prefill

Decode capacity requirement:
  model weights
  + runtime workspace
  + FP16 KV for input and generated output tokens
```

Each decode topology must fit its complete local weight and KV working set in
the assigned GPUs with the declared runtime reserve. An infeasible split is
reported as `capacity_infeasible`; batch size must not be silently reduced.

`P12:D4` is a required boundary case for 235B. Its B8 capacity margin must be
reported explicitly for 90k/8k and 114k/5k.

## System Combination

For matched-batch latency:

```text
TTFT_disaggregated =
    PLENA prefill latency
  + FP16 prompt-KV handoff latency
  + A100 first-token decode latency

E2E_disaggregated =
    PLENA prefill latency
  + FP16 prompt-KV handoff latency
  + complete A100 decode latency

system energy =
    PLENA prefill energy
  + handoff energy
  + A100 decode energy
```

The aggregated baseline uses the same global batch, token counts, compute
precision, and decode-KV precision.

For steady-state serving:

```text
batch service interval =
    max(prefill service interval,
        handoff service interval,
        decode service interval)

goodput =
    requests satisfying both TTFT and TPOT SLO / second
```

The system-level evaluator exhaustively enumerates the declared P:D splits,
legal decode topologies, and runtime batches for every retained fixed PLENA
candidate. This small provisioning search does not use Optuna.

## Final Selection And Outputs

Each model report contains:

- Matched-B8 aggregated and disaggregated results for every declared split.
- The latency-energy Pareto frontier.
- Minimum energy under `1.00x`, `1.25x`, and `1.50x` A100 latency bounds.
- TTFT, TPOT, E2E latency, and goodput envelopes for B1/B2/B4/B8.
- Goodput/W, requests/J, input tokens/J, and total tokens/J.
- Prefill, handoff, and decode latency and energy breakdowns.
- Decode capacity, PLENA area/HBM, topology, and extrapolation status.
- The same fixed hardware evaluated on all three workloads.

The headline claim should use this form:

> PLENA disaggregation reduces system energy while satisfying a stated
> latency or serving SLO.

It must not default to claiming that PLENA has lower raw TTFT than A100.

## Validation And Claim Boundaries

- Every evaluated system satisfies `P + D = R_total`.
- Aggregated and disaggregated systems use identical model, token, batch, and
  quantization semantics.
- Decode weights, runtime workspace, and complete FP16 KV fit on the assigned
  GPUs.
- Infeasible systems remain visible as infeasible and are not repaired by an
  implicit batch or precision change.
- One fixed PLENA hardware design is selected per model.
- System energy equals the sum of prefill, handoff, and decode energy.
- Results using more than eight A100s are marked as replica extrapolations.
- Model fidelity, ideal-II1 timing, power assumptions, and excluded physical
  costs accompany every selected result.
- Cross-model results are not treated as an iso-resource comparison because
  32B and 235B use different total deployment budgets.

The experiment therefore measures the latency cost and system-energy benefit
of disaggregation without assuming that energy efficiency automatically
implies lower latency.

## Local DSE Readiness Check

The five budget-conditioned studies were exercised locally with the formal
`90k input / 8k output / batch 8` workload before launching the large remote
runs. These are infrastructure smokes, not converged Pareto results.

| Study | Local smoke | Result | Cache condition |
|---|---:|---:|---|
| 32B P2:D6 | 16 COMPLETE / 8 workers | 77.6 s, 0 PRUNED, 0 FAIL | 12 cold reports |
| 32B P4:D4 | 8 COMPLETE / 8 workers | about 16 s, 0 PRUNED, 0 FAIL | shared 32B cache |
| 235B P4:D12 | 64 COMPLETE / 8 workers | 84.3 s, 0 PRUNED, 0 FAIL | 51 cold reports |
| 235B P8:D8 | 8 COMPLETE / 8 workers | about 16 s, 0 PRUNED, 0 FAIL | shared 235B cache |
| 235B P12:D4 | 8 COMPLETE / 8 workers | about 16 s, 0 PRUNED, 0 FAIL | shared 235B cache |

A fixed representative 235B P4:D12 point completed cold in 34.4 s with a
0.68 GiB peak RSS. Its modeled prefill result was 146.85 s and 45.99 kJ. A
warm evaluation of the same compiler report took about 5 s end to end; the
cached objective phases themselves took about 0.5 s.

Two search-control defects were found and fixed:

1. Optuna 4.2.1 could pass PRUNED multi-objective trials with `values=None`
   into its hypervolume weighting path, causing a repeated `None * float`
   failure. Non-COMPLETE trials now receive an infeasible constraint value.
2. Startup and random sampling could select a DP/TP/EP topology, or even a
   physical chip count, that could never hold the selected precision's
   replicated weights and prefill KV working set. The canonical domain is now
   conditioned on precision and exact HBM capacity before CostEmitter runs.

Before conditioning, the fractions of otherwise legal topology/profile pairs
that met HBM capacity were:

```text
32B P2:  90.2%       32B P4:  100.0%
235B P4: 43.9%       235B P8: 63.5%       235B P12: 75.7%
```

Every validated precision profile has at least one feasible topology in its
declared budget. Capacity conditioning therefore removes no deployable point;
it only prevents structural PRUNEs. The final 64-point 235B P4 test achieved a
100% effective completion rate.

The first cold 32B `MLEN=512` reports remain the expensive case: roughly
41 s in long-context lowering and 55-68 s in HBM V4 on one worker. Cross-budget
content-addressed reuse reduces matching reports to sub-second objective work.
Formal runs should therefore execute in this order and share one cache per
model:

```text
32B:  P2 then P4
235B: P4 then P8 then P12
```

On Tromokratis, the initial provisional expectation is tens of minutes for
the first 4096-point study of each model and substantially less for later
budgets that reuse compiler/V4 reports. This estimate must be updated after
the first 512 COMPLETE points; it is not a measured remote runtime claim.
