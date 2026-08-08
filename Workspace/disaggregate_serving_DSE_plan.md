# Disaggregated Serving DSE Experiment Plan v2

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

Each formal search used:

```text
16,384 COMPLETE trials
2,048 TPE startup trials
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

`normalized_latency` and `normalized_energy` are currently objective-field
names only. They contain the unnormalized prefill latency and energy values;
no A100 result is divided into either objective.

All validated precision profiles with `accuracy > 0.9` remain eligible. The
candidate set is not restricted to profiles having an identical accuracy
score.

The five searches are candidate-generation runs, not five final chip designs.
For each model, the union of their Pareto candidates is deduplicated by the
complete per-chip architecture signature. Each unique candidate is then
re-scored across every declared `P:D` split and legal decode topology for the
primary `90k/8k, B8` workload.

The final system selector chooses the minimum-system-energy combination that
satisfies:

```text
E2E_disaggregated <= 1.25 x E2E_aggregated_A100
```

The PLENA microarchitecture belonging to that combination is fixed for the
model. The same hardware is subsequently evaluated under all other splits,
runtime batches, and holdout workloads. The report also runs the selector at
`1.00x` and `1.50x` E2E bounds.

The fixed per-chip signature includes:

```text
precision/datapath configuration
MLEN and VLEN
BLEN
Matrix SRAM organization
softmax row-lane count
```

The number of identical PLENA chips and their runtime DP/TP/EP mapping may
change with deployment. Hardware parameters must not change across P:D
splits, runtime batches, or holdout workloads after final selection.

Compiler and model caches should be shared across the budget-conditioned runs
when their semantic keys match. Search studies and result directories remain
separate because their area, HBM, and topology constraints differ.

## A100 Measurements

Measurements use one 8 x A100 80 GB RunPod node and a frozen vLLM software
stack. Both models use W4A16 compute and FP16 decode KV unless a failure is
explicitly reported. The following are recorded in every result and remain
fixed within a model campaign:

```text
vLLM and container version
model and tokenizer revision
W4A16 quantization backend
CUDA and driver version
max_model_len
gpu_memory_utilization
scheduler, KV-block, and execution settings
```

The frozen checkpoints are:

```text
Qwen3-32B:
  Qwen/Qwen3-32B-AWQ

Qwen3-235B-A22B:
  QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ
```

The resolved Hugging Face commit SHA is stored after the first download.
AWQ-Marlin is attempted first; if either model fails preflight, every formal
point uses the same AWQ backend instead. The two models must use one vLLM
version and one frozen container image digest.

The 32B short workload uses native RoPE. Its 90k and 114k workloads use static
YaRN with factor 4, original context 32,768, and `max_model_len=131072`. The
235B checkpoint uses its native 262,144-token context without RoPE scaling.

Prompts use deterministic valid token IDs. Decoding is greedy, ignores EOS,
and generates exactly the declared output length.

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

Every run records four synchronized boundaries:

```text
request_start
prefill complete / first output token
first normal decode iteration complete
last output token
```

vLLM does not import external PLENA KV in this experiment. The harness defines
`imported_kv_decode_proxy` as one measured normal decode iteration plus the
measured generation interval after the first output token. This proxy is used
for disaggregated composition and is never labelled as a real KV-import run.
The A100 prefill phase must not be added after PLENA prefill.

Formal energy is total board energy integrated over all allocated GPUs.
NVML's total-energy counter is preferred and is cross-checked with fixed-rate
power sampling. Idle-subtracted dynamic energy is reported as a secondary
diagnostic, not used by the primary selector.

Every 90k/8k screening point generates all 8,000 output tokens once after one
same-shape warmup. Screening therefore does not extrapolate long decode from a
short output. The best topology for each declared system, plus any runner-up
within 5%, is then repeated until it has three formal measurements. The median
of those confirmation runs is used in the comparison and the spread is
reported.

Screening uses a GPU-aware parent scheduler. Reusable engine groups are split
across disjoint physical GPU sets, allowing up to `8 x TP1`, `4 x TP2`, `2 x
TP4`, or `1 x TP8` on the eight-GPU node, including compatible mixed-TP
packing. Points in the same shard reuse one model load. This concurrency is
used only for candidate discovery. Confirmation and 114k
holdout points run in isolation by default so their formal latency and energy
are not biased by unrelated co-tenants. The short sweep reuses each loaded
engine while different non-overlapping TP groups run concurrently.

Physical GPU placement, worker overlap windows and process return status are
recorded separately from the semantic point fingerprint. A capacity failure
removes only that point and releases its GPU allocation; it does not terminate
the remaining campaign.

The inexpensive 1.4k/0.2k workload may sweep all topology points with three
measurements. The 114k/5k holdout runs only the primary-selected topologies,
again with three measurements. It does not repeat the full topology sweep.

The implemented harness is located at
`analytic_models/serving_benchmark/`. Its frozen manifest expands to 42 formal
points and seven compatibility/capacity preflight points. Formal points are grouped
into ten reusable vLLM engine configurations, so changing local batch does not
reload the same checkpoint. The `inventory`, `preflight`, `run`, `aggregate`,
and `replica-check` commands provide hardware admission, environment locking,
resumable execution, validation, and synchronized dual-TP4 measurement.

Each measured repetition retains compressed engine events, per-token timing,
20 Hz board telemetry, NVML total-energy deltas, output-token hashes, and
best-effort DCGM NVLink fields 1011/1012. Missing DCGM profiling privileges are
reported as unavailable rather than replaced with estimated traffic.

Model-load and 32B YaRN compatibility probes are mandatory. Maximum-shape
capacity probes are diagnostic: an OOM is retained as `capacity_infeasible`
evidence and does not incorrectly force a quantization-backend fallback.

### Qwen3-32B topology candidates

Reusable per-replica curves are measured at:

```text
TP1: local batch 1, 2
TP2: local batch 2, 3, 4
TP4: local batch 4, 8
TP8: local batch 8
```

They reconstruct the following formal systems:

```text
D4:
  TP4 x DP1: local batch 8
  TP2 x DP2: local batch 4 + 4
  TP1 x DP4: local batch 2 + 2 + 2 + 2

D6:
  TP2 x DP3: local batch 3 + 3 + 2
  TP1 x DP6: local batch 2 + 2 + 1 + 1 + 1 + 1

aggregated R8:
  TP8 x DP1: local batch 8
  TP4 x DP2: local batch 4 x 2
  TP2 x DP4: local batch 2 x 4
  TP1 x DP8: local batch 1 x 8
```

Decode capacity is checked independently for each topology.

### Qwen3-235B-A22B topology candidates

Reusable per-replica curves are measured at:

```text
TP4: local batch 2, 3, 4, 8
TP8: local batch 4, 8
```

The formal mappings are:

```text
D4:  TP4 x DP1, local batch 8
D8:  TP4 x DP2, local batch 4 x 2
     TP8 x DP1, local batch 8
D12: TP4 x DP3, local batch 3 + 3 + 2
R16: TP4 x DP4, local batch 2 x 4
     TP8 x DP2, local batch 4 x 2
```

A single TP4 replica and two concurrent TP4 replicas are measured on the
8-GPU node to check that replica scaling does not expose a host, PCIe, or
power-management bottleneck. TP8 configurations are measured directly on all
eight GPUs.

The 12-GPU and 16-GPU results are then reconstructed as independent replicas:

```text
global latency = max(replica latency)

global energy =
    sum(active replica energy)
  + sum(replica idle power x (global latency - replica latency))
```

These results must be labelled `measured-replica extrapolation`; they are not
direct 12-GPU or 16-GPU measurements.

If the two-replica measurement differs from independent-replica
reconstruction by more than 5%, the observed concurrency correction is
applied and reported. Uncorrected independent-replica extrapolation is not
used in that case.

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
  + A100 first decode-step latency from imported KV

E2E_disaggregated =
    PLENA prefill latency
  + FP16 prompt-KV handoff latency
  + complete A100 decode-from-KV latency

system energy =
    PLENA prefill energy
  + handoff energy
  + A100 decode energy
```

The handoff planner assigns each complete request's prompt KV to its decode
replica and schedules the resulting transfers against the selected PLENA and
decode endpoint ports. Handoff latency is the slowest routed transfer, not
aggregate bytes divided by an unrelated system-wide bandwidth. RunPod's
internal A100 NVLink traffic is not presented as a measurement of the
PLENA-to-A100 handoff link.

The aggregated baseline uses the same global batch, token counts, W4A16 A100
compute, FP16 decode KV, and vLLM runtime settings.

For steady-state serving:

```text
batch service interval =
    max(prefill service interval,
        handoff service interval,
        decode service interval)

goodput =
    requests satisfying both TTFT and TPOT SLO / second
```

The formal relative serving SLO is:

```text
TTFT_disaggregated <= 1.25 x matched aggregated-A100 TTFT
TPOT_disaggregated <= 1.10 x matched aggregated-A100 TPOT
```

This is a deterministic three-stage analytical serving envelope. Stages are
dependency-serial within one batch, while different batches may occupy
prefill, handoff, and decode concurrently. It assumes paced admission and
does not claim to model queueing delay, online continuous batching, or a real
distributed scheduler.

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
- The measured prefill, first-decode-step, and remaining-decode phase times and
  energies reconstruct the complete GPU run within measurement resolution.
- Disaggregated composition includes no A100 prefill latency or energy after
  PLENA prefill.
- Aggregated and disaggregated systems use identical model, token counts, and
  global batch. Aggregated A100 and A100 decode use the same W4A16 backend and
  FP16 KV format.
- PLENA may use any validated DSE precision profile with `accuracy > 0.9`.
  Its accuracy score and complete precision profile accompany every result;
  the comparison does not require identical PLENA and A100 internal
  quantization.
- PLENA hands off FP16-formatted KV but does not claim that it is bitwise
  identical to KV generated by A100 prefill.
- Decode weights, runtime workspace, and complete FP16 KV fit on the assigned
  GPUs.
- Infeasible systems remain visible as infeasible and are not repaired by an
  implicit batch or precision change.
- TP/DP reconstruction conserves the global request count, token count, and
  total measured replica energy, including idle tails for uneven replicas.
- One fixed PLENA hardware design is selected per model.
- Holdout workloads and runtime batches may change deployment topology but do
  not trigger a new hardware search.
- System energy equals the sum of prefill, handoff, and decode energy.
- Results using more than eight A100s are marked as replica extrapolations.
- Model fidelity, ideal-II1 timing, power assumptions, and excluded physical
  costs accompany every selected result.
- Cross-model results are not treated as an iso-resource comparison because
  32B and 235B use different total deployment budgets.
- This revision does not alter the existing PLENA power or clock-gating
  semantics.

The experiment therefore measures the latency cost and system-energy benefit
of disaggregation without assuming that energy efficiency automatically
implies lower latency.

## Formal DSE Campaign Status

All five budget-conditioned candidate-generation studies completed with
`16,384 COMPLETE` trials and `2,048` TPE startup trials per study. The repaired
Pareto export contains the following numbers of unique objective-space points:

| Model and budget | Unique Pareto points |
|---|---:|
| Qwen3-32B P2:D6 | 5 |
| Qwen3-32B P4:D4 | 9 |
| Qwen3-235B-A22B P4:D12 | 10 |
| Qwen3-235B-A22B P8:D8 | 12 |
| Qwen3-235B-A22B P12:D4 | 14 |

These counts retain all precision profiles above the accuracy threshold; they
are not produced by matching candidates to one accuracy value. The completed
campaigns now serve as the PLENA candidate source for the RunPod measurement
and system-composition stage. Historical smoke timings and provisional remote
runtime estimates are intentionally excluded from the formal experiment
specification.
