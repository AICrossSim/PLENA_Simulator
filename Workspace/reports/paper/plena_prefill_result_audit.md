# Audit Notes for Published PLENA Prefill Results

## Purpose

These notes support a future prefill-optimized PLENA paper. They are not an
allegation of misconduct. They identify ambiguities that must be resolved
before reusing TTFT numbers from [PLENA v3](https://arxiv.org/html/2509.09505v3)
as a baseline.

## Main Reproducibility Concern

The published selected configuration reports `BLEN=32`, `MLEN=VLEN=2048`,
4-bit W/A/KV and 16 accelerators at 1 GHz. One accelerator therefore contains

```text
MLEN * BLEN = 65,536 MACs
```

and 16 accelerators provide at most 1.048576e15 MAC/s under the direct
one-MAC-per-PE-per-cycle interpretation. The paper's area/throughput table is
consistent with this convention: `0.237 mm2 * 34.49 TOPS/mm2` is about
8.17 TOPS, matching 4,096 MACs at 1 GHz when a MAC is counted as two operations.

Using only dense model weight MACs as a conservative floor gives the following
checks. Attention and all non-matmul work are omitted, so these are lower
bounds rather than full latency predictions.

| Case | Conservative compute floor | Published TTFT | Required multiple of nominal peak |
|---|---:|---:|---:|
| Qwen-32B, 90k prompt, batch 64 | 175.8 s | 90.71 s | 1.94x |
| Qwen-32B, 114k prompt, batch 64 | 222.7 s | 108.1 s | 2.06x |
| Llama-70B, 90k prompt, batch 16 | 96.1 s | 43.43 s | 2.21x |
| Llama-70B, 114k prompt, batch 16 | 121.8 s | 69.10 s | 1.76x |

Under the straightforward interpretation of batch, accelerator count and PE
throughput, these TTFT values require more than nominal aggregate peak even
before attention, normalization, memory and control are included.

## Plausible Explanations Requiring Clarification

The discrepancy could come from an undocumented convention rather than a
numerical error. A reproducible comparison needs answers to:

1. Does reported batch size mean concurrent requests, aggregate tokens across
   replicas, or a throughput-normalized batch?
2. Is TTFT per request, per accelerator group, or after hidden model/tensor
   replication beyond the stated 16 accelerators?
3. Does one PE perform more than one independent MAC per cycle in the reported
   configuration?
4. Are model weights, sequence length or layer count scaled in the simulator?
5. Is the reported number latency, amortized throughput converted to latency,
   or a mixture of prefill and decode conventions?

## Validation Evidence in the Paper

The paper describes the simulator as cycle-accurate in some locations and
cycle-approximate in others. The reported simulator validation is based on a
small set of single-layer Llama-70B trials and an average error, without a
shape-by-shape error table or a tail metric. That evidence is insufficient to
establish accuracy for 90k-114k token, multi-accelerator TTFT.

## Recommended Thesis Position

Do not copy the published TTFT as a trusted ground truth. Use it as a reported
baseline with an explicit reproducibility caveat, and report the new system in
three layers:

1. Direct RTL evidence: opcode timing and DC area anchors.
2. Calibrated surrogate evidence: holdout errors, domains and unsupported
   fractions.
3. End-to-end DSE estimates: compiler schedule, workload semantics and all
   extrapolation warnings.

For the new paper, include a throughput sanity floor for every headline TTFT:

```text
minimum dense MAC work / aggregate peak MAC throughput <= reported latency
```

Also publish the exact interpretation of batch, device replication, frequency,
precision throughput, prompt length, decoder-layer count and whether lm_head is
included. This makes the prefill result independently checkable even when the
full simulator cannot be released.

## Current Project Boundary

The current PLENA Simulator work improves several known sources of optimism or
pessimism: Matrix timing is BLEN-based and RTL-calibrated, DMA waits for actual
completion, V4 matches production gather/scatter semantics, compiler padding
has been compacted, and the area model uses structural instance counts.

It still does not justify an absolute silicon TTFT claim. Production opcodes
and sizes remain partly extrapolated, V4 is post-hoc, stage roofline omits a
full online schedule, and 1 GHz is an assumed conversion rather than timing
closure. The appropriate claim is a calibrated and auditable architectural
estimate with reported uncertainty, not cycle-exact measured hardware.
