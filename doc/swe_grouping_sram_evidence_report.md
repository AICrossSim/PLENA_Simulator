# PLENA-MoE SWE Expert Grouping and SRAM Evidence Report

## 1. Executive Result

The current evidence supports one large effect and rejects one proposed SRAM
effect:

1. Grouping all token rows routed to the same expert and keeping each weight
   panel stationary removes repeated expert-weight fetches. On representative
   Rust/Ramulator layer-step replays this improves cycles by `1.274x--3.108x`.
2. Enlarging a two-panel ping-pong organization into a generic cross-expert
   pool of 4 or 8 panels does not improve cycles further. Its best incremental
   gain over ordinary ping-pong is only `1.0024x--1.0028x`.

Therefore the recommended version-one mechanism is:

```text
runtime route coalescer
  -> deterministic unique-expert job queue
  -> MX-aware tile-major weight layout
  -> two-panel ping-pong prefetch
  -> keep one expert panel resident across all of its token-row groups
  -> deterministic route-weighted scatter/combine
```

A deep elastic SRAM panel pool should not be presented as the architectural
contribution. The measured benefit is too small even before physical tag,
bank-conflict, allocator, and arbitration costs are charged.

## 2. Truth Levels and Scope

The measurements below intentionally use three different truth labels.

| Evidence | What was covered | What it proves | What it does not prove |
|---|---|---|---|
| Full-population route audit | All 2,294 archived SWE-bench issue prompts, all 16 saved decode tokens, all MoE layers | Exact routed-expert population and exact formula/64-B-rounded weight traffic for the saved archive | Full generation to EOS or a complete agent trajectory |
| Representative Compiler + Rust/Ramulator replay | One exact saved layer-step per model and batch | Comparative cycles and HBM bytes for pair-major versus grouped execution | End-to-end request latency or RTL-calibrated absolute cycles |
| Nonzero SRAM mechanism gate | One exact B8 SWE route per model, reduced functional dimensions | Numerical correctness, exact bytes, deterministic cycles, and isolated panel-placement effect | Full-model absolute timing or physical SRAM bank/port timing |

The archived B2/B4/B8/B16 groups are static cohorts. They are not claimed to be
a full continuous-batching arrival/retirement trace. A separate older pilot
contains all turns from 9 SWE-bench Verified tasks, but the full 500-task agent
trajectory corpus is not present on this machine.

## 3. Full Archived SWE Decode Population

The following table covers every saved decode layer-step in the two 2,294-prompt
archives. `M=1..4` is the fraction of active routed experts receiving at most
four token rows. Byte reduction compares pair-major repeated loading against
one physical expert-weight load per layer-step, including an unchanged shared
expert load in both cases.

| Model | Batch | Layer-step windows | Active experts with M=1..4 | Grouped row occupancy | Physical weight bytes reduced | Weight-traffic ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B-FP8 | 2 | 734,080 | 100.00% | 36.61% | 29.85% | 1.426x |
| Qwen3.5-35B-A3B-FP8 | 4 | 367,360 | 100.00% | 56.81% | 54.30% | 2.188x |
| Qwen3.5-35B-A3B-FP8 | 8 | 183,680 | 64.50% | 69.20% | 72.21% | 3.599x |
| Qwen3.5-35B-A3B-FP8 | 16 | 92,160 | 47.68% | 77.77% | 83.22% | 5.959x |
| DeepSeek-V2-Lite-Chat | 2 | 477,152 | 100.00% | 32.15% | 19.07% | 1.236x |
| DeepSeek-V2-Lite-Chat | 4 | 238,784 | 100.00% | 45.99% | 42.13% | 1.728x |
| DeepSeek-V2-Lite-Chat | 8 | 119,392 | 79.83% | 58.89% | 62.08% | 2.637x |
| DeepSeek-V2-Lite-Chat | 16 | 59,904 | 60.89% | 70.32% | 76.23% | 4.206x |

This table establishes that duplicate weight transfer is widespread in the
saved SWE decode archive. The traffic ratio is not itself a latency speedup.

## 4. Representative Rust/Ramulator Cycle Replay

Each row replays one deterministic exact route slice selected from the full
archive. Both variants use the same single matrix engine and Ramulator-backed
64-B HBM requests. The only semantic change is pair-major versus same-expert
grouped execution with one weight load per expert.

| Model | Batch | Pair-major cycles | Expert-grouped cycles | Cycle speedup | HBM bytes reduced |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B-FP8 | 2 | 17,387,390 | 12,276,966 | 1.416x | 29.41% |
| Qwen3.5-35B-A3B-FP8 | 4 | 33,774,877 | 15,428,030 | 2.189x | 54.55% |
| Qwen3.5-35B-A3B-FP8 | 8 | 67,425,182 | 24,922,982 | 2.705x | 72.31% |
| Qwen3.5-35B-A3B-FP8 | 16 | 134,690,778 | 43,332,083 | 3.108x | 82.95% |
| DeepSeek-V2-Lite-Chat | 2 | 38,735,228 | 30,408,728 | 1.274x | 21.43% |
| DeepSeek-V2-Lite-Chat | 4 | 72,021,984 | 41,542,187 | 1.734x | 42.31% |
| DeepSeek-V2-Lite-Chat | 8 | 143,327,861 | 64,723,918 | 2.214x | 62.00% |
| DeepSeek-V2-Lite-Chat | 16 | 285,677,820 | 108,818,763 | 2.625x | 76.53% |

These are comparative Rust simulation cycles, not end-to-end GPU measurements
and not absolute RTL timing. Timing uses zero-valued tensors because values do
not alter the instruction or HBM request schedule; numerical correctness is
checked separately with nonzero tensors below.

## 5. Fixed-Capacity Matrix-SRAM Ablation

All rows below use the same 512-KiB Matrix-SRAM capacity, one matrix engine,
exact nonzero MX weights, random nonzero activations, 64-B HBM requests, and
three exactly repeatable runs. Both models transfer exactly `8,404,992` HBM
bytes in every organization.

| Model | Blocking one-panel | One-expert ping-pong | Cross-expert pool 2 | Pool 4 | Pool 8 | Pool best vs ping-pong |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 SWE B8 route | 3,185,289 | 3,150,189 | 3,142,506 | 3,142,506 | 3,142,506 | 1.0024x |
| DeepSeek-V2-Lite SWE B8 route | 2,812,254 | 2,779,754 | 2,772,071 | 2,772,071 | 2,772,071 | 1.0028x |

Qwen rel-RMS is `0.003411`; DeepSeek rel-RMS is `0.001144`. Cycles, HBM bytes,
and output hashes are identical across all three repeats.

Interpretation:

- One-panel to two-panel ping-pong gives about `1.011x`; this is a small but
  repeatable overlap benefit.
- Interleaving two expert streams gives only another `0.24%--0.28%`.
- Four and eight slots produce exactly the same cycles as two slots.
- Because bank/port conflicts are not yet physically modeled, these figures are
  optimistic for the more complex pool. The generic deep pool is rejected.

## 6. Weight Layout and Asynchronous Prefetch Controls

A separate Shared-FFN microbenchmark isolates exact MX tile layout, physical
64-B scale-burst coalescing, and dependency-aware asynchronous prefetch:

| Case | Baseline cycles | Final cycles | Total speedup | Physical HBM bytes reduced |
|---|---:|---:|---:|---:|
| DeepSeek-style M=4, H=I=512 | 272,998 | 220,653 | 1.237x | 43.64% |
| DeepSeek-style M=64, H=I=512 | 3,458,631 | 3,266,859 | 1.059x | 42.00% |
| Qwen sigmoid shared-gate smoke, M=4, H=I=64 | 6,600 | 6,133 | 1.076x | 43.62% |

This is a mechanism ablation, not a complete Transformer or SWE trajectory.

## 7. Why the Earlier Lane-Splitting DSE Did Not Win

The older MNK/lane replay explored 10,989 legal analytical architectures and
shortlisted ten candidates. Its Rust result set is incomplete overall
(`429/720` reports), so no all-phase aggregate is valid. The common complete
decode subset contains 24 windows times all 10 architectures.

On that fair decode intersection, the best non-baseline candidates are only
about `1.000x`; several small-core DeepSeek candidates are substantially worse.
The reason is physical: splitting the same 4,096 PEs does not reduce expert
weight bytes or increase total HBM bandwidth. It only changes where the same
memory-bound work waits. This experiment is distinct from grouping, which
actually removes duplicate transfers and therefore produces the large gains in
Section 4.

## 8. Architecture Decision

### Selected version-one dataflow

1. Router output enters a runtime route coalescer. It creates one descriptor per
   unique expert with deterministic token IDs, route weights, and sequence ID.
2. The HBM layout stores each MX weight tile contiguously and coalesces scale
   slices sharing the same physical 64-B burst.
3. Two four-cell Matrix-SRAM panels form ping-pong storage. One panel computes
   while the next exact tensor slice fills.
4. A ready expert panel stays resident while every `ceil(M/BLEN)` token-row
   group consumes it. This is panel-stationary reuse and is the source of the
   large byte reduction.
5. Shared experts use the same path with all token rows. Results carry
   `{token_id, route_slot}` into deterministic route weighting and combine.

### Rejected version-one mechanisms

- A deeper generic multi-expert panel pool: measured incremental gain below
  `0.3%`, with no benefit from slots 4 or 8.
- Compute-lane splitting as the primary speedup: it preserves bytes and total
  bandwidth and showed no meaningful gain on the complete decode intersection.
- Static HBM channel binding: not modeled or justified; it can strand bandwidth
  when shared or hot experts need all channels.

### Remaining SRAM research hypothesis

An SRAM-specific paper contribution still needs a mechanism that changes a
material bottleneck. The strongest next hypothesis is temporal expert-tile
residency across adjacent decode steps, selected by measured route reuse. It can
reduce HBM bytes rather than merely queueing the same bytes more deeply. It must
be evaluated against the two-panel baseline with finite capacity, replacement,
bank conflicts, and shared-expert interference.

## 9. Required Next Validation

1. Move grouping from the fixed-route compiler oracle into the device-selected
   runtime route coalescer; no host round trip is allowed.
2. Replay complete requests to EOS with real continuous batching. The current
   2,294-prompt archive stops after 16 saved decode tokens.
3. Add bank/port conflict, panel allocator, and stage critical-path counters.
   Current global cycles and bytes are valid for the modeled resources, but
   overlapping stage attribution is incomplete.
4. Calibrate matrix, load/compute overlap, and combine primitives against RTL.
5. Measure cross-step expert-tile reuse before proposing an expert cache or
   residency-aware SRAM organization.

## 10. Evidence Paths

- Full route population:
  `/scratch/shared/mcl123/plena/outputs/swe_grouping_sram_20260903/full_population/`
- Representative cycle replay:
  `/scratch/shared/mcl123/plena/outputs/swe_grouping_sram_20260903/representative_cycle_replay/`
- SRAM depth sweep:
  `/scratch/shared/mcl123/plena/outputs/swe_grouping_sram_20260903/sram_ring_sweep/`
- Cross-expert panel pool:
  `/scratch/shared/mcl123/plena/outputs/swe_grouping_sram_20260903/panel_pool_sweep/`
- Layout/async ablation:
  `/scratch/shared/mcl123/plena/outputs/layout_async_ablation_20260903/`
- Older MNK/lane DSE:
  `/scratch/shared/mcl123/plena/outputs/mnk_dse_real_swe_20260827/`

No result in this report is an RTL signoff or an end-to-end GPU speedup claim.
