# Research question: when does heterogeneous compute help MoE?

Status: hypothesis and experiment plan, not an established novelty or speedup
claim. Scope: Compiler export contracts and Rust numerical/timing simulation.

## The story in plain language

MoE sends different numbers of tokens to different experts. A wide compute
organization may waste work on a lightly used expert. A narrower organization
may avoid that padding, but dividing a chip also divides its local supply paths
and introduces additional scheduling and feedback constraints. Less wasted
arithmetic does not automatically mean an earlier completed MoE output.

The current B8 shape diagnostic makes the issue concrete: after revising the
large core, heterogeneous useful/issued MACs reached 99.58%, yet the operator
took 0.828061 ms versus the matched single core's 0.717776 ms with per-channel
DMA. The revision helped the heterogeneous configuration, but eliminating nearly
all padding did not produce a win. This is a bounded operator experiment with
synthetic values and an archived routing window, not a general impossibility
result. It motivates measuring where recovered arithmetic capacity is lost.

The research question is: **with the same fixed weight image, HBM bandwidth,
multiplier count and declared storage/port budgets, can coordinated tile shape,
operand delivery and independent-output scheduling turn reduced padding into
lower complete-operator latency as expert popularity changes?**

## What is engineering groundwork, and what might become a contribution?

| Part | Current research status |
|---|---|
| Add a core or private SRAM | Architecture choice; not sufficient novelty. |
| Immutable full expert bank, checksums, exact numerical oracle | Necessary experimental validity, not a speedup mechanism. |
| Separate temporal token rows Mt from physical output/reduction lanes P/R | Enables a fair shape experiment; configurable tiles alone are not a new idea. |
| Count decoded-weight and accumulator ports, storage and feedback | Model fidelity needed to support conclusions, not automatically a contribution. |
| Interleave independent outputs and prioritize current operand demand | Concrete mechanisms to test; both have broad prior art. |
| Preserve original scale ownership while reading different views | Required correctness. A distinct efficient multi-view delivery mechanism, with measured benefits and costs, could support a contribution; correct indexing alone does not. |

A possible central contribution is a **specific bounded operand-admission and
output-scheduling mechanism** that adapts to heterogeneous shapes without
repacking weights for known future routes. Its case must rest on a mechanism
that previous designs do not already provide, rather than a list of familiar
features. The current implementation is a platform for testing that claim.

## Closely related work already rules out broad novelty claims

This is an initial primary-source check, not an exhaustive novelty search.

- [HarMoEny](https://arxiv.org/abs/2506.12417) combines dynamic token redistribution
  and asynchronous expert prefetching on multiple GPUs. “MoE load balancing
  plus prefetch” cannot be our distinct claim.
- [Expert Streaming](https://arxiv.org/abs/2603.27624) schedules fine-grained
  expert streams across chiplets to overlap computation and communication.
  “Fine-grained expert scheduling” and overlap alone are insufficient.
- [ThAME](https://arxiv.org/abs/2607.17074) combines heterogeneous 3D memory/compute
  organization and an MoE-specific communication network. “Heterogeneous MoE
  hardware” is already an established direction. Our proposed setting keeps
  conventional shared HBM, rather than claiming its memory technology.
- [TEMPO](https://arxiv.org/abs/2608.13057) explicitly models weight-streaming
  and padded-compute regimes for expert-parallel dispatch. A finish-time cost
  model or a win/loss regime diagram is not by itself a new contribution.

Differences in platform are not enough to establish novelty. Before writing a
claim, compare the actual algorithms, constraints and access mechanisms in
these papers and their relevant cited predecessors.

## Experiments that can support or reject the story

1. Freeze one complete expert bank per model before choosing route windows.
   Include a previously inactive expert becoming hot. First use independent
   cold-start runs; persistent warm SRAM/cache state is a separate experiment.
2. Compare an optimized single core, a homogeneous pair and a heterogeneous
   pair. Give all organizations the same DMA/scheduling options and report
   configured resources separately from occupied bytes. Equal MACs/SRAM are
   not a claim of equal physical area or energy.
3. Isolate shape, output interleave, DMA priority and view/transpose changes.
   A corrected timing model must not be ranked against an older model as an
   architectural speedup. Use a one-active-output control inside the new model.
4. Start with all-small groups, mixed groups, one dominant expert and changing
   hot experts. Include shared experts only with their actual semantics and
   dimensions; the first bank version has a common expert hidden dimension.
5. Measure complete operator latency, useful/issued MACs, HBM bytes and channel
   traffic, local-port service, feedback waits and scheduling cost. Overlapping
   waits are not additive components of wall time. Fit any placement heuristic
   on training windows and freeze it before holdout; do not inspect future
   Ramulator completion times when making scheduling decisions.
6. Publish both wins and losses. If improved delivery helps the single core just
   as much, the contribution is not evidence for heterogeneous cores. If no
   reproducible win remains over strong controls, revise the mechanism or frame
   the work as a bounded design study rather than inventing a positive result.

The first full-dimension probe is deliberately a small frozen configuration
comparison, not a complete DSE and not a paper result. Synthetic numerical
weights/inputs with archived routing windows test a MoE operator; they do not
establish end-to-end model throughput, model accuracy, energy or silicon area.
