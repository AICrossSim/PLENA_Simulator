# Batch and real-operand evaluation contract — 2026-09-21

This study continues the **physical spatial-M experiment** frozen on September
19. It does not reinterpret the earlier temporal-M/Q32/Step2 native-engine
measurements. All performance numbers are Rust simulation cycles under explicit
timing assumptions, not GPU, RTL, silicon, native HBM, or whole-model latency.

## Architecture and timing

Hardware notation is **M lanes × N lanes × K reduction lanes**. The five
organizations are `[6]`, `[3,3]`, `[4,2]`, `[2,2,2]`, and `[1,1,1,1,1,1]`, with
N=4, K=512 throughout. Each has 12,288 parallel multipliers. This is equal
arithmetic capacity, **not demonstrated equal area, power, or achievable clock**.

Default result latency is 25 cycles and issue interval is 1 cycle. The pure
compute model removes data/control transport and retains finite pipeline
capacity, actual K-dependency ordering, final drain, and M/N/K tail masking.
The finite model additionally charges the decoded operand interfaces:

| Shared resource | Default |
|---|---:|
| Weight-source delivery | 1,024 B/cycle |
| Activation source | 6,144 B/cycle |
| Accumulator RMW port | 192 B/cycle |
| Control ports | 1 |
| Descriptor issue / weight installation / retirement | 2 / 3 / 2 cycles |
| Descriptor records / ready window | 256 / 32 |
| Private weight storage, aggregate | 48 KiB |
| Private activation staging, aggregate | 12 KiB |
| Pipeline result storage, aggregate | 2,400 B |
| Output backing accumulator | 2 MiB |

Persistent `tile_cohort`, affinity selection, weight retention and multicast are
used in every primary finite comparison. Every positive data request is rounded
to 32 B. This source transports **decoded BF16**, not compressed MX element/scale
streams; no new native HBM controller, Ramulator model, or codec is asserted.
Each invocation also pays the existing one-cycle local commit sequencer. This
remains charged in zero-global-control oracles. Because N/K lanes are identical
across organizations, one expert's unique weight tile count is identical; this
does not inherit the old study's differently sized N/K tiles.

`pinned_expert` assigns each expert to one core with the existing deterministic
work estimate. `tile_stealing` allows ready M-row tiles to migrate across cores,
preserving each output row's K dependency and common accumulator backing.
Rows using different expert weights cannot share one matrix invocation. Both
policies are reported, including strong uniform multi-core comparisons. No
per-window fastest-policy selection is presented as an implemented scheduler.

## Inputs and coverage

`inputs/archive_inventory.json` hashes all nine original route archives:
Qwen3.5-35B-A3B-FP8, DeepSeek-V2-Lite-Chat and Nemotron3-Nano-30B-A3B, each with
BFCL/GPQA/SWE captures. All **29,495,540 valid routed pairs** were inventoried.
`population_by_batch.csv` profiles the entire valid archive for batch 2/4/8/16,
including all layers/steps, and explicitly accounts for incomplete batch tails.
This population analysis is not a full-archive latency simulation.

Timing windows are selected **before looking at results**. For each archive:

* Primary: first 16 valid streams at decode step 0, first MoE layer.
* Holdout: next 16 valid streams at decode step 8, middle MoE layer.
* Each group supplies nested prefixes of 2, 4, 8 and 16 streams.

Thus 72 timing windows are simulated. Stream/sample/layer IDs, route weights,
original capture batch, valid masks, and NPZ SHA256 are recorded. These are
rebatches of actual decisions, not new model runs at each batch size. Timing
simulations use **full original matrix dimensions**, not the previous uniform
512×2048 surrogate for all models:

| Model | Hidden H | Routed intermediate D | Shared intermediate Ds | Routed top-k |
|---|---:|---:|---:|---:|
| Qwen3.5 | 2,048 | 512 | 512 | 8 |
| DeepSeek-V2-Lite | 2,048 | 1,408 | 2,816 | 6 |
| Nemotron3-Nano | 2,688 | 1,856 | 3,712 | 6 |

For an expert receiving Me tokens, gate/up are `(Me, D, H)` and down is
`(Me, H, D)` in workload M/N/K notation. Shared GEMMs use M=batch and D=Ds.
Gate and up have identical timing-only shapes and are measured once for the
route study; their duration is counted **twice** in serial phase sums. Shared
shapes, independent of routing, are measured once per model/batch/configuration.

The NPZ archives do not contain activation or weight values. Their numerical
verification flag is therefore **off**, explicitly. They are not represented as
whole-model or actual-weight numerical runs.

## Real-value connected MoE test

`real_inputs/manifest.json` records a separate actual-value experiment using
the local original DeepSeek-V2-Lite-Chat BF16 checkpoint. The first 16 examples
of pinned BFCL v3 `simple` data provide full prompts plus tool schemas. No
prompt is truncated. The official installed Transformers DeepSeek implementation
executes embedding, complete decoder 0, and decoder 1 attention/post-attention
normalization on CPU. The last prompt token's activation is captured, and the
original layer-1 router computes its six expert IDs and FP32 route weights.
This is **last-token prefill**, not the older archived decode capture or a full
autoregressive trajectory. Dataset URL/commit, token IDs, model tensor hashes,
versions, and selected weights are preserved.

For each batch 2/4/8/16 and all five organizations × two dispatch policies:

1. Rust loads the **actual BF16** X and gate/up weights with SHA256 and exact
   shape checks, consumes weights through the finite weight slots, and executes
   all full-size GEMMs.
2. Host PyTorch executes BF16 SiLU and multiply on Rust's outputs.
3. Rust executes actual down weights with those produced intermediates.
4. Shared experts follow the same three Rust GEMMs and nonlinear operations.
5. Original top-k slot order is restored; FP32 route weighting/reduction and
   BF16 rounding/addition produce the final MoE output.

There are six individually timed GEMM phases per configuration. Their serial
sum is explicitly named **GEMM phase sum**, not MoE end-to-end latency. Routing,
nonlinear operations, scatter/combine, across-phase state residency and
concurrent shared/routed phase scheduling have no complete hardware timing
model here. Numerical completion of the MoE layer does not imply those timing
gaps are implemented. No model accuracy or task success claim is made.

Actual X/W loading is opt-in (`--operands`). The manifest fixes X=[M,K] and
W=[N,K], row-major little-endian BF16. Wrong lengths, hashes, expert ordering,
nonfinite operands, unknown schema, disabled verification, and input-overwriting
output paths are rejected. Defaults and the frozen binary remain unchanged.

## Numerical and timing validation

* Each performance/value point is repeated twice and full output hashes must
  match exactly. The additional frozen timing replay is a one-run regression.
* Rust's independent recursive FP32 reduction reference must match its iterative
  K512 datapath bit-for-bit. K tiles accumulate in ascending order. This is an
  explicit arithmetic contract; arbitrary PyTorch reduction order need not be
  bit-exact to it.
* Independent FP64 matrix products check each actual-value projection. The
  maximum error divided by sum(abs(X·W)) must be ≤5e-6 (predeclared).
* Final connected MoE relative L2 error versus the independent PyTorch model
  equation must be ≤1% (predeclared), with measured error and exact-element
  fraction reported. All five architectures and both policies must produce
  identical final BF16 and intermediate FP32 bits for a given batch.
* All actual-value phase event hashes must match a replay on the frozen finite
  binary with values disabled. Functional support must not silently alter timing.
* Drains, descriptor/weight/activation/result peaks, MAC coverage, source/delivery
  bytes and control service formulas are audited. Full traces independently
  verify row/K coverage, dependencies, slot ownership and port reservations.

## Sensitivities and interpretation

The main route scan includes control/weight 2×2 timing oracles, four separate
finite resource improvements, and both dispatch policies. The additional scan
uses weight rates 256/512/2048/4096 B/cycle and pipeline points L1/II1 and
L25/II25 alongside default L25/II1. These pipeline points are **sensitivity
assumptions**, not measured implementation feasibility. Result-register storage
changes with ceil(L/II) and is reported.

The activation/RMW 4× byte-rate probes are timing-inert for these tile sizes:
each request was already charged the minimum one cycle. They therefore do NOT
exclude transaction-rate or latency bottlenecks. `operand_ports/` adds 160×2
actual-value gate runs covering all batches/organizations/policies: zero X
timing, zero RMW timing, both, and optimistic sub-beat request packing at the
same aggregate byte rates. Their actual FP32/BF16 outputs must match charged.
Non-monotonic oracle points are retained, not treated as strict upper bounds.

Oracles remove only named timing charges; slots, data values, dependencies,
ownership rules and byte counters remain. Online ordering/cache hits can change
as events move, so byte/invocation deltas are reported. These are same-workload
policy counterfactuals, not fixed-trace causal bounds or realizable designs.

`breakdown/` reconstructs primary BFCL gate/up traces for all models/batches/
organizations/policies. It splits the existing weight-wait category into source
queue/transfer, distribution, and control installation queue/service. The
mutually exclusive issue-state categories sum to wall cycles and reproduce the
original counters. MAC pipeline occupancy and port-service occupancy overlap
these categories and **must not be added to them**. Neither an idle issue actor
nor a nonempty pipeline alone establishes compute-bound or memory-bound.

`real_breakdown/` additionally reconstructs B8/B16 fixed-ownership actual-input
cases on `[6]`, `[3,3]`, `[4,2]`. Their event hashes match the corresponding
numeric run. Gate and up share an identical timing trace (explicitly checked),
so gate has multiplicity two in six-phase state summaries. Each phase's
last-finishing core may differ; such a summed row is not one core's timeline
and remains a serial-phase diagnostic, not full-layer critical-path attribution.

Performance conclusions must include losing cases. The candidate set is five
organizations, not exhaustive M/N/K DSE. Primary-window selection, if used, is
frozen on disjoint holdout windows. Per-window minima are labeled optimistic
configuration selection. No result can establish asymmetric cores' necessity
without beating the strong uniform and scheduling alternatives under supported
resource/timing assumptions.
