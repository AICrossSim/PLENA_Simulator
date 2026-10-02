# Supply-first v3 implementation contract

The executable architecture selector is `arch=joint_v1|supply_v3`.
`joint_v1` retains the prior complete-expert dispatcher, private weight slots,
landing credit release, staged Gate/Up/SiLU/Down, and ordered combine semantics.
`supply_v3` is an independent finite-resource analytical event model. Its HBM
model is aggregate bandwidth and response latency, not native Ramulator or RTL.
All cycles are 1 ns at the stated 1 GHz clock. Host execution time is separate.

The authoritative supplied specification is
`v3_reference/TASK_SUPPLY_FIRST_V3.md`; reference budgets are first-order
estimates and are kept separate from measured simulator executions.

## Data and arithmetic

Physical array dimensions use M x N x K, with N_tile=4 and K_tile=512.
Weights are stored W[N,K]. Projections are X W_gate^T, X W_up^T, and
Z W_down^T, with low-rank reconstruction X A B. Main MX integer blocks share
an exponent along 32 K elements. A uses MXINT4 by default; MXINT8 split and
BF16 reference formats are explicit. B is BF16 by default. Gate/Up correction
precedes SiLU; U and Z are BF16, partial reductions and combine are FP32.
Rank lane capacity is L times the number of K segments. Tail and streamed-Z
rank-only issues are charged. No unused main MX lane executes BF16 factors
without the optional P1 slack path.

## Finite supply resources

One request controller arbitrates consumers; it can generate multiple 32-B
requests per cycle subject to wire service, credits, DMA readiness and landing
reservation. Responses enter a finite ingress FIFO. Ingress release and
landing release are independently configurable. The byte-addressed pool has
128 banks, 16 B per bank per cycle; reads and writes contend. A reservation
persists until WOR consumes its last referenced bytes. Requests and returns
must fully drain before completion.

Core-local WOR retains a tile across its M blocks; XOR retains activations
across WS column groups or an IS K segment. P1 decodes to BF16 local operand
storage; P2 retains MX format. Operand fills, local reads, accumulator access,
vector operations, combine operations and control service gate execution.
Full and streamed Z have separate schedules and capacity requirements.
Gate/Up prepass FP32 scratch is bounded by a rank group; U_d partials are
stored through charged activation banks. Full-Z Down completes and drains
one bounded N group (up to 32 columns at G=8) at a time. Streamed Down atomically
drains each K-segment and lane-tail group into shared combine; it has no free
private full Y array.
Current and Next share execution only when both actual live accumulator
footprints and separate Z/U addresses fit the frozen physical capacity. Resume
requires the complete next group to be ready and its WOR space reservable.
The default switches when Current cannot progress or the other context reaches
its bounded age; an alternating policy is a separately charged diagnostic.
Suspended contexts cannot retain incomplete future WOR frames. Current's
immediate group has a progress reservation; Next looks ahead only to its next
complete group when another compute context is present. DMA overlap does not
give either context another operand frame.
Quotas are soft targets within the fixed physical pool. If a stage transition
shrinks a quota below Next's already held bytes, Current's immediate required
group can borrow actually unused pool capacity. Future lookahead cannot borrow;
all existing byte leases, bank service and request limits remain enforced.
The bounded admissions are reported as `quota_progress_borrows`.

Pool admission reserves an entire bounded operand group transactionally using
the actual physical free ranges. Speculative groups must leave realizable
placements for selected Current heads, and Next also reserves its finite
accumulator/Z/U context arenas after protecting Current activation. The
one-tile diagnostic preserves a full future Current frame. Missing heads may
resume a fully ready parked context in both pool modes; live WOR frames cannot
be evicted. Offloaded frames wait for helper availability and cannot pin the
helper's independent task. D032 records the post-heldout correctness amendment
and the required complete new-engine rerun.

Current installation and Next promotion require successful transactional
allocation of the actual contiguous accumulator/Z/U ranges. Aggregate free
bytes do not prove physical eligibility. A bound context that cannot acquire
its ranges stays queued and cannot exclude helper service by occupying Current.
No live context is compacted or moved to satisfy admission. D034 records the
fragmented-context offload failure and the required new-engine validation.

BF16 U is stored in a fixed K-segment rank permutation and is read through the
physical activation banks. Two bounded local rank-input SRAM cache entries per
core use the existing control reserve; external reads, local reads and fills
have finite ports. Rank operand lookup is not an uncharged host-array operation.
SiLU, BF16 U stores, U_d prepass deltas, Down combine and helper returns read
their RF/SRAM source through the same finite private accumulator port used by
RMW, after preceding commits. Helper outputs lease the helper's existing arena
and pay the helper source read, cross-core copy and owner RMW. Its fixed width is
max(128,32*M_c) B/cycle. Non-inline WS additionally writes and reads a bounded
FP32 group scratch slice in the existing private arena. Non-inline IS reads
the existing Gate/Up FP32 backing; no extra whole-expert backing is assumed.
The helper's RF lease ends only after the corresponding U store, U_d backing
update or returned delta has finished all real source/destination transfers.
It does not remain pinned through unrelated main GEMMs; a whole-arena IS expert
can then run before the owner's next offloaded group without circular waiting.

## Runtime and fairness

Every policy sees Shared-first, otherwise expert-ID ordered descriptors in a
bounded eight-entry window. Current/Next lookahead, quota constraints and
ownership apply to all policies. `supply_ipd` adds shape affinity, calibrated
remaining-service estimates, a tail guard and charged recovery. Predictions
use completed past work only. Shared splitting is an optional charged
diagnostic, not the default. One layer's gate-weighted rank allocation uses
development-calibrated tail-error tables and a bounded lambda controller.
The Joint baseline retains critical-anchor pairing and aging; it is not an
earliest-finish alias. Charged paired proposals persist until installed or
legally cancelled. Hardware rank energies are BF16 RNE per projection, costs
are actual uint32 factor bytes, and lambda fits and feeds back routed bytes
separately from Shared's fixed rank cost.

The quota-off ablation retains only Current's immediate group and prohibits
Next DMA; the pipeline-off ablation waits the previous MAC/accumulator commit
before subsequent operand reads or issues. Both use finite physical resources,
with no injected delay standing in for the disabled mechanism.

Main multiplier budgets are 12,288 (M6) or 16,384 (M8). Rank BF16 multipliers,
register bytes, SRAM bytes and port widths are separately reported. The total
storage ceiling is 2,158,592 B. Equal-port comparisons use total pool read
1,024 B/cycle, X read 640 B/cycle, equal WOR capacity and decoder throughput.
Demand-sized comparisons report their differing ports as costs.
Accumulator, rank-input and auxiliary port widths are also reported and charged
in the area proxy; the stated equal-port group refers to the specified pool,
X, WOR and decoder budgets, not unreported identical ports everywhere.
The canonical on-chip movement total includes every measured storage endpoint:
pool and ingress reads/writes, WOR/XOR fills and array broadcast reads, rank
cache and external U transfers, private source/RMW access, Z writes, combine
RMW and explicit cross-core copies. The category sum is checked exactly.
The earlier incomplete subset is retained only as a diagnostic. Internal
individual-PE wiring is outside this endpoint metric.

## Correctness and evaluation

Timing correctness includes unique ownership, ascending K accumulation,
rank capacity, finite storage, mutually exclusive state counters, no implicit
weight refetch, and request drain. Seeded payload replay uses timed issue
order and compares with `expert_ffn_hw` and actual-order `combine_hw` or
`combine_partial_hw` (relative error
1e-5), then FP64 quantized arithmetic (5e-3). Accuracy of quantization is
evaluated separately on hash-verified pretrained weights and routed inputs.

Development data and held-out request IDs are disjoint. PREREG_V3 is committed
before held-out timing. Every legal final point is run twice and raw JSON
equality is required. Missing real captures are reported as missing; constructed
workloads never replace real workloads without an explicit label. N1–N5 use
the supplied fixed thresholds, including negative outcomes.

The original legacy full-window output-retention layout cannot fit the declared
T64/T96 mixed windows in its frozen private SRAM. D033 introduces a separately
labeled finite outer token-chunk capacity extension for those formerly
unsupported cases. Full-layer X/final Y/routes stay resident; unchanged legacy
kernels run serially on the largest fitting chunks, and all repeated HBM reads
and descriptor setup are charged. Supported old plans and their timing remain
unchanged. Large-window comparison tables explicitly report this extension;
no point is silently removed, given extra storage, or described as an original
full-window legacy measurement.
