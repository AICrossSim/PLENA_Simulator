# Spatial-M finite operand fabric v1

Scope: extend the independent spatial-M experiment with finite **operand delivery,
control and accumulator service**. This is a decoded BF16 interface model, not
Ramulator, a native HBM controller, an MX codec or a measured silicon design.
The prior compute-only binary/defaults and archived results remain unchanged.

## Equal aggregate budgets

Use 12,288 multipliers, N lanes=4, K lanes=512 and total M lanes=6. All shapes
receive the same shared weight-source, activation and accumulator byte rates,
control ports, 256 active invocation descriptors, and 2 MiB accumulator capacity.
Default analytical rates: 1,024 / 6,144 / 192 B per cycle respectively; 32 B
transaction rounding. These are sensitivities, not calibrated chip parameters.

Weights reside in finite private **holding slots**: 2*M slots/core, 12 slots
total, each reserving N*K*2=4,096 B (48 KiB total). Slots retain a weight tile
until LRU replacement, can feed successive MAC invocations without refetch,
and cannot be evicted while loading or referenced by a staged invocation.
Activation staging is two full M*K BF16 tiles/core, 12 KiB aggregate. Pending
MAC results reserve ceil(L/II)*M*N*4 B/core, 2,400 B aggregate at L25/II1.
Accumulator bytes include every output in the input window and must fit 2 MiB.
Descriptor count and storage peaks are checked; metadata/control/interconnect
area is not synthesized. Holding slots are an abstract register/banked operand
interface, not a claim that a conventional SRAM provides free wide reads.

## Causal execution

Admission reserves a stage, descriptor, row/K ownership, weight slot and result
credit. Reserving result credit before a bundled invocation prevents circular
wait between a full result queue and an unissued member of the same cohort.
The control issue service must complete before activation/weight requests start.
On a weight miss, source service precedes delivery and control installation;
MAC issue requires that installation, activation transfer and issue service all
complete. A hit reuses the same actual cached BF16 values, not a computed byte
discount. MAC completion precedes shared accumulator RMW service, which precedes
control completion and visible commit. Next K waits for that visible commit.
Result storage and descriptors remain allocated until commit. Data are never
released merely because a future service interval was reserved.

Every port serializes its reservations. All occupancy/ownership still holds in
explicit zero-time oracle runs. Only port timing is bypassed; 32 B byte counts,
logical consumers, cache tags, numerical operations and finite capacities remain.
The online scheduler/cache may make different decisions when timing changes,
so total traffic need not remain equal. These are same-policy counterfactuals,
not frozen-request causal bounds; report traffic changes alongside timing.

## Broadcast, reuse and control

Misses for an identical (expert, N tile, K segment) may
share one source transfer, with one destination per participating core. Each
receiver reserves its own slot. A later miss may join a still-in-flight source
read before it completes; the fanout is fixed at source completion.
Broadcast adds ceil(log2(fanout)) cycles by
default; delivered bytes count every receiver, source bytes count only one
transfer. This represents a fixed up-to-six-way multicast capability shared by
all organizations. Different weights never coalesce. Retention and broadcast
can be disabled independently, for ablation.

Both FIFO and reuse/cohort-affinity selectors share a 32-group ready lookahead,
partitioned in proportion to physical M (floor division, unused remainder not
borrowed). Pinned engines inspect only their assigned experts in that allocation.
Affinity keeps a partially consumed cohort at the front, first prefers a resident tag and otherwise the most
recently selected weight group so multiple engines can share one transfer.
The core visit order rotates. Whole-expert pinning and tile stealing apply to
every architecture. These are explicit bounded heuristics, not optimal scheduling.

Control costs: 2 cycles for issue, 3 for a tile install, 2 for completion.
`invocation` charges per invocation and per installed receiver.
`cohort` bundles same-round invocations with the same weight key into one issue
descriptor; install is one service per actual source transfer and completion is
one service after all members' accumulator services finish. Members keep their
own row ownership and finite result storage until the common commit. Thus cohort
control reduces descriptor work but may delay faster members. Both modes are
measured, so a small-core penalty cannot silently rely on unbatched control.

The stronger `tile_cohort` control creates one persistent descriptor for the
entire (expert,N tile,K segment), including M blocks admitted on later cycles.
Its header is read once; each member then uses the local sequencer. RMW results
become dependency-visible after a one-cycle local update and release their
individual result/activation metadata credit. The descriptor retires with one
global completion service after all Me rows have completed. Thus a Me32 cohort
can exceed the number of simultaneous result entries without deadlocking.
Persistent descriptors and invocation records share the same 256-record budget.
At most half of that pool may hold persistent headers; the remainder preserves
progress credits. Already-open cohorts have a bounded resume bitmap and receive
priority over new admissions, even if their next M block is outside the new-work
lookahead. Without this rule FIFO can fill the pool with unfinished cohorts and
deadlock. An explicit small-credit/long-N regression and the formerly failing B8
request check this rule.
Both older control modes are retained as diagnostics; architecture selection
includes this persistent mode to avoid a weak small-engine baseline.

## Operand interface sensitivity

The default activation/RMW port permits one packet at a time, charging
ceil(bytes/rate) cycles per request. Equal byte rates under this contract do NOT
mean small requests can be packed together. `packed_operand_ports` is a separate,
optimistic banked/gather interface: consecutive requests share a beat using
non-overlapping byte offsets; aggregate byte rate is unchanged. Source weight
reads and control ports retain their original mutual exclusion. The trace records
byte_start/end/rate, and service busy cycles count the union of occupied cycles,
not the sum of overlapping request intervals. Local dependency updates still
serialize at one invocation/core/cycle. Bank conflicts and gather routing are not
modeled, so packed results are an interface sensitivity, not demonstrated PPA.
Both interfaces are compared across all shapes; they must not be mixed in one
architecture speedup ratio.

No cross-expert packing occurs inside one invocation. Different experts may
occupy consecutive pipeline invocations. No arbitrary expert drain is imposed.
No cross-batch repartition costs or RTL/PPA/energy conclusions are modeled.

## Evaluation gate

First test port mutual exclusion, actual gating, finite slot/descriptor lifetimes,
cache reuse, multicast payload identity, K commit order, tails, exact numerics,
and repeat identity. Then test all ordered six-lane partitions used previously,
both policies and control modes. Add FIFO/affinity, broadcast/reuse, bandwidth and
zero-control/zero-delivery oracle ablations on a bounded common set of inputs.
Use B8 for selection and only B32 tokens 8–31 for disjoint-token validation.
These remain one archive family, not independent model coverage.

Report end-to-end cycles, source/delivered weight bytes, activation/RMW bytes,
service reservations, cache/descriptor/stage peaks and per-core exclusive issue
states. Busy service totals may overlap and must not be added into wall time.
Per-core state partitions sum to elapsed cycles; neither they nor resource
service totals alone prove a global memory-bound classification.
