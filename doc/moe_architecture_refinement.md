# MoE architecture refinement: compute, SRAM, DMA and dispatch

Status: **design proposal, not implemented or performance-validated** (2026-09-07).
The runnable reference is Simulator `bead42b2` and Compiler `c45cf7b`.
The [existing review guide](moe_normal_review.md) describes that implementation.
This document defines the missing contracts and the order for implementing them
in Compiler and Simulator. Passing a numerical smoke test does not freeze an
architecture, its port structure, or its arithmetic pipeline.

## Goal and evidence boundary

Determine whether heterogeneous compute shapes improve fixed-route MoE execution
when task partitioning, operand delivery, accumulation and shared memory service
are designed together. Hold workload, numerical contract and declared resources
constant; measure complete operator latency, not only occupied multipliers.

The existing negative result applies to **whole-expert dispatch, square tiles,
fixed activation allocation and the current analytical timing rules**. It does
not test arbitrary splitting of one expert across cores. The performance cases
cover routed-expert gate/up/down; the independent shared-expert path is exercised
by functional tests. Initial input HBM transfers, runtime router execution and
output HBM stores are excluded from the present timing boundary and must remain
labelled as such. Input gather and activation supply are included.

## 1. Separate the three meanings of matrix size

| Meaning | Contract |
|---|---|
| Workload | `X[Mw,Kw] * W[Nw,Kw]^T -> Y[Mw,Nw]`; Mw is the number of routed token rows for this expert. |
| Scheduled tile | Explicit ranges `(m0, mt), (n0, nt), (k0, kt)`, with valid tails. |
| Compute organization | Modeled output lanes `P` and reduction width `R`, with `P*R` multipliers. Token rows are serviced over time. |

Today nominal tile capacities are `Mt=Nt=P=BLEN` and `Kt=R=MLEN`; valid extents
`mt,nt,kt` may be smaller at tails. These equalities are a mapping restriction,
not the definition of MoE or proof that a three-dimensional multiplier array
exists. Gate/up use `(Mw,Nw,Kw)=(Me,F,D)`; down uses `(Me,D,F)`.
Compiler descriptors must retain full dimensions and ranges independently of
which core is eventually chosen.

## 2. Make task partitioning an explicit experiment axis

| Mode | Ownership and additional costs |
|---|---|
| Whole expert — implemented | One core owns an expert through gather, gate/up, SwiGLU, down and output copy. Dispatch may choose a different core for the next expert. |
| M partition — proposed later | Different token rows go to different cores. Outputs remain independent, but repeated weight reads, activation copies and lost weight reuse must be counted. |
| N partition — proposed later | Different output columns of a projection go to different cores. Gate/up partitions must align for SwiGLU; down needs the complete required Z features, so gathering/replication and barriers must be represented. |
| K partition — outside the next experiment | Requires partial-sum communication and a separately specified floating-point reduction order. It cannot reuse the current exact oracle without justification. |

First refine the whole-expert reference so memory/scheduling changes can be
isolated. A claim about intra-expert splitting requires a separate executable
graph with the transfers above. A task is not migrated after execution starts
in the next implementation; pre-dispatch placement is a different decision.

## 3. Give Compiler and runtime different responsibilities

Compiler supplies validated tensor regions, token/route order, GEMM ranges,
element/scale format and logical producer-consumer dependencies. For a tile:

```
element address = element_base + output_row * element_stride + k0
scale address   = scale_base   + output_row * scale_stride   + k0 / block_size
```

The next contract keeps `k0` block-aligned and charges tail sectors explicitly.
Runtime selects a core, reserves physical storage, creates the concrete request
fragments and assigns destination addresses. The HBM command scheduler sees
memory requests; the task dispatcher decides which core executes an expert.
HBM request priority must not silently change that ownership.

## 4. Specify storage lifetime before scheduling it

A resident weight tile has the following proposed observable states:

```
FREE -> RESERVED -> FETCHING -> PACKED_READY -> DECODE_QUEUED
     -> READY -> CONSUMING -> FREE
```

- Reserve the complete slot before issuing a request. Track element and scale
  completion separately; both must cover the same valid reduction range.
- Decode only after the required packed bytes have been copied. Compute requires
  decoded readiness, activation readiness and an available output context.
- A READY tile still owns its slot. Release it after the last admitted consumer
  finishes reading/capturing the weight operands. Independently reserved result
  and accumulator state may remain in flight until result completion. No silent
  eviction, free reload or persistent weight cache.
- A destination is `(core, slot, packed-or-BF16, row, column)`, not just an HBM
  source address. Copies to two destinations are charged twice.

Existing weight SRAM budgets are 64 KiB single, or 48/16 KiB large/small.
Four slots actually occupy 50 KiB single, or 37.5/12.5 KiB large/small:
`slot_bytes = BLEN * (MLEN/8) * 25` includes packed E4M3, E8M0 and decoded BF16.
Configured capacity and occupied bytes must be reported separately.

The complete expert accumulator data is already reserved. Introduce a bounded
`OutputContext(job, projection, m0, n0, next_k, ready_time, state)` table for active
outputs; explicitly charge its encoded bytes against the accumulator/control
budget. The current host `ready[Mb*Nb]` vector is not a complete metadata budget.
The first interleave experiment admits at most two N tiles per core; the context
count is bounded by those tiles and the admitted token blocks. Reject an expert
if data, pipeline storage and metadata together exceed capacity.

## 5. Close the operand-port and accumulation contracts

These are blocking architecture decisions, not parameters to hide behind a
successful functional test:

| Path | Existing model | Required refinement |
|---|---|---|
| Activation -> compute | Aggregate per-core supply; single 1024 versus 768+256 BF16 elements/cycle. | Declare local bank/read limits and the destination interface. Fixed allocations remain a control; dynamic borrowing needs a real shared arbiter and transport budget. |
| Decoded weight -> compute | Values are directly consumed from a host vector. | Choose SRAM streaming or stationary operand latches; declare preload/read bandwidth, latch storage if present, and contention with decode writes. |
| Returned bytes -> SRAM | Copy bank selected by `(HBM source address / 64) % 8`. | Define destination bank mapping and ports; source-address striping is not a destination-SRAM conflict model. |
| Accumulator | Full data capacity plus analytical result-ready timestamps. | Explicit read/write service, result-queue capacity, forwarding rules and same-output feedback latency. |

Do not add operand latches or crossbars without charging their storage and service.
If an access model cannot feed the claimed multiplier rate, report its sustainable
rate instead. The current small core takes 4 service cycles rather than its ideal
2 under `max(BLEN,ceil(BLEN*MLEN/activation_supply))`; changing its supply is a
named design change, not an HBM optimization.

The current oracle performs ascending-global-K FP32 multiply/add, whereas the
timing model includes a `ceil(log2 MLEN)` term. That does not define the numerical
behavior of a parallel reduction tree. Preserve current arithmetic and feedback
latency for the first scheduling comparison. A chosen reduction structure must
subsequently specify rounding/order and get its own independent numerical check;
never shorten feedback latency or relax accuracy solely to obtain a speedup.

## 6. Schedule independent outputs without breaking dependencies

First compare the current N->K->M order with a two-N-tile active window. Only
issue a context when its weight/activation data are ready, its next K range is
correct, its accumulator feedback is ready, and all operand/result resources
have been reserved. Select the oldest legal context with deterministic ties.

The four weight slots can hold independent N tiles rather than only successive K
tiles of one output. Preserve every scalar output's ascending K order and the
current full feedback delay in this comparison. When no context is legal, wait
for a real readiness event. Do not manufacture overlap by subtracting stalls.
Reserve/admit each active context's earliest unfinished K fragment before
speculative later-K fragments. Never fill every slot with future fragments while
the predecessor needed for progress has neither a resident nor a reserved slot.
Ready-context selection alone does not guarantee this progress invariant.

## 7. Make the DMA frontend aware of tile demand

The existing per-channel mutex removes cross-channel blocking, but holds its
position across native rejection retries. Element/scale fragments are initially
created in address order. This is not progress-aware arbitration.

Proposed frontend rules:

1. Keep non-issued fragments in already-budgeted resident-tile descriptors.
   Each has core/tile identity, element-or-scale tag, destination and age.
2. Distinguish current demand from future prefetch. A fragment that completes a
   demanded tile is useful whether it contains elements or scales; all scales
   do not receive unconditional priority. An unissued coalesced line/sector
   inherits the strongest current consumer demand and relevant age, so demand
   merged into prefetch cannot remain at prefetch priority. Destination copies
   remain separately charged; accepted requests are not preempted.
3. At each legal channel issue opportunity, choose among eligible fragments.
   Demand wins over ordinary prefetch; use per-core byte fairness and bounded
   age promotion to prevent starvation. Expose the age threshold as a parameter.
4. A rejected attempt remains queued and can be reconsidered at the next legal
   opportunity. Accepted native requests are neither cancelled nor preempted.
5. Reserve response/destination space before submission. Drain returns without
   requiring a new request credit, so request pressure cannot deadlock response
   completion. Release each waiter/line/slot at its own defined lifetime boundary.

Retain 128 logical line/waiter credits, 256 native 32-byte trackers and 44 KiB
total frontend storage, including staging. Coalescing remains in-flight only.
Current accounting reserves 32,512 B single or 32,832 B dual within that frontend
budget. Allocate and validate new queue heads, age fields and completion bitmaps
within the remaining capacity; do not use unbounded host queues as free hardware.
Keep the native HBM configuration and command scheduler fixed for this control.

## 8. Refine core assignment only after the estimator is observable

Keep the M-threshold/work-conserving policy as a baseline. A proposed alternative
compares estimated finish times using known shapes, tile counts/tails, current
core work, resident data, channel queues and shared vector service. It must use
only current state and completed-service statistics, never future native return
times or the measured answer of that same candidate run.

Bound candidate scanning and account for scheduler service and descriptor space.
Record both candidate estimates, the choice and actual completion. Freeze any
fitted parameters before held-out windows. Treat this as a heuristic whose error
is measured, not an optimal scheduler. Do not change dispatch, DMA arbitration
and memory size simultaneously and attribute the result to one component.

## 9. Instrument the evidence needed to distinguish causes

Record tile milestones: slot reservation, last element copy, last scale copy,
decode queue/start/end, first compute, last consumer and release. Record request
credit wait, lookup, native admission, response and destination-copy service.
Separate vector service into gather/decode/SwiGLU/combine. Track per-core/channel
demand and prefetch bytes, queue peaks and oldest request age.

Trace collection may stream to a bounded host sink; state used to make scheduling
decisions must remain inside declared modeled storage. Use per-core timelines
and final completion time: overlapping waits cannot be summed into latency.
The current `weight_ready_wait_ps` alone cannot identify a bad HBM controller.

## 10. Implementation and acceptance order

1. Add observation-only milestones and audit storage accounting; reproduce the
   present configurations without changing issue order or latency rules. Retain
   this version as the legacy timing control.
2. Instantiate the operand/accumulator ports and bounded result contexts from
   section 5 as a separately labelled model. Check insufficient-port cases;
   added service constraints may change latency. Do not mix that correction with
   a scheduling speedup or require it to reproduce the legacy timing.
3. Implement bounded independent-output scheduling, preserving existing feedback
   and arithmetic. Test one-context and multiple-context cases to distinguish
   unavoidable dependencies from waiting that can legally overlap.
4. Implement demand-aware DMA separately; test delayed scale, same-channel
   prefetch blocking, asymmetric consumers, full queues and starvation bounds.
5. Compare dispatch policies and activation allocations only after their inputs
   and service constraints are explicit. Then consider M/N task partitions.
6. Search SRAM sizes and core shapes under the resulting contract, with the same
   policy options available to single and dual cores. Transpose/Attention uses a
   later access contract; adding it does not resolve the normal-path gaps above.

For every stage: exact current-oracle checks, tails, deterministic repetition,
byte conservation and finite-capacity/backpressure assertions precede latency
ranking. Include all-small-M, mixed/hot experts, one dominant expert, demand
blocked by prefetch, and insufficient-port cases. Publish both wins and losses.
Make compact routing inputs and the fixture-generation recipe available before
claiming external reproduction of the complete performance table.
