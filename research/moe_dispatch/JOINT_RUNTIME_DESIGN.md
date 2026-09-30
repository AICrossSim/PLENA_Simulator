# Bounded reservation-aware pair runtime — implementation contract

Status: research candidate; performance and novelty are hypotheses until tested.
Scope: Compiler address/capacity plans and the existing Rust analytical FFN
engine. No RTL, native Ramulator, routing computation, or full-model timing.

## Question and mechanism

Can delaying irreversible assignment until prefetch is useful, while selecting
complementary tasks from a finite visible window, recover opportunities lost by
head-only, eager Current/Next dispatch on a fixed heterogeneous accelerator?

The compiler supplies expert shapes, weight addresses, route references and
per-core storage plans. A runtime descriptor is NOT a weight payload. Router
outputs are already known; no expert ID is predicted and no selected route is
dropped. Hardware remains fixed across workloads.

The common descriptor stream provides Shared first, followed by routed experts
in expert-ID order. This is legal at the modeled FFN boundary where input X and
routes are available. Every comparison receives the SAME stream. A separate
Shared-last development ablation measures the input-order artifact. There is
no free full-layer scan by the runtime.

    Routed descriptors (Me,H,F,addresses,input references)
                           |
                     finite window <=8
                           |
             charged snapshot + resource legality masks
                           |
               critical anchor + companion comparison
                           |
              predicted prefetch-lead admission gate
                           |
            revalidate / commit up to two distinct owners
                    /                      \
             Current + Next          Current + Next
             private W/X/acc          private W/X/acc
                    \                      /
                 existing shared credit/stock DMA
                           |
                        shared HBM

Data direction is HBM -> reserved private W -> operand feed -> MAC -> private
accumulator -> existing vector and output-copy path. The drawing above shows
control connectivity, not that weight data travels through the dispatcher.

## Concrete interface and datapath

Dispatch granularity in this study is **one complete expert FFN**. It is not an
individual token, a router top-k selection, or an independently migrated K
segment. The expert computes Gate and Up, forms the nonlinear intermediate,
then computes Down. The local engine tiles each projection; the dispatcher
does not place each matrix tile independently.

Physical dimensions are always written **M x N x K**, with N=4 and K=512.
M6 comparisons have 12,288 multipliers; input/weight are BF16 and accumulation
is FP32. No new quantization mechanism is introduced.

| M6 organization | Core shapes | Private W slots | X double buffers | Total arena |
|---|---|---|---|---|
| Single | 6x4x512 | 10 | 12 KiB | 2 MiB |
| Homogeneous | 3x4x512 + 3x4x512 | 5 + 5 | 6 + 6 KiB | 1 + 1 MiB |
| Heterogeneous | 4x4x512 + 2x4x512 | 5 + 5 | 8 + 4 KiB | M-proportional, 2 MiB total |

One full W tile is 4x512x2 = 4,096 B, independent of core M. A full X tile is
M x512x2 B; each core has two such buffers. The W budget also includes an 8 KiB
shared return region, giving 48 KiB total. The arena includes intermediates,
retained outputs and the control reserve; it is not all available for live
partial sums. Capacity and bank/port counts remain distinct constraints.

The Compiler emits shapes, route-row references, tensor addresses/strides,
allocation bounds and future-workspace fit information. It does not select
experts or hard-code task owners from the evaluation trace. Runtime uses these
descriptors and actual resource state. The task lifetime is:

    Waiting: no owner, no reserved W slot
       -> Proposed: charged decision snapshot, still no ownership
       -> Committed Next: immutable owner + one reserved W landing slot
       -> Current: acquire workspace, stream tiled Gate/Up/Down computation
       -> Finished: retain required output for the existing combine operation

One Current and at most one Next are allowed per core. Binding is distinct
from issue: execution still requires actual W/X readiness, output capacity and
the previous K dependency. The no-prefetch ablation keeps the reservation
protocol but does not issue Next DMA before promotion. A forecast never
substitutes for a real completion event.

## Bounded hardware policy

1. Accept at most one descriptor per cycle. Initially allow the finite window
   to fill, or stop at the known stream end. Do not inspect unseen descriptors.
2. Snapshot potential owners separately from commit-ready owners on the
   mutually exclusive controller port. Potential ownership checks vacant Next
   and the compiler's complete future-workspace fit. Commit additionally needs
   a private weight landing slot and enough prefetch lead. Current and Next do not acquire
   overlapping expert workspaces: Next acquires its workspace at promotion.
3. Among tasks with potential owners, select an aged task after eight bypass rounds;
   otherwise choose the task with the largest minimum estimated standalone
   service. This critical anchor prevents pure shortest-job pair scoring from
   always postponing Shared or hot experts. This is a heuristic, not optimality.
4. Compare anchor/companion orientations across the visible window, prioritizing
   two proposed placements when two distinct tasks can eventually use the cores.
   Minimize predicted pair completion, then prefer admitting more estimated
   work, then smaller finish sum. Commit only the due subset of that proposal:
   a hot anchor can wait unbound for the busy large core while a cold companion
   starts on the small core. Removing the busy core before comparing assignments
   would incorrectly force the anchor onto the small core. Comparing a singleton's completion directly
   with a two-task completion would compare different work and is prohibited.
5. With late binding enabled, a busy core admits Next only near the estimated
   first-tile fetch lead plus decision service and a fixed 64-cycle margin.
   Idle cores remain admissible. This estimate never overrides actual readiness.
   An inaccurate estimate can stall performance but cannot authorize execution.
6. Pay scan/comparison/commit service BEFORE state changes. Freeze selected
   identities during service; recheck legality at completion. Canceled proposals
   issue no DMA. Commit ownership and one landing slot atomically, then stream through the existing
   Next protocol. No migration, cancellation of accepted DMA, or free refill.
7. Existing per-core four-Me-bin EWMA adjusts observed service costs only.
   No unseen-core labels, future event times, or heldout-trained constants.
   Feedback resets at each independent window; this is within-window learning.

The choice policy leaves per-output ascending K accumulation and route-order
combination unchanged. It neither co-packs rows with different expert weights
into a single GEMM nor assumes small experts share weights.

## Storage and service ledger

The additional logical state reservation is 256 B: at most eight 24-B snapshot
entries plus 64 B of scan/winner registers. Service/finish estimates use bounded
32-bit representations; saturation must be observable. Age/bookkeeping reuses
fields in the existing 64-B task descriptors. Diagnostic histories are observer
output, never hardware inputs. Existing service-feedback state remains 96 B.

One 24-B entry packs task ID (32 bits), potential/due masks and age (32 bits
including padding), and four 32-bit service/finish values. A feasible 64-B
header packs a 64-bit start time; one 64-bit ready/retry time (mutually exclusive
states); two 32-bit remaining-work snapshots; 32-bit input generation plus
mask/count/anchor/flags; proposed indices and a committed-subset mask; scan
cursors/comparison/service counts; and 32/33/33-bit comparison scores. Comparison
temporaries are reused between scan and commit. The Rust host's `Vec`/`usize`
objects and observer audit are not the hardware layout. This is a logical
register budget, not a physical area or timing closure result.

The old 4,096-B control reserve becomes 4,352 B **inside the existing 2 MiB
accumulator/intermediate/output arena**, reducing payload capacity by 256 B.
All policies, including baselines, reserve the same additional bytes. Aggregate
arena capacity and SRAM ports do not increase. Compiler reruns concrete address
allocation, per-core fit checks and headroom accounting.

The Compiler's `ownership_policy` metadata describes the joint capability when
`joint_state_bytes=256`. Comparison policies reserve the same bytes but execute
their own configured protocol. In particular, legacy FIFO/dynamic binding does
not acquire a W slot atomically just because this metadata field is present;
the runtime policy and its event trace determine actual behavior.

At minimum charge scan 4+4*visible_tasks*cores cycles, two cycles per compared
orientation, and four per committed task. Existing promotion, feedback, DMA,
SRAM banking and operand service remain charged. No single-cycle or area/power
claim is made without a synthesized implementation.

## Frozen comparisons

Keep the six previously selected physical/G configurations: M6 [6],[3,3],[4,2]
and M8 [8],[4,4],[5,3]. Compare only within each MAC budget. All use the same
mechanism, controller reserve, precision, bandwidth, credits and response time.
Primary policy comparisons use whole experts, with tail_partition=false for all;
report prior selected tail behavior separately rather than hiding that baseline.

Compare FIFO, dynamic earliest-finish, feedback earliest-finish and joint.
Ablate joint late binding, pair selection, feedback and successor prefetch
individually. Baselines receive the same prefetch and stock arbitration except
the explicitly named no-prefetch ablation. Development uses only prior design
requests. Fresh evaluation uses twelve disjoint windows (three datasets x
B2/B4/B8/B16), layer13 step7, excluding every previous study request. Select
windows by deterministic request hash before timing, never by hotness or result.

Each accepted timing point runs twice with identical complete raw JSON. Check
weight bytes, useful MACs, finite capacity, K ordering, unique ownership and
request drain. Replay small seeded numerical tensors through the ACTUAL timed
DMA/issue trace; real-sized timing uses captured routes and shapes, not trained
model weight/activation tensors.

Report latency, same-policy organization comparisons, matched-hardware policy
speedups, legal-core choices, reorder/aging events, charged scan cost, binding
prediction error, prefetch readiness, SRAM peaks and supply lower bounds. Never
sum overlapping wait counters into wall time. With 256 32-B credits and 64-ns
minimum residency, sustained supply cannot exceed 128 B/ns before landing:
the controller cannot manufacture bandwidth or reduce required weight bytes.

`next_prefetch_tiles` is the historical name for reserved Next first tiles;
joint reserves them even in the no-prefetch ablation. It is not a count of DMA
requests issued before promotion. Use ready/inflight-at-promotion observations
and the actual DMA trace when interpreting prefetch overlap.

## Research claim boundary

Hot/cold heterogeneous expert placement, hardware earliest-finish scheduling,
finite-window scheduling and adaptive prefetching have prior art. The candidate
claim is the measured tradeoff between retaining assignment choice and committing
finite private storage early enough to hide supply latency. An incremental
combination is not automatically a publishable novelty. Test whether the
combined mechanism gives an independent benefit over strong common baselines;
report failures and bottleneck bounds without changing hardware per test.
