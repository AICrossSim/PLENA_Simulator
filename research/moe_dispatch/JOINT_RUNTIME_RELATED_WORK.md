# Bounded joint MoE runtime: literature and independent design review

Date: 2026-09-30. This is a research/design review, not a performance result or a first-of-kind claim. Sources below are primary papers or official implementation documentation. Existing analytical results remain frozen; this document does not modify paper prose or implementation.

## Main finding

The defensible research question is narrower than “heterogeneous cores need a scheduler”: **can a finite hardware task window retain useful assignment choices until private-SRAM/DMA ownership must become irreversible, without losing the lead time required to prefetch?** Joint selection, reservation, and commitment can address a measured failure of the existing Current/Next contract. A better estimator alone cannot help when only one core is legally selectable.

This is a candidate contribution to evaluate, not a guaranteed acceleration or proof that heterogeneous shapes are necessary.

## Primary-source comparison

| Source | Verified mechanism | What transfers to this project | Novelty boundary / limitation |
|---|---|---|---|
| [Duplex, MICRO 2024, sections V-B and V-C](https://arxiv.org/html/2409.01141v1) | Uses token-count-dependent xPU/Logic-PIM processing-time tables, moves fewer-token experts progressively to Logic-PIM, and manages memory regions to limit bank conflicts. | Evaluate actual service costs on both engines; preserve same-expert token batching; include placement and memory contention. | Hot/cold heterogeneous expert mapping already exists. Logic-PIM's extra internal bandwidth differs from two cores sharing one fixed HBM interface. |
| [Heterogeneous Dataflow Accelerators / Herald, HPCA 2021](https://arxiv.org/abs/1909.07437) | Co-optimizes heterogeneous sub-accelerator resource partitioning and layer scheduling, exploiting complementary dataflows. | Treat resource split and mapping jointly; require complementary strengths and strong monolithic/reconfigurable baselines. | Neither “single shape is insufficient” nor heterogeneous partition DSE alone is new. |
| [Hardware HEFT_RT, VLSI-SoC 2022](https://arxiv.org/abs/2207.11360) | Implements a runtime heterogeneous earliest-finish-time scheduler in hardware. | A bounded hardware service-time comparator is plausible; explicitly charge decision and state-update latency. | Predicting completion and choosing the earlier engine is established; do not claim unsynthesized subcycle timing or negligible area. |
| [CUTLASS grouped kernel schedulers, NVIDIA documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) | Persistent workers repeatedly select tiles from a group of GEMMs; device scheduling and host precomputation are alternatives. | Finite descriptors, persistent workers, finer task granularity, and explicit scheduling overhead. | A grouped GEMM does not imply arbitrary expert rows can share one ordinary matrix multiply's weight broadcast. Low-intensity jobs can be scheduler dominated. |
| [Cache-Conscious Wavefront Scheduling, MICRO 2012, author page](https://engineering.purdue.edu/tgrogers/publication/rogers-micro-2012/) | Throttles active wavefronts based on lost data locality to avoid cache thrashing. | Admission may deliberately wait to preserve W/X reuse; maximal concurrency need not maximize performance. | Scratchpad residency is not a cache replacement policy. Need explicit slot ownership and dependencies, not a renamed CCWS counter. |
| [Feedback-Directed Prefetching, HPCA 2007](https://hps.ece.utexas.edu/pub/srinath_hpca07.pdf) | Uses accuracy, timeliness, and pollution feedback to control prefetch aggressiveness. | Track late arrival, excessive early slot occupancy, and demand interference. For known routed tasks, timing usefulness matters more than expert-ID accuracy. | EWMA, confidence counters, urgency, and feedback-directed prefetch are existing techniques; correctness does not imply prefetch usefulness. |
| [Gemmini architecture, official repository](https://github.com/ucb-bar/gemmini#major-components) | Separate load/store/execute controllers operate concurrently; a ROB enforces cross-controller hazards. | Separate selection/ownership, DMA admission, and local issue. Scoreboards and ownership invariants must remain independent of predictions. | Decoupled access/execute and ping-pong buffering are foundations rather than proposed novelty. |
| [ST-MoE, June 2026 preprint, sections 4.2–4.3](https://arxiv.org/html/2606.15453v1) | CCT/HT predict future expert IDs; an Expert Mapping Unit maps selected experts using prefetched availability, with configurable local PE dataflow. | Learn from confidence and correctness-preserving miss handling; distinguish predicted expert ID from predicted execution time. | ST-MoE already includes runtime mapping and staging, not only expert-ID prediction. Its published EPU area is not an estimate of this controller. |

Another close work is [HDA-MoE, September 2026 preprint, section V-C](https://arxiv.org/html/2609.08682v1): predicted compute hotspots guide budgeted pre-broadcast, then token dispatch chooses a lightly loaded node that already holds the expert. It explicitly uses heuristics rather than online global optimization. Its distributed 3D NMP deployment and routing-adaptation portions differ from this fixed-routing, private-SRAM design; nonetheless, generic compute/data-aware dispatch and prefetch are overlapping ideas.

## Candidate implementation reviewed

- Eight-entry task window; select a critical anchor by largest minimum standalone service across cores, with aging after eight binding rounds.
- Select a companion and compare both two-core orientations by predicted maximum finish time.
- Commit a Next task only when Current's predicted remaining time is no greater than first-tile fetch estimate plus scan lead plus 64 ns.
- Current/Next ownership immutable after commit. Reserve a free W landing slot; verify workspace fits when it becomes Current.
- Existing stock-based DMA arbitration remains unchanged.
- Charge sequential control service: `4 + 4*W*C`, plus two cycles per pair comparison and four per commit.
- New state target 256 B, charged within the total 2 MiB arena.
- Normalize Shared-first input for every policy and include an order-only baseline.

## Required corrections or explicit contracts

1. **Use the same work set in pair/single comparisons.** Comparing one task's finish to two tasks' maximum finish naturally favors not assigning the second task. Either require a pair when both engines and two tasks are admissible, or include deferred companion completion in the single-placement alternative. The first option is a bounded heuristic, not an optimal scheduler.

2. **Select the anchor from admissible candidates.** Aged or large tasks that cannot currently fit must not prevent other legal work from proceeding. Distinguish temporary infeasibility from a task that can never fit any core. Permanently impossible input should fail with a resource diagnostic.

3. **Aging by binding rounds is not a wall-time progress guarantee.** It prevents continued successful bindings from perpetually bypassing an old task. If no binding occurs, that age does not change. Require a progress invariant and independent watchdog; do not assume aging alone prevents deadlock.

4. **Reserve future resources without blocking Current's progress.** Next may reserve its first W landing slot, but Current must retain enough slots to execute and retire the active group. For G4 and five slots this is exactly one extra slot; verify every supported group/slot allocation and phase. Future workspace capacity can be promised after Current completes, but its occupied data cannot be overwritten now.

5. **Revalidate after the charged scan.** New data, retirements, other control service, or task completion may invalidate the scoring snapshot. A stale prediction may be suboptimal; stale resource/ownership checks are incorrect. Commit must atomically recheck task status, core status, slot and workspace promises.

6. **Check the late-binding floor.** The current remaining-time estimator has an output-drain floor. If that floor exceeds `first_fetch + scan_lead + 64`, prefetch never commits until the old task ends. This is legal but can erase the intended overlap. Report floor-blocked cycles and actual remaining time at commit.

7. **Account for finite input visibility.** One descriptor per cycle and an eight-entry window do not mean eight choices exist on every decision. Decide whether initial fill waits are allowed and charge them; freeze the visible set at the scan start or charge any rescan. Never inspect tasks beyond the window for scoring.

8. **Make the state ledger exact.** The prior dual-core control state occupied 4,032 B of a 4 KiB subreserve, leaving only 64 B. A fresh 256 B cannot fit that same subreserve without reclaiming fields. It can remain within the total 2 MiB arena by increasing the control partition and reducing available data capacity, followed by new fit/peak validation. Charge the same reserve to comparison policies.

9. **Separate predicted execution service from common supply.** Two independent finish estimates may each assume bandwidth they cannot both receive. At minimum show a shared-traffic floor, outstanding-request state and sensitivity. Current service-progress estimation is not equivalent to hardware observing exact future simulator events.

10. **Shared-first is a separate optimization.** If Shared input is available at the same boundary, early enqueue is reasonable for all policies. If not, retain its true release dependency. Results must distinguish descriptor reordering from the new runtime. Also test an ID permutation/alternate order diagnostic to expose order dependence.

## Strong baselines and ablations

All use identical capacity/ports/credits, immutable post-DMA ownership, timing charges, and routing outputs.

1. Original FIFO and dynamic ECT with old order: provenance reference only.
2. Shared-first + original dynamic ECT: isolates input order.
3. Same finite window + oldest-ready/ECT: isolates visibility from joint matching.
4. Same window + largest-service-first/ECT: isolates the anchor priority from pair assignment.
5. Joint matching without late commit, then with late commit: isolates commitment timing.
6. Joint matching with fixed model, then measured feedback: isolates actual prediction benefit.
7. Apply the same mechanism to single, homogeneous and heterogeneous organizations; preserve frozen hardware points.

An oracle may enumerate assignments on a small bounded set for attribution, but is not a deployable controller and must not donate future timings to runtime policy.

## Diagnostics that make the result causal

- Per decision: visible window length, anchor/companion IDs and shapes, age, legal-core mask before/after late gate, predicted finish for both orientations, rejected reason, control cycles and commit cycle.
- Counts of zero/one/two legal cores; fraction of decisions in which at least two task/core pairings are feasible.
- Pair-versus-greedy assignment disagreements and realized effects. Offline alternate-placement regret must be explicitly marked counterfactual and include supply contention.
- Initial window-fill delay, late-gate delay, aged bypass counts, infeasible-head bypasses, stale scan retries, and watchdog/progress checks.
- First-prefetch issue, arrival and consumption times; Current blocked by Next slot/credit reservation; ready-data inventory separately from in-flight inventory.
- Final-core gap and which expert/phase causes the tail; prediction mean error and maximum underestimation.
- Exact HBM bytes/transactions, X/Z copies, peak live bytes and result correctness. Overlapping stall counters must not be summed into global wall time.
- Paired-run determinism, frozen parameter selection and new held-out windows; do not choose the winning policy per test workload.

## Falsifiable contribution statement

“A bounded runtime preserves heterogeneous assignment choices until data ownership must be committed, jointly matching ready MoE tasks to engines while reserving finite local storage and scheduling prefetch lead time.”

The claim is supported only if the controller measurably reduces premature-binding regret or final-core tail versus equally informed ECT/list scheduling, after paying state/control costs. Heterogeneous necessity additionally requires beating the same optimized homogeneous and monolithic configurations. Where the monolithic configuration already reaches the fixed-byte supply lower bound, no scheduler can promise large acceleration without changing that bound.
