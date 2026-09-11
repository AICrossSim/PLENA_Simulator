# Fixed-route MoE bottleneck diagnosis

This experiment identifies service limits in the numerical Compiler-export / Rust
MoE runner. It does not search for a new architecture or claim equal-cost speedups.
The existing compiler workload, block8 weight bank, core geometry, SRAM capacity,
finite credits, route tuples, and each core's expert order are fixed.

Run with Python 3.11 after building `moe_dual_normal` and running the tests recorded
by the runner. A fresh output directory archives the executable, native library,
patch, input hashes, and complete run manifest. An existing frozen manifest resumes
atomic result checkpoints and revalidates every saved result; it does not silently
replace the frozen binary or experimental cases.
The main campaign used 8 host worker processes after an initial 2-worker pass;
this changes host concurrency only. Every point is repeated twice and checked against independent numerical golden
outputs. Baselines must reproduce the previous frozen total time exactly.

## Interventions

All rates default to one. HBM profiles and the ideal-memory backend are applied
by the `moe_dual_normal` binary; callers embedding `execute` must construct the
matching memory backend themselves. Factors are explicit counterfactual service
changes:

- MAC: divide issue service and reduction/pipeline feedback delay by the factor.
  Activation delivery remains independent: issue service is the maximum of MAC
  service and activation service. The core clock and physical P/R/Mt do not change.
- Activation: divide activation delivery service only.
- Weight / accumulator: divide serialized SRAM-port service after the original
  element-rate ceiling. This is not an increase in SRAM capacity.
- Scheduler: divide output-pool visits and expert-dispatch service; preserve fixed
  ownership/order. The legacy engine has no modeled per-visit scheduler charge.
- DMA: divide lookup, copy, and native frontend admission service. Preserve queue
  sizes, credits and HBM-domain timing. Cache service (disabled here) is unchanged.
- Vector: divide shared decode, gather, SwiGLU, copy and combine service. Weight
  SRAM service uses the core clock and is not accidentally scaled with decode.
- HBM column: halve same-direction column recurrence constraints, leaving response
  latency, row timings, native command clock and DMA admission unchanged. This
  increases a timing ceiling; it does not promise twice the achieved bandwidth.
- HBM return: halve RD-command-to-response delay; retain ACT/PRE constraints.
- HBM all timing: halve bank/column/response delays, preserving the memory command
  clock and refresh intervals. This still retains native command issue limits.
- HBM clock: halve only the memory-domain period (1 ns to 0.5 ns), with the same
  cycle constraints, geometry and 32-byte transactions. Both physical latency and
  bandwidth change, as does refresh cadence in physical time. It is not a JEDEC
  device specification. The main core and DMA clocks stay fixed.
- Ideal HBM: remove native DRAM and native ingress timing, but use the same real
  encoded bytes and keep upstream DMA, decode, ports, scheduling and arithmetic.
- Ideal supply64: ideal HBM plus 64x faster non-MAC service, with original MAC issue
  and feedback delay. Finite slots, dependencies and nonzero residual service
  remain. This is an oracle approaching compute limits, not a feasible design.

`compute_busy_ps` includes activation-limited issue intervals. It is not pure MAC
activity. Original `mac_utilization` uses nominal PE/clock and must not be treated
as utilization of a counterfactually accelerated MAC. Resource waits overlap;
never add them or subtract them from wall time to predict a speedup.

## Comparison boundary

Three isolated expert groups (Me=1,8,32) diagnose the single N3 core. Two archived
multi-expert windows (B8,B32) compare single N3, single Q32, heterogeneous N2 and
heterogeneous Q32 under the same total 4096 PEs, 64 KiB weight SRAM and fixed port
budgets. Both organizations receive each intervention. Work-conserving baseline
ownership is frozen before changing timing; these are sensitivity measurements,
not claims of an optimal rebalanced scheduler.

Gate/up shapes are `(Me, K, N)=(Me,2048,512)`; down is `(Me,512,2048)`.
P/R/Mt: single `(8,512,4)`; heterogeneous `(6,512,4)+(4,256,1)`.
P and R are physical spatial lanes; Mt is temporal batching. Small Me alone does
not waste a physical M dimension in this implementation. These aligned N/K cases
therefore cannot establish benefits from avoiding N/K tail padding.

## Follow-up and aggregation

`adaptive.py` restores the existing work-conserving dispatcher for the two route
windows, four organizations/modes and three interventions (HBM clock + DMA 2x,
all services 2x, ideal supply64), repeated twice: 48 additional numerical runs.
This diagnoses whether fixed ownership hides a placement problem; it does not
search for optimal placement. The baseline work-conserving runs already exist in
the main regression.

After both manifests report `passed`, run `analyze.py`, `plot.py`, and `report.py`
with the project Python 3.11 environment. These write the sensitivity tables,
per-core service audit, organization comparisons, standalone PNG/SVG and Chinese
report into the frozen output directory.
