# Bounded output scheduling over finite operand storage

Status: implementation and frozen validation complete. All 210 native main
experiments passed. The scan-based output pool remains opt-in: it does not
establish a stable latency benefit or a heterogeneous-core advantage.

## Question and scope

Does separating independent output completion from resident weight storage
reduce normal-path MoE latency when weights, SRAM, ports and HBM are finite?
The same mechanism is offered to a single core, a homogeneous pair and a
heterogeneous pair. A generic scheduling benefit is not a heterogeneous win.

The previous normal V2 engine gives each active N group its own operand latch
and weight-slot share. A group cannot be reused until all its K contributions
complete. The new optional pool retains a bounded number of output records,
while a separate bounded operand-stage pool serves ready work. It preserves the
entire expert's M-row weight reuse and the ascending-K FP32 numerical contract.

The existing immutable Compiler weight-bank/workload contract is sufficient:
physical output records and operand stages are runtime architectural resources,
not a reason to change source weight addresses or regenerate a Compiler bank.
No new transpose, Attention, cross-core task partitioning, local accumulator
dataflow or HBM command scheduling is included in this first intervention.

## Executable resource contract

`CoreRefinement.output_pool` is optional. Omitting it preserves the original
normal engine. When present, it specifies `output_contexts` (Q, 1–256),
`operand_stages` (1 or 2), and positive `scheduler_cycles` per visited descriptor.
Q counts independent `(M block, N band)` records. Admission reserves all
`ceil(Me / Mt)` M blocks for an N band together. An expert that cannot fit one
complete cohort is ineligible for that core; fixed placement reports an error
instead of silently changing ownership.

```text
HBM element + scale -> finite weight slots -> finite operand stages -> MAC
                                                                     |
                         Q output records <---- pending FP32 writeback
                              |
                   same output's next K may proceed
```

A weight tile serves every M block in its admitted band before releasing its
slot and operand stage. Its output records remain live until their dependent
FP32 writes finish. Each record allows one pending update and preserves the
original ascending global K order. The last completion retires the band through
a bounded event queue. Writeback already owns its destination; it never waits
for a new weight slot, operand stage, or DMA credit.

All storage is charged within existing per-core budgets:

| State | Reserved bytes |
|---|---:|
| Output and slot/stage control records | `128*Q + 64*(weight_slots + operand_stages)` |
| Independent pending FP32 results | `Q*Mt*P*4` |
| BF16 operand latches | `operand_stages*P*R*2` |
| One resident weight tile | `3.125*P*R` for the current block8 format |

The last row includes packed elements, packed scales and decoded BF16, with
block/tail rounding enforced by code. The existing full FP32 accumulator and
other live operator storage are charged separately. This is a capacity and
service model, not a transistor-area model.

The scheduler uses round-robin current-K demand and charges descriptor visits.
It does not speculatively preload a band's future K. Scheduler visits, operand
preload and MAC issue share one serial actor; background HBM loads and FP32
writebacks advance independently. Thus legacy-to-pool compares an admission
and scheduling change as well as output ownership. Q8/Q16/Q32 isolates capacity
within this one pool policy.

Each gate/up/down projection reports tile-load count and latency, useful/issued
MACs, local port service, waits, scheduler work and peak occupancy. A tile's load
latency starts when loading is spawned and ends when decoded data is ready; it
includes delays inside the load path, but not waiting before tile admission.
These observations do not affect scheduling. Projection end drains outstanding
loads and feedback before the next record begins.

## Frozen configurations

Here `P` is physical parallel output width, `R` is reduction width (the current
MLEN), and `Mt` is a temporal M-row block. Mt is not another physical PE axis.
The underlying expert GEMMs use `(Me,K,N)`, where Me varies with routing;
`D=2048`, `F=512`, gate/up use `(Me,2048,512)`, and down uses `(Me,512,2048)`.

| Organization | Cores `(P,R,Mt)` | Weight SRAM | Accumulator elements/cycle |
|---|---|---|---|
| Single | `(8,512,4)` | 64 KiB | 8 |
| Homogeneous pair | two `(4,512,4)` | 32 + 32 KiB | 4 + 4 |
| Heterogeneous pair | `(6,512,4)` + `(4,256,1)` | 48 + 16 KiB | 4 + 4 |

Every organization has 4096 modeled PEs, 4 MiB private vector SRAM in total,
1 MiB private accumulator SRAM in total, and the same aggregate activation and
weight port rate (1024 BF16 elements/cycle each). Shared settings are 128 DMA
credits, 8 KiB staging, 44 KiB frontend SRAM, the same per-channel DMA policy,
HBM configuration, and 1 ns core clock. Pool Q8/Q16/Q32 all use three weight slots
and two operand stages per core. Legacy N2 uses two operand latches; the single
N3 control uses three, within the same 64 KiB weight-SRAM budget.

The bank holds all 256 experts at fixed addresses. Archived B8 routes select 17
experts (64 assignments); B32 selects 23 (256 assignments). Inputs and weight
values are synthetic, and no shared expert is enabled in these performance
windows. Shared-expert and irregular-tail correctness have separate tests.

Fixed threshold placement assigns Me>=8 to core 0 and the remaining experts
to core 1. The work-conserving control lets an idle core take a fitting expert,
preferring its own threshold class and then job order. It does not predict
completion times or jointly optimize channel/port service. Its one-cycle job
selection is the existing abstract dispatcher model, shared by all controls.

## Comparison boundaries

- Complete numerical normal MoE operator with ready BF16 input/routes, including
  gather, gate/up, SwiGLU, down, materialization and combine. Native Ramulator
  supplies weights. Setup/image loading is outside simulated time.
- One fixed full expert bank, with synthetic numerical values and archived
  routing windows. Service microbenchmarks use declared generated routing and
  inputs against that same bank. Each invocation begins with cold state.
- Exact BF16 outputs, deterministic repeated timing/native counters, source and
  artifact hashes, finite resource checks and drained native requests.
- State occupancy and service wait counters may overlap. Per-core waits are not
  additive components of operator time; service curves are not whole-model speed.

## Experimental sequence

1. A separate accumulator-port diagnosis changes only 4+4 to 6+2 on the frozen
   P6/R512 + P4/R256 normal-N2 architecture, with an equal-resource single-core
   control. Its work-conserving dispatcher may change which expert each core
   obtains as timing changes. It is a port intervention, not a fixed assignment
   per-core microbenchmark and not an architecture search.
2. Characterize the old and pooled normal paths on Me=1,2,8,32 plus a concurrent
   two-expert case. Keep threshold assignment fixed so pool capacity does not
   silently change which core can execute a large expert. Report which core
   actually executes: one-expert dual-core cases are service characterizations,
   not evidence for using all aggregate PEs.
3. Compare fixed B8/B32 windows at Q8/16/32 under the same assignment rule, then
   use a predeclared Q32 work-conserving confirmation where every core can fit
   every selected expert. This separates within-core mechanism changes from
   capacity-dependent task eligibility.
4. Include a budget-legal three-N-group single-core baseline as well as normal
   N2. The single core receives the pool mechanism too. All organizations retain
   matched aggregate PE, SRAM and modeled port budgets.

The hypothesis is rejected for a tested case if the pool remains underused,
extra scheduling or traffic offsets overlap, or a stronger single control closes
the apparent gap. Further local accumulation is justified only by a measured
remaining accumulator-service limit and requires a separate resource contract.

## Validation and results

The isolated port experiment completed 12 native executions with exact outputs,
deterministic repeats, unchanged bank/traffic and resource/native-drain gates.
It used the already frozen `repro_02` release binary, independently of the new
pool implementation.

| Window | Single acc8 (ms) | Pair acc4+4 (ms) | Pair acc6+2 (ms) |
|---|---:|---:|---:|
| B8 | 0.717776 | 0.828061 | 0.765839 |
| B32 | 1.180535 | 1.319428 | 1.233640 |

The changed partition reduces pair latency by 7.51%/6.50%, but the pair remains
6.70%/4.50% slower than this single control. Shared HBM bytes stay unchanged.
The large accumulator serves faster while the small one serves slower; the
work-conserving dispatcher also changes assignments. This is an end-to-end
response to one configuration change, not a fixed-job per-core service ratio
and not a proof of a heterogeneous win.

[Frozen port hypothesis and results](/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/accumulator_split_control/summary.json).

The main experiment completed 210 native executions: 130 service cases, 52
fixed-placement comparisons and 28 predeclared work-conserving confirmations.
All exact BF16, deterministic repeated timing/counter, source/bank identity,
capacity and native-drain gates passed. The native binary and sources were
unchanged across phases. Validation also includes 45 Rust tests, 54 Python
evidence tests, Clippy with warnings denied, 28 unchanged legacy native
comparisons, and 28 new-pool native smoke executions. An independent tail-aware
analytic audit reconciled 2,634 representative projections (7,902 checks) with
zero tile-load or SRAM-port service differences.

The work-conserving confirmation is the useful cross-organization comparison:

| Organization / schedule | B8 (ms) | B32 (ms) |
|---|---:|---:|
| Single N2 | 0.717776 | 1.180535 |
| Single N3 | 0.582155 | 1.003224 |
| Single pool Q32 | 0.596846 | 1.177071 |
| Homogeneous N2 | 0.741689 | 1.220115 |
| Homogeneous pool Q32 | 0.661646 | 1.225583 |
| Heterogeneous N2 | 0.828061 | 1.319428 |
| Heterogeneous pool Q32 | 0.745106 | 1.433075 |

The heterogeneous pool reduces its own B8 latency by 10.02%, but increases B32
latency by 8.61%. It remains slower than single N3 in both windows. Single Q8
in the separate fixed-placement B8 sweep is 1.15% faster than N3; in B32, even
the fastest tested single pool (Q32) is 17.33% slower than N3. This is not a
stable mechanism gain. N3 is a strong tested control, not a claimed global
optimum. Actual WC jobs all fit at least two N bands in Q32, so a Q64 rerun was
not justified as a capacity correction.

Two observations explain why a broad "more contexts is faster" claim fails:

- Me1 single Q8 records 20.914 us of scheduler service over 30.299 us total.
  Weight-wait time falls, but descriptor processing overlaps HBM work. This is
  not evidence that HBM became six times faster. Me32 single Q32 records
  38.463 us of scheduler work and takes 97.115 us versus N3's 70.532 us, despite
  identical useful work and weight/accumulator port service volumes.
- Fixed threshold placement misjudges per-expert cost. In B8 it sends 75% of
  MAC work and 88.2% of weight bytes to the core with 25% of the PEs. In B32 that
  core has only 7.0% of MAC work but still 47.8% of weight bytes. A small Me does
  not eliminate the expert's fixed weight transfer. WC improves this allocation
  but still does not produce a heterogeneous win.

These observations motivate a different controller organization and a better
task cost model. They do not prove that all output decoupling, heterogeneous
shapes or HBM controller designs are ineffective. The old path remains the
default; the candidate is retained for reproducible research.

Evidence root: `/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909`.
[Full result tables and CSVs](/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/final_report/REPORT.md),
[Chinese explanation](/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/final_report/RESULT_ZH.md),
[independent service audit](/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/service_audit/full_audit.json).
Earlier `repro_02` measurements remain archived separately from this build.

## Reproduction

The release executable and native library are archived in `repro_01`, alongside
source copies, hashes and build provenance. `prepared/prepared.json` freezes
the Compiler/exporter inputs, full bank, all configurations and the 210-run
plan. The executable SHA256 is
`9e5f4374a10bd537a76b8dfd2937e62fa9db6e56751075a9dbfc96d16bddf3fb`.

From this worktree, the following reruns the whole frozen plan in a fresh UUID
directory. Use the explicit result path it returns, not `latest_run.json` when
independent phases run concurrently.

```sh
LD_LIBRARY_PATH=/scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/repro_01 \
  /scratch/shared/mcl123/plena/venvs/plena-py311/bin/python \
  transactional_emulator/testbench/moe_refinement/output_pool_probe.py run \
  --prepared-root /scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/prepared \
  --binary /scratch/shared/mcl123/plena/outputs/moe_output_pool_20260909/repro_01/moe_dual_normal \
  --phase all
```

`analyze_output_pool.py` accepts the prepared manifest, one or more explicit
completed `--result` paths and a fresh `--output-dir`. It requires all frozen
cases, the exact run count, matching binaries/libraries and source snapshots.
`audit_output_pool_service.py` independently recomputes tile counts and port
service; `plot_output_pool.py` plots the consolidated `points.csv`.

## Design decision criteria

Increasing Q is not always increasing concurrency relative to legacy N2.
For Me=32 and Mt=4, Q8 admits one N band, whereas N2 admits two. With Mt=1,
Q32 likewise admits only one band at Me=32. Work-conserving Q32 ensures each
selected job is eligible on each core, but does not ensure legacy-equivalent
N-band concurrency. Reports therefore retain the M cohort size and actual
admission bound for every projection.

If descriptor service dominates, the next intervention should be a bounded
ready-queue controller, tested separately from asynchronous operand refill:

- Store current-K and next-K ready-M links in the charged context records;
  retire each whole M cohort before releasing its weight tile.
- Maintain finite load-eligible, decoded-ready, free-slot/stage and ready-stage
  queues with membership bits. Advance them on actual load/feedback events.
- Reserve completion credits before launch. Bound result-completion events by
  Q and load-completion events by weight slots; draining them must not require
  another operand or output allocation.
- Charge descriptor reads/updates and finite FIFO ingress/egress service. A
  concrete candidate reserves another `256 + 16*(Q + weight_slots)` bytes from
  the existing budget, with at most one 64-byte descriptor update per cycle.
- Initially preserve the current-K demand policy and serial operand refill.
  This isolates controller organization; a later independent stage-fill actor
  must still contend for the same weight SRAM port.

This is a follow-up design specification, not an implemented or measured
ready-queue engine. The first success condition is fewer descriptor visits per
real tile/update and lower complete-operator time at unchanged bytes and exact
outputs. The same controller must be offered to single and paired cores.

Local accumulation requires a different justification. For reduction length
`L=ceil(K/R)`, this model currently serves approximately `2*L*Me*N` FP32 elements
per projection, including final materialization. Keeping partial sums locally
can reduce backing-SRAM traffic to `2*Me*N`, but local partial-sum read/write
traffic still exists and needs a finite service contract. Current accumulator
SRAM already has zero fixed access latency, so simply renaming it a smaller
local buffer at the same port rate provides no automatic speed benefit.
At R512, gate/up have L4 and down L1: the full Qwen FFN's backing traffic could
at most halve under this change, not imply a fourfold operator speedup.
