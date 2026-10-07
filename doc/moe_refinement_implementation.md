# MoE normal-path refinement: implemented contract and validation

Status (2026-09-09): implementation in the local Compiler/Simulator refinement
worktrees; the frozen `repro_02` release executable passed the validation below. This is
an operator-level numerical and timing experiment. The
[research question](moe_research_story.md) separates engineering groundwork from
an unproven novelty claim. The [earlier proposal](moe_architecture_refinement.md)
remains a historical design plan, not a description of every implemented feature.

## Goal and execution boundary

Test whether independently chosen token blocks and compute shapes help when
weight delivery, accumulator service and independent-output scheduling have
explicit finite resources. Compare organizations on the same encoded expert
bank, route window, numerical contract, HBM configuration and aggregate budgets.
An older timing model is not the performance baseline for this corrected model.

The timed operator starts with BF16 inputs and expert routes ready. It includes
input gather, routed gate/up, SwiGLU, down, output copying and weighted combine;
weight traffic goes through native Ramulator. Router execution, initial input
HBM transfers, final output HBM stores and an entire model are outside this
boundary. Each native invocation starts with cold simulated state.

## What changed

### One immutable expert bank, many route windows

Compiler `moe_bank_export.py` creates a complete expert catalog and encoded
`weights.bin` before selecting windows. Workload V2 references the bank manifest
by SHA-256 and carries its complete catalog. A new window changes inputs/routes,
not expert addresses, encoded values or scale ownership. The first format is
local `plena_e4m3_e8m0_block8`, output-major `[N,K]`, with a separate E8M0 scale
plane and one scale per eight source-row K elements. All experts currently share
one input dimension D and expert hidden dimension F.

Rust checks manifest/image hashes, exact catalog agreement, dimensions, format,
address ownership, strides and bounds before simulation. Loading and hashing
the image are setup work, excluded from simulated operator time. This establishes
a stable weight image; it does not create a persistent warm SRAM/cache between runs.

### Temporal token blocks are independent of physical lanes

For `X[Me,K] * W[N,K]^T`, Me is the routed token count for that expert. In normal
V2, `m_rows=Mt` controls token scheduling, `blen=P` specifies output lanes and
`mlen=R` specifies reduction lanes. Modeled multipliers are `P*R`, not `Mt*P*R`.
Gate/up use logical `(Me,K,N)=(Me,D,F)` and down uses `(Me,F,D)`.

`tail_policy=valid_rows` charges only valid token rows at the M tail; padded P/R
lanes are still counted as issued arithmetic. The `padded` policy remains a
control. The numerical operation keeps ascending global K and explicit FP32
multiply/add order, with BF16 projection outputs, so changing Mt does not change
the reference arithmetic order.

### Finite local delivery and output lifetime

- Each core has one finite aggregate decoded-weight SRAM read/write service.
  Decode placement and reads into stationary BF16 operand latches contend for
  it. Latch capacity is reserved inside the existing weight SRAM budget.
- Each core has one finite aggregate FP32 accumulator read/write service.
  The first K contribution injects zero; later contributions read the previous
  partial sum. Every result must finish its modeled feedback delay and writeback
  before that output context can consume its next K contribution.
- One or two independent N groups may be resident. Their token-block contexts,
  pending FP32 result storage and pipeline storage are charged against the
  accumulator budget. Ready groups alternate deterministically; a dependent
  output cannot bypass its own feedback. Every active N group owns at least one
  weight slot, preventing future tiles from occupying all slots needed by a
  predecessor.
- After all FP32 writebacks, projection materialization reads the accumulator
  through its finite port, then uses the shared vector service for BF16
  conversion/copy. SwiGLU/down cannot consume the output before both complete.

These are explicit aggregate port models, not bank-level SRAM calibration.
The compute fill/feedback expression remains analytical (`ceil(log2 R)` plus
the configured pipeline overhead); exact outputs do not validate a particular
physical parallel reduction tree.

### Demand-aware DMA is connected to the normal V2 path

`per_channel` preserves the control policy. Optional `demand_aware` propagates
monotonic priority tokens from resident tiles through coalesced sectors and
memory wrappers to native submission. A line needed by any demanding consumer
inherits that priority. No task placement decision uses future DRAM completion
times.

Each channel reselects pending requests once per issue cycle. Requests aged by
128 issue cycles are promoted, then demand precedes prefetch; queue order breaks
ties. A rejected candidate yields one arbitration to another pending request
when available. Already accepted DRAM commands are neither cancelled nor
preempted. The same 256 native tracker permits cover queuing through response,
including cancelled callers. Native priority state costs 4,224 additional bytes
at eight channels; bounded consumer-priority handles are also reserved by the
DMA frontend. This is pending-request arbitration, not a per-core bandwidth
guarantee or a redesign of native DRAM command scheduling.

### Transpose is a tested standalone primitive

`matrix_view.rs` maps a normal/transposed crop back to source coordinates before
looking up element and scale bytes. It handles non-square extents, unaligned
block origins, padded strides and tails without regrouping scales. A separate
finite BF16 transpose buffer models diagonal banking, padded storage, shared
bank read/write ports, fill/read cycles and release ownership.

The normal MoE engine does not yet consume arbitrary views or this transpose
buffer. No Attention operator graph is implemented by these primitives.

## Frozen full-Qwen probe

The prepared probe uses **E=256 experts, D=2048, F=512**. The complete encoded
bank is **905,969,664 bytes (864 MiB)** and is identical for the archived B8 and
B32 input/route windows. Inactive experts remain in the bank. Image size is not
the number of bytes read by an active window. Inputs/routes are archived;
weights are synthetic and the windows contain routed experts only.

The six prepared JSON files are `normal_v2_n{1,2}_{single,homogeneous,heterogeneous}`:

| Organization | Per-core `(Mt,P,R)` | Multipliers | Activation / weight SRAM port supply, BF16 elements/cycle | Accumulator port, FP32 elements/cycle |
|---|---|---:|---|---|
| Single | `(4,8,512)` | 4096 | 1024 / 1024 | 8 |
| Homogeneous pair | `(4,4,512)` + `(4,4,512)` | 2048 + 2048 | 512 / 512 each | 4 each |
| Heterogeneous pair | `(4,4,768)` + `(1,4,256)` | 3072 + 1024 | 768 / 768; 256 / 256 | 4 each |

For each organization, `n1` and `n2` differ only in one versus two active N
groups. Both reserve enough latch capacity for two groups to keep provisioned
storage constant. The small core in this probe is `(P,R)=(4,256)`, distinct from
the older square-tile experiment's `(2,512)`.

| Provisioned storage | Single | Homogeneous pair | Heterogeneous pair |
|---|---|---|---|
| Weight SRAM, including latches | 64 KiB | 32 + 32 KiB | 48 + 16 KiB |
| Latches within that budget | 16 KiB | 8 + 8 KiB | 12 + 4 KiB |
| Input/intermediate SRAM | 4 MiB | 2 + 2 MiB | 2 + 2 MiB |
| Accumulator/control budget | 1 MiB | 512 + 512 KiB | 512 + 512 KiB |

All six use three weight slots per core, no read cache, valid-row tails,
work-conserving whole-expert dispatch with threshold 8, a 1 ns core clock and
16 configured pipeline-overhead cycles. Shared resources are 128 logical DMA
credits, 8 KiB DMA staging, 44 KiB frontend SRAM, 16 KiB dispatch queue, 4 MiB
combine SRAM and a 512-element/cycle vector service. Aggregate activation and
weight-port supply are each 1024 elements/cycle; aggregate accumulator supply
is eight elements/cycle. These resource equalities are not an iso-area claim.

The first full probe uses **per-channel DMA, lookup II=1 cycle, sector reads and
coalescing enabled, fair credits disabled**. A second full-dimension probe changes
only the issue policy to demand-aware in copies of the same configurations. All
organizations share the same eight-channel native HBM2 model: two 64-bit
pseudo-channels per channel at 2 Gbit/s/pin, **256 GB/s theoretical total**.
That peak is not a measured sustained application throughput.

## Reproduction and acceptance gates

The release executable/library and source identities are archived in
`/scratch/shared/mcl123/plena/outputs/moe_refinement_20260909/repro_02`.
Commands below use the existing local Python/codec environment. A new smoke
output directory must not already exist. The full probe command reuses the
prepared immutable bank and six frozen configurations.

```bash
PLENA_BASE=/scratch/shared/mcl123/plena
PLENA_SIM="$PLENA_BASE/review_20260909/simulator-moe-refinement"
PLENA_COMPILER="$PLENA_BASE/review_20260909/compiler-moe-refinement"
PLENA_REPRO="$PLENA_BASE/outputs/moe_refinement_20260909/repro_02"
PLENA_PYTHON="$PLENA_BASE/venvs/plena-py311/bin/python"
export PYTHONPATH="$PLENA_BASE/review_20260905/simulator-moe-review/PLENA_Tools"
export LD_LIBRARY_PATH="$PLENA_REPRO${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PLENA_PYTHON" "$PLENA_SIM/transactional_emulator/testbench/moe_refinement/fixed_bank_smoke.py" \
  --compiler "$PLENA_COMPILER" --binary "$PLENA_REPRO/moe_dual_normal" \
  --output-dir "$PLENA_BASE/outputs/moe_refinement_20260909/smoke_manual"

"$PLENA_PYTHON" "$PLENA_SIM/transactional_emulator/testbench/moe_refinement/fixed_bank_probe.py" \
  --run-only --models qwen --compiler "$PLENA_COMPILER" \
  --binary "$PLENA_REPRO/moe_dual_normal" \
  --source-root "$PLENA_BASE/outputs/moe_full_shape_benefit_20260905" \
  --output-dir "$PLENA_BASE/outputs/moe_refinement_20260909/fixed_bank_full_qwen"

"$PLENA_PYTHON" "$PLENA_SIM/transactional_emulator/testbench/moe_refinement/demand_probe.py" \
  --prepared-root "$PLENA_BASE/outputs/moe_refinement_20260909/fixed_bank_full_qwen" \
  --binary "$PLENA_REPRO/moe_dual_normal" \
  --output-dir "$PLENA_BASE/outputs/moe_refinement_20260909/demand_manual"
```

The small smoke uses D=17, F=19 and two route windows, including an expert
absent from the first window becoming active in the second. It checks legacy
controls plus normal V2 organizations, one/two N groups and both DMA policies.
Six malformed-bank/window cases must fail without publishing a successful
report. Module tests additionally cover scale ownership, transpose ports,
limited accumulation/weight service, independent-output ordering and native
priority/credit lifetimes.

Every reported comparison requires exact BF16 agreement with an independent
reference, identical bank/workload/configuration identities, declared SRAM and
aggregate resource limits, native request accounting and drained state, and
deterministic repeated counters/timing. Each full probe completed **24 native runs**:
two windows × two N-group settings × three organizations × two repeats; both
DMA policies therefore account for 48 runs.
Report useful MAC utilization separately from compute busy fraction, and do
not sum overlapping per-core waits as a wall-time breakdown.

**Validated results:** 39 focused Compiler tests, 35 MoE Rust tests, 16 memory
and 10 Ramulator tests, and 41 Python evidence-gate tests passed. Rust Clippy
checks passed. The small integration smoke passed 56 positive native runs and
six malformed-input rejection checks. The two full probes passed all 48 runs.
The separately recorded rectangular-shape diagnostic passed 16 additional runs,
bringing the total to **120 successful native executions and six rejection checks**.

In the fixed full-dimension configurations, independent-N scheduling reduced
latency for every organization (about 1.40–1.56x). Changing only DMA priority
changed elapsed time by 0–0.34%. The selected heterogeneous shape remains slower
than the corresponding single core. These effects must not be conflated.
See the [complete initial probe report](/scratch/shared/mcl123/plena/outputs/moe_refinement_20260909/INITIAL_PROBE_REPORT.md)
and [integration smoke evidence](/scratch/shared/mcl123/plena/outputs/moe_arch_refinement_20260909/smoke_v2/summary.json).

The incomplete `repro_01` full probe omitted final accumulator/vector drain
service and was superseded; its timings are not performance evidence.

## Remaining architecture questions

Dispatch is still threshold/work-conserving at whole-expert granularity. There
is no cross-core M/N/K split, migration, finish-time placement heuristic, mixed-F
expert catalog, persistent warm state, arbitrary-view MoE integration or
Attention execution. Aggregate ports, analytical feedback and current DMA
selection need controlled sensitivity studies before making physical-throughput
claims. This six-configuration probe is not a DSE. A size or scheduling benefit
must survive the same optimization on the single-core and homogeneous controls;
these implementation features alone do not establish novelty.

## Next controlled experiments

The newly legal large shape `P=6,R=512` is a separately recorded diagnostic,
selected after observing `R=768` padding. It preserves 3072 large-core
multipliers and the same storage/port budgets, but introduces small N tails and
more K accumulator updates. All 16 diagnostic runs passed. Useful/issued MACs
reached 99.58% (B8) and 99.45% (B32), but this heterogeneous configuration still
took 11.64–16.10% longer than its paired single-core control across both windows
and DMA policies. Its measurements remain separate from the six original
configurations; see the [diagnostic evidence](/scratch/shared/mcl123/plena/outputs/moe_refinement_20260909/nonsquare_probe/summary.json).

Next freeze a bounded DSE over **both** compute organization and supply:
`Mt/P/R`, activation/accumulator port allocation, weight slot partition and
independent-output count. Give the single and homogeneous controls the same
options. Keep the full bank fixed and select on training windows before testing
held-out routes. A finish-time placement heuristic is a separate mechanism and
ablation; the present threshold/work-conserving dispatcher is its control.
Do not attribute a future scheduling improvement to more HBM bandwidth or to
heterogeneity unless those isolated comparisons support it.
