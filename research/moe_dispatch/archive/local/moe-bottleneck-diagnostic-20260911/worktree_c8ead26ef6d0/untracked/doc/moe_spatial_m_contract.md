# Spatial-M compute experiment contract v1

Status: independent compute-only mechanism experiment, not a change to the
accepted normal-buffer MoE engine or its frozen results. No RTL or PPA claim.

## Hardware and tensor dimensions

An expert GEMM is X[Me,K] W[K,N] -> Y[Me,N]. `m_lanes`, `n_lanes`,
`k_lanes` are **physical** multiplier dimensions. One engine contains
`m_lanes*n_lanes*k_lanes` multipliers. Its spatial M rows consume distinct
activation rows and broadcast the same weight tile. This is a bank of parallel
dot-product datapaths; it is not automatically a conventional 2-D systolic
array. K=512 means 512 physical reduction lanes here, not just tensor length.

The motivating configurations are [6], [3,3], [4,2], all with n_lanes=4 and
k_lanes=512: 12,288 total multipliers. They must not be compared directly with
the old 4,096-multiplier temporal-M configuration.

## Issue, pipeline, dependencies

Each invocation contains one expert, one output N tile, one K segment, and up
to m_lanes distinct rows. Different experts cannot share an invocation.
Different experts MAY occupy consecutive pipeline invocations: the single
engine is not artificially forced to drain before switching experts.

`issue_interval_cycles` and `result_latency_cycles` are independent inputs.
Completion is issue+latency. Results commit before selection at the same cycle.
The next K segment of an output waits for its previous segment to commit.
Independent outputs may overlap, including outputs from different experts.
Each engine has ceil(latency/II) finite pipeline entries, each reserving a full
m_lanes*n_lanes FP32 partial-result register tile. No unbounded pending result
queue is permitted. The engine advances to actual issue/completion events;
elapsed cycles are not computed as a fixed multiple of wave count.

The primary diagnostic latency is 25 cycles, II=1. Also measure latency1/II1
and latency25/II25. These are explicit analytical sensitivities, NOT measured
or synthesized pipeline parameters. The fixed latency isolates spatial shape
from implementation-dependent timing; a later hardware model must replace it.

M/N/K tails retain the full physical tile service. Masked lanes do not execute
useful MACs but consume issue capacity. No `valid_rows` latency shortcut.

## Numerical contract

Inputs and weights are deterministic, expert-specific BF16 values expanded to
FP32. Each K segment performs a zero-padded balanced FP32 reduction over
k_lanes, then adds to the FP32 output accumulator. Segments commit in increasing
K order. Output is FP32 (with BF16 bits provided for checking). There is no new
quantizer or MX codec in this experiment: operands are already decoded.

The fixture values are dyadic and small enough for exact FP32 sums. Therefore
an independent scalar ascending-K reference must match bit-for-bit despite
using a different reduction organization. Dedicated tests also check tree
rounding behavior. Trace timing-only runs must be explicitly labeled and cannot
be counted as numerical full-model execution.

## Scheduling and challengers

Default: pin each whole expert to one engine using identical deterministic
longest-work-first assignment for all shapes. Estimated per-engine load is
ceil(Me/m_lanes)*ceil(N/n_lanes)*ceil(K/k_lanes) issue slots. An assigned engine
can switch experts between invocations, including while earlier results remain
in flight. There is no fixed cost or delay for switching in this ideal model.

Also evaluate tile stealing: engines may take distinct ready row/N/K work from
the same expert, without cross-expert packing inside one invocation. This is a
stronger scheduling baseline and is enabled identically for every organization.
Splitting across engines can imply extra weight delivery in a physical design;
this compute-only diagnostic deliberately removes that cost.

The batch-partition oracle tries all positive two-way M splits of the fixed
budget and selects the best measured execution for the current batch. It has
zero partition/reconfiguration cost, retains the same total multipliers and
executes the same pipeline model. It is a bounded challenger, not a proof of
optimality over arbitrary cycle-by-cycle reconfiguration or expert fusion.

## Boundary and metrics

All operands are ready at ideal interfaces. HBM/DMA, decode, SRAM ports, routing,
scheduler service, and nonlinear FFN operations are outside this first-stage
experiment. No total-SRAM/area equality or full-chip speedup claim follows from
equal multiplier count. Record required operand and result-register resources.

Report actual elapsed cycles, invocation count per engine, issue timestamps,
completion timestamps, valid/issued MACs, tail MAC slots, pipeline peak, and
dependency checks. Separate:

* occupied-invocation utilization = useful MACs / padded issued MAC slots;
* elapsed utilization = useful MACs / (total multipliers * elapsed cycles), II=1;
* drained pipeline latency from steady-state issue throughput.

Do not invent one global wave count for asynchronous, variable-width engines.
Report per-engine invocations and distinct issue times. Hand-derived toy waves
are explanatory quantities with explicitly synchronized assumptions.

## Required mechanism gates

For one N/K tile, [4,2] must have fewer issues than [3,3] on Me=[4,2]. Me=[3,3]
must admit a homogeneous win under pinned ownership. Me=[5,1] and [2,2] are
non-win/equality controls. Test nondivisible N/K and multiple K dependencies,
consecutive expert switching, finite pipeline bounds, exact values and repeat
identity. Test tile stealing and the partition oracle, even if they erase the
fixed heterogeneous advantage.

Only after these gates, run a bounded trace-driven shape sweep. Trace train and
held-out windows must be separate. If independent evaluation coverage is too
small to support a general claim, report that limitation and do not launch a
large DSE or memory extension automatically.
