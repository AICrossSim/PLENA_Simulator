# Resource-derived MoE analytical model, 2026-09-24

This is a specified candidate microarchitecture and a max-plus/cycle timing model.
It is NOT a calibrated model of existing PLENA RTL, a conventional 2-D systolic
array, silicon, or complete model inference. No RTL work is required or performed.
The original 2026-09-23 results and defaults remain frozen.

## Four distinct dimensions

* Workload GEMM: X[Me,K] times W[N,K]^T; Me comes from actual routed token counts.
* Instruction segment: up to Mc rows, Nt=4 columns, Kt=512 reduction elements.
* Physical reduction parallelism: Pk, a power of two dividing Kt. This is explicit.
* Physical multipliers: sum(Mc)*Nt*Pk. For 6 / 3+3 / 4+2 this is 12,288 at
  Pk=512, or 3,072 at Pk=128. Compare organizations only within the same Pk.

K_tile=512 alone does not specify physical multipliers. Pk=512 retains the previous
candidate's fully parallel reduction interpretation, now with explicit services;
Pk=128 evaluates a folded candidate, not an equal-compute speedup over Pk=512.
Neither is automatically the existing MXINT mini-systolic implementation.

## Arithmetic recurrence

The datapath is a bank of Mc*Nt dot products. Each feed consumes Pk products;
G=Kt/Pk feeds complete one segment. Intra-group and inter-group sums are balanced
FP32 trees; for power-of-two grouping the numerical association equals the same
512-leaf tree. K segments update each private output in increasing order.

For group g, actual feed completion A_g comes from the W/X bank commands. The
segment becomes available at:

    dot_done = A_last + L_mul + log2(Pk)*L_add + log2(G)*L_add

The physical minimum input occupation is G cycles, not an independently supplied
whole-tile II. Bank service, context exhaustion, source readiness, and K feedback
can make actual first-feed intervals longer. Arithmetic pipelines can overlap
independent outputs; the model does not serialize complete tiles to favor 4+2.
Cross-group results occupy finite per-core contexts until the output RMW commits.

L_mul/L_add are primitive hypotheses, not measured or synthesized values. The
nominal pair is 2/2 cycles; 1/1 and 4/4 are sensitivity points. At Pk=512 the
nominal arithmetic tail is 20 cycles after the last feed, derived from 9 tree
levels, not an arbitrary whole-tile latency. Frequency is also a hypothesis;
cycles are primary, ms is explicitly converted at the assumed 1 GHz.

## Banked SRAM and register resources

Each private bank has one shared read/write command port and a 128-bit word in
the nominal configuration. Word address modulo private bank count selects a bank.
W is slot-major with 512-element rows; X is stage-major; FP32 outputs/metadata are
band-major with a 32 B per-row record (16 B data,16 B metadata).

For n_b words requested from bank b at t:

    start_b = max(t, bank_free_b)
    bank_free_b = start_b + n_b
    read_done_b = bank_free_b + L_read - 1

Reads and writes contend. Read latency does not monopolize the command port after
issue. Output RMW uses an explicitly conservative bank-lock policy: each word
holds its bank for L_read+L_add+1 cycles. There are word_bytes/4 FP32 add lanes per
accumulator bank. Other banks progress independently. A pipelined/forwarding RMW
controller would be a separate mechanism, not free here.

The first K segment logically selects a zero previous sum using its K-index flag;
no bulk, untimed clear of 2 MiB is required. The current conservative RMW service
still charges the bank read even though its payload is ignored for first K.
Completion is acknowledged at the whole issued M-cohort's last RMW completion;
earlier per-row acknowledgments would be a separately evaluated optimization.

Nominal aggregate resources for all three organizations:

|Resource|Total|Partition|
|---|---:|---|
|Weight storage including ingress|48 KiB|8 KiB shared ingress +40 KiB private tiles|
|W banks|64 x128-bit 1RW|64 single,32+32 dual|
|Private W slots|10 total x4 KiB|10 single,5+5 dual|
|X storage|12 KiB|2 stages/core, proportional to Mc|
|X banks|24 x128-bit 1RW|4 per physical M row|
|Accumulator including metadata/control|2 MiB|proportional to Mc;4 KiB reserved control|
|Accumulator banks|12 x128-bit 1RW|2 per physical M row|
|Reduction/result contexts|8/core|partial vectors scale with Mc|

The operand gather registers, response registers, arithmetic pipeline registers,
cross-group state, and controller state are separately reported in resource_bill.
They are not silently called SRAM or omitted to assert equal area. Arithmetic
register counts are lower bounds on 32-bit data registers, not a gate-level area
estimate; FP internal bits, interconnect, clocking, and SRAM macro overhead need
separate implementation evidence. Equal MAC count/capacity is not equal PPA.
The half/double-bank sweeps hold byte capacities but change port/bank organization;
they are not claimed equal-area configurations.

X enters from an on-chip producer (not HBM). A finite-word bus is backpressured
before each word; a one-cycle bus register then writes its private bank. There is
no uncharged whole-tile return buffer. Two finite X slots retain tags after reads,
allowing exact local reuse; otherwise input transfer is counted again. A W gather
register retains one Pk group and is reused by a following matching M block.

## Native HBM and backpressure

Unchanged HBM preset/addresses:8 controllers,32 B sectors,256 credits,16 MiB fixed
expert stride, BF16 row-major W[N,K] padded to32 B; no quantization/layout change.
Native delivery now reports each returning sector. It writes the reserved private
slot through the same 1RW banks that serve compute reads. The sector's source
credit is retained until the bank write completes. The 256x32 B bounded response
storage is charged INSIDE the48 KiB weight budget. W becomes readable only after
all sectors and destination writes complete. No tile-sized unaccounted assembly
buffer is assumed. Request/response metadata is timing state, not hidden payload.

This bridge is a transaction-level candidate: response storage is a credit-indexed
sector holding structure, sized for the configured native interface, not a
synthesized controller. Its data register/selection implementation and physical
timing remain uncalibrated. Source credit conservation, byte counts, destination
bank contention and final drain are checked. Legacy Native callers keep their
old behavior unless consumer backpressure is explicitly enabled.

Two nonphysical diagnostics are clearly labeled:
* onchip_oracle: removes HBM service, retains bank writes/reads, control and dependencies.
* compute_oracle: also removes operand-bank/control service but retains finite
  front-end transitions, contexts, arithmetic pipeline and ordered result feedback.
Neither is a real end-to-end configuration or literally only useful MAC time.

## Compiler mapping and scope

The schedule adapter consumes frozen real jobs/Me. It maps whole N bands with a
deterministic load estimate ceil(Me/Mc)*ceil(K/Kt), constrained by private output
capacity. An expert-pinned option supports motivation/counterexample fixtures.
The mapping does not use future HBM completions or measured policy results.
There is no claim of optimal mapping, runtime work stealing, or main-Compiler ISA
integration. Host lists stand for a compiled address program, not free runtime
search over all work. Resident K-major windows and W/X/context state are bounded.
Each W descriptor costs ceil(descriptor_bytes/control_word_bytes) cycles on one
shared port; local sequencers/bitmaps advance at most once per cycle.

For each core, up to16 owned N bands are traversed in K-major order; each W tile
serves all M blocks before release. The FIFO local policy may wait for a K
dependency. It is a declared conservative baseline, not a claim to the best core
scheduler. Output storage is held to phase end. Shared HBM and private X/W/outputs
are preserved; no uncharged partial-sum migration or cross-expert row packing.

Real study: DeepSeek-V2-Lite-Chat first MoE layer, BFCL captured B2/B4/B8/B16.
Six independent routed/shared gate/up/down GEMMs. Router/nonlinear/combine and
cross-phase traffic/residency are excluded. Their sum is not layer/model E2E.

## Verification and interpretation

Check bank scheduling against an independent explicit per-cycle occupied-slot
allocator; check read/write conflicts, atomic RMW, stream writes, folded arithmetic,
positive/negative shape cases, tail K/N/M, exact BF16/FP32 values, capacity, K order,
native bytes/credit/consumer drain, and repeat hashes. Primitive timing checks
validate implementation against this contract, not the contract against silicon.

Report arithmetic-active cycles separately from issue-decision states: they
overlap. Per-bank waits, control services and DMA latency must not be summed into
wall time. Source timing sensitivity changes queues and may change later events;
oracle differences are interventions, not additive causal latency breakdowns.

Before making a target-chip selection, establish the intended physical Pk,
operator implementation, SRAM macro/bank feasibility and clock. If ranking varies
over stated candidate assumptions, report that uncertainty rather than choosing
the nominal winner as a validated hardware recommendation.
