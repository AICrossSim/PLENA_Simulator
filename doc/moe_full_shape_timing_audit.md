# Full-shape MoE timing and evidence audit

## Mapping to the 0904 meeting

The meeting's final first milestone is normal buffers on both cores, independent
accumulators, simultaneous memory loading, comparison with grouped single-core
execution at equal total multipliers, and reported utilization. Transpose,
attention, many small cores and shared SRAM are subsequent work. This experiment
implements that first numerical comparison through `moe_dual_normal`; it is not
an implementation of the complete drawn architecture in the legacy ISA or RTL.

The core exposes B=BLEN and Kt=MLEN, with square M/N macrotiles B×B. The full
operator is `[M,D] × [D,F]`, then `[M,F] × [F,D]`. A macro tile issues B²Kt MACs
on BKt multipliers. Thus BKt, rather than B²Kt, is the counted multiplier budget.
This is a restricted shape family, not an arbitrary independent M/K/N DSE.

## Existing code versus the experiment

`transactional_emulator/src/matrix_machine.rs::mm` reads an MLEN-square matrix
SRAM view, selects BLEN columns, calls
`core.compute(SYSTOLIC_PROCESSING_OVERHEAD + mlen)`, reads BLEN vector rows and
updates the accumulator. `matrix_core.rs` records core geometry and applies its
configured latency multiplier. The default overhead in `load_config.rs` is zero;
other run configurations may override it.

The new experiment consumes a finite BLEN×MLEN weight tile. Its pipelined mode
models BLEN issue cycles plus ceil(log2(MLEN))+16 result-latency cycles. It tracks
dependent accumulator reuse and the final drain. This is a proposed pipeline
service model, **not a measured throughput property of the existing RTL**.

The conservative sensitivity uses MLEN+16 cycles, serialized per instruction.
It checks sensitivity to the old instruction-service form while preserving the
predeclared overhead. It is not a cycle-equivalent legacy MatrixMachine replay:
that machine has different SRAM read operations, default overhead and control.

The local RTL `src/matrix_machine/rtl/matrix_machine.sv` instantiates an MCU with
KLEN=BLEN and MLEN-wide operand interfaces. This supports reporting geometry and
port requirements; it does not validate the new pipeline's issue interval,
adder-tree arithmetic or exact banking. RTL simulation/synthesis remains needed.

## Common budgets and differing port assumptions

Every main configuration has 4096 matrix multipliers, 4 MiB private activation
SRAM, 1 MiB private accumulator/pipeline SRAM, 64 KiB private weight SRAM and
40 KiB private cache (including tags). A pair divides each budget equally. The
shared resources are 4 MiB input/combine SRAM, 16 KiB job descriptors, 64 loader
credits with 4 KiB burst staging, one 512-element/cycle vector actor, one
one-cycle dispatcher and one eight-channel Ramulator HBM2 instance. Clock is
1 ns and matrix overhead is 16 cycles. The cache-disabled comparison sets all
cache capacities to zero, with every other resource unchanged.

Each active core assumes a private MLEN-wide BF16 activation read path, BLEN
output lanes, two weight slots and one 64-byte cache port. Aggregate assumed
activation width is therefore 128/256/512/1024 for the four single geometries,
256 for the homogeneous pair, and 320 for the heterogeneous pair. Two cache
ports versus one, duplicated control, banking, and the different wide read paths
are not area-equivalent. Equal multiplier count and capacity is **not iso-area**.
SRAM bank conflicts and detailed accumulator read/write port conflicts are not
modeled; the finite capacity/ownership checks do not establish physical timing.

The shared vector actor charges gather, MX decode/placement, SwiGLU, copy and
combine at its configured operator throughput. Those latencies are analytical,
not calibrated exp/decode hardware timing. The real HBM requests and contention
do use the repository Ramulator backend and actual encoded memory bytes.

## Data and numerical evidence

Full fixtures retain the original route archive's D/F dimensions and validate
both the original NPZ SHA-256 and the archived token/expert/slot/weight slice.
Batch 8 and 32 select valid captured decode decisions; batch 32 re-batches
decisions from a batch-16 capture. Neither is a fresh full-model forward, and
batch 32 is not prefill. Only active routed experts are executed in this campaign.
Shared experts remain covered by the smaller functional tests, not full-shape
model-level performance measurements.

Inputs and weights are reproducibly generated synthetic values. Actual PLENA
MX quantization is used; packing retains the reference arithmetic but batches
its scalar range checks. Tests compare every finite E4M3 field encoding with the
reference packer, tail layout/decoding with the old exporter, and complete
outputs with the independent scalar oracle. Full-shape NumPy arithmetic only
vectorizes independent outputs; ascending-K FP32 multiply/add order and BF16
stage boundaries remain explicit. It does not call a BLAS reduction.

Time starts with inputs/routes ready and weights resident in HBM. It includes
gather, actual weight transfer, numerical FFN and weighted combine, but excludes
router execution, initial placement, host grouping, output HBM store, other
layers and full request latency. Each configuration repeats twice with exact
deterministic result checks and hash-bound numerical/resource gates. Interrupted
attempt directories are retained and excluded; only a final passing comparison
manifest authorizes a timing observation.

## Interpreting a benefit

Compare each pair against the fastest of all four tested singles. Also compare
the heterogeneous pair against the homogeneous pair; a win over one single
does not isolate heterogeneity. The static-threshold pair tests queue stranding,
and cache-disabled/serialized comparisons test dependence on caching and timing.
These are finite configurations and route windows. No result establishes the
globally optimal partition, a full-model speedup, iso-area benefit or novelty.
