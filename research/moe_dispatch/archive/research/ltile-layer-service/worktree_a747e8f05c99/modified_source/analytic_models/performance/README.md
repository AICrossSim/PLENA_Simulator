# Compiled analytical decode model

`ltile_decode unified` composes the same instruction service model used by the
connected numerical sublayer runner. It predicts decode time; it does not run
the complete model numerically, certify task quality, or estimate power/TTFT.
The older `ltile_decode` entry point remains for historical reproduction.

## Execution contract

`ExecutionProfile` is the common input to the numerical runner and analytical
services: L=256, update II=2/latency=6, BF16 state and tree reduction, FP32 fused
update intermediates, bounded SFU lanes, Matrix service, DMA credits and codec
buffers. Profile/source/assembly/memory hashes travel with results. Arithmetic
uses the compiled rational delta producer, not free GPU-prepared coefficients.

The analytical input is a compiler schedule with concrete local addresses.
`ltile_cost` counts instruction, SRAM, arithmetic and dependency services;
Ramulator receives the ordered DMA addresses and intervening compute delays.
`ltile_execution.price_program` joins the ledgers. No measured cycles are inputs
to prediction. The Rust and Python memory implementations share Ramulator, so
agreement verifies integration rather than independently validating DRAM.
The exclusive total is `issue + scalar + sram + arithmetic + dependency + dma`.
`frontend` is a convenience subtotal (`issue + scalar`), not an extra term.

`ltile_layers` produces connected Mamba/KDA sublayers from shapes. Coefficient
packing reuses repeated masks, source rows and common output-row contributions;
the old recurrent ISA instead caches compact
coefficients and broadcasts using ordinary Vector instructions. Projections
share Matrix weight panels across batch requests. `lower_resident_projection`
keeps one output row per request and a bounded cache of full input windows,
then writes each completed output row once. It owns Vector rows 0..57;
rows 58..63 are reserved for the 24 KiB weight decoder. Cache misses still DMA
full owned windows; there is no assumed unaligned read or left shift. Panels
that exceed Matrix capacity fall back to serial requests with weight rereads.
This schedule is shared by all four comparison arms. Historical `stream`
projection and `pattern` gather remain selectable through `build_layer` and
`build_batch` for ablation and reproduction. These are legal, conservative
compiled schedules, not a proof of globally optimal Matrix scheduling.

The default comparisons are:

| Control | Recurrence | Purpose |
| --- | --- | --- |
| `old_isa` | BF16 ordinary updates and SRAM tree | Optimized software recurrence |
| `row` | Fused FP32 update, BF16 tree, explicit row control | Same arithmetic/data path as FSM |
| `fsm` | Same fused operations, bounded traversal | Incremental control benefit |
| `native` | Same fused arithmetic/tree, original compact coefficient layout | Descriptor supply versus software packing |

All share projections, memory, capacity, SFU and weight decoding. `old_isa` is
the old **recurrent** ISA on this common platform, not unmodified original PLENA.
Do not describe old-ISA versus fused-update comparisons as identical arithmetic.
`fsm` versus `native` isolates supply at the same arithmetic. `row` versus
`native` changes both control and supply and is not a pure FSM ablation.

## Weight and memory accounting

NVFP4 is block16 E2M1 with E4M3 block scales and a tensor scale. Matrix packet
padding, global-scale traffic, finite unpack/scale service and input/output
staging are charged. The decoder throughput is a candidate parameter, not RTL
evidence; BF16 numerical sublayers still use offline decoded weights. Decoder
buffers occupy existing Vector SRAM during projection and cannot overlap its
live rows. No activation/state format changes when weight storage changes.

The 16-controller profile is a **new common 32 GiB / 256 GB/s candidate** at
1 GHz. DMA 32/32 is also a candidate, not an audited original PLENA capability.
Both comparison arms use the same settings. Capacity comes from the Ramulator
organization and a persistent tensor inventory, not bytes accessed per step.
Capacity-invalid points are skipped; `--include-infeasible` only emits conditional
diagnostics, never single-device results. No external weight supply is free.

## Run

From the Simulator root, with Compiler and archived routing data available:

```sh
nix develop -c python -m analytic_models.performance.ltile_dma \
  --prepare "$OUTPUT/memory32" --controllers 16
python -m analytic_models.performance.ltile_decode unified \
  --compiler "$COMPILER" --memory-root "$OUTPUT/memory32" \
  --campaign "$AGENTIC_CAMPAIGN" --output "$OUTPUT/decode"
```

The default grid is B1/2/4/8/16 and contexts 4096/32768/131072. Output separates
batch step time, per-request rate, aggregate TPS, exclusive operator/cycle
components, HBM traffic, tensor inventory and capacity. Nemotron uses archived
B1 routes grouped into batches; context sweeps reuse those routes. Kimi routing
is a min/max occupancy scenario. This is not new batched GPU capture.
Context denotes the number of keys attended in the modeled step, including
the current appended key. Its zero-based append position is `context - 1`.

`--codec-lanes 128|256|512`, `--sfu-scale 1|2` and `--dma-window 1..64` expose
uncertain service assumptions without changing one comparison arm in isolation.
`--supply native` is the default; `--supply packed` retains the three-arm graph.

## Validation and limits

Use `transactional_emulator.testbench.models.unified_service_test` for numerical machine-code checks
of shared weight panels, private requests, K/N tails, coefficient packing,
compact old-ISA broadcast, softmax, positive score normalization, dot and ReLU².
Its `--only attention` case connects prepared-Q/static-KV QK, softmax and PV
in a single program with no host writes between stages (two queries, 4K keys).
This verifies the attention core, not incremental KV packing, query projection,
RoPE, MLA, routing or a complete attention block.
`--only experts` connects fixed-route expert up projection, ReLU², down
projection and weighted combine for two tokens sharing one of three experts.
Weights and activations flow through the program; dynamic router selection
remains outside that test.
`--only resident` checks cache misses at B16, K/N tails across output-row
boundaries, oversized-panel fallback, and repeated gather subsets under cache
pressure, through actual machine code. It compares all cost components and
HBM transfers, not just arithmetic output.
`transactional_emulator.testbench.models.connected_peripheral_test` adds
continuous machine-code programs for projected GQA with private KV append,
representative MLA, fixed-route experts with weighted combine, decoded NVFP4
weight projections, and two-token private KDA states with reversed request order.
`--case attention --model-shape --batch 2 --keys 4096` checks Nemotron GQA
dimensions; `--case experts --model-shape --batch 16` checks its 2688/1856 expert
widths and cross-row combine. These use synthetic values. The expert case is a
three-expert subset with fixed routing, not a full top6/router quality test.
NVFP4 decoding is independently checked in Python before Rust projection;
the decoder itself is not an executed Rust instruction.

Native Mamba and KDA B1/2/4/8/16 real-weight sublayers have executed with private
request storage and shared weight panels. Their seven timing components and
physical HBM bytes match the shared analytical service. Batch members repeat
the captured B1 input; independent private-state tests use distinct values.
Exact agreement is with the declared arithmetic, not original-model FP32
accuracy or long-sequence task quality.

Whole-model results are deliberately `analytical_candidate`. A service cache
starts Ramulator afresh between independent operators; that is not equivalent
to arbitrary continuous whole-model memory history. Held-out composition and
address-phase checks must be reported separately from recurrent calibration.
KV append now compiles gather plus masked Vector read/modify/write instructions,
with owned rows, immutable masks, and measured DMA. Router selection remains a
named finite orchestration budget; its 0x/2x sensitivity is reported. Connected
programs test representative peripheral boundaries, while full-shape coverage,
global buffer relocation and routing quality remain incomplete. The existing
64-bit HBM base registers can support addressing beyond the local 32-bit GP
offsets, but a complete global placement/base-switch schedule is not yet emitted.
No offload-free execution or certified end-to-end speedup follows from this
candidate table alone.

## Candidate coefficient and state access study

`ltile_access` is an isolated, uncalibrated UPDATE-path design experiment. It
does not replace the compiled/Rust-checked R3 service model. It enumerates the
actual 64 banks of 32 BF16 elements, keeps both layouts inside the same 512 KiB
state allocation, and compares native head-major coefficient sectors with
row-packed coefficient pairs. A 32-element coefficient sector is explicitly
staged and charged; it is not a free arbitrary gather port.

The finite scheduler accounts for input slots, reserved output credits,
update II/latency, alignment delay, shared or 1R/1W arbitration, and write
acceptance. It retains destination metadata until the last write completes.
Periodic port blocking is a synthetic sensitivity, not measured DMA overlap.
State words still cost 64 bytes each: widening a read reduces service waves,
not the number of bank words or automatically the number of execution cycles.

The numerical check compares synthetic native-SRAM coefficient fetches with
independently packed BF16 operands and the same separate FP32 update arithmetic.
It is not Rust execution, a real-model precision gate, or a complete KDA pass.
Projection, coefficient production/DMA, dot/tree/residual, ISA encoding and
integrated routing/PPA remain outside this candidate. Preserve that scope when
using the generated buffer, wave-count or latency tables.

The separate compiled native path now lives in Compiler `ltile_native.py`,
Rust `accelerator/native_coeff.rs` and the existing `v2.rs` execution loop.
`build_layer/build_batch(..., native_coefficients=True)` emits CCFG descriptors
and native EXEC forms instead of `pack_update/pack_dot`. The unified decode
entry calls this same path through `Services.layer(..., supply="native")`;
configuration and result files explicitly record the selected supply. Small
dt/beta/skip gathering is still executed and charged.

`ltile_cost` models those executed operations with two finite state-input slots,
bounded result credits, real coefficient bank words, incremental addressing,
registered alignment/selection, shared Matrix service and the existing BF16
Vector tree. `Machine.native_read_width/native_result_slots` propagate through
the common execution profile. SRAM capacities and DMA service are unchanged.

`native_supply_test` runs real checkpoint-derived full-sublayer fixtures;
batch requests share projection weight panels but own private state. Its
BF16 weight execution must not be described as a runtime NVFP4 codec. Fixture
allocation holes preserve historical addresses only; native code never reads
or writes the removed packing arrays. The test compares all remaining arrays,
not merely a final checksum. Exact same-arithmetic agreement is a mechanism
test, not a new long-sequence task-quality result.
`--mapped-archive` validates full reconstructed image hashes before removing
duplicate raw images. The resulting archive depends on the recorded immutable
B1 fixtures; full-array comparison counts include unchanged weights.
