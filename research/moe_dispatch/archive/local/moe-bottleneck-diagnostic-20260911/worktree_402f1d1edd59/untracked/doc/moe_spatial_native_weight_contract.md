# Spatial-M native weight timing bridge

This opt-in bridge connects the existing physical-M finite fabric to the real
Ramulator C API. It does **not** implement a complete MoE layer or transformer
model timing graph. The previous default finite-interface model is unchanged.

## Measurement boundaries and units

- `total_cycles`: core cycles from the ready GEMM job list to all output commits,
  descriptor retirements and weight requests drained.
- `total_time_ps = total_cycles * core_period_ps`.
- `time_us = total_time_ps / 1e6`; `time_ms = total_time_ps / 1e9`.
- Default core period 1000 ps (1 GHz) is an explicit architecture assumption,
  not a synthesis result or measured silicon frequency.
- Six isolated routed/shared gate/up/down calls may be added as a **serial GEMM
  phase sum**. Each starts a fresh native memory model; this is not the critical
  path of a complete layer, nor a warm-state cross-phase simulation.

Previous tables used cycles or thousands of cycles. For example, 408.5 thousand
cycles means approximately 408,500 cycles, 408.5 us at the assumed 1 GHz, or
817 us at 500 MHz. Those previous results did not have native HBM timing.

## Live causal chain

Finite admission / output credit and weight-slot reservation
→ control issue service
→ 2D BF16 weight descriptor
→ finite accepted-sector credits and per-channel issue arbitration
→ native HBM2 controller acceptance, queueing, bank/row scheduling and callback
→ finite decoded weight delivery port
→ control install
→ MAC issue (also requires activation and preceding K result readiness)
→ accumulator port / commit.

The simulator advances the raw native memory clock in picoseconds. At each core
edge it processes callbacks for all elapsed memory edges, then attempts at most
one sector per channel. New descriptors are attempted on the following core
edge. Sampling rounds completion up to a core edge. No precomputed traffic trace
or post-hoc addition of independently measured memory time is used.

A multicast joiner shares its tile's outstanding transfer. Source completion
must precede delivery and install; weight slots remain reserved while waiting.
The original result registers, SRAM capacities, control costs, dependency rules,
and BF16/FP32 numerical datapath remain active.

## Native configuration and addresses

- Existing HBM2 preset: 2000 MT/s, 1000 ps memory period, HBM12 controller,
  FRFCFS, open-row policy, all-bank refresh, MOP4CLXOR local mapper.
- Default 8 channels; 32 B native sectors; channel = `(address >> 5) & 7`.
- 256 accepted outstanding sectors; per-channel tile-order FIFO admission.
  This is a new synchronous bridge, **not** a claim of identical arbitration to
  the older async normal-core runner's frontend.
- W is uncompressed BF16 `[N,K]`; row stride `ceil(2*K/32)*32` bytes.
- Expert base = logical expert ID × 16 MiB. IDs do not compact with routing.
  The real DeepSeek shared expert's old test sentinel 10000 is explicitly mapped
  to physical bank ID 64. Routed IDs 0–63 stay unchanged.
- Native row-tail sector padding is charged; native traffic need not equal the
  old interface's aggregate tile rounding for arbitrary tails.
- Each tested projection is isolated: gate/up/down do not share a persistent
  physical bank or warm controller instance across phases.
- SHA256-verified BF16 operand files supply numerical values; Ramulator models
  request timing, not stored data contents. Tile values cannot be installed
  before the corresponding native reads return.

The finite decoded interface remains 1024 B/core-cycle downstream of native
memory. It is not a declaration that HBM sustains 1024 B/core-cycle. Native
latency must not be compared to old finite latency as a same-bandwidth speedup.

Host-expanded sector addresses represent 2D descriptors. This prototype does
not claim a synthesized DMA frontend, final metadata area, equal-area results,
or fully calibrated controller implementation cost.

## Deliberately not included in this bridge

X and output accumulation use the existing on-chip activation and accumulator
ports. No native activation spilling, input upload, output writeback, MX/scale
codec, router projection, softmax/top-k, token rearrangement, SiLU, merge,
attention, normalization or cross-layer KV traffic is added here.

Existing numerical MoE tests perform nonlinear and weighted-merge operations on
the host and use actual router outputs. That establishes numerical coverage,
not simulated hardware timing for those operations.

A future complete-layer runner needs a single persistent timing graph and
explicit resource specifications for router/vector operations, dispatch/merge,
interphase storage, shared/routed overlap, and input/output residency. A whole
transformer model also needs attention, norms, KV/cache policy, every layer and
output head. Neither can be replaced by summing a router estimate into this table.

## Verification

Native requests must all complete; C API served counts must match wrapper
completions. SRAM peaks and sector credits stay within limits. Outputs must match
both the independent arithmetic reference and the previously archived real BF16
projections exactly. Repeated native runs must match all result bytes, including
native telemetry and request-order hashes. Defaults must reproduce the frozen
finite-interface output without any changes.
