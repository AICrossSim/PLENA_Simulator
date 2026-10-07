# Bounded X reuse in the structural candidate model

This mechanism applies equally to M=6, 3+3 and 4+2 with Nt=4, Kt=512.
It changes the compiler's local loop traversal and exact-tag X-slot reuse only.
It does not change whole-N-band ownership, HBM preset/layout/precision, native
credits, private payload capacities, arithmetic latencies or accumulator rules.
It is a candidate compiler schedule adapter plus simulator mechanism, not a
main-Compiler ISA integration, calibrated chip model or complete model inference.

## Frozen baseline and opt-in mode

`Hardware.x_reuse_bands = 0` (also the serde default) retains the frozen tile-major
execution and prefetch policy. `x_reuse_bands = 4` selects the new policy:

1. Keep the original per-core weight descriptor order.
2. If ceil(Me / Mc) <= x_slots_per_core, all the expert's X rows for one K segment
   fit in the local slots: retain the tile-major M traversal and W gather reuse.
3. Otherwise group at most four consecutive same-expert, same-K weight tiles.
   Traverse M blocks outside the group's N bands. Every W tile remains reserved
   until its final M-block operand read. All K updates of an output remain local
   and in increasing K order, waiting for the preceding commit as before.
4. Use the existing two X slots and tag lookup. If the desired tag is still in a
   busy slot, wait for its operand read to finish; do not duplicate the same X
   into the other slot. A free matching slot reuses data; a miss goes through the
   unchanged shared on-chip producer bus and private 1RW banks.

This is a static loop sequencer, NOT ready-window out-of-order issue or an
urgency-aware HBM arbiter. It cannot skip the scheduled target because another
tile happens to return sooner. Future HBM completion times never guide mapping.
Only one task preparation is allowed per core per cycle, as in the baseline.
The target tile is resolved through a bounded private-slot sequence-tag lookup.
The group length fits the existing 32-byte descriptor; its existing control-port
service is preserved. This is a candidate timing assumption, not a synthesized
claim about comparator timing.

Grouping must fit both the private W slots and the bounded N-band window; invalid
requests fail. No payload is shared between cores, and no output is moved. The
final short group does not load extra tiles. Groups cannot cross expert or K
boundaries. The scheme can trade fewer X writes for more W gather reads and
longer weight-slot residency; no latency improvement is guaranteed.

## Resource accounting

The previous 4 KiB control reservation inside the 2 MiB output/accumulator budget
is unchanged. New control state is conservatively charged as four 64-bit cursor
registers/core plus two 32-bit tags/private W slot. Runtime counters used solely
for simulator diagnostics are not proposed hardware state.

| Organization | Added state | Old control state | New state | Old / new reserve headroom |
|---|---:|---:|---:|---:|
| 6, 10 private W slots | 112 B | 1088 B | 1200 B | 3008 / 2896 B |
| 3+3 or 4+2, 5+5 slots | 144 B | 1728 B | 1872 B | 2368 / 2224 B |

Each core keeps 2 X stages; weights keep 10 or 5+5 slots; the 8 KiB shared native
response area is still inside the 48 KiB W budget. All original bank conflicts,
finite result contexts and accumulator RMW service remain charged. No equal-area
claim is made from equal multiplier and SRAM counts.

## Measurements

New counters expose X tag hits/misses and bytes per core; cycles waiting for a
busy matching X tag; and front-task weight waits split into transport-not-complete
and arrived-but-private-write-not-complete. Transport includes submission/native
queueing and DRAM: it must NOT be labeled pure HBM latency. Busy-tag cycles can
overlap operand feeding. Arithmetic activity, bank service waits and frontend
states are different axes and cannot be summed into a wall-time breakdown.

Input study: frozen DeepSeek-V2-Lite-Chat first MoE layer, BFCL B2/B4/B8/B16,
shared/routed gate/up/down (six separately cold-started GEMMs). Native HBM covers
weight reads; X moves on chip. Router, nonlinear, combine and inter-phase residency
are excluded. Cycles are primary; milliseconds assume 1 GHz.

New reports reside in outputs/moe_x_reuse_20260925. Each point repeats twice,
checks exact outputs against frozen operands/reference and old report, and checks
HBM bytes, K order, storage/context limits and native/consumer drain. Mode 0 must
exactly reproduce old cycles, frontend states, bank service and native report.
