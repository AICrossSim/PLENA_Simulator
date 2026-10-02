# Supply-first v3 engine: implemented contract and limits

This document describes the Rust cycle/event analytical engine, not synthesized RTL, native Ramulator timing, or a complete autoregressive model. The shared HBM path models finite 32-byte requests, a fixed response latency, aggregate bandwidth, ingress credits and return storage. Actual on-chip bank contention, byte leases, dependency timing, finite operand storage and control service are scheduled rather than appended to an analytical latency formula.

## Fixed hardware and ownership

- Main physical dimensions are `M × N × K`, with `N=4`, `K=512`; supported main examples are `[6]`, `[3,3]`, `[4,2]`, plus fixed-budget M8 variants.
- Physical capacity uses a fixed `t_chunk` of 128 or 96 according to the precision/rank/bandwidth working point. Batch size never changes the hardware capacity.
- The installed WOR capacity is 18 aggregate slots. `wor_tiles` defines the maximum executable N-group, not installed capacity. Slot bytes depend on the frozen format. Pool bytes, WOR slot bytes, actual live bytes and peak live bytes are separate fields.
- Generic WS and IS use one finite private accumulator arena per core. Two live execution contexts share that arena; all combined footprints must fit. A helper offload obtains a lease from the helper's same arena and releases it on completion. It cannot overlap an incompatible cold task.
- The specialized streaming core admits at most four rows and holds two whole-K-slice input buffers, totaling 8 KiB. Generic switchable cores admit IS when `Me <= M` and the actual full projection's partial sums fit their fixed private arena; otherwise they use WS.
- The private accumulator/source port width is fixed at `max(128,32*M)` bytes per cycle. Partial-sum RMW, consumer reads, deferred SiLU transfers and helper-return accesses share this port. Width and capacity are reported per core. No independent free source port is assumed.
- Two finite U-cache entries per core are banked SRAM inside the existing 16 KiB control reserve, not extra SRAM and not free register files. Local reads/writes have explicit fixed bandwidth and real fill/issue delays. A calibrated rank LUT occupies 6,186 bytes in this reserve when enabled; energy entries are BF16 and costs are uint32.

## Scheduled consumers of committed partial sums

Every row below waits for pending dot/accumulator commits and pays the finite source port before using committed data.

| Actor | Actual scheduled source and destination |
|---|---|
| Inline SiLU/vector | Read Gate/Up FP32 from the existing private arena, execute vector service, write BF16 Z to the shared activation arena. IS reads existing full-width G/U backing; WS reads its bounded output group. |
| Non-inline SiLU, WS | Read the original RF group, write a second group-sized FP32 scratch slice in the same private arena, read that slice, execute vector service, write Z. This is a finite group-local deferred variant, not an unbudgeted whole-expert global G/U array. |
| Non-inline SiLU, IS | Read the existing full-width FP32 IS backing; no duplicate whole-expert scratch allocation. |
| `UStore`, Gate/Up prepass | Read the owner's or helper's private FP32 prepass output, execute BF16 conversion, write packed logical-rank-permuted U through shared activation ports. |
| `UAccumulate`, Down prepass | Read the private RF delta, read/write the allocated shared FP32 U_d backing, then convert/store BF16 U after the prepass completes. |
| `Combine` | Read the private Down group, acquire the global combine RMW lease, read and write the output values atomically. Addition order is the actual lease/write-completion order. |
| Offload tail return | Read the helper's leased RF delta, pay cross-core copy service, read the owner's old partial sum, and write the merged partial sum through the owner's same finite private port. |

## Dataflow and progress

WS Gate/Up uses at most eight N tiles in a bounded group and reuses each X slice across that group. Each resident weight tile serves the required M blocks before release. A_gu rank prepasses use a bounded rank-group outer loop and ascending K inner loop so FP32 U intermediates fit the charged group arena.

Full-Z WS Down executes `for N_group: for K_segment: issue group; rank tails; Combine group`. A default G=8 group covers at most 32 output columns. Streamed WS Down executes a bounded N group for each available K segment, then performs a partial Combine; late correction also combines bounded groups. It never retains an unbudgeted private `Me × hidden` output. IS keeps one token slice resident while the K segment's N tiles stream through.

Current/Next may alternate only at a physically clean group boundary. A context cannot park partially decoded future WOR data that removes the other context's required working slots. Current reserves enough WOR slots to finish its complete group. Next admission is bounded to its next complete group and does not require both contexts' complete groups to fit the shared pool simultaneously. Pool byte leases are retained until actual last consumers or discarded responses drain; they are not released early to avoid a stall.

Pool quotas are targets for unreserved storage. When a stage transition shrinks a quota below already held Next bytes, the Current's immediate required group may borrow actually free pool bytes; this cannot admit future lookahead or Next beyond its quota, never exceeds physical pool capacity, and retains all existing leases. `quota_progress_borrows` counts these bounded admissions.

All numerical paths retain a 20-cycle default dot latency even under ideal supply. Per-output K order and pending-result availability gate issue; one-cycle issue throughput is not one-cycle dot completion. The oracle `dot_latency=1` is explicit and is not a physical performance claim.

## Mechanism switches

| Switch | Disabled behavior |
|---|---|
| Byte pool | Uses the same aggregate capacity as fixed 4 KiB slots in static equal per-core partitions. Reserved and wire-payload bytes are separate; short compressed tiles still occupy a complete slot. Releases and discarded responses retire reserved bytes. Current progress borrowing cannot cross another core's partition. This is a same-capacity slot-policy comparison, not the old legacy five-slot hardware. |
| Wide ports | Uses modeled narrower physical widths, reported in the ledger. |
| X reuse | Invalidates/loads X according to the non-reusing issue stream; actual bank transfers remain charged. |
| Weight reuse | Each real M block reads/fills/decodes the weight again through finite ports; WOR is retired after that M block and refilled before the next one. The final refill releases the pool lease; prior copies retain it. The weight's HBM acquisition remains unique. |
| Supply pipeline | Current-only exact head issue; next operand reads wait for previous dot and accumulator completion. No lookahead preloads or fake post-hoc latency gap. |
| Inline SiLU | Uses the actual finite deferred group-local source/copy/read actor described above. |
| Prefetch quota | Current-only exact current group admission. Next cannot borrow the pool when this switch is off. `little` and `fixed_one_tile` are distinct enabled policies. |
| IPD dispatch | Uses the selected real FIFO, random, EFT or joint policy. Joint uses a paid window snapshot, critical/aged anchor, maximum-cardinality pair and minimum maximum completion time, preserving up to two assignments until legal revalidation. It is not an EFT alias. |

Dynamic ranks use the same common per-expert candidate rank as the compiler quality policy, clip each projection to capacity, read BF16 energy/uint32 cost entries with charged comparison service and update lambda from routed-only factor bytes by default. Shared's fixed cost is reported separately. A lambda excursion outside its calibration bounds resets to lambda0.

## Numerical and reporting meaning

The optional numerical payload executes main, A and B arithmetic on the actual timed issue trajectory and executes SiLU/Combine on the corresponding completed actors. Gold output is only an independent comparison target. Packed MX reconstruction is independently tested in the compiler before supplying dequantized tensors. Physical DMA values remain timing payloads, so this is not a raw memory-bit RTL simulator.

`comp_equal_bytes` is an equal-total-wire diagnostic, not an identical-layout arithmetic-only experiment. Each compensation mode retains its native tile size and fetch order; the engine enumerates legal modes for that expert/organization, takes the maximum native total, and appends real 32-byte-request padding groups at the expert's end to smaller plans. Padding occupies HBM credits, ingress, pool storage and pool-read bandwidth, but does not occupy WOR, decode, or arithmetic issue. Its position can affect the completion tail. `unique_weight_bytes` excludes this padding. Any N2 conclusion must distinguish equal total wire bytes from equal per-tile bytes, equal fetch ordering and equal physical formats.

`onchip_traffic_bytes` and `all_onchip_traffic_bytes` now sum all measured SRAM/RF endpoint reads/writes, ingress FIFO read/write, WOR/XOR broadcast reads and explicit cross-core wire copies. `onchip_movement_breakdown_bytes` sums exactly to this total. Individual PE internal wires are outside this statistic. `legacy_onchip_subset_bytes` preserves the prior incomplete subset only as a diagnostic.

`group_wall_elapsed_histogram` contains elapsed first-to-last group issue time, including context gaps. `issue_to_accumulator_completion_histogram` is scheduled issue-to-commit latency. `pool_to_wor_elapsed_histogram` measures actual pool-read start to decoded WOR readiness, including contention and decoder queueing. None is relabeled as a bare arithmetic service latency.

The engine asserts request drainage, byte conservation, actual storage capacity, atomic Combine ownership, per-output K order, routed gather addresses and release of all byte/operand/context leases before returning a successful report. Main and auxiliary useful/padding MACs are separate; rank computation is not labeled tail waste.
