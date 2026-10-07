# Private-core spatial-M experiment (2026-09-22)

This is a new opt-in architecture. Legacy requests omit `fabric.private_memories`
and retain their frozen shared-accumulator implementation. No RTL or PPA claim.

## Datapath and ownership

Shared native HBM2 -> shared finite weight DMA -> private weight SRAM write port
-> private weight SRAM read port -> core MAC pipeline -> private FP32 accumulator.
The shared on-chip activation source similarly feeds each private activation
stage through a write port, then a read port. Each SRAM is modeled as 1R1W.
The accumulator is one serialized RMW port **per core**, not a global port.
The shared charged descriptor/control port remains finite and unchanged.

An output row/N-tile is assigned once, before its first K segment. All K segments
and their ordered FP32 RMWs stay on that core. Pinned-expert mode retains whole
expert assignment. The `tile_stealing` API in private mode steals **unowned output
blocks only**; it cannot steal intermediate K segments or access another core's
partial sums. This restriction is reported as output-block affinity. Completed
outputs remain in their private accumulator until phase drain; host gathering is
untimed readout, not a simulated inter-core transfer or next-phase delivery.

Actual BF16 activation values are copied on the private write-completion event.
MAC consumes those staged values and its own decoded weight slot. Numerical
FP32 partial sums are backed by separate per-core maps. A merged output exists
only after drain for comparison to the independent reference.

Each persistent output row/N-tile reserves `4*N_tile + 16` bytes in its core's
accumulator budget: padded FP32 values plus a 16-byte address/progress/flags
record. Software lookup structures represent these charged records and existing
queue descriptors; they are not a synthesized associative lookup circuit.
Selector/record service uses the existing charged control model. Metadata never
silently borrows another core's capacity. Infeasible pinned placements are errors.

## Primary equal-budget comparison

Hardware order is M_tile x N_tile x K_tile; N_tile=4, K_tile=512.
All have 12,288 multipliers, result latency 25 cycles, II=1, assumed 1 GHz.

|Resource|Single 6|Homogeneous 3+3|Heterogeneous 4+2|
|---|---:|---:|---:|
|Private weight SRAM / slots|48 KiB / 12|24+24 KiB / 6+6|24+24 KiB / 6+6|
|Private activation storage, 2 stages|12 KiB|6+6 KiB|8+4 KiB|
|Reserved in-flight FP32 result storage|2400 B|1200+1200 B|1600+800 B|
|Private accumulator incl. records|2 MiB|1+1 MiB|1,398,112 + 699,040 B (~2:1)|
|Accumulator RMW bytes/cycle|192|96+96|128+64|
|Local weight read AND write B/cycle|8192 each|4096+4096 each|4096+4096 each|
|Local activation read AND write B/cycle|6144 each|3072+3072 each|4096+2048 each|

Weight tile size depends on N*K, so it stays 4 KiB for every core. Equal weight
slots isolate M partitioning. Activation and result storage/ports scale with M.
Accumulator capacity is not forced to scale with M: it depends on the live
output set. The initial equal-capacity probe rejected B16 pinned 4+2 routed-down: the big core needs 1,081,344 B for its 66 token rows versus a 1,048,576 B allocation. The small core needs 491,520 B for 30 rows. The primary 4+2 comparison therefore partitions the same 2 MiB budget 2:1, rounded to 32 B; this design decision is explicit, and the rejected equal-capacity probe is retained. This is a complete core-plus-memory comparison, not an experiment changing only the MAC M dimension.
Aggregate bytes and service rates are equal, but replicated peripherals/ports
are not proven equal area. Wide SRAM interfaces are explicit architectural
assumptions requiring later banking/timing validation, not measured macros.

HBM configuration, BF16 weights/layout, shared delivery 1024 B/cycle, shared
activation ingress 6144 B/cycle, 256 descriptors and 32-key lookahead stay fixed.
These interface rates are **not HBM peak bandwidth**. Equal compute and modeled
storage/port budgets do not mean equal synthesized area or energy.

## Evaluation boundary

Reuse frozen DeepSeek-V2-Lite-Chat first-MoE-layer BFCL activations/routing and
checkpoint BF16 weights, B=2/4/8/16. Run routed/shared gate/up/down (six GEMMs),
three organizations, two assignment policies, twice. Native HBM feeds weight
requests live; activation source is on-chip. Independent numerical reference,
frozen output equality, native drain, capacity and service serialization audits
must pass. Sum of six cold-start phases is **not full layer/model latency**:
router, SiLU/gating, cross-phase activation delivery, combine and final output
export are not included. Never present these times as full end-to-end inference.
