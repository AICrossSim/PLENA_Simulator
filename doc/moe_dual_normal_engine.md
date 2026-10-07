# Normal-buffer MoE engine contract

Implementation: `transactional_emulator/src/moe_normal/`; entry point:
`transactional_emulator/src/bin/moe_dual_normal.rs`. This is a dedicated numerical
architecture experiment. It does not change the legacy ISA decoder or claim to
execute the current RTL's datapath and timing.

## Public interface and data boundary

`run(workload, architecture, Arc<dyn memory::ErasedMemoryModel>, hbm_len)` validates
the complete workload, creates a `runtime::Executor`, and returns a `RunReport`.
`execute` provides the same future for a caller that already owns an executor.
`validate` is a read-only preflight entry point. Version 1 JSON types live in
`types.rs`; unknown fields fail except the explicitly accepted, uninterpreted
`metadata` and `grouped_routes` values.

One HBM object is shared by every core and every outstanding request. The CLI
uses actual `MemoryBacked` bytes with a shared Ramulator timing model. The engine
requires the complete 64-byte-aligned image length and checks every matrix's
element/scale address range before any read. `MemoryBacked`'s zero-on-out-of-range
behavior is not an accepted substitute for valid data.

Inputs and routes are ready at time zero. Runtime code rebuilds expert groups
from the route list; it does not trust supplied grouped-route annotations. It
sorts routes by `(token, slot)`, groups them by expert, and assigns an entire group
to `large_core` when `M >= dispatch_threshold`, otherwise to `small_core`.
This is the default `threshold` policy. With `work_conserving`, each free core
claims a fitting job from a shared ready list: its own M class first, then any
other fitting group, with ascending job id breaking ties. A running group is
not migrated or split. The shared dispatcher has one permit and charges
`dispatch_cycles` per successful claim. Each ready-job descriptor reserves 64
bytes against `dispatch_queue_bytes`; initialization/group construction is
outside measured time. An optional shared
expert adds one all-token job after the routed jobs. Repeated `(token, expert)`
routes with different slots are supported as distinct rows; repeated
`(token, slot)` is an error. No expert is permanently bound to a physical core.

Measured time includes input gather, MX reads and decode, gate/up/down numerical
computation, SwiGLU, result copy, and deterministic weighted combine. It excludes
initial input/weight placement, router execution, host grouping/dispatch
construction, and the final output HBM store. This is one fixed-route MoE stage,
not an agent trajectory or a complete autoregressive model execution.

## Numerical operations and weight layout

Matrices are output-major `[N, K]`. `MatrixRegion` carries shape, element base,
scale base, and independent row strides. K is padded to a multiple of eight in
HBM; each row has `ceil(K / 8)` one-byte scales. Gate and Up are `[E, D]`, Down is
`[D, E]`. The format is the repository's local E4M3/E8M0 block-8 convention,
including its existing subnormal semantics. It is not an OCP MX conformance
claim.

HBM bytes are decoded through `quantize::DataType::convert_bits_to_f32`, then
rounded into actual `Vec<half::bf16>` normal-buffer storage. Decode uses scalar
FP32 temporaries rather than an uncharged full FP32 tile. Non-finite decoded
weights and results return errors.

Each projection accumulates in FP32, with separate multiply/add in ascending K,
then rounds its complete result once to BF16. Gate and Up remain BF16.
`Z = BF16((gate / (1 + exp(-gate))) * up)` uses FP32 intermediate arithmetic.
Down emits BF16. Weighted route outputs accumulate in FP32 in sorted token/slot
order, with the shared expert last for each token, then round to BF16.
`output_f32` is a decoded view of the final BF16 output;
`pre_round_output_f32` is the combine sum before final rounding. Scalar dot-product
order is an explicit numerical contract; the RTL's reduction-tree rounding has
not been checked against it.

## Finite physical data storage

Let D be model width, E expert hidden width, M one expert group's row count,
B the core's BLEN, Kt its MLEN, and P the configured pipeline overhead plus
`ceil(log2(Kt))`.

| Region | Capacity required while active | Lifetime |
|---|---:|---|
| Core vector SRAM | `2 * M * (2D + 3E)` bytes | BF16 X, Gate, Up, Z and output reserved together for one job |
| Core accumulator/result pipeline | `4 * M * max(D,E) + 4 * B * P` bytes | One FP32 projection result plus fixed in-flight result registers |
| One weight slot | `B * Kt * 2 + B * Kt + B * Kt / 8` bytes | Decoded BF16 plus packed elements/scales |
| Core weight SRAM | Two complete weight slots | Current values remain owned until all M blocks have consumed them |
| Global DMA staging | `64 * global_dma_credits` bytes | Loader transaction credit held through cache lookup or HBM miss, insertion and tile copy |
| Per-core read cache | `80 * floor(read_cache_bytes / 80)` bytes | Each resident FIFO entry holds 64 data bytes and 16 tag/control bytes |
| Shared dispatcher descriptors | `64 * job_count` bytes | Reserved for all expert groups before launch |
| Shared input/combine SRAM | `8 * T * D + 2 * R * D` bytes | Ready input BF16, output FP32 sum, final BF16, all R BF16 route results |

Here T is token count and R includes T additional rows when a shared expert is
present. SRAM admission is checked before execution; insufficient capacity
fails explicitly. V0 does not spill an expert group or tile M for capacity.
The reported shared-SRAM peak is the full reservation: all route output slots
are reserved before any task starts. It is a conservative live-data policy.

Per-core X/Gate/Up/Z/output are actual separate BF16 vectors, and the FP32
accumulator is an actual per-core vector. The result pipeline uses readiness
timestamps over that accumulator, with its separately charged register capacity;
it is not a second unbounded collection of results. Core buffers are freed only
after their output has been copied into shared reorder storage. Reordering can
therefore absorb out-of-order completions without hidden unlimited output memory.

Pending DMA request descriptors and route/scoreboard metadata are host control
structures. They hold no uncharged tensor payload. Their finite descriptor count
depends on the current tiles/workload; control-SRAM area and RTL descriptor queue
sizing beyond the explicitly charged job descriptors are not modeled and must
be added before an area claim. The one-cycle ready selection is an analytical
priority-selection assumption, not a synthesized scheduler.

## Double buffering, HBM competition and vector work

Each core owns at most two weight-slot reservations: one consumed tile and one
loading/ready tile. The loader may prefetch the next N/K tile while the current
tile is being consumed across all M blocks. An optional finite per-core FIFO
line cache retains read-only weight bursts across tiles/jobs. Zero bytes disables
it; a nonzero capacity below one 80-byte entry fails validation. Entries are
evicted in insertion order, not access order. Duplicate concurrent misses are
not coalesced; all actual duplicate HBM requests are charged.
The tile releases its slot only after its last operand has entered computation.

Element and scale reads are coalesced by 64-byte address within a tile, including
unaligned subranges. All cores share one DMA semaphore. A permit covers the real
cache lookup, HBM read on a miss, cache insertion, and copying the 64-byte response
into the finite packed tile. Each cache has one serialized port, charging one
cycle per lookup and per insertion, shared by both weight slots. A permit
is not released early into an unbounded response queue. Request/byte counters are
updated on actual reads, and tests compare them with `memory::WithStats`.
`global_dma_inflight_peak` is the peak number of loader transaction credits,
including cache hits; it is an upper bound, not the exact number of HBM misses
outstanding. Cache requests equal hits plus actual HBM bytes divided by 64.

One shared vector actor provides configured `vector_elements_per_cycle` service
for input gather, weight decode/BF16 placement, SwiGLU, result copy, weighted
combine, and final rounding. Each whole operation takes
`ceil(elements / lanes)` cycles and holds the actor for that service. These are
analytical operator-throughput assumptions, not measurements of a real exponent
or decode unit. All cores and the baseline use the same actor definition.

## Computation latency and issue interval

Square macro tiles use `M = N = BLEN`, `K = MLEN`; multiplier count is
`BLEN * MLEN`. MLEN must be a positive multiple of eight and divisible by BLEN.
The latter retains the existing square MatrixMachine mapping: `mm()` slices
BLEN-wide column groups from an MLEN-square SRAM tile (`matrix_machine.rs`, the
column-offset decode and slice in `mm`). The V0 local loops could support other
ratios, but admitting those would require a separately stated mapping extension.
Non-power-of-two MLEN is accepted; reduction latency uses ceiling log2.

In the default `pipelined` mode each issued macro tile consumes BLEN cycles at the multiplier resource. Its
result becomes ready after a further `ceil(log2(MLEN)) + mac_pipeline_cycles`.
A timestamp per M/N output block prevents the next K contribution from reading
that accumulator before its previous contribution is complete. Independent
output blocks can issue while earlier results remain in flight; their complete
FP32 output storage is already reserved. The projection waits for the final
result to drain before publishing BF16 output.

The `legacy_serialized` sensitivity instead charges `MLEN + mac_pipeline_cycles`
per macro tile and no additional pipeline delay/storage. It follows the old
MatrixMachine's instruction-service formula, but does not reproduce that
machine's SRAM operations, ISA or exact configured overhead. Comparisons must
use the same timing mode and overhead for every architecture. See
[timing audit](moe_full_shape_timing_audit.md).

Thus service interval and completion latency are separate. The engine neither
charges every tile an unconditional full pipeline drain nor assumes infinitely
many accumulator contexts. `compute_busy_ps` measures multiplier service;
`accumulator_dependency_stall_ps`, `pipeline_drain_ps` and
`weight_ready_wait_ps` explain other time. Vector-wait counters include loader
and core waits and can overlap; do not add all reported waits to obtain total
time. Actual elapsed time comes from the shared executor.

This service model assumes each active core can supply MLEN BF16 input values
per cycle, broadcast to BLEN output lanes with a resident weight tile. It does
not yet model SRAM bank conflicts, crossbar arbitration, or SRAM/accumulator
read/write port implementation. Those bandwidth resources must be listed when
comparing architectures, even when their multiplier totals are equal.

## Error handling and verification limits

Preflight rejects unsupported schema, invalid or duplicate route slots,
non-finite ready inputs/route weights, malformed matrix shapes/strides, invalid
HBM ranges, invalid core geometry, insufficient finite storage, and model-local
counter/duration overflow. Core queues are all awaited even when one returns a
numerical error. Loader errors occur after their issued reads have drained;
slot reservations release through ownership/drop. An early backend panic is a
backend failure, not a successful numerical result; the memory trait has no
recoverable read-error return type.

Unit tests use nonzero encoded weights with nonzero padding canaries, a shared
timed memory backing, and an independent scalar golden. They cover different
core shapes, non-power-of-two MLEN, three projections and SwiGLU, duplicate expert
routes, shared expert, route-order invariance, out-of-order completion, credit
limits, SRAM failures, malformed addresses and extreme clock rejection. The CLI
additionally checks the compiler-exported nonzero fixture against a separate
golden under Ramulator.

The engine establishes a finite numerical experiment for the first meeting
milestone. It does not establish an RTL-ready complete chip, a full-model
speedup, an iso-area result, or a final optimal big/small split. Transpose SRAM,
additional small cores, shared on-chip storage, realistic scheduler/descriptor
costs, and hardware-calibrated timing remain explicit later work.
