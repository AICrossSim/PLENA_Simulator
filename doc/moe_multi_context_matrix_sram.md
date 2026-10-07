# PLENA-MoE Multi-Context Matrix SRAM Design

## 1. What Is Proven Today

The current behavioral simulator constructs Matrix SRAM with `MLEN=64`,
`MATRIX_SRAM_SIZE=4096`, and BF16 storage. `MatrixSram::new()` creates
`4096 / 64 = 64` independently addressed cells. Each cell stores one
`64 x 64` BF16 tile, so the modeled payload is:

```text
64 cells x 64 x 64 elements x 2 bytes = 512 KiB
```

The compiler default `mram_tile_capacity=4` exposes only four cells to one
projection schedule. One compiler panel is therefore:

```text
HBM:  4 x 4,608 B MXFP8+scale tiles = 18,432 B
MRAM: 4 x 8,192 B BF16 cells       = 32,768 B
```

This is a simulator-derived contract, not yet an RTL SRAM-macro signoff.
`MatrixSram::size_in_bytes()` currently returns element count despite its name,
so reports must calculate the BF16 byte capacity explicitly.

## 2. Separation of Contributions

Two mechanisms solve different problems and must be evaluated separately:

1. **Expert grouping and weight reuse**: group all rows routed to the same
   expert, load that expert's weight once, and issue `ceil(M/BLEN)` row groups.
   This reduces physical HBM bytes.
2. **Multi-context Matrix SRAM**: keep pending/ready panels for several expert
   streams and allow DMA for one context while another context computes. This
   does not reduce bytes; it can reduce blocking and improve overlap.

Any speedup claim must identify which mechanism produced it.

## 3. Explored Multi-Context Organization

Preserve the same 64-cell, 512-KiB payload and add banked ownership metadata:

```text
                 asynchronous DMA descriptor queue
                              |
                              v
  +-------------------------------------------------------------+
  | 64 x (64x64 BF16) Matrix-SRAM cells, divided into banks     |
  |                                                             |
  | Context tag: {expert, projection, k_chunk, n_block, epoch}  |
  | State: FREE -> FILLING -> READY -> READING -> FREE          |
  |                                                             |
  | Ctx 0: panel A[4 cells] | panel B[4 cells]  (ping/pong)     |
  | Ctx 1: panel A[4 cells] | panel B[4 cells]                  |
  | ...                                                         |
  | Ctx 7: panel A[4 cells] | panel B[4 cells]                  |
  +---------------------------+---------------------------------+
                              |
                       Matrix issue arbiter
                              |
                         existing matrix unit
```

One explored alternative lets the cells form an elastic tagged pool rather
than permanently belonging to an expert. With four-cell panels, unchanged
capacity permits:

| Active contexts | Maximum panel depth per context |
|---:|---:|
| 1 | 16 |
| 2 | 8 |
| 4 | 4 |
| 8 | 2 |

The scheduler may use fewer contexts and deeper lookahead for hot/shared
experts, or eight ping-pong contexts for a queue of cold experts. This is a
storage policy, not a claim that eight compute cores exist. Section 8 records
that the event-level experiment did not find enough incremental benefit to
select this organization for version one.

### 3.1 Concrete bank placement

The evaluated physical candidate is eight 64-KiB banks, each holding eight
`64 x 64` BF16 cells. The 16 four-cell panels are phase-colored:

```text
panel 0,2,4,... : tile 0..3 -> banks 0..3
panel 1,3,5,... : tile 0..3 -> banks 4..7
row within bank : panel_id / 2
```

Each context receives one even and one odd panel. While the matrix unit reads
one color, DMA fills the opposite color. The allocator may lend either color
to any expert context, but cannot move a live panel or overwrite a non-`FREE`
tag. If the only free destination conflicts with the bank being read, DMA
stalls and charges a bank-conflict cycle. Thus the proposal requires neither a
larger payload nor an unpriced multi-port SRAM.

This mapping was a candidate to measure, not a fixed architectural conclusion.
An interleaved mapping and a static per-expert partition are required controls;
phase coloring survives only if it lowers measured conflict/stall cycles.

## 4. Dataflow

1. The compiler groups route pairs by expert and emits deterministic jobs.
2. A DMA descriptor names the HBM tensor slice and destination context tag.
3. The destination panel moves to `FILLING`; 64-B requests complete through
   Ramulator and the panel becomes `READY`.
4. The issue arbiter selects a ready panel whose accumulator dependency is
   available. The panel moves to `READING`.
5. While that panel computes, DMA fills the other panel of the same context or
   a different expert context.
6. After the final consumer, the panel becomes `FREE`; completion order may be
   arbitrary because tags, not physical order, identify the tensor slice.
7. Expert outputs retain token IDs and route weights and enter the existing
   deterministic scatter/combine path.

No reorder buffer is required for mathematical correctness. A small completion
table indexed by `{token_id, route_slot}` is sufficient; retirement may wait
until every contribution for a token is present.

## 5. Scheduling Policy

Use a work-conserving two-level policy:

```text
DMA priority:
  context with the smallest ready-compute runway first
  tie-break by monotonically increasing sequence_id

Compute priority:
  READY panel that continues an open accumulator first
  then oldest-ready sequence_id
```

This avoids nondeterministic timing while favoring both starvation prevention
and accumulator locality. Shared experts are ordinary tagged streams, but the
compiler may grant them more lookahead because they are guaranteed to execute.

## 6. Required Banking Contract

The current Rust `Vec<Mutex<Cell<_>>>` is functional storage, not a physical
multi-port SRAM model. The proposal is valid only after choosing and charging
for a banking organization. The minimum experiment set is:

| Candidate | Contexts | Depth | Required simultaneous operations |
|---|---:|---:|---|
| Current blocking | 1 | 1 | one DMA write or one matrix read |
| Single ping-pong | 1 | 2 | DMA write + matrix read to different banks |
| Static four-context | 4 | 2 | one DMA write + one matrix read, fixed ownership |
| Static eight-context | 8 | 2 | one DMA write + one matrix read, fixed ownership |
| Elastic tagged | 1-8 | 2-16 | one DMA write + one matrix read, dynamic ownership |

The first implementation should require only one aggregate read and one write
per cycle to different banks. It must stall on a same-bank conflict; it must
not assume a free 8R/8W macro or eight independent HBM ports.

## 7. Validation Funnel

### Gate A: Capacity and trace pressure

Run every saved Qwen and DeepSeek SWE decode route window. Report active expert
jobs, one-wave coverage, and scheduling waves for each organization. This only
proves that a layout fits and quantifies queue pressure.

### Gate B: Byte truth

For pair-major and expert-grouped programs, require formula physical bytes to
equal emulator `WithStats` bytes exactly. SRAM organization alone must leave
bytes unchanged.

### Gate C: Functional truth

Use nonzero inputs and weights, including groups larger than `BLEN=4`, and
require deterministic repeat-3 output with `rel_rms < 0.01`.

### Gate D: Event-level timing

Compare the following with the same route, bytes, 512-KiB payload, and matrix
unit:

```text
A: pair-major + blocking
B: expert-grouped + blocking
C: expert-grouped + one-context ping-pong
D: expert-grouped + static multi-context ping-pong
E: expert-grouped + elastic tagged contexts
```

Report total cycles, HBM starvation, overlap, bank conflicts, context
occupancy, and completion skew. `C-B` measures asynchronous prefetch;
`E-C` measures multi-context ownership. A design that only improves an
analytical capacity metric but not event-level makespan is rejected.

Use the same real-route windows to sweep the unchanged 16-panel payload:

| Organization | Live panels | Bank groups | Question isolated |
|---|---:|---:|---|
| Blocking | 1 | 1 | current load-then-compute behavior |
| Ping-pong | 2 | 2 | does one-panel lookahead hide latency? |
| Four-panel | 4 | 4 | does additional request concurrency help? |
| Eight-panel | 8 | 8 | does deeper lookahead help or only consume SRAM? |

The primary comparison is absolute makespan across rows, not the replay
engine's candidate/baseline ratio, because each row deliberately changes the
buffer organization. Require repeat-3 determinism, equal HBM bytes and useful
MACs, zero residual events, and a monotonicity explanation whenever a deeper
buffer is slower.

### Gate E: RTL calibration

Calibrate one-, two-, and four-panel load/compute overlap and same-bank
conflicts in RTL before making an absolute-cycle claim. Until then, Rust values
are comparative emulator timing.

## 8. Measured Fixed-Capacity Result

The first event-level panel-depth experiment is now complete for one exact
Qwen SWE route and one exact DeepSeek SWE route. Both use the same 512-KiB
Matrix-SRAM payload at every depth, exact nonzero MX weights, random nonzero
activations, 64-B Ramulator requests, and three deterministic repeats. The
functional dimensions are intentionally reduced to `H=512`, routed `I=256`,
and shared `I=512`, so these runs validate the mechanism and numerical path;
they are not full-model absolute timing points.

| Model | Depth 1 cycles | Depth 2 cycles | Depth 4 cycles | Depth 8 cycles | Depth 1 -> 2 | Depth 2 -> 8 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 SWE B8 route | 3,185,289 | 3,150,189 | 3,147,589 | 3,146,406 | 1.0111x | 1.0012x |
| DeepSeek-V2-Lite SWE B8 route | 2,812,254 | 2,779,754 | 2,777,154 | 2,775,971 | 1.0117x | 1.0014x |

All rows transfer exactly `8,404,992` physical HBM bytes. Qwen rel-RMS is
`0.003411`; DeepSeek rel-RMS is `0.001144`. Cycle and byte counters are
identical across all three repeats.

The result supports a modest `H2`: two panels remove most exposed DMA wait.
It does not support deeper lookahead as a meaningful contribution. Depths four
and eight improve less than 0.15% beyond depth two, even before charging a
physical SRAM bank/port penalty.

The fixed-capacity cross-expert pool experiment is also complete. It changes
panel issue order from one-expert-at-a-time to
`(k_chunk, output_column, expert)` while preserving one matrix engine, the same
512-KiB Matrix SRAM, exact tensor slices, and identical HBM bytes:

| Model | Blocking | One-expert ping-pong | Cross-expert pool 2 | Pool 4 | Pool 8 | Best pool vs ping-pong |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 SWE B8 route | 3,185,289 | 3,150,189 | 3,142,506 | 3,142,506 | 3,142,506 | 1.0024x |
| DeepSeek-V2-Lite SWE B8 route | 2,812,254 | 2,779,754 | 2,772,071 | 2,772,071 | 2,772,071 | 1.0028x |

Every row transfers exactly `8,404,992` physical HBM bytes and passes the same
nonzero functional and repeat-three gates. The pool adds only 0.24--0.28%
beyond ordinary ping-pong, and slots four and eight add nothing beyond slots
two. This rejects `H3` as a primary contribution even before charging physical
tag, allocator, bank-conflict, and arbitration costs.

## 9. Recommended Version-One Architecture

The evidence currently supports the following narrow implementation:

1. A runtime route coalescer produces deterministic expert descriptors
   `{expert_id, token_rows, route_weights, sequence_id}`. Static trace sorting
   is only an experiment oracle; production hardware cannot assume the
   compiler knows router outcomes in advance.
2. Weights use the validated tile-major MX layout so element and scale bursts
   can be coalesced without changing logical parameters.
3. Two phase-colored four-cell panels implement ping-pong prefetch at unchanged
   Matrix-SRAM capacity. One panel is `READING` while the other is `FILLING`.
4. Once an expert panel is ready, it remains weight-stationary while all
   `ceil(M/BLEN)` token-row groups consume it. This is the mechanism that
   removes duplicate expert-weight traffic.
5. Shared experts use the same descriptor and data path with all token rows;
   routed outputs retain `{token_id, route_slot}` for deterministic combine.

The unused Matrix-SRAM cells remain architecturally available for other tiles,
accumulator pressure, or future residency policies. They should not be turned
into a deeper generic expert-panel queue: the measured pool saturates at two
slots and its incremental gain is too small to survive realistic metadata,
bank, and arbitration costs.

## 10. Novelty Boundary

Banked scratchpads, tagged buffers, ping-pong prefetch, and expert grouping are
not independently novel. The defensible research contribution would have to
be their PLENA-specific co-design and measured policy:

```text
real routed-MoE skew
  -> compiler expert grouping and exact MX layout
  -> two-panel, panel-stationary use of existing Matrix-SRAM cells
  -> deterministic dependency-aware DMA/compute scheduling
  -> event-level and RTL-calibrated evidence under fixed bytes/capacity
```

The paper should claim this joint mechanism only if the ablation shows a
repeatable makespan gain beyond grouping alone.

Three hypotheses must remain separate in the paper:

1. `H1` expert grouping reduces bytes and BLEN padding. This is already
   directly testable in the compiler plus functional emulator.
2. `H2` nonblocking ping-pong overlaps useful compute with the next load. This
   needs an event-level timing delta at unchanged bytes.
3. `H3` elastic multi-context ownership outperforms fixed ping-pong. The
   measured 0.24--0.28% delta rejects this hypothesis for the current design.

The completed depth sweep rejects **deep ring buffering by itself** as a paper
claim. Generic expert grouping and two-entry ping-pong are also established
techniques. The remaining defensible research hypothesis is narrower:

```text
runtime route coalescing
  + MX-aware physical tile packing and 64-B scale-burst coalescing
  + panel-stationary reuse across all rows of one routed expert
  + deterministic issue on PLENA's flattened systolic array
```

This joint mechanism is only paper-worthy if the runtime device-selected path
achieves the fixed-route gains without a host round trip and if RTL confirms
the load/compute overlap. Until those gates pass, describe `H1+H2` as a strong
compiler/emulator optimization, not a new SRAM architecture. A tagged
multi-context pool is therefore not part of the recommended architecture. A
future SRAM-specific contribution must instead demonstrate a material new
effect, such as cross-step expert-tile residency that reduces HBM bytes, and
must beat two-panel ping-pong after physical bank/port costs.

### Related-work pressure test

The isolated ingredients already have close precedents:

- [MegaBlocks](https://arxiv.org/abs/2211.15841) and
  [ScatterMoE](https://arxiv.org/abs/2403.08245) remove padding/copy overhead
  with block-sparse or scattered expert execution.
- [MoEpic](https://arxiv.org/abs/2509.08342) explores expert splitting,
  caching, and prefetch overlap for offloaded MoE inference.
- [HyperParallel-MoE](https://arxiv.org/abs/2605.23764) uses tile-level,
  dependency-preserving scheduling across matrix/vector resources.
- [MONET](https://doi.org/10.23919/DATE69613.2026.11539142) proposes
  reconfigurable systolic PE islands and a MoE-oriented interconnect.

Therefore the document deliberately does not call a tagged pool, grouped
GEMM, ping-pong buffering, or reconfigurable compute individually novel. The
novelty test is narrower: does PLENA's unusual MX-to-BF16 tile expansion and
fixed 64-cell Matrix SRAM create a measurable scheduling problem for real
Shared-MoE routes, and does one deterministic compiler/runtime policy solve it
under unchanged SRAM bytes, HBM bytes, and matrix resources? If the event-level
ablation cannot isolate that gain, the SRAM mechanism should remain an
engineering optimization rather than a paper contribution.
