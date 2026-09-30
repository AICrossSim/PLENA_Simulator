# Fixed-hardware MoE study — preregistered methods

This study uses the independent Rust analytical engine, not native Ramulator,
RTL or measured silicon. Its boundary starts with captured routes and upstream
X already available and ends with the combined output in private SRAM. Router,
attention, network communication, and full model inference are excluded.

## Audit and reproducibility

The historical reference is Compiler `6f38f52`, Simulator `346f6986`.
Before modification, B8 stock arbitration was reproduced twice: `6` 2.038015 ms,
`3+3` 2.341945 ms, `4+2` 2.099873 ms. These belong to the old captured window,
not to the new request-disjoint test suite. New resource support must preserve
these cycles and all per-core counters under the historical defaults.

Each study phase snapshots the binary, Compiler and Simulator source, inputs,
configuration and SHA256 manifest. Each accepted point runs twice and requires
identical complete raw JSON. Host runtime is not simulated latency.

## Dimensions and timing

Dimensions are **M × N × K**. Physical N=4, K=512, while M varies. A projection
is X[Me,K] W[K,N] → Y[Me,N]. A full physical issue contains M_core×4×512 MACs;
useful MACs use the valid M/N/K extents. Padding inside issued tiles is separate
from time when a whole core has no task.

The arithmetic model has one-cycle minimum initiation and a 20-cycle dot tail
after operand feed. Real feed time comes from private SRAM bank reservations;
full X requires 16 bank service cycles plus read latency. At most eight results
are in flight per core. The next K update of an output waits for its previous
RMW commit. Independent outputs may overlap. N-group retirement waits for its
results and conversion. Thus `waves × pipeline depth` is not a latency formula.

The compute-only CLI is explicitly a nonphysical **single-projection oracle**:
resident operands, II=1, completion latency=21, eight result contexts per core,
ascending K, and a group retirement boundary. It removes memory, control, vector
and output-copy costs. It is not full FFN timing or an implementation result.
Full diagnostic FFNs execute Gate, Up, SiLU/product, Down, result movement and
weighted combination in the existing engine.

## Resource envelope

| Resource | M6 group | M8 group |
|---|---:|---:|
| Physical multipliers | 12,288 | 16,384 |
| Shared return + private W | 8 + 40 KiB | 8 + 40 KiB |
| Private W slots, 4 KiB each | 10 total | 10 total |
| X staging | 12 KiB | 16 KiB |
| Output/workspace/control | 2 MiB | 2 MiB |
| W/X/accumulator banks | 64/24/12 | 64/32/16 |
| HBM issue / response / credits | 256 B/ns / 64 ns / 256 | same |

W, X and accumulator are separate bank arrays. Each bank serves a 16-byte word
per ordinary read or write cycle; reads add one cycle. A FP32 accumulator RMW
occupies five cycles per word. Addresses map by `floor(address/16) % banks`.
Capacity and port counts are independently checked. SRAM macros, routing and
frequency have not been synthesized: equal resource budgets are not equal area.

X has two physical buffers per core, each M_core×512×2 B. A current X tile is
reused over the resident N group before changing M/K. This existing reuse is
unchanged in all study policies. Weights occupy a reserved slot from admission
through the last operand read. A 32B HBM credit returns only after its response
lands safely in that slot; return and private W storage are not counted twice.
No phase in this study expands the credit pool.

The 2 MiB arena includes original X, route metadata, persistent component and
final output inboxes, current expert X/Gate-Z/Up/Y, bounded partial sums and
control. Gate storage is overwritten by Z only after the required source reads.
Each output's K order is preserved. Results retain the historical layer lifetime.
A design unable to store a workload is a capacity rejection, not slow latency.

## Finite design space

M6 shapes: 6, 3+3, 4+2, 5+1. M8 shapes: 8, 4+4, 5+3, 6+2, 7+1.
Only within-group speedups are meaningful. No more than two cores, precision,
transpose, N/K variation or larger storage is introduced.

The predeclared N-group candidates are G=2 and G=4; G=1 is outside this bounded
search. Each core must retain G resident slots plus one Next slot. Enumerate all
integer dual-core slot splits satisfying that condition (3+7 through 7+3 for
G2; 5+5 for G4), including asymmetric memory for homogeneous compute. Mirrored
homogeneous assignments are the same physical point and are deduplicated.

Accumulator capacity and bank split independently choose proportional-to-M or
equal allocation. Capacity rounds to 32B, with the deterministic remainder
assigned in core order; bank splits round to integer banks similarly. W banks
remain 32+32 (64 single); X banks remain 4 per M lane. These choices generate
132 unique physical/group points, not a global optimum over all architectures.

Each core reserves an equal share of the existing 4 KiB control arena. This is
necessary for M=1 cores: control records do not shrink in proportion to MACs.
Every point, including FIFO, reserves the candidate's 96B feedback state inside
that same 4 KiB. Dual-core control occupancy is 4032B, leaving 64B. Compiler
reports the per-core placement. Pipeline state replication is reported as an
implementation cost proxy, not silently called area-equivalent.

## Runtime and common supply

The first-stage task is a whole expert. It retains one owner through Gate/Up,
activation and Down. Pending FIFO is eight entries; each core has one Current
and one immutable Next. There is no free task migration. Next may reserve one
existing W slot and prefetch its first tile before Current retires, but obtains
workspace only at promotion. Common supply uses stock-cycle arbitration with
the existing aging and tie rotation. Surplus rules 1–4 remain off.

All shapes receive three policies:

* `fifo`: FIFO head to a legal idle core first, round-robin tie; otherwise legal
  Next capacity. It is work-conserving under the same finite Current/Next rules.
* `dynamic`: compare predicted completion for legal cores. Shape service sums
  issue feed estimates and group tails, takes max with a credit-aware weight
  bandwidth estimate, then includes vector and startup terms. Remaining work
  uses issued/total issue counters, not future simulator event timestamps.
* `feedback`: the same decision with a Q8 service multiplier for four Me bins
  (1, 2–4, 5–16, >16), separately per core. At Current completion, measured/raw
  service ratio is clamped to [0.25,4], then updated with EWMA 3/4 old + 1/4 new.
  Only observed chosen-core completions train it. No counterfactual labels,
  expert-ID prediction, future routing, or per-test parameter selection is used.

Feedback initializes to 1.0 for every independent window. It can learn within a
window but is not a persistent cross-request predictor in this evaluation.
The update occupies two cycles of the existing mutually exclusive control
port; the multiplier changes only when that service completes. Ordinary bind
decision cost is 4+4×eligible_cores cycles, promotion and tile admission retain
their existing charges. Bounded decision costs gate execution, not just stats.
Binding error is actual completion minus the predicted absolute completion at
binding; report mean absolute error and largest positive error (underestimate).

## Workloads and selection

Mechanism pairs [4,2], [3,3], [2,2], [8,2], [1,1], [8,8] are never used for
hardware selection. Single projection N128/K512 and synthetic full FFN H512/F128
are different timing scopes and reported separately.

Real routes are archived DeepSeek-V2-Lite BFCL, GPQA and SWE decode captures:
H2048, routed F1408, Shared F2816, top-k6, no Shared sigmoid gate. Use layers1/13,
decode steps0/7, B2/4/8/16. Independently selected requests are offline rebatched;
these are not new batched inference executions. Route IDs and scores are real;
pretrained weight/activation numerical payloads are not executed at large sizes.

Split by SHA256 of dataset/request identity before timing: buckets0–5 design,
6–7 validation, 8–9 heldout. Windows are request-disjoint, including different
batches. Design has four BFCL windows, validation four GPQA, heldout eight
BFCL/SWE. Dataset, layer and batch effects are not independently factored in
this modest suite; conclusions apply to these windows only.

Nemotron's archived relu2 graph and Qwen's FP8/shared-gate graph do not match
this BF16 SwiGLU implementation and are excluded, rather than relabeled as
equivalent model validation. Prefill aggregate counts lack the required ordered
token routes. This study cannot establish cross-model or prefill robustness.

Before DSE, enumerate all legal owners of each two-expert toy and a four-expert
subset (three routed plus Shared) of one design window, preserving per-core
input order. This is an assignment oracle over whole experts, not global
optimal scheduling. Do not compare that partial workload with full-layer data.
The assignment oracle did expose residual whole-expert imbalance: in its
four-expert subset the best M6 homogeneous/heterogeneous assignments leave
433706/152830 cycles between core finishes. Before design or test timing, the
conditional second task granularity was therefore implemented and added to
the preregistration (the original plan is retained).

`tail_partition` holds only the final, still-unbound FIFO expert until both
Current/Next pipelines drain, then binds two disjoint output-column partitions.
Gate/Up share the same F partition; Down has a proportional H partition. The
compiler deterministically divides 4-column units by M width, so a partition
contains every Me row and all K segments of each owned output. There are at
most two partitions per projection; this is **coarse tail elasticity**, not a
general fine-grained group-stealing queue. Local G2/G4 tiling remains bounded.

Both cores read their own X (all copy/read/write ports charged), retain local Z,
copy the other required Z columns over the charged shared on-chip path, and
wait for complete Z before Down. No already-bound task migrates. Waiting for
both cores and loss of final Next prefetch are real costs. The paired bind
charges 16 control cycles. Pair-decision state uses part of the common 96B
reservation; feedback does not train whole-expert service bins on split tasks.
Single cores naturally have no pair. The rule is applied to FIFO, dynamic and
feedback equally. Isolate task-granularity-only, policy-only and both changes.

The 132 physical/group points expand to 260 hardware/granularity candidates
(single-core no-op variants deduplicated). All three policies run for every
design point. Selection freezes both hardware and this granularity rule;
heldout evaluation additionally runs both granularities on each frozen dual
hardware, so a policy change is not confused with a task-size change.

The selection function is fixed before timing in `selection_plan.json`:
rank hardware by design geometric-mean feedback latency; validate the best
three per budget/architecture; choose lowest validation geometric mean (ID for
deterministic ties). No invented worst-regression cutoff. Report worst-case
and Pareto behavior alongside the mean. Freeze six designs before any heldout
timing. Test every frozen design under all three policies; never swap physical
configurations per window. All full design results remain available, including
the strongest FIFO/dynamic results within the same declared search.

## Checks and interpretation

Small timed trace replay checks BF16 numerical outputs, FP32 K accumulation,
weighted route combination, shared addition, M/N/K tails and physical buffer
contents. All 132 resource points receive small replay coverage; representative
larger-tail tests cover every M partition. This is not full pretrained model
numerical validation. All large timing points check work, byte, owner, capacity,
slot and final drain invariants and identical repeats.

Front states are exclusive observations at each core's instruction front, but
can overlap arithmetic and activity on another core. Bank wait sums and control
service sums must never be added to obtain wall latency. Report useful/padded
MACs, X/Z/result movement, HBM bytes/transactions, credit peaks, actual binding
records and per-core finish separately. Do not turn a losing shape or unchanged
dispatch into a claimed contribution.
