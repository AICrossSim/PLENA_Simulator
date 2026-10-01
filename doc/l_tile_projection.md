# Projection and native recurrent sublayers

## Main design decision, 2026-10-01

Projection now uses a software mapping of the common Matrix path in the main
research candidate. Do not include the 16 KiB replay buffer, compact-input
slice hardware, M_MM.P or S4 segmented reduction in that candidate's cost or
speedup. Their implementations and earlier measurements remain historical
hardware ablations. The controlled comparison is:

```
same software-selected projection + ordinary Vector recurrence
same software-selected projection + native L_TILE recurrence
```

The [completed study](../artifacts/projection_pipeline/software_study/README.md)
is reproduced with `analytic_models.performance.projection_software`.
It searches request groups and software-owned Vector SRAM rows, then evaluates
static-transposed weights through the existing `M_TMV` opcode. It executes
emitted memory traces through Ramulator and reprices each complete sublayer.
Reloads between request groups are explicit. The resident search retains K256
rounding; the transposed K1024 candidate has a separately tested numerical
contract. Default resources are unchanged, and reserved codec rows can be
released only for BF16 execution.
The common rectangular Matrix-view substrate is still present in both arms;
these are not measurements of the best unmodified original PLENA.

The original packed M_MM emitter also had a separate address bug: its packed
K slices advanced by BLEN rather than BLEN*MLEN, contrary to the row-granular
decoder. This is corrected in `aten/plena/isa_matrix.py`. The `legacy_mm`
machine check uses the original M_MM/M_MM_WO instructions, one 8 KiB Matrix
tile, B1/2/4/8/16 and K=256 at MLEN=64/BLEN=4. Partial sums survive two
weight-tile loads; 1,984 active values and 320 padded values are checked exactly. This is functional
evidence for existing resources, not calibration of full-size M_MM timing.
Its packed weight image contains 262,144 bytes for 32,768 logical BF16 weight
bytes. This deliberately padded correctness fixture is not a bandwidth-optimal
projection. Its legacy cycles do not enter any performance comparison.

The remaining sections retain the history of the projection hardware study.
The design rationale and current evidence boundary are collected below.

## Earlier published implementation checkpoint

Use `feat/matrix-sram-lcompute` in both repositories. This Simulator's
`PLENA_Compiler` gitlink pins the exact companion implementation. Existing
review PR branches remain historical mechanism-only reviews.

| Area | Current implementation | Boundary |
| --- | --- | --- |
| Compiler/Rust | Mamba/KDA coefficient producers, native supply, finite M_MM.P projection, private request state | One recurrent sublayer; outer residual and FFN/MoE excluded |
| Python timing | The same compiled instructions, finite SRAM/compute services and address-based Ramulator DMA | Analytical prediction; shares the memory model with Rust |
| Batch evidence | Mamba/KDA B1/2/4/8/16 machine executions; exact declared-arithmetic checks and separate cycle/traffic calibration | Captured B1 inputs repeated per batch; distinct-input Matrix checks are separate |
| Peripheral evidence | Connected representative attention/MLA, fixed-route expert MLP and combine | Does not certify all shapes, dynamic routing or complete models |
| Whole-model decode | Compositional candidate model with capacity/coverage limits | Not a validated original-PLENA or silicon speedup |
| Overlap | Separate bounded protocol probe | Production retires serially; no integrated concurrent engine implementation |

Compact results, resource requirements, provenance and numerical scope are in
[`artifacts/projection_pipeline`](../artifacts/projection_pipeline/README.md).
Historical experiments elsewhere in the repository retain their own contracts;
their speedups are not interchangeable with these tables.

## Execution and hardware contract

The default recurrent working point is L=256, update II=2, update latency=6,
BF16 state/RN writeback, FP32 update intermediates and a BF16 pairwise tree.
Compact coefficients pass through bounded sector/refill/broadcast services.
The evaluated Matrix geometry has 4096 multipliers, 1 MiB Matrix SRAM with 64
banks and 256 KiB Vector SRAM. Clock=1 GHz and 32/32 DMA credits are explicit
candidate parameters, not achieved integrated timing or audited original RTL.

Projection ablations progressively enable replay, compact input slices and
up-to-four-request M_MM.P packets. Their extra payload is 22.25 KiB, excluding
metadata, control, selection and the separate recurrent hardware. The tables
execute BF16 weights; runtime NVFP4 decoding is not a Rust-validated instruction.
The analytical NVFP4 service is a separate finite-throughput assumption.

The default comparison called `old_isa` changes recurrence to ordinary Vector
instructions. It still uses common Matrix-view/Softplus services. A true original
PLENA comparison needs legal original DMA/addressing, best M_MM/M_MV selection,
nonlinear software mapping and matched numerical execution. That gate is not
passed. Neither the model nor these tables claim an optimal projection mapping.

## Reproduce from a checkout

Initialize the pinned dependencies and enter the repository's existing Nix
environment. Use a new output directory on a filesystem with room for builds
and temporary memory traces. No GPU or checkpoint download is required for
the unit tests and analytical sweep.

```sh
git submodule update --init --recursive
nix develop --no-write-lock-file
export PLENA_COMPILER_ROOT="$PWD/PLENA_Compiler"
export PYTHONPATH="$PWD:$PLENA_COMPILER_ROOT:$PYTHONPATH"
export CARGO_TARGET_DIR=/tmp/plena-projection-target
export RUN_ROOT=/tmp/plena-projection-run
python -m pytest analytic_models/performance -q
cargo fmt --manifest-path transactional_emulator/Cargo.toml --all -- --check
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
  cargo test --manifest-path transactional_emulator/Cargo.toml --workspace -- --test-threads=1
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
  cargo clippy --manifest-path transactional_emulator/Cargo.toml --workspace --all-targets -- -D warnings
python -m analytic_models.performance.ltile_dma --prepare "$RUN_ROOT/memory16" --controllers 16
python -m analytic_models.performance.projection_campaign \
  --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/smoke" \
  --models mamba --batches 1 --stages baseline batch
```

Omit `--models`, `--batches` and `--stages` for the 40-case Mamba/KDA × B1..16 ×
four-stage sweep. A second output with `--control old_isa --stages batch`
produces the shared-projection recurrence comparison. That comparison changes
the arithmetic contract and is not a pure FSM ablation. Output directories
must be new; the runner never overwrites a completed campaign.

Each run saves `summary.csv`, `operators.csv`, per-case resource/source hashes
and a manifest. The exclusive total is issue + scalar + SRAM + arithmetic +
dependency + DMA. Operator totals already include DMA; do not add it twice.
`frontend` is issue+scalar, not another exclusive component. Totals are batch
step cycles, and milliseconds are cycles / 1e6 at this 1 GHz working point.

For ten small numerical executions with distinct request inputs, K/N tails,
one/four-request packets and no checkpoint:

```sh
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
  cargo build --release --manifest-path transactional_emulator/Cargo.toml
python -m transactional_emulator.testbench.models.unified_service_test \
    --runtime "$CARGO_TARGET_DIR/release/transactional_emulator" \
    --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/projection-check" --only projection
```

Real-weight sublayer execution uses
`transactional_emulator.testbench.models.projection_pipeline_test --help`.
It requires external captured fixtures (`--fixtures`) with initial and reference
HBM images. The small result archive does not contain model weights or raw HBM
images, so it is not a self-contained substitute for those numerical fixtures.
Synthetic checks and analytical sweeps work without them.

## Projection mapping and reduction study

The [completed 120-case study](../artifacts/projection_pipeline/projection_study/README.md)
separates two changes. Both keep the recurrent
arithmetic, SRAM capacities, DMA credits and BF16 K256 rounding contract fixed.

**Compiler-only K/N panel tiling.** `projection_n_panel_tile=1/2/4/8` selects
the number of N32 panels that share a K2048 input chunk. The Compiler retains
one output row per request, loads each weight packet once, and reuses each
uncached input chunk across the selected output panels. At most 64 K256xN32
views occupy the existing 1 MiB Matrix SRAM. No additional hardware is needed
relative to the existing M_MM.P candidate. The original schedule remains a
candidate: larger panel groups can worsen DRAM locality when all inputs already
fit. This full-SRAM projection schedule does not assume simultaneous resident
recurrent state or concurrent projection/recurrence execution.

**Fixed reduction segments.** `matrix.projection_segments=2/4` is a distinct
hardware candidate, not a Compiler-only gain. A K256 packet uses only 64 of the
256 four-by-four mini-arrays in the current K1024 reduction geometry. Fixed
segments assign different N4 outputs to otherwise unused groups while sharing
the input. Four segments produce N16 per wave instead of N4; an N32 packet
therefore takes two waves instead of eight. The multiplier count remains 4096.
The segment count is a static candidate implementation parameter. The same
M_MM.P machine code runs on each candidate; this experiment does not implement
a new instruction for arbitrary runtime switching between segment counts.

Each segment keeps the original local accumulation and BF16 tree order. Its
root is selected into the existing upper tree, with unused operands zeroed,
and traverses the original upper levels serially. The model retains these
zero-add rounding operations rather than substituting a different dot product.
The final per-request BF16 output merge also remains unchanged.

The candidate explicitly charges operand-latch bandwidth, two distribution
cycles per wave, two selection and two collection cycles per root, and the
serialized upper-tree arithmetic. It assumes no overlap between waves. Matrix
bank reads, Vector input reads and output read-modify-write remain explicit.
Four 4x4 BF16 roots require 128 bytes of new capture storage, plus unpriced
tags, masks, selection, broadcast and control logic. Existing 16 KiB replay,
4 KiB row transfer, 2 KiB compact inputs and 256-byte result storage remain;
the common Matrix operand latches are a further 8 KiB on each input side.
This is a capacity inventory, not an area or power result. The routing delay
and its effect on the original full-width Matrix path still require RTL checks.

This study does not establish a globally optimal Matrix architecture. The
original PLENA paper already describes PE-local output stationarity and long-K
accumulation. The local RTL checked at `2c5a5f4` uses a small MXINT/E6M5 test
configuration and a fixed-point accumulator; it does not certify this BF16
K1024 profile or bubble-free tile throughput. The default K256 BF16 execution
contract is a controlled research reference, not the best proven original
PLENA implementation. Long-K accumulation and projection precision therefore
remain separate comparisons.

WS, IS and OS need not be mutually exclusive hardware modes: weights are
reused across requests in Matrix SRAM, inputs across output panels in Vector
SRAM, and partial results within the Matrix/Vector execution contract. The
Compiler selects bounded loops for the actual shape and effective batch;
for MoE that is the expert's token count. A generic mode-switching network is
not justified by this experiment.

Relevant prior work includes [PLENA](https://arxiv.org/html/2509.09505v2),
[LoopTree](https://arxiv.org/abs/2409.13625),
[HLX](https://doi.org/10.1145/3725843.3756115), and
[MAERI](https://anands09.github.io/papers/maeri_asplos2018.pdf).
Tiling, hybrid support, fusion and segmented reduction alone are not new claims.
The research question is whether a bounded mapping between batch-shared
projection weights and request-private recurrent state improves the complete
sublayer without expensive operand materialization or sacrificing weight reuse.

## Next work, not claimed as implemented

1. Establish the best legal original Matrix and ordinary Vector baseline.
2. Evaluate request/head-group residency jointly with weight reuse.
3. Specify and charge an actual bounded Vector/Matrix SRAM handoff interface.
4. Validate the final arithmetic over long real-input chains and task quality.
5. Validate whole-model composition, capacity, routing and held-out operators.
6. Integrate the candidate datapath before making timing, area or power claims.

WS/IS/OS template selection and direct producer-consumer handoff remain design
candidates. The standalone protocol probe is not evidence of zero-cost overlap.

## Per-projection selection and complete batch comparison

The [2026-09-29 comparison](../artifacts/projection_pipeline/final_comparison/README.md)
records four explicitly named arms for both recurrent sublayers at B1/2/4/8/16.
The new `--tune-panels` policy generates four legal uniform Compiler programs,
selects a panel schedule for each projection from their analytical stage costs,
then recompiles and reprices the entire mixed program. Memory history can make
the combination worse; in that case it retains the best uniform program.
It does not add stage-wise minima or consume measured Rust timing as a prediction.
Selections are included in the immutable execution profile and cache identity.

```sh
python -m analytic_models.performance.projection_campaign \
  --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/tuned" \
  --stages batch --segments 4 --tune-panels --workers 4
python -m transactional_emulator.testbench.models.unified_service_test \
  --only mixed --runtime "$CARGO_TARGET_DIR/release/transactional_emulator" \
  --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/mixed-validation"
```

The machine runner scopes the pinned libtorch loader path to the Rust child;
do not prepend that path to Python's environment. The five new dependent
projection tests use distinct requests, different schedules in consecutive
operators, K/N tails and B16 cache pressure. All 10,974 active output values
match the independent reference exactly, and all seven cycle components match.
The full analytical suite passed 282 tests with nine skips before raising the
interpreter safety limit for the ordinary-Vector KDA B8/B16 programs. The
affected interpreter/projection tests then passed 75 tests with six skips.
The raised instruction limit changes no modeled hardware resource or cycle cost.

## Publication checks

The current projection study passed 329 Rust workspace tests, formatting and
clippy across all targets, and 276 analytical Python tests with nine explicit
skips. Its 24 representative machine-code cases made 53,637 exact
output-value comparisons against an independent reference and matched all seven cycle components. Compiler tests
focused on the affected interfaces passed 111 cases. These are implementation
checks, not integrated RTL timing or long-chain model quality acceptance.

The earlier publication checkpoint passed 328 Rust tests and 268 analytical
Python tests with nine explicit skips. Its
portable campaign reproduced Mamba B1 at 3,063,999 baseline and 2,909,800 batch
cycles. These checks do not replace the archived full-shape numerical campaign
or change its one-token scope. Compiler-wide legacy failures are recorded in
the companion Compiler's `doc/l_tile_projection.md`.
The portable `--only projection` numerical runner also passed all ten
B1/2/4/8/16 × one/four-request cases with distinct inputs and K2305/N65 tails,
including independent cycle and traffic reconciliation.

## Dataflow design after removing projection hardware extensions

### The implemented projection alternative

For `Y = X W`, store each static N32-by-K packet with K contiguous within
each output row. The original transposed opcode reads those rows into the
common Matrix operand registers. Four output rows are read **serially**, with
bank and port service charged, before the same array/tree evaluates them.
K1024 uses the existing four-by-1024 weight operand capacity (8 KiB in the
common reference); it does not add the historical 16 KiB replay buffer.

The emitted loop is:

```
retain legal aligned input windows in existing Vector SRAM
for each N32 panel:
    load its K packets once into existing Matrix SRAM
    for each request:
        M_TMV over K packets, preserving the request's output partial sums
        M_MV_WO to its private output row
store completed output rows
```

A physical word's 32 K weights are consumed by the current output dot.
The old K256 column mapping read a word for eight output groups. This reduces
word transfers, but not automatically eight times the elapsed read service:
the new schedule still serializes logical rows. Its larger benefit comes from
fewer padded K packets, fewer whole-tree reductions and fewer full-row input
transfers. Static weight transposition is an offline model preparation cost;
activations are not transposed by an unpriced runtime operation.

`M_TMV` already existed, but its old Simulator timing was coarse. The new
bounded timing branch uses exactly the common Matrix geometry and rounding
model, while leaving the legacy unprofiled branch unchanged. This is
Simulator reference development, not evidence that the original RTL has
already passed these timings. Rectangular views remain a common research
interface; **zero additional projection payload relative to that interface**
must not be reworded as zero modification to the original PLENA RTL.

K256 and K1024 independently match their declared BF16 machine references.
Their reduction grouping differs. Small numerical tests and one-token layer
checks cannot certify long-chain quality. Transposed NVFP4 is rejected until
block/scale placement and conversion are separately validated.

This study uses one projection template per campaign. Small-K operators may
prefer M_MV, and a legal full-size M_MM mapping may improve batch utilization.
Those alternatives need complete emitted-program comparisons before selecting
per-operator templates. In particular, the existing V_SHFT_V is a zero-filled
right shift; it is not a free left-shift selector for the upper half of an
input row. No such selector is silently assumed here.

A concrete example is K1024 by N32. The old template emits four K256
packets, each sending four useful output columns through the full reduction
tree. The transposed template emits one K1024 packet with the same eight
N4 output groups. All 1024 K lanes now carry useful operands. Under the
common reference, the compute/read service is four times (256 + 40) cycles
versus once (256 + 40), before instruction setup and DMA. The bank-word
traffic falls eightfold, but the elapsed packet service only fourfold.
The complete layer improvement is smaller because weights still traverse HBM
and other operators remain. The N32-by-K orientation also fits K1024 into
the bounded rectangular view without pretending that 1024 logical rows fit
the 256-row Matrix storage.

Projection computes `Y[B,N] = X[B,K] W[K,N]`. Its two reusable operands have
very different lifetimes from recurrence: weights are shared across requests;
recurrent state belongs to one request, layer and token chain. Keeping a
request's state resident across the entire projection can evict shared weights.
Processing a complete layer for each small request group can reload those
weights. Neither policy is universally preferable.

The recommended design has three responsibilities:

| Responsibility | Existing resource / mechanism | Implementation boundary |
| --- | --- | --- |
| Dense projections, experts and output head | Common Matrix array, operand registers and partial sums | Compiler selects M_MV or static-transposed M_TMV; original M_MM functionality is checked but its full-size timing still needs calibration |
| Input reuse and intermediate lifetimes | Existing Vector SRAM, with explicit row ownership | Request groups and row budgets are implemented; arbitrary unaligned slices are not assumed |
| Request-private recurrence | Shared banked Matrix SRAM and native L_TILE access/update path | Finite coefficient sectors, state/result slots, shared-port service and BF16 tree are modeled and executed |

A legal serial schedule is the initial implementation, not a free-overlap
estimate:

```
projection weights in Matrix SRAM + inputs in Vector SRAM
  -> existing Matrix instructions, retain partials as the template permits
  -> committed projection output
  -> existing conv / nonlinear / normalization instructions
  -> compact recurrent coefficients
  -> shared Matrix SRAM reassigned to the current request's state tile
  -> descriptor reads + head broadcast + fused update + existing Vector tree
  -> BF16 state writeback and recurrent output
  -> existing Matrix output projection
```

Until an on-chip producer-consumer interface is actually emitted and executed,
its explicit transfers remain in the ledger. This diagram does not certify
zero-copy Matrix-to-recurrent handoff, simultaneous engines or full-model
weight residency.

### Compiler decisions, not a WS/IS/OS mode switch

At the SRAM level, the Compiler may keep a weight panel while serving several
requests, and retain input rows while traversing output columns. At the PE
level, original M_MM can retain output partial sums across K. These choices
can coexist. Loop order alone does not turn a fixed output-stationary PE into
a different physical dataflow.

Choose mappings from the legal instruction interface, then check:

1. Weight panel plus any concurrently live state/coefficients fits Matrix SRAM.
2. Input windows, private outputs, codec workspace and temporaries fit Vector SRAM.
3. Only the actual number of PE partial sums stays live; switching output tiles
   cannot silently preserve another accumulator set.
4. Every SRAM access uses the real alignment, bank and port rules. The current
   M_MV ABI accepts a full aligned Vector row and only consumes its K prefix.
   Logical tensor size alone therefore understates its resident input cost.
5. The emitted numerical order is unchanged, or the new order receives separate
   numerical validation. Moving cross-K reduction changes rounding boundaries.
6. Choose the fastest complete emitted program, with a fallback to the old
   schedule. Do not sum independently optimal stage times.

For MoE, the effective matrix batch is the number of tokens assigned to an
expert. Apply the same mapping rules to that count, rather than assuming that
all experts receive the model batch size. Dynamic routing integration remains
outside this sublayer study.

The resident bounded search varies request tile 1/2/4/8/16 and 58/64 Vector
rows. The transposed candidate additionally uses K1024. This is a finite
search, not a proof of the optimum across all Matrix instructions and layouts.
It deliberately does not invent M_MM service from peak MAC counts.
At 64 rows, the six decoder rows are released for BF16-only projection. NVFP4
must retain the reservation and account for decoding; the BF16 results cannot
be relabeled as NVFP4 results.

### Recurrent hardware that remains

The mechanism worth evaluating is direct consumption of compact producer
outputs through a bounded banked-SRAM interface. A view supplies addresses,
head sharing and lane placement; it does not store an expanded tensor.
Compile-time phase placement must be separated from the hardware access path:
fixed diagonal wiring and a different legal base can already give identical
coordinates. D=D' remains a useful negative control.

| Part | Purpose | Explicit cost / limit |
| --- | --- | --- |
| View registers, coefficient descriptors and finite controller | Traverse regular state/coefficient operands | Registers, counters, configuration/issue cycles; not a general dynamic scheduler |
| Per-bank address/mask selection and cyclic lane alignment | Read and return the requested logical lanes | Shared existing ports; physical words and conflicting rounds are charged |
| Sector refill and head broadcast | Consume compact coefficients without per-token expanded arrays | 4 KiB sector payload and selection/refill cycles |
| State and invariant/residual holding | Decouple restricted reads from L=256 consumption | 2 KiB state slots plus 4 KiB invariant/residual payload |
| Update lanes | FP32 `S - delta*S + b*x`, then BF16 commit | L=256 working point, II=2, latency=6; arithmetic and integrated wiring still require implementation cost evidence |
| Result slots and destination tags | Hold committed values until write acceptance | Four 256-element BF16 slots: 2 KiB, plus tags/masks/credits |
| Dot reduction | Reuse the Vector pairwise BF16 tree | Existing SRAM and arithmetic service; no dedicated FP32 dot-context SRAM by default |

The recurrent payload budget is 12 KiB, excluding pipeline registers,
descriptors, tags and control. It is not a cache: there is no associative
lookup, replacement policy or hidden capacity. Removing the historical
projection extensions removes 22.25 KiB of their payload, plus the optional
128-byte S4 roots. These are capacity differences, not area or power savings.
The common Matrix operands and SRAM are still required.

At L=256 and II=2, the update array accepts at most 128 elements per cycle
on average. A 512-element state slot takes at least four issue cycles to
consume, and a logical 2048-element row needs at least 16. A conflict-free
bank read does not update all 128 state rows in one cycle. The two input
slots can hide only delays allowed by the shared read ports and finite
refill schedule; they do not guarantee zero stalls.

Persistent state stays in HBM. Only the active group occupies Matrix SRAM.
Mamba's configured group holds 32 heads of 128x64 BF16 state; KDA holds 16 heads
of 128x128 BF16 state. Each group is 512 KiB. Do not keep multiple requests'
groups resident if their complete live sets exceed capacity.

KDA must finish its prediction reduction before forming the BF16 residual.
The old BF16 state is retained until the subsequent update can legally consume
it. A controller cannot replace that dependency with a readiness counter.
Mamba's shared B/C groups and output normalization group constrain a possible
handoff schedule, but do not require every independent recurrence head to wait
for all projection outputs.

### An additional candidate: forward the committed update to readout

A focused next experiment can forward the **rounded BF16 update result** to
readout before releasing its result slot. Mamba's update and readout would
consume one state scan rather than two. KDA retains its prediction/residual
barrier, then combines update and readout, reducing three scans to two. This
reduces Matrix SRAM reads, not automatically state HBM traffic.

This is a design candidate, not implemented performance in the software study.
It needs an explicit consumer for both the SRAM write and the BF16-product
path. A slot is released only after both accept the value. If multiplication
resources are shared with update, the product consumes their service time;
if separate, the extra multiplier cost must be reported. Backpressure must
preserve the payload and destination tag. The order of BF16 tree leaves and
merges must match the unfused path.

Forwarding the committed BF16 value preserves the state-rounding boundary.
Distributing the algebra, for example reading out `decay*S + outer` without
that intermediate BF16 rounding, is not generally bit-equivalent. This is a
reason to test a specific implementation, not proof of novelty or speedup.
Retain it only if the saved bank service outweighs additional arbitration,
product service and holding costs in complete sublayers.

### Prior work and the defensible research claim

| Primary source | What is already established | Consequence for this work |
| --- | --- | --- |
| [PLENA, Section 3.2](https://arxiv.org/html/2509.09505v2#S3.SS2) | Flattened output-stationary arrays, long-K partial sums and separate final reduction | Do not claim these as new projection hardware; first exploit and calibrate them |
| [Timeloop](https://accelergy.mit.edu/timeloop.pdf) and [mapping specification](https://timeloop.csail.mit.edu/previous_versions/timeloop-accelergy-v3/timeloop/input-formats/mapping) | Mapping through loop factors, order, spatial distribution and retention | Generic loop search and WS/IS/OS labels are established methods |
| [Gemmini implementation](https://github.com/ucb-bar/gemmini/blob/master/README.md) | Configurable dataflows, explicit scratchpad/accumulator resources and finite access/execute queues | A mode change or overlap claim needs the supporting datapath and storage |
| [NVIDIA tensor maps](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) and [CuTe layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) | Descriptor-based multidimensional transfers, shared-memory swizzles and compile-time layout composition | Neither an affine descriptor nor bank-aware layout is new alone; distinguish this recurrent operand-consumption path from a tensor-copy engine |
| [LoopTree](https://arxiv.org/html/2409.13625v2) | Joint tiling, retention and recomputation; fusion can worsen traffic under small buffers | Joint scheduling is required; fusion itself is not a new contribution |
| [HLX, MICRO 2025](https://doi.org/10.1145/3725843.3756115) | Unified hybrid Transformer-Mamba architecture with pipelined dataflows | Hybrid support and pipeline overlap alone are insufficient novelty |
| [MARCA, ICCAD 2024](https://arxiv.org/html/2409.11440v1) | Reconfigurable reduction/bypass PEs and intra/inter-operator buffer management | Reusing compute for elementwise work is feasible in some architectures; dedicated lanes are not universally necessary |
| [HEMERA, July 2026 preprint](https://arxiv.org/html/2607.22022v1) | Separate dense projection and streaming recurrence engines with a shared local buffer | A two-engine block diagram or local exchange is not by itself a new contribution; its CIM/tiered-memory setting differs from this shared-SRAM decode study |
| [Persistent-state linear-attention accelerator](https://arxiv.org/html/2603.05931v1) | On-chip GDN state and a fused two-scan design, including an algebraic transformation | State residency and scan fusion already have close prior art; compare capacity, arithmetic boundaries and complete operator scope |

The proposed research question is specific: **Can compact coefficients and
request-private recurrent state be consumed directly through existing shared
SRAM capacity, while retaining an effective software mapping for batch-shared
projection weights?** The possible contribution is the restricted view-access
mechanism and its resource/precision-aware mapping, demonstrated against a
strong projection baseline. This is a hypothesis; neither a new name nor the
presence of a descriptor establishes novelty.

Measure three comparisons separately:

- Best legal software projection versus earlier software projection, recurrence
  fixed: Compiler benefit, no new projection hardware.
- Same projection and same recurrent arithmetic, software materialization versus
  native supply: effect of the operand-access mechanism.
- Same projection, ordinary Vector recurrence versus the full recurrent path:
  total extension benefit, including the declared arithmetic change.

Finally, compare equal lane counts/layouts for bank attribution and equal
arithmetic/ports for FSM attribution. Report a zero improvement where no extra
bank rounds exist. Full sublayer, kernel and whole-model ratios are separate;
this study does not certify TTFT, quality preservation, PPA or a GPU speedup.
