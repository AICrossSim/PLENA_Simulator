# Production-DMA HBM Service Model V4: Design, Calibration, and Validation Report

## 1. Document Information

| Item | Value |
|---|---|
| Report date | 2026-07-17 |
| Scope | Post-hoc prediction of production DMA completion latency |
| Model artifact | `analytic_models/performance/calibration/hbm_dma_service_v4.json` |
| Calibration ID | `hbm-production-dma-v4-1b00531de9a61298` |
| Model kind | `production_dma_occurrence_residual_v4` |
| DMA semantic version | `production-dma-lines-v2` |
| Feature semantic version | `production-dma-targeted-row-hit-v3` |
| Ramulator configuration | `HBM2_2Gbps`, `MOP4CLXOR` mapping |
| Logical request line | 64 bytes |
| Native physical burst | 16 bytes |
| Promotion status | Accepted as the default DSE memory shadow |
| Explicitly out of scope | Numerical correctness, comparison tolerance, online cross-queue memory simulation |

The primary model target is one production DMA instruction completion
interval. Reported nanoseconds use the current 1 ns transactional cycle. This
does not prove that the RTL closes timing at 1 GHz.

## 2. Executive Summary

The historical V3 HBM service model was not sufficiently accurate for use as a
formal DSE objective. Its generic holdout P95 absolute error was 39.03%. The
main issue was not merely regression quality: the calibration target did not
always execute the same request semantics as the production transactional DMA
path. The largest discrepancy was in stores. For a Qwen-like `H_STORE_V` with
`dim=512`, `amount=64`, and 128 channels, the old raw calibration measured
12,289 ns, while the production scatter path completed in 52 ns. The old label
included 36,864 bytes of reads and 36,864 bytes of writes; the production path
recognized complete line coverage and issued zero read bytes and 36,864 write
bytes.

V4 removes this calibration-to-production mismatch. It uses the exact packed
MX layout and the same 64-byte gather/scatter line planner as the emulator. A
store performs one read-modify-write only for a partially covered line; a fully
covered line skips the read; overlapping element and scale fragments are
merged. Rust executes the production DMA path and records completion callbacks.
Python independently reconstructs the physical request manifest. A calibration
point is rejected if line counts, byte counts, or the canonical manifest hash
do not agree.

V4 is fitted on 2,592 generic production-DMA measurements plus 36 targeted
row-state anchors. It uses a physically motivated transfer floor and a
nonnegative residual model, separately for each opcode and HBM channel count.
The final 585-point generic holdout results are:

```text
Median absolute error:           3.48%
P95 absolute error:             18.10%
Maximum absolute error:         42.18%
Weighted MAPE:                   2.80%
H_STORE_V P95 absolute error:   19.92%
Request-manifest mismatches:         0
```

Promotion also required four fixed-point, arrival-aware production Qwen trace
replays. Total HBM work error ranges from 0.79% to 11.70%, and scheduled
makespan error ranges from 0.086% to 0.573%. The makespan figures are not a
claim of sub-percent memory-model error: the tested traces are compute-heavy,
so HBM work error is the more direct memory-model metric.

The accepted V4 artifact is integrated into CostEmitter and the Qwen3-32B DSE
as a memory shadow. The formal DSE latency objective remains calibrated
`rtl-v1` compute resource work with the legacy bandwidth guard. V4 is not yet
used as the default formal objective because it remains a post-hoc surrogate
and does not maintain online Ramulator state across simultaneously outstanding
DMA queues.

## 3. Research Question and Claim Boundary

### 3.1 Research question

The design answers the following question:

> Given a compiler-emitted production DMA occurrence, its packed precision
> layout, physical addresses, and HBM channel count, can a lightweight model
> predict the completion time produced by the current transactional
> gather/scatter plus Ramulator path accurately enough to serve as a DSE memory
> shadow?

### 3.2 What V4 predicts

V4 predicts:

- `H_PREFETCH_M` completion latency.
- `H_PREFETCH_V` completion latency.
- `H_STORE_V` completion latency, including partial-line read-modify-write.
- Production packed MXINT/MXFP element and scale traffic.
- Critical-channel, bank-group, bank, and row-state effects visible in the
  calibrated request geometry.
- The DMA/SRAM transfer drain represented by the production calibration path.

### 3.3 What V4 does not predict exactly

V4 does not claim:

- Online scheduling between multiple simultaneously outstanding DMA queues.
- A globally exact 64-layer Ramulator row and queue state in the fast DSE mode.
- HBM PHY, package, stack, or thermal behavior.
- Numerical model accuracy or quantization correctness.
- RTL Fmax or timing closure.
- Accuracy outside the recorded feature domain without extrapolation risk.

The numerical executor, golden data, comparison tolerance, and PASS threshold
were not changed during this work.

## 4. Why V3 Was Not an Adequate Production Model

### 4.1 Target mismatch

V3 was calibrated from request patterns that could differ from the production
`gather/scatter` implementation. A regression model cannot repair a systematic
label mismatch: even a perfect fit to those labels would predict the wrong
runtime behavior.

The store example that exposed the problem is:

```text
Opcode:            H_STORE_V
dim:               512 elements
amount:             64 rows
channels:          128

Old raw target:
  latency:       12,289 ns
  read bytes:    36,864
  write bytes:   36,864

Production DMA:
  latency:           52 ns
  read bytes:         0
  write bytes:   36,864
```

The production store completely covered the affected physical lines. Reading
those lines before writing them was therefore unnecessary. The old target
modeled a sequential read-plus-write behavior and overestimated service time by
more than two orders of magnitude.

### 4.2 Aggregation mismatch

V3 could predict a compressed stream and divide the result by stream
multiplicity. This assumes that every occurrence has the same addresses,
request conflicts, and row-state history. Compiler-generated Qwen streams do
not satisfy that assumption. V4 predicts one occurrence at a time and verifies
that every predicted occurrence is consumed exactly once.

### 4.3 Feature mismatch

A channel-average request count is insufficient for irregular MOP4CLXOR
layouts. Completion is controlled by the busiest channel or pseudochannel, not
the arithmetic average across all configured channels. Large strides can also
concentrate accesses into a subset of banks or bank groups. V4 therefore uses
critical-path occupancy features instead of only global request counts.

### 4.4 Row-state mismatch

Early V4 candidates still had a long-tail error because they treated all
low-request-count accesses similarly. Inspection showed two physically
different regimes:

- A first access to a closed bank, which incurs an initial row miss.
- A repeated access to an already open row, which is a true warm row hit.

An inherited switch to a different row is not a warm hit; it remains a row
conflict. The final V4 model adds targeted anchors and uses a separate small
warm residual model only for the strict no-miss, no-inherited-conflict regime.

## 5. Production DMA Semantics

### 5.1 Packed layout

The request planner receives the logical transfer and the actual memory
format. For MX formats it computes:

```text
element row bytes = dim * element_bits / 8
scale row bytes   = (dim / block_size) * scale_bits / 8
```

Element and scale arrays have independent physical bases. Logical offsets are
converted into packed byte ranges before physical lines are generated. This is
necessary for MXINT2 and MXINT4, where one logical element occupies less than
one byte.

### 5.2 Read planning

For gather operations, all touched byte ranges are mapped to 64-byte physical
lines. Duplicate lines are merged. Each line is requested once even if
multiple logical fragments refer to it.

### 5.3 Store planning

For scatter operations, the planner first unions all byte-coverage masks for a
physical line:

```text
full line coverage:
  read phase  = none
  write phase = one line

partial line coverage:
  read phase  = one line
  merge patch = local read-modify-write
  write phase = one line
```

All partial-line reads complete before the coalesced write phase is submitted.
Writes wait for Ramulator completion callbacks rather than returning when the
request is merely accepted.

### 5.4 Canonical request manifest

Every occurrence is represented as an ordered manifest:

```text
read_lines
write_lines
full_lines
partial_lines
payload_read_bytes
payload_write_bytes
physical read/write bytes
request_manifest_hash
```

The canonical hash is `FNV-1a 64 v1` over semantic-versioned ordered line
addresses. Rust and Python independently generate the manifest. The fit aborts
on any mismatch. This parity check prevents a fast analytical request builder
from silently drifting away from production DMA behavior.

### 5.5 Change-to-evidence matrix

| Correction or design change | Principal implementation | Direct validation evidence |
|---|---|---|
| Coalesce duplicate 64-byte reads | `transactional_emulator/lib/memory/src/chunked.rs` | Duplicate-line gather unit test |
| Merge overlapping store fragments | `chunked.rs::plan_scatter` | Same-line and element/scale overlap tests |
| Skip read for full-line store | `chunked.rs::scatter` | Full-line store manifest test: 0 reads, 1 write |
| Read-modify-write partial lines once | `chunked.rs::scatter` | Partial and cross-line scatter tests |
| Wait for write completion and drain | Ramulator wrapper and transactional runner | Read/write callback and zero-pending drain tests |
| Account for packed MXINT/MXFP bytes | `transactional_emulator/src/dma.rs` | MXINT4 offsets and MX element/scale payload tests |
| Execute the real production DMA calibration target | `src/dma_calibration.rs`, `src/bin/hbm_dma_calibration.rs` | Three completion samples for every one of 2,628 points |
| Independently rebuild request geometry in Python | `analytic_models/performance/hbm_service_v4.py` | Zero count/byte/hash mismatches across all points |
| Replace average bandwidth with critical-channel floor | `occurrence_features()` | Generic layout holdout and large-stride cases |
| Model opcode/channel behavior separately | `HbmServiceModelV4.group_key()` and fitted coefficient groups | Per-group holdout diagnostics and four channel-diverse Qwen traces |
| Preserve production row state across occurrences | `Mop4clxorRowState`, `V4DmaServiceProvider` | 36 row-state anchors and arrival-aware system replay |
| Separate strict warm and cold/mixed regimes | warm and cold residual coefficient sets | Exact-row-hit P95 3.94%; runtime regime checks |
| Enforce physical monotonicity | theoretical floor and nonnegative ridge | Unit trend checks and artifact coefficient constraints |
| Track semantic compatibility | model compatibility metadata and fixture hash | Loader rejects semantic or geometry mismatch |
| Predict each dynamic occurrence | `V4DmaServiceProvider` occurrence sequences | Provider consumption checks and exact traffic parity |
| Add practical decoder-layer scaling | CostEmitter one-layer schedule and stage multipliers | Two-layer differential error 1.82% with exact request/byte parity |
| Integrate accepted V4 into DSE | `compiler_cost_model.py`, `run_optuna_dse.py` | 3-trial single-worker and 12-trial four-worker smoke tests |

This table is also the audit boundary: a code change without a corresponding
test or artifact should not be treated as validated merely because the final
aggregate metric remains low.

## 6. V4 Model Design

### 6.1 Grouped occurrence model

V4 fits one model for each:

```text
opcode x HBM channel count
```

The nine cold/mixed groups are:

```text
H_PREFETCH_M x {8, 32, 128 channels}
H_PREFETCH_V x {8, 32, 128 channels}
H_STORE_V    x {8, 32, 128 channels}
```

This separation is intentional. Startup cost, channel occupancy, and
bank-conflict behavior do not scale through a single channel-count
coefficient. The warm residual is separately fitted for vector prefetch and
store groups where row-hit anchors are available.

### 6.2 Physical transfer floor

Each 64-byte line consists of four 16-byte native HBM bursts. MOP4CLXOR maps
each burst to a channel, pseudochannel, bank group, bank, row, and column. For
each sequential read or write phase, V4 computes:

```text
phase_floor_bursts = max(
    busiest_channel_bursts,
    2 * busiest_pseudochannel_commands
)

theoretical_phase_floor =
    burst_service_time * (read_floor_bursts + write_floor_bursts)
```

Read and write floors are summed because a partial-line scatter waits for the
read phase before issuing writes. The model prediction is never allowed below
this floor.

### 6.3 Residual features

For one occurrence, the cold/mixed residual feature vector is:

```text
read_phase_startup
write_phase_startup
read_write_turnaround
read_channel_tail
write_channel_tail
read_bankgroup_serial
write_bankgroup_serial
read_bank_serial
write_bank_serial
read_row_miss
write_row_miss
read_row_conflict
write_row_conflict
sram_dma_drain
```

The feature roles are:

- Phase startup terms capture fixed command and controller costs.
- Turnaround captures the boundary between store read and write phases.
- Channel-tail terms measure critical-channel imbalance above the average.
- Bank-group and bank terms measure command serialization pressure.
- Row miss and conflict terms distinguish closed-bank activation from row
  replacement.
- `sram_dma_drain` combines `log2(amount + 1)` with request density per
  channel, representing transfer setup and drain behavior not included in the
  HBM bus floor.

The strict warm feature vector is intentionally smaller:

```text
read_phase_startup
write_phase_startup
read_row_conflict
write_row_conflict
```

The physical transfer floor already represents unavoidable data movement. A
large warm feature set made one coefficient vector compromise between two
different physical regimes and increased tail error.

### 6.4 Regression form

For group `g`, the cold/mixed model is:

```text
latency_g = max(
    floor,
    floor + beta_g^T x
)

subject to beta_g >= 0
```

The warm regime uses the same form with its reduced feature vector and separate
coefficients. Coefficients are fitted with nonnegative ridge regression. The
training residual is weighted by inverse observed latency so that small
control-sized transfers and large production transfers both affect the fit.

Nonnegative coefficients are a deliberate interpretability constraint. More
contention, more row conflicts, or more serialized work must not reduce
predicted service time. The theoretical floor prevents the regression from
predicting a latency below unavoidable bus occupancy.

### 6.5 Domain tracking

The artifact stores the training min/max for every feature, request format
signatures, and row-state regime. Runtime predictions report:

```text
calibration_in_domain
domain issues
extrapolation ratio
row-state regime
```

Out-of-domain points are not silently treated as calibrated. In the formal
`compute-objective` DSE mode they may remain as memory-shadow observations; in
the experimental integrated-memory objective they can be pruned.

## 7. Calibration Dataset

### 7.1 Dataset size and split

```text
Generic production-DMA points:       2,592
Targeted row-state anchors:             36
Total points:                         2,628

Generic training points:              2,007
Row-state training anchors:              36
Total training points:                2,043
Generic holdout points:                 585
```

Every point is measured three times with no hidden warmup run. The median
production DMA completion cycle is the fit target.

### 7.2 Covered hardware dimensions

```text
HBM channels: 8, 32, 128
Dimensions:   64, 128, 256, 512, 1024, 2048
Opcodes:      H_PREFETCH_M, H_PREFETCH_V, H_STORE_V
```

Matrix amounts are derived from `dim/4`, `dim/2`, and `dim`. Vector/store
amounts use unique valid values from `4, 16, 64, 256, 1024, dim`.

### 7.3 Covered precision geometries

The request geometry covers:

```text
4-bit element, 8-bit scale, block 64
8-bit element, 8-bit scale, block 64
8-bit element, 8-bit scale, block 8
```

These geometries represent:

- MXINT4 and MXFP4 (`E1M2`, `E2M1`) where storage geometry is equivalent.
- MXINT8 and MXFP8 (`E4M3`, `E5M2`) at block 64.
- The existing block-8 MXFP8 transactional configuration.

The service model predicts physical request behavior. Formats with identical
element, scale, and block widths therefore share request geometry even if their
numerical exponent/mantissa interpretation differs.

### 7.4 Covered layouts

Each generic geometry includes:

```text
aligned contiguous
32-byte-offset contiguous
aligned 2x stride
32-byte-offset large stride
```

This layout set intentionally exercises line alignment, partial lines,
channel balance, bank conflicts, and sparse channel utilization.

### 7.5 Targeted row-state anchors

The 36 anchors contain repeated vector prefetch/store accesses across channel
counts and representative precision geometries. They distinguish exact open-row
hits from initial conflicts. Of these, 35 are exact-row-hit observations and
one is an initial-conflict stress point.

## 8. Validation Methodology

Validation is organized as a ladder. Passing a higher level requires the
lower-level semantics to be correct first.

### 8.1 Level 1: request-planner unit tests

The production planner is checked for:

- A fully covered store line: zero reads and one write.
- A partial store line: one read and one write.
- Overlapping element/scale fragments: one coalesced physical line operation.
- Cross-line fragments: correct splitting and coverage masks.
- Duplicate gather lines: one physical read.
- MXINT and MXFP packed-byte accounting.
- Completion-aware Ramulator writes and final drain.

These tests establish that the calibration target executes the intended DMA
semantics. They do not, by themselves, validate the regression.

### 8.2 Level 2: Rust/Python request parity

For all 2,628 points, the Rust production execution and Python analytical
planner must match:

```text
read line count
write line count
full/partial line count
read/write bytes
native burst count
canonical request-manifest hash
```

Result:

```text
Point IDs matched:                 2,628 / 2,628
Three samples present:            2,628 / 2,628
Request-manifest mismatches:                  0
```

This is important evidence that prediction error is regression error rather
than hidden request-generation drift.

### 8.3 Level 3: generic held-out prediction

The 585 generic holdout points are not used to fit coefficients.

| Metric | Result | Acceptance | Status |
|---|---:|---:|---|
| Median absolute error | 3.48% | <= 10% | PASS |
| P95 absolute error | 18.10% | <= 25% | PASS |
| Maximum absolute error | 42.18% | <= 60% | PASS |
| Weighted MAPE | 2.80% | Reported | - |
| `H_STORE_V` P95 | 19.92% | <= 20% | PASS |
| Manifest mismatches | 0 | 0 | PASS |

The maximum error shows that V4 is not uniformly precise. P95 is the main tail
metric used for promotion because isolated geometries can still be difficult
for a post-hoc linear residual model.

### 8.4 Level 4: row-state anchor diagnostics

| Anchor subset | Samples | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| All row-state anchors | 36 | approximately 0% | 5.69% | 36.19% |
| Exact row hits | 35 | approximately 0% | 3.94% | 6.21% |
| Initial-conflict stress | 1 | 36.19% | 36.19% | 36.19% |

The single initial-conflict point remains difficult, but runtime regime
selection routes inherited initial conflicts to the cold/mixed model rather
than incorrectly labeling them as fully warm. This classification change was
the main row-state correction that reduced the generic tail.

### 8.5 Level 5: production Qwen system traces

The system validation does not rely only on the original transactional DMA
durations. The observed trace provides the first arrival-time seed. The full
production DMA sequence is then replayed through Ramulator until bounded
arrival, physical delay, HBM work, and makespan deltas converge. Every replay
also verifies full request-manifest parity.

Per-opcode and aggregate errors are:

| Production trace | `H_PREFETCH_M` | `H_PREFETCH_V` | `H_STORE_V` | Total HBM work | Makespan |
|---|---:|---:|---:|---:|---:|
| Qwen3-32B, seq128, M256/B64, E1M2 b64, c8 | 0.74% | 9.44% | 6.37% | 0.79% | 0.208% |
| Qwen3-32B, seq482, M512/B64, E4M3 b8, c128 | 3.47% | 3.37% | 8.11% | 3.38% | 0.086% |
| Qwen3-8B, seq128, M256/B32, MXINT8 b64, c32 | 11.61% | 14.03% | 11.72% | 11.70% | 0.573% |
| Qwen3-8B, seq64, M128/B16, MXINT4 b64, c128 | 6.22% | 8.46% | 6.43% | 6.07% | 0.158% |

Acceptance requirements and results:

```text
All required cases present:                    PASS
Arrival-aware replay converged:                PASS
Request-manifest parity:                       PASS
Each opcode work error <= 25%:                 PASS
Total HBM work error <= 20%:                   PASS
Scheduled makespan error <= 10%:               PASS
Numerical path/correctness gate unchanged:     PASS
```

The total HBM work error is the correct metric for assessing V4 memory service
accuracy. The much smaller makespan error occurs because compute dominates much
of the critical path; it must not be presented as the standalone memory-model
error.

### 8.6 Level 6: fast DSE scaling validation

`full-global-stateful` prediction walks every dynamic DMA occurrence in global
issue order with one persistent MOP4CLXOR row state. It is the validation mode
and the only mode allowed to drive scheduled replay.

For practical DSE, `one-layer-stateful-scaled` predicts one complete decoder
layer statefully, scales only `layer/*` stages by the decoder-layer count, and
keeps global prologue/epilogue stages single-counted.

A two-layer Qwen3-8B differential test uses:

```text
seq_len=64, batch=1
MLEN=VLEN=128, BLEN=16
MXINT4, block=64, channels=128
```

Results:

```text
Full-global-stateful HBM work:       1,947,874.101 ns
One-layer-scaled HBM work:           1,983,392.735 ns
Relative HBM work difference:               1.823%

DMA occurrence count:                   exact match
Read requests and bytes:                exact match
Write requests and bytes:               exact match
```

The remaining difference is attributable to the second layer inheriting the
first layer's global open-row history. It is not caused by missing DMA
occurrences or incorrect traffic scaling.

### 8.7 Regression and integration tests

Completed verification includes:

```text
cargo fmt --check --all:                        PASS
cargo check --workspace:                        PASS
cargo test --workspace:                         PASS
Python performance-model suite:          74 tests PASS
Final V4/CostEmitter/DSE regression:      29 tests PASS
Python static compilation:                      PASS
git diff --check:                               PASS
```

The historical V3 reproduction test remains, but it now explicitly expects a
DMA semantic/geometry compatibility mismatch. V3 numerical reproduction is
preserved for historical comparison without presenting it as production-DMA
compatible.

## 9. Why the P95 Error Improved

The reduction from approximately 39.03% V3 P95 to 18.10% V4 P95 has five
causes.

### 9.1 Correct prediction target

V4 trains directly on production DMA completion cycles. It does not reuse the
old sequential raw-store labels. This removes a systematic error that no
feature engineering could have fixed.

### 9.2 Exact request parity

Rust and Python must agree on the full physical manifest before fitting. This
eliminates hidden differences in packed bytes, scale traffic, line alignment,
and read-modify-write behavior.

### 9.3 Critical-path rather than average traffic

The transfer floor and residual features use the busiest mapped channel,
pseudochannel, bank group, and bank. This is more representative for strided
layouts than dividing total requests by the channel count.

### 9.4 Opcode and channel specialization

Nine separate models prevent a single global coefficient vector from
compromising between read, store, and different channel-count regimes.

### 9.5 Explicit row-state regimes

The 36 targeted anchors distinguish true repeated-row hits from cold misses and
inherited conflicts. The small warm model avoids forcing one large cold feature
set to explain low-latency row-hit behavior.

The P95 improvement is measured on a held-out set. It is not a training-error
comparison and is not caused by the one-layer DSE scaling optimization.

## 10. Runtime Cost and DSE Integration

### 10.1 Exact global-state cost

For Qwen3-32B, batch 16, seq 482, 64 decoder layers:

```text
Dynamic DMA occurrences: approximately 746,056
Observed diagnostic runtime before interruption: > 6 min 25 s
Peak resident memory before interruption:        approximately 13.4 GiB
Completion status:                               not completed
```

This negative result is retained deliberately. It demonstrates that exact
global occurrence expansion is not suitable for each DSE trial.

### 10.2 Fast scaled cost

The one-layer-stateful-scaled path processes approximately 14,437 representative
DMA occurrences:

```text
Cold command runtime: approximately 24.8 s
Peak resident memory: approximately 944 MiB
```

A per-process CostEmitter cache reuses reports for repeated identical hardware
and precision configurations.

Measured DSE smoke results:

```text
3 trials / 1 worker:   3 complete, 0 failed
12 trials / 4 workers: 12 complete, 0 failed, approximately 61.7 s wall time
```

### 10.3 Current DSE semantics

The Qwen3-32B DSE defaults to:

```text
compiler cost mode:          compute-objective
compute timing:              rtl-v1 calibrated resource work
memory artifact:             accepted V4
V4 evaluation:              one-layer-stateful-scaled
formal latency objective:    compute resource work only
memory role:                 shadow
bandwidth guard:             legacy bandwidth prune
```

Each trial records:

```text
compiler_compute_latency_ms
compiler_memory_latency_ms
compiler_serial_latency_ms
compiler_memory_evaluation_mode
calibration_in_domain
domain and extrapolation issues
memory model/calibration identity
```

The experimental `objective` mode can include V4 memory in serial latency, but
it is not the recommended formal result. Out-of-domain memory predictions are
pruned in that mode.

## 11. Design Rationale for Thesis Use

### 11.1 Why use a surrogate instead of raw bandwidth

A bytes/bandwidth equation cannot represent startup latency, partial-line
read-modify-write, channel imbalance, bank serialization, row misses, or row
conflicts. These effects are visible in production request geometry and matter
for small or irregular DMA operations common in transformer execution.

### 11.2 Why retain a theoretical floor

A purely statistical model can predict physically impossible latency below
the time required to transfer the busiest channel's bursts. The floor embeds a
minimal hardware constraint while leaving Ramulator-specific residual cost to
calibration.

### 11.3 Why use nonnegative coefficients

The model is intended for design-space exploration and extrapolation. Negative
contention coefficients may reduce training error while producing incorrect
trends when hardware dimensions change. Nonnegative fitting preserves the
interpretation that additional serialized work cannot improve service time.

### 11.4 Why use semantic hashes

Model validity depends on DMA request semantics, not only on coefficient file
paths. The artifact records semantic versions, request size, burst size,
Ramulator preset, mapper, fixture hash, calibration ID, promotion evidence, and
evidence SHA-256. A code refactor that changes physical request geometry can
therefore invalidate the artifact explicitly.

### 11.5 Why keep V4 as a shadow

The generic and system results support using V4 as a significantly more
informative memory estimate than V3. They do not demonstrate exact online
interaction among multiple concurrent DMA queues. Keeping compute resource work
as the formal objective avoids turning an acknowledged post-hoc approximation
into an unsupported full-system claim.

### 11.6 Why provide two fidelity modes

The exact global mode is valuable for validation but too expensive for large
DSE studies. The one-layer scaling mode exposes its approximation explicitly,
preserves exact traffic counts, and has a measured two-layer error. This is more
defensible than silently truncating the trace or using an unvalidated aggregate
division.

## 12. Limitations and Threats to Validity

1. **Post-hoc scheduling.** V4 does not update one online Ramulator instance as
   multiple DMA engines and compute resources progress concurrently.
2. **Calibration domain.** Production Qwen traces contain some feature values
   outside generic training bounds. These are reported but still involve
   extrapolation.
3. **Limited system cases.** Four Qwen traces cover multiple model sizes,
   precisions, channels, and shapes, but they are not an exhaustive workload
   suite.
4. **Compute dominance.** Small makespan error can hide larger memory-work error.
5. **Layer scaling.** The fast mode does not preserve inherited row history
   across all 64 decoder layers.
6. **No completed exact 64-layer run.** The attempted global evaluation was
   interrupted because of time and memory growth; it must not be reported as a
   passed validation.
7. **Fixed Ramulator setup.** Results apply to `HBM2_2Gbps + MOP4CLXOR` and the
   configured channel counts. They are not a physical A100 HBM topology model.
8. **No Fmax validation.** Cycle-to-time conversion assumes 1 ns per cycle.
9. **Correctness excluded.** Numerical quantization and comparison semantics
   were intentionally held constant.

## 13. Reproduction

### 13.1 Generate the plan

```bash
python -m analytic_models.performance.compiler_cost_model \
  generate-service-v4-plan \
  --row-state-anchors \
  --output analytic_models/performance/calibration/hbm_dma_v4_plan.json
```

### 13.2 Run production DMA calibration

```bash
nix develop --command bash -lc '
cd transactional_emulator
cargo run --release --bin hbm_dma_calibration -- \
  --input ../analytic_models/performance/calibration/hbm_dma_v4_plan.json \
  --output ../analytic_models/performance/calibration/hbm_dma_v4_results.json
'
```

### 13.3 Fit and validate a candidate

```bash
python -m analytic_models.performance.compiler_cost_model \
  fit-service-v4 \
  --plan analytic_models/performance/calibration/hbm_dma_v4_plan.json \
  --results analytic_models/performance/calibration/hbm_dma_v4_results.json \
  --output /tmp/hbm_dma_service_v4_candidate.json \
  --validation-output /tmp/hbm_dma_service_v4_validation.json
```

### 13.4 Promotion

The candidate is promoted only after the generic holdout and all required
system cases pass:

```bash
python transactional_emulator/testbench/rtl_timing/\
aggregate_hbm_v4_system_validation.py \
  --generic-validation /tmp/hbm_dma_service_v4_validation.json \
  --case qwen3_32b_seq482_m512_b64_mxfp_e4m3_b8_c128=analytic_models/performance/calibration/hbm_dma_service_v4_system_qwen3_32b_e4m3_b8_c128.json \
  --case qwen3_8b_seq64_m128_b16_mxint4_b64_c128=analytic_models/performance/calibration/hbm_dma_service_v4_system_qwen3_8b_mxint4_c128.json \
  --case qwen3_8b_seq128_m256_b32_mxint8_b64_c32=analytic_models/performance/calibration/hbm_dma_service_v4_system_qwen3_8b_mxint8_c32.json \
  --case qwen3_32b_seq128_m256_b64_mxfp_e1m2_b64_c8=analytic_models/performance/calibration/hbm_dma_service_v4_system_qwen3_32b_e1m2_c8.json \
  --output /tmp/hbm_dma_service_v4_system_validation.json \
  --model /tmp/hbm_dma_service_v4_candidate.json \
  --promoted-model-output analytic_models/performance/calibration/hbm_dma_service_v4.json
```

The exact system-case arguments are retained in the validation tooling and
individual system artifacts.

### 13.5 DSE usage

```bash
nix develop --command bash -lc '
source .venv/bin/activate
python Workspace/qwen3_32b_dense_analytic/run_optuna_dse.py \
  --n-trials 1000 \
  --workers 8 \
  --compiler-cost-mode compute-objective \
  --compiler-compute-timing rtl-v1 \
  --compiler-v4-memory-evaluation one-layer-stateful-scaled \
  --legacy-bandwidth-prune \
  --run-dir Workspace/qwen3_32b_dense_analytic/runs/v4_dse
'
```

## 14. Evidence Artifacts

| Artifact | Purpose |
|---|---|
| `hbm_dma_v4_plan.json` | Complete deterministic calibration plan and expected manifests |
| `hbm_dma_v4_results.json` | Three production DMA measurements per point |
| `hbm_dma_service_v4.json` | Accepted fitted model, domains, semantics, and promotion metadata |
| `hbm_dma_service_v4_validation.json` | Generic holdout and row-state diagnostics |
| `hbm_dma_service_v4_system_validation.json` | Aggregated four-case promotion evidence |
| `hbm_dma_service_v4_system_qwen3_*.json` | Per-Qwen production replay evidence |
| `hbm_dma_service_v4_scaling_validation.json` | Full-stateful versus one-layer-scaled differential check |

All official artifacts are under:

```text
analytic_models/performance/calibration/
```

Important identities:

```text
Calibration ID:
  hbm-production-dma-v4-1b00531de9a61298

Promotion evidence SHA-256:
  2bd6248babfb64edfe07f7a2f406fae1787a6a5d4470c9f1cddb82656c3aecfc

Scaling-validation model SHA-256:
  1567e2fb8e77c7792c15f16211ab08e3de2f9ff3fe69d83a2c021e7ecd6253f6
```

## 15. Defensible Claims

The following statements are supported by current evidence:

1. V4 uses the same packed request semantics as the production transactional
   DMA path, with zero manifest mismatches across 2,628 calibration points.
2. V4 reduces generic holdout P95 absolute error from the historical V3 value
   of 39.03% to 18.10% against production-DMA completion labels.
3. Four production Qwen traces have total HBM work error between 0.79% and
   11.70% under arrival-aware replay.
4. The fast layer-scaled path preserves request/byte counts exactly and differs
   from a two-layer global-stateful prediction by 1.82% in aggregate HBM work.
5. V4 is suitable as an accepted DSE memory shadow for the tested semantic and
   hardware domain.

The following statements are not supported and should not appear in a paper:

1. V4 is a cycle-exact full-system memory simulator.
2. Scheduled makespan is universally accurate to below 1%.
3. The fast scaled mode reproduces exact 64-layer global row history.
4. V4 models a physical A100 HBM2e topology.
5. V4 validates numerical quantization accuracy or RTL timing closure.

## 16. Recommended Next Work

1. Add online Ramulator state integration for the HBM queues used by scheduled
   CostEmitter replay.
2. Add more memory-bound production traces so makespan validation is not
   dominated by compute.
3. Extend system validation to more batch sizes and sequence lengths inside
   and outside the current feature domain.
4. Complete a resource-controlled multi-layer global-stateful run and compare
   layer-scaled error beyond two layers.
5. Refit only if additional production labels reduce current tail errors
   without weakening manifest parity or physical monotonicity.
