# Projection and native recurrent sublayers

## Implementation checkpoint

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
