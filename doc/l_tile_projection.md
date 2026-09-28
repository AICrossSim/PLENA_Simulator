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
export PYTHONPATH="$PWD:$PLENA_COMPILER_ROOT"
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
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" \
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

## Next work, not claimed as implemented

1. Establish the best legal original Matrix and ordinary Vector baseline.
2. Evaluate request/head-group residency jointly with weight reuse.
3. Specify and charge an actual bounded Vector/Matrix SRAM handoff interface.
4. Validate the final arithmetic over long real-input chains and task quality.
5. Validate whole-model composition, capacity, routing and held-out operators.
6. Integrate the candidate datapath before making timing, area or power claims.

WS/IS/OS template selection and direct producer-consumer handoff remain design
candidates. The standalone protocol probe is not evidence of zero-cost overlap.

## Publication checks

The Rust workspace passed 328 tests, formatting and clippy across all targets.
The analytical Python suite passed 268 tests with nine explicit skips. The new
portable campaign reproduced Mamba B1 at 3,063,999 baseline and 2,909,800 batch
cycles. These checks do not replace the archived full-shape numerical campaign
or change its one-token scope. Compiler-wide legacy failures are recorded in
the companion Compiler's `doc/l_tile_projection.md`.
The portable `--only projection` numerical runner also passed all ten
B1/2/4/8/16 × one/four-request cases with distinct inputs and K2305/N65 tails,
including independent cycle and traffic reconciliation.
