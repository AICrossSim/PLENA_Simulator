# Reproduce the fixed-design study

Use the paired local research branches `research/moe-robust-fixed-design`.
Simulator pins the Compiler submodule; `PLENA_DISPATCH_COMPILER` is optional
and only needed when developing both repositories in separate worktrees.
No native Simulator build, external network access or model weight download is
needed for these analytical timings.

## Build and verify

From the Simulator root, with Python 3.11+, NumPy and a Rust 2024 toolchain:

```sh
export CARGO_TARGET_DIR=/tmp/plena-robust-target
cargo test --manifest-path research/moe_dispatch/rust/Cargo.toml
cargo build --release --manifest-path research/moe_dispatch/rust/Cargo.toml
export PLENA_DISPATCH_TEST_BINARY="$CARGO_TARGET_DIR/release/moe-dispatch-analytical-v1"
python -m unittest discover -s research/moe_dispatch -p 'test*.py'
python -m unittest discover -s PLENA_Compiler/research/moe_dispatch -p 'test*.py'
```

The execution environment used:

```sh
export RUSTUP_HOME=/scratch/shared/mcl123/plena/rustup
export CARGO_HOME=/scratch/shared/mcl123/plena/cargo
export PATH="$CARGO_HOME/bin:$PATH"
```

Python executable:
`/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python`.

## Inputs and full campaign

The archived `inputs/*_workloads.json` and `workload_manifest.json` are sufficient
to replay the timing campaign. They retain captured IDs, route scores, shapes
and addresses, but do not contain pretrained numerical weights/activations.

To regenerate input windows from the original route archives, call
`robust_workloads.prepare(route_root, output_root / "inputs")`. `route_root` is
`/scratch/shared/mcl123/plena/outputs/real_shared_moe_routes_20260819`.
The workload manifest records SHA256 and request selection. Reproduction does
not require gathering new routing traces or replacing the stored inputs.

With `$study_output` pointing at a separate result directory containing `inputs`:

```sh
python research/moe_dispatch/robust_study.py plan --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY"
python research/moe_dispatch/robust_study.py diagnostic --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY" --workers 28
python research/moe_dispatch/robust_compute_assignment.py --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY"
python research/moe_dispatch/robust_study.py elasticity --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY" --workers 28
python research/moe_dispatch/robust_study.py design --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY" --workers 28
python research/moe_dispatch/robust_study.py validation --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY" --workers 28
python research/moe_dispatch/robust_study.py heldout --output "$study_output" --binary "$PLENA_DISPATCH_TEST_BINARY" --workers 28
python research/moe_dispatch/robust_report.py --output "$study_output"
```

Stages are dependent and must run in this order. Validation writes
`frozen_designs.json`; heldout reads that file and never reselects. The plan
records the selection objective before timing. Output points carry input,
configuration, source bundle and binary hashes plus repeat receipts.

For a single archived point, invoke its recorded binary with `workload.json`,
`config.json`, and a fresh report path; compare that report with the stored
`report_repeat1.json`. Relocating the archive changes point-directory paths,
not the simulator's JSON result. `raw_directory` columns retain original
execution paths; resolve their suffix below the archived study root.

## Scope of the archive

The authoritative rows are those referenced by each final stage CSV/JSON.
Unreferenced point directories can include superseded development checks and
must not be mixed into the formal dataset. Historical reports outside this
new study directory remain unchanged. `METHODS.md` distinguishes projection
oracle, whole FFN analytical timing, and the partial real-route diagnostic.
