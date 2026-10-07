# Reproduce this study

Workspace: `/scratch/shared/mcl123/plena`.
Python: `venvs/plena-py311/bin/python` (NumPy, PyTorch, Transformers, safetensors).
New source: `review_20260921/simulator-moe-batch-real`.
The frozen finite performance binary is the **v3** executable in the 20260919
fabric archive. The opt-in actual-operand binary is
`repro/moe_spatial_fabric_real`. SHA256 and source snapshots are in `repro/`.

The commands below are examples from the workspace root. Existing receipts are
checked before reuse where a runner supports resume. Use a fresh output root by
editing the common OUT constant and dependent input paths for a new experiment;
do not overwrite frozen results while changing parameters.

```bash
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/prepare_routes.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/profile_full_archive.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/run_routes.py --workers 8
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/run_sensitivities.py
```

The original nine NPZ archives are required; their paths/hashes are in
`inputs/archive_inventory.json`. Window selection and the complete route/shape
manifest are in `inputs/route_windows.json`. `population_by_batch.csv` is the
whole-archive **distribution**; the simulator's timing coverage is 72 windows.

The real checkpoint capture is already preserved in `real_inputs/capture.pt`
and `x16.bf16`. To recreate it using the exact full prompts and local original
DeepSeek checkpoint, run `capture_real.py`. The BFCL source file's pinned GitHub
URL, commit and SHA are in `inputs/BFCL_v3_simple.json.provenance.json`. The
capture manifest records every token ID, lack of truncation, versions, source
tensor name/shape/hash and the last-token prefill scope. No GPU is required.

Large raw weight exports live in a task-specific `/tmp` directory. They can be
recreated from the original safetensors without rerunning the model prefix:

```bash
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/restore_operands.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/run_real.py --workers 4
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/real_compute_and_timing_audit.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/run_operand_ports.py
```

Each real configuration contains six requests, six external-operand manifests,
input activations including actual generated down-projection inputs, compressed
reports, two-run hashes, FP64 projection errors, cross-architecture value hashes,
and final connected MoE comparison against PyTorch. Missing hashes are not filled
with estimates. All source/output tensor byte orders and arithmetic conventions
are defined in `METHODS.md`.

After the corresponding compact route campaign is complete, reconstruct the
full traces and waiting categories, then generate the final tables:

```bash
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/run_breakdown.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/trace_real_reversals.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/verify_regression.py
venvs/plena-py311/bin/python outputs/moe_spatial_batch_real_20260921/scripts/summarize.py
```

Rust build environment is the existing frozen
`outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step0/repro/build-env.sh`.
Source that environment and run from the new worktree's
`transactional_emulator` directory:

```bash
cargo test --workspace
cargo clippy --release --bin moe_spatial_fabric -- -D warnings
cargo build --release --bin moe_spatial_fabric
```

This uses the existing pinned Nix/Rust/native-library environment. It is not a
claim that a machine without those dependencies can build completely offline.
Test logs, input rejection regression, frozen complete-output regression and
actual-value/frozen-timing event equality are archived.

Timing excludes a complete model, original HBM/Ramulator, the MX codec,
cross-phase nonlinear/router/merge timing and physical implementation costs.
The real layer's six-phase sum must never be relabeled whole-layer latency.
