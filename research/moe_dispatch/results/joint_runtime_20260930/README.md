# Frozen joint-runtime evidence — 2026-09-30

This is a published snapshot of an already completed experiment. No engine,
hardware allocation, result or acceptance threshold changed during publication.
The pinned Compiler submodule contains the matching planner.

## Read first

- `REPORT_ZH.md`: independent test results and claim limits.
- `frozen_hardware.json`: all six exact resource allocations.
- `all_results.csv`: 960 development/order/test points; each ran twice.
- `policy_summary.csv`, `architecture_same_policy.csv`: paired comparisons.
- `shared_b16_capacity_diagnostic.json`: why frozen 5+3 cannot place B16 Shared
  on the larger core. This is a storage feasibility issue, not a preference.
- `mechanism/REPORT_ZH.md`: 120 synthetic complete-FFN points, each run twice;
  includes both the 4,2 benefit and the 3,3 counterexample.
- `FILES_SHA256.json`: hashes of the copied evidence, including replay cases.

All times are analytical Rust MoE-FFN times, including modeled data movement,
expert projections and combine. Routes and input activations are available at
entry. They are not router/full-model, native Ramulator, RTL or silicon times.
Physical dimensions are M x N x K, with N=4 and K=512. Compare M6 designs only
with M6, and M8 only with M8. Every cycle is modeled as 1 ns.

Supply is shared 256 B/ns with a 64 ns response and 256 credits of 32 B each;
credit occupancy alone bounds sustained supply to at most 128 B/ns before
landing overhead. The study retains these settings rather than silently
increasing memory resources.

## Build and replay the six included complete cases

From the Simulator repository root:

```bash
git submodule update --init PLENA_Compiler
export CARGO_TARGET_DIR=/tmp/plena-joint-review-target
cargo build --release --locked --manifest-path research/moe_dispatch/rust/Cargo.toml
python3 research/moe_dispatch/results/joint_runtime_20260930/replay_cases.py \
  --binary "$CARGO_TARGET_DIR/release/moe-dispatch-analytical-v1"
```

The script verifies evidence hashes, runs each case twice in a temporary output
directory, and compares each complete JSON report against the original bytes.
Cases cover BFCL B8 for all M6 organizations and BFCL B16 for all M8 organizations.
These are reproducibility samples, not a replacement for the full result table.
No model-weight downloads or original capture files are needed.

## Rerun the full independent test

Copy `plan.json`, `prepare_receipt.json`, `frozen_hardware.json`,
`workload_manifest.json`, and `inputs/` into a new writable output directory.
Run the following from the repository root, without invoking `prepare` (which
would reselect inputs):

```bash
python3 research/moe_dispatch/joint_study.py run \
  --output /path/to/new-output --split test --policies all --workers 4 \
  --binary "$CARGO_TARGET_DIR/release/moe-dispatch-analytical-v1"
```

Original point manifests and receipts retain their original absolute paths and
source/binary hashes as provenance. Use the relative replay paths above on a
different machine. A rebuilt executable can have a different binary hash;
complete report equality is checked independently.

## Archive boundary

This Git snapshot contains code, extracted route inputs, all-point result tables,
diagnostic summaries and six pairs of raw reports. It does not contain all raw
campaigns, build outputs, Git bundles or model tensors. The complete original
delivery is preserved locally as `plena-moe-joint-runtime-20260930.tar.gz`, SHA256
`0029effd43fed86567303ee341c21665012535da06c9c5fe03057dd5118e9ece`.
Earlier surplus-admission tables and methods remain alongside this directory;
their workload selection differs and their times must not be mixed into this study.
