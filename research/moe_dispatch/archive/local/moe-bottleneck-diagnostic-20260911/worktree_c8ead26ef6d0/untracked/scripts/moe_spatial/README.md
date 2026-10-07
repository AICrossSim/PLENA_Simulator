# Spatial-M compute mechanism experiment

This directory is an independent experiment for **physical M row lanes**. Read
`../../doc/moe_spatial_m_contract.md` first. It does not change `moe_normal`, its
Compiler contract, the accepted Step1/Step2 controls or the old oracle results.

From `transactional_emulator`, build and test:

```bash
source /scratch/shared/mcl123/plena/outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step0/repro/build-env.sh
cargo test --workspace
cargo clippy --bin moe_spatial_m -- -D warnings
cargo build --release --bin moe_spatial_m
```

From the repository root, run into a **new** output directory (the runner refuses
to reuse its `requests`, `reports` or `inputs` subdirectories):

```bash
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial/run_experiment.py \
  --binary /tmp/plena-moe-dual-core-target/release/moe_spatial_m \
  --output /tmp/plena-spatial-rerun/results \
  --windows /scratch/shared/mcl123/plena/outputs/moe_refinement_20260909/fixed_bank_full_qwen/windows
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial/summarize.py \
  --results /tmp/plena-spatial-rerun/results
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial/plot_results.py \
  --results /tmp/plena-spatial-rerun/results
```

The run produces 202 points with two identical repetitions each. Positive and
negative numerical mechanism tests execute first; the small trace sweep is
gated on their success. Both asymmetric core-index orders are measured because
FIFO tile stealing uses core index to break same-cycle ties. No parameter is
fitted to produce a positive result.

The trace sweep covers only a total physical M budget of six, fixed N=4/K=512
lanes, and the documented dispatch policies. “Best” means best observed in that
bounded space. The batch oracle enumerates all positive two-way partitions and
assumes free repartitioning between batches; it is not a general optimum over
arbitrary dynamic array partitioning or all possible schedules.

B8's routes are exactly the prefix of the B32 archive. B8 selects the shape;
validation uses only B32 tokens 8–31. These tokens are disjoint, but still from
the same archive family. Full B32 is labeled an overlapping reference.

`run_experiment.py` independently audits every invocation and checks actual
numerical runs against an integer reference. It saves each request, a compressed
complete report, hashes for both repetitions and separate CSV tables.
The output distinguishes 154 numerical points (generated BF16 operands) from
48 shape-only points. None is a full-model execution or native HBM test.

`summarize.py` derives the compute lower bound, selection/validation results and
the Chinese report. Its fixed report prose describes this version of the
experiment; update and review it if the experiment contract changes.

Primary timing is the explicitly assumed L25/II1 pipeline, with L1/II1 and
L25/II25 sensitivity points. All operands/accumulator interfaces are ideal;
finite in-flight result storage and per-output K ordering remain enforced.
Reported logical weight elements are core operand consumption, **not** HBM
traffic: caches and broadcast may reduce physical movement.
