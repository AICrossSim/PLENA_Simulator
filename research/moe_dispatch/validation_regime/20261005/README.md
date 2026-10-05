# Regime evaluation, 2026-10-05

The six-regime analytical campaign and independent audits are complete. None of
six heterogeneous candidates clears the preregistered 5% reduction against
**both** the optimized single and optimized homogeneous baseline. Candidate
native calibration is therefore skipped by the approved gate. No hardware
victory, trained-model quantization accuracy or synthesis-based area/energy
saving is claimed. The compiler is unchanged in this campaign.

Read `REPORT_ZH.md` and `COMPARISON.csv` first. Milliseconds use assumed 1 GHz;
the table averages the same 135 captured post-router FFN windows geometrically.
It excludes Router, Attention and complete generation. This is a phase-level
fluid analytical model, not cycle-exact native measurements. Every model point
is repeated twice; 71 tests pass; 945 frozen v6 windows reproduce exactly.

PM/PN/PK vary, including independent PK32/64/128/256/512/1024. The selected
single is 6x16x128 and the selected homogeneous pair is 6x16x64+6x16x64.
Private capacities and W/X/accumulator banks are refined independently under
fixed total budgets. Hardware and runtime freeze across all batches within a
regime before heldout evaluation. This is bounded hierarchical joint search,
not an exhaustive optimum over every possible controller or dataflow.

The owner is a whole expert; cross-core N-band splitting, elastic bank leasing
and a new trained dispatch predictor are not part of this search. W8/W4 are
transport/decoder assumptions with BF16 operand storage, not a QERA/LQER path
or a validated quantized inference implementation. Details and finite-decoder
sensitivity appear in the report. Service counters overlap; do not add them
into a wall-clock breakdown. Equal capacity and MACs do not mean equal area.

## Reproduce

From the simulator repository root in this research branch, install NumPy,
pytest and matplotlib; optional quant probes also need PyTorch, safetensors and
the actual DeepSeek-V2-Lite pretrained weights. Choose empty directories below.

```sh
mkdir /absolute/path/to/regime-inputs

tar -xzf research/moe_dispatch/validation_regime/20261005/captured_inputs.tar.gz \
  -C /absolute/path/to/regime-inputs

tar -xzf research/moe_dispatch/validation_regime/20261005/compatibility_baseline.tar.gz \
  -C /absolute/path/to/regime-inputs

python -m research.moe_dispatch.regime.run \
  --inputs /absolute/path/to/regime-inputs/inputs \
  --old-system /absolute/path/to/regime-inputs/old-system \
  --out /absolute/path/to/new-results --workers 32
```

Optionally add `--weights /absolute/path/to/deepseek-v2-lite-chat` for the
numerical probe. Without weights the runner explicitly leaves task accuracy
unqualified. The runner stops if checks fail, verifies repeats, freezes source
and writes a manifest. `REPRODUCTION_INPUTS.json` hashes both input bundles.
The environment and original base commits are in `ENVIRONMENT.json`.

Compact summaries, frozen selections, all near-best1% sets, gate statistics,
per-batch attribution and audit receipts are versioned here. The full source,
inputs, every evaluated candidate and selected raw result are preserved in the
local durable archive located by `ARCHIVE_RECEIPT.json`; its tar members were
verified against all 163 manifest hashes. Old frozen reports are not replaced.
