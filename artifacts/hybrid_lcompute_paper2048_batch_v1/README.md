# Hybrid L-Compute paper-2048 batch campaign

Re-generated on 2026-09-05 from the corrected Compiler/Simulator sources.
This remains a **historical architecture control**, not the current Matrix-SRAM
`L_TILE` result. Current claims use
[`MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md`](../../docs/MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md).

Historical Vector/output-SRAM L_CFG experiment at the paper-2048 point; includes B1/B2/B4/B8/B16 routing bounds and pinned Nemotron measured-routing replay.

The shared workload/cycle fixes are included: generic convolution/state MACs
cost VLEN-wide MUL then ADD passes (one cycle per pass by default), independent
of Matrix BLEN; Nemotron includes the final RMSNorm after its 52 blocks.
This is an analytic fully packed arithmetic proxy, not a measured RTL schedule.
Nemotron uses mixed NVFP4/BF16 checkpoint weights, Kimi mixed MXFP4/BF16;
activation/state precision remains that of this historical experiment and must
not be equated with the current uniformly BF16 Matrix-SRAM path.

The batch model repeats the single-request recurrent body with private state
and KV per request; it does not claim a batched Rust lowering. Routing bounds
are separate from the measured B1 replay, and Kimi has no measured GPU routing.
The `--measured-routing` switch is required to regenerate the replay tables.

Weights are symbolic and the official 52/93-layer schedules are analytical.
These results do not establish full-checkpoint Rust execution, RTL/PPA,
GPU-relative speedup or energy. Old narrative figures in historical reports
predate this re-generation; use these machine-readable files for corrected
values of this historical control.

Reproduce from the Simulator root (the Compiler submodule must match the
current source pin):

```bash
nix develop --no-write-lock-file --command python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --hardware-profile paper2048 --batch-sweep --measured-routing \
  --json-out artifacts/hybrid_lcompute_paper2048_batch_v1/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_paper2048_batch_v1/tables
```

Canonical report hash embedded in `campaign.json`:

```text
20749fc4f40f7bee4a5d9c1ff0ff3aafd96d5f969d453b2973dc9d2df561bc5a
```

File SHA256 values:

```text
76ac1b2ff67eea7cd58119a5c7f6ab3f4a8fff78ad72cef1d98236140e5d8271  campaign.json
988390c9e96795f88913657aae564feea1ccc095247e05f15280abb3fa3d5abf  tables/ablation.csv
a9af53b4b363f1f656d0e8ed7769a4a89356d52d7411aec4a0ef5776d2740bca  tables/batch_dse.csv
ddb0fc1dfa410a40720d18b478ab6d7b2c932ccfe9b5d337108e70b97a7c6b0c  tables/dse.csv
cac127bf63ec200ca432ca26e71e0642d20e5787c2f52cb894b620707c372819  tables/measured_routing_dse.csv
b7f4de588709110e290cb39ef3e5140dd6c1ee2dabbe2fc2a7578642f304504c  tables/precision.csv
6f12e60c541b0e8ee63ff232fee79f0857c440e6aa7f93b76d5b43fa120f2e2b  tables/schedule_validation.csv
```
