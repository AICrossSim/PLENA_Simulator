# Hybrid L-Compute packet campaign

Re-generated on 2026-09-05 from the corrected Compiler/Simulator sources.
This remains a **historical architecture control**, not the current Matrix-SRAM
`L_TILE` result. Current claims use
[`MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md`](../../docs/MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md).

64-lane historical Vector/output-SRAM L_CFG experiment; S16/S128 prefill and 4/32-token decode.

The shared workload/cycle fixes are included: generic convolution/state MACs
cost VLEN-wide MUL then ADD passes (one cycle per pass by default), independent
of Matrix BLEN; Nemotron includes the final RMSNorm after its 52 blocks.
This is an analytic fully packed arithmetic proxy, not a measured RTL schedule.
Nemotron uses mixed NVFP4/BF16 checkpoint weights, Kimi mixed MXFP4/BF16;
activation/state precision remains that of this historical experiment and must
not be equated with the current uniformly BF16 Matrix-SRAM path.

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
  --long \
  --json-out artifacts/hybrid_lcompute_packet_v2/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_packet_v2/tables
```

Canonical report hash embedded in `campaign.json`:

```text
7ea44bac902b7b7a68451c54c802dea4bb19c97d89f781407f6741364fb3b35b
```

File SHA256 values:

```text
661006a94ee0184dd13c38be9371f9d9797349add8f0ff2af1d08722c73c4cce  campaign.json
7d5b94ff7bbe2fd554bb270fa260bc4e61209ac86af85c997135eaaa18c85207  tables/ablation.csv
4a0a4b5af30fbd6f92b56717daa262ce9cf57dd6825ca91431a442a1401e77e1  tables/dse.csv
fcd7fca8f8a09b896a219e5f74455334f03cf593f7b9228b6c6d112efb39f743  tables/precision.csv
52e208612583704a383c65681e3912fa7a70f7c9931e164cf5d8ed367a23edde  tables/schedule_validation.csv
```
