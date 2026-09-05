# Hybrid L-Compute paper-2048 campaign

Re-generated on 2026-09-05 from the corrected Compiler/Simulator sources.
This remains a **historical architecture control**, not the current Matrix-SRAM
`L_TILE` result. Current claims use
[`MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md`](../../docs/MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md).

Historical Vector/output-SRAM L_CFG experiment at MLEN=VLEN=2048 and BLEN=32; includes the exact 64/128/256/512/1024/2048 lane sweep.

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
  --hardware-profile paper2048 --long --lane-sweep \
  --json-out artifacts/hybrid_lcompute_paper2048_v1/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_paper2048_v1/tables
```

Canonical report hash embedded in `campaign.json`:

```text
58854c00d57fbe6c2dba703d9c2959871f930a1affd28e588573a3b2d28f9252
```

File SHA256 values:

```text
8fe87427578f0f4c7837713703a9f55e240ecef7ba6a4ef9a5d41eccace85c2b  campaign.json
9db8f07bb13ec5e195441efc446b68f70f5151baaaf09bdfec3e5cb36015e931  tables/ablation.csv
ddb0fc1dfa410a40720d18b478ab6d7b2c932ccfe9b5d337108e70b97a7c6b0c  tables/dse.csv
f80d0b02dd578fcae17a70bb387fa5dbc5909aa1c2717bbb12323770b7e225d5  tables/lane_dse.csv
b7f4de588709110e290cb39ef3e5140dd6c1ee2dabbe2fc2a7578642f304504c  tables/precision.csv
52e208612583704a383c65681e3912fa7a70f7c9931e164cf5d8ed367a23edde  tables/schedule_validation.csv
```
