# Hybrid L-Compute paper-2048 campaign

This artifact retargets the hybrid Compiler/Simulator campaign to the final
system point reported by the PLENA paper: `BLEN=32`, `MLEN=VLEN=2048`, 1 GHz.
The added bank geometry is an architecture candidate, not a fact claimed by
the paper: 32 banks x 64 FP16 elements, 2 read ports and 1 write port per bank.

Nemotron uses exact 64-element recurrent rows. Kimi uses its natural
128-element recurrent rows, so the ordinary KDA baseline is not unfairly
split into two 64-element operations. `L_CFG` defines model-independent
affine views; each existing Vector instruction explicitly selects its input
views with a three-slot Vector consumer mask; slot 3 is reserved for Matrix
writeback. Those views may combine bank-word atoms from
several rows into one 2048-element packet. The executable affine mapping
stores those 32 words in one physical bank row; the row-major control uses 32
padded short-row locations.

The campaign covers the official 52-layer Nemotron and 93-layer Kimi
schedules, S16/S128 prefill, 4/32-token decode, A-J ablation, exact
64/128/256/512/1024/2048 lane recompilation, hardware DSE, precision traffic,
and schedule validation. Weights remain symbolic. This is not RTL PPA or a
full-checkpoint numerical execution.

Run from the Simulator root:

```bash
PYTHONPATH="$PWD:$PWD/PLENA_Compiler" \
  .venv/bin/python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --hardware-profile paper2048 \
  --long --lane-sweep \
  --json-out artifacts/hybrid_lcompute_paper2048_v1/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_paper2048_v1/tables
```

The canonical hash embedded in `campaign.json` is:

```text
0b9c659e91852c96e17e75c1826dadcb5c9798604a7cee1abef30c6a48fe5e9d
```

File SHA256 values:

```text
4727f9f336129b6b467673e3cab4c8a32519a16a9fa306b5ee3d4f2226ce81e8  campaign.json
0260045f8bde01d58f4649fb03342e573a0b4b4798eb4aa0f5c6f4a216a6624b  tables/ablation.csv
6db51b6a639ee21ef692ad235c66b39ba0e73b3a4c208e65eb9af39e89331243  tables/dse.csv
990f022fe76179c056bf034dab2e000188488f89270c0621022bd52a758e9bc8  tables/lane_dse.csv
ef3c2c8ffcfdf9acb5460d63302a62a6ef28ed193e710704f6d5aca758a6dc6b  tables/precision.csv
ff6e091807088ca1d2f0eda6cf34e430a2c6707ef8bdb87052ab5c64a7a6e5b4  tables/schedule_validation.csv
```
