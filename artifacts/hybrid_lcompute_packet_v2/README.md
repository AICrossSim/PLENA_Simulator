# Hybrid L-Compute packet campaign

Generated from the checked-in Compiler submodule and Simulator with:

```bash
PYTHONPATH="$PWD:$PWD/PLENA_Compiler" \
  .venv/bin/python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --json-out artifacts/hybrid_lcompute_packet_v2/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_packet_v2/tables \
  --long
```

Scope: official Nemotron 52-layer and Kimi 93-layer shapes, symbolic weights,
S16/S128 prefill, 4/32-token decode, A-J ablation, DSE, precision traffic, and
schedule validation. This is a Compiler/Simulator result, not RTL PPA or a
full-checkpoint numerical execution.

The canonical hash embedded in `campaign.json` is:

```text
ec266bef46611daaae1982b8335d6f0c3ad78550b77436c6efd035e592e68d4d
```

File SHA256 values:

```text
860a4bbc8ed1f45c7cb6602672d3a23745a1f6f95d43c7108a34fd1fa57b4e03  campaign.json
d043a7d26dd0096fff9769fe65142032a8f2152ede4c9965216e955717cd9e96  tables/ablation.csv
2072d2fd4831e474be5a3eb5134d0ce0f4986d394d47f232f929af076fdfe49e  tables/dse.csv
1c43aed1a134a58c52345c6f68a0bbeeb7a3a7e3b8c936fa0a15efa3c8d4c66f  tables/precision.csv
d45d8f0bd51a8a69e9765507f34a04684630ade7c9220ee73a945005f4939726  tables/schedule_validation.csv
```
