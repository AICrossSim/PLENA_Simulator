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
a07ef4516e05a26420ea5e63994fdec960ccd0bc3dc688a6e7c734c1f384b5cd
```

File SHA256 values:

```text
410888ff47ca331bc4da861e5c406b2ed29338b303be451b961d7eaaa41c3990  campaign.json
f965c08a1bea1f4e9a96df0e319ff2cd6e7a1a9bdea92f3c64add6974facf44c  tables/ablation.csv
3e5ec2ebe6d19e46dd3a35a2d260e73d427c5ca3079f68d4798d8d08cf70affe  tables/dse.csv
f7b7d440ea2f1c9d704d4e7d798a358ab5501e78a8d8fbbf602610b04d2e2545  tables/precision.csv
ff6e091807088ca1d2f0eda6cf34e430a2c6707ef8bdb87052ab5c64a7a6e5b4  tables/schedule_validation.csv
```
