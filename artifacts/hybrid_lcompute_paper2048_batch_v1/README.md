# Hybrid L-Compute paper-2048 batch campaign

This artifact sweeps full-model decode at `B=1/2/4/8/16`, context 2048 and 32 generated tokens. It uses the official 52-layer Nemotron 3 and 93-layer Kimi K3 schedules on one shared Matrix/Vector/HBM/banked-output timeline.

The batch model fetches each touched weight tensor once per batched stage, scales Matrix work with B, and keeps recurrent state and KV private to every request. The exact Compiler single-request recurrent body is repeated B times; no unimplemented cross-request state packet is assumed.

Two MoE routing bounds are reported because Kimi batch routing has not been measured: `full_overlap` minimizes routed-weight traffic and `maximum_distinct` maximizes it.

Reproduce from the Simulator root:

```bash
.venv/bin/python -m analytic_models.performance.hybrid_lcompute_campaign \
  --compiler-root PLENA_Compiler \
  --hardware-profile paper2048 \
  --batch-sweep \
  --json-out artifacts/hybrid_lcompute_paper2048_batch_v1/campaign.json \
  --csv-dir artifacts/hybrid_lcompute_paper2048_batch_v1/tables
```

Canonical report hash embedded in `campaign.json`:

```text
28d885896f7e5f553e11a298c3d2c79e862c3cb65bd297014d96fcb99f575342
```

File SHA256 values:

```text
61c09a11fb875e39cb807beb79d866fcdd577c51afd722b9f3fcdfafb2a901f7  campaign.json
f4c365cd23cff6278448b8885ad08288972e742eb0b5d2f7add26818747595ef  tables/ablation.csv
dc9900ddde00eb0ec479628af8dbb8cc6f277e7f5f8543c758fb90969d1c6de0  tables/batch_dse.csv
6db51b6a639ee21ef692ad235c66b39ba0e73b3a4c208e65eb9af39e89331243  tables/dse.csv
ef3c2c8ffcfdf9acb5460d63302a62a6ef28ed193e710704f6d5aca758a6dc6b  tables/precision.csv
5ee6192517065493b0d821305a9548efc6af161860236769c068713f20d20f17  tables/schedule_validation.csv
```

This is a Compiler/Simulator pre-RTL estimate with symbolic weights, not RTL timing, PPA, energy, or full-checkpoint numerical execution.
