# Merge compatibility repairs — originals retained

Only active compatibility code was repaired; historical final result tables remain unchanged.

- `nemotron3_workload.py`: restore distinct OCP MXFP8 block-32 storage alongside existing configurable MX8. This is an existing general regression profile, not a round-two precision experiment; all new MoE experiments remain BF16.
- `gpu_sources.json`: pin restored exact historical measured routing fixture from e93b832f6586. SHA matches Git bytes; no synthetic values inserted.
- `kimi_k3_workload.py`: restore prior prefill chunk prepare/recurrence work while retaining later precision-contract additions.
- `hybrid_lcompute_campaign.py`: validate decode and prefill with their corresponding existing stage names. Runtime recurrence replacement remains decode-only.

The `.before` copies retain the pre-repair merged versions. Attempts and final actual unit receipts live under `results/E0/tests/`.
