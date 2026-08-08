# RunPod A100 Serving Benchmark

This harness measures the A100 phases needed by the disaggregated-serving
study. It uses offline vLLM, deterministic token IDs, exact output lengths,
NVML board-energy counters, and a simultaneous 20 Hz power trace.

It does **not** import PLENA KV into vLLM. `imported_kv_decode_proxy` removes
the measured A100 prefill phase and adds one measured normal decode iteration
to represent the first token produced from imported KV.

## RunPod sequence

Use an on-demand Secure Cloud pod with 8 x A100 SXM 80 GB, an 80 GB container
disk, and a 500 GB network volume mounted at `/workspace`. Store the checkout,
Hugging Face cache, and all results on `/workspace`; network-volume pods must
be terminated rather than stopped.

```bash
export HF_HOME=/workspace/huggingface
export VLLM_CACHE_ROOT=/workspace/vllm-cache
export RESULTS=/workspace/plena_runpod_a100_v1

python -m pip install -r analytic_models/serving_benchmark/requirements-runpod.txt

python -m analytic_models.serving_benchmark inventory \
  --output "$RESULTS/inventory.json"

python -m analytic_models.serving_benchmark preflight \
  --inventory "$RESULTS/inventory.json" \
  --output-root "$RESULTS/preflight" \
  --environment-lock "$RESULTS/environment.lock.json" \
  --image-digest 'sha256:<digest-from-RunPod-template>'

python -m analytic_models.serving_benchmark run \
  --environment-lock "$RESULTS/environment.lock.json" \
  --image-digest 'sha256:<same-digest>' \
  --measurement-stage screening \
  --output-root "$RESULTS/screening"

python -m analytic_models.serving_benchmark aggregate \
  --output-root "$RESULTS/screening"
```

Screening runs all 14 primary-workload topology/local-batch points once after
their warmup. The default `auto` scheduler shards each reusable engine group
across the number of replicas that can physically fit: up to eight TP1, four
TP2, two TP4, or one TP8 worker can occupy the eight-GPU node. Points assigned
to one shard reuse its loaded model. Mixed power-of-two TP sizes are packed
whenever their GPU sets do not overlap. Each worker gets a
separate CUDA visibility mask and rendezvous port, while NVML/DCGM retain the
physical GPU IDs. Capacity failures terminate only the affected point.

`auto` uses sharded-engine GPU concurrency for screening, engine-group
concurrency for the inexpensive short sweep, and isolated execution for
confirmation and holdout measurements. This keeps candidate discovery fast
without using co-tenant measurements as the final formal values.
`--execution-mode gpu-parallel` forces point-level concurrency for any stage;
`--execution-mode sequential` disables it. The available devices and process
cap can be changed
with:

```text
--physical-gpu-pool 0,1,2,3,4,5,6,7
--max-concurrent-engines 8
```

Scheduler assignments and start/end timestamps are retained in
`active_schedule.json` and `run_state.json`. Physical placement is diagnostic
metadata and does not alter a point's resume fingerprint.

Select the best topology for each declared system and confirm
only those points (plus a runner-up when it is within 5%):

```bash
python -m analytic_models.serving_benchmark run \
  --environment-lock "$RESULTS/environment.lock.json" \
  --image-digest 'sha256:<same-digest>' \
  --measurement-stage confirmation \
  --point-ids '<comma-separated-selected-primary-points>' \
  --output-root "$RESULTS/confirmation"
```

`confirmation`, `short-sweep`, and selected `holdout` points use three full
measurements. `holdout` requires explicit point IDs so the 114k/5k workload is
never accidentally run over the complete topology matrix.

After the corresponding independent TP4 point is complete, run the two-replica
check on disjoint halves of the node:

```bash
python -m analytic_models.serving_benchmark replica-check \
  --environment-lock "$RESULTS/environment.lock.json" \
  --image-digest 'sha256:<same-digest>' \
  --formal-output-root "$RESULTS/confirmation" \
  --output-root "$RESULTS/replica_checks/32b-primary-tp4-b4" \
  --point-id qwen3-32b.primary-90000x8000.tp4.b4
```

The output reports latency and energy correction factors and flags deviations
above 5%.

Each stage must use its own output directory. `run` is resumable. A completed point is skipped only when its manifest,
checkpoint revision, backend, and environment hashes all match. Use
`--models`, `--workloads`, or `--point-ids` to execute a subset. Do not alter
the image or Python environment after preflight.

## Fidelity and validation

- Prefix caching, speculative decoding, CPU offload, and swap are disabled.
- Sampling is greedy with EOS ignored and an exact output-token count.
- The first vLLM output marks the prefill/first-token boundary. The next
  engine iteration is the measured normal decode-step proxy.
- Any multi-token engine step or token-count mismatch fails the point.
- Aggregation warns above 5% repeat CV or 3% disagreement between NVML's
  total-energy counter and integrated 20 Hz samples.
- RunPod-internal NVLink is only evidence for A100 TP/DP. It is not the
  modeled PLENA-to-A100 handoff link.
