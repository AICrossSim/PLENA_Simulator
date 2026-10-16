#!/usr/bin/env bash
# Run from the portable code root containing research/moe_dispatch/gpu_hotcold.
# CUDA_VISIBLE_DEVICES must identify an idle, authorized device.
set -euo pipefail
benchmark_python=${BENCHMARK_PYTHON:-python3}
campaign_root=${1:?Usage: run_campaign.sh OUTPUT_ROOT [ROUTES_JSON]}
routes_path=${2:-"$campaign_root/input/heldout_routes.json"}
mkdir -p "$campaign_root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$benchmark_python" -m research.moe_dispatch.gpu_hotcold.traces --validate "$routes_path"
for run_index in 1 2; do
  order_seed=$((20261012 + run_index))
  "$benchmark_python" -m research.moe_dispatch.gpu_hotcold.benchmark \
    --routes "$routes_path" --output "$campaign_root/repeat$run_index" \
    --warmup 20 --samples 100 --correctness all --order-seed "$order_seed" \
    --modes eager,graph --paths vllm_grouped,vllm_split,triton_grouped,triton_split \
    > "$campaign_root/repeat$run_index.log" 2>&1
done
