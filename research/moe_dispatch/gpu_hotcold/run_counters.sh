#!/usr/bin/env bash
# Run on a different idle GPU from the main timing campaign.
set -euo pipefail
counter_root=${1:?Output root required}
counter_python=${BENCHMARK_PYTHON:-python3}
counter_ncu=${NCU:-ncu}
mkdir -p "$counter_root/ncu" "$counter_root/kineto"
counter_metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
for counter_batch in 2 4 8 16 64 96 128; do
  for counter_path in vllm_grouped vllm_split; do
    counter_stem="$counter_root/ncu/B${counter_batch}_${counter_path}"
    # The initial B2 grouped pilot is already a complete counter capture.
    if [ ! -f "$counter_stem.ncu-rep" ]; then
      "$counter_ncu" --target-processes all --nvtx --nvtx-include 'measured_moe/' \
        --clock-control none --metrics "$counter_metrics" --csv --page raw \
        --log-file "$counter_stem.csv" --export "$counter_stem" \
        "$counter_python" -m research.moe_dispatch.gpu_hotcold.profile_counter \
        --routes "$counter_root/input/heldout_routes.json" \
        --batch "$counter_batch" --path "$counter_path" --output "$counter_stem.json" \
        > "$counter_stem.log" 2>&1
    fi
  done
  "$counter_python" -m research.moe_dispatch.gpu_hotcold.profile_counter \
    --routes "$counter_root/input/heldout_routes.json" --batch "$counter_batch" \
    --path vllm_grouped --torch-profiler \
    --output "$counter_root/kineto/B${counter_batch}_vllm_grouped.json" \
    > "$counter_root/kineto/B${counter_batch}.log" 2>&1
done
