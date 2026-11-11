#!/usr/bin/env bash
# GPU baseline: MoE-layer kernels on the held-out traces, the decode-step ablation in vLLM, and the summary.
# usage: run.sh <output dir> <trace dir> <DeepSeek-V2-Lite-Chat dir>
# FlashInfer compiles kernels on first use and needs nvcc and ninja on PATH.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <output dir> <trace dir> <deepseek>" >&2; exit 2; }
out=$1 traces=$2 deepseek=$3
pkg=complementary_arrays.gpu.bench
export VLLM_TUNED_CONFIG_FOLDER=$out/kernels/tuned_configs

# One process per kernel, so that a failing kernel does not stop the others.
for model in qwen deepseek; do
  for kernel in bf16:triton bf16:flashinfer bf16:trtllm w8g128:triton w8g128:marlin w4g32:marlin; do
    python -m $pkg.moe_layer --traces "$traces" --out "$out/kernels" --models "$model" --kernels "$kernel" ||
      echo "failed: $model $kernel" >&2
  done
done

for quantization in none fp8; do
  for variant in full no_moe; do
    python -m $pkg.e2e_moe_share --model "$deepseek" --out "$out/e2e" --quantization "$quantization" --variant "$variant"
  done
done

python -m $pkg.report --kernels "$out/kernels" --e2e "$out/e2e" --out "$out/report.md"
