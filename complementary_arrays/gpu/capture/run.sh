#!/usr/bin/env bash
# Routing captures of the three models: dev/held-out traces for the simulator and the sealed request group.
# usage: run.sh <output root> <Qwen3.5 dir> <DeepSeek dir> <Qwen3-Next dir> <BFCL jsonl> <SWE-bench jsonl> <GPQA csv>
# FP8_KERNEL may name a local snapshot of kernels-community/finegrained-fp8 for Qwen3-Next.
set -euo pipefail
[ $# -eq 7 ] || { echo "usage: $0 <output root> <qwen35> <deepseek> <qwen3next> <bfcl> <swe> <gpqa>" >&2; exit 2; }
out=$1 qwen35=$2 deepseek=$3 qwen3next=$4 bfcl=$5 swe=$6 gpqa=$7
pkg=complementary_arrays.gpu.capture
sources=$out/sources

python -m $pkg.prepare_sources --bfcl "$bfcl" --swe "$swe" --gpqa "$gpqa" --out "$sources"
python -m $pkg.capture_routes --model-kind qwen35 --model "$qwen35" --sources "$sources" --out "$out/capture/qwen35"
# DeepSeek's chat template accepts system messages anywhere, so they are passed unchanged.
python -m $pkg.capture_routes --model-kind deepseek --model "$deepseek" --sources "$sources" \
  --out "$out/capture/deepseek" --system-messages keep
python -m $pkg.capture_qwen3next --model "$qwen3next" --sources "$sources" --out "$out/capture/qwen3next" \
  ${FP8_KERNEL:+--fp8-kernel "$FP8_KERNEL"}
for capture in qwen35 deepseek qwen3next; do
  python -m $pkg.convert_traces "$out/capture/$capture" --out "$out/traces"
done

# The simulator reads the sealed group directly from its capture shards.
python -m $pkg.capture_sealed --model-kind qwen35 --model "$qwen35" --sources "$sources" \
  --exclude "$out"/capture/*/manifest.json --out "$out/sealed/qwen35_bf16"
python -m $pkg.capture_sealed --model-kind deepseek --model "$deepseek" --sources "$sources" \
  --exclude "$out"/capture/*/manifest.json --out "$out/sealed/deepseek_bf16"
