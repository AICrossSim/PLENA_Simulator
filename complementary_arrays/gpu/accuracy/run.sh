#!/usr/bin/env bash
# Accuracy of the three models under the RTN expert-weight formats, and the summary tables.
# usage: run.sh <output dir> <task data dir> <Qwen3.5 dir> <DeepSeek dir> <Qwen3-Next dir>
# FP8_KERNEL may name a local snapshot of kernels-community/finegrained-fp8 for Qwen3-Next.
set -euo pipefail
[ $# -eq 5 ] || { echo "usage: $0 <output dir> <task data dir> <qwen35> <deepseek> <qwen3next>" >&2; exit 2; }
out=$1 data=$2 qwen35=$3 deepseek=$4 qwen3next=$5
evaluate=(python -m complementary_arrays.gpu.accuracy.evaluate --data "$data" --out "$out")

"${evaluate[@]}" --model-kind qwen35 --model "$qwen35"
"${evaluate[@]}" --model-kind deepseek --model "$deepseek"
"${evaluate[@]}" --model-kind qwen3next --model "$qwen3next" --tasks lambada,mmlu,arc_c,gsm8k \
  ${FP8_KERNEL:+--fp8-kernel "$FP8_KERNEL"}
python -m complementary_arrays.gpu.accuracy.summarize "$out"
