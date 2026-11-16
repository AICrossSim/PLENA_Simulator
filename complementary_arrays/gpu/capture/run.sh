#!/usr/bin/env bash
# Routing captures of Qwen3.5-35B-A3B and DeepSeek-V2-Lite: the request pool, vLLM captures of every request (sampled with
# reference and INT4 weights, greedy on the greedy subset), the engine check against the transformers path, and the
# large pool for batch 256 and 512.
# usage: run.sh <output root> <Qwen3.5-35B-A3B-FP8 dir> <DeepSeek-V2-Lite-Chat dir> <BFCL jsonl> <SWE-bench jsonl> <GPQA csv>
#   EXCLUDE: manifests of earlier captures whose requests must not enter the pool (default: the transformers captures
#   made below). VLLM_PYTHON: interpreter of the vLLM environment (gpu/requirements-vllm.txt), HF_PYTHON: that of
#   gpu/requirements.txt (request files, pools, transformers capture). The pools read WildChat-1M, SuperGPQA,
#   SWE-bench_bm25_27K, MATH-500, AIME 2024/2025, MATH and LiveCodeBench from the Hugging Face hub.
set -euo pipefail
[ $# -eq 6 ] || { echo "usage: $0 <output root> <qwen35> <deepseek> <bfcl> <swe> <gpqa>" >&2; exit 2; }
out=$1 bfcl=$4 swe=$5 gpqa=$6
declare -A model=([qwen35]=$2 [deepseek]=$3)
VLLM_PYTHON=${VLLM_PYTHON:-python} HF_PYTHON=${HF_PYTHON:-python}
pkg=complementary_arrays.gpu.capture
sources=$out/sources

$HF_PYTHON -m $pkg.prepare_sources --bfcl "$bfcl" --swe "$swe" --gpqa "$gpqa" --out "$sources"

# transformers path with vLLM's fused experts: reference of the engine check (held-out batch-8 groups)
$HF_PYTHON -m $pkg.capture_routes --model-kind qwen35 --model "${model[qwen35]}" --sources "$sources" --out "$out/hf/qwen35"
$HF_PYTHON -m $pkg.capture_routes --model-kind deepseek --model "${model[deepseek]}" --sources "$sources" \
  --out "$out/hf/deepseek" --system-messages keep
for kind in qwen35 deepseek; do $HF_PYTHON -m $pkg.convert_traces "$out/hf/$kind" --out "$out/hf/traces"; done

tokenizers=(--tokenizer "qwen35=${model[qwen35]}" --tokenizer "deepseek=${model[deepseek]}")
read -r -a exclude <<< "${EXCLUDE:-$out/hf/qwen35/manifest.json $out/hf/deepseek/manifest.json}"
$HF_PYTHON -m $pkg.pool "$out/pool" --sources "$sources" "${tokenizers[@]}" --exclude "${exclude[@]}"

for kind in qwen35 deepseek; do
  for weights in ref int4; do
    $VLLM_PYTHON -m $pkg.capture "$kind" "$weights" "$out/pool/pool_$kind.pkl" "${model[$kind]}" "$out/capture"
    $VLLM_PYTHON -m $pkg.capture "$kind" "$weights" "$out/pool/pool_$kind.pkl" "${model[$kind]}" "$out/capture" --greedy
  done
  $VLLM_PYTHON -m $pkg.verify "$kind" "${model[$kind]}" "$sources" "$out/hf/traces" "$out/verify_$kind.json"
done
$VLLM_PYTHON -m $pkg.summary "$out/capture"

# batch 256 and 512: 27,000 further requests, 16 random decode steps stored per request
$HF_PYTHON -m $pkg.pool_large "$out/pool_large" --sources "$sources" "${tokenizers[@]}" --exclude "${exclude[@]}" \
  --previous "$out/pool/pool_manifest.json"
for kind in qwen35 deepseek; do
  for weights in ref int4; do
    $VLLM_PYTHON -m $pkg.capture "$kind" "$weights" "$out/pool_large/pool_$kind.pkl" "${model[$kind]}" \
      "$out/capture_large" --keep-steps 16
  done
done
$VLLM_PYTHON -m $pkg.summary "$out/capture_large"
