#!/usr/bin/env bash
# Accuracy of Qwen3.5-35B-A3B and DeepSeek-V2-Lite-Chat with quantized expert weights, and the summary tables.
# usage: run.sh <out> <task data dir> <GPQA Diamond csv> <LiveCodeBench repo at 28fef95> <Qwen3.5 FP8 dir> <DeepSeek dir>
#   HF_PYTHON: environment of gpu/requirements.txt (emulated formats, task files); VLLM_PYTHON: gpu/requirements-vllm.txt;
#   EVAL_PYTHON: bfcl-eval 2026.3.23, math-verify 0.9.0, openai; PORT (default 18100). LiveCodeBench is scored with
#   rootless podman. SWE-bench (Qwen3.5) needs podman with its Docker API socket and mini-swe-agent 2.4.6 + swebench 5.0.2
#   (SWE_PYTHON); it runs when SWE=1, with the servers on this machine.
set -uo pipefail
[ $# -eq 6 ] || { echo "usage: $0 <out> <data> <gpqa csv> <lcb repo> <qwen35> <deepseek>" >&2; exit 2; }
out=$1 data=$2 gpqa=$3 lcb=$4
declare -A model=([qwen35]=$5 [deepseek]=$6)
HF_PYTHON=${HF_PYTHON:-python} VLLM_PYTHON=${VLLM_PYTHON:-python} EVAL_PYTHON=${EVAL_PYTHON:-python} PORT=${PORT:-18100}
pkg=complementary_arrays.gpu.accuracy
here=$(dirname "$(readlink -f "$0")")
tasks=$out/tasks

# 1. emulated formats (transformers): standard tasks with a 1,024-token GSM8K cap, and GSM8K with 4,096 tokens
for kind in qwen35 deepseek; do
  $HF_PYTHON -m $pkg.evaluate --model-kind $kind --model "${model[$kind]}" --data "$data" --out "$out/emulated"
  $HF_PYTHON -m $pkg.evaluate --model-kind $kind --model "${model[$kind]}" --data "$data" --out "$out/gsm4096" \
    --formats bf16,w4g32,nvfp4 --tasks gsm8k --max-new 4096
done

# 2. GPU kernels through a vLLM server: one session per task group
$HF_PYTHON -m $pkg.prepare "$tasks" --gpqa "$gpqa"
session() {  # <kind> <mode> <run dir> <server max length or ""> <kv cache or ""> <task ...>
  local kind=$1 mode=$2 dir=$3 maxlen=$4 kv=$5
  shift 5
  mkdir -p "$dir"
  $VLLM_PYTHON -m $pkg.serve "$kind" "$mode" "$PORT" --model "${model[$kind]}" --work "$out/work" \
    ${maxlen:+--max-model-len $maxlen} ${kv:+--kv-cache $kv} > "$dir/server_$(date +%s).log" 2>&1 &
  local pid=$!
  for _ in $(seq 240); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 5; done
  for t in "$@"; do
    case $t in
      bfcl) $EVAL_PYTHON -m $pkg.bfcl "$kind" "$mode" "$PORT" "$dir" "$out/work/modeldirs/vllm_${kind}_$mode" > "$dir/bfcl.log" 2>&1 ;;
      *_long) $EVAL_PYTHON -m $pkg.generate "$kind" "$mode" "${t%_long}" "$PORT" "$dir" --data "$tasks" --max-tokens 81920 \
        --tag long > "$dir/$t.log" 2>&1 ;;
      *) $EVAL_PYTHON -m $pkg.generate "$kind" "$mode" "$t" "$PORT" "$dir" --data "$tasks" > "$dir/$t.log" 2>&1 ;;
    esac
  done
  kill "$pid"
  wait "$pid"
}
for mode in bf16 fp8_w8a16 nvfp4_w4a16; do
  d=$out/runs/qwen35_$mode
  session qwen35 $mode "$d" "" "" gpqa math500
  session qwen35 $mode "$d" "" "" bfcl
  session qwen35 $mode "$d" 90112 "" aime_long
  session qwen35 $mode "$d" 90112 "" lcb_long
  bash "$here/lcb_score.sh" "$lcb" "$tasks" "$d" long
  d=$out/runs/deepseek_$mode
  session deepseek $mode "$d" "" "" gpqa math500 aime lcb
  session deepseek $mode "$d" "" "" bfcl
  bash "$here/lcb_score.sh" "$lcb" "$tasks" "$d"
done

# 3. emulated KV-cache formats with NVFP4 weights
for kv in mxint8 mxint4; do
  d=$out/kv_$kv/qwen35_nvfp4_w4a16
  session qwen35 nvfp4_w4a16 "$d" 90112 $kv gpqa math500 aime_long
  session qwen35 nvfp4_w4a16 "$d" 90112 $kv lcb_long
  bash "$here/lcb_score.sh" "$lcb" "$tasks" "$d" long
  d=$out/kv_$kv/deepseek_nvfp4_w4a16
  session deepseek nvfp4_w4a16 "$d" "" $kv gpqa math500 aime lcb
  bash "$here/lcb_score.sh" "$lcb" "$tasks" "$d"
done
# DeepSeek-V2-Lite's 4-bit KV-cache loss by part: perplexity with the latent or the rotary key alone in MXINT4
$HF_PYTHON -m $pkg.perplexity windows --model "${model[deepseek]}" --data "$data" --out "$out/ppl/windows_deepseek.json"
for kv in "" mxint4 mxint4_c mxint4_pe; do
  $VLLM_PYTHON -m $pkg.perplexity score deepseek fp8_w8a16 "$out/ppl/windows_deepseek.json" \
    "$out/ppl/deepseek_fp8_w8a16_${kv:-bf16}.json" --model "${model[deepseek]}" --work "$out/work" \
    ${kv:+--kv-cache $kv} --eager --stats
done

# 4. SWE-bench Verified (Qwen3.5): one server per format with reasoning and tool-call parsing
if [ "${SWE:-0}" = 1 ]; then
  endpoints=() pids=()
  for i in 0 1 2; do
    mode=$(echo bf16 fp8_w8a16 nvfp4_w4a16 | cut -d' ' -f$((i + 1))) port=$((28001 + i))
    CUDA_VISIBLE_DEVICES=$i $VLLM_PYTHON -m $pkg.serve qwen35 $mode $port --model "${model[qwen35]}" --work "$out/work" \
      --max-model-len 131072 -- --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      > "$out/swe_server_$mode.log" 2>&1 &
    pids+=($!) endpoints+=(--endpoint "$mode=http://127.0.0.1:$port/v1")
  done
  ${SWE_PYTHON:-python} -m $pkg.swebench "$out/swebench" "${endpoints[@]}"
  kill "${pids[@]}"
fi

$HF_PYTHON -m $pkg.summarize "$out/accuracy" --emulated "$out/emulated" --gsm4096 "$out/gsm4096" --runs "$out/runs" \
  --tasks "$tasks" --kv mxint8="$out/kv_mxint8" --kv mxint4="$out/kv_mxint4" --swebench "$out/swebench"
