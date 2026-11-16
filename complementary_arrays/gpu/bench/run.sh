#!/usr/bin/env bash
# B200 measurements of the GPU comparison on one GPU: read bandwidth, MoE-layer wall time (full, shared expert zeroed and
# MoE zeroed back to back per format and engine), kernel times, board energy (steady state and closed batches), then the
# summary.
# usage: run.sh <out> <Qwen3.5-35B-A3B-FP8 dir> <DeepSeek-V2-Lite-Chat dir> <pool dir> <capture dir>
#   pool dir: pool_<model>.pkl (capture/pool.py); capture dir: the reference sampled capture (capture/capture.py).
#   VLLM_PYTHON and SGLANG_PYTHON: interpreters of the two environments (gpu/requirements.txt). SGLang's JIT kernels
#   need CUDA_HOME pointing at a CUDA 13 toolkit (e.g. the nvidia/cu13 package of its environment) and nvcc on PATH.
set -uo pipefail
[ $# -eq 5 ] || { echo "usage: $0 <out> <qwen35> <deepseek> <pool dir> <capture dir>" >&2; exit 2; }
out=$1 pool=$4 capture=$5
declare -A model=([qwen35]=$2 [deepseek]=$3)
VLLM_PYTHON=${VLLM_PYTHON:-python} SGLANG_PYTHON=${SGLANG_PYTHON:-python}
export HF_HUB_OFFLINE=1 SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
pkg=complementary_arrays.gpu.bench
VLLM_FORMATS="fp8_w8a8 fp8_w8a16 nvfp4_w4a16 nvfp4_w4a4 nvfp4_w4a4_sh"
SGLANG_FORMATS="fp8_w8a8 nvfp4_w4a4 nvfp4_w4a16"

py() { if [ "$1" = vllm ]; then echo "$VLLM_PYTHON"; else echo "$SGLANG_PYTHON"; fi; }
run() {  # one process per measurement, so that a failing run does not stop the others
  local tool=$1 stack=$2 kind=$3 mode=$4 dir=$5
  shift 5
  $(py "$stack") -m $pkg.$tool "$stack" "$kind" "$mode" --model "${model[$kind]}" --pool "$pool/pool_$kind.pkl" \
    --capture "$capture" --out "$dir" --work "$out/work" "$@" || echo "failed: $tool $stack $kind $mode $*" >&2
}

$VLLM_PYTHON -m $pkg.bandwidth "$out/bandwidth.json"

for kind in qwen35 deepseek; do
  for stack in vllm sglang; do
    formats=$VLLM_FORMATS
    [ "$stack" = sglang ] && formats=$SGLANG_FORMATS
    for mode in $formats; do
      d=$out/stream/${stack}_$mode
      run stream "$stack" "$kind" bf16 "$d" --moe no_moe
      run stream "$stack" "$kind" "$mode" "$d"
      # vLLM's NVFP4 W4A4 keeps the shared expert in BF16; its split comes from nvfp4_w4a4_sh
      [ "$stack/$mode" = vllm/nvfp4_w4a4 ] || run stream "$stack" "$kind" "$mode" "$d" --no-shared
      run kernel_time "$stack" "$kind" "$mode" "$out/kernels"
    done
  done
done

# energy: the faster engine of each format, with the MoE-zeroed run of the same engine in the same directory
for spec in qwen35:sglang:fp8_w8a8,nvfp4_w4a4,nvfp4_w4a16 qwen35:vllm:fp8_w8a16,nvfp4_w4a4_sh,nvfp4_w4a16 \
  deepseek:vllm:fp8_w8a8,fp8_w8a16,nvfp4_w4a4,nvfp4_w4a4_sh,nvfp4_w4a16 deepseek:sglang:nvfp4_w4a16; do
  IFS=: read -r kind stack modes <<< "$spec"
  for method in steady closed; do
    d=$out/energy/$method/${kind}_$stack
    run energy "$stack" "$kind" bf16 "$d" --moe no_moe --method $method
    for mode in ${modes//,/ }; do run energy "$stack" "$kind" "$mode" "$d" --method $method; done
  done
done

python -m $pkg.report "$out/b200" --stream "$out/stream" --kernels "$out/kernels" --energy "$out/energy"
