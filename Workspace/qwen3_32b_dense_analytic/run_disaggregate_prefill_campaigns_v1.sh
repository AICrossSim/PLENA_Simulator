#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
WORKERS=${PLENA_DSE_WORKERS:-auto}
RUN_ROOT=${PLENA_DSE_RUNS:-"$REPO_ROOT/Workspace/dse_runs"}
CACHE_ROOT=${PLENA_DSE_CACHE:-"$REPO_ROOT/Workspace/.cache/dse"}
CAMPAIGN_ROOT="$RUN_ROOT/disaggregate_prefill_dse_config_v1"
GLOBAL_CACHE="$CACHE_ROOT/plena_cross_study_cache_v1"
DSE="$REPO_ROOT/Workspace/qwen3_32b_dense_analytic/run_optuna_dse.py"

mkdir -p "$CAMPAIGN_ROOT/logs" "$GLOBAL_CACHE"

COMMON_ARGS=(
    --target-complete-trials 16384
    --max-total-attempts 49152
    --tpe-startup-trials 2048
    --tpe-ei-candidates 128
    --workers "$WORKERS"
    --artifact-retention compact
    --allowed-weight-element-bits 4
    --softmax-row-lanes 1,2,4,8,16
    --chip-counts 1,2,4,8,16
    --multi-chip-model tile-aware-dp-tp-ep-v4
    --input-seq-len 90000
    --output-seq-len 8000
    --latency-batch-size 8
    --min-accuracy 0.9
    --global-cache-dir "$GLOBAL_CACHE"
)

run_campaign() {
    local name=$1
    local model_config=$2
    local prefill_budget=$3
    local decode_chips=$4
    shift 4

    echo "[$(date --iso-8601=seconds)] starting $name"
    "$PYTHON_BIN" "$DSE" \
        "${COMMON_ARGS[@]}" \
        --model-config "$model_config" \
        --reference-a100-count "$prefill_budget" \
        --decode-chip-count "$decode_chips" \
        --run-dir "$CAMPAIGN_ROOT/$name" \
        "$@" \
        2>&1 | tee -a "$CAMPAIGN_ROOT/logs/$name.log"
    echo "[$(date --iso-8601=seconds)] completed $name"
}

run_campaign \
    qwen3_32b_90k8_p2_r16_w4_v1 \
    "$REPO_ROOT/Workspace/qwen3_32b_dense_analytic/qwen3-32b.json" \
    2 6

run_campaign \
    qwen3_32b_90k8_p4_r16_w4_v1 \
    "$REPO_ROOT/Workspace/qwen3_32b_dense_analytic/qwen3-32b.json" \
    4 4

MOE_ARGS=(
    --moe-routing-mode fixed-balanced
    --moe-layer-scaling repeat-fixed-balanced
)

run_campaign \
    qwen3_235b_90k8_p4_r16_w4_v1 \
    "$REPO_ROOT/Workspace/qwen3_235b_a22b_analytic/qwen3-235b-a22b-instruct.json" \
    4 12 \
    "${MOE_ARGS[@]}"

run_campaign \
    qwen3_235b_90k8_p8_r16_w4_v1 \
    "$REPO_ROOT/Workspace/qwen3_235b_a22b_analytic/qwen3-235b-a22b-instruct.json" \
    8 8 \
    "${MOE_ARGS[@]}"

run_campaign \
    qwen3_235b_90k8_p12_r16_w4_v1 \
    "$REPO_ROOT/Workspace/qwen3_235b_a22b_analytic/qwen3-235b-a22b-instruct.json" \
    12 4 \
    "${MOE_ARGS[@]}"
