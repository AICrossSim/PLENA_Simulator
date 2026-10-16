#!/usr/bin/env bash
set -euo pipefail
precision="$1"
campaign_root="$2"
profile_python="${PROFILE_PYTHON:-/home/mcl123/envs/plena-nemotron-vllm/bin/python}"
cd "$campaign_root/code"
for repeat in 1 2; do
    "$profile_python" -m research.moe_dispatch.decode_precision_profile.profile \
        --capture "$campaign_root/capture_bf16" \
        --model /home/mcl123/models/deepseek-v2-lite-chat \
        --out "$campaign_root/profiles/$precision/repeat$repeat" \
        --precision "$precision" --repeat "$repeat" --warmup 10 --samples 50
done
