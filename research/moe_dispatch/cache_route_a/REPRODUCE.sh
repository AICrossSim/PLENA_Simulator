#!/usr/bin/env bash
set -euo pipefail
# Run from the archive root; provide a fresh output directory.
experiment_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PLENA_PYTHON:-python3.11}
result_dir=${1:?Provide a fresh output directory}
cd "$experiment_root/code"
"$python_bin" -m research.moe_dispatch.cache_route_a.run --inputs "$experiment_root/inputs.json" --out "$result_dir"
"$python_bin" -m research.moe_dispatch.cache_route_a.independent_audit --out "$result_dir"
"$python_bin" -m research.moe_dispatch.cache_route_a.report --input "$result_dir/per_window.csv" --output "$result_dir/report"
