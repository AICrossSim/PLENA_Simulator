#!/usr/bin/env bash
set -euo pipefail
mdpu_repo=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$mdpu_repo"
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
mdpu_python=${MDPU_PYTHON:-/tmp/plena-round2-venv/bin/python}
mdpu_output=${MDPU_OUTPUT:-/home/mcl123/plena-outputs/mdpu_compute_reproduction}
"$mdpu_python" -m unittest research.moe_dispatch.mdpu_compute.test_domain research.moe_dispatch.mdpu_compute.test_runtime research.moe_dispatch.mdpu_compute.test_report -v
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.run --out "$mdpu_output" --workers 8
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.run --guarded --out "$mdpu_output/guarded" --workers 8
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.audit --out "$mdpu_output"
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.audit --out "$mdpu_output/guarded"
"$mdpu_python" research/moe_dispatch/mdpu_compute/rtl/validate_rtl.py --output "$mdpu_output/rtl"
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.report --input "$mdpu_output/results.jsonl" --out "$mdpu_output"
"$mdpu_python" -m research.moe_dispatch.mdpu_compute.report --input "$mdpu_output/guarded/results.jsonl" --out "$mdpu_output/guarded"
