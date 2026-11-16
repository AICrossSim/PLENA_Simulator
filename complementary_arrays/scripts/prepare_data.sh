#!/usr/bin/env bash
# Snapshot files of both models and both routing captures from the per-request routing captures, checked against
# snapshots.sha256 (content digests of the files the paper used).
# usage: prepare_data.sh <data root> <capture directory>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <capture directory>" >&2; exit 2; }
data=$1 captures=$2
here=$(cd "$(dirname "$0")" && pwd)
for model in qwen deepseek; do
  for route in int4 ref; do
    python -m complementary_arrays.dse.snapshots build "$model" "$route" --captures "$captures" --data "$data"
  done
done
python -m complementary_arrays.dse.snapshots check "$here/snapshots.sha256" --data "$data"
