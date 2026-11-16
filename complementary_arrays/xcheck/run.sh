#!/usr/bin/env bash
# Closed-loop HBM cross-check of sampled windows: every window of the given sample tags runs through Ramulator HBM2
# (512 channels, 2,134 MT/s; extra latency 65, no issue cap, layout 0, look 512), JOBS windows at a time.
# usage: run.sh <sample dir> <run dir> <tag_B<batch>> [...]     JOBS (default 4), ENGINE_DIR (optional), CHECK=1 runs the
# ideal HBM stream instead (must reproduce the samples), KEEP_REQUESTS=1 keeps the per-request tables of the logs.
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <sample dir> <run dir> <tag_B<batch>> [...]" >&2; exit 2; }
samples=$1 runs=$2
shift 2
mkdir -p "$runs"
args=(--extra 65 --cap 0 --layout 0 --look 512)
[ "${CHECK:-0}" = 1 ] && args=(--ideal)
[ -n "${ENGINE_DIR:-}" ] && args+=(--engine-dir "$ENGINE_DIR")
for tag in "$@"; do
  n=$(python -c "import numpy as np, sys; print(len(np.load(sys.argv[1])['out']))" "$samples/$tag.npz")
  for ((w = 0; w < n; w++)); do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-4}" ]; do sleep 2; done
    (
      OMP_NUM_THREADS=1 python -m complementary_arrays.xcheck.run "$samples" "$tag" "$w" "$runs/${tag}_w$w.npz" "${args[@]}" \
        >> "$runs/log.txt" 2>&1
      [ "${KEEP_REQUESTS:-0}" = 1 ] || rm -f "$runs/${tag}_w${w}_w0_req.csv"
    ) &
  done
done
wait
