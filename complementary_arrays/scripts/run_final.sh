#!/usr/bin/env bash
# Final chips, sealed-group runs, bootstrap intervals and the result table (after run_search.sh).
# usage: run_final.sh <data root> <output root>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <output root>" >&2; exit 2; }
data=$1 out=$2 jobs=${JOBS:-4} threads=${THREADS:-3}
export OMP_NUM_THREADS=$threads
python -m complementary_arrays.dse.select --out "$out"
for m in qwen deepseek; do for f in w8g128 bf16 w4g32; do for p in 98304 196608; do echo "$m $f $p"; done; done; done \
  | xargs -P "$jobs" -L 1 python -m complementary_arrays.dse.sealed --data "$data" --out "$out" --threads "$threads"
# Bootstrap intervals depend on the BLAS thread count; the reference results used 3.
OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 python -m complementary_arrays.dse.bootstrap --data "$data" --out "$out"
python -m complementary_arrays.dse.export --out "$out"
