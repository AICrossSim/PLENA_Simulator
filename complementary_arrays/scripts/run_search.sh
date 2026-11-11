#!/usr/bin/env bash
# All searches: top10 for every organisation and hetopt for the het organisations, then the fullhom and hetcap controls
# (two streams; hetcap reads the top10 hom chip of its cell). JOBS processes in parallel, THREADS engine threads each;
# ENGINE_OPTIONS (e.g. "--systolic 3 --drain-form tree") are passed to every search.
# usage: run_search.sh <data root> <output root>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <output root>" >&2; exit 2; }
data=$1 out=$2 jobs=${JOBS:-4} threads=${THREADS:-3}
export OMP_NUM_THREADS=$threads
read -r -a options <<< "${ENGINE_OPTIONS:-}"
search=(python -m complementary_arrays.dse.search --data "$data" --out "$out" --threads "$threads" "${options[@]}")
cells2=$(for m in qwen deepseek qwen3next; do for f in w8g128 bf16 w4g32; do for p in 98304 196608; do echo "$m $f $p"; done; done; done)
cells4=$(for m in qwen deepseek; do for f in w8g128 bf16; do for p in 98304 196608; do echo "$m $f $p"; done; done; done)
{
  while read -r c; do for o in single2 hom11 het11; do echo "$c $o --stage top10"; done; echo "$c het11 --stage hetopt"; done <<< "$cells2"
  while read -r c; do for o in single4 hom22 het22; do echo "$c $o --stage top10"; done; echo "$c het22 --stage hetopt"; done <<< "$cells4"
} | xargs -P "$jobs" -L 1 "${search[@]}"
while read -r c; do echo "$c hom11 --stage fullhom"; echo "$c het11 --stage hetcap"; done <<< "$cells2" | xargs -P "$jobs" -L 1 "${search[@]}"
