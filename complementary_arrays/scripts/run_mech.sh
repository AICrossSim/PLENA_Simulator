#!/usr/bin/env bash
# Mechanism data of the final chips (after run_final.sh): stall partitions, weight-port ladders and event timelines.
# usage: run_mech.sh <data root> <output root>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <output root>" >&2; exit 2; }
data=$1 out=$2 jobs=${JOBS:-4} threads=${THREADS:-3}
export OMP_NUM_THREADS=$threads
roots="--data $data --out $out"
ladder_cells="qwen w8g128 98304
qwen bf16 98304
deepseek w8g128 98304"
{
  for m in qwen deepseek; do for f in w8g128 bf16; do for p in 98304 196608; do for o in single2 hom11 het11; do
    echo "complementary_arrays.mech.stall $m $f $p $o $roots"
  done; done; done; done
  while read -r c; do for o in single2 hom11 het11; do echo "complementary_arrays.mech.ladder chip $c $o --threads $threads $roots"; done; done <<< "$ladder_cells"
  for f in w8g128 bf16; do for o in hom11 het11; do echo "complementary_arrays.mech.timeline qwen $f 98304 $o $roots"; done; done
} | xargs -P "$jobs" -L 1 python -m
while read -r c; do python -m complementary_arrays.mech.ladder merge $c --out "$out"; done <<< "$ladder_cells"
