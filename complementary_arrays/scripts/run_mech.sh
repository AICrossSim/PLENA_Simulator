#!/usr/bin/env bash
# After run_final.sh: multiplier-cycle accounting of the final chips and of the batch-128 designs, the single-layer
# timeline of the batch-128 designs, and ablations of the final chips (two-ended dispatch on every chip, first-come
# first-served weight port, no HBM prefetch). MECH_CELLS: cells (default: CELLS of run_search.sh); the batch-128 designs
# are those of PERBATCH_CELLS.
# usage: run_mech.sh <data root> <output root>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <output root>" >&2; exit 2; }
data=$1 out=$2 jobs=${JOBS:-4} threads=${THREADS:-3}
cells=${MECH_CELLS:-${CELLS:-$'qwen w4g32p 98304\nqwen fp8b128 98304\ndeepseek w4g32p 98304\ndeepseek fp8b128 98304'}}
pbcells=${PERBATCH_CELLS:-"qwen w4g32p 98304"}
export OMP_NUM_THREADS=$threads
roots="--data $data --out $out --threads $threads"
mech=complementary_arrays.mech
{
  while read -r model fmt P; do
    final="--chips $out/final/${model}_${fmt}_H8_P${P} --version final"
    echo "$mech.stall $model $fmt $P $final $roots"
    echo "$mech.ablate $model $fmt $P pol13 policy=13 $final $roots"
    echo "$mech.ablate $model $fmt $P fcfs port_arbitration=fcfs $final $roots"
    echo "$mech.ablate $model $fmt $P nopf prefetch=false $final $roots"
  done <<< "$cells"
  while read -r model fmt P; do
    echo "$mech.stall $model $fmt $P --chips $out/perbatch/${model}_${fmt}_H8_P${P}/B128 --version perbatch-B128 --batches 128 $roots"
  done <<< "$pbcells"
} | xargs -P "$jobs" -L 1 python -m
while read -r model fmt P; do
  python -m $mech.timeline "$model" "$fmt" "$P" --chips "$out/perbatch/${model}_${fmt}_H8_P${P}/B128" \
    --version perbatch-B128 --set s1_b128 --batch 128 --data "$data" --out "$out"
done <<< "$pbcells"
