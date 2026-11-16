#!/usr/bin/env bash
# After run_search.sh: designs for each batch size (seeded with the final chip and the tuned top10 list of each
# organisation), the evaluation of the final chips (eval, sealed and large-batch snapshots) and of the per-batch designs,
# bootstrap intervals and the result tables.
# CELLS as in run_search.sh; PERBATCH_CELLS: cells that get per-batch designs (default: Qwen3.5 NVFP4 at 96K);
# REPLICATES bootstrap replicates (default 200).
# usage: run_final.sh <data root> <output root> <capture directory>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <data root> <output root> <capture directory>" >&2; exit 2; }
data=$1 out=$2 captures=$3 jobs=${JOBS:-4} threads=${THREADS:-3} reps=${REPLICATES:-200}
cells=${CELLS:-$'qwen w4g32p 98304\nqwen fp8b128 98304\ndeepseek w4g32p 98304\ndeepseek fp8b128 98304'}
pbcells=${PERBATCH_CELLS:-"qwen w4g32p 98304"}
export OMP_NUM_THREADS=$threads
roots=(--data "$data" --out "$out" --threads "$threads")
pkg=complementary_arrays.dse
while read -r model fmt P; do
  cell=${model}_${fmt}_H8_P${P}
  for B in 2 4 8 16 32 64 96 128; do
    for o in single2 hom11 het11; do
      echo "$model $fmt $P $o $B --full-budget --tune-extra redeliv_packed=0,1 --seeds final=$out/final/$cell,$out/search/top10/$cell"
    done
  done
done <<< "$pbcells" | xargs -P "$jobs" -L 1 python -m $pkg.perbatch "${roots[@]}"
{
  while read -r model fmt P; do
    for s in s1_eval s1_sealed s1_big; do echo "$model $fmt $P $s --chips $out/final/${model}_${fmt}_H8_P${P} --version final"; done
  done <<< "$cells"
  while read -r model fmt P; do
    for B in 2 4 8 16 32 64 96 128; do
      echo "$model $fmt $P s1_b$B --chips $out/perbatch/${model}_${fmt}_H8_P${P}/B$B --version perbatch-B$B"
    done
  done <<< "$pbcells"
} | xargs -P "$jobs" -L 1 python -m $pkg.evaluate "${roots[@]}"
step=$(( (reps + jobs - 1) / jobs ))
while read -r model fmt P; do
  chips=$out/final/${model}_${fmt}_H8_P${P}
  for ((i = 0; i < reps; i += step)); do
    echo "run $model $fmt $P --name final --chips $chips --first $i --last $((i + step - 1 < reps - 1 ? i + step - 1 : reps - 1))"
    echo "run $model $fmt $P --name big --chips $chips --batches 256 512 --seed-label bootbig --first $i --last $((i + step - 1 < reps - 1 ? i + step - 1 : reps - 1))"
  done
done <<< "$cells" | xargs -P "$jobs" -L 1 python -m $pkg.bootstrap "${roots[@]}" --captures "$captures"
while read -r model fmt P; do
  python -m $pkg.bootstrap merge "$model" "$fmt" "$P" --name final --out "$out"
  python -m $pkg.bootstrap merge "$model" "$fmt" "$P" --name big --batches 256 512 --out "$out"
done <<< "$cells"
python -m $pkg.export --version final --bootstrap final --out "$out"
python -m $pkg.export --version final --set s1_big --bootstrap big --out "$out"
