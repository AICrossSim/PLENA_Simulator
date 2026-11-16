#!/usr/bin/env bash
# Search and final chips of every cell: top10 for the three organisations, hetopt and the sharded wide pair search for
# the pair, the prior stage when SEEDS names earlier designs ([name=]directories of <org>.json; they also compete as they
# are in the selection), a first selection, the re-tuning round of the pair from that selection, and the final
# selection. Every chip uses the whole multiplier budget and tunes the re-read format; the wide pair search also tunes
# the placement of the shared expert.
# CELLS: lines "model fmt P" (default: both models in NVFP4 and FP8 at 96K multipliers); JOBS processes in parallel,
# THREADS engine threads each; HETX_SHARDS shards of the wide pair search.
# usage: run_search.sh <data root> <output root>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 <data root> <output root>" >&2; exit 2; }
data=$1 out=$2 jobs=${JOBS:-4} threads=${THREADS:-3} shards=${HETX_SHARDS:-8}
cells=${CELLS:-$'qwen w4g32p 98304\nqwen fp8b128 98304\ndeepseek w4g32p 98304\ndeepseek fp8b128 98304'}
export OMP_NUM_THREADS=$threads
read -r -a seeds <<< "${SEEDS:-}"
roots=(--data "$data" --out "$out" --threads "$threads")
search=(python -m complementary_arrays.dse.search "${roots[@]}" --full-budget)
packed=(--tune-extra redeliv_packed=0,1)
place=(--tune-extra redeliv_packed=0,1 shared_place=0,1,2)
{
  while read -r c; do
    for o in single2 hom11 het11; do echo "$c $o --stage top10 ${packed[*]}"; done
    echo "$c het11 --stage hetopt ${packed[*]}"
  done <<< "$cells"
} | xargs -P "$jobs" -L 1 "${search[@]}"
{
  while read -r c; do
    for ((s = 0; s < shards; s++)); do
      echo "$c het11 --stage hetx$s --parts $shards --probes top10 hetopt ${place[*]}"
    done
    if [ ${#seeds[@]} -gt 0 ]; then
      for o in single2 hom11 het11; do echo "$c $o --stage prior --seeds ${seeds[*]} ${packed[*]}"; done
    fi
  done <<< "$cells"
} | xargs -P "$jobs" -L 1 "${search[@]}"
select_all() {  # $1 = extra pair stages
  while read -r model fmt P; do
    cell=${model}_${fmt}_H8_P${P}
    for o in single2 hom11 het11; do
      stages=(top10)
      [ "$o" = het11 ] && stages+=(hetopt)
      [ -f "$out/search/prior/$cell/$o.json" ] && stages+=(prior)
      [ "$o" = het11 ] && stages+=($(cd "$out/search" && ls -d hetx[0-9]* $1 2>/dev/null))
      measure=()
      [ ${#seeds[@]} -gt 0 ] && measure=(--measure "${seeds[@]}")
      python -m complementary_arrays.dse.select "$model" "$fmt" "$P" "$o" --stages "${stages[@]}" "${measure[@]}" \
        --budget "${roots[@]}"
    done
  done <<< "$cells"
}
select_all ""
while read -r model fmt P; do
  for part in 0 1; do
    echo "$model $fmt $P het11 --stage retune$part --parts 2 -K 6 --from-stages $(cd "$out/search" && ls -d hetx[0-9]* | tr '\n' ' ') --reference $out/final/${model}_${fmt}_H8_P${P}/het11.json ${place[*]}"
  done
done <<< "$cells" | xargs -P "$jobs" -L 1 "${search[@]}"
select_all "retune[0-9]*"
