#!/usr/bin/env bash
# sweep_ttft_tpot.sh -- enumerate the synthesizable (board, preset, case, phase)
# matrix and emulate each combo to capture TTFT (prefill) vs TPOT (decode) latency.
#
# LATENCY-ONLY: decode TPOT cost is op-count-based and independent of KV values, so
# decode runs go through run_model.py --past-len (which implies --no-verify). Each
# run writes <build_dir>/rust_emulator_run_stats.json (sim_latency_ns). Because the
# build-dir name does NOT encode the board (and the A7 boards share an identical
# 100 MHz latency profile), we snapshot each run's stats into a board-tagged file
# under build/ttft_tpot_results/ and drop a sweep_meta.json sidecar in the build dir
# so aggregate_ttft_tpot.py can recover (board, preset, case, phase, clock).
#
# Matrix (synthesizable only):
#   nexys_video, custom_a7 : native_16x16x16_b1, native_16x16x8_b1, native_64x64x16_b1
#   v80                    : the above + native_256x256x64_b1
#   cases  : decoder (lang), vision-layers (vision)
#   phases : prefill (--seq-len 64, past 0)  ->  TTFT
#            decode  (--past-len 1024, seq 1) -> TPOT
#
# Concurrency is capped at <=2 (default; override with SWEEP_JOBS): concurrent
# libtorch runs thrash a shared box, so OMP_NUM_THREADS=1 is forced too.
#
# Env overrides (space-separated lists) to scope the sweep down:
#   SWEEP_BOARDS   default: "nexys_video custom_a7 v80"
#   SWEEP_PRESETS  default: "native_16x16x16_b1 native_16x16x8_b1 native_64x64x16_b1"
#                  (v80 ALSO gets native_256x256x64_b1 unless overridden)
#   SWEEP_V80_EXTRA_PRESETS default: "native_256x256x64_b1" (set empty to disable)
#   SWEEP_CASES    default: "decoder vision-layers"
#   SWEEP_PHASES   default: "prefill decode"
#   SWEEP_PAST_LEN default: 1024    SWEEP_PREFILL_SEQ_LEN default: 64
#   SWEEP_LAYERS   default: 1        SWEEP_JOBS default: 2 (max concurrent runs)
#   SWEEP_COMPILE_ONLY default: 0 (set 1 to skip the emulator -- smoke only)
set -euo pipefail

# --- resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # transactional_emulator/
RESULTS_DIR="${SCRIPT_DIR}/build/ttft_tpot_results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# --- conda env ---------------------------------------------------------------
# conda's activate.d hooks reference unbound vars (e.g. ADDR2LINE) which trip the
# `set -u` above, so relax nounset just for the source+activate, then restore it.
set +u
# shellcheck disable=SC1091
source /home/khl22/miniconda3/etc/profile.d/conda.sh
conda activate plena
set -u

# --- libtorch thread cap (concurrent runs thrash otherwise) ------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE

# --- matrix (env-overridable) ------------------------------------------------
SWEEP_BOARDS="${SWEEP_BOARDS:-nexys_video custom_a7 v80}"
SWEEP_PRESETS="${SWEEP_PRESETS:-native_16x16x16_b1 native_16x16x8_b1 native_64x64x16_b1}"
SWEEP_V80_EXTRA_PRESETS="${SWEEP_V80_EXTRA_PRESETS-native_256x256x64_b1}"
SWEEP_CASES="${SWEEP_CASES:-decoder vision-layers}"
SWEEP_PHASES="${SWEEP_PHASES:-prefill decode}"
SWEEP_PAST_LEN="${SWEEP_PAST_LEN:-1024}"
SWEEP_PREFILL_SEQ_LEN="${SWEEP_PREFILL_SEQ_LEN:-64}"
SWEEP_LAYERS="${SWEEP_LAYERS:-1}"
SWEEP_JOBS="${SWEEP_JOBS:-2}"
# clamp to a sane max to guard against fork-bomb/thrash (default stays 2)
SWEEP_JOBS=$(( SWEEP_JOBS > 4 ? 4 : SWEEP_JOBS ))
SWEEP_COMPILE_ONLY="${SWEEP_COMPILE_ONLY:-0}"
NICK="smolvlm2"

# clock per board (MHz) -- used by the snapshot meta; the aggregator also reads
# the YAML directly, this is just a convenience tag.
board_clock_mhz() {
  case "$1" in
    v80) echo 400 ;;
    nexys_video|custom_a7) echo 100 ;;
    *) echo 100 ;;
  esac
}

# presets applicable to a given board (v80 gets the extra large preset)
presets_for_board() {
  local board="$1"
  echo "${SWEEP_PRESETS}"
  if [[ "${board}" == "v80" && -n "${SWEEP_V80_EXTRA_PRESETS}" ]]; then
    echo "${SWEEP_V80_EXTRA_PRESETS}"
  fi
}

# --- one combo ---------------------------------------------------------------
# args: board preset case phase
run_combo() {
  local board="$1" preset="$2" case="$3" phase="$4"
  # decode/TPOT is decoder-only: the vision encoder has no KV cache and ignores
  # past_len, so a "vision decode" run is a degenerate seq_len=1 vision prefill
  # mislabeled as TPOT. Skip those combos outright.
  if [[ "${phase}" == "decode" && "${case}" != "decoder" ]]; then
    echo "[SKIP] decode/TPOT only meaningful for case=decoder: ${board}/${preset}/${case}"
    return 0
  fi
  local tag="${board}__${preset}__${case}__${phase}"
  local log="${LOG_DIR}/${tag}.log"
  local clock; clock="$(board_clock_mhz "${board}")"

  # phase -> run_model args + the build-dir suffix run_model.py uses.
  local phase_args=() dir_suffix=""
  if [[ "${phase}" == "decode" ]]; then
    phase_args=(--past-len "${SWEEP_PAST_LEN}")   # seq-len defaults to 1; implies --no-verify
    dir_suffix="_decode_p${SWEEP_PAST_LEN}"
  else
    phase_args=(--seq-len "${SWEEP_PREFILL_SEQ_LEN}")
  fi
  local compile_args=()
  [[ "${SWEEP_COMPILE_ONLY}" == "1" ]] && compile_args=(--compile-only)

  local build_dir="${SCRIPT_DIR}/build/${NICK}_${preset}_${case}${dir_suffix}"

  echo "[RUN ] ${tag}  (clock=${clock}MHz build=${build_dir##*/})"
  set +e
  (
    cd "${EMU_DIR}" && \
    python testbench/run_model.py "${NICK}" \
      --config "${preset}" \
      --case "${case}" \
      --layers "${SWEEP_LAYERS}" \
      --board "${board}" \
      "${phase_args[@]}" \
      "${compile_args[@]}" \
      --threads 1
  ) >"${log}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[FAIL] ${tag}  rc=${rc}  (see ${log})"
    tail -n 15 "${log}" | sed 's/^/    | /' || true
    return 0   # do not abort the whole sweep on one failure
  fi

  # Snapshot the per-board stats so the next board's run cannot clobber it, and
  # drop a sidecar tagging the board/phase/clock for the aggregator.
  local stats="${build_dir}/rust_emulator_run_stats.json"
  cat >"${build_dir}/sweep_meta.json" <<EOF
{
  "board": "${board}",
  "preset": "${preset}",
  "case": "${case}",
  "phase": "${phase}",
  "clock_mhz": ${clock},
  "past_len": $([[ "${phase}" == "decode" ]] && echo "${SWEEP_PAST_LEN}" || echo 0),
  "build_dir": "${build_dir}"
}
EOF
  if [[ "${SWEEP_COMPILE_ONLY}" == "1" ]]; then
    echo "[OK  ] ${tag}  (compile-only smoke; no stats)"
    return 0
  fi
  if [[ ! -f "${stats}" ]]; then
    echo "[WARN] ${tag}  completed rc=0 but no rust_emulator_run_stats.json (slow/aborted emulator?)"
    return 0
  fi
  # board-tagged snapshot = {stats..., __sweep__: meta}
  python - "${stats}" "${build_dir}/sweep_meta.json" "${RESULTS_DIR}/${tag}.json" <<'PY'
import json, sys
stats_p, meta_p, out_p = sys.argv[1:4]
with open(stats_p) as f: stats = json.load(f)
with open(meta_p) as f: meta = json.load(f)
stats["__sweep__"] = meta
with open(out_p, "w") as f: json.dump(stats, f, indent=2, sort_keys=True)
PY
  local lat_ns; lat_ns=$(python -c "import json;print(json.load(open('${stats}')).get('sim_latency_ns'))" 2>/dev/null || echo "?")
  echo "[OK  ] ${tag}  sim_latency_ns=${lat_ns}"
}

# --- job-slot loop: cap concurrency at SWEEP_JOBS ---------------------------
declare -a PIDS=()
wait_for_slot() {
  while (( ${#PIDS[@]} >= SWEEP_JOBS )); do
    local alive=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then alive+=("${pid}"); fi
    done
    PIDS=("${alive[@]}")
    (( ${#PIDS[@]} >= SWEEP_JOBS )) && wait -n 2>/dev/null || true
  done
}

echo "=== TTFT/TPOT sweep ==="
echo "boards : ${SWEEP_BOARDS}"
echo "presets: ${SWEEP_PRESETS}   (v80 extra: ${SWEEP_V80_EXTRA_PRESETS:-<none>})"
echo "cases  : ${SWEEP_CASES}"
echo "phases : ${SWEEP_PHASES}   (decode past_len=${SWEEP_PAST_LEN}, prefill seq_len=${SWEEP_PREFILL_SEQ_LEN})"
echo "jobs   : ${SWEEP_JOBS}  compile_only=${SWEEP_COMPILE_ONLY}"
echo "results-> ${RESULTS_DIR}"
echo

N_TOTAL=0
for board in ${SWEEP_BOARDS}; do
  for preset in $(presets_for_board "${board}"); do
    for case in ${SWEEP_CASES}; do
      for phase in ${SWEEP_PHASES}; do
        wait_for_slot
        run_combo "${board}" "${preset}" "${case}" "${phase}" &
        PIDS+=("$!")
        N_TOTAL=$((N_TOTAL + 1))
      done
    done
  done
done

# drain
wait
echo
echo "=== sweep complete: ${N_TOTAL} combos enqueued ==="
echo "aggregate with: python ${SCRIPT_DIR}/aggregate_ttft_tpot.py"
