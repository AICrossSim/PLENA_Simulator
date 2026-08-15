#!/usr/bin/env bash
set -euo pipefail

ROOT=${PLENA_SIM_ROOT:-/home/yh3525/FYP/PLENA_Simulator}
CURRENT_RTL_ROOT=${PLENA_RTL_ROOT:-/home/yh3525/FYP/PLENA_RTL}
RTL_V5_ROOT=${RTL_V5_ROOT:-/tmp/PLENA_RTL_rtl_v5_c77b5c5}
RUN_ROOT=${RUN_ROOT:-/tmp/rtl_v6_physical_calibration_v1}
WORKER_ROOT=${WORKER_ROOT:-/tmp/rtl_v6_physical_workers_v1}
AREA_WORKERS=${AREA_WORKERS:-auto}
BASELINE_WORKERS=${BASELINE_WORKERS:-11}
POWER_MAP_WORKERS=${POWER_MAP_WORKERS:-2}
POWER_ACTIVITY_WORKERS=${POWER_ACTIVITY_WORKERS:-2}
POWER_HEAVY_ACTIVITY_WORKERS=${POWER_HEAVY_ACTIVITY_WORKERS:-2}
POWER_WORKERS=${POWER_WORKERS:-2}
POWER_MEMORY_RESERVE_GIB=${POWER_MEMORY_RESERVE_GIB:-18}
POWER_TMP_RESERVE_GIB=${POWER_TMP_RESERVE_GIB:-6}
POWER_CPU_CAPACITY=${POWER_CPU_CAPACITY:-16}
POWER_VERILATOR_JOBS=${POWER_VERILATOR_JOBS:-2}
PROMOTE=${PROMOTE:-0}
RUN_TIMING_SHADOW=${RUN_TIMING_SHADOW:-0}

CURRENT_RUN="$RUN_ROOT/current_area"
BASELINE_RUN="$RUN_ROOT/rtl_v5_baseline"
TIMING_RUN="$RUN_ROOT/timing"
POWER_RUN="$RUN_ROOT/power"
LOG="$RUN_ROOT/orchestrator.log"

mkdir -p "$RUN_ROOT"
cd "$ROOT"

prepend_first_store_bin() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      export PATH="$candidate:$PATH"
      return 0
    fi
  done
  return 1
}

# Avoid evaluating the full flake for every mapping/activity subprocess.  The
# host intentionally lacks a system C++ runtime and RTL toolchain, so expose
# only the immutable Nix store paths needed by the existing virtualenv.
for gcc_runtime in \
  /nix/store/*gcc-14.3.0-lib/lib \
  /nix/store/*gcc-14.4.0-lib/lib \
  /nix/store/*gcc*-lib/lib; do
  if [[ -e "$gcc_runtime/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$gcc_runtime:${LD_LIBRARY_PATH:-}"
    break
  fi
done
prepend_first_store_bin /nix/store/*verilator*/bin || true
prepend_first_store_bin /nix/store/*just*/bin || true
prepend_first_store_bin /nix/store/*gcc-14.3.0/bin || true
prepend_first_store_bin /nix/store/*gcc-wrapper*/bin || true
prepend_first_store_bin /nix/store/*gnumake*/bin || true
export IN_NIX_SHELL=1

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="$ROOT:$ROOT/PLENA_Compiler"

if [[ ! -d "$CURRENT_RTL_ROOT" ]]; then
  printf 'missing current RTL source: %s\n' "$CURRENT_RTL_ROOT" >&2
  exit 2
fi
if [[ ! -d "$RTL_V5_ROOT" ]]; then
  printf 'missing isolated rtl-v5 source: %s\n' "$RTL_V5_ROOT" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

fit_optional() {
  local label=$1
  shift
  if "$@" >>"$LOG" 2>&1; then
    log "$label candidate passed its fail-closed checks"
  else
    log "$label candidate remains uncalibrated; inspect diagnostics"
  fi
}

log "starting 56-point bounded-width current rtl-v6 area campaign"
if ! python analytic_models/area_new/scripts/run_area_calibration_scheduler.py \
  --plan analytic_models/area_new/config/rtl_v6_softmax_calibration_v1.yaml \
  --run-dir "$CURRENT_RUN" \
  --worker-root "$WORKER_ROOT/current_area" \
  --workers "$AREA_WORKERS" \
  --resume \
  --cleanup-worker-builds >>"$LOG" 2>&1; then
  log "parallel area campaign reported failures; retrying failed points serially"
fi

python analytic_models/area_new/scripts/run_area_calibration_scheduler.py \
  --plan analytic_models/area_new/config/rtl_v6_softmax_calibration_v1.yaml \
  --run-dir "$CURRENT_RUN" \
  --worker-root "$WORKER_ROOT/current_area_retry" \
  --workers 1 \
  --resume \
  --retry-failed \
  --cleanup-worker-builds >>"$LOG" 2>&1

log "starting eight matched rtl-v5 VectorMachine baselines"
export PLENA_RTL_NIX_ROOT="$CURRENT_RTL_ROOT"
python analytic_models/area_new/scripts/run_vector_machine_calibration.py \
  --preset rtl-v6-paired-baseline-v1 \
  --rtl-root "$RTL_V5_ROOT" \
  --run-dir "$BASELINE_RUN" \
  --worker-root "$WORKER_ROOT/rtl_v5_baseline" \
  --workers "$BASELINE_WORKERS" \
  --resume \
  --cleanup-worker-builds \
  --no-copy-to-calibration >>"$LOG" 2>&1

area_fit=(
  python analytic_models/area_new/scripts/fit_vector_rtl_v6_delta.py
  --current-csv "$CURRENT_RUN/calibration_points.csv"
  --baseline-csv "$BASELINE_RUN/calibration_points.csv"
  --output "$CURRENT_RUN/vector_rtl_v6_delta_candidate.json"
  --diagnostics-csv "$CURRENT_RUN/vector_rtl_v6_delta_diagnostics.csv"
)
if [[ "$PROMOTE" == 1 ]]; then
  area_fit+=(--promote-to-calibration)
fi
fit_optional area "${area_fit[@]}"

if [[ "$RUN_TIMING_SHADOW" == 1 ]]; then
  log "starting optional nine-point production VectorMachine timing shadow"
  python analytic_models/area_new/scripts/run_area_calibration_scheduler.py \
    --plan analytic_models/area_new/config/rtl_v6_softmax_timing_v1.yaml \
    --run-dir "$TIMING_RUN" \
    --worker-root "$WORKER_ROOT/timing" \
    --workers "$AREA_WORKERS" \
    --resume \
    --cleanup-worker-builds >>"$LOG" 2>&1

  fit_optional timing \
    python transactional_emulator/testbench/rtl_timing/build_rtl_v6_timing_artifact.py \
    --timing-csv "$TIMING_RUN/calibration_points.csv" \
    --pipeline-audit transactional_emulator/calibration/rtl_v6_reduction_pipeline_audit.json \
    --output "$TIMING_RUN/rtl_v6_vector_timing_shadow.json"
else
  log "skipping physical timing sweep; 1 GHz remains an architectural assumption"
fi

log "starting module-level rtl-v6 activity power campaign"
python analytic_models/power/scripts/run_rtl_activity_power_calibration.py \
  --run-dir "$POWER_RUN" \
  --worker-root "$WORKER_ROOT/power" \
  --plan-version softmax-v6 \
  --map-workers "$POWER_MAP_WORKERS" \
  --activity-workers "$POWER_ACTIVITY_WORKERS" \
  --heavy-activity-workers "$POWER_HEAVY_ACTIVITY_WORKERS" \
  --power-workers "$POWER_WORKERS" \
  --memory-reserve-gib "$POWER_MEMORY_RESERVE_GIB" \
  --tmp-reserve-gib "$POWER_TMP_RESERVE_GIB" \
  --cpu-capacity "$POWER_CPU_CAPACITY" \
  --verilator-jobs "$POWER_VERILATOR_JOBS" \
  --resume >>"$LOG" 2>&1

power_fit=(
  python analytic_models/power/scripts/fit_vector_rtl_v6_power_delta.py
  --input "$POWER_RUN/power_calibration_points.csv"
  --output "$POWER_RUN/vector_rtl_v6_power_delta_candidate.json"
  --csv-output "$POWER_RUN/vector_rtl_v6_power_delta_points.csv"
)
if [[ "$PROMOTE" == 1 ]]; then
  power_fit+=(--promote-to-calibration)
fi
fit_optional power "${power_fit[@]}"

log "rtl-v6 non-top-level physical calibration complete"
