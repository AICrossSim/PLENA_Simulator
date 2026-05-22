#!/usr/bin/env bash
# SSB stepwise driver — run each kernel of the SingleStreamBlock as its
# own emulator invocation, double-compared (local + global golden).
#
# Per step:
#   1. prepare <step>   — compile this one kernel, build the sim env, and
#                         seed the input HBM from the PREVIOUS step's real
#                         hbm_dump snapshot (first step uses block inputs).
#   2. emulator run     — cargo run, writes transactional_emulator/hbm_dump.bin
#   3. compare <step>   — recompute LOCAL golden from prev REAL HBM output,
#                         pull GLOBAL golden from the ideal chain, compare
#                         the same output bytes against both, snapshot the
#                         HBM image forward to .ssb_steps/<step>.hbm.bin.
#
# Stops if a step's LOCAL cosine < 0.85 (that kernel's own math is wrong)
# or any sub-step fails.
#
# Usage:
#   ./run_ssb_stepwise.sh                 # all steps in order
#   ./run_ssb_stepwise.sh layernorm modulate   # just these
set -euo pipefail

# Repo root = three levels up from this script
# (.../transactional_emulator/testbench/run_ssb_stepwise.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/tvm_ssb_stepwise_test.py"
STEP_DIR="$SCRIPT_DIR/.ssb_steps"
# Same nix gcc libstdc++ the justfile puts on LD_LIBRARY_PATH for torch.
LDLP="/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib"

# Default step order (must match STEP_ORDER in the Python driver).
DEFAULT_STEPS=(layernorm modulate linear_q)
if [ "$#" -gt 0 ]; then
  STEPS=("$@")
else
  STEPS=("${DEFAULT_STEPS[@]}")
fi

run_py() {  # run the Python driver with the testbench env
  LD_LIBRARY_PATH="$LDLP" \
  PYTHONPATH="$REPO_ROOT/compiler:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$DRIVER" "$@"
}

run_emulator() {
  local asm_path="$SCRIPT_DIR/build/generated_machine_code.mem"
  local data_path="$SCRIPT_DIR/build/hbm_for_behave_sim.bin"
  local fp_path="$SCRIPT_DIR/build/fp_sram.bin"
  local int_path="$SCRIPT_DIR/build/int_sram.bin"
  ( cd "$REPO_ROOT/transactional_emulator" && \
    RUST_BACKTRACE=1 cargo run --release -- \
      --opcode "$asm_path" --hbm "$data_path" \
      --fpsram "$fp_path" --intsram "$int_path" --quiet )
}

# Fresh report for this run.
mkdir -p "$STEP_DIR"
: > "$STEP_DIR/stepwise_report.txt"

echo "SSB stepwise: ${STEPS[*]}"
for step in "${STEPS[@]}"; do
  echo
  echo "######################## $step ########################"
  echo "[1/3] prepare $step"
  run_py prepare "$step"

  echo "[2/3] emulator run $step"
  run_emulator

  echo "[3/3] compare $step (local + global)"
  if ! run_py compare "$step"; then
    echo "STOP: $step failed local-cosine threshold (or compare error)."
    exit 1
  fi
done

echo
echo "================ stepwise report ================"
cat "$STEP_DIR/stepwise_report.txt"
echo "================================================="
