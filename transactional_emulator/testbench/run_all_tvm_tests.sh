#!/usr/bin/env bash
# Run every TVM testbench one by one via `just build-emulator-debug <arg>`,
# scrape each run's build/comparison_report.txt for the global cosine, and
# flag any kernel whose cosine < 0.85 (FAIL) — that's a real problem.
#
# Usage:
#   ./run_all_tvm_tests.sh                 # all kernels in TESTS below
#   ./run_all_tvm_tests.sh layernorm_min rope_min   # just these args
#
# Each `just build-emulator-debug` wipes build/ and re-runs the full
# compile -> sim -> compare pipeline, so the report is fresh per kernel.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORT="$SCRIPT_DIR/build/comparison_report.txt"
THRESH=0.85
SUMMARY="$SCRIPT_DIR/all_tests_summary.txt"
# nix gcc libstdc++ for torch's C extensions (same path justfile + the
# stepwise driver use; the direnv shell sets this but a child shell of
# this script may not inherit it).
LDLP="${LD_LIBRARY_PATH:-/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib}"

# The just args (kernel names) to run. Single-kernel testbenches that
# produce a comparison_report. Chained / staged drivers (single_stream_block,
# ssb_*) and pure-bisect variants are intentionally left out — add them as
# CLI args if you want them.
TESTS=(
  layernorm_min
  rmsnorm_min
  modulate_min
  rope_min
  gelu_min
  silu_min
  residual_gate_min
  linear_min
  flash_attention_min
)
if [ "$#" -gt 0 ]; then
  TESTS=("$@")
fi

: > "$SUMMARY"
echo "Running ${#TESTS[@]} TVM testbenches; flagging cosine < $THRESH"
echo

pass=0; fail=0; noreport=0
for arg in "${TESTS[@]}"; do
  echo "============================================================"
  echo ">>> $arg"
  echo "============================================================"

  # Run via `just` from the repo root. The child shell inherits this
  # script's PATH, so as long as you launch the script from inside the
  # direnv/nix shell (where `just` is on PATH), it resolves here too.
  ( cd "$REPO_ROOT" && just build-emulator-debug "$arg" ) \
    > "$SCRIPT_DIR/build_${arg}.log" 2>&1
  rc=$?

  if [ ! -f "$REPORT" ]; then
    echo "  [no report]  (just rc=$rc — see build_${arg}.log)"
    printf '%-26s NO_REPORT  (rc=%s)\n' "$arg" "$rc" >> "$SUMMARY"
    noreport=$((noreport+1))
    continue
  fi

  # Scrape the global cosine. Report line: "Global cosine similarity: 0.999614"
  cos=$(grep -iE 'Global cosine similarity' "$REPORT" | head -1 \
        | grep -oE '[0-9]+\.[0-9]+' | head -1)
  nrmse=$(grep -iE 'NRMSE' "$REPORT" | head -1 \
        | grep -oE '[0-9]+\.[0-9]+' | head -1)

  if [ -z "$cos" ]; then
    echo "  [no cosine in report]  (rc=$rc)"
    printf '%-26s NO_COSINE  (rc=%s)\n' "$arg" "$rc" >> "$SUMMARY"
    noreport=$((noreport+1))
    continue
  fi

  # cosine < THRESH ?  (awk for float compare)
  if awk "BEGIN{exit !($cos < $THRESH)}"; then
    verdict="FAIL"
    fail=$((fail+1))
    echo "  *** FAIL  cosine=$cos  NRMSE=${nrmse}%  (< $THRESH)"
  else
    verdict="PASS"
    pass=$((pass+1))
    echo "  PASS  cosine=$cos  NRMSE=${nrmse}%"
  fi
  printf '%-26s %-4s  cosine=%s  NRMSE=%s%%\n' \
    "$arg" "$verdict" "$cos" "${nrmse}" >> "$SUMMARY"

  # Snapshot this kernel's report so the next `just` (which wipes build/)
  # doesn't clobber it.
  cp -f "$REPORT" "$SCRIPT_DIR/report_${arg}.txt" 2>/dev/null || true
done

echo
echo "==================== SUMMARY ===================="
cat "$SUMMARY"
echo "================================================="
echo "PASS=$pass  FAIL=$fail  NO_REPORT=$noreport"
[ "$fail" -eq 0 ] && [ "$noreport" -eq 0 ]
