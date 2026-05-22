#!/usr/bin/env bash
# OOO-branch regression suite.
#
# Runs a curated set of tilelang verify scripts against the PLENA emulator
# and prints a one-line summary per test. Use to detect numerical drift when
# merging features, working on Step 2.2 (per-unit dispatcher), or any other
# main.rs / load_config.rs change.
#
# Coverage:
#   flash_v2_bmm_verify   — BMM Q@K^T, 2961 instructions, prefetch + matrix
#                           + writeback; deterministic, ~30s. PASS canary.
#   linear_verify         — basic linear+bias kernel, ~10s. Fast smoke test.
#   regime_sweep          — 6 (regime × KIT) shapes, BMM-heavy, ~5min.
#                           Primary OOO sensitivity test — bit-identical MAE
#                           across runs proves the dispatcher isn't shuffling
#                           data.
#
# Skipped (known broken on yw/ooo_arch):
#   bmm_via_parallel_jit_verify  — kernel-side: relies on the *old*
#       constructor default `bmm_scale=0.25`. Post bmm_scale=1.0 fix on the
#       simulator side, this test FAILs (max_err=1.7481, 49% match) until
#       the tilelang side emits an explicit C_SET_BMM_SCALE. Tracked
#       separately; do NOT include here.
#
# Usage:
#   ./tools/regression_suite.sh                # run + print summary
#   ./tools/regression_suite.sh --baseline     # also write baseline file
#   ./tools/regression_suite.sh --diff         # diff against saved baseline
#
# Assumes:
#   * Gateway running on 127.0.0.1:7878 (start via start_online_sim.sh or
#     by launching `transactional_emulator --gateway` directly).
#   * $PLENA_SIM points to PLENA_Simulator checkout (defaults to the
#     enclosing repo of this script).
#   * tilelang_for_plena checked out next to PLENA_Simulator at the
#     same parent dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLENA_SIM="${PLENA_SIM:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TILELANG="${TILELANG_ROOT:-$(cd "$PLENA_SIM/../tilelang_for_plena" 2>/dev/null && pwd || true)}"

if [[ -z "${TILELANG:-}" || ! -d "$TILELANG" ]]; then
    echo "ERROR: tilelang_for_plena not found at \$PLENA_SIM/../tilelang_for_plena" >&2
    echo "       Set \$TILELANG_ROOT explicitly." >&2
    exit 1
fi

PLENA_CONFIG_PATH="${PLENA_CONFIG:-$PLENA_SIM/configs/config_2.toml}"
if [[ ! -f "$PLENA_CONFIG_PATH" ]]; then
    echo "ERROR: PLENA_CONFIG not found at $PLENA_CONFIG_PATH" >&2
    exit 1
fi

BASELINE="$SCRIPT_DIR/regression_baseline.txt"
RESULTS="/tmp/plena_regression_$(date +%Y%m%d_%H%M%S).txt"

mode="run"
case "${1:-}" in
    --baseline) mode="baseline" ;;
    --diff)     mode="diff" ;;
    "")         mode="run" ;;
    *) echo "unknown arg: $1 (use --baseline or --diff)" >&2 ; exit 2 ;;
esac

# Sanity: gateway listening?
if ! lsof -nP -iTCP:7878 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: no PLENA gateway on 127.0.0.1:7878" >&2
    echo "  start with:" >&2
    echo "    cd $PLENA_SIM/transactional_emulator && ./start_online_sim.sh --background" >&2
    echo "  or direct:" >&2
    echo "    cd $PLENA_SIM/transactional_emulator && \\" >&2
    echo "      DYLD_LIBRARY_PATH=\$(find /nix/store -path '*ramulator2*/lib' -print -quit):\$(find target -path '*torch-sys-*/out/libtorch/libtorch/lib' -print -quit) \\" >&2
    echo "      ./target/release/transactional_emulator --gateway --bind 127.0.0.1:7878 --quiet > /tmp/plena_emulator.log 2>&1 &" >&2
    exit 1
fi

cd "$TILELANG"
# _sim_env.sh resolves PLENA_CONFIG: if it's a bare name it looks up
# $PLENA_SIM/configs/<name>.toml and `exit 1`s on miss; unset defaults to
# "config_1" which we don't ship on yw/ooo_arch. Force the resolution
# target to "config_2" (the only config that exists here) before sourcing,
# regardless of what the parent shell had.
export PLENA_CONFIG=config_2
# shellcheck source=/dev/null
source playground/_sim_env.sh

echo "================================================================="
echo "OOO-branch regression suite"
echo "  PLENA_SIM       = $PLENA_SIM"
echo "  TILELANG_ROOT   = $TILELANG"
echo "  PLENA_CONFIG    = $PLENA_CONFIG_PATH"
echo "  BASELINE        = $BASELINE"
echo "  RESULTS         = $RESULTS"
echo "  MODE            = $mode"
echo "================================================================="

# ---------------------------------------------------------------------------
# Per-test runner: extracts one key metric line per test for diff-friendly
# comparison. Format:  "<test> <status> <metric>"
# ---------------------------------------------------------------------------

run_one() {
    local test_id="$1" ; shift
    local script="$1" ; shift
    local extract="$1" ; shift
    # remaining args are passed to the script

    echo "------ $test_id ------" >&2
    local out
    # plena_sim_run_python ends with `exec uv run python ...` which would
    # replace our shell. Wrap in a subshell so the exec only kills the
    # subshell, not our outer script.
    out=$( ( plena_sim_run_python "$script" "$@" ) 2>&1 ) || true
    # Extract per-test metric using a caller-supplied awk/grep snippet.
    local line
    line=$(echo "$out" | eval "$extract" || true)
    if [[ -z "$line" ]]; then
        line="$test_id NO_OUTPUT (test runner produced no parseable metric)"
    fi
    echo "$line" | tee -a "$RESULTS"
}

# Header
echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) — PLENA OOO regression suite" > "$RESULTS"
echo "# HEAD = $(git -C "$PLENA_SIM" rev-parse --short HEAD) $(git -C "$PLENA_SIM" log -1 --pretty='%s')" >> "$RESULTS"
echo "" >> "$RESULTS"

run_one "flash_v2_bmm" \
    "playground/ops/attention/flash_v2_bmm_verify.py" \
    "grep -E 'All Values Pass|PASS: codegen' | head -2 | tr '\n' ' ' | sed 's/^/flash_v2_bmm /'" \
    --run --quiet
# NOTE: deliberately *excluding* the 'Simulation completed. Latency Xns' line.
# Step 1's per-tile fanout spawns multiple oneshot tasks per H_PREFETCH; the
# executor's ordering of those tasks at the same simulated time is
# implementation-defined, which produces ~30ns of run-to-run latency jitter
# on a 17ms kernel (sub-PPM). That noise would mask the real regression
# signal we care about — bit-identical MSE/MAE/max_err from regime_sweep.

run_one "linear" \
    "playground/models/qwen3_vl/kernels/linear_verify.py" \
    "grep -E 'max abs err|cos sim|PASS|FAIL' | head -3 | tr '\n' ' ' | sed 's/^/linear /'"

run_one "regime_sweep" \
    "playground/models/qwen3_vl/kernels/verification/linear_bias/regime_sweep.py" \
    "grep -E 'regime_[ab]_kit.*MSE|Total: [0-9]+ PASS' | sed -E 's/  took [0-9.]+s$//' | sed 's/^/regime_sweep /'" \
    --run

echo ""
echo "================================================================="
echo "Results written to: $RESULTS"
echo "================================================================="

case "$mode" in
    baseline)
        cp "$RESULTS" "$BASELINE"
        echo "Baseline updated: $BASELINE"
        ;;
    diff)
        if [[ ! -f "$BASELINE" ]]; then
            echo "No baseline at $BASELINE — run with --baseline first." >&2
            exit 3
        fi
        echo "Diff vs baseline:"
        # Skip the first two metadata lines (timestamp + git HEAD) so a
        # baseline run on commit A vs a check run on commit B doesn't show
        # spurious diffs.
        if diff -u <(tail -n +3 "$BASELINE") <(tail -n +3 "$RESULTS"); then
            echo "(identical — no numerical drift)"
        else
            echo ""
            echo "*** REGRESSION: numbers diverged from baseline ***" >&2
            exit 4
        fi
        ;;
esac
