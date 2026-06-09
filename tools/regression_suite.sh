#!/usr/bin/env bash
# OOO-branch regression suite.
#
# Runs a curated set of tilelang verify scripts against the PLENA emulator
# and prints a one-line summary per test. Use to detect numerical drift when
# merging features, working on Step 2.2 (per-unit dispatcher), or any other
# main.rs / load_config.rs change.
#
# Coverage:
#   flash_per_head_v3     — production text-prefill attention (per-head
#                           online-softmax flash, H_Q=4 H_KV=2 S_Q=256
#                           S_KV=1024 HD=128). Exercises matrix+vector+
#                           scalar units, prefetch, HBM writeback. HBM-
#                           authoritative verify.
#   linear                — text-prefill linear+bias kernel. Fast smoke.
#   regime_sweep          — 6 (regime × KIT) linear_bias shapes, ~5min.
#                           Primary OOO sensitivity test — bit-identical MAE
#                           across runs proves the dispatcher isn't shuffling
#                           data.
#
# Retired:
#   flash_v2_bmm_verify   — bmm-based flash kernel no longer used by any
#       model pipeline (2026-06); replaced by flash_per_head_v3 above.
#   bmm_via_parallel_jit_verify  — kernel-side: relies on the *old*
#       constructor default `bmm_scale=0.25`; FAILs since the simulator's
#       bmm_scale=1.0 fix until tilelang emits C_SET_BMM_SCALE.
#
# Usage:
#   ./tools/regression_suite.sh                # run + print summary
#   ./tools/regression_suite.sh --baseline     # also write baseline file
#   ./tools/regression_suite.sh --diff         # diff against saved baseline
#
# Assumes:
#   * Gateway running on 127.0.0.1:7979 (the ooo_arch worktree default —
#     see ../transactional_emulator/start_online_sim.sh for the
#     online_emulator-vs-ooo_arch port layout). Start via
#     start_online_sim.sh or by launching
#     `transactional_emulator --gateway` directly. Override via
#     `EMU_PORT=<n> tools/regression_suite.sh ...` if you need to test
#     against a non-default gateway.
#   * $PLENA_SIM points to PLENA_Simulator checkout (defaults to the
#     enclosing repo of this script).
#   * tilelang_for_plena checked out next to PLENA_Simulator at the
#     same parent dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pin PLENA_SIM to THIS worktree before sourcing _sim_env.sh — the
# default in playground/_sim_env.sh points at the sibling
# `PLENA_Simulator-yw-online_emulator` checkout, which would silently
# resolve PLENA_CONFIG=config_2 to a TOML on the wrong branch and run
# our regression against that one's gateway+binary. Exporting ensures
# the value survives into the child `uv run python` env.
export PLENA_SIM="${PLENA_SIM:-$(cd "$SCRIPT_DIR/.." && pwd)}"
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

# Gateway port. 7979 = ooo_arch worktree default; override with EMU_PORT=...
EMU_PORT="${EMU_PORT:-7979}"

BASELINE="$SCRIPT_DIR/regression_baseline.txt"
RESULTS="/tmp/plena_regression_$(date +%Y%m%d_%H%M%S).txt"

mode="run"
case "${1:-}" in
    --baseline) mode="baseline" ;;
    --diff)     mode="diff" ;;
    "")         mode="run" ;;
    *) echo "unknown arg: $1 (use --baseline or --diff)" >&2 ; exit 2 ;;
esac

# Sanity: gateway listening on the expected port?
if ! lsof -nP -iTCP:${EMU_PORT} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: no PLENA gateway on 127.0.0.1:${EMU_PORT}" >&2
    echo "  start with:" >&2
    echo "    cd $PLENA_SIM/transactional_emulator && ./start_online_sim.sh --background" >&2
    echo "  or direct:" >&2
    echo "    cd $PLENA_SIM/transactional_emulator && \\" >&2
    echo "      DYLD_LIBRARY_PATH=\$(find /nix/store -path '*ramulator2*/lib' -print -quit):\$(find target -path '*torch-sys-*/out/libtorch/libtorch/lib' -print -quit) \\" >&2
    echo "      ./target/release/transactional_emulator --gateway --bind 127.0.0.1:${EMU_PORT} --quiet > /tmp/plena_emulator.log 2>&1 &" >&2
    exit 1
fi

# Drive tilelang client scripts to this gateway. Each verify.py reads
# PLENA_SIM_PORT (defaults to 7878 in the script — wrong for us). Pin it
# to EMU_PORT so every plena_sim_run_python invocation targets the
# ooo_arch gateway, not whatever else is listening on 7878.
export PLENA_SIM_PORT="${EMU_PORT}"

cd "$TILELANG"
# _sim_env.sh resolves PLENA_CONFIG: if it's a bare name it looks up
# $PLENA_SIM/configs/<name>.toml and `exit 1`s on miss; unset defaults to
# "config_1" which we don't ship on yw/ooo_arch. Force the resolution
# target to "config_2" (the only config that exists here) before sourcing,
# regardless of what the parent shell had.
export PLENA_CONFIG=config_2
# shellcheck source=/dev/null
source playground/_sim_env.sh

# `_sim_env.sh` defaults PLENA_SIM_SKIP_CONFIG_HANDSHAKE=1 for legacy
# compatibility. We NEED the handshake — without it the gateway spawns
# a backend with no per-session config (mlen=32 defaults) and every
# kernel run fails the kernel-config marker check.
#
# CAUTION 1: must come AFTER `source _sim_env.sh` because _sim_env.sh
# unconditionally `export`s SKIP=1 via `${VAR:-1}` (which fills in even
# an empty-string VAR), undoing any earlier unset.
# CAUTION 2: server.py uses `if not os.environ.get(...)`, which is
# truthy for ANY non-empty string — including "0". The only way to
# re-enable the handshake is to UNSET the variable, not set it to "0".
unset PLENA_SIM_SKIP_CONFIG_HANDSHAKE

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

# flash_per_head_v3: the production text-prefill attention path
# (per-head online-softmax flash, H_Q=4 H_KV=2 S_Q=256 S_KV=1024 HD=128).
# Replaced flash_v2_bmm (2026-06): the bmm-based flash kernel is no
# longer used by any model pipeline. HBM-authoritative verify — the
# MSE/MAE/max_err lines come from the decoded O_pad in the HBM dump.
run_one "flash_per_head_v3" \
    "playground/models/qwen3_vl/kernels/plena_text_prefill_flash_per_head_v3_verify.py" \
    "grep -E 'MSE|MAE|Max abs err|Allclose PASS|PASS: codegen' | head -5 | tr '\n' ' ' | sed -E 's/ +/ /g; s/^/flash_per_head_v3 /'" \
    --run
# NOTE: deliberately *excluding* any wall-clock/latency lines (see
# regime_sweep note below): executor task ordering at equal simulated
# instants produces sub-PPM latency jitter that would mask the real
# regression signal — bit-identical MSE/MAE/max_err.

# NOTE: tilelang renamed the verify scripts in 2026-Q1 — kept the test
# IDs ("linear", "regime_sweep") stable for baseline continuity but
# updated the paths. The old `kernels/linear_verify.py` is now
# `plena_text_prefill_linear_verify.py`; the old
# `verification/linear_bias/regime_sweep.py` is now under
# `verification/plena_vision_prefill_linear_bias/`.
run_one "linear" \
    "playground/models/qwen3_vl/kernels/plena_text_prefill_linear_verify.py" \
    "grep -E 'max abs err|cos sim|PASS|FAIL' | head -3 | tr '\n' ' ' | sed 's/^/linear /'"

# Per-case verdict lines look like (post tilelang 2026-Q1 refactor —
# MSE= was replaced by cos= / MAE= / max_err=):
#   [PASS ] regime_a_kit1_K1024  shape=(8, 1024)  cos=0.9999  MAE=1.409e-02  max_err=7.910e-02  max|ref|=...  took 30.1s
# Strip the trailing wall-clock field — it's run-to-run jitter, not signal.
run_one "regime_sweep" \
    "playground/models/qwen3_vl/kernels/verification/plena_vision_prefill_linear_bias/regime_sweep.py" \
    "grep -E '\[(PASS|PASSish|FAIL) *\] regime_[ab]_kit|Total: [0-9]+ PASS' | sed -E 's/ +took [0-9.]+s\$//' | sed 's/^/regime_sweep /'" \
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
