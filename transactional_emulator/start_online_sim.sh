#!/usr/bin/env bash
# Launch the PLENA online emulator + Flask Web GUI.
#
# Server topology (default: gateway):
#   --gateway       : multi-tenant. Listens on $EMU_PORT, spawns an isolated
#                     `--serve` backend subprocess per client session, reaps
#                     the backend on client disconnect. Each backend has its
#                     own EmulatorState (no cross-session interference). This
#                     is the default.
#   --serve         : single-tenant. Original single-backend service — all
#                     clients share one EmulatorState. Cheaper if you only
#                     ever have one process driving the simulator.
#
# Usage:
#   ./start_online_sim.sh                        # foreground, gateway + monitor GUI
#   ./start_online_sim.sh --serve                # foreground, single-tenant
#   ./start_online_sim.sh --dev                  # foreground, dev GUI mode
#   ./start_online_sim.sh --background           # detach, log to /tmp/plena_*.log
#   ./start_online_sim.sh --background --dev     # detached dev mode
#
# GUI modes:
#   monitor : read-only live view of emulator state, sessions, history.
#             Default. Safe to leave open while jobs hit the simulator.
#   dev     : full interactive console (load files, execute, reset, ASM input).
#
# Stop background instances with: ./stop_online_sim.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

EMU_BIN="$SCRIPT_DIR/target/release/transactional_emulator"
VENV_PY="$PROJECT_ROOT/.venv/bin/python3"
WEBGUI="$SCRIPT_DIR/demo/webgui.py"

EMU_HOST="${EMU_HOST:-127.0.0.1}"
# Port layout — keep ooo_arch separate from the sibling yw/online_emulator
# worktree so both can run simultaneously without stomping on each other:
#   * online_emulator worktree: gateway 7878 / webgui 5001 (upstream defaults)
#   * ooo_arch worktree (this one): gateway 7979 / webgui 5002
# The "79" in 7979 mnemonics the OOO branch; 5002 steps past macOS AirPlay
# (5000) and the online_emulator GUI (5001). Override via EMU_PORT/WEB_PORT.
EMU_PORT="${EMU_PORT:-7979}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5002}"

if [[ ! -x "$EMU_BIN" ]]; then
  echo "ERROR: transactional_emulator release binary missing. Build with 'cargo build --release' from transactional_emulator/." >&2
  exit 1
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: project venv not found. Create .venv at the repository root." >&2
  exit 1
fi

# --- dyld setup: the release binary needs both libramulator.dylib (from
# the nix store) and libtorch (from the cargo build dir). We can't bake
# these into the binary via @rpath because they live at locations the
# Cargo build doesn't know about. Discover them at launch time and
# prepend to DYLD_LIBRARY_PATH so dyld can resolve @rpath/* references.
# Both can be overridden via $RAMULATOR_LIB / $PLENA_EMULATOR_LIBTORCH if
# the auto-discovery picks the wrong one.
if [[ -z "${RAMULATOR_LIB:-}" ]]; then
  _RAMULATOR_DYLIB="$(find /nix/store -maxdepth 5 -path '*ramulator2-*/lib/libramulator.dylib' -print -quit 2>/dev/null || true)"
  if [[ -n "$_RAMULATOR_DYLIB" ]]; then
    RAMULATOR_LIB="$(dirname "$_RAMULATOR_DYLIB")"
  fi
fi
if [[ ! -f "${RAMULATOR_LIB:-}/libramulator.dylib" ]]; then
  echo "ERROR: libramulator.dylib not found. Install ramulator2 (nix) or set RAMULATOR_LIB=/path/to/lib." >&2
  exit 1
fi
if [[ -z "${PLENA_EMULATOR_LIBTORCH:-}" ]]; then
  PLENA_EMULATOR_LIBTORCH="$(find "$SCRIPT_DIR/target" -maxdepth 8 -path '*torch-sys-*/out/libtorch/libtorch/lib' -print -quit 2>/dev/null || true)"
fi
if [[ -z "${PLENA_EMULATOR_LIBTORCH:-}" || ! -d "$PLENA_EMULATOR_LIBTORCH" ]]; then
  echo "ERROR: cargo-built libtorch dir not found under $SCRIPT_DIR/target. Run 'cargo build --release' once so torch-sys downloads it, or set PLENA_EMULATOR_LIBTORCH=/path/to/libtorch/lib." >&2
  exit 1
fi
export DYLD_LIBRARY_PATH="$RAMULATOR_LIB:$PLENA_EMULATOR_LIBTORCH:${DYLD_LIBRARY_PATH:-}"

# The emulator looks for plena_settings.toml at $cwd/../plena_settings.toml,
# so we must launch from inside transactional_emulator/.
#
# Optional override: set $PLENA_CONFIG to either a named config (e.g.
# "config_2", resolved to ../configs/config_2.toml) or an absolute /
# relative .toml path. See load_config.rs::load_config for the resolution
# rules. Use it like:
#
#   PLENA_CONFIG=config_2 ./start_online_sim.sh --background
#   PLENA_CONFIG=/abs/path/to/custom.toml ./start_online_sim.sh
#
# Unset → falls back to ../plena_settings.toml.
cd "$SCRIPT_DIR"
if [[ -n "${PLENA_CONFIG:-}" ]]; then
  echo "PLENA_CONFIG=${PLENA_CONFIG} (emulator will resolve via load_config.rs)"
else
  echo "PLENA_CONFIG unset → emulator falls back to ../plena_settings.toml"
fi

# Per-worktree pid/log paths — keyed by EMU_PORT so the ooo_arch worktree
# (7979) and the sibling yw/online_emulator worktree (7878) don't fight
# over the same /tmp/plena_emulator.pid (which would let either branch's
# stop script kill the other's process).
PID_FILE_EMU="/tmp/plena_emulator_${EMU_PORT}.pid"
PID_FILE_WEB="/tmp/plena_webgui_${WEB_PORT}.pid"
LOG_EMU="/tmp/plena_emulator_${EMU_PORT}.log"
LOG_WEB="/tmp/plena_webgui_${WEB_PORT}.log"

run_mode="foreground"       # foreground | background
gui_mode="monitor"          # monitor | dev
server_mode="gateway"       # gateway | serve
for arg in "$@"; do
  case "$arg" in
    --background) run_mode="background" ;;
    --dev)        gui_mode="dev" ;;
    --monitor)    gui_mode="monitor" ;;
    --gateway)    server_mode="gateway" ;;
    --serve)      server_mode="serve" ;;
    -h|--help)
      sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Translate to the emulator binary flag.
case "$server_mode" in
  gateway) emu_server_flag="--gateway" ;;
  serve)   emu_server_flag="--serve" ;;
esac

wait_for_port() {
  local host="$1" port="$2"
  for _ in {1..30}; do
    if nc -z "$host" "$port" 2>/dev/null; then return 0; fi
    sleep 0.3
  done
  return 1
}

if [[ "$run_mode" == "background" ]]; then
  : > "$LOG_EMU"
  : > "$LOG_WEB"

  # NOTE: don't use `nohup` here. On macOS, /usr/bin/nohup is SIP-protected
  # which causes dyld to strip the entire DYLD_* environment in the child
  # process — and the emulator binary needs DYLD_LIBRARY_PATH set above so
  # it can resolve @rpath/libramulator.dylib + @rpath/libtorch*.dylib.
  # Plain `&` + `disown` keeps the env intact and still survives this
  # shell's exit, which is what we want for --background mode.
  "$EMU_BIN" $emu_server_flag --bind "$EMU_HOST:$EMU_PORT" >"$LOG_EMU" 2>&1 </dev/null &
  EMU_PID=$!
  disown $EMU_PID 2>/dev/null || true
  echo "$EMU_PID" > "$PID_FILE_EMU"
  echo "Emulator started (PID $EMU_PID, $server_mode mode) -> $EMU_HOST:$EMU_PORT (log: $LOG_EMU)"

  if ! wait_for_port "$EMU_HOST" "$EMU_PORT"; then
    echo "ERROR: emulator did not open $EMU_HOST:$EMU_PORT; see $LOG_EMU" >&2
    exit 1
  fi

  # Web GUI is a Python process — `nohup` would be fine here (no DYLD
  # dependency in the parent) but we use the same pattern for symmetry
  # and to keep the cleanup path simple.
  "$VENV_PY" "$WEBGUI" \
    --listen-host "$WEB_HOST" --listen-port "$WEB_PORT" \
    --emulator-host "$EMU_HOST" --emulator-port "$EMU_PORT" \
    --mode "$gui_mode" \
    >"$LOG_WEB" 2>&1 </dev/null &
  WEB_PID=$!
  disown $WEB_PID 2>/dev/null || true
  echo "$WEB_PID" > "$PID_FILE_WEB"
  echo "Web GUI  started (PID $WEB_PID, $gui_mode mode) -> http://$WEB_HOST:$WEB_PORT (log: $LOG_WEB)"
  echo
  echo "Open http://$WEB_HOST:$WEB_PORT in your browser."
  echo "Stop with: ./stop_online_sim.sh"
  exit 0
fi

# Foreground mode: emulator backgrounded under THIS shell, Flask in foreground.
# Ctrl+C kills both.
cleanup() {
  echo
  echo "Shutting down..."
  [[ -n "${EMU_PID:-}" ]] && kill "$EMU_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$EMU_BIN" $emu_server_flag --bind "$EMU_HOST:$EMU_PORT" &
EMU_PID=$!
echo "Emulator (PID $EMU_PID, $server_mode mode) on $EMU_HOST:$EMU_PORT"

if ! wait_for_port "$EMU_HOST" "$EMU_PORT"; then
  echo "ERROR: emulator did not open $EMU_HOST:$EMU_PORT" >&2
  exit 1
fi

echo "Web GUI starting on http://$WEB_HOST:$WEB_PORT ($gui_mode mode) ..."
echo "Press Ctrl+C to stop both services."
echo
exec "$VENV_PY" "$WEBGUI" \
  --listen-host "$WEB_HOST" --listen-port "$WEB_PORT" \
  --emulator-host "$EMU_HOST" --emulator-port "$EMU_PORT" \
  --mode "$gui_mode"
