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

# This script lives at <repo>/gui/, so PROJECT_ROOT is one level up.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

EMU_BIN="$PROJECT_ROOT/transactional_emulator/target/release/transactional_emulator"
VENV_PY="$PROJECT_ROOT/.venv/bin/python3"
WEBGUI="$SCRIPT_DIR/webgui.py"

EMU_HOST="${EMU_HOST:-127.0.0.1}"
EMU_PORT="${EMU_PORT:-7878}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5001}"   # 5000 is taken by macOS AirPlay

if [[ ! -x "$EMU_BIN" ]]; then
  echo "ERROR: transactional_emulator release binary missing at $EMU_BIN." >&2
  echo "       Build it with 'cargo build --release' from transactional_emulator/." >&2
  exit 1
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: project venv not found. Create .venv at the repository root." >&2
  exit 1
fi

# The emulator resolves plena_settings.toml at $cwd/../plena_settings.toml.
# $SCRIPT_DIR is <repo>/gui, whose parent is the repo root holding the TOML,
# so launching from here lets both the gateway and any spawned --serve
# backend (which inherits this cwd) find the settings file.
cd "$SCRIPT_DIR"

PID_FILE_EMU="/tmp/plena_emulator.pid"
PID_FILE_WEB="/tmp/plena_webgui.pid"
LOG_EMU="/tmp/plena_emulator.log"
LOG_WEB="/tmp/plena_webgui.log"

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

  nohup "$EMU_BIN" $emu_server_flag --bind "$EMU_HOST:$EMU_PORT" >"$LOG_EMU" 2>&1 </dev/null &
  EMU_PID=$!
  echo "$EMU_PID" > "$PID_FILE_EMU"
  echo "Emulator started (PID $EMU_PID, $server_mode mode) -> $EMU_HOST:$EMU_PORT (log: $LOG_EMU)"

  if ! wait_for_port "$EMU_HOST" "$EMU_PORT"; then
    echo "ERROR: emulator did not open $EMU_HOST:$EMU_PORT; see $LOG_EMU" >&2
    exit 1
  fi

  nohup "$VENV_PY" "$WEBGUI" \
    --listen-host "$WEB_HOST" --listen-port "$WEB_PORT" \
    --emulator-host "$EMU_HOST" --emulator-port "$EMU_PORT" \
    --mode "$gui_mode" \
    >"$LOG_WEB" 2>&1 </dev/null &
  WEB_PID=$!
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
