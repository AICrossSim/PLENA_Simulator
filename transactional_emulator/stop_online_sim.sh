#!/usr/bin/env bash
# Stop the PLENA online emulator + Flask Web GUI started in --background mode.
#
# Reads EMU_PORT / WEB_PORT (defaults match start_online_sim.sh: 7979 / 5002
# for ooo_arch worktree) so the pid files we look up match what start
# wrote. Override via `EMU_PORT=7878 ./stop_online_sim.sh` to target the
# sibling yw/online_emulator worktree from this checkout.

EMU_PORT="${EMU_PORT:-7979}"
WEB_PORT="${WEB_PORT:-5002}"

PID_FILE_EMU="/tmp/plena_emulator_${EMU_PORT}.pid"
PID_FILE_WEB="/tmp/plena_webgui_${WEB_PORT}.pid"

stop_one() {
  local pid_file="$1"
  local label="$2"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "$label (PID $pid) stopped."
    else
      echo "$label (PID $pid) not running."
    fi
    rm -f "$pid_file"
  else
    echo "$label: no pid file at $pid_file"
  fi
}

stop_one "$PID_FILE_WEB" "Web GUI"
stop_one "$PID_FILE_EMU" "Emulator"
