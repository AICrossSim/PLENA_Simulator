#!/usr/bin/env bash
# Stop the PLENA online emulator + Flask Web GUI started in --background mode.

PID_FILE_EMU="/tmp/plena_emulator.pid"
PID_FILE_WEB="/tmp/plena_webgui.pid"

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
