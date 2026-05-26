#!/usr/bin/env bash
# Serve the PLENA Web GUI for sharing via a public tunnel (ngrok / Cloudflare).
#
# Common hardening (both modes):
#   * Served by gunicorn (production WSGI), not the Werkzeug dev server.
#   * HTTP basic auth is REQUIRED: set PLENA_WEBGUI_PASSWORD or this refuses
#     to start.
#   * The Flask port and the emulator port bind to 127.0.0.1 only. Nothing is
#     opened to the network directly — a tunnel/reverse proxy fronts it and
#     terminates HTTPS.
#
# The GUI is the INTERACTIVE dev console: logged-in users can load files,
# execute opcode batches / ASM, reset, and read memory. The emulator runs as a
# single warm --serve backend. SECURITY: even with auth, anyone who has the
# password can run code on the simulator and read files the server can reach.
# File-load/execute paths are confined to PLENA_WEBGUI_FILE_ROOT (default: the
# repo root) to prevent reading arbitrary server files; widen/narrow it
# deliberately. Only share a link with people you trust, and use a strong,
# rotated password.
#
# Usage:
#   export PLENA_WEBGUI_PASSWORD='choose-a-strong-password'
#   ./serve_public.sh                 # interactive console
#   ./serve_public.sh --no-emulator   # gunicorn only (emulator already running)
#
# Then, in another shell, expose port $WEB_PORT with a tunnel (see below).
# Stop with Ctrl+C.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

EMU_BIN="$PROJECT_ROOT/transactional_emulator/target/release/transactional_emulator"
VENV_PY="$PROJECT_ROOT/.venv/bin/python3"

# Bind everything to localhost; the tunnel is the only thing that faces outward.
# The emulator port is ALWAYS localhost-only (never override). The web port
# defaults to localhost too, but may be set to 0.0.0.0 when running inside a
# container whose port is published to the host loopback (so the host's tunnel
# can reach it) — auth is still required in that case.
EMU_HOST="127.0.0.1"
EMU_PORT="${EMU_PORT:-7878}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5001}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"

start_emulator=1
for arg in "$@"; do
  case "$arg" in
    --no-emulator) start_emulator=0 ;;
    # Deprecated: the GUI is always the interactive console now. Accepted as a
    # no-op so existing launch commands / container Cmds keep working.
    --dev|--monitor) echo "note: $arg is deprecated and ignored (the GUI is always the interactive console)" >&2 ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "${PLENA_WEBGUI_PASSWORD:-}" ]]; then
  echo "ERROR: refusing to serve a public link without a password." >&2
  echo "       export PLENA_WEBGUI_PASSWORD='...' (and optionally PLENA_WEBGUI_USERNAME)." >&2
  exit 1
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: project venv not found at $VENV_PY." >&2
  exit 1
fi
if ! "$VENV_PY" -c "import gunicorn" 2>/dev/null; then
  echo "ERROR: gunicorn not installed. Run: uv pip install -e \".[gui]\"" >&2
  exit 1
fi

export PLENA_EMULATOR_HOST="$EMU_HOST"
export PLENA_EMULATOR_PORT="$EMU_PORT"
# PLENA_WEBGUI_PASSWORD / PLENA_WEBGUI_USERNAME are inherited by gunicorn.

# Confine load_*/execute_file paths to a directory so the console cannot read
# arbitrary server files. Default to the repo root; override by exporting
# PLENA_WEBGUI_FILE_ROOT before launching.
export PLENA_WEBGUI_FILE_ROOT="${PLENA_WEBGUI_FILE_ROOT:-$PROJECT_ROOT}"

# Emulator must run with cwd whose parent holds plena_settings.toml; $SCRIPT_DIR
# (<repo>/gui) satisfies that and is inherited by any spawned backend.
cd "$SCRIPT_DIR"

# The dev console drives a single warm shared backend so loads/executes and the
# subsequent memory reads land on the same emulator state (and it avoids a
# per-command backend cold-start under gateway).
emu_flag="--serve"

EMU_PID=""
cleanup() {
  echo; echo "Shutting down..."
  [[ -n "$EMU_PID" ]] && kill "$EMU_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$start_emulator" -eq 1 ]]; then
  if [[ ! -x "$EMU_BIN" ]]; then
    echo "ERROR: emulator binary missing at $EMU_BIN (cargo build --release)." >&2
    exit 1
  fi
  "$EMU_BIN" $emu_flag --bind "$EMU_HOST:$EMU_PORT" --quiet &
  EMU_PID=$!
  echo "Emulator (PID $EMU_PID, ${emu_flag#--}) on $EMU_HOST:$EMU_PORT"
  for _ in {1..30}; do
    nc -z "$EMU_HOST" "$EMU_PORT" 2>/dev/null && break
    sleep 0.3
  done
fi

echo
echo "!!! Serving the INTERACTIVE dev console — logged-in users can run code and"
echo "!!! load files on this server. File paths confined to: $PLENA_WEBGUI_FILE_ROOT"
echo "!!! Share only with trusted people; use a strong password."
echo "Listening (basic-auth required) on http://$WEB_HOST:$WEB_PORT"
echo "Expose it publicly with ONE of (run where the tunnel binary lives,"
echo "targeting the loopback port — if in a container, publish it to the host first):"
echo "  ngrok:       ngrok http $WEB_PORT"
echo "  cloudflared: cloudflared --config /tmp/cf_empty.yml tunnel --url http://127.0.0.1:$WEB_PORT"
echo "               (the empty --config avoids inheriting any system"
echo "                /etc/cloudflared/config.yml ingress rules: 'touch /tmp/cf_empty.yml')"
echo "The tunnel gives you an HTTPS link; viewers log in with your username/password."
echo "Press Ctrl+C to stop."
echo

exec "$VENV_PY" -m gunicorn \
  --workers "$GUNICORN_WORKERS" --threads "$GUNICORN_THREADS" \
  --bind "$WEB_HOST:$WEB_PORT" \
  wsgi:app
