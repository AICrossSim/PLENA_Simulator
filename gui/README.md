# PLENA Simulator Web GUI

A Flask front-end for the transactional emulator's **online service** mode. It
talks to the emulator over a line-delimited JSON TCP protocol, so the GUI is a
thin client — all simulation happens in the Rust emulator exactly as in batch
mode (same ISA, same `bf16` numerics, same `plena_settings.toml`).

```
gui/
├── webgui.py            # Flask app (monitor + dev views)
├── emulator_client.py   # TCP JSON client for the emulator service
├── templates/           # monitor.html, webgui.html
├── python_client_demo.py
├── start_online_sim.sh  # launch emulator backend + Flask front-end
└── stop_online_sim.sh   # stop a --background launch
```

## Prerequisites

1. **Build the emulator** (provides the `--serve` / `--gateway` service modes):

   ```bash
   cd transactional_emulator && cargo build --release
   ```

2. **Install the GUI dependency** (Flask) into the project venv:

   ```bash
   uv pip install -e ".[gui]"     # or: pip install -e ".[gui]"
   ```

The emulator binary links against the project's PyTorch/libtorch, so run
everything inside the project's dev environment (nix shell or the `dev`
Docker service) where that toolchain is available.

## Quick start

```bash
cd gui
./start_online_sim.sh            # gateway + read-only monitor GUI (default)
# open http://127.0.0.1:5001
```

Other launch modes:

```bash
./start_online_sim.sh --dev         # interactive console (load/execute/read)
./start_online_sim.sh --serve       # single shared backend instead of gateway
./start_online_sim.sh --background   # detach; logs to /tmp/plena_*.log
./stop_online_sim.sh                 # stop a --background launch
```

Override hosts/ports with env vars: `EMU_HOST`, `EMU_PORT` (default 7878),
`WEB_HOST`, `WEB_PORT` (default 5001).

## GUI modes

- **monitor** (default): read-only live view of sessions, history, server
  status, and execution progress. Issues only session-only commands, so under
  `--gateway` it never spawns a per-session backend. Safe to leave open.
- **dev**: full interactive console — load HBM/SRAM/VRAM files, execute opcode
  batches/files, assemble and run ASM snippets (via `PLENA_Compiler`), read
  memory, and render VRAM/MRAM/HBM heatmaps.

## Server topologies

- `--gateway` (default): multi-tenant. Each client session gets an isolated
  `--serve` backend subprocess (its own `EmulatorState`/HBM), spawned lazily on
  the first hardware-touching command and reaped on disconnect.
- `--serve`: single-tenant. All clients share one `EmulatorState`.

## Sharing a public link (remote server + tunnel)

To let external people view the GUI over the internet, serve it **read-only**
behind a password and a tunnel. Never expose the raw emulator port (`7878`)
publicly — the emulator protocol has no access control of its own.

`serve_public.sh` enforces a safe shape: gunicorn (production WSGI), HTTP basic
auth required, everything bound to `127.0.0.1` so only the tunnel faces outward.
It serves **monitor** (read-only) by default.

### Interactive (`--dev`) over a public link

`./serve_public.sh --dev` serves the full interactive console (load files,
execute opcodes/ASM, reset, read memory) instead of monitor. This is powerful
and risky: anyone who has the password can run code on the simulator and read
files the server process can reach. Mitigations applied automatically in
`--dev`:

- File-load/execute paths are **confined** to `PLENA_WEBGUI_FILE_ROOT` (default:
  the repo root) — attempts to read `/etc/...`, `~/.ssh`, etc. are rejected.
  Override the root deliberately if your data lives elsewhere.
- The emulator runs as a single warm `--serve` backend (no per-command
  cold start). Execute actions allow up to `PLENA_WEBGUI_EXEC_TIMEOUT` seconds
  (default 300) for long kernels.

Only share a `--dev` link with people you trust, use a strong/rotated password,
and prefer monitor mode for anything truly public.

```bash
# On the remote server (inside the project's dev env):
uv pip install -e ".[gui]"                 # flask + gunicorn
export PLENA_WEBGUI_PASSWORD='choose-a-strong-password'
export PLENA_WEBGUI_USERNAME='plena'        # optional, defaults to "plena"
cd gui
./serve_public.sh                           # emulator (gateway) + gunicorn on 127.0.0.1:5001
```

Then expose port 5001 with a tunnel (in another shell) — this is what gives you
the shareable HTTPS link:

```bash
ngrok http 5001
# or (the empty --config avoids inheriting any system /etc/cloudflared/config.yml
# ingress rules, which would otherwise 404 every request and ignore --url):
touch /tmp/cf_empty.yml
cloudflared --config /tmp/cf_empty.yml tunnel --url http://127.0.0.1:5001
```

The `https://….trycloudflare.com` URL is **random and changes every run** — if a
previously-shared link stops working with a 404, the tunnel was restarted and the
old hostname is dead; re-share the current one. Login is the browser's native
basic-auth popup (username/password), not a field on the page.

Share the tunnel's `https://…` URL plus the username/password. Viewers get the
live read-only monitor; they cannot load files, execute code, or reach the
emulator directly.

Notes:
- The tunnel terminates HTTPS, which is what makes basic-auth safe (it's
  plaintext otherwise). Don't expose port 5001 directly without TLS.
- `serve_public.sh --no-emulator` serves only the GUI if the emulator is
  already running elsewhere on `127.0.0.1:7878`.
- For a permanent deployment instead of a tunnel, front gunicorn with
  nginx/Caddy (TLS + the same basic-auth) on the server's domain.
- Basic auth is also available to the dev-server launcher: set
  `PLENA_WEBGUI_PASSWORD` before `start_online_sim.sh` and it is enforced too.

## Running the emulator service directly

The GUI is optional — the service protocol can be driven by any TCP client:

```bash
cd transactional_emulator
./target/release/transactional_emulator --serve --bind 127.0.0.1:7878 --quiet
```

See `python_client_demo.py` and `emulator_client.py` for a minimal client.
Batch mode is unchanged: omit `--serve`/`--gateway` and pass
`--opcode/--hbm/--fpsram` as before.
