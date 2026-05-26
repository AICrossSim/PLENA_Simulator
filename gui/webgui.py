#!/usr/bin/env python3

import argparse
import array
import atexit
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# The Werkzeug dev server logs every HTTP request to stderr. With the workload
# runner polling /api/job once a second that floods the terminal. Demote it to
# WARNING and silence Flask's own info logger as well.
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("flask.app").setLevel(logging.WARNING)

# Imported after the logging tweak above so werkzeug/flask loggers are already
# demoted by the time their modules initialize.
from emulator_client import EmulatorClient, parse_opcode_batch  # noqa: E402
from flask import (  # noqa: E402
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
)

# This file lives at <repo>/gui/webgui.py, so the repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "PLENA_Tools"
COMPILER_ROOT = PROJECT_ROOT / "PLENA_Compiler"
# The emulator runs batch workloads with this as its cwd, so its dump artifacts
# (vram_dump.bin, *_touches.bin, hbm_dump.bin, ...) land here.
EMU_DIR = PROJECT_ROOT / "transactional_emulator"

# Put the repo root, PLENA_Tools, and PLENA_Compiler on sys.path so the WebGUI
# can import plena_* helpers and the compiler's `assembler` package directly,
# regardless of whether the project was `pip install -e .`'d.
for import_path in (PROJECT_ROOT, TOOLS_ROOT, COMPILER_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


def resolve_allowed_file_path(raw_path: str) -> str:
    """Resolve a user-supplied path for a load_*/execute_file dev action.

    When ``PLENA_WEBGUI_FILE_ROOT`` is set (strongly recommended for any
    deployment reachable beyond localhost, e.g. behind a public tunnel), the
    resolved path must live inside that directory; relative paths are taken
    relative to it, and the absolute resolved path is returned for sending to
    the emulator. This stops the dev console's file-load actions — which would
    otherwise read any path the server process can — from exfiltrating
    arbitrary server files (``/etc/...``, ``~/.ssh``, credentials, ...).

    With the env var unset, paths pass through unchanged so local/trusted use
    is unaffected.
    """
    root = os.environ.get("PLENA_WEBGUI_FILE_ROOT")
    if not root:
        return raw_path
    if not raw_path:
        raise ValueError("path is empty")
    root_p = Path(root).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root_p / candidate
    candidate = candidate.resolve()
    if candidate != root_p and root_p not in candidate.parents:
        raise ValueError(f"path {raw_path!r} is outside the allowed directory ({root_p})")
    return str(candidate)


# ===========================================================================
# Transformer-block workload runner (dev mode)
#
# The "Transformer Block Test" panel runs the repo's existing testbench
# scripts as subprocesses. Per-run knobs are passed as PLENA_TB_* env vars
# (read by transactional_emulator/testbench/gui_params.py); hardware tile
# dimensions / precision are written into plena_settings.toml first. Each
# script compiles the layer, runs the emulator, and prints a PASSED/FAILED
# line which we surface in the live log.
# ===========================================================================

# workload key -> definition. `script` is relative to PROJECT_ROOT. `params`
# are the PLENA_TB_* knobs the script honours. `models` (if present) makes it a
# model workload whose selected key is passed as argv to the script.
WORKLOADS: dict[str, dict[str, Any]] = {
    "linear": {
        "label": "Linear projection",
        "script": "transactional_emulator/testbench/aten/linear_test.py",
        "heavy": False,
        "params": ["batch_size", "in_features", "out_features"],
    },
    "rms_norm": {
        "label": "RMSNorm",
        "script": "transactional_emulator/testbench/aten/rms_norm_test.py",
        "heavy": False,
        "params": ["batch_size", "hidden_size"],
    },
    "layer_norm": {
        "label": "LayerNorm",
        "script": "transactional_emulator/testbench/aten/layer_norm_test.py",
        "heavy": False,
        "params": ["batch_size", "hidden_size"],
    },
    "ffn": {
        "label": "FFN / MLP (SwiGLU)",
        "script": "transactional_emulator/testbench/aten/ffn_test.py",
        "heavy": False,
        "params": ["batch_size", "hidden_size", "inter_dim"],
    },
    "rope": {
        "label": "RoPE (rotary embedding)",
        "script": "transactional_emulator/testbench/aten/rope_test.py",
        "heavy": False,
        "params": ["seq_len"],
    },
    "softmax": {
        "label": "Online softmax",
        "script": "transactional_emulator/testbench/aten/fpvar_softmax_test.py",
        "heavy": False,
        "params": [],
    },
    "flash_attention": {
        "label": "Flash attention (GQA)",
        "script": "transactional_emulator/testbench/aten/flash_attention_gqa_test.py",
        "heavy": False,
        "params": ["batch_size", "seq_len"],
    },
    "ffn_model": {
        "label": "FFN — real model weights",
        "script": "transactional_emulator/testbench/models/multi_model_ffn_test.py",
        "heavy": True,
        "params": ["batch_size"],
        "models": ["smollm2-135m", "clm60m", "smolvlm2"],
    },
    "decoder_model": {
        "label": "Decoder layer — real model weights",
        "script": "transactional_emulator/testbench/models/multi_model_decoder_test.py",
        "heavy": True,
        "params": ["seq_len"],
        "models": ["smollm2-135m", "llada-8b"],
    },
}

# Optional per-run integer knobs that map to PLENA_TB_* env vars. mlen/blen/hlen
# additionally drive the plena_settings.toml rewrite below.
_RUN_INT_KNOBS = (
    "mlen", "blen", "hlen", "vlen",
    "batch_size", "seq_len", "hidden_size", "inter_dim",
    "in_features", "out_features", "layer_idx",
)


def _settings_path() -> Path:
    return PROJECT_ROOT / "plena_settings.toml"


def write_hw_config(params: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the [TRANSACTIONAL] section of plena_settings.toml.

    Writes the tile dimensions (and optional precision) the next workload run
    will use. The emulator asserts ``MLEN == HLEN * BROADCAST_AMOUNT``, so
    BROADCAST_AMOUNT is recomputed from mlen/hlen and the divisibility is
    validated up front. Returns the dict of values actually applied.
    """
    import tomlkit  # available in the project env

    def _as_int(key: str) -> int | None:
        raw = params.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        return int(str(raw).strip())

    mlen = _as_int("mlen")
    blen = _as_int("blen")
    hlen = _as_int("hlen")
    vlen = _as_int("vlen")

    # Tile dimensions must be powers of two (hardware constraint).
    def _is_pow2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0

    for dim_name, dim_val in (("mlen", mlen), ("blen", blen), ("hlen", hlen), ("vlen", vlen)):
        if dim_val is not None and not _is_pow2(dim_val):
            raise ValueError(f"{dim_name} ({dim_val}) must be a power of two")

    if mlen is not None:
        if mlen <= 0:
            raise ValueError("mlen must be positive")
        if blen is not None and mlen % blen != 0:
            raise ValueError(f"mlen ({mlen}) must be a multiple of blen ({blen})")
        if hlen is not None and mlen % hlen != 0:
            raise ValueError(f"mlen ({mlen}) must be a multiple of hlen ({hlen})")

    path = _settings_path()
    with open(path) as fh:
        doc = tomlkit.load(fh)

    cfg = doc["TRANSACTIONAL"]["CONFIG"]
    applied: dict[str, Any] = {}

    if mlen is not None:
        cfg["MLEN"]["value"] = mlen
        # VLEN must equal mlen for the emulator's row-alignment check.
        cfg["VLEN"]["value"] = vlen if vlen is not None else mlen
        applied["mlen"] = mlen
        applied["vlen"] = cfg["VLEN"]["value"]
    elif vlen is not None:
        cfg["VLEN"]["value"] = vlen
        applied["vlen"] = vlen
    if blen is not None:
        cfg["BLEN"]["value"] = blen
        applied["blen"] = blen
    if hlen is not None:
        cfg["HLEN"]["value"] = hlen
        applied["hlen"] = hlen
        base_mlen = mlen if mlen is not None else int(cfg["MLEN"]["value"])
        if base_mlen % hlen != 0:
            raise ValueError(f"mlen ({base_mlen}) must be a multiple of hlen ({hlen})")
        cfg["BROADCAST_AMOUNT"]["value"] = base_mlen // hlen
        applied["broadcast_amount"] = base_mlen // hlen

    # Optional precision (advanced). Only touched when explicitly provided.
    prec = doc["TRANSACTIONAL"]["PRECISION"]
    sram_exp = _as_int("sram_exp")
    sram_man = _as_int("sram_man")
    if sram_exp is not None or sram_man is not None:
        for tname in ("MATRIX_SRAM_TYPE", "VECTOR_SRAM_TYPE"):
            dt = prec[tname]["DATA_TYPE"]
            if sram_exp is not None:
                dt["exponent"] = sram_exp
            if sram_man is not None:
                dt["mantissa"] = sram_man
        applied["sram_precision"] = f"e{sram_exp}m{sram_man}"
    # HBM MX element precision is per-tensor-class: W (weight), KV (key/value
    # cache, both matrix- and vector-side), A (activation). Block size and the
    # scale bitwidth are shared across all four MX types.
    mx_block = _as_int("hbm_mx_block")
    mx_scale_exp = _as_int("hbm_mx_scale_exp")
    mx_scale_man = _as_int("hbm_mx_scale_man")
    # group -> the TOML MX types it drives
    mx_groups = {
        "w": ("HBM_M_WEIGHT_TYPE",),
        "kv": ("HBM_M_KV_TYPE", "HBM_V_KV_TYPE"),
        "a": ("HBM_V_ACT_TYPE",),
    }
    mx_elem = {
        g: (_as_int(f"hbm_mx_{g}_exp"), _as_int(f"hbm_mx_{g}_man")) for g in mx_groups
    }
    elem_touched = any(e is not None or m is not None for e, m in mx_elem.values())
    if elem_touched or mx_block is not None or mx_scale_exp is not None or mx_scale_man is not None:
        for g, tnames in mx_groups.items():
            e_exp, e_man = mx_elem[g]
            for tname in tnames:
                t = prec[tname]
                if mx_block is not None:
                    t["block"] = mx_block
                if e_exp is not None:
                    t["ELEM"]["exponent"] = e_exp
                if e_man is not None:
                    t["ELEM"]["mantissa"] = e_man
                if mx_scale_exp is not None:
                    t["SCALE"]["exponent"] = mx_scale_exp
                if mx_scale_man is not None:
                    t["SCALE"]["mantissa"] = mx_scale_man
        applied["hbm_mx"] = {
            "w": mx_elem["w"], "kv": mx_elem["kv"], "a": mx_elem["a"],
            "block": mx_block, "scale_exp": mx_scale_exp, "scale_man": mx_scale_man,
        }

    int_width = _as_int("hbm_int_width")
    if int_width is not None:
        prec["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"] = int_width
        applied["hbm_int_width"] = int_width

    # Atomic write so a concurrent emulator read never sees a half-written file.
    tmp = path.with_suffix(".toml.tmp")
    with open(tmp, "w") as fh:
        tomlkit.dump(doc, fh)
    os.replace(tmp, path)
    return applied


def read_hw_config() -> dict[str, Any]:
    """Read the tile dims + key precisions from plena_settings.toml for prefill.

    This is the source of truth for what the next workload run will use (the
    live emulator's get_config returns runtime register state, not these). Falls
    back to the transactional defaults if the file can't be parsed.
    """
    defaults = {
        "mlen": 64, "blen": 4, "hlen": 16, "vlen": 64, "broadcast_amount": 4,
        "sram_exp": 8, "sram_man": 7,
        "hbm_mx_w_exp": 4, "hbm_mx_w_man": 3,
        "hbm_mx_kv_exp": 4, "hbm_mx_kv_man": 3,
        "hbm_mx_a_exp": 4, "hbm_mx_a_man": 3,
        "hbm_mx_block": 8, "hbm_mx_scale_exp": 8, "hbm_mx_scale_man": 0,
        "hbm_int_width": 32,
    }
    try:
        import tomlkit

        doc = tomlkit.load(open(_settings_path()))
        cfg = doc["TRANSACTIONAL"]["CONFIG"]
        prec = doc["TRANSACTIONAL"]["PRECISION"]
        sd = prec["MATRIX_SRAM_TYPE"]["DATA_TYPE"]
        w_elem = prec["HBM_M_WEIGHT_TYPE"]["ELEM"]
        kv_elem = prec["HBM_M_KV_TYPE"]["ELEM"]
        a_elem = prec["HBM_V_ACT_TYPE"]["ELEM"]
        scale = prec["HBM_M_WEIGHT_TYPE"]["SCALE"]
        return {
            "mlen": int(cfg["MLEN"]["value"]),
            "blen": int(cfg["BLEN"]["value"]),
            "hlen": int(cfg["HLEN"]["value"]),
            "vlen": int(cfg["VLEN"]["value"]),
            "broadcast_amount": int(cfg["BROADCAST_AMOUNT"]["value"]),
            "sram_exp": int(sd["exponent"]),
            "sram_man": int(sd["mantissa"]),
            "hbm_mx_w_exp": int(w_elem["exponent"]),
            "hbm_mx_w_man": int(w_elem["mantissa"]),
            "hbm_mx_kv_exp": int(kv_elem["exponent"]),
            "hbm_mx_kv_man": int(kv_elem["mantissa"]),
            "hbm_mx_a_exp": int(a_elem["exponent"]),
            "hbm_mx_a_man": int(a_elem["mantissa"]),
            "hbm_mx_block": int(prec["HBM_M_WEIGHT_TYPE"]["block"]),
            "hbm_mx_scale_exp": int(scale["exponent"]),
            "hbm_mx_scale_man": int(scale["mantissa"]),
            "hbm_int_width": int(prec["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"]),
        }
    except Exception:
        return defaults


# The batch-mode emulator binary dumps full register / SRAM / VRAM / HBM
# contents to stdout (main.rs ~2506-2567), which buries the actual test result
# under thousands of array lines. We filter those blocks out of the run log
# while keeping the summary lines (HBM Statistics, Latency) and the PASS/FAIL.
_RE_REG_DUMP = re.compile(r"^\s*(gp\d+|scale)\s*=")
_RE_CONTENTS_HDR = re.compile(r"(Vector|Matrix|INT|FP|VRAM|HBM) SRAM Contents:|Contents:\s*$")
_RE_ARRAYISH = re.compile(r"^[\s\[\]\d.,eExXa-fA-F+\-]*(\.\.\.)?[\s\[\]\d.,eExXa-fA-F+\-]*$")


class _NoiseFilter:
    """Stateful per-run filter that drops the emulator's bulk memory dumps."""

    def __init__(self) -> None:
        self._in_dump = False

    def keep(self, line: str) -> bool:
        s = line.strip()
        if self._in_dump:
            if s.startswith("Tensor[["):
                self._in_dump = False
                return False
            if s == "" or _RE_ARRAYISH.match(s):
                return False
            # A non-array line ends the dump; re-evaluate it normally.
            self._in_dump = False
        if "Contents:" in s and _RE_CONTENTS_HDR.search(s):
            self._in_dump = True
            return False
        if _RE_REG_DUMP.match(s):
            return False
        return True


class _Job:
    """A single background workload run with an appended-line log buffer."""

    def __init__(self, job_id: str, workload: str, model: str | None, cmd: list[str]):
        self.id = job_id
        self.workload = workload
        self.model = model
        self.cmd = cmd
        self.lines: list[str] = []
        self.status = "running"  # running | passed | failed | error | stopped
        self.returncode: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def append(self, text: str) -> None:
        with self.lock:
            self.lines.append(text)

    def snapshot(self, offset: int) -> dict[str, Any]:
        with self.lock:
            total = len(self.lines)
            new = self.lines[offset:] if offset < total else []
            return {
                "id": self.id,
                "workload": self.workload,
                "model": self.model,
                "status": self.status,
                "returncode": self.returncode,
                "running": self.status == "running",
                "elapsed": round((self.ended or time.time()) - self.started, 1),
                "lines": new,
                "next_offset": total,
            }


class WorkloadRunner:
    """Single-flight runner: at most one workload subprocess at a time.

    The emulator state and plena_settings.toml are process-global, so running
    two workloads concurrently would corrupt each other.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._current: str | None = None

    def _python(self) -> str:
        return shutil.which("python3") or sys.executable

    def start(self, workload: str, model: str | None, knobs: dict[str, str]) -> _Job:
        spec = WORKLOADS[workload]
        with self._lock:
            cur = self._jobs.get(self._current) if self._current else None
            if cur is not None and cur.status == "running":
                raise RuntimeError(
                    f"a workload is already running ({cur.workload}); stop it or wait for it to finish"
                )
            script = (PROJECT_ROOT / spec["script"]).resolve()
            if PROJECT_ROOT not in script.parents:
                raise ValueError("workload script escapes the project root")
            cmd = [self._python(), str(script)]
            if "models" in spec:
                if model not in spec["models"]:
                    raise ValueError(f"unknown model {model!r} for workload {workload!r}")
                cmd.append(model)
            job_id = uuid.uuid4().hex[:12]
            job = _Job(job_id, workload, model, cmd)
            self._jobs[job_id] = job
            self._current = job_id

        env = dict(os.environ)
        # Make the testbench packages importable regardless of editable installs.
        extra = f"{PROJECT_ROOT}:{TOOLS_ROOT}:{COMPILER_ROOT}"
        env["PYTHONPATH"] = f"{extra}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["RUST_BACKTRACE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        for key, val in knobs.items():
            if val is not None and str(val).strip() != "":
                env[f"PLENA_TB_{key.upper()}"] = str(val).strip()

        job.append(f"$ PLENA_TB_* = {{ {', '.join(f'{k}={v}' for k, v in sorted(knobs.items()) if str(v).strip())} }}")
        job.append(f"$ {' '.join(job.cmd)}")
        job.append("")

        def _run() -> None:
            try:
                proc = subprocess.Popen(
                    job.cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                job.proc = proc
                assert proc.stdout is not None
                noise = _NoiseFilter()
                for line in proc.stdout:
                    stripped = line.rstrip("\n")
                    if noise.keep(stripped):
                        job.append(stripped)
                proc.wait()
                job.returncode = proc.returncode
                joined = "\n".join(job.lines)
                if proc.returncode == 0 and "PASSED" in joined:
                    job.status = "passed"
                elif job.status != "stopped":
                    job.status = "failed" if proc.returncode == 0 else "error"
            except Exception as exc:  # noqa: BLE001
                job.append(f"[runner error] {exc}")
                job.status = "error"
            finally:
                job.ended = time.time()

        threading.Thread(target=_run, name=f"workload-{job_id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self) -> bool:
        with self._lock:
            job = self._jobs.get(self._current) if self._current else None
        if job is None or job.status != "running":
            return False
        job.status = "stopped"
        proc = job.proc
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except (ProcessLookupError, PermissionError):
                pass
        job.append("[stopped by user]")
        return True


def create_app(
    default_emulator_host: str = "127.0.0.1",
    default_emulator_port: int = 7878,
) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "PLENA_WEBGUI_SECRET_KEY",
        "plena-transactional-emulator-webgui-dev",
    )
    app.config["DEFAULT_EMULATOR_HOST"] = default_emulator_host
    app.config["DEFAULT_EMULATOR_PORT"] = default_emulator_port
    # Re-read templates from disk when they change so edits to webgui.html are
    # picked up without restarting gunicorn (the Python module still requires a
    # restart). Cheap mtime check per render; fine for this low-traffic GUI.
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Optional HTTP basic auth, enforced only when PLENA_WEBGUI_PASSWORD is set.
    # This is a guard for exposing the GUI beyond localhost (e.g. behind a
    # public tunnel): with no password set, behavior is unchanged for local
    # use. Use constant-time comparison to avoid leaking the secret via timing.
    # Note: basic auth is plaintext over HTTP, so only meaningful behind TLS
    # (the tunnel/reverse proxy must terminate HTTPS).
    auth_user = os.environ.get("PLENA_WEBGUI_USERNAME", "plena")
    auth_password = os.environ.get("PLENA_WEBGUI_PASSWORD")
    if auth_password:

        @app.before_request
        def _require_basic_auth() -> Response | None:
            creds = request.authorization
            ok = (
                creds is not None
                and creds.type == "basic"
                and hmac.compare_digest(creds.username or "", auth_user)
                and hmac.compare_digest(creds.password or "", auth_password)
            )
            if not ok:
                return Response(
                    "Authentication required.",
                    401,
                    {"WWW-Authenticate": 'Basic realm="PLENA WebGUI"'},
                )
            return None

    # One persistent labeled EmulatorClient per (host, port), shared across
    # every request from the dev console. Crucially, this means a load_*/
    # execute_* in one request and the memory reads (heatmaps + content) in
    # the next land on the SAME gateway session — i.e. the same backend and
    # the same memory. A fresh ephemeral client per request would instead
    # spawn a new (empty) backend for every command, so the heatmaps would
    # always show zeros. Keyed because the user may flip endpoints via the
    # form. The long read timeout covers slow execute_* kernels.
    exec_timeout = float(os.environ.get("PLENA_WEBGUI_EXEC_TIMEOUT", "300"))
    dev_clients: dict[tuple[str, int], EmulatorClient] = {}
    dev_clients_lock = threading.Lock()

    def get_persistent_client(host: str, port: int) -> EmulatorClient:
        key = (host, port)
        with dev_clients_lock:
            client = dev_clients.get(key)
            if client is None:
                client = EmulatorClient(
                    host=host,
                    port=port,
                    timeout=exec_timeout,
                    label="webgui-dev",
                    auto_label=True,
                )
                dev_clients[key] = client
            # If the persistent socket was dropped (server restart, network
            # blip, fallback after a transient error), reopen it so the next
            # request is again labeled+persistent and lands on the same
            # session/backend.
            if client._sock is None:
                try:
                    client.connect()
                except OSError:
                    # Leave _sock as None; individual send_command calls fall
                    # back to ephemeral and surface their own errors.
                    pass
            return client

    def _close_dev_clients() -> None:
        with dev_clients_lock:
            for client in dev_clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            dev_clients.clear()

    atexit.register(_close_dev_clients)

    # In-memory rolling log of emulator command exchanges (request +
    # response/error), newest last. Server-side so it survives the full-page
    # reload each POST triggers and never bloats the session cookie. Shared
    # across browser tabs, which suits a single-user dev console.
    CONSOLE_LOG_LIMIT = 50
    console_log: list[dict[str, Any]] = []
    console_log_lock = threading.Lock()

    def append_console_log(action: str, req: Any, resp: Any = None, error: str | None = None) -> None:
        with console_log_lock:
            console_log.append(
                {
                    "ts": time.strftime("%H:%M:%S"),
                    "action": action,
                    "request": req,
                    "response": resp,
                    "error": error,
                }
            )
            if len(console_log) > CONSOLE_LOG_LIMIT:
                del console_log[: len(console_log) - CONSOLE_LOG_LIMIT]

    def render_console_log() -> str:
        with console_log_lock:
            entries = list(console_log)
        if not entries:
            return "// no commands run yet"
        blocks: list[str] = []
        for entry in entries:
            head = f"[{entry['ts']}] {entry['action']}"
            req_json = json.dumps(entry["request"], indent=2, sort_keys=True)
            block = f"{head}\n>>> request\n{req_json}"
            if entry.get("error"):
                block += f"\n<<< error\n{entry['error']}"
            else:
                resp_json = json.dumps(entry["response"], indent=2, sort_keys=True)
                block += f"\n<<< response\n{resp_json}"
            blocks.append(block)
        return "\n\n".join(blocks)

    # Persisted form values for the memory views. The *_heatmap_* keys drive
    # the raw text dumps. The touch heatmaps no longer take addr/row controls —
    # they render the whole write-touch map read from the dump files.
    HEATMAP_DEFAULTS = {
        "vram_content_addr": "0",
        "vram_content_rows": "8",
        "hbm_content_addr": "0",
        "hbm_content_len": "256",
    }

    def connection_settings() -> tuple[str, int]:
        host = session.get("emulator_host", app.config["DEFAULT_EMULATOR_HOST"])
        port = session.get("emulator_port", app.config["DEFAULT_EMULATOR_PORT"])
        return str(host), int(port)

    def update_connection_from_form() -> tuple[str, int]:
        current_host, current_port = connection_settings()
        host = request.form.get("emulator_host", current_host).strip() or current_host
        try:
            port = int(request.form.get("emulator_port", str(current_port)))
        except ValueError:
            port = current_port

        session["emulator_host"] = host
        session["emulator_port"] = port
        return host, port

    def try_fetch_snapshot(client: EmulatorClient, cmd: str) -> tuple[Any | None, str | None]:
        try:
            return client.send_command(cmd), None
        except Exception as exc:
            return None, str(exc)

    def compile_asm_snippet(asm_text: str) -> list[str]:
        asm_source = asm_text.strip()
        if not asm_source:
            raise ValueError("ASM snippet is empty")

        isa_path = COMPILER_ROOT / "doc" / "operation.svh"
        config_path = COMPILER_ROOT / "doc" / "configuration.svh"
        if not isa_path.exists() or not config_path.exists():
            raise RuntimeError("Compiler assets are missing. Ensure the `PLENA_Compiler` submodule is initialized.")

        try:
            from assembler.assembly_to_binary import AssemblyToBinary
        except Exception as exc:
            raise RuntimeError(f"Failed to import PLENA compiler assembler: {exc}") from exc

        with tempfile.TemporaryDirectory(prefix="plena_webgui_asm_") as tempdir:
            tempdir_path = Path(tempdir)
            asm_path = tempdir_path / "snippet.asm"
            mem_path = tempdir_path / "snippet.mem"
            asm_path.write_text(asm_source + "\n", encoding="utf-8")

            assembler = AssemblyToBinary(str(isa_path), str(config_path))
            binary_instructions = assembler.generate_binary(str(asm_path), str(mem_path))

        return [f"0x{instruction:08X}" for instruction in binary_instructions]

    def parse_int_literal(text: str, default: int) -> int:
        try:
            return int(str(text).strip(), 0)
        except (TypeError, ValueError):
            return default

    def clamp_int(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    def heatmap_form_values() -> dict[str, str]:
        values = {}
        for key, default in HEATMAP_DEFAULTS.items():
            values[key] = str(session.get(key, default))
        return values

    def stored_form_value(key: str, default: str = "") -> str:
        return request.form.get(key, session.get(key, default))

    def update_heatmap_form_values() -> dict[str, str]:
        values = heatmap_form_values()
        for key in HEATMAP_DEFAULTS:
            if key in request.form:
                raw = request.form.get(key, values[key]).strip()
                values[key] = raw or values[key]
                session[key] = values[key]
        return values

    # --- write-touch heatmaps -------------------------------------------------
    # The heatmaps colour each hardware write-unit (a VRAM row, an MRAM tile, or
    # a 64-byte HBM chunk) by how many times the last workload wrote it. The
    # counts come from the emulator's *_touches.bin dump files (one little-endian
    # u32 per unit), NOT from the live session — touch counts are a property of
    # the batch run, so reading the dump files is the only way to surface them.
    def _read_touches(path: Path) -> "array.array[int]":
        """Load a *_touches.bin file as a C-backed u32 array (fast for large
        VRAM maps, which can be tens of MB)."""
        counts = array.array("I")
        try:
            data = path.read_bytes()
        except OSError:
            return counts
        usable = (len(data) // 4) * 4
        if usable:
            counts.frombytes(data[:usable])
        return counts

    def _build_touch_heatmap(
        counts: "array.array[int]",
        unit_stride: int,
        unit_label: str,
        mem: str,
        cols: int = 32,
        max_cells: int = 8192,
    ) -> dict[str, Any]:
        """Lay a per-unit touch-count array out as a wrapped grid.

        Each cell is one write-unit, coloured by its write count. Only the first
        ``max_cells`` units are rendered (a large memory is mostly untouched and
        the used region sits at low addresses); trailing untouched units within
        that window are trimmed. Aggregate stats are computed over ALL units via
        C-level array ops so a multi-MB map stays cheap to render.
        """
        total_units = len(counts)
        total_writes = sum(counts) if total_units else 0
        max_value = max(counts) if total_units else 0
        touched_units = (total_units - counts.count(0)) if total_units else 0

        window = counts[:max_cells]
        last_touched = 0
        for i, c in enumerate(window):
            if c:
                last_touched = i + 1
        shown = min(max(last_touched, cols), len(window)) if window else 0
        rows = (shown + cols - 1) // cols if shown else 0

        values: list[list[int]] = []
        addrs: list[list[int]] = []
        in_range: list[list[bool]] = []
        for r in range(rows):
            vrow: list[int] = []
            arow: list[int] = []
            irow: list[bool] = []
            for c in range(cols):
                i = r * cols + c
                real = i < shown
                irow.append(real)
                vrow.append(window[i] if real else 0)
                arow.append(i * unit_stride if real else 0)
            values.append(vrow)
            addrs.append(arow)
            in_range.append(irow)

        return {
            "kind": "touch",
            "mem": mem,
            "unit_label": unit_label,
            "unit_stride": unit_stride,
            "rows": rows,
            "cols": cols,
            "values": values,
            "addrs": addrs,
            "in_range": in_range,
            "total_units": total_units,
            "touched_units": touched_units,
            "total_writes": total_writes,
            "max_value": max_value,
        }

    def fetch_vram_touches(
        config_data: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        counts = _read_touches(EMU_DIR / "vram_touches.bin")
        if not counts:
            return None, "No VRAM touch map yet — run a workload first."
        stride = int((config_data or {}).get("vlen") or 0) or 1
        return _build_touch_heatmap(counts, stride, "row", "vram"), None

    def fetch_mram_touches(
        config_data: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        counts = _read_touches(EMU_DIR / "mram_touches.bin")
        if not counts:
            return None, "No MRAM touch map yet — run a workload first."
        tile_side = int((config_data or {}).get("mlen") or 0) or 1
        return _build_touch_heatmap(counts, tile_side * tile_side, "tile", "mram"), None

    def fetch_hbm_touches() -> tuple[dict[str, Any] | None, str | None]:
        counts = _read_touches(EMU_DIR / "hbm_touches.bin")
        if not counts:
            return None, "No HBM touch map yet — run a workload first."
        return _build_touch_heatmap(counts, 64, "chunk", "hbm"), None

    def _fmt_value(value: float) -> str:
        return str(value) if float(value).is_integer() else f"{value:.6g}"

    def fetch_vram_content(
        client: EmulatorClient,
        config_data: dict[str, Any] | None,
        form_values: dict[str, str],
    ) -> tuple[str | None, str | None]:
        """Raw VRAM contents as text: one aligned row of values per line."""
        if config_data is None:
            return None, "Configuration unavailable; cannot determine VRAM row stride."
        row_stride = int(config_data.get("vlen") or 0)
        if row_stride <= 0:
            return None, "Invalid VLEN in configuration; cannot read VRAM content."

        requested_addr = parse_int_literal(form_values["vram_content_addr"], 0)
        rows = clamp_int(parse_int_literal(form_values["vram_content_rows"], 8), 1, 64)
        start_addr = (requested_addr // row_stride) * row_stride

        lines: list[str] = []
        data_type = ""
        for row_idx in range(rows):
            addr = start_addr + row_idx * row_stride
            sample = client.send_command("read_vram", addr=addr)
            if not data_type:
                data_type = str(sample.get("data_type", ""))
            values = [_fmt_value(float(v)) for v in sample.get("values", [])]
            lines.append(f"0x{addr:08X} [{addr:>10}]  " + "  ".join(values))

        header = (
            f"VRAM  start=0x{start_addr:08X}  rows={rows}  "
            f"stride={row_stride}  dtype={data_type or 'unknown'}"
        )
        return header + "\n" + ("-" * len(header)) + "\n" + "\n".join(lines), None

    def fetch_hbm_content(
        form_values: dict[str, str],
    ) -> tuple[str | None, str | None]:
        """HBM hex dump (16 bytes + ASCII per line), read from the run's
        hbm_dump.bin.

        The live session's HBM is empty after a batch workload (HBM isn't loaded
        back into the session), so the bytes come from the dump file the batch
        run wrote — that is the actual HBM content the workload produced.
        """
        path = EMU_DIR / "hbm_dump.bin"
        if not path.exists():
            return None, "No HBM dump yet — run a workload first."
        start_addr = parse_int_literal(form_values["hbm_content_addr"], 0)
        length = clamp_int(parse_int_literal(form_values["hbm_content_len"], 256), 1, 4096)
        try:
            dumped = path.stat().st_size
            with open(path, "rb") as fh:
                fh.seek(max(0, start_addr))
                raw_bytes = fh.read(length)
        except OSError as exc:
            return None, f"Could not read HBM dump: {exc}"

        lines: list[str] = []
        for offset in range(0, len(raw_bytes), 16):
            chunk = raw_bytes[offset : offset + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            hex_part = f"{hex_part:<47}"  # pad to 16 byte columns
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"0x{start_addr + offset:08X}  {hex_part}  |{ascii_part}|")
        if not lines:
            lines.append("(start address is past the end of the dumped region)")

        header = f"HBM  start=0x{start_addr:08X}  bytes={len(raw_bytes)}  (of {dumped} dumped)"
        return header + "\n" + ("-" * len(header)) + "\n" + "\n".join(lines), None

    def read_run_asm() -> tuple[str | None, str | None]:
        """The assembly the most recent workload compiled and executed.

        Testbench scripts write `generated_asm_code.asm` into their build dir
        before invoking the emulator; the newest one is the last run's program.
        """
        candidates = list(EMU_DIR.glob("testbench/**/generated_asm_code.asm"))
        if not candidates:
            return None, "No assembly yet — run a workload first."
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            text = newest.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"Could not read assembly: {exc}"
        rel = newest.relative_to(PROJECT_ROOT)
        return f"; {rel}\n; {text.count(chr(10)) + 1} lines\n\n{text}", None

    def fetch_server_status(client: EmulatorClient) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("get_server_status"), None
        except Exception as exc:
            return None, str(exc)

    # View-only POST actions: they only update persisted form values for the
    # memory views (the data itself is re-read on every render below), so they
    # are not real emulator commands and never go into the console log.
    VIEW_ONLY_ACTIONS = {
        "refresh_vram_heatmap",
        "refresh_mram_heatmap",
        "refresh_hbm_heatmap",
        "refresh_vram_content",
        "refresh_hbm_content",
    }

    def _handle_post() -> tuple[str, int, str | None, bool, str | None, dict[str, str]]:
        """Run an action-tagged form POST on the shared persistent session.

        Used by both the main page and the manual console: the command lands on
        the one persistent session (so loads/executes and later memory reads
        share a backend) and is recorded in the console log. Returns the
        resolved endpoint, the action outcome, and the current memory-view form
        values.
        """
        host, port = connection_settings()
        last_action: str | None = None
        last_ok = False
        last_error: str | None = None
        hv = heatmap_form_values()

        if request.method == "POST":
            host, port = update_connection_from_form()
            hv = update_heatmap_form_values()
            if "asm_snippet" in request.form:
                session["asm_snippet"] = request.form.get("asm_snippet", "")
            client = get_persistent_client(host, port)
            action = request.form.get("action", "").strip() or "ping"
            last_action = action

            req: Any = None
            resp: Any = None
            err: str | None = None
            try:
                if action == "ping":
                    req = {"cmd": "ping"}
                    resp = client.send_command("ping")
                elif action == "reset":
                    req = {"cmd": "reset"}
                    resp = client.send_command("reset")
                elif action == "get_config":
                    req = {"cmd": "get_config"}
                    resp = client.send_command("get_config")
                elif action == "get_state":
                    req = {"cmd": "get_state"}
                    resp = client.send_command("get_state")
                elif action == "load_hbm":
                    path = resolve_allowed_file_path(request.form.get("hbm_path", "").strip())
                    req = {"cmd": "load_hbm_file", "path": path}
                    resp = client.send_command("load_hbm_file", path=path)
                elif action == "load_fpsram":
                    path = resolve_allowed_file_path(request.form.get("fpsram_path", "").strip())
                    req = {"cmd": "load_fp_sram_file", "path": path}
                    resp = client.send_command("load_fp_sram_file", path=path)
                elif action == "load_intsram":
                    path = resolve_allowed_file_path(request.form.get("intsram_path", "").strip())
                    req = {"cmd": "load_int_sram_file", "path": path}
                    resp = client.send_command("load_int_sram_file", path=path)
                elif action == "load_vram":
                    path = resolve_allowed_file_path(request.form.get("vram_path", "").strip())
                    req = {"cmd": "load_vram_file", "path": path}
                    resp = client.send_command("load_vram_file", path=path)
                elif action == "execute_file":
                    path = resolve_allowed_file_path(request.form.get("opcode_file", "").strip())
                    req = {"cmd": "execute_file", "path": path}
                    resp = client.send_command("execute_file", path=path)
                elif action == "execute_batch":
                    opcode_text = request.form.get("opcode_batch", "")
                    opcodes = parse_opcode_batch(opcode_text)
                    req = {"cmd": "execute_batch", "opcodes": opcodes}
                    resp = client.send_command("execute_batch", opcodes=opcodes)
                elif action == "execute_asm":
                    asm_snippet = request.form.get("asm_snippet", "")
                    opcodes = compile_asm_snippet(asm_snippet)
                    req = {
                        "cmd": "execute_batch",
                        "source": "asm_snippet",
                        "asm": asm_snippet,
                        "opcodes": opcodes,
                    }
                    resp = client.send_command("execute_batch", opcodes=opcodes)
                elif action == "read_memory":
                    memory_space = request.form.get("memory_space", "vram")
                    addr = int(request.form.get("memory_addr", "0"), 0)
                    length = int(request.form.get("memory_len", "64"), 0)
                    if memory_space == "vram":
                        req = {"cmd": "read_vram", "addr": addr}
                        resp = client.send_command("read_vram", addr=addr)
                    elif memory_space == "mram":
                        req = {"cmd": "read_mram", "addr": addr}
                        resp = client.send_command("read_mram", addr=addr)
                    else:
                        req = {"cmd": "read_hbm", "addr": addr, "len": length}
                        resp = client.send_command("read_hbm", addr=addr, len=length)
                elif action in VIEW_ONLY_ACTIONS:
                    # The new form values were already persisted by
                    # update_heatmap_form_values(); the render below re-reads.
                    pass
                else:
                    err = f"Unknown action: {action}"
            except Exception as exc:
                err = str(exc)

            last_error = err
            last_ok = err is None and resp is not None
            if action not in VIEW_ONLY_ACTIONS:
                append_console_log(action, req, resp, err)

        return host, port, last_action, last_ok, last_error, hv

    @app.route("/", methods=["GET", "POST"])
    @app.route("/dev", methods=["GET", "POST"])
    def index():
        host, port, last_action, last_ok, last_error, current_heatmap_form_values = _handle_post()

        # All memory reads below go through the same persistent session as the
        # action just handled, so the views reflect real memory rather than the
        # zeros a fresh ephemeral session would report.
        client = get_persistent_client(host, port)

        config_data, config_error = try_fetch_snapshot(client, "get_config")
        state_data, state_error = try_fetch_snapshot(client, "get_state")
        status_data, _ = fetch_server_status(client)
        cfg = config_data if isinstance(config_data, dict) else None

        # Write-touch heatmaps read the batch run's *_touches.bin dump files.
        try:
            vram_heatmap_data, vram_heatmap_error = fetch_vram_touches(cfg)
        except Exception as exc:
            vram_heatmap_data, vram_heatmap_error = None, str(exc)
        try:
            mram_heatmap_data, mram_heatmap_error = fetch_mram_touches(cfg)
        except Exception as exc:
            mram_heatmap_data, mram_heatmap_error = None, str(exc)
        try:
            hbm_heatmap_data, hbm_heatmap_error = fetch_hbm_touches()
        except Exception as exc:
            hbm_heatmap_data, hbm_heatmap_error = None, str(exc)

        # VRAM content still comes from the live session (the run's vram_dump.bin
        # is loaded into it); HBM content comes from the run's hbm_dump.bin file.
        vram_content_text = None
        vram_content_error = None
        hbm_content_text = None
        hbm_content_error = None
        try:
            vram_content_text, vram_content_error = fetch_vram_content(
                client, cfg, current_heatmap_form_values
            )
        except Exception as exc:
            vram_content_error = str(exc)
        if vram_content_error and config_error and config_data is None:
            vram_content_error = config_error
        try:
            hbm_content_text, hbm_content_error = fetch_hbm_content(current_heatmap_form_values)
        except Exception as exc:
            hbm_content_error = str(exc)

        run_asm_text, run_asm_error = read_run_asm()

        return render_template(
            "webgui.html",
            emulator_host=host,
            emulator_port=port,
            last_action=last_action,
            last_ok=last_ok,
            last_error=last_error,
            config_json=json.dumps(config_data, indent=2, sort_keys=True) if config_data is not None else "",
            config_error=config_error,
            config_data=config_data if isinstance(config_data, dict) else None,
            workloads=WORKLOADS,
            hw_config=read_hw_config(),
            state_data=state_data,
            state_json=json.dumps(state_data, indent=2, sort_keys=True) if state_data is not None else "",
            state_error=state_error,
            vram_heatmap_data=vram_heatmap_data,
            vram_heatmap_error=vram_heatmap_error,
            mram_heatmap_data=mram_heatmap_data,
            mram_heatmap_error=mram_heatmap_error,
            hbm_heatmap_data=hbm_heatmap_data,
            hbm_heatmap_error=hbm_heatmap_error,
            vram_content_text=vram_content_text,
            vram_content_error=vram_content_error,
            hbm_content_text=hbm_content_text,
            hbm_content_error=hbm_content_error,
            run_asm_text=run_asm_text,
            run_asm_error=run_asm_error,
            status_data=status_data,
            form_values={
                "hbm_path": request.form.get("hbm_path", ""),
                "fpsram_path": request.form.get("fpsram_path", ""),
                "intsram_path": request.form.get("intsram_path", ""),
                "vram_path": request.form.get("vram_path", ""),
                "opcode_file": request.form.get("opcode_file", ""),
                "opcode_batch": request.form.get("opcode_batch", ""),
                "asm_snippet": stored_form_value("asm_snippet", ""),
                "memory_space": request.form.get("memory_space", "vram"),
                "memory_addr": request.form.get("memory_addr", "0"),
                "memory_len": request.form.get("memory_len", "64"),
                **current_heatmap_form_values,
            },
        )

    @app.route("/manual", methods=["GET", "POST"])
    def manual_view():
        """Low-level control page: endpoint, preload, execute, probe.

        Shares the persistent session (and thus the memory state) with the main
        page's views via _handle_post().
        """
        host, port, _last_action, last_ok, last_error, _hv = _handle_post()
        client = get_persistent_client(host, port)
        status_data, _ = fetch_server_status(client)
        return render_template(
            "manual.html",
            emulator_host=host,
            emulator_port=port,
            last_ok=last_ok,
            last_error=last_error,
            console_log_text=render_console_log(),
            status_data=status_data,
            form_values={
                "hbm_path": request.form.get("hbm_path", ""),
                "fpsram_path": request.form.get("fpsram_path", ""),
                "intsram_path": request.form.get("intsram_path", ""),
                "vram_path": request.form.get("vram_path", ""),
                "opcode_file": request.form.get("opcode_file", ""),
                "opcode_batch": request.form.get("opcode_batch", ""),
                "asm_snippet": stored_form_value("asm_snippet", ""),
                "memory_space": request.form.get("memory_space", "vram"),
                "memory_addr": request.form.get("memory_addr", "0"),
                "memory_len": request.form.get("memory_len", "64"),
            },
        )

    # Memory dumps a batch workload run leaves in the emulator dir. The run is a
    # separate process, so loading these into the GUI's persistent session is
    # the only way its output reaches the memory views. HBM/MRAM aren't here:
    # HBM is only dumped without --quiet (and is huge) and there is no
    # load_mram_file command.
    _RUN_DUMPS = (
        ("vram", "load_vram_file", PROJECT_ROOT / "transactional_emulator" / "vram_dump.bin"),
        ("fpsram", "load_fp_sram_file", PROJECT_ROOT / "transactional_emulator" / "fpsram_dump.bin"),
    )

    @app.route("/api/load_run_output", methods=["POST"])
    def api_load_run_output() -> Any:
        """Load the last workload run's VRAM/FP-SRAM dumps into the session."""
        host, port = connection_settings()
        client = get_persistent_client(host, port)
        loaded: list[str] = []
        errors: dict[str, str] = {}
        for name, cmd, path in _RUN_DUMPS:
            if not path.exists():
                errors[name] = "dump file not found (run a workload first)"
                continue
            req = {"cmd": cmd, "path": str(path)}
            try:
                resp = client.send_command(cmd, path=str(path))
                loaded.append(name)
                append_console_log(f"load_run_{name}", req, resp)
            except Exception as exc:
                errors[name] = str(exc)
                append_console_log(f"load_run_{name}", req, None, str(exc))
        return jsonify({"ok": True, "loaded": loaded, "errors": errors})

    # --------------------------------------------------------------- workloads
    workload_runner = WorkloadRunner()

    @app.route("/api/run_workload", methods=["POST"])
    def api_run_workload() -> Any:
        payload = request.get_json(silent=True) or request.form.to_dict()
        workload = str(payload.get("workload", "")).strip()
        if workload not in WORKLOADS:
            return jsonify({"ok": False, "error": f"unknown workload {workload!r}"}), 400
        spec = WORKLOADS[workload]
        model = str(payload.get("model", "")).strip() or None

        # Apply the hardware config (tile dims + optional precision) first so the
        # spawned emulator/compiler see consistent values.
        try:
            applied = write_hw_config(payload)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"config: {exc}"}), 400

        # Collect the PLENA_TB_* knobs this workload understands (plus the tile
        # dims, which a few scripts read directly).
        knobs: dict[str, str] = {}
        for key in (*spec.get("params", []), "mlen", "blen"):
            val = payload.get(key)
            if val is not None and str(val).strip() != "":
                knobs[key] = str(val).strip()

        try:
            job = workload_runner.start(workload, model, knobs)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "job_id": job.id, "applied": applied})

    @app.route("/api/job/<job_id>", methods=["GET"])
    def api_job(job_id: str) -> Any:
        try:
            offset = int(request.args.get("offset", "0"))
        except ValueError:
            offset = 0
        job = workload_runner.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "unknown job"}), 404
        return jsonify({"ok": True, **job.snapshot(offset)})

    @app.route("/api/stop_workload", methods=["POST"])
    def api_stop_workload() -> Any:
        return jsonify({"ok": True, "stopped": workload_runner.stop()})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask Web GUI for the transactional emulator online service.")
    parser.add_argument(
        "--listen-host",
        default="127.0.0.1",
        help="Flask bind host",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=5000,
        help="Flask bind port",
    )
    parser.add_argument(
        "--emulator-host",
        default="127.0.0.1",
        help="Default online emulator host shown in the UI",
    )
    parser.add_argument(
        "--emulator-port",
        type=int,
        default=7878,
        help="Default online emulator port shown in the UI",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(
        default_emulator_host=args.emulator_host,
        default_emulator_port=args.emulator_port,
    )
    # Suppress Flask's banner ("WARNING: This is a development server...")
    # and Click's run announcement, leaving only our own one-line startup
    # message above so the terminal stays clean.
    import flask.cli

    flask.cli.show_server_banner = lambda *_args, **_kw: None
    print(
        f"[plena-webgui] serving dev console on http://{args.listen_host}:{args.listen_port}",
        flush=True,
    )
    app.run(host=args.listen_host, port=args.listen_port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
