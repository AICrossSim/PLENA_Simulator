#!/usr/bin/env python3

import argparse
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

# The Werkzeug dev server logs every HTTP request to stderr. With the
# monitor polling several times a second that floods the terminal. Demote
# it to WARNING and silence Flask's own info logger as well.
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("flask.app").setLevel(logging.WARNING)

# Imported after the logging tweak above so werkzeug/flask loggers are already
# demoted by the time their modules initialize.
from emulator_client import EmulatorClient, parse_opcode_batch  # noqa: E402
from flask import (  # noqa: E402
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# This file lives at <repo>/gui/webgui.py, so the repo root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "PLENA_Tools"
COMPILER_ROOT = PROJECT_ROOT / "PLENA_Compiler"

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
    mx_exp = _as_int("hbm_mx_exp")
    mx_man = _as_int("hbm_mx_man")
    mx_block = _as_int("hbm_mx_block")
    if mx_exp is not None or mx_man is not None or mx_block is not None:
        for tname in ("HBM_M_WEIGHT_TYPE", "HBM_M_KV_TYPE", "HBM_V_ACT_TYPE", "HBM_V_KV_TYPE"):
            t = prec[tname]
            if mx_block is not None:
                t["block"] = mx_block
            if mx_exp is not None:
                t["ELEM"]["exponent"] = mx_exp
            if mx_man is not None:
                t["ELEM"]["mantissa"] = mx_man
        applied["hbm_mx"] = {"exp": mx_exp, "man": mx_man, "block": mx_block}

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
        "hbm_mx_exp": 4, "hbm_mx_man": 3, "hbm_mx_block": 8,
    }
    try:
        import tomlkit

        doc = tomlkit.load(open(_settings_path()))
        cfg = doc["TRANSACTIONAL"]["CONFIG"]
        prec = doc["TRANSACTIONAL"]["PRECISION"]
        sd = prec["MATRIX_SRAM_TYPE"]["DATA_TYPE"]
        elem = prec["HBM_M_WEIGHT_TYPE"]["ELEM"]
        return {
            "mlen": int(cfg["MLEN"]["value"]),
            "blen": int(cfg["BLEN"]["value"]),
            "hlen": int(cfg["HLEN"]["value"]),
            "vlen": int(cfg["VLEN"]["value"]),
            "broadcast_amount": int(cfg["BROADCAST_AMOUNT"]["value"]),
            "sram_exp": int(sd["exponent"]),
            "sram_man": int(sd["mantissa"]),
            "hbm_mx_exp": int(elem["exponent"]),
            "hbm_mx_man": int(elem["mantissa"]),
            "hbm_mx_block": int(prec["HBM_M_WEIGHT_TYPE"]["block"]),
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
    mode: str = "monitor",
) -> Flask:
    if mode not in {"monitor", "dev"}:
        raise ValueError(f"Unknown mode: {mode!r}; expected 'monitor' or 'dev'")

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "PLENA_WEBGUI_SECRET_KEY",
        "plena-transactional-emulator-webgui-dev",
    )
    app.config["DEFAULT_EMULATOR_HOST"] = default_emulator_host
    app.config["DEFAULT_EMULATOR_PORT"] = default_emulator_port
    app.config["WEBGUI_MODE"] = mode
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
    # all /api/snapshot calls so the webgui appears as a single stable
    # session in the monitor instead of churning through a new id every
    # poll. Keyed because the user may flip endpoints via the form.
    monitor_clients: dict[tuple[str, int], EmulatorClient] = {}
    monitor_clients_lock = threading.Lock()

    def get_monitor_client(host: str, port: int) -> EmulatorClient:
        key = (host, port)
        with monitor_clients_lock:
            client = monitor_clients.get(key)
            if client is None:
                client = EmulatorClient(
                    host=host,
                    port=port,
                    timeout=3.0,
                    label="webgui-monitor",
                    auto_label=True,
                )
                monitor_clients[key] = client
            # If the persistent socket was dropped (server restart,
            # network blip, fallback after a transient error), reopen it
            # so the next request is again labeled+persistent.
            if client._sock is None:
                try:
                    client.connect()
                except OSError:
                    # Leave _sock as None; individual send_command calls
                    # will fall back to ephemeral and surface their own
                    # errors via the snapshot payload.
                    pass
            return client

    def _close_monitor_clients() -> None:
        with monitor_clients_lock:
            for client in monitor_clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            monitor_clients.clear()

    atexit.register(_close_monitor_clients)

    HEATMAP_DEFAULTS = {
        "vram_heatmap_addr": "0",
        "vram_heatmap_rows": "8",
        "mram_heatmap_addr": "0",
        "mram_heatmap_tiles": "1",
        "hbm_heatmap_addr": "0",
        "hbm_heatmap_rows": "16",
        "hbm_heatmap_cols": "32",
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

    def fetch_vram_heatmap(
        client: EmulatorClient,
        config_data: dict[str, Any] | None,
        form_values: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if config_data is None:
            return None, "Configuration unavailable; cannot determine VRAM row stride."

        row_stride = int(config_data.get("vlen") or 0)
        if row_stride <= 0:
            return None, "Invalid VLEN in configuration; cannot render VRAM heatmap."

        requested_addr = parse_int_literal(form_values["vram_heatmap_addr"], 0)
        rows = clamp_int(parse_int_literal(form_values["vram_heatmap_rows"], 8), 1, 32)
        start_addr = (requested_addr // row_stride) * row_stride

        samples: list[dict[str, Any]] = []
        flat_values: list[float] = []
        data_type = ""
        for row_idx in range(rows):
            addr = start_addr + row_idx * row_stride
            sample = client.send_command("read_vram", addr=addr)
            values = [float(value) for value in sample.get("values", [])]
            flat_values.extend(values)
            samples.append({"addr": addr, "values": values})
            if not data_type:
                data_type = str(sample.get("data_type", ""))

        cols = max((len(sample["values"]) for sample in samples), default=0)
        matrix = [sample["values"] + [0.0] * (cols - len(sample["values"])) for sample in samples]
        min_value = min(flat_values, default=0.0)
        max_value = max(flat_values, default=0.0)

        return {
            "kind": "vram",
            "requested_addr": requested_addr,
            "start_addr": start_addr,
            "row_stride": row_stride,
            "rows": rows,
            "cols": cols,
            "data_type": data_type,
            "row_addrs": [sample["addr"] for sample in samples],
            "values": matrix,
            "min_value": min_value,
            "max_value": max_value,
        }, None

    def fetch_hbm_heatmap(
        client: EmulatorClient,
        form_values: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        start_addr = parse_int_literal(form_values["hbm_heatmap_addr"], 0)
        rows = clamp_int(parse_int_literal(form_values["hbm_heatmap_rows"], 16), 1, 64)
        cols = clamp_int(parse_int_literal(form_values["hbm_heatmap_cols"], 32), 4, 128)
        total_len = rows * cols

        sample = client.send_command("read_hbm", addr=start_addr, len=total_len)
        raw_hex = str(sample.get("hex", ""))
        raw_bytes = bytes.fromhex(raw_hex) if raw_hex else b""
        matrix = []
        for row_idx in range(rows):
            offset = row_idx * cols
            row_bytes = list(raw_bytes[offset : offset + cols])
            if len(row_bytes) < cols:
                row_bytes.extend([0] * (cols - len(row_bytes)))
            matrix.append(row_bytes)

        return {
            "kind": "hbm",
            "start_addr": start_addr,
            "rows": rows,
            "cols": cols,
            "len": total_len,
            "row_addrs": [start_addr + row_idx * cols for row_idx in range(rows)],
            "values": matrix,
            "min_value": min(raw_bytes, default=0),
            "max_value": max(raw_bytes, default=0),
        }, None

    def fetch_mram_heatmap(
        client: EmulatorClient,
        config_data: dict[str, Any] | None,
        form_values: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if config_data is None:
            return None, "Configuration unavailable; cannot determine MRAM tile size."

        tile_side = int(config_data.get("mlen") or 0)
        if tile_side <= 0:
            return None, "Invalid MLEN in configuration; cannot render MRAM heatmap."

        tile_stride = tile_side * tile_side
        requested_addr = parse_int_literal(form_values["mram_heatmap_addr"], 0)
        tiles = clamp_int(parse_int_literal(form_values["mram_heatmap_tiles"], 1), 1, 8)
        start_addr = (requested_addr // tile_stride) * tile_stride

        row_addrs: list[int] = []
        tile_bases_by_row: list[int] = []
        matrix: list[list[float]] = []
        flat_values: list[float] = []
        data_type = ""

        for tile_idx in range(tiles):
            tile_base = start_addr + tile_idx * tile_stride
            sample = client.send_command("read_mram", addr=tile_base)
            values = [float(value) for value in sample.get("values", [])]
            flat_values.extend(values)
            if not data_type:
                data_type = str(sample.get("data_type", ""))

            padded_values = values + [0.0] * max(0, tile_stride - len(values))
            for row_idx in range(tile_side):
                start = row_idx * tile_side
                end = start + tile_side
                matrix.append(padded_values[start:end])
                row_addrs.append(tile_base + start)
                tile_bases_by_row.append(tile_base)

        return {
            "kind": "mram",
            "requested_addr": requested_addr,
            "start_addr": start_addr,
            "tile_side": tile_side,
            "tile_stride": tile_stride,
            "tiles": tiles,
            "rows": len(matrix),
            "cols": tile_side,
            "data_type": data_type,
            "row_addrs": row_addrs,
            "tile_bases_by_row": tile_bases_by_row,
            "values": matrix,
            "min_value": min(flat_values, default=0.0),
            "max_value": max(flat_values, default=0.0),
        }, None

    def fetch_sessions(client: EmulatorClient) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("list_sessions"), None
        except Exception as exc:
            return None, str(exc)

    def fetch_history(client: EmulatorClient, limit: int = 30) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("get_history", limit=limit), None
        except Exception as exc:
            return None, str(exc)

    def fetch_closed_sessions(client: EmulatorClient, limit: int = 30) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("list_closed_sessions", limit=limit), None
        except Exception as exc:
            return None, str(exc)

    def fetch_execution_progress(
        client: EmulatorClient,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("get_execution_progress"), None
        except Exception as exc:
            return None, str(exc)

    def fetch_session_modules(
        client: EmulatorClient, session_id: int, limit: int = 20
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("list_session_modules", id=session_id, limit=limit), None
        except Exception as exc:
            return None, str(exc)

    def fetch_server_status(client: EmulatorClient) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return client.send_command("get_server_status"), None
        except Exception as exc:
            return None, str(exc)

    @app.route("/", methods=["GET"])
    def root():
        if app.config["WEBGUI_MODE"] == "monitor":
            return redirect(url_for("monitor"))
        # The dev view's endpoint is the function name, `dev_view`.
        return redirect(url_for("dev_view"))

    @app.route("/api/session_state", methods=["GET"])
    def api_session_state():
        """On-demand drill-in: state + config + progress for one session.

        Fired by the monitor UI when the user expands a session row. We
        go through the monitor's persistent client; the gateway routes
        each `get_session_*` command to the target session's backend via
        a short-lived proxy connection.
        """
        host = request.args.get("host") or app.config["DEFAULT_EMULATOR_HOST"]
        try:
            port = int(request.args.get("port") or app.config["DEFAULT_EMULATOR_PORT"])
        except (TypeError, ValueError):
            port = app.config["DEFAULT_EMULATOR_PORT"]
        try:
            sess_id = int(request.args.get("id", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid id"}), 400

        client = get_monitor_client(host, port)

        def fetch(cmd):
            try:
                return client.send_command(cmd, id=sess_id), None
            except Exception as exc:
                return None, str(exc)

        state, state_err = fetch("get_session_state")
        config, config_err = fetch("get_session_config")
        progress, progress_err = fetch("get_session_progress")
        modules, modules_err = fetch_session_modules(client, sess_id, limit=64)
        return jsonify(
            {
                "id": sess_id,
                "state": state,
                "state_error": state_err,
                "config": config,
                "config_error": config_err,
                "progress": progress,
                "progress_error": progress_err,
                "modules": modules,
                "modules_error": modules_err,
            }
        )

    @app.route("/api/snapshot", methods=["GET"])
    def api_snapshot():
        """JSON snapshot used by the monitor view for live polling.

        Uses one short-lived but labeled persistent connection so the GUI
        appears as a single labeled session rather than five ephemeral ones.
        """
        host = request.args.get("host") or app.config["DEFAULT_EMULATOR_HOST"]
        try:
            port = int(request.args.get("port") or app.config["DEFAULT_EMULATOR_PORT"])
        except (TypeError, ValueError):
            port = app.config["DEFAULT_EMULATOR_PORT"]

        # Reuse the persistent monitor client so the webgui shows up as
        # one stable session in /list_sessions instead of one per poll.
        client = get_monitor_client(host, port)
        # Cheap session-only commands first — these never touch the state
        # lock, so they return promptly even while a heavy execute_* holds
        # the write lock and is grinding through a kernel.
        status_data, status_error = fetch_server_status(client)
        progress_data, progress_error = fetch_execution_progress(client)
        sessions_data, sessions_error = fetch_sessions(client)
        closed_data, closed_error = fetch_closed_sessions(client, limit=30)
        # History deliberately not fetched: the monitor UI now surfaces the
        # same info per-session (last_cmd, cmds_executed) inline.
        history_data, history_error = None, None

        # State/config aren't session-only commands. Under the gateway they
        # would force the monitor's idle session to spawn a per-session
        # backend (~1 GB HBM) just so we can poll. In monitor mode we
        # deliberately skip them — the GUI is about "who's running what",
        # not detailed emulator state. Dev mode keeps the fetches because
        # the dev page actually needs them. While an execute_* is in
        # progress they would also block on the write lock; skip then too.
        is_running = bool(progress_data and progress_data.get("running"))
        skip_state = app.config["WEBGUI_MODE"] == "monitor" or is_running
        if skip_state:
            state_data, state_error = (
                None,
                (
                    "skipped in monitor mode (use dev mode to see emulator state)"
                    if app.config["WEBGUI_MODE"] == "monitor"
                    else "skipped (execute in progress)"
                ),
            )
            config_data, config_error = None, None
        else:
            state_data, state_error = try_fetch_snapshot(client, "get_state")
            config_data, config_error = try_fetch_snapshot(client, "get_config")

        return jsonify(
            {
                "connection": {"host": host, "port": port},
                "status": status_data,
                "status_error": status_error,
                "progress": progress_data,
                "progress_error": progress_error,
                "sessions": sessions_data,
                "sessions_error": sessions_error,
                "closed_sessions": closed_data,
                "closed_sessions_error": closed_error,
                "history": history_data,
                "history_error": history_error,
                "state": state_data,
                "state_error": state_error,
                "config": config_data,
                "config_error": config_error,
            }
        )

    @app.route("/monitor", methods=["GET", "POST"])
    def monitor():
        # The monitor template renders an empty shell and pulls everything
        # client-side via /api/snapshot, so we deliberately do NOT open an
        # emulator connection here. That keeps the page load free of an
        # extra throwaway TCP session.
        host, port = connection_settings()
        if request.method == "POST":
            host, port = update_connection_from_form()
        return render_template(
            "monitor.html",
            emulator_host=host,
            emulator_port=port,
            dev_available=True,
        )

    @app.route("/dev", methods=["GET", "POST"])
    def dev_view():
        host, port = connection_settings()
        last_action = None
        last_request = None
        last_response = None
        last_error = None
        current_heatmap_form_values = heatmap_form_values()

        if request.method == "POST":
            host, port = update_connection_from_form()
            current_heatmap_form_values = update_heatmap_form_values()
            if "asm_snippet" in request.form:
                session["asm_snippet"] = request.form.get("asm_snippet", "")
            # Interactive actions can run a real kernel; allow a generous read
            # timeout (override via PLENA_WEBGUI_EXEC_TIMEOUT) so long executes
            # complete instead of tripping the default 10s client timeout.
            exec_timeout = float(os.environ.get("PLENA_WEBGUI_EXEC_TIMEOUT", "300"))
            client = EmulatorClient(host=host, port=port, timeout=exec_timeout)
            action = request.form.get("action", "").strip() or "ping"
            last_action = action

            try:
                if action == "ping":
                    last_request = {"cmd": "ping"}
                    last_response = client.send_command("ping")
                elif action == "reset":
                    last_request = {"cmd": "reset"}
                    last_response = client.send_command("reset")
                elif action == "get_config":
                    last_request = {"cmd": "get_config"}
                    last_response = client.send_command("get_config")
                elif action == "get_state":
                    last_request = {"cmd": "get_state"}
                    last_response = client.send_command("get_state")
                elif action == "load_hbm":
                    path = resolve_allowed_file_path(request.form.get("hbm_path", "").strip())
                    last_request = {"cmd": "load_hbm_file", "path": path}
                    last_response = client.send_command("load_hbm_file", path=path)
                elif action == "load_fpsram":
                    path = resolve_allowed_file_path(request.form.get("fpsram_path", "").strip())
                    last_request = {"cmd": "load_fp_sram_file", "path": path}
                    last_response = client.send_command("load_fp_sram_file", path=path)
                elif action == "load_intsram":
                    path = resolve_allowed_file_path(request.form.get("intsram_path", "").strip())
                    last_request = {"cmd": "load_int_sram_file", "path": path}
                    last_response = client.send_command("load_int_sram_file", path=path)
                elif action == "load_vram":
                    path = resolve_allowed_file_path(request.form.get("vram_path", "").strip())
                    last_request = {"cmd": "load_vram_file", "path": path}
                    last_response = client.send_command("load_vram_file", path=path)
                elif action == "execute_file":
                    path = resolve_allowed_file_path(request.form.get("opcode_file", "").strip())
                    last_request = {"cmd": "execute_file", "path": path}
                    last_response = client.send_command("execute_file", path=path)
                elif action == "execute_batch":
                    opcode_text = request.form.get("opcode_batch", "")
                    opcodes = parse_opcode_batch(opcode_text)
                    last_request = {"cmd": "execute_batch", "opcodes": opcodes}
                    last_response = client.send_command("execute_batch", opcodes=opcodes)
                elif action == "execute_asm":
                    asm_snippet = request.form.get("asm_snippet", "")
                    opcodes = compile_asm_snippet(asm_snippet)
                    last_request = {
                        "cmd": "execute_batch",
                        "source": "asm_snippet",
                        "asm": asm_snippet,
                        "opcodes": opcodes,
                    }
                    last_response = client.send_command("execute_batch", opcodes=opcodes)
                elif action == "read_memory":
                    memory_space = request.form.get("memory_space", "vram")
                    addr = int(request.form.get("memory_addr", "0"), 0)
                    length = int(request.form.get("memory_len", "64"), 0)
                    if memory_space == "vram":
                        last_request = {"cmd": "read_vram", "addr": addr}
                        last_response = client.send_command("read_vram", addr=addr)
                    elif memory_space == "mram":
                        last_request = {"cmd": "read_mram", "addr": addr}
                        last_response = client.send_command("read_mram", addr=addr)
                    else:
                        last_request = {"cmd": "read_hbm", "addr": addr, "len": length}
                        last_response = client.send_command("read_hbm", addr=addr, len=length)
                elif action == "refresh_vram_heatmap":
                    requested_addr = parse_int_literal(current_heatmap_form_values["vram_heatmap_addr"], 0)
                    rows = clamp_int(
                        parse_int_literal(current_heatmap_form_values["vram_heatmap_rows"], 8),
                        1,
                        32,
                    )
                    last_request = {
                        "cmd": "read_vram_window",
                        "start_addr": requested_addr,
                        "rows": rows,
                    }
                    last_response = {
                        "message": "VRAM heatmap refresh requested",
                        "start_addr": requested_addr,
                        "rows": rows,
                    }
                elif action == "refresh_hbm_heatmap":
                    requested_addr = parse_int_literal(current_heatmap_form_values["hbm_heatmap_addr"], 0)
                    rows = clamp_int(
                        parse_int_literal(current_heatmap_form_values["hbm_heatmap_rows"], 16),
                        1,
                        64,
                    )
                    cols = clamp_int(
                        parse_int_literal(current_heatmap_form_values["hbm_heatmap_cols"], 32),
                        4,
                        128,
                    )
                    last_request = {
                        "cmd": "read_hbm_window",
                        "addr": requested_addr,
                        "rows": rows,
                        "cols": cols,
                    }
                    last_response = {
                        "message": "HBM heatmap refresh requested",
                        "addr": requested_addr,
                        "rows": rows,
                        "cols": cols,
                    }
                elif action == "refresh_mram_heatmap":
                    requested_addr = parse_int_literal(current_heatmap_form_values["mram_heatmap_addr"], 0)
                    tiles = clamp_int(
                        parse_int_literal(current_heatmap_form_values["mram_heatmap_tiles"], 1),
                        1,
                        8,
                    )
                    last_request = {
                        "cmd": "read_mram_window",
                        "start_addr": requested_addr,
                        "tiles": tiles,
                    }
                    last_response = {
                        "message": "MRAM heatmap refresh requested",
                        "start_addr": requested_addr,
                        "tiles": tiles,
                    }
                else:
                    last_error = f"Unknown action: {action}"
            except Exception as exc:
                last_error = str(exc)

        client = EmulatorClient(host=host, port=port)
        config_data, config_error = try_fetch_snapshot(client, "get_config")
        state_data, state_error = try_fetch_snapshot(client, "get_state")
        sessions_data, _ = fetch_sessions(client)
        status_data, _ = fetch_server_status(client)
        vram_heatmap_data = None
        vram_heatmap_error = None
        mram_heatmap_data = None
        mram_heatmap_error = None
        hbm_heatmap_data = None
        hbm_heatmap_error = None

        try:
            vram_heatmap_data, vram_heatmap_error = fetch_vram_heatmap(
                client,
                config_data if isinstance(config_data, dict) else None,
                current_heatmap_form_values,
            )
        except Exception as exc:
            vram_heatmap_error = str(exc)
        if vram_heatmap_error and config_error and config_data is None:
            vram_heatmap_error = config_error

        try:
            mram_heatmap_data, mram_heatmap_error = fetch_mram_heatmap(
                client,
                config_data if isinstance(config_data, dict) else None,
                current_heatmap_form_values,
            )
        except Exception as exc:
            mram_heatmap_error = str(exc)
        if mram_heatmap_error and config_error and config_data is None:
            mram_heatmap_error = config_error

        try:
            hbm_heatmap_data, hbm_heatmap_error = fetch_hbm_heatmap(
                client,
                current_heatmap_form_values,
            )
        except Exception as exc:
            hbm_heatmap_error = str(exc)

        return render_template(
            "webgui.html",
            emulator_host=host,
            emulator_port=port,
            last_action=last_action,
            last_request_json=json.dumps(last_request, indent=2, sort_keys=True) if last_request is not None else "",
            last_response_json=json.dumps(last_response, indent=2, sort_keys=True) if last_response is not None else "",
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
            sessions_data=sessions_data,
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

    # --------------------------------------------------------------- workloads
    workload_runner = WorkloadRunner()

    def _dev_only() -> Response | None:
        if app.config["WEBGUI_MODE"] != "dev":
            return jsonify({"ok": False, "error": "workload runner is dev-mode only"}), 403
        return None

    @app.route("/api/run_workload", methods=["POST"])
    def api_run_workload() -> Any:
        guard = _dev_only()
        if guard is not None:
            return guard
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
        guard = _dev_only()
        if guard is not None:
            return guard
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
        guard = _dev_only()
        if guard is not None:
            return guard
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
        "--mode",
        choices=("monitor", "dev"),
        default="monitor",
        help=(
            "UI mode. `monitor` (default): read-only live view of emulator state, "
            "sessions, and history. `dev`: interactive control panel "
            "(load files, execute, reset)."
        ),
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
        mode=args.mode,
    )
    # Suppress Flask's banner ("WARNING: This is a development server...")
    # and Click's run announcement, leaving only our own one-line startup
    # message above. Lets the terminal stay clean while the monitor polls
    # several times a second.
    import flask.cli

    flask.cli.show_server_banner = lambda *_args, **_kw: None
    print(
        f"[plena-webgui] serving {args.mode!r} on http://{args.listen_host}:{args.listen_port}",
        flush=True,
    )
    app.run(host=args.listen_host, port=args.listen_port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
