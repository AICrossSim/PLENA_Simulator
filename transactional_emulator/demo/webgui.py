#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from emulator_client import EmulatorClient, EmulatorServiceError, parse_opcode_batch
from flask import Flask, render_template, request, session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "tools"

for import_path in (PROJECT_ROOT, TOOLS_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


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
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def compile_asm_snippet(asm_text: str) -> list[str]:
        asm_source = asm_text.strip()
        if not asm_source:
            raise ValueError("ASM snippet is empty")

        isa_path = PROJECT_ROOT / "compiler" / "doc" / "operation.svh"
        config_path = PROJECT_ROOT / "compiler" / "doc" / "configuration.svh"
        if not isa_path.exists() or not config_path.exists():
            raise RuntimeError(
                "Compiler assets are missing. Ensure the `compiler` submodule is initialized."
            )

        try:
            from compiler.assembler.assembly_to_binary import AssemblyToBinary
        except Exception as exc:  # noqa: BLE001
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
        matrix = [
            sample["values"] + [0.0] * (cols - len(sample["values"]))
            for sample in samples
        ]
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

    @app.route("/", methods=["GET", "POST"])
    def index():
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
            client = EmulatorClient(host=host, port=port)
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
                    path = request.form.get("hbm_path", "").strip()
                    last_request = {"cmd": "load_hbm_file", "path": path}
                    last_response = client.send_command("load_hbm_file", path=path)
                elif action == "load_fpsram":
                    path = request.form.get("fpsram_path", "").strip()
                    last_request = {"cmd": "load_fp_sram_file", "path": path}
                    last_response = client.send_command("load_fp_sram_file", path=path)
                elif action == "load_intsram":
                    path = request.form.get("intsram_path", "").strip()
                    last_request = {"cmd": "load_int_sram_file", "path": path}
                    last_response = client.send_command("load_int_sram_file", path=path)
                elif action == "load_vram":
                    path = request.form.get("vram_path", "").strip()
                    last_request = {"cmd": "load_vram_file", "path": path}
                    last_response = client.send_command("load_vram_file", path=path)
                elif action == "execute_file":
                    path = request.form.get("opcode_file", "").strip()
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
                    requested_addr = parse_int_literal(
                        current_heatmap_form_values["vram_heatmap_addr"], 0
                    )
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
                    requested_addr = parse_int_literal(
                        current_heatmap_form_values["hbm_heatmap_addr"], 0
                    )
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
                    requested_addr = parse_int_literal(
                        current_heatmap_form_values["mram_heatmap_addr"], 0
                    )
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
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        client = EmulatorClient(host=host, port=port)
        config_data, config_error = try_fetch_snapshot(client, "get_config")
        state_data, state_error = try_fetch_snapshot(client, "get_state")
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
        except Exception as exc:  # noqa: BLE001
            vram_heatmap_error = str(exc)
        if vram_heatmap_error and config_error and config_data is None:
            vram_heatmap_error = config_error

        try:
            mram_heatmap_data, mram_heatmap_error = fetch_mram_heatmap(
                client,
                config_data if isinstance(config_data, dict) else None,
                current_heatmap_form_values,
            )
        except Exception as exc:  # noqa: BLE001
            mram_heatmap_error = str(exc)
        if mram_heatmap_error and config_error and config_data is None:
            mram_heatmap_error = config_error

        try:
            hbm_heatmap_data, hbm_heatmap_error = fetch_hbm_heatmap(
                client,
                current_heatmap_form_values,
            )
        except Exception as exc:  # noqa: BLE001
            hbm_heatmap_error = str(exc)

        return render_template(
            "webgui.html",
            emulator_host=host,
            emulator_port=port,
            last_action=last_action,
            last_request_json=json.dumps(last_request, indent=2, sort_keys=True)
            if last_request is not None
            else "",
            last_response_json=json.dumps(last_response, indent=2, sort_keys=True)
            if last_response is not None
            else "",
            last_error=last_error,
            config_json=json.dumps(config_data, indent=2, sort_keys=True)
            if config_data is not None
            else "",
            config_error=config_error,
            state_data=state_data,
            state_json=json.dumps(state_data, indent=2, sort_keys=True)
            if state_data is not None
            else "",
            state_error=state_error,
            vram_heatmap_data=vram_heatmap_data,
            vram_heatmap_error=vram_heatmap_error,
            mram_heatmap_data=mram_heatmap_data,
            mram_heatmap_error=mram_heatmap_error,
            hbm_heatmap_data=hbm_heatmap_data,
            hbm_heatmap_error=hbm_heatmap_error,
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

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flask Web GUI for the transactional emulator online service."
    )
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
    app.run(host=args.listen_host, port=args.listen_port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
