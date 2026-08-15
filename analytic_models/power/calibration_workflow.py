"""Prepare and ingest reproducible DC/SAIF power-calibration runs."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import re
import shlex
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifact_catalog import (
    ARTIFACT_CATALOG_SCHEMA,
    CONTEXT_ARTIFACT_KINDS,
    POINT_ARTIFACT_KINDS,
    validate_artifact_catalog,
)
from .calibration_manifest import (
    CALIBRATION_MANIFEST_SCHEMA,
    HARDWARE_FP_BINDING,
    MEASUREMENT_COLUMNS,
    MX_BLOCK_SIZE,
    SELECTOR_SIGNATURE,
    VECTOR_FP,
    CalibrationPoint,
    build_manifest,
    manifest_hash,
    manifest_payload,
)

MODEL_WORKFLOW_SCHEMA = "plena-power-calibration-workflow"
RUN_SPEC_SCHEMA = "plena-power-dc-run"
ACTIVITY_REQUEST_SCHEMA = "plena-power-activity-request"
ACTIVITY_TRACE_SCHEMA = "plena-power-decode-trace"
EXACT_ACTIVITY_TRACE_SCHEMA = "plena-exact-decode-trace"
TRACE_PAIR_SCHEMA = "plena-power-trace-pair"
LIBRARY_MANIFEST_SCHEMA = "plena-dc-library-manifest"
TOOL_CONTEXT_SCHEMA = "plena-dc-tool-context"
TOOL_LOG_INDEX_SCHEMA = "plena-power-tool-log-index"
RTL_SOURCE_MANIFEST_SCHEMA = "decode-rtl-source-manifest"
EXACT_REQUEST_SCHEMA = "plena-exact-dc-request"
EXACT_WORKFLOW_SCHEMA = "plena-exact-dc-workflow"
EXACT_ANCHOR_SCHEMA = "decode-exact-dc-anchor-index"
RTL_SPECIALIZATION_SCHEMA = "decode-rtl-dc-specialization"
COMPILER_BINDING_SCHEMA = "plena-compiler-precision-binding"

_DYNAMIC_COMPONENTS = frozenset({"array", "vector", "selector"})
_DC_COMPONENTS = frozenset(
    {"array", "vector", "selector", "fixed", "chip_leakage"}
)
_TRACE_COMPONENTS = frozenset({"cycle", "latency"})
_POWER_SCALE = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9}
_NUMBER = r"([-+]?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
_SKIP_RTL = frozenset({"bram.sv", "fake_hbm.sv"})
# Simulation-only memory and bus models. They carry string parameters and
# file-backed initialisation that no synthesis front end accepts, and no
# synthesised module instantiates them.
_SIMULATION_ONLY_RTL = frozenset(
    {
        "fake_hbm_4port.sv",
        "fake_hbm_5port.sv",
        "fake_instr_mem.sv",
        "peripheral_system.sv",
        # A testbench harness that instantiates the matrix machine with its own
        # parameters. Analyzing it elaborates a second, conflicting variant of
        # the compute hierarchy that cannot be linked against the real one.
        "matrix_machine_tb_wrapper.sv",
    }
)
_SKIP_SYNTHESIS_RTL = (
    _SKIP_RTL | _SIMULATION_ONLY_RTL | frozenset({"fp_rounding.sv"})
)
# Packages that are imported rather than included, so the file defining them
# has to be analyzed in its own right and ahead of every user.
_ANALYZED_PACKAGE_HEADERS = ("memory/HBM/TileLink_Lib/prim_util_pkg.svh",)
_SYNTHESIS_RELATIVE_DIRS = (
    "basic_components/mx_fp_operation/rtl",
    "basic_components/int_operation/rtl",
    "basic_components/fp_operation/rtl",
    "basic_components/fixed_operation/rtl",
    "basic_components/conversion/rtl",
    "basic_components/common/rtl",
    "basic_components/cast/rtl",
    "basic_components/buffer/rtl",
    "basic_components/gemv/rtl",
    "basic_components/systolic_gemm_mx/rtl",
    "basic_components/systolic_gemm_mxint/rtl",
    "basic_components/synopsis_ip_inst/rtl",
    "basic_components/synopsis/rtl",
    "basic_components/hadamard_transform/rtl",
    "memory/matrix_sram/rtl",
    "memory/vector_sram/rtl",
    "memory/scalar_sram/rtl",
    "memory/HBM/rtl",
    "memory/HBM/TileLink_Lib",
    "matrix_machine/rtl",
    "frontend/rtl",
    "scalar_machine/rtl",
    "vector_machine/rtl",
    "control/rtl",
    "core/rtl",
)
_FIXED_BLACKBOXES = (
    "matrix_machine",
    "vector_machine",
    "matrix_sram_without_rounding",
    "fp_vector_sram",
    "scalar_sram",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    if executable:
        temporary.chmod(
            temporary.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
    temporary.replace(path)


def _write_json(path: Path, body: Mapping[str, Any]) -> None:
    payload = dict(body)
    payload["content_hash"] = _content_hash(payload)
    _write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )


def _load_hashed_json(
    path: Path,
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    body = dict(raw)
    observed = str(body.pop("content_hash", ""))
    if _content_hash(body) != observed:
        raise ValueError(f"{path} content hash mismatch")
    if schema is not None and body.get("schema_version") != schema:
        raise ValueError(f"{path} uses an unsupported schema")
    return {**body, "content_hash": observed}


def _validate_sha256(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_identifier(value: object, label: str) -> str:
    token = str(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", token) is None:
        raise ValueError(f"{label} is not a valid RTL identifier")
    return token


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {resolved}")
    return resolved


def _require_source_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _ensure_empty_target(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"workflow path is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(f"workflow directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _copy_verified(source: Path, destination: Path) -> None:
    _write_bytes(destination, source.read_bytes())
    if _sha256_file(source) != _sha256_file(destination):
        raise ValueError(f"copy verification failed for {source}")


@functools.lru_cache(maxsize=None)
def _rtl_source_files(rtl_root: Path) -> tuple[Path, ...]:
    src = rtl_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"RTL source directory is missing: {src}")
    selected: list[Path] = []
    for path in src.rglob("*"):
        if not path.is_file() or path.suffix not in {".sv", ".v", ".svh", ".vh"}:
            continue
        relative = path.relative_to(src)
        if path.name in _SKIP_RTL:
            continue
        if relative.parts[0] == "definitions":
            selected.append(path)
        elif "rtl" in relative.parts or "TileLink_Lib" in relative.parts:
            selected.append(path)
    files = tuple(sorted(set(selected)))
    if not files:
        raise ValueError("RTL source census is empty")
    return files


def build_rtl_source_manifest(rtl_root: str | Path) -> dict[str, Any]:
    root = Path(rtl_root).resolve()
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in _rtl_source_files(root)
    }
    body = {
        "schema_version": RTL_SOURCE_MANIFEST_SCHEMA,
        "files": dict(sorted(files.items())),
        "source_tree_sha256": _content_hash(
            {"files": dict(sorted(files.items()))}
        ),
    }
    return {**body, "content_hash": _content_hash(body)}


def _validate_rtl_source_manifest(
    manifest: Mapping[str, Any],
    rtl_root: Path,
) -> None:
    if manifest.get("schema_version") != RTL_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("RTL source manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("RTL source manifest file census is empty")
    observed: dict[str, str] = {}
    for name, digest in files.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("RTL source manifest path is not confined")
        source = (rtl_root / relative).resolve()
        try:
            source.relative_to(rtl_root)
        except ValueError as exc:
            raise ValueError("RTL source path escapes the source root") from exc
        observed[str(name)] = _sha256_file(
            _require_source_file(source, "RTL source")
        )
        if observed[str(name)] != _validate_sha256(digest, "RTL source hash"):
            raise ValueError(f"RTL source changed after preparation: {name}")
    expected_tree = _content_hash({"files": dict(sorted(observed.items()))})
    if manifest.get("source_tree_sha256") != expected_tree:
        raise ValueError("RTL source-tree identity mismatch")


def create_library_manifest(
    output: str | Path,
    *,
    library_id: str,
    process_corner: str,
    operating_condition: str,
    library_files: Sequence[str | Path],
) -> Path:
    if not library_id.strip() or not process_corner.strip():
        raise ValueError("library and process-corner identities are required")
    if not operating_condition.strip():
        raise ValueError("operating condition is required")
    records = []
    for raw_path in sorted({str(Path(path).resolve()) for path in library_files}):
        path = _require_file(raw_path, "library file")
        records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError("at least one characterized library file is required")
    body = {
        "schema_version": LIBRARY_MANIFEST_SCHEMA,
        "library_id": library_id.strip(),
        "process_corner": process_corner.strip(),
        "operating_condition": operating_condition.strip(),
        "files": records,
    }
    target = Path(output).resolve()
    _write_json(target, body)
    return target


def _load_library_manifest(path: str | Path) -> dict[str, Any]:
    source = _require_file(path, "library manifest")
    value = _load_hashed_json(source, schema=LIBRARY_MANIFEST_SCHEMA)
    if set(value) != {
        "schema_version",
        "library_id",
        "process_corner",
        "operating_condition",
        "files",
        "content_hash",
    }:
        raise ValueError("library manifest fields differ from the schema")
    records = value["files"]
    if not isinstance(records, list) or not records:
        raise ValueError("library manifest contains no files")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("library file record fields differ from the schema")
        library = _require_file(record["path"], "library file")
        if (
            isinstance(record["size_bytes"], bool)
            or int(record["size_bytes"]) != library.stat().st_size
            or _sha256_file(library) != record["sha256"]
        ):
            raise ValueError(f"library file identity mismatch: {library}")
    return value


def _format_fields(token: str) -> tuple[bool, int, int, int]:
    value = token.upper()
    if value.startswith("MXINT"):
        return True, int(value.removeprefix("MXINT")), 1, 2
    value = value.removeprefix("MXFP_")
    match = re.fullmatch(r"E(\d+)M(\d+)", value)
    if match is None:
        raise ValueError(f"unsupported matrix format {token!r}")
    exponent, mantissa = (int(part) for part in match.groups())
    return False, 8, exponent, mantissa


def _vector_fields(token: str) -> tuple[int, int]:
    value = token.upper()
    if value == "BF16":
        return 8, 7
    match = re.fullmatch(r"FP_E(\d+)M(\d+)", value)
    if match is None:
        raise ValueError(f"unsupported vector format {token!r}")
    return tuple(int(part) for part in match.groups())


def _point_formats(point: CalibrationPoint) -> dict[str, str]:
    if point.component == "array":
        _, operands = point.signature.split(":", 1)
        left, activation = operands.split("x", 1)
        matrix = left
        return {
            "weight": matrix,
            "activation": activation,
            "key": matrix,
            "value": matrix,
            "vector": HARDWARE_FP_BINDING,
        }
    if point.component == "vector":
        return {
            "weight": "MXINT8",
            "activation": "MXINT8",
            "key": "MXINT8",
            "value": "MXINT8",
            "vector": point.signature.removeprefix("VECTOR:"),
        }
    return {
        "weight": "MXINT4",
        "activation": "MXINT4",
        "key": "MXINT4",
        "value": "MXINT4",
        "vector": HARDWARE_FP_BINDING,
    }


def _precision_replacements(formats: Mapping[str, str]) -> dict[str, int]:
    weight = _format_fields(formats["weight"])
    activation = _format_fields(formats["activation"])
    key = _format_fields(formats["key"])
    vector_exp, vector_mant = _vector_fields(formats["vector"])
    if weight[0] != activation[0] or weight[0] != key[0]:
        raise ValueError("one calibration run cannot mix MXINT and MXFP")
    return {
        "ACT_MXFP_MANT_WIDTH": activation[3],
        "ACT_MXFP_EXP_WIDTH": activation[2],
        "ACT_MX_INT_WIDTH": activation[1],
        "ACT_MX_SCALE_WIDTH": 8,
        "ACT_MX_INT_ENABLE": int(activation[0]),
        "KV_MX_MANT_WIDTH": key[3],
        "KV_MX_EXP_WIDTH": key[2],
        "KV_MX_INT_WIDTH": key[1],
        "KV_MX_SCALE_WIDTH": 8,
        "KV_MX_INT_ENABLE": int(key[0]),
        "WT_MX_MANT_WIDTH": weight[3],
        "WT_MX_EXP_WIDTH": weight[2],
        "WT_MX_INT_ENABLE": int(weight[0]),
        "WT_MX_INT_WIDTH": weight[1],
        "WT_MX_SCALE_WIDTH": 8,
        "MX_SCALE_WIDTH": 8,
        "BLOCK_DIM": MX_BLOCK_SIZE,
        "V_FP_EXP_WIDTH": vector_exp,
        "V_FP_MANT_WIDTH": vector_mant,
        "M_FP_EXP_WIDTH": vector_exp,
        "M_FP_MANT_WIDTH": vector_mant,
        "S_FP_EXP_WIDTH": vector_exp,
        "S_FP_MANT_WIDTH": vector_mant,
    }


def _replace_localparams(template: str, values: Mapping[str, int]) -> str:
    rendered = template
    for name, value in values.items():
        pattern = re.compile(
            rf"(?m)^(\s*localparam\s+{re.escape(name)}\s*=\s*)[^;]+;"
        )
        rendered, count = pattern.subn(rf"\g<1>{value};", rendered)
        if count != 1:
            raise ValueError(f"definition template contains {count} values for {name}")
    return rendered


def _render_definitions(
    rtl_root: Path,
    run_dir: Path,
    *,
    formats: Mapping[str, str],
    mlen: int,
    blen: int,
    hlen: int,
) -> tuple[Path, Path]:
    if (
        mlen <= 0
        or blen <= 0
        or hlen <= 0
        or mlen % blen
        or mlen % hlen
        or blen > hlen
    ):
        raise ValueError("calibration geometry is invalid")
    definitions = rtl_root / "src" / "definitions"
    precision = _replace_localparams(
        (definitions / "precision.svh").read_text(encoding="utf-8"),
        _precision_replacements(formats),
    )
    configuration = _replace_localparams(
        (definitions / "configuration.svh").read_text(encoding="utf-8"),
        {"MLEN": mlen, "BLEN": blen, "HLEN": hlen, "VLEN": mlen},
    )
    precision_path = run_dir / "definitions" / "precision.svh"
    configuration_path = run_dir / "definitions" / "configuration.svh"
    _write_bytes(precision_path, precision.encode())
    _write_bytes(configuration_path, configuration.encode())
    return precision_path, configuration_path


def _selector_wrapper(
    *,
    enabled: bool,
    mlen: int,
    hlen: int,
    element_width: int,
) -> str:
    implementation = (
        """
    packed_kv_head_selector #(
        .ELEMENT_WIDTH(ELEMENT_WIDTH),
        .MLEN(MLEN),
        .HLEN(HLEN),
        .SELECTOR_WIDTH(SELECTOR_WIDTH)
    ) selector_impl (
        .packed_row(packed_row),
        .selector(selector),
        .selected_head(selected_head),
        .selector_valid(selector_valid)
    );"""
        if enabled
        else """
    assign selected_head = packed_row[0 +: HLEN];
    assign selector_valid = (selector == '0);"""
    )
    return f"""`timescale 1ns / 1ps

module calibration_selector_top #(
    parameter int ELEMENT_WIDTH = {element_width},
    parameter int MLEN = {mlen},
    parameter int HLEN = {hlen},
    parameter int SELECTOR_WIDTH = 4
) (
    input logic [MLEN-1:0][ELEMENT_WIDTH-1:0] packed_row,
    input logic [SELECTOR_WIDTH-1:0] selector,
    output logic [HLEN-1:0][ELEMENT_WIDTH-1:0] selected_head,
    output logic selector_valid
);
{implementation}
endmodule
"""


def _tcl_quote(path: Path) -> str:
    text = str(path)
    if "{" in text or "}" in text or "\n" in text:
        raise ValueError(f"path cannot be represented safely in Tcl: {path}")
    return "{" + text + "}"


def _render_library_setup(
    path: Path,
    library_manifest: Mapping[str, Any],
) -> None:
    libraries = [
        Path(str(record["path"])).resolve()
        for record in library_manifest["files"]
    ]
    lines = ["set calibration_target_libraries [list \\"]
    for library in libraries:
        lines.append(f"    {_tcl_quote(library)} \\")
    lines.extend(
        (
            "]",
            "set target_library $calibration_target_libraries",
            'set link_library [concat [list "*"] '
            "$calibration_target_libraries]",
            "if {[info exists synthetic_library] "
            '&& $synthetic_library ne ""} {',
            "    set link_library [concat $link_library "
            "$synthetic_library]",
            "}",
            "foreach calibration_library "
            "$calibration_target_libraries {",
            "    set calibration_library_dir "
            "[file dirname $calibration_library]",
            "    if {[lsearch -exact $search_path "
            "$calibration_library_dir] < 0} {",
            "        lappend search_path $calibration_library_dir",
            "    }",
            "}",
        )
    )
    _write_bytes(path, ("\n".join(lines) + "\n").encode())


def _render_rtl_filelist(
    rtl_root: Path,
    run_dir: Path,
    *,
    selector_only: bool,
) -> Path:
    definitions = rtl_root / "src" / "definitions"
    if selector_only:
        files = (
            rtl_root
            / "src"
            / "basic_components"
            / "systolic_gemm_mxint"
            / "rtl"
            / "packed_kv_head_selector.sv",
            run_dir / "calibration_selector_top.sv",
        )
        search_paths = (
            run_dir,
            files[0].parent,
        )
    else:
        synthesis_directories = tuple(
            (rtl_root / "src" / relative).resolve()
            for relative in _SYNTHESIS_RELATIVE_DIRS
        )
        directory_order = {
            directory: index
            for index, directory in enumerate(synthesis_directories)
        }
        candidates = [
            path
            for path in _rtl_source_files(rtl_root)
            if (
                path.suffix in {".sv", ".v"}
                and path.parent.resolve() in directory_order
                and path.name not in _SKIP_SYNTHESIS_RTL
                and path.stat().st_size > 0
            )
        ]
        # Analyze in declared directory order rather than by path. A package
        # such as tl_pkg reaches the work library only through the `include`
        # inside the first analyzed file that uses it, so TileLink_Lib must
        # precede core/rtl, whose top-level ports expand tl_pkg types.
        original = sorted(
            candidates,
            key=lambda path: (directory_order[path.parent.resolve()], path.name),
        )
        files = (
            definitions / "global_define.vh",
            run_dir / "definitions" / "precision.svh",
            run_dir / "definitions" / "configuration.svh",
            definitions / "operation.svh",
            *(
                _require_source_file(
                    rtl_root / "src" / relative,
                    "analyzed package header",
                )
                for relative in _ANALYZED_PACKAGE_HEADERS
            ),
            *original,
        )
        search_paths = (
            run_dir / "definitions",
            definitions,
            *synthesis_directories,
        )
    lines = ["set calibration_search_paths [list \\"]
    for directory in search_paths:
        resolved = directory.resolve()
        try:
            relative = resolved.relative_to(run_dir.resolve())
        except ValueError:
            rendered = _tcl_quote(resolved)
        else:
            rendered = (
                '"$::env(CAL_RUN_DIR)/'
                + relative.as_posix()
                + '"'
            )
        lines.append(f"    {rendered} \\")
    lines.extend(
        (
            "]",
            "set search_path "
            "[concat $calibration_search_paths $search_path]",
        )
    )
    lines.append("set calibration_rtl_files [list \\")
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(run_dir.resolve())
        except ValueError:
            rendered = _tcl_quote(resolved)
        else:
            rendered = (
                '"$::env(CAL_RUN_DIR)/'
                + relative.as_posix()
                + '"'
            )
        lines.append(f"    {rendered} \\")
    lines.append("]")
    target = run_dir / "rtl_files.tcl"
    _write_bytes(target, ("\n".join(lines) + "\n").encode())
    return target


def _render_constraints(path: Path, *, clock_ns: float = 1.0) -> None:
    if not math.isclose(clock_ns, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("power calibration uses one common 1 ns constraint")
    text = """set_units -time ns
set calibration_clock_period_ns 1.0
set clock_ports [get_ports -quiet clk]
if {[sizeof_collection $clock_ports] > 0} {
    create_clock -period $calibration_clock_period_ns $clock_ports
    set_clock_uncertainty -setup 0.1 [get_clocks clk]
    set_drive 0 $clock_ports
    set non_clock_inputs [remove_from_collection [all_inputs] $clock_ports]
    if {[sizeof_collection $non_clock_inputs] > 0} {
        set_input_delay 0.08 -clock clk $non_clock_inputs
    }
    set_output_delay 0.05 -clock clk [all_outputs]
} else {
    set_max_delay $calibration_clock_period_ns \
        -from [all_inputs] -to [all_outputs]
}
set_load 0.005 [all_outputs]
set_max_transition 0.5 [current_design]
"""
    _write_bytes(path, text.encode())


def _artifact_record(path: Path, kind: str, root: Path) -> dict[str, Any]:
    source = _require_file(path, kind)
    relative = source.relative_to(root)
    return {
        "kind": kind,
        "path": relative.as_posix(),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _run_command(
    *,
    synopsys_dir: Path,
    environment_script: Path,
    dc_setup: Path,
    tcl_script: Path,
    point_id: str,
    top_module: str,
    operating_condition: str,
    requires_saif: bool,
    blackboxes: Sequence[str],
    expected_tool_version: str,
    source_tree_sha256: str,
    constraints_sha256: str,
    library_manifest_sha256: str,
    dc_setup_sha256: str,
    library_setup_sha256: str,
    compiler_binding_sha256: str | None = None,
) -> str:
    q = shlex.quote
    saif_setup = ""
    saif_bindings = (
        'saif_sha="$(sha256sum "$saif_path" | awk \'{print $1}\')"\n'
        'trace_sha="$(sha256sum "$trace_path" | awk \'{print $1}\')"\n'
        if requires_saif
        else 'saif_sha=""\ntrace_sha=""\n'
    )
    if requires_saif:
        saif_setup = """saif_path="${run_dir}/inputs/activity.saif"
trace_path="${run_dir}/inputs/decode_trace.json"
test -s "$saif_path"
test -s "$trace_path"
export CAL_SAIF_PATH="$saif_path"
export CAL_SAIF_INSTANCE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["saif_instance"])' "$trace_path")"
"""
    blackbox_value = ",".join(blackboxes)
    compiler_line = (
        f'printf "COMPILER_BINDING_SHA256: {compiler_binding_sha256}\\n"\n'
        if compiler_binding_sha256
        else ""
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
log_path="$run_dir/outputs/logs/dc.log"
success_path="$run_dir/outputs/DC_SUCCESS"
if [ -e "$success_path" ]; then
    echo "Refusing to overwrite a completed run: $run_dir" >&2
    exit 3
fi
mkdir -p "$run_dir/outputs/logs" "$run_dir/outputs/reports" \
    "$run_dir/outputs/netlist" "$run_dir/work"
exec > >(tee "$log_path") 2>&1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV _PYTHON_SYSCONFIGDATA_NAME \
    _PYTHON_HOST_PLATFORM PYTHONNOUSERSITE PYTHONHASHSEED
source {q(str(environment_script))}
observed_version="$(dc_shell -version 2>&1 || true)"
case "$observed_version" in
    *{q(expected_tool_version)}*) ;;
    *)
        echo "Unexpected Design Compiler version: $observed_version" >&2
        exit 4
        ;;
esac
{saif_setup}{saif_bindings}export CAL_RUN_DIR="$run_dir"
export CAL_TOP_MODULE={q(top_module)}
export CAL_RTL_FILELIST="$run_dir/rtl_files.tcl"
export CAL_CONSTRAINTS="$run_dir/constraints.sdc"
export CAL_DC_SETUP={q(str(dc_setup))}
export CAL_LIBRARY_SETUP="$run_dir/library_setup.tcl"
export CAL_OPERATING_CONDITION={q(operating_condition)}
export CAL_POINT_ID={q(point_id)}
export CAL_CLOCK_PERIOD_NS=1.0
export CAL_REQUIRES_SAIF={"1" if requires_saif else "0"}
export CAL_COMPILE_MODE=normal
export CAL_BLACKBOX_DESIGNS={q(blackbox_value)}
cd {q(str(synopsys_dir))}
dc_shell -no_init -f {q(str(tcl_script))} < /dev/null
netlist_path="$run_dir/outputs/netlist/{top_module}_mapped.v"
test -s "$netlist_path"
netlist_sha="$(sha256sum "$netlist_path" | awk '{{print $1}}')"
printf "RTL_SOURCE_TREE_SHA256: {source_tree_sha256}\\n"
printf "CONSTRAINTS_SHA256: {constraints_sha256}\\n"
printf "LIBRARY_MANIFEST_SHA256: {library_manifest_sha256}\\n"
printf "DC_SETUP_SHA256: {dc_setup_sha256}\\n"
printf "LIBRARY_SETUP_SHA256: {library_setup_sha256}\\n"
if [ -n "$saif_sha" ]; then
    printf "SAIF_SHA256: %s\\n" "$saif_sha"
    printf "DECODE_TRACE_SHA256: %s\\n" "$trace_sha"
fi
{compiler_line}printf "NETLIST_SHA256: %s\\n" "$netlist_sha"
: > "$success_path"
"""


def _run_scope(point: CalibrationPoint) -> tuple[str | None, bool, tuple[str, ...]]:
    if point.component == "array":
        return "matrix_machine", True, ()
    if point.component == "vector":
        return "vector_machine", True, ()
    if point.component == "selector":
        return "calibration_selector_top", True, ()
    if point.component == "fixed":
        return "plena", False, _FIXED_BLACKBOXES
    if point.component == "chip_leakage":
        return "plena", False, ()
    if point.component in _TRACE_COMPONENTS:
        return None, False, ()
    raise ValueError(f"unsupported calibration component {point.component!r}")


def prepare_model_workflow(
    output: str | Path,
    *,
    rtl_root: str | Path,
    library_manifest: str | Path,
    synopsys_environment: str | Path,
    dc_tool_version: str,
    synopsys_setup: str | Path | None = None,
) -> Path:
    """Create every scheduled run without executing synthesis."""

    target = _ensure_empty_target(output)
    rtl = Path(rtl_root).resolve()
    library = _load_library_manifest(library_manifest)
    environment = _require_file(
        synopsys_environment,
        "Synopsys environment script",
    )
    setup = _require_file(
        synopsys_setup or rtl / "tools" / "synopsys" / ".synopsys_dc.setup",
        "Design Compiler setup",
    )
    synopsys_dir = setup.parent
    tcl_script = _require_file(
        rtl / "tools" / "synopsys" / "power_calibration.tcl",
        "power calibration Tcl",
    )
    if not dc_tool_version.strip():
        raise ValueError("Design Compiler version identity is required")

    rtl_manifest = build_rtl_source_manifest(rtl)
    context = target / "context"
    context.mkdir(parents=True, exist_ok=True)
    _write_bytes(
        context / "rtl_source_manifest.json",
        (json.dumps(rtl_manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _copy_verified(
        Path(library_manifest).resolve(),
        context / "library_manifest.json",
    )
    _render_library_setup(context / "library_setup.tcl", library)
    _render_constraints(context / "constraints.sdc")
    tool_body = {
        "schema_version": TOOL_CONTEXT_SCHEMA,
        "dc_tool_version": dc_tool_version.strip(),
        "environment_script": str(environment),
        "environment_script_sha256": _sha256_file(environment),
        "dc_setup": str(setup),
        "dc_setup_sha256": _sha256_file(setup),
        "calibration_tcl": str(tcl_script),
        "calibration_tcl_sha256": _sha256_file(tcl_script),
        "library_setup_sha256": _sha256_file(
            context / "library_setup.tcl"
        ),
        "library_id": library["library_id"],
        "process_corner": library["process_corner"],
        "operating_condition": library["operating_condition"],
        "clock_period_ns": 1.0,
        "area_unit": "um2",
    }
    _write_json(context / "tool_context.json", tool_body)

    manifest_body = manifest_payload()
    _write_bytes(
        target / "calibration_manifest.json",
        (json.dumps(manifest_body, indent=2, sort_keys=True) + "\n").encode(),
    )
    with (target / "measurements.template.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for point in build_manifest():
            writer.writerow(
                {
                    "status": point.status,
                    "point_id": point.point_id,
                    "split": point.split,
                    "component": point.component,
                    "signature": point.signature,
                    "MLEN": point.mlen,
                    "BLEN": point.blen,
                    "selector_enabled": point.selector_enabled,
                    "clock_ns": point.clock_ns,
                    "MX_BLOCK_SIZE": point.mx_block_size,
                    "hardware_fp_binding": point.hardware_fp_binding,
                    "activity_class": point.activity_class or "",
                }
            )

    commands = []
    constraints_sha = _sha256_file(context / "constraints.sdc")
    library_sha = _sha256_file(context / "library_manifest.json")
    dc_setup_sha = _sha256_file(setup)
    library_setup_sha = _sha256_file(context / "library_setup.tcl")
    selector_compatible: dict[tuple[int, int], list[str]] = {}
    for scheduled in build_manifest():
        if scheduled.component == "selector":
            selector_compatible.setdefault(
                (scheduled.mlen, scheduled.blen),
                [],
            ).append(scheduled.point_id)
    selector_compatible = {
        geometry: sorted(point_ids)
        for geometry, point_ids in selector_compatible.items()
    }
    for point in build_manifest():
        run_dir = target / "runs" / point.point_id
        run_dir.mkdir(parents=True)
        top_module, requires_saif, blackboxes = _run_scope(point)
        formats = _point_formats(point)
        hlen = point.blen
        definition_hashes: dict[str, str] = {}
        if point.component in _DC_COMPONENTS:
            precision, configuration = _render_definitions(
                rtl,
                run_dir,
                formats=formats,
                mlen=point.mlen,
                blen=point.blen,
                hlen=hlen,
            )
            definition_hashes = {
                "precision.svh": _sha256_file(precision),
                "configuration.svh": _sha256_file(configuration),
            }
            selector_only = point.component == "selector"
            if selector_only:
                wrapper = _selector_wrapper(
                    enabled=point.selector_enabled,
                    mlen=point.mlen,
                    hlen=hlen,
                    element_width=4,
                )
                _write_bytes(
                    run_dir / "calibration_selector_top.sv",
                    wrapper.encode(),
                )
            filelist = _render_rtl_filelist(
                rtl,
                run_dir,
                selector_only=selector_only,
            )
            _copy_verified(
                context / "constraints.sdc",
                run_dir / "constraints.sdc",
            )
            _copy_verified(
                context / "library_setup.tcl",
                run_dir / "library_setup.tcl",
            )
            run_spec_body = {
                "schema_version": RUN_SPEC_SCHEMA,
                "calibration_manifest_hash": manifest_hash(),
                "point": asdict(point),
                "top_module": top_module,
                "formats": dict(formats),
                "geometry": {
                    "MLEN": point.mlen,
                    "BLEN": point.blen,
                    "HLEN": hlen,
                    "VLEN": point.mlen,
                },
                "clock_period_ns": 1.0,
                "compile_mode": "normal",
                "requires_saif": requires_saif,
                "blackbox_designs": list(blackboxes),
                "rtl_source_tree_sha256": rtl_manifest[
                    "source_tree_sha256"
                ],
                "definition_sha256": definition_hashes,
                "rtl_filelist_sha256": _sha256_file(filelist),
                "constraints_sha256": constraints_sha,
                "library_manifest_sha256": library_sha,
                "dc_setup_sha256": dc_setup_sha,
                "library_setup_sha256": library_setup_sha,
                "dc_tool_version": dc_tool_version.strip(),
                "library_id": library["library_id"],
                "process_corner": library["process_corner"],
                "operating_condition": library["operating_condition"],
                "measurement_state": "scheduled",
            }
            _write_json(run_dir / "run_spec.json", run_spec_body)
            if requires_saif:
                activity_body = {
                    "schema_version": ACTIVITY_REQUEST_SCHEMA,
                    "point_id": point.point_id,
                    "compatible_point_ids": (
                        selector_compatible[(point.mlen, point.blen)]
                        if point.component == "selector"
                        else [point.point_id]
                    ),
                    "component": point.component,
                    "signature": point.signature,
                    "activity_class": point.activity_class,
                    "geometry": {
                        "MLEN": point.mlen,
                        "BLEN": point.blen,
                    },
                    "formats": dict(formats),
                    "clock_period_ns": 1.0,
                    "required_outputs": {
                        "saif": "inputs/activity.saif",
                        "decode_trace": "inputs/decode_trace.json",
                    },
                    "decode_trace_schema": ACTIVITY_TRACE_SCHEMA,
                    "measurement_state": "awaiting_activity",
                }
                _write_json(
                    run_dir / "activity_request.json",
                    activity_body,
                )
            run_script = _run_command(
                synopsys_dir=synopsys_dir,
                environment_script=environment,
                dc_setup=setup,
                tcl_script=tcl_script,
                point_id=point.point_id,
                top_module=str(top_module),
                operating_condition=library["operating_condition"],
                requires_saif=requires_saif,
                blackboxes=blackboxes,
                expected_tool_version=dc_tool_version.strip(),
                source_tree_sha256=rtl_manifest["source_tree_sha256"],
                constraints_sha256=constraints_sha,
                library_manifest_sha256=library_sha,
                dc_setup_sha256=dc_setup_sha,
                library_setup_sha256=library_setup_sha,
            )
            _write_bytes(
                run_dir / "run.sh",
                run_script.encode(),
                executable=True,
            )
            commands.append(f"runs/{point.point_id}/run.sh")
        else:
            trace_body = {
                "schema_version": TRACE_PAIR_SCHEMA,
                "point": asdict(point),
                "required_outputs": (
                    {
                        "rtl_trace": "inputs/rtl_trace.json",
                        "emulator_trace": "inputs/emulator_trace.json",
                    }
                    if point.component == "cycle"
                    else {
                        "measured_trace": "inputs/measured_trace.json",
                        "analytical_trace": "inputs/analytical_trace.json",
                    }
                ),
                "measurement_state": "awaiting_traces",
            }
            _write_json(run_dir / "trace_request.json", trace_body)

    workflow_body = {
        "schema_version": MODEL_WORKFLOW_SCHEMA,
        "calibration_manifest_schema": CALIBRATION_MANIFEST_SCHEMA,
        "calibration_manifest_hash": manifest_hash(),
        "point_count": len(build_manifest()),
        "dc_run_count": sum(
            point.component in _DC_COMPONENTS for point in build_manifest()
        ),
        "trace_point_count": sum(
            point.component in _TRACE_COMPONENTS for point in build_manifest()
        ),
        "requested_vector_fp": [
            value for value in VECTOR_FP if value != "BF16"
        ],
        "bf16_vector_control": "BF16",
        "rtl_root": str(rtl),
        "rtl_source_tree_sha256": rtl_manifest["source_tree_sha256"],
        "dc_tool_version": dc_tool_version.strip(),
        "library_id": library["library_id"],
        "process_corner": library["process_corner"],
        "operating_condition": library["operating_condition"],
        "clock_period_ns": 1.0,
        "mx_block_size": MX_BLOCK_SIZE,
        "measurement_state": "scheduled",
        "commands": commands,
    }
    _write_json(target / "workflow_manifest.json", workflow_body)
    return target


def _one_match(pattern: str, text: str, label: str) -> tuple[str, ...]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{label} must occur exactly once")
    value = matches[0]
    return (value,) if isinstance(value, str) else tuple(value)


def _parse_dc_report(path: Path) -> dict[str, float]:
    text = _require_file(path, "DC report").read_text(
        encoding="utf-8",
        errors="strict",
    )
    area = float(
        _one_match(
            rf"^\s*Total\s+cell\s+area\s*:\s*{_NUMBER}\s*$",
            text,
            "total cell area",
        )[0]
    )
    dynamic_value, dynamic_unit = _one_match(
        rf"^\s*Total\s+Dynamic\s+Power\s*=\s*{_NUMBER}\s*"
        rf"(W|mW|uW|nW)(?:\s+\([^)]*\))?\s*$",
        text,
        "total dynamic power",
    )
    leakage_value, leakage_unit = _one_match(
        rf"^\s*Cell\s+Leakage\s+Power\s*=\s*{_NUMBER}\s*"
        rf"(W|mW|uW|nW)(?:\s+\([^)]*\))?\s*$",
        text,
        "cell leakage power",
    )
    values = {
        "area_mm2": area / 1e6,
        "dynamic_power_w": (
            float(dynamic_value) * _POWER_SCALE[dynamic_unit]
        ),
        "leakage_power_w": (
            float(leakage_value) * _POWER_SCALE[leakage_unit]
        ),
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("DC area and power values must be finite and positive")
    return values


def _parse_timing_report(path: Path) -> dict[str, float]:
    text = _require_file(path, "timing report").read_text(
        encoding="utf-8",
        errors="strict",
    )
    period = float(
        _one_match(
            rf"^\s*Clock\s+period\s*:\s*{_NUMBER}\s*ns\s*$",
            text,
            "clock period",
        )[0]
    )
    met = re.findall(
        rf"^\s*slack\s+\(MET\)\s+{_NUMBER}\s*$",
        text,
        flags=re.MULTILINE,
    )
    violated = re.findall(
        rf"^\s*slack\s+\(VIOLATED\)\s+{_NUMBER}\s*$",
        text,
        flags=re.MULTILINE,
    )
    if len(met) + len(violated) != 1:
        raise ValueError("timing slack must occur exactly once")
    slack = float((met or violated)[0])
    if (
        not math.isclose(period, 1.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isfinite(slack)
        or slack < 0
        or violated
    ):
        raise ValueError("DC point does not meet the common 1 ns constraint")
    return {"clock_period_ns": period, "worst_slack_ns": slack}


def _log_binding(path: Path, label: str) -> str:
    text = _require_file(path, "synthesis log").read_text(
        encoding="utf-8",
        errors="strict",
    )
    return _one_match(
        rf"^\s*{re.escape(label)}\s*:\s*([0-9a-f]{{64}})\s*$",
        text,
        label,
    )[0]


def _load_activity_trace(
    path: Path,
    *,
    point: CalibrationPoint,
    saif_sha256: str,
    rtl_source_tree_sha256: str,
) -> dict[str, Any]:
    trace = _load_hashed_json(path, schema=ACTIVITY_TRACE_SCHEMA)
    if set(trace) != {
        "schema_version",
        "compatible_point_ids",
        "activity_class",
        "events",
        "cycles",
        "clock_ns",
        "saif_sha256",
        "saif_source_id",
        "activity_generator",
        "saif_instance",
        "rtl_source_tree_sha256",
        "stimulus_id",
        "content_hash",
    }:
        raise ValueError("decode trace fields differ from the schema")
    compatible = trace["compatible_point_ids"]
    if (
        not isinstance(compatible, list)
        or point.point_id not in compatible
        or len(compatible) != len(set(compatible))
    ):
        raise ValueError("decode trace does not cover this calibration point")
    if trace["activity_class"] != point.activity_class:
        raise ValueError("decode trace activity class mismatch")
    for name in ("events", "cycles"):
        value = trace[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"decode trace {name} must be a positive integer")
    if not math.isclose(
        float(trace["clock_ns"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("decode trace clock differs from 1 ns")
    if trace["saif_sha256"] != saif_sha256:
        raise ValueError("decode trace SAIF hash mismatch")
    if trace["rtl_source_tree_sha256"] != rtl_source_tree_sha256:
        raise ValueError("decode trace RTL source-tree mismatch")
    for name in (
        "saif_source_id",
        "activity_generator",
        "saif_instance",
        "stimulus_id",
    ):
        if not str(trace[name]).strip():
            raise ValueError(f"decode trace {name} must be non-empty")
    return trace


def _load_scalar_trace(
    path: Path,
    *,
    point_id: str,
    metric: str,
) -> float:
    trace = _load_hashed_json(path)
    if set(trace) != {
        "schema_version",
        "point_id",
        metric,
        "source_id",
        "command",
        "source_sha256",
        "content_hash",
    }:
        raise ValueError(f"{metric} trace fields differ from the schema")
    if trace["point_id"] != point_id:
        raise ValueError(f"{metric} trace point identity mismatch")
    value = trace[metric]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{metric} must be finite and positive")
    if not str(trace["source_id"]).strip() or not str(trace["command"]).strip():
        raise ValueError(f"{metric} trace provenance is incomplete")
    _validate_sha256(trace["source_sha256"], f"{metric} source")
    return float(value)


def _validate_run_spec(
    run_dir: Path,
    point: CalibrationPoint,
    workflow: Mapping[str, Any],
    rtl_manifest: Mapping[str, Any],
    library_sha256: str,
    constraints_sha256: str,
    dc_setup_sha256: str,
    library_setup_sha256: str,
) -> dict[str, Any]:
    spec = _load_hashed_json(
        run_dir / "run_spec.json",
        schema=RUN_SPEC_SCHEMA,
    )
    if spec["point"] != asdict(point):
        raise ValueError("run specification point mismatch")
    if spec["measurement_state"] != "scheduled":
        raise ValueError("prepared run cannot claim a measured state")
    if spec["calibration_manifest_hash"] != manifest_hash():
        raise ValueError("run calibration manifest mismatch")
    if (
        spec["rtl_source_tree_sha256"]
        != rtl_manifest["source_tree_sha256"]
        or spec["library_manifest_sha256"] != library_sha256
        or spec["constraints_sha256"] != constraints_sha256
        or spec["dc_setup_sha256"] != dc_setup_sha256
        or spec["library_setup_sha256"] != library_setup_sha256
        or spec["dc_tool_version"] != workflow["dc_tool_version"]
        or spec["library_id"] != workflow["library_id"]
        or spec["process_corner"] != workflow["process_corner"]
        or spec["operating_condition"] != workflow["operating_condition"]
        or float(spec["clock_period_ns"]) != 1.0
    ):
        raise ValueError("run specification context mismatch")
    for name, digest in spec["definition_sha256"].items():
        source = run_dir / "definitions" / name
        if _sha256_file(_require_file(source, "generated definition")) != digest:
            raise ValueError("generated definition changed after preparation")
    if _sha256_file(run_dir / "rtl_files.tcl") != spec[
        "rtl_filelist_sha256"
    ]:
        raise ValueError("RTL file list changed after preparation")
    if _sha256_file(
        _require_file(run_dir / "library_setup.tcl", "library setup")
    ) != library_setup_sha256:
        raise ValueError("library setup changed after preparation")
    return spec


def ingest_model_workflow(
    workflow_root: str | Path,
    *,
    measurements_output: str | Path,
    artifact_catalog_output: str | Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Revalidate real outputs and populate existing fit inputs."""

    root = Path(workflow_root).resolve()
    workflow = _load_hashed_json(
        root / "workflow_manifest.json",
        schema=MODEL_WORKFLOW_SCHEMA,
    )
    if (
        workflow["calibration_manifest_hash"] != manifest_hash()
        or int(workflow["point_count"]) != len(build_manifest())
        or workflow["measurement_state"] != "scheduled"
    ):
        raise ValueError("workflow manifest differs from the power schedule")
    rtl_manifest = _load_hashed_json(
        root / "context" / "rtl_source_manifest.json",
        schema=RTL_SOURCE_MANIFEST_SCHEMA,
    )
    _validate_rtl_source_manifest(
        rtl_manifest,
        Path(workflow["rtl_root"]).resolve(),
    )
    library = _load_library_manifest(
        root / "context" / "library_manifest.json"
    )
    if (
        library["library_id"] != workflow["library_id"]
        or library["process_corner"] != workflow["process_corner"]
        or library["operating_condition"] != workflow["operating_condition"]
    ):
        raise ValueError("workflow and library identities differ")
    tool_context = _load_hashed_json(
        root / "context" / "tool_context.json",
        schema=TOOL_CONTEXT_SCHEMA,
    )
    for name in ("environment_script", "dc_setup", "calibration_tcl"):
        source = _require_file(tool_context[name], name)
        if _sha256_file(source) != tool_context[f"{name}_sha256"]:
            raise ValueError(f"{name} changed after preparation")
    library_setup = _require_file(
        root / "context" / "library_setup.tcl",
        "library setup",
    )
    library_setup_sha = _sha256_file(library_setup)
    if library_setup_sha != tool_context["library_setup_sha256"]:
        raise ValueError("library setup changed after preparation")
    dc_setup_sha = tool_context["dc_setup_sha256"]
    constraints = _require_file(
        root / "context" / "constraints.sdc",
        "common constraints",
    )
    constraints_sha = _sha256_file(constraints)
    library_path = root / "context" / "library_manifest.json"
    library_sha = _sha256_file(library_path)

    rows: list[dict[str, Any]] = []
    catalog_records: list[dict[str, Any]] = []
    synthesis_logs: list[dict[str, str]] = []
    incomplete: list[str] = []
    selector_activity: dict[
        tuple[int, int], tuple[str, str]
    ] = {}

    for point in build_manifest():
        run_dir = root / "runs" / point.point_id
        base = {
            "status": "scheduled",
            "point_id": point.point_id,
            "split": point.split,
            "component": point.component,
            "signature": point.signature,
            "MLEN": point.mlen,
            "BLEN": point.blen,
            "selector_enabled": point.selector_enabled,
            "clock_ns": point.clock_ns,
            "MX_BLOCK_SIZE": point.mx_block_size,
            "hardware_fp_binding": point.hardware_fp_binding,
            "activity_class": point.activity_class or "",
        }
        if point.component in _DC_COMPONENTS:
            _validate_run_spec(
                run_dir,
                point,
                workflow,
                rtl_manifest,
                library_sha,
                constraints_sha,
                dc_setup_sha,
                library_setup_sha,
            )
            if not (run_dir / "outputs" / "DC_SUCCESS").is_file():
                incomplete.append(point.point_id)
                rows.append(base)
                continue
            top_module, requires_saif, _ = _run_scope(point)
            dc_report = _require_file(
                run_dir / "outputs" / "reports" / "dc_report.rpt",
                "DC report",
            )
            timing_report = _require_file(
                run_dir / "outputs" / "reports" / "timing.rpt",
                "timing report",
            )
            synthesis_log = _require_file(
                run_dir / "outputs" / "logs" / "dc.log",
                "synthesis log",
            )
            netlist = _require_file(
                run_dir
                / "outputs"
                / "netlist"
                / f"{top_module}_mapped.v",
                "mapped netlist",
            )
            metrics = _parse_dc_report(dc_report)
            _parse_timing_report(timing_report)
            expected_log = {
                "RTL_SOURCE_TREE_SHA256": rtl_manifest[
                    "source_tree_sha256"
                ],
                "CONSTRAINTS_SHA256": constraints_sha,
                "LIBRARY_MANIFEST_SHA256": library_sha,
                "DC_SETUP_SHA256": dc_setup_sha,
                "LIBRARY_SETUP_SHA256": library_setup_sha,
                "NETLIST_SHA256": _sha256_file(netlist),
            }
            for label, expected in expected_log.items():
                if _log_binding(synthesis_log, label) != expected:
                    raise ValueError(
                        f"synthesis log binding mismatch for {point.point_id}"
                    )
            artifacts = [
                _artifact_record(dc_report, "dc_report", root),
                _artifact_record(synthesis_log, "synthesis_log", root),
            ]
            row = {
                **base,
                "status": "complete",
                "dc_tool_version": workflow["dc_tool_version"],
                "library_id": workflow["library_id"],
                "process_corner": workflow["process_corner"],
            }
            if point.component in {"array", "vector", "fixed", "selector"}:
                row["area_mm2"] = metrics["area_mm2"]
            if point.component == "chip_leakage":
                row["leakage_power_w"] = metrics["leakage_power_w"]
            if requires_saif:
                saif = _require_file(
                    run_dir / "inputs" / "activity.saif",
                    "SAIF",
                )
                trace_path = _require_file(
                    run_dir / "inputs" / "decode_trace.json",
                    "decode trace",
                )
                saif_sha = _sha256_file(saif)
                trace = _load_activity_trace(
                    trace_path,
                    point=point,
                    saif_sha256=saif_sha,
                    rtl_source_tree_sha256=rtl_manifest[
                        "source_tree_sha256"
                    ],
                )
                trace_sha = _sha256_file(trace_path)
                if (
                    _log_binding(synthesis_log, "SAIF_SHA256")
                    != saif_sha
                    or _log_binding(
                        synthesis_log,
                        "DECODE_TRACE_SHA256",
                    )
                    != trace_sha
                ):
                    raise ValueError(
                        f"activity log binding mismatch for {point.point_id}"
                    )
                row.update(
                    events=trace["events"],
                    cycles=trace["cycles"],
                    dynamic_power_w=metrics["dynamic_power_w"],
                    activity_class=trace["activity_class"],
                    saif_sha256=saif_sha,
                    decode_trace_sha256=trace_sha,
                    saif_source_id=trace["saif_source_id"],
                    activity_generator=trace["activity_generator"],
                )
                artifacts.extend(
                    (
                        _artifact_record(saif, "saif", root),
                        _artifact_record(
                            trace_path,
                            "decode_trace",
                            root,
                        ),
                    )
                )
                if point.component == "selector":
                    key = (point.mlen, point.blen)
                    identity = (saif_sha, trace_sha)
                    previous = selector_activity.setdefault(key, identity)
                    if previous != identity:
                        raise ValueError(
                            "selector off/on runs must use identical activity"
                        )
            if set(item["kind"] for item in artifacts) != (
                POINT_ARTIFACT_KINDS[point.component]
            ):
                raise ValueError("DC artifact coverage differs from the catalog")
            catalog_records.append(
                {"point_id": point.point_id, "artifacts": artifacts}
            )
            synthesis_logs.append(
                {
                    "point_id": point.point_id,
                    "sha256": _sha256_file(synthesis_log),
                }
            )
            rows.append(row)
            continue

        request = _load_hashed_json(run_dir / "trace_request.json")
        if request["point"] != asdict(point):
            raise ValueError("trace request point mismatch")
        required = request["required_outputs"]
        paths = {
            kind: run_dir / relative for kind, relative in required.items()
        }
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
            incomplete.append(point.point_id)
            rows.append(base)
            continue
        artifacts = []
        row = {**base, "status": "complete"}
        if point.component == "cycle":
            row["rtl_cycles"] = _load_scalar_trace(
                paths["rtl_trace"],
                point_id=point.point_id,
                metric="cycles",
            )
            row["emulator_cycles"] = _load_scalar_trace(
                paths["emulator_trace"],
                point_id=point.point_id,
                metric="cycles",
            )
        else:
            row["measured_latency_s"] = _load_scalar_trace(
                paths["measured_trace"],
                point_id=point.point_id,
                metric="latency_s",
            )
            row["analytical_latency_s"] = _load_scalar_trace(
                paths["analytical_trace"],
                point_id=point.point_id,
                metric="latency_s",
            )
        for kind, path in paths.items():
            artifacts.append(_artifact_record(path, kind, root))
        if set(item["kind"] for item in artifacts) != (
            POINT_ARTIFACT_KINDS[point.component]
        ):
            raise ValueError("trace artifact coverage differs from the catalog")
        catalog_records.append(
            {"point_id": point.point_id, "artifacts": artifacts}
        )
        rows.append(row)

    if require_complete and incomplete:
        raise RuntimeError(
            f"{len(incomplete)} calibration points remain incomplete"
        )
    tool_log_body = {
        "schema_version": TOOL_LOG_INDEX_SCHEMA,
        "dc_tool_version": workflow["dc_tool_version"],
        "logs": sorted(synthesis_logs, key=lambda value: value["point_id"]),
    }
    _write_json(root / "context" / "tool_log_index.json", tool_log_body)
    context_artifacts = [
        _artifact_record(constraints, "constraints", root),
        _artifact_record(library_path, "library_manifest", root),
        _artifact_record(
            root / "context" / "rtl_source_manifest.json",
            "rtl_source_manifest",
            root,
        ),
        _artifact_record(
            root / "context" / "tool_log_index.json",
            "tool_log",
            root,
        ),
    ]
    if set(item["kind"] for item in context_artifacts) != (
        CONTEXT_ARTIFACT_KINDS
    ):
        raise ValueError("context artifact coverage differs from the catalog")
    catalog_body = {
        "schema_version": ARTIFACT_CATALOG_SCHEMA,
        "context_artifacts": context_artifacts,
        "records": sorted(
            catalog_records,
            key=lambda value: value["point_id"],
        ),
    }
    catalog_target = Path(artifact_catalog_output).resolve()
    _write_json(catalog_target, catalog_body)
    measurements_target = Path(measurements_output).resolve()
    measurements_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = measurements_target.with_suffix(
        measurements_target.suffix + ".tmp"
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(measurements_target)
    complete_rows = [row for row in rows if row["status"] == "complete"]
    validate_artifact_catalog(catalog_target, complete_rows)
    return {
        "point_count": len(rows),
        "complete_count": len(complete_rows),
        "incomplete_count": len(incomplete),
        "incomplete_point_ids": incomplete,
        "measurements": str(measurements_target),
        "artifact_catalog": str(catalog_target),
    }


def _load_compiler_binding(path: str | Path) -> dict[str, Any]:
    source = _require_file(path, "compiler precision binding")
    value = _load_hashed_json(source, schema=COMPILER_BINDING_SCHEMA)
    required = {
        "schema_version",
        "profile_id",
        "profile",
        "target",
        "evidence_target",
        "matrix_binding_mode",
        "format_descriptors",
        "format_binding_ids",
        "runtime_precision_contract",
        "binding_id",
        "content_hash",
    }
    if set(value) != required:
        raise ValueError("compiler precision binding fields differ from the schema")
    body = {
        key: value[key]
        for key in value
        if key not in {"binding_id", "content_hash"}
    }
    if value["binding_id"] != "cpb-" + _content_hash(body):
        raise ValueError("compiler precision binding identity mismatch")
    if value["profile_id"] != "dqp-" + _content_hash(value["profile"]):
        raise ValueError("compiler profile identity mismatch")
    runtime = value["runtime_precision_contract"]
    if not isinstance(runtime, Mapping):
        raise ValueError("compiler runtime precision contract is missing")
    runtime_body = dict(runtime)
    runtime_hash = str(runtime_body.pop("content_hash", ""))
    if _content_hash(runtime_body) != runtime_hash:
        raise ValueError("compiler runtime precision contract hash mismatch")
    semantics = runtime.get("matrix_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("compiler matrix semantics are missing")
    semantics_body = dict(semantics)
    semantics_hash = str(semantics_body.pop("content_hash", ""))
    if _content_hash(semantics_body) != semantics_hash:
        raise ValueError("compiler matrix semantics hash mismatch")
    return value


def _load_exact_request(path: str | Path) -> dict[str, Any]:
    value = _load_hashed_json(
        _require_file(path, "exact DC request"),
        schema=EXACT_REQUEST_SCHEMA,
    )
    if set(value) != {
        "schema_version",
        "model_name",
        "model_revision",
        "candidate",
        "workload",
        "timing_evidence_id",
        "layout_id",
        "traffic_ledger_id",
        "top_module",
        "compiler_binding_path",
        "activity_class",
        "activity_coverage",
        "content_hash",
    }:
        raise ValueError("exact DC request fields differ from the schema")
    for name in (
        "model_name",
        "model_revision",
        "timing_evidence_id",
        "layout_id",
        "traffic_ledger_id",
        "activity_class",
    ):
        if not str(value[name]).strip():
            raise ValueError(f"exact DC request {name} is empty")
    value["top_module"] = _validate_identifier(
        value["top_module"],
        "exact top module",
    )
    candidate = value["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BATCH",
        "HBM_CHANNELS",
        "HBM_GENERATION",
        "CHIP_COUNT",
    }:
        raise ValueError("exact candidate fields differ from the schema")
    for name in (
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BATCH",
        "HBM_CHANNELS",
        "CHIP_COUNT",
    ):
        raw = candidate[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"exact candidate {name} is invalid")
    if (
        candidate["VLEN"] != candidate["MLEN"]
        or candidate["MLEN"] % candidate["BLEN"]
        or candidate["MLEN"] % candidate["HLEN"]
        or candidate["BLEN"] > candidate["HLEN"]
        or not str(candidate["HBM_GENERATION"]).strip()
    ):
        raise ValueError("exact candidate geometry is invalid")
    workload = value["workload"]
    if (
        not isinstance(workload, Mapping)
        or workload.get("scope") != "steady_state_cached_q1"
        or workload.get("query_length") != 1
        or workload.get("admission_included") is not False
    ):
        raise ValueError("exact workload must exclude decode admission")
    coverage = value["activity_coverage"]
    if (
        not isinstance(coverage, list)
        or set(coverage) != {"linear", "qk", "pv", "vector", "selector"}
        or len(coverage) != 5
    ):
        raise ValueError("exact activity coverage is incomplete")
    return value


def prepare_exact_workflow(
    output: str | Path,
    *,
    request: str | Path,
    rtl_root: str | Path,
    library_manifest: str | Path,
    synopsys_environment: str | Path,
    dc_tool_version: str,
    synopsys_setup: str | Path | None = None,
) -> Path:
    """Prepare one selected-candidate full-chip DC/SAIF anchor."""

    target = _ensure_empty_target(output)
    request_value = _load_exact_request(request)
    binding = _load_compiler_binding(request_value["compiler_binding_path"])
    candidate = request_value["candidate"]
    compiler_target = binding["target"]
    if (
        compiler_target.get("mlen") != candidate["MLEN"]
        or compiler_target.get("blen") != candidate["BLEN"]
        or compiler_target.get("hlen") != candidate["HLEN"]
        or compiler_target.get("block_size") != 8
        or compiler_target.get("packed_kv") is not True
    ):
        raise ValueError("compiler binding target differs from exact candidate")
    profile = binding["profile"]
    formats = {
        "weight": profile["weight_format"],
        "activation": profile["activation_format"],
        "key": profile["key_format"],
        "value": profile["value_format"],
        "vector": profile["vector_format"],
    }
    if formats["key"] != formats["value"]:
        raise ValueError("current exact RTL specialization requires K=V")

    rtl = Path(rtl_root).resolve()
    library = _load_library_manifest(library_manifest)
    environment = _require_file(
        synopsys_environment,
        "Synopsys environment script",
    )
    setup = _require_file(
        synopsys_setup or rtl / "tools" / "synopsys" / ".synopsys_dc.setup",
        "Design Compiler setup",
    )
    synopsys_dir = setup.parent
    tcl_script = _require_file(
        rtl / "tools" / "synopsys" / "power_calibration.tcl",
        "power calibration Tcl",
    )
    if not dc_tool_version.strip():
        raise ValueError("Design Compiler version identity is required")

    context = target / "context"
    context.mkdir(parents=True)
    rtl_manifest = build_rtl_source_manifest(rtl)
    _write_bytes(
        context / "rtl_source_manifest.json",
        (json.dumps(rtl_manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _copy_verified(
        Path(library_manifest).resolve(),
        context / "library_manifest.json",
    )
    _render_library_setup(context / "library_setup.tcl", library)
    _copy_verified(
        Path(request).resolve(),
        context / "exact_request.json",
    )
    _copy_verified(
        Path(request_value["compiler_binding_path"]).resolve(),
        context / "compiler_precision_binding.json",
    )
    _render_constraints(context / "constraints.sdc")
    tool_body = {
        "schema_version": TOOL_CONTEXT_SCHEMA,
        "dc_tool_version": dc_tool_version.strip(),
        "environment_script": str(environment),
        "environment_script_sha256": _sha256_file(environment),
        "dc_setup": str(setup),
        "dc_setup_sha256": _sha256_file(setup),
        "calibration_tcl": str(tcl_script),
        "calibration_tcl_sha256": _sha256_file(tcl_script),
        "library_setup_sha256": _sha256_file(
            context / "library_setup.tcl"
        ),
        "library_id": library["library_id"],
        "process_corner": library["process_corner"],
        "operating_condition": library["operating_condition"],
        "clock_period_ns": 1.0,
        "area_unit": "um2",
    }
    _write_json(context / "tool_context.json", tool_body)

    run_id = "exact-" + _content_hash(
        {
            "request_hash": request_value["content_hash"],
            "profile_id": binding["profile_id"],
            "candidate": candidate,
            "rtl_source_tree_sha256": rtl_manifest[
                "source_tree_sha256"
            ],
            "library_manifest_sha256": _sha256_file(
                context / "library_manifest.json"
            ),
        }
    )
    run_dir = target / "run"
    run_dir.mkdir()
    precision, configuration = _render_definitions(
        rtl,
        run_dir,
        formats=formats,
        mlen=candidate["MLEN"],
        blen=candidate["BLEN"],
        hlen=candidate["HLEN"],
    )
    filelist = _render_rtl_filelist(
        rtl,
        run_dir,
        selector_only=False,
    )
    _copy_verified(
        context / "constraints.sdc",
        run_dir / "constraints.sdc",
    )
    _copy_verified(
        context / "library_setup.tcl",
        run_dir / "library_setup.tcl",
    )
    constraints_sha = _sha256_file(context / "constraints.sdc")
    library_sha = _sha256_file(context / "library_manifest.json")
    dc_setup_sha = _sha256_file(setup)
    library_setup_sha = _sha256_file(context / "library_setup.tcl")
    compiler_sha = _sha256_file(
        context / "compiler_precision_binding.json"
    )
    run_spec_body = {
        "schema_version": RUN_SPEC_SCHEMA,
        "run_id": run_id,
        "kind": "selected_candidate_exact_anchor",
        "profile_id": binding["profile_id"],
        "binding_id": binding["binding_id"],
        "candidate": candidate,
        "workload": request_value["workload"],
        "top_module": request_value["top_module"],
        "formats": formats,
        "clock_period_ns": 1.0,
        "compile_mode": "normal",
        "requires_saif": True,
        "blackbox_designs": [],
        "rtl_source_tree_sha256": rtl_manifest[
            "source_tree_sha256"
        ],
        "definition_sha256": {
            "precision.svh": _sha256_file(precision),
            "configuration.svh": _sha256_file(configuration),
        },
        "rtl_filelist_sha256": _sha256_file(filelist),
        "constraints_sha256": constraints_sha,
        "library_manifest_sha256": library_sha,
        "dc_setup_sha256": dc_setup_sha,
        "library_setup_sha256": library_setup_sha,
        "compiler_binding_sha256": compiler_sha,
        "dc_tool_version": dc_tool_version.strip(),
        "library_id": library["library_id"],
        "process_corner": library["process_corner"],
        "operating_condition": library["operating_condition"],
        "measurement_state": "scheduled",
    }
    _write_json(run_dir / "run_spec.json", run_spec_body)
    activity_body = {
        "schema_version": ACTIVITY_REQUEST_SCHEMA,
        "point_id": run_id,
        "compatible_point_ids": [run_id],
        "component": "full_chip_exact_anchor",
        "signature": binding["profile_id"],
        "activity_class": request_value["activity_class"],
        "activity_coverage": request_value["activity_coverage"],
        "candidate": candidate,
        "workload": request_value["workload"],
        "formats": formats,
        "clock_period_ns": 1.0,
        "required_outputs": {
            "saif": "inputs/activity.saif",
            "decode_trace": "inputs/decode_trace.json",
        },
        "decode_trace_schema": EXACT_ACTIVITY_TRACE_SCHEMA,
        "measurement_state": "awaiting_activity",
    }
    _write_json(run_dir / "activity_request.json", activity_body)
    run_script = _run_command(
        synopsys_dir=synopsys_dir,
        environment_script=environment,
        dc_setup=setup,
        tcl_script=tcl_script,
        point_id=run_id,
        top_module=request_value["top_module"],
        operating_condition=library["operating_condition"],
        requires_saif=True,
        blackboxes=(),
        expected_tool_version=dc_tool_version.strip(),
        source_tree_sha256=rtl_manifest["source_tree_sha256"],
        constraints_sha256=constraints_sha,
        library_manifest_sha256=library_sha,
        dc_setup_sha256=dc_setup_sha,
        library_setup_sha256=library_setup_sha,
        compiler_binding_sha256=compiler_sha,
    )
    _write_bytes(
        run_dir / "run.sh",
        run_script.encode(),
        executable=True,
    )
    workflow_body = {
        "schema_version": EXACT_WORKFLOW_SCHEMA,
        "run_id": run_id,
        "request_sha256": _sha256_file(context / "exact_request.json"),
        "profile_id": binding["profile_id"],
        "binding_id": binding["binding_id"],
        "candidate": candidate,
        "workload": request_value["workload"],
        "rtl_root": str(rtl),
        "rtl_source_tree_sha256": rtl_manifest[
            "source_tree_sha256"
        ],
        "dc_tool_version": dc_tool_version.strip(),
        "library_id": library["library_id"],
        "process_corner": library["process_corner"],
        "operating_condition": library["operating_condition"],
        "clock_period_ns": 1.0,
        "measurement_state": "scheduled",
        "command": "run/run.sh",
    }
    _write_json(target / "workflow_manifest.json", workflow_body)
    return target


def _load_exact_activity_trace(
    path: Path,
    *,
    run_id: str,
    request: Mapping[str, Any],
    saif_sha256: str,
    rtl_source_tree_sha256: str,
) -> dict[str, Any]:
    trace = _load_hashed_json(path, schema=EXACT_ACTIVITY_TRACE_SCHEMA)
    if set(trace) != {
        "schema_version",
        "compatible_point_ids",
        "activity_class",
        "activity_coverage",
        "candidate",
        "workload",
        "cycles",
        "clock_ns",
        "saif_sha256",
        "saif_source_id",
        "activity_generator",
        "saif_instance",
        "rtl_source_tree_sha256",
        "stimulus_id",
        "content_hash",
    }:
        raise ValueError("exact decode trace fields differ from the schema")
    if trace["compatible_point_ids"] != [run_id]:
        raise ValueError("exact decode trace run identity mismatch")
    if (
        trace["activity_class"] != request["activity_class"]
        or trace["activity_coverage"] != request["activity_coverage"]
        or trace["candidate"] != request["candidate"]
        or trace["workload"] != request["workload"]
    ):
        raise ValueError("exact decode trace workload binding mismatch")
    if (
        isinstance(trace["cycles"], bool)
        or not isinstance(trace["cycles"], int)
        or trace["cycles"] <= 0
        or not math.isclose(
            float(trace["clock_ns"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("exact decode trace timing is invalid")
    if (
        trace["saif_sha256"] != saif_sha256
        or trace["rtl_source_tree_sha256"] != rtl_source_tree_sha256
    ):
        raise ValueError("exact decode trace provenance mismatch")
    for name in (
        "saif_source_id",
        "activity_generator",
        "saif_instance",
        "stimulus_id",
    ):
        if not str(trace[name]).strip():
            raise ValueError(f"exact decode trace {name} is empty")
    return trace


def _exact_artifact(
    path: Path,
    *,
    kind: str,
    root: Path,
) -> dict[str, Any]:
    source = _require_file(path, kind)
    return {
        "kind": kind,
        "path": source.relative_to(root).as_posix(),
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def ingest_exact_workflow(
    workflow_root: str | Path,
    *,
    output: str | Path,
    software_root: str | Path,
) -> Path:
    """Build and independently validate one exact-anchor index."""

    root = Path(workflow_root).resolve()
    workflow = _load_hashed_json(
        root / "workflow_manifest.json",
        schema=EXACT_WORKFLOW_SCHEMA,
    )
    if (
        workflow["measurement_state"] != "scheduled"
        or float(workflow["clock_period_ns"]) != 1.0
    ):
        raise ValueError("exact workflow was not prepared under the 1 ns contract")
    request = _load_exact_request(root / "context" / "exact_request.json")
    binding = _load_compiler_binding(
        root / "context" / "compiler_precision_binding.json"
    )
    if (
        binding["profile_id"] != workflow["profile_id"]
        or binding["binding_id"] != workflow["binding_id"]
        or request["candidate"] != workflow["candidate"]
        or request["workload"] != workflow["workload"]
    ):
        raise ValueError("exact workflow identities differ")

    rtl_manifest = _load_hashed_json(
        root / "context" / "rtl_source_manifest.json",
        schema=RTL_SOURCE_MANIFEST_SCHEMA,
    )
    _validate_rtl_source_manifest(
        rtl_manifest,
        Path(workflow["rtl_root"]).resolve(),
    )
    if (
        rtl_manifest["source_tree_sha256"]
        != workflow["rtl_source_tree_sha256"]
    ):
        raise ValueError("exact workflow RTL identity mismatch")
    library = _load_library_manifest(
        root / "context" / "library_manifest.json"
    )
    if (
        library["library_id"] != workflow["library_id"]
        or library["process_corner"] != workflow["process_corner"]
        or library["operating_condition"] != workflow["operating_condition"]
    ):
        raise ValueError("exact workflow library identity mismatch")
    tool_context = _load_hashed_json(
        root / "context" / "tool_context.json",
        schema=TOOL_CONTEXT_SCHEMA,
    )
    for name in ("environment_script", "dc_setup", "calibration_tcl"):
        source = _require_file(tool_context[name], name)
        if _sha256_file(source) != tool_context[f"{name}_sha256"]:
            raise ValueError(f"{name} changed after exact preparation")
    library_setup = _require_file(
        root / "context" / "library_setup.tcl",
        "library setup",
    )
    library_setup_sha = _sha256_file(library_setup)
    if library_setup_sha != tool_context["library_setup_sha256"]:
        raise ValueError("library setup changed after exact preparation")
    dc_setup_sha = tool_context["dc_setup_sha256"]

    run_dir = root / "run"
    spec = _load_hashed_json(
        run_dir / "run_spec.json",
        schema=RUN_SPEC_SCHEMA,
    )
    if (
        spec["run_id"] != workflow["run_id"]
        or spec["profile_id"] != workflow["profile_id"]
        or spec["candidate"] != workflow["candidate"]
        or spec["measurement_state"] != "scheduled"
        or spec["rtl_source_tree_sha256"]
        != rtl_manifest["source_tree_sha256"]
        or spec["library_manifest_sha256"]
        != _sha256_file(root / "context" / "library_manifest.json")
        or spec["constraints_sha256"]
        != _sha256_file(root / "context" / "constraints.sdc")
        or spec["dc_setup_sha256"] != dc_setup_sha
        or spec["library_setup_sha256"] != library_setup_sha
        or spec["dc_tool_version"] != workflow["dc_tool_version"]
        or spec["library_id"] != workflow["library_id"]
        or spec["process_corner"] != workflow["process_corner"]
        or spec["operating_condition"] != workflow["operating_condition"]
        or float(spec["clock_period_ns"]) != 1.0
    ):
        raise ValueError("exact run specification mismatch")
    for name, digest in spec["definition_sha256"].items():
        if _sha256_file(run_dir / "definitions" / name) != digest:
            raise ValueError("exact generated definition changed")
    if _sha256_file(run_dir / "rtl_files.tcl") != spec[
        "rtl_filelist_sha256"
    ]:
        raise ValueError("exact RTL file list changed")
    if _sha256_file(
        _require_file(run_dir / "library_setup.tcl", "library setup")
    ) != library_setup_sha:
        raise ValueError("exact library setup changed")
    if not (run_dir / "outputs" / "DC_SUCCESS").is_file():
        raise RuntimeError("exact DC run is incomplete")

    reports = run_dir / "outputs" / "reports"
    area_report = _require_file(reports / "area.rpt", "area report")
    power_report = _require_file(reports / "power.rpt", "power report")
    timing_report = _require_file(reports / "timing.rpt", "timing report")
    _parse_dc_report(run_dir / "outputs" / "reports" / "dc_report.rpt")
    _parse_timing_report(timing_report)
    synthesis_log = _require_file(
        run_dir / "outputs" / "logs" / "dc.log",
        "synthesis log",
    )
    netlist = _require_file(
        run_dir
        / "outputs"
        / "netlist"
        / f"{request['top_module']}_mapped.v",
        "mapped netlist",
    )
    constraints = _require_file(
        run_dir / "constraints.sdc",
        "constraints",
    )
    saif = _require_file(run_dir / "inputs" / "activity.saif", "SAIF")
    trace_path = _require_file(
        run_dir / "inputs" / "decode_trace.json",
        "decode trace",
    )
    saif_sha = _sha256_file(saif)
    _load_exact_activity_trace(
        trace_path,
        run_id=workflow["run_id"],
        request=request,
        saif_sha256=saif_sha,
        rtl_source_tree_sha256=rtl_manifest["source_tree_sha256"],
    )
    compiler_path = root / "context" / "compiler_precision_binding.json"
    library_path = root / "context" / "library_manifest.json"
    source_manifest_path = root / "context" / "rtl_source_manifest.json"
    expected_log = {
        "RTL_SOURCE_TREE_SHA256": rtl_manifest["source_tree_sha256"],
        "COMPILER_BINDING_SHA256": _sha256_file(compiler_path),
        "CONSTRAINTS_SHA256": _sha256_file(constraints),
        "LIBRARY_MANIFEST_SHA256": _sha256_file(library_path),
        "DC_SETUP_SHA256": dc_setup_sha,
        "LIBRARY_SETUP_SHA256": library_setup_sha,
        "SAIF_SHA256": saif_sha,
        "DECODE_TRACE_SHA256": _sha256_file(trace_path),
        "NETLIST_SHA256": _sha256_file(netlist),
    }
    for label, digest in expected_log.items():
        if _log_binding(synthesis_log, label) != digest:
            raise ValueError(f"exact synthesis log binding mismatch: {label}")

    specialization_hashes = {
        "compiler_precision_binding": _sha256_file(compiler_path),
        "constraints": _sha256_file(constraints),
        "decode_trace": _sha256_file(trace_path),
        "library_manifest": _sha256_file(library_path),
        "saif": saif_sha,
        "synthesis_log": _sha256_file(synthesis_log),
        "synthesized_netlist": _sha256_file(netlist),
    }
    profile = binding["profile"]
    runtime = binding["runtime_precision_contract"]
    specialization_without_id = {
        "schema_version": RTL_SPECIALIZATION_SCHEMA,
        "profile_id": binding["profile_id"],
        "binding_id": binding["binding_id"],
        "target": request["candidate"],
        "format_bindings": {
            "weight": profile["weight_format"],
            "activation": profile["activation_format"],
            "key": profile["key_format"],
            "value": profile["value_format"],
            "vector": profile["vector_format"],
        },
        "rtl_precision_parameters": runtime["rtl_precision_parameters"],
        "matrix_semantics_sha256": runtime["matrix_semantics"][
            "content_hash"
        ],
        "rtl_source_tree_sha256": rtl_manifest["source_tree_sha256"],
        "artifact_sha256": specialization_hashes,
        "build_command": ["bash", "run/run.sh"],
        "selector_enabled": True,
    }
    specialization_body = {
        **specialization_without_id,
        "specialization_id": (
            "rtl-specialization-"
            + _content_hash(specialization_without_id)
        ),
    }
    specialization_path = root / "context" / "rtl_specialization.json"
    _write_json(specialization_path, specialization_body)

    artifact_paths = {
        "area_report": area_report,
        "compiler_precision_binding": compiler_path,
        "constraints": constraints,
        "decode_trace": trace_path,
        "library_manifest": library_path,
        "power_report": power_report,
        "rtl_source_manifest": source_manifest_path,
        "rtl_specialization": specialization_path,
        "saif": saif,
        "synthesis_log": synthesis_log,
        "synthesized_netlist": netlist,
        "timing_report": timing_report,
    }
    artifacts = [
        _exact_artifact(path, kind=kind, root=root)
        for kind, path in sorted(artifact_paths.items())
    ]
    record_body = {
        "profile_id": binding["profile_id"],
        "candidate": request["candidate"],
        "workload": request["workload"],
        "timing_evidence_id": request["timing_evidence_id"],
        "layout_id": request["layout_id"],
        "traffic_ledger_id": request["traffic_ledger_id"],
        "activity_coverage": sorted(request["activity_coverage"]),
        "artifacts": artifacts,
    }
    record = {
        **record_body,
        "record_hash": _content_hash(record_body),
    }
    index_body = {
        "schema_version": EXACT_ANCHOR_SCHEMA,
        "model_name": request["model_name"],
        "model_revision": request["model_revision"],
        "rtl_source_tree_sha256": rtl_manifest["source_tree_sha256"],
        "metric_scope": "decode_chip_through_final_rmsnorm",
        "workload": request["workload"],
        "synthesis_context": {
            "dc_tool_version": workflow["dc_tool_version"],
            "library_id": workflow["library_id"],
            "process_corner": workflow["process_corner"],
            "clock_period_ns": 1.0,
            "mx_block_size": 8,
            "power_analysis_mode": "saif_annotated",
            "area_unit": "um2",
        },
        "records": [record],
    }
    target = Path(output).resolve()
    if target.parent != root:
        raise ValueError("exact anchor index must be written in the workflow root")
    _write_json(target, index_body)

    import sys

    software = Path(software_root).resolve()
    if not (software / "decode_dse" / "hardware" / "dc_anchor.py").is_file():
        raise FileNotFoundError("software exact-anchor validator is missing")
    if str(software) not in sys.path:
        sys.path.insert(0, str(software))
    from decode_dse.hardware.dc_anchor import ExactDCAnchorIndex

    ExactDCAnchorIndex.load(
        target,
        model_name=request["model_name"],
        model_revision=request["model_revision"],
        workload=request["workload"],
        rtl_source_tree_sha256=rtl_manifest["source_tree_sha256"],
    )
    return target


def create_exact_request(
    output: str | Path,
    *,
    model_name: str,
    model_revision: str,
    candidate_path: str | Path,
    workload_path: str | Path,
    timing_evidence_id: str,
    layout_id: str,
    traffic_ledger_id: str,
    top_module: str,
    compiler_binding_path: str | Path,
    activity_class: str,
) -> Path:
    candidate = json.loads(
        _require_file(candidate_path, "candidate JSON").read_text(
            encoding="utf-8"
        )
    )
    workload = json.loads(
        _require_file(workload_path, "workload JSON").read_text(
            encoding="utf-8"
        )
    )
    body = {
        "schema_version": EXACT_REQUEST_SCHEMA,
        "model_name": model_name,
        "model_revision": model_revision,
        "candidate": candidate,
        "workload": workload,
        "timing_evidence_id": timing_evidence_id,
        "layout_id": layout_id,
        "traffic_ledger_id": traffic_ledger_id,
        "top_module": top_module,
        "compiler_binding_path": str(
            _require_file(
                compiler_binding_path,
                "compiler precision binding",
            )
        ),
        "activity_class": activity_class,
        "activity_coverage": [
            "linear",
            "qk",
            "pv",
            "vector",
            "selector",
        ],
    }
    target = Path(output).resolve()
    _write_json(target, body)
    _load_exact_request(target)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and ingest PLENA DC/SAIF calibration runs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    library = commands.add_parser("library-manifest")
    library.add_argument("--output", required=True)
    library.add_argument("--library-id", required=True)
    library.add_argument("--process-corner", required=True)
    library.add_argument("--operating-condition", required=True)
    library.add_argument("--file", action="append", required=True)

    prepare = commands.add_parser("prepare-model")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--rtl-root", required=True)
    prepare.add_argument("--library-manifest", required=True)
    prepare.add_argument("--synopsys-environment", required=True)
    prepare.add_argument("--dc-tool-version", required=True)
    prepare.add_argument("--synopsys-setup")

    ingest = commands.add_parser("ingest-model")
    ingest.add_argument("--workflow", required=True)
    ingest.add_argument("--measurements", required=True)
    ingest.add_argument("--artifact-catalog", required=True)
    ingest.add_argument("--require-complete", action="store_true")

    request = commands.add_parser("exact-request")
    request.add_argument("--output", required=True)
    request.add_argument("--model-name", required=True)
    request.add_argument("--model-revision", required=True)
    request.add_argument("--candidate-json", required=True)
    request.add_argument("--workload-json", required=True)
    request.add_argument("--timing-evidence-id", required=True)
    request.add_argument("--layout-id", required=True)
    request.add_argument("--traffic-ledger-id", required=True)
    request.add_argument("--top-module", default="plena")
    request.add_argument("--compiler-binding", required=True)
    request.add_argument(
        "--activity-class",
        default="qwen3_32b_decode_q1_full_chip",
    )

    prepare_exact = commands.add_parser("prepare-exact")
    prepare_exact.add_argument("--output", required=True)
    prepare_exact.add_argument("--request", required=True)
    prepare_exact.add_argument("--rtl-root", required=True)
    prepare_exact.add_argument("--library-manifest", required=True)
    prepare_exact.add_argument("--synopsys-environment", required=True)
    prepare_exact.add_argument("--dc-tool-version", required=True)
    prepare_exact.add_argument("--synopsys-setup")

    ingest_exact = commands.add_parser("ingest-exact")
    ingest_exact.add_argument("--workflow", required=True)
    ingest_exact.add_argument("--output", required=True)
    ingest_exact.add_argument("--software-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "library-manifest":
        output = create_library_manifest(
            args.output,
            library_id=args.library_id,
            process_corner=args.process_corner,
            operating_condition=args.operating_condition,
            library_files=args.file,
        )
        print(output)
        return 0
    if args.command == "prepare-model":
        output = prepare_model_workflow(
            args.output,
            rtl_root=args.rtl_root,
            library_manifest=args.library_manifest,
            synopsys_environment=args.synopsys_environment,
            dc_tool_version=args.dc_tool_version,
            synopsys_setup=args.synopsys_setup,
        )
        print(output)
        return 0
    if args.command == "ingest-model":
        status = ingest_model_workflow(
            args.workflow,
            measurements_output=args.measurements,
            artifact_catalog_output=args.artifact_catalog,
            require_complete=args.require_complete,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if args.command == "exact-request":
        output = create_exact_request(
            args.output,
            model_name=args.model_name,
            model_revision=args.model_revision,
            candidate_path=args.candidate_json,
            workload_path=args.workload_json,
            timing_evidence_id=args.timing_evidence_id,
            layout_id=args.layout_id,
            traffic_ledger_id=args.traffic_ledger_id,
            top_module=args.top_module,
            compiler_binding_path=args.compiler_binding,
            activity_class=args.activity_class,
        )
        print(output)
        return 0
    if args.command == "prepare-exact":
        output = prepare_exact_workflow(
            args.output,
            request=args.request,
            rtl_root=args.rtl_root,
            library_manifest=args.library_manifest,
            synopsys_environment=args.synopsys_environment,
            dc_tool_version=args.dc_tool_version,
            synopsys_setup=args.synopsys_setup,
        )
        print(output)
        return 0
    if args.command == "ingest-exact":
        output = ingest_exact_workflow(
            args.workflow,
            output=args.output,
            software_root=args.software_root,
        )
        print(output)
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVITY_REQUEST_SCHEMA",
    "ACTIVITY_TRACE_SCHEMA",
    "EXACT_ACTIVITY_TRACE_SCHEMA",
    "EXACT_REQUEST_SCHEMA",
    "LIBRARY_MANIFEST_SCHEMA",
    "MODEL_WORKFLOW_SCHEMA",
    "RUN_SPEC_SCHEMA",
    "build_rtl_source_manifest",
    "create_exact_request",
    "create_library_manifest",
    "ingest_exact_workflow",
    "ingest_model_workflow",
    "prepare_exact_workflow",
    "prepare_model_workflow",
]
