"""Fail-closed compiler-to-emulator evidence hook for PackedKV profiles."""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
COMPILER_ROOT = SIMULATOR_ROOT / "compiler"
EMULATOR_ROOT = SIMULATOR_ROOT / "transactional_emulator"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from compiler.aten.packedkv_profile_hook import (  # noqa: E402
    ACCUMULATOR_RULE,
    COMPILER_CAPABILITY_SCOPE,
    COMPILER_TARGET_MODE,
    EVIDENCE_TARGET_SCHEMA,
    LONG_CONTEXT_GEOMETRY,
    MATRIX_SEMANTICS_SCHEMA,
    MX_PHYSICAL_SEMANTICS_SCHEMA,
    MXINT2_ACTIVATION_SCOPE,
    OUTPUT_RULE,
    PROFILE_SCHEMA,
    TARGET,
    _assemble_trace,
    _build_binding,
    _canonical_bytes,
    _compile_trace,
    _content_hash,
    _profile_support,
    _with_content_hash,
)
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from transactional_emulator.tools.packedkv_numeric import (  # noqa: E402
    MXINT_SCALE_RULE,
    MX_PHYSICAL_SEMANTICS_ID,
    canonical_mxint_vectors,
    decode_mx,
    decode_vector_row,
    encode_mx,
    encode_vector_rows,
    matrix_accumulate_partials,
    matrix_format,
    round_float,
    vector_format,
)

REQUEST_SCHEMA = "decode-stage-hook-request/v1"
RESULT_SCHEMA = "decode-stage-hook-result/v1"
BUNDLE_SCHEMA = "plena-emulator-compiler-bundle/v2"
RUN_METRICS_SCHEMA = "plena-emulator-packedkv-run-metrics/v2"
REJECTION_SCHEMA = "plena-emulator-profile-rejection/v2"
TRACE_BATCHES = (1, 2, 4)
SCALED_GEOMETRY = dict(LONG_CONTEXT_GEOMETRY)
SCALED_GEOMETRY["batch"] = 1
SCALED_MLEN = int(SCALED_GEOMETRY["mlen"])
SCALED_BLEN = int(SCALED_GEOMETRY["blen"])
SCALED_HLEN = int(SCALED_GEOMETRY["hlen"])
SCALED_KV_HEADS = int(SCALED_GEOMETRY["kv_heads"])
HBM_SIZE_BYTES = 64 * 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class HookError(RuntimeError):
    """Raised when measured evidence cannot be produced or trusted."""


def _expected_compiler_target(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_TARGET_SCHEMA,
        "target_mode": COMPILER_TARGET_MODE,
        "capability_scope": COMPILER_CAPABILITY_SCOPE,
        "source_tree_sha256": request["source_tree_sha256"],
        "mxint2_activation_scope": MXINT2_ACTIVATION_SCOPE,
        "rtl_deployment_supports_mxint2_activation": False,
        "common_deployment_valid": False,
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HookError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HookError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HookError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HookError(f"{path} must contain a JSON object")
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HookError(
                f"immutable output already exists with different content: {path}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, _canonical_bytes(value) + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    token = str(value)
    if not _HASH_RE.fullmatch(token):
        raise HookError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _ensure_confined(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HookError(f"path escapes artifact directory: {path}") from exc


def _artifact(path: Path, kind: str) -> dict[str, str]:
    payload_hash = _sha256_file(path)
    if path.stat().st_size <= 0:
        raise HookError(f"artifact is empty: {path}")
    return {
        "artifact_id": "sha256-" + payload_hash,
        "kind": kind,
        "path": str(path.resolve()),
    }


def _validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "stage",
        "manifest_hash",
        "profile_id",
        "profile",
        "target",
        "source_tree_sha256",
        "hook_template_hash",
        "environment_sha256",
        "content_hash",
    }
    if set(value) != expected:
        raise HookError("emulator request fields differ from the schema")
    if value["schema_version"] != REQUEST_SCHEMA:
        raise HookError(f"unsupported request schema {value['schema_version']!r}")
    if value["stage"] != "emulator":
        raise HookError(f"emulator hook cannot execute stage {value['stage']!r}")
    for field in (
        "manifest_hash",
        "source_tree_sha256",
        "hook_template_hash",
        "environment_sha256",
        "content_hash",
    ):
        _require_hash(value[field], field)
    if value["content_hash"] != _content_hash(value):
        raise HookError("request content_hash does not match its canonical body")
    profile = value["profile"]
    if not isinstance(profile, dict):
        raise HookError("profile must be an object")
    profile_id = "dqp-" + hashlib.sha256(_canonical_bytes(profile)).hexdigest()
    if value["profile_id"] != profile_id:
        raise HookError("profile_id does not match the canonical profile")
    if value["target"] != TARGET:
        raise HookError("request target differs from the PackedKV deployment target")
    return dict(value)


def _existing_result(
    result_path: Path,
    request_hash: str,
    artifact_root: Path,
) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    result = _load_json(result_path)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise HookError("existing result has an unsupported schema")
    if result.get("stage") != "emulator":
        raise HookError("existing result has the wrong stage")
    if result.get("content_hash") != _content_hash(result):
        raise HookError("existing result content hash is invalid")
    if result.get("request_content_hash") != request_hash:
        raise HookError("existing result belongs to a different request")
    tests = result.get("tests")
    if not isinstance(tests, list) or not tests:
        raise HookError("existing result has no tests")
    names = [test.get("name") for test in tests if isinstance(test, dict)]
    if len(names) != len(tests) or len(names) != len(set(names)):
        raise HookError("existing result contains malformed or repeated tests")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HookError("existing result has no artifacts")
    ids: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise HookError("existing result contains a malformed artifact")
        path = Path(str(artifact.get("path", "")))
        _ensure_confined(path, artifact_root)
        if not path.is_file() or path.stat().st_size <= 0:
            raise HookError(f"existing artifact is missing or empty: {path}")
        artifact_id = "sha256-" + _sha256_file(path)
        if artifact.get("artifact_id") != artifact_id:
            raise HookError(f"existing artifact hash is invalid: {path}")
        ids.append(artifact_id)
    if len(ids) != len(set(ids)):
        raise HookError("existing result repeats an artifact ID")
    return result


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        return "[" + ", ".join(json.dumps(item) for item in value) + "]"
    raise HookError(f"unsupported TOML scalar {value!r}")


def _toml_table(
    lines: list[str],
    path: str,
    values: Mapping[str, Any],
    *,
    array: bool = False,
) -> None:
    lines.append(f"[[{path}]]" if array else f"[{path}]")
    for key, value in values.items():
        lines.append(f"{key} = {_toml_scalar(value)}")
    lines.append("")


def _emit_mx_type(lines: list[str], path: str, value: Mapping[str, Any]) -> None:
    _toml_table(lines, path, {"format": value["format"], **(
        {"block": value["block"]} if value["format"] == "Mx" else {}
    )})
    if value["format"] == "Plain":
        _toml_table(lines, f"{path}.DATA_TYPE", value["DATA_TYPE"])
    else:
        _toml_table(lines, f"{path}.ELEM", value["ELEM"])
        _toml_table(lines, f"{path}.SCALE", value["SCALE"])


def _settings_payload(binding: Mapping[str, Any]) -> bytes:
    runtime = binding["runtime_precision_contract"]
    precision = runtime["emulator_precision"]
    semantics = precision["MATRIX_SEMANTICS"]
    lines: list[str] = []
    config = {
        "DC_EN": 1,
        "MAX_LOOP_INSTRUCTIONS": 1_000_000,
        "BLEN": SCALED_BLEN,
        "HLEN": SCALED_HLEN,
        "MLEN": SCALED_MLEN,
        "VLEN": SCALED_MLEN,
        "BROADCAST_AMOUNT": SCALED_MLEN // SCALED_HLEN,
        "HBM_SIZE": HBM_SIZE_BYTES,
        "HBM_GEN": "HBM2",
        "HBM_CHANNELS": 8,
        "MATRIX_SRAM_SIZE": 4096,
        "VECTOR_SRAM_SIZE": 2048,
        "HBM_M_Prefetch_Amount": SCALED_MLEN,
        "HBM_V_Prefetch_Amount": SCALED_BLEN,
        "HBM_V_Writeback_Amount": SCALED_BLEN,
    }
    for key, value in config.items():
        _toml_table(lines, f"TRANSACTIONAL.CONFIG.{key}", {"value": value})
    for key in (
        "MATRIX_SRAM_TYPE",
        "VECTOR_SRAM_TYPE",
        "HBM_M_WEIGHT_TYPE",
        "HBM_M_KV_TYPE",
        "HBM_V_ACT_TYPE",
        "HBM_V_KV_TYPE",
    ):
        _emit_mx_type(lines, f"TRANSACTIONAL.PRECISION.{key}", precision[key])
    _emit_mx_type(
        lines,
        "TRANSACTIONAL.PRECISION.HBM_V_INT_TYPE",
        {
            "format": "Plain",
            "DATA_TYPE": {"type": "Int", "width": 32},
        },
    )
    _toml_table(
        lines,
        "TRANSACTIONAL.PRECISION.SCALAR_FP",
        precision["SCALAR_FP"],
    )
    nested = {
        "profile_contract",
        "operation_bindings",
        "fixed_accumulator_bank",
        "instruction_reduction",
        "matrix_storage_fp",
        "mxint_pipeline",
        "mxfp_pipeline",
        "mixed_family",
        "packedkv_selector_rtl_capability",
        "numerical_trace_conformance",
    }
    _toml_table(
        lines,
        "TRANSACTIONAL.PRECISION.MATRIX_SEMANTICS",
        {key: value for key, value in semantics.items() if key not in nested},
    )
    _toml_table(
        lines,
        "TRANSACTIONAL.PRECISION.MATRIX_SEMANTICS.profile_contract",
        semantics["profile_contract"],
    )
    for operation in semantics["operation_bindings"]:
        _toml_table(
            lines,
            "TRANSACTIONAL.PRECISION.MATRIX_SEMANTICS.operation_bindings",
            operation,
            array=True,
        )
    for key in (
        "fixed_accumulator_bank",
        "instruction_reduction",
        "matrix_storage_fp",
        "mxint_pipeline",
        "mxfp_pipeline",
        "mixed_family",
        "packedkv_selector_rtl_capability",
        "numerical_trace_conformance",
    ):
        _toml_table(
            lines,
            f"TRANSACTIONAL.PRECISION.MATRIX_SEMANTICS.{key}",
            semantics[key],
        )
    _toml_table(
        lines,
        "TRANSACTIONAL.PRECISION.MX_PHYSICAL_SEMANTICS",
        runtime["physical_semantics"],
    )
    latency_names = (
        "SYSTOLIC_PROCESSING_OVERHEAD",
        "VECTOR_ADD_CYCLES",
        "VECTOR_MUL_CYCLES",
        "VECTOR_EXP_CYCLES",
        "VECTOR_PREFIX_SCAN_CYCLES",
        "VECTOR_SHIFT_CYCLES",
        "VECTOR_RECI_CYCLES",
        "VECTOR_MAX_CYCLES",
        "VECTOR_SUM_CYCLES",
        "SCALAR_FP_LONGEST_OPERATE_CYCLES",
        "SCALAR_FP_BASIC_CYCLES",
        "SCALAR_FP_EXP_CYCLES",
        "SCALAR_FP_SQRT_CYCLES",
        "SCALAR_FP_RECI_CYCLES",
        "SCALAR_INT_BASIC_CYCLES",
    )
    for name in latency_names:
        _toml_table(
            lines,
            f"TRANSACTIONAL.LATENCY.{name}",
            {"dc_lib_en": 1, "dc_lib_dis": 1},
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _vector_zero_preload(row_count: int, vector_token: str) -> bytes:
    fmt = vector_format(vector_token)
    return encode_vector_rows(
        [(0.0,) * SCALED_MLEN for _ in range(max(1, row_count))],
        fmt,
        SCALED_MLEN,
    )


def _bf16_round(value: float) -> float:
    return round_float(value, vector_format("BF16"))


def _write_trace_files(
    root: Path,
    name: str,
    assembly: str,
) -> tuple[Path, Path, dict[str, Any]]:
    trace_root = root / name
    trace_root.mkdir(parents=True, exist_ok=True)
    assembly_path = trace_root / "program.asm"
    _write_immutable(assembly_path, assembly.encode("utf-8"))
    machine, assembler_metrics = _assemble_trace(assembly_path)
    if not assembler_metrics["execution_contract_valid"]:
        raise HookError(
            f"{name} emits an unsupported execution contract: "
            f"{assembler_metrics['emulator_unsupported_opcodes']} / "
            f"{assembler_metrics['rtl_decoder_unsupported_opcodes']} / "
            f"{assembler_metrics['cross_target_operand_violations']}"
        )
    machine_path = trace_root / "program.mem"
    _write_immutable(machine_path, machine)
    return assembly_path, machine_path, assembler_metrics


def _place(hbm: bytearray, base: int, payload: bytes) -> None:
    end = base + len(payload)
    if base < 0 or end > len(hbm):
        raise HookError("HBM payload exceeds its measured reservation")
    hbm[base:end] = payload


def _linear_bundle(
    root: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    descriptors = binding["format_descriptors"]
    profile = binding["profile"]
    compiler = PlenaCompiler(
        mlen=SCALED_MLEN,
        blen=SCALED_BLEN,
        hbm_element_width=int(descriptors["weight"]["element_bits"]),
        hbm_block_size=8,
        hbm_scale_width=8,
        hbm_v_prefetch_amount=SCALED_BLEN,
    )
    compiler.hlen = SCALED_HLEN
    compiler.broadcast_amount = SCALED_MLEN // SCALED_HLEN
    activation_input = compiler.input(
        "linear_activation",
        shape=(SCALED_BLEN, SCALED_MLEN),
        physical_shape=(SCALED_BLEN, SCALED_MLEN),
        hbm_element_width=int(descriptors["activation"]["element_bits"]),
        precision_role="activation",
    )
    activation = compiler.load_batch(activation_input, name="linear_activation_vram")
    weight = compiler.input(
        "linear_weight",
        shape=(SCALED_MLEN, SCALED_MLEN),
        physical_shape=(SCALED_MLEN, SCALED_MLEN),
        precision_role="weight",
    )
    output = compiler.linear_projection(
        activation,
        weight,
        name="linear_output",
        physical_shape=(SCALED_BLEN, SCALED_MLEN),
    )
    assembly_path, machine_path, assembler = _write_trace_files(
        root,
        "linear",
        compiler.compile(),
    )
    a_values = [
        ((-1.0) ** (row + col)) * (0.25 + 0.125 * (col % 4))
        for row in range(SCALED_BLEN)
        for col in range(SCALED_MLEN)
    ]
    w_values = [
        (
            ((-1.0) ** (row + col)) * (0.5 if row == col else 0.25)
            if row % 8 == col % 8
            else 0.0
        )
        for row in range(SCALED_MLEN)
        for col in range(SCALED_MLEN)
    ]
    a_fmt = matrix_format(profile["activation_format"])
    w_fmt = matrix_format(profile["weight_format"])
    a_layout = compiler.get_hbm_layout(activation_input.name)
    w_layout = compiler.get_hbm_layout(weight.name)
    a_image = encode_mx(
        a_values,
        a_fmt,
        hbm_row_bytes=a_layout.row_bytes,
    )
    w_image = encode_mx(
        w_values,
        w_fmt,
        hbm_row_bytes=w_layout.row_bytes,
    )
    if len(a_image.payload) != activation_input.hbm_size:
        raise HookError("linear activation image differs from compiler allocation")
    if len(w_image.payload) != weight.hbm_size:
        raise HookError("linear weight image differs from compiler allocation")
    hbm = bytearray(
        max(
            activation_input.hbm_addr + activation_input.hbm_size,
            weight.hbm_addr + weight.hbm_size,
        )
    )
    _place(hbm, activation_input.hbm_addr, a_image.payload)
    _place(hbm, weight.hbm_addr, w_image.payload)
    hbm_path = root / "linear" / "hbm.bin"
    _write_immutable(hbm_path, bytes(hbm))
    vram_path = root / "linear" / "vram.bin"
    _write_immutable(
        vram_path,
        _vector_zero_preload(1, profile["vector_format"]),
    )
    a_decoded = list(decode_mx(a_image, a_fmt))
    vector = vector_format(profile["vector_format"])
    a_vector = [round_float(value, vector) for value in a_decoded]
    a_matrix = list(
        decode_mx(
            encode_mx(a_vector, a_fmt, hbm_row_bytes=a_layout.row_bytes),
            a_fmt,
        )
    )
    w_matrix = [_bf16_round(value) for value in decode_mx(w_image, w_fmt)]
    expected_rows: list[list[float]] = []
    for row in range(SCALED_BLEN):
        expected_row: list[float] = []
        for col in range(SCALED_MLEN):
            value = sum(
                a_matrix[row * SCALED_MLEN + index]
                * w_matrix[index * SCALED_MLEN + col]
                for index in range(SCALED_MLEN)
            )
            expected_row.append(
                matrix_accumulate_partials((value,), vector)
            )
        expected_rows.append(expected_row)
    golden = {
        "output_base_elements": compiler.get_vram_addr(output.name),
        "expected_rows": expected_rows,
        "weight_hbm_base": weight.hbm_addr,
        "activation_hbm_base": activation_input.hbm_addr,
    }
    golden_path = root / "linear" / "golden.json"
    _write_json(golden_path, golden)
    return {
        "name": "linear",
        "assembly_path": str(assembly_path),
        "machine_path": str(machine_path),
        "hbm_path": str(hbm_path),
        "vram_path": str(vram_path),
        "golden_path": str(golden_path),
        "assembler_metrics": assembler,
        "output_base_elements": golden["output_base_elements"],
        "expected_rows": expected_rows,
        "activation_prefetch_precision_valid": (
            "H_PREFETCH_V" in assembler["opcode_histogram"]
        ),
        "weight_prefetch_precision_valid": (
            0 in assembler["decoded_h_prefetch_m_precision_funct1"]
        ),
    }


def _roundtrip_bundle(
    root: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    descriptors = binding["format_descriptors"]
    profile = binding["profile"]
    kv_bits = int(descriptors["value"]["element_bits"])
    compiler = PlenaCompiler(
        mlen=SCALED_MLEN,
        blen=SCALED_BLEN,
        hbm_element_width=kv_bits,
        hbm_block_size=8,
        hbm_scale_width=8,
        hbm_v_prefetch_amount=SCALED_BLEN,
        hbm_v_writeback_amount=SCALED_BLEN,
    )
    source = compiler.input(
        "roundtrip_source",
        shape=(SCALED_BLEN, SCALED_MLEN),
        physical_shape=(SCALED_BLEN, SCALED_MLEN),
        hbm_element_width=kv_bits,
        precision_role="value",
    )
    resident = compiler.load_batch(source, name="roundtrip_vram")
    destination = compiler.store(
        resident,
        name="roundtrip_destination",
        precision=1,
        hbm_element_width=kv_bits,
        hbm_block_size=8,
        hbm_scale_width=8,
        precision_role="value",
    )
    assembly_path, machine_path, assembler = _write_trace_files(
        root,
        "roundtrip",
        compiler.compile(),
    )
    pattern = (1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, -0.0)
    values = list(pattern * (SCALED_BLEN * SCALED_MLEN // len(pattern)))
    fmt = matrix_format(profile["value_format"])
    layout = compiler.get_hbm_layout(source.name)
    image = encode_mx(values, fmt, hbm_row_bytes=layout.row_bytes)
    if len(image.payload) != source.hbm_size:
        raise HookError("roundtrip source differs from compiler allocation")
    hbm = bytearray(destination.hbm_addr + destination.hbm_size)
    _place(hbm, source.hbm_addr, image.payload)
    hbm_path = root / "roundtrip" / "hbm.bin"
    _write_immutable(hbm_path, bytes(hbm))
    vram_path = root / "roundtrip" / "vram.bin"
    _write_immutable(
        vram_path,
        _vector_zero_preload(1, profile["vector_format"]),
    )
    return {
        "name": "roundtrip",
        "assembly_path": str(assembly_path),
        "machine_path": str(machine_path),
        "hbm_path": str(hbm_path),
        "vram_path": str(vram_path),
        "assembler_metrics": assembler,
        "source_base": source.hbm_addr,
        "destination_base": destination.hbm_addr,
        "payload_bytes": len(image.payload),
        "element_plane_bytes": len(image.element_plane),
        "scale_plane_bytes": len(image.scale_plane),
        "source_payload_sha256": hashlib.sha256(image.payload).hexdigest(),
    }


def _attention_values(batch_size: int) -> tuple[list[float], list[float]]:
    k = [0.0] * (batch_size * SCALED_MLEN * SCALED_MLEN)
    v = [0.0] * (batch_size * SCALED_MLEN * SCALED_MLEN)
    for batch in range(batch_size):
        row = batch * SCALED_MLEN * SCALED_MLEN
        for selector in range(SCALED_KV_HEADS):
            sign = 1.0 if selector == 0 else -1.0
            amplitude = 0.25 * (batch + 1)
            for lane in range(SCALED_HLEN):
                offset = row + selector * SCALED_HLEN + lane
                k[offset] = sign * (0.25 + 0.125 * (lane % 4))
                v[offset] = sign * amplitude * (1.0 if lane % 2 == 0 else 0.5)
    return k, v


def _attention_bundle(
    root: Path,
    binding: Mapping[str, Any],
    batch_size: int,
    source_tree_sha256: str,
) -> dict[str, Any]:
    scaled_binding = {**binding, "target": dict(SCALED_GEOMETRY)}
    assembly, metrics, trace_contract, recipe = _compile_trace(
        scaled_binding,
        batch_size=batch_size,
        production_source_tree_sha256=source_tree_sha256,
        cache_tokens=1,
        cache_rows_per_batch=SCALED_MLEN,
        trace_scope="scaled_emulator_numerical",
    )
    name = f"attention-b{batch_size}"
    assembly_path, machine_path, assembler = _write_trace_files(
        root,
        name,
        assembly,
    )
    metrics.update(assembler)
    metrics["machine_word_count_valid"] = (
        metrics["machine_word_count"] == metrics["assembly_instruction_count"]
    )
    if (
        metrics["q_len"] != 1
        or metrics["cache_position"] != metrics["cache_tokens"] - 1
        or trace_contract["cache_position"] != metrics["cache_position"]
        or recipe["cache_position"] != metrics["cache_position"]
    ):
        raise HookError("compiler trace does not describe a cached tail q_len=1 step")
    trace_contract_path = root / name / "trace-contract.json"
    recipe_path = root / name / "input-recipe.json"
    _write_json(trace_contract_path, trace_contract)
    _write_json(recipe_path, recipe)
    profile = binding["profile"]
    kv_fmt = matrix_format(profile["key_format"])
    vector = vector_format(profile["vector_format"])
    k_values, v_values = _attention_values(batch_size)
    key_layout = metrics["role_hbm_layouts"]["key"]
    value_layout = metrics["role_hbm_layouts"]["value"]
    k_image = encode_mx(
        k_values,
        kv_fmt,
        hbm_row_bytes=32,
    )
    v_image = encode_mx(
        v_values,
        kv_fmt,
        hbm_row_bytes=32,
    )
    if len(k_image.payload) != key_layout["total_bytes"]:
        raise HookError("attention K image differs from compiler allocation")
    if len(v_image.payload) != value_layout["total_bytes"]:
        raise HookError("attention V image differs from compiler allocation")
    hbm_end = max(
        metrics["key_hbm_base"] + len(k_image.payload),
        metrics["value_hbm_base"] + len(v_image.payload),
    )
    hbm = bytearray(hbm_end)
    _place(hbm, metrics["key_hbm_base"], k_image.payload)
    _place(hbm, metrics["value_hbm_base"], v_image.payload)
    hbm_path = root / name / "hbm.bin"
    _write_immutable(hbm_path, bytes(hbm))
    q_rows = batch_size * SCALED_MLEN * SCALED_KV_HEADS
    query_rows = [
        [0.0] * SCALED_MLEN
        for _ in range(q_rows)
    ]
    for region in recipe["memory_images"][0]["active_regions"]:
        row_index = int(region["base_address_elements"]) // SCALED_MLEN
        query_rows[row_index] = [
            0.25 if lane % 2 == 0 else -0.25
            for lane in range(SCALED_MLEN)
        ]
    vram_path = root / name / "vram.bin"
    _write_immutable(
        vram_path,
        encode_vector_rows(query_rows, vector, SCALED_MLEN),
    )
    decoded_v = [_bf16_round(value) for value in decode_mx(v_image, kv_fmt)]
    output_image = next(
        item for item in recipe["memory_images"] if item["name"] == "O"
    )
    output_base = int(output_image["base_address"])
    group_stride = batch_size * SCALED_MLEN * SCALED_MLEN
    batch_stride = SCALED_MLEN * SCALED_MLEN
    expected: list[dict[str, Any]] = []
    for batch in range(batch_size):
        row_base = batch * SCALED_MLEN * SCALED_MLEN
        for selector in range(SCALED_KV_HEADS):
            head = decoded_v[
                row_base
                + selector * SCALED_HLEN:
                row_base
                + (selector + 1) * SCALED_HLEN
            ]
            row = [
                matrix_accumulate_partials((value,), vector)
                for value in head + head
            ]
            address = (
                output_base
                + selector * group_stride
                + batch * batch_stride
            )
            expected.append(
                {
                    "batch": batch,
                    "selector": selector,
                    "address_elements": address,
                    "values": row,
                }
            )
    golden_path = root / name / "golden.json"
    _write_json(
        golden_path,
        {
            "batch_size": batch_size,
            "q_len": 1,
            "cache_tokens": metrics["cache_tokens"],
            "cache_position": metrics["cache_position"],
            "expected_rows": expected,
        },
    )
    writeout_stride_valid = all(
        match.group(1) != "gp0"
        for match in re.finditer(
            r"^M_MM_WO\s+gp\d+,\s+(gp\d+),",
            assembly,
            flags=re.MULTILINE,
        )
    ) and "M_MM_WO" in assembler["opcode_histogram"]
    return {
        "name": name,
        "batch_size": batch_size,
        "assembly_path": str(assembly_path),
        "machine_path": str(machine_path),
        "hbm_path": str(hbm_path),
        "vram_path": str(vram_path),
        "golden_path": str(golden_path),
        "trace_contract_path": str(trace_contract_path),
        "recipe_path": str(recipe_path),
        "assembler_metrics": assembler,
        "compiler_metrics": metrics,
        "expected_rows": expected,
        "writeout_stride_valid": writeout_stride_valid,
    }


def _worker_compile(
    request_path: Path,
    output_dir: Path,
    bundle_path: Path,
) -> None:
    request = _validate_request(_load_json(request_path))
    supported, reason, descriptors = _profile_support(
        request["profile"],
        request["target"],
    )
    if not supported:
        raise HookError(f"worker cannot compile unsupported profile: {reason}")
    binding = _build_binding(request, descriptors)
    if binding.get("evidence_target") != _expected_compiler_target(request):
        raise HookError("compiler binding has the wrong evidence target")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.toml"
    _write_immutable(settings_path, _settings_payload(binding))
    traces = {
        "linear": _linear_bundle(output_dir, binding),
        "roundtrip": _roundtrip_bundle(output_dir, binding),
        "attention": [
            _attention_bundle(
                output_dir,
                binding,
                batch_size,
                request["source_tree_sha256"],
            )
            for batch_size in TRACE_BATCHES
        ],
    }
    binding_path = output_dir / "binding.json"
    _write_json(binding_path, binding)
    bundle = _with_content_hash(
        {
            "schema_version": BUNDLE_SCHEMA,
            "profile_id": request["profile_id"],
            "request_content_hash": request["content_hash"],
            "binding": binding,
            "binding_path": str(binding_path.resolve()),
            "settings_path": str(settings_path.resolve()),
            "traces": traces,
        }
    )
    _write_json(bundle_path, bundle)


def _resolve_binary(explicit: Path | None) -> tuple[Path, dict[str, Any] | None]:
    binary = (
        explicit.resolve()
        if explicit is not None
        else EMULATOR_ROOT / "target" / "release" / "transactional_emulator"
    )
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary, None
    cargo = shutil.which("cargo")
    if cargo is None:
        raise HookError(
            f"transactional emulator binary is unavailable at {binary} and cargo is not installed"
        )
    lock_path = EMULATOR_ROOT / "target" / ".packedkv-build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary, None
        started = time.monotonic()
        completed = subprocess.run(
            [cargo, "build", "--release", "--locked"],
            cwd=EMULATOR_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        record = {
            "command": [cargo, "build", "--release", "--locked"],
            "cwd": str(EMULATOR_ROOT),
            "return_code": completed.returncode,
            "wall_seconds": time.monotonic() - started,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
        }
        if completed.returncode or not binary.is_file():
            raise HookError(
                "cargo could not build the transactional emulator "
                f"(exit {completed.returncode})"
            )
        return binary, record


def _emulator_environment(binary: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["RUST_BACKTRACE"] = "1"
    environment["OMP_WAIT_POLICY"] = "PASSIVE"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    pattern = str(
        binary.parent
        / "build"
        / "torch-sys-*"
        / "out"
        / "libtorch"
        / "libtorch"
        / "lib"
    )
    libraries = glob.glob(pattern)
    if libraries:
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            libraries[0] + (":" + previous if previous else "")
        )
    return environment


def _run_emulator(
    binary: Path,
    settings_path: Path,
    trace: Mapping[str, Any],
    run_root: Path,
    *,
    dump_hbm: bool = False,
    timeout_seconds: float,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    op_stats = run_root / "op-stats.jsonl"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    command = [
        str(binary),
        "--opcode",
        str(Path(trace["machine_path"]).resolve()),
        "--hbm",
        str(Path(trace["hbm_path"]).resolve()),
        "--fpsram",
        str((run_root.parents[1] / "fpsram.bin").resolve()),
        "--vram",
        str(Path(trace["vram_path"]).resolve()),
        "--settings",
        str(settings_path.resolve()),
        "--hbm-size",
        str(HBM_SIZE_BYTES),
        "--op-stats",
        str(op_stats.resolve()),
        "--blocking-prefetch",
        "--log-level",
        "debug" if dump_hbm else "warn",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=run_root,
            check=False,
            capture_output=True,
            text=True,
            env=_emulator_environment(binary),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise HookError(f"{trace['name']} emulator run timed out") from exc
    _write_immutable(stdout_path, completed.stdout.encode("utf-8"))
    _write_immutable(stderr_path, completed.stderr.encode("utf-8"))
    if completed.returncode:
        raise HookError(
            f"{trace['name']} emulator run exited with {completed.returncode}"
        )
    output = run_root / "vram_dump.bin"
    if not output.is_file() or output.stat().st_size <= 0:
        raise HookError(f"{trace['name']} did not produce a VRAM dump")
    if not op_stats.is_file() or op_stats.stat().st_size <= 0:
        raise HookError(f"{trace['name']} did not produce opcode statistics")
    hbm_dump = run_root / "hbm_dump.bin"
    if dump_hbm and (
        not hbm_dump.is_file() or hbm_dump.stat().st_size != HBM_SIZE_BYTES
    ):
        raise HookError("physical roundtrip did not produce the bounded HBM dump")
    aggregate = None
    for line in op_stats.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("aggregate") is True:
            aggregate = item
    if not isinstance(aggregate, dict):
        raise HookError(f"{trace['name']} op-stats has no aggregate record")
    observed_opcodes = {item["op"] for item in aggregate["ops"]}
    expected_opcodes = set(trace["assembler_metrics"]["opcode_histogram"])
    missing_opcodes = sorted(expected_opcodes - observed_opcodes)
    if missing_opcodes:
        raise HookError(
            f"{trace['name']} did not execute emitted opcodes {missing_opcodes}"
        )
    return {
        "name": trace["name"],
        "command": command,
        "cwd": str(run_root),
        "return_code": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "binary_sha256": _sha256_file(binary),
        "settings_sha256": _sha256_file(settings_path),
        "machine_code_sha256": _sha256_file(Path(trace["machine_path"])),
        "hbm_preload_sha256": _sha256_file(Path(trace["hbm_path"])),
        "vram_preload_sha256": _sha256_file(Path(trace["vram_path"])),
        "output_sha256": _sha256_file(output),
        "op_stats_sha256": _sha256_file(op_stats),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
        "output_path": str(output),
        "op_stats_path": str(op_stats),
        "hbm_dump_path": str(hbm_dump) if dump_hbm else None,
        "observed_opcodes": sorted(observed_opcodes),
        "opcode_coverage_valid": not missing_opcodes,
        "aggregate": aggregate,
    }


def _compare_rows(
    output_path: Path,
    expected: Sequence[Mapping[str, Any]],
    vector_token: str,
) -> tuple[bool, float, list[dict[str, Any]]]:
    payload = output_path.read_bytes()
    fmt = vector_format(vector_token)
    maximum = 0.0
    rows: list[dict[str, Any]] = []
    for item in expected:
        address = int(item["address_elements"])
        if address % SCALED_MLEN:
            raise HookError("golden VRAM address is not row aligned")
        observed = decode_vector_row(
            payload,
            address // SCALED_MLEN,
            fmt,
            SCALED_MLEN,
        )
        reference = tuple(float(value) for value in item["values"])
        errors = [abs(left - right) for left, right in zip(observed, reference)]
        row_max = max(errors, default=0.0)
        maximum = max(maximum, row_max)
        rows.append(
            {
                "address_elements": address,
                "observed": list(observed),
                "expected": list(reference),
                "max_abs_error": row_max,
            }
        )
    return maximum <= 1e-6, maximum, rows


def _physical_roundtrip(
    trace: Mapping[str, Any],
    run: Mapping[str, Any],
) -> dict[str, Any]:
    dump = Path(str(run["hbm_dump_path"])).read_bytes()
    source = int(trace["source_base"])
    destination = int(trace["destination_base"])
    size = int(trace["payload_bytes"])
    source_payload = dump[source:source + size]
    destination_payload = dump[destination:destination + size]
    element_bytes = int(trace["element_plane_bytes"])
    return {
        "valid": source_payload == destination_payload,
        "source_sha256": hashlib.sha256(source_payload).hexdigest(),
        "destination_sha256": hashlib.sha256(destination_payload).hexdigest(),
        "element_plane_valid": (
            source_payload[:element_bytes] == destination_payload[:element_bytes]
        ),
        "scale_plane_valid": (
            source_payload[element_bytes:] == destination_payload[element_bytes:]
        ),
        "payload_bytes": size,
    }


def _fpsram_payload() -> bytes:
    values = [0.0, 1.0 / math.sqrt(SCALED_HLEN), -60000.0]
    values.extend([0.0] * (256 - len(values)))
    return b"".join(struct.pack("<e", value) for value in values)


def _compiler_worker_command(
    request_path: Path,
    compiled_root: Path,
    bundle_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-compile",
        "--request",
        str(request_path.resolve()),
        "--artifact-dir",
        str(compiled_root.resolve()),
        "--bundle",
        str(bundle_path.resolve()),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=SIMULATOR_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    record = {
        "command": command,
        "cwd": str(SIMULATOR_ROOT),
        "return_code": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
    }
    if completed.returncode or not bundle_path.is_file():
        raise HookError(
            "compiler worker failed: "
            + completed.stderr.strip()[-1000:]
        )
    return command, record


def _result_body(
    request: Mapping[str, Any],
    tests: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _with_content_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "stage": "emulator",
            "manifest_hash": request["manifest_hash"],
            "profile_id": request["profile_id"],
            "request_content_hash": request["content_hash"],
            "observed_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "tests": list(tests),
            "artifacts": list(artifacts),
        }
    )


def run_hook(
    request_path: Path,
    result_path: Path,
    artifact_dir: Path,
    *,
    emulator_binary: Path | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    request = _validate_request(_load_json(request_path))
    artifact_root = artifact_dir.resolve()
    existing = _existing_result(
        result_path,
        request["content_hash"],
        artifact_root,
    )
    if existing is not None:
        return existing
    artifact_root.mkdir(parents=True, exist_ok=True)
    profile_root = artifact_root / request["profile_id"]
    profile_root.mkdir(parents=True, exist_ok=True)
    supported, reason, descriptors = _profile_support(
        request["profile"],
        request["target"],
    )
    if not supported:
        evidence_target = _expected_compiler_target(request)
        rejection = _with_content_hash(
            {
                "schema_version": REJECTION_SCHEMA,
                "profile_id": request["profile_id"],
                "reason_code": reason,
                "profile": request["profile"],
                "target": request["target"],
                "evidence_target": evidence_target,
            }
        )
        rejection_path = profile_root / "rejection.json"
        _write_json(rejection_path, rejection)
        result = _result_body(
            request,
            [
                {
                    "name": "profile_support",
                    "passed": False,
                    "metrics": {
                        "reason_code": reason,
                        "weight_format": request["profile"]["weight_format"],
                        "activation_format": request["profile"]["activation_format"],
                        "kv_format": request["profile"]["key_format"],
                        "vector_format": request["profile"]["vector_format"],
                        "block_size": request["profile"]["block_size"],
                        "evidence_target": evidence_target,
                    },
                }
            ],
            [_artifact(rejection_path, "emulator_profile_rejection")],
        )
        _write_json(result_path, result)
        return result

    binary, build_record = _resolve_binary(emulator_binary)
    compiled_root = profile_root / "compiled"
    bundle_path = profile_root / "compiler-bundle.json"
    _, compiler_record = _compiler_worker_command(
        request_path,
        compiled_root,
        bundle_path,
    )
    bundle = _load_json(bundle_path)
    if (
        bundle.get("schema_version") != BUNDLE_SCHEMA
        or bundle.get("profile_id") != request["profile_id"]
        or bundle.get("request_content_hash") != request["content_hash"]
        or bundle.get("content_hash") != _content_hash(bundle)
    ):
        raise HookError("compiler worker bundle is invalid")
    binding = bundle["binding"]
    if binding != _build_binding(request, descriptors):
        raise HookError("compiler worker binding differs from the canonical binding")
    evidence_target = _expected_compiler_target(request)
    if binding.get("evidence_target") != evidence_target:
        raise HookError("compiler worker evidence target is invalid")
    settings_path = Path(bundle["settings_path"])
    if not settings_path.is_file():
        raise HookError("compiler worker did not materialize emulator settings")
    fpsram_path = profile_root / "fpsram.bin"
    _write_immutable(fpsram_path, _fpsram_payload())
    traces = bundle["traces"]
    run_root = profile_root / "runs"
    runs: dict[str, dict[str, Any]] = {}
    roundtrip_trace = traces["roundtrip"]
    runs["roundtrip"] = _run_emulator(
        binary,
        settings_path,
        roundtrip_trace,
        run_root / "roundtrip",
        dump_hbm=True,
        timeout_seconds=timeout_seconds,
    )
    roundtrip = _physical_roundtrip(
        roundtrip_trace,
        runs["roundtrip"],
    )
    linear_trace = traces["linear"]
    runs["linear"] = _run_emulator(
        binary,
        settings_path,
        linear_trace,
        run_root / "linear",
        timeout_seconds=timeout_seconds,
    )
    linear_expected = [
        {
            "address_elements": int(linear_trace["output_base_elements"])
            + row * SCALED_MLEN,
            "values": values,
        }
        for row, values in enumerate(linear_trace["expected_rows"])
    ]
    linear_pass, linear_error, linear_rows = _compare_rows(
        Path(runs["linear"]["output_path"]),
        linear_expected,
        request["profile"]["vector_format"],
    )
    attention_results: dict[str, Any] = {}
    for trace in traces["attention"]:
        key = str(trace["batch_size"])
        run = _run_emulator(
            binary,
            settings_path,
            trace,
            run_root / trace["name"],
            timeout_seconds=timeout_seconds,
        )
        runs[trace["name"]] = run
        passed, maximum, rows = _compare_rows(
            Path(run["output_path"]),
            trace["expected_rows"],
            request["profile"]["vector_format"],
        )
        attention_results[key] = {
            "passed": passed,
            "max_abs_error": maximum,
            "rows": rows,
        }
    semantics = binding["runtime_precision_contract"]["matrix_semantics"]
    physical = binding["runtime_precision_contract"]["physical_semantics"]
    canonical = canonical_mxint_vectors()
    physical_contract_valid = (
        physical["schema_version"] == MX_PHYSICAL_SEMANTICS_SCHEMA
        and physical["schema_version"] == MX_PHYSICAL_SEMANTICS_ID
        and physical["mxint_scale_rule"] == MXINT_SCALE_RULE
        and physical["mxint_encoding"] == "sign_magnitude"
        and physical["mxint_canonical_zero"] == "positive_zero"
        and physical["block_size"] == 8
        and canonical["maximum_e8m0_finite"]
        and canonical["zero_times_maximum_scale_is_zero"]
    )
    common = {
        "evidence_target": evidence_target,
        "common_deployment_valid": False,
        "rtl_deployment_valid": False,
        "machine_code_executed": True,
        "runtime_profile_binding_valid": True,
        "matrix_semantics_binding_valid": (
            semantics["schema_version"] == MATRIX_SEMANTICS_SCHEMA
            and semantics["source_profile_schema"] == PROFILE_SCHEMA
            and semantics["content_hash"]
            == binding["runtime_precision_contract"][
                "accumulator_storage_policy"
            ]["family_semantics_sha256"]
        ),
        "structural_precision_binding_valid": True,
        "numerical_trace_conformance": "not_run",
        "matrix_semantics_sha256": semantics["content_hash"],
        "physical_semantics_sha256": physical["content_hash"],
        "physical_semantics_binding_valid": physical_contract_valid,
        "mxint_sign_magnitude_valid": physical_contract_valid,
        "mxint_range_safe_scale_valid": physical_contract_valid,
        "canonical_zero_valid": physical_contract_valid,
        "e8m0_code255_valid": physical_contract_valid,
        "native_block8_valid": True,
        "activation_matrix_port_conversion_valid": True,
        "vector_rounding_independent_valid": True,
        "independent_reference_valid": True,
        "numerical_validation": True,
    }
    tests: list[dict[str, Any]] = [
        {
            "name": "profile_support",
            "passed": True,
            "metrics": {
                "binding_id": binding["binding_id"],
                "weight_format": request["profile"]["weight_format"],
                "activation_format": request["profile"]["activation_format"],
                "kv_format": request["profile"]["key_format"],
                "vector_format": request["profile"]["vector_format"],
                "block_size": request["profile"]["block_size"],
                "profile_schema": PROFILE_SCHEMA,
                "matrix_semantics_schema": MATRIX_SEMANTICS_SCHEMA,
                "physical_semantics_schema": MX_PHYSICAL_SEMANTICS_SCHEMA,
                "evidence_target": evidence_target,
                "common_deployment_valid": False,
                "rtl_deployment_valid": False,
            },
        },
        {
            "name": "linear_w_a",
            "passed": bool(linear_pass and physical_contract_valid),
            "metrics": {
                **common,
                "weight_precision_valid": linear_trace[
                    "weight_prefetch_precision_valid"
                ],
                "allclose_pass": linear_pass,
                "max_abs_error": linear_error,
                "output_rows": linear_rows,
                "return_code": runs["linear"]["return_code"],
                "opcode_coverage_valid": runs["linear"][
                    "opcode_coverage_valid"
                ],
                "machine_code_sha256": runs["linear"][
                    "machine_code_sha256"
                ],
                "input_sha256": runs["linear"]["hbm_preload_sha256"],
                "output_sha256": runs["linear"]["output_sha256"],
            },
        },
    ]
    for trace in traces["attention"]:
        batch_size = int(trace["batch_size"])
        result = attention_results[str(batch_size)]
        run = runs[trace["name"]]
        compiler_metrics = trace["compiler_metrics"]
        batch_slabs = {
            mapping["batch"]: mapping["cache_slab_element_offset"]
            for mapping in compiler_metrics["trace_contract"]["slab_mappings"]
        } if "trace_contract" in compiler_metrics else {}
        slabs_disjoint = (
            len(batch_slabs) == batch_size
            if batch_slabs
            else compiler_metrics["batch_slab_mapping_valid"]
        )
        passed = bool(
            result["passed"]
            and roundtrip["valid"]
            and trace["writeout_stride_valid"]
            and compiler_metrics["q_len"] == 1
            and compiler_metrics["cache_position"]
            == compiler_metrics["cache_tokens"] - 1
            and compiler_metrics["selector_sequence_valid"]
            and compiler_metrics["batch_slab_mapping_valid"]
            and run["opcode_coverage_valid"]
            and physical_contract_valid
        )
        tests.append(
            {
                "name": f"packedkv_q1_batch_{batch_size}",
                "passed": passed,
                "metrics": {
                    **common,
                    "batch_size": batch_size,
                    "q_len": 1,
                    "cache_tokens": compiler_metrics["cache_tokens"],
                    "cache_position": compiler_metrics["cache_position"],
                    "block_size": 8,
                    "return_code": run["return_code"],
                    "kv_image_roundtrip_valid": roundtrip["valid"],
                    "kv_element_plane_roundtrip_valid": roundtrip[
                        "element_plane_valid"
                    ],
                    "kv_scale_plane_roundtrip_valid": roundtrip[
                        "scale_plane_valid"
                    ],
                    "kv_prefetch_precision_valid": compiler_metrics[
                        "kv_prefetch_precision_valid"
                    ],
                    "selector_sequence_valid": compiler_metrics[
                        "selector_sequence_valid"
                    ],
                    "batch_slab_mapping_valid": compiler_metrics[
                        "batch_slab_mapping_valid"
                    ],
                    "batch_slabs_disjoint": slabs_disjoint,
                    "writeout_stride_valid": trace[
                        "writeout_stride_valid"
                    ],
                    "physical_byte_addressing_valid": compiler_metrics[
                        "physical_byte_addressing_valid"
                    ],
                    "allclose_pass": result["passed"],
                    "max_abs_error": result["max_abs_error"],
                    "output_rows": result["rows"],
                    "op_stats_recorded": True,
                    "opcode_coverage_valid": run[
                        "opcode_coverage_valid"
                    ],
                    "machine_code_sha256": run[
                        "machine_code_sha256"
                    ],
                    "input_sha256": run["hbm_preload_sha256"],
                    "output_sha256": run["output_sha256"],
                    "op_stats_sha256": run["op_stats_sha256"],
                    "evidence_scope": "scaled_operator_numerical_conformance",
                    "full_geometry_timing_evidence": False,
                },
            }
        )
    run_metrics = _with_content_hash(
        {
            "schema_version": RUN_METRICS_SCHEMA,
            "profile_id": request["profile_id"],
            "binding_id": binding["binding_id"],
            "source_tree_sha256": request["source_tree_sha256"],
            "compiler_worker": compiler_record,
            "emulator_build": build_record,
            "emulator_binary": str(binary.resolve()),
            "emulator_binary_sha256": _sha256_file(binary),
            "settings_sha256": _sha256_file(settings_path),
            "binding_sha256": _sha256_file(Path(bundle["binding_path"])),
            "matrix_semantics_sha256": semantics["content_hash"],
            "physical_semantics_sha256": physical["content_hash"],
            "physical_roundtrip": roundtrip,
            "runs": runs,
        }
    )
    run_metrics_path = profile_root / "run-metrics.json"
    _write_json(run_metrics_path, run_metrics)
    golden_path = profile_root / "golden.json"
    _write_json(
        golden_path,
        {
            "schema_version": "plena-emulator-independent-golden/v1",
            "linear": linear_rows,
            "attention": attention_results,
            "physical_roundtrip": roundtrip,
        },
    )
    representative = next(
        trace for trace in traces["attention"] if trace["batch_size"] == 4
    )
    representative_run = runs[representative["name"]]
    artifacts = [
        _artifact(settings_path, "emulator_settings"),
        _artifact(Path(representative["machine_path"]), "emulator_machine_code"),
        _artifact(Path(representative["hbm_path"]), "emulator_hbm_preload"),
        _artifact(Path(representative["vram_path"]), "emulator_vram_preload"),
        _artifact(golden_path, "emulator_golden"),
        _artifact(Path(representative_run["output_path"]), "emulator_output"),
        _artifact(run_metrics_path, "emulator_run_metrics"),
        _artifact(Path(representative_run["op_stats_path"]), "emulator_op_stats"),
    ]
    artifact_ids = [artifact["artifact_id"] for artifact in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise HookError("required emulator artifacts are not independently hashed")
    result = _result_body(request, tests, artifacts)
    _write_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--emulator-binary", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--worker-compile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.worker_compile:
            if args.bundle is None:
                raise HookError("--worker-compile requires --bundle")
            _worker_compile(args.request, args.artifact_dir, args.bundle)
        else:
            if args.result is None:
                raise HookError("emulator hook requires --result")
            run_hook(
                args.request,
                args.result,
                args.artifact_dir,
                emulator_binary=args.emulator_binary,
                timeout_seconds=args.timeout_seconds,
            )
    except Exception as exc:
        print(f"emulator PackedKV hook failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
