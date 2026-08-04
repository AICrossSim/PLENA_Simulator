"""Structured Ramulator DMA calibration for the request-latency model.

Each measured descriptor uses a fresh emulator process by default so
Ramulator state cannot leak between observations.  The process fixes the
hardware settings that are read once at startup, and ``--blocking-prefetch``
makes prefetch instruction timing include its DMA service.  Per-instruction
``--op-stats`` records retain the exact physical traffic and latency.

The resulting CSV is consumed by ``analytic_models.disagg_serve.memory``.  It
contains the compiler-visible descriptor as well as measured Ramulator traffic
and time, so the fitted request model can be reproduced without inferring
tensor geometry from aggregate bandwidth.

Usage from the simulator repository root::

    python transactional_emulator/testbench/calibration/dma_microbench.py

Use ``--dry-run`` to review the complete Cartesian sweep without allocating
HBM or launching the emulator.  ``--max-configurations`` provides a bounded
smoke run while preserving the same deterministic ordering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import tomlkit

_ROOT_HINT = Path(__file__).resolve().parents[3]
if str(_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(_ROOT_HINT))

from runtime_paths import settings_path, simulator_root

_HERE = Path(__file__).resolve().parent
_REPO = simulator_root()
_EMULATOR = _REPO / "transactional_emulator"

MLEN = 64
VLEN = 64
SCALE_PLANE_OFFSET = 64 * 1024 * 1024
HBM_IMAGE_BYTES = 128 * 1024 * 1024
BASE_ADDRESS = 1024 * 1024
PIN_RATE_GBPS = {"HBM2": 2.0, "HBM3": 2.0}
OPS = ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
VECTOR_OPS = ("H_PREFETCH_V", "H_STORE_V")


@dataclass(frozen=True)
class PrecisionSpec:
    name: str
    family: str
    element_bits: int
    exponent_bits: int | None = None
    mantissa_bits: int | None = None
    block_size: int = 8
    scale_bits: int = 8

    @property
    def element_scale_ratio(self) -> int:
        numerator = self.element_bits * self.block_size
        if numerator % self.scale_bits:
            raise ValueError(f"precision {self.name} has a fractional scale stride")
        return numerator // self.scale_bits

    def mx_config(self) -> dict[str, Any]:
        if self.family == "mxint":
            element = {"type": "Int", "width": self.element_bits}
        elif self.family == "mxfp":
            if self.exponent_bits is None or self.mantissa_bits is None:
                raise ValueError(f"precision {self.name} lacks an FP field layout")
            element = {
                "type": "Fp",
                "sign": True,
                "exponent": self.exponent_bits,
                "mantissa": self.mantissa_bits,
            }
        else:
            raise ValueError(f"unsupported precision family {self.family!r}")
        return {
            "format": "Mx",
            "block": self.block_size,
            "ELEM": element,
            "SCALE": {
                "type": "Fp",
                "sign": False,
                "exponent": self.scale_bits,
                "mantissa": 0,
            },
        }


PRECISIONS = {
    spec.name: spec
    for spec in (
        PrecisionSpec("mxfp4_e2m1", "mxfp", 4, 2, 1),
        PrecisionSpec("mxfp8_e4m3", "mxfp", 8, 4, 3),
        PrecisionSpec("mxint4", "mxint", 4),
        PrecisionSpec("mxint8", "mxint", 8),
    )
}


@dataclass(frozen=True)
class SweepConfiguration:
    hbm_generation: str
    channels: int
    rows: int
    precision: PrecisionSpec


@dataclass(frozen=True)
class PlannedRequest:
    opcode: str
    address: int
    rows: int
    elements_per_row: int
    stride_bytes: int
    alignment_bytes: int
    replica: int
    precision: PrecisionSpec

    @property
    def direction(self) -> str:
        return "write" if self.opcode == "H_STORE_V" else "read"

    @property
    def tensor(self) -> str:
        return {
            "H_PREFETCH_M": "matrix_weight",
            "H_PREFETCH_V": "vector_activation",
            "H_STORE_V": "vector_writeback",
        }[self.opcode]

    @property
    def scale_address(self) -> int:
        return self.address + SCALE_PLANE_OFFSET

    @property
    def scale_stride_bytes(self) -> int:
        return self.stride_bytes // self.precision.element_scale_ratio

    @property
    def element_bytes_per_row(self) -> int:
        return self.elements_per_row * self.precision.element_bits // 8


@dataclass(frozen=True)
class ProcessMeasurement:
    """Measured observations and the complete receipt for one emulator process."""

    rows: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def _parse_csv_ints(value: str, *, name: str, allow_zero: bool = False) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {qualifier} integers")
    return values


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def plan_requests(
    configuration: SweepConfiguration,
    *,
    stride_multipliers: Sequence[int],
    alignments: Sequence[int],
    replicas: int,
) -> tuple[PlannedRequest, ...]:
    """Create deterministic, non-overlapping physical request descriptors."""

    if replicas <= 0:
        raise ValueError("replicas must be positive")
    # H_PREFETCH_M moves whole MLEN tiles, so it is measured only at row counts
    # that are MLEN multiples.  The vector engines carry an independent transfer
    # amount and decode issues theirs far below MLEN, so smaller row counts
    # measure H_PREFETCH_V and H_STORE_V alone.
    opcodes = OPS if configuration.rows % MLEN == 0 else VECTOR_OPS
    ratio = configuration.precision.element_scale_ratio
    row_bytes = VLEN * configuration.precision.element_bits // 8
    if any(multiplier <= 0 for multiplier in stride_multipliers):
        raise ValueError("stride multipliers must be positive")
    if any(alignment < 0 or alignment >= 64 for alignment in alignments):
        raise ValueError("alignments must lie in [0, 64)")

    cursor = BASE_ADDRESS
    requests = []
    for opcode in opcodes:
        for multiplier in stride_multipliers:
            stride = row_bytes * multiplier
            if stride % ratio:
                raise ValueError("physical stride does not align with the scale plane")
            footprint = (configuration.rows - 1) * stride + row_bytes
            for alignment in alignments:
                for replica in range(replicas):
                    base = _align_up(cursor, 64) + alignment
                    requests.append(
                        PlannedRequest(
                            opcode=opcode,
                            address=base,
                            rows=configuration.rows,
                            elements_per_row=VLEN,
                            stride_bytes=stride,
                            alignment_bytes=alignment,
                            replica=replica,
                            precision=configuration.precision,
                        )
                    )
                    cursor = base + footprint + 4096
    if cursor + SCALE_PLANE_OFFSET >= HBM_IMAGE_BYTES:
        raise ValueError(
            "sweep program exceeds the bounded HBM image; reduce rows, stride, or replicas"
        )
    return tuple(requests)


def build_asm(requests: Sequence[PlannedRequest]) -> str:
    """Emit one instruction for every planned physical request."""

    from asm_templates._imm import load_large_int_str

    lines = [
        "; Structured DMA calibration",
        load_large_int_str(2, SCALE_PLANE_OFFSET).rstrip("\n"),
        "C_SET_SCALE_REG gp2",
        "S_ADDI_INT gp4, gp0, 0",
        "S_ADDI_INT gp5, gp0, 0",
    ]
    active_stride = None
    for request in requests:
        if request.stride_bytes != active_stride:
            lines.extend(
                (
                    load_large_int_str(3, request.stride_bytes).rstrip("\n"),
                    "C_SET_STRIDE_REG gp3",
                )
            )
            active_stride = request.stride_bytes
        lines.extend(
            (
                load_large_int_str(1, request.address).rstrip("\n"),
                "C_SET_ADDR_REG a1, gp0, gp1",
                f"{request.opcode} gp5, gp4, a1, 1, 0",
            )
        )
    lines.append("C_BREAK")
    return "\n".join(lines) + "\n"


def patch_settings(
    base_toml: str,
    configuration: SweepConfiguration,
) -> str:
    """Apply one process-wide sweep configuration to the emulator TOML."""

    document = tomlkit.loads(base_toml)
    transactional = document["TRANSACTIONAL"]
    config = transactional["CONFIG"]
    config["HBM_GEN"]["value"] = configuration.hbm_generation
    config["HBM_CHANNELS"]["value"] = configuration.channels
    if configuration.rows % MLEN == 0:
        config["HBM_M_Prefetch_Amount"]["value"] = configuration.rows
        config["MATRIX_SRAM_SIZE"]["value"] = max(
            int(config["MATRIX_SRAM_SIZE"]["value"]),
            configuration.rows * MLEN,
        )
    config["HBM_V_Prefetch_Amount"]["value"] = configuration.rows
    config["HBM_V_Writeback_Amount"]["value"] = configuration.rows
    config["VECTOR_SRAM_SIZE"]["value"] = max(
        int(config["VECTOR_SRAM_SIZE"]["value"]),
        configuration.rows,
    )
    precision = transactional["PRECISION"]
    for key in ("HBM_M_WEIGHT_TYPE", "HBM_V_ACT_TYPE"):
        precision[key] = configuration.precision.mx_config()
    return tomlkit.dumps(document)


def parse_instruction_stats(
    path: Path,
    requests: Sequence[PlannedRequest],
) -> tuple[dict[str, Any], ...]:
    """Match ordered target instructions to their planned descriptors."""

    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("op") in OPS and not record.get("aggregate"):
                records.append(record)
    if len(records) != len(requests):
        raise RuntimeError(
            f"op-stats contains {len(records)} DMA records for {len(requests)} requests"
        )
    for request, record in zip(requests, records):
        if record["op"] != request.opcode:
            raise RuntimeError(
                f"op-stats order mismatch: expected {request.opcode}, found {record['op']}"
            )
        if int(record["dt_ps"]) <= 0:
            raise RuntimeError(f"Ramulator reported zero service time for {request.opcode}")
    return tuple(records)


def observation_row(
    configuration: SweepConfiguration,
    request: PlannedRequest,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine a physical descriptor with its measured Ramulator outcome."""

    return {
        "schema_version": "plena-dma-request-calibration",
        "opcode": request.opcode,
        "hbm_generation": configuration.hbm_generation,
        "channels": configuration.channels,
        "address": request.address,
        "rows": request.rows,
        "elements_per_row": request.elements_per_row,
        "stride_bytes": request.stride_bytes,
        "alignment_bytes": request.alignment_bytes,
        "element_bits": request.precision.element_bits,
        "direction": request.direction,
        "pin_rate_gbps": PIN_RATE_GBPS[configuration.hbm_generation],
        "tensor": request.tensor,
        "precision": request.precision.name,
        "scale_bits": request.precision.scale_bits,
        "block_size": request.precision.block_size,
        "scale_address": request.scale_address,
        "scale_stride_bytes": request.scale_stride_bytes,
        "partial_write_rmw": request.opcode == "H_STORE_V",
        "replica": request.replica,
        "dt_ps": int(record["dt_ps"]),
        "measured_s": int(record["dt_ps"]) / 1e12,
        "physical_read_bytes": int(record.get("hbm_issue_rd", 0)),
        "physical_write_bytes": int(record.get("hbm_issue_wr", 0)),
        "instruction_pc": int(record["pc"]),
        "evidence_tier": "ramulator2_simulated",
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _source_tree_sha256(root: Path, suffixes: set[str]) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "__pycache__" not in path.parts
        and "target" not in path.parts
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _command_text(command: Sequence[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        return f"unavailable (exit {completed.returncode}): {completed.stderr.strip()}"
    return completed.stdout.strip()


def _planned_request_dict(request: PlannedRequest) -> dict[str, Any]:
    return {
        "opcode": request.opcode,
        "address": request.address,
        "rows": request.rows,
        "elements_per_row": request.elements_per_row,
        "stride_bytes": request.stride_bytes,
        "alignment_bytes": request.alignment_bytes,
        "replica": request.replica,
        "direction": request.direction,
        "tensor": request.tensor,
        "precision": request.precision.name,
        "element_bits": request.precision.element_bits,
        "scale_bits": request.precision.scale_bits,
        "block_size": request.precision.block_size,
        "scale_address": request.scale_address,
        "scale_stride_bytes": request.scale_stride_bytes,
    }


def _receipt_context(
    *,
    binary: Path,
    base_toml: str,
    summary: Mapping[str, Any],
    receipt_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    compiler = _REPO / "compiler"
    ramulator = _EMULATOR / "lib" / "ramulator"
    tracked_environment = {
        name: environment.get(name)
        for name in (
            "LD_LIBRARY_PATH",
            "OMP_NUM_THREADS",
            "PLENA_SIMULATOR_PATH",
            "PYTHONPATH",
        )
    }
    return {
        "schema_version": "plena-dma-request-calibration-receipt",
        "invocation": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "receipt_path": str(receipt_path.resolve()),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": tracked_environment,
        "toolchain": {
            "python": sys.version,
            "python_executable": sys.executable,
            "rustc": _command_text(("rustc", "--version", "--verbose")),
            "cargo": _command_text(("cargo", "--version", "--verbose")),
            "linked_libraries": _command_text(("ldd", str(binary))),
        },
        "emulator": {
            "path": str(binary.resolve()),
            "sha256": _file_sha256(binary),
            "ramulator_source_sha256": _source_tree_sha256(
                ramulator,
                {".rs", ".toml", ".lock"},
            ),
        },
        "compiler": {
            "path": str(compiler.resolve()),
            "revision": _command_text(("git", "-C", str(compiler), "rev-parse", "HEAD")),
            "status": _command_text(
                ("git", "-C", str(compiler), "status", "--short", "--untracked-files=all")
            ),
            "source_sha256": _source_tree_sha256(
                compiler,
                {".py", ".svh", ".json"},
            ),
        },
        "settings": {
            "path": str(settings_path().resolve()),
            "sha256": _sha256(base_toml.encode("utf-8")),
        },
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "sweep_plan": dict(summary),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _emulator_binary() -> Path:
    binary = _EMULATOR / "target" / "release" / "transactional_emulator"
    if not binary.is_file():
        raise RuntimeError(
            "release emulator is absent; run cargo build --release in transactional_emulator"
        )
    inputs: Iterable[Path] = (
        *_EMULATOR.glob("Cargo*.toml"),
        *_EMULATOR.glob("Cargo.lock"),
        *_EMULATOR.glob("src/**/*.rs"),
        *_EMULATOR.glob("lib/**/*.rs"),
        *_EMULATOR.glob("lib/**/Cargo.toml"),
    )
    newer = sorted(path for path in inputs if path.stat().st_mtime_ns > binary.stat().st_mtime_ns)
    if newer:
        sample = ", ".join(str(path.relative_to(_REPO)) for path in newer[:3])
        raise RuntimeError(f"release emulator is stale relative to {sample}")
    return binary


def _configuration_grid(
    generations: Sequence[str],
    channels: Sequence[int],
    row_counts: Sequence[int],
    precisions: Sequence[PrecisionSpec],
) -> tuple[SweepConfiguration, ...]:
    configurations = []
    for generation in generations:
        if generation not in PIN_RATE_GBPS:
            raise ValueError(f"unsupported HBM generation {generation!r}")
        for channel_count in channels:
            if channel_count <= 0 or channel_count & (channel_count - 1):
                raise ValueError("HBM channel counts must be powers of two")
            for rows in row_counts:
                for precision in precisions:
                    configurations.append(
                        SweepConfiguration(generation, channel_count, rows, precision)
                    )
    return tuple(configurations)


def _run_configuration(
    configuration: SweepConfiguration,
    requests: Sequence[PlannedRequest],
    *,
    base_toml: str,
    binary: Path,
    temporary: Path,
    hbm_path: Path,
    fpsram_path: Path,
    intsram_path: Path,
    environment: Mapping[str, str],
    assembler: Any,
    request_tag: str = "",
    assembler_lock: threading.Lock | None = None,
) -> ProcessMeasurement:
    stem = (
        f"{configuration.hbm_generation}_{configuration.channels}_"
        f"{configuration.rows}_{configuration.precision.name}{request_tag}"
    )
    asm_path = temporary / f"{stem}.asm"
    opcode_path = temporary / f"{stem}.mem"
    toml_path = temporary / f"{stem}.toml"
    stats_path = temporary / f"{stem}.jsonl"
    asm_path.write_text(build_asm(requests), encoding="utf-8")
    if assembler_lock is None:
        assembler.generate_binary(str(asm_path), str(opcode_path))
    else:
        with assembler_lock:
            assembler.generate_binary(str(asm_path), str(opcode_path))
    toml_path.write_text(patch_settings(base_toml, configuration), encoding="utf-8")
    command = [
        str(binary),
        "--opcode",
        str(opcode_path),
        "--hbm",
        str(hbm_path),
        "--fpsram",
        str(fpsram_path),
        "--intsram",
        str(intsram_path),
        "--hbm-size",
        "128M",
        "--settings",
        str(toml_path),
        "--hbm-gen",
        configuration.hbm_generation,
        "--hbm-channels",
        str(configuration.channels),
        "--log-level",
        "warn",
        "--blocking-prefetch",
        "--op-stats",
        str(stats_path),
    ]
    process = subprocess.run(command, env=environment, capture_output=True, text=True)
    if process.returncode:
        sys.stderr.write(process.stderr[-4000:] + "\n")
        raise RuntimeError(f"emulator failed for {stem}")
    records = parse_instruction_stats(stats_path, requests)
    rows = tuple(
        observation_row(configuration, request, record)
        for request, record in zip(requests, records)
    )
    stats_jsonl = stats_path.read_text(encoding="utf-8")
    receipt = {
        "process_id": stem,
        "configuration": {
            "hbm_generation": configuration.hbm_generation,
            "channels": configuration.channels,
            "rows": configuration.rows,
            "precision": configuration.precision.name,
        },
        "requests": [_planned_request_dict(request) for request in requests],
        "command": command,
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "artifacts": {
            "assembly_sha256": _file_sha256(asm_path),
            "opcode_sha256": _file_sha256(opcode_path),
            "settings_sha256": _file_sha256(toml_path),
            "op_stats_sha256": _sha256(stats_jsonl.encode("utf-8")),
        },
        "op_stats_jsonl": stats_jsonl,
        "observations": list(rows),
    }
    return ProcessMeasurement(rows=rows, receipt=receipt)


def _dry_run_summary(
    configurations: Sequence[SweepConfiguration],
    requests_per_configuration: Sequence[int],
    output: Path,
    receipt: Path,
    *,
    request_isolation: bool,
    workers: int,
) -> dict[str, Any]:
    return {
        "schema_version": "plena-dma-request-sweep-plan",
        "configuration_count": len(configurations),
        "observation_count": sum(requests_per_configuration),
        "output": str(output.resolve()),
        "receipt": str(receipt.resolve()),
        "hbm_image_bytes": HBM_IMAGE_BYTES,
        "generations": sorted({item.hbm_generation for item in configurations}),
        "channels": sorted({item.channels for item in configurations}),
        "row_counts": sorted({item.rows for item in configurations}),
        "precisions": sorted({item.precision.name for item in configurations}),
        "opcodes": list(OPS),
        "vector_only_row_counts": sorted(
            {item.rows for item in configurations if item.rows % MLEN}
        ),
        "request_isolation": request_isolation,
        "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-counts", default="4,16,64,256")
    parser.add_argument("--stride-multipliers", default="1,4,32")
    parser.add_argument("--alignments", default="0,16,32,48")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--channels", default="8,32,128")
    parser.add_argument("--generations", default="HBM2,HBM3")
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--max-configurations", type=int)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--isolate-requests",
        dest="isolate_requests",
        action="store_true",
        default=True,
        help="launch a fresh Ramulator process for every descriptor",
    )
    execution.add_argument(
        "--batched",
        dest="isolate_requests",
        action="store_false",
        help="batch descriptors for a fast smoke run with state carry-over",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO
        / "analytic_models"
        / "disagg_serve"
        / "calibration_dma_requests.csv",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="complete process-level receipt (defaults beside --out)",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    row_counts = _parse_csv_ints(args.row_counts, name="row counts")
    stride_multipliers = _parse_csv_ints(
        args.stride_multipliers,
        name="stride multipliers",
    )
    alignments = _parse_csv_ints(args.alignments, name="alignments", allow_zero=True)
    channels = _parse_csv_ints(args.channels, name="channels")
    generations = tuple(item.strip() for item in args.generations.split(",") if item.strip())
    try:
        precisions = tuple(
            PRECISIONS[item.strip()]
            for item in args.precisions.split(",")
            if item.strip()
        )
    except KeyError as error:
        raise ValueError(f"unknown precision {error.args[0]!r}") from error
    configurations = _configuration_grid(generations, channels, row_counts, precisions)
    if args.max_configurations is not None:
        if args.max_configurations <= 0:
            raise ValueError("max configurations must be positive")
        configurations = configurations[: args.max_configurations]

    plans = tuple(
        plan_requests(
            configuration,
            stride_multipliers=stride_multipliers,
            alignments=alignments,
            replicas=args.replicas,
        )
        for configuration in configurations
    )
    receipt_path = args.receipt or args.out.with_suffix(".receipt.json")
    summary = _dry_run_summary(
        configurations,
        tuple(len(requests) for requests in plans),
        args.out,
        receipt_path,
        request_isolation=args.isolate_requests,
        workers=args.workers,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return

    binary = _emulator_binary()
    sys.path.insert(0, str(_REPO / "compiler"))
    from assembler.assembly_to_binary import AssemblyToBinary

    assembler = AssemblyToBinary(
        str(_REPO / "compiler" / "doc" / "operation.svh"),
        str(_REPO / "compiler" / "doc" / "configuration.svh"),
    )
    base_toml = settings_path().read_text(encoding="utf-8")
    environment = dict(os.environ)
    libtorch = list(
        (_EMULATOR / "target" / "release" / "build").glob(
            "torch-sys-*/out/libtorch/libtorch/lib"
        )
    )
    if libtorch:
        environment["LD_LIBRARY_PATH"] = (
            f"{libtorch[0]}:{environment.get('LD_LIBRARY_PATH', '')}"
        )
    environment.setdefault("OMP_NUM_THREADS", "1")
    receipt_context = _receipt_context(
        binary=binary,
        base_toml=base_toml,
        summary=summary,
        receipt_path=receipt_path,
        environment=environment,
    )

    rows: list[dict[str, Any]] = []
    process_receipts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="plena_dma_") as temporary_name:
        temporary = Path(temporary_name)
        hbm_path = temporary / "hbm.bin"
        with hbm_path.open("wb") as handle:
            handle.truncate(HBM_IMAGE_BYTES)
        fpsram_path = temporary / "fp_sram.bin"
        intsram_path = temporary / "int_sram.bin"
        fpsram_path.write_bytes(b"\x00" * 8)
        intsram_path.write_bytes(b"\x00" * 8)

        assembler_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (configuration, requests) in enumerate(
                zip(configurations, plans),
                start=1,
            ):
                if args.isolate_requests:
                    def run_request(
                        indexed_request: tuple[int, PlannedRequest],
                    ) -> ProcessMeasurement:
                        request_index, request = indexed_request
                        return _run_configuration(
                            configuration,
                            (request,),
                            base_toml=base_toml,
                            binary=binary,
                            temporary=temporary,
                            hbm_path=hbm_path,
                            fpsram_path=fpsram_path,
                            intsram_path=intsram_path,
                            environment=environment,
                            assembler=assembler,
                            request_tag=f"_{request_index}",
                            assembler_lock=assembler_lock,
                        )

                    results = tuple(pool.map(run_request, enumerate(requests)))
                else:
                    results = (
                        _run_configuration(
                            configuration,
                            requests,
                            base_toml=base_toml,
                            binary=binary,
                            temporary=temporary,
                            hbm_path=hbm_path,
                            fpsram_path=fpsram_path,
                            intsram_path=intsram_path,
                            environment=environment,
                            assembler=assembler,
                        ),
                    )
                measured = tuple(row for result in results for row in result.rows)
                process_receipts.extend(result.receipt for result in results)
                rows.extend(measured)
                mean_ns = sum(row["dt_ps"] for row in measured) / len(measured) / 1000
                print(
                    f"[dma] {index}/{len(configurations)} "
                    f"{configuration.hbm_generation}/{configuration.channels}ch "
                    f"rows={configuration.rows} {configuration.precision.name}: "
                    f"{len(measured)} requests, mean={mean_ns:.1f} ns",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("DMA sweep produced no observations")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dma] wrote {len(rows)} observations -> {args.out}", flush=True)

    receipt = {
        **receipt_context,
        "calibration": {
            "path": str(args.out.resolve()),
            "sha256": _file_sha256(args.out),
            "observation_count": len(rows),
        },
        "process_count": len(process_receipts),
        "processes": process_receipts,
    }
    receipt["content_hash"] = _sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    _write_json_atomic(receipt_path, receipt)
    print(
        f"[dma] wrote {len(process_receipts)} process receipts -> {receipt_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
