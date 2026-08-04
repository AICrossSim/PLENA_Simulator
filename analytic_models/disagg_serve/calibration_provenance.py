"""Audit retained aggregate and structured HBM calibration evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

CALIBRATION_AUDIT_SCHEMA = "plena-memory-calibration-audit"
CALIBRATION_EVIDENCE_GRADE = "aggregate_csv_without_raw_run_receipts"
CALIBRATION_PUBLICATION_RECEIPT_COMPLETE = False
REQUEST_CALIBRATION_EVIDENCE_GRADE = (
    "ramulator2_structured_csv_with_process_receipts"
)
REQUEST_CALIBRATION_PUBLICATION_RECEIPT_COMPLETE = True

CSV_COMMIT = {
    "revision": "8ec79640a448ca5a72e7c1f64e9b6b4178026563",
    "authored_at": "2026-07-23T20:03:27+01:00",
    "committed_at": "2026-07-23T20:03:27+01:00",
    "subject": (
        "analytic: disagg_serve package "
        "(calibrated memory, KV hand-off, area, tests)"
    ),
}
HARNESS_COMMIT = {
    "revision": "d91f0ea4d38cc97354ca4f444877d24f2b730995",
    "authored_at": "2026-07-23T20:05:09+01:00",
    "committed_at": "2026-07-23T20:05:09+01:00",
    "subject": (
        "testbench: effective-bandwidth calibration harnesses "
        "+ compiler pointer bump"
    ),
}
HISTORICAL_SETTINGS_SHA256 = (
    "15ea43d5d1c1a51ea5036dd30dbad9d7f0cdb89fae413be278ee0ffa7b79365f"
)

_CSV_SPECS = {
    "kernel": {
        "path": "analytic_models/disagg_serve/calibration_bw.csv",
        "sha256": (
            "60b84763bfa3e382bbe37febbe5e4583471753ac8d3e790476af53e283902555"
        ),
        "headers": (
            "kv_size",
            "hbm_gen",
            "channels",
            "op",
            "count",
            "bytes",
            "dt_ps",
            "achieved_gbps",
            "sim_latency_ns",
        ),
        "axes": {
            "kv_size": (128, 256, 512, 1024, 2048),
            "hbm_gen": ("HBM2", "HBM3"),
            "channels": (8, 16, 32),
            "op": ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V", "_total"),
        },
        "key_fields": ("kv_size", "hbm_gen", "channels", "op"),
        "integer_fields": ("kv_size", "channels", "count", "bytes", "dt_ps"),
    },
    "dma": {
        "path": "analytic_models/disagg_serve/calibration_dma.csv",
        "sha256": (
            "a43a79856524f5921c5b518a9dcf92707f41a796c33268ac6910bd38857c763d"
        ),
        "headers": (
            "hbm_gen",
            "channels",
            "amount",
            "count",
            "bytes_per_transfer",
            "dt_ps_per_transfer",
            "achieved_gbps",
        ),
        "axes": {
            "hbm_gen": ("HBM2", "HBM3"),
            "channels": (8, 16, 32),
            "amount": (64, 128, 256, 512, 1024, 2048, 4096),
        },
        "key_fields": ("hbm_gen", "channels", "amount"),
        "integer_fields": ("channels", "amount", "count"),
    },
}

_HARNESS_SPECS = {
    "kernel": {
        "path": "transactional_emulator/testbench/calibration/kernel_sweep.py",
        "sha256": (
            "0d06f13fde49cb437143cfc2e6f5dc06bced5806b8d980f039045bda4c977891"
        ),
    },
    "dma": {
        "path": "transactional_emulator/testbench/calibration/dma_microbench.py",
        "sha256": (
            "cdf6a0b2c96d63a371f3f6c99d0239cc7d8450c4f25b5ada509dfe88f881338c"
        ),
    },
}

#: MLEN of the calibrated array; H_PREFETCH_M transfers whole MLEN tiles.
MATRIX_TILE_ROWS = 64
VECTOR_DMA_OPCODES = ("H_PREFETCH_V", "H_STORE_V")

_REQUEST_CALIBRATION_SPEC = {
    "path": "analytic_models/disagg_serve/calibration_dma_requests.csv",
    "sha256": (
        "0cc4ebbcb834ce795d11db7d371f61d1b7461f12f87dafc11ce9258fad71c103"
    ),
    "validation_path": (
        "analytic_models/disagg_serve/calibration_dma_requests.validation.json"
    ),
    "validation_sha256": (
        "71dd780130cbd1176095da26a2cf2161b41a18c1708b063e4aa3c5eb7041c1de"
    ),
    "harness_path": (
        "transactional_emulator/testbench/calibration/dma_microbench.py"
    ),
    "harness_sha256": (
        "298df1ea1d05ce0d0731568d9619d829b51254916e22b0477a23e528e7357297"
    ),
    "receipt_path": (
        "analytic_models/disagg_serve/calibration_dma_requests.receipt.json"
    ),
    "receipt_sha256": (
        "caca72d38cd2acfb9a4b7ad9f019bffdfda2f0d8225e0632b1a480c2fbbf6b16"
    ),
}

MISSING_AGGREGATE_RUN_RECEIPTS = (
    "exact invocation argv and working directory",
    "per-run op-stats JSONL",
    "captured stdout, stderr, and exit status",
    "emulator binary SHA-256",
    "compiler revision actually used by each run",
    "settings file SHA-256 actually used by each run",
    "Rust, Python, libtorch, and Ramulator toolchain receipt",
    "host and environment-variable receipt",
)

_REQUEST_INTEGER_FIELDS = {
    "channels",
    "address",
    "rows",
    "elements_per_row",
    "stride_bytes",
    "alignment_bytes",
    "element_bits",
    "scale_bits",
    "block_size",
    "scale_address",
    "scale_stride_bytes",
    "replica",
    "dt_ps",
    "physical_read_bytes",
    "physical_write_bytes",
    "instruction_pc",
}
_REQUEST_FLOAT_FIELDS = {"pin_rate_gbps", "measured_s"}
_REQUEST_BOOLEAN_FIELDS = {"partial_write_rmw"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_hash", None)
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coerce(value: str, field: str, integer_fields: Sequence[str]) -> Any:
    return int(value) if field in integer_fields else value


def _audit_csv(root: Path, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    payload = path.read_bytes()
    digest = _sha256(payload)
    if digest != spec["sha256"]:
        raise ValueError(f"{name} calibration CSV SHA-256 mismatch")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(spec["headers"]):
            raise ValueError(f"{name} calibration CSV headers differ")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{name} calibration CSV is empty")

    integer_fields = tuple(spec["integer_fields"])
    axes = {
        field: tuple(values)
        for field, values in dict(spec["axes"]).items()
    }
    observed_axes = {
        field: tuple(
            sorted(
                {
                    _coerce(row[field], field, integer_fields)
                    for row in rows
                }
            )
        )
        for field in axes
    }
    if observed_axes != axes:
        raise ValueError(f"{name} calibration grid axes differ")

    key_fields = tuple(spec["key_fields"])
    keys = [
        tuple(_coerce(row[field], field, integer_fields) for field in key_fields)
        for row in rows
    ]
    expected_keys = set(itertools.product(*(axes[field] for field in key_fields)))
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError(f"{name} calibration grid is incomplete or duplicated")
    for row in rows:
        for field in spec["headers"]:
            if field == "hbm_gen" or field == "op":
                continue
            numeric = float(row[field])
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(
                    f"{name} calibration has invalid {field} value"
                )

    return {
        "path": str(spec["path"]),
        "sha256": digest,
        "row_count": len(rows),
        "headers": list(spec["headers"]),
        "observed_grid": {
            field: list(values) for field, values in observed_axes.items()
        },
        "cartesian_grid_complete": True,
        "raw_run_receipts_retained": False,
    }


def _typed_request_row(
    row: Mapping[str, Any],
    headers: Sequence[str],
) -> dict[str, Any]:
    """Normalize CSV and JSON receipt rows to one exact representation."""

    if set(row) != set(headers):
        raise ValueError("structured request receipt row fields differ")
    normalized: dict[str, Any] = {}
    for field in headers:
        value = row[field]
        if field in _REQUEST_INTEGER_FIELDS:
            normalized[field] = int(value)
        elif field in _REQUEST_FLOAT_FIELDS:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("structured request receipt has non-finite data")
            normalized[field] = numeric
        elif field in _REQUEST_BOOLEAN_FIELDS:
            if isinstance(value, bool):
                normalized[field] = value
            elif value in ("True", "False"):
                normalized[field] = value == "True"
            else:
                raise ValueError("structured request receipt boolean differs")
        elif not isinstance(value, str):
            raise ValueError("structured request receipt text field differs")
        else:
            normalized[field] = value
    return normalized


def _request_row_fingerprint(
    row: Mapping[str, Any],
    headers: Sequence[str],
) -> str:
    return json.dumps(
        _typed_request_row(row, headers),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _audit_request_receipt(
    root: Path,
    *,
    spec: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, str]],
    headers: Sequence[str],
    expected_axes: Mapping[str, set[Any]],
) -> dict[str, Any]:
    """Verify every retained process receipt against its CSV observation."""

    receipt_path = root / str(spec["receipt_path"])
    receipt_payload = receipt_path.read_bytes()
    receipt_digest = _sha256(receipt_payload)
    if receipt_digest != spec["receipt_sha256"]:
        raise ValueError("structured request receipt SHA-256 mismatch")
    receipt = json.loads(receipt_payload)
    if not isinstance(receipt, dict) or (
        receipt.get("schema_version")
        != "plena-dma-request-calibration-receipt"
    ):
        raise ValueError("structured request receipt schema differs")
    content_hash = receipt.get("content_hash")
    if not _is_sha256(content_hash) or content_hash != _canonical_hash(receipt):
        raise ValueError("structured request receipt content hash differs")

    invocation = receipt.get("invocation")
    if (
        not isinstance(invocation, list)
        or not invocation
        or not all(isinstance(argument, str) and argument for argument in invocation)
        or not isinstance(receipt.get("working_directory"), str)
        or not receipt["working_directory"]
    ):
        raise ValueError("structured request invocation receipt is incomplete")

    expected_count = len(csv_rows)
    plan = receipt.get("sweep_plan")
    expected_configurations = math.prod(
        len(expected_axes[field])
        for field in (
            "hbm_generation",
            "channels",
            "rows",
            "precision",
        )
    )
    if not isinstance(plan, dict) or (
        plan.get("schema_version") != "plena-dma-request-sweep-plan"
        or int(plan.get("configuration_count", -1)) != expected_configurations
        or int(plan.get("observation_count", -1)) != expected_count
        or plan.get("request_isolation") is not True
        or set(plan.get("generations", ())) != expected_axes["hbm_generation"]
        or set(plan.get("channels", ())) != expected_axes["channels"]
        or set(plan.get("row_counts", ())) != expected_axes["rows"]
        or set(plan.get("precisions", ())) != expected_axes["precision"]
        or set(plan.get("opcodes", ())) != expected_axes["opcode"]
    ):
        raise ValueError("structured request sweep-plan receipt differs")

    calibration = receipt.get("calibration")
    if not isinstance(calibration, dict) or (
        calibration.get("sha256") != spec["sha256"]
        or int(calibration.get("observation_count", -1)) != expected_count
    ):
        raise ValueError("structured request output binding differs")
    harness = receipt.get("harness")
    if not isinstance(harness, dict) or harness.get("sha256") != spec["harness_sha256"]:
        raise ValueError("structured request harness receipt differs")

    emulator = receipt.get("emulator")
    if (
        not isinstance(emulator, dict)
        or not _is_sha256(emulator.get("sha256"))
        or not _is_sha256(emulator.get("ramulator_source_sha256"))
        or not isinstance(emulator.get("path"), str)
        or not emulator["path"]
    ):
        raise ValueError("structured request emulator receipt is incomplete")
    retained_binary = (
        root / "transactional_emulator" / "target" / "release"
        / "transactional_emulator"
    )
    current_binary_verified = retained_binary.is_file()
    if current_binary_verified and _sha256(retained_binary.read_bytes()) != emulator["sha256"]:
        raise ValueError("retained emulator binary differs from its run receipt")

    settings = receipt.get("settings")
    compiler = receipt.get("compiler")
    if (
        not isinstance(settings, dict)
        or not _is_sha256(settings.get("sha256"))
        or not isinstance(compiler, dict)
        or not isinstance(compiler.get("revision"), str)
        or len(compiler["revision"]) != 40
        or not _is_sha256(compiler.get("source_sha256"))
        or not isinstance(compiler.get("status"), str)
    ):
        raise ValueError("structured request source receipt is incomplete")
    toolchain = receipt.get("toolchain")
    if not isinstance(toolchain, dict) or any(
        not isinstance(toolchain.get(field), str) or not toolchain[field]
        for field in ("python", "python_executable", "rustc", "cargo", "linked_libraries")
    ):
        raise ValueError("structured request toolchain receipt is incomplete")
    if (
        "libramulator" not in toolchain["linked_libraries"]
        or "libtorch" not in toolchain["linked_libraries"]
    ):
        raise ValueError("structured request linked-library receipt differs")
    environment = receipt.get("environment")
    host = receipt.get("host")
    if (
        not isinstance(environment, dict)
        or "PYTHONPATH" not in environment
        or "LD_LIBRARY_PATH" not in environment
        or not isinstance(host, dict)
        or not all(field in host for field in ("platform", "machine", "processor"))
    ):
        raise ValueError("structured request host/environment receipt is incomplete")

    processes = receipt.get("processes")
    if (
        not isinstance(processes, list)
        or int(receipt.get("process_count", -1)) != expected_count
        or len(processes) != expected_count
    ):
        raise ValueError("structured request process receipt count differs")
    process_ids: set[str] = set()
    receipt_rows: list[Mapping[str, Any]] = []
    for process in processes:
        if not isinstance(process, dict):
            raise ValueError("structured request process receipt differs")
        process_id = process.get("process_id")
        if not isinstance(process_id, str) or not process_id or process_id in process_ids:
            raise ValueError("structured request process IDs are invalid or duplicated")
        process_ids.add(process_id)
        if process.get("return_code") != 0 or not all(
            isinstance(process.get(field), str) for field in ("stdout", "stderr")
        ):
            raise ValueError("structured request process did not complete cleanly")
        command = process.get("command")
        if (
            not isinstance(command, list)
            or not command
            or command[0] != emulator["path"]
            or "--blocking-prefetch" not in command
            or "--op-stats" not in command
        ):
            raise ValueError("structured request process command differs")
        requests = process.get("requests")
        observations = process.get("observations")
        if (
            not isinstance(requests, list)
            or len(requests) != 1
            or not isinstance(observations, list)
            or len(observations) != 1
            or not isinstance(requests[0], dict)
            or not isinstance(observations[0], dict)
        ):
            raise ValueError("structured request process isolation differs")
        request = requests[0]
        observation = observations[0]
        if any(observation.get(field) != value for field, value in request.items()):
            raise ValueError("structured request descriptor receipt differs")
        configuration = process.get("configuration")
        expected_configuration = {
            "hbm_generation": observation.get("hbm_generation"),
            "channels": observation.get("channels"),
            "rows": observation.get("rows"),
            "precision": observation.get("precision"),
        }
        if configuration != expected_configuration:
            raise ValueError("structured request process configuration differs")

        raw_stats = process.get("op_stats_jsonl")
        artifacts = process.get("artifacts")
        if not isinstance(raw_stats, str) or not isinstance(artifacts, dict):
            raise ValueError("structured request raw artifacts are incomplete")
        if _sha256(raw_stats.encode("utf-8")) != artifacts.get("op_stats_sha256"):
            raise ValueError("structured request op-stats hash differs")
        if any(
            not _is_sha256(artifacts.get(field))
            for field in (
                "assembly_sha256",
                "opcode_sha256",
                "settings_sha256",
                "op_stats_sha256",
            )
        ):
            raise ValueError("structured request artifact hash is incomplete")
        records = [json.loads(line) for line in raw_stats.splitlines() if line]
        targets = [
            record
            for record in records
            if record.get("pc") == observation.get("instruction_pc")
            and record.get("op") == observation.get("opcode")
        ]
        if len(targets) != 1:
            raise ValueError("structured request target op-stat is absent or duplicated")
        target = targets[0]
        expected_stats = {
            "dt_ps": observation.get("dt_ps"),
            "hbm_rd": observation.get("physical_read_bytes"),
            "hbm_wr": observation.get("physical_write_bytes"),
            "hbm_issue_rd": observation.get("physical_read_bytes"),
            "hbm_issue_wr": observation.get("physical_write_bytes"),
        }
        if any(target.get(field) != value for field, value in expected_stats.items()):
            raise ValueError("structured request target op-stat differs")
        receipt_rows.append(observation)

    csv_fingerprints = sorted(
        _request_row_fingerprint(row, headers) for row in csv_rows
    )
    receipt_fingerprints = sorted(
        _request_row_fingerprint(row, headers) for row in receipt_rows
    )
    if receipt_fingerprints != csv_fingerprints:
        raise ValueError("structured request process receipts do not reproduce the CSV")

    return {
        "path": str(spec["receipt_path"]),
        "sha256": receipt_digest,
        "content_hash": content_hash,
        "process_count": len(processes),
        "unique_process_ids": len(process_ids),
        "successful_process_count": len(processes),
        "raw_op_stats_retained": True,
        "observations_match_csv": True,
        "fresh_process_per_descriptor": True,
        "emulator_binary_sha256": emulator["sha256"],
        "current_emulator_binary_verified": current_binary_verified,
        "compiler_revision": compiler["revision"],
        "compiler_source_sha256": compiler["source_sha256"],
        "settings_sha256": settings["sha256"],
        "harness_sha256": harness["sha256"],
        "toolchain_receipt_retained": True,
        "host_environment_receipt_retained": True,
        "publication_receipt_complete": True,
    }


def _audit_request_calibration(root: Path) -> dict[str, Any]:
    """Validate the complete structured grid and its retained holdout report."""

    spec = _REQUEST_CALIBRATION_SPEC
    path = root / str(spec["path"])
    payload = path.read_bytes()
    digest = _sha256(payload)
    if digest != spec["sha256"]:
        raise ValueError("structured request calibration SHA-256 mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    expected_headers = (
        "schema_version",
        "opcode",
        "hbm_generation",
        "channels",
        "address",
        "rows",
        "elements_per_row",
        "stride_bytes",
        "alignment_bytes",
        "element_bits",
        "direction",
        "pin_rate_gbps",
        "tensor",
        "precision",
        "scale_bits",
        "block_size",
        "scale_address",
        "scale_stride_bytes",
        "partial_write_rmw",
        "replica",
        "dt_ps",
        "measured_s",
        "physical_read_bytes",
        "physical_write_bytes",
        "instruction_pc",
        "evidence_tier",
    )
    if tuple(reader.fieldnames or ()) != expected_headers:
        raise ValueError("structured request calibration headers differ")

    keys = set()
    observed = {
        "hbm_generation": set(),
        "channels": set(),
        "rows": set(),
        "precision": set(),
        "opcode": set(),
        "stride_multiplier": set(),
        "alignment_bytes": set(),
        "replica": set(),
    }
    for row in rows:
        if row["schema_version"] != "plena-dma-request-calibration":
            raise ValueError("structured request calibration schema differs")
        if row["evidence_tier"] != "ramulator2_simulated":
            raise ValueError("structured request evidence tier differs")
        element_bytes = int(row["elements_per_row"]) * int(row["element_bits"]) // 8
        stride_bytes = int(row["stride_bytes"])
        if element_bytes <= 0 or stride_bytes % element_bytes:
            raise ValueError("structured request stride is inconsistent")
        stride_multiplier = stride_bytes // element_bytes
        key = (
            row["hbm_generation"],
            int(row["channels"]),
            int(row["rows"]),
            row["precision"],
            row["opcode"],
            stride_multiplier,
            int(row["alignment_bytes"]),
            int(row["replica"]),
        )
        if key in keys:
            raise ValueError("structured request calibration contains duplicates")
        keys.add(key)
        for field, value in zip(observed, key):
            observed[field].add(value)
        dt_ps = int(row["dt_ps"])
        if dt_ps <= 0 or not math.isclose(
            float(row["measured_s"]),
            dt_ps / 1e12,
            rel_tol=0.0,
            abs_tol=1e-18,
        ):
            raise ValueError("structured request latency is inconsistent")
        read_bytes = int(row["physical_read_bytes"])
        write_bytes = int(row["physical_write_bytes"])
        if row["opcode"] == "H_STORE_V":
            if read_bytes <= 0 or write_bytes <= 0 or read_bytes != write_bytes:
                raise ValueError("structured store traffic is inconsistent")
        elif read_bytes <= 0 or write_bytes != 0:
            raise ValueError("structured prefetch traffic is inconsistent")

    expected_axes = {
        "hbm_generation": {"HBM2", "HBM3"},
        "channels": {8, 32, 128},
        "rows": {4, 16, 64, 256},
        "precision": {"mxfp4_e2m1", "mxfp8_e4m3", "mxint4", "mxint8"},
        "opcode": {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"},
        "stride_multiplier": {1, 4, 32},
        "alignment_bytes": {0, 16, 32, 48},
        "replica": {0, 1, 2},
    }
    # H_PREFETCH_M moves whole MLEN tiles and is measured only at row counts
    # that are MLEN multiples. The vector engines carry an independent transfer
    # amount, so H_PREFETCH_V and H_STORE_V cover every row count including the
    # small ones the decode program issues.
    matrix_rows = {
        value for value in expected_axes["rows"] if value % MATRIX_TILE_ROWS == 0
    }
    expected_count = (
        math.prod(
            len(values)
            for field, values in expected_axes.items()
            if field not in ("rows", "opcode")
        )
        * (len(matrix_rows) + len(expected_axes["rows"]) * len(VECTOR_DMA_OPCODES))
    )
    if observed != expected_axes or len(rows) != expected_count or len(keys) != expected_count:
        raise ValueError("structured request calibration grid is incomplete")
    for key in keys:
        if key[4] == "H_PREFETCH_M" and key[2] not in matrix_rows:
            raise ValueError("matrix prefetch measured outside whole MLEN tiles")

    validation_path = root / str(spec["validation_path"])
    validation_payload = validation_path.read_bytes()
    if _sha256(validation_payload) != spec["validation_sha256"]:
        raise ValueError("structured request validation SHA-256 mismatch")
    validation = json.loads(validation_payload)
    if (
        validation.get("schema_version")
        != "plena-request-latency-validation"
        or int(validation.get("training_count", -1))
        + int(validation.get("holdout_count", -1))
        != expected_count
        or float(validation.get("holdout_fraction", 0.0)) < 0.15
        or validation.get("split_unit") != "descriptor_fingerprint"
    ):
        raise ValueError("structured request validation contract differs")
    for field in (
        "median_absolute_error_percent",
        "p95_absolute_error_percent",
        "p99_absolute_error_percent",
        "worst_absolute_error_percent",
    ):
        value = float(validation[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError("structured request validation metric is invalid")

    harness_path = root / str(spec["harness_path"])
    harness_digest = _sha256(harness_path.read_bytes())
    if harness_digest != spec["harness_sha256"]:
        raise ValueError("structured request harness SHA-256 mismatch")
    receipt_audit = _audit_request_receipt(
        root,
        spec=spec,
        csv_rows=rows,
        headers=expected_headers,
        expected_axes=expected_axes,
    )
    return {
        "path": str(spec["path"]),
        "sha256": digest,
        "row_count": len(rows),
        "observed_grid": {
            field: sorted(values) for field, values in observed.items()
        },
        "cartesian_grid_complete": True,
        "measurement_isolation": "fresh_process_per_descriptor",
        "measurement_evidence_tier": "ramulator2_simulated",
        "evidence_grade": REQUEST_CALIBRATION_EVIDENCE_GRADE,
        "publication_receipt_complete": (
            REQUEST_CALIBRATION_PUBLICATION_RECEIPT_COMPLETE
        ),
        "validation": validation,
        "validation_path": str(spec["validation_path"]),
        "validation_sha256": str(spec["validation_sha256"]),
        "harness_path": str(spec["harness_path"]),
        "harness_sha256": harness_digest,
        "receipt": receipt_audit,
        "raw_run_receipts_retained": True,
    }


def _git(
    root: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode:
        stderr = completed.stderr if text else completed.stderr.decode()
        raise RuntimeError(f"git evidence check failed: {stderr.strip()}")
    return completed.stdout


def _verify_commit(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    fields = str(
        _git(
            root,
            "show",
            "-s",
            "--format=%H%x00%aI%x00%cI%x00%s",
            expected["revision"],
        )
    ).rstrip("\n").split("\0")
    observed = dict(
        zip(("revision", "authored_at", "committed_at", "subject"), fields)
    )
    if observed != dict(expected):
        raise ValueError(f"commit evidence differs for {expected['revision']}")
    return observed


def _verify_historical_evidence(root: Path) -> dict[str, Any]:
    csv_commit = _verify_commit(root, CSV_COMMIT)
    harness_commit = _verify_commit(root, HARNESS_COMMIT)
    for spec in _CSV_SPECS.values():
        payload = _git(
            root,
            "show",
            f"{CSV_COMMIT['revision']}:{spec['path']}",
            text=False,
        )
        if _sha256(bytes(payload)) != spec["sha256"]:
            raise ValueError("historical calibration CSV hash differs")
    for spec in _HARNESS_SPECS.values():
        payload = _git(
            root,
            "show",
            f"{HARNESS_COMMIT['revision']}:{spec['path']}",
            text=False,
        )
        if _sha256(bytes(payload)) != spec["sha256"]:
            raise ValueError("historical calibration harness hash differs")

    ancestor = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            CSV_COMMIT["revision"],
            HARNESS_COMMIT["revision"],
        ),
        check=False,
        capture_output=True,
    )
    if ancestor.returncode:
        raise ValueError("calibration CSV commit is not an ancestor of harness")
    revision_distance = int(
        str(
            _git(
                root,
                "rev-list",
                "--count",
                f"{CSV_COMMIT['revision']}..{HARNESS_COMMIT['revision']}",
            )
        ).strip()
    )
    settings_hashes = []
    for revision in (CSV_COMMIT["revision"], HARNESS_COMMIT["revision"]):
        payload = _git(root, "show", f"{revision}:plena_settings.toml", text=False)
        settings_hashes.append(_sha256(bytes(payload)))
    if tuple(settings_hashes) != (
        HISTORICAL_SETTINGS_SHA256,
        HISTORICAL_SETTINGS_SHA256,
    ):
        raise ValueError("candidate historical settings hashes differ")
    compiler_revisions = {}
    for label, revision in (
        ("aggregate_csv_commit", CSV_COMMIT["revision"]),
        ("harness_commit", HARNESS_COMMIT["revision"]),
    ):
        fields = str(_git(root, "ls-tree", revision, "compiler")).split()
        if len(fields) < 3 or fields[1] != "commit":
            raise ValueError("compiler submodule evidence is unavailable")
        compiler_revisions[label] = fields[2]

    return {
        "git_objects_verified": True,
        "aggregate_csv_commit": csv_commit,
        "reproduction_harness_commit": harness_commit,
        "csv_commit_is_harness_ancestor": True,
        "revision_distance": revision_distance,
        "authored_time_delta_seconds": 102,
        "candidate_settings_sha256": HISTORICAL_SETTINGS_SHA256,
        "candidate_settings_were_bound_to_runs": False,
        "candidate_compiler_revisions": compiler_revisions,
        "candidate_compiler_revision_changed": (
            len(set(compiler_revisions.values())) != 1
        ),
        "compiler_revision_was_bound_to_runs": False,
        "exact_measurement_input_binding": "unknown",
    }


def build_calibration_audit(
    repository: str | os.PathLike[str],
    *,
    verify_git: bool = True,
) -> dict[str, Any]:
    """Build a deterministic audit without upgrading aggregate CSVs to receipts."""

    root = Path(repository).resolve()
    aggregate = {
        name: _audit_csv(root, name, spec)
        for name, spec in _CSV_SPECS.items()
    }
    structured_request = _audit_request_calibration(root)
    harnesses = {}
    for name, spec in _HARNESS_SPECS.items():
        path = root / str(spec["path"])
        digest = _sha256(path.read_bytes())
        harnesses[name] = {
            "path": str(path),
            "historical_repository_path": str(spec["path"]),
            "sha256": digest,
            "introduced_by": HARNESS_COMMIT["revision"],
            "matches_historical_source": digest == spec["sha256"],
            "historically_bound_to_aggregate_csv": False,
            "status": "current_reproduction_harness_only",
        }

    historical = (
        _verify_historical_evidence(root)
        if verify_git
        else {
            "git_objects_verified": False,
            "exact_measurement_input_binding": "unknown",
        }
    )
    audit = {
        "schema_version": CALIBRATION_AUDIT_SCHEMA,
        "evidence_grade": CALIBRATION_EVIDENCE_GRADE,
        "publication_receipt_complete": (
            CALIBRATION_PUBLICATION_RECEIPT_COMPLETE
        ),
        "aggregate_measurements": aggregate,
        "structured_request_measurement": structured_request,
        "evidence_grades": {
            "aggregate": CALIBRATION_EVIDENCE_GRADE,
            "structured_request": REQUEST_CALIBRATION_EVIDENCE_GRADE,
        },
        "historical_evidence": historical,
        "current_reproduction_harnesses": harnesses,
        "publication_receipts": {
            "aggregate": False,
            "structured_request": True,
        },
        "missing_run_receipt_scope": "aggregate_measurements_only",
        "missing_run_receipts": list(MISSING_AGGREGATE_RUN_RECEIPTS),
        "permitted_use": [
            "aggregate model fitting",
            "reported holdout validation",
            "request-level model fitting",
            "sensitivity analysis",
        ],
        "unsupported_claims": [
            "exact historical replay",
            "complete publication-grade calibration provenance",
        ],
        "required_remediation": (
            "rerun the historical aggregate harnesses while retaining "
            "immutable per-run receipts; the structured request sweep is complete"
        ),
    }
    audit["content_hash"] = _canonical_hash(audit)
    return audit


def write_calibration_audit(
    path: str | os.PathLike[str],
    audit: Mapping[str, Any],
) -> Path:
    """Atomically create an immutable audit or verify an identical one."""

    value = dict(audit)
    if value.get("content_hash") != _canonical_hash(value):
        raise ValueError("calibration audit content hash differs")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to replace a different audit: {destination}"
            )
        return destination
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(Path(__file__).parents[2]))
    parser.add_argument("--output")
    parser.add_argument("--without-git-verification", action="store_true")
    args = parser.parse_args(argv)
    audit = build_calibration_audit(
        args.repository,
        verify_git=not args.without_git_verification,
    )
    if args.output:
        output = write_calibration_audit(args.output, audit)
        print(output)
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
