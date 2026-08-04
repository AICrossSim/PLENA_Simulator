"""Audit aggregate area calibration data and unavailable raw DC receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import estimate_area
from .matrix import estimate_matrix_machine_area

AREA_AUDIT_SCHEMA = "plena-area-calibration-audit"
AREA_EVIDENCE_GRADE = "aggregate_area_tables_without_raw_dc_reports"
AREA_PUBLICATION_RECEIPT_COMPLETE = False

_CALIBRATION_FILES = {
    "full_chip_anchors.csv": {
        "sha256": (
            "8a388d40f9567900652ca9991e50385ec2cf4bd76cfa9fef9ce48986ffd1d884"
        ),
        "rows": 17,
    },
    "matrix_machine_mxint.csv": {
        "sha256": (
            "ace5b1e8fe4b778eb624f094ef028367a33ceb148a61fae8cfae378947fb8f40"
        ),
        "rows": 48,
    },
    "matrix_machine_mxfp.csv": {
        "sha256": (
            "2c951a7a200fb6d95d957bd2d7ce11acfd808658972e9507050e853f6a01f1fd"
        ),
        "rows": 23,
    },
    "asap7_sram_macro_table.csv": {
        "sha256": (
            "45c7e031b2c32efd5083379fdcdcd32361af39dbc930d474756ec56ea111bf07"
        ),
        "rows": 36,
    },
    "matrix_structural_coefficients.json": {
        "sha256": (
            "bfe51635c498088d40d905f3cec62fbd5ed916863ed4911d8a6530c61eea2b41"
        ),
    },
    "ASAP7_SRAM_LICENSE": {
        "sha256": (
            "f729eadf714ca54a7b665f9094f9687ba7b7f2c30e6c294431d1f0e2b1144245"
        ),
    },
}
_RAW_PATH_FIELDS = ("report_dir", "summary_log", "_source_csv")
_GATE_NAMES = ("holdout", "anchor", "monotonic", "precision", "full_chip")


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


def _config(
    mlen: int,
    blen: int,
    activation: str = "MXINT4",
    key_value: str = "MXINT4",
    weight: str = "MXINT4",
) -> dict[str, Any]:
    return {
        "MLEN": mlen,
        "BLEN": blen,
        "VLEN": mlen,
        "ACT_WIDTH": activation,
        "KV_WIDTH": key_value,
        "WEIGHT_WIDTH": weight,
        "FP_SETTING": "FP_E5M6",
    }


def _aggregate_gates(coefficients: Mapping[str, Any]) -> dict[str, Any]:
    reports = coefficients["report"]
    holdout_reports = {
        "matrix_mxint": reports["mxint"],
        "matrix_mxfp": reports["mxfp"],
        **reports["full_chip_blocks"],
    }
    holdout = all(
        float(report["holdout_fraction"]) >= 0.25
        and float(report["p95_abs_error_pct"]) <= 10.0
        for report in holdout_reports.values()
    )

    anchor = (
        estimate_matrix_machine_area(
            _config(1024, 4),
            corner="reference",
        )["area"]
        / 1e6
    )
    anchor_error = abs(anchor - 0.237) / 0.237 * 100.0
    anchor_passed = anchor_error <= 1.0

    shapes = ((16, 4), (32, 4), (64, 8), (256, 8), (1024, 4), (2048, 32))
    areas = tuple(
        estimate_matrix_machine_area(_config(mlen, blen))["area"]
        for mlen, blen in shapes
    )
    monotonic = all(right > left for left, right in zip(areas, areas[1:]))

    precision_areas = tuple(
        estimate_matrix_machine_area(
            _config(1024, 4, token, token, "MXINT4")
        )["area"]
        for token in ("MXINT2", "MXINT4", "MXINT8")
    )
    precision = (
        precision_areas[0] < precision_areas[1] < precision_areas[2]
    )

    whole_chip = estimate_area(_config(1024, 4))
    required_blocks = {
        "MatrixMachine",
        "VectorMachine",
        "ScalarMachine",
        "HBMInterface",
        "TopOverhead",
        "MatrixSRAM",
        "VectorSRAM",
        "ScalarIntSRAM",
        "ScalarFPSRAM",
    }
    full_chip = (
        float(whole_chip["area"]) > 0
        and float(whole_chip["matrix_machine_area"]) > 0
        and required_blocks == set(whole_chip["breakdown"])
        and math.isclose(
            float(whole_chip["area"]),
            sum(float(value) for value in whole_chip["breakdown"].values()),
            rel_tol=1e-12,
        )
    )
    gates = {
        "holdout": {
            "passed": holdout,
            "minimum_fraction": min(
                float(report["holdout_fraction"])
                for report in holdout_reports.values()
            ),
            "maximum_p95_abs_error_pct": max(
                float(report["p95_abs_error_pct"])
                for report in holdout_reports.values()
            ),
            "required_minimum_fraction": 0.25,
            "p95_limit_pct": 10.0,
            "per_block": {
                name: {
                    "holdout_rows": int(report["holdout_rows"]),
                    "n_rows": int(report["n_rows"]),
                    "holdout_fraction": float(report["holdout_fraction"]),
                    "median_abs_error_pct": float(
                        report["median_abs_error_pct"]
                    ),
                    "p95_abs_error_pct": float(report["p95_abs_error_pct"]),
                }
                for name, report in holdout_reports.items()
            },
        },
        "anchor": {
            "passed": anchor_passed,
            "observed_mm2": anchor,
            "expected_mm2": 0.237,
            "error_pct": anchor_error,
            "limit_pct": 1.0,
        },
        "monotonic": {"passed": monotonic},
        "precision": {"passed": precision},
        "full_chip": {"passed": full_chip},
    }
    if tuple(gates) != _GATE_NAMES or not all(
        gate["passed"] for gate in gates.values()
    ):
        raise ValueError("aggregate area validation gates do not all pass")
    return {
        "passed": 5,
        "total": 5,
        "basis": "retained aggregate tables and fitted coefficient artifact",
        "gates": gates,
    }


def _raw_run_root(value: str) -> str:
    parts = Path(value).parts
    if len(parts) >= 4 and parts[2] == "runs":
        return str(Path(*parts[:4]))
    return str(Path(value).parent)


def build_area_calibration_audit(
    repository: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build a deterministic grade without inferring absent synthesis receipts."""

    root = Path(repository).resolve()
    calibration = root / "analytic_models" / "area" / "calibration"
    files: dict[str, Any] = {}
    raw_paths: set[str] = set()
    raw_roots: set[str] = set()
    for name, spec in _CALIBRATION_FILES.items():
        path = calibration / name
        payload = path.read_bytes()
        digest = _sha256(payload)
        if digest != spec["sha256"]:
            raise ValueError(f"area calibration SHA-256 mismatch for {name}")
        record: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "sha256": digest,
        }
        if name.endswith(".csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != spec["rows"]:
                raise ValueError(f"area calibration row count differs for {name}")
            record["row_count"] = len(rows)
            if name != "asap7_sram_macro_table.csv":
                if any(row.get("status") != "complete" for row in rows):
                    raise ValueError(f"area calibration has incomplete rows in {name}")
                for row in rows:
                    area = float(row["area_um2"])
                    if not math.isfinite(area) or area <= 0:
                        raise ValueError(f"area calibration has invalid area in {name}")
                    for field in _RAW_PATH_FIELDS:
                        value = row.get(field, "")
                        if value:
                            raw_paths.add(value)
                            raw_roots.add(_raw_run_root(value))
        files[name] = record

    coefficients = json.loads(
        (calibration / "matrix_structural_coefficients.json").read_text(
            encoding="utf-8"
        )
    )
    resolved = tuple(root / path for path in sorted(raw_paths))
    existing = tuple(path for path in resolved if path.exists())
    if existing:
        raise ValueError(
            "audit constants require absent raw DC paths; update the evidence grade"
        )

    audit = {
        "schema_version": AREA_AUDIT_SCHEMA,
        "evidence_grade": AREA_EVIDENCE_GRADE,
        "publication_receipt_complete": AREA_PUBLICATION_RECEIPT_COMPLETE,
        "aggregate_calibration_files": files,
        "aggregate_validation": _aggregate_gates(coefficients),
        "raw_dc_receipts": {
            "reference_count": len(raw_paths),
            "existing_reference_count": 0,
            "missing_reference_count": len(raw_paths),
            "referenced_run_roots": sorted(raw_roots),
            "raw_reports_retained_in_workspace": False,
        },
        "permitted_use": [
            "aggregate structural area fitting",
            "aggregate holdout and shape validation",
            "area sensitivity analysis",
        ],
        "unsupported_claims": [
            "raw DC report provenance is complete",
            "historical synthesis commands are exactly replayable",
        ],
        "required_remediation": (
            "recover referenced DC report trees or rerun synthesis with "
            "content-addressed commands, toolchain, logs, and reports"
        ),
    }
    audit["content_hash"] = _canonical_hash(audit)
    return audit


def write_area_calibration_audit(
    path: str | os.PathLike[str],
    audit: Mapping[str, Any],
) -> Path:
    """Atomically create an immutable audit or verify an identical one."""

    value = dict(audit)
    if value.get("content_hash") != _canonical_hash(value):
        raise ValueError("area calibration audit content hash differs")
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
    args = parser.parse_args(argv)
    audit = build_area_calibration_audit(args.repository)
    if args.output:
        print(write_area_calibration_audit(args.output, audit))
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
