#!/usr/bin/env python3
"""Validate composed area estimates against held-out full-chip DC anchors.

The comparison separates logic from SRAM scope. DC full-chip reports contain
logic and SRAM wrapper/conversion logic but no SRAM bitcells; the proxy adds
ASAP7 SRAM macros. Logic error is compared directly, while composite error adds
the same macro estimate to both sides. Committed hierarchy columns make module
validation reproducible without retaining large DC report directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"
DEFAULT_ANCHORS = CALIBRATION_DIR / "full_chip_anchors.csv"
DEFAULT_SPLIT = CALIBRATION_DIR / "full_chip_validation_split.json"

INTEGER_FIELDS = {
    "MLEN",
    "VLEN",
    "BLEN",
    "HLEN",
    "MATRIX_SRAM_DEPTH",
    "VECTOR_SRAM_DEPTH",
    "INT_SRAM_DEPTH",
    "FP_SRAM_DEPTH",
    "INT_DATA_WIDTH",
    "MX_SCALE_WIDTH",
    "HBM_M_Prefetch_Amount",
    "HBM_V_Prefetch_Amount",
    "HBM_V_Writeback_Amount",
}


def _load_estimator():
    from analytic_models.area_new import estimate_area

    return estimate_area


def _config_from_row(row: dict[str, str]) -> dict[str, Any]:
    config: dict[str, Any] = {key: value for key, value in row.items() if value != ""}
    for key in INTEGER_FIELDS:
        if key in config:
            config[key] = int(config[key])
    return config


def _metrics(rows: list[dict[str, Any]], predicted_key: str, actual_key: str) -> dict[str, float]:
    errors = [
        (float(row[predicted_key]) - float(row[actual_key])) / float(row[actual_key])
        for row in rows
    ]
    return {
        "mape_pct": 100.0 * sum(abs(error) for error in errors) / len(errors),
        "max_abs_error_pct": 100.0 * max(abs(error) for error in errors),
        "mean_signed_error_pct": 100.0 * sum(errors) / len(errors),
    }


def parse_full_chip_hierarchy(report: Path) -> dict[str, float]:
    """Parse absolute hierarchy cell area values from a DC ``area.rpt`` file.

    DC wraps long hierarchy names onto a separate line, so the parser supports
    both inline and pending-name row formats.
    """
    if not report.exists():
        raise FileNotFoundError(report)
    rows: dict[str, float] = {}
    pending_name: str | None = None
    numeric_line = re.compile(r"^\s+([0-9.]+)\s+([0-9.]+)\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+")
    inline_line = re.compile(
        r"^(.{1,34}?)(\s+[0-9.]+\s+[0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+)"
    )
    for line in report.read_text(errors="ignore").splitlines():
        if (
            not line.strip()
            or line.startswith("-")
            or line.startswith("Hierarchical")
            or line.startswith("Global")
            or line.startswith("Total")
            or line.startswith("Design")
        ):
            continue
        inline = inline_line.match(line)
        if inline and inline.group(1).strip():
            match = numeric_line.match(inline.group(2))
            if match:
                rows[inline.group(1).strip()] = float(match.group(1))
                pending_name = None
                continue
        match = numeric_line.match(line)
        if match and pending_name:
            rows[pending_name] = float(match.group(1))
            pending_name = None
            continue
        if not re.search(r"[0-9]+\.[0-9]+", line):
            pending_name = line.strip()
    return rows


def _resolve_report(source: dict[str, str]) -> Path:
    report = Path(source["report_dir"]) / "area.rpt"
    if report.exists():
        return report
    report = ROOT / report
    if report.exists():
        return report
    raise FileNotFoundError(f"area.rpt not found for {source['point_id']}: {report}")


def _hierarchy_from_source(source: dict[str, str]) -> dict[str, float]:
    embedded = {
        "plena": "hier_plena_area",
        "matrix_machine_init": "hier_matrix_machine_area",
        "vector_machine_init": "hier_vector_machine_area",
        "scalar_machine_init": "hier_scalar_machine_area",
        "scalar_machine_init/fp_scalar_sram": "hier_scalar_fp_sram_wrapper_area",
        "scalar_machine_init/int_scalar_sram": "hier_scalar_int_sram_wrapper_area",
        "hbm_interface_init": "hier_hbm_system_area",
        "matrix_sram": "hier_matrix_sram_wrapper_area",
        "vector_sram": "hier_vector_sram_wrapper_area",
    }
    if all(source.get(field, "") != "" for field in embedded.values()):
        return {name: float(source[field]) for name, field in embedded.items()}
    return parse_full_chip_hierarchy(_resolve_report(source))


def _logic_module_predictions(estimate: dict[str, Any]) -> dict[str, float]:
    breakdown = estimate["area_breakdown"]
    return {
        "MatrixMachine": float(breakdown["MatrixMachine"]),
        "VectorMachine": float(breakdown["VectorMachine"]),
        "ScalarMachineLogic": sum(
            float(breakdown.get(key, 0.0))
            for key in ("ScalarIntLogic", "ScalarFPLogic", "ScalarControl", "ScalarVectorBufferLogic")
        ),
        "HBMSystem": sum(float(value) for key, value in breakdown.items() if key.startswith("HBM")),
    }


def _module_validation_rows(
    source: dict[str, str],
    raw_estimate: dict[str, Any],
    corrected_estimate: dict[str, Any],
) -> list[dict[str, Any]]:
    hierarchy = _hierarchy_from_source(source)
    predicted = _logic_module_predictions(raw_estimate)
    scalar_sram_wrapper = sum(
        hierarchy.get(key, 0.0)
        for key in (
            "scalar_machine_init/fp_scalar_sram",
            "scalar_machine_init/int_scalar_sram",
        )
    )
    actual = {
        "MatrixMachine": hierarchy["matrix_machine_init"],
        "VectorMachine": hierarchy["vector_machine_init"],
        "ScalarMachineLogic": hierarchy["scalar_machine_init"] - scalar_sram_wrapper,
        "HBMSystem": hierarchy["hbm_interface_init"],
    }
    dc_total = hierarchy["plena"]
    actual_aggregate_residual = dc_total - sum(actual.values())
    predicted_aggregate_residual = float(corrected_estimate["area_breakdown"]["FullChipTopResidual"])
    rows: list[dict[str, Any]] = []

    def append(module: str, predicted_area: float, actual_area: float, scope: str, comparable: bool, note: str) -> None:
        rows.append(
            {
                "point_id": source["point_id"],
                "module": module,
                "predicted_area_um2": predicted_area,
                "dc_hierarchy_area_um2": actual_area,
                "error_pct": 100.0 * (predicted_area - actual_area) / actual_area if comparable else "",
                "comparison_scope": scope,
                "comparable": comparable,
                "note": note,
            }
        )

    for module in ("MatrixMachine", "VectorMachine", "ScalarMachineLogic", "HBMSystem"):
        append(module, predicted[module], actual[module], "logic", True, "Direct module-level logic comparison.")
    append(
        "FullChipTopResidual",
        predicted_aggregate_residual,
        actual_aggregate_residual,
        "aggregate residual",
        True,
        "DC remainder after Matrix/Vector/Scalar/HBM logic; includes SRAM wrappers and top-level integration.",
    )

    sram_predictions = raw_estimate["sram"]["area_sram_breakdown"]
    append(
        "MatrixSRAM",
        float(sram_predictions["MatrixSRAM"]),
        hierarchy.get("matrix_sram", 0.0),
        "scope mismatch",
        False,
        "Proxy is an ASAP7 bitcell macro; DC value is synthesized wrapper/conversion logic without bitcells.",
    )
    append(
        "VectorSRAM",
        float(sram_predictions["VectorSRAM"]),
        hierarchy.get("vector_sram", 0.0),
        "scope mismatch",
        False,
        "Proxy is an ASAP7 bitcell macro; DC value is synthesized wrapper/conversion logic without bitcells.",
    )
    append(
        "ScalarSRAM",
        float(sram_predictions["ScalarIntSRAM"]) + float(sram_predictions["ScalarFPSRAM"]),
        scalar_sram_wrapper,
        "scope mismatch",
        False,
        "Proxy is an ASAP7 bitcell macro; DC value is wrapper logic without bitcells.",
    )
    return rows


def _module_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | str]]:
    output: dict[str, dict[str, float | int | str]] = {}
    modules = sorted({str(row["module"]) for row in rows})
    for module in modules:
        selected = [row for row in rows if row["module"] == module]
        comparable = [row for row in selected if row["comparable"]]
        if not comparable:
            output[module] = {
                "points": len(selected),
                "status": "not_comparable",
                "reason": str(selected[0]["note"]),
            }
            continue
        errors = [float(row["error_pct"]) for row in comparable]
        status = "aggregate_correction" if module == "FullChipTopResidual" else "comparable"
        output[module] = {
            "points": len(comparable),
            "status": status,
            "mape_pct": sum(abs(error) for error in errors) / len(errors),
            "max_abs_error_pct": max(abs(error) for error in errors),
            "mean_signed_error_pct": sum(errors) / len(errors),
        }
    return output


def build_validation(
    anchors_path: Path,
    split_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate training/validation totals and held-out module errors.

    Validation IDs are loaded from a committed split manifest and are never
    used to fit the full-chip top residual coefficient.
    """
    split = json.loads(split_path.read_text())
    validation_ids = set(split["validation_point_ids"])
    estimator = _load_estimator()
    rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    coefficient: float | None = None
    with anchors_path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            config = _config_from_row(source)
            raw_estimate = estimator(config, apply_top_residual=False)
            corrected_estimate = estimator(config)
            dc_logic = float(source["area_um2"])
            macro_sram = float(corrected_estimate["sram_macro_area"])
            predicted_logic = float(raw_estimate["logic_area_before_top_residual"])
            corrected_logic = predicted_logic + float(corrected_estimate["full_chip_top_residual"]["area"])
            current_coefficient = float(corrected_estimate["full_chip_top_residual"]["coefficients"]["logic_fraction"])
            if coefficient is None:
                coefficient = current_coefficient
            elif coefficient != current_coefficient:
                raise ValueError("top residual coefficient changed across validation points")
            rows.append(
                {
                    "point_id": source["point_id"],
                    "split": "validation" if source["point_id"] in validation_ids else "training",
                    "MLEN": int(source["MLEN"]),
                    "VLEN": int(source["VLEN"]),
                    "BLEN": int(source["BLEN"]),
                    "HLEN": int(source["HLEN"]),
                    "ACT_WIDTH": source["ACT_WIDTH"],
                    "KV_WIDTH": source["KV_WIDTH"],
                    "WEIGHT_WIDTH": source["WEIGHT_WIDTH"],
                    "FP_SETTING": source["FP_SETTING"],
                    "dc_logic_wrapper_um2": dc_logic,
                    "predicted_logic_um2": predicted_logic,
                    "sram_macro_um2": macro_sram,
                    "raw_residual_um2": dc_logic - predicted_logic,
                    "top_residual_um2": corrected_logic - predicted_logic,
                    "corrected_logic_um2": corrected_logic,
                }
            )
            if source["point_id"] in validation_ids:
                module_rows.extend(_module_validation_rows(source, raw_estimate, corrected_estimate))

    found_validation = {row["point_id"] for row in rows if row["split"] == "validation"}
    missing = validation_ids - found_validation
    if missing:
        raise ValueError(f"validation point IDs are absent from {anchors_path}: {sorted(missing)}")
    training = [row for row in rows if row["split"] == "training"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not training or not validation:
        raise ValueError("both training and validation full-chip points are required")

    for row in rows:
        predicted_logic = float(row["predicted_logic_um2"])
        dc_logic = float(row["dc_logic_wrapper_um2"])
        macro_sram = float(row["sram_macro_um2"])
        corrected_logic = float(row["corrected_logic_um2"])
        row.update(
            {
                "raw_logic_error_pct": 100.0 * (predicted_logic - dc_logic) / dc_logic,
                "corrected_logic_error_pct": 100.0 * (corrected_logic - dc_logic) / dc_logic,
                "dc_composite_with_macro_um2": dc_logic + macro_sram,
                "predicted_composite_with_macro_um2": predicted_logic + macro_sram,
                "corrected_composite_with_macro_um2": corrected_logic + macro_sram,
            }
        )

    summary = {
        "metadata": {
            "area_proxy_scope": "module logic models plus ASAP7 SRAM macros",
            "dc_area_scope": "logic plus SRAM wrappers; no SRAM bitcell macros",
            "comparison_method": (
                "Compare proxy logic excluding SRAM macros with full-chip DC. For composite total, "
                "add the same SRAM macro estimate to both DC and proxy logic."
            ),
            "top_residual_model": "top_residual_um2 = coefficient * predicted_module_logic_um2",
            "validation_points_are_excluded_from_fit": True,
        },
        "training_points": len(training),
        "validation_points": len(validation),
        "top_residual_coefficient": coefficient,
        "training": {
            "raw_logic": _metrics(training, "predicted_logic_um2", "dc_logic_wrapper_um2"),
            "corrected_logic": _metrics(training, "corrected_logic_um2", "dc_logic_wrapper_um2"),
        },
        "validation": {
            "raw_logic": _metrics(validation, "predicted_logic_um2", "dc_logic_wrapper_um2"),
            "corrected_logic": _metrics(validation, "corrected_logic_um2", "dc_logic_wrapper_um2"),
            "raw_composite": _metrics(
                validation,
                "predicted_composite_with_macro_um2",
                "dc_composite_with_macro_um2",
            ),
            "corrected_composite": _metrics(
                validation,
                "corrected_composite_with_macro_um2",
                "dc_composite_with_macro_um2",
            ),
        },
        "module_validation": _module_metrics(module_rows),
    }
    return rows, module_rows, summary


def _write_report(
    rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "full_chip_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "full_chip_module_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module_rows[0]))
        writer.writeheader()
        writer.writerows(module_rows)
    (output_dir / "full_chip_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    validation = [row for row in rows if row["split"] == "validation"]
    metrics = summary["validation"]
    lines = [
        "# Full-Chip Area Proxy Validation",
        "",
        "The five latest full-chip points are a strict holdout set and are not used to fit the top residual.",
        "Full-chip DC area contains synthesized logic and SRAM wrappers, but no SRAM bitcell macros.",
        "",
        "## Summary",
        "",
        f"- Training anchors: {summary['training_points']}",
        f"- Validation anchors: {summary['validation_points']}",
        f"- Top residual: `area = module_logic * (1 + {summary['top_residual_coefficient']:.6f})`",
        f"- Raw logic MAPE: {metrics['raw_logic']['mape_pct']:.3f}%",
        f"- Corrected logic MAPE: {metrics['corrected_logic']['mape_pct']:.3f}%",
        f"- Corrected logic maximum error: {metrics['corrected_logic']['max_abs_error_pct']:.3f}%",
        f"- Raw composite MAPE: {metrics['raw_composite']['mape_pct']:.3f}%",
        f"- Corrected composite MAPE: {metrics['corrected_composite']['mape_pct']:.3f}%",
        "",
        "## Module Summary",
        "",
        "| Module | Status | MAPE | Maximum error | Mean signed error |",
        "|---|---|---:|---:|---:|",
    ]
    for module, values in summary["module_validation"].items():
        if values["status"] in {"comparable", "aggregate_correction"}:
            lines.append(
                f"| {module} | {values['status']} | {values['mape_pct']:.3f}% | "
                f"{values['max_abs_error_pct']:.3f}% | {values['mean_signed_error_pct']:.3f}% |"
            )
        else:
            lines.append(f"| {module} | scope mismatch | N/A | N/A | N/A |")
    lines.extend([
        "",
        "## Validation Points",
        "",
        "| Point | DC logic (um2) | Raw prediction (um2) | Corrected prediction (um2) | Corrected error |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in validation:
        lines.append(
            f"| {row['point_id']} | {row['dc_logic_wrapper_um2']:.3f} | "
            f"{row['predicted_logic_um2']:.3f} | {row['corrected_logic_um2']:.3f} | "
            f"{row['corrected_logic_error_pct']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Per-Module Validation",
            "",
            "| Point | Module | Prediction (um2) | DC hierarchy (um2) | Error |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in module_rows:
        if not row["comparable"]:
            continue
        lines.append(
            f"| {row['point_id']} | {row['module']} | {row['predicted_area_um2']:.3f} | "
            f"{row['dc_hierarchy_area_um2']:.3f} | {row['error_pct']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## SRAM Scope-Mismatch Diagnostics",
            "",
            "These values are reported for completeness, but their differences are not model errors: the proxy column",
            "contains physical ASAP7 SRAM bitcell macros while the DC column contains wrapper/conversion logic only.",
            "",
            "| Point | SRAM | Macro proxy (um2) | DC wrapper only (um2) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in module_rows:
        if row["comparable"]:
            continue
        lines.append(
            f"| {row['point_id']} | {row['module']} | {row['predicted_area_um2']:.3f} | "
            f"{row['dc_hierarchy_area_um2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The corrected composite area is `corrected logic + ASAP7 SRAM macro area`. It remains a calibrated proxy,",
            "not a placed-and-routed full-chip signoff result, and excludes HBM stacks, PHY, package, and physical overhead.",
            "",
        ]
    )
    (output_dir / "full_chip_validation_report.md").write_text("\n".join(lines))


def main() -> int:
    """Generate CSV, JSON, and Markdown validation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "Workspace"
        / "area_new_validation"
        / f"full_chip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    rows, module_rows, summary = build_validation(args.anchors, args.split)
    _write_report(rows, module_rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote validation artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
