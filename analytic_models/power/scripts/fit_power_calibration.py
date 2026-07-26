#!/usr/bin/env python3
# ruff: noqa: E402
"""Fit the RTL-activity calibrated on-chip action-energy candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.power.power_model import DEFAULT_LOGIC_COEFFICIENTS


TRAIN_PATTERNS = {"random", "low-toggle"}
VALIDATION_PATTERNS = {"representative-qwen", "mixed-kernel-holdout"}
MATRIX_ALIAS = "matrix.matrix_vector_bit_product"
MATRIX_PRIMARY = "matrix.active_mac_bit_product"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.inf
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _nnls(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls

        return nnls(matrix, target)[0]
    except ImportError:
        coefficients = np.zeros(matrix.shape[1], dtype=float)
        norm = float(np.linalg.norm(matrix, ord=2))
        step = 1.0 / max(norm * norm, 1e-12)
        for _ in range(100_000):
            updated = np.maximum(0.0, coefficients - step * matrix.T @ (matrix @ coefficients - target))
            if np.linalg.norm(updated - coefficients) <= 1e-11 * max(1.0, np.linalg.norm(coefficients)):
                return updated
            coefficients = updated
        return coefficients


def _read_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            latest[(raw.get("point_key", ""), raw.get("scenario", ""))] = raw

    rows: list[dict[str, Any]] = []
    for raw in latest.values():
        if raw.get("status") != "complete" or not raw.get("features_json"):
            continue
        payload = json.loads(raw["features_json"])
        dynamic = {str(key): float(value) for key, value in payload.get("dynamic_features", {}).items()}
        clock = {str(key): float(value) for key, value in payload.get("clock_features", {}).items()}
        # Early harnesses grouped these two physically different blocks under
        # one clock feature.  They differ by roughly 40x in mapped area, so a
        # shared coefficient makes the otherwise exact Control replay appear
        # thousands of percent too high.  The component label disambiguates
        # the existing rows without changing any measured energy.
        if str(raw.get("component")) in {"control", "hbm"}:
            shared = clock.pop("control_hbm", 0.0)
            clock[
                "control_frontend" if raw.get("component") == "control" else "hbm_controller"
            ] = shared
        # The RTL uses the same bit-product leaf structure for GEMM and
        # GEMV. Combining them avoids an unidentifiable duplicate column;
        # the fitted coefficient is copied back to both public keys.
        dynamic[MATRIX_PRIMARY] = dynamic.get(MATRIX_PRIMARY, 0.0) + dynamic.pop(MATRIX_ALIAS, 0.0)
        rows.append(
            {
                **raw,
                "dynamic_features": dynamic,
                "clock_features": clock,
                "total_dynamic_target_pj": float(raw["window_dynamic_energy_pj"]),
                "logic_area_um2": float(raw.get("logic_area_um2") or 0.0),
                "leakage_power_mw": float(raw.get("leakage_power_mw") or 0.0),
                # The leaf bundle exists to identify output/reduction action
                # energy.  It is not a MatrixMachine shape holdout and its
                # idle clock network is not representative of that machine.
                "holdout": (
                    bool(int(raw.get("holdout") or 0))
                    and raw.get("point_id") != "power_matrix_leaf_bundle"
                ),
                "repeat_count": int(raw["repeat_count"]),
                "measurement_start_ns": float(payload["measurement_start_ns"]),
                "measurement_end_ns": float(payload["measurement_end_ns"]),
            }
        )
    return rows


def _pair_incremental(rows: list[dict[str, Any]]) -> None:
    idle: dict[tuple[str, int, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["point_key"], row["repeat_count"],
            row["measurement_start_ns"], row["measurement_end_ns"],
        )
        if row["pattern"] == "idle":
            idle[key] = row
    missing: list[str] = []
    for row in rows:
        if row["pattern"] == "idle":
            row["incremental_target_pj"] = 0.0
            continue
        key = (
            row["point_key"], row["repeat_count"],
            row["measurement_start_ns"], row["measurement_end_ns"],
        )
        baseline = idle.get(key)
        if baseline is None:
            missing.append(f"{row['point_id']}/{row['scenario']}")
            continue
        row["matched_idle_energy_pj"] = baseline["total_dynamic_target_pj"]
        row["incremental_target_pj"] = row["total_dynamic_target_pj"] - baseline["total_dynamic_target_pj"]
    if missing:
        raise ValueError(f"active rows lack exact matched idle windows: {missing[:10]}")
    negative = [row for row in rows if row["pattern"] != "idle" and row["incremental_target_pj"] <= 0]
    if negative:
        labels = [f"{row['point_id']}/{row['scenario']}" for row in negative]
        raise ValueError(f"active energy is not above matched idle: {labels[:10]}")


def _matrix(rows: list[dict[str, Any]], field: str, names: list[str]) -> np.ndarray:
    return np.asarray([[row[field].get(name, 0.0) for name in names] for row in rows], dtype=float)


def _fit(rows: list[dict[str, Any]], *, field: str, target: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    names = sorted({name for row in rows for name, value in row[field].items() if value != 0})
    if not names:
        raise ValueError(f"no active features in {field}")
    matrix = _matrix(rows, field, names)
    missing_columns = [name for index, name in enumerate(names) if np.all(matrix[:, index] == 0)]
    if missing_columns:
        raise ValueError(f"zero feature columns: {missing_columns}")
    target_values = np.asarray([row[target] for row in rows], dtype=float)
    scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-18)
    normalized = matrix / scale
    coefficients = _nnls(normalized, target_values) / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = math.inf if singular[-1] <= 1e-15 else float(singular[0] / singular[-1])
    correlations: list[dict[str, Any]] = []
    if len(rows) > 1:
        corr = np.corrcoef(matrix, rowvar=False) if len(names) > 1 else np.asarray([[1.0]])
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                value = float(corr[left, right])
                if math.isfinite(value) and abs(value) >= 0.95:
                    correlations.append({"left": names[left], "right": names[right], "correlation": value})
    return names, coefficients, {
        "condition_number": condition,
        "rank": int(np.linalg.matrix_rank(normalized)),
        "feature_count": len(names),
        "zero_coefficients": [name for name, value in zip(names, coefficients) if value <= 1e-18],
        "highly_correlated_features": correlations,
    }


def _predict(row: dict[str, Any], names: list[str], coefficients: np.ndarray, field: str) -> float:
    return float(sum(row[field].get(name, 0.0) * coefficients[index] for index, name in enumerate(names)))


def _diagnose(
    rows: list[dict[str, Any]], dynamic_names: list[str], dynamic_coefficients: np.ndarray,
    clock_names: list[str], clock_coefficients: np.ndarray,
) -> tuple[list[float], list[dict[str, Any]]]:
    errors: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        dynamic_prediction = _predict(row, dynamic_names, dynamic_coefficients, "dynamic_features")
        clock_prediction = _predict(row, clock_names, clock_coefficients, "clock_features")
        if row["pattern"] == "idle":
            target = row["total_dynamic_target_pj"]
            prediction = clock_prediction
        else:
            target = row["incremental_target_pj"]
            prediction = dynamic_prediction
        error = abs(prediction - target) / max(abs(target), 1e-12) * 100.0
        errors.append(error)
        diagnostics.append(
            {
                "point_id": row["point_id"], "component": row["component"],
                "scenario": row["scenario"], "target_pj": target,
                "prediction_pj": prediction, "absolute_percentage_error": error,
            }
        )
    return errors, diagnostics


def _total_errors(
    rows: list[dict[str, Any]], dynamic_names: list[str], dynamic_coefficients: np.ndarray,
    clock_names: list[str], clock_coefficients: np.ndarray,
) -> tuple[list[float], list[dict[str, Any]]]:
    errors: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        prediction = _predict(row, dynamic_names, dynamic_coefficients, "dynamic_features") + _predict(
            row, clock_names, clock_coefficients, "clock_features"
        )
        target = row["total_dynamic_target_pj"]
        error = abs(prediction - target) / max(target, 1e-12) * 100.0
        errors.append(error)
        diagnostics.append(
            {
                "point_id": row["point_id"], "component": row["component"],
                "scenario": row["scenario"], "target_pj": target,
                "prediction_pj": prediction, "absolute_percentage_error": error,
            }
        )
    return errors, diagnostics


def _random_r2(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_point: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row["pattern"] == "random":
            by_point.setdefault(row["point_id"], []).append((row["repeat_count"], row["incremental_target_pj"]))
    result: dict[str, float] = {}
    for point, samples in by_point.items():
        if len(samples) < 3:
            result[point] = -math.inf
            continue
        x = np.asarray([sample[0] for sample in samples], dtype=float)
        y = np.asarray([sample[1] for sample in samples], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        residual = float(np.sum((y - (slope * x + intercept)) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        result[point] = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total if total else -math.inf
    return result


def _component_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in diagnostics:
        grouped.setdefault(row["component"], []).append(row["absolute_percentage_error"])
    return {
        component: {
            "rows": len(errors), "median_ape": median(errors),
            "p95_ape": _percentile(errors, 95), "max_ape": max(errors),
        }
        for component, errors in sorted(grouped.items())
    }


def _validation_markdown(
    validation: dict[str, Any], coefficients: dict[str, Any]
) -> str:
    """Render a compact, reproducible candidate-validation report."""

    status = "PASS" if validation["accepted"] else "FAIL"
    lines = [
        "# RTL-Activity Power Candidate Validation",
        "",
        f"**Promotion status:** {status}",
        "",
        "This report evaluates RTL VCD activity replayed on mapped Design "
        "Compiler netlists. Gate-level simulation, CTS, routed parasitics, "
        "external HBM/PHY, package power, and SRAM leakage are outside scope.",
        "",
        "## Error Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Train median APE | {validation['train_median_ape']:.3f}% |",
        f"| Train P95 APE | {validation['train_p95_ape']:.3f}% |",
        f"| Mapped holdout median APE | {validation['holdout_median_ape']:.3f}% |",
        f"| Mapped holdout P95 APE | {validation['holdout_p95_ape']:.3f}% |",
        f"| Clock holdout median APE | {validation['clock_holdout_median_ape']:.3f}% |",
        f"| Clock holdout P95 APE | {validation['clock_holdout_p95_ape']:.3f}% |",
        f"| Qwen/mixed median total error | {validation['qwen_mixed_median_total_error']:.3f}% |",
        f"| Qwen/mixed maximum total error | {validation['qwen_mixed_max_total_error']:.3f}% |",
        "",
        "## Acceptance Gates",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    for name, passed in validation["acceptance"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Component Validation",
            "",
            "| Component | Rows | Median APE | P95 APE | Maximum APE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for component, metrics in validation["component_holdout"].items():
        lines.append(
            f"| {component} | {metrics['rows']} | "
            f"{metrics['median_ape']:.3f}% | {metrics['p95_ape']:.3f}% | "
            f"{metrics['max_ape']:.3f}% |"
        )
    lines.extend(["", "## Identifiability", ""])
    for label in ("dynamic", "clock"):
        diagnostic = validation[f"{label}_fit_diagnostics"]
        condition = diagnostic["condition_number"]
        condition_text = "infinite" if not math.isfinite(condition) else f"{condition:.6g}"
        lines.extend(
            [
                f"- **{label.title()} matrix:** rank "
                f"{diagnostic['rank']}/{diagnostic['feature_count']}, condition "
                f"number {condition_text}.",
                f"- **{label.title()} zero coefficients:** "
                f"{', '.join(diagnostic['zero_coefficients']) or 'none'}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Fitted Coefficients",
            "",
            "### Incremental dynamic energy",
            "",
            "| Feature | pJ per feature unit |",
            "|---|---:|",
        ]
    )
    for name, value in sorted(coefficients["dynamic_pj"].items()):
        lines.append(f"| `{name}` | {float(value):.9g} |")
    lines.extend(
        [
            "",
            "### Idle clock energy",
            "",
            "| Feature | pJ per cycle-feature |",
            "|---|---:|",
        ]
    )
    for name, value in sorted(coefficients["clock_pj_per_cycle"].items()):
        lines.append(f"| `{name}` | {float(value):.9g} |")
    lines.extend(
        [
            "",
            f"Logic leakage reference: "
            f"`{float(coefficients['logic_leakage_mw_per_um2']):.9g} mW/um^2`.",
            "",
            "A PASS promotes only an `rtl_activity_calibrated_candidate`; it "
            "does not imply gate-level or signoff validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(args.points)
    _pair_incremental(rows)
    dynamic_train = [row for row in rows if not row["holdout"] and row["pattern"] in TRAIN_PATTERNS]
    clock_train = [row for row in rows if not row["holdout"] and row["pattern"] == "idle"]
    holdout = [row for row in rows if row["holdout"] and row["pattern"] != "idle"]
    holdout_clock = [row for row in rows if row["holdout"] and row["pattern"] == "idle"]
    system_validation = [row for row in rows if row["pattern"] in VALIDATION_PATTERNS]
    if not dynamic_train or not clock_train or not holdout or not holdout_clock or not system_validation:
        raise ValueError("fit requires complete train, mapped holdout, Qwen/mixed, and matched idle rows")

    dynamic_names, dynamic_coefficients, dynamic_fit = _fit(
        dynamic_train, field="dynamic_features", target="incremental_target_pj"
    )
    clock_names, clock_coefficients, clock_fit = _fit(
        clock_train, field="clock_features", target="total_dynamic_target_pj"
    )
    required_dynamic = set(DEFAULT_LOGIC_COEFFICIENTS["dynamic_pj"]) - {MATRIX_ALIAS}
    missing_dynamic = required_dynamic - set(dynamic_names)
    missing_clock = set(DEFAULT_LOGIC_COEFFICIENTS["clock_pj_per_cycle"]) - set(clock_names)
    if missing_dynamic or missing_clock:
        raise ValueError(f"missing action family coverage: dynamic={sorted(missing_dynamic)}, clock={sorted(missing_clock)}")

    train_errors, train_diag = _diagnose(
        dynamic_train, dynamic_names, dynamic_coefficients, clock_names, clock_coefficients
    )
    holdout_errors, holdout_diag = _diagnose(
        holdout, dynamic_names, dynamic_coefficients, clock_names, clock_coefficients
    )
    clock_errors, clock_diag = _diagnose(
        holdout_clock, dynamic_names, dynamic_coefficients, clock_names, clock_coefficients
    )
    system_errors, system_diag = _total_errors(
        system_validation, dynamic_names, dynamic_coefficients, clock_names, clock_coefficients
    )
    random_r2 = _random_r2(rows)

    leakage_ratios = [
        row["leakage_power_mw"] / row["logic_area_um2"]
        for row in rows if not row["holdout"] and row["logic_area_um2"] > 0
    ]
    acceptance = {
        "holdout_median_error_le_15pct": median(holdout_errors) <= 15.0,
        "holdout_p95_error_le_30pct": _percentile(holdout_errors, 95) <= 30.0,
        "clock_holdout_median_error_le_15pct": median(clock_errors) <= 15.0,
        "clock_holdout_p95_error_le_30pct": _percentile(clock_errors, 95) <= 30.0,
        "qwen_mixed_max_error_le_20pct": max(system_errors) <= 20.0,
        "all_random_r2_ge_0p95": bool(random_r2) and min(random_r2.values()) >= 0.95,
        "no_missing_action_family": not missing_dynamic and not missing_clock,
        "logic_leakage_reference_available": bool(leakage_ratios),
    }
    accepted = all(acceptance.values())

    payload = json.loads(json.dumps(DEFAULT_LOGIC_COEFFICIENTS))
    fitted_dynamic = {name: float(value) for name, value in zip(dynamic_names, dynamic_coefficients)}
    fitted_dynamic[MATRIX_ALIAS] = fitted_dynamic[MATRIX_PRIMARY]
    payload["dynamic_pj"].update(fitted_dynamic)
    payload["clock_pj_per_cycle"].update(
        {name: float(value) for name, value in zip(clock_names, clock_coefficients)}
    )
    if leakage_ratios:
        payload["logic_leakage_mw_per_um2"] = median(leakage_ratios)
    payload.update(
        {
            "calibration_status": "rtl_activity_calibrated_candidate" if accepted else "rtl_activity_candidate_failed_validation",
            "gate_level_validation": "not_run_by_scope",
            "training_rows": len(dynamic_train), "holdout_rows": len(holdout),
            "source_points": str(args.points.resolve()),
            "power_scope": "mapped_logic_rtl_activity_no_cts_plus_macro_sram_dynamic",
        }
    )
    payload["provenance"] = {
        "dynamic": "nonnegative active-minus-exact-matched-idle slopes from RTL VCD replayed on mapped DC",
        "clock": "nonnegative idle-window energy from RTL VCD replayed on mapped DC",
        "matrix_gemm_gemv": "shared bit-product coefficient",
        "leakage": "median mapped-logic leakage divided by mapped cell area",
    }
    validation = {
        "model": "onchip_action_energy_v1", "accepted": accepted,
        "calibration_status": payload["calibration_status"],
        "gate_level_validation": "not_run_by_scope", "acceptance": acceptance,
        "dynamic_fit_diagnostics": dynamic_fit, "clock_fit_diagnostics": clock_fit,
        "train_median_ape": median(train_errors), "train_p95_ape": _percentile(train_errors, 95),
        "holdout_median_ape": median(holdout_errors), "holdout_p95_ape": _percentile(holdout_errors, 95),
        "clock_holdout_median_ape": median(clock_errors), "clock_holdout_p95_ape": _percentile(clock_errors, 95),
        "qwen_mixed_median_total_error": median(system_errors),
        "qwen_mixed_max_total_error": max(system_errors),
        "random_slope_r2": random_r2,
        "component_holdout": _component_summary(holdout_diag + clock_diag + system_diag),
        "train_points": train_diag, "holdout_points": holdout_diag,
        "clock_holdout_points": clock_diag, "qwen_mixed_points": system_diag,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_validation_markdown(validation, payload))
    if args.promote:
        if not accepted:
            raise RuntimeError("RTL-activity candidate failed validation; default artifact was not changed")
        calibration_dir = Path(__file__).resolve().parents[1] / "calibration"
        promoted = {
            calibration_dir / "logic_energy_v1.json": payload,
            calibration_dir / "power_validation_v1.json": validation,
        }
        for destination, artifact in promoted.items():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, destination)
    print(json.dumps({"accepted": accepted, "training_rows": len(dynamic_train), "holdout_rows": len(holdout)}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
