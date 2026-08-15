#!/usr/bin/env python3
"""Fit and validate the rtl-v6 VectorMachine physical area increment.

The fit is deliberately fail closed.  Wrapper and leaf synthesis establish a
nonnegative structural model; independently synthesized rtl-v5/rtl-v6
VectorMachine pairs validate that model in production context.  An artifact is
promoted only when every planned train/holdout set is present and all published
error limits pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytic_models.area_new.scripts.run_matrix_machine_calibration import (
    fit_nonnegative,
)

CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"

FEATURES = (
    "vlen_fp",
    "extra_rows",
    "extra_rows_vlen",
    "extra_rows_vlen_exp",
    "extra_rows_vlen_mant",
    "row_fp",
    "write_fp",
    "bank_count",
    "scoreboard_depth",
    "const",
)
COEFFICIENT_KEYS = tuple(f"{name}_um2" for name in FEATURES)
PRODUCTION_FEATURES = (
    "vlen_fp",
    "extra_rows",
    "extra_rows_vlen",
    "const",
)

REQUIRED_COUNTS = {
    ("state-simd-leaf", "train"): 8,
    ("state-simd-leaf", "holdout"): 2,
    ("packed-pv-leaf", "train"): 8,
    ("packed-pv-leaf", "holdout"): 2,
    ("banked-integration-wrapper", "train"): 16,
    ("banked-integration-wrapper", "width-holdout"): 8,
    ("banked-integration-wrapper", "precision-holdout"): 4,
    ("paired-production-vector-current", "train"): 6,
    ("paired-production-vector-current", "holdout"): 2,
    ("paired-production-vector-baseline", "train"): 6,
    ("paired-production-vector-baseline", "holdout"): 2,
}


def _read_complete(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "complete":
                continue
            key = str(row.get("point_key") or row.get("job_key") or row.get("point_id"))
            latest[key] = row
    return list(latest.values())


def _integer(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    return int(float(value)) if value not in (None, "") else default


def structural_features(row: dict[str, Any]) -> dict[str, float]:
    """Return the nonnegative rtl-v6 physical feature vector for one point."""

    lanes = _integer(row, "ROW_LANES", _integer(row, "SOFTMAX_ROW_LANES", 1))
    vlen = _integer(row, "VLEN", 1)
    fp_width = _integer(row, "fp_width", 1)
    exp_width = _integer(row, "V_FP_EXP_WIDTH", 5)
    mant_width = _integer(row, "V_FP_MANT_WIDTH", max(1, fp_width - exp_width - 1))
    write_lanes = _integer(row, "WRITE_LANES", min(16, vlen))
    scoreboard_depth = _integer(row, "scoreboard_depth", 32)
    extra_rows = max(0, lanes - 1)
    return {
        "vlen_fp": float(vlen * fp_width),
        "extra_rows": float(extra_rows),
        "extra_rows_vlen": float(extra_rows * vlen),
        "extra_rows_vlen_exp": float(extra_rows * vlen * exp_width),
        "extra_rows_vlen_mant": float(extra_rows * vlen * mant_width),
        "row_fp": float(lanes * fp_width),
        "write_fp": float(write_lanes * fp_width),
        # E5M6 and E6M5 mapped to nearly identical packed-PV lanes, whereas
        # the E8M5 implementation paid extra exponent-path logic.  Preserve
        # that measured threshold without introducing a high-order fit.
        "write_exp_excess": float(write_lanes * max(0, exp_width - 6)),
        "bank_count": float(lanes),
        "scoreboard_depth": float(scoreboard_depth),
        "const": 1.0,
    }


def _predict(features: dict[str, float], coefficients: dict[str, float]) -> float:
    return sum(
        features[name] * coefficients[f"{name}_um2"] for name in FEATURES
    )


def _errors(actual_predicted: Iterable[tuple[float, float]]) -> dict[str, float]:
    values = sorted(
        abs(predicted - actual) / max(abs(actual), 1e-9) * 100.0
        for actual, predicted in actual_predicted
    )
    if not values:
        return {"count": 0, "median_pct": math.inf, "p95_pct": math.inf, "max_pct": math.inf}
    midpoint = len(values) // 2
    median = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    p95_index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return {
        "count": len(values),
        "median_pct": median,
        "p95_pct": values[p95_index],
        "max_pct": values[-1],
    }


def _role(rows: Iterable[dict[str, Any]], role: str, split: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("calibration_role") == role
        and row.get("calibration_split") == split
    ]


def _wrapper_fit_split(row: dict[str, Any]) -> str | None:
    """Assign a formal fit split without changing collection point keys."""

    if row.get("calibration_role") != "banked-integration-wrapper":
        return None
    vlen = _integer(row, "VLEN")
    setting = str(row.get("FP_SETTING"))
    lanes = _integer(row, "ROW_LANES")
    if vlen in {16, 32} and setting in {"FP_E5M6", "FP_E8M5"}:
        return "train"
    if vlen == 64 and setting in {"FP_E5M6", "FP_E8M5"}:
        return "width-holdout"
    if vlen in {32, 64} and setting == "FP_E6M5" and lanes in {2, 8}:
        return "precision-holdout"
    if vlen > 64:
        return "supplementary-extrapolation"
    return None


def _leaf_fit(
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for row in train:
        features = structural_features(row)
        normalized.append(
            {
                **{name: features[name] for name in feature_names},
                "area_um2": float(row["area_um2"]),
            }
        )
    values, train_mape = fit_nonnegative(normalized, list(feature_names))
    coefficients = dict(zip(feature_names, values, strict=True))
    holdout_error = _errors(
        (
            float(row["area_um2"]),
            sum(
                structural_features(row)[name] * coefficients[name]
                for name in feature_names
            ),
        )
        for row in holdout
    )
    return coefficients, {"mape_pct": train_mape}, holdout_error


def _pair_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        _integer(row, "VLEN"),
        str(row.get("FP_SETTING")),
        _integer(row, "ROW_LANES", _integer(row, "SOFTMAX_ROW_LANES", 1)),
    )


def _paired_deltas(
    current: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_key = {_pair_key(row): row for row in baseline}
    deltas: list[dict[str, Any]] = []
    for row in current:
        key = _pair_key(row)
        before = baseline_by_key.get(key)
        if before is None:
            continue
        delta = float(row["area_um2"]) - float(before["area_um2"])
        deltas.append(
            {
                **dict(row),
                "baseline_area_um2": float(before["area_um2"]),
                "current_area_um2": float(row["area_um2"]),
                "delta_area_um2": delta,
                "baseline_point_id": before.get("point_id"),
            }
        )
    return deltas


def _monotonic(coefficients: dict[str, float]) -> bool:
    for exp_width, mant_width in ((5, 6), (6, 5), (8, 5)):
        fp_width = 1 + exp_width + mant_width
        # Increasing row parallelism at a fixed vector width cannot reduce
        # physical logic area.
        for vlen in (16, 32, 64, 128):
            previous = -math.inf
            for lanes in (1, 2, 4, 8):
                features = structural_features(
                    {
                        "ROW_LANES": lanes,
                        "VLEN": vlen,
                        "fp_width": fp_width,
                        "V_FP_EXP_WIDTH": exp_width,
                        "V_FP_MANT_WIDTH": mant_width,
                        "WRITE_LANES": min(16, vlen),
                        "scoreboard_depth": 32,
                    }
                )
                value = _predict(features, coefficients)
                if lanes > 1 and value + 1e-9 < previous:
                    return False
                previous = value
        # Increasing VLEN at a fixed row tier cannot reduce area either.
        for lanes in (1, 2, 4, 8):
            previous = -math.inf
            for vlen in (16, 32, 64, 128):
                features = structural_features(
                    {
                        "ROW_LANES": lanes,
                        "VLEN": vlen,
                        "fp_width": fp_width,
                        "V_FP_EXP_WIDTH": exp_width,
                        "V_FP_MANT_WIDTH": mant_width,
                        "WRITE_LANES": min(16, vlen),
                        "scoreboard_depth": 32,
                    }
                )
                value = _predict(features, coefficients)
                if vlen > 16 and value + 1e-9 < previous:
                    return False
                previous = value
    return True


def fit_artifact(
    current_csv: Path, baseline_csv: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current_rows = _read_complete(current_csv)
    baseline_rows = _read_complete(baseline_csv)
    rows = current_rows + baseline_rows

    wrapper_by_split = {
        split: [row for row in current_rows if _wrapper_fit_split(row) == split]
        for split in (
            "train",
            "width-holdout",
            "precision-holdout",
            "supplementary-extrapolation",
        )
    }
    counts: dict[str, int] = {}
    for role, split in REQUIRED_COUNTS:
        if role == "banked-integration-wrapper":
            counts[f"{role}/{split}"] = len(wrapper_by_split[split])
        else:
            counts[f"{role}/{split}"] = len(_role(rows, role, split))
    failures = [
        f"missing {role}/{split}: expected {expected}, observed {counts[f'{role}/{split}']}"
        for (role, split), expected in REQUIRED_COUNTS.items()
        if counts[f"{role}/{split}"] != expected
    ]

    state_coeffs, state_train, state_holdout = _leaf_fit(
        _role(rows, "state-simd-leaf", "train"),
        _role(rows, "state-simd-leaf", "holdout"),
        feature_names=("row_fp", "const"),
    )
    pv_coeffs, pv_train, pv_holdout = _leaf_fit(
        _role(rows, "packed-pv-leaf", "train"),
        _role(rows, "packed-pv-leaf", "holdout"),
        feature_names=("write_fp", "write_exp_excess", "const"),
    )

    wrapper_train = wrapper_by_split["train"]
    wrapper_width_holdout = wrapper_by_split["width-holdout"]
    wrapper_precision_holdout = wrapper_by_split["precision-holdout"]
    wrapper_supplementary = wrapper_by_split["supplementary-extrapolation"]
    fit_rows = []
    for row in wrapper_train:
        fit_rows.append(
            {
                **structural_features(row),
                "area_um2": float(row["area_um2"]),
            }
        )
    wrapper_coefficient_values, wrapper_train_mape = fit_nonnegative(
        fit_rows, list(FEATURES)
    )
    wrapper_coefficients = dict(
        zip(COEFFICIENT_KEYS, wrapper_coefficient_values, strict=True)
    )
    wrapper_width_holdout_error = _errors(
        (
            float(row["area_um2"]),
            _predict(structural_features(row), wrapper_coefficients),
        )
        for row in wrapper_width_holdout
    )
    wrapper_precision_holdout_error = _errors(
        (
            float(row["area_um2"]),
            _predict(structural_features(row), wrapper_coefficients),
        )
        for row in wrapper_precision_holdout
    )
    wrapper_supplementary_error = _errors(
        (
            float(row["area_um2"]),
            _predict(structural_features(row), wrapper_coefficients),
        )
        for row in wrapper_supplementary
    )

    paired_train = _paired_deltas(
        _role(rows, "paired-production-vector-current", "train"),
        _role(rows, "paired-production-vector-baseline", "train"),
    )
    paired_holdout = _paired_deltas(
        _role(rows, "paired-production-vector-current", "holdout"),
        _role(rows, "paired-production-vector-baseline", "holdout"),
    )
    production_fit_rows = [
        {
            **{
                name: structural_features(row)[name]
                for name in PRODUCTION_FEATURES
            },
            "area_um2": row["delta_area_um2"],
        }
        for row in paired_train
    ]
    production_values, production_train_mape = fit_nonnegative(
        production_fit_rows, list(PRODUCTION_FEATURES)
    )
    coefficients = {key: 0.0 for key in COEFFICIENT_KEYS}
    for name, value in zip(PRODUCTION_FEATURES, production_values, strict=True):
        coefficients[f"{name}_um2"] = value
    paired_train_error = _errors(
        (row["delta_area_um2"], _predict(structural_features(row), coefficients))
        for row in paired_train
    )
    paired_holdout_error = _errors(
        (row["delta_area_um2"], _predict(structural_features(row), coefficients))
        for row in paired_holdout
    )

    checks = {
        "state_holdout": state_holdout,
        "packed_pv_holdout": pv_holdout,
        "wrapper_width_holdout": wrapper_width_holdout_error,
        "wrapper_precision_holdout": wrapper_precision_holdout_error,
        "wrapper_vlen128_supplementary": wrapper_supplementary_error,
        "paired_machine_train": paired_train_error,
        "paired_machine_holdout": paired_holdout_error,
        "all_coefficients_nonnegative": all(value >= 0.0 for value in coefficients.values()),
        "area_monotonic": _monotonic(coefficients),
    }
    for label, metrics in (
        ("state", state_holdout),
        ("packed_pv", pv_holdout),
        ("wrapper_width", wrapper_width_holdout_error),
        ("wrapper_precision", wrapper_precision_holdout_error),
    ):
        if metrics["median_pct"] > 5.0 or metrics["p95_pct"] > 10.0:
            failures.append(f"{label} holdout exceeds 5% median/10% P95: {metrics}")
    if paired_holdout_error["max_pct"] > 10.0:
        failures.append(
            "paired-machine holdout exceeds 10%: "
            f"{paired_holdout_error}"
        )
    if paired_train_error["median_pct"] > 5.0 or paired_train_error["p95_pct"] > 10.0:
        failures.append(
            "paired-machine train fit exceeds 5% median/10% P95: "
            f"{paired_train_error}"
        )
    if not checks["all_coefficients_nonnegative"]:
        failures.append("one or more fitted coefficients are negative")
    if not checks["area_monotonic"]:
        failures.append("fitted area is not monotonic in R and VLEN")
    if any(row["delta_area_um2"] <= 0.0 for row in paired_train + paired_holdout):
        failures.append("one or more production rtl-v6 paired deltas are nonpositive")

    promoted = not failures
    status = (
        "fitted_from_paired_rtl_v6_dc"
        if promoted
        else "rtl_v6_dc_candidate_not_promoted"
    )
    artifact = {
        "schema_version": "vector_rtl_v6_delta_v3",
        "metadata": {
            "status": status,
            "model": "paired_production_nonnegative_structural_fit_v3",
            "technology": "ASAP7 TT 0.7 V 25 C",
            "clock_period_ps": 1000,
            "compile_mode": "normal",
            "current_csv": str(current_csv),
            "baseline_csv": str(baseline_csv),
            "required_and_observed_counts": counts,
            "wrapper_train_mape_pct": wrapper_train_mape,
            "production_train_mape_pct": production_train_mape,
            "production_features": list(PRODUCTION_FEATURES),
            "logic_fit_vlen": [16, 32],
            "logic_holdout_vlen": [64],
            "logic_calibration_max_vlen": 64,
            "large_width_semantics": (
                "VLEN>64 is a structural banked-logic extrapolation; SRAM "
                "macro tiling remains exact"
            ),
            "checks": checks,
            "failures": failures,
            "sram_semantics": (
                "Vector and state SRAM bit capacity is excluded from this logic "
                "overlay and charged by the ASAP7 macro tiler"
            ),
        },
        "coefficients": coefficients,
        "wrapper_coefficients": wrapper_coefficients,
        "component_leaf_fits": {
            "state_simd": {
                "coefficients": state_coeffs,
                "train": state_train,
                "holdout": state_holdout,
            },
            "packed_pv": {
                "coefficients": pv_coeffs,
                "train": pv_train,
                "holdout": pv_holdout,
            },
        },
        "paired_measurements": paired_train + paired_holdout,
    }
    diagnostics = []
    for row in (
        wrapper_train
        + wrapper_width_holdout
        + wrapper_precision_holdout
        + wrapper_supplementary
    ):
        predicted = _predict(structural_features(row), wrapper_coefficients)
        diagnostics.append(
            {
                "point_id": row.get("point_id"),
                "role": row.get("calibration_role"),
                "collection_split": row.get("calibration_split"),
                "fit_split": _wrapper_fit_split(row),
                "actual_area_um2": float(row["area_um2"]),
                "predicted_area_um2": predicted,
                "error_pct": (
                    predicted - float(row["area_um2"])
                ) / max(float(row["area_um2"]), 1e-9) * 100.0,
            }
        )
    for row in paired_train + paired_holdout:
        predicted = _predict(structural_features(row), coefficients)
        actual = float(row["delta_area_um2"])
        diagnostics.append(
            {
                "point_id": row.get("point_id"),
                "role": row.get("calibration_role"),
                "collection_split": row.get("calibration_split"),
                "fit_split": (
                    "production-train"
                    if row.get("calibration_split") == "train"
                    else "production-holdout"
                ),
                "actual_area_um2": actual,
                "predicted_area_um2": predicted,
                "error_pct": (predicted - actual) / max(actual, 1e-9) * 100.0,
            }
        )
    return artifact, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument(
        "--promote-to-calibration",
        action="store_true",
        help="copy a passing artifact into the runtime calibration directory",
    )
    args = parser.parse_args()

    artifact, diagnostics = fit_artifact(args.current_csv, args.baseline_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "point_id",
        "role",
        "collection_split",
        "fit_split",
        "actual_area_um2",
        "predicted_area_um2",
        "error_pct",
    )
    with args.diagnostics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)

    promoted = artifact["metadata"]["status"] == "fitted_from_paired_rtl_v6_dc"
    if args.promote_to_calibration:
        if not promoted:
            raise RuntimeError(
                "refusing to promote rtl-v6 area calibration: "
                + "; ".join(artifact["metadata"]["failures"])
            )
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            args.output,
            CALIBRATION_DIR / "vector_rtl_v6_delta_coefficients.json",
        )
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
