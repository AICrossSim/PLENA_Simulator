#!/usr/bin/env python3
"""Fit the hierarchy-supervised MatrixMachine structural area model v4.

The fitter treats the RTL generate census as fixed.  It never asks regression
to rediscover PE or reduction instance counts.  Nonnegative regressions are
used only for the area of one replicated structural feature and for residual
peripheral logic.  MatrixMachine holdouts are grouped by (MLEN, BLEN), so a
shape cannot leak through another precision row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_matrix_machine_calibration import _fit_nonnegative, parse_hierarchy_area
except ModuleNotFoundError:
    from .run_matrix_machine_calibration import _fit_nonnegative, parse_hierarchy_area

from analytic_models.area_new import matrix_structural_model


CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"
DEFAULT_OUTPUT = CALIBRATION_DIR / "matrix_structural_v4_coefficients.json"
FP_WIDTH = 12

COMPONENT_FEATURES = {
    "array": ["streamer_operand_bits", "scale_distribution_bits", "array_fixed"],
    "reduce": ["edge_width", "tree_depth_width", "scale_edges", "active_outputs"],
    "output_accumulator": ["cell_fp_width", "cells"],
    "output_conversion": ["cell_acc_width", "cell_fp_width", "cells"],
    "result_buffer": ["matrix_fp_bits", "tile_fp_bits", "mlen_fp_bits"],
    "io_pipeline": ["lane_bits", "lanes", "fixed"],
    "control": ["shape_log", "mlen", "fixed"],
}

HIERARCHY_TARGETS = {
    "array": "hier_array_area",
    "reduce": "hier_reduce_area",
    "output_accumulator": "hier_output_accumulator_area",
    "output_conversion": "hier_output_conversion_area",
    "result_buffer": "hier_result_buffer_area",
    "io_pipeline": "hier_io_pipeline_area",
    "control": "hier_control_area",
}


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value in {None, ""} else float(value)


def _read_latest_complete(paths: Iterable[Path]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "complete":
                    continue
                latest[str(row.get("point_key") or row.get("point_id"))] = dict(row)
    return list(latest.values())


def _report_path(row: Mapping[str, Any]) -> Path | None:
    value = row.get("report_dir")
    if not value:
        return None
    directory = Path(str(value))
    if not directory.is_absolute():
        directory = ROOT / directory
    report = directory / "area.rpt"
    return report if report.exists() else None


def _enrich_hierarchy(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("level") != "matrix_machine":
            continue
        report = _report_path(row)
        if report is not None:
            row.update(parse_hierarchy_area(report))


def _precision_key(mode: str, row: Mapping[str, Any]) -> str:
    if mode == "mxint":
        return f"t{int(_number(row, 'T_BITS'))}_l{int(_number(row, 'L_BITS'))}"
    return (
        f"te{int(_number(row, 'T_EXP'))}m{int(_number(row, 'T_MANT'))}_"
        f"le{int(_number(row, 'L_EXP'))}m{int(_number(row, 'L_MANT'))}"
    )


def _fit_pe_lookup(
    mode: str,
    pe_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float] | float]:
    by_key_depth: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in pe_rows:
        if row.get("mode") != mode or row.get("level") != "pe":
            continue
        depth = int(_number(row, "ACC_DEPTH", 16))
        by_key_depth[(_precision_key(mode, row), depth)].append(_number(row, "area_um2"))

    lookup: dict[str, float] = {}
    for key, depth in by_key_depth:
        if mode == "mxfp" or depth == 16:
            lookup[key] = statistics.mean(by_key_depth[(key, depth)])
    if mode == "mxfp":
        return lookup, 0.0

    slopes: dict[str, float] = {}
    all_slopes: list[float] = []
    for key, base in lookup.items():
        samples: list[float] = []
        for (sample_key, depth), areas in by_key_depth.items():
            if sample_key != key or depth == 16:
                continue
            delta = math.log2(depth) - math.log2(16)
            if delta > 0:
                samples.append(max(0.0, (statistics.mean(areas) - base) / delta))
        if samples:
            slopes[key] = statistics.mean(samples)
            all_slopes.extend(samples)
    slopes["default"] = statistics.mean(all_slopes) if all_slopes else 0.0
    return lookup, slopes


def _pe_area(mode: str, row: Mapping[str, Any], model: Mapping[str, Any], blen: int) -> float:
    key = _precision_key(mode, row)
    base = float(model["pe_lookup"][key])
    if mode != "mxint":
        return base
    slope_spec = model.get("pe_depth_slope", 0.0)
    slope = (
        float(slope_spec.get(key, slope_spec.get("default", 0.0)))
        if isinstance(slope_spec, Mapping)
        else float(slope_spec)
    )
    return max(0.0, base + slope * (math.log2(blen) - math.log2(16)))


def _feature_values(
    mode: str,
    row: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    mlen: int | None = None,
    blen: int | None = None,
) -> dict[str, dict[str, float]]:
    b = int(blen if blen is not None else _number(row, "BLEN", _number(row, "BLOCK_DIM")))
    m = int(mlen if mlen is not None else _number(row, "MLEN", b))
    if b <= 0 or m <= 0 or m % b:
        raise ValueError(f"illegal structural shape M={m}, B={b}")
    scale = int(_number(row, "scale_width", 8))
    counts = matrix_structural_model.structural_counts(m, b, FP_WIDTH)
    splits = counts["array_count"]
    log_splits = math.log2(splits) if splits > 1 else 0.0
    if mode == "mxint":
        t_width = int(_number(row, "T_BITS"))
        l_width = int(_number(row, "L_BITS"))
        accumulator_width = (
            t_width
            + l_width
            + max(1, math.ceil(math.log2(b)))
            + int(model.get("mxint_max_shift", 16))
            + max(1, math.ceil(math.log2(splits + 1)))
        )
    else:
        t_width = 1 + int(_number(row, "T_EXP")) + int(_number(row, "T_MANT"))
        l_width = 1 + int(_number(row, "L_EXP")) + int(_number(row, "L_MANT"))
        accumulator_width = int(model.get("acc_fp_width", 16))
    pe_area = _pe_area(mode, row, model, b)
    active_outputs = b * b if splits > 1 else 0
    return {
        "array": {
            "pe_grid": counts["pe_count"] * pe_area,
            "streamer_operand_bits": counts["pe_count"] * (t_width + l_width),
            "scale_distribution_bits": m * scale,
            "array_fixed": float(splits),
        },
        "reduce": {
            "edge_width": counts["reduce_node_count"] * accumulator_width,
            "tree_depth_width": active_outputs * log_splits * accumulator_width,
            "scale_edges": counts["reduce_node_count"] * scale,
            "active_outputs": float(active_outputs),
        },
        "output_accumulator": {
            "cell_fp_width": counts["output_cell_count"] * FP_WIDTH,
            "cells": float(counts["output_cell_count"]),
        },
        "output_conversion": {
            "cell_acc_width": counts["output_cell_count"] * accumulator_width,
            "cell_fp_width": counts["output_cell_count"] * FP_WIDTH,
            "cells": float(counts["output_cell_count"]),
        },
        "result_buffer": {
            "matrix_fp_bits": float(counts["mxfp_block_buffer_bits"] if mode == "mxfp" else 0),
            "tile_fp_bits": float(counts["output_cell_count"] * FP_WIDTH if mode == "mxfp" else 0),
            "mlen_fp_bits": float(m * FP_WIDTH),
        },
        "io_pipeline": {
            "lane_bits": m * (t_width + l_width + 2 * scale + FP_WIDTH),
            "lanes": float(m),
            "fixed": 1.0,
        },
        "control": {
            "shape_log": 1.0 + math.log2(m) + math.log2(b) + log_splits,
            "mlen": float(m),
            "fixed": 1.0,
        },
    }


def _fit_named(features: list[dict[str, float]], targets: list[float], names: list[str]) -> dict[str, float]:
    if not features or not any(target > 0 for target in targets):
        return {name: 0.0 for name in names}
    x = [[float(item[name]) for name in names] for item in features]
    values = _fit_nonnegative(x, targets)
    return {name: float(value) for name, value in zip(names, values)}


def _leaf_reduce_observations(
    mode: str,
    leaf_rows: list[dict[str, Any]],
    model: Mapping[str, Any],
) -> tuple[list[dict[str, float]], list[float]]:
    features: list[dict[str, float]] = []
    targets: list[float] = []
    for row in leaf_rows:
        if row.get("mode") != mode or row.get("level") != "reduce_leaf":
            continue
        b = int(_number(row, "COMPUTE_DIM"))
        s = int(_number(row, "SYS_ARRAY_AMOUNT"))
        features.append(_feature_values(mode, row, model, mlen=b * s, blen=b)["reduce"])
        targets.append(_number(row, "area_um2"))
    return features, targets


def _fit_family(
    mode: str,
    pe_rows: list[dict[str, Any]],
    mini_rows: list[dict[str, Any]],
    mm_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pe_lookup, depth_slope = _fit_pe_lookup(mode, pe_rows + leaf_rows)
    model: dict[str, Any] = {
        "pe_lookup": pe_lookup,
        "pe_depth_reference": 16,
        "pe_depth_slope": depth_slope,
        "pe_area_floor": min(pe_lookup.values()) * 0.5,
        "mxint_max_shift": 16,
        "acc_fp_width": 16,
    }

    array_features: list[dict[str, float]] = []
    array_residuals: list[float] = []
    for row in mini_rows:
        if row.get("mode") != mode:
            continue
        feature = _feature_values(mode, row, model)["array"]
        array_features.append(feature)
        array_residuals.append(max(0.0, _number(row, "area_um2") - feature["pe_grid"]))
    for row in mm_rows:
        feature = _feature_values(mode, row, model)["array"]
        array_features.append(feature)
        array_residuals.append(max(0.0, _number(row, "hier_array_area") - feature["pe_grid"]))
    array_residual = _fit_named(
        array_features,
        array_residuals,
        COMPONENT_FEATURES["array"],
    )
    model["array"] = {"pe_grid": 1.0, **array_residual}

    for component in [
        "reduce",
        "output_accumulator",
        "output_conversion",
        "result_buffer",
        "io_pipeline",
        "control",
    ]:
        features = [_feature_values(mode, row, model)[component] for row in mm_rows]
        targets = [_number(row, HIERARCHY_TARGETS[component]) for row in mm_rows]
        if component == "reduce":
            leaf_features, leaf_targets = _leaf_reduce_observations(mode, leaf_rows, model)
            features.extend(leaf_features)
            targets.extend(leaf_targets)
        elif component == "output_conversion" and mode == "mxint":
            for row in leaf_rows:
                if row.get("mode") != mode or row.get("level") != "output_conversion":
                    continue
                features.append(
                    {
                        "cell_acc_width": _number(row, "ACC_WIDTH"),
                        "cell_fp_width": 1 + _number(row, "FP_EXP_WIDTH") + _number(row, "FP_MANT_WIDTH"),
                        "cells": 1.0,
                    }
                )
                targets.append(_number(row, "area_um2"))
        model[component] = _fit_named(features, targets, COMPONENT_FEATURES[component])
    return model


def _row_inputs(mode: str, row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": mode,
        "MLEN": int(_number(row, "MLEN")),
        "BLEN": int(_number(row, "BLEN")),
        "scale_width": int(_number(row, "scale_width", 8)),
        "FP_SETTING": "FP_E5M6",
    }
    for key in ("T_BITS", "L_BITS", "T_EXP", "T_MANT", "L_EXP", "L_MANT"):
        if row.get(key) not in {None, ""}:
            result[key.lower()] = int(_number(row, key))
    return result


def _errors(actual: list[float], predicted: list[float]) -> list[float]:
    return [abs(p - a) / max(abs(a), 1e-9) * 100.0 for a, p in zip(actual, predicted)]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _diagnose_family(mode: str, rows: list[dict[str, Any]], model: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    component_errors: dict[str, list[float]] = defaultdict(list)
    total_actual: list[float] = []
    total_predicted: list[float] = []
    for row in rows:
        estimate = matrix_structural_model._evaluate_family(mode, _row_inputs(mode, row), model)
        actual = _number(row, "area_um2")
        total_actual.append(actual)
        total_predicted.append(float(estimate["area"]))
        item: dict[str, Any] = {
            "mode": mode,
            "point_id": row.get("point_id"),
            "MLEN": row.get("MLEN"),
            "BLEN": row.get("BLEN"),
            "actual_area_um2": actual,
            "predicted_area_um2": estimate["area"],
            "error_pct": (float(estimate["area"]) - actual) / max(actual, 1e-9) * 100.0,
        }
        for component, target in HIERARCHY_TARGETS.items():
            predicted_key = f"{component}_area"
            if component == "array":
                predicted_key = "array_stack_area"
            elif component == "reduce":
                predicted_key = "reduce_tree_area"
            actual_component = _number(row, target)
            predicted_component = float(estimate["breakdown"].get(predicted_key, 0.0))
            if actual_component > 0:
                component_errors[component].append(
                    abs(predicted_component - actual_component) / actual_component * 100.0
                )
            item[f"actual_{component}_um2"] = actual_component
            item[f"predicted_{component}_um2"] = predicted_component
        diagnostics.append(item)
    errors = _errors(total_actual, total_predicted)
    summary = {
        "train_rows": len(rows),
        "train_mape_pct": statistics.mean(errors) if errors else float("nan"),
        "train_median_error_pct": statistics.median(errors) if errors else float("nan"),
        "train_p95_error_pct": _percentile(errors, 0.95),
        "component_median_error_pct": {
            key: statistics.median(values) if values else None
            for key, values in component_errors.items()
        },
    }
    return summary, diagnostics


def _grouped_holdout(
    mode: str,
    pe_rows: list[dict[str, Any]],
    mini_rows: list[dict[str, Any]],
    mm_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float], list[float], list[dict[str, Any]]]:
    groups = sorted({(int(_number(row, "MLEN")), int(_number(row, "BLEN"))) for row in mm_rows})
    ensemble: list[dict[str, Any]] = []
    errors: list[float] = []
    actual_over_predicted: list[float] = []
    rows_out: list[dict[str, Any]] = []
    for group in groups:
        train = [row for row in mm_rows if (int(_number(row, "MLEN")), int(_number(row, "BLEN"))) != group]
        holdout = [row for row in mm_rows if row not in train]
        if not train or not holdout:
            continue
        model = _fit_family(mode, pe_rows, mini_rows, train, leaf_rows)
        ensemble.append(model)
        for row in holdout:
            estimate = matrix_structural_model._evaluate_family(mode, _row_inputs(mode, row), model)
            actual = _number(row, "area_um2")
            error = abs(float(estimate["area"]) - actual) / max(actual, 1e-9) * 100.0
            errors.append(error)
            actual_over_predicted.append(
                actual / max(float(estimate["area"]), 1e-9)
            )
            rows_out.append(
                {
                    "mode": mode,
                    "holdout_MLEN": group[0],
                    "holdout_BLEN": group[1],
                    "point_id": row.get("point_id"),
                    "actual_area_um2": actual,
                    "predicted_area_um2": estimate["area"],
                    "absolute_error_pct": error,
                }
            )
    return ensemble, errors, actual_over_predicted, rows_out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _invariant_audit(artifact: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    shapes = [(m, b) for m in [128, 256, 512, 1024, 2048] for b in [4, 8, 16, 32, 64, 128, 256, 512, 1024] if b <= m and m % b == 0]
    family_profiles = {
        "mxint": [
            {"t_bits": 4, "l_bits": 2},
            {"t_bits": 4, "l_bits": 4},
            {"t_bits": 4, "l_bits": 8},
            {"t_bits": 8, "l_bits": 2},
            {"t_bits": 8, "l_bits": 4},
            {"t_bits": 8, "l_bits": 8},
        ],
        "mxfp": [
            {"t_exp": 1, "t_mant": 2, "l_exp": 1, "l_mant": 2},
            {"t_exp": 2, "t_mant": 1, "l_exp": 2, "l_mant": 1},
            {"t_exp": 4, "t_mant": 3, "l_exp": 4, "l_mant": 3},
            {"t_exp": 5, "t_mant": 2, "l_exp": 5, "l_mant": 2},
        ],
    }
    values: dict[tuple[str, int, int, int], float] = {}
    for mode, profiles in family_profiles.items():
        for index, profile in enumerate(profiles):
            for m, b in shapes:
                inputs = {"mode": mode, "MLEN": m, "BLEN": b, "scale_width": 8, "FP_SETTING": "FP_E5M6", **profile}
                estimate = matrix_structural_model.estimate(inputs, artifact)
                area = float(estimate["area"])
                values[(mode, index, m, b)] = area
                if not math.isfinite(area) or area < 0:
                    issues.append(f"non-finite/negative area {mode} profile={index} M={m} B={b}")
                if m == b and estimate["breakdown"]["reduce_tree_area"] != 0.0:
                    issues.append(f"nonzero S=1 reduction {mode} profile={index} M=B={m}")
        for index in range(len(profiles)):
            for b in [4, 8, 16, 32, 64, 128]:
                series = [values[(mode, index, m, b)] for m in [128, 256, 512, 1024, 2048] if b <= m]
                if any(right + 1e-6 < left for left, right in zip(series, series[1:])):
                    issues.append(f"non-monotonic MLEN {mode} profile={index} B={b}")
            for m in [128, 256, 512, 1024, 2048]:
                valid_b = [b for b in [4, 8, 16, 32, 64, 128, 256, 512, 1024] if b <= m]
                series = [values[(mode, index, m, b)] for b in valid_b]
                if any(right + 1e-6 < left for left, right in zip(series, series[1:])):
                    issues.append(f"non-monotonic BLEN {mode} profile={index} M={m}")
        # Precision monotonicity is a partial order.  MXFP formats with the
        # same total width (for example E1M2 and E2M1) are not ordered because
        # exponent/mantissa allocation can change the synthesized leaf area in
        # either direction.  Wider T and L operands must, however, never make
        # the estimated datapath smaller.
        def operand_widths(profile: Mapping[str, int]) -> tuple[int, int]:
            if mode == "mxint":
                return int(profile["t_bits"]), int(profile["l_bits"])
            return (
                1 + int(profile["t_exp"]) + int(profile["t_mant"]),
                1 + int(profile["l_exp"]) + int(profile["l_mant"]),
            )

        for low_index, low_profile in enumerate(profiles):
            low_t, low_l = operand_widths(low_profile)
            for high_index, high_profile in enumerate(profiles):
                high_t, high_l = operand_widths(high_profile)
                if (low_t, low_l) == (high_t, high_l):
                    continue
                if low_t <= high_t and low_l <= high_l:
                    for m, b in shapes:
                        if values[(mode, low_index, m, b)] > values[(mode, high_index, m, b)] + 1e-6:
                            issues.append(
                                f"precision inversion {mode} low={low_index} high={high_index} "
                                f"M={m} B={b}"
                            )
    return {"pass": not issues, "issue_count": len(issues), "issues": issues[:100]}


def _calibrated_ordering_audit(
    mode: str,
    rows: list[dict[str, Any]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that predictions preserve measured ordering at each DC shape."""

    grouped: dict[tuple[int, int], list[tuple[float, float, str]]] = defaultdict(list)
    for row in rows:
        estimate = matrix_structural_model._evaluate_family(
            mode, _row_inputs(mode, row), model
        )
        grouped[(int(_number(row, "MLEN")), int(_number(row, "BLEN")))].append(
            (
                _number(row, "area_um2"),
                float(estimate["area"]),
                str(row.get("point_id")),
            )
        )
    violations: list[dict[str, Any]] = []
    comparisons = 0
    for shape, points in grouped.items():
        for left_index, left in enumerate(points):
            for right in points[left_index + 1 :]:
                actual_delta = left[0] - right[0]
                if abs(actual_delta) <= 1e-9:
                    continue
                comparisons += 1
                predicted_delta = left[1] - right[1]
                if actual_delta * predicted_delta < 0:
                    violations.append(
                        {
                            "MLEN": shape[0],
                            "BLEN": shape[1],
                            "left": left[2],
                            "right": right[2],
                            "actual_delta_um2": actual_delta,
                            "predicted_delta_um2": predicted_delta,
                        }
                    )
    return {
        "pass": not violations,
        "comparisons": comparisons,
        "violation_count": len(violations),
        "violations": violations[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-mxint", type=Path, default=CALIBRATION_DIR / "pe_mxint.csv")
    parser.add_argument("--pe-mxfp", type=Path, default=CALIBRATION_DIR / "pe_mxfp.csv")
    parser.add_argument("--mini-mxint", type=Path, default=CALIBRATION_DIR / "mini_array_mxint.csv")
    parser.add_argument("--mini-mxfp", type=Path, default=CALIBRATION_DIR / "mini_array_mxfp.csv")
    parser.add_argument("--matrix-mxint", type=Path, default=CALIBRATION_DIR / "matrix_machine_mxint.csv")
    parser.add_argument("--matrix-mxfp", type=Path, default=CALIBRATION_DIR / "matrix_machine_mxfp.csv")
    parser.add_argument("--leaf-points", type=Path, default=CALIBRATION_DIR / "matrix_structural_leaf_points.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics-dir", type=Path, default=CALIBRATION_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pe_rows = _read_latest_complete([args.pe_mxint, args.pe_mxfp])
    mini_rows = _read_latest_complete([args.mini_mxint, args.mini_mxfp, args.leaf_points])
    mm_rows = _read_latest_complete([args.matrix_mxint, args.matrix_mxfp])
    leaf_rows = _read_latest_complete([args.leaf_points])
    _enrich_hierarchy(mm_rows)

    try:
        artifact_source = str(args.output.resolve().relative_to(ROOT))
    except ValueError:
        artifact_source = str(args.output)
    artifact: dict[str, Any] = {
        "model_version": matrix_structural_model.MODEL_VERSION,
        "source": artifact_source,
        "units": "um2",
        "uncertainty_basis": (
            "grouped_MLEN_BLEN_holdout_coefficient_ensemble_and_residual_scale"
        ),
        "families": {},
    }
    all_diagnostics: list[dict[str, Any]] = []
    all_holdout: list[dict[str, Any]] = []
    for mode in ("mxint", "mxfp"):
        mode_pe = [row for row in pe_rows if row.get("mode") == mode]
        mode_mini = [row for row in mini_rows if row.get("mode") == mode and row.get("level") == "mini_array"]
        mode_mm = [row for row in mm_rows if row.get("mode") == mode]
        model = _fit_family(mode, mode_pe, mode_mini, mode_mm, leaf_rows)
        summary, diagnostics = _diagnose_family(mode, mode_mm, model)
        ensemble, holdout_errors, holdout_scales, holdout_rows = _grouped_holdout(
            mode, mode_pe, mode_mini, mode_mm, leaf_rows
        )
        summary.update(
            {
                "grouped_holdout_rows": len(holdout_errors),
                "grouped_holdout_median_error_pct": statistics.median(holdout_errors) if holdout_errors else None,
                "grouped_holdout_p95_error_pct": _percentile(holdout_errors, 0.95) if holdout_errors else None,
            }
        )
        artifact["families"][mode] = {
            "coefficients": model,
            "ensemble": ensemble,
            "uncertainty_scale": {
                "p10": min(1.0, _percentile(holdout_scales, 0.10)),
                "p50": 1.0,
                "p90": max(1.0, _percentile(holdout_scales, 0.90)),
                "basis": "grouped_holdout_actual_over_predicted",
            },
            "diagnostics": summary,
            "calibrated_ordering": _calibrated_ordering_audit(
                mode, mode_mm, model
            ),
        }
        all_diagnostics.extend(diagnostics)
        all_holdout.extend(holdout_rows)

    audit = _invariant_audit(artifact)
    artifact["physical_invariants"] = audit
    artifact["acceptance"] = {
        "train_mape_le_10pct": all(
            family["diagnostics"]["train_mape_pct"] <= 10.0
            for family in artifact["families"].values()
        ),
        "grouped_holdout_median_le_12pct": all(
            (family["diagnostics"]["grouped_holdout_median_error_pct"] or float("inf")) <= 12.0
            for family in artifact["families"].values()
        ),
        "grouped_holdout_p95_le_20pct": all(
            (family["diagnostics"]["grouped_holdout_p95_error_pct"] or float("inf")) <= 20.0
            for family in artifact["families"].values()
        ),
        "physical_invariants": audit["pass"],
        "calibrated_shape_ordering": all(
            family["calibrated_ordering"]["pass"]
            for family in artifact["families"].values()
        ),
    }
    artifact["acceptance"]["pass"] = all(artifact["acceptance"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.diagnostics_dir / "matrix_structural_v4_diagnostics.csv", all_diagnostics)
    _write_csv(args.diagnostics_dir / "matrix_structural_v4_grouped_holdout.csv", all_holdout)
    summary_path = args.diagnostics_dir / "matrix_structural_v4_validation.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_version": artifact["model_version"],
                "families": {
                    mode: {
                        **value["diagnostics"],
                        "calibrated_ordering": value["calibrated_ordering"],
                    }
                    for mode, value in artifact["families"].items()
                },
                "physical_invariants": audit,
                "acceptance": artifact["acceptance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(json.loads(summary_path.read_text()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
