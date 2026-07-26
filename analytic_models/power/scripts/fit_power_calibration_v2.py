#!/usr/bin/env python3
# ruff: noqa: E402
"""Fit the family-separated RTL-activity power candidate v2.

The v1 fit treated low-toggle, random, and transformer-like operands as
samples from one linear model.  That made operand activity look like a shape
coefficient and left mixed Vector/Scalar microkernels rank deficient.  This
fitter instead extracts one slope per mapped configuration and hardware
microkernel.  Qwen-like slopes define the nominal model; low and random
slopes define an empirical activity envelope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import sys
import time
from itertools import combinations, pairwise
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.power.scripts.run_power_calibration import (
    HBM_V2_MICROKERNELS,
)
from analytic_models.power.scripts.run_rtl_activity_power_calibration import (
    _power_group_fields,
)


VECTOR_FAMILY = {
    "add_vv": "lane_add_sub_vv",
    "add_vf": "lane_add_sub_vf",
    "add_vseg": "lane_add_sub_vseg",
    "mul_vv": "lane_multiply_vv",
    "mul_vf": "lane_multiply_vf",
    "mul_vseg": "lane_multiply_vseg",
    "exp": "lane_sfu_exp",
    "reciprocal": "lane_sfu_reciprocal",
    "reduce_sum": "reduction_sum_full",
    "reduce_max": "reduction_max_full",
    "reduce_sum_seg": "reduction_sum_segment",
    "reduce_max_seg": "reduction_max_segment",
    "reduce_sum_segs": "reduction_sum_segments",
    "reduce_max_segs": "reduction_max_segments",
    "shift": "lane_movement_shift",
}
SCALAR_FAMILY = {
    "fp_alu": "fp_add_sub_move",
    "fp_mul": "fp_multiply",
    "fp_exp": "fp_sfu_exp",
    "fp_reciprocal": "fp_sfu_reciprocal",
    "fp_sqrt": "fp_sfu_sqrt",
    "fp_rsqrt": "fp_sfu_rsqrt",
    "int_alu": "integer_alu",
    "int_mul": "integer_multiply",
    "register_access": "register_or_sram_access",
}
LANE_ACCESS_FAMILY = {
    "lane_load": "vector_lane_load",
    "lane_store": "vector_lane_store",
}
CORNER = {
    "process": "ASAP7_TT",
    "voltage_v": 0.7,
    "temperature_c": 25.0,
    "clock_period_ps": 1000,
}


def _percentile(values: Iterable[float], percentile: float) -> float:
    materialized = list(values)
    return math.inf if not materialized else float(np.percentile(materialized, percentile))


def _latest_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            latest[(raw.get("point_key", ""), raw.get("scenario", ""))] = raw
    rows: list[dict[str, Any]] = []
    for raw in latest.values():
        if raw.get("status") != "complete" or not raw.get("features_json"):
            continue
        sidecar = json.loads(raw["features_json"])
        group_fields: dict[str, Any]
        if raw.get("clock_network_energy_pj") and raw.get("nonclock_dynamic_energy_pj"):
            group_fields = {
                name: float(raw[name])
                for name in (
                    "clock_network_dynamic_power_mw",
                    "register_dynamic_power_mw",
                    "combinational_dynamic_power_mw",
                    "nonclock_dynamic_power_mw",
                    "clock_network_energy_pj",
                    "register_dynamic_energy_pj",
                    "combinational_dynamic_energy_pj",
                    "nonclock_dynamic_energy_pj",
                )
            }
        else:
            report = Path(raw.get("power_report", ""))
            if not report.exists():
                raise ValueError(
                    f"missing power-group data and report for {raw.get('point_id')}/{raw.get('scenario')}"
                )
            group_fields = _power_group_fields(
                report.read_text(errors="ignore"), float(raw["window_ns"])
            )
        rows.append(
            {
                **raw,
                "holdout": bool(int(raw.get("holdout") or 0)),
                "repeat_count": int(raw["repeat_count"]),
                "microkernel": raw.get("microkernel") or sidecar.get("microkernel", "mixed"),
                "window_energy_pj": float(raw["window_dynamic_energy_pj"]),
                "window_cycles": int(sidecar["measurement_cycles"]),
                "accepted_actions": int(sidecar.get("accepted_actions", 0)),
                "logic_area_um2": float(raw.get("logic_area_um2") or 0.0),
                "leakage_power_mw": float(raw.get("leakage_power_mw") or 0.0),
                "params": dict(sidecar.get("params", {})),
                "dynamic_features": {
                    str(name): float(value)
                    for name, value in sidecar.get("dynamic_features", {}).items()
                },
                "mix_action_counts": {
                    str(name): int(value)
                    for name, value in sidecar.get("mix_action_counts", {}).items()
                },
                "qwen_mix_semantic_hash": sidecar.get("qwen_mix_semantic_hash"),
                **group_fields,
            }
        )
    return rows


def _pair_idle(rows: list[dict[str, Any]]) -> None:
    idle: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["pattern"] == "idle":
            idle[(row["point_key"], row["repeat_count"])] = row
    missing: list[str] = []
    for row in rows:
        if row["pattern"] == "idle":
            row["incremental_energy_pj"] = 0.0
            row["normalized_dynamic_energy_pj"] = row["window_energy_pj"]
            row["excluded_clock_delta_pj"] = 0.0
            row["nonclock_residual_positive"] = True
            continue
        baseline = idle.get((row["point_key"], row["repeat_count"]))
        if baseline is None or baseline["window_cycles"] != row["window_cycles"]:
            missing.append(f"{row['point_id']}/{row['scenario']}")
            continue
        # DC's pre-CTS ``clock_network`` group is mostly sequential-cell
        # clock-pin internal power.  It can decrease when data/state activity
        # changes, masking a positive datapath increment.  v2 therefore keeps
        # the matched idle total as the baseline and fits only the measured
        # non-clock residual.  No negative value is clipped or hidden.
        row["incremental_energy_pj"] = (
            row["nonclock_dynamic_energy_pj"]
            - baseline["nonclock_dynamic_energy_pj"]
        )
        row["normalized_dynamic_energy_pj"] = (
            baseline["window_energy_pj"] + row["incremental_energy_pj"]
        )
        row["excluded_clock_delta_pj"] = (
            row["clock_network_energy_pj"]
            - baseline["clock_network_energy_pj"]
        )
        row["nonclock_residual_positive"] = row["incremental_energy_pj"] > 0
    if missing:
        raise ValueError(f"missing exact matched idle windows: {missing[:12]}")


def _linear_slope(samples: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Return nonnegative slope, intercept, and R² for action-window energy."""

    x = np.asarray([sample[0] for sample in samples], dtype=float)
    y = np.asarray([sample[1] for sample in samples], dtype=float)
    if len(samples) == 1:
        return max(0.0, y[0] / max(x[0], 1.0)), 0.0, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total if total else -math.inf
    return max(0.0, float(slope)), float(intercept), r2


def _extract_slopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["pattern"] == "idle" or row["microkernel"] in {"idle", "mixed"}:
            continue
        groups[(row["point_key"], row["pattern"], row["microkernel"])].append(row)
    slopes: list[dict[str, Any]] = []
    for (_, pattern, microkernel), samples in sorted(groups.items()):
        ordered = sorted(samples, key=lambda row: row["accepted_actions"])
        fit_samples = [
            (row["accepted_actions"], row["incremental_energy_pj"])
            for row in ordered
        ]
        slope, intercept, r2 = _linear_slope(fit_samples)
        exemplar = ordered[0]
        slopes.append(
            {
                "point_id": exemplar["point_id"],
                "point_key": exemplar["point_key"],
                "component": exemplar["component"],
                "holdout": exemplar["holdout"],
                "pattern": pattern,
                "microkernel": microkernel,
                "params": exemplar["params"],
                "energy_per_action_pj": slope,
                "startup_pj": intercept,
                "r2": r2,
                "sample_count": len(samples),
                "dynamic_features_per_action": {
                    name: value / max(1, exemplar["accepted_actions"])
                    for name, value in exemplar["dynamic_features"].items()
                },
            }
        )
    return slopes


def _fp_key(params: Mapping[str, Any], component: str) -> str:
    prefix = "V_FP" if component == "vector" else "S_FP"
    return f"FP_E{int(params[f'{prefix}_EXP_WIDTH'])}M{int(params[f'{prefix}_MANT_WIDTH'])}"


def _fp_width(format_key: str) -> int:
    match = re.fullmatch(r"FP_E(\d+)M(\d+)", format_key)
    if not match:
        raise ValueError(f"invalid FP format {format_key}")
    return 1 + int(match.group(1)) + int(match.group(2))


def _nnls_ridge(matrix: np.ndarray, target: np.ndarray, ridge: float = 1e-9) -> tuple[np.ndarray, dict[str, Any]]:
    scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-18)
    normalized = matrix / scale
    augmented = np.vstack((normalized, math.sqrt(ridge) * np.eye(normalized.shape[1])))
    augmented_target = np.concatenate((target, np.zeros(normalized.shape[1])))
    try:
        from scipy.optimize import nnls

        normalized_coefficients = nnls(augmented, augmented_target)[0]
    except ImportError:
        normalized_coefficients = np.zeros(normalized.shape[1])
        step = 1.0 / max(float(np.linalg.norm(augmented, ord=2)) ** 2, 1e-12)
        for _ in range(100_000):
            update = np.maximum(
                0.0,
                normalized_coefficients
                - step * augmented.T @ (augmented @ normalized_coefficients - augmented_target),
            )
            if np.linalg.norm(update - normalized_coefficients) < 1e-12:
                break
            normalized_coefficients = update
    coefficients = normalized_coefficients / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = math.inf if singular[-1] <= 1e-15 else float(singular[0] / singular[-1])
    return coefficients, {
        "rank": int(np.linalg.matrix_rank(normalized)),
        "feature_count": int(matrix.shape[1]),
        "condition_number": condition,
        "zero_coefficients": [index for index, value in enumerate(coefficients) if value <= 1e-18],
    }


def _matrix_mode(params: Mapping[str, Any]) -> str:
    return str(params.get("mode", "mxint"))


def _matrix_widths(params: Mapping[str, Any]) -> tuple[int, int]:
    if _matrix_mode(params) == "mxint":
        return int(params["T_BITS"]), int(params["L_BITS"])
    return (
        1 + int(params["T_EXP"]) + int(params["T_MANT"]),
        1 + int(params["L_EXP"]) + int(params["L_MANT"]),
    )


def _fit_matrix_pe(
    slopes: list[dict[str, Any]], *, include_holdout: bool
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    result: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, Any] = {}
    for mode in ("mxint", "mxfp"):
        selected = [
            row
            for row in slopes
            if row["component"] == "matrix"
            and row["pattern"] == "representative-qwen"
            and row["microkernel"] == "array_compute"
            and _matrix_mode(row["params"]) == mode
            and (include_holdout or not row["holdout"])
        ]
        if len(selected) < 3:
            raise ValueError(f"matrix {mode} needs at least three Qwen-like precision points")
        matrix_rows: list[list[float]] = []
        targets: list[float] = []
        for row in selected:
            t_width, l_width = _matrix_widths(row["params"])
            block = int(row["params"]["BLOCK_DIM"])
            if mode == "mxint":
                # A mini-array action includes fixed launch work, B cycles of
                # feed/control work, and B^3 PE-MAC cycles.  Separating these
                # terms prevents tiny B anchors from inflating the PE energy
                # used to extrapolate larger arrays.
                matrix_rows.append(
                    [
                        1.0,
                        float(block),
                        float(block**3),
                        float(block**3 * t_width * l_width),
                        float(block**3 * (t_width + l_width)),
                    ]
                )
                targets.append(row["energy_per_action_pj"])
            else:
                # MXFP currently has precision diversity only at B=4.  Keep
                # its B^3 PE extrapolation identifiable instead of inventing
                # launch/feed coefficients that the data cannot separate.
                matrix_rows.append([1.0, t_width * l_width, t_width + l_width])
                targets.append(row["energy_per_action_pj"] / max(1, block**3))
        coefficients, diagnostic = _nnls_ridge(
            np.asarray(matrix_rows, dtype=float), np.asarray(targets, dtype=float)
        )
        if mode == "mxint":
            result[mode] = {
                "slice_fixed": float(coefficients[0]),
                "feed_cycle": float(coefficients[1]),
                "base": float(coefficients[2]),
                "bit_product": float(coefficients[3]),
                "width_sum": float(coefficients[4]),
            }
        else:
            result[mode] = {
                "slice_fixed": 0.0,
                "feed_cycle": 0.0,
                "base": float(coefficients[0]),
                "bit_product": float(coefficients[1]),
                "width_sum": float(coefficients[2]),
            }
        diagnostics[mode] = {**diagnostic, "point_count": len(selected)}
    return result, diagnostics


def _vector_scale(microkernel: str, params: Mapping[str, Any]) -> int:
    vlen = int(params["VLEN"])
    if microkernel in {"reduce_sum", "reduce_max"}:
        return max(1, (vlen - 1) * int(math.log2(max(2, vlen))))
    if microkernel in {"reduce_sum_seg", "reduce_max_seg"}:
        # Selecting one segment still toggles the full-width input/select and
        # tree-routing network in the current VectorMachine.
        return max(1, vlen * int(math.log2(max(2, vlen))))
    if microkernel in {"reduce_sum_segs", "reduce_max_segs"}:
        # Compact multi-segment outputs reuse intermediate tree levels; the
        # measured activity follows instantiated lanes rather than one full
        # independent tree per result segment.
        return vlen
    return vlen


def _median_tables(
    slopes: list[dict[str, Any]], *, include_holdout: bool
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    vector_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    scalar_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in slopes:
        if row["pattern"] != "representative-qwen" or (row["holdout"] and not include_holdout):
            continue
        microkernel = row["microkernel"]
        if row["component"] == "vector" and microkernel in VECTOR_FAMILY:
            family = VECTOR_FAMILY[microkernel]
            key = _fp_key(row["params"], "vector")
            vector_values[(family, key)].append(
                row["energy_per_action_pj"] / _vector_scale(microkernel, row["params"])
            )
        elif row["component"] == "vector" and microkernel in LANE_ACCESS_FAMILY:
            key = _fp_key(row["params"], "vector")
            scalar_values[(LANE_ACCESS_FAMILY[microkernel], key)].append(
                row["energy_per_action_pj"] / int(row["params"]["VLEN"])
            )
        elif row["component"] == "scalar" and microkernel in SCALAR_FAMILY:
            family = SCALAR_FAMILY[microkernel]
            key = (
                str(int(row["params"]["INT_DATA_WIDTH"]))
                if family.startswith("integer")
                else _fp_key(row["params"], "scalar")
            )
            scalar_values[(family, key)].append(row["energy_per_action_pj"])

    def materialize(values: Mapping[tuple[str, str], list[float]]) -> dict[str, dict[str, float]]:
        tables: dict[str, dict[str, float]] = defaultdict(dict)
        for (family, key), samples in values.items():
            tables[family][key] = median(samples)
        for family, table in tables.items():
            table["default"] = median(table.values())
        return dict(tables)

    return materialize(vector_values), materialize(scalar_values)


def _lookup(table: Mapping[str, float], key: str) -> float:
    if key in table:
        return float(table[key])
    if key.isdigit():
        widths = sorted((int(name), float(value)) for name, value in table.items() if str(name).isdigit())
        if widths:
            target = int(key)
            return min(widths, key=lambda item: abs(item[0] - target))[1]
        return float(table.get("default", 0.0))
    parsed_anchors = [
        (int(match.group(1)), int(match.group(2)), float(value))
        for name, value in table.items()
        if (match := re.fullmatch(r"FP_E(\d+)M(\d+)", str(name)))
    ]
    target_match = re.fullmatch(r"FP_E(\d+)M(\d+)", key)
    if target_match and len(parsed_anchors) >= 2:
        # E5M6/E6M5 have the same total width but expose the very different
        # exponent-side and mantissa-side switching costs.  Use that physical
        # information before falling back to total-width interpolation.
        for left, right in combinations(parsed_anchors, 2):
            determinant = left[0] * right[1] - right[0] * left[1]
            if determinant == 0:
                continue
            exp_c = (left[2] * right[1] - right[2] * left[1]) / determinant
            mant_c = (left[0] * right[2] - right[0] * left[2]) / determinant
            if exp_c >= 0 and mant_c >= 0:
                return (
                    exp_c * int(target_match.group(1))
                    + mant_c * int(target_match.group(2))
                )
    anchors = sorted(
        (_fp_width(f"FP_E{exp}M{mant}"), value)
        for exp, mant, value in parsed_anchors
    )
    if not anchors:
        return float(table.get("default", 0.0))
    target = _fp_width(key)
    if target <= anchors[0][0]:
        return anchors[0][1] * target / anchors[0][0]
    if target >= anchors[-1][0]:
        return anchors[-1][1] * target / anchors[-1][0]
    for left, right in pairwise(anchors):
        if left[0] <= target <= right[0]:
            ratio = (target - left[0]) / (right[0] - left[0])
            return left[1] + ratio * (right[1] - left[1])
    return float(table.get("default", 0.0))


def _hierarchy_child_dynamic_mw(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    children: dict[str, float] = {}
    pattern = re.compile(
        r"^\s+(u_cvt|u_fp|u_int)\s+\([^)]*\)\s+"
        r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+",
        re.MULTILINE,
    )
    for match in pattern.finditer(path.read_text(errors="ignore")):
        children[match.group(1)] = float(match.group(2)) + float(match.group(3))
    return children


def _fit_leaf_coefficients(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_scenario = {
        row["scenario"]: row
        for row in rows
        if row["point_id"] == "power_matrix_leaf_bundle"
    }
    samples: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for active in by_scenario.values():
        if active["pattern"] != "representative-qwen" or active["microkernel"] == "mixed":
            continue
        idle = by_scenario.get(f"idle_{active['repeat_count']}")
        if idle is None:
            continue
        active_hierarchy = _hierarchy_child_dynamic_mw(
            Path(active["power_report"]).with_name("power.hierarchy.rpt")
        )
        idle_hierarchy = _hierarchy_child_dynamic_mw(
            Path(idle["power_report"]).with_name("power.hierarchy.rpt")
        )
        for child in ("u_int", "u_fp", "u_cvt"):
            if child in active_hierarchy and child in idle_hierarchy:
                incremental = (
                    active_hierarchy[child] - idle_hierarchy[child]
                ) * float(active["window_ns"])
                samples[child].append((active["accepted_actions"], incremental))
    if set(samples) != {"u_int", "u_fp", "u_cvt"}:
        raise ValueError("matrix leaf hierarchy replay does not cover int/fp reduce and conversion")
    slopes = {child: _linear_slope(values)[0] for child, values in samples.items()}
    return {
        "mxint_reduce_node_bit": slopes["u_int"] / (4 * 16),
        "mxfp_reduce_node_bit": slopes["u_fp"] / (4 * 16),
        "mxint_output_bit": slopes["u_cvt"] / 12,
        # No standalone MXFP conversion exists in this bundle.  The FP reduce
        # output register/normalizer is the closest measured leaf and remains
        # explicitly disclosed in provenance.
        "mxfp_output_bit": slopes["u_fp"] / (4 * 16),
    }


def _fit_hbm(rows: list[dict[str, Any]], *, include_holdout: bool) -> tuple[dict[str, float], dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["component"] == "hbm"
        and row["pattern"] == "representative-qwen"
        and row["microkernel"] in HBM_V2_MICROKERNELS
        and (include_holdout or not row["holdout"])
    ]
    if len(selected) < 3:
        raise ValueError("HBM fit needs all three production DMA microkernels")
    # The v2 dataset intentionally reuses only one fixed-amount configuration
    # per precision family. Three opcode observations cannot identify three
    # startup terms plus shared line and byte terms. The previous five-column
    # NNLS was therefore rank deficient and assigned arbitrary transfer cost.
    # Fit the observable quantity instead: energy per accepted logical lane
    # for each production DMA family. Physical-line/byte decomposition remains
    # explicitly unavailable until an amount sweep is collected.
    values: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        amount_key = {
            "matrix_prefetch": "HBM_M_Prefetch_Amount",
            "vector_prefetch": "HBM_V_Prefetch_Amount",
            "vector_writeback": "HBM_V_Writeback_Amount",
        }[row["microkernel"]]
        amount = max(1, int(row["params"][amount_key]))
        values[row["microkernel"]].append(row["energy_per_action_pj"] / amount)
    fitted = {family: median(values[family]) for family in HBM_V2_MICROKERNELS}
    fitted.update(
        {
            "line": 0.0,
            "byte": 0.0,
            "default": median(fitted.values()),
        }
    )
    return fitted, {
        "rank": len(HBM_V2_MICROKERNELS),
        "feature_count": len(HBM_V2_MICROKERNELS),
        "condition_number": 1.0,
        "zero_coefficients": [],
        "point_count": len(selected),
        "coefficient_semantics": "pj_per_accepted_logical_lane",
        "unidentified_features": ["physical_line", "useful_byte"],
    }


def _clock_component_key(row: Mapping[str, Any]) -> str:
    component = str(row["component"])
    if component == "hbm":
        return "hbm_controller"
    if component == "matrix" and not row["params"].get("leaf_bundle"):
        return f"matrix.{_matrix_mode(row['params'])}"
    return component


def _clock_models(
    rows: list[dict[str, Any]], *, include_holdout: bool
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Fit idle energy/cycle as mapped area density plus fixed clock glue.

    Total mapped area contains large combinational FP/INT datapaths that do not
    receive a clock. A density-only model therefore makes wider combinational
    units look as if they also added proportional clock load. The nonnegative
    intercept captures fixed controller/register-file clocking while retaining
    area scaling for replicated lane and array state.
    """

    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row["pattern"] != "idle" or row["point_key"] in seen:
            continue
        if row["params"].get("leaf_bundle"):
            continue
        if row["holdout"] and not include_holdout:
            continue
        if row["logic_area_um2"] <= 0 or row["window_cycles"] <= 0:
            continue
        seen.add(row["point_key"])
        samples[_clock_component_key(row)].append(
            (
                row["logic_area_um2"],
                row["window_energy_pj"] / row["window_cycles"],
            )
        )
    required = {
        "matrix.mxint", "matrix.mxfp", "vector", "scalar", "control",
        "hbm_controller",
    }
    if required - samples.keys():
        raise ValueError(
            f"missing idle component clock models: {sorted(required - samples.keys())}"
        )
    density: dict[str, float] = {}
    fixed: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for component, points in sorted(samples.items()):
        if len(points) == 1:
            area, energy = points[0]
            density[component] = energy / area
            fixed[component] = 0.0
            diagnostics[component] = {
                "point_count": 1,
                "rank": 1,
                "feature_count": 1,
                "condition_number": 1.0,
                "zero_coefficients": [1],
            }
            continue
        matrix = np.asarray([[area, 1.0] for area, _ in points], dtype=float)
        target = np.asarray([energy for _, energy in points], dtype=float)
        coefficients, diagnostic = _nnls_ridge(matrix, target, ridge=1e-8)
        density[component] = float(coefficients[0])
        fixed[component] = float(coefficients[1])
        diagnostics[component] = {**diagnostic, "point_count": len(points)}
    return density, fixed, diagnostics


def _activity_envelope(slopes: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    indexed = {
        (row["point_key"], row["microkernel"], row["pattern"]): row
        for row in slopes
    }
    ratios: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    diagnostics: list[dict[str, Any]] = []
    for row in slopes:
        if row["pattern"] != "representative-qwen" or row["sample_count"] < 3:
            continue
        low = indexed.get((row["point_key"], row["microkernel"], "low-toggle"))
        high = indexed.get((row["point_key"], row["microkernel"], "random"))
        if low is None or high is None or row["energy_per_action_pj"] <= 0:
            continue
        component = row["component"]
        microkernel = row["microkernel"]
        if component == "matrix":
            key = f"matrix.array_compute.{_matrix_mode(row['params'])}"
        elif component == "vector" and microkernel in VECTOR_FAMILY:
            key = f"vector.{VECTOR_FAMILY[microkernel]}"
        elif component == "vector":
            key = f"scalar.{LANE_ACCESS_FAMILY[microkernel]}"
        elif component == "scalar":
            key = f"scalar.{SCALAR_FAMILY[microkernel]}"
        elif component == "hbm":
            key = f"hbm_controller.{microkernel}"
        else:
            key = "control.frontend_issue"
        low_raw = low["energy_per_action_pj"] / row["energy_per_action_pj"]
        high_raw = high["energy_per_action_pj"] / row["energy_per_action_pj"]
        ratios[key]["low"].append(min(1.0, low_raw, high_raw))
        ratios[key]["high"].append(max(1.0, low_raw, high_raw))
        diagnostics.append(
            {"family": key, "point_id": row["point_id"], "low_raw": low_raw, "high_raw": high_raw}
        )
    envelope = {
        key: {"low": median(values["low"]), "nominal": 1.0, "high": median(values["high"])}
        for key, values in ratios.items()
    }
    if any(row["component"] == "hbm" for row in slopes):
        hbm_low = [value["low"] for key, value in envelope.items() if key.startswith("hbm_controller.")]
        hbm_high = [value["high"] for key, value in envelope.items() if key.startswith("hbm_controller.")]
        if hbm_low and hbm_high:
            envelope["hbm_controller.physical_transfer"] = {
                "low": median(hbm_low), "nominal": 1.0, "high": median(hbm_high)
            }
    return envelope, diagnostics


def _build_model(rows: list[dict[str, Any]], slopes: list[dict[str, Any]], *, include_holdout: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_pe, matrix_fit = _fit_matrix_pe(slopes, include_holdout=include_holdout)
    vector, scalar = _median_tables(slopes, include_holdout=include_holdout)
    leaf = _fit_leaf_coefficients(rows)
    hbm, hbm_fit = _fit_hbm(slopes, include_holdout=include_holdout)
    clocks, clock_fixed, clock_fit = _clock_models(
        rows, include_holdout=include_holdout
    )
    envelope, envelope_diag = _activity_envelope(slopes)
    leakage = [
        row["leakage_power_mw"] / row["logic_area_um2"]
        for row in rows
        if row["logic_area_um2"] > 0 and (include_holdout or not row["holdout"])
    ]
    model = {
        "model": "onchip_action_energy_v2",
        "calibration_status": "rtl_activity_candidate_pending_validation_v2",
        "gate_level_validation": "not_run_by_scope",
        "corner": CORNER,
        "dynamic_nominal_pj": {
            "matrix": {
                "mxint": {
                    "pe_cycle": matrix_pe["mxint"],
                    "reduce_node_bit": leaf["mxint_reduce_node_bit"],
                    "output_bit": leaf["mxint_output_bit"],
                },
                "mxfp": {
                    "pe_cycle": matrix_pe["mxfp"],
                    "reduce_node_bit": leaf["mxfp_reduce_node_bit"],
                    "output_bit": leaf["mxfp_output_bit"],
                },
            },
            "vector": vector,
            "scalar": scalar,
            "control": {"frontend_issue": median(
                row["energy_per_action_pj"]
                for row in slopes
                if row["component"] == "control" and row["pattern"] == "representative-qwen"
            )},
            "hbm_controller": hbm,
        },
        "activity_envelope": envelope,
        "clock_pj_per_cycle_um2": clocks,
        "clock_fixed_pj_per_cycle": clock_fixed,
        "logic_leakage_mw_per_um2": median(leakage),
        "grouped_holdout_residual": {"p90_relative": 0.0},
        "calibration_domain": {
            "fp_formats": ["FP_E5M6", "FP_E6M5", "FP_E8M5"],
            "matrix_block_dim": [2, 4, 8, 16],
            "vector_vlen": [16, 32, 64],
            "activity_semantics": "qwen_like_p50_with_low_random_empirical_envelope",
        },
        "power_scope": "mapped_logic_rtl_activity_no_cts_plus_macro_sram_dynamic",
        "provenance": {
            "nominal": (
                "Qwen-like per-microkernel non-clock dynamic minus matched-idle "
                "non-clock slopes"
            ),
            "activity_envelope": "per-family low-toggle and random slope ratios",
            "clock": (
                "mapped component area times a nonnegative idle density plus a "
                "nonnegative fixed peripheral clock term; Matrix MXINT/MXFP are "
                "fit separately"
            ),
            "dc_clock_group_normalization": (
                "activity-dependent pre-CTS clock_network changes are excluded; "
                "normalized active target equals matched idle total plus active "
                "non-clock minus idle non-clock"
            ),
            "matrix_leaf": "hierarchical u_int/u_fp/u_cvt power from matrix leaf bundle",
            "vector_lane_access": "measured in VectorMachine because S_LD/ST_VLANE datapaths reside there",
            "mxfp_output": "uses measured FP reduction output normalization as the nearest available leaf proxy",
            "matrix_array": (
                "MXINT uses nonnegative fixed launch + B feed + B^3 PE-MAC "
                "terms; MXFP retains B^3 PE scaling because all available "
                "MXFP anchors use B=4"
            ),
            "vector_reduction": (
                "full/single-segment routing scales with VLEN*log2(VLEN); "
                "compact multi-segment activity scales with VLEN"
            ),
            "hbm_controller": (
                "per-opcode energy per accepted logical lane; fixed-amount "
                "data cannot identify separate physical-line/byte terms"
            ),
        },
    }
    return model, {
        "matrix_fit": matrix_fit,
        "hbm_fit": hbm_fit,
        "clock_fit": clock_fit,
        "activity_envelope_points": envelope_diag,
    }


def _predict_slope(row: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    dynamic = model["dynamic_nominal_pj"]
    component = row["component"]
    microkernel = row["microkernel"]
    params = row["params"]
    if component == "matrix" and microkernel == "array_compute":
        mode = _matrix_mode(params)
        t_width, l_width = _matrix_widths(params)
        pe = dynamic["matrix"][mode]["pe_cycle"]
        per_pe = pe["base"] + pe["bit_product"] * t_width * l_width + pe["width_sum"] * (t_width + l_width)
        block = int(params["BLOCK_DIM"])
        return (
            pe.get("slice_fixed", 0.0)
            + block * pe.get("feed_cycle", 0.0)
            + block**3 * per_pe
        )
    if component == "vector" and microkernel in VECTOR_FAMILY:
        family = VECTOR_FAMILY[microkernel]
        return _vector_scale(microkernel, params) * _lookup(
            dynamic["vector"][family], _fp_key(params, "vector")
        )
    if component == "vector" and microkernel in LANE_ACCESS_FAMILY:
        return int(params["VLEN"]) * _lookup(
            dynamic["scalar"][LANE_ACCESS_FAMILY[microkernel]],
            _fp_key(params, "vector"),
        )
    if component == "scalar" and microkernel in SCALAR_FAMILY:
        family = SCALAR_FAMILY[microkernel]
        key = str(int(params["INT_DATA_WIDTH"])) if family.startswith("integer") else _fp_key(params, "scalar")
        return _lookup(dynamic["scalar"][family], key)
    if component == "control":
        return float(dynamic["control"]["frontend_issue"])
    if component == "hbm":
        coefficients = dynamic["hbm_controller"]
        amount_key = {
            "matrix_prefetch": "HBM_M_Prefetch_Amount",
            "vector_prefetch": "HBM_V_Prefetch_Amount",
            "vector_writeback": "HBM_V_Writeback_Amount",
        }[microkernel]
        return int(params[amount_key]) * float(coefficients[microkernel])
    return 0.0


def _prediction_errors(
    slopes: list[dict[str, Any]], model: Mapping[str, Any], *, holdout_only: bool
) -> tuple[list[float], list[dict[str, Any]]]:
    errors: list[float] = []
    details: list[dict[str, Any]] = []
    for row in slopes:
        if row["pattern"] != "representative-qwen" or row["microkernel"] == "leaf_bundle":
            continue
        if holdout_only and not row["holdout"]:
            continue
        prediction = _predict_slope(row, model)
        target = row["energy_per_action_pj"]
        error = abs(prediction - target) / max(target, 1e-12)
        errors.append(error)
        details.append(
            {
                "point_id": row["point_id"], "component": row["component"],
                "microkernel": row["microkernel"], "target_pj_per_action": target,
                "prediction_pj_per_action": prediction, "absolute_percentage_error": error * 100,
            }
        )
    return errors, details


def _clock_errors(rows: list[dict[str, Any]], model: Mapping[str, Any]) -> list[float]:
    result: list[float] = []
    seen: set[str] = set()
    for row in rows:
        if not row["holdout"] or row["pattern"] != "idle" or row["point_key"] in seen:
            continue
        # ``matrix_leaf_bundle`` is a synthetic mapping convenience used to
        # isolate reduction/conversion hierarchy slopes.  It is not a runtime
        # component and has no corresponding area_new clock model, so treating
        # its aggregate idle power as a MatrixMachine holdout is meaningless.
        if row["params"].get("leaf_bundle"):
            continue
        seen.add(row["point_key"])
        component = _clock_component_key(row)
        prediction = (
            row["window_cycles"] * row["logic_area_um2"]
            * model["clock_pj_per_cycle_um2"][component]
            + row["window_cycles"]
            * model["clock_fixed_pj_per_cycle"].get(component, 0.0)
        )
        result.append(abs(prediction - row["window_energy_pj"]) / max(row["window_energy_pj"], 1e-12))
    return result


def _predict_qwen_mix_incremental(row: Mapping[str, Any], model: Mapping[str, Any]) -> float | None:
    """Predict the harness mixed-Qwen window from measured family slopes."""

    counts = dict(row.get("mix_action_counts") or {})
    if sum(counts.values()) != int(row["accepted_actions"]):
        return None
    component = row["component"]
    params = row["params"]
    if component == "matrix":
        if params.get("leaf_bundle"):
            return None
        return sum(
            instances * _predict_slope({**row, "microkernel": microkernel}, model)
            for microkernel, instances in counts.items()
        )
    if component == "vector":
        return sum(
            instances * _predict_slope({**row, "microkernel": microkernel}, model)
            for microkernel, instances in counts.items()
        )
    if component == "scalar":
        return sum(
            instances * _predict_slope({**row, "microkernel": microkernel}, model)
            for microkernel, instances in counts.items()
        )
    if component == "control":
        return sum(counts.values()) * float(model["dynamic_nominal_pj"]["control"]["frontend_issue"])
    if component == "hbm":
        incremental = sum(
            instances
            * _predict_slope({**row, "microkernel": microkernel}, model)
            for microkernel, instances in counts.items()
        )
        return incremental
    return None


def _qwen_mix_errors(
    rows: list[dict[str, Any]], model: Mapping[str, Any]
) -> tuple[list[float], list[dict[str, Any]]]:
    errors: list[float] = []
    details: list[dict[str, Any]] = []
    for row in rows:
        if row["pattern"] != "representative-qwen" or row["microkernel"] != "mixed":
            continue
        incremental = _predict_qwen_mix_incremental(row, model)
        if incremental is None:
            continue
        component = _clock_component_key(row)
        idle = (
            row["window_cycles"]
            * row["logic_area_um2"]
            * float(model["clock_pj_per_cycle_um2"][component])
            + row["window_cycles"]
            * float(model["clock_fixed_pj_per_cycle"].get(component, 0.0))
        )
        prediction = incremental + idle
        target = row["normalized_dynamic_energy_pj"]
        error = abs(prediction - target) / max(target, 1e-12)
        errors.append(error)
        details.append(
            {
                "point_id": row["point_id"],
                "component": row["component"],
                "target_normalized_dynamic_pj": target,
                "raw_dc_total_dynamic_pj": row["window_energy_pj"],
                "excluded_clock_delta_pj": row["excluded_clock_delta_pj"],
                "prediction_normalized_dynamic_pj": prediction,
                "absolute_percentage_error": error * 100,
            }
        )
    return errors, details


def _write_markdown(validation: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# RTL-Activity Power Model v2 Validation",
        "",
        f"**Promotion status:** {'PASS' if validation['accepted'] else 'FAIL'}",
        "",
        "Qwen-like microkernel slopes define nominal P50. Low-toggle and random "
        "slopes define an empirical activity envelope; they are not training "
        "samples for the nominal coefficients.",
        "Pre-CTS DC clock-network variation is retained for audit but excluded "
        "from action slopes. Validation uses matched-idle total plus the measured "
        "non-clock active residual.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Qwen component holdout median APE | {validation['holdout_median_ape']:.3f}% |",
        f"| Qwen component holdout P95 APE | {validation['holdout_p95_ape']:.3f}% |",
        f"| Idle clock holdout median APE | {validation['clock_holdout_median_ape']:.3f}% |",
        f"| Idle clock holdout P95 APE | {validation['clock_holdout_p95_ape']:.3f}% |",
        f"| Minimum action slope R² | {validation['minimum_action_slope_r2']:.6f} |",
        f"| Cached power evaluation median | {validation['cached_power_evaluation_median_ms']:.3f} ms |",
        f"| Non-positive non-clock residual rows | {len(validation['nonpositive_nonclock_rows'])} |",
        "",
        "## Acceptance Gates",
        "",
    ]
    for name, passed in validation["acceptance"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "The result is RTL VCD activity replayed on mapped DC netlists. It "
            "does not include gate-level timing activity, CTS, routed parasitics, "
            "external HBM/PHY/package power, KV links, or SRAM leakage.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def _numeric_leaves(value: Any) -> Iterable[float]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, (int, float)):
        yield float(value)


def _matrix_invariants(model: Mapping[str, Any]) -> dict[str, bool]:
    matrix = model["dynamic_nominal_pj"]["matrix"]
    monotonic = True
    block_monotonic = True
    for mode in ("mxint", "mxfp"):
        pe = matrix[mode]["pe_cycle"]
        previous = -math.inf
        for width in range(2, 17):
            energy = pe["base"] + pe["bit_product"] * width * width + pe["width_sum"] * 2 * width
            monotonic &= energy >= previous
            previous = energy
        previous = -math.inf
        for block in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            per_pe = (
                pe["base"]
                + pe["bit_product"] * 16
                + pe["width_sum"] * 8
            )
            energy = (
                pe.get("slice_fixed", 0.0)
                + block * pe.get("feed_cycle", 0.0)
                + block**3 * per_pe
            )
            block_monotonic &= energy >= previous
            previous = energy
    return {
        "single_k_split_reduce_energy_is_zero_by_formula": True,
        "matrix_precision_energy_monotonic": monotonic,
        "matrix_block_energy_monotonic": block_monotonic,
        "matrix_structural_coefficients_nonnegative": all(
            value >= 0 and math.isfinite(value) for value in _numeric_leaves(matrix)
        ),
    }


def _cached_evaluation_benchmark_ms(artifact: Path, repeats: int = 50) -> float:
    """Measure the actual cached estimator path on a representative trace."""

    from analytic_models.power import estimate_onchip_power

    config = {
        "MLEN": 512,
        "VLEN": 512,
        "BLEN": 64,
        "HLEN": 128,
        "ACT_WIDTH": "MXFP_E4M3",
        "KV_WIDTH": "MXFP_E4M3",
        "WEIGHT_WIDTH": "MXFP_E4M3",
        "FP_SETTING": "FP_E5M6",
        "MX_SCALE_WIDTH": 8,
        "INT_DATA_WIDTH": 32,
        "MATRIX_SRAM_DEPTH": 1024,
        "VECTOR_SRAM_DEPTH": 1024,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 1600,
        "HBM_M_Prefetch_Amount": 512,
        "HBM_V_Prefetch_Amount": 64,
        "HBM_V_Writeback_Amount": 64,
        "CLOCK_PERIOD_PS": 1000,
        "SEQ_LEN": 482,
        "BATCH_SIZE": 16,
    }
    trace = {
        "schema_version": 4,
        "energy_actions": [
            {"stage": "layer/matrix", "component": "matrix", "action": "array_compute", "count": 1024},
            {"stage": "layer/matrix", "component": "matrix", "action": "cross_k_reduce", "count": 1024},
            {"stage": "layer/vector", "component": "vector", "action": "lane_multiply_vv", "count": 4096},
            {"stage": "layer/vector", "component": "vector", "action": "reduction_sum_segment", "count": 4096, "segment_log2": 7},
            {"stage": "layer/scalar", "component": "scalar", "action": "fp_sfu_reciprocal", "count": 4096},
            {"stage": "layer/control", "component": "control", "action": "frontend_issue", "count": 32768},
            {"stage": "layer/hbm", "component": "hbm_controller", "action": "dma_prefetch_vector", "count": 128, "bytes": 65536},
        ],
    }
    timing = {
        "compute_pipeline_makespan_cycles": 10_000_000,
        "hbm_read_bytes": 1_048_576,
        "hbm_write_bytes": 262_144,
        "hbm_read_requests": 16_384,
        "hbm_write_requests": 4_096,
    }
    for _ in range(3):
        estimate_onchip_power(
            config, trace, timing, logic_coefficients_path=artifact
        )
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        estimate_onchip_power(
            config, trace, timing, logic_coefficients_path=artifact
        )
        samples.append((time.perf_counter() - started) * 1_000.0)
    return median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--envelope-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    rows = _latest_rows(args.points)
    _pair_idle(rows)
    slopes = _extract_slopes(rows)
    training_model, fit_diagnostics = _build_model(rows, slopes, include_holdout=False)
    final_model, final_fit_diagnostics = _build_model(rows, slopes, include_holdout=True)
    holdout_errors, holdout_details = _prediction_errors(
        slopes, training_model, holdout_only=True
    )
    component_holdout: dict[str, dict[str, float]] = {}
    by_component: dict[str, list[float]] = defaultdict(list)
    for detail in holdout_details:
        by_component[detail["component"]].append(
            detail["absolute_percentage_error"] / 100.0
        )
    for component, errors in sorted(by_component.items()):
        component_holdout[component] = {
            "count": len(errors),
            "median_ape": median(errors) * 100,
            "p95_ape": _percentile(errors, 95) * 100,
            "max_ape": max(errors) * 100,
        }
    clock_errors = _clock_errors(rows, training_model)
    slope_r2 = {
        f"{row['point_id']}/{row['microkernel']}": row["r2"]
        for row in slopes
        if row["pattern"] == "representative-qwen" and row["sample_count"] >= 3
    }
    required_vector = set(VECTOR_FAMILY.values())
    required_scalar = set(SCALAR_FAMILY.values()) | set(LANE_ACCESS_FAMILY.values())
    missing = {
        "vector": sorted(required_vector - final_model["dynamic_nominal_pj"]["vector"].keys()),
        "scalar": sorted(required_scalar - final_model["dynamic_nominal_pj"]["scalar"].keys()),
    }
    qwen_mix_errors, qwen_mix_details = _qwen_mix_errors(rows, training_model)
    active_rows = [row for row in rows if row["pattern"] != "idle"]
    nonpositive_nonclock_rows = [
        {
            "point_id": row["point_id"],
            "scenario": row["scenario"],
            "nonclock_residual_pj": row["incremental_energy_pj"],
        }
        for row in active_rows
        if not row["nonclock_residual_positive"]
    ]
    excluded_clock_deltas = [row["excluded_clock_delta_pj"] for row in active_rows]
    expected_mix_rows = [
        row for row in rows
        if row["pattern"] == "representative-qwen"
        and row["microkernel"] == "mixed"
        and not row["params"].get("leaf_bundle")
    ]
    matrix_invariants = _matrix_invariants(final_model)
    acceptance = {
        "each_component_qwen_holdout_median_le_15pct": bool(component_holdout) and all(
            metrics["median_ape"] <= 15.0 for metrics in component_holdout.values()
        ),
        "each_component_qwen_holdout_p95_le_30pct": bool(component_holdout) and all(
            metrics["p95_ape"] <= 30.0 for metrics in component_holdout.values()
        ),
        "idle_clock_holdout_median_le_15pct": bool(clock_errors) and median(clock_errors) <= 0.15,
        "idle_clock_holdout_p95_le_25pct": bool(clock_errors) and _percentile(clock_errors, 95) <= 0.25,
        "all_action_family_slope_r2_ge_0p95": bool(slope_r2) and min(slope_r2.values()) >= 0.95,
        "no_missing_action_family": not any(missing.values()),
        "all_active_nonclock_residuals_positive": not nonpositive_nonclock_rows,
        "no_negative_or_nonfinite_coefficient": all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in _numeric_leaves(final_model["dynamic_nominal_pj"])
        ) and all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in final_model["clock_pj_per_cycle_um2"].values()
        ) and all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in final_model["clock_fixed_pj_per_cycle"].values()
        ),
        "qwen_mix_uses_costtrace_counts": bool(expected_mix_rows) and len(qwen_mix_errors) == len(expected_mix_rows),
        "qwen_mix_total_error_le_20pct": bool(qwen_mix_errors) and max(qwen_mix_errors) <= 0.20,
        "matrix_structural_invariants": all(matrix_invariants.values()),
    }
    residual_p90 = _percentile(holdout_errors, 90) if holdout_errors else 0.0
    final_model["grouped_holdout_residual"] = {"p90_relative": residual_p90}
    final_model["source_points"] = str(args.points.resolve())
    final_model["calibration_status"] = "rtl_activity_candidate_pending_validation_v2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_model, indent=2, sort_keys=True) + "\n")
    cached_evaluation_ms = _cached_evaluation_benchmark_ms(args.output)
    acceptance["cached_power_evaluation_lt_10ms"] = cached_evaluation_ms < 10.0
    accepted = all(acceptance.values())
    final_model["calibration_status"] = (
        "rtl_activity_calibrated_candidate_v2"
        if accepted
        else "rtl_activity_candidate_failed_validation_v2"
    )
    validation = {
        "model": "onchip_action_energy_v2",
        "accepted": accepted,
        "calibration_status": final_model["calibration_status"],
        "acceptance": acceptance,
        "missing_action_families": missing,
        "holdout_median_ape": _percentile(holdout_errors, 50) * 100,
        "holdout_p95_ape": _percentile(holdout_errors, 95) * 100,
        "clock_holdout_median_ape": _percentile(clock_errors, 50) * 100,
        "clock_holdout_p95_ape": _percentile(clock_errors, 95) * 100,
        "minimum_action_slope_r2": min(slope_r2.values()) if slope_r2 else -math.inf,
        "cached_power_evaluation_median_ms": cached_evaluation_ms,
        "action_slope_r2": slope_r2,
        "qwen_mix_errors": qwen_mix_errors,
        "qwen_mix_points": qwen_mix_details,
        "nonpositive_nonclock_rows": nonpositive_nonclock_rows,
        "excluded_clock_delta_pj": {
            "median": _percentile(excluded_clock_deltas, 50),
            "p05": _percentile(excluded_clock_deltas, 5),
            "p95": _percentile(excluded_clock_deltas, 95),
            "minimum": min(excluded_clock_deltas, default=0.0),
            "maximum": max(excluded_clock_deltas, default=0.0),
            "semantics": "audited_only_not_in_action_fit",
        },
        "matrix_invariants": matrix_invariants,
        "holdout_points": holdout_details,
        "component_holdout": component_holdout,
        "training_fit_diagnostics": fit_diagnostics,
        "final_fit_diagnostics": final_fit_diagnostics,
        "gate_level_validation": "not_run_by_scope",
        "promotion_note": "Coefficient promotion is followed by non-objective MXINT/MXFP DSE smoke",
    }

    args.output.write_text(json.dumps(final_model, indent=2, sort_keys=True) + "\n")
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    envelope_rows = []
    for family, values in sorted(final_model["activity_envelope"].items()):
        envelope_rows.append({"family": family, **values})
    args.envelope_output.parent.mkdir(parents=True, exist_ok=True)
    with args.envelope_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("family", "low", "nominal", "high"))
        writer.writeheader()
        writer.writerows(envelope_rows)
    _write_markdown(validation, args.markdown_output)

    if args.promote:
        if not accepted:
            raise RuntimeError("v2 candidate failed validation; default artifact was not changed")
        destination = REPO_ROOT / "analytic_models/power/calibration/logic_energy_v2.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(final_model, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    print(json.dumps({"accepted": accepted, "slopes": len(slopes), "holdout_rows": len(holdout_errors)}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
