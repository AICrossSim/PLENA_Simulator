"""Physically compositional MatrixMachine area model.

The model mirrors the generate-time census in the PLENA RTL instead of
regressing total area against interchangeable shape polynomials.  A logical
MatrixMachine contains ``MLEN / BLEN`` systolic slices.  Each slice owns a
``BLEN x BLEN`` PE grid, while cross-K reduction only exists between slices.

MXINT and MXFP deliberately use different output-buffer terms because the
current RTL implementations differ: MXINT drains a BLEN-squared FP accumulator
bank into an MLEN-wide result register, whereas MXFP assembles BLEN-by-MLEN
blocks before draining them.  All returned areas are square micrometres.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_VERSION = "matrix_machine_structural_census_v4"
DEFAULT_ARTIFACT = Path(__file__).with_name("calibration") / "matrix_structural_v4_coefficients.json"


def load_artifact(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the structural artifact, returning ``None`` for legacy installs."""

    selected = path or os.environ.get("PLENA_AREA_NEW_MATRIX_STRUCTURAL_COEFFICIENTS")
    artifact_path = Path(selected) if selected else DEFAULT_ARTIFACT
    if not artifact_path.exists():
        return None
    with artifact_path.open() as handle:
        artifact = json.load(handle)
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Unsupported MatrixMachine structural artifact version: "
            f"{artifact.get('model_version')!r}"
        )
    return artifact


def _fp_width(inputs: Mapping[str, Any]) -> int:
    if "FP_EXP_WIDTH" in inputs and "FP_MANT_WIDTH" in inputs:
        return 1 + int(inputs["FP_EXP_WIDTH"]) + int(inputs["FP_MANT_WIDTH"])
    token = str(inputs.get("FP_SETTING", "FP_E5M6")).upper().replace("FP_", "")
    if token.startswith("E") and "M" in token:
        exp, mant = token[1:].split("M", 1)
        return 1 + int(exp) + int(mant)
    return 1 + int(inputs.get("S_FP_EXP_WIDTH", 5)) + int(inputs.get("S_FP_MANT_WIDTH", 6))


def _precision_key(mode: str, inputs: Mapping[str, Any]) -> str:
    if mode == "mxint":
        return f"t{int(inputs['t_bits'])}_l{int(inputs['l_bits'])}"
    return (
        f"te{int(inputs['t_exp'])}m{int(inputs['t_mant'])}_"
        f"le{int(inputs['l_exp'])}m{int(inputs['l_mant'])}"
    )


def structural_counts(mlen: int, blen: int, fp_width: int) -> dict[str, int]:
    """Return exact RTL replication counts for one legal shape."""

    if mlen <= 0 or blen <= 0:
        raise ValueError(f"MLEN and BLEN must be positive, got {mlen}, {blen}")
    if mlen % blen:
        raise ValueError(f"MLEN must be divisible by BLEN, got {mlen}, {blen}")
    splits = mlen // blen
    return {
        "array_count": splits,
        "pe_count": mlen * blen,
        "reduce_node_count": blen * blen * max(splits - 1, 0),
        "output_cell_count": blen * blen,
        "result_buffer_bits": mlen * blen * fp_width,
        "mxint_result_register_bits": mlen * fp_width,
        "mxfp_block_buffer_bits": mlen * blen * fp_width,
    }


def _dot(coefficients: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(float(coefficients.get(name, 0.0)) * value for name, value in features.items())


def _evaluate_family(
    mode: str,
    inputs: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    mlen = int(inputs["MLEN"])
    blen = int(inputs["BLEN"])
    scale_width = int(inputs.get("scale_width", inputs.get("MX_SCALE_WIDTH", 8)))
    fp_width = _fp_width(inputs)
    counts = structural_counts(mlen, blen, fp_width)
    splits = counts["array_count"]
    log_splits = math.log2(splits) if splits > 1 else 0.0

    if mode == "mxint":
        t_width = int(inputs["t_bits"])
        l_width = int(inputs["l_bits"])
        accumulator_width = (
            t_width
            + l_width
            + max(1, math.ceil(math.log2(blen)))
            + int(model.get("mxint_max_shift", 16))
            + max(1, math.ceil(math.log2(splits + 1)))
        )
    else:
        t_width = 1 + int(inputs["t_exp"]) + int(inputs["t_mant"])
        l_width = 1 + int(inputs["l_exp"]) + int(inputs["l_mant"])
        accumulator_width = int(inputs.get("ACC_FP_WIDTH", model.get("acc_fp_width", 16)))

    key = _precision_key(mode, inputs)
    pe_lookup = model.get("pe_lookup", {})
    if key not in pe_lookup:
        raise ValueError(f"No {mode} PE calibration for precision capability {key}")
    pe_area = float(pe_lookup[key])
    if mode == "mxint":
        reference_depth = int(model.get("pe_depth_reference", 16))
        depth_delta = math.log2(blen) - math.log2(reference_depth)
        depth_slopes = model.get("pe_depth_slope", 0.0)
        if isinstance(depth_slopes, Mapping):
            depth_slope = float(depth_slopes.get(key, depth_slopes.get("default", 0.0)))
        else:
            depth_slope = float(depth_slopes)
        pe_area += depth_slope * depth_delta
        pe_area = max(float(model.get("pe_area_floor", 0.0)), pe_area)

    array_features = {
        "pe_grid": counts["pe_count"] * pe_area,
        "streamer_operand_bits": counts["pe_count"] * (t_width + l_width),
        "scale_distribution_bits": mlen * scale_width,
        "array_fixed": float(splits),
    }
    array_stack = _dot(model["array"], array_features)

    reduce_width = accumulator_width if mode == "mxint" else int(model.get("acc_fp_width", 16))
    active_reduce_outputs = blen * blen if splits > 1 else 0
    reduce_features = {
        "edge_width": counts["reduce_node_count"] * reduce_width,
        "tree_depth_width": active_reduce_outputs * log_splits * reduce_width,
        "scale_edges": counts["reduce_node_count"] * scale_width,
        "active_outputs": float(active_reduce_outputs),
    }
    reduce_tree = _dot(model["reduce"], reduce_features)

    output_accumulator_features = {
        "cell_fp_width": counts["output_cell_count"] * fp_width,
        "cells": float(counts["output_cell_count"]),
    }
    output_accumulator = _dot(model["output_accumulator"], output_accumulator_features)

    output_conversion_features = {
        "cell_acc_width": counts["output_cell_count"] * accumulator_width,
        "cell_fp_width": counts["output_cell_count"] * fp_width,
        "cells": float(counts["output_cell_count"]),
    }
    output_conversion = _dot(model["output_conversion"], output_conversion_features)

    if mode == "mxint":
        result_features = {
            "matrix_fp_bits": 0.0,
            "tile_fp_bits": 0.0,
            "mlen_fp_bits": float(counts["mxint_result_register_bits"]),
        }
    else:
        result_features = {
            "matrix_fp_bits": float(counts["mxfp_block_buffer_bits"]),
            "tile_fp_bits": float(counts["output_cell_count"] * fp_width),
            "mlen_fp_bits": float(mlen * fp_width),
        }
    result_buffer = _dot(model["result_buffer"], result_features)

    io_features = {
        "lane_bits": mlen * (t_width + l_width + 2 * scale_width + fp_width),
        "lanes": float(mlen),
        "fixed": 1.0,
    }
    io_pipeline = _dot(model["io_pipeline"], io_features)

    control_features = {
        "shape_log": 1.0 + math.log2(mlen) + math.log2(blen) + log_splits,
        "mlen": float(mlen),
        "fixed": 1.0,
    }
    control = _dot(model["control"], control_features)

    breakdown = {
        "array_stack_area": max(0.0, array_stack),
        "reduce_tree_area": max(0.0, reduce_tree),
        "output_accumulator_area": max(0.0, output_accumulator),
        "output_conversion_area": max(0.0, output_conversion),
        "result_buffer_area": max(0.0, result_buffer),
        "io_pipeline_area": max(0.0, io_pipeline),
        "control_area": max(0.0, control),
    }
    breakdown["matrix_machine_area"] = sum(breakdown.values())
    return {
        "area": breakdown["matrix_machine_area"],
        "breakdown": breakdown,
        "structural_counts": counts,
        "pe_area": pe_area,
        "precision_key": key,
        "accumulator_width": accumulator_width,
        "feature_values": {
            "array": array_features,
            "reduce": reduce_features,
            "output_accumulator": output_accumulator_features,
            "output_conversion": output_conversion_features,
            "result_buffer": result_features,
            "io_pipeline": io_features,
            "control": control_features,
        },
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate(inputs: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate nominal area and grouped-shape holdout uncertainty.

    The interval combines two complementary signals.  Refitting coefficients
    after withholding each ``(MLEN, BLEN)`` group captures coefficient
    sensitivity, while the held-out actual/predicted ratios retain residual
    model error that a coefficient ensemble alone can underestimate far from
    the calibration domain.  The interval is a regression uncertainty proxy,
    not a process-variation or physical-design signoff interval.
    """

    mode = str(inputs["mode"])
    family = artifact["families"][mode]
    nominal = _evaluate_family(mode, inputs, family["coefficients"])
    ensemble_areas = [nominal["area"]]
    for candidate in family.get("ensemble", []):
        ensemble_areas.append(_evaluate_family(mode, inputs, candidate)["area"])
    uncertainty_scale = family.get("uncertainty_scale", {})
    residual_p10 = nominal["area"] * float(uncertainty_scale.get("p10", 1.0))
    residual_p90 = nominal["area"] * float(uncertainty_scale.get("p90", 1.0))
    p10 = min(nominal["area"], _quantile(ensemble_areas, 0.10), residual_p10)
    p90 = max(nominal["area"], _quantile(ensemble_areas, 0.90), residual_p90)
    return {
        "area": nominal["area"],
        "area_proxy": nominal["area"],
        "area_model": MODEL_VERSION,
        "breakdown": nominal["breakdown"],
        **nominal["breakdown"],
        "structural_counts": nominal["structural_counts"],
        "features": nominal["feature_values"],
        "inputs": dict(inputs),
        "precision_key": nominal["precision_key"],
        "pe_area": nominal["pe_area"],
        "accumulator_width": nominal["accumulator_width"],
        "area_uncertainty_p10": p10,
        "area_uncertainty_p50": nominal["area"],
        "area_uncertainty_p90": p90,
        "uncertainty_basis": artifact.get(
            "uncertainty_basis",
            "grouped_shape_holdout_coefficient_ensemble_and_residual_scale",
        ),
        "coefficient_artifact": artifact.get("source", str(DEFAULT_ARTIFACT)),
    }
