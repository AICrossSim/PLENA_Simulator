"""Compose calibrated PLENA logic and SRAM area models.

The public :func:`estimate_area` path combines:

1. precision-aware MatrixMachine logic;
2. fitted VectorMachine, ScalarMachine, and HBM interface logic;
3. a full-chip integration residual fitted only against logic/wrapper area;
4. ASAP7 SRAM macro tilings added after the residual.

Every area value is in square micrometres (um^2). Callers must divide by
``1e6`` to report square millimetres. The proxy is suitable for DSE ranking and
early die-size comparisons, not place-and-route signoff.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import (
    hbm_model,
    matrix_structural_model,
    mxint_model,
    mxfp_model,
    scalar_model,
    top_model,
    vector_model,
)
from .precision import derive_compute_sides, parse_precision
from .sram_model import estimate_sram_area

CALIBRATION_DIR = Path(__file__).with_name("calibration")


def _matrix_calibration_assessment(
    *,
    mode: str,
    mlen: int,
    blen: int,
) -> dict[str, Any]:
    """Describe whether a MatrixMachine shape is covered by DC anchors.

    Both families were synthesized over MLEN 16--64, BLEN 4--16, and at
    least two flattened K-splits.  In particular, no ``BLEN == MLEN`` point
    was used for regression.  Reporting that fact is preferable to silently
    comparing two independently extrapolated family equations.
    """

    max_k_splits = 16 if mode == "mxint" else 8
    k_splits = mlen / blen
    domain = {
        "MLEN": {"min": 16, "max": 64},
        "BLEN": {"min": 4, "max": 16},
        "matrix_k_splits": {"min": 2, "max": max_k_splits},
    }
    warnings: list[str] = []
    if not 16 <= mlen <= 64:
        warnings.append(f"MLEN={mlen} is outside the DC calibration range [16, 64]")
    if not 4 <= blen <= 16:
        warnings.append(f"BLEN={blen} is outside the DC calibration range [4, 16]")
    if not 2 <= k_splits <= max_k_splits:
        warnings.append(
            f"MLEN/BLEN={k_splits:g} is outside the {mode} calibration "
            f"range [2, {max_k_splits}]"
        )
    return {
        "calibration_domain": domain,
        "calibration_in_domain": not warnings,
        "calibration_status": "calibrated" if not warnings else "structural_extrapolation",
        "calibration_warnings": warnings,
        "matrix_k_splits": k_splits,
        "extrapolation_ratios": {
            "MLEN": max(1.0, mlen / 64.0),
            "BLEN": max(1.0, blen / 16.0),
            "matrix_k_splits": max(1.0, k_splits / max_k_splits),
        },
    }


def _load_coeffs(mode: str, explicit_path: str | Path | None = None) -> dict[str, float] | None:
    """Load MatrixMachine coefficients for one numeric family."""
    env_var = "PLENA_AREA_NEW_MXINT_COEFFICIENTS" if mode == "mxint" else "PLENA_AREA_NEW_MXFP_COEFFICIENTS"
    path = explicit_path or os.environ.get(env_var)
    if path is None:
        path = CALIBRATION_DIR / f"{mode}_model_coefficients.json"
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        raw = json.load(f)
    coeffs = raw.get("coefficients", raw)
    return {str(k): float(v) for k, v in coeffs.items()}


def _fp_parts(config: dict[str, Any]) -> tuple[int, int]:
    """Resolve generic FP exponent/mantissa widths for compatibility terms."""
    if "FP_EXP_WIDTH" in config and "FP_MANT_WIDTH" in config:
        return int(config["FP_EXP_WIDTH"]), int(config["FP_MANT_WIDTH"])
    if "FP_SETTING" in config:
        token = str(config["FP_SETTING"]).upper().replace("FP_", "")
        if token.startswith("E") and "M" in token:
            exp, mant = token[1:].split("M", 1)
            return int(exp), int(mant)
    return int(config.get("S_FP_EXP_WIDTH", 5)), int(config.get("S_FP_MANT_WIDTH", 6))


def _legacy_remaining_area(config: dict[str, Any], sides: dict[str, Any]) -> dict[str, Any]:
    """Estimate units lacking fitted artifacts with the original proxy.

    Each legacy term is removed as soon as its precision-aware replacement is
    available, preventing double counting during incremental model rollout.
    SRAM is never taken from the legacy proxy.
    """
    try:
        from analytic_models.area.area_proxy import estimate_area as estimate_legacy_area
        from analytic_models.area.area_proxy import load_area_units
    except Exception:  # pragma: no cover - optional compatibility path
        return {"area": 0.0, "breakdown": {}, "inputs": {}}

    act = parse_precision(config["ACT_WIDTH"], default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)))
    kv = parse_precision(config["KV_WIDTH"], default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)))
    fp_exp, fp_mant = _fp_parts(config)
    mlen = int(config["MLEN"])
    blen = int(config["BLEN"])
    vlen = int(config.get("VLEN", mlen))
    block_dim = int(config.get("BLOCK_DIM", blen))
    scale_width = int(sides.get("scale_width", config.get("MX_SCALE_WIDTH", 8)))
    t_width = int(sides["t_width"])
    legacy_config = {
        "MLEN": mlen,
        "BLEN": blen,
        "VLEN": vlen,
        "BLOCK_DIM": block_dim,
        "ACT_ELEMENT_WIDTH": act.element_width,
        "KV_ELEMENT_WIDTH": kv.element_width,
        "FP_EXP_WIDTH": fp_exp,
        "FP_MANT_WIDTH": fp_mant,
        "INT_DATA_WIDTH": int(config.get("INT_DATA_WIDTH", 32)),
        "INT_SRAM_DEPTH": int(config.get("INT_SRAM_DEPTH", 32)),
        "FP_SRAM_DEPTH": int(config.get("FP_SRAM_DEPTH", 512)),
        "MATRIX_SRAM_DEPTH": int(config.get("MATRIX_SRAM_DEPTH", config.get("MATRIX_SRAM_SIZE", max(32, 2 * mlen)))),
        "VECTOR_SRAM_DEPTH": int(config.get("VECTOR_SRAM_DEPTH", config.get("VECTOR_SRAM_SIZE", 1024))),
        "WT_MX_EXP_WIDTH": 0,
        "WT_MX_MANT_WIDTH": max(0, t_width - 1),
        "MX_SCALE_WIDTH": scale_width,
        "HBM_ELE_WIDTH": int(config.get("HBM_ELE_WIDTH", mlen)),
        "HBM_SCALE_WIDTH": int(config.get("HBM_SCALE_WIDTH", max(1, mlen // max(block_dim, 1)) * scale_width)),
    }
    legacy = estimate_legacy_area(legacy_config)
    legacy_breakdown = legacy.get("area_proxy_breakdown", {})
    out = {
        "VectorMachineLegacy": float(legacy_breakdown.get("VectorMachine", 0.0)),
        "HBMSystemLegacy": float(legacy_breakdown.get("HBMSystem", 0.0)),
    }
    units = load_area_units()
    scalar = units.get("ScalarMachine", {}).get("Coefficients", {})
    fp_width = fp_exp + fp_mant + 1
    if scalar:
        out["ScalarMachineLogicLegacy"] = (
            float(scalar.get("P1", 0.0)) * fp_width
            + float(scalar.get("P2", 0.0)) * legacy_config["INT_DATA_WIDTH"]
            + float(scalar.get("P5", 0.0))
        )
    return {"area": sum(out.values()), "breakdown": out, "inputs": legacy_config}


def estimate_matrix_machine_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate precision-aware MatrixMachine logic area in um^2.

    Required keys are ``ACT_WIDTH``, ``KV_WIDTH``, ``MLEN``, and ``BLEN``;
    ``WEIGHT_WIDTH`` defaults to MXINT4. The precision bridge maps software
    knobs to asymmetric T/L PE capabilities before selecting MXINT or MXFP.
    """
    sides = derive_compute_sides(
        config["ACT_WIDTH"],
        config["KV_WIDTH"],
        config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    inputs = dict(config)
    inputs.update(sides)
    inputs["MLEN"] = int(config["MLEN"])
    inputs["BLEN"] = int(config["BLEN"])
    structural_artifact = None
    if coefficients_path is None:
        structural_artifact = matrix_structural_model.load_artifact()
    else:
        candidate_path = Path(coefficients_path)
        if candidate_path.exists():
            with candidate_path.open() as handle:
                candidate = json.load(handle)
            if candidate.get("model_version") == matrix_structural_model.MODEL_VERSION:
                structural_artifact = candidate
    if structural_artifact is not None:
        result = matrix_structural_model.estimate(inputs, structural_artifact)
    else:
        coeffs = _load_coeffs(str(sides["mode"]), coefficients_path)
        if sides["mode"] == "mxint":
            result = mxint_model.estimate(inputs, coeffs)
        else:
            result = mxfp_model.estimate(inputs, coeffs)
    result.update(
        _matrix_calibration_assessment(
            mode=str(sides["mode"]),
            mlen=inputs["MLEN"],
            blen=inputs["BLEN"],
        )
    )
    return result


def estimate_area(
    config: dict[str, Any],
    *,
    matrix_coefficients_path: str | Path | None = None,
    sram_coefficients_path: str | Path | None = None,
    vector_coefficients_path: str | Path | None = None,
    scalar_coefficients_path: str | Path | None = None,
    hbm_coefficients_path: str | Path | None = None,
    top_residual_coefficients_path: str | Path | None = None,
    apply_top_residual: bool = True,
) -> dict[str, Any]:
    """Estimate complete PLENA proxy area in um^2.

    Coefficient paths are injectable for fitting and validation. In normal DSE
    use they should be omitted so committed calibration artifacts are loaded.
    ``apply_top_residual=False`` exposes the raw module sum for validation; it
    must not be used for reported full-chip estimates.

    Returns:
        Total area, flat breakdown, model-specific results, coefficient
        provenance, raw logic subtotal, top residual, and SRAM macro subtotal.
    """
    sides = derive_compute_sides(
        config["ACT_WIDTH"],
        config["KV_WIDTH"],
        config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    matrix = estimate_matrix_machine_area(config, coefficients_path=matrix_coefficients_path)
    sram = estimate_sram_area(config, coefficients_path=sram_coefficients_path)
    legacy = _legacy_remaining_area(config, sides)
    vector = None
    scalar = None
    hbm = None
    legacy_breakdown = dict(legacy["breakdown"])
    if vector_model.has_fitted_coefficients(vector_coefficients_path):
        vector = vector_model.estimate_vector_machine_area(config, coefficients_path=vector_coefficients_path)
        legacy_breakdown.pop("VectorMachineLegacy", None)
    if scalar_model.has_fitted_coefficients(scalar_coefficients_path):
        scalar = scalar_model.estimate_scalar_machine_area(config, coefficients_path=scalar_coefficients_path)
        legacy_breakdown.pop("ScalarMachineLogicLegacy", None)
    if hbm_model.has_fitted_coefficients(hbm_coefficients_path):
        hbm = hbm_model.estimate_hbm_system_area(config, coefficients_path=hbm_coefficients_path)
        legacy_breakdown.pop("HBMSystemLegacy", None)
    logic_breakdown = {
        "MatrixMachine": float(matrix["area"]),
        **({"VectorMachine": float(vector["area"])} if vector else {}),
        **({key: float(value) for key, value in scalar["breakdown"].items()} if scalar else {}),
        **({key: float(value) for key, value in hbm["breakdown"].items()} if hbm else {}),
        **{key: float(value) for key, value in legacy_breakdown.items()},
    }
    # The residual is calibrated against logic/wrapper area. Applying it before
    # SRAM macro insertion avoids incorrectly scaling physical bitcell macros.
    logic_area = sum(logic_breakdown.values())
    top_residual = top_model.estimate_full_chip_top_residual(
        logic_area,
        coefficients_path=top_residual_coefficients_path,
    )
    if not apply_top_residual:
        top_residual = {**top_residual, "area": 0.0, "disabled_for_call": True}
    sram_breakdown = {key: float(value) for key, value in sram["area_sram_breakdown"].items()}
    breakdown = {
        **logic_breakdown,
        "FullChipTopResidual": float(top_residual["area"]),
        **sram_breakdown,
    }
    total = sum(breakdown.values())
    matrix_p10 = float(matrix.get("area_uncertainty_p10", matrix["area"]))
    matrix_p90 = float(matrix.get("area_uncertainty_p90", matrix["area"]))
    non_matrix_logic_area = logic_area - float(matrix["area"])
    logic_area_p10 = non_matrix_logic_area + matrix_p10
    logic_area_p90 = non_matrix_logic_area + matrix_p90
    top_residual_p10 = top_model.estimate_full_chip_top_residual(
        logic_area_p10,
        coefficients_path=top_residual_coefficients_path,
    )
    top_residual_p90 = top_model.estimate_full_chip_top_residual(
        logic_area_p90,
        coefficients_path=top_residual_coefficients_path,
    )
    if not apply_top_residual:
        top_residual_p10 = {**top_residual_p10, "area": 0.0, "disabled_for_call": True}
        top_residual_p90 = {**top_residual_p90, "area": 0.0, "disabled_for_call": True}
    sram_area = sum(sram_breakdown.values())
    total_p10 = logic_area_p10 + float(top_residual_p10["area"]) + sram_area
    total_p90 = logic_area_p90 + float(top_residual_p90["area"]) + sram_area
    return {
        "area": total,
        "area_proxy": total,
        "area_model": "plena_structural_matrix_logic_top_residual_and_sram_v4",
        "area_uncertainty_p10": min(total, total_p10),
        "area_uncertainty_p50": total,
        "area_uncertainty_p90": max(total, total_p90),
        "area_breakdown": breakdown,
        "logic_area_before_top_residual": logic_area,
        "full_chip_top_residual": top_residual,
        "full_chip_top_residual_uncertainty": {
            "p10": top_residual_p10,
            "p50": top_residual,
            "p90": top_residual_p90,
        },
        "sram_macro_area": sram_area,
        "matrix_machine": matrix,
        "area_calibration_in_domain": bool(matrix["calibration_in_domain"]),
        "area_extrapolation_warnings": list(matrix["calibration_warnings"]),
        "vector_machine": vector,
        "scalar_machine": scalar,
        "hbm_system": hbm,
        "sram": sram,
        "legacy_remaining": {**legacy, "breakdown": legacy_breakdown, "area": sum(legacy_breakdown.values())},
    }
