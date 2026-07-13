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

from . import hbm_model, mxint_model, mxfp_model, scalar_model, top_model, vector_model
from .precision import derive_compute_sides, parse_precision
from .sram_model import estimate_sram_area

CALIBRATION_DIR = Path(__file__).with_name("calibration")


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
    coeffs = _load_coeffs(str(sides["mode"]), coefficients_path)
    if sides["mode"] == "mxint":
        return mxint_model.estimate(inputs, coeffs)
    return mxfp_model.estimate(inputs, coeffs)


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
    return {
        "area": total,
        "area_proxy": total,
        "area_model": "plena_precision_aware_logic_top_residual_and_sram_v2",
        "area_breakdown": breakdown,
        "logic_area_before_top_residual": logic_area,
        "full_chip_top_residual": top_residual,
        "sram_macro_area": sum(sram_breakdown.values()),
        "matrix_machine": matrix,
        "vector_machine": vector,
        "scalar_machine": scalar,
        "hbm_system": hbm,
        "sram": sram,
        "legacy_remaining": {**legacy, "breakdown": legacy_breakdown, "area": sum(legacy_breakdown.values())},
    }
