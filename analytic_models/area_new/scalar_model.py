"""Precision-aware ScalarMachine logic area proxy.

The model separates integer ALU, floating-point ALU/SFU, vector-sized scalar
buffer/control, and fixed control area. Hierarchy-supervised coefficients are
fitted from DC reports with scalar SRAM submodules excluded; scalar SRAM
bitcells are added independently by :mod:`sram_model`. Areas are in um^2.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_COEFFICIENTS_PATH = CALIBRATION_DIR / "scalar_model_coefficients.json"

DEFAULT_SCALAR_COEFFICIENTS = {
    "a_int_mul": 0.02,
    "a_int_lin": 8.0,
    "a_fp_quad": 0.5,
    "a_fp_lin": 25.0,
    "a_exp": 10.0,
    "a_fp_alu_const": 0.0,
    "a_fp_sfu_const": 0.0,
    "a_const": 500.0,
}

_FP_RE = re.compile(r"^(?:FP_)?E(\d+)M(\d+)$", re.IGNORECASE)


def parse_scalar_fp_setting(config: dict[str, Any]) -> tuple[int, int, str]:
    """Return scalar FP widths with explicit RTL keys taking precedence."""
    if "S_FP_EXP_WIDTH" in config and "S_FP_MANT_WIDTH" in config:
        exp = int(config["S_FP_EXP_WIDTH"])
        mant = int(config["S_FP_MANT_WIDTH"])
        return exp, mant, f"FP_E{exp}M{mant}"
    if "FP_EXP_WIDTH" in config and "FP_MANT_WIDTH" in config:
        exp = int(config["FP_EXP_WIDTH"])
        mant = int(config["FP_MANT_WIDTH"])
        return exp, mant, f"FP_E{exp}M{mant}"
    token = str(config.get("FP_SETTING", "FP_E5M6")).strip().upper()
    match = _FP_RE.match(token)
    if not match:
        raise ValueError(f"unsupported FP_SETTING for scalar model: {config.get('FP_SETTING')!r}")
    exp = int(match.group(1))
    mant = int(match.group(2))
    return exp, mant, f"FP_E{exp}M{mant}"


def coefficient_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve coefficient path using argument, environment, then default."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_SCALAR_COEFFICIENTS")
    return Path(path) if path else DEFAULT_COEFFICIENTS_PATH


def has_fitted_coefficients(explicit_path: str | Path | None = None) -> bool:
    """Return whether a valid locally fitted coefficient artifact is present."""
    path = coefficient_path(explicit_path)
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    status = str(raw.get("metadata", {}).get("status", ""))
    return status == "fitted_from_local_plena_rtl_synth"


def load_coefficients(explicit_path: str | Path | None = None) -> tuple[dict[str, float], str]:
    """Load fitted coefficients or return monotonic bootstrap coefficients."""
    path = coefficient_path(explicit_path)
    if not path.exists():
        return dict(DEFAULT_SCALAR_COEFFICIENTS), "bootstrap_default"
    with path.open() as f:
        raw = json.load(f)
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(path)


def scalar_features(int_width: int, fp_exp: int, fp_mant: int, mlen: int = 16, vlen: int = 16) -> dict[str, float]:
    """Construct integer, FP, and vector-shape regression features.

    Quadratic width terms approximate arithmetic operators. ``min/max`` vector
    features represent RTL structures whose sizing follows the smaller/larger
    of MLEN and VLEN rather than one dimension unconditionally.
    """
    fp_width = 1 + fp_exp + fp_mant
    return {
        "int_mul": float(int_width * int_width),
        "int_lin": float(int_width),
        "fp_quad": float(fp_width * fp_width),
        "fp_lin": float(fp_width),
        "fp_exp": float(fp_exp),
        "mlen": float(mlen),
        "vlen": float(vlen),
        "mlen_buffer": float(mlen * fp_width),
        "vlen_output": float(vlen * fp_width),
        "min_vector": float(min(mlen, vlen)),
        "max_vector": float(max(mlen, vlen)),
        "min_vector_buffer": float(min(mlen, vlen) * fp_width),
        "max_vector_buffer": float(max(mlen, vlen) * fp_width),
        # Backward-compatible alias for older coefficient experiments.
        "vector_buffer": float(mlen * fp_width),
        "const": 1.0,
    }


def evaluate_breakdown(features: dict[str, float], coeffs: dict[str, float]) -> dict[str, float]:
    """Evaluate nonnegative ScalarMachine hierarchy terms in um^2."""
    int_logic = (
        coeffs.get("a_int_mul", 0.0) * features["int_mul"]
        + coeffs.get("a_int_lin", 0.0) * features["int_lin"]
    )
    fp_logic = (
        coeffs.get("a_fp_quad", 0.0) * features["fp_quad"]
        + coeffs.get("a_fp_lin", 0.0) * features["fp_lin"]
        + coeffs.get("a_exp", 0.0) * features["fp_exp"]
        + coeffs.get("a_fp_alu_const", 0.0) * features["const"]
        + coeffs.get("a_fp_sfu_const", 0.0) * features["const"]
    )
    vector_buffer = (
        coeffs.get("a_mlen_buffer", coeffs.get("a_vector_buffer", 0.0)) * features.get("mlen_buffer", features.get("vector_buffer", 0.0))
        + coeffs.get("a_vlen_output", 0.0) * features.get("vlen_output", 0.0)
        + coeffs.get("a_min_vector_buffer", 0.0) * features.get("min_vector_buffer", 0.0)
        + coeffs.get("a_max_vector_buffer", 0.0) * features.get("max_vector_buffer", 0.0)
        + coeffs.get("a_mlen", 0.0) * features.get("mlen", 0.0)
        + coeffs.get("a_vlen", 0.0) * features.get("vlen", 0.0)
        + coeffs.get("a_min_vector", 0.0) * features.get("min_vector", 0.0)
        + coeffs.get("a_max_vector", 0.0) * features.get("max_vector", 0.0)
    )
    control = coeffs.get("a_const", 0.0) * features["const"]
    return {
        "ScalarIntLogic": max(0.0, int_logic),
        "ScalarFPLogic": max(0.0, fp_logic),
        "ScalarVectorBufferLogic": max(0.0, vector_buffer),
        "ScalarControl": max(0.0, control),
    }


def estimate_scalar_machine_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate ScalarMachine logic area and return its hierarchy breakdown."""
    int_width = int(config.get("INT_DATA_WIDTH", 32))
    mlen = int(config.get("MLEN", config.get("VLEN", 16)))
    vlen = int(config.get("VLEN", config.get("MLEN", 16)))
    fp_exp, fp_mant, fp_setting = parse_scalar_fp_setting(config)
    fp_width = 1 + fp_exp + fp_mant
    coeffs, source = load_coefficients(coefficients_path)
    features = scalar_features(int_width, fp_exp, fp_mant, mlen, vlen)
    breakdown = evaluate_breakdown(features, coeffs)
    area = sum(breakdown.values())
    fitted = source != "bootstrap_default"
    return {
        "area": area,
        "area_proxy": area,
        "area_model": "scalar_machine_precision_proxy_v1" if fitted else "scalar_machine_precision_proxy_v1_bootstrap",
        "coefficients_source": source,
        "coefficients": coeffs,
        "features": features,
        "breakdown": breakdown,
        "inputs": {
            "INT_DATA_WIDTH": int_width,
            "MLEN": mlen,
            "VLEN": vlen,
            "FP_SETTING": fp_setting,
            "S_FP_EXP_WIDTH": fp_exp,
            "S_FP_MANT_WIDTH": fp_mant,
            "fp_width": fp_width,
        },
    }
