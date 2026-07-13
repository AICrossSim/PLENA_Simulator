"""Precision-aware VectorMachine logic area proxy.

The fitted model uses ``VLEN`` and the vector FP exponent/mantissa widths. It
captures replicated element logic, reduction-tree growth, buffers, and fixed
control from Synopsys DC hierarchy reports. SRAM is intentionally excluded and
is modeled separately by :mod:`sram_model`. Returned area is in um^2.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_COEFFICIENTS_PATH = CALIBRATION_DIR / "vector_model_coefficients.json"

DEFAULT_VECTOR_COEFFICIENTS = {
    "a_exp_lane": 8.0,
    "a_mant_lane": 80.0,
    "e_const": 500.0,
}

_FP_RE = re.compile(r"^(?:FP_)?E(\d+)M(\d+)$", re.IGNORECASE)


def parse_fp_setting(config: dict[str, Any]) -> tuple[int, int, str]:
    """Return VectorMachine FP widths from DSE or RTL-style config.

    Explicit ``V_FP_*`` keys take precedence over generic FP keys and the
    software ``FP_SETTING`` token.
    """
    if "V_FP_EXP_WIDTH" in config and "V_FP_MANT_WIDTH" in config:
        exp = int(config["V_FP_EXP_WIDTH"])
        mant = int(config["V_FP_MANT_WIDTH"])
        return exp, mant, f"FP_E{exp}M{mant}"
    if "FP_EXP_WIDTH" in config and "FP_MANT_WIDTH" in config:
        exp = int(config["FP_EXP_WIDTH"])
        mant = int(config["FP_MANT_WIDTH"])
        return exp, mant, f"FP_E{exp}M{mant}"
    token = str(config.get("FP_SETTING", "FP_E5M6")).strip().upper()
    match = _FP_RE.match(token)
    if not match:
        raise ValueError(f"unsupported FP_SETTING for vector model: {config.get('FP_SETTING')!r}")
    exp = int(match.group(1))
    mant = int(match.group(2))
    return exp, mant, f"FP_E{exp}M{mant}"


def coefficient_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve coefficient path using argument, environment, then default."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_VECTOR_COEFFICIENTS")
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
        return dict(DEFAULT_VECTOR_COEFFICIENTS), "bootstrap_default"
    with path.open() as f:
        raw = json.load(f)
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(path)


def vector_features(vlen: int, fp_exp: int, fp_mant: int) -> dict[str, float]:
    """Construct physically motivated VectorMachine regression features.

    Lane terms scale linearly with ``VLEN``; tree terms additionally scale with
    ``log2(VLEN)`` to approximate hierarchical reduction wiring and operators.
    """
    fp_width = 1 + fp_exp + fp_mant
    log_vlen = math.log2(max(vlen, 2))
    return {
        "lane_quad": float(vlen * fp_width * fp_width),
        "tree": float(vlen * fp_width * log_vlen),
        "lane_linear": float(vlen * fp_width),
        "control": float(log_vlen * fp_width),
        "exp_lane": float(vlen * fp_exp),
        "mant_lane": float(vlen * fp_mant),
        "exp_tree": float(vlen * fp_exp * log_vlen),
        "mant_tree": float(vlen * fp_mant * log_vlen),
        "vlen": float(vlen),
        "fp_width": float(fp_width),
        "const": 1.0,
    }


def evaluate_area(features: dict[str, float], coeffs: dict[str, float]) -> float:
    """Evaluate the coefficient schema identified by its feature keys.

    Schema detection preserves compatibility with the hierarchy and legacy
    fitting experiments while the committed model uses direct-feature v2.
    """
    if "direct_mant_lane" in coeffs:
        return (
            coeffs.get("direct_exp_lane", 0.0) * features["exp_lane"]
            + coeffs.get("direct_mant_lane", 0.0) * features["mant_lane"]
            + coeffs.get("direct_exp_tree", 0.0) * features["exp_tree"]
            + coeffs.get("direct_mant_tree", 0.0) * features["mant_tree"]
            + coeffs.get("direct_lane_quad", 0.0) * features["lane_quad"]
            + coeffs.get("direct_vlen", 0.0) * features["vlen"]
            + coeffs.get("direct_const", 0.0) * features["const"]
        )
    if "element_exp_lane" in coeffs:
        return (
            coeffs.get("element_exp_lane", 0.0) * features["exp_lane"]
            + coeffs.get("element_mant_lane", 0.0) * features["mant_lane"]
            + coeffs.get("reduction_exp_tree", 0.0) * features["exp_tree"]
            + coeffs.get("reduction_mant_tree", 0.0) * features["mant_tree"]
            + coeffs.get("buffer_vlen", 0.0) * features["vlen"]
            + coeffs.get("buffer_width", 0.0) * features["fp_width"]
            + coeffs.get("top_const", 0.0) * features["const"]
        )
    return (
        coeffs.get("a_exp_lane", 0.0) * features["exp_lane"]
        + coeffs.get("a_mant_lane", 0.0) * features["mant_lane"]
        + coeffs.get("a_lane_quad", 0.0) * features["lane_quad"]
        + coeffs.get("b_tree", 0.0) * features["tree"]
        + coeffs.get("c_lane_linear", 0.0) * features["lane_linear"]
        + coeffs.get("d_control", 0.0) * features["control"]
        + coeffs.get("e_const", 0.0) * features["const"]
    )


def estimate_vector_machine_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate VectorMachine logic area in um^2.

    Returns a serializable result containing the fitted area, equation family,
    feature values, coefficient provenance, and an explanatory breakdown.
    """
    vlen = int(config["VLEN"])
    exp, mant, fp_setting = parse_fp_setting(config)
    fp_width = 1 + exp + mant
    coeffs, source = load_coefficients(coefficients_path)
    features = vector_features(vlen, exp, mant)
    area = max(0.0, evaluate_area(features, coeffs))
    fitted = source != "bootstrap_default"
    if "direct_mant_lane" in coeffs:
        breakdown = {
            "VectorLaneMantissaLogic": coeffs.get("direct_mant_lane", 0.0) * features["mant_lane"],
            "VectorLaneExponentLogic": coeffs.get("direct_exp_lane", 0.0) * features["exp_lane"],
            "VectorReductionLogic": coeffs.get("direct_exp_tree", 0.0) * features["exp_tree"]
            + coeffs.get("direct_mant_tree", 0.0) * features["mant_tree"],
            "VectorLaneQuadraticLogic": coeffs.get("direct_lane_quad", 0.0) * features["lane_quad"],
            "VectorControl": coeffs.get("direct_vlen", 0.0) * features["vlen"]
            + coeffs.get("direct_const", 0.0),
        }
        area_model = "vector_machine_direct_feature_proxy_v2"
    elif "element_exp_lane" in coeffs:
        breakdown = {
            "VectorElementUnit": coeffs.get("element_exp_lane", 0.0) * features["exp_lane"]
            + coeffs.get("element_mant_lane", 0.0) * features["mant_lane"],
            "VectorReductionUnit": coeffs.get("reduction_exp_tree", 0.0) * features["exp_tree"]
            + coeffs.get("reduction_mant_tree", 0.0) * features["mant_tree"],
            "VectorBuffers": coeffs.get("buffer_vlen", 0.0) * features["vlen"]
            + coeffs.get("buffer_width", 0.0) * features["fp_width"],
            "VectorTopControl": coeffs.get("top_const", 0.0),
        }
        area_model = "vector_machine_hierarchy_proxy_v2"
    else:
        breakdown = {"VectorMachine": area}
        area_model = "vector_machine_precision_proxy_v1" if fitted else "vector_machine_precision_proxy_v1_bootstrap"
    return {
        "area": area,
        "area_proxy": area,
        "area_model": area_model,
        "coefficients_source": source,
        "coefficients": coeffs,
        "breakdown": breakdown,
        "features": features,
        "inputs": {
            "VLEN": vlen,
            "FP_SETTING": fp_setting,
            "V_FP_EXP_WIDTH": exp,
            "V_FP_MANT_WIDTH": mant,
            "fp_width": fp_width,
        },
    }
