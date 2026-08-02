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
DEFAULT_RTL_V3_DELTA_PATH = CALIBRATION_DIR / "vector_rtl_v3_delta_coefficients.json"
DEFAULT_RTL_V4_DELTA_PATH = CALIBRATION_DIR / "vector_rtl_v4_delta_coefficients.json"
DEFAULT_RTL_V5_DELTA_PATH = CALIBRATION_DIR / "vector_rtl_v5_delta_coefficients.json"

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


def load_rtl_v3_delta(
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, float] | None, str | None]:
    """Load the paired old/new rtl-v3 delta artifact when available."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_VECTOR_RTL_V3_DELTA")
    resolved = Path(path) if path else DEFAULT_RTL_V3_DELTA_PATH
    if not resolved.exists():
        return None, None
    with resolved.open() as handle:
        raw = json.load(handle)
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(resolved)


def _vector_scalar_area_version(config: dict[str, Any]) -> str:
    """Return the requested Vector/Scalar RTL generation."""
    version = config.get(
        "vector_scalar_area_version",
        config.get("VECTOR_SCALAR_AREA_VERSION", config.get("vector_scalar_schedule", "rtl-v3")),
    )
    return str(version).strip().lower()


def _use_rtl_v3_delta(config: dict[str, Any]) -> bool:
    """Apply the cumulative rtl-v3 hardware delta to rtl-v3 and later RTL."""
    return _vector_scalar_area_version(config) in {"rtl-v3", "rtl-v4", "rtl-v5"}


def load_rtl_v4_delta(
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, float] | None, str | None, str]:
    """Load the paired rtl-v4 compact-stat delta after physical calibration.

    Missing or non-promoted artifacts are deliberately not interpreted as zero
    area.  The caller reports a pending calibration status instead.
    """
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_VECTOR_RTL_V4_DELTA")
    resolved = Path(path) if path else DEFAULT_RTL_V4_DELTA_PATH
    if not resolved.exists():
        return None, None, "recalibration_pending_rtl_v4"
    with resolved.open() as handle:
        raw = json.load(handle)
    status = str(raw.get("metadata", {}).get("status", ""))
    if status not in {
        "fitted_from_paired_rtl_v4_dc",
        "calibrated_paired_rtl_v4_delta",
    }:
        return None, str(resolved), status or "rtl_v4_artifact_not_promoted"
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(resolved), status


def load_rtl_v5_delta(
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, float] | None, str | None, str]:
    """Load the structural compact-lane overlay used by rtl-v5."""

    path = explicit_path or os.environ.get("PLENA_AREA_NEW_VECTOR_RTL_V5_DELTA")
    resolved = Path(path) if path else DEFAULT_RTL_V5_DELTA_PATH
    if not resolved.exists():
        return None, None, "recalibration_pending_rtl_v5"
    with resolved.open() as handle:
        raw = json.load(handle)
    status = str(raw.get("metadata", {}).get("status", ""))
    if status not in {
        "structural_extrapolation_from_compact_leaf_dc",
        "fitted_from_paired_rtl_v5_dc",
    }:
        return None, str(resolved), status or "rtl_v5_artifact_not_promoted"
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(resolved), status


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
    rtl_v3_delta_path: str | Path | None = None,
    rtl_v4_delta_path: str | Path | None = None,
    rtl_v5_delta_path: str | Path | None = None,
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
    baseline_area = max(0.0, evaluate_area(features, coeffs))
    area = baseline_area
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
    delta_coeffs, delta_source = load_rtl_v3_delta(rtl_v3_delta_path)
    rtl_v3_delta = 0.0
    if delta_coeffs is not None and _use_rtl_v3_delta(config):
        rtl_v3_delta = max(
            0.0,
            delta_coeffs.get("delta_vlen_width", 0.0) * features["lane_linear"]
            + delta_coeffs.get("delta_const", 0.0),
        )
        area += rtl_v3_delta
        breakdown["VectorRTLv3SegmentParallelDelta"] = rtl_v3_delta
        area_model += "_rtl_v3_delta_overlay"
    area_version = _vector_scalar_area_version(config)
    rtl_v4_delta = 0.0
    rtl_v4_source: str | None = None
    rtl_v4_status = "not_requested"
    rtl_v4_coeffs: dict[str, float] | None = None
    if area_version == "rtl-v4":
        rtl_v4_coeffs, rtl_v4_source, rtl_v4_status = load_rtl_v4_delta(
            rtl_v4_delta_path
        )
        if rtl_v4_coeffs is not None:
            rtl_v4_delta = max(
                0.0,
                rtl_v4_coeffs.get("compact_stats_simd_const", 0.0)
                + rtl_v4_coeffs.get("compact_stats_simd_fp_width", 0.0)
                * features["fp_width"]
                + rtl_v4_coeffs.get("reduction_overwrite_control_const", 0.0),
            )
            area += rtl_v4_delta
            breakdown["CompactStatsSIMD"] = max(
                0.0,
                rtl_v4_coeffs.get("compact_stats_simd_const", 0.0)
                + rtl_v4_coeffs.get("compact_stats_simd_fp_width", 0.0)
                * features["fp_width"],
            )
            breakdown["ReductionOverwriteControl"] = max(
                0.0,
                rtl_v4_coeffs.get("reduction_overwrite_control_const", 0.0),
            )
            area_model += "_rtl_v4_delta_overlay"
    rtl_v5_delta = 0.0
    rtl_v5_source: str | None = None
    rtl_v5_status = "not_requested"
    rtl_v5_coeffs: dict[str, float] | None = None
    compact_stats_lanes = int(config.get("COMPACT_STATS_LANES", 16))
    if area_version == "rtl-v5":
        rtl_v5_coeffs, rtl_v5_source, rtl_v5_status = load_rtl_v5_delta(
            rtl_v5_delta_path
        )
        if rtl_v5_coeffs is not None:
            compact_area = max(
                0.0,
                rtl_v5_coeffs.get("compact_stats_fixed_control_um2", 0.0)
                + compact_stats_lanes
                * rtl_v5_coeffs.get("compact_stats_per_lane_um2", 0.0),
            )
            overwrite_area = max(
                0.0,
                rtl_v5_coeffs.get("reduction_overwrite_control_const", 0.0),
            )
            rtl_v5_delta = compact_area + overwrite_area
            area += rtl_v5_delta
            breakdown["CompactStatsSIMD"] = compact_area
            breakdown["ReductionOverwriteControl"] = overwrite_area
            area_model += "_rtl_v5_structural_lane_overlay"
    delta_warnings: list[str] = []
    if rtl_v3_delta > 0.0 and not 16 <= vlen <= 64:
        delta_warnings.append(
            f"rtl-v3 VectorMachine delta VLEN={vlen} is outside paired DC range [16, 64]"
        )
    if rtl_v3_delta > 0.0 and not 12 <= fp_width <= 14:
        delta_warnings.append(
            f"rtl-v3 VectorMachine delta fp_width={fp_width} is outside paired DC range [12, 14]"
        )
    if area_version == "rtl-v4" and rtl_v4_coeffs is None:
        delta_warnings.append(
            "rtl-v4 compact-stat SIMD and reduction-overwrite area calibration "
            f"is not promoted ({rtl_v4_status}); reported area includes the "
            "cumulative rtl-v3 delta but excludes the unknown rtl-v4 increment"
        )
    if area_version == "rtl-v5" and rtl_v5_coeffs is None:
        delta_warnings.append(
            "rtl-v5 compact-stat lane scaling calibration is unavailable; "
            "reported area excludes the rtl-v5 compact SIMD increment"
        )
    elif (
        area_version == "rtl-v5"
        and rtl_v5_status != "fitted_from_paired_rtl_v5_dc"
    ):
        delta_warnings.append(
            "rtl-v5 32/64-lane area uses structural leaf extrapolation; "
            "paired VectorMachine calibration remains pending"
        )
    return {
        "area": area,
        "area_proxy": area,
        "area_model": area_model,
        "coefficients_source": source,
        "coefficients": coeffs,
        "rtl_v3_delta_area": rtl_v3_delta,
        "rtl_v3_delta_coefficients_source": delta_source,
        "rtl_v3_delta_coefficients": delta_coeffs,
        "rtl_v4_delta_area": rtl_v4_delta,
        "rtl_v4_delta_coefficients_source": rtl_v4_source,
        "rtl_v4_delta_coefficients": rtl_v4_coeffs,
        "rtl_v4_delta_status": rtl_v4_status,
        "rtl_v5_delta_area": rtl_v5_delta,
        "rtl_v5_delta_coefficients_source": rtl_v5_source,
        "rtl_v5_delta_coefficients": rtl_v5_coeffs,
        "rtl_v5_delta_status": rtl_v5_status,
        "vector_scalar_area_calibration_status": (
            rtl_v5_status
            if area_version == "rtl-v5"
            else rtl_v4_status
            if area_version == "rtl-v4"
            else (
                "calibrated_rtl_v3_delta_overlay"
                if rtl_v3_delta > 0.0
                else "calibrated_pre_rtl_v3"
            )
        ),
        "rtl_v3_delta_calibration_in_domain": not delta_warnings,
        "rtl_v3_delta_calibration_warnings": delta_warnings,
        "rtl_v3_delta_extrapolation_ratio": max(1.0, vlen / 64.0),
        "breakdown": breakdown,
        "features": features,
        "inputs": {
            "VLEN": vlen,
            "FP_SETTING": fp_setting,
            "V_FP_EXP_WIDTH": exp,
            "V_FP_MANT_WIDTH": mant,
            "fp_width": fp_width,
            "COMPACT_STATS_LANES": compact_stats_lanes,
            "vector_scalar_area_version": (
                area_version
            ),
        },
    }
