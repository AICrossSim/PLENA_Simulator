"""Precision-aware on-chip HBM interface logic area proxy.

This model covers PLENA's ``hbm_sys`` RTL: packing/dequant paths, scale lanes,
address generation, prefetch/writeback control, and fixed interface logic. It
does not include HBM PHYs, memory stacks, package area, channel count, capacity,
or Ramulator timing. Returned area is synthesized 7 nm logic area in um^2.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .precision import derive_compute_sides, parse_precision

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_COEFFICIENTS_PATH = CALIBRATION_DIR / "hbm_model_coefficients.json"

DEFAULT_HBM_COEFFICIENTS = {
    "a_ele": 0.5,
    "a_scale": 5.0,
    "a_m_path": 1.0,
    "a_v_path": 1.0,
    "a_scale_path": 5.0,
    "a_addr": 10.0,
    "a_load": 100.0,
    "a_write": 100.0,
    "a_const": 1000.0,
}


def coefficient_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve coefficient path using argument, environment, then default."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_HBM_COEFFICIENTS")
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
        return dict(DEFAULT_HBM_COEFFICIENTS), "bootstrap_default"
    with path.open() as f:
        raw = json.load(f)
    coeffs = raw.get("coefficients", raw)
    return {key: float(value) for key, value in coeffs.items()}, str(path)


def _ceil_log2(value: int) -> int:
    return int(math.ceil(math.log2(max(value, 1))))


def _next_power_of_two(value: int) -> int:
    return 1 << _ceil_log2(value)


def _fp_width(config: dict[str, Any]) -> int:
    if "V_FP_EXP_WIDTH" in config and "V_FP_MANT_WIDTH" in config:
        return 1 + int(config["V_FP_EXP_WIDTH"]) + int(config["V_FP_MANT_WIDTH"])
    if "FP_EXP_WIDTH" in config and "FP_MANT_WIDTH" in config:
        return 1 + int(config["FP_EXP_WIDTH"]) + int(config["FP_MANT_WIDTH"])
    token = str(config.get("FP_SETTING", "FP_E5M6")).upper().replace("FP_", "")
    if token.startswith("E") and "M" in token:
        exp, mant = token[1:].split("M", 1)
        return 1 + int(exp) + int(mant)
    return 12


def hbm_features(config: dict[str, Any]) -> dict[str, float]:
    """Derive RTL-shaped features for the on-chip HBM interface.

    The interface lane width is the maximum payload width routed through the
    shared path. ``HBM_ELE_WIDTH`` follows the RTL power-of-two rounding rule.
    Prefetch amounts enter logarithmically because they primarily resize
    counters/control, not the external HBM storage.
    """
    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    blen = int(config["BLEN"])
    block_dim = int(config.get("BLOCK_DIM", blen))
    scale_width = int(config.get("MX_SCALE_WIDTH", 8))
    m_load = int(config.get("HBM_M_Prefetch_Amount", config.get("M_LOAD", mlen)))
    v_load = int(config.get("HBM_V_Prefetch_Amount", config.get("V_LOAD", 4)))
    v_write = int(config.get("HBM_V_Writeback_Amount", config.get("V_WRITE", 4)))

    act = parse_precision(config["ACT_WIDTH"], default_scale_width=scale_width)
    kv = parse_precision(config["KV_WIDTH"], default_scale_width=scale_width)
    wt = parse_precision(config.get("WEIGHT_WIDTH", "MXINT4"), default_scale_width=scale_width)
    sides = derive_compute_sides(config["ACT_WIDTH"], config["KV_WIDTH"], config.get("WEIGHT_WIDTH", "MXINT4"), default_scale_width=scale_width)

    wt_width = int(wt.element_width)
    act_width = int(act.element_width)
    kv_width = int(kv.element_width)
    v_fp_width = _fp_width(config)
    mx_lane_width = max(wt_width, act_width, kv_width)
    hbm_lane_width = max(mx_lane_width, v_fp_width)
    hbm_ele_width_raw = hbm_lane_width * mlen
    # Match configuration.svh: reserve two packed payloads and round the
    # physical interface to a power of two.
    hbm_ele_width = _next_power_of_two(hbm_ele_width_raw * 2)
    hbm_scale_width = scale_width * max(1, mlen // max(block_dim, 1))
    source_width = 4 + _ceil_log2(max(1, hbm_lane_width * mlen // 16))
    hbm_addr_width = int(config.get("HBM_ADDR_WIDTH", 128))

    return {
        "hbm_ele_width": float(hbm_ele_width),
        "hbm_scale_width": float(hbm_scale_width),
        "m_path": float(mlen * (wt_width + kv_width)),
        "v_path": float(vlen * (act_width + kv_width)),
        "scale_path": float(((mlen + vlen) / max(blen, 1)) * scale_width),
        "addr": float(source_width * hbm_addr_width),
        "load": float(math.log2(max(m_load, 0) + 1) + math.log2(max(v_load, 0) + 1)),
        "write": float(math.log2(max(v_write, 0) + 1)),
        "const": 1.0,
        "mlen": float(mlen),
        "vlen": float(vlen),
        "blen": float(blen),
        "block_dim": float(block_dim),
        "wt_width": float(wt_width),
        "act_width": float(act_width),
        "kv_width": float(kv_width),
        "v_fp_width": float(v_fp_width),
        "hbm_lane_width": float(hbm_lane_width),
        "hbm_ele_width_raw": float(hbm_ele_width_raw),
        "source_width": float(source_width),
        "hbm_addr_width": float(hbm_addr_width),
        "m_load": float(m_load),
        "v_load": float(v_load),
        "v_write": float(v_write),
        "mode": str(sides["mode"]),
    }


def evaluate_breakdown(features: dict[str, float], coeffs: dict[str, float]) -> dict[str, float]:
    """Evaluate fitted HBM interface terms and clamp numerical negatives."""
    matrix_path = (
        coeffs.get("a_ele", 0.0) * features["hbm_ele_width"]
        + coeffs.get("a_m_path", 0.0) * features["m_path"]
        + coeffs.get("a_scale", 0.0) * features["hbm_scale_width"]
    )
    vector_path = coeffs.get("a_v_path", 0.0) * features["v_path"]
    scale_path = coeffs.get("a_scale_path", 0.0) * features["scale_path"]
    address_control = coeffs.get("a_addr", 0.0) * features["addr"]
    prefetch_writeback = (
        coeffs.get("a_load", 0.0) * features["load"]
        + coeffs.get("a_write", 0.0) * features["write"]
    )
    fixed_control = coeffs.get("a_const", 0.0) * features["const"]
    return {
        "HBMMatrixPath": max(0.0, matrix_path),
        "HBMVectorPath": max(0.0, vector_path),
        "HBMScalePath": max(0.0, scale_path),
        "HBMAddressControl": max(0.0, address_control),
        "HBMPrefetchWritebackControl": max(0.0, prefetch_writeback),
        "HBMFixedControl": max(0.0, fixed_control),
    }


def estimate_hbm_system_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate on-chip HBM interface logic area in um^2.

    ``HBM_CHANNELS`` is intentionally absent: it is a transactional latency
    setting and the current RTL top does not instantiate one PHY per modeled
    Ramulator channel.
    """
    coeffs, source = load_coefficients(coefficients_path)
    features = hbm_features(config)
    breakdown = evaluate_breakdown(features, coeffs)
    area = sum(breakdown.values())
    fitted = source != "bootstrap_default"
    return {
        "area": area,
        "area_proxy": area,
        "area_model": "hbm_system_interface_proxy_v1" if fitted else "hbm_system_interface_proxy_v1_bootstrap",
        "coefficients_source": source,
        "coefficients": coeffs,
        "features": features,
        "breakdown": breakdown,
        "inputs": {
            "MLEN": int(features["mlen"]),
            "VLEN": int(features["vlen"]),
            "BLEN": int(features["blen"]),
            "BLOCK_DIM": int(features["block_dim"]),
            "WT_ELEMENT_WIDTH": int(features["wt_width"]),
            "ACT_ELEMENT_WIDTH": int(features["act_width"]),
            "KV_ELEMENT_WIDTH": int(features["kv_width"]),
            "V_FP_ELEMENT_WIDTH": int(features["v_fp_width"]),
            "HBM_ELE_WIDTH": int(features["hbm_ele_width"]),
            "HBM_SCALE_WIDTH": int(features["hbm_scale_width"]),
            "SourceWidth": int(features["source_width"]),
            "HBM_M_Prefetch_Amount": int(features["m_load"]),
            "HBM_V_Prefetch_Amount": int(features["v_load"]),
            "HBM_V_Writeback_Amount": int(features["v_write"]),
        },
    }


__all__ = ["estimate_hbm_system_area", "has_fitted_coefficients", "hbm_features"]
