"""Structural-census area model for PLENA's digital HBM interface.

The census follows ``memory/HBM/rtl/hbm_sys.sv``: matrix and vector lane
endpoints, precision-width adapters, block-scale buffers, optional MXFP
converters, and the three transfer counters. It models controller/interface
logic only; HBM stacks and package interposers are outside the silicon area.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .evidence import aggregate_dc_evidence
from .matrix import load_calibration_artifact, load_pdk_scale
from .precision import derive_compute_sides, parse_precision

FEATURE_NAMES = (
    "lane_endpoints",
    "narrow_adapter_bits",
    "mxfp_converter_lanes",
    "scale_buffer_bits",
    "transfer_counter_bits",
    "fixed_control",
)

CALIBRATION_DOMAIN = {
    "MLEN": [16, 32, 64],
    "VLEN": [16, 32, 64],
    "uniform_precision": ["MXINT4", "MXINT8", "MXFP_E4M3"],
    "prefetch_amount": [16],
}


def _counter_bits(value: int) -> int:
    if value <= 0:
        raise ValueError("HBM transfer amounts must be positive")
    return max(1, math.ceil(math.log2(value + 1)))


def structural_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return exact HBM lane, adapter, scale, and counter counts."""

    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    block_dim = int(config.get("BLOCK_DIM", config.get("BLEN", 4)))
    if mlen <= 0 or vlen <= 0 or block_dim <= 0:
        raise ValueError("MLEN, VLEN, and BLOCK_DIM must be positive")
    sides = derive_compute_sides(
        config["ACT_WIDTH"],
        config["KV_WIDTH"],
        config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    activation = parse_precision(config["ACT_WIDTH"])
    key_value = parse_precision(config["KV_WIDTH"])
    weight = parse_precision(config.get("WEIGHT_WIDTH", "MXINT4"))
    narrow_adapter_bits = 0
    mxfp_converter_lanes = 0
    if sides["mode"] == "mxint":
        narrow_adapter_bits = mlen * (
            (8 - key_value.element_width) + (8 - weight.element_width)
        ) + vlen * (
            (8 - activation.element_width)
            + (8 - key_value.element_width)
            + (8 - weight.element_width)
        )
    else:
        mxfp_converter_lanes = mlen + vlen
    matrix_prefetch = int(config.get("HBM_M_Prefetch_Amount", 16))
    vector_prefetch = int(config.get("HBM_V_Prefetch_Amount", 16))
    vector_writeback = int(config.get("HBM_V_Writeback_Amount", 16))
    return {
        "lane_endpoints": mlen + vlen,
        "narrow_adapter_bits": narrow_adapter_bits,
        "mxfp_converter_lanes": mxfp_converter_lanes,
        "scale_buffer_bits": math.ceil(mlen / block_dim) * int(sides["scale_width"])
        + math.ceil(vlen / block_dim) * int(sides["scale_width"]),
        "transfer_counter_bits": _counter_bits(matrix_prefetch)
        + _counter_bits(vector_prefetch)
        + _counter_bits(vector_writeback),
    }


def feature_row(config: Mapping[str, Any]) -> dict[str, float]:
    """Evaluate the HBM-interface structural features for a configuration."""

    counts = structural_counts(config)
    return {
        **{
            name: float(counts[name])
            for name in FEATURE_NAMES
            if name != "fixed_control"
        },
        "fixed_control": 1.0,
    }


def _uniform_precision(config: Mapping[str, Any]) -> bool:
    values = {
        str(config["ACT_WIDTH"]).upper().replace("MXINT_", "MXINT"),
        str(config["KV_WIDTH"]).upper().replace("MXINT_", "MXINT"),
        str(config.get("WEIGHT_WIDTH", "MXINT4"))
        .upper()
        .replace("MXINT_", "MXINT"),
    }
    return (
        len(values) == 1
        and next(iter(values)) in CALIBRATION_DOMAIN["uniform_precision"]
    )


def estimate_hbm_interface_area(
    config: Mapping[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    corner: str = "reference",
) -> dict[str, Any]:
    """Estimate digital HBM controller/interface area in square micrometres."""

    if corner not in {"dc", "reference"}:
        raise ValueError("corner must be 'dc' or 'reference'")
    artifact = load_calibration_artifact(coefficients_path)
    coefficients = artifact.get("full_chip", {}).get("hbm_interface")
    if not coefficients:
        raise FileNotFoundError("full-chip HBM-interface coefficients are unavailable")
    features = feature_row(config)
    dc_area = sum(
        float(coefficients.get(name, 0.0)) * features[name] for name in FEATURE_NAMES
    )
    scale = load_pdk_scale(coefficients_path) if corner == "reference" else 1.0
    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    transfer_amounts = {
        int(config.get("HBM_M_Prefetch_Amount", 16)),
        int(config.get("HBM_V_Prefetch_Amount", 16)),
        int(config.get("HBM_V_Writeback_Amount", 16)),
    }
    extrapolated = (
        mlen not in CALIBRATION_DOMAIN["MLEN"]
        or vlen not in CALIBRATION_DOMAIN["VLEN"]
        or not _uniform_precision(config)
        or transfer_amounts != {16}
    )
    return {
        "area": dc_area * scale,
        "area_dc_corner": dc_area,
        "pdk_scale": scale,
        "area_model": f"hbm_interface_structural_census_{corner}",
        "counts": structural_counts(config),
        "evidence": aggregate_dc_evidence(
            "calibration/full_chip_anchors.csv:hier_hbm_system_area",
            extrapolated=extrapolated,
            calibration_domain=CALIBRATION_DOMAIN,
        ),
    }
