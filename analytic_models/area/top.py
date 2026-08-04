"""Top-level integration and SRAM-periphery area for the PLENA chip.

The calibration target is the non-overlapping residual after MatrixMachine,
VectorMachine, ScalarMachine, and HBM-system hierarchy totals are removed from
the synthesized ``plena`` total. The structural features represent top-level
fanout trees, precision-dependent routed bits, matrix-slice controls, and fixed
integration logic. SRAM bitcell macro area is added separately.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .evidence import aggregate_dc_evidence
from .matrix import load_calibration_artifact, load_pdk_scale
from .precision import derive_compute_sides

FEATURE_NAMES = (
    "routing_tree_edges",
    "wide_routing_bit_edges",
    "mxfp_routing_lanes",
    "slice_controls",
    "fixed_control",
)

CALIBRATION_DOMAIN = {
    "MLEN": [16, 32, 64],
    "VLEN": [16, 32, 64],
    "BLEN": [4, 8, 16],
    "uniform_precision": ["MXINT4", "MXINT8", "MXFP_E4M3"],
}


def structural_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return top-level routing-tree and slice-control counts."""

    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    blen = int(config["BLEN"])
    if mlen <= 0 or vlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError(
            "top-level geometry needs positive MLEN/VLEN/BLEN and MLEN%BLEN==0"
        )
    sides = derive_compute_sides(
        config["ACT_WIDTH"],
        config["KV_WIDTH"],
        config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    lanes = mlen + vlen
    tree_depth = math.ceil(math.log2(max(mlen, vlen))) if max(mlen, vlen) > 1 else 0
    widest_side = max(int(sides["t_width"]), int(sides["l_width"]))
    return {
        "routing_tree_edges": lanes * tree_depth,
        "wide_routing_bit_edges": lanes * tree_depth * max(widest_side - 4, 0),
        "mxfp_routing_lanes": lanes if sides["mode"] == "mxfp" else 0,
        "slice_controls": mlen // blen,
    }


def feature_row(config: Mapping[str, Any]) -> dict[str, float]:
    """Evaluate top-level structural features for a configuration."""

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


def estimate_top_area(
    config: Mapping[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    corner: str = "reference",
) -> dict[str, Any]:
    """Estimate top integration/periphery area in square micrometres."""

    if corner not in {"dc", "reference"}:
        raise ValueError("corner must be 'dc' or 'reference'")
    artifact = load_calibration_artifact(coefficients_path)
    coefficients = artifact.get("full_chip", {}).get("top")
    if not coefficients:
        raise FileNotFoundError("full-chip top-level coefficients are unavailable")
    features = feature_row(config)
    dc_area = sum(
        float(coefficients.get(name, 0.0)) * features[name] for name in FEATURE_NAMES
    )
    scale = load_pdk_scale(coefficients_path) if corner == "reference" else 1.0
    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    blen = int(config["BLEN"])
    extrapolated = (
        mlen not in CALIBRATION_DOMAIN["MLEN"]
        or vlen not in CALIBRATION_DOMAIN["VLEN"]
        or blen not in CALIBRATION_DOMAIN["BLEN"]
        or not _uniform_precision(config)
    )
    return {
        "area": dc_area * scale,
        "area_dc_corner": dc_area,
        "pdk_scale": scale,
        "area_model": f"top_structural_census_{corner}",
        "counts": structural_counts(config),
        "evidence": aggregate_dc_evidence(
            "calibration/full_chip_anchors.csv:plena_area_minus_hierarchy_blocks",
            extrapolated=extrapolated,
            calibration_domain=CALIBRATION_DOMAIN,
        ),
    }
