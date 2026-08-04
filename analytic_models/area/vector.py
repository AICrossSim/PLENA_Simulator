"""Structural-census area model for the PLENA VectorMachine.

The census follows the RTL replication in ``vector_machine/rtl``: floating-
point element lanes scale with VLEN, reduction and prefix networks scale with
their exact edge counts, and head masks scale with ``VLEN / HLEN``. Unit areas
are fitted to the retained hierarchy totals in ``full_chip_anchors.csv``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .evidence import STRUCTURAL_ESTIMATE, aggregate_dc_evidence
from .matrix import load_calibration_artifact, load_pdk_scale

FEATURE_NAMES = (
    "element_and_reduction_bits",
    "prefix_edge_bits",
    "mask_heads",
    "fixed_control",
)

CALIBRATION_DOMAIN = {
    "VLEN": [16, 32, 64],
    "HLEN": [8, 32, 64],
    "FP_SETTING": ["FP_E5M6"],
}

RTL_REDUCTION_LEVEL_LATENCY = 7
RTL_SEGMENT_FIFO_MARGIN = 2
RTL_FP_OPERAND_WIDTH = 3


def _fp_width(config: Mapping[str, Any]) -> int:
    token = str(config.get("FP_SETTING", "FP_E5M6")).upper().replace("FP_", "")
    if not token.startswith("E") or "M" not in token:
        raise ValueError(f"unsupported VectorMachine FP setting: {token}")
    exponent, mantissa = token[1:].split("M", 1)
    return 1 + int(exponent) + int(mantissa)


def structural_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return exact lane and network counts for one VectorMachine shape."""

    vlen = int(config.get("VLEN", config["MLEN"]))
    hlen = int(config.get("HLEN", config.get("BLEN", 1)))
    if vlen <= 0 or hlen <= 0 or vlen % hlen:
        raise ValueError(f"need VLEN>0, HLEN>0, VLEN%HLEN==0; got {vlen}, {hlen}")
    return {
        "lanes": vlen,
        "reduction_edges": max(vlen - 1, 0),
        "prefix_edges": vlen * math.ceil(math.log2(vlen)) if vlen > 1 else 0,
        "mask_heads": vlen // hlen,
    }


def segment_parallel_reduction_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the exact census of the opt-in segmented-output extension.

    The reduction arithmetic is not duplicated: partitioning ``VLEN`` lanes
    into independent trees still instantiates ``VLEN`` adders and ``VLEN`` max
    units because every segment receives the common accumulator seed.  The
    extension cost is the fixed-latency result FIFO and the packet-drain logic
    in ``fp_reduction_compute_unit.sv`` and ``vector_machine.sv``.
    """

    base = structural_counts(config)
    vlen = base["lanes"]
    hlen = int(config.get("HLEN", config.get("BLEN", 1)))
    segments = int(config.get("REDUCTION_SEGMENTS", 1))
    if segments < 1 or segments > vlen or vlen % segments:
        raise ValueError(
            "REDUCTION_SEGMENTS must be a positive divisor of VLEN no larger than VLEN"
        )
    if segments > 1 and segments != vlen // hlen:
        raise ValueError(
            "segmented reduction requires REDUCTION_SEGMENTS == VLEN / HLEN"
        )

    enabled = segments > 1
    fp_width = _fp_width(config)
    segment_width = vlen // segments
    tree_levels = math.ceil(math.log2(segment_width + 1))
    fifo_packets = (
        RTL_SEGMENT_FIFO_MARGIN + tree_levels * RTL_REDUCTION_LEVEL_LATENCY
        if enabled
        else 0
    )
    fifo_result_words = fifo_packets * segments
    fifo_data_bits = fifo_result_words * fp_width
    fifo_pointer_width = math.ceil(math.log2(fifo_packets)) if enabled else 0
    fifo_count_width = math.ceil(math.log2(fifo_packets + 1)) if enabled else 0

    # The core has read/write pointers plus FIFO and in-flight credit counters.
    fifo_control_bits = (
        2 * fifo_pointer_width + 2 * fifo_count_width if enabled else 0
    )
    drain_index_bits = math.ceil(math.log2(segments)) if enabled else 0
    drain_mux_equivalent_bits = (segments - 1) * fp_width if enabled else 0
    destination_adder_bits = (
        int(config.get("FP_OPERAND_WIDTH", RTL_FP_OPERAND_WIDTH))
        if enabled
        else 0
    )
    if enabled and destination_adder_bits <= 0:
        raise ValueError("FP_OPERAND_WIDTH must be positive")
    equivalent_bits = (
        fifo_data_bits
        + fifo_control_bits
        + drain_mux_equivalent_bits
        + drain_index_bits
        + destination_adder_bits
    )
    return {
        "segments": segments,
        "segment_width": segment_width,
        "tree_levels": tree_levels,
        "fp_adders": vlen,
        "fp_max_units": vlen,
        "arithmetic_units_added": 0,
        "output_fifo_packets": fifo_packets,
        "output_fifo_result_words": fifo_result_words,
        "output_fifo_data_bits": fifo_data_bits,
        "output_fifo_pointer_bits": 2 * fifo_pointer_width,
        "output_fifo_count_bits": 2 * fifo_count_width,
        "output_fifo_control_bits": fifo_control_bits,
        "drain_mux_inputs": segments if enabled else 0,
        "drain_mux_equivalent_bits": drain_mux_equivalent_bits,
        "drain_index_bits": drain_index_bits,
        "destination_adder_bits": destination_adder_bits,
        "estimated_equivalent_bits": equivalent_bits,
    }


def feature_row(config: Mapping[str, Any]) -> dict[str, float]:
    """Evaluate the VectorMachine structural features for a configuration."""

    counts = structural_counts(config)
    fp_width = _fp_width(config)
    return {
        "element_and_reduction_bits": float(
            (counts["lanes"] + counts["reduction_edges"]) * fp_width
        ),
        "prefix_edge_bits": float(counts["prefix_edges"] * fp_width),
        "mask_heads": float(counts["mask_heads"]),
        "fixed_control": 1.0,
    }


def estimate_vector_area(
    config: Mapping[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    corner: str = "reference",
) -> dict[str, Any]:
    """Estimate VectorMachine logic area in square micrometres."""

    if corner not in {"dc", "reference"}:
        raise ValueError("corner must be 'dc' or 'reference'")
    artifact = load_calibration_artifact(coefficients_path)
    coefficients = artifact.get("full_chip", {}).get("vector")
    if not coefficients:
        raise FileNotFoundError("full-chip VectorMachine coefficients are unavailable")
    features = feature_row(config)
    calibrated_dc_area = sum(
        float(coefficients.get(name, 0.0)) * features[name] for name in FEATURE_NAMES
    )
    scale = load_pdk_scale(coefficients_path) if corner == "reference" else 1.0
    segment_counts = segment_parallel_reduction_counts(config)
    proxy_coefficient_name = "element_and_reduction_bits"
    proxy_coefficient = float(coefficients[proxy_coefficient_name])
    structural_dc_area = (
        segment_counts["estimated_equivalent_bits"] * proxy_coefficient
    )
    structural_area = structural_dc_area * scale
    calibrated_area = calibrated_dc_area * scale
    enhanced_area = calibrated_area + structural_area
    segment_enabled = segment_counts["segments"] > 1
    vlen = int(config.get("VLEN", config["MLEN"]))
    hlen = int(config.get("HLEN", config.get("BLEN", 1)))
    extrapolated = (
        vlen not in CALIBRATION_DOMAIN["VLEN"]
        or hlen not in CALIBRATION_DOMAIN["HLEN"]
        or str(config.get("FP_SETTING", "FP_E5M6")).upper()
        not in CALIBRATION_DOMAIN["FP_SETTING"]
    )
    baseline_evidence = aggregate_dc_evidence(
        "calibration/full_chip_anchors.csv:hier_vector_machine_area",
        extrapolated=extrapolated,
        calibration_domain=CALIBRATION_DOMAIN,
    )
    evidence = dict(baseline_evidence)
    if segment_enabled:
        evidence.update(
            {
                "tier": STRUCTURAL_ESTIMATE,
                "structural_extrapolation": True,
                "calibrated_baseline_tier": baseline_evidence["tier"],
                "structural_extension": "segment_parallel_reduction",
            }
        )
    return {
        # ``area`` is the selectable chip-building total.  The retained fit is
        # kept separately so enabling a new RTL structure cannot be free.
        "area": enhanced_area,
        "area_dc_corner": calibrated_dc_area + structural_dc_area,
        "calibrated_area": calibrated_area,
        "calibrated_area_dc_corner": calibrated_dc_area,
        "enhanced_area": enhanced_area,
        "pdk_scale": scale,
        "area_model": f"vector_structural_census_{corner}",
        "counts": structural_counts(config),
        "segment_parallel_reduction_counts": segment_counts,
        "segment_parallel_reduction_area_accounting": {
            "status": "structural_census_only",
            "evidence_tier": STRUCTURAL_ESTIMATE,
            "enabled": segment_enabled,
            "included_in_calibrated_area": False,
            "included_in_enhanced_area": segment_enabled,
            "structural_estimate_area": structural_area,
            "structural_estimate_area_dc_corner": structural_dc_area,
            "proxy_coefficient_name": proxy_coefficient_name,
            "proxy_coefficient_area_per_equivalent_bit_dc": proxy_coefficient,
            "coefficient_use": "extrapolated_from_closest_retained_per_bit_fit",
            "reason": (
                "no retained DC points contain the opt-in segmented-output "
                "FIFO and drain path"
            ),
        },
        "evidence": evidence,
    }
