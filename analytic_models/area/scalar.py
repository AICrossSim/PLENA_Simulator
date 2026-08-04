"""Structural-census area model for the PLENA ScalarMachine.

The RTL contains fixed integer and floating-point execution logic plus an
MLEN-wide floating-point vector buffer and a VLEN counter. The aggregate DC
points vary those replicated structures at the fixed INT32/FP-E5M6 datapath.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .evidence import STRUCTURAL_ESTIMATE, aggregate_dc_evidence
from .matrix import load_calibration_artifact, load_pdk_scale

FEATURE_NAMES = ("fp_vector_buffer_bits", "vector_counter_bits", "fixed_control")

RTL_FP_ROB_DEPTH = 4
RTL_FP_OPERAND_WIDTH = 3
RTL_INT_OPERAND_WIDTH = 4
RTL_INT_DATA_WIDTH = 32
RTL_LOOP_COUNT_WIDTH = 22
RTL_MAX_LOOP_DEPTH = 4
RTL_LOOP_ADDRESS_STREAMS = 4

CALIBRATION_DOMAIN = {
    "MLEN": [16, 32, 64],
    "VLEN": [16, 32, 64],
    "INT_DATA_WIDTH": [32],
    "FP_SETTING": ["FP_E5M6"],
}


def _fp_width(config: Mapping[str, Any]) -> int:
    token = str(config.get("FP_SETTING", "FP_E5M6")).upper().replace("FP_", "")
    if not token.startswith("E") or "M" not in token:
        raise ValueError(f"unsupported ScalarMachine FP setting: {token}")
    exponent, mantissa = token[1:].split("M", 1)
    return 1 + int(exponent) + int(mantissa)


def fp_issue_pipeline_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the exact sequential and interface census for scalar FP issue.

    These counts mirror ``scalar_fp_reorder_buffer.sv`` and the two execution-
    unit reservation bits in ``scalar_machine.sv``.  They are deliberately
    reported separately from the retained baseline DC fit: no new DC result is
    available on this host, so control area must not be silently invented.
    """

    depth = int(config.get("SCALAR_FP_ROB_DEPTH", RTL_FP_ROB_DEPTH))
    register_address_bits = int(
        config.get("FP_OPERAND_WIDTH", RTL_FP_OPERAND_WIDTH)
    )
    if depth < 2:
        raise ValueError("SCALAR_FP_ROB_DEPTH must be at least two")
    if register_address_bits <= 0:
        raise ValueError("FP_OPERAND_WIDTH must be positive")

    tag_bits = math.ceil(math.log2(depth))
    occupancy_bits = math.ceil(math.log2(depth + 1))
    entry_bits = _fp_width(config) + register_address_bits + 2
    entry_storage_bits = depth * entry_bits
    pointer_bits = 2 * tag_bits
    reorder_sequential_bits = entry_storage_bits + pointer_bits + occupancy_bits
    return {
        "fp_reorder_entries": depth,
        "fp_reorder_entry_bits": entry_bits,
        "fp_reorder_entry_storage_bits": entry_storage_bits,
        "fp_reorder_pointer_bits": pointer_bits,
        "fp_reorder_occupancy_bits": occupancy_bits,
        "fp_reorder_sequential_bits": reorder_sequential_bits,
        "fp_issue_reservation_bits": 2,
        "fp_pending_write_mask_bits": 1 << register_address_bits,
        "fp_completion_channels": 2,
        "fp_retirement_ports": 1,
    }


def loop_address_generator_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the exact state and datapath census of ``loop_address_generator.sv``."""

    address_width = int(config.get("INT_DATA_WIDTH", RTL_INT_DATA_WIDTH))
    register_index_width = int(
        config.get("INT_OPERAND_WIDTH", RTL_INT_OPERAND_WIDTH)
    )
    loop_count_width = int(config.get("IMM_WIDTH", RTL_LOOP_COUNT_WIDTH))
    loop_depth = int(config.get("MAX_LOOP_DEPTH", RTL_MAX_LOOP_DEPTH))
    address_streams = int(
        config.get("LOOP_ADDRESS_STREAMS", RTL_LOOP_ADDRESS_STREAMS)
    )
    if min(
        address_width,
        register_index_width,
        loop_count_width,
        loop_depth,
        address_streams,
    ) <= 0:
        raise ValueError("loop address-generator dimensions must be positive")
    if address_width < loop_count_width:
        raise ValueError("INT_DATA_WIDTH must cover IMM_WIDTH")

    register_count = 1 << register_index_width
    stream_entry_bits = register_index_width + loop_count_width + 1
    shadow_value_bits = register_count * address_width
    shadow_valid_bits = register_count
    pending_state_bits = address_streams * stream_entry_bits
    frame_counter_bits = loop_depth * (
        loop_count_width + register_index_width
    )
    frame_stream_bits = loop_depth * address_streams * stream_entry_bits
    frame_state_bits = frame_counter_bits + frame_stream_bits
    stack_pointer_bits = math.ceil(math.log2(loop_depth + 1))
    status_bits = 2
    sequential_bits = (
        shadow_value_bits
        + shadow_valid_bits
        + pending_state_bits
        + frame_state_bits
        + stack_pointer_bits
        + status_bits
    )

    # Wide stored values use the retained data-bit proxy. Register selectors,
    # valids, pointers and status use the retained scalar control-bit proxy.
    storage_state_bits = (
        shadow_value_bits
        + address_streams * loop_count_width
        + loop_depth * loop_count_width
        + loop_depth * address_streams * loop_count_width
    )
    control_state_bits = sequential_bits - storage_state_bits
    address_adder_bits = address_streams * address_width
    counter_decrement_bits = loop_count_width
    shadow_read_mux_equivalent_bits = (
        3 * (register_count - 1) * address_width
    )
    pending_comparator_bits = address_streams * register_index_width
    datapath_equivalent_bits = (
        address_adder_bits
        + counter_decrement_bits
        + shadow_read_mux_equivalent_bits
    )
    control_equivalent_bits = control_state_bits + pending_comparator_bits
    return {
        "address_width": address_width,
        "register_index_width": register_index_width,
        "register_count": register_count,
        "loop_count_width": loop_count_width,
        "max_loop_depth": loop_depth,
        "address_streams_per_loop": address_streams,
        "shadow_value_bits": shadow_value_bits,
        "shadow_valid_bits": shadow_valid_bits,
        "shadow_state_bits": shadow_value_bits + shadow_valid_bits,
        "pending_entry_bits": stream_entry_bits,
        "pending_state_bits": pending_state_bits,
        "frame_counter_bits": frame_counter_bits,
        "frame_stream_bits": frame_stream_bits,
        "frame_state_bits": frame_state_bits,
        "stack_pointer_bits": stack_pointer_bits,
        "status_bits": status_bits,
        "sequential_bits": sequential_bits,
        "storage_state_bits": storage_state_bits,
        "control_state_bits": control_state_bits,
        "address_adders": address_streams,
        "address_adder_bits": address_adder_bits,
        "counter_decrementers": 1,
        "counter_decrement_bits": counter_decrement_bits,
        "shadow_read_ports": 3,
        "shadow_read_mux_inputs": register_count,
        "shadow_read_mux_equivalent_bits": shadow_read_mux_equivalent_bits,
        "pending_register_comparators": address_streams,
        "pending_comparator_bits": pending_comparator_bits,
        "datapath_equivalent_bits": datapath_equivalent_bits,
        "storage_equivalent_bits": storage_state_bits
        + datapath_equivalent_bits,
        "control_equivalent_bits": control_equivalent_bits,
    }


def structural_counts(config: Mapping[str, Any]) -> dict[str, int]:
    """Return exact replicated storage/control counts from ``scalar_machine.sv``."""

    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    if mlen <= 0 or vlen <= 0:
        raise ValueError("MLEN and VLEN must be positive")
    return {
        "fp_vector_buffer_elements": mlen,
        "vector_counter_bits": math.ceil(math.log2(vlen)) + 1 if vlen > 1 else 1,
    }


def feature_row(config: Mapping[str, Any]) -> dict[str, float]:
    """Evaluate the ScalarMachine structural features for a configuration."""

    counts = structural_counts(config)
    return {
        "fp_vector_buffer_bits": float(
            counts["fp_vector_buffer_elements"] * _fp_width(config)
        ),
        "vector_counter_bits": float(counts["vector_counter_bits"]),
        "fixed_control": 1.0,
    }


def estimate_scalar_area(
    config: Mapping[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    corner: str = "reference",
) -> dict[str, Any]:
    """Estimate ScalarMachine logic area in square micrometres."""

    if corner not in {"dc", "reference"}:
        raise ValueError("corner must be 'dc' or 'reference'")
    artifact = load_calibration_artifact(coefficients_path)
    coefficients = artifact.get("full_chip", {}).get("scalar")
    if not coefficients:
        raise FileNotFoundError("full-chip ScalarMachine coefficients are unavailable")
    features = feature_row(config)
    calibrated_dc_area = sum(
        float(coefficients.get(name, 0.0)) * features[name] for name in FEATURE_NAMES
    )
    scale = load_pdk_scale(coefficients_path) if corner == "reference" else 1.0
    issue_counts = fp_issue_pipeline_counts(config)
    issue_enabled = config.get("SCALAR_FP_ISSUE_PIPELINE", False)
    if not isinstance(issue_enabled, bool):
        raise TypeError("SCALAR_FP_ISSUE_PIPELINE must be boolean")
    loop_counts = loop_address_generator_counts(config)
    loop_enabled = config.get("ENABLE_LOOP_ADDRESS_GENERATOR", False)
    if not isinstance(loop_enabled, bool):
        raise TypeError("ENABLE_LOOP_ADDRESS_GENERATOR must be boolean")

    # The closest retained ScalarMachine fits distinguish wide data storage
    # from counter/control state.  Reuse those per-bit coefficients as an
    # explicitly extrapolated proxy; this is not presented as a new DC point.
    storage_coefficient_name = "fp_vector_buffer_bits"
    control_coefficient_name = "vector_counter_bits"
    storage_coefficient = float(coefficients[storage_coefficient_name])
    control_coefficient = float(coefficients[control_coefficient_name])
    storage_equivalent_bits = issue_counts["fp_reorder_entry_storage_bits"]
    control_equivalent_bits = (
        issue_counts["fp_reorder_pointer_bits"]
        + issue_counts["fp_reorder_occupancy_bits"]
        + issue_counts["fp_issue_reservation_bits"]
    )
    issue_structural_dc_area = (
        storage_equivalent_bits * storage_coefficient
        + control_equivalent_bits * control_coefficient
    )
    issue_structural_area = issue_structural_dc_area * scale

    loop_storage_equivalent_bits = loop_counts["storage_equivalent_bits"]
    loop_control_equivalent_bits = loop_counts["control_equivalent_bits"]
    loop_structural_dc_area = (
        loop_storage_equivalent_bits * storage_coefficient
        + loop_control_equivalent_bits * control_coefficient
    )
    loop_structural_area = loop_structural_dc_area * scale
    calibrated_area = calibrated_dc_area * scale
    selected_structural_dc_area = (
        (issue_structural_dc_area if issue_enabled else 0.0)
        + (loop_structural_dc_area if loop_enabled else 0.0)
    )
    selected_structural_area = (
        (issue_structural_area if issue_enabled else 0.0)
        + (loop_structural_area if loop_enabled else 0.0)
    )
    enhanced_area = calibrated_area + selected_structural_area
    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    extrapolated = (
        mlen not in CALIBRATION_DOMAIN["MLEN"]
        or vlen not in CALIBRATION_DOMAIN["VLEN"]
        or int(config.get("INT_DATA_WIDTH", 32)) != 32
        or str(config.get("FP_SETTING", "FP_E5M6")).upper()
        not in CALIBRATION_DOMAIN["FP_SETTING"]
    )
    baseline_evidence = aggregate_dc_evidence(
        "calibration/full_chip_anchors.csv:hier_scalar_machine_area",
        extrapolated=extrapolated,
        calibration_domain=CALIBRATION_DOMAIN,
    )
    evidence = dict(baseline_evidence)
    enabled_extensions = []
    if issue_enabled:
        enabled_extensions.append("scalar_fp_issue_pipeline")
    if loop_enabled:
        enabled_extensions.append("loop_address_generator")
    if enabled_extensions:
        evidence.update(
            {
                "tier": STRUCTURAL_ESTIMATE,
                "structural_extrapolation": True,
                "calibrated_baseline_tier": baseline_evidence["tier"],
                "structural_extensions": enabled_extensions,
            }
        )
    return {
        # ``area`` is the selectable chip-building total.  The retained fit is
        # kept separately so enabling the new issue pipeline cannot be free.
        "area": enhanced_area,
        "area_dc_corner": calibrated_dc_area + selected_structural_dc_area,
        "calibrated_area": calibrated_area,
        "calibrated_area_dc_corner": calibrated_dc_area,
        "enhanced_area": enhanced_area,
        "pdk_scale": scale,
        "area_model": f"scalar_structural_census_{corner}",
        "counts": structural_counts(config),
        "fp_issue_pipeline_counts": issue_counts,
        "fp_issue_pipeline_area_accounting": {
            "status": "structural_census_only",
            "evidence_tier": STRUCTURAL_ESTIMATE,
            "enabled": issue_enabled,
            "included_in_calibrated_area": False,
            "included_in_enhanced_area": issue_enabled,
            "structural_estimate_area": issue_structural_area,
            "structural_estimate_area_dc_corner": issue_structural_dc_area,
            "storage_equivalent_bits": storage_equivalent_bits,
            "control_equivalent_bits": control_equivalent_bits,
            "storage_proxy_coefficient_name": storage_coefficient_name,
            "storage_proxy_coefficient_area_per_bit_dc": storage_coefficient,
            "control_proxy_coefficient_name": control_coefficient_name,
            "control_proxy_coefficient_area_per_bit_dc": control_coefficient,
            "coefficient_use": "extrapolated_from_closest_retained_per_bit_fits",
            "reason": "no retained DC points contain the scalar FP issue pipeline",
        },
        "loop_address_generator_counts": loop_counts,
        "loop_address_generator_area_accounting": {
            "status": "structural_census_only",
            "evidence_tier": STRUCTURAL_ESTIMATE,
            "enabled": loop_enabled,
            "extrapolated": True,
            "included_in_calibrated_area": False,
            "included_in_enhanced_area": loop_enabled,
            "structural_estimate_area": loop_structural_area,
            "structural_estimate_area_dc_corner": loop_structural_dc_area,
            "storage_equivalent_bits": loop_storage_equivalent_bits,
            "control_equivalent_bits": loop_control_equivalent_bits,
            "storage_proxy_coefficient_name": storage_coefficient_name,
            "storage_proxy_coefficient_area_per_bit_dc": storage_coefficient,
            "control_proxy_coefficient_name": control_coefficient_name,
            "control_proxy_coefficient_area_per_bit_dc": control_coefficient,
            "coefficient_use": "extrapolated_from_closest_retained_per_bit_fits",
            "reason": "no retained DC points contain the loop address generator",
        },
        "evidence": evidence,
    }
