"""Validation gates for the precision-aware full-chip area model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from . import estimate_area, estimate_system_area
from .fit import build_artifact
from .hbm_interface import estimate_hbm_interface_area
from .link import estimate_link_phy_area
from .matrix import estimate_matrix_machine_area
from .scalar import (
    estimate_scalar_area,
    fp_issue_pipeline_counts,
    loop_address_generator_counts,
)
from .top import estimate_top_area
from .vector import estimate_vector_area, segment_parallel_reduction_counts


def _config(
    mlen: int = 64,
    blen: int = 8,
    precision: str = "MXINT4",
) -> dict[str, object]:
    return {
        "MLEN": mlen,
        "BLEN": blen,
        "VLEN": mlen,
        "HLEN": max(blen, min(64, mlen)),
        "BLOCK_DIM": blen,
        "ACT_WIDTH": precision,
        "KV_WIDTH": precision,
        "WEIGHT_WIDTH": precision,
        "FP_SETTING": "FP_E5M6",
        "INT_DATA_WIDTH": 32,
        "MATRIX_SRAM_DEPTH": max(32, 4 * mlen),
        "VECTOR_SRAM_DEPTH": max(32, 4 * mlen),
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 68,
        "HBM_M_Prefetch_Amount": 16,
        "HBM_V_Prefetch_Amount": 16,
        "HBM_V_Writeback_Amount": 16,
    }


def _precision(bits: int) -> dict[str, object]:
    token = f"MXINT{bits}"
    return {
        "attn_elem": bits,
        "ffn_elem": bits,
        "kv_elem": bits,
        "attn_label": token,
        "ffn_label": token,
        "kv_label": token,
        "m_bits": bits,
    }


def _hardware() -> SimpleNamespace:
    return SimpleNamespace(
        MLEN=1024,
        BLEN=4,
        VLEN=1024,
        HLEN=128,
        MATRIX_SRAM_SIZE=4096,
        VECTOR_SRAM_SIZE=4096,
        HBM_M_Prefetch_Amount=1024,
        HBM_V_Prefetch_Amount=16,
        HBM_V_Writeback_Amount=16,
    )


def test_matrix_reference_anchor_is_preserved() -> None:
    observed = estimate_matrix_machine_area(_config(1024, 4))["area"] / 1e6
    assert observed == pytest.approx(0.237, rel=1e-12)


def test_vector_census_tracks_lane_replication() -> None:
    small = estimate_vector_area(_config(32, 4))["area"]
    large = estimate_vector_area(_config(64, 4))["area"]
    assert large > small > 0.0


def test_segment_parallel_reduction_census_matches_the_rtl_structure() -> None:
    config = _config(64, 4)
    config.update({"HLEN": 32, "REDUCTION_SEGMENTS": 2})
    counts = segment_parallel_reduction_counts(config)
    assert counts == {
        "segments": 2,
        "segment_width": 32,
        "tree_levels": 6,
        "fp_adders": 64,
        "fp_max_units": 64,
        "arithmetic_units_added": 0,
        "output_fifo_packets": 44,
        "output_fifo_result_words": 88,
        "output_fifo_data_bits": 1056,
        "output_fifo_pointer_bits": 12,
        "output_fifo_count_bits": 12,
        "output_fifo_control_bits": 24,
        "drain_mux_inputs": 2,
        "drain_mux_equivalent_bits": 12,
        "drain_index_bits": 1,
        "destination_adder_bits": 3,
        "estimated_equivalent_bits": 1096,
    }

    estimate = estimate_vector_area(config)
    accounting = estimate["segment_parallel_reduction_area_accounting"]
    assert accounting["status"] == "structural_census_only"
    assert accounting["evidence_tier"] == "declared_structural_estimate"
    assert accounting["included_in_calibrated_area"] is False
    assert accounting["included_in_enhanced_area"] is True
    assert accounting["structural_estimate_area"] > 0.0
    assert estimate["enhanced_area"] == pytest.approx(
        estimate["calibrated_area"] + accounting["structural_estimate_area"]
    )
    assert estimate["area"] == pytest.approx(estimate["enhanced_area"])


def test_segment_parallel_reduction_cost_is_in_the_enabled_chip_total() -> None:
    baseline_config = _config(64, 4)
    baseline_config["HLEN"] = 32
    enhanced_config = dict(baseline_config)
    enhanced_config["REDUCTION_SEGMENTS"] = 2

    baseline = estimate_area(baseline_config)
    enhanced = estimate_area(enhanced_config)
    extension = enhanced["vector_machine"][
        "segment_parallel_reduction_area_accounting"
    ]["structural_estimate_area"]
    assert extension > 0.0
    assert enhanced["area"] - baseline["area"] == pytest.approx(extension)
    assert enhanced["evidence_tier"] == "declared_structural_estimate"


def test_scalar_census_tracks_vector_buffer() -> None:
    small = estimate_scalar_area(_config(32, 4))["area"]
    large = estimate_scalar_area(_config(64, 4))["area"]
    assert large > small > 0.0


def test_scalar_fp_issue_census_matches_the_rtl_structure() -> None:
    counts = fp_issue_pipeline_counts(_config())
    assert counts == {
        "fp_reorder_entries": 4,
        "fp_reorder_entry_bits": 17,
        "fp_reorder_entry_storage_bits": 68,
        "fp_reorder_pointer_bits": 4,
        "fp_reorder_occupancy_bits": 3,
        "fp_reorder_sequential_bits": 75,
        "fp_issue_reservation_bits": 2,
        "fp_pending_write_mask_bits": 8,
        "fp_completion_channels": 2,
        "fp_retirement_ports": 1,
    }
    evidence = estimate_scalar_area(_config())
    assert evidence["fp_issue_pipeline_counts"] == counts
    accounting = evidence["fp_issue_pipeline_area_accounting"]
    assert accounting["status"] == "structural_census_only"
    assert accounting["evidence_tier"] == "declared_structural_estimate"
    assert accounting["enabled"] is False
    assert accounting["included_in_calibrated_area"] is False
    assert accounting["included_in_enhanced_area"] is False
    assert accounting["storage_equivalent_bits"] == 68
    assert accounting["control_equivalent_bits"] == 9
    assert accounting["structural_estimate_area"] > 0.0
    assert evidence["area"] == pytest.approx(evidence["calibrated_area"])


def test_scalar_fp_issue_cost_is_in_the_enabled_chip_total() -> None:
    baseline_config = _config(64, 4)
    enhanced_config = dict(baseline_config)
    enhanced_config["SCALAR_FP_ISSUE_PIPELINE"] = True

    baseline = estimate_area(baseline_config)
    enhanced = estimate_area(enhanced_config)
    extension = enhanced["scalar_machine"][
        "fp_issue_pipeline_area_accounting"
    ]["structural_estimate_area"]
    assert extension > 0.0
    assert enhanced["area"] - baseline["area"] == pytest.approx(extension)
    assert enhanced["scalar_machine"]["enhanced_area"] == pytest.approx(
        enhanced["scalar_machine"]["calibrated_area"] + extension
    )
    assert enhanced["evidence_tier"] == "declared_structural_estimate"


def test_loop_address_generator_census_matches_the_rtl_structure() -> None:
    counts = loop_address_generator_counts(_config())
    assert counts == {
        "address_width": 32,
        "register_index_width": 4,
        "register_count": 16,
        "loop_count_width": 22,
        "max_loop_depth": 4,
        "address_streams_per_loop": 4,
        "shadow_value_bits": 512,
        "shadow_valid_bits": 16,
        "shadow_state_bits": 528,
        "pending_entry_bits": 27,
        "pending_state_bits": 108,
        "frame_counter_bits": 104,
        "frame_stream_bits": 432,
        "frame_state_bits": 536,
        "stack_pointer_bits": 3,
        "status_bits": 2,
        "sequential_bits": 1177,
        "storage_state_bits": 1040,
        "control_state_bits": 137,
        "address_adders": 4,
        "address_adder_bits": 128,
        "counter_decrementers": 1,
        "counter_decrement_bits": 22,
        "shadow_read_ports": 3,
        "shadow_read_mux_inputs": 16,
        "shadow_read_mux_equivalent_bits": 1440,
        "pending_register_comparators": 4,
        "pending_comparator_bits": 16,
        "datapath_equivalent_bits": 1590,
        "storage_equivalent_bits": 2630,
        "control_equivalent_bits": 153,
    }

    config = _config()
    config["ENABLE_LOOP_ADDRESS_GENERATOR"] = True
    estimate = estimate_scalar_area(config)
    accounting = estimate["loop_address_generator_area_accounting"]
    assert accounting["status"] == "structural_census_only"
    assert accounting["evidence_tier"] == "declared_structural_estimate"
    assert accounting["extrapolated"] is True
    assert accounting["included_in_calibrated_area"] is False
    assert accounting["included_in_enhanced_area"] is True
    assert accounting["structural_estimate_area"] > 0.0
    assert estimate["enhanced_area"] == pytest.approx(
        estimate["calibrated_area"] + accounting["structural_estimate_area"]
    )


def test_loop_address_generator_cost_is_in_the_enhanced_chip_total() -> None:
    baseline_config = _config(64, 4)
    enhanced_config = dict(baseline_config)
    enhanced_config["ENABLE_LOOP_ADDRESS_GENERATOR"] = True

    baseline = estimate_area(baseline_config)
    enhanced = estimate_area(enhanced_config)
    accounting = enhanced["scalar_machine"][
        "loop_address_generator_area_accounting"
    ]
    extension = accounting["structural_estimate_area"]
    assert extension > 0.0
    assert enhanced["area"] - baseline["area"] == pytest.approx(extension)
    assert enhanced["enhancement_area"] == pytest.approx(extension)
    assert enhanced["calibrated_chip_area"] + extension == pytest.approx(
        enhanced["enhanced_chip_area"]
    )
    assert enhanced["evidence_tier"] == "declared_structural_estimate"


def test_hbm_interface_has_explicit_precision_response() -> None:
    narrow = estimate_hbm_interface_area(_config(64, 8, "MXINT4"))["area"]
    wide = estimate_hbm_interface_area(_config(64, 8, "MXINT8"))["area"]
    assert narrow > wide > 0.0


def test_top_overhead_is_fitted_and_positive() -> None:
    estimate = estimate_top_area(_config(64, 8))
    assert estimate["area"] > 0.0
    assert estimate["evidence"]["raw_dc_reports_available"] is False


def test_link_projection_matches_published_density_scaling() -> None:
    estimate = estimate_link_phy_area()
    expected = 7200.0 / (552.0 * (5.0 / 7.0) ** 2)
    assert estimate["area_mm2"] == pytest.approx(expected)
    assert estimate["area_mm2"] == pytest.approx(25.5652173913)
    assert estimate["evidence"]["synthesized_for_plena"] is False


def test_full_chip_breakdown_is_complete_and_additive() -> None:
    estimate = estimate_area(_config(1024, 4))
    required = {
        "MatrixMachine",
        "VectorMachine",
        "ScalarMachine",
        "HBMInterface",
        "TopOverhead",
        "MatrixSRAM",
        "VectorSRAM",
        "ScalarIntSRAM",
        "ScalarFPSRAM",
    }
    assert set(estimate["breakdown"]) == required
    assert sum(estimate["breakdown"].values()) == pytest.approx(estimate["area"])
    logic_base = sum(
        estimate["breakdown"][name]
        for name in ("MatrixMachine", "VectorMachine", "ScalarMachine", "HBMInterface")
    )
    assert estimate["top_overhead_ratio"] == pytest.approx(
        estimate["top_overhead_area"] / logic_base
    )


def test_system_area_counts_chip_side_ports() -> None:
    chip = estimate_area(_config(1024, 4))
    system = estimate_system_area(_config(1024, 4), chip_count=4, ports_per_chip=2)
    expected = 4 * (chip["area"] + 2 * system["link_phy_area_per_port"])
    assert system["area"] == pytest.approx(expected)


def test_every_calibrated_block_has_at_least_quarter_holdout() -> None:
    reports = build_artifact()["report"]
    holdouts = {
        "matrix_mxint": reports["mxint"],
        "matrix_mxfp": reports["mxfp"],
        **reports["full_chip_blocks"],
    }
    assert all(report["holdout_fraction"] >= 0.25 for report in holdouts.values())
    assert max(report["p95_abs_error_pct"] for report in holdouts.values()) <= 10.0
    assert reports["full_chip_blocks"]["full_chip"]["p95_abs_error_pct"] <= 1.0


def test_disaggregated_bridge_returns_full_chip_breakdown() -> None:
    from analytic_models.disagg_serve.area import area_mm2

    estimate = area_mm2("calibrated", _hardware(), _precision(4), return_breakdown=True)
    assert estimate["area_mm2"] > 0.237
    assert "VectorMachine" in estimate["breakdown_mm2"]
    assert math.isclose(sum(estimate["breakdown_mm2"].values()), estimate["area_mm2"])


def test_disaggregated_bridge_propagates_loop_address_generator_enablement() -> None:
    from analytic_models.disagg_serve.area import area_mm2

    baseline_hardware = _hardware()
    enhanced_hardware = _hardware()
    enhanced_hardware.ENABLE_LOOP_ADDRESS_GENERATOR = True
    baseline = area_mm2(
        "calibrated", baseline_hardware, _precision(4), return_breakdown=True
    )
    enhanced = area_mm2(
        "calibrated", enhanced_hardware, _precision(4), return_breakdown=True
    )
    extension = enhanced["scalar_machine"][
        "loop_address_generator_area_accounting"
    ]["structural_estimate_area"]
    assert enhanced["area_mm2"] - baseline["area_mm2"] == pytest.approx(
        extension / 1e6
    )
    assert enhanced["evidence_tier"] == "declared_structural_estimate"


def test_disaggregated_bridge_propagates_decode_path_enhancements() -> None:
    from analytic_models.disagg_serve.area import area_mm2

    hardware = _hardware()
    hardware.REDUCTION_SEGMENTS = hardware.VLEN // hardware.HLEN
    hardware.SCALAR_FP_ISSUE_PIPELINE = True
    estimate = area_mm2(
        "calibrated", hardware, _precision(4), return_breakdown=True
    )

    vector_accounting = estimate["vector_machine"][
        "segment_parallel_reduction_area_accounting"
    ]
    scalar_accounting = estimate["scalar_machine"][
        "fp_issue_pipeline_area_accounting"
    ]
    assert vector_accounting["enabled"] is True
    assert vector_accounting["included_in_enhanced_area"] is True
    assert scalar_accounting["enabled"] is True
    assert scalar_accounting["included_in_enhanced_area"] is True
    assert estimate["evidence_tier"] == "declared_structural_estimate"


def test_iso_area_solver_makes_precision_a_geometry_parameter() -> None:
    from analytic_models.disagg_serve.area import solve_area_budget

    narrow = solve_area_budget(
        20.0,
        _hardware(),
        _precision(4),
        mlen_candidates=(128, 256, 512, 1024),
        blen_candidates=(4, 8, 16, 32, 64),
        hidden_size=8192,
    )
    wide = solve_area_budget(
        20.0,
        _hardware(),
        _precision(8),
        mlen_candidates=(128, 256, 512, 1024),
        blen_candidates=(4, 8, 16, 32, 64),
        hidden_size=8192,
    )
    assert narrow["matrix_multipliers"] > wide["matrix_multipliers"]
    assert narrow["area_mm2"] <= 20.0
    assert wide["area_mm2"] <= 20.0
    assert narrow["VLEN"] == narrow["MLEN"]
    assert wide["VLEN"] == wide["MLEN"]


def test_solver_rejects_an_unreachable_budget() -> None:
    from analytic_models.disagg_serve.area import solve_area_budget

    with pytest.raises(ValueError, match="no legal geometry fits"):
        solve_area_budget(
            0.001,
            _hardware(),
            _precision(4),
            mlen_candidates=(128,),
            blen_candidates=(4,),
        )
