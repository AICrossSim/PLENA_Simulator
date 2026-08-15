"""Unit and integration tests for area equations, parsers, and job bookkeeping."""

from __future__ import annotations

import csv
import json
import math

import pytest

from analytic_models.area_new import (
    PrecisionError,
    derive_compute_sides,
    estimate_area,
    estimate_address_generation_unit_area,
    estimate_hbm_system_area,
    estimate_matrix_machine_area,
    estimate_scalar_machine_area,
    estimate_sram_area,
    estimate_vector_machine_area,
    parse_precision,
)
from analytic_models.area_new.mxint_model import pe_area as mxint_pe_area
from analytic_models.area_new.mxint_model import estimate as estimate_mxint_matrix
from analytic_models.area_new.matrix_structural_model import (
    MODEL_VERSION as MATRIX_STRUCTURAL_MODEL_VERSION,
    load_artifact as load_matrix_structural_artifact,
    structural_counts,
)
from analytic_models.area_new.scripts.calibration_csv import latest_by_key, latest_complete_rows, write_rows
from analytic_models.area_new.scripts.calibration_runtime import classify_failure, stable_job_key
from analytic_models.area_new.scripts.merge_calibration_attempts import stable_key as calibration_attempt_key
from analytic_models.area_new.scripts.run_matrix_machine_calibration import parse_hierarchy_area
from analytic_models.area_new.scripts.run_vector_machine_calibration import (
    ASAP7_TARGET_LIBRARIES,
    Point as VectorPoint,
    build_plan as build_vector_plan,
    patch_vector_config,
    prepare_worker_techlib,
)
from analytic_models.area_new.scripts.fit_vector_rtl_v6_delta import (
    fit_artifact as fit_vector_rtl_v6_artifact,
    structural_features as rtl_v6_structural_features,
)
from analytic_models.area_new import scalar_model, vector_model
from analytic_models.area_new.scripts.validate_full_chip_area_proxy import (
    _hierarchy_from_source,
    parse_full_chip_hierarchy,
)
from transactional_emulator.testbench.rtl_timing.build_rtl_v6_timing_artifact import (
    build as build_rtl_v6_timing_artifact,
)


def test_parse_mxint_and_mxfp() -> None:
    assert parse_precision("MXINT2").bits == 2
    assert parse_precision("MXINT_4").bits == 4
    p = parse_precision("MXFP_E4M3")
    assert (p.exp, p.mant, p.element_width) == (4, 3, 8)
    q = parse_precision({"kind": "MXFP", "exp": 5, "mant": 2, "scale_width": 8})
    assert q.name == "MXFP_E5M2"


def test_prepare_worker_techlib_links_external_databases(tmp_path) -> None:
    techlib_root = tmp_path / "technology-lib"
    library_dir = techlib_root / "asap7" / "asap7sc7p5t_28"
    library_dir.mkdir(parents=True)
    for name in ASAP7_TARGET_LIBRARIES:
        (library_dir / name).write_bytes(b"db")

    worker_rtl = tmp_path / "worker"
    (worker_rtl / "tools" / "synopsys").mkdir(parents=True)
    prepare_worker_techlib(worker_rtl, techlib_root)

    target = worker_rtl / "tools" / "synopsys" / "lib"
    assert target.is_symlink()
    assert target.resolve() == techlib_root.resolve()


def test_prepare_worker_techlib_rejects_incomplete_database_set(tmp_path) -> None:
    techlib_root = tmp_path / "technology-lib"
    (techlib_root / "asap7" / "asap7sc7p5t_28").mkdir(parents=True)
    worker_rtl = tmp_path / "worker"

    with pytest.raises(FileNotFoundError, match="technology library is incomplete"):
        prepare_worker_techlib(worker_rtl, techlib_root)


def test_calibration_attempt_key_accepts_scheduler_and_vector_rows() -> None:
    assert calibration_attempt_key({"job_key": "scheduler-key"}) == "scheduler-key"
    assert calibration_attempt_key({"point_key": "vector-key"}) == "vector-key"
    assert calibration_attempt_key({}) == ""


def test_derive_mxint_sides() -> None:
    sides = derive_compute_sides("MXINT2", "MXINT4", "MXINT4")
    assert sides["mode"] == "mxint"
    assert sides["t_bits"] == 4
    assert sides["l_bits"] == 2
    sides = derive_compute_sides("MXINT8", "MXINT4", "MXINT4")
    assert sides["t_bits"] == 4
    assert sides["l_bits"] == 8
    sides = derive_compute_sides("MXINT4", "MXINT8", "MXINT4")
    assert sides["t_bits"] == 8
    assert sides["l_bits"] == 4


def test_derive_mxfp_sides() -> None:
    sides = derive_compute_sides("MXFP_E4M3", "MXFP_E5M2", "MXFP_E4M3")
    assert sides["mode"] == "mxfp"
    assert (sides["t_exp"], sides["t_mant"]) == (5, 3)
    assert (sides["l_exp"], sides["l_mant"]) == (4, 3)


def test_reject_mixed_mode_and_unsupported_act3() -> None:
    with pytest.raises(PrecisionError):
        derive_compute_sides("MXINT4", "MXFP_E4M3", "MXINT4")
    with pytest.raises(PrecisionError):
        derive_compute_sides("MXINT3", "MXINT4", "MXINT4")


def test_mxint_pe_area_monotonic() -> None:
    assert mxint_pe_area(4, 2) < mxint_pe_area(4, 4) < mxint_pe_area(4, 8)
    assert mxint_pe_area(4, 4) < mxint_pe_area(8, 8)


def test_estimate_matrix_machine_area() -> None:
    small = estimate_matrix_machine_area(
        {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "MLEN": 16, "BLEN": 4}
    )
    large = estimate_matrix_machine_area(
        {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "MLEN": 32, "BLEN": 4}
    )
    assert small["area"] > 0
    assert large["area"] > small["area"]


def test_loop_agu_paired_delta_is_mode_selective() -> None:
    default = estimate_address_generation_unit_area(
        {"INT_DATA_WIDTH": 32}
    )
    current = estimate_address_generation_unit_area(
        {"address_generation_mode": "loop-agu-v1", "INT_DATA_WIDTH": 32}
    )
    legacy = estimate_address_generation_unit_area(
        {"address_generation_mode": "legacy", "INT_DATA_WIDTH": 32}
    )

    assert default["area"] == pytest.approx(current["area"])
    assert default["inputs"]["address_generation_mode"] == "loop-agu-v1"
    assert current["area"] == pytest.approx(1891.25927)
    assert current["breakdown"]["AguAffineSidecar"] == pytest.approx(1722.568673)
    assert current["calibration_status"] == "mapped_dc_paired_delta"
    assert current["timing"]["timing_status"] == "met_with_low_margin"
    assert legacy["area"] == 0.0

    with pytest.raises(ValueError, match="unsupported address_generation_mode"):
        estimate_address_generation_unit_area(
            {"address_generation_mode": "loop-agu-v2", "INT_DATA_WIDTH": 32}
        )


def test_complete_area_includes_agu_only_for_agu_mode() -> None:
    common = {
        "MLEN": 16,
        "VLEN": 16,
        "BLEN": 4,
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "FP_SETTING": "FP_E5M6",
        "MATRIX_SRAM_DEPTH": 32,
        "VECTOR_SRAM_DEPTH": 32,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
    }
    current = estimate_area({**common, "address_generation_mode": "loop-agu-v1"})
    legacy = estimate_area({**common, "address_generation_mode": "legacy"})

    assert current["area_breakdown"]["AddressGenerationUnit"] == pytest.approx(
        1891.25927
    )
    assert legacy["area_breakdown"]["AddressGenerationUnit"] == 0.0
    assert current["area"] > legacy["area"]


def test_matrix_precision_trends_hold_in_calibrated_shape_domain() -> None:
    common = {"MLEN": 64, "BLEN": 16}
    mxint4 = estimate_matrix_machine_area(
        {
            **common,
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
        }
    )
    mxint8 = estimate_matrix_machine_area(
        {
            **common,
            "ACT_WIDTH": "MXINT8",
            "KV_WIDTH": "MXINT8",
            "WEIGHT_WIDTH": "MXINT8",
        }
    )
    mxfp4 = estimate_matrix_machine_area(
        {
            **common,
            "ACT_WIDTH": "MXFP_E1M2",
            "KV_WIDTH": "MXFP_E1M2",
            "WEIGHT_WIDTH": "MXFP_E1M2",
        }
    )
    mxfp8 = estimate_matrix_machine_area(
        {
            **common,
            "ACT_WIDTH": "MXFP_E4M3",
            "KV_WIDTH": "MXFP_E4M3",
            "WEIGHT_WIDTH": "MXFP_E4M3",
        }
    )

    assert mxint4["area"] < mxint8["area"]
    assert mxfp4["area"] < mxfp8["area"]
    assert mxint4["area"] < mxfp4["area"]
    assert mxint4["calibration_in_domain"] is True


def test_single_k_split_matrix_shape_is_marked_extrapolated() -> None:
    result = estimate_matrix_machine_area(
        {
            "MLEN": 512,
            "BLEN": 512,
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
        }
    )

    assert result["calibration_in_domain"] is False
    assert result["matrix_k_splits"] == 1
    assert result["reduce_tree_area"] == 0.0
    assert any("MLEN/BLEN=1" in item for item in result["calibration_warnings"])


def test_structural_matrix_exact_census_and_single_split_reduction() -> None:
    counts = structural_counts(1024, 1024, 12)
    assert counts["array_count"] == 1
    assert counts["pe_count"] == 1024 * 1024
    assert counts["reduce_node_count"] == 0
    assert counts["output_cell_count"] == 1024 * 1024
    assert counts["result_buffer_bits"] == 1024 * 1024 * 12


def test_installed_matrix_model_is_structural_v4() -> None:
    result = estimate_matrix_machine_area(
        {
            "MLEN": 128,
            "BLEN": 16,
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
        }
    )
    assert result["area_model"] == MATRIX_STRUCTURAL_MODEL_VERSION
    assert result["structural_counts"]["pe_count"] == 128 * 16
    assert result["area_uncertainty_p10"] < result["area"]
    assert result["area_uncertainty_p90"] > result["area"]
    assert "residual_scale" in result["uncertainty_basis"]


def test_structural_matrix_calibration_acceptance_is_committed() -> None:
    artifact = load_matrix_structural_artifact()
    assert artifact is not None
    assert artifact["acceptance"]["pass"] is True
    assert artifact["physical_invariants"]["pass"] is True
    for family in artifact["families"].values():
        diagnostics = family["diagnostics"]
        assert diagnostics["train_mape_pct"] <= 10.0
        assert diagnostics["grouped_holdout_median_error_pct"] <= 12.0
        assert diagnostics["grouped_holdout_p95_error_pct"] <= 20.0


def test_matrix_machine_hierarchy_parser(tmp_path) -> None:
    report = tmp_path / "area.rpt"
    report.write_text(
        """
Hierarchical area distribution
------------------------------
matrix_machine                    10709.5785    100.0    97.7006   109.0584  0.0000  matrix_machine
gen_mxint_systolic_mcu_matrix_compute_unit
                                  10286.5398     96.0    16.4754     3.7908  0.0000  mxint_systolic_mcu_1
gen_mxint_systolic_mcu_matrix_compute_unit/cross_k_reduce
                                   3825.0338     35.7  2306.2644   532.4616  0.0000  mxint_sum_across_1
gen_mxint_systolic_mcu_matrix_compute_unit/g_acc_row_0__g_acc_col_0__acc
                                    579.3948      5.4   100.0000   200.0000  0.0000  fp_fix_accumulator_1
gen_mxint_systolic_mcu_matrix_compute_unit/g_fp_row_0__g_fp_col_0__acc_to_fp
                                    120.2704      1.0    50.0000    70.0000  0.0000  mxint_acc_2_fp_1
matrix_element_buffer                 23.0801      0.2     4.0000    19.0000  0.0000  register_slice_wo_hs_1
result_buffer                         68.7447      0.6    12.0000    56.0000  0.0000  register_slice_wo_hs_2
"""
    )
    parsed = parse_hierarchy_area(report)
    assert parsed["hier_total_area"] == pytest.approx(10709.5785)
    assert parsed["hier_compute_unit_area"] == pytest.approx(10286.5398)
    assert parsed["hier_reduce_area"] == pytest.approx(3825.0338)
    assert parsed["hier_accum_area"] == pytest.approx(579.3948)
    assert parsed["hier_output_accumulator_area"] == pytest.approx(579.3948)
    assert parsed["hier_output_conversion_area"] == pytest.approx(120.2704)
    assert parsed["hier_result_buffer_area"] == pytest.approx(68.7447)
    assert parsed["hier_io_pipeline_area"] == pytest.approx(23.0801)
    assert parsed["hier_top_glue_area"] == pytest.approx(423.0387)


def test_mxint_matrix_v2_breakdown_adds_up() -> None:
    coeffs = {
        "pe_c0": 1.0,
        "pe_c_tl": 1.0,
        "pe_c_sum": 1.0,
        "mini_pe_scale": 1.0,
        "mini_a_scale": 1.0,
        "mini_a_grid": 0.0,
        "mini_a0": 0.0,
        "mm_reduce_c": 2.0,
        "mm_accum_c": 3.0,
        "mm_top_c": 4.0,
    }
    result = estimate_mxint_matrix({"t_bits": 4, "l_bits": 2, "MLEN": 16, "BLEN": 4, "scale_width": 8}, coeffs)
    breakdown = result["breakdown"]
    assert result["area_model"] == "matrix_machine_mxint_hierarchy_residual_v2"
    total_parts = (
        breakdown["mini_array_stack_area"]
        + breakdown["reduce_tree_area"]
        + breakdown["accumulator_grid_area"]
        + breakdown["top_glue_area"]
    )
    assert breakdown["matrix_machine_area"] == pytest.approx(total_parts)


def test_estimate_sram_area_width_mapping() -> None:
    small = estimate_sram_area(
        {
            "ACT_WIDTH": "MXINT2",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 32,
            "VLEN": 32,
            "BLEN": 4,
            "MATRIX_SRAM_DEPTH": 64,
            "VECTOR_SRAM_DEPTH": 64,
            "INT_DATA_WIDTH": 16,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        }
    )
    large = estimate_sram_area(
        {
            "ACT_WIDTH": "MXINT8",
            "KV_WIDTH": "MXINT8",
            "WEIGHT_WIDTH": "MXINT8",
            "MLEN": 32,
            "VLEN": 32,
            "BLEN": 4,
            "MATRIX_SRAM_DEPTH": 64,
            "VECTOR_SRAM_DEPTH": 64,
            "INT_DATA_WIDTH": 64,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        }
    )
    assert small["area"] > 0
    assert large["area"] > small["area"]
    assert small["area_sram_inputs"]["matrix"]["element_width"] == 4
    assert large["area_sram_inputs"]["matrix"]["element_width"] == 8
    assert small["area_sram_model"] in {"asap7_sram_macro_tiling", "fitted_linear_coefficients"}


def test_rtl_v3_vector_scalar_delta_overlays_are_explicit(tmp_path) -> None:
    vector_delta = tmp_path / "vector_delta.json"
    vector_delta.write_text(
        json.dumps({"coefficients": {"delta_vlen_width": 0.5, "delta_const": 10.0}})
    )
    scalar_delta = tmp_path / "scalar_delta.json"
    scalar_delta.write_text(
        json.dumps(
            {
                "coefficients": {
                    "delta_vlen_width": 0.25,
                    "delta_fp_width": 2.0,
                    "delta_const": 5.0,
                }
            }
        )
    )
    common = {"VLEN": 32, "FP_SETTING": "FP_E5M6"}
    vector_old = vector_model.estimate_vector_machine_area(
        {**common, "vector_scalar_area_version": "rtl-v2"},
        rtl_v3_delta_path=vector_delta,
    )
    vector_new = vector_model.estimate_vector_machine_area(
        {**common, "vector_scalar_area_version": "rtl-v3"},
        rtl_v3_delta_path=vector_delta,
    )
    assert vector_new["area"] - vector_old["area"] == pytest.approx(0.5 * 32 * 12 + 10.0)
    assert vector_new["vector_scalar_area_calibration_status"] == "calibrated_rtl_v3_delta_overlay"

    scalar_config = {**common, "MLEN": 32, "INT_DATA_WIDTH": 32}
    scalar_old = scalar_model.estimate_scalar_machine_area(
        {**scalar_config, "vector_scalar_area_version": "rtl-v2"},
        rtl_v3_delta_path=scalar_delta,
    )
    scalar_new = scalar_model.estimate_scalar_machine_area(
        {**scalar_config, "vector_scalar_area_version": "rtl-v3"},
        rtl_v3_delta_path=scalar_delta,
    )
    assert scalar_new["area"] - scalar_old["area"] == pytest.approx(
        0.25 * 32 * 12 + 2.0 * 12 + 5.0
    )
    assert scalar_new["breakdown"]["ScalarRTLv3PipelineDelta"] > 0.0


def test_installed_rtl_v3_delta_calibration_is_active() -> None:
    vector = estimate_vector_machine_area(
        {"VLEN": 32, "FP_SETTING": "FP_E5M6", "vector_scalar_area_version": "rtl-v3"}
    )
    scalar = estimate_scalar_machine_area(
        {
            "MLEN": 32,
            "VLEN": 32,
            "INT_DATA_WIDTH": 32,
            "FP_SETTING": "FP_E5M6",
            "vector_scalar_area_version": "rtl-v3",
        }
    )

    assert vector["rtl_v3_delta_area"] > 0.0
    assert scalar["rtl_v3_delta_area"] > 0.0
    assert vector["vector_scalar_area_calibration_status"] == "calibrated_rtl_v3_delta_overlay"
    assert scalar["vector_scalar_area_calibration_status"] == "calibrated_rtl_v3_delta_overlay"


def test_rtl_v4_vector_area_is_cumulative_and_never_silently_zero(tmp_path) -> None:
    vector_v3_delta = tmp_path / "vector_v3_delta.json"
    vector_v3_delta.write_text(
        json.dumps({"coefficients": {"delta_vlen_width": 0.5, "delta_const": 10.0}})
    )
    missing_v4 = tmp_path / "missing_v4.json"
    common = {"VLEN": 32, "FP_SETTING": "FP_E5M6"}
    v3 = vector_model.estimate_vector_machine_area(
        {**common, "vector_scalar_area_version": "rtl-v3"},
        rtl_v3_delta_path=vector_v3_delta,
    )
    pending_v4 = vector_model.estimate_vector_machine_area(
        {**common, "vector_scalar_area_version": "rtl-v4"},
        rtl_v3_delta_path=vector_v3_delta,
        rtl_v4_delta_path=missing_v4,
    )
    assert pending_v4["area"] == pytest.approx(v3["area"])
    assert pending_v4["rtl_v3_delta_area"] == pytest.approx(v3["rtl_v3_delta_area"])
    assert pending_v4["rtl_v4_delta_area"] == 0.0
    assert pending_v4["vector_scalar_area_calibration_status"] == "recalibration_pending_rtl_v4"
    assert any("excludes the unknown rtl-v4 increment" in item for item in pending_v4["rtl_v3_delta_calibration_warnings"])

    vector_v4_delta = tmp_path / "vector_v4_delta.json"
    vector_v4_delta.write_text(
        json.dumps(
            {
                "metadata": {"status": "fitted_from_paired_rtl_v4_dc"},
                "coefficients": {
                    "compact_stats_simd_const": 100.0,
                    "compact_stats_simd_fp_width": 2.0,
                    "reduction_overwrite_control_const": 3.0,
                },
            }
        )
    )
    calibrated_v4 = vector_model.estimate_vector_machine_area(
        {**common, "vector_scalar_area_version": "rtl-v4"},
        rtl_v3_delta_path=vector_v3_delta,
        rtl_v4_delta_path=vector_v4_delta,
    )
    assert calibrated_v4["area"] - v3["area"] == pytest.approx(100.0 + 2.0 * 12 + 3.0)
    assert calibrated_v4["breakdown"]["CompactStatsSIMD"] == pytest.approx(124.0)
    assert calibrated_v4["breakdown"]["ReductionOverwriteControl"] == pytest.approx(3.0)


def test_installed_rtl_v4_delta_calibration_is_active() -> None:
    vector = estimate_vector_machine_area(
        {
            "VLEN": 2048,
            "FP_SETTING": "FP_E5M6",
            "vector_scalar_area_version": "rtl-v4",
        }
    )

    assert vector["rtl_v4_delta_area"] > 0.0
    assert vector["breakdown"]["CompactStatsSIMD"] > 0.0
    assert (
        vector["vector_scalar_area_calibration_status"]
        == "fitted_from_paired_rtl_v4_dc"
    )


def test_rtl_v5_compact_area_scales_with_configured_lane_tier() -> None:
    areas = []
    for lanes in (4, 8, 16, 32, 64):
        vector = estimate_vector_machine_area(
            {
                "VLEN": 8192,
                "FP_SETTING": "FP_E5M6",
                "COMPACT_STATS_LANES": lanes,
                "vector_scalar_area_version": "rtl-v5",
            }
        )
        assert vector["rtl_v5_delta_area"] > 0.0
        assert vector["inputs"]["COMPACT_STATS_LANES"] == lanes
        areas.append(vector["breakdown"]["CompactStatsSIMD"])
    assert areas == sorted(areas)
    assert len(set(areas)) == len(areas)


def test_rtl_v6_softmax_row_area_scales_with_lane_tier() -> None:
    totals = []
    for lanes in (1, 2, 4, 8):
        vector = estimate_vector_machine_area(
            {
                "VLEN": 2048,
                "HLEN": 128,
                "FP_SETTING": "FP_E5M6",
                "COMPACT_STATS_LANES": 16,
                "SOFTMAX_ROW_LANES": lanes,
                "vector_scalar_area_version": "rtl-v6",
            }
        )
        assert vector["rtl_v6_delta_status"] == "fitted_from_paired_rtl_v6_dc"
        assert vector["rtl_v6_delta_coefficients_source"].endswith(
            "vector_rtl_v6_delta_coefficients.json"
        )
        assert vector["breakdown"]["PackedPVAccumulator"] > 0.0
        assert vector["breakdown"]["SoftmaxStateSIMD"] > 0.0
        assert vector["rtl_v6_breakdown_fidelity"] == (
            "paired_total_leaf_guided_component_allocation"
        )
        assert sum(
            vector["breakdown"][name]
            for name in (
                "SoftmaxAuxRowSlices",
                "SoftmaxCommonRowLogic",
                "SoftmaxStateSIMD",
                "BankedVectorSRAMControl",
                "PackedPVAccumulator",
            )
        ) == pytest.approx(vector["rtl_v6_delta_area"])
        totals.append(vector["rtl_v6_delta_area"])
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_rtl_v6_ablation_feature_gates_remove_disabled_components() -> None:
    base = {
        "VLEN": 2048,
        "HLEN": 128,
        "BLEN": 128,
        "FP_SETTING": "FP_E6M5",
        "COMPACT_STATS_LANES": 16,
        "SOFTMAX_ROW_LANES": 1,
        "vector_scalar_area_version": "rtl-v6",
    }
    state_only = estimate_vector_machine_area(
        {
            **base,
            "ENABLE_SOFTMAX_MULTIROW": False,
            "ENABLE_SOFTMAX_STATE_SIMD": True,
            "ENABLE_PACKED_PV_ACCUMULATION": False,
        }
    )
    direct_only = estimate_vector_machine_area(
        {
            **base,
            "ENABLE_SOFTMAX_MULTIROW": False,
            "ENABLE_SOFTMAX_STATE_SIMD": False,
            "ENABLE_PACKED_PV_ACCUMULATION": True,
        }
    )
    assert state_only["breakdown"]["SoftmaxStateSIMD"] > 0
    assert state_only["breakdown"]["PackedPVAccumulator"] == 0
    assert state_only["breakdown"]["SoftmaxAuxRowSlices"] == 0
    assert direct_only["breakdown"]["SoftmaxStateSIMD"] == 0
    assert direct_only["breakdown"]["PackedPVAccumulator"] > 0
    assert direct_only["breakdown"]["SoftmaxCommonRowLogic"] == 0


def test_rtl_v6_area_plan_has_explicit_train_holdout_evidence() -> None:
    points = build_vector_plan("rtl-v6-area-v1")
    counts: dict[tuple[str, str], int] = {}
    for point in points:
        key = (
            str(point.params["calibration_role"]),
            str(point.params["calibration_split"]),
        )
        counts[key] = counts.get(key, 0) + 1
    assert len(points) == 56
    assert max(int(point.params["VLEN"]) for point in points) == 64
    assert counts == {
        ("state-simd-leaf", "train"): 8,
        ("state-simd-leaf", "holdout"): 2,
        ("packed-pv-leaf", "train"): 8,
        ("packed-pv-leaf", "holdout"): 2,
        ("banked-integration-wrapper", "train"): 24,
        ("banked-integration-wrapper", "holdout"): 4,
        ("paired-production-vector-current", "train"): 6,
        ("paired-production-vector-current", "holdout"): 2,
    }

    baseline = build_vector_plan("rtl-v6-paired-baseline-v1")
    assert len(baseline) == 8
    assert all(
        point.params["calibration_role"]
        == "paired-production-vector-baseline"
        for point in baseline
    )


def test_rtl_v6_promoted_structural_coefficients_drive_area(tmp_path) -> None:
    artifact = tmp_path / "rtl_v6.json"
    artifact.write_text(
        json.dumps(
            {
                "metadata": {"status": "fitted_from_paired_rtl_v6_dc"},
                "coefficients": {
                    "vlen_fp_um2": 0.05,
                    "extra_rows_um2": 10.0,
                    "extra_rows_vlen_um2": 1.0,
                    "extra_rows_vlen_exp_um2": 0.1,
                    "extra_rows_vlen_mant_um2": 0.2,
                    "row_fp_um2": 2.0,
                    "write_fp_um2": 3.0,
                    "bank_count_um2": 4.0,
                    "scoreboard_depth_um2": 5.0,
                    "const_um2": 6.0,
                },
            }
        )
    )
    result = estimate_vector_machine_area(
        {
            "VLEN": 64,
            "BLEN": 16,
            "FP_SETTING": "FP_E5M6",
            "SOFTMAX_ROW_LANES": 4,
            "SOFTMAX_SCOREBOARD_DEPTH": 32,
            "vector_scalar_area_version": "rtl-v6",
        },
        rtl_v6_delta_path=artifact,
    )
    fp_width = 12
    expected = (
        64 * fp_width * 0.05
        + 3 * 10.0
        + 3 * 64 * 1.0
        + 3 * 64 * 5 * 0.1
        + 3 * 64 * 6 * 0.2
        + 4 * fp_width * 2.0
        + 16 * fp_width * 3.0
        + 4 * 4.0
        + 32 * 5.0
        + 6.0
    )
    assert result["rtl_v6_delta_status"] == "fitted_from_paired_rtl_v6_dc"
    assert result["rtl_v6_delta_area"] == pytest.approx(expected)
    assert "paired_dc_structural_overlay" in result["area_model"]

    extrapolated = estimate_vector_machine_area(
        {
            "VLEN": 128,
            "BLEN": 16,
            "FP_SETTING": "FP_E5M6",
            "SOFTMAX_ROW_LANES": 4,
            "vector_scalar_area_version": "rtl-v6",
        },
        rtl_v6_delta_path=artifact,
    )
    assert extrapolated["rtl_v6_logic_calibration_max_vlen"] == 64
    assert extrapolated["rtl_v6_large_width_banked_logic_extrapolation"]


def test_rtl_v6_state_bank_is_separate_from_scalar_fp_sram() -> None:
    result = estimate_sram_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 64,
            "VLEN": 64,
            "BLEN": 16,
            "MATRIX_SRAM_DEPTH": 128,
            "VECTOR_SRAM_DEPTH": 128,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 10,
            "SOFTMAX_STATE_BANK_ENTRIES": 512,
            "SOFTMAX_ROW_LANES": 4,
            "INT_DATA_WIDTH": 32,
            "FP_SETTING": "FP_E5M6",
        },
        sram_port_model="ideal-dual-port",
    )
    state = result["area_sram_inputs"]["softmax_state"]
    assert state["entries"] == 512
    assert state["depth"] == 128
    assert state["row_lanes"] == 4
    assert state["entry_width"] == 25
    assert result["area_sram_breakdown"]["SoftmaxStateBank"] > 0.0
    assert result["area_sram_breakdown"]["SoftmaxStatisticBank"] > 0.0
    assert result["area_sram_breakdown"]["SoftmaxFactorBank"] > 0.0


def test_rtl_v6_vector_sram_banking_preserves_bits_and_charges_macro_rounding() -> None:
    base = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 2048,
        "VLEN": 2048,
        "BLEN": 1024,
        "MATRIX_SRAM_DEPTH": 4096,
        "VECTOR_SRAM_DEPTH": 257,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
        "INT_DATA_WIDTH": 32,
        "FP_SETTING": "FP_E5M6",
    }
    banked_areas = []
    logical_bits = set()
    for lanes in (1, 2, 4, 8):
        result = estimate_sram_area(
            {**base, "VECTOR_SRAM_ROW_BANKS": lanes},
            sram_port_model="ideal-dual-port",
        )
        banking = result["vector_sram_banking"]
        assert banking["physical_bank_count"] == lanes
        assert sum(banking["physical_bank_depths"]) == base["VECTOR_SRAM_DEPTH"]
        assert banking["storage_replication_factor"] == 1
        assert banking["covered_capacity_bits"] >= banking["logical_bits"]
        logical_bits.add(banking["logical_bits"])
        banked_areas.append(banking["selected_banked_area_um2"])
    assert len(logical_bits) == 1
    assert banked_areas == sorted(banked_areas)


def test_softmax_row_lanes_select_physical_vector_sram_banks() -> None:
    config = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 512,
        "VLEN": 512,
        "BLEN": 64,
        "MATRIX_SRAM_DEPTH": 1024,
        "VECTOR_SRAM_DEPTH": 257,
        "SOFTMAX_ROW_LANES": 8,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
        "INT_DATA_WIDTH": 32,
        "FP_SETTING": "FP_E5M6",
    }
    result = estimate_sram_area(config, sram_port_model="ideal-dual-port")
    assert result["vector_sram_banking"]["physical_bank_count"] == 8


def test_large_row_lane_analysis_preserves_logical_vector_sram_bits() -> None:
    base = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 2048,
        "VLEN": 2048,
        "BLEN": 128,
        "MATRIX_SRAM_DEPTH": 4096,
        "VECTOR_SRAM_DEPTH": 257,
        "SOFTMAX_STATE_BANK_ENTRIES": 32768,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
        "INT_DATA_WIDTH": 32,
        "FP_SETTING": "FP_E6M5",
    }
    r8 = estimate_sram_area(
        {**base, "SOFTMAX_ROW_LANES": 8},
        sram_port_model="ideal-dual-port",
    )
    r32 = estimate_sram_area(
        {**base, "SOFTMAX_ROW_LANES": 32},
        sram_port_model="ideal-dual-port",
    )
    b8 = r8["vector_sram_banking"]
    b32 = r32["vector_sram_banking"]
    assert b8["logical_bits"] == b32["logical_bits"]
    assert b32["physical_bank_count"] == 32
    assert b32["row_bank_fidelity"] == (
        "structural_extrapolation_not_isa_encodable"
    )
    assert b32["selected_banked_area_um2"] > b8["selected_banked_area_um2"]


def test_rtl_v6_vector_banks_keep_dual_port_shadow_separate() -> None:
    config = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 512,
        "VLEN": 512,
        "BLEN": 64,
        "MATRIX_SRAM_DEPTH": 1024,
        "VECTOR_SRAM_DEPTH": 257,
        "VECTOR_SRAM_ROW_BANKS": 4,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
        "INT_DATA_WIDTH": 32,
        "FP_SETTING": "FP_E5M6",
    }
    ideal = estimate_sram_area(config, sram_port_model="ideal-dual-port")
    replicated = estimate_sram_area(config, sram_port_model="replicated-single-port")
    ideal_banking = ideal["vector_sram_banking"]
    replicated_banking = replicated["vector_sram_banking"]
    assert ideal_banking["storage_replication_factor"] == 1
    assert replicated_banking["storage_replication_factor"] == 1
    assert math.isclose(
        replicated_banking["selected_banked_area_um2"],
        2.0 * ideal_banking["selected_banked_area_um2"],
    )


def test_estimate_sram_area_macro_tiling_has_details() -> None:
    result = estimate_sram_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 64,
            "VLEN": 64,
            "BLEN": 16,
            "MATRIX_SRAM_DEPTH": 512,
            "VECTOR_SRAM_DEPTH": 512,
            "INT_DATA_WIDTH": 32,
            "INT_SRAM_DEPTH": 256,
            "FP_SRAM_DEPTH": 256,
        }
    )
    if result["area_sram_model"] == "asap7_sram_macro_tiling":
        assert result["area_sram_macro_tiling"]["matrix"]["macro"].startswith("srambank_")
        assert result["area_sram_macro_tiling"]["matrix"]["tile_count"] >= 1
        assert result["area_sram_breakdown"]["MatrixSRAM"] > 0


def test_ideal_dual_port_sram_removes_macro_replication_only() -> None:
    config = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 64,
        "VLEN": 64,
        "BLEN": 8,
        "MATRIX_SRAM_DEPTH": 128,
        "VECTOR_SRAM_DEPTH": 32,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 256,
        "INT_DATA_WIDTH": 32,
        "FP_SETTING": "FP_E6M5",
    }
    replicated = estimate_sram_area(config)
    ideal = estimate_sram_area(
        config, sram_port_model="ideal-dual-port"
    )

    for name in ("matrix", "vector"):
        assert (
            replicated["area_sram_macro_tiling"][name]["port_copies"] == 2
        )
        assert ideal["area_sram_macro_tiling"][name]["port_copies"] == 1
    for name in ("scalar_int", "scalar_fp"):
        assert (
            replicated["area_sram_breakdown"][
                "ScalarIntSRAM" if name == "scalar_int" else "ScalarFPSRAM"
            ]
            == ideal["area_sram_breakdown"][
                "ScalarIntSRAM" if name == "scalar_int" else "ScalarFPSRAM"
            ]
        )
    assert ideal["area"] <= replicated["area"]
    assert ideal["dual_port_area_savings_um2"] > 0


def test_estimate_total_area_includes_sram() -> None:
    result = estimate_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 16,
            "VLEN": 16,
            "BLEN": 4,
            "MATRIX_SRAM_DEPTH": 32,
            "VECTOR_SRAM_DEPTH": 32,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        }
    )
    assert result["area"] > result["area_breakdown"]["MatrixMachine"]
    assert "MatrixSRAM" in result["area_breakdown"]


def test_full_chip_top_residual_scales_logic_only(tmp_path) -> None:
    coeff_path = tmp_path / "top_coefficients.json"
    coeff_path.write_text(
        json.dumps(
            {
                "metadata": {"model_version": "test_top_residual"},
                "coefficients": {"logic_fraction": 0.1},
            }
        )
    )
    config = {
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "MLEN": 16,
        "VLEN": 16,
        "BLEN": 4,
        "MATRIX_SRAM_DEPTH": 32,
        "VECTOR_SRAM_DEPTH": 32,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 32,
    }
    raw = estimate_area(config, top_residual_coefficients_path=coeff_path, apply_top_residual=False)
    corrected = estimate_area(config, top_residual_coefficients_path=coeff_path)
    expected_residual = raw["logic_area_before_top_residual"] * 0.1
    assert corrected["area_breakdown"]["FullChipTopResidual"] == pytest.approx(expected_residual)
    assert corrected["area"] - raw["area"] == pytest.approx(expected_residual)
    assert corrected["sram_macro_area"] == pytest.approx(raw["sram_macro_area"])


def test_full_chip_hierarchy_parser(tmp_path) -> None:
    report = tmp_path / "area.rpt"
    report.write_text(
        """
Hierarchical area distribution
------------------------------
plena                             1000.0000    100.0    10.0000  20.0000  0.0000 plena
matrix_machine_init                400.0000     40.0    10.0000  20.0000  0.0000 matrix_machine
scalar_machine_init/fp_scalar_sram
                                      0.2500      0.0     0.2500   0.0000  0.0000 scalar_sram_1
"""
    )
    parsed = parse_full_chip_hierarchy(report)
    assert parsed["plena"] == pytest.approx(1000.0)
    assert parsed["matrix_machine_init"] == pytest.approx(400.0)
    assert parsed["scalar_machine_init/fp_scalar_sram"] == pytest.approx(0.25)


def test_full_chip_hierarchy_uses_embedded_calibration_values() -> None:
    source = {
        "hier_plena_area": "100.0",
        "hier_matrix_machine_area": "40.0",
        "hier_vector_machine_area": "20.0",
        "hier_scalar_machine_area": "10.0",
        "hier_scalar_fp_sram_wrapper_area": "1.0",
        "hier_scalar_int_sram_wrapper_area": "2.0",
        "hier_hbm_system_area": "15.0",
        "hier_matrix_sram_wrapper_area": "7.0",
        "hier_vector_sram_wrapper_area": "5.0",
        "report_dir": "/does/not/exist",
        "point_id": "embedded",
    }
    hierarchy = _hierarchy_from_source(source)
    assert hierarchy["plena"] == pytest.approx(100.0)
    assert hierarchy["matrix_machine_init"] == pytest.approx(40.0)
    assert hierarchy["scalar_machine_init/fp_scalar_sram"] == pytest.approx(1.0)


def test_estimate_vector_machine_area_monotonic() -> None:
    small = estimate_vector_machine_area({"VLEN": 64, "FP_SETTING": "FP_E3M2"})
    wide = estimate_vector_machine_area({"VLEN": 64, "FP_SETTING": "FP_E5M6"})
    large = estimate_vector_machine_area({"VLEN": 128, "FP_SETTING": "FP_E5M6"})
    assert small["area"] > 0
    assert wide["area"] > small["area"]
    assert large["area"] > wide["area"]
    assert wide["inputs"]["fp_width"] == 12


def test_estimate_scalar_machine_area_monotonic() -> None:
    small = estimate_scalar_machine_area({"INT_DATA_WIDTH": 16, "FP_SETTING": "FP_E5M6"})
    wide_int = estimate_scalar_machine_area({"INT_DATA_WIDTH": 64, "FP_SETTING": "FP_E5M6"})
    wide_fp = estimate_scalar_machine_area({"INT_DATA_WIDTH": 16, "FP_SETTING": "FP_E8M5"})
    assert small["area"] > 0
    assert wide_int["area"] > small["area"]
    assert wide_fp["area"] > small["area"]
    assert wide_fp["inputs"]["S_FP_EXP_WIDTH"] == 8


def test_scheduler_job_key_is_stable() -> None:
    point = VectorPoint(
        "vector_v16_fp_e5m6",
        "vector_machine",
        "vector_machine",
        {"VLEN": 16, "FP_SETTING": "FP_E5M6", "V_FP_EXP_WIDTH": 5, "V_FP_MANT_WIDTH": 6},
    )
    assert stable_job_key("vector_machine", point) == stable_job_key("vector_machine", point)
    changed = VectorPoint(
        "vector_v32_fp_e5m6",
        "vector_machine",
        "vector_machine",
        {"VLEN": 32, "FP_SETTING": "FP_E5M6", "V_FP_EXP_WIDTH": 5, "V_FP_MANT_WIDTH": 6},
    )
    assert stable_job_key("vector_machine", point) != stable_job_key("vector_machine", changed)


def test_calibration_csv_latest_and_complete_only(tmp_path) -> None:
    rows = [
        {"job_key": "a", "status": "failed", "area_um2": ""},
        {"job_key": "a", "status": "complete", "area_um2": "10"},
        {"job_key": "b", "status": "failed", "area_um2": ""},
    ]
    latest = latest_by_key(rows)
    assert latest["a"]["status"] == "complete"
    complete = latest_complete_rows(rows, lambda row: row["job_key"])
    assert complete == [{"job_key": "a", "status": "complete", "area_um2": "10"}]
    out = tmp_path / "rows.csv"
    write_rows(out, complete, ["job_key", "status", "area_um2"])
    with out.open(newline="") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 1
    assert written[0]["job_key"] == "a"


def test_calibration_failure_classification() -> None:
    assert classify_failure({"status": "complete"}) == ""
    assert classify_failure({"status": "failed", "failure_reason": "SEC-50 Unable to obtain license"}) == "license_busy"
    assert classify_failure({"status": "failed", "failure_reason": "ValueError unsupported precision"}) == "config_error"
    assert classify_failure({"status": "failed", "failure_reason": "synth failed with exit code 1"}) == "synth_failed"


def test_estimate_hbm_system_area_monotonic() -> None:
    small = estimate_hbm_system_area(
        {
            "ACT_WIDTH": "MXINT2",
            "KV_WIDTH": "MXINT2",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 64,
            "VLEN": 64,
            "BLEN": 16,
            "HBM_M_Prefetch_Amount": 64,
            "HBM_V_Prefetch_Amount": 64,
            "HBM_V_Writeback_Amount": 64,
        }
    )
    large = estimate_hbm_system_area(
        {
            "ACT_WIDTH": "MXINT8",
            "KV_WIDTH": "MXINT8",
            "WEIGHT_WIDTH": "MXINT8",
            "MLEN": 128,
            "VLEN": 128,
            "BLEN": 16,
            "HBM_M_Prefetch_Amount": 128,
            "HBM_V_Prefetch_Amount": 128,
            "HBM_V_Writeback_Amount": 128,
        }
    )
    assert small["area"] > 0
    assert large["area"] > small["area"]
    assert large["inputs"]["HBM_M_Prefetch_Amount"] == 128


def test_total_area_uses_fitted_vector_proxy(tmp_path) -> None:
    coeff_path = tmp_path / "vector_model_coefficients.json"
    coeff_path.write_text(
        json.dumps(
            {
                "metadata": {"status": "fitted_from_local_plena_rtl_synth"},
                "coefficients": {
                    "a_lane_quad": 0.0,
                    "b_tree": 0.0,
                    "c_lane_linear": 1.0,
                    "d_control": 0.0,
                    "e_const": 10.0,
                },
            }
        )
    )
    result = estimate_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 16,
            "VLEN": 16,
            "BLEN": 4,
            "FP_SETTING": "FP_E3M2",
            "MATRIX_SRAM_DEPTH": 32,
            "VECTOR_SRAM_DEPTH": 32,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        },
        vector_coefficients_path=coeff_path,
    )
    assert "VectorMachine" in result["area_breakdown"]
    assert "VectorMachineLegacy" not in result["area_breakdown"]
    assert result["vector_machine"]["coefficients_source"] == str(coeff_path)


def test_total_area_uses_fitted_scalar_proxy(tmp_path) -> None:
    coeff_path = tmp_path / "scalar_model_coefficients.json"
    coeff_path.write_text(
        json.dumps(
            {
                "metadata": {"status": "fitted_from_local_plena_rtl_synth"},
                "coefficients": {
                    "a_int_mul": 0.0,
                    "a_int_lin": 1.0,
                    "a_fp_quad": 0.0,
                    "a_fp_lin": 2.0,
                    "a_exp": 3.0,
                    "a_const": 10.0,
                },
            }
        )
    )
    result = estimate_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 16,
            "VLEN": 16,
            "BLEN": 4,
            "FP_SETTING": "FP_E5M6",
            "INT_DATA_WIDTH": 32,
            "MATRIX_SRAM_DEPTH": 32,
            "VECTOR_SRAM_DEPTH": 32,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        },
        scalar_coefficients_path=coeff_path,
    )
    assert "ScalarIntLogic" in result["area_breakdown"]
    assert "ScalarFPLogic" in result["area_breakdown"]
    assert "ScalarControl" in result["area_breakdown"]
    assert "ScalarMachineLogicLegacy" not in result["area_breakdown"]
    assert result["scalar_machine"]["coefficients_source"] == str(coeff_path)


def test_total_area_uses_fitted_hbm_proxy(tmp_path) -> None:
    coeff_path = tmp_path / "hbm_model_coefficients.json"
    coeff_path.write_text(
        json.dumps(
            {
                "metadata": {"status": "fitted_from_local_plena_rtl_synth"},
                "coefficients": {
                    "a_ele": 0.0,
                    "a_scale": 0.0,
                    "a_m_path": 1.0,
                    "a_v_path": 2.0,
                    "a_scale_path": 0.0,
                    "a_addr": 0.0,
                    "a_load": 0.0,
                    "a_write": 0.0,
                    "a_const": 10.0,
                },
            }
        )
    )
    result = estimate_area(
        {
            "ACT_WIDTH": "MXINT4",
            "KV_WIDTH": "MXINT4",
            "WEIGHT_WIDTH": "MXINT4",
            "MLEN": 16,
            "VLEN": 16,
            "BLEN": 4,
            "MATRIX_SRAM_DEPTH": 32,
            "VECTOR_SRAM_DEPTH": 32,
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": 32,
        },
        hbm_coefficients_path=coeff_path,
    )
    assert "HBMMatrixPath" in result["area_breakdown"]
    assert "HBMVectorPath" in result["area_breakdown"]
    assert "HBMSystemLegacy" not in result["area_breakdown"]
    assert result["hbm_system"]["coefficients_source"] == str(coeff_path)


def test_rtl_v6_area_fit_requires_and_accepts_complete_paired_dataset(
    tmp_path,
) -> None:
    current_path = tmp_path / "current.csv"
    baseline_path = tmp_path / "baseline.csv"
    current_points = build_vector_plan("rtl-v6-area-v1")
    baseline_points = build_vector_plan("rtl-v6-paired-baseline-v1")
    feature_coefficients = {
        "vlen_fp": 0.02,
        "extra_rows": 4.0,
        "extra_rows_vlen": 0.2,
        "extra_rows_vlen_exp": 0.03,
        "extra_rows_vlen_mant": 0.04,
        "row_fp": 1.5,
        "write_fp": 2.0,
        "bank_count": 3.0,
        "scoreboard_depth": 0.5,
        "const": 120.0,
    }
    production_coefficients = {
        "vlen_fp": 0.2,
        "extra_rows": 4.0,
        "extra_rows_vlen": 0.4,
        "const": 120.0,
    }

    def area_for(point: VectorPoint, *, baseline: bool = False) -> float:
        row = dict(point.params)
        features = rtl_v6_structural_features(row)
        if baseline:
            return 100_000.0 + 10.0 * float(row["VLEN"])
        role = str(row["calibration_role"])
        if role == "state-simd-leaf":
            return 2.0 * features["row_fp"] + 75.0
        if role == "packed-pv-leaf":
            return 3.0 * features["write_fp"] + 45.0
        wrapper_delta = sum(
            features[name] * feature_coefficients[name]
            for name in feature_coefficients
        )
        if role == "paired-production-vector-current":
            production_delta = sum(
                features[name] * production_coefficients[name]
                for name in production_coefficients
            )
            return (
                100_000.0
                + 10.0 * float(row["VLEN"])
                + production_delta
            )
        return wrapper_delta

    fieldnames = sorted(
        {
            "point_id",
            "point_key",
            "status",
            "area_um2",
            *(
                key
                for point in current_points + baseline_points
                for key in point.params
            ),
        }
    )

    def write_points(path, points, *, baseline: bool) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in points:
                writer.writerow(
                    {
                        "point_id": point.point_id,
                        "point_key": point.point_key,
                        "status": "complete",
                        "area_um2": area_for(point, baseline=baseline),
                        **point.params,
                    }
                )

    write_points(current_path, current_points, baseline=False)
    write_points(baseline_path, baseline_points, baseline=True)
    artifact, diagnostics = fit_vector_rtl_v6_artifact(
        current_path, baseline_path
    )

    assert artifact["metadata"]["status"] == "fitted_from_paired_rtl_v6_dc"
    assert not artifact["metadata"]["failures"]
    assert artifact["metadata"]["checks"]["area_monotonic"]
    assert len(diagnostics) == 36
    assert artifact["metadata"]["checks"]["packed_pv_holdout"]["median_pct"] <= 5.0
    assert artifact["metadata"]["checks"]["paired_machine_holdout"]["max_pct"] <= 10.0
    assert artifact["metadata"]["logic_fit_vlen"] == [16, 32]
    assert artifact["metadata"]["logic_holdout_vlen"] == [64]


def test_rtl_v6_timing_plan_covers_tiers_and_period_sensitivity() -> None:
    points = build_vector_plan("rtl-v6-timing-v1")

    assert len(points) == 9
    assert {
        (
            int(point.params["SOFTMAX_ROW_LANES"]),
            int(point.params["clock_period_ps"]),
        )
        for point in points
    } == {
        (lanes, period)
        for lanes in (1, 4, 8)
        for period in (1000, 1250, 1500)
    }
    assert all(point.module == "vector_machine" for point in points)
    assert all(
        point.params["calibration_role"] == "production-vector-timing"
        for point in points
    )


@pytest.mark.parametrize(
    ("module", "variant", "parameters"),
    (
        (
            "vector_machine_rtl_v6_integration_wrapper",
            "rtl-v6-production-vector-sram-integration",
            {"ROW_LANES": 8, "STATE_ENTRIES": 96, "SRAM_DEPTH": 192},
        ),
        (
            "packed_pv_accumulator",
            "rtl-v6-packed-pv-production-leaf",
            {"EXP_WIDTH": 6, "MANT_WIDTH": 5, "VLEN": 64, "WRITE_LANES": 32},
        ),
    ),
)
def test_rtl_v6_production_variants_specialize_dc_worker_source(
    tmp_path, module, variant, parameters,
) -> None:
    definitions = tmp_path / "src/definitions"
    definitions.mkdir(parents=True)
    (definitions / "configuration.svh").write_text("localparam VLEN = 16;\n")
    (definitions / "precision.svh").write_text(
        "localparam V_FP_EXP_WIDTH = 5;\n"
        "localparam V_FP_MANT_WIDTH = 6;\n"
    )
    relative = (
        "src/vector_machine/rtl/vector_machine_rtl_v6_integration_wrapper.sv"
        if module == "vector_machine_rtl_v6_integration_wrapper"
        else "src/matrix_machine/rtl/packed_pv_accumulator.sv"
    )
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text(
        "module test #(\n"
        + ",\n".join(
            f"parameter int {name} = 1" for name in parameters
        )
        + "\n); endmodule\n"
    )
    params = {
        "VLEN": 64,
        "V_FP_EXP_WIDTH": 6,
        "V_FP_MANT_WIDTH": 5,
        "rtl_variant": variant,
        **parameters,
    }

    patch_vector_config(VectorPoint("point", module, module, params), tmp_path)

    specialized = source.read_text()
    for name, value in parameters.items():
        assert f"parameter int {name} = {value}" in specialized


def test_rtl_v6_timing_artifact_keeps_wns_separate_from_ii(tmp_path) -> None:
    timing_csv = tmp_path / "timing.csv"
    fields = (
        "point_key",
        "point_id",
        "status",
        "SOFTMAX_ROW_LANES",
        "clock_period_ps",
        "wns_ns",
        "report_dir",
    )
    with timing_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lanes in (1, 4, 8):
            for period in (1000, 1250, 1500):
                writer.writerow(
                    {
                        "point_key": f"r{lanes}_{period}",
                        "point_id": f"r{lanes}_{period}",
                        "status": "complete",
                        "SOFTMAX_ROW_LANES": lanes,
                        "clock_period_ps": period,
                        "wns_ns": (period - 1200 - lanes * 10) / 1000.0,
                        "report_dir": "reports",
                    }
                )
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps(
            {
                "independent_row_ii_one": True,
                "measurements": [{"independent_ii_cycles": 1}],
            }
        )
    )

    artifact = build_rtl_v6_timing_artifact(
        timing_csv, pipeline_audit=pipeline
    )

    assert artifact["calibration_status"] == "production_vector_timing_candidate"
    assert artifact["tiers"]["1"]["functional_independent_ii_cycles"] == 1
    assert artifact["tiers"]["4"]["functional_independent_ii_cycles"] is None
    assert artifact["tiers"]["8"]["minimum_closed_period_ps"] == 1500
