"""Unit and integration tests for area equations, parsers, and job bookkeeping."""

from __future__ import annotations

import csv
import json

import pytest

from analytic_models.area_new import (
    PrecisionError,
    derive_compute_sides,
    estimate_area,
    estimate_hbm_system_area,
    estimate_matrix_machine_area,
    estimate_scalar_machine_area,
    estimate_sram_area,
    estimate_vector_machine_area,
    parse_precision,
)
from analytic_models.area_new.mxint_model import pe_area as mxint_pe_area
from analytic_models.area_new.mxint_model import estimate as estimate_mxint_matrix
from analytic_models.area_new.scripts.calibration_csv import latest_by_key, latest_complete_rows, write_rows
from analytic_models.area_new.scripts.calibration_runtime import classify_failure, stable_job_key
from analytic_models.area_new.scripts.run_matrix_machine_calibration import parse_hierarchy_area
from analytic_models.area_new.scripts.run_vector_machine_calibration import Point as VectorPoint
from analytic_models.area_new.scripts.validate_full_chip_area_proxy import (
    _hierarchy_from_source,
    parse_full_chip_hierarchy,
)


def test_parse_mxint_and_mxfp() -> None:
    assert parse_precision("MXINT2").bits == 2
    assert parse_precision("MXINT_4").bits == 4
    p = parse_precision("MXFP_E4M3")
    assert (p.exp, p.mant, p.element_width) == (4, 3, 8)
    q = parse_precision({"kind": "MXFP", "exp": 5, "mant": 2, "scale_width": 8})
    assert q.name == "MXFP_E5M2"


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
"""
    )
    parsed = parse_hierarchy_area(report)
    assert parsed["hier_total_area"] == pytest.approx(10709.5785)
    assert parsed["hier_compute_unit_area"] == pytest.approx(10286.5398)
    assert parsed["hier_reduce_area"] == pytest.approx(3825.0338)
    assert parsed["hier_accum_area"] == pytest.approx(579.3948)
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
