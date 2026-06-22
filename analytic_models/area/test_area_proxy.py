from __future__ import annotations

import json
import math

import pytest

from analytic_models.area.area_proxy import AreaFormulaError, estimate_area


BASE_CONFIG = {
    "MLEN": 1024,
    "BLEN": 64,
    "VLEN": 1024,
    "WT_MX_MANT_WIDTH": 1,
    "WT_MX_EXP_WIDTH": 2,
    "KV_ELEMENT_WIDTH": 8,
    "BLOCK_DIM": 8,
    "MX_SCALE_WIDTH": 8,
    "ACT_ELEMENT_WIDTH": 8,
    "FP_EXP_WIDTH": 3,
    "FP_MANT_WIDTH": 4,
    "HBM_ELE_WIDTH": 512,
    "HBM_SCALE_WIDTH": 512,
    "MATRIX_SRAM_DEPTH": 4096,
    "VECTOR_SRAM_DEPTH": 4096,
    "INT_DATA_WIDTH": 32,
    "INT_SRAM_DEPTH": 256,
    "FP_SRAM_DEPTH": 256,
}


def test_area_proxy_is_stable_and_matches_legacy_formula_total():
    first = estimate_area(BASE_CONFIG)
    second = estimate_area(BASE_CONFIG)

    assert first["area"] == second["area"]
    assert first["area_proxy_breakdown"] == second["area_proxy_breakdown"]
    assert math.isclose(first["area"], 845199773.68)


def test_area_proxy_changes_with_trial_knobs():
    base = estimate_area(BASE_CONFIG)["area"]
    larger = estimate_area({**BASE_CONFIG, "MLEN": 2048})["area"]
    wider_int = estimate_area({**BASE_CONFIG, "INT_DATA_WIDTH": 64})["area"]

    assert larger != base
    assert wider_int != base


def test_area_proxy_breakdown_sums_to_total():
    result = estimate_area(BASE_CONFIG)

    assert math.isclose(sum(result["area_proxy_breakdown"].values()), result["area"])


def test_area_proxy_rejects_unsafe_formula(tmp_path):
    units = {
        "BadUnit": {
            "Coefficients": {},
            "Relationship": "__import__('os').system('echo bad')",
        }
    }
    units_path = tmp_path / "units.json"
    units_path.write_text(json.dumps(units))

    with pytest.raises(AreaFormulaError):
        estimate_area(BASE_CONFIG, units_path)
