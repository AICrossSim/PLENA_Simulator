from __future__ import annotations

import pytest

from analytic_models.dse.report_rtl_v6_single_layer_ab import (
    MODELS,
    _derive,
    _validate_actual_invariants,
)


def _row(model: str, arm: str, lanes: int, latency: float, area: float) -> dict:
    return {
        "model": model,
        "arm": arm,
        "row_lanes": lanes,
        "latency_ms": latency,
        "system_energy_mj": latency * 2,
        "core_area_mm2": area,
        "state_fp_loads": 10,
        "state_fp_stores": 10,
        "pv_shift_ops": 10,
        "pv_vector_add_ops": 10,
        "result_fidelity": "costemitter_actual_rtl_isa_supported",
        "qk_compute_count": 3,
        "pv_compute_count": 4,
        "hbm_physical_read_bytes": 5,
        "hbm_physical_write_bytes": 6,
        "layer_dma_manifest_hash": "sha256:test",
    }


def test_marginal_return_only_compares_combined_r_tiers() -> None:
    model = MODELS[0].key
    rows = [
        _row(model, "rtl_v5", 1, 20.0, 10.0),
        _row(model, "state_only", 1, 15.0, 10.1),
        _row(model, "combined_r1", 1, 12.0, 10.2),
        _row(model, "combined_r2", 2, 8.0, 11.2),
    ]
    _derive(rows)
    assert rows[1]["marginal_speedup_vs_previous"] is None
    assert rows[2]["marginal_speedup_vs_previous"] is None
    assert rows[3]["marginal_speedup_vs_previous"] == pytest.approx(1.5)
    assert rows[3]["marginal_latency_saved_per_mm2"] == pytest.approx(4.0)


def test_actual_invariant_audit_rejects_dma_drift() -> None:
    rows = []
    for model in MODELS:
        rows.extend(
            [
                _row(model.key, "rtl_v5", 1, 20.0, 10.0),
                _row(model.key, "combined_r1", 1, 12.0, 10.2),
            ]
        )
    rows[1]["layer_dma_manifest_hash"] = "sha256:drift"
    with pytest.raises(RuntimeError, match="invariant drift"):
        _validate_actual_invariants(rows)
