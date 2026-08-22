from __future__ import annotations

import pytest

from .report_primary_system_search import (
    _aggregate_endpoints,
    reconstruct_replica_phase,
)


def _row(*, latency: float, energy_mj: float, idle_power: float) -> dict[str, object]:
    return {
        "median_full_request_latency_s": latency,
        "median_complete_energy_mj": energy_mj,
        "median_imported_kv_decode_proxy_latency_s": latency,
        "median_imported_kv_decode_proxy_energy_mj": energy_mj,
        "median_first_decode_iteration_latency_s": 0.5,
        "median_idle_total_board_power_w": idle_power,
        "local_batch_size": 1,
        "output_tokens": 10,
        "median_mean_request_tpot_s": 0.1,
        "median_p95_request_tpot_s": 0.2,
        "median_median_request_ttft_s": 1.0,
        "request_tpot_samples_s": [0.1],
        "request_visible_ttft_samples_s": [1.0],
    }


def test_reconstruct_replica_phase_charges_idle_tail() -> None:
    result = reconstruct_replica_phase(
        [
            _row(latency=10.0, energy_mj=10_000.0, idle_power=100.0),
            _row(latency=8.0, energy_mj=8_000.0, idle_power=50.0),
        ],
        phase="decode",
        fidelity="test",
    )

    assert result["latency_s"] == pytest.approx(10.0)
    assert result["active_energy_j"] == pytest.approx(18.0)
    assert result["idle_tail_energy_j"] == pytest.approx(100.0)
    assert result["energy_j"] == pytest.approx(118.0)
    assert result["global_batch_size"] == 2
    assert result["global_output_tokens"] == 20
    assert result["mean_request_tpot_s"] == pytest.approx(0.1)


def test_aggregate_endpoints_can_choose_different_topologies() -> None:
    endpoints = _aggregate_endpoints(
        [
            {
                "topology": "fast",
                "output_tokens_per_s": 2.0,
                "output_tokens_per_j": 1.0,
            },
            {
                "topology": "efficient",
                "output_tokens_per_s": 1.5,
                "output_tokens_per_j": 3.0,
            },
        ]
    )

    assert endpoints["maximum_output_tps"]["topology"] == "fast"
    assert endpoints["maximum_output_tokens_per_j"]["topology"] == "efficient"
