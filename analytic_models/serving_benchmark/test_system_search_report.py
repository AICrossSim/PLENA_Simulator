from __future__ import annotations

import pytest

from .report_primary_system_search import (
    _aggregate_candidates,
    _aggregate_endpoints,
    _annotate_fixed_batch_ttft,
    _fixed_batch_ttft_evidence,
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
    assert result["idle_power_w"] == pytest.approx(150.0)
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


def test_aggregate_candidates_uses_workload_output_tokens() -> None:
    candidates = _aggregate_candidates(
        {
            "t": {
                "latency_s": 4.0,
                "energy_j": 8.0,
                "global_batch_size": 2,
            }
        },
        batch_size=2,
        output_tokens_per_request=100,
    )

    assert candidates[0]["output_tokens_per_s"] == pytest.approx(50.0)
    assert candidates[0]["output_tokens_per_j"] == pytest.approx(25.0)


def test_fixed_batch_ttft_replaces_full_generation_admission_diagnostic() -> None:
    topologies = {
        "TP4xDP2_B4": {
            "scheduler_admitted_ttft_exact_or_proxy_s": 60.0,
            "scheduler_admitted_ttft_source": "full_generation",
        }
    }
    _annotate_fixed_batch_ttft(
        topologies,
        audit_rows=[
            {
                "point_id": "m.primary-90000x1.tp4.b4",
                "model": "m",
                "input_tokens": 90_000,
                "output_tokens": 1,
                "tensor_parallel_size": 4,
                "local_batch_size": 4,
                "validation_status": "pass",
                "median_scheduler_admitted_ttft_best_available_s": 21.0,
                "scheduler_admitted_ttft_sources": "inferred_serial_staircase",
                "scheduler_admitted_ttft_fidelities": "proxy_from_serial_staircase",
                "median_median_request_ttft_s": 52.0,
                "median_p95_request_ttft_s": 84.0,
                "median_batch_first_token_barrier_latency_s": 84.0,
            }
        ],
        model="m",
        input_tokens=90_000,
        topology_shapes={"TP4xDP2_B4": (4, 4)},
    )

    row = topologies["TP4xDP2_B4"]
    assert row["full_generation_scheduler_admitted_ttft_exact_or_proxy_s"] == 60.0
    assert row["scheduler_admitted_ttft_exact_or_proxy_s"] == 21.0
    assert row["scheduler_admitted_ttft_source"] == "inferred_serial_staircase"
    assert row["fixed_batch_first_token_barrier_s"] == 84.0


def test_fixed_batch_ttft_rejected_proxy_is_unavailable() -> None:
    evidence = _fixed_batch_ttft_evidence(
        {
            "point_id": "m.short.tp8.b8",
            "validation_status": "pass",
            "median_scheduler_admitted_ttft_best_available_s": float("nan"),
            "scheduler_admitted_ttft_sources": "unavailable",
            "scheduler_admitted_ttft_fidelities": "unavailable",
            "median_median_request_ttft_s": 0.6,
            "median_p95_request_ttft_s": 0.9,
            "median_batch_first_token_barrier_latency_s": 0.9,
        }
    )

    assert evidence["scheduler_admitted_ttft_exact_or_proxy_s"] is None
    assert evidence["scheduler_admitted_ttft_source"] == "unavailable_fixed_batch_audit_rejected"
