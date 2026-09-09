from __future__ import annotations

import math

import pytest

from .serving_queue import (
    ServingQueueConfig,
    ServingRequest,
    poisson_arrival_times,
    simulate_sensitivity,
    simulate_serving_queue,
)


def request(
    request_id: str,
    arrival_s: float,
    *,
    prefill_s: float = 1.0,
    decode_tpot_s: float = 0.1,
    generation_tokens: int = 3,
    kv_buffer_bytes: int = 100,
) -> ServingRequest:
    return ServingRequest(
        request_id=request_id,
        arrival_s=arrival_s,
        prompt_tokens=128,
        generation_tokens=generation_tokens,
        prefill_s=prefill_s,
        decode_tpot_s=decode_tpot_s,
        transfer_streamed_s=0.05,
        transfer_bulk_s=0.2,
        host_write_s=0.3,
        host_read_s=0.3,
        admission_s=0.1,
        kv_buffer_bytes=kv_buffer_bytes,
        admission_bytes=150,
        prefill_energy_j=2.0,
        decode_energy_per_token_j=0.5,
        handoff_energy_j=0.25,
    )


def config(regime: str, **overrides) -> ServingQueueConfig:
    values = {
        "regime": regime,
        "prefill_replicas": 1,
        "decode_replicas": 1,
        "link_ports": 1,
        "kv_buffer_capacity_bytes": 1000 if regime != "back_pressure" else 0,
        "max_inflight_requests": 32,
        "ttft_slo_s": 10.0,
        "tpot_slo_s": 10.0,
    }
    values.update(overrides)
    return ServingQueueConfig(**values)


def test_fully_pipelined_first_token_and_service_timing() -> None:
    result = simulate_serving_queue(
        [request("r0", 0.0)], config("fully_pipelined")
    )
    row = result["requests"][0]
    assert row["ttft_s"] == pytest.approx(1.0)
    assert row["first_decode_token_s"] == pytest.approx(1.25)
    assert row["completion_s"] == pytest.approx(1.35)
    assert row["decode_service_tpot_s"] == pytest.approx(0.1)
    assert row["observed_tpot_s"] == pytest.approx(0.175)
    assert result["resources"]["peak_kv_buffer_bytes"] == 100
    assert result["completed_requests"] == 1
    assert result["model_scope"]["standalone_input_provenance_rankable"] is False
    assert result["model_scope"]["external_input_receipt_required"] is True


def test_backpressure_stalls_prefill_and_respects_link_ports() -> None:
    requests = [
        request("r0", 0.0, generation_tokens=12),
        request("r1", 0.0, generation_tokens=12),
        request("r2", 0.0, generation_tokens=12),
    ]
    result = simulate_serving_queue(requests, config("back_pressure"))
    assert result["completed_requests"] == 3
    assert result["rejected_requests"] == 0
    assert result["resources"]["prefill_stall_fraction"] > 0
    assert result["resources"]["peak_kv_buffer_bytes"] == 0
    assert 0 < result["resources"]["link_utilization"] <= 1
    assert all(
        row.get("prefill_blocked_s", 0) >= 0 for row in result["requests"]
    )


def test_host_buffer_capacity_rejection_is_visible_and_costed() -> None:
    requests = [
        request("r0", 0.0, prefill_s=0.1, generation_tokens=20),
        request("r1", 0.0, prefill_s=0.1, generation_tokens=20),
        request("r2", 0.0, prefill_s=0.1, generation_tokens=20),
    ]
    result = simulate_serving_queue(
        requests,
        config(
            "host_buffered",
            prefill_replicas=3,
            kv_buffer_capacity_bytes=100,
        ),
    )
    assert result["completed_requests"] == 1
    assert result["rejected_requests"] == 2
    assert result["rejection_reasons"] == {"kv_buffer_capacity": 2}
    assert result["energy"]["rejected_prefill_j"] == pytest.approx(4.0)


def test_host_writes_and_reads_share_link_port_capacity() -> None:
    result = simulate_serving_queue(
        [
            request("r0", 0.0, prefill_s=0.1),
            request("r1", 0.0, prefill_s=0.1),
        ],
        config(
            "host_buffered",
            prefill_replicas=2,
            decode_replicas=2,
            link_ports=1,
        ),
    )
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["r0"]["host_read_start_s"] == pytest.approx(0.4)
    assert rows["r0"]["host_read_finish_s"] == pytest.approx(0.7)
    assert rows["r1"]["transfer_start_s"] == pytest.approx(0.7)
    assert rows["r1"]["host_read_start_s"] == pytest.approx(1.0)
    assert result["resources"]["link_utilization"] <= 1.0
    assert "decode reads have priority" in result["model_scope"]["queue_policy"]


def test_host_buffer_remains_reserved_until_read_completion() -> None:
    result = simulate_serving_queue(
        [
            request("r0", 0.0, prefill_s=0.1),
            request("r1", 0.35, prefill_s=0.1),
        ],
        config(
            "host_buffered",
            prefill_replicas=2,
            decode_replicas=2,
            link_ports=1,
            kv_buffer_capacity_bytes=100,
        ),
    )
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["r0"]["host_read_start_s"] == pytest.approx(0.4)
    assert rows["r0"]["host_read_finish_s"] == pytest.approx(0.7)
    assert rows["r0"]["buffer_leave_s"] == pytest.approx(0.7)
    assert rows["r1"]["status"] == "rejected"
    assert rows["r1"]["rejection_reason"] == "kv_buffer_capacity"
    assert result["resources"]["peak_kv_buffer_bytes"] == 100


def test_slo_goodput_uses_observed_tpot_including_handoff_gap() -> None:
    result = simulate_serving_queue(
        [request("r0", 0.0)],
        config("fully_pipelined", tpot_slo_s=0.15),
    )
    assert result["latency"]["decode_service_tpot_s"]["mean"] == pytest.approx(0.1)
    assert result["latency"]["observed_tpot_s"]["mean"] == pytest.approx(0.175)
    assert result["slo_attainment_fraction"] == 0.0
    assert result["slo_goodput_requests_per_s"] == 0.0


def test_poisson_trace_is_reproducible_and_monotonic() -> None:
    left = poisson_arrival_times(rate_per_s=2.0, count=20, seed=7)
    right = poisson_arrival_times(rate_per_s=2.0, count=20, seed=7)
    assert left == right
    assert all(math.isfinite(value) for value in left)
    assert all(next_value > value for value, next_value in zip(left, left[1:]))


def test_invalid_generation_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefill token"):
        request("r0", 0.0, generation_tokens=1)


def test_sensitivity_crosses_rates_regimes_and_replica_pairs() -> None:
    rows = simulate_sensitivity(
        [request("template", 0.0)],
        rates_per_s=(0.5, 1.0),
        regimes=("fully_pipelined", "back_pressure"),
        replica_pairs=((1, 1), (2, 1)),
        base_config=config("fully_pipelined"),
        request_count=4,
        seed=17,
    )
    assert len(rows) == 8
    assert len({row["content_hash"] for row in rows}) == 8
    assert {row["sensitivity"]["arrival_rate_per_s"] for row in rows} == {
        0.5,
        1.0,
    }
