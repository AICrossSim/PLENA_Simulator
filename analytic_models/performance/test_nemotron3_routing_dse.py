import copy
import json

import pytest

from analytic_models.performance.nemotron3_routing_dse import (
    PINNED_TRACE,
    RoutingTraceError,
    build_report,
    load_routing_trace,
    simulate_routed_expert_lru,
)
from transactional_emulator.testbench.model_configs.loader import load_model_config


def _arch():
    return load_model_config("nemotron3_nano_30b_a3b").arch


def test_trace_preserves_all_exact_decode_topk_events() -> None:
    trace = load_routing_trace()
    assert len(trace["decode_topk_by_step"]) == 127
    assert sum(len(experts) for step in trace["decode_topk_by_step"] for experts in step) == 127 * 23 * 6
    by_length = trace["prefill_active_experts_by_sequence_length"]
    assert {tokens: sum(map(len, layers)) for tokens, layers in by_length.items()} == {
        "128": 2185,
        "2048": 2807,
        "8192": 2881,
    }


def test_profile_driven_lru_reproduces_capacity_knees() -> None:
    report = build_report(_arch(), capacity_entries=(23, 92, 138, 2944))
    expert_id = {item["capacity_entries"]: item for item in report["routed_expert"]["access_orders"]["expert_id"]}
    topk_rank = {item["capacity_entries"]: item for item in report["routed_expert"]["access_orders"]["topk_rank"]}

    assert expert_id[23]["hit_rate"] == 0.0
    assert expert_id[92]["hit_rate"] == 0.0
    assert expert_id[138]["hit_rate"] == pytest.approx(0.6811023622)
    assert topk_rank[138]["hit_rate"] == pytest.approx(0.7211000799)
    assert expert_id[2944]["hit_rate"] == pytest.approx(0.9996005934)
    assert expert_id[138]["weight_read_bytes"] < expert_id[92]["weight_read_bytes"]


def test_lru_counts_every_miss_as_one_expert_weight_read() -> None:
    trace = load_routing_trace()
    result = simulate_routed_expert_lru(trace, entry_bytes=1234, capacity_entries=0)
    assert result.accesses == 127 * 23 * 6
    assert result.misses == result.accesses
    assert result.weight_read_bytes == result.misses * 1234


def test_trace_rejects_duplicate_topk_ids(tmp_path) -> None:
    trace = copy.deepcopy(load_routing_trace(PINNED_TRACE))
    trace["decode_topk_by_step"][0][0][1] = trace["decode_topk_by_step"][0][0][0]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(trace))
    with pytest.raises(RoutingTraceError, match="must be unique"):
        load_routing_trace(path)
