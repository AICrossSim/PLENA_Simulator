"""Collectives and pipeline transfers are DeepStack's, called with the per-device shapes."""

from __future__ import annotations

import pytest
from distributed_fixtures import UNBOUNDED, gpt_oss, llama, perf_model, precision, switch_noc

from analytic_models.distributed import ModelSpec, ParallelPlan, estimate_distributed, routing_rows
from analytic_models.distributed._deepstack import (
    Modeling_Granularity,
    TrafficMatrix,
    all_gather_wrapper,
    all_reduce_ring,
    all_reduce_wrapper,
    build_route_stats_from_extended,
    ep_all_to_all_wrapper,
    get_extend_max_routes_with_traffic,
)

GRANULARITY = Modeling_Granularity("coarse", False, False)


@pytest.fixture(scope="module")
def perf():
    return perf_model()


@pytest.fixture(scope="module")
def prec():
    return precision()


def _run(model, plan, noc, perf, prec, **kwargs):
    options = {"batch_size": 4, "input_seq_len": 256, "output_seq_len": 2, **kwargs}
    return estimate_distributed(model, plan, noc, perf, prec, UNBOUNDED, **options)


def test_tp_all_reduce_is_deepstacks_wrapper_twice_per_layer(perf, prec) -> None:
    noc = switch_noc(8)
    plan = ParallelPlan(tp=8)
    result = _run(llama(), plan, noc, perf, prec)
    activation = int(4 * 4096 * prec.activation_bytes)
    hop, link, _ = all_reduce_wrapper(None, plan.dense_scheme(), noc.hierarchy, GRANULARITY, "tp", activation)

    stage = result.first_token_decode.stages[0]
    assert set(stage.comm_seconds) == {"tp_all_reduce"}
    assert stage.comm_seconds["tp_all_reduce"] == pytest.approx(2 * 32 * (hop + link), rel=1e-12)
    # Without overlap a layer takes its compute (memory is unbounded here) plus its collectives.
    assert stage.seconds == pytest.approx(stage.compute_seconds + stage.comm_seconds["tp_all_reduce"], rel=1e-12)


def test_non_power_of_two_tp_groups_use_the_ring(perf, prec) -> None:
    model = ModelSpec.from_hf_config(
        {
            "hidden_size": 1536,
            "num_attention_heads": 24,
            "num_key_value_heads": 6,
            "num_hidden_layers": 2,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        },
        name="fictional-24-head",
    )
    noc = switch_noc(6)
    plan = ParallelPlan(tp=6)
    result = _run(model, plan, noc, perf, prec)
    activation = int(4 * 1536 * prec.activation_bytes)
    hop, link, _ = all_reduce_ring(None, plan.dense_scheme(), noc.hierarchy, GRANULARITY, "tp", activation)
    stage = result.first_token_decode.stages[0]
    assert stage.comm_seconds["tp_all_reduce"] == pytest.approx(2 * 2 * (hop + link), rel=1e-12)


def test_comm_overlap_hides_collectives_behind_compute(perf, prec) -> None:
    noc = switch_noc(8)
    serial = _run(llama(), ParallelPlan(tp=8), noc, perf, prec)
    hidden = _run(llama(), ParallelPlan(tp=8), noc, perf, prec, comm_overlap="full")
    for a, b in ((serial.prefill, hidden.prefill), (serial.first_token_decode, hidden.first_token_decode)):
        overlapped, sequential = b.stages[0], a.stages[0]
        assert overlapped.seconds < sequential.seconds
        assert overlapped.seconds >= max(overlapped.compute_seconds, sum(overlapped.comm_seconds.values()))
    assert hidden.tps > serial.tps


def test_pipeline_transfer_is_the_slowest_stage_boundary(perf, prec) -> None:
    noc = switch_noc(4, groups=2)
    plan = ParallelPlan(tp=2, pp=2)
    result = _run(llama(), plan, noc, perf, prec, batch_size=8)
    scheme = plan.dense_scheme()
    micro_per_device = 4  # ceil(8 / pp)
    for phase_result, q_len in ((result.prefill, 256), (result.first_token_decode, 1)):
        tm = TrafficMatrix(scheme.world_size())
        nbytes = int(micro_per_device * q_len * 4096 * prec.activation_bytes)
        tm.add_intra_group_traffic_pair_bulk("pp", [[nbytes, 0, 1]], tp=2, ep=1, sp=1, cp=1, dp=1, pp=2)
        hop, link, _, _ = get_extend_max_routes_with_traffic(tm, noc.hierarchy)
        assert phase_result.transfer_seconds == pytest.approx(hop + link, rel=1e-12)
        assert phase_result.transfer_seconds > 0


def test_context_parallel_collectives(perf, prec) -> None:
    noc = switch_noc(2)
    plan = ParallelPlan(cp=2)
    result = _run(llama(), plan, noc, perf, prec)
    scheme = plan.dense_scheme()
    head_dim, kv_heads = 128, 8
    combine = int(4 * 32 * head_dim * prec.activation_bytes)
    hop, link, _ = all_reduce_wrapper(None, scheme, noc.hierarchy, GRANULARITY, "cp", combine)
    assert result.first_token_decode.stages[0].comm_seconds["cp_attention_combine"] == pytest.approx(
        32 * (hop + link), rel=1e-12
    )
    gathered = int(2 * 4 * 256 * kv_heads * head_dim * prec.kv_write_bytes)
    hop, link, _ = all_gather_wrapper(None, scheme, noc.hierarchy, GRANULARITY, "cp", gathered)
    assert result.prefill.stages[0].comm_seconds["cp_kv_all_gather"] == pytest.approx(32 * (hop + link), rel=1e-12)

    single = _run(llama(), ParallelPlan(), switch_noc(1), perf, prec)
    assert result.prefill.stages[0].compute_seconds < single.prefill.stages[0].compute_seconds
    assert result.kv_cache_bytes_per_device == pytest.approx(single.kv_cache_bytes_per_device / 2, rel=1e-12)


def test_expert_dispatch_is_deepstacks_all_to_all(perf, prec) -> None:
    noc = switch_noc(8)
    plan = ParallelPlan(ep=8)
    model = gpt_oss()
    result = _run(model, plan, noc, perf, prec, batch_size=16)
    scheme = plan.moe_scheme(model.num_experts)
    rows = routing_rows("balanced", phase="decode", tokens=16, num_experts=32, top_k=4)
    hop, link, _ = ep_all_to_all_wrapper(
        scheme, noc.hierarchy, GRANULARITY, int(2880 * prec.activation_bytes), rows, 16, 1, 32, 4, 1.0
    )
    stage = result.first_token_decode.stages[0]
    assert set(stage.comm_seconds) == {"ep_dispatch", "ep_combine"}
    assert stage.comm_seconds["ep_dispatch"] == pytest.approx(24 * (hop + link), rel=1e-12)
    assert stage.comm_seconds["ep_combine"] == stage.comm_seconds["ep_dispatch"]


def test_noc_energy_per_decode_token(perf, prec) -> None:
    noc = switch_noc(8)
    plan = ParallelPlan(tp=8)
    result = _run(llama(), plan, noc, perf, prec)
    activation = int(4 * 4096 * prec.activation_bytes)
    hop, link, traffic = all_reduce_wrapper(None, plan.dense_scheme(), noc.hierarchy, GRANULARITY, "tp", activation)
    per_call = build_route_stats_from_extended([traffic], noc.hierarchy, hop + link, noc.energy).total_noc_energy_pj
    assert per_call > 0
    # 32 layers with two all-reduces each, one pipeline stage, four tokens per period.
    assert result.noc_energy_pj_per_decode_token == pytest.approx(2 * 32 * per_call / 4, rel=1e-12)

    no_energy = _run(llama(), plan, switch_noc(8, energy=False), perf, prec)
    assert no_energy.noc_energy_pj_per_decode_token is None
    assert no_energy.tps == pytest.approx(result.tps, rel=1e-12)


def test_pipeline_energy_counts_every_stage_boundary(perf, prec) -> None:
    noc = switch_noc(4)
    plan = ParallelPlan(pp=4)
    result = _run(llama(), plan, noc, perf, prec, batch_size=8)
    scheme = plan.dense_scheme()
    nbytes = int(2 * 4096 * prec.activation_bytes)
    expected = 0.0
    for stage in range(3):
        tm = TrafficMatrix(scheme.world_size())
        tm.add_intra_group_traffic_pair_bulk("pp", [[nbytes, stage, stage + 1]], tp=1, ep=1, sp=1, cp=1, dp=1, pp=4)
        hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, noc.hierarchy)
        expected += build_route_stats_from_extended(
            [traffic], noc.hierarchy, hop + link, noc.energy
        ).total_noc_energy_pj
    assert result.noc_energy_pj_per_decode_token == pytest.approx(expected / 2, rel=1e-12)
