"""Composition, pipelining, footprints and validation of the multi-chip estimate."""

from __future__ import annotations

import json
import math

import pytest
from distributed_fixtures import UNBOUNDED, gpt_oss, llama, perf_model, precision, switch_noc

from analytic_models.distributed import ParallelPlan, estimate_distributed
from analytic_models.stacked_dram import FixedBandwidthMemory


@pytest.fixture(scope="module")
def perf():
    return perf_model()


@pytest.fixture(scope="module")
def prec():
    return precision()


def _run(model, plan, noc, perf, prec, memory=UNBOUNDED, **kwargs):
    options = {"batch_size": 8, "input_seq_len": 256, "output_seq_len": 3, **kwargs}
    return estimate_distributed(model, plan, noc, perf, prec, memory, **options)


def test_data_parallel_replicas_add_throughput(perf, prec) -> None:
    replicated = _run(llama(), ParallelPlan(dp=4), switch_noc(4), perf, prec, batch_size=16)
    single = _run(llama(), ParallelPlan(), switch_noc(1), perf, prec, batch_size=4)
    assert replicated.sequences_per_device == 4
    assert replicated.ttft_seconds == pytest.approx(single.ttft_seconds, rel=1e-12)
    assert replicated.tps == pytest.approx(4 * single.tps, rel=1e-12)
    assert replicated.tps_per_sequence == pytest.approx(single.tps_per_sequence, rel=1e-12)


def test_pipeline_period_latency_and_rates(perf, prec) -> None:
    result = _run(llama(), ParallelPlan(pp=2), switch_noc(2), perf, prec, batch_size=7)
    assert result.micro_batch == 4
    for phase in (result.prefill, result.first_token_decode):
        assert [stage.layers for stage in phase.stages] == [16, 16]
        slowest = max(stage.seconds for stage in phase.stages)
        assert phase.period_seconds == pytest.approx(slowest + phase.transfer_seconds, rel=1e-12)
        assert phase.latency_seconds == pytest.approx(
            sum(stage.seconds for stage in phase.stages) + phase.transfer_seconds, rel=1e-12
        )
    # The first stage also runs the embedding lookup.
    assert result.prefill.bottleneck_stage == 0
    assert result.ttft_seconds == pytest.approx(
        result.prefill.latency_seconds + result.first_token_decode.latency_seconds, rel=1e-12
    )
    assert result.tps == pytest.approx(4 * 3 / result.decode_seconds, rel=1e-12)
    assert result.tps_per_sequence == pytest.approx(3 / (2 * result.decode_seconds), rel=1e-12)
    assert any("not a multiple of pp=2" in warning for warning in result.warnings)


def test_uneven_pipeline_stages(perf, prec) -> None:
    result = _run(llama(), ParallelPlan(pp=3), switch_noc(3), perf, prec, batch_size=6)
    assert [stage.layers for stage in result.prefill.stages] == [11, 11, 10]
    empty = _run(llama(), ParallelPlan(pp=12), switch_noc(12), perf, prec, batch_size=12, output_seq_len=1)
    assert [stage.layers for stage in empty.prefill.stages] == [3] * 10 + [2, 0]
    assert any("leave pipeline stages empty" in warning for warning in empty.warnings)


def test_tensor_parallel_footprint_halves(perf, prec) -> None:
    single = _run(llama(), ParallelPlan(), switch_noc(1), perf, prec)
    split = _run(llama(), ParallelPlan(tp=2), switch_noc(2), perf, prec)
    assert split.weight_bytes_per_device == pytest.approx(single.weight_bytes_per_device / 2, rel=1e-12)
    assert split.kv_cache_bytes_per_device == pytest.approx(single.kv_cache_bytes_per_device / 2, rel=1e-12)
    staged = _run(llama(), ParallelPlan(pp=2), switch_noc(2), perf, prec)
    assert staged.weight_bytes_per_device == pytest.approx(single.weight_bytes_per_device / 2, rel=1e-12)


def test_kv_heads_are_replicated_beyond_their_count(perf, prec) -> None:
    result = _run(llama(), ParallelPlan(tp=16), switch_noc(16), perf, prec)
    eight = _run(llama(), ParallelPlan(tp=8), switch_noc(8), perf, prec)
    assert result.kv_cache_bytes_per_device == pytest.approx(eight.kv_cache_bytes_per_device, rel=1e-12)
    assert any("KV heads are replicated" in warning for warning in result.warnings)


def test_sliding_layers_keep_only_their_window(perf, prec) -> None:
    result = _run(gpt_oss(), ParallelPlan(), switch_noc(1), perf, prec, batch_size=2, input_seq_len=1000)
    held = 1000 + 3
    expected = 12 * 2 * 2 * 8 * 64 * (held + min(held, 128)) * prec.kv_write_bytes
    assert result.kv_cache_bytes_per_device == pytest.approx(expected, rel=1e-12)


def test_capacity_check(perf, prec) -> None:
    single = _run(llama(), ParallelPlan(), switch_noc(1), perf, prec)
    needed = single.weight_bytes_per_device + single.kv_cache_bytes_per_device
    small = FixedBandwidthMemory(name="small", bandwidth_bytes_per_s=1e12, capacity_bytes=math.ceil(needed / 2))
    large = FixedBandwidthMemory(name="large", bandwidth_bytes_per_s=1e12, capacity_bytes=math.ceil(needed))
    assert _run(llama(), ParallelPlan(), switch_noc(1), perf, prec, memory=small).fits_in_memory is False
    assert _run(llama(), ParallelPlan(tp=2), switch_noc(2), perf, prec, memory=small).fits_in_memory is True
    result = _run(llama(), ParallelPlan(), switch_noc(1), perf, prec, memory=large)
    assert result.fits_in_memory is True
    assert result.memory_capacity_bytes == math.ceil(needed)


def test_idle_replicas_are_flagged(perf, prec) -> None:
    result = _run(llama(), ParallelPlan(dp=4), switch_noc(4), perf, prec, batch_size=2)
    assert result.sequences_per_device == 1
    assert any("idle" in warning for warning in result.warnings)


def test_result_is_json_serialisable(perf, prec) -> None:
    result = _run(gpt_oss(), ParallelPlan(tp=2, ep=4, pp=1), switch_noc(8), perf, prec, routing="random")
    data = json.loads(json.dumps(result.to_dict()))
    assert data["moe"]["ep"] == 8
    assert data["prefill"]["stages"][0]["comm_seconds"]["ep_dispatch"] > 0


@pytest.mark.parametrize(
    ("model", "plan", "devices", "kwargs", "message"),
    [
        (llama, ParallelPlan(tp=2), 4, {}, "uses 2 devices"),
        (llama, ParallelPlan(tp=3), 3, {}, "does not divide"),
        (llama, ParallelPlan(ep=2), 2, {}, "needs a model with MoE"),
        (llama, ParallelPlan(pp=33), 33, {}, "exceeds the 32 layers"),
        (llama, ParallelPlan(cp=2), 2, {"input_seq_len": 1}, "exceeds the 1-token prompt"),
        (gpt_oss, ParallelPlan(cp=2), 2, {}, "dense full-attention models only"),
        (gpt_oss, ParallelPlan(ep=3), 3, {}, "do not split evenly"),
        (gpt_oss, ParallelPlan(ep=2), 2, {"routing": "uniform"}, "unknown routing mode"),
        (llama, ParallelPlan(), 1, {"comm_overlap": "partial"}, "comm_overlap"),
        (llama, ParallelPlan(), 1, {"overlap_policy": "magic"}, "overlap_policy"),
    ],
)
def test_invalid_plans_are_rejected(perf, prec, model, plan, devices, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _run(model(), plan, switch_noc(devices), perf, prec, **kwargs)
