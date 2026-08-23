from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from .hybrid_system_timeline import (
    HybridSystemTimelineModel,
    ModelFamily,
    Resource,
    SystemDesign,
)
from .kimi_k3_workload import KimiK3Architecture
from .nemotron3_dse import ProjectionLayout, StateCachePolicy
from .nemotron3_workload import InferencePhase, Precision, storage_bytes


@pytest.fixture(scope="module")
def decode_reports():
    design = SystemDesign()
    return {
        model: HybridSystemTimelineModel(model, design).simulate(
            InferencePhase.DECODE,
            context_length=16,
            decode_tokens=2,
            include_embedding=False,
            include_lm_head=False,
        )
        for model in ModelFamily
    }


@pytest.fixture(scope="module")
def prefill_reports():
    design = SystemDesign()
    return {
        model: HybridSystemTimelineModel(model, design).simulate(
            InferencePhase.PREFILL,
            sequence_length=16,
            include_embedding=False,
            include_lm_head=False,
        )
        for model in ModelFamily
    }


def _layer_counts(report) -> Counter[str]:
    unique = {(stage.layer_type, stage.layer_id) for stage in report.stages if stage.layer_id >= 0}
    return Counter(layer_type for layer_type, _ in unique)


def test_decode_executes_complete_hybrid_backbones(decode_reports) -> None:
    assert _layer_counts(decode_reports[ModelFamily.NEMOTRON3]) == {
        "mamba": 23,
        "moe": 23,
        "attention": 6,
    }
    assert _layer_counts(decode_reports[ModelFamily.KIMI_K3]) == {
        "kda": 69,
        "mla": 24,
        "latent_moe": 92,
        "attn_res": 93,
        "dense": 1,
    }


def test_multi_token_decode_grows_context_and_reports_tpot(decode_reports) -> None:
    for report in decode_reports.values():
        assert [token.context_length for token in report.tokens] == [16, 17]
        assert all(token.cycles > 0 for token in report.tokens)
        document = report.to_dict(include_stages=False)
        assert document["metrics"]["tpot_cycles"] == pytest.approx(sum(token.cycles for token in report.tokens) / 2)
        assert document["metrics"]["ttft_cycles"] is None


def test_prefill_uses_mamba_and_kda_chunk_algorithms(prefill_reports) -> None:
    nemotron_names = Counter(stage.name for stage in prefill_reports[ModelFamily.NEMOTRON3].stages)
    assert nemotron_names["mamba_chunk_intra_cb"] == 23
    assert nemotron_names["mamba_chunk_scan_compose"] == 23

    kimi_names = Counter(stage.name for stage in prefill_reports[ModelFamily.KIMI_K3].stages)
    assert kimi_names["kda_chunk_prepare"] == 69
    assert kimi_names["kda_chunk_recurrence_output"] == 69
    document = prefill_reports[ModelFamily.KIMI_K3].to_dict(include_stages=False)
    assert document["metrics"]["prompt_tokens"] == 16
    assert document["metrics"]["ttft_cycles"] > 0
    assert document["metrics"]["tpot_cycles"] is None


def test_nemotron_prefill_routing_matches_the_requested_sequence_length() -> None:
    model = HybridSystemTimelineModel(ModelFamily.NEMOTRON3, SystemDesign())
    s128 = model._nemotron_prefill_active_experts(128, decode_warm_start=False)
    fallback = model._nemotron_prefill_active_experts(16, decode_warm_start=False)

    assert sum(map(len, s128)) == 2185
    assert sum(map(len, fallback)) == 2804
    assert (
        model._moe_cache(
            InferencePhase.PREFILL,
            prompt_tokens=128,
        ).routing_source
        == "exact_nemotron_prefill_s128_active_set"
    )


def test_kimi_mla_keeps_only_compressed_cache(decode_reports) -> None:
    report = decode_reports[ModelFamily.KIMI_K3]
    arch = KimiK3Architecture()
    writes = sum(stage.logical_hbm_write_bytes for stage in report.stages if stage.name == "mla_kv_latent_projection")
    expected_per_token = len(arch.mla_layer_numbers) * storage_bytes(arch.mla_cache_elements_per_token, Precision.BF16)
    assert writes == 2 * expected_per_token
    expanded_per_token = len(arch.mla_layer_numbers) * storage_bytes(
        arch.kda.num_heads * (arch.mla_q_head_dim + arch.v_head_dim),
        Precision.BF16,
    )
    assert expected_per_token < expanded_per_token / 50


def test_all_stage_types_share_one_hbm_and_sram_server(decode_reports) -> None:
    report = decode_reports[ModelFamily.NEMOTRON3]
    hbm_users = {
        stage.layer_type for stage in report.stages if any(span.resource == Resource.HBM for span in stage.spans)
    }
    sram_users = {
        stage.layer_type for stage in report.stages if any(span.resource == Resource.SRAM for span in stage.spans)
    }
    assert {"mamba", "attention", "moe"} <= hbm_users
    assert {"mamba", "attention", "moe"} <= sram_users
    assert report.resource_queue_wait_cycles[Resource.HBM] > 0


@pytest.mark.parametrize("model,cache_mib", [(ModelFamily.NEMOTRON3, 64), (ModelFamily.KIMI_K3, 512)])
def test_full_state_cache_removes_per_token_state_hbm(model, cache_mib) -> None:
    base = SystemDesign()
    cached = replace(
        base,
        state_cache_bytes=cache_mib * 1024 * 1024,
        state_cache_policy=StateCachePolicy.PINNED,
    )
    no_cache_report = HybridSystemTimelineModel(model, base).simulate(
        InferencePhase.DECODE,
        context_length=16,
        decode_tokens=2,
        include_embedding=False,
        include_lm_head=False,
    )
    cached_report = HybridSystemTimelineModel(model, cached).simulate(
        InferencePhase.DECODE,
        context_length=16,
        decode_tokens=2,
        include_embedding=False,
        include_lm_head=False,
    )
    assert cached_report.state_cache.hit_rate == 1.0
    assert cached_report.state_cache.final_flush_bytes > 0
    assert sum(stage.logical_hbm_read_bytes for stage in cached_report.stages) < sum(
        stage.logical_hbm_read_bytes for stage in no_cache_report.stages
    )


@pytest.mark.parametrize("model", list(ModelFamily))
def test_consumer_aware_layout_reaches_ideal_bank_service(model) -> None:
    row = SystemDesign(projection_layout=ProjectionLayout.ROW_MAJOR)
    skew = replace(row, projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED)
    row_report = HybridSystemTimelineModel(model, row).simulate(
        InferencePhase.DECODE,
        context_length=16,
        decode_tokens=1,
        include_embedding=False,
        include_lm_head=False,
    )
    skew_report = HybridSystemTimelineModel(model, skew).simulate(
        InferencePhase.DECODE,
        context_length=16,
        decode_tokens=1,
        include_embedding=False,
        include_lm_head=False,
    )
    row_service = sum(stage.bank_service_cycles for stage in row_report.stages)
    skew_service = sum(stage.bank_service_cycles for stage in skew_report.stages)
    assert skew_service < row_service
    assert sum(stage.bank_stall_cycles for stage in row_report.stages) > 0
    assert sum(stage.bank_stall_cycles for stage in skew_report.stages) == 0


def test_mx_storage_contracts_keep_formats_distinct() -> None:
    assert storage_bytes(128, Precision.MX8) == 129
    assert storage_bytes(128, Precision.MXFP8) == 132
    assert storage_bytes(128, Precision.MXFP4) == 68
    assert storage_bytes(128, Precision.NVFP4) == 72
