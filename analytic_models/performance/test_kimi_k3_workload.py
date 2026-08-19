from __future__ import annotations

from .b200_formal_campaign import PINNED_SUMMARY
from .kimi_k3_model import _parser, build_document
from .kimi_k3_workload import (
    KimiK3Architecture,
    KimiK3HybridWorkloadModel,
    KimiK3KdaWorkloadModel,
    default_kimi_k3_scenario,
)
from .nemotron3_workload import InferencePhase, Precision, storage_bytes


MIB = 1024 * 1024


def test_official_layer_patterns_are_disjoint_and_complete() -> None:
    arch = KimiK3Architecture()
    assert len(arch.kda_layer_numbers) == 69
    assert len(arch.mla_layer_numbers) == 24
    assert set(arch.kda_layer_numbers).isdisjoint(arch.mla_layer_numbers)
    assert set(arch.kda_layer_numbers) | set(arch.mla_layer_numbers) == set(range(1, 94))
    assert arch.dense_ffn_layer_numbers == (1,)
    assert len(arch.moe_layer_numbers) == 92


def test_profiled_mixed_state_capacity_matches_real_kimi_k3() -> None:
    arch = KimiK3Architecture()
    assert arch.recurrent_state_bytes() == 414 * MIB
    assert arch.conv_state_bytes(Precision.BF16) == int(19.40625 * MIB)


def test_decode_charges_one_state_read_and_write_per_kda_layer() -> None:
    arch = KimiK3Architecture()
    scenario = default_kimi_k3_scenario(InferencePhase.DECODE, batch_size=1)
    report = KimiK3KdaWorkloadModel(arch).build(scenario)
    traffic = report.total_traffic
    expected = arch.recurrent_state_bytes() + arch.conv_state_bytes()
    assert traffic.state_read_bytes == expected
    assert traffic.state_write_bytes == expected
    assert report.to_dict()["layer_counts"] == {"kda": 69}


def test_prefill_reads_no_initial_state_and_commits_final_state() -> None:
    arch = KimiK3Architecture()
    scenario = default_kimi_k3_scenario(InferencePhase.PREFILL, sequence_length=128)
    report = KimiK3KdaWorkloadModel(arch).build(scenario)
    assert report.total_traffic.state_read_bytes == 0
    assert report.total_traffic.state_write_bytes == arch.recurrent_state_bytes() + arch.conv_state_bytes()


def test_recurrent_core_counts_three_state_sized_mac_passes() -> None:
    arch = KimiK3Architecture()
    report = KimiK3KdaWorkloadModel(arch).build(default_kimi_k3_scenario())
    one_layer = [stage for stage in report.stages if stage.layer_id == 0 and stage.resource == "state"]
    assert sum(stage.macs for stage in one_layer) == 3 * arch.kda.state_elements


def test_cli_is_explicitly_kda_only() -> None:
    document = build_document(_parser().parse_args([]))
    assert document["scope"] == "text_backbone_kda_mixers_only"
    assert set(document["excluded"]) == {"MLA", "LatentMoE", "dense FFN", "AttnRes", "vision tower"}
    assert document["architecture"]["kda_state_mib_per_request"] == 414.0
    assert document["architecture"]["kda_conv_state_mib_per_request"] == 19.40625
    assert document["architecture"]["state_precision"] == "fp32"
    assert document["architecture"]["conv_state_precision"] == "bf16"
    profile = document["gpu_profile"]
    assert profile["revision"] == "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    assert profile["observed_state"]["persistent_mib_for_69_layers"] == 433.40625
    assert profile["latency"][4] == {
        "phase": "decode",
        "batch": 1,
        "sequence": 1,
        "median_us": 565.0,
        "p95_us": 567.0,
    }
    assert "PLENA stage cycles" in profile["not_calibrated"]


def test_full_workload_contains_every_real_kimi_component() -> None:
    arch = KimiK3Architecture()
    scenario = default_kimi_k3_scenario(InferencePhase.DECODE, batch_size=1)
    report = KimiK3HybridWorkloadModel(arch).build(scenario)
    names = [stage.name for stage in report.stages]
    assert names.count("kda_qkv_projection") == 69
    assert names.count("mla_q_low_rank_projection") == 24
    assert names.count("dense_situ_ffn") == 1
    assert names.count("latent_moe_router_top16") == 92
    assert names.count("attn_res_capture_prefix") == 8
    assert names.count("attn_res_output") == 1
    assert names.count("final_rms_norm") == 1
    assert report.total_traffic.state_read_bytes == (
        arch.recurrent_state_bytes() + arch.conv_state_bytes()
    )
    assert report.total_traffic.kv_write_bytes == (
        24 * storage_bytes(arch.mla_cache_elements_per_token, Precision.BF16)
    )


def test_cli_full_scope_excludes_only_the_vision_tower() -> None:
    document = build_document(_parser().parse_args(["--scope", "full"]))
    assert document["scope"] == "full_93_layer_text_backbone"
    assert document["excluded"] == ["vision tower"]
    assert document["workload"]["layer_counts"]["kda"] == 69
    assert document["workload"]["layer_counts"]["mla"] == 24


def test_cli_attaches_formal_stage_evidence_without_claiming_plena_calibration() -> None:
    document = build_document(
        _parser().parse_args(
            ["--formal-b200-campaign-summary", str(PINNED_SUMMARY)]
        )
    )
    assert document["formal_b200_campaign"]["decode_b8_to_b1_kernel_time_ratio"] > 1
    assert document["calibration"]["campaign_status"] == "complete"
    assert document["calibration"]["plena_timing"] == "uncalibrated_gpu_time_is_not_plena_cycles"
    assert "62-74%" in document["modeling_assumptions"]["kda_bottleneck"]
