from __future__ import annotations

from .kimi_k3_model import _parser, build_document
from .kimi_k3_workload import KimiK3Architecture, KimiK3KdaWorkloadModel, default_kimi_k3_scenario
from .nemotron3_workload import InferencePhase, Precision


MIB = 1024 * 1024


def test_official_layer_patterns_are_disjoint_and_complete() -> None:
    arch = KimiK3Architecture()
    assert len(arch.kda_layer_numbers) == 69
    assert len(arch.mla_layer_numbers) == 24
    assert set(arch.kda_layer_numbers).isdisjoint(arch.mla_layer_numbers)
    assert set(arch.kda_layer_numbers) | set(arch.mla_layer_numbers) == set(range(1, 94))
    assert arch.dense_ffn_layer_numbers == (1,)
    assert len(arch.moe_layer_numbers) == 92


def test_bf16_state_capacity_matches_real_kimi_k3() -> None:
    arch = KimiK3Architecture()
    assert arch.recurrent_state_bytes(Precision.BF16) == 207 * MIB
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
    assert document["architecture"]["kda_state_mib_per_request"] == 207.0
