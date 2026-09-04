from __future__ import annotations

import json
from pathlib import Path

import pytest

from .agentic_matrix_lcompute_campaign import (
    WEIGHT_PRECISION_SCENARIOS,
    _d_prime_bank_control,
    _summary_rows,
    _timeline_endpoints,
)
from .hybrid_routing import RoutingProfile, RoutingStep
from .matrix_lcompute_campaign import _routing_for_scenario
from .nemotron3_workload import InferencePhase


ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts/matrix_lcompute_agentic_v1"


def test_timeline_endpoints_separate_serial_execution_from_optimistic_overlap() -> None:
    endpoints = _timeline_endpoints(
        {
            "cycles": 100,
            "hbm_cycles": 10,
            "matrix_cycles": 20,
            "vector_cycles": 30,
            "lcompute_cycles": 40,
        }
    )
    assert endpoints == {
        "strict_serial_cycles": 100,
        "ideal_resource_overlap_lower_bound_cycles": 70,
        "resource_cycles": {
            "hbm": 10,
            "matrix": 20,
            "vector_plus_lcompute": 70,
        },
    }


def test_timeline_endpoints_reject_a_hidden_overlap_credit() -> None:
    with pytest.raises(AssertionError, match="strict-serial timeline"):
        _timeline_endpoints(
            {
                "cycles": 99,
                "hbm_cycles": 10,
                "matrix_cycles": 20,
                "vector_cycles": 30,
                "lcompute_cycles": 40,
            }
        )


def test_strict_routing_cannot_silently_fall_back_to_the_expert_bound() -> None:
    profile = RoutingProfile(
        model_key="nemotron3",
        model_id="test",
        revision="test",
        batch_size=1,
        context_length=128,
        expert_count=128,
        top_k=6,
        source_sha256="a" * 64,
        steps=(
            RoutingStep(
                phase="decode",
                index=0,
                token_count=1,
                active_experts_by_layer=((1, (2, 7, 9)),),
            ),
        ),
    )
    exact = _routing_for_scenario(
        model="nemotron3",
        phase=InferencePhase.DECODE,
        batch_size=1,
        context_length=128,
        sequence_length=1,
        decode_index=0,
        profile=profile,
        strict_profile=True,
    )
    assert exact == (None, ((1, 3),))

    with pytest.raises(ValueError, match="strict routing profile"):
        _routing_for_scenario(
            model="nemotron3",
            phase=InferencePhase.DECODE,
            batch_size=1,
            context_length=129,
            sequence_length=1,
            decode_index=0,
            profile=profile,
            strict_profile=True,
        )
    assert _routing_for_scenario(
        model="nemotron3",
        phase=InferencePhase.DECODE,
        batch_size=1,
        context_length=129,
        sequence_length=1,
        decode_index=0,
        profile=profile,
        strict_profile=False,
    ) == (6, ())


def test_d_prime_remains_a_bank_only_control() -> None:
    control = _d_prime_bank_control(
        {
            "fixed_phased_bank_control": {
                "nemotron3": {
                    "mapping": {"fixed_alpha": 1},
                    "service_cycles": 1,
                    "ideal_cycles": 1,
                    "bank_stall_cycles": 0,
                    "roundtrip_values_checked": 262_144,
                    "same_physical_coordinates_as_compact_tile_phase": True,
                    "compact_phase_vs_explicit_bases_bank_speedup": 1.0,
                }
            }
        },
        {"bank_stall_cycles": 0, "matrix_sram_service_cycles": 1234},
    )
    assert control["scope"].startswith("bank service only")
    assert control["workload_matrix_service_cycles"] == 1234
    assert control["D_vs_D_prime_pure_bank_speedup"] == 1.0
    assert "timeline" not in control


def test_checked_agentic_artifact_is_self_consistent() -> None:
    campaign = json.loads((ARTIFACT_ROOT / "campaign.json").read_text())
    assert campaign["contract"] == "nemotron-agentic-matrix-lcompute-dse-v2"
    assert campaign["routing_contract"]["fallback_allowed"] is False
    assert campaign["group_count"] == len(campaign["groups"]) == 93
    assert campaign["summary"] == _summary_rows(campaign["groups"])
    assert campaign["source"]["timing_and_routing_tokens_identical_samples"] == 3
    assert campaign["source"]["timing_and_routing_replay_window_identical_samples"] == 20
    assert campaign["source"]["routing_event_accounting"] == {
        "raw_events": 140_921,
        "conservation_validated_events": 140_921,
        "prefill_events": 1_104,
        "decode_events": 139_817,
        "fully_validated_decode_events": 139_817,
        "used_decode_events": 35_328,
        "ignored_decode_events": 104_489,
    }
    assert campaign["source"]["gpu_global_aggregates"]["batch_b16"]["trial_measurements"] == 60
    assert set(campaign["weight_precision_contract"]) == {
        name for name, _precision, _description in WEIGHT_PRECISION_SCENARIOS
    }
    accounting = campaign["source"]["routing_event_accounting"]
    assert accounting["raw_events"] == accounting["prefill_events"] + accounting["decode_events"]
    assert accounting["decode_events"] == (accounting["used_decode_events"] + accounting["ignored_decode_events"])
    assert all(
        group["plena"]["D_prime_bank_control"]["D_vs_D_prime_pure_bank_speedup"] == 1.0 for group in campaign["groups"]
    )
    readme = (ARTIFACT_ROOT / "README.md").read_text()
    full_identical = campaign["source"]["timing_and_routing_tokens_identical_samples"]
    sample_count = campaign["source"]["sample_count"]
    assert f"{full_identical}/{sample_count}" in readme


def test_summary_marks_low_sample_p95_as_exploratory(monkeypatch: pytest.MonkeyPatch) -> None:
    template = {
        "benchmark": "fixture",
        "batch_size": 16,
        "D_bank_stall_cycles": 0,
    }
    metric_names = (
        "padding_fraction",
        "active_experts_median",
        "active_experts_p95",
        "gpu_ttft_ms_median",
        "gpu_itl_ms_median",
        "gpu_e2e_ms_median",
        "gpu_batch_throughput_tokens_s_median",
        "gpu_batch_energy_joules_median",
        "A_original_cycles",
        "B_arlo_cycles",
        "C_fixed_cycles",
        "D_phased_cycles",
        "D_speedup_vs_A",
        "D_speedup_vs_B",
        "D_speedup_vs_B_ideal_resource_overlap",
        "D_speedup_vs_C",
        "D_tpot_ms_proxy",
        "D_tpot_ms_ideal_resource_overlap_lower_bound",
        "D_aggregate_throughput_tokens_s_proxy",
        "D_aggregate_throughput_tokens_s_ideal_resource_overlap_upper_bound",
        "B_ideal_resource_overlap_cycles",
        "D_ideal_resource_overlap_cycles",
        "D_vs_D_prime_pure_bank_speedup",
    )
    for name in metric_names:
        template[name] = 1.0
    for precision, _value, _description in WEIGHT_PRECISION_SCENARIOS:
        for suffix in (
            "D_speedup_vs_B_serial",
            "D_speedup_vs_B_ideal_overlap",
            "D_logical_weight_read_bytes",
            "D_physical_hbm_read_bytes",
        ):
            template[f"{precision}_{suffix}"] = 1.0

    monkeypatch.setattr(
        "analytic_models.performance.agentic_matrix_lcompute_campaign._flatten_group",
        lambda row: row,
    )
    summaries = _summary_rows([{**template, "group": index} for index in range(3)])
    assert len(summaries) == 2
    assert all(row["group_count"] == 3 for row in summaries)
    assert all(row["p95_status"] == "exploratory_low_n" for row in summaries)
