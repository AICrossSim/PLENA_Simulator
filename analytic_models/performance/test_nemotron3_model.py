import json

import pytest

import analytic_models.performance.nemotron3_model as nemotron3_model
from analytic_models.performance.b200_formal_campaign import PINNED_SUMMARY
from analytic_models.performance.nemotron3_model import _parser, build_document, main


def test_workload_cli_document_uses_real_hybrid_model() -> None:
    args = _parser().parse_args(["--mode", "workload", "--body-only"])
    document = build_document(args)

    assert document["calibration"]["status"] == "uncalibrated_no_gpu_or_rtl"
    assert document["workload"]["layer_counts"] == {"mamba": 23, "moe": 23, "attention": 6}


def test_sweep_cli_builds_only_valid_cache_combinations() -> None:
    args = _parser().parse_args(
        [
            "--mode",
            "sweep",
            "--body-only",
            "--decode-tokens",
            "2",
            "--sweep-layouts",
            "row_major,group_major_skewed",
            "--sweep-broadcasts",
            "0,1",
            "--sweep-cache-mib",
            "0,64",
            "--sweep-cache-policies",
            "none,lru",
            "--sweep-state-dim-lanes",
            "8",
        ]
    )
    document = build_document(args)

    assert document["design_count"] == 8
    assert all(result["metrics"]["calibrated"] is False for result in document["results"])
    assert document["results"] == sorted(document["results"], key=lambda result: result["metrics"]["total_cycles"])


def test_cli_writes_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "report.json"
    status = main(["--mode", "dse", "--body-only", "--json-out", str(output)])

    assert status == 0
    parsed = json.loads(output.read_text())
    assert parsed["mode"] == "dse"
    assert len(parsed["results"]) == 1


def test_gpu_microprofile_validates_workload_without_claiming_plena_calibration(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        nemotron3_model,
        "build_microprofile_report",
        lambda _: {
            "observed_precision": {"ssm_state": "bfloat16"},
            "profile_scope": "standalone mixer",
        },
    )
    args = _parser().parse_args(
        [
            "--mode",
            "dse",
            "--body-only",
            "--state-precision",
            "bf16",
            "--gpu-microprofile-dir",
            str(tmp_path),
        ]
    )
    document = build_document(args)

    assert document["calibration"]["status"] == "gpu_workload_validated_plena_uncalibrated"
    assert document["calibration"]["state_precision_match"] is True
    assert document["results"][0]["metrics"]["calibrated"] is False


def test_formal_campaign_attaches_real_routing_but_keeps_cycles_uncalibrated() -> None:
    args = _parser().parse_args(
        [
            "--mode",
            "dse",
            "--body-only",
            "--weight-precision",
            "nvfp4",
            "--formal-b200-campaign-summary",
            str(PINNED_SUMMARY),
            "--formal-checkpoint-weight-map",
        ]
    )
    document = build_document(args)

    assert document["formal_b200_campaign"]["routing"]["event_count"] == 3013
    assert document["calibration"]["routing"] == "3013_real_layer_step_events_non_uniform"
    assert document["calibration"]["formal_weight_precision_match"] is True
    assert document["calibration"]["transactional_weight_precision_match"] is False
    assert document["calibration"]["plena_timing"] == "uncalibrated_gpu_time_is_not_plena_cycles"
    capacity = document["formal_routing_capacity_guardrail"]
    assert capacity["routed_expert_mib_per_layer"] == pytest.approx(
        (2 * 2688 * 1856 // 2 + (2 * 2688 * 1856 + 15) // 16) / (1024**2)
    )
    assert capacity["one_hottest_routed_expert_for_all_moe_layers_mib"] == pytest.approx(
        capacity["routed_expert_mib_per_layer"] * 23
    )
    assert capacity["one_hottest_expert_per_layer_assignment_coverage"] == pytest.approx(
        32785 / 300150
    )
    assert "not a cache-hit prediction" in capacity["interpretation"]
    assert document["results"][0]["metrics"]["calibrated"] is False
    guardrail = document["formal_b200_guardrail"]
    assert guardrail["gpu_decode_itl_median_ms"] == 4.047566
    assert guardrail["candidate_latency_over_gpu"] > 1
    assert guardrail["minimum_hbm_bytes_per_cycle_to_match_gpu_ignoring_compute"] > 300
    assert "not calibrated speedup" in guardrail["interpretation"]


def test_uniform_nvfp4_is_not_mislabeled_as_the_formal_mixed_checkpoint() -> None:
    args = _parser().parse_args(
        [
            "--mode",
            "dse",
            "--body-only",
            "--weight-precision",
            "nvfp4",
            "--formal-b200-campaign-summary",
            str(PINNED_SUMMARY),
        ]
    )
    document = build_document(args)

    assert document["calibration"]["formal_weight_precision_match"] is False


def test_formal_weight_map_requires_the_campaign_contract() -> None:
    args = _parser().parse_args(["--formal-checkpoint-weight-map"])
    with pytest.raises(ValueError, match="requires --formal-b200-campaign-summary"):
        build_document(args)
