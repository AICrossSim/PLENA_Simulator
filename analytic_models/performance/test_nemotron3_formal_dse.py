import pytest

from analytic_models.performance.nemotron3_formal_dse import build_report, render_markdown


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_formal_report_keeps_gpu_evidence_separate_from_plena_cycles(report) -> None:
    assert report["status"] == "workload_and_gpu_baseline_calibrated_plena_timing_uncalibrated"
    assert report["evidence"]["campaign_status"] == "complete"
    assert report["evidence"]["plena_cycle_calibrated"] is False
    assert report["evidence"]["rtl_ppa_calibrated"] is False
    assert all(
        item["metrics"]["calibrated"] is False
        for item in report["system_dse"]["decode_b1_context2048_steps127"]
    )


def test_decode_logical_traffic_crosschecks_complete_b200_ncu(report) -> None:
    layers = report["gpu_logical_traffic_crosscheck"]["decode_step_s2048"]["layer_types"]
    assert layers["mamba"]["physical_to_logical_read_ratio"] == pytest.approx(0.9846727713)
    assert layers["attention"]["physical_to_logical_read_ratio"] == pytest.approx(1.0033220873)
    assert layers["moe"]["physical_to_logical_read_ratio"] == pytest.approx(1.0734392812)
    assert (
        report["gpu_logical_traffic_crosscheck"]["prefill_s128"]["layer_types"]["mamba"][
            "physical_to_logical_read_ratio"
        ]
        == pytest.approx(1.6782662342)
    )


def test_formal_dse_preserves_exact_state_and_routing_capacity_knees(report) -> None:
    state = report["mamba_state_cache"]["requirements"]
    assert state["fp32"]["required_bytes"] == 50_495_488
    assert state["fp32"]["required_mib"] == 48.15625
    assert state["fp32"]["integer_mib_to_fit"] == 49
    assert state["bf16"]["required_bytes"] == 25_247_744

    routed = report["moe_weight_cache"]["routed_expert"]["access_orders"]["expert_id"]
    by_capacity = {item["capacity_entries"]: item for item in routed}
    assert by_capacity[92]["hit_rate"] == 0.0
    assert by_capacity[138]["hit_rate"] == pytest.approx(0.6811023622)
    assert by_capacity[256]["hit_rate"] == pytest.approx(0.9242268629)

    event_patterns = report["moe_event_dse"]["routing_patterns"]
    assert event_patterns["138"]["misses"] == by_capacity[138]["misses"]
    assert event_patterns["138"]["hits"] == by_capacity[138]["hits"]


def test_formal_dse_couples_cache_misses_to_expert_m_k_timeline(report) -> None:
    records = [
        row
        for row in report["moe_event_dse"]["records"]
        if row["capacity_entries"] == 138 and row["shared_resident"]
    ]
    assert {row["calibration"]["name"] for row in records} == {
        "ideal_geometry_hbm64",
        "transferred_shared_moe",
    }
    for calibration in {row["calibration"]["name"] for row in records}:
        rows = [row for row in records if row["calibration"]["name"] == calibration]
        baseline = next(
            row
            for row in rows
            if row["candidate"]["name"] == "baseline_4x1024__expert"
        )
        winner = min(rows, key=lambda row: row["rank"])
        assert winner["candidate"]["name"] == "row_1_1_1_1__k_split"
        assert winner["speedup_vs_baseline"] > 1
        assert winner["hbm_bytes"] == baseline["hbm_bytes"]
    assert all(
        "calibrated_for_nemotron" in row["calibration"] for row in records
    )


def test_l_compute_and_amdahl_numbers_are_labeled_as_local_bounds(report) -> None:
    layout = report["l_compute_layout"]["nemotron3_mamba_decode"]
    assert layout["comparison"]["read_write_service_speedup"] == pytest.approx(2.2808641975)
    assert all(case["roundtrip_ok"] for case in layout["cases"])
    guardrail = report["guardrails"]["nemotron_decode"]
    assert guardrail["optimistic_whole_system_speedup_upper_bound"] == pytest.approx(1.2510005916)
    assert "not a PLENA-vs-B200 speedup claim" in guardrail["meaning"]
    assert "不能声称相对 B200 加速" in render_markdown(report)
