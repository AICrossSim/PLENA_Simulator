from __future__ import annotations

from .report_fixed_hardware_cross_workloads import (
    _find_measurement_row,
    _fixed_hardware_mismatches,
    _flat_row,
)


def test_find_measurement_row_accepts_recorded_warning() -> None:
    row = _find_measurement_row(
        [
            {
                "model": "m",
                "tensor_parallel_size": 2,
                "local_batch_size": 4,
                "validation_status": "warning",
            }
        ],
        model="m",
        tp=2,
        local_batch=4,
    )

    assert row["validation_status"] == "warning"


def test_fixed_hardware_mismatches_ignores_policy_name_but_checks_tiles() -> None:
    primary = {
        "precision_profile": "q",
        "MLEN": 1024,
        "VLEN": 1024,
        "BLEN": 128,
        "INT_DATA_WIDTH": 32,
        "softmax_row_lanes": 8,
        "matrix_sram_tiles": 5,
        "physical_chip_count": 32,
        "dp_degree": 4,
        "tp_degree": 8,
        "ep_degree": 1,
        "nvlink_port_count": 1,
        "total_silicon_area_mm2": 100.0,
    }
    holdout = {
        **primary,
        "matrix_sram_policy": "raw-tiles",
        "MATRIX_SRAM_TILES": 5,
    }

    assert _fixed_hardware_mismatches(primary, holdout) == {}

    holdout["MATRIX_SRAM_TILES"] = 6
    assert _fixed_hardware_mismatches(primary, holdout) == {
        "matrix_sram_tiles": (5, 6)
    }


def test_flat_row_prefers_fixed_batch_ttft_for_report() -> None:
    system = {
        "system_split": "P2:D6",
        "decode_topology": "decode",
        "projected_static_batch_request_ttft_s": 70.0,
        "e2e_latency_s": 300.0,
        "projected_pipeline_output_tokens_per_s": 200.0,
        "projected_output_tokens_per_j": 0.1,
        "energy_per_output_token_j": 10.0,
        "energy_per_request_j": 80_000.0,
        "bottleneck_stage": "decode",
        "prefill_interval_s": 60.0,
        "kv_handoff_interval_s": 0.1,
        "decode_interval_s": 250.0,
        "cross_stage_idle_energy_j": 1.0,
    }
    baseline = {
        "topology": "a100",
        "scheduler_admitted_ttft_exact_or_proxy_s": 21.0,
        "scheduler_admitted_ttft_source": "inferred_serial_staircase",
        "scheduler_admitted_ttft_fidelity": "proxy_from_serial_staircase",
        "fixed_batch_ttft_point_id": "audit",
        "median_request_visible_ttft_s": 59.0,
        "p95_request_visible_ttft_s": 80.0,
        "latency_s": 250.0,
        "output_tokens_per_s": 180.0,
        "output_tokens_per_j": 0.05,
        "energy_per_output_token_j": 20.0,
        "energy_per_request_j": 160_000.0,
        "fidelity": "measured",
    }

    row = _flat_row(
        model="qwen3-32b",
        workload_name="primary-90000x8000",
        objective="maximum_output_tps",
        system=system,
        baseline=baseline,
    )

    assert row["a100_scheduler_admitted_ttft_s"] == 21.0
    assert row["a100_request_visible_median_ttft_s"] == 59.0
    assert row["a100_ttft_s"] == 21.0
    assert row["a100_ttft_source"] == "one_token_fixed_batch_measurement"


def test_flat_row_falls_back_to_topology_matched_measured_ttft() -> None:
    system = {
        "system_split": "P2:D6",
        "decode_topology": "decode",
        "projected_static_batch_request_ttft_s": 1.2,
        "e2e_latency_s": 3.5,
        "projected_pipeline_output_tokens_per_s": 590.0,
        "projected_output_tokens_per_j": 0.3,
        "energy_per_output_token_j": 3.0,
        "energy_per_request_j": 600.0,
        "bottleneck_stage": "decode",
        "prefill_interval_s": 0.8,
        "kv_handoff_interval_s": 0.001,
        "decode_interval_s": 2.7,
        "cross_stage_idle_energy_j": 1.0,
    }
    baseline = {
        "topology": "a100",
        "scheduler_admitted_ttft_exact_or_proxy_s": None,
        "scheduler_admitted_ttft_source": "unavailable",
        "scheduler_admitted_ttft_fidelity": "unavailable",
        "fixed_batch_ttft_point_id": None,
        "median_request_visible_ttft_s": 0.576,
        "p95_request_visible_ttft_s": 0.661,
        "latency_s": 2.3,
        "output_tokens_per_s": 684.0,
        "output_tokens_per_j": 0.29,
        "energy_per_output_token_j": 3.4,
        "energy_per_request_j": 680.0,
        "fidelity": "measured",
    }

    row = _flat_row(
        model="qwen3-32b",
        workload_name="short-1400x200",
        objective="maximum_output_tps",
        system=system,
        baseline=baseline,
    )

    assert row["a100_ttft_s"] == 0.576
    assert row["a100_ttft_source"] == "topology_matched_measured_median"
