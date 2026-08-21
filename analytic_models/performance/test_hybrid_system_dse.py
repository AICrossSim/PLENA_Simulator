from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from .hybrid_system_dse import (
    build_report,
    formal_precision,
    persistent_state_bytes,
    precision_error_sweep,
)
from .hybrid_system_timeline import ModelFamily


LONG_PRECISION = Path(__file__).parent / "profiles" / "hybrid_state_precision_long_v1.json"
PINNED_DSE = Path(__file__).parent / "profiles" / "hybrid_system_dse_quick_v1.json"


@pytest.fixture(scope="module")
def report():
    return build_report(
        context_length=16,
        decode_tokens=2,
        prefill_tokens=16,
    )


def test_formal_state_capacities_match_real_dtypes() -> None:
    assert persistent_state_bytes(ModelFamily.NEMOTRON3, formal_precision(ModelFamily.NEMOTRON3)) == 50_495_488
    assert persistent_state_bytes(ModelFamily.KIMI_K3, formal_precision(ModelFamily.KIMI_K3)) == 454_459_392


def test_report_contains_full_prefill_decode_dse_and_ablation(report) -> None:
    assert report["status"] == "pre_rtl_dse_not_rtl_calibrated"
    for model in ModelFamily:
        section = report["models"][model.value]
        assert section["full_model"]["prefill"]["ttft_us"] > 0
        assert section["full_model"]["decode"]["tpot_us"] > 0
        assert section["dse"]["candidate_count"] == 11
        assert section["dse"]["pareto_designs"]
        assert section["ablation"]["code_generation_ablation"]["timing_delta"] is None
        assert section["ablation"]["cycle_ablations"][0]["ablation"] == "all_features"
    recommendation = report["shared_device_recommendation"]
    assert recommendation["one_device_for_both_models"] is True
    assert recommendation["parameters"]["state_cache_mib"] == 32
    assert recommendation["parameters"]["projection_fifo_values"] == 64


def test_layout_and_state_cache_ablations_are_physical(report) -> None:
    for model in ModelFamily:
        rows = {row["ablation"]: row for row in report["models"][model.value]["ablation"]["cycle_ablations"]}
        full = rows["all_features"]
        row_major = rows["without_l_compute_layout"]
        no_cache = rows["without_state_cache"]
        no_fusion = rows["without_fused_layer_dataflow"]
        assert full["bank_stall_cycles"] == 0
        assert row_major["bank_stall_cycles"] > 0
        assert no_cache["logical_hbm_read_bytes"] > full["logical_hbm_read_bytes"]
        assert no_fusion["logical_hbm_read_bytes"] > full["logical_hbm_read_bytes"]


def test_mixed_precision_keeps_model_specific_weight_formats(report) -> None:
    nemotron = report["models"]["nemotron3"]["mixed_precision"]["records"][0]
    kimi = report["models"]["kimi_k3"]["mixed_precision"]["records"][0]
    assert nemotron["precision"]["weight"] == "nvfp4"
    assert nemotron["precision"]["conv_state"] == "fp32"
    assert kimi["precision"]["weight"] == "mxfp4"
    assert kimi["precision"]["activation"] == "mxfp8"
    assert kimi["precision"]["conv_state"] == "bf16"


def test_data_coverage_does_not_present_synthetic_kimi_routing_as_measured(report) -> None:
    assert "missing" in report["data_coverage"]["kimi_k3_full_routing"]
    source = report["models"]["kimi_k3"]["full_model"]["decode"]["moe_weight_cache"]["routing_source"]
    assert source.endswith("not_empirical")


def test_state_precision_error_sweep_runs_both_recurrences() -> None:
    report = precision_error_sweep((4,))
    assert report["nemotron3_mamba2"][0]["experiment"]["tokens"] == 4
    assert report["kimi_k3_kda"][0]["experiment"]["tokens"] == 4
    for model in ("nemotron3_mamba2", "kimi_k3_kda"):
        assert {row["storage"] for row in report[model][0]["results"]} == {
            "bf16",
            "fp16",
            "mx8_b128",
        }


def test_pinned_long_sequence_precision_evidence_is_complete_and_finite() -> None:
    assert (
        hashlib.sha256(LONG_PRECISION.read_bytes()).hexdigest()
        == "82a5faca6edb974fd108b69497668996e059904d1f25b29b16e3130194eeb0e6"
    )
    report = json.loads(LONG_PRECISION.read_text())
    assert report["token_counts"] == [2048, 8192, 32768]
    for model in ("nemotron3_mamba2", "kimi_k3_kda"):
        assert [run["experiment"]["tokens"] for run in report[model]] == report["token_counts"]
        for run in report[model]:
            for result in run["results"]:
                assert all(
                    math.isfinite(value)
                    for metric in ("state_error", "output_error")
                    for value in result[metric].values()
                )


def test_pinned_default_dse_is_reproducible_and_uses_one_device() -> None:
    assert (
        hashlib.sha256(PINNED_DSE.read_bytes()).hexdigest()
        == "41d6b57fe05b11cbfe34a53b36dc29843d6f3802f1fb9c0e9debd24fab784b77"
    )
    report = json.loads(PINNED_DSE.read_text())
    assert report["scenario"] == {
        "batch_size": 1,
        "context_length": 2048,
        "decode_tokens": 4,
        "prefill_tokens": 128,
    }
    recommendation = report["shared_device_recommendation"]
    assert recommendation["one_device_for_both_models"] is True
    assert recommendation["parameters"]["state_cache_mib"] == 32
