from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .b200_formal_campaign import (
    PINNED_SUMMARY,
    B200CampaignFormatError,
    build_report,
    crosscheck_local_kda_stage2,
)
from .b200_campaign_raw import build_normalized_profile


def _optional_profile_root(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else None


LOCAL_KDA_STAGE2 = _optional_profile_root("PLENA_KDA_STAGE2_ROOT")
RAW_CAMPAIGN = _optional_profile_root("PLENA_B200_CAMPAIGN_ROOT")
PINNED_ROUTING = PINNED_SUMMARY.with_name("nemotron3_decode_routing_trace.json")


def test_pinned_campaign_proves_matrix_dominance_and_routing_skew() -> None:
    report = build_report()
    cases = {case["case"]: case for case in report["kda"]["cases"]}

    assert cases["prefill_b1_s2048"]["matrix_path_time_fraction"] == pytest.approx(
        (1.27754 + 0.47699 + 0.48832) / 3.01731
    )
    assert cases["decode_b1"]["matrix_path_time_fraction"] == pytest.approx(
        (0.15228 + 0.06119 + 0.05453) / 0.35995
    )
    assert cases["decode_b8"]["matrix_path_time_fraction"] == pytest.approx(
        (0.14849 + 0.05705 + 0.05085) / 0.41188
    )
    assert cases["decode_b1"]["state_core_time_fraction"] == pytest.approx(0.01808 / 0.35995)
    assert report["kda"]["decode_b8_to_b1_state_core_read_ratio"] == pytest.approx(
        51_194_100 / 6_463_840
    )

    nemotron = report["nemotron"]
    assert nemotron["model"].endswith("30B-A3B-NVFP4")
    assert nemotron["revision"] == "ce1b118ae66ec705d02c241525192832eb045fd3"
    assert nemotron["moe_to_mamba_prefill_dram_read_ratio"] == pytest.approx(
        13_462_514_240 / 1_509_389_580
    )
    assert nemotron["moe_to_mamba_decode_dram_read_ratio"] == pytest.approx(
        1_114_920_410 / 873_460_520
    )
    assert nemotron["ncu"]["decode_step_s2048"]["layer_types"]["moe"][
        "duration_fraction"
    ] == pytest.approx(0.6036138119708677)
    assert nemotron["routing"]["decode_max_hotspot_count"] == 2139
    assert nemotron["routing"]["decode_max_hotspot_to_mean"] == pytest.approx(2139 / 101.953125)
    assert nemotron["routing"]["one_hottest_expert_per_layer_assignment_coverage"] == pytest.approx(
        32785 / 300150
    )
    assert nemotron["routing"]["decode_generation"]["recurrent_decode_steps"] == 127
    assert report["campaign_status"] == "complete"
    assert report["evidence_boundaries"]["workload_calibrated"] is True
    assert report["evidence_boundaries"]["plena_cycle_calibrated"] is False
    assert "not implemented" in report["precision_scope"]["transactional_emulator"]
    assert report["precision_scope"]["compiler_binary"].endswith("not NVFP4")


def test_campaign_rejects_stage_totals_that_do_not_match(tmp_path: Path) -> None:
    document = json.loads(PINNED_SUMMARY.read_text())
    document["kda"]["cases"]["decode_b1"]["stages"]["qkv_projection"]["time_ms"] += 0.1
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(document))
    with pytest.raises(B200CampaignFormatError, match="decode_b1 time"):
        build_report(source)


def test_campaign_rejects_internally_inconsistent_routing(tmp_path: Path) -> None:
    document = json.loads(PINNED_SUMMARY.read_text())
    document["nemotron"]["routing"]["cases"]["prefill_s128"]["routed_assignments"] -= 1
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(document))
    with pytest.raises(B200CampaignFormatError, match="routed assignments"):
        build_report(source)


@pytest.mark.skipif(
    LOCAL_KDA_STAGE2 is None or not LOCAL_KDA_STAGE2.exists(),
    reason="set PLENA_KDA_STAGE2_ROOT to cross-check the optional raw archive",
)
def test_local_kda_stage2_independently_matches_the_formal_core_traffic() -> None:
    assert LOCAL_KDA_STAGE2 is not None
    report = crosscheck_local_kda_stage2(LOCAL_KDA_STAGE2)

    assert report["status"] == "local_gpu2_kda_core_subset_matches_formal_gpu3_summary"
    assert [case["kernel_calls"] for case in report["cases"]] == [3, 2, 2]
    assert max(case["absolute_delta_mib"] for case in report["cases"]) < 0.01


@pytest.mark.skipif(
    RAW_CAMPAIGN is None or not RAW_CAMPAIGN.exists(),
    reason="set PLENA_B200_CAMPAIGN_ROOT to rebuild from the optional raw archive",
)
def test_raw_campaign_rebuilds_the_pinned_contract_and_routing_trace() -> None:
    assert RAW_CAMPAIGN is not None
    profile, routing = build_normalized_profile(RAW_CAMPAIGN)
    pinned = json.loads(PINNED_SUMMARY.read_text())
    # Archive metadata is supplied only when ingesting the transferred tarball.
    profile["source"]["archive"] = pinned["source"]["archive"]

    assert profile == pinned
    assert routing == json.loads(PINNED_ROUTING.read_text())
