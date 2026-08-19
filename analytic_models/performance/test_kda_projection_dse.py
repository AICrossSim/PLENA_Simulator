from __future__ import annotations

import json
from pathlib import Path

import pytest

from .b200_formal_campaign import build_report as build_formal_campaign_report
from .kda_projection_dse import build_report


FIXTURE = Path(__file__).parents[2] / "transactional_emulator/testdata/projection_scatter_v1_kimi_k3_decode.json"


def test_k8_rotation_eliminates_kda_packet_bank_stalls() -> None:
    report = build_report(json.loads(FIXTURE.read_text()))
    assert report["candidate_count"] == 256
    assert report["baseline"]["service_cycles"] == 7776
    assert report["baseline"]["stall_cycles"] == 1536
    assert report["selected"]["k_rotation"] == 8
    assert report["selected"]["decay_rotation"] == 0
    assert report["selected"]["service_cycles"] == 6240
    assert report["selected"]["stall_cycles"] == 0
    assert report["selected"]["write_service_cycles"] == 3078
    assert report["selected"]["write_stall_cycles"] == 0
    assert report["selected"]["roundtrip"]["read_values"] == 49248
    assert report["read_service_cycle_reduction_percent"] == 100 * (7776 - 6240) / 7776
    assert report["total_service_cycle_reduction_percent"] == 100 * ((7776 + 3078) - (6240 + 3078)) / (7776 + 3078)


def test_formal_profile_bounds_system_impact_without_claiming_plena_speedup() -> None:
    campaign = build_formal_campaign_report()
    report = build_report(json.loads(FIXTURE.read_text()), formal_campaign=campaign)
    cases = {item["case"]: item for item in report["formal_b200_context"]["cases"]}
    reduction = report["total_service_cycle_reduction_percent"] / 100
    exposed = cases["decode_b1"]["b200_matrix_path_time_fraction"]

    assert cases["decode_b1"]["optimistic_speedup_ceiling"] == pytest.approx(1 / (1 - exposed * reduction))
    assert cases["decode_b1"]["optimistic_speedup_ceiling"] < 1.12
    assert "deliberately optimistic" in report["formal_b200_context"]["interpretation"]
