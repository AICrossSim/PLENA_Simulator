from __future__ import annotations

import pytest

from .layout_mode_dse import build_report


def test_dense_diagonal_balances_write_and_column_read_conflicts() -> None:
    cases = {
        case["layout"]: case
        for case in build_report()["dense_column_ablation"]["cases"]
    }
    assert cases["row_major"]["write_service_cycles"] == 128
    assert cases["row_major"]["read_service_cycles"] == 2048
    assert cases["transpose"]["write_service_cycles"] == 2048
    assert cases["transpose"]["read_service_cycles"] == 128
    assert cases["diagonal_custom"]["write_service_cycles"] == 128
    assert cases["diagonal_custom"]["read_service_cycles"] == 128
    assert all(case["hbm_repack_bytes"] == 0 for case in cases.values())


def test_real_mamba_and_kda_layouts_roundtrip_and_reach_read_ideal() -> None:
    report = build_report()
    mamba_row, mamba_skew = report["nemotron3_mamba_decode"]["cases"]
    kda_row, kda_skew = report["kimi_k3_kda_decode"]["cases"]

    assert mamba_row["roundtrip_ok"] and mamba_skew["roundtrip_ok"]
    assert mamba_row["read_service_cycles"] == 53_176
    assert mamba_skew["read_service_cycles"] == 14_904
    assert mamba_skew["read_stall_cycles"] == 0
    assert report["nemotron3_mamba_decode"]["comparison"][
        "read_service_speedup"
    ] == pytest.approx(53_176 / 14_904)

    assert kda_row["roundtrip_ok"] and kda_skew["roundtrip_ok"]
    assert kda_row["read_service_cycles"] == 536_544
    assert kda_skew["read_service_cycles"] == 430_560
    assert kda_skew["read_stall_cycles"] == 0
    assert report["kimi_k3_kda_decode"]["comparison"][
        "read_write_service_reduction_percent"
    ] == pytest.approx(14.151465)
