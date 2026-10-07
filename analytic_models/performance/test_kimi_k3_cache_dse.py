from __future__ import annotations

from .kimi_k3_cache_dse import MIB, build_report, evaluate_capacity


def test_32_mib_holds_exactly_five_real_kda_states() -> None:
    point = evaluate_capacity(32, policy="pinned")
    assert point.entry_bytes == 6_586_368
    assert point.resident_layers == 5
    assert point.hits == 10
    assert point.misses == 128


def test_layer_major_lru_thrashes_until_every_state_fits() -> None:
    partial = evaluate_capacity(256, policy="lru")
    full = evaluate_capacity(512, policy="lru")
    assert partial.resident_layers < 69
    assert partial.hits == 0
    assert full.resident_layers == 69
    assert full.hit_rate == 1.0


def test_report_uses_profiled_fp32_plus_bf16_state_capacity() -> None:
    report = build_report()
    assert report["state_contract"]["total_mib_per_request"] == 433.40625
    assert report["state_contract"]["entry_mib"] == 6.28125
    zero = next(
        point
        for point in report["points"]
        if point["capacity_mib"] == 0 and point["policy"] == "streaming"
    )
    assert zero["hbm_read_bytes"] == 2 * 69 * int(6.28125 * MIB)


def test_streaming_never_claims_resident_entries() -> None:
    point = evaluate_capacity(512, policy="streaming")
    assert point.resident_layers == 0
    assert point.hits == 0
    assert point.misses == point.accesses
