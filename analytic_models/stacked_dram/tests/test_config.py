"""Configuration semantics, checked with conspicuously fictional values."""

from __future__ import annotations

from dataclasses import replace

import pytest
from stacked_dram_fixtures import fictional_dram, fictional_thermal, fictional_timing

from analytic_models.stacked_dram import BufferingPolicy, DramEnergyConfig, dram_connectivity_efficiency


def test_peak_bandwidth_composes_connected_layers_channels_and_clock() -> None:
    dram = fictional_dram()
    assert dram.peak_bandwidth_bytes_per_s() == pytest.approx(3 * 7 * 5.5 * 1.25 * 345_678_901.0)
    assert dram.peak_bandwidth_bytes_per_s(connected_layers=6) == pytest.approx(6 * 7 * 5.5 * 1.25 * 345_678_901.0)


def test_direct_peak_bandwidth_is_an_explicit_alternative() -> None:
    dram = fictional_dram(
        bytes_per_channel_transfer=None,
        transfers_per_memory_clock=None,
        memory_frequency_hz=None,
        direct_peak_bandwidth_per_connected_layer_bytes_per_s=246_813_579.0,
    )
    assert dram.peak_bandwidth_bytes_per_s(connected_layers=6) == pytest.approx(6 * 246_813_579.0)


def test_capacity_and_wave_follow_layers_and_channels() -> None:
    dram = fictional_dram()
    assert dram.capacity_bytes == 10 * 123_456_789
    assert dram.wave_bytes == 96 * 7


@pytest.mark.parametrize(
    ("total_layers", "connected_layers", "expected"),
    [(10, 1, 1.0), (10, 5, 1.0), (10, 7, 0.748), (10, 10, 0.37)],
)
def test_connectivity_efficiency_interpolates_to_the_fully_connected_endpoint(
    total_layers: int, connected_layers: int, expected: float
) -> None:
    efficiency = dram_connectivity_efficiency(total_layers, connected_layers, fully_connected_efficiency=0.37)
    assert efficiency == pytest.approx(expected)


def test_bank_timing_derives_row_cycles_efficiency_and_latency() -> None:
    timing = fictional_timing()
    assert timing.sectors_per_row == 55
    assert timing.row_read_cycles == 165
    assert timing.full_row_cycles == 206
    assert timing.fully_connected_efficiency == pytest.approx(165 / 206)
    assert timing.round_trip_latency_seconds == pytest.approx(137.5 / 654_321_987.0)

    dram = fictional_dram(fully_connected_efficiency=None, bank_timing=timing)
    assert dram.resolved_fully_connected_efficiency == pytest.approx(165 / 206)


@pytest.mark.parametrize(
    "changes",
    [
        {"direct_peak_bandwidth_per_connected_layer_bytes_per_s": 246_813_579.0},
        {"bytes_per_channel_transfer": None},
        {"memory_frequency_hz": None},
        {"bank_timing": fictional_timing()},
        {"fully_connected_efficiency": None},
    ],
)
def test_exactly_one_bandwidth_form_and_one_efficiency_form(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        fictional_dram(**changes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_layers", 0),
        ("total_layers", True),
        ("connected_layers", 11),
        ("channels_per_connected_layer", 2.5),
        ("bytes_per_channel_transfer", 0.0),
        ("transfers_per_memory_clock", float("nan")),
        ("memory_frequency_hz", float("inf")),
        ("capacity_per_layer_bytes", -1),
        ("transaction_bytes", 0),
        ("fully_connected_efficiency", 1.01),
        ("fully_connected_efficiency", 0.0),
        ("apply_wave_quantization", 1),
    ],
)
def test_dram_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fictional_dram(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_bytes", 0),
        ("sector_bytes", 64),
        ("sector_cycles", 0),
        ("recharge_cycles", -1),
        ("round_trip_latency_cycles", float("nan")),
        ("latency_clock_hz", 0.0),
    ],
)
def test_timing_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        fictional_timing(**{field: value})


def test_with_layers_defaults_to_a_fully_connected_stack() -> None:
    dram = fictional_dram()
    assert (dram.with_layers(12).total_layers, dram.with_layers(12).connected_layers) == (12, 12)
    taller = dram.with_layers(12, 11)
    assert (taller.total_layers, taller.connected_layers) == (12, 11)
    with pytest.raises(ValueError):
        dram.with_layers(4, 5)


def test_buffering_latency_comes_from_the_policy_or_the_bank_timing() -> None:
    own = BufferingPolicy(
        requesters=2,
        buffer_bytes_per_requester=4_096,
        buffering_factor=1.75,
        round_trip_latency_cycles=17.5,
        round_trip_latency_clock_hz=0.25e9,
    )
    assert own.round_trip_latency_seconds(None) == pytest.approx(17.5 / 0.25e9)

    inherited = BufferingPolicy(requesters=2, buffer_bytes_per_requester=4_096, buffering_factor=1.75)
    timing = fictional_timing()
    assert inherited.round_trip_latency_seconds(timing) == pytest.approx(timing.round_trip_latency_seconds)
    with pytest.raises(ValueError):
        inherited.round_trip_latency_seconds(None)
    with pytest.raises(ValueError):
        BufferingPolicy(
            requesters=2, buffer_bytes_per_requester=4_096, buffering_factor=1.75, round_trip_latency_cycles=17.5
        )


def test_thermal_scale_follows_the_stack_resistance() -> None:
    thermal = fictional_thermal()
    assert thermal.frequency_scale(3) == pytest.approx(1.0)

    resistance = 0.42 + 0.007 * 10
    baseline = 0.42 + 0.007 * 3
    dynamic = 73.0 * baseline / resistance - 8.0
    assert thermal.frequency_scale(10) == pytest.approx((dynamic / (73.0 - 8.0)) ** (1 / 2.5))
    assert thermal.frequency_scale(1) > 1.0


def test_thermal_policy_without_dynamic_budget_is_an_error() -> None:
    thermal = fictional_thermal(resistance_per_layer_c_per_w=10.0)
    with pytest.raises(ValueError, match="no dynamic-power budget"):
        thermal.frequency_scale(40)
    with pytest.raises(ValueError):
        fictional_thermal(static_power_w=73.0)


def test_energy_counts_bits_in_each_direction() -> None:
    energy = DramEnergyConfig(read_pj_per_bit=2.47, write_pj_per_bit=2.71)
    assert energy.energy_pj(100, 10) == pytest.approx(8 * (100 * 2.47 + 10 * 2.71))
    with pytest.raises(ValueError):
        replace(energy, read_pj_per_bit=-1.0)
