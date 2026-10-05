"""Derived bandwidth, buffering and quantisation of ``StackedDramModel``."""

from __future__ import annotations

import json

import pytest
from stacked_dram_fixtures import fictional_dram, fictional_thermal, fictional_timing

from analytic_models.stacked_dram import (
    BufferingPolicy,
    DramEnergyConfig,
    FixedBandwidthMemory,
    StackedDramModel,
)


def test_effective_bandwidth_applies_connectivity_efficiency() -> None:
    model = StackedDramModel(name="fictional", config=fictional_dram(connected_layers=7))
    assert model.connectivity_efficiency == pytest.approx(0.748)
    assert model.effective_bandwidth_bytes_per_s == pytest.approx(model.peak_bandwidth_bytes_per_s * 0.748)
    assert model.usable_bandwidth_bytes_per_s == model.effective_bandwidth_bytes_per_s
    assert model.littles_law() is None


def test_littles_law_leaves_a_large_buffer_unlimited() -> None:
    timing = fictional_timing()
    dram = fictional_dram(fully_connected_efficiency=None, bank_timing=timing)
    policy = BufferingPolicy(requesters=3, buffer_bytes_per_requester=10**9, buffering_factor=1.75)
    model = StackedDramModel(name="fictional", config=dram, buffering=policy)

    bound = model.littles_law()
    assert bound is not None and not bound.limited
    required = model.effective_bandwidth_bytes_per_s / 3 * timing.round_trip_latency_seconds * 1.75
    assert bound.required_buffer_bytes_per_requester == pytest.approx(required)
    assert model.usable_bandwidth_bytes_per_s == model.effective_bandwidth_bytes_per_s


def test_littles_law_scales_bandwidth_by_the_buffer_shortfall() -> None:
    timing = fictional_timing()
    dram = fictional_dram(fully_connected_efficiency=None, bank_timing=timing)
    policy = BufferingPolicy(requesters=3, buffer_bytes_per_requester=1_000, buffering_factor=1.75)
    model = StackedDramModel(name="fictional", config=dram, buffering=policy)

    bound = model.littles_law()
    assert bound is not None and bound.limited
    expected = model.effective_bandwidth_bytes_per_s * 1_000 / bound.required_buffer_bytes_per_requester
    assert model.usable_bandwidth_bytes_per_s == pytest.approx(expected)


def test_wave_quantisation_rounds_each_transfer_to_whole_waves() -> None:
    plain = StackedDramModel(name="fictional", config=fictional_dram())
    assert plain.quantize_bytes(1) == 1

    quantised = StackedDramModel(name="fictional", config=fictional_dram(apply_wave_quantization=True))
    wave = 96 * 7
    assert quantised.quantize_bytes(1) == wave
    assert quantised.quantize_bytes(wave) == wave
    assert quantised.quantize_bytes(wave + 1) == 2 * wave
    assert quantised.quantize_bytes(0) == 0


def test_thermal_policy_scales_the_compute_clock() -> None:
    model = StackedDramModel(name="fictional", config=fictional_dram(), thermal=fictional_thermal())
    assert model.compute_frequency_scale == pytest.approx(fictional_thermal().frequency_scale(10))
    assert StackedDramModel(name="fictional", config=fictional_dram()).compute_frequency_scale == 1.0


def test_with_layers_keeps_the_policies() -> None:
    model = StackedDramModel(
        name="fictional",
        config=fictional_dram(),
        thermal=fictional_thermal(),
        energy=DramEnergyConfig(read_pj_per_bit=2.47, write_pj_per_bit=2.71),
        provenance={"source": "test"},
    )
    taller = model.with_layers(12, 8)
    assert (taller.config.total_layers, taller.config.connected_layers) == (12, 8)
    assert taller.thermal == model.thermal and taller.energy == model.energy
    assert taller.compute_frequency_scale == pytest.approx(fictional_thermal().frequency_scale(12))
    assert dict(taller.provenance) == {"source": "test"}


def test_describe_is_json_serialisable() -> None:
    model = StackedDramModel(
        name="fictional",
        config=fictional_dram(),
        buffering=BufferingPolicy(
            requesters=3,
            buffer_bytes_per_requester=54_321,
            buffering_factor=1.75,
            round_trip_latency_cycles=17.5,
            round_trip_latency_clock_hz=0.25e9,
        ),
        energy=DramEnergyConfig(read_pj_per_bit=2.47, write_pj_per_bit=2.71),
    )
    summary = json.loads(json.dumps(model.describe()))
    assert summary["kind"] == "stacked_dram"
    assert summary["capacity_bytes"] == 10 * 123_456_789
    assert summary["littles_law"]["limited"] is False


def test_model_rejects_misplaced_policies() -> None:
    with pytest.raises(TypeError):
        StackedDramModel(name="fictional", config=fictional_dram(), thermal=fictional_timing())
    with pytest.raises(ValueError):
        StackedDramModel(name=" ", config=fictional_dram())


def test_fixed_bandwidth_memory() -> None:
    memory = FixedBandwidthMemory(name="baseline", bandwidth_bytes_per_s=98_765_432_100.0, capacity_bytes=None)
    assert memory.usable_bandwidth_bytes_per_s == 98_765_432_100.0
    assert memory.compute_frequency_scale == 1.0
    assert memory.quantize_bytes(7) == 7
    assert memory.energy_pj(1, 1) is None

    granular = FixedBandwidthMemory(name="baseline", bandwidth_bytes_per_s=1.0, capacity_bytes=1, transaction_bytes=64)
    assert granular.quantize_bytes(65) == 128
    with pytest.raises(ValueError):
        FixedBandwidthMemory(name="baseline", bandwidth_bytes_per_s=0.0, capacity_bytes=None)
