from __future__ import annotations

import torch

from transactional_emulator.testbench.emulator_runner import (
    _comparison_summary,
    _parse_lstream_packet_counters,
    _parse_matrix_view_packet_counters,
)


def test_parse_lstream_packet_counters_from_prefixed_rust_log() -> None:
    line = (
        "2026-09-01 INFO runner: L-stream packet counters "
        "packet_reads=192 packet_writes=128 packet_bank_words=4160 "
        "packet_service_cycles=320 packet_bandwidth_floor_cycles=320 "
        "packet_conflict_stall_cycles=0 packet_lane_restore_values=12288"
    )

    assert _parse_lstream_packet_counters(line) == {
        "packet_reads": 192,
        "packet_writes": 128,
        "packet_bank_words": 4160,
        "packet_service_cycles": 320,
        "packet_bandwidth_floor_cycles": 320,
        "packet_conflict_stall_cycles": 0,
        "packet_lane_restore_values": 12288,
    }


def test_parse_lstream_packet_counters_rejects_incomplete_line() -> None:
    assert _parse_lstream_packet_counters("packet_reads=1 packet_writes=1") is None


def test_parse_matrix_view_packet_counters_from_prefixed_rust_log() -> None:
    line = (
        "2026-09-01 INFO runner: Matrix-view packet counters "
        "packets=17 values=128 bank_words=32 service_cycles=17 "
        "ideal_cycles=17 bank_stall_cycles=0"
    )

    assert _parse_matrix_view_packet_counters(line) == {
        "packets": 17,
        "values": 128,
        "bank_words": 32,
        "service_cycles": 17,
        "ideal_cycles": 17,
        "bank_stall_cycles": 0,
    }


def test_parse_matrix_view_packet_counters_ignores_ansi_colour_codes() -> None:
    line = (
        "\x1b[2m2026-09-01T00:00:00Z\x1b[0m \x1b[32mINFO\x1b[0m "
        "Matrix-view packet counters packets=17 values=128 bank_words=32 "
        "service_cycles=17 ideal_cycles=17 bank_stall_cycles=0"
    )

    assert _parse_matrix_view_packet_counters(line) == {
        "packets": 17,
        "values": 128,
        "bank_words": 32,
        "service_cycles": 17,
        "ideal_cycles": 17,
        "bank_stall_cycles": 0,
    }


def test_parse_matrix_view_packet_counters_rejects_incomplete_line() -> None:
    assert _parse_matrix_view_packet_counters("packets=1 values=64") is None


def test_comparison_summary_excludes_full_tensors() -> None:
    results = {
        "mse": 1.0,
        "max_error": 2.0,
        "allclose_match_rate": 100.0,
        "allclose_pass": True,
        "golden_shape": (4, 8),
        "simulated_shape": (32,),
        "golden_values": torch.ones(32),
        "simulated_values": torch.ones(32),
        "errors": torch.zeros(32),
    }
    params = {"atol": 0.01, "rtol": 0.02}

    summary = _comparison_summary(results, params)

    assert summary == {
        "mse": 1.0,
        "max_error": 2.0,
        "allclose_match_rate": 100.0,
        "allclose_pass": True,
        "golden_shape": [4, 8],
        "simulated_shape": [32],
        "comparison_params": params,
    }
    assert "golden_values" not in summary
    assert "errors" not in summary
