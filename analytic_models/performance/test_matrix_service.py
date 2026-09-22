from dataclasses import replace

import pytest

from .matrix_service import MatrixService, matrix_cost


def test_tail_shape_does_not_receive_unattainable_fractional_tile_throughput():
    small = matrix_cost(1, 1, 1)
    full = matrix_cost(4, 4, 1024)
    assert small["total"] == full["total"]
    assert small["logical_macs"] == 1
    assert small["issued_macs"] == full["issued_macs"] == 16384


def test_feedback_and_write_acceptance_are_part_of_completion():
    fast = MatrixService(mac_latency=1)
    slow = replace(fast, mac_latency=4)
    assert matrix_cost(4, 4, 1024, slow)["arithmetic"] > matrix_cost(4, 4, 1024, fast)["arithmetic"]
    a = matrix_cost(5, 5, 1025, slow)
    b = matrix_cost(5, 5, 1025, slow, write_stall=7)
    assert b["total"] - a["total"] == 4 * 7
    assert b["arithmetic"] == a["arithmetic"]


def test_cross_array_reduction_is_not_free_area_or_storage():
    h = MatrixService()
    r = h.resources()
    assert r["multipliers"] == 4096
    assert r["cross_array_adders"] == 4080
    assert r["pe_accumulator_bytes"] == 8192
    assert replace(h, accumulator="FP32").resources()["pe_accumulator_bytes"] == 16384


def test_finite_operand_capacity_and_invalid_geometries_fail_closed():
    with pytest.raises(ValueError, match="Matrix SRAM"):
        MatrixService(matrix_capacity_bytes=1024)
    with pytest.raises(ValueError, match="power-of-two"):
        MatrixService(reduction_lanes=12)
    with pytest.raises(ValueError):
        matrix_cost(0, 4, 1024)
