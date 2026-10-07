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


def test_isa_projection_requires_a_service_and_prices_final_write_ports():
    from .ltile_cost import assembly_cost

    with pytest.raises(ValueError, match="explicit bounded"):
        assembly_cost("M_MV 0, gp0, gp0, 0\n")
    h = MatrixService(vector_read_elements=128, vector_write_elements=256)
    cost = assembly_cost("M_MV_WO gp0, 0\n", matrix_service=h)
    assert cost.sram == 16 + 8
    assert cost.total == 1 + 16 + 8


def test_norm_reduction_contract_cannot_silently_inherit_fp32_or_write_f0():
    from .ltile_cost import Machine, assembly_cost

    with pytest.raises(ValueError, match="reduction contract"):
        assembly_cost("V_RED_SUM f1, gp0, 0\n")
    h = Machine(reduction_tree_bf16=True)
    with pytest.raises(ValueError, match="reduction contract"):
        assembly_cost("V_RED_SUM f0, gp0, 0\n", h)
    cost = assembly_cost("V_RED_SUM f1, gp0, 0\n", h)
    assert (cost.issue, cost.sram, cost.arithmetic) == (1, 1, 12)


def test_nvfp4_fixture_decoder_covers_signed_codes_and_block_global_scales():
    import numpy as np
    from transactional_emulator.testbench.models.recurrent_layer_test import decode_nvfp4

    codes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
    packed = np.array([codes + codes], dtype=np.uint8)
    actual = decode_nvfp4(packed, np.array([[0.5, 2]], dtype=np.float32), 3)
    values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    expected = np.concatenate([values * 1.5, values * 6])[:, None]
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    with pytest.raises(ValueError, match="block16"):
        decode_nvfp4(packed, np.ones((1, 1)), 3)
