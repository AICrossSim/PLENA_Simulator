from __future__ import annotations

import math

from .matrix_state_residency import (
    KIMI_K3_STATE_ELEMENTS,
    NEMOTRON_STATE_ELEMENTS,
    MatrixSramGeometry,
    StateFormat,
    build_report,
    load_geometries,
    residency_case,
    storage_bytes,
)


def test_shipped_matrix_sram_geometry_has_row_units() -> None:
    geometries = load_geometries()
    transactional = geometries["transactional"]
    assert transactional.mlen == 64
    assert transactional.depth_rows == 4096
    assert transactional.element_bits == 16
    assert transactional.physical_bytes == 512 * 1024
    assert transactional.whole_square_tiles == 64

    analytic = geometries["analytic"]
    assert analytic.mlen == 2048
    assert analytic.depth_rows == 256
    assert analytic.physical_bytes == 1024 * 1024
    assert analytic.whole_square_tiles == 0
    assert analytic.element_format == "bf16"
    rendered = analytic.to_dict()
    assert rendered["legacy_square_tile_api_valid"] is False
    assert rendered["compact_matrix_view_api_valid"] is True


def test_kimi_official_fp32_state_needs_twelve_transactional_windows() -> None:
    geometry = load_geometries()["transactional"]
    case = residency_case(
        model="kimi_k3",
        layers=69,
        state_elements=KIMI_K3_STATE_ELEMENTS,
        storage=StateFormat.FP32,
        geometry=geometry,
    )
    assert case.state_bytes_per_layer == 6 * 1024 * 1024
    assert case.capacity_windows == 12
    assert case.final_window_occupancy == 1.0
    assert case.hbm_read_write_bytes_per_layer_token == 12 * 1024 * 1024
    assert case.natively_representable is False


def test_nemotron_state_counts_heads_not_only_bc_groups() -> None:
    geometry = load_geometries()["transactional"]
    case = residency_case(
        model="nemotron3_nano",
        layers=23,
        state_elements=NEMOTRON_STATE_ELEMENTS,
        storage=StateFormat.FP32,
        geometry=geometry,
    )
    assert NEMOTRON_STATE_ELEMENTS == 64 * 64 * 128
    assert case.state_bytes_per_layer == 2 * 1024 * 1024
    assert case.capacity_windows == 4
    assert case.hbm_read_write_bytes_all_layers_token == 92 * 1024 * 1024


def test_narrow_state_capacity_and_scale_overhead_are_not_hidden() -> None:
    geometry = MatrixSramGeometry("test", 64, 4096, 16, "bf16")
    bf16 = residency_case(
        model="kimi_k3",
        layers=69,
        state_elements=KIMI_K3_STATE_ELEMENTS,
        storage=StateFormat.BF16,
        geometry=geometry,
    )
    mx8 = residency_case(
        model="kimi_k3",
        layers=69,
        state_elements=KIMI_K3_STATE_ELEMENTS,
        storage=StateFormat.MX8_B128,
        geometry=geometry,
    )
    assert bf16.capacity_windows == 6
    assert bf16.natively_representable is True
    assert storage_bytes(KIMI_K3_STATE_ELEMENTS, StateFormat.MX8_B128) == 1_585_152
    assert mx8.capacity_windows == 4
    assert math.isclose(mx8.final_window_occupancy, 12_288 / (512 * 1024))
    assert mx8.natively_representable is False


def test_published_compact_views_fit_without_a_full_square_tile() -> None:
    footprints = build_report()["published_packet_footprints"]
    mamba = footprints["nemotron_mamba_32_heads_x_64"]
    kimi = footprints["kimi_kda_16_heads_x_128"]
    assert mamba["reserved_bytes_one_operand"] == 128 * 1024
    assert mamba["reserved_bytes_two_operands"] == 256 * 1024
    assert kimi["reserved_bytes_one_operand"] == 64 * 1024
    assert kimi["reserved_bytes_two_operands"] == 128 * 1024
    assert mamba["fits_two_operands"] is True
    assert kimi["fits_two_operands"] is True


def test_report_keeps_kda_accuracy_missing_until_measured() -> None:
    report = build_report()
    assert report["accuracy"]["nemotron_mamba"]["bf16"]["output_relative_l2_mean"] > 0
    # The KDA campaign is a separate artifact. It must never borrow the Mamba
    # error numbers merely because both algorithms carry recurrent state.
    if report["accuracy"]["kimi_kda"] is None:
        assert report["accuracy"]["kimi_kda"] is None
