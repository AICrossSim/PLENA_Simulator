from __future__ import annotations

from dataclasses import replace

from .state_engine import (
    BankedStateLayout,
    RecurrentStateEngineModel,
    StateEngineDesign,
    StateGeometry,
    StateSramLayout,
    StateStorage,
)
from .state_engine_model import _parser, build_document


def test_profiled_kda_head_tile_is_fp32_and_twice_mamba_bf16_size() -> None:
    assert StateGeometry.nemotron3_mamba2().bytes_per_head == 32 * 1024
    kda = StateGeometry.kimi_k3_kda()
    assert kda.bytes_per_head == 64 * 1024
    assert kda.conv_precision == StateStorage.BF16


def test_dual_axis_mapping_is_bijective_without_state_duplication() -> None:
    geometry = StateGeometry.kimi_k3_kda()
    design = StateEngineDesign(layout=StateSramLayout.DUAL_AXIS_CYCLIC)
    layout = BankedStateLayout(geometry, design)
    addresses = {
        (layout.address(row, column).bank, layout.address(row, column).offset)
        for row in range(geometry.rows)
        for column in range(geometry.columns)
    }
    assert len(addresses) == geometry.elements_per_head


def test_row_major_conflicts_but_dual_axis_serves_a_full_lane_tile() -> None:
    for geometry in (StateGeometry.nemotron3_mamba2(), StateGeometry.kimi_k3_kda()):
        row = RecurrentStateEngineModel(geometry, StateEngineDesign()).bank_stats()
        cyclic = RecurrentStateEngineModel(
            geometry,
            StateEngineDesign(layout=StateSramLayout.DUAL_AXIS_CYCLIC),
        ).bank_stats()
        assert row.stall_cycles > 0
        assert cyclic.stall_cycles == 0
        assert cyclic.service_cycles == cyclic.ideal_cycles


def test_kda_two_sram_passes_still_use_one_hbm_read_and_write() -> None:
    geometry = StateGeometry.kimi_k3_kda()
    result = RecurrentStateEngineModel(
        geometry,
        StateEngineDesign(layout=StateSramLayout.DUAL_AXIS_CYCLIC),
    ).evaluate()
    assert geometry.sram_passes == 2
    assert result.bank_stats.values == 2 * geometry.elements_per_layer
    assert result.hbm_read_bytes == geometry.persistent_bytes_per_layer
    assert result.hbm_write_bytes == geometry.persistent_bytes_per_layer
    assert geometry.conv_bytes_per_layer == 288 * 1024


def test_ping_pong_head_tiles_overlap_state_dma() -> None:
    geometry = StateGeometry.kimi_k3_kda()
    base = StateEngineDesign(layout=StateSramLayout.DUAL_AXIS_CYCLIC)
    single = RecurrentStateEngineModel(geometry, replace(base, head_tile_slots=1)).evaluate()
    ping_pong = RecurrentStateEngineModel(geometry, replace(base, head_tile_slots=2)).evaluate()
    assert ping_pong.total_cycles == max(ping_pong.compute_cycles, ping_pong.state_sram_cycles, ping_pong.hbm_cycles)
    assert single.total_cycles > ping_pong.total_cycles
    assert ping_pong.head_tile_sram_bytes == 128 * 1024


def test_resident_state_removes_hbm_traffic_not_recurrent_work() -> None:
    geometry = StateGeometry.nemotron3_mamba2()
    model = RecurrentStateEngineModel(
        geometry,
        StateEngineDesign(layout=StateSramLayout.DUAL_AXIS_CYCLIC),
    )
    streamed = model.evaluate()
    resident = model.evaluate(state_resident=True)
    assert resident.compute_cycles == streamed.compute_cycles
    assert resident.state_sram_cycles == streamed.state_sram_cycles
    assert resident.hbm_read_bytes == resident.hbm_write_bytes == 0
    assert resident.total_cycles <= streamed.total_cycles


def test_mx8_counts_scales_for_recurrent_and_conv_state() -> None:
    fp32 = StateGeometry.nemotron3_mamba2(StateStorage.FP32)
    mx8 = StateGeometry.nemotron3_mamba2(StateStorage.MX8_B128)
    assert mx8.state_scale_bytes_per_layer == 64 * 64
    assert mx8.conv_bytes_per_layer == 6144 * (4 + 1)
    assert fp32.persistent_bytes_per_layer / mx8.persistent_bytes_per_layer > 3.9


def test_common_state_dse_scans_resource_parameters() -> None:
    args = _parser().parse_args(
        [
            "--algorithm",
            "kda",
            "--layout",
            "dual_axis_cyclic",
            "--sweep-head-lanes",
            "1,2,4",
            "--sweep-banks",
            "16,32,64",
            "--sweep-head-tile-slots",
            "1,2",
        ]
    )
    document = build_document(args)
    assert len(document["results"]) == 18
    resources = {
        (
            result["metrics"]["total_fma_lanes"],
            result["metrics"]["total_state_banks"],
            result["metrics"]["head_tile_sram_bytes"],
        )
        for result in document["results"]
    }
    assert len(resources) == 18
