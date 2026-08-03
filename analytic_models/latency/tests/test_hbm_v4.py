from __future__ import annotations

from compiler.aten.isa_builder import DmaTransfer, RepeatAxis
from compiler.aten.program_sink import CostTrace, TraceDma

from analytic_models.latency.hbm_v4 import (
    DEFAULT_HBM_V4_CALIBRATION,
    HbmPrecisionConfig,
    HbmServiceModelV4,
    HbmV4Config,
    HbmV4MemoryProvider,
    MemoryFormat,
    plan_dma_request_manifest,
    request_manifest_fixture_hash,
)


def _model() -> HbmServiceModelV4:
    zeros = {
        name: 0.0
        for name in (
            "read_phase_startup",
            "write_phase_startup",
            "read_write_turnaround",
            "read_channel_tail",
            "write_channel_tail",
            "read_bankgroup_serial",
            "write_bankgroup_serial",
            "read_bank_serial",
            "write_bank_serial",
            "read_row_miss",
            "write_row_miss",
            "read_row_conflict",
            "write_row_conflict",
            "sram_dma_drain",
        )
    }
    return HbmServiceModelV4(
        calibration_id="unit-test",
        coefficients={f"{opcode}:c8": zeros for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")},
        domains={},
        warm_coefficients={},
        compatibility={},
        metadata={},
    )


def _precision(element_bits: int = 8) -> HbmPrecisionConfig:
    mx = MemoryFormat("mxfp", element_bits, 8, 8, f"MX{element_bits}")
    return HbmPrecisionConfig(
        weight=mx,
        matrix_kv=mx,
        activation=mx,
        vector_kv=mx,
        integer=MemoryFormat("plain", 32, name="INT32"),
    )


def test_request_planner_matches_immutable_fixture() -> None:
    assert request_manifest_fixture_hash() == "ccc23894a6bbaa5edbec4d5fffa77b5f41304688bab6bde3e03132cef26a8d0a"


def test_checked_in_calibration_matches_main_dma_semantics() -> None:
    model = HbmServiceModelV4.load(DEFAULT_HBM_V4_CALIBRATION)
    assert model.calibration_id == "hbm-production-dma-v4-1b00531de9a61298"
    assert model.metadata["training_samples"] == 2043
    assert model.metadata["holdout_samples"] == 585


def test_partial_store_generates_read_modify_write() -> None:
    transfer = DmaTransfer(
        opcode="H_STORE_V",
        direction="write",
        role="activation",
        element_base_bytes=32,
        scale_base_bytes=(1 << 20) + 32,
        dim=128,
        amount=1,
        stride_bytes=64,
        rstride=1,
        write_amount=1,
    )
    manifest = plan_dma_request_manifest(transfer, MemoryFormat("mxint", 4, 8, 64))
    assert manifest.write_lines
    assert manifest.read_lines
    assert set(manifest.read_lines).issubset(manifest.write_lines)
    assert manifest.partial_lines == len(manifest.read_lines)


def test_scalar_provider_preserves_stage_and_affine_occurrence_accounting() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        role="activation",
        element_base_bytes=0,
        scale_base_bytes=1 << 20,
        dim=64,
        amount=2,
        stride_bytes=64,
        rstride=1,
        element_bytes=1,
    )
    event = TraceDma(
        stage="decoder/layer/attention",
        transfer=transfer,
        multiplicity=3,
        repeat_axes=(
            RepeatAxis.from_mapping(
                "row_group",
                3,
                {"element_base_bytes": 128, "scale_base_bytes": 16},
            ),
        ),
    )
    trace = CostTrace(
        schema_version="test",
        isa_hash="isa",
        compiler_hash="compiler",
        instructions=(),
        dma_events=(event,),
        metadata={},
    )
    report = HbmV4MemoryProvider(
        _model(), _precision(), HbmV4Config(8), aggregation="scalar"
    ).estimate(trace)
    assert report.total_picos == report.by_stage_picos["decoder/layer/attention"]
    assert report.by_opcode_picos["H_PREFETCH_V"] == report.total_picos
    assert report.provenance["occurrence_count"] == 3
    assert report.physical_read_bytes == report.read_requests * 64
    assert report.physical_write_bytes == 0
    assert report.payload_read_bytes < report.physical_read_bytes


def test_sufficient_statistics_matches_scalar_for_affine_and_prefix_dma() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        role="weight",
        element_base_bytes=0,
        scale_base_bytes=1 << 20,
        dim=64,
        amount=1,
        stride_bytes=64,
        rstride=1,
        precision="weight",
        element_bytes=1,
        memory_object="K",
    )
    layer = RepeatAxis.from_mapping("decoder_layer", 2, {})
    events = tuple(
        TraceDma(
            stage="decoder/layer/attention",
            transfer=transfer,
            multiplicity=2 * visible * 3,
            repeat_axes=(
                layer,
                RepeatAxis.from_mapping(
                    "visible_k_block",
                    visible,
                    {"element_base_bytes": 64, "scale_base_bytes": 8},
                ),
            ),
        )
        for visible in (1, 2, 3)
    )
    trace = CostTrace("test", "isa", "compiler", (), events, {})
    scalar = HbmV4MemoryProvider(
        _model(), _precision(), HbmV4Config(8), aggregation="scalar"
    ).estimate(trace)
    grouped = HbmV4MemoryProvider(
        _model(), _precision(), HbmV4Config(8), aggregation="sufficient-statistics"
    ).estimate(trace)
    assert grouped.total_picos == scalar.total_picos
    assert grouped.by_stage_picos == scalar.by_stage_picos
    assert grouped.by_opcode_picos == scalar.by_opcode_picos
    assert grouped.physical_read_bytes == scalar.physical_read_bytes
    assert grouped.payload_read_bytes == scalar.payload_read_bytes
    assert grouped.read_requests == scalar.read_requests
    assert grouped.traffic_breakdown == scalar.traffic_breakdown
    assert grouped.provenance["logical_occurrence_count"] == 36
    assert grouped.provenance["prefix_stream_count_folded"] == 3
    assert grouped.provenance["occurrences_elided"] > 0


def test_main_address_layout_fails_closed_for_repacked_precision() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        role="weight",
        element_base_bytes=0,
        scale_base_bytes=1 << 20,
        dim=64,
        amount=1,
        stride_bytes=64,
        rstride=1,
        precision="weight",
        element_bytes=1,
    )
    try:
        _precision(4).for_transfer(transfer)
    except ValueError as error:
        assert "main lowering must describe repacked addresses" in str(error)
    else:
        raise AssertionError("4-bit format must not reuse 8-bit compiler addresses")
