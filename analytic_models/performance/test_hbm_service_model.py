from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from analytic_models.performance.hbm_service_calibration import (
    generate_hbm_service_calibration_plan,
)
from analytic_models.performance.hbm_service_model import (
    HbmConfig,
    MemoryFormat,
    MemoryPrecisionConfig,
    PhysicalDmaStream,
    PhysicalRepeatAxis,
    build_physical_memory_work,
    clear_physical_memory_work_cache,
    mapper_statistics,
    sample_stream_requests,
    summarize_physical_stream,
)
from analytic_models.performance.compiler_cost_model import (
    TransactionalCycleModel,
    evaluate_compiler_cost,
)
from compiler.aten.cost_emitter import CostTrace, MemoryEvent
from compiler.aten.cost_frontend import compile_native_decoder_cost_trace
from compiler.aten.isa_builder import DmaTransfer
from compiler.aten.tests.test_cost_frontend import _qwen3_32b, _target_hardware


ROOT = Path(__file__).resolve().parents[2]
V3_ARTIFACT = ROOT / "analytic_models/performance/calibration/hbm_service_global_v3.json"
V3_VALIDATION = (
    ROOT / "analytic_models/performance/calibration/hbm_service_global_v3_validation.json"
)
TARGET_SETTINGS = (
    ROOT
    / "Workspace/qwen3_32b_transactional_prefetch_sweep/runs/"
    "gqa_logical_kv_optimized_20260710/trial_0000/plena_settings.toml"
)


def _stream(
    *,
    opcode: str = "H_PREFETCH_M",
    direction: str = "read",
    axes: tuple[PhysicalRepeatAxis, ...] = (),
) -> PhysicalDmaStream:
    return PhysicalDmaStream(
        stage="test",
        opcode=opcode,
        direction=direction,
        precision_role="weight" if opcode == "H_PREFETCH_M" else "activation",
        format_signature="mx:e4:s8:b64",
        element_base=0,
        scale_base=1 << 20,
        dim=128,
        amount=128,
        stride_bytes=64,
        rstride=1,
        write_amount=128 if opcode == "H_PREFETCH_M" else 1,
        axes=axes,
        multiplicity=1 if not axes else 8,
        stream_index=0,
        source="test",
    )


def test_memory_precision_support_and_mxint3_rejection() -> None:
    config = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXINT8",
            "activation": "MXFP4",
            "kv": "MXFP_E4M3",
            "block": 64,
            "integer_bits": 16,
        }
    )

    assert config.weight.request_signature() == "mx:e8:s8:b64"
    assert config.activation.request_signature() == "mx:e4:s8:b64"
    assert config.matrix_kv.request_signature() == "mx:e8:s8:b64"
    assert config.integer.request_signature() == "plain:e16"
    with pytest.raises(ValueError, match="MXINT3 is unsupported"):
        MemoryFormat.parse("MXINT3")


def test_mxint4_prefetch_and_store_request_geometry() -> None:
    fmt = MemoryFormat.parse("MXINT4", default_block=64)
    hbm = HbmConfig(128)
    prefetch = summarize_physical_stream(_stream(), fmt, hbm)
    store = summarize_physical_stream(
        _stream(opcode="H_STORE_V", direction="write"), fmt, hbm
    )

    assert prefetch.read_requests == 256
    assert prefetch.physical_read_bytes == 16_384
    assert prefetch.write_requests == 0
    assert store.read_requests == 256
    assert store.write_requests == 256
    assert store.rmw_requests == 256
    assert store.physical_read_bytes == 16_384
    assert store.physical_write_bytes == 16_384


@pytest.mark.parametrize(
    ("precision", "expected_read_requests"),
    (
        ("MXINT4", 256),
        ("MXFP_E1M2", 256),
        ("MXINT8", 384),
        ("MXFP_E4M3", 384),
    ),
)
def test_all_v3_mx_formats_pack_to_expected_requests(
    precision: str, expected_read_requests: int
) -> None:
    fmt = MemoryFormat.parse(precision, default_block=64)
    geometry = summarize_physical_stream(_stream(), fmt, HbmConfig(128))

    assert geometry.read_requests == expected_read_requests
    assert geometry.physical_read_bytes == expected_read_requests * 64


def test_unaligned_mxint4_store_accounts_for_rmw_requests() -> None:
    fmt = MemoryFormat.parse("MXINT4", default_block=64)
    stream = replace(
        _stream(opcode="H_STORE_V", direction="write"),
        element_base=32,
        scale_base=(1 << 20) + 32,
    )
    geometry = summarize_physical_stream(stream, fmt, HbmConfig(128))

    assert geometry.read_requests == 384
    assert geometry.write_requests == 384
    assert geometry.rmw_requests == 384


def test_compressed_stream_matches_expanded_requests_and_mapper_stats() -> None:
    fmt = MemoryFormat.parse("MXINT4", default_block=64)
    hbm = HbmConfig(32)
    axis = PhysicalRepeatAxis("instruction", 8, 16_384, 256)
    compressed = _stream(axes=(axis,))
    compressed_geometry = summarize_physical_stream(compressed, fmt, hbm)

    expanded_requests = []
    read_requests = write_requests = 0
    for index in range(axis.count):
        item = replace(
            compressed,
            element_base=compressed.element_base + index * axis.element_byte_delta,
            scale_base=compressed.scale_base + index * axis.scale_byte_delta,
            axes=(),
            multiplicity=1,
        )
        geometry = summarize_physical_stream(item, fmt, hbm)
        read_requests += geometry.read_requests
        write_requests += geometry.write_requests
        expanded_requests.extend(sample_stream_requests(item, fmt))

    compressed_mapper = mapper_statistics(
        sample_stream_requests(compressed, fmt, limit=2048), hbm.channels
    )
    expanded_mapper = mapper_statistics(expanded_requests, hbm.channels)
    assert compressed_geometry.read_requests == read_requests
    assert compressed_geometry.write_requests == write_requests
    assert compressed_mapper == expanded_mapper


def test_physical_work_cache_distinguishes_memory_event_subsets() -> None:
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXINT4",
            "activation": "MXINT4",
            "kv": "MXINT4",
            "block": 64,
            "integer_bits": 16,
        }
    )

    def trace_for(*, amount: int, stream_index: int) -> CostTrace:
        transfer = DmaTransfer(
            opcode="H_PREFETCH_V",
            direction="read",
            precision="activation",
            precision_role="activation",
            element_base=0,
            scale_base=1 << 20,
            dim=64,
            amount=amount,
            stride=64,
            rstride=1,
            source="cache_regression",
            memory_object="cache_regression",
            logical_object_elements=64 * amount,
            logical_element_offset=0,
            logical_scale_offset=0,
            logical_stride=64,
        )
        return CostTrace(
            memory_events=[
                MemoryEvent(
                    stage="test",
                    transfer=transfer,
                    multiplicity=1,
                    stream_index=stream_index,
                )
            ],
            metadata={"config_hash": "same-config", "num_layers": 1},
        )

    small_trace = trace_for(amount=1, stream_index=0)
    large_trace = trace_for(amount=8, stream_index=1)
    clear_physical_memory_work_cache()

    small = build_physical_memory_work(small_trace, precision, HbmConfig(128))
    large = build_physical_memory_work(large_trace, precision, HbmConfig(128))
    small_again = build_physical_memory_work(small_trace, precision, HbmConfig(128))

    assert large.read_requests > small.read_requests
    assert large.read_bytes > small.read_bytes
    assert small_again == small


def test_v3_plan_covers_active_domain_and_caps_request_work() -> None:
    plan = generate_hbm_service_calibration_plan()

    assert plan["schema_version"] == 3
    assert len(plan["patterns"]) == 1536
    assert {pattern["channels"] for pattern in plan["patterns"]} == {8, 32, 128}
    assert {pattern["transfer"]["dim"] for pattern in plan["patterns"]} == {
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    }
    assert {pattern["stream_family"] for pattern in plan["patterns"]} == {
        "single",
        "reuse",
        "affine",
        "nested",
    }
    assert {tuple(pattern["equivalent_formats"]) for pattern in plan["patterns"]} == {
        ("MXINT4", "MXFP4"),
        ("MXINT8", "MXFP8"),
        ("MXFP8_BLOCK8",),
    }
    grouped_splits = {}
    for pattern in plan["patterns"]:
        grouped_splits.setdefault(pattern["split_group"], set()).add(pattern["split"])
    assert all(len(splits) == 1 for splits in grouped_splits.values())
    assert any(pattern["run_transactional"] for pattern in plan["patterns"])


def test_v3_calibration_request_audit_and_median_threshold() -> None:
    validation = json.loads(V3_VALIDATION.read_text())
    artifact = json.loads(V3_ARTIFACT.read_text())

    assert validation["physical_request_byte_mismatches"] == 0
    assert validation["request_level_sample_count"] == 1536
    assert validation["transactional_dma_sample_count"] == 363
    assert validation["transactional_dma_max_latency_error_percent"] == 0
    assert artifact["metadata"]["training_samples"] + artifact["metadata"][
        "holdout_samples"
    ] == 1536
    assert artifact["metadata"]["fit_target"] == "raw_ramulator_service_time"
    assert validation["median_absolute_error_percent"] <= 15


def test_historical_v3_target_replays_but_is_outside_production_dma_domain() -> None:
    trace = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        seq_len=482,
        batch_size=16,
        num_layers=1,
    )
    mxfp8 = MemoryFormat("mxfp", 8, 8, 8, "MXFP8_BLOCK8")
    precision = MemoryPrecisionConfig(
        weight=mxfp8,
        activation=mxfp8,
        matrix_kv=mxfp8,
        vector_kv=mxfp8,
        integer=MemoryFormat("plain", 32, name="INT32"),
    )
    report = evaluate_compiler_cost(
        trace,
        TransactionalCycleModel.load(TARGET_SETTINGS),
        V3_ARTIFACT,
        precision,
        compute_timing_mode="legacy",
    )

    assert report.serial_latency_ns == pytest.approx(495_666_456, rel=0.10)
    assert report.hbm_read_bytes == 9_841_459_200
    assert report.hbm_write_bytes == 25_165_824
    # V3 was fitted before the production gather/scatter manifest semantics
    # used by V4.  Keep its historical numerical regression, but do not bless
    # the stale compatibility hashes or allow it to masquerade as in-domain.
    assert not report.calibration_in_domain
    assert {
        "dma_semantics_hash=mismatch",
        "request_geometry_hash=mismatch",
    }.issubset(report.compatibility["domain_issues"])
    assert report.memory_model_version == "global_v3"
    assert report.stage_bound
    for stage, serial in report.stage_latency_ns.items():
        compute = report.stage_compute_latency_ns.get(stage, 0.0)
        memory = report.hbm_stage_latency_ns.get(stage, 0.0)
        assert serial == pytest.approx(compute + memory)
        assert report.stage_roofline_latency_ns[stage] == max(compute, memory)
