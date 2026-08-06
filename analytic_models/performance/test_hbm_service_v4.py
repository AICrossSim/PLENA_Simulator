from __future__ import annotations

from types import SimpleNamespace

import pytest

from compiler.aten.cost_emitter import CostTrace, MemoryEvent
from compiler.aten.isa_builder import DmaTransfer, RepeatAxis

from analytic_models.performance.hbm_service_model import (
    HbmConfig,
    MemoryFormat,
    MemoryPrecisionConfig,
    PhysicalDmaStream,
    PhysicalRepeatAxis,
)
from analytic_models.performance.hbm_service_v4 import (
    DMA_SEMANTIC_VERSION,
    HbmServiceModelV4,
    LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION,
    Mop4clxorRowState,
    V4DmaServiceProvider,
    _iter_schedule_dma_stream_indices,
    _schedule_dma_count,
    combined_request_manifest_hash,
    generate_hbm_service_v4_plan,
    occurrence_features,
    plan_dma_request_manifest,
    scale_hbm_service_v4_work_by_stage,
    stream_occurrence_transfer,
)


def _transfer(*, dim: int, element_base: int, scale_base: int) -> dict[str, int | str]:
    return {
        "opcode": "H_STORE_V",
        "direction": "write",
        "element_base": element_base,
        "scale_base": scale_base,
        "dim": dim,
        "amount": 1,
        "stride_bytes": dim // 2,
        "rstride": 1,
        "write_amount": 1,
    }


def test_full_line_store_skips_read() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    manifest = plan_dma_request_manifest(
        _transfer(dim=128, element_base=0, scale_base=60), fmt
    )

    assert manifest.read_lines == ()
    assert manifest.write_lines == (0,)
    assert manifest.full_lines == 1
    assert manifest.partial_lines == 0


def test_partial_line_store_reads_then_writes_once() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    manifest = plan_dma_request_manifest(
        _transfer(dim=64, element_base=0, scale_base=16), fmt
    )

    assert manifest.read_lines == (0,)
    assert manifest.write_lines == (0,)
    assert manifest.full_lines == 0
    assert manifest.partial_lines == 1


def test_overlapping_element_and_scale_patches_coalesce() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    manifest = plan_dma_request_manifest(
        _transfer(dim=128, element_base=0, scale_base=32), fmt
    )

    assert len(manifest.write_lines) == 1
    assert manifest.request_manifest_hash.startswith("fnv1a64:")
    assert combined_request_manifest_hash((manifest,)) == manifest.request_manifest_hash


def test_v4_default_plan_has_expected_2592_occurrence_points() -> None:
    plan = generate_hbm_service_v4_plan(repetitions=1)

    assert plan["schema_version"] == 4
    assert plan["dma_semantic_version"] == DMA_SEMANTIC_VERSION
    assert len(plan["patterns"]) == 2592
    assert {pattern["channels"] for pattern in plan["patterns"]} == {8, 32, 128}
    assert all(not pattern["run_raw"] for pattern in plan["patterns"])
    assert all(pattern["run_transactional"] for pattern in plan["patterns"])


def test_v4_row_state_plan_adds_only_targeted_vector_anchors() -> None:
    plan = generate_hbm_service_v4_plan(
        repetitions=1, include_row_state_anchors=True
    )
    anchors = [
        pattern
        for pattern in plan["patterns"]
        if pattern["stream_family"] == "row_hit_anchor"
    ]

    assert len(plan["patterns"]) == 2628
    assert len(anchors) == 36
    assert {pattern["transfer"]["opcode"] for pattern in anchors} == {
        "H_PREFETCH_V",
        "H_STORE_V",
    }
    assert all(pattern["split"] == "train" for pattern in anchors)


def test_compressed_schedule_dma_order_expands_only_memory_repeats() -> None:
    instruction = lambda stream: SimpleNamespace(  # noqa: E731
        opcode="H_PREFETCH_V", memory_stream_index=stream
    )
    compute = SimpleNamespace(opcode="V_ADD_VV", memory_stream_index=None)
    inner = SimpleNamespace(children=(instruction(1), compute, instruction(2)))
    repeated = SimpleNamespace(count=3, body=inner)
    schedule = SimpleNamespace(children=(compute, instruction(0), repeated))

    assert _schedule_dma_count(schedule) == 7
    assert list(_iter_schedule_dma_stream_indices(schedule)) == [
        0,
        1,
        2,
        1,
        2,
        1,
        2,
    ]


def test_mapper_row_translation_preserves_v4_features() -> None:
    fmt = MemoryFormat("mxfp", 8, 8, 8, "MXFP_E4M3")
    transfer = {
        "opcode": "H_STORE_V",
        "direction": "write",
        "element_base": 0x123400,
        "scale_base": 0x987600,
        "dim": 512,
        "amount": 64,
        "stride_bytes": 512,
        "rstride": 1,
        "write_amount": 1,
    }
    for channels in (8, 32, 128):
        mapper_row_period = 16_384 * channels
        translated = dict(transfer)
        translated["element_base"] += 3 * mapper_row_period
        translated["scale_base"] += 3 * mapper_row_period
        baseline = occurrence_features(
            plan_dma_request_manifest(transfer, fmt), transfer, channels
        )
        shifted = occurrence_features(
            plan_dma_request_manifest(translated, fmt), translated, channels
        )
        assert shifted == baseline


def _affine_v4_test_trace() -> CostTrace:
    q_axis = RepeatAxis(
        name="q_block",
        count=8,
        element_base_delta=512 * 512,
        scale_base_delta=512 * 64,
    )
    k_axis = RepeatAxis(
        name="k_block",
        count=4,
        element_base_delta=512 * 512,
        scale_base_delta=512 * 64,
    )
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        precision="matrix_kv",
        precision_role="kv",
        element_base=0x200000,
        scale_base=0x800000,
        dim=512,
        amount=512,
        stride=512,
    )
    return CostTrace(
        memory_events=[
            MemoryEvent(
                "layer/attention",
                transfer,
                q_axis.count * k_axis.count,
                enclosing_axes=(q_axis, k_axis),
                stream_index=0,
            )
        ],
        metadata={"num_layers": 1},
    )


def _affine_v4_test_model() -> HbmServiceModelV4:
    return HbmServiceModelV4(
        calibration_id="affine-test",
        coefficients={
            HbmServiceModelV4.group_key("H_PREFETCH_M", 128): {
                "read_phase_startup": 2.0,
                "read_channel_tail": 0.25,
                "read_row_conflict": 0.5,
            },
            HbmServiceModelV4.group_key("H_PREFETCH_V", 128): {
                "read_phase_startup": 2.0,
                "read_channel_tail": 0.25,
                "read_row_conflict": 0.5,
            },
            HbmServiceModelV4.group_key("H_STORE_V", 128): {
                "read_phase_startup": 2.0,
                "write_phase_startup": 3.0,
                "read_write_turnaround": 1.0,
                "read_channel_tail": 0.25,
                "write_channel_tail": 0.5,
                "read_row_conflict": 0.5,
                "write_row_conflict": 0.75,
            },
        },
        domains={},
    )


def _assert_v4_work_equal(left, right) -> None:
    for name in (
        "read_bytes",
        "write_bytes",
        "payload_read_bytes",
        "payload_write_bytes",
        "read_requests",
        "write_requests",
        "occurrence_count",
        "logical_occurrence_count",
        "calibration_in_domain",
        "domain_issues",
        "row_state_regime_counts",
        "stage_occurrence_count",
        "stage_row_state_regime_counts",
        "traffic_breakdown",
    ):
        assert getattr(left, name) == getattr(right, name), name
    for name in (
        "latency_ns",
        "theoretical_floor_ns",
        "max_extrapolation_ratio",
    ):
        assert getattr(left, name) == pytest.approx(
            getattr(right, name), rel=0.0, abs=1e-9
        ), name
    for name in (
        "opcode_latency_ns",
        "stage_latency_ns",
        "stage_theoretical_floor_ns",
        "stage_opcode_latency_ns",
    ):
        assert getattr(left, name) == pytest.approx(
            getattr(right, name), rel=0.0, abs=1e-9
        ), name


def test_affine_geometry_grouping_matches_literal_cold_occurrences() -> None:
    trace = _affine_v4_test_trace()
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    grouped = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(group_cold_geometries=True)
    literal = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(group_cold_geometries=False)

    _assert_v4_work_equal(grouped, literal)
    assert grouped.aggregation == "affine_feature_grouped_v2"
    assert grouped.exact_feature_equivalence is True
    assert grouped.unique_address_geometry_count == grouped.unique_geometry_count
    assert grouped.unique_feature_signature_count > 0
    # Overlapping element/scale read lines are unioned before vectorized
    # MOP4CLXOR mapping, so this formerly conservative fixture no longer
    # requires scalar manifest construction.
    assert grouped.scalar_fallback_count == 0
    assert grouped.occurrences_elided > 0


def test_v4_grouped_progress_callback_is_monotonic() -> None:
    trace = _affine_v4_test_trace()
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    updates = []
    V4DmaServiceProvider(
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
        prepare_global_row_state=False,
    ).aggregate(
        progress_callback=updates.append,
        geometry_batch_size=1,
    )

    assert updates
    progress = [int(update["progress_done"]) for update in updates]
    assert progress == sorted(progress)
    assert progress[-1] > 0
    assert all(update["phase"] == "v4_aggregation" for update in updates)


def test_sufficient_statistics_read_backend_matches_scalar_planner() -> None:
    axis = RepeatAxis(
        name="translated_rows",
        count=32,
        element_base_delta=4096,
        scale_base_delta=512,
    )
    trace = CostTrace(
        memory_events=[
            MemoryEvent(
                "layer/attention",
                DmaTransfer(
                    opcode="H_PREFETCH_V",
                    direction="read",
                    precision="vector_integer",
                    precision_role="integer",
                    element_base=0x200000,
                    scale_base=1 << 36,
                    dim=512,
                    amount=64,
                    stride=512,
                ),
                axis.count,
                enclosing_axes=(axis,),
                stream_index=0,
            )
        ],
        metadata={"num_layers": 1},
    )
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    vectorized = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="sufficient-statistics-v2")
    scalar = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="scalar-v1")

    _assert_v4_work_equal(vectorized, scalar)
    assert vectorized.scalar_fallback_count == 0
    assert vectorized.exact_feature_equivalence is True


def test_sufficient_statistics_partial_store_backend_matches_scalar_planner() -> None:
    axis = RepeatAxis(
        name="translated_rows",
        count=32,
        element_base_delta=4096,
        scale_base_delta=512,
        logical_element_delta=4096,
    )
    trace = CostTrace(
        memory_events=[
            MemoryEvent(
                "layer/attention",
                DmaTransfer(
                    opcode="H_STORE_V",
                    direction="write",
                    precision="vector_integer",
                    precision_role="integer",
                    element_base=0x200008,
                    scale_base=(1 << 36) + 16,
                    dim=16,
                    amount=1,
                    stride=16,
                    write_amount=1,
                    memory_object="partial-store-fixture",
                    logical_object_elements=32 * 4096 + 64,
                    logical_element_offset=8,
                    logical_stride=16,
                ),
                axis.count,
                enclosing_axes=(axis,),
                stream_index=0,
            )
        ],
        metadata={"num_layers": 1},
    )
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    vectorized = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="sufficient-statistics-v2")
    scalar = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="scalar-v1")

    _assert_v4_work_equal(vectorized, scalar)
    assert vectorized.read_requests > 0
    assert vectorized.write_requests > 0
    assert vectorized.scalar_fallback_count == 0
    assert vectorized.exact_feature_equivalence is True


def test_sufficient_statistics_mx_store_with_scale_drift_matches_scalar() -> None:
    axis = RepeatAxis(
        name="translated_rows",
        count=32,
        element_base_delta=4096,
        scale_base_delta=512,
        logical_element_delta=4096,
    )
    trace = CostTrace(
        memory_events=[
            MemoryEvent(
                "layer/attention",
                DmaTransfer(
                    opcode="H_STORE_V",
                    direction="write",
                    precision="vector_activation",
                    precision_role="activation",
                    element_base=0,
                    scale_base=0,
                    dim=64,
                    amount=1,
                    stride=64,
                    write_amount=1,
                    memory_object="partial-mx-store-fixture",
                    logical_object_elements=32 * 4096 + 128,
                    logical_element_offset=8,
                    logical_stride=64,
                ),
                axis.count,
                enclosing_axes=(axis,),
                stream_index=0,
            )
        ],
        metadata={"num_layers": 1},
    )
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E5M2",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 64,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    vectorized = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="sufficient-statistics-v2")
    scalar = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(aggregation_backend="scalar-v1")

    _assert_v4_work_equal(vectorized, scalar)
    assert vectorized.scalar_fallback_count == 0


def test_causal_prefix_family_folding_matches_literal_occurrences() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        precision="matrix_kv",
        precision_role="kv",
        element_base=0,
        scale_base=0,
        dim=512,
        amount=512,
        stride=512,
        memory_object="causal-prefix-k",
        logical_object_elements=128 * 512 * 512,
        logical_element_offset=0,
        logical_scale_offset=0,
        logical_stride=512,
    )
    events = []
    for count in range(1, 9):
        axes = (
            ()
            if count == 1
            else (
                RepeatAxis(
                    name="streaming_kv_block",
                    count=count,
                    element_base_delta=512 * 512,
                    scale_base_delta=512 * 64,
                    logical_element_delta=512 * 512,
                    logical_scale_delta=512 * 64,
                ),
            )
        )
        events.append(
            MemoryEvent(
                "layer/attention",
                transfer,
                count,
                enclosing_axes=axes,
                stream_index=count - 1,
            )
        )
    trace = CostTrace(memory_events=events, metadata={"num_layers": 1})
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    folded = V4DmaServiceProvider(
        *provider_args,
        prepare_global_row_state=False,
    ).aggregate(aggregation_backend="sufficient-statistics-v2")
    literal = V4DmaServiceProvider(
        *provider_args,
        prepare_global_row_state=False,
    ).aggregate(group_cold_geometries=False)

    _assert_v4_work_equal(folded, literal)
    assert folded.prefix_stream_family_count == 1
    assert folded.prefix_stream_count_folded == 7


def test_overlapping_mx_envelope_uses_exact_relative_row_groups() -> None:
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E1M2",
            "activation": "MXFP_E1M2",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 64,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider = V4DmaServiceProvider(
        _affine_v4_test_trace(),
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
        prepare_global_row_state=False,
    )
    fmt = precision.weight
    stream = PhysicalDmaStream(
        stage="layer/ffn",
        opcode="H_PREFETCH_M",
        direction="read",
        precision_role="weight",
        format_signature=fmt.request_signature(),
        element_base=0x100000000,
        scale_base=0x100310000,
        dim=512,
        amount=512,
        stride_bytes=2560,
        rstride=1,
        write_amount=512,
        axes=(
            PhysicalRepeatAxis("output_row_tile", 17, 0, 0),
            PhysicalRepeatAxis("k_tile", 2, 1_310_720, 40_960),
            PhysicalRepeatAxis("decoder_layer", 4, 21_626_880, 21_626_880),
        ),
        multiplicity=17 * 2 * 4,
        stream_index=99,
        source="relative-row-test",
    )
    assert not provider._cold_stream_regions_are_disjoint(stream, fmt)

    grouped, used_scalar_fallback = provider._cold_geometry_groups(
        stream,
        fmt,
    )
    grouped_counts = {
        key: count
        for key, (_transfer, count) in grouped.items()
    }
    literal_counts = {}
    for occurrence in range(stream.multiplicity):
        key = provider._key(
            stream,
            fmt,
            stream_occurrence_transfer(stream, occurrence),
        )
        literal_counts[key] = literal_counts.get(key, 0) + 1

    assert used_scalar_fallback is False
    assert grouped_counts == literal_counts
    assert len(grouped_counts) < stream.multiplicity


def test_stage_scaling_matches_direct_grouped_multiplier() -> None:
    trace = _affine_v4_test_trace()
    precision = MemoryPrecisionConfig.from_mapping(
        {
            "weight": "MXFP_E4M3",
            "activation": "MXFP_E4M3",
            "kv": "MXFP_E4M3",
            "internal_fp": "FP_E4M3",
            "block": 8,
            "scale_bits": 8,
            "integer_bits": 32,
        }
    )
    provider_args = (
        trace,
        precision,
        HbmConfig(channels=128),
        _affine_v4_test_model(),
        1000,
    )
    one = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate()
    scaled = scale_hbm_service_v4_work_by_stage(
        one, {"layer/attention": 64}
    )
    direct = V4DmaServiceProvider(
        *provider_args, prepare_global_row_state=False
    ).aggregate(stage_multipliers={"layer/attention": 64})

    _assert_v4_work_equal(scaled, direct)


def test_native_bursts_can_map_one_line_to_multiple_channels() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    transfer = {
        "opcode": "H_PREFETCH_V",
        "direction": "read",
        "element_base": 0,
        "scale_base": 1 << 20,
        "dim": 128,
        "amount": 1,
        "stride_bytes": 64,
        "rstride": 1,
        "write_amount": 1,
    }
    manifest = plan_dma_request_manifest(transfer, fmt)
    features = occurrence_features(manifest, transfer, channels=128)

    # A 64-byte API line becomes four 16-byte Ramulator transfers.  Its
    # theoretical phase floor is therefore based on native channel mapping,
    # not on mapping the line base once and charging four cycles to it.
    assert features.theoretical_phase_floor_ns >= 1.0
    assert features.theoretical_phase_floor_ns < 4.0 * len(manifest.read_lines)


def test_warmed_rows_use_separate_residual_coefficients() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    transfer = {
        "opcode": "H_PREFETCH_V",
        "direction": "read",
        "element_base": 0,
        "scale_base": 1 << 20,
        "dim": 128,
        "amount": 1,
        "stride_bytes": 64,
        "rstride": 1,
        "write_amount": 1,
    }
    manifest = plan_dma_request_manifest(transfer, fmt)
    group = HbmServiceModelV4.group_key("H_PREFETCH_V", 128)
    model = HbmServiceModelV4(
        calibration_id="test",
        coefficients={
            group: {
                "read_phase_startup": 10.0,
                "write_phase_startup": 10.0,
            }
        },
        warm_coefficients={
            group: {
                "read_phase_startup": 5.0,
                "write_phase_startup": 0.0,
                "read_row_conflict": 0.0,
                "write_row_conflict": 0.0,
            }
        },
        domains={},
    )
    row_state = Mop4clxorRowState(128)

    cold = model.predict_manifest(
        "H_PREFETCH_V", transfer, fmt, 128, manifest, row_state=row_state
    )
    warm = model.predict_manifest(
        "H_PREFETCH_V", transfer, fmt, 128, manifest, row_state=row_state
    )

    assert cold.row_state_regime == "cold_or_mixed"
    assert warm.row_state_regime == "fully_warm"
    assert warm.latency_ns < cold.latency_ns


def test_open_but_conflicting_rows_use_cold_or_mixed_model() -> None:
    """An open bank is not warm when the next access targets another row."""

    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    transfer = {
        "opcode": "H_PREFETCH_V",
        "direction": "read",
        "element_base": 0,
        "scale_base": 1 << 20,
        "dim": 128,
        "amount": 1,
        "stride_bytes": 64,
        "rstride": 1,
        "write_amount": 1,
    }
    group = HbmServiceModelV4.group_key("H_PREFETCH_V", 8)
    model = HbmServiceModelV4(
        calibration_id="test",
        coefficients={
            group: {
                "read_phase_startup": 10.0,
                "read_row_conflict": 2.0,
            }
        },
        warm_coefficients={
            group: {
                "read_phase_startup": 1.0,
                "write_phase_startup": 0.0,
                "read_row_conflict": 0.0,
                "write_row_conflict": 0.0,
            }
        },
        domains={},
    )
    row_state = Mop4clxorRowState(8)
    first_manifest = plan_dma_request_manifest(transfer, fmt)
    first = model.predict_manifest(
        "H_PREFETCH_V", transfer, fmt, 8, first_manifest, row_state=row_state
    )

    # MOP4CLXOR maps a translation by this period to the same banks and
    # columns but a different DRAM row.
    conflicting = dict(transfer)
    mapper_row_period = 16_384 * 8
    conflicting["element_base"] += mapper_row_period
    conflicting["scale_base"] += mapper_row_period
    conflict_manifest = plan_dma_request_manifest(conflicting, fmt)
    conflict_features = occurrence_features(
        conflict_manifest, conflicting, 8, row_state=row_state
    )

    # Recreate the state because feature extraction updates it in place.
    row_state = Mop4clxorRowState(8)
    model.predict_manifest(
        "H_PREFETCH_V", transfer, fmt, 8, first_manifest, row_state=row_state
    )
    second = model.predict_manifest(
        "H_PREFETCH_V",
        conflicting,
        fmt,
        8,
        conflict_manifest,
        row_state=row_state,
    )

    assert first.row_state_regime == "cold_or_mixed"
    assert conflict_features.values["read_row_miss"] == 0.0
    assert conflict_features.values["read_initial_row_conflict"] > 0.0
    assert conflict_features.values["read_row_conflict"] > 0.0
    assert second.row_state_regime == "cold_or_mixed"

    legacy_model = HbmServiceModelV4(
        calibration_id="legacy-test",
        coefficients=model.coefficients,
        warm_coefficients=model.warm_coefficients,
        domains={},
        compatibility={
            "feature_semantic_version": LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION
        },
    )
    legacy_state = Mop4clxorRowState(8)
    legacy_model.predict_manifest(
        "H_PREFETCH_V", transfer, fmt, 8, first_manifest, row_state=legacy_state
    )
    legacy_second = legacy_model.predict_manifest(
        "H_PREFETCH_V",
        conflicting,
        fmt,
        8,
        conflict_manifest,
        row_state=legacy_state,
    )
    assert legacy_second.row_state_regime == "fully_warm"


def test_zero_width_domain_reports_finite_extrapolation_ratio() -> None:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    transfer = {
        "opcode": "H_PREFETCH_V",
        "direction": "read",
        "element_base": 0,
        "scale_base": 1 << 20,
        "dim": 128,
        "amount": 1,
        "stride_bytes": 64,
        "rstride": 1,
        "write_amount": 1,
    }
    group = HbmServiceModelV4.group_key("H_PREFETCH_V", 8)
    model = HbmServiceModelV4(
        calibration_id="domain-test",
        coefficients={group: {}},
        domains={
            group: {
                "features": {
                    "read_phase_startup": {"min": 0.0, "max": 0.0}
                }
            }
        },
    )

    prediction = model.predict_occurrence(
        "H_PREFETCH_V", transfer, fmt, channels=8
    )

    assert prediction.extrapolation_ratio == 2.0
