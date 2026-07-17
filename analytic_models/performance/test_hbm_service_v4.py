from __future__ import annotations

from types import SimpleNamespace

from analytic_models.performance.hbm_service_model import MemoryFormat
from analytic_models.performance.hbm_service_v4 import (
    DMA_SEMANTIC_VERSION,
    HbmServiceModelV4,
    LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION,
    Mop4clxorRowState,
    _iter_schedule_dma_stream_indices,
    _schedule_dma_count,
    combined_request_manifest_hash,
    generate_hbm_service_v4_plan,
    occurrence_features,
    plan_dma_request_manifest,
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
