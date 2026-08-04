from __future__ import annotations

import math
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from .memory import (
    DMAObservation,
    DMARequestDescriptor,
    REQUEST_FEATURES,
    RequestLatencyModel,
    RequestModelFit,
    _mop4clxor_coordinates,
    dma_request_features,
    expand_dma_requests,
    request_holdout_report,
)
from .calibration_provenance import build_calibration_audit
from transactional_emulator.testbench.calibration.dma_microbench import (
    PRECISIONS,
    SweepConfiguration,
    build_asm,
    patch_settings,
    plan_requests,
)
from runtime_paths import settings_path


def _descriptor(index: int) -> DMARequestDescriptor:
    return DMARequestDescriptor(
        opcode="H_PREFETCH_M",
        hbm_generation="HBM2",
        channels=8,
        address=64 * (index * 37 + index % 5),
        rows=1 + index % 4,
        elements_per_row=64 * (1 + index % 5),
        stride_bytes=64 * (1 + (index * 3) % 11),
        element_bits=4,
        direction="read",
        pin_rate_gbps=2.0,
        tensor="weight",
        scale_bits=8,
        block_size=8,
        scale_address=1 << 20 | (index * 64),
        scale_stride_bytes=64 * (1 + index % 3),
    )


def _observations() -> tuple[DMAObservation, ...]:
    coefficients = dict(
        zip(
            REQUEST_FEATURES,
            (
                9e-9,
                1.1e-9,
                0.6e-9,
                0.7e-9,
                0.4e-9,
                2.2e-9,
                0.3e-9,
                3.1e-9,
                0.2e-9,
                1.4e-9,
            ),
        )
    )
    result = []
    for index in range(30):
        descriptor = _descriptor(index)
        features = dma_request_features(descriptor)
        residual = sum(
            coefficients[name] * value
            for name, value in zip(REQUEST_FEATURES, features.values)
        )
        result.append(
            DMAObservation(
                descriptor=descriptor,
                measured_s=features.critical_channel_floor_s + residual,
            )
        )
    return tuple(result)


def test_request_expansion_includes_scale_plane_and_store_rmw() -> None:
    descriptor = DMARequestDescriptor(
        opcode="H_STORE_V",
        hbm_generation="HBM2",
        channels=8,
        address=17,
        rows=2,
        elements_per_row=64,
        stride_bytes=80,
        element_bits=4,
        direction="write",
        pin_rate_gbps=2.0,
        tensor="activation",
        scale_bits=8,
        block_size=8,
        scale_address=4099,
        scale_stride_bytes=9,
        partial_write_rmw=True,
    )
    requests = expand_dma_requests(descriptor)
    features = dma_request_features(descriptor)
    assert {request.tensor for request in requests} == {
        "activation",
        "activation_scale",
    }
    assert features.physical_read_bytes == features.physical_write_bytes
    assert features.physical_read_bytes > 0
    assert features.to_dict()["read_modify_write_requests"] == len(requests) / 2


def test_nonnegative_request_model_recovers_synthetic_latency() -> None:
    observations = _observations()
    model = RequestLatencyModel.fit(observations, ridge=1e-24)
    assert all(
        coefficient >= 0
        for fit in model.to_dict()["fits"]
        for coefficient in fit["coefficients_s"].values()
    )
    for observation in observations:
        predicted = model.predict(observation.descriptor)
        assert math.isclose(predicted, observation.measured_s, rel_tol=1e-5)


def test_request_model_reports_at_least_fifteen_percent_holdout() -> None:
    report = request_holdout_report(
        _observations(),
        holdout_fraction=0.20,
        ridge=1e-24,
    )
    assert report["holdout_fraction"] >= 0.15
    assert report["split_unit"] == "descriptor_fingerprint"
    assert report["fit_objective"] == "relative_error_nonnegative_ridge"
    assert report["p95_absolute_error_percent"] < 0.01
    assert report["worst_absolute_error_percent"] >= report[
        "p95_absolute_error_percent"
    ]


def test_request_descriptor_rejects_non_power_of_two_channels() -> None:
    with pytest.raises(ValueError, match="power of two"):
        DMARequestDescriptor(
            opcode="H_PREFETCH_V",
            hbm_generation="HBM2",
            channels=12,
            address=0,
            rows=1,
            elements_per_row=64,
            stride_bytes=64,
            element_bits=8,
            direction="read",
            pin_rate_gbps=2.0,
        )


def test_request_model_rejects_uncalibrated_pin_rate() -> None:
    model = RequestLatencyModel.fit(_observations(), ridge=1e-24)
    with pytest.raises(ValueError, match="cannot extrapolate"):
        model.predict(replace(_descriptor(0), pin_rate_gbps=3.2))


def test_request_stream_carries_rows_without_retaining_cross_call_state() -> None:
    first = DMARequestDescriptor(
        opcode="H_PREFETCH_M",
        hbm_generation="HBM2",
        channels=8,
        address=0,
        rows=1,
        elements_per_row=64,
        stride_bytes=64,
        element_bits=8,
        direction="read",
        pin_rate_gbps=2.0,
        tensor="weight",
    )
    # Address bit 17 is the first row bit in the pinned 8-channel mapper, so
    # this descriptor touches the same four channel-bank pairs in a new row.
    next_row = replace(first, address=1 << 17)
    coefficients = [0.0] * len(REQUEST_FEATURES)
    coefficients[REQUEST_FEATURES.index("row_conflicts")] = 2e-9
    model = RequestLatencyModel(
        (
            RequestModelFit(
                opcode="H_PREFETCH_M",
                hbm_generation="HBM2",
                channels=8,
                coefficients_s=tuple(coefficients),
                training_points=1,
            ),
        ),
        "request-latency-" + "b" * 64,
    )
    runs = ((first, 1), (first, 2), (next_row, 1))

    initial = model.predict_stream(runs)
    repeated = model.predict_stream(runs)

    assert initial == repeated
    assert [item.carried_row_conflicts for item in initial] == [0, 0, 4]
    assert initial[-1].seconds - initial[-1].isolated_seconds == pytest.approx(
        8e-9
    )


def test_request_stream_rejects_mixed_operating_points() -> None:
    descriptor = _descriptor(0)
    model = RequestLatencyModel.fit(_observations(), ridge=1e-24)
    with pytest.raises(ValueError, match="cannot mix HBM operating points"):
        model.predict_stream(
            ((descriptor, 1), (replace(descriptor, hbm_generation="HBM3"), 1))
        )


def test_mop4clxor_matches_pinned_hbm_mapping() -> None:
    assert _mop4clxor_coordinates(0, 8) == (0, 0, 0)
    assert _mop4clxor_coordinates(16, 8) == (1, 0, 0)
    assert _mop4clxor_coordinates(32, 8) == (2, 0, 0)
    assert _mop4clxor_coordinates(48, 8) == (3, 0, 0)
    assert _mop4clxor_coordinates(64, 8) == (1, 0, 0)
    assert _mop4clxor_coordinates(128, 8) == (2, 0, 0)
    assert _mop4clxor_coordinates(192, 8) == (3, 0, 0)


def test_structured_dma_plan_matches_emulator_settings() -> None:
    configuration = SweepConfiguration("HBM3", 32, 64, PRECISIONS["mxint4"])
    requests = plan_requests(
        configuration,
        stride_multipliers=(1, 4),
        alignments=(0, 16),
        replicas=2,
    )
    assert len(requests) == 3 * 2 * 2 * 2
    assert all(request.scale_stride_bytes * 4 == request.stride_bytes for request in requests)
    assert [line.split()[0] for line in build_asm(requests).splitlines() if line.startswith("H_")] == [
        request.opcode for request in requests
    ]

    patched = tomllib.loads(
        patch_settings(settings_path().read_text(encoding="utf-8"), configuration)
    )["TRANSACTIONAL"]
    assert patched["CONFIG"]["HBM_CHANNELS"]["value"] == 32
    assert patched["CONFIG"]["HBM_M_Prefetch_Amount"]["value"] == 64
    assert patched["PRECISION"]["HBM_M_WEIGHT_TYPE"]["ELEM"] == {
        "type": "Int",
        "width": 4,
    }


def test_structured_calibration_audit_binds_grid_and_validation() -> None:
    audit = build_calibration_audit(Path(__file__).parents[2], verify_git=False)
    request = audit["structured_request_measurement"]

    assert request["row_count"] == 8_640
    assert request["cartesian_grid_complete"] is True
    assert request["measurement_evidence_tier"] == "ramulator2_simulated"
    assert request["publication_receipt_complete"] is True
    assert request["raw_run_receipts_retained"] is True
    assert request["receipt"]["process_count"] == 8_640
    assert request["receipt"]["unique_process_ids"] == 8_640
    assert request["receipt"]["successful_process_count"] == 8_640
    assert request["receipt"]["observations_match_csv"] is True
    assert request["receipt"]["current_emulator_binary_verified"] is True
    assert request["validation"]["holdout_fraction"] >= 0.15
    # The holdout split and the fit are both deterministic given the pinned
    # CSV, so this metric is exactly reproducible at 23.403%. The bound sits
    # just above it to catch any material regression in the request model.
    assert request["validation"]["p95_absolute_error_percent"] < 24.0
    assert audit["publication_receipt_complete"] is False
    assert audit["publication_receipts"] == {
        "aggregate": False,
        "structured_request": True,
    }
    assert audit["missing_run_receipt_scope"] == "aggregate_measurements_only"
