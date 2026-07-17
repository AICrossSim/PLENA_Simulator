"""Generate and consume ordered transactional-DMA calibration streams."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from compiler.aten.isa_builder import DmaTransfer, RepeatAxis

from .hbm_cost import (
    HbmCalibration,
    HbmFormat,
    StreamCalibrationSample,
    dma_stream_geometry,
    file_sha256,
    fit_hbm_stream_calibration,
)


PLAN_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 2


def _group_split(
    seed: int,
    cell: str,
    amount_multiplier: int,
    amount_multipliers: Sequence[int],
    stream_length: int,
    stream_lengths: Sequence[int],
) -> str:
    interior_lengths = sorted(set(stream_lengths))[1:-1]
    if stream_length not in interior_lengths:
        return "train"
    ordered_amounts = sorted(set(amount_multipliers))
    target_holdouts = max(1, round(0.2 * len(ordered_amounts) * len(stream_lengths)))
    target_holdouts = min(target_holdouts, len(ordered_amounts) * len(interior_lengths))
    candidates = [
        (amount, length) for length in interior_lengths for amount in ordered_amounts
    ]
    rotation = int.from_bytes(
        hashlib.sha256(f"{seed}:{cell}".encode()).digest()[:4], "little"
    ) % len(candidates)
    selected = {
        candidates[(rotation + index) % len(candidates)] for index in range(target_holdouts)
    }
    return "holdout" if (amount_multiplier, stream_length) in selected else "train"


def _conditioner_addresses(seed: int, pattern_id: str, count: int = 16) -> list[int]:
    result = []
    state = hashlib.sha256(f"{seed}:conditioner:{pattern_id}".encode()).digest()
    for index in range(count):
        state = hashlib.sha256(state + index.to_bytes(4, "little")).digest()
        result.append((int.from_bytes(state[:4], "little") % (1 << 20)) * 64)
    return result


def _instruction_delta(
    family: str,
    *,
    dim: int,
    amount: int,
    stride: int,
    block: int,
) -> tuple[int, int]:
    footprint = max(dim, amount * stride)
    if family == "reuse":
        return 0, 0
    if family == "affine":
        return footprint, math.ceil(footprint / block)
    if family == "far":
        element_delta = (1 << 20) + footprint
        return element_delta, math.ceil(element_delta / block)
    raise ValueError(f"unknown instruction delta family {family!r}")


def generate_dma_calibration_plan(
    *,
    seed: int = 20260711,
    repetitions: int = 3,
    warmup: int = 0,
    channels: Sequence[int] = (8, 32, 128),
    dimensions: Sequence[int] = (64, 128, 256, 512),
    amount_multipliers: Sequence[int] = (1, 2, 4),
    stream_lengths: Sequence[int] = (1, 8, 64),
    model_column_strides: Sequence[int] = (5120, 8192, 25600),
    alignment_pairs: Sequence[tuple[int, int]] = ((0, 0), (32, 32)),
    delta_families: Sequence[str] = ("reuse", "affine", "far"),
    nested_stream_shapes: Sequence[tuple[int, int]] = (),
    nested_opcodes: Sequence[str] = ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"),
    nested_delta_families: Sequence[str] = ("affine",),
    include_base_streams: bool = True,
    include_standard_strides: bool = True,
) -> dict[str, Any]:
    if repetitions <= 0 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    if any(
        value <= 0
        for values in (
            channels,
            dimensions,
            amount_multipliers,
            stream_lengths,
            model_column_strides,
        )
        for value in values
    ):
        raise ValueError("calibration dimensions and repeat counts must be positive")
    if any(value < 0 or value >= 64 for pair in alignment_pairs for value in pair):
        raise ValueError("calibration alignments must be byte residues in [0, 64)")
    if not delta_families or set(delta_families) - {"reuse", "affine", "far"}:
        raise ValueError(f"unsupported delta families: {delta_families!r}")
    if any(outer <= 0 or inner <= 0 for outer, inner in nested_stream_shapes):
        raise ValueError("nested stream axis counts must be positive")
    if set(nested_opcodes) - {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
        raise ValueError(f"unsupported nested opcodes: {nested_opcodes!r}")
    if set(nested_delta_families) - {"affine", "far"}:
        raise ValueError(
            f"nested streams support only affine/far inner deltas: {nested_delta_families!r}"
        )
    fmt = HbmFormat("mxfp8_e4m3_e8m0", element_bits=8, scale_bits=8, block=8)
    opcode_precision = {
        "H_PREFETCH_M": "weight",
        "H_PREFETCH_V": "activation",
        "H_STORE_V": "activation",
    }
    patterns = []
    for channel_count in channels:
        for dim in dimensions:
            for amount_multiplier in amount_multipliers:
                amount = dim * amount_multiplier
                stride_options = [
                    *(
                        [("contiguous", dim), ("2x", 2 * dim)]
                        if include_standard_strides
                        else []
                    ),
                    *(
                        (f"model_column_{model_stride}", max(model_stride, dim))
                        for model_stride in model_column_strides
                    ),
                ]
                for stride_name, stride in stride_options:
                    for element_alignment, scale_alignment in alignment_pairs:
                        alignment_name = f"e{element_alignment}_s{scale_alignment}"
                        for opcode, precision in opcode_precision.items():
                            direction = "write" if opcode == "H_STORE_V" else "read"
                            for stream_length in stream_lengths:
                                stream_cell = (
                                    f"{opcode}:d{dim}:{stride_name}:{alignment_name}"
                                )
                                family_digest = hashlib.sha256(
                                    (
                                        f"{seed}:{opcode}:{dim}:{amount_multiplier}:"
                                        f"{stride_name}:{alignment_name}"
                                    ).encode()
                                ).digest()
                                delta_family = (
                                    "reuse",
                                    "affine",
                                    "far",
                                )[int.from_bytes(family_digest[:2], "little") % 3]
                                if stream_length == 1:
                                    delta_family = "reuse"
                                element_delta, scale_delta = _instruction_delta(
                                    delta_family,
                                    dim=dim,
                                    amount=amount,
                                    stride=stride,
                                    block=fmt.block,
                                )
                                group = (
                                    f"{opcode}:d{dim}:a{amount_multiplier}:{stride_name}:"
                                    f"{alignment_name}:n{stream_length}:{delta_family}"
                                )
                                pattern_id = f"{group}:c{channel_count}"
                                address_cell = (
                                    f"{opcode}:c{channel_count}:d{dim}:"
                                    f"{stride_name}:{alignment_name}"
                                )
                                address_digest = hashlib.sha256(address_cell.encode()).digest()
                                address_base = (
                                    int.from_bytes(address_digest[:4], "little") % (1 << 16)
                                ) * 64
                                transfer = DmaTransfer(
                                    opcode=opcode,
                                    direction=direction,
                                    precision=precision,
                                    element_base=address_base + element_alignment,
                                    scale_base=(1 << 26) + scale_alignment,
                                    dim=dim,
                                    amount=amount,
                                    stride=stride,
                                    rstride=1,
                                    write_amount=dim if opcode == "H_PREFETCH_M" else 1,
                                    geometry_fidelity="exact",
                                    source="transactional_dma_calibration_v2",
                                )
                                axis = RepeatAxis(
                                    "stream_instruction",
                                    stream_length,
                                    element_base_delta=element_delta,
                                    scale_base_delta=scale_delta,
                                )
                                patterns.append(
                                    {
                                        "id": pattern_id,
                                        "group": group,
                                        "split_group": group,
                                        "split": _group_split(
                                            seed,
                                            stream_cell,
                                            amount_multiplier,
                                            amount_multipliers,
                                            stream_length,
                                            stream_lengths,
                                        ),
                                        "channels": channel_count,
                                        "repetitions": repetitions,
                                        "warmup": warmup,
                                        "precision": precision,
                                        "format": asdict(fmt),
                                        "transfer": asdict(transfer),
                                        "repeat_axes": [asdict(axis)],
                                        "stream_instruction_count": stream_length,
                                        "conditioner_addresses": _conditioner_addresses(
                                            seed, f"c{channel_count}"
                                        ),
                                    }
                                )
    expanded_patterns = []
    for pattern in patterns:
        stream_length = int(pattern["stream_instruction_count"])
        if stream_length == 1:
            expanded_patterns.append(pattern)
            continue
        group_prefix = pattern["group"].rsplit(":n", 1)[0]
        transfer = pattern["transfer"]
        holdout_rotation = int.from_bytes(
            hashlib.sha256(f"{seed}:{group_prefix}".encode()).digest()[:4], "little"
        ) % len(delta_families)
        holdout_families = {
            delta_families[(holdout_rotation + index) % len(delta_families)]
            for index in range(min(2, len(delta_families)))
        }
        for delta_family in delta_families:
            expanded = deepcopy(pattern)
            element_delta, scale_delta = _instruction_delta(
                delta_family,
                dim=int(transfer["dim"]),
                amount=int(transfer["amount"]),
                stride=int(transfer["stride"]),
                block=fmt.block,
            )
            group = f"{group_prefix}:n{stream_length}:{delta_family}"
            expanded["group"] = group
            expanded["split_group"] = group
            expanded["id"] = f"{group}:c{pattern['channels']}"
            if pattern["split"] == "holdout":
                expanded["split"] = (
                    "holdout" if delta_family in holdout_families else "train"
                )
            expanded["repeat_axes"][0]["element_base_delta"] = element_delta
            expanded["repeat_axes"][0]["scale_base_delta"] = scale_delta
            expanded_patterns.append(expanded)
    nested_patterns = []
    templates = [
        pattern
        for pattern in patterns
        if int(pattern["stream_instruction_count"]) == 1
        and pattern["transfer"]["opcode"] in nested_opcodes
    ]
    for template in templates:
        transfer = template["transfer"]
        for outer_count, inner_count in nested_stream_shapes:
            for delta_family in nested_delta_families:
                element_delta, scale_delta = _instruction_delta(
                    delta_family,
                    dim=int(transfer["dim"]),
                    amount=int(transfer["amount"]),
                    stride=int(transfer["stride"]),
                    block=fmt.block,
                )
                nested = deepcopy(template)
                shape_name = (
                    f"nested_o{outer_count}_i{inner_count}:{delta_family}"
                )
                group_prefix = template["group"].rsplit(":n", 1)[0]
                group = f"{group_prefix}:{shape_name}"
                nested["group"] = group
                nested["split_group"] = group
                nested["split"] = "train"
                nested["id"] = f"{group}:c{template['channels']}"
                nested["repeat_axes"] = [
                    asdict(RepeatAxis("outer_reuse", outer_count)),
                    asdict(
                        RepeatAxis(
                            f"inner_{delta_family}",
                            inner_count,
                            element_base_delta=element_delta,
                            scale_base_delta=scale_delta,
                        )
                    ),
                ]
                nested["stream_instruction_count"] = outer_count * inner_count
                nested_patterns.append(nested)
    output_patterns = [
        *(expanded_patterns if include_base_streams else []),
        *nested_patterns,
    ]
    if not output_patterns:
        raise ValueError("calibration plan has no enabled patterns")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "seed": seed,
        "ramulator_preset": "HBM2_2Gbps",
        "request_bytes": 64,
        "geometry_fidelity": "exact",
        "dma_format_constraint": "mxfp8_e4m3_e8m0_block8",
        "patterns": output_patterns,
    }


def write_dma_calibration_plan(path: str | Path, **kwargs) -> dict[str, Any]:
    plan = generate_dma_calibration_plan(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _load_json(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value)
    return json.loads(path.read_text()), file_sha256(path)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fit_hbm_calibration_from_ramulator(
    plan_value: str | Path | Mapping[str, Any],
    results_value: str | Path | Mapping[str, Any],
    *,
    ridge: float = 1e-8,
    refit_full: bool = True,
) -> tuple[HbmCalibration, dict[str, Any]]:
    plan, plan_hash = _load_json(plan_value)
    results, results_hash = _load_json(results_value)
    if int(plan.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration plan schema {plan.get('schema_version')!r}")
    if int(results.get("schema_version", -1)) != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration result schema {results.get('schema_version')!r}")
    result_by_id = {entry["id"]: entry for entry in results["patterns"]}

    training: list[StreamCalibrationSample] = []
    holdout: list[StreamCalibrationSample] = []
    channels = set()
    signatures = set()
    byte_mismatches = []
    raw_transactional_errors = []
    for pattern in plan["patterns"]:
        try:
            result = result_by_id[pattern["id"]]
        except KeyError as exc:
            raise ValueError(f"Ramulator results are missing pattern {pattern['id']!r}") from exc
        transfer_data = dict(pattern["transfer"])
        transfer_data["axes"] = tuple(
            RepeatAxis(**axis) for axis in transfer_data.get("axes", ())
        )
        transfer = DmaTransfer(**transfer_data)
        axes = tuple(RepeatAxis(**axis) for axis in pattern["repeat_axes"])
        fmt = HbmFormat(**pattern["format"])
        conditioner = pattern.get("conditioner_addresses", [])
        geometry = dma_stream_geometry(
            transfer,
            axes,
            fmt,
            previous_address=conditioner[-1] if conditioner else None,
        )
        channels.add(int(pattern["channels"]))
        signatures.add(fmt.signature())
        observed_read_bytes = int(result["transactional_read_bytes"])
        observed_write_bytes = int(result["transactional_write_bytes"])
        transactional_ns = float(result["transactional_dma_median_latency_ns"])
        raw_ns = float(result["raw_ramulator_median_latency_ns"])
        raw_transactional_errors.append(100.0 * abs(raw_ns - transactional_ns) / transactional_ns)
        if (geometry.read_bytes, geometry.write_bytes) != (
            observed_read_bytes,
            observed_write_bytes,
        ):
            byte_mismatches.append(
                {
                    "id": pattern["id"],
                    "expected": [geometry.read_bytes, geometry.write_bytes],
                    "observed": [observed_read_bytes, observed_write_bytes],
                }
            )
        sample = StreamCalibrationSample(
            opcode=transfer.opcode,
            geometry=geometry,
            channels=int(pattern["channels"]),
            observed_ns=transactional_ns,
            source=pattern["id"],
        )
        (holdout if pattern["split"] == "holdout" else training).append(sample)
    if byte_mismatches:
        raise ValueError(
            f"transactional DMA byte audit failed for {len(byte_mismatches)} patterns: "
            f"{byte_mismatches[:3]}"
        )

    root = Path(__file__).resolve().parents[2]
    dma_source = root / "transactional_emulator/src/dma.rs"
    calibration_source = root / "transactional_emulator/src/dma_calibration.rs"
    compatibility = {
        "ramulator_preset": plan["ramulator_preset"],
        "request_bytes": int(plan["request_bytes"]),
        "channels": sorted(channels),
        "precision_signatures": sorted(signatures),
        "dma_semantics_hash": file_sha256(dma_source) if dma_source.exists() else "unknown",
        "trace_schema_version": 2,
        "dma_format_constraint": plan["dma_format_constraint"],
    }
    metadata = {
        "seed": int(plan["seed"]),
        "source_kind": "transactional_dma_ordered_stream_driver",
        "driver": results.get("driver", "unknown"),
        "plan_sha256": plan_hash,
        "results_sha256": results_hash,
        "driver_source_sha256": (
            file_sha256(calibration_source) if calibration_source.exists() else "unknown"
        ),
        "training_pattern_count": len(training),
        "holdout_pattern_count": len(holdout),
    }
    calibration = fit_hbm_stream_calibration(
        training,
        ridge=ridge,
        compatibility=compatibility,
        metadata=metadata,
        store_samples=False,
    )

    errors = []
    error_details = []
    by_opcode: dict[str, list[float]] = {}
    predictor_errors: dict[str, list[float]] = {
        "selected": [],
        "global_linear": [],
        "local_linear": [],
    }
    for sample in holdout:
        stream_model = calibration.stream_models[sample.opcode]
        features = sample.geometry.features(sample.channels)
        predictions = {
            "selected": calibration.predict_stream_ns(
                sample.opcode, sample.geometry, sample.channels
            ),
            "global_linear": stream_model.global_model.predict_ns(features),
        }
        local_model = stream_model.local_models.get(
            sample.geometry.calibration_cell(sample.channels)
        )
        predictions["local_linear"] = (
            local_model.predict_ns(features)
            if local_model is not None
            else predictions["global_linear"]
        )
        predicted = predictions["selected"]
        error = 100.0 * abs(predicted - sample.observed_ns) / sample.observed_ns
        errors.append(error)
        for predictor, value in predictions.items():
            predictor_errors[predictor].append(
                100.0 * abs(value - sample.observed_ns) / sample.observed_ns
            )
        error_details.append(
            {
                "id": sample.source,
                "opcode": sample.opcode,
                "observed_ns": sample.observed_ns,
                "predicted_ns": predicted,
                "absolute_error_percent": error,
            }
        )
        by_opcode.setdefault(sample.opcode, []).append(error)
    validation = {
        "holdout_pattern_count": len(holdout),
        "dma_byte_mismatch_count": 0,
        "raw_transactional_pattern_count": len(raw_transactional_errors),
        "raw_transactional_median_error_percent": statistics.median(
            raw_transactional_errors
        ),
        "raw_transactional_max_error_percent": max(raw_transactional_errors),
        "median_absolute_error_percent": statistics.median(errors) if errors else None,
        "p95_absolute_error_percent": _percentile(errors, 0.95),
        "max_absolute_error_percent": max(errors) if errors else None,
        "by_opcode": {
            opcode: {
                "count": len(values),
                "median_absolute_error_percent": statistics.median(values),
                "p95_absolute_error_percent": _percentile(values, 0.95),
                "max_absolute_error_percent": max(values),
            }
            for opcode, values in sorted(by_opcode.items())
        },
        "predictor_diagnostics": {
            predictor: {
                "median_absolute_error_percent": statistics.median(values),
                "p95_absolute_error_percent": _percentile(values, 0.95),
                "max_absolute_error_percent": max(values),
            }
            for predictor, values in predictor_errors.items()
        },
        "worst_patterns": sorted(
            error_details, key=lambda value: value["absolute_error_percent"], reverse=True
        )[:24],
    }
    if refit_full:
        final_metadata = {
            **metadata,
            "validation_fit_pattern_count": len(training),
            "final_refit_pattern_count": len(training) + len(holdout),
            "validation_holdout_metrics": {
                "median_absolute_error_percent": validation[
                    "median_absolute_error_percent"
                ],
                "p95_absolute_error_percent": validation[
                    "p95_absolute_error_percent"
                ],
                "max_absolute_error_percent": validation[
                    "max_absolute_error_percent"
                ],
            },
        }
        calibration = fit_hbm_stream_calibration(
            [*training, *holdout],
            ridge=ridge,
            compatibility=compatibility,
            metadata=final_metadata,
            store_samples=False,
        )
    return calibration, validation


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "fit_hbm_calibration_from_ramulator",
    "generate_dma_calibration_plan",
    "write_dma_calibration_plan",
]
