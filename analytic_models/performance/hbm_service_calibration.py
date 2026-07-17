"""Deterministic V3 global HBM service calibration workflow."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .hbm_service_model import (
    HbmConfig,
    HbmServiceModel,
    HbmServiceSample,
    MemoryFormat,
    PhysicalDmaStream,
    PhysicalRepeatAxis,
    summarize_physical_stream,
    validate_holdout,
)


CALIBRATION_SCHEMA_VERSION = 3
DEFAULT_SEED = 20260711
DEFAULT_CHANNELS = (8, 32, 128)
DEFAULT_DIMENSIONS = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_DSE_AMOUNTS = (2, 4, 8, 16, 32, 64, 128)
DEFAULT_MAX_PATTERNS = 1536


def _conditioner_addresses(seed: int, channel_count: int, count: int = 16) -> list[int]:
    state = hashlib.sha256(f"{seed}:conditioner:c{channel_count}".encode()).digest()
    result = []
    for index in range(count):
        state = hashlib.sha256(state + index.to_bytes(4, "little")).digest()
        result.append((int.from_bytes(state[:4], "little") % (1 << 20)) * 64)
    return result


def _group_split(seed: int, group: str) -> str:
    digest = hashlib.sha256(f"{seed}:split:{group}".encode()).digest()
    return "holdout" if int.from_bytes(digest[:4], "little") % 5 == 0 else "train"


def _address_base(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:address:{identity}".encode()).digest()
    return (int.from_bytes(digest[:4], "little") % (1 << 20)) * 64


def _stream_axes(
    family: str,
    *,
    element_footprint: int,
    scale_footprint: int,
    instruction_count: int,
) -> tuple[PhysicalRepeatAxis, ...]:
    element_delta = max(64, ((element_footprint + 63) // 64) * 64)
    scale_delta = max(64, ((scale_footprint + 63) // 64) * 64)
    if family == "single":
        return ()
    if family == "reuse":
        return (PhysicalRepeatAxis("reuse", instruction_count, 0, 0),)
    if family == "affine":
        return (PhysicalRepeatAxis("affine", instruction_count, element_delta, scale_delta),)
    if family == "nested":
        outer = max(1, min(8, math.isqrt(instruction_count)))
        inner = max(1, instruction_count // outer)
        return (
            PhysicalRepeatAxis("outer_reuse", outer, 0, 0),
            PhysicalRepeatAxis("inner_affine", inner, element_delta, scale_delta),
        )
    raise ValueError(f"unknown calibration stream family {family!r}")


def generate_hbm_service_calibration_plan(
    *,
    seed: int = DEFAULT_SEED,
    repetitions: int = 1,
    warmup: int = 0,
    channels: Sequence[int] = DEFAULT_CHANNELS,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    dse_amounts: Sequence[int] = DEFAULT_DSE_AMOUNTS,
    max_patterns: int = DEFAULT_MAX_PATTERNS,
) -> dict[str, Any]:
    if repetitions <= 0 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    if max_patterns <= 0 or max_patterns > DEFAULT_MAX_PATTERNS:
        raise ValueError(f"max_patterns must be in [1, {DEFAULT_MAX_PATTERNS}]")
    if not channels or not dimensions or not dse_amounts:
        raise ValueError("calibration channels, dimensions and amounts cannot be empty")

    physical_formats = (
        (
            MemoryFormat("mxint", 4, 8, 64, "MXINT4"),
            ("MXINT4", "MXFP4"),
        ),
        (
            MemoryFormat("mxint", 8, 8, 64, "MXINT8"),
            ("MXINT8", "MXFP8"),
        ),
        (
            MemoryFormat("mxfp", 8, 8, 8, "MXFP8_BLOCK8"),
            ("MXFP8_BLOCK8",),
        ),
    )
    opcodes = (
        ("H_PREFETCH_M", "read", "weight"),
        ("H_PREFETCH_V", "read", "activation"),
        ("H_STORE_V", "write", "activation"),
    )
    families = ("single", "reuse", "affine", "nested")
    patterns = []
    for channel_count in channels:
        for dim_index, dim in enumerate(dimensions):
            for fmt, equivalent_formats in physical_formats:
                element_row_bytes = dim * fmt.element_bits // 8
                scale_row_bytes = (dim // fmt.block) * fmt.scale_bits // 8
                for opcode, direction, role in opcodes:
                    for amount_variant in range(3):
                        requested_amount = (
                            dim * (1 << amount_variant)
                            if opcode == "H_PREFETCH_M"
                            else dse_amounts[
                                (dim_index * 3 + amount_variant) % len(dse_amounts)
                            ]
                        )
                        for family_index, family in enumerate(families):
                            if family == "single":
                                logical_stride = dim
                                alignment = 0
                                requested_instruction_count = 1
                            elif family == "reuse":
                                logical_stride = dim
                                alignment = 32
                                requested_instruction_count = 8
                            elif family == "affine":
                                logical_stride = 2 * dim
                                alignment = 0
                                requested_instruction_count = 64
                            else:
                                logical_stride = max(8192, dim)
                                alignment = 32
                                requested_instruction_count = 64
                            element_requests_per_row = (
                                alignment + element_row_bytes + 63
                            ) // 64
                            if direction == "read":
                                requests_per_row = element_requests_per_row + 1
                            else:
                                scale_requests_per_row = (
                                    alignment + scale_row_bytes + 63
                                ) // 64
                                requests_per_row = 2 * (
                                    element_requests_per_row + scale_requests_per_row
                                )
                            instruction_count = min(
                                requested_instruction_count,
                                max(1, 2048 // max(1, requests_per_row)),
                            )
                            amount = min(
                                requested_amount,
                                max(1, 2048 // max(1, requests_per_row * instruction_count)),
                            )
                            stride_bytes = logical_stride * fmt.element_bits // 8
                            scale_stride = logical_stride * fmt.scale_bits // fmt.block // 8
                            element_footprint = amount * stride_bytes + element_row_bytes
                            scale_footprint = amount * scale_stride + scale_row_bytes
                            axes = _stream_axes(
                                family,
                                element_footprint=element_footprint,
                                scale_footprint=scale_footprint,
                                instruction_count=instruction_count,
                            )
                            group = (
                                f"{opcode}:{fmt.request_signature()}:d{dim}:a{requested_amount}:"
                                f"s{logical_stride}:r{alignment}:{family}"
                            )
                            identity = f"{group}:c{channel_count}:v{amount_variant}:f{family_index}"
                            element_base = _address_base(seed, identity) + alignment
                            scale_base = (1 << 28) + _address_base(seed, f"scale:{identity}") + alignment
                            multiplicity = math.prod(axis.count for axis in axes) if axes else 1
                            pattern_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
                            patterns.append(
                                {
                                    "id": f"v3-{pattern_id}",
                                    "group": group,
                                    "split_group": group,
                                    "split": _group_split(seed, group),
                                    "channels": int(channel_count),
                                    "repetitions": repetitions,
                                    "warmup": warmup,
                                    "precision_role": role,
                                    "equivalent_formats": list(equivalent_formats),
                                    "format": asdict(fmt),
                                    "transfer": {
                                        "opcode": opcode,
                                        "direction": direction,
                                        "precision": role,
                                        "element_base": element_base,
                                        "scale_base": scale_base,
                                        "dim": int(dim),
                                        "amount": int(amount),
                                        "stride_bytes": stride_bytes,
                                        "rstride": 1,
                                        "write_amount": int(dim if opcode == "H_PREFETCH_M" else 1),
                                    },
                                    "repeat_axes": [asdict(axis) for axis in axes],
                                    "stream_instruction_count": multiplicity,
                                    "requested_amount": int(requested_amount),
                                    "stream_family": family,
                                    "conditioner_addresses": _conditioner_addresses(
                                        seed, int(channel_count)
                                    ),
                                    "run_transactional": bool(
                                        fmt.family == "mxfp"
                                        and fmt.element_bits == 8
                                        and fmt.scale_bits == 8
                                        and fmt.block == 8
                                        and (
                                            opcode != "H_PREFETCH_M"
                                            or amount % dim == 0
                                        )
                                    ),
                                }
                            )
    if len(patterns) > max_patterns:
        patterns.sort(
            key=lambda pattern: hashlib.sha256(
                f"{seed}:select:{pattern['id']}".encode()
            ).digest()
        )
        patterns = patterns[:max_patterns]
    patterns.sort(key=lambda pattern: pattern["id"])
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "seed": seed,
        "ramulator_preset": "HBM2_2Gbps",
        "mapper": "MOP4CLXOR",
        "request_bytes": 64,
        "max_sampled_requests_per_geometry": 2048,
        "geometry_fidelity": "physical_request_exact",
        "patterns": patterns,
    }


def write_hbm_service_calibration_plan(path: str | Path, **kwargs) -> dict[str, Any]:
    plan = generate_hbm_service_calibration_plan(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _load_json(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_from_pattern(pattern: Mapping[str, Any]) -> tuple[PhysicalDmaStream, MemoryFormat]:
    transfer = pattern["transfer"]
    axes = tuple(PhysicalRepeatAxis(**axis) for axis in pattern.get("repeat_axes", ()))
    multiplicity = math.prod(axis.count for axis in axes) if axes else 1
    return (
        PhysicalDmaStream(
            stage="calibration",
            opcode=str(transfer["opcode"]),
            direction=str(transfer["direction"]),
            precision_role=str(pattern["precision_role"]),
            format_signature=MemoryFormat(**pattern["format"]).request_signature(),
            element_base=int(transfer["element_base"]),
            scale_base=int(transfer["scale_base"]),
            dim=int(transfer["dim"]),
            amount=int(transfer["amount"]),
            stride_bytes=int(transfer["stride_bytes"]),
            rstride=int(transfer.get("rstride", 1)),
            write_amount=int(transfer.get("write_amount", 1)),
            axes=axes,
            multiplicity=multiplicity,
            stream_index=0,
            source="hbm_service_calibration_v3",
        ),
        MemoryFormat(**pattern["format"]),
    )


def fit_hbm_service_model_from_ramulator(
    plan_value: str | Path | Mapping[str, Any],
    results_value: str | Path | Mapping[str, Any],
    *,
    ridge: float = 1e-8,
) -> tuple[HbmServiceModel, dict[str, Any]]:
    plan = _load_json(plan_value)
    results = _load_json(results_value)
    if int(plan.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported V3 calibration plan schema {plan.get('schema_version')!r}")
    if int(results.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported V3 calibration result schema {results.get('schema_version')!r}"
        )
    by_id = {item["id"]: item for item in results["patterns"]}
    samples = []
    raw_sample_count = 0
    transactional_sample_count = 0
    byte_mismatches = []
    transactional_latency_errors = []
    channels = set()
    dimensions = set()
    signatures = set()
    for pattern in plan["patterns"]:
        if pattern["id"] not in by_id:
            raise ValueError(f"Ramulator results are missing pattern {pattern['id']!r}")
        result = by_id[pattern["id"]]
        stream, fmt = _stream_from_pattern(pattern)
        hbm = HbmConfig(int(pattern["channels"]))
        geometry = summarize_physical_stream(stream, fmt, hbm)
        observed_bytes = (
            int(result["request_read_bytes"]),
            int(result["request_write_bytes"]),
        )
        expected_bytes = (geometry.physical_read_bytes, geometry.physical_write_bytes)
        if observed_bytes != expected_bytes:
            byte_mismatches.append(
                {"id": pattern["id"], "expected": expected_bytes, "observed": observed_bytes}
            )
        raw_sample_count += 1
        raw_latency = float(result["raw_ramulator_median_latency_ns"])
        samples.append(
            HbmServiceSample(
                geometry=geometry,
                channels=hbm.channels,
                observed_latency_ns=raw_latency,
                split=str(pattern["split"]),
                family=str(pattern["group"]),
            )
        )
        transactional_latency = result.get("transactional_dma_median_latency_ns")
        if transactional_latency is not None:
            transactional_sample_count += 1
            transactional_bytes = (
                int(result["transactional_read_bytes"]),
                int(result["transactional_write_bytes"]),
            )
            if transactional_bytes != expected_bytes:
                byte_mismatches.append(
                    {
                        "id": pattern["id"],
                        "expected": expected_bytes,
                        "observed_transactional": transactional_bytes,
                    }
                )
            transactional_latency_errors.append(
                100.0
                * abs(float(transactional_latency) - raw_latency)
                / max(float(transactional_latency), 1.0)
            )
        channels.add(hbm.channels)
        dimensions.add(stream.dim)
        signatures.add(fmt.request_signature())
    if byte_mismatches:
        raise ValueError(
            f"V3 physical request audit failed for {len(byte_mismatches)} patterns: "
            f"{byte_mismatches[:3]}"
        )
    if not samples:
        raise ValueError("V3 calibration results contain no raw Ramulator service samples")
    compatibility = {
        "trace_schema_version": 3,
        "ramulator_preset": plan["ramulator_preset"],
        "mapper": plan["mapper"],
        "request_bytes": int(plan["request_bytes"]),
        "dma_semantics_hash": _file_sha256(
            Path(__file__).resolve().parents[2] / "transactional_emulator/src/dma.rs"
        ),
        "request_geometry_hash": _file_sha256(
            Path(__file__).resolve().with_name("hbm_service_model.py")
        ),
        "calibration_driver_hash": _file_sha256(
            Path(__file__).resolve().parents[2]
            / "transactional_emulator/src/bin/hbm_dma_calibration.rs"
        ),
        "domain": {
            "channels": sorted(channels),
            "dimensions": sorted(dimensions),
            "request_signatures": sorted(signatures),
        },
    }
    model = HbmServiceModel.fit(
        samples,
        ridge=ridge,
        compatibility=compatibility,
        metadata={
            "seed": int(plan["seed"]),
            "training_samples": sum(sample.split == "train" for sample in samples),
            "holdout_samples": sum(sample.split == "holdout" for sample in samples),
            "source_kind": "format_generic_raw_requests_with_production_dma_audit",
            "fit_target": "raw_ramulator_service_time",
            "request_level_samples": raw_sample_count,
            "transactional_dma_samples": transactional_sample_count,
            "transactional_dma_max_latency_error_percent": max(
                transactional_latency_errors, default=0.0
            ),
            "relative_error_weighted_fit": True,
        },
    )
    validation = validate_holdout(model, samples)
    validation["physical_request_byte_mismatches"] = 0
    validation["request_level_sample_count"] = raw_sample_count
    validation["transactional_dma_sample_count"] = transactional_sample_count
    validation["transactional_dma_max_latency_error_percent"] = max(
        transactional_latency_errors, default=None
    )
    return model, validation


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "fit_hbm_service_model_from_ramulator",
    "generate_hbm_service_calibration_plan",
    "write_hbm_service_calibration_plan",
]
