"""Physical DMA geometry and calibrated Ramulator timing surrogate."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


REQUEST_BYTES = 64
CALIBRATION_SCHEMA_VERSION = 2
_CHANNEL_ANCHORS = (8, 32, 128)
_DIM_ANCHORS = (64, 128, 256, 512)
_STRIDE_CLASSES = ("contiguous", "2x", "model_column")
_ALIGNMENT_CLASSES = ("aligned", "misaligned")
_BASE_FEATURE_NAMES = (
    "bias",
    "read_requests",
    "write_requests",
    "request_work_per_channel",
    "request_work_per_sqrt_channel",
    "row_element_requests",
    "parallel_request_depth",
    "parallel_depth_per_channel",
    "amount",
    "dim",
    "element_bytes_per_row",
    "scale_bytes_per_row",
    "strided",
    "misaligned",
    "aligned",
    "inverse_channels",
    "row_span_per_channel",
)
_RATE_FEATURE_NAMES = tuple(
    f"request_rate_c{channels}_d{dim}_{stride}_{alignment}"
    for channels in _CHANNEL_ANCHORS
    for dim in _DIM_ANCHORS
    for stride in _STRIDE_CLASSES
    for alignment in _ALIGNMENT_CLASSES
)
_FIXED_FEATURE_NAMES = tuple(name.replace("request_rate", "fixed") for name in _RATE_FEATURE_NAMES)
FEATURE_NAMES = _BASE_FEATURE_NAMES + _RATE_FEATURE_NAMES + _FIXED_FEATURE_NAMES
STREAM_FEATURE_NAMES = (
    "bias",
    "sqrt_request_work",
    "request_work",
    "request_work_per_channel",
    "read_requests",
    "write_requests",
    "sqrt_instruction_count",
    "instruction_count",
    "sqrt_unique_blocks",
    "unique_blocks",
    "reuse_requests",
    "reuse_ratio",
    "address_span_per_channel",
    "mean_adjacent_gap",
    "max_adjacent_gap",
    "initial_address_gap",
    "amount",
    "dim",
    "stride_bytes",
    "strided",
    "misaligned",
    "inverse_channels",
)
LOCAL_STREAM_FEATURE_NAMES = (
    "bias",
    "sqrt_instruction_count",
    "instruction_count",
    "sqrt_request_work",
    "request_work",
)


@dataclass(frozen=True)
class HbmFormat:
    name: str
    element_bits: int
    scale_bits: int = 0
    block: int = 1

    @property
    def is_mx(self) -> bool:
        return self.scale_bits > 0

    def signature(self) -> str:
        if self.is_mx:
            return f"mx:e{self.element_bits}:s{self.scale_bits}:b{self.block}"
        return f"plain:e{self.element_bits}"


@dataclass(frozen=True)
class DmaGeometry:
    read_requests: int
    write_requests: int
    read_bytes: int
    write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    element_bytes_per_row: int
    scale_bytes_per_row: int
    dim: int
    amount: int
    stride_bytes: int
    element_alignment: int
    scale_alignment: int
    row_element_requests: int
    parallel_request_depth: int
    row_span_bytes: int
    strided: bool
    misaligned: bool

    @property
    def request_work(self) -> int:
        return self.read_requests + self.write_requests

    @property
    def stride_class(self) -> str:
        if self.stride_bytes == self.element_bytes_per_row:
            return "contiguous"
        if self.stride_bytes == 2 * self.element_bytes_per_row:
            return "2x"
        return "model_column"

    @property
    def alignment_class(self) -> str:
        return "misaligned" if self.misaligned else "aligned"

    def calibration_cell(self, channels: int) -> str:
        return f"c{channels}:d{self.dim}:{self.stride_class}:{self.alignment_class}"

    def scaled(self, multiplicity: int) -> DmaGeometry:
        if multiplicity < 0:
            raise ValueError(f"multiplicity must be nonnegative, got {multiplicity}")
        return DmaGeometry(
            read_requests=self.read_requests * multiplicity,
            write_requests=self.write_requests * multiplicity,
            read_bytes=self.read_bytes * multiplicity,
            write_bytes=self.write_bytes * multiplicity,
            payload_read_bytes=self.payload_read_bytes * multiplicity,
            payload_write_bytes=self.payload_write_bytes * multiplicity,
            element_bytes_per_row=self.element_bytes_per_row,
            scale_bytes_per_row=self.scale_bytes_per_row,
            dim=self.dim,
            amount=self.amount,
            stride_bytes=self.stride_bytes,
            element_alignment=self.element_alignment,
            scale_alignment=self.scale_alignment,
            row_element_requests=self.row_element_requests,
            parallel_request_depth=self.parallel_request_depth,
            row_span_bytes=self.row_span_bytes,
            strided=self.strided,
            misaligned=self.misaligned,
        )

    def features(self, channels: int) -> dict[str, float]:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        request_work = self.request_work
        stride_class = self.stride_class
        alignment_class = self.alignment_class
        features = {
            "bias": 1.0,
            "read_requests": float(self.read_requests),
            "write_requests": float(self.write_requests),
            "request_work_per_channel": request_work / channels,
            "request_work_per_sqrt_channel": request_work / math.sqrt(channels),
            "row_element_requests": float(self.row_element_requests),
            "parallel_request_depth": float(self.parallel_request_depth),
            "parallel_depth_per_channel": self.parallel_request_depth / channels,
            "amount": float(self.amount),
            "dim": float(self.dim),
            "element_bytes_per_row": float(self.element_bytes_per_row),
            "scale_bytes_per_row": float(self.scale_bytes_per_row),
            "strided": float(self.strided),
            "misaligned": float(self.misaligned),
            "aligned": float(not self.misaligned),
            "inverse_channels": 1.0 / channels,
            "row_span_per_channel": self.row_span_bytes / channels,
        }
        features.update(dict.fromkeys(_RATE_FEATURE_NAMES, 0.0))
        features.update(dict.fromkeys(_FIXED_FEATURE_NAMES, 0.0))
        rate_name = (
            f"request_rate_c{channels}_d{self.dim}_{stride_class}_{alignment_class}"
        )
        if rate_name in features:
            features[rate_name] = float(request_work)
            features[rate_name.replace("request_rate", "fixed")] = 1.0
        return features


@dataclass(frozen=True)
class DmaStreamGeometry:
    instruction_count: int
    read_requests: int
    write_requests: int
    read_bytes: int
    write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    unique_blocks: int
    address_span_bytes: int
    mean_adjacent_gap_bytes: float
    max_adjacent_gap_bytes: int
    initial_address_gap_bytes: int
    final_element_base: int
    final_scale_base: int
    dim: int
    amount: int
    stride_bytes: int
    element_alignment: int
    scale_alignment: int
    strided: bool
    misaligned: bool

    @property
    def request_work(self) -> int:
        return self.read_requests + self.write_requests

    @property
    def reuse_requests(self) -> int:
        return max(0, self.request_work - self.unique_blocks)

    @property
    def reuse_ratio(self) -> float:
        if not self.request_work:
            return 0.0
        return self.reuse_requests / self.request_work

    @property
    def stride_class(self) -> str:
        if not self.strided:
            return "contiguous"
        if self.stride_bytes == 2 * self.dim:
            return "2x"
        return "model_column"

    @property
    def alignment_class(self) -> str:
        return f"ea{self.element_alignment}:sa{self.scale_alignment}"

    def calibration_cell(self, channels: int) -> str:
        return (
            f"c{channels}:d{self.dim}:s{self.stride_bytes}:"
            f"{self.alignment_class}"
        )

    def features(self, channels: int) -> dict[str, float]:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        return {
            "bias": 1.0,
            "sqrt_request_work": math.sqrt(self.request_work),
            "request_work": float(self.request_work),
            "request_work_per_channel": self.request_work / channels,
            "read_requests": float(self.read_requests),
            "write_requests": float(self.write_requests),
            "sqrt_instruction_count": math.sqrt(self.instruction_count),
            "instruction_count": float(self.instruction_count),
            "sqrt_unique_blocks": math.sqrt(self.unique_blocks),
            "unique_blocks": float(self.unique_blocks),
            "reuse_requests": float(self.reuse_requests),
            "reuse_ratio": self.reuse_ratio,
            "address_span_per_channel": self.address_span_bytes / channels,
            "mean_adjacent_gap": self.mean_adjacent_gap_bytes,
            "max_adjacent_gap": float(self.max_adjacent_gap_bytes),
            "initial_address_gap": float(self.initial_address_gap_bytes),
            "amount": float(self.amount),
            "dim": float(self.dim),
            "stride_bytes": float(self.stride_bytes),
            "strided": float(self.strided),
            "misaligned": float(self.misaligned),
            "inverse_channels": 1.0 / channels,
        }


def _data_type_bits(data_type: Mapping[str, Any]) -> int:
    kind = str(data_type.get("type", ""))
    if kind == "Fp":
        return int(bool(data_type.get("sign", False))) + int(data_type["exponent"]) + int(
            data_type["mantissa"]
        )
    if kind == "Int":
        return int(data_type["width"])
    raise ValueError(f"unsupported HBM data type: {data_type!r}")


def parse_hbm_format(name: str, section: Mapping[str, Any]) -> HbmFormat:
    fmt = str(section.get("format", ""))
    if fmt == "Mx":
        return HbmFormat(
            name=name,
            element_bits=_data_type_bits(section["ELEM"]),
            scale_bits=_data_type_bits(section["SCALE"]),
            block=int(section["block"]),
        )
    if fmt == "Plain":
        return HbmFormat(name=name, element_bits=_data_type_bits(section["DATA_TYPE"]))
    raise ValueError(f"unsupported HBM format {fmt!r} for {name}")


def load_transactional_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    try:
        return data["TRANSACTIONAL"]
    except KeyError as exc:
        raise ValueError(f"{path} has no TRANSACTIONAL section") from exc


def hbm_formats_from_settings(settings: Mapping[str, Any]) -> dict[str, HbmFormat]:
    precision = settings["PRECISION"]
    return {
        "matrix": parse_hbm_format("HBM_M_WEIGHT_TYPE", precision["HBM_M_WEIGHT_TYPE"]),
        "weight": parse_hbm_format("HBM_M_WEIGHT_TYPE", precision["HBM_M_WEIGHT_TYPE"]),
        "matrix_kv": parse_hbm_format("HBM_M_KV_TYPE", precision["HBM_M_KV_TYPE"]),
        "activation": parse_hbm_format("HBM_V_ACT_TYPE", precision["HBM_V_ACT_TYPE"]),
        "vector_kv": parse_hbm_format("HBM_V_KV_TYPE", precision["HBM_V_KV_TYPE"]),
        "integer": parse_hbm_format("HBM_V_INT_TYPE", precision["HBM_V_INT_TYPE"]),
    }


def _blocks_touched(address: int, length: int) -> int:
    if address < 0 or length < 0:
        raise ValueError(f"address and length must be nonnegative, got {address}, {length}")
    if length == 0:
        return 0
    return ((address % REQUEST_BYTES) + length + REQUEST_BYTES - 1) // REQUEST_BYTES


def dma_geometry(transfer: Any, hbm_format: HbmFormat) -> DmaGeometry:
    """Mirror ``transactional_emulator/src/dma.rs`` request construction."""
    dim = int(transfer.dim)
    amount = int(transfer.amount)
    if dim <= 0 or amount <= 0:
        raise ValueError(f"DMA dim and amount must be positive, got {dim}, {amount}")
    if (hbm_format.element_bits * dim) % 8:
        raise ValueError("DMA element row is not byte aligned")
    element_len = hbm_format.element_bits * dim // 8
    scale_len = 0
    if hbm_format.is_mx:
        if dim % hbm_format.block:
            raise ValueError(f"dim={dim} must be divisible by MX block={hbm_format.block}")
        scale_bits = hbm_format.scale_bits * (dim // hbm_format.block)
        if scale_bits % 8:
            raise ValueError("DMA scale row is not byte aligned")
        scale_len = scale_bits // 8

    stride = int(transfer.stride) if int(transfer.rstride) == 1 else dim
    scale_stride = stride / hbm_format.block if hbm_format.is_mx else 0.0
    element_base = int(transfer.element_base)
    scale_base = int(transfer.scale_base)
    direction = str(transfer.direction)

    read_requests = 0
    write_requests = 0
    payload_read = 0
    payload_write = 0
    max_element_requests = 0
    element_min: int | None = None
    element_max = 0
    scale_min: int | None = None
    scale_max = 0
    misaligned = False

    for index in range(amount):
        element_address = element_base + index * stride
        scale_address = scale_base + int(index * scale_stride)
        element_requests = _blocks_touched(element_address, element_len)
        max_element_requests = max(max_element_requests, element_requests)
        misaligned |= bool(element_address % REQUEST_BYTES)
        element_min = element_address if element_min is None else min(element_min, element_address)
        element_max = max(element_max, element_address + element_len)

        if direction == "read":
            read_requests += element_requests
            payload_read += element_len
            if scale_len:
                # transfer_mx_from_hbm intentionally emits one scale ChunkRead
                # per row and clamps it to the containing 64-byte block.
                read_requests += 1
                payload_read += min(scale_len, REQUEST_BYTES - scale_address % REQUEST_BYTES)
                misaligned |= bool(scale_address % REQUEST_BYTES)
                scale_min = scale_address if scale_min is None else min(scale_min, scale_address)
                scale_max = max(
                    scale_max,
                    scale_address + min(scale_len, REQUEST_BYTES - scale_address % REQUEST_BYTES),
                )
        elif direction == "write":
            # write_unaligned performs one read-modify-write per touched block.
            read_requests += element_requests
            write_requests += element_requests
            payload_read += element_requests * REQUEST_BYTES
            payload_write += element_len
            if scale_len:
                scale_requests = _blocks_touched(scale_address, scale_len)
                read_requests += scale_requests
                write_requests += scale_requests
                payload_read += scale_requests * REQUEST_BYTES
                payload_write += scale_len
                misaligned |= bool(scale_address % REQUEST_BYTES)
                scale_min = scale_address if scale_min is None else min(scale_min, scale_address)
                scale_max = max(scale_max, scale_address + scale_len)
        else:
            raise ValueError(f"unsupported DMA direction {direction!r}")

    element_span = element_max - (element_min or 0)
    scale_span = scale_max - (scale_min or 0)
    row_span = max(element_span, scale_span)
    row_requests = max_element_requests
    contiguous = stride == element_len
    parallel_depth = read_requests if direction == "read" else 1
    return DmaGeometry(
        read_requests=read_requests,
        write_requests=write_requests,
        read_bytes=read_requests * REQUEST_BYTES,
        write_bytes=write_requests * REQUEST_BYTES,
        payload_read_bytes=payload_read,
        payload_write_bytes=payload_write,
        element_bytes_per_row=element_len,
        scale_bytes_per_row=scale_len,
        dim=dim,
        amount=amount,
        stride_bytes=stride,
        element_alignment=element_base % REQUEST_BYTES,
        scale_alignment=scale_base % REQUEST_BYTES,
        row_element_requests=row_requests,
        parallel_request_depth=parallel_depth,
        row_span_bytes=row_span,
        strided=not contiguous,
        misaligned=misaligned,
    )


def dma_geometry_classes(
    transfer: Any,
    axes: Sequence[Any],
    hbm_format: HbmFormat,
) -> list[tuple[DmaGeometry, int]]:
    """Compress an affine DMA stream by its element/scale 64-byte residues."""
    states: Counter[tuple[int, int]] = Counter(
        {(int(transfer.element_base) % REQUEST_BYTES, int(transfer.scale_base) % REQUEST_BYTES): 1}
    )
    for axis in axes:
        axis_states: Counter[tuple[int, int]] = Counter()
        for index in range(int(axis.count)):
            axis_states[
                (
                    index * int(axis.element_base_delta) % REQUEST_BYTES,
                    index * int(axis.scale_base_delta) % REQUEST_BYTES,
                )
            ] += 1
        combined: Counter[tuple[int, int]] = Counter()
        for (element_residue, scale_residue), state_count in states.items():
            for (element_delta, scale_delta), axis_count in axis_states.items():
                combined[
                    (
                        (element_residue + element_delta) % REQUEST_BYTES,
                        (scale_residue + scale_delta) % REQUEST_BYTES,
                    )
                ] += state_count * axis_count
        states = combined

    base_element_residue = int(transfer.element_base) % REQUEST_BYTES
    base_scale_residue = int(transfer.scale_base) % REQUEST_BYTES
    return [
        (
            dma_geometry(
                replace(
                    transfer,
                    element_base=(
                        int(transfer.element_base)
                        + (element_residue - base_element_residue) % REQUEST_BYTES
                    ),
                    scale_base=(
                        int(transfer.scale_base)
                        + (scale_residue - base_scale_residue) % REQUEST_BYTES
                    ),
                ),
                hbm_format,
            ),
            count,
        )
        for (element_residue, scale_residue), count in sorted(states.items())
    ]


def _stream_offsets(axes: Sequence[Any]) -> Iterable[tuple[int, int]]:
    def visit(axis_index: int, element_delta: int, scale_delta: int):
        if axis_index == len(axes):
            yield element_delta, scale_delta
            return
        axis = axes[axis_index]
        for index in range(int(axis.count)):
            yield from visit(
                axis_index + 1,
                element_delta + index * int(axis.element_base_delta),
                scale_delta + index * int(axis.scale_base_delta),
            )

    yield from visit(0, 0, 0)


def _dma_request_blocks(transfer: Any, hbm_format: HbmFormat) -> tuple[list[int], list[int]]:
    dim = int(transfer.dim)
    amount = int(transfer.amount)
    element_len = hbm_format.element_bits * dim // 8
    scale_len = hbm_format.scale_bits * (dim // hbm_format.block) // 8 if hbm_format.is_mx else 0
    stride = int(transfer.stride) if int(transfer.rstride) == 1 else dim
    scale_stride = stride / hbm_format.block if hbm_format.is_mx else 0.0
    reads: list[int] = []
    writes: list[int] = []
    for index in range(amount):
        element_address = int(transfer.element_base) + index * stride
        scale_address = int(transfer.scale_base) + int(index * scale_stride)
        element_blocks = range(
            element_address // REQUEST_BYTES,
            (element_address + element_len - 1) // REQUEST_BYTES + 1,
        )
        if transfer.direction == "read":
            reads.extend(block * REQUEST_BYTES for block in element_blocks)
            if scale_len:
                reads.append(scale_address // REQUEST_BYTES * REQUEST_BYTES)
        elif transfer.direction == "write":
            touched = [block * REQUEST_BYTES for block in element_blocks]
            if scale_len:
                touched.extend(
                    block * REQUEST_BYTES
                    for block in range(
                        scale_address // REQUEST_BYTES,
                        (scale_address + scale_len - 1) // REQUEST_BYTES + 1,
                    )
                )
            reads.extend(touched)
            writes.extend(touched)
        else:
            raise ValueError(f"unsupported DMA direction {transfer.direction!r}")
    return reads, writes


def _compressed_transition_gaps(axes: Sequence[Any]) -> tuple[float, int]:
    instruction_count = math.prod(int(axis.count) for axis in axes) if axes else 1
    if instruction_count <= 1:
        return 0.0, 0
    weighted_gap = 0
    max_gap = 0
    inner_element_extent = 0
    inner_scale_extent = 0
    for axis_index in range(len(axes) - 1, -1, -1):
        axis = axes[axis_index]
        element_jump = int(axis.element_base_delta) - inner_element_extent
        scale_jump = int(axis.scale_base_delta) - inner_scale_extent
        gap = max(abs(element_jump), abs(scale_jump))
        prefix_count = math.prod(int(outer.count) for outer in axes[:axis_index])
        transitions = prefix_count * (int(axis.count) - 1)
        weighted_gap += gap * transitions
        max_gap = max(max_gap, gap)
        inner_element_extent += (int(axis.count) - 1) * int(axis.element_base_delta)
        inner_scale_extent += (int(axis.count) - 1) * int(axis.scale_base_delta)
    return weighted_gap / (instruction_count - 1), max_gap


def dma_stream_geometry(
    transfer: Any,
    axes: Sequence[Any],
    hbm_format: HbmFormat,
    *,
    previous_address: int | None = None,
    exact_enumeration_limit: int = 8,
) -> DmaStreamGeometry:
    """Summarize one ordered affine DMA stream without expanding large repeats."""
    instruction_count = math.prod(int(axis.count) for axis in axes) if axes else 1
    if instruction_count <= 0:
        raise ValueError("DMA stream repeat axes must have positive counts")
    geometry_classes = dma_geometry_classes(transfer, axes, hbm_format)
    if sum(count for _, count in geometry_classes) != instruction_count:
        raise ValueError("DMA geometry classes do not cover the complete stream")

    totals: Counter[str] = Counter()
    for geometry, count in geometry_classes:
        for name in (
            "read_requests",
            "write_requests",
            "read_bytes",
            "write_bytes",
            "payload_read_bytes",
            "payload_write_bytes",
        ):
            totals[name] += int(getattr(geometry, name)) * count

    final_element_base = int(transfer.element_base) + sum(
        (int(axis.count) - 1) * int(axis.element_base_delta) for axis in axes
    )
    final_scale_base = int(transfer.scale_base) + sum(
        (int(axis.count) - 1) * int(axis.scale_base_delta) for axis in axes
    )
    initial_gap = 0
    if previous_address is not None:
        initial_gap = min(
            abs(int(transfer.element_base) - previous_address),
            abs(int(transfer.scale_base) - previous_address),
        )

    if instruction_count <= exact_enumeration_limit:
        unique_blocks: set[int] = set()
        previous_bases: tuple[int, int] | None = None
        gaps = []
        for element_delta, scale_delta in _stream_offsets(axes):
            current = replace(
                transfer,
                element_base=int(transfer.element_base) + element_delta,
                scale_base=int(transfer.scale_base) + scale_delta,
            )
            reads, writes = _dma_request_blocks(current, hbm_format)
            unique_blocks.update(reads)
            unique_blocks.update(writes)
            current_bases = (int(current.element_base), int(current.scale_base))
            if previous_bases is not None:
                gaps.append(
                    max(
                        abs(current_bases[0] - previous_bases[0]),
                        abs(current_bases[1] - previous_bases[1]),
                    )
                )
            previous_bases = current_bases
        if unique_blocks:
            address_span = max(unique_blocks) + REQUEST_BYTES - min(unique_blocks)
        else:
            address_span = 0
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        max_gap = max(gaps, default=0)
        unique_block_count = len(unique_blocks)
    else:
        base_reads, base_writes = _dma_request_blocks(transfer, hbm_format)
        base_unique_blocks = len(set(base_reads) | set(base_writes))
        varying_instruction_count = math.prod(
            int(axis.count)
            for axis in axes
            if int(axis.element_base_delta) or int(axis.scale_base_delta)
        )
        varying_instruction_count = max(1, varying_instruction_count)
        unique_block_count = min(
            totals["read_requests"] + totals["write_requests"],
            base_unique_blocks * varying_instruction_count,
        )
        base_geometry = dma_geometry(transfer, hbm_format)
        address_span = max(
            final_element_base - int(transfer.element_base),
            final_scale_base - int(transfer.scale_base),
        ) + base_geometry.row_span_bytes
        mean_gap, max_gap = _compressed_transition_gaps(axes)

    base_geometry = dma_geometry(transfer, hbm_format)
    return DmaStreamGeometry(
        instruction_count=instruction_count,
        read_requests=totals["read_requests"],
        write_requests=totals["write_requests"],
        read_bytes=totals["read_bytes"],
        write_bytes=totals["write_bytes"],
        payload_read_bytes=totals["payload_read_bytes"],
        payload_write_bytes=totals["payload_write_bytes"],
        unique_blocks=unique_block_count,
        address_span_bytes=address_span,
        mean_adjacent_gap_bytes=mean_gap,
        max_adjacent_gap_bytes=max_gap,
        initial_address_gap_bytes=initial_gap,
        final_element_base=final_element_base,
        final_scale_base=final_scale_base,
        dim=int(transfer.dim),
        amount=int(transfer.amount),
        stride_bytes=base_geometry.stride_bytes,
        element_alignment=base_geometry.element_alignment,
        scale_alignment=base_geometry.scale_alignment,
        strided=base_geometry.strided,
        misaligned=base_geometry.misaligned,
    )


@dataclass(frozen=True)
class CalibrationSample:
    opcode: str
    geometry: DmaGeometry
    channels: int
    observed_ns: float
    source: str


@dataclass(frozen=True)
class StreamCalibrationSample:
    opcode: str
    geometry: DmaStreamGeometry
    channels: int
    observed_ns: float
    source: str


@dataclass(frozen=True)
class LocalLinearSurrogate:
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_rmse_ns: float
    sample_count: int

    def predict_ns(self, request_work: int) -> float:
        values = (1.0, math.sqrt(request_work), float(request_work))
        return max(0.0, sum(c * v / s for c, v, s in zip(self.coefficients, values, self.feature_scales)))


@dataclass(frozen=True)
class StreamLinearSurrogate:
    feature_names: tuple[str, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_rmse_ns: float
    sample_count: int

    def predict_ns(self, features: Mapping[str, float]) -> float:
        return max(
            0.0,
            sum(
                coefficient * features[name] / scale
                for name, scale, coefficient in zip(
                    self.feature_names,
                    self.feature_scales,
                    self.coefficients,
                    strict=True,
                )
            ),
        )


@dataclass(frozen=True)
class OpcodeSurrogate:
    feature_names: tuple[str, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_rmse_ns: float
    sample_count: int
    local_models: Mapping[str, LocalLinearSurrogate] = field(default_factory=dict)
    integration_bias_ns: float = 0.0
    integration_scale: float = 1.0

    def predict_ns(self, geometry: DmaGeometry, channels: int) -> float:
        local = self.local_models.get(geometry.calibration_cell(channels))
        if local is not None:
            base_value = local.predict_ns(geometry.request_work)
        else:
            features = geometry.features(channels)
            base_value = sum(
                coefficient * features[name] / scale
                for name, scale, coefficient in zip(
                    self.feature_names, self.feature_scales, self.coefficients, strict=True
                )
            )
        return max(0.0, self.integration_bias_ns + self.integration_scale * float(base_value))


@dataclass(frozen=True)
class StreamCurvePoint:
    cell: str
    amount: int
    instruction_count: int
    mean_adjacent_gap_bytes: float
    max_adjacent_gap_bytes: int
    reuse_ratio: float
    observed_ns: float


@dataclass(frozen=True)
class StreamOpcodeSurrogate:
    global_model: StreamLinearSurrogate
    local_models: Mapping[str, StreamLinearSurrogate] = field(default_factory=dict)
    curve_points: tuple[StreamCurvePoint, ...] = ()

    def predict_ns(self, geometry: DmaStreamGeometry, channels: int) -> float:
        points = [
            point
            for point in self.curve_points
            if point.cell == geometry.calibration_cell(channels)
            and point.amount == geometry.amount
        ]
        if points:
            by_count: dict[int, list[StreamCurvePoint]] = {}
            for point in points:
                by_count.setdefault(point.instruction_count, []).append(point)

            def transition_distance(point: StreamCurvePoint) -> float:
                if point.instruction_count == 1 or geometry.instruction_count == 1:
                    return 0.0
                return (
                    abs(
                        math.log1p(point.mean_adjacent_gap_bytes)
                        - math.log1p(geometry.mean_adjacent_gap_bytes)
                    )
                    + 0.25
                    * abs(
                        math.log1p(point.max_adjacent_gap_bytes)
                        - math.log1p(geometry.max_adjacent_gap_bytes)
                    )
                    + abs(point.reuse_ratio - geometry.reuse_ratio)
                )

            selected_with_distance = {
                count: min(
                    ((transition_distance(point), point) for point in values),
                    key=lambda value: value[0],
                )
                for count, values in by_count.items()
            }
            selected = {
                count: point
                for count, (distance, point) in selected_with_distance.items()
                if count == 1 or distance <= 1.0
            }
            if len(selected) < 2:
                selected = {
                    count: point
                    for count, (_, point) in sorted(
                        selected_with_distance.items(), key=lambda value: value[1][0]
                    )[:2]
                }
            curve = sorted(
                (count, point.observed_ns) for count, point in selected.items()
            )
            target = geometry.instruction_count
            if target in selected:
                return selected[target].observed_ns
            lower = [point for point in curve if point[0] < target]
            upper = [point for point in curve if point[0] > target]
            if lower and upper:
                left = lower[-1]
                right = upper[0]
            elif len(lower) >= 2:
                left, right = lower[-2:]
            elif len(upper) >= 2:
                left, right = upper[:2]
            else:
                left = right = curve[0]
            if left[0] == right[0]:
                return max(0.0, left[1] * target / left[0])
            slope = (right[1] - left[1]) / (right[0] - left[0])
            return max(0.0, left[1] + (target - left[0]) * slope)
        features = geometry.features(channels)
        model = self.local_models.get(geometry.calibration_cell(channels), self.global_model)
        return model.predict_ns(features)


def _stream_model_to_dict(model: StreamLinearSurrogate) -> dict[str, Any]:
    return {
        "feature_names": list(model.feature_names),
        "feature_scales": list(model.feature_scales),
        "coefficients": list(model.coefficients),
        "training_rmse_ns": model.training_rmse_ns,
        "sample_count": model.sample_count,
    }


def _stream_model_from_dict(data: Mapping[str, Any]) -> StreamLinearSurrogate:
    return StreamLinearSurrogate(
        feature_names=tuple(data["feature_names"]),
        feature_scales=tuple(float(value) for value in data["feature_scales"]),
        coefficients=tuple(float(value) for value in data["coefficients"]),
        training_rmse_ns=float(data["training_rmse_ns"]),
        sample_count=int(data["sample_count"]),
    )


@dataclass(frozen=True)
class HbmCalibration:
    calibration_id: str
    models: Mapping[str, OpcodeSurrogate]
    compatibility: Mapping[str, Any]
    metadata: Mapping[str, Any]
    stream_models: Mapping[str, StreamOpcodeSurrogate] = field(default_factory=dict)

    def predict_ns(self, opcode: str, geometry: DmaGeometry, channels: int) -> float:
        try:
            model = self.models[opcode]
        except KeyError as exc:
            raise ValueError(f"HBM calibration {self.calibration_id!r} has no model for {opcode}") from exc
        return model.predict_ns(geometry, channels)

    def predict_stream_ns(
        self,
        opcode: str,
        geometry: DmaStreamGeometry,
        channels: int,
    ) -> float:
        try:
            model = self.stream_models[opcode]
        except KeyError as exc:
            raise ValueError(
                f"HBM calibration {self.calibration_id!r} has no stream model for {opcode}"
            ) from exc
        return model.predict_ns(geometry, channels)

    def assert_compatible(
        self,
        *,
        channels: int,
        format_signatures: Iterable[str],
        dma_semantics_hash: str | None = None,
        trace_schema_version: int | None = None,
    ) -> None:
        errors = []
        compatible_channels = {int(value) for value in self.compatibility.get("channels", [])}
        if compatible_channels and channels not in compatible_channels:
            errors.append(f"channels={channels} not in calibrated set {sorted(compatible_channels)}")
        expected_request_bytes = int(self.compatibility.get("request_bytes", REQUEST_BYTES))
        if expected_request_bytes != REQUEST_BYTES:
            errors.append(f"request bytes {expected_request_bytes} != {REQUEST_BYTES}")
        calibrated_formats = set(self.compatibility.get("precision_signatures", []))
        unknown_formats = set(format_signatures) - calibrated_formats
        if calibrated_formats and unknown_formats:
            errors.append(f"uncalibrated precision formats {sorted(unknown_formats)}")
        calibrated_hash = self.compatibility.get("dma_semantics_hash")
        if dma_semantics_hash and calibrated_hash not in {None, "unknown", dma_semantics_hash}:
            errors.append("transactional DMA semantics hash differs from calibration")
        calibrated_trace_schema = self.compatibility.get("trace_schema_version")
        if (
            trace_schema_version is not None
            and calibrated_trace_schema is not None
            and int(calibrated_trace_schema) != trace_schema_version
        ):
            errors.append(
                f"trace schema {trace_schema_version} != calibrated schema "
                f"{calibrated_trace_schema}"
            )
        if errors:
            raise ValueError(
                f"HBM calibration {self.calibration_id!r} is incompatible: " + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_id": self.calibration_id,
            "compatibility": dict(self.compatibility),
            "metadata": dict(self.metadata),
            "models": {
                opcode: {
                    "feature_names": list(model.feature_names),
                    "feature_scales": list(model.feature_scales),
                    "coefficients": list(model.coefficients),
                    "training_rmse_ns": model.training_rmse_ns,
                    "sample_count": model.sample_count,
                    "integration_bias_ns": model.integration_bias_ns,
                    "integration_scale": model.integration_scale,
                    "local_models": {
                        cell: {
                            "feature_scales": list(local.feature_scales),
                            "coefficients": list(local.coefficients),
                            "training_rmse_ns": local.training_rmse_ns,
                            "sample_count": local.sample_count,
                        }
                        for cell, local in sorted(model.local_models.items())
                    },
                }
                for opcode, model in sorted(self.models.items())
            },
            "stream_models": {
                opcode: {
                    "global_model": _stream_model_to_dict(model.global_model),
                    "local_models": {
                        cell: _stream_model_to_dict(local)
                        for cell, local in sorted(model.local_models.items())
                    },
                    "curve_points": [asdict(point) for point in model.curve_points],
                }
                for opcode, model in sorted(self.stream_models.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HbmCalibration:
        schema_version = int(data.get("schema_version", -1))
        if schema_version not in {1, CALIBRATION_SCHEMA_VERSION}:
            raise ValueError(f"unsupported HBM calibration schema {data.get('schema_version')!r}")
        models = {
            opcode: OpcodeSurrogate(
                feature_names=tuple(model["feature_names"]),
                feature_scales=tuple(float(value) for value in model["feature_scales"]),
                coefficients=tuple(float(value) for value in model["coefficients"]),
                training_rmse_ns=float(model["training_rmse_ns"]),
                sample_count=int(model["sample_count"]),
                integration_bias_ns=float(model.get("integration_bias_ns", 0.0)),
                integration_scale=float(model.get("integration_scale", 1.0)),
                local_models={
                    cell: LocalLinearSurrogate(
                        feature_scales=tuple(float(value) for value in local["feature_scales"]),
                        coefficients=tuple(float(value) for value in local["coefficients"]),
                        training_rmse_ns=float(local["training_rmse_ns"]),
                        sample_count=int(local["sample_count"]),
                    )
                    for cell, local in model.get("local_models", {}).items()
                },
            )
            for opcode, model in data.get("models", {}).items()
        }
        stream_models = {
            opcode: StreamOpcodeSurrogate(
                global_model=_stream_model_from_dict(model["global_model"]),
                local_models={
                    cell: _stream_model_from_dict(local)
                    for cell, local in model.get("local_models", {}).items()
                },
                curve_points=tuple(
                    StreamCurvePoint(
                        cell=str(point["cell"]),
                        amount=int(point["amount"]),
                        instruction_count=int(point["instruction_count"]),
                        mean_adjacent_gap_bytes=float(
                            point.get("mean_adjacent_gap_bytes", 0.0)
                        ),
                        max_adjacent_gap_bytes=int(
                            point.get("max_adjacent_gap_bytes", 0)
                        ),
                        reuse_ratio=float(point.get("reuse_ratio", 0.0)),
                        observed_ns=float(point["observed_ns"]),
                    )
                    for point in model.get("curve_points", ())
                ),
            )
            for opcode, model in data.get("stream_models", {}).items()
        }
        return cls(
            calibration_id=str(data["calibration_id"]),
            models=models,
            compatibility=dict(data["compatibility"]),
            metadata=dict(data.get("metadata", {})),
            stream_models=stream_models,
        )

    @classmethod
    def load(cls, path: str | Path) -> HbmCalibration:
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _fit_nonnegative_ridge(matrix: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    """Solve NNLS ridge with projected FISTA for the sparse feature basis."""
    feature_count = matrix.shape[1]
    if not feature_count:
        return np.zeros(0, dtype=float)
    vector = np.full(feature_count, 1.0 / math.sqrt(feature_count))
    for _ in range(32):
        candidate = matrix.T @ (matrix @ vector) + ridge * vector
        norm = np.linalg.norm(candidate)
        if norm == 0:
            return np.zeros(feature_count, dtype=float)
        vector = candidate / norm
    lipschitz = float(vector @ (matrix.T @ (matrix @ vector)) + ridge)
    coefficients = np.zeros(feature_count, dtype=float)
    accelerated = coefficients.copy()
    momentum = 1.0
    for _ in range(20_000):
        gradient = matrix.T @ (matrix @ accelerated - targets) + ridge * accelerated
        updated = np.maximum(0.0, accelerated - gradient / lipschitz)
        if np.linalg.norm(updated - coefficients) <= 1e-9 * (1.0 + np.linalg.norm(coefficients)):
            coefficients = updated
            break
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        coefficients = updated
        momentum = next_momentum
    return coefficients


def fit_hbm_calibration(
    samples: Sequence[CalibrationSample],
    *,
    ridge: float = 1e-8,
    compatibility: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    store_samples: bool = True,
) -> HbmCalibration:
    grouped: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.opcode, []).append(sample)
    required = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}
    if missing := required - grouped.keys():
        raise ValueError(f"calibration samples are missing opcodes: {sorted(missing)}")

    models: dict[str, OpcodeSurrogate] = {}
    for opcode, opcode_samples in sorted(grouped.items()):
        rows = [sample.geometry.features(sample.channels) for sample in opcode_samples]
        matrix = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
        targets = np.asarray([sample.observed_ns for sample in opcode_samples], dtype=float)
        scales = np.sqrt(np.mean(matrix * matrix, axis=0))
        scales[scales == 0] = 1.0
        normalized = matrix / scales
        coefficients = _fit_nonnegative_ridge(normalized, targets, ridge)
        predictions = normalized @ coefficients
        rmse = math.sqrt(float(np.mean((predictions - targets) ** 2)))
        local_samples: dict[str, list[CalibrationSample]] = {}
        for sample in opcode_samples:
            local_samples.setdefault(sample.geometry.calibration_cell(sample.channels), []).append(sample)
        local_models = {}
        for cell, cell_samples in local_samples.items():
            distinct_work = {sample.geometry.request_work for sample in cell_samples}
            use_prefetch_curve = opcode != "H_STORE_V" and len(distinct_work) > 1
            local_matrix = np.asarray(
                [
                    [
                        float(opcode != "H_STORE_V"),
                        math.sqrt(sample.geometry.request_work) if use_prefetch_curve else 0.0,
                        sample.geometry.request_work
                        if use_prefetch_curve or opcode == "H_STORE_V"
                        else 0.0,
                    ]
                    for sample in cell_samples
                ],
                dtype=float,
            )
            local_targets = np.asarray([sample.observed_ns for sample in cell_samples], dtype=float)
            local_scales = np.sqrt(np.mean(local_matrix * local_matrix, axis=0))
            local_scales[local_scales == 0] = 1.0
            local_normalized = local_matrix / local_scales
            local_coefficients = _fit_nonnegative_ridge(local_normalized, local_targets, ridge)
            local_predictions = local_normalized @ local_coefficients
            local_models[cell] = LocalLinearSurrogate(
                feature_scales=tuple(float(value) for value in local_scales),
                coefficients=tuple(float(value) for value in local_coefficients),
                training_rmse_ns=math.sqrt(float(np.mean((local_predictions - local_targets) ** 2))),
                sample_count=len(cell_samples),
            )
        models[opcode] = OpcodeSurrogate(
            feature_names=FEATURE_NAMES,
            feature_scales=tuple(float(value) for value in scales),
            coefficients=tuple(float(value) for value in coefficients),
            training_rmse_ns=rmse,
            sample_count=len(opcode_samples),
            local_models=local_models,
        )

    identity = {
        "samples": [
            {
                "opcode": sample.opcode,
                "channels": sample.channels,
                "observed_ns": sample.observed_ns,
                "source": sample.source,
                "geometry": asdict(sample.geometry),
            }
            for sample in samples
        ],
        "ridge": ridge,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    artifact_metadata = {**dict(metadata or {}), "ridge": ridge}
    if store_samples:
        artifact_metadata["samples"] = identity["samples"]
    else:
        artifact_metadata["sample_count"] = len(samples)
        artifact_metadata["sample_manifest_sha256"] = hashlib.sha256(
            json.dumps(identity["samples"], sort_keys=True).encode()
        ).hexdigest()
    return HbmCalibration(
        calibration_id=f"hbm-surrogate-{digest}",
        models=models,
        compatibility=dict(compatibility or {}),
        metadata=artifact_metadata,
    )


def _fit_stream_linear_model(
    samples: Sequence[StreamCalibrationSample],
    feature_names: Sequence[str],
    ridge: float,
) -> StreamLinearSurrogate:
    rows = [sample.geometry.features(sample.channels) for sample in samples]
    matrix = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    targets = np.asarray([sample.observed_ns for sample in samples], dtype=float)
    scales = np.sqrt(np.mean(matrix * matrix, axis=0))
    scales[scales == 0] = 1.0
    normalized = matrix / scales
    # Stream latencies span four orders of magnitude. Relative weighting keeps
    # short startup-dominated streams from being ignored by long DMA samples.
    weights = 1.0 / np.maximum(targets, 1.0)
    coefficients = _fit_nonnegative_ridge(
        normalized * weights[:, None],
        targets * weights,
        ridge,
    )
    predictions = normalized @ coefficients
    return StreamLinearSurrogate(
        feature_names=tuple(feature_names),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        training_rmse_ns=math.sqrt(float(np.mean((predictions - targets) ** 2))),
        sample_count=len(samples),
    )


def fit_hbm_stream_calibration(
    samples: Sequence[StreamCalibrationSample],
    *,
    ridge: float = 1e-8,
    compatibility: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    store_samples: bool = False,
) -> HbmCalibration:
    grouped: dict[str, list[StreamCalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.opcode, []).append(sample)
    required = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}
    if missing := required - grouped.keys():
        raise ValueError(f"stream calibration samples are missing opcodes: {sorted(missing)}")

    stream_models = {}
    for opcode, opcode_samples in sorted(grouped.items()):
        global_model = _fit_stream_linear_model(opcode_samples, STREAM_FEATURE_NAMES, ridge)
        by_cell: dict[str, list[StreamCalibrationSample]] = {}
        for sample in opcode_samples:
            by_cell.setdefault(
                sample.geometry.calibration_cell(sample.channels), []
            ).append(sample)
        local_models = {
            cell: _fit_stream_linear_model(cell_samples, LOCAL_STREAM_FEATURE_NAMES, ridge)
            for cell, cell_samples in by_cell.items()
            if len(cell_samples) >= 3
            and len({sample.geometry.instruction_count for sample in cell_samples}) >= 2
        }
        stream_models[opcode] = StreamOpcodeSurrogate(
            global_model=global_model,
            local_models=local_models,
            curve_points=tuple(
                StreamCurvePoint(
                    cell=sample.geometry.calibration_cell(sample.channels),
                    amount=sample.geometry.amount,
                    instruction_count=sample.geometry.instruction_count,
                    mean_adjacent_gap_bytes=sample.geometry.mean_adjacent_gap_bytes,
                    max_adjacent_gap_bytes=sample.geometry.max_adjacent_gap_bytes,
                    reuse_ratio=sample.geometry.reuse_ratio,
                    observed_ns=sample.observed_ns,
                )
                for sample in opcode_samples
            ),
        )

    identity_samples = [
        {
            "opcode": sample.opcode,
            "channels": sample.channels,
            "observed_ns": sample.observed_ns,
            "source": sample.source,
            "geometry": asdict(sample.geometry),
        }
        for sample in samples
    ]
    identity = {"samples": identity_samples, "ridge": ridge, "kind": "ordered_dma_stream"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    artifact_metadata = {**dict(metadata or {}), "ridge": ridge}
    if store_samples:
        artifact_metadata["samples"] = identity_samples
    else:
        artifact_metadata["sample_count"] = len(samples)
        artifact_metadata["sample_manifest_sha256"] = hashlib.sha256(
            json.dumps(identity_samples, sort_keys=True).encode()
        ).hexdigest()
    return HbmCalibration(
        calibration_id=f"hbm-transactional-stream-surrogate-{digest}",
        models={},
        compatibility=dict(compatibility or {}),
        metadata=artifact_metadata,
        stream_models=stream_models,
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "REQUEST_BYTES",
    "CalibrationSample",
    "DmaGeometry",
    "DmaStreamGeometry",
    "HbmCalibration",
    "HbmFormat",
    "OpcodeSurrogate",
    "StreamCalibrationSample",
    "StreamCurvePoint",
    "StreamOpcodeSurrogate",
    "dma_geometry",
    "dma_geometry_classes",
    "dma_stream_geometry",
    "file_sha256",
    "fit_hbm_calibration",
    "fit_hbm_stream_calibration",
    "hbm_formats_from_settings",
    "load_transactional_toml",
    "parse_hbm_format",
]
