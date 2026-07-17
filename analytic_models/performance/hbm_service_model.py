"""Precision-independent physical HBM work and global service model.

V3 deliberately separates the compiler's logical DMA trace from a cost-only
physical layout.  Normal compilation keeps its existing MXFP8 byte layout;
this module repacks logical objects for the requested DSE precision and never
feeds those addresses back to the compiler.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from compiler.aten.cost_emitter import CostTrace, MemoryEvent


REQUEST_BYTES = 64
TRACE_SCHEMA_VERSION = 3
SERVICE_MODEL_SCHEMA_VERSION = 3
MAX_SAMPLED_REQUESTS = 2048
DEFAULT_SAMPLED_REQUESTS = 256
SUPPORTED_MX_WIDTHS = {4, 8}
_PHYSICAL_WORK_CACHE_LIMIT = 256
_PHYSICAL_WORK_CACHE: OrderedDict[tuple[Any, ...], PhysicalMemoryWork] = OrderedDict()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_bits_to_bytes(bits: int) -> int:
    return (bits + 7) // 8


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memory_events_cache_digest(events: Sequence[MemoryEvent]) -> str:
    """Return a stable identity for the event subset used to build HBM work.

    ``config_hash`` identifies the compiler/hardware configuration, but a
    caller may intentionally evaluate only part of that configuration's cost
    trace.  The scheduled-shadow V3 provider does this once per DMA stream.
    Including the actual compressed events prevents a full trace and its
    single-event subsets from aliasing in ``_PHYSICAL_WORK_CACHE``.
    """

    payload = json.dumps(
        [event.to_dict() for event in events],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MemoryFormat:
    family: str
    element_bits: int
    scale_bits: int = 0
    block: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        family = self.family.lower()
        object.__setattr__(self, "family", family)
        if self.element_bits == 3 or (family == "mxint" and self.element_bits == 3):
            raise ValueError("MXINT3 is unsupported by Compiler Cost Memory V3")
        if family in {"mxint", "mxfp"}:
            if self.element_bits not in SUPPORTED_MX_WIDTHS:
                raise ValueError(
                    f"V3 supports only 4/8-bit MX formats, got {family}{self.element_bits}"
                )
            if self.scale_bits <= 0 or self.block <= 0:
                raise ValueError("MX formats require positive scale_bits and block")
        elif family == "plain":
            if self.element_bits <= 0:
                raise ValueError("plain format element_bits must be positive")
            if self.scale_bits or self.block != 1:
                raise ValueError("plain formats cannot have an MX scale stream")
        else:
            raise ValueError(f"unsupported memory format family {self.family!r}")

    @property
    def is_mx(self) -> bool:
        return self.family in {"mxint", "mxfp"}

    def request_signature(self) -> str:
        if self.is_mx:
            return f"mx:e{self.element_bits}:s{self.scale_bits}:b{self.block}"
        return f"plain:e{self.element_bits}"

    @classmethod
    def parse(
        cls,
        value: MemoryFormat | str | Mapping[str, Any],
        *,
        default_block: int = 64,
        default_scale_bits: int = 8,
    ) -> MemoryFormat:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.upper().replace("_", "")
            if text.startswith("MXINT"):
                width = int(text.removeprefix("MXINT"))
                return cls("mxint", width, default_scale_bits, default_block, text)
            if text in {"MXFP4", "MXFP8"}:
                width = int(text.removeprefix("MXFP"))
                return cls("mxfp", width, default_scale_bits, default_block, text)
            if text.startswith("MXFPE") and "M" in text:
                exponent, mantissa = text.removeprefix("MXFPE").split("M", 1)
                width = 1 + int(exponent) + int(mantissa)
                return cls("mxfp", width, default_scale_bits, default_block, text)
            if text.startswith("INT"):
                return cls("plain", int(text.removeprefix("INT")), name=text)
            raise ValueError(f"unsupported memory precision {value!r}")
        if not isinstance(value, Mapping):
            raise TypeError(f"memory format must be a string or mapping, got {type(value).__name__}")
        family = str(value.get("family", value.get("kind", value.get("type", "")))).lower()
        family = family.replace("_", "")
        if family.startswith("mxint"):
            suffix = family.removeprefix("mxint")
            width = int(suffix or value.get("width", value.get("bits")))
            family = "mxint"
        elif family.startswith("mxfp"):
            suffix = family.removeprefix("mxfp")
            if suffix.isdigit():
                width = int(suffix)
            elif "width" in value or "bits" in value:
                width = int(value.get("width", value.get("bits")))
            else:
                width = 1 + int(value["exp"]) + int(value["mant"])
            family = "mxfp"
        elif family in {"plain", "int", "integer"}:
            family = "plain"
            width = int(value.get("width", value.get("bits")))
        else:
            raise ValueError(f"unsupported memory format mapping {value!r}")
        return cls(
            family=family,
            element_bits=width,
            scale_bits=(
                0
                if family == "plain"
                else int(value.get("scale_bits", value.get("scale_width", default_scale_bits)))
            ),
            block=1 if family == "plain" else int(value.get("block", default_block)),
            name=str(value.get("name", "")),
        )


@dataclass(frozen=True)
class MemoryPrecisionConfig:
    weight: MemoryFormat
    activation: MemoryFormat
    matrix_kv: MemoryFormat
    vector_kv: MemoryFormat
    integer: MemoryFormat

    @classmethod
    def active_mxint4(
        cls,
        *,
        block: int = 64,
        scale_bits: int = 8,
        integer_bits: int = 32,
    ) -> MemoryPrecisionConfig:
        mxint4 = MemoryFormat("mxint", 4, scale_bits, block, "MXINT4")
        return cls(
            weight=mxint4,
            activation=mxint4,
            matrix_kv=mxint4,
            vector_kv=mxint4,
            integer=MemoryFormat("plain", integer_bits, name=f"INT{integer_bits}"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MemoryPrecisionConfig:
        block = int(value.get("block", value.get("mx_block", 64)))
        scale_bits = int(value.get("scale_bits", value.get("scale_width", 8)))
        integer_bits = int(value.get("integer_bits", value.get("int_data_width", 32)))
        if "ACT_WIDTH" in value or "KV_WIDTH" in value:
            weight_value = value.get("weight", value.get("weight_precision", "MXINT4"))
            activation_value = value.get("ACT_WIDTH", "MXINT4")
            kv_value = value.get("KV_WIDTH", activation_value)
            return cls(
                weight=MemoryFormat.parse(
                    weight_value, default_block=block, default_scale_bits=scale_bits
                ),
                activation=MemoryFormat.parse(
                    activation_value, default_block=block, default_scale_bits=scale_bits
                ),
                matrix_kv=MemoryFormat.parse(
                    kv_value, default_block=block, default_scale_bits=scale_bits
                ),
                vector_kv=MemoryFormat.parse(
                    kv_value, default_block=block, default_scale_bits=scale_bits
                ),
                integer=MemoryFormat("plain", integer_bits, name=f"INT{integer_bits}"),
            )
        defaults: dict[str, Any] = {
            "weight": "MXINT4",
            "activation": "MXINT4",
            "matrix_kv": value.get("kv", "MXINT4"),
            "vector_kv": value.get("kv", "MXINT4"),
            "integer": f"INT{integer_bits}",
        }
        return cls(
            **{
                role: MemoryFormat.parse(
                    value.get(role, default),
                    default_block=block,
                    default_scale_bits=scale_bits,
                )
                for role, default in defaults.items()
            }
        )

    def for_role(self, role: str, opcode: str | None = None) -> MemoryFormat:
        aliases = {
            "matrix": "weight",
            "kv": "matrix_kv" if opcode == "H_PREFETCH_M" else "vector_kv",
            "key_value": "matrix_kv" if opcode == "H_PREFETCH_M" else "vector_kv",
        }
        normalized = aliases.get(role, role)
        try:
            return getattr(self, normalized)
        except AttributeError as exc:
            raise ValueError(f"unknown memory precision role {role!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {role: asdict(getattr(self, role)) for role in self.__dataclass_fields__}


@dataclass(frozen=True)
class HbmConfig:
    channels: int
    request_bytes: int = REQUEST_BYTES
    channel_bandwidth_bytes_per_ns: float = 16.0
    mapper: str = "MOP4CLXOR"
    preset: str = "HBM2_2Gbps"

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.channels & (self.channels - 1):
            raise ValueError(f"MOP4CLXOR requires a power-of-two channel count, got {self.channels}")
        if self.request_bytes != REQUEST_BYTES:
            raise ValueError(f"V3 request geometry requires 64-byte requests, got {self.request_bytes}")
        if self.channel_bandwidth_bytes_per_ns <= 0:
            raise ValueError("channel bandwidth must be positive")
        if self.mapper != "MOP4CLXOR":
            raise ValueError(f"unsupported HBM mapper {self.mapper!r}")


@dataclass(frozen=True)
class PhysicalRepeatAxis:
    name: str
    count: int
    element_byte_delta: int
    scale_byte_delta: int


@dataclass(frozen=True)
class PhysicalDmaStream:
    stage: str
    opcode: str
    direction: str
    precision_role: str
    format_signature: str
    element_base: int
    scale_base: int
    dim: int
    amount: int
    stride_bytes: int
    rstride: int
    write_amount: int
    axes: tuple[PhysicalRepeatAxis, ...]
    multiplicity: int
    stream_index: int
    source: str

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.opcode,
            self.precision_role,
            self.format_signature,
            self.dim,
            self.amount,
            self.stride_bytes,
            self.element_base % REQUEST_BYTES,
            self.scale_base % REQUEST_BYTES,
            tuple(
                (
                    axis.count,
                    axis.element_byte_delta % REQUEST_BYTES,
                    axis.scale_byte_delta % REQUEST_BYTES,
                )
                for axis in self.axes
            ),
        )


@dataclass(frozen=True)
class MapperStats:
    sampled_requests: int
    active_channels: int
    mean_channel_load: float
    max_channel_load: int
    channel_imbalance: float
    row_hits: int
    row_misses: int
    row_conflicts: int
    address_span_bytes: int
    unique_blocks: int
    reuse_ratio: float


@dataclass(frozen=True)
class PhysicalGeometryWork:
    signature: str
    stage: str
    opcode: str
    precision_role: str
    format_signature: str
    dim: int
    amount: int
    stride_bytes: int
    dma_count: int
    read_requests: int
    write_requests: int
    rmw_requests: int
    physical_read_bytes: int
    physical_write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    active_channels: int
    max_channel_load: float
    mean_channel_load: float
    channel_imbalance: float
    row_hit_rate: float
    row_miss_rate: float
    row_conflict_rate: float
    address_span_bytes: int
    reuse_ratio: float
    queue_depth: float
    sampled_requests: int
    stream_count: int = 1

    @property
    def request_work(self) -> int:
        return self.read_requests + self.write_requests

    def features(self, channels: int) -> dict[str, float]:
        request_work = float(self.request_work)
        channel_tail = request_work * self.max_channel_load
        normalized_queue_depth = self.queue_depth / channels
        row_miss_depth = max(0.0, normalized_queue_depth * self.row_miss_rate)
        row_conflict_depth = max(0.0, normalized_queue_depth * self.row_conflict_rate)
        read_scale_requests = (
            float(self.dma_count * self.amount)
            if self.write_requests == 0 and self.format_signature.startswith("mx:")
            else 0.0
        )
        scale_stride_bytes = 0.0
        element_bits = 8
        if self.format_signature.startswith("mx:"):
            fields = {
                item[0]: int(item[1:])
                for item in self.format_signature.removeprefix("mx:").split(":")
            }
            element_bits = fields["e"]
            logical_stride = self.stride_bytes * 8 / fields["e"]
            scale_stride_bytes = logical_stride * fields["s"] / fields["b"] / 8
        elif self.format_signature.startswith("plain:e"):
            element_bits = int(self.format_signature.removeprefix("plain:e"))
        read_only_requests = self.read_requests if self.write_requests == 0 else 0
        read_requests_per_dma = read_only_requests / max(1, self.dma_count)
        row_element_requests = math.ceil(self.dim * element_bits / 8 / REQUEST_BYTES)
        cold_dma_count = min(self.dma_count, self.stream_count)
        marginal_dma_count = max(0, self.dma_count - cold_dma_count)
        return {
            "startup_prefetch_m": float(cold_dma_count if self.opcode == "H_PREFETCH_M" else 0),
            "startup_prefetch_v": float(cold_dma_count if self.opcode == "H_PREFETCH_V" else 0),
            "startup_store_v": float(cold_dma_count if self.opcode == "H_STORE_V" else 0),
            "marginal_prefetch_m": float(
                marginal_dma_count if self.opcode == "H_PREFETCH_M" else 0
            ),
            "marginal_prefetch_v": float(
                marginal_dma_count if self.opcode == "H_PREFETCH_V" else 0
            ),
            "marginal_store_v": float(
                marginal_dma_count if self.opcode == "H_STORE_V" else 0
            ),
            "read_requests": float(read_only_requests) * (1.0 - self.reuse_ratio),
            "read_scale_requests": read_scale_requests
            * (1.0 - self.reuse_ratio)
            / channels,
            "read_scale_reuse_requests": read_scale_requests
            * self.reuse_ratio
            / channels,
            "read_scale_reuse_pressure": read_scale_requests
            / max(1.0, scale_stride_bytes),
            "read_reuse_requests": float(read_only_requests) * self.reuse_ratio,
            "read_sqrt_requests": float(self.dma_count) * math.sqrt(read_requests_per_dma),
            "read_sqrt_channel_requests": float(self.dma_count)
            * math.sqrt(read_requests_per_dma / channels),
            "read_row_width": float(self.dma_count * row_element_requests)
            if read_only_requests
            else 0.0,
            "inverse_channel_dma": float(self.dma_count) / channels
            if read_only_requests
            else 0.0,
            "write_requests": float(self.write_requests),
            "rmw_requests": float(self.rmw_requests),
            "channel_tail_requests": channel_tail * (1.0 - self.reuse_ratio),
            "reuse_channel_tail_requests": channel_tail * self.reuse_ratio,
            "row_miss_requests": float(self.dma_count) * math.sqrt(row_miss_depth),
            "row_conflict_requests": float(self.dma_count) * row_conflict_depth,
            "queue_depth_work": float(self.dma_count) * math.sqrt(normalized_queue_depth),
        }


@dataclass(frozen=True)
class PhysicalMemoryWork:
    geometries: tuple[PhysicalGeometryWork, ...]
    precision_config: MemoryPrecisionConfig
    hbm_config: HbmConfig
    logical_object_count: int

    @property
    def read_requests(self) -> int:
        return sum(item.read_requests for item in self.geometries)

    @property
    def write_requests(self) -> int:
        return sum(item.write_requests for item in self.geometries)

    @property
    def read_bytes(self) -> int:
        return sum(item.physical_read_bytes for item in self.geometries)

    @property
    def write_bytes(self) -> int:
        return sum(item.physical_write_bytes for item in self.geometries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_MODEL_SCHEMA_VERSION,
            "precision_config": self.precision_config.to_dict(),
            "hbm_config": asdict(self.hbm_config),
            "logical_object_count": self.logical_object_count,
            "read_requests": self.read_requests,
            "write_requests": self.write_requests,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "geometries": [asdict(item) for item in self.geometries],
        }


@dataclass
class _LogicalObject:
    name: str
    role: str
    first_index: int
    explicit_elements: int | None = None
    minimum_offset: int | None = None
    required_end: int = 0
    origin: int = 0
    base: int = 0
    element_bytes: int = 0
    scale_base: int = 0
    instance_stride: int = 0
    layer_count: int = 1


def _axis_logical_delta(axis: Any) -> int:
    value = getattr(axis, "logical_element_delta", None)
    return int(axis.element_base_delta if value is None else value)


def _event_role(event: MemoryEvent) -> str:
    return str(event.transfer.precision_role or event.transfer.precision)


def _event_object(event: MemoryEvent) -> str:
    fallback = f"stream:{event.stream_index}:{_event_role(event)}:{event.transfer.opcode}"
    return str(event.transfer.memory_object or event.transfer.source or fallback)


def _event_logical_offset(event: MemoryEvent) -> int:
    value = event.transfer.logical_element_offset
    return int(event.transfer.element_base if value is None else value)


def _logical_to_byte_offset(elements: int, element_bits: int, *, label: str) -> int:
    bits = elements * element_bits
    if bits % 8:
        raise ValueError(f"{label} is not byte aligned after {element_bits}-bit packing: {elements}")
    return bits // 8


def _build_object_layout(
    events: Sequence[MemoryEvent],
    precision: MemoryPrecisionConfig,
) -> dict[str, _LogicalObject]:
    objects: dict[str, _LogicalObject] = {}
    for index, event in enumerate(events):
        name = _event_object(event)
        role = _event_role(event)
        obj = objects.setdefault(name, _LogicalObject(name=name, role=role, first_index=index))
        if obj.role != role:
            existing_format = precision.for_role(obj.role, event.transfer.opcode)
            new_format = precision.for_role(role, event.transfer.opcode)
            if existing_format.request_signature() != new_format.request_signature():
                raise ValueError(
                    f"logical memory object {name!r} is written/read with incompatible "
                    f"precision roles {obj.role!r}/{role!r}"
                )
        explicit = event.transfer.logical_object_elements
        if explicit is not None:
            explicit = int(explicit)
            if explicit <= 0:
                raise ValueError(f"logical object {name!r} has invalid size {explicit}")
            obj.explicit_elements = max(obj.explicit_elements or 0, explicit)
        offset = _event_logical_offset(event)
        obj.minimum_offset = offset if obj.minimum_offset is None else min(obj.minimum_offset, offset)
        stride = int(event.transfer.logical_stride or event.transfer.stride)
        row_stride = stride if int(event.transfer.rstride) == 1 else int(event.transfer.dim)
        instruction_end = offset + (int(event.transfer.amount) - 1) * row_stride + int(
            event.transfer.dim
        )
        for axis in event.enclosing_axes:
            if axis.name == "decoder_layer":
                obj.layer_count = max(obj.layer_count, int(axis.count))
        axis_end = sum(
            (int(axis.count) - 1) * _axis_logical_delta(axis)
            for axis in event.enclosing_axes
            if axis.name != "decoder_layer"
        )
        obj.required_end = max(obj.required_end, instruction_end + axis_end)

    cursor = 0
    for obj in sorted(objects.values(), key=lambda item: item.first_index):
        obj.origin = 0 if obj.explicit_elements is not None else int(obj.minimum_offset or 0)
        extent = max(obj.explicit_elements or 0, obj.required_end - obj.origin)
        fmt = precision.for_role(obj.role)
        obj.base = _align_up(cursor, REQUEST_BYTES)
        obj.element_bytes = _ceil_bits_to_bytes(extent * fmt.element_bits)
        obj.scale_base = obj.base + obj.element_bytes
        scale_bytes = (
            _ceil_bits_to_bytes(math.ceil(extent / fmt.block) * fmt.scale_bits)
            if fmt.is_mx
            else 0
        )
        obj.instance_stride = _align_up(obj.element_bytes + scale_bytes, REQUEST_BYTES)
        cursor = obj.base + obj.instance_stride * obj.layer_count
    return objects


def _physical_stream(
    event: MemoryEvent,
    obj: _LogicalObject,
    precision: MemoryPrecisionConfig,
) -> PhysicalDmaStream:
    transfer = event.transfer
    role = _event_role(event)
    fmt = precision.for_role(role, transfer.opcode)
    logical_offset = _event_logical_offset(event) - obj.origin
    logical_stride = int(transfer.logical_stride or transfer.stride)
    element_offset = _logical_to_byte_offset(
        logical_offset, fmt.element_bits, label=f"{obj.name} element offset"
    )
    stride_bytes = _logical_to_byte_offset(
        logical_stride, fmt.element_bits, label=f"{obj.name} stride"
    )
    scale_offset = (
        _ceil_bits_to_bytes((logical_offset // fmt.block) * fmt.scale_bits) if fmt.is_mx else 0
    )
    axes = []
    for axis in event.enclosing_axes:
        if axis.name == "decoder_layer":
            axes.append(
                PhysicalRepeatAxis(
                    name=axis.name,
                    count=int(axis.count),
                    element_byte_delta=obj.instance_stride,
                    scale_byte_delta=obj.instance_stride,
                )
            )
            continue
        logical_delta = _axis_logical_delta(axis)
        axes.append(
            PhysicalRepeatAxis(
                name=axis.name,
                count=int(axis.count),
                element_byte_delta=_logical_to_byte_offset(
                    logical_delta,
                    fmt.element_bits,
                    label=f"{obj.name}/{axis.name} repeat delta",
                ),
                scale_byte_delta=(
                    _ceil_bits_to_bytes((logical_delta // fmt.block) * fmt.scale_bits)
                    if fmt.is_mx
                    else 0
                ),
            )
        )
    if not axes and event.multiplicity > 1:
        axes.append(
            PhysicalRepeatAxis(
                name="stream_instruction",
                count=event.multiplicity,
                element_byte_delta=0,
                scale_byte_delta=0,
            )
        )
    axis_product = math.prod(axis.count for axis in axes) if axes else 1
    if axis_product != event.multiplicity:
        raise ValueError(
            f"physical stream {event.stream_index} axes cover {axis_product}, "
            f"expected {event.multiplicity}"
        )
    return PhysicalDmaStream(
        stage=event.stage,
        opcode=transfer.opcode,
        direction=transfer.direction,
        precision_role=role,
        format_signature=fmt.request_signature(),
        element_base=obj.base + element_offset,
        scale_base=obj.scale_base + scale_offset,
        dim=int(transfer.dim),
        amount=int(transfer.amount),
        stride_bytes=stride_bytes,
        rstride=int(transfer.rstride),
        write_amount=int(transfer.write_amount),
        axes=tuple(axes),
        multiplicity=event.multiplicity,
        stream_index=event.stream_index,
        source=transfer.source or "unspecified",
    )


def _axis_residue_distribution(axis: PhysicalRepeatAxis) -> Counter[tuple[int, int]]:
    result: Counter[tuple[int, int]] = Counter()
    element_delta = axis.element_byte_delta % REQUEST_BYTES
    scale_delta = axis.scale_byte_delta % REQUEST_BYTES
    period = 1
    while period <= REQUEST_BYTES * REQUEST_BYTES:
        if (period * element_delta) % REQUEST_BYTES == 0 and (
            period * scale_delta
        ) % REQUEST_BYTES == 0:
            break
        period += 1
    full, tail = divmod(axis.count, period)
    for index in range(period):
        count = full + int(index < tail)
        if count:
            result[
                (
                    index * element_delta % REQUEST_BYTES,
                    index * scale_delta % REQUEST_BYTES,
                )
            ] += count
    return result


def _stream_residues(stream: PhysicalDmaStream) -> Counter[tuple[int, int]]:
    states: Counter[tuple[int, int]] = Counter(
        {(stream.element_base % REQUEST_BYTES, stream.scale_base % REQUEST_BYTES): 1}
    )
    for axis in stream.axes:
        additions = _axis_residue_distribution(axis)
        combined: Counter[tuple[int, int]] = Counter()
        for (element, scale), count in states.items():
            for (element_delta, scale_delta), axis_count in additions.items():
                combined[
                    (
                        (element + element_delta) % REQUEST_BYTES,
                        (scale + scale_delta) % REQUEST_BYTES,
                    )
                ] += count * axis_count
        states = combined
    if sum(states.values()) != stream.multiplicity:
        raise ValueError(f"residue compression lost DMA multiplicity for stream {stream.stream_index}")
    return states


def _periodic_rows(amount: int, stride: int, element_residue: int, scale_stride: int):
    period = 1
    while period <= REQUEST_BYTES:
        if (period * stride) % REQUEST_BYTES == 0 and (
            period * scale_stride
        ) % REQUEST_BYTES == 0:
            break
        period += 1
    full, tail = divmod(amount, period)
    for index in range(period):
        count = full + int(index < tail)
        if count:
            yield (
                (element_residue + index * stride) % REQUEST_BYTES,
                index * scale_stride % REQUEST_BYTES,
                count,
            )


def _instruction_geometry(
    stream: PhysicalDmaStream,
    fmt: MemoryFormat,
    element_residue: int,
    scale_residue: int,
) -> Counter[str]:
    if stream.dim <= 0 or stream.amount <= 0:
        raise ValueError("DMA dim and amount must be positive")
    element_len = _ceil_bits_to_bytes(stream.dim * fmt.element_bits)
    if stream.dim * fmt.element_bits % 8:
        raise ValueError("DMA element row is not byte aligned")
    scale_len = 0
    if fmt.is_mx:
        if stream.dim % fmt.block:
            raise ValueError(f"dim={stream.dim} must be divisible by block={fmt.block}")
        if (stream.dim // fmt.block) * fmt.scale_bits % 8:
            raise ValueError("DMA scale row is not byte aligned")
        scale_len = (stream.dim // fmt.block) * fmt.scale_bits // 8
    stride = stream.stride_bytes if stream.rstride == 1 else element_len
    scale_stride = (
        _ceil_bits_to_bytes((stride * 8 // fmt.element_bits // fmt.block) * fmt.scale_bits)
        if fmt.is_mx
        else 0
    )
    totals: Counter[str] = Counter()
    for row_element_residue, row_scale_delta, count in _periodic_rows(
        stream.amount, stride, element_residue, scale_stride
    ):
        row_scale_residue = (scale_residue + row_scale_delta) % REQUEST_BYTES
        element_requests = (row_element_residue + element_len + REQUEST_BYTES - 1) // REQUEST_BYTES
        if stream.direction == "read":
            totals["read_requests"] += count * (element_requests + int(bool(scale_len)))
            totals["payload_read_bytes"] += count * (
                element_len + min(scale_len, REQUEST_BYTES - row_scale_residue)
            )
        elif stream.direction == "write":
            scale_requests = (
                (row_scale_residue + scale_len + REQUEST_BYTES - 1) // REQUEST_BYTES
                if scale_len
                else 0
            )
            touched = element_requests + scale_requests
            totals["read_requests"] += count * touched
            totals["write_requests"] += count * touched
            totals["rmw_requests"] += count * touched
            totals["payload_read_bytes"] += count * touched * REQUEST_BYTES
            totals["payload_write_bytes"] += count * (element_len + scale_len)
        else:
            raise ValueError(f"unsupported DMA direction {stream.direction!r}")
    return totals


def _flat_axis_offset(axes: Sequence[PhysicalRepeatAxis], index: int) -> tuple[int, int]:
    element = 0
    scale = 0
    remainder = index
    for axis in reversed(axes):
        axis_index = remainder % axis.count
        remainder //= axis.count
        element += axis_index * axis.element_byte_delta
        scale += axis_index * axis.scale_byte_delta
    return element, scale


def _even_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit <= 1:
        return [0]
    return sorted({round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def _sample_instruction_requests(
    stream: PhysicalDmaStream,
    fmt: MemoryFormat,
    *,
    instruction_index: int,
    limit: int,
) -> list[tuple[str, int]]:
    if limit <= 0:
        return []
    rows_per_instruction = max(1, min(stream.amount, limit // 2))
    row_indices = _even_indices(stream.amount, rows_per_instruction)
    element_len = stream.dim * fmt.element_bits // 8
    scale_len = (stream.dim // fmt.block) * fmt.scale_bits // 8 if fmt.is_mx else 0
    stride = stream.stride_bytes if stream.rstride == 1 else element_len
    logical_stride = stride * 8 // fmt.element_bits
    scale_stride = logical_stride * fmt.scale_bits // fmt.block if fmt.is_mx else 0
    requests: list[tuple[str, int]] = []
    element_delta, scale_delta = _flat_axis_offset(stream.axes, instruction_index)
    for row in row_indices:
        element_address = stream.element_base + element_delta + row * stride
        scale_address = stream.scale_base + scale_delta + row * scale_stride
        first = element_address // REQUEST_BYTES
        last = (element_address + element_len - 1) // REQUEST_BYTES
        element_blocks = _even_indices(last - first + 1, min(last - first + 1, 16))
        touched = [(first + index) * REQUEST_BYTES for index in element_blocks]
        if scale_len:
            if stream.direction == "read":
                touched.append(scale_address // REQUEST_BYTES * REQUEST_BYTES)
            else:
                scale_first = scale_address // REQUEST_BYTES
                scale_last = (scale_address + scale_len - 1) // REQUEST_BYTES
                touched.extend(
                    (scale_first + index) * REQUEST_BYTES
                    for index in _even_indices(
                        scale_last - scale_first + 1,
                        min(scale_last - scale_first + 1, 16),
                    )
                )
        if stream.direction == "read":
            requests.extend(("read", address) for address in touched)
        else:
            for address in touched:
                requests.append(("read", address))
                requests.append(("write", address))
        if len(requests) >= limit:
            return requests[:limit]
    return requests[:limit]


def sample_stream_requests(
    stream: PhysicalDmaStream,
    fmt: MemoryFormat,
    *,
    limit: int = DEFAULT_SAMPLED_REQUESTS,
) -> list[tuple[str, int]]:
    """Return a deterministic bounded sample of physical 64-byte requests."""
    if limit <= 0 or limit > MAX_SAMPLED_REQUESTS:
        if limit > MAX_SAMPLED_REQUESTS:
            raise ValueError(f"request sample limit exceeds {MAX_SAMPLED_REQUESTS}: {limit}")
        return []
    instruction_indices = _even_indices(stream.multiplicity, min(stream.multiplicity, 16))
    per_instruction_limit = max(1, limit // len(instruction_indices))
    requests = []
    for instruction_index in instruction_indices:
        requests.extend(
            _sample_instruction_requests(
                stream,
                fmt,
                instruction_index=instruction_index,
                limit=per_instruction_limit,
            )
        )
    return requests[:limit]


def mop4clxor_map(address: int, channels: int) -> tuple[int, int, int, int, int, int]:
    """Mirror Ramulator2's HBM2 MOP4CLXOR native-transfer mapper.

    The emulator memory API coalesces traffic into 64-byte lines, but the HBM2
    wrapper submits four 16-byte native transfers for each line.  Ramulator
    maps each native transfer independently, so mapper-level feature
    extraction must accept 16-byte alignment rather than only line bases.
    """
    native_transfer_bytes = 16
    if address < 0 or address % native_transfer_bytes:
        raise ValueError(
            "mapper address must be nonnegative and 16-byte aligned, "
            f"got {address}"
        )
    channel_bits = int(math.log2(channels))
    level_bits = (channel_bits, 1, 2, 2)
    value = address >> 4  # HBM2 internal prefetch 2 * configured 64-bit channel width.
    column = value & 0b11
    value >>= 2
    levels = []
    for bits in level_bits:
        levels.append(value & ((1 << bits) - 1))
        value >>= bits
    column |= (value & 0b111) << 2
    value >>= 3
    row = value
    xor_index = 0
    for index, bits in enumerate(level_bits):
        levels[index] ^= (column >> xor_index) & ((1 << bits) - 1)
        xor_index += bits
    return levels[0], levels[1], levels[2], levels[3], row, column


def mapper_statistics(
    requests: Sequence[tuple[str, int]],
    channels: int,
) -> MapperStats:
    if not requests:
        return MapperStats(0, 0, 0.0, 0, 0.0, 0, 0, 0, 0, 0, 0.0)
    channel_load: Counter[int] = Counter()
    open_rows: dict[tuple[int, int, int, int], int] = {}
    row_hits = row_misses = row_conflicts = 0
    addresses = []
    for _, address in requests:
        channel, pseudochannel, bankgroup, bank, row, _ = mop4clxor_map(address, channels)
        channel_load[channel] += 1
        addresses.append(address)
        bank_key = (channel, pseudochannel, bankgroup, bank)
        previous = open_rows.get(bank_key)
        if previous is None:
            row_misses += 1
        elif previous == row:
            row_hits += 1
        else:
            row_conflicts += 1
        open_rows[bank_key] = row
    sampled = len(requests)
    active = len(channel_load)
    mean = sampled / active if active else 0.0
    maximum = max(channel_load.values(), default=0)
    imbalance = maximum / mean if mean else 0.0
    unique = len(set(addresses))
    return MapperStats(
        sampled_requests=sampled,
        active_channels=active,
        mean_channel_load=mean,
        max_channel_load=maximum,
        channel_imbalance=imbalance,
        row_hits=row_hits,
        row_misses=row_misses,
        row_conflicts=row_conflicts,
        address_span_bytes=max(addresses) - min(addresses) + REQUEST_BYTES,
        unique_blocks=unique,
        reuse_ratio=max(0.0, (sampled - unique) / sampled),
    )


def summarize_physical_stream(
    stream: PhysicalDmaStream,
    fmt: MemoryFormat,
    hbm: HbmConfig,
    *,
    occurrence_count: int = 1,
) -> PhysicalGeometryWork:
    residues = _stream_residues(stream)
    totals: Counter[str] = Counter()
    for (element_residue, scale_residue), multiplicity in residues.items():
        geometry = _instruction_geometry(stream, fmt, element_residue, scale_residue)
        for name, value in geometry.items():
            totals[name] += value * multiplicity * occurrence_count
    samples = sample_stream_requests(stream, fmt)
    mapper = mapper_statistics(samples, hbm.channels)
    instruction_mappers = [
        mapper_statistics(
            _sample_instruction_requests(
                stream,
                fmt,
                instruction_index=index,
                limit=DEFAULT_SAMPLED_REQUESTS,
            ),
            hbm.channels,
        )
        for index in _even_indices(stream.multiplicity, min(stream.multiplicity, 8))
    ]
    instruction_sampled = sum(item.sampled_requests for item in instruction_mappers)
    mean_max_channel_share = sum(
        item.max_channel_load / max(1, item.sampled_requests) for item in instruction_mappers
    ) / len(instruction_mappers)
    mean_channel_share = sum(
        item.mean_channel_load / max(1, item.sampled_requests) for item in instruction_mappers
    ) / len(instruction_mappers)
    row_hits = sum(item.row_hits for item in instruction_mappers)
    row_misses = sum(item.row_misses for item in instruction_mappers)
    row_conflicts = sum(item.row_conflicts for item in instruction_mappers)
    request_work = totals["read_requests"] + totals["write_requests"]
    digest = hashlib.sha256(repr(stream.signature).encode()).hexdigest()[:16]
    return PhysicalGeometryWork(
        signature=digest,
        stage=stream.stage,
        opcode=stream.opcode,
        precision_role=stream.precision_role,
        format_signature=stream.format_signature,
        dim=stream.dim,
        amount=stream.amount,
        stride_bytes=stream.stride_bytes,
        dma_count=stream.multiplicity * occurrence_count,
        read_requests=int(totals["read_requests"]),
        write_requests=int(totals["write_requests"]),
        rmw_requests=int(totals["rmw_requests"]),
        physical_read_bytes=int(totals["read_requests"] * REQUEST_BYTES),
        physical_write_bytes=int(totals["write_requests"] * REQUEST_BYTES),
        payload_read_bytes=int(totals["payload_read_bytes"]),
        payload_write_bytes=int(totals["payload_write_bytes"]),
        active_channels=max(item.active_channels for item in instruction_mappers),
        max_channel_load=mean_max_channel_share,
        mean_channel_load=mean_channel_share,
        channel_imbalance=(
            mean_max_channel_share / mean_channel_share if mean_channel_share else 0.0
        ),
        row_hit_rate=row_hits / max(1, instruction_sampled),
        row_miss_rate=row_misses / max(1, instruction_sampled),
        row_conflict_rate=row_conflicts / max(1, instruction_sampled),
        address_span_bytes=mapper.address_span_bytes,
        reuse_ratio=mapper.reuse_ratio,
        queue_depth=request_work / max(1, stream.multiplicity * occurrence_count),
        sampled_requests=mapper.sampled_requests,
        stream_count=occurrence_count,
    )


def build_physical_memory_work(
    trace: CostTrace,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any],
    hbm_config: HbmConfig | Mapping[str, Any],
) -> PhysicalMemoryWork:
    precision = (
        precision_config
        if isinstance(precision_config, MemoryPrecisionConfig)
        else MemoryPrecisionConfig.from_mapping(precision_config)
    )
    hbm = hbm_config if isinstance(hbm_config, HbmConfig) else HbmConfig(**hbm_config)
    events = sorted(trace.memory_events, key=lambda event: event.stream_index)
    config_hash = trace.metadata.get("config_hash")
    cache_key = None
    if config_hash:
        cache_key = (
            "physical_memory_work_v2",
            str(config_hash),
            int(trace.metadata.get("num_layers", 1)),
            _memory_events_cache_digest(events),
            json.dumps(precision.to_dict(), sort_keys=True, separators=(",", ":")),
            hbm,
        )
        cached = _PHYSICAL_WORK_CACHE.get(cache_key)
        if cached is not None:
            _PHYSICAL_WORK_CACHE.move_to_end(cache_key)
            return cached
    streams = build_physical_dma_streams(trace, precision)
    grouped: dict[tuple[Any, ...], tuple[PhysicalDmaStream, MemoryFormat, int]] = {}
    for stream, fmt in streams:
        existing = grouped.get(stream.signature)
        if existing is None:
            grouped[stream.signature] = (stream, fmt, 1)
        else:
            representative, previous_format, occurrences = existing
            grouped[stream.signature] = (representative, previous_format, occurrences + 1)
    geometries = tuple(
        summarize_physical_stream(stream, fmt, hbm, occurrence_count=occurrences)
        for stream, fmt, occurrences in grouped.values()
    )
    result = PhysicalMemoryWork(
        geometries=geometries,
        precision_config=precision,
        hbm_config=hbm,
        logical_object_count=len({_event_object(event) for event in events}),
    )
    if cache_key is not None:
        _PHYSICAL_WORK_CACHE[cache_key] = result
        _PHYSICAL_WORK_CACHE.move_to_end(cache_key)
        while len(_PHYSICAL_WORK_CACHE) > _PHYSICAL_WORK_CACHE_LIMIT:
            _PHYSICAL_WORK_CACHE.popitem(last=False)
    return result


def build_physical_dma_streams(
    trace: CostTrace,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any],
) -> tuple[tuple[PhysicalDmaStream, MemoryFormat], ...]:
    """Materialize each compressed DMA stream using the production layout.

    V3 consumes these streams only after aggregating equal geometries.  V4
    needs the ungrouped stream and its repeat axes so it can predict one DMA
    occurrence at a time.  Keeping layout construction here gives both models
    one implementation for object placement, packed byte strides, and layer
    offsets.
    """

    precision = (
        precision_config
        if isinstance(precision_config, MemoryPrecisionConfig)
        else MemoryPrecisionConfig.from_mapping(precision_config)
    )
    events = sorted(trace.memory_events, key=lambda event: event.stream_index)
    objects = _build_object_layout(events, precision)
    return tuple(
        (
            stream,
            precision.for_role(stream.precision_role, stream.opcode),
        )
        for event in events
        for stream in (
            _physical_stream(event, objects[_event_object(event)], precision),
        )
    )


def clear_physical_memory_work_cache() -> None:
    _PHYSICAL_WORK_CACHE.clear()


SERVICE_FEATURE_NAMES = (
    "startup_prefetch_m",
    "startup_prefetch_v",
    "startup_store_v",
    "marginal_prefetch_m",
    "marginal_prefetch_v",
    "marginal_store_v",
    "read_requests",
    "read_scale_requests",
    "read_scale_reuse_requests",
    "read_scale_reuse_pressure",
    "read_reuse_requests",
    "read_sqrt_requests",
    "read_sqrt_channel_requests",
    "read_row_width",
    "inverse_channel_dma",
    "write_requests",
    "rmw_requests",
    "channel_tail_requests",
    "reuse_channel_tail_requests",
    "row_miss_requests",
    "row_conflict_requests",
    "queue_depth_work",
)


@dataclass(frozen=True)
class HbmServiceSample:
    geometry: PhysicalGeometryWork
    channels: int
    observed_latency_ns: float
    split: str
    family: str


def _nonnegative_ridge(
    matrix: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    *,
    iterations: int = 20_000,
) -> np.ndarray:
    if matrix.ndim != 2 or targets.ndim != 1 or len(matrix) != len(targets):
        raise ValueError("invalid nonnegative ridge input shapes")
    scales = np.sqrt(np.mean(matrix * matrix, axis=0))
    scales[scales == 0] = 1.0
    normalized = matrix / scales
    coefficients = np.zeros(normalized.shape[1], dtype=float)
    prediction = normalized @ coefficients
    for _ in range(iterations):
        previous = coefficients.copy()
        for column in range(normalized.shape[1]):
            values = normalized[:, column]
            residual = targets - prediction + values * coefficients[column]
            updated = max(0.0, float(values @ residual) / float(values @ values + ridge))
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        if np.max(np.abs(coefficients - previous)) < 1e-10:
            break
    return coefficients / scales


@dataclass(frozen=True)
class HbmServicePrediction:
    latency_ns: float
    theoretical_floor_ns: float
    opcode_latency_ns: Mapping[str, float]
    stage_latency_ns: Mapping[str, float]
    calibration_in_domain: bool
    domain_issues: tuple[str, ...]


@dataclass(frozen=True)
class HbmServiceModel:
    calibration_id: str
    coefficients: Mapping[str, float]
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        samples: Sequence[HbmServiceSample],
        *,
        ridge: float = 1e-8,
        metadata: Mapping[str, Any] | None = None,
        compatibility: Mapping[str, Any] | None = None,
    ) -> HbmServiceModel:
        training = [sample for sample in samples if sample.split == "train"]
        if not training:
            raise ValueError("global HBM service fit requires training samples")
        matrix = np.asarray(
            [
                [sample.geometry.features(sample.channels)[name] for name in SERVICE_FEATURE_NAMES]
                for sample in training
            ],
            dtype=float,
        )
        targets = np.asarray([sample.observed_latency_ns for sample in training], dtype=float)
        base_weights = 1.0 / np.maximum(targets, 1.0)
        emphasis = np.ones_like(targets)
        coefficients = np.zeros(matrix.shape[1], dtype=float)
        for _ in range(5):
            weights = base_weights * np.sqrt(emphasis)
            coefficients = _nonnegative_ridge(
                matrix * weights[:, np.newaxis],
                targets * weights,
                ridge,
            )
            relative_error = np.abs(matrix @ coefficients - targets) / np.maximum(targets, 1.0)
            emphasis = np.clip(relative_error / 0.25, 1.0, 4.0)
        coefficient_map = dict(zip(SERVICE_FEATURE_NAMES, coefficients.tolist(), strict=True))
        identity = {
            "coefficients": coefficient_map,
            "compatibility": dict(compatibility or {}),
            "metadata": dict(metadata or {}),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        return cls(
            calibration_id=f"hbm-global-v3-{digest}",
            coefficients=coefficient_map,
            compatibility=dict(compatibility or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> HbmServiceModel:
        data = json.loads(Path(path).read_text())
        if int(data.get("schema_version", -1)) != SERVICE_MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported HBM service model schema {data.get('schema_version')!r}")
        if tuple(data.get("feature_names", ())) != SERVICE_FEATURE_NAMES:
            raise ValueError("HBM service model feature schema does not match the current evaluator")
        return cls(
            calibration_id=str(data["calibration_id"]),
            coefficients={name: float(value) for name, value in data["coefficients"].items()},
            compatibility=dict(data.get("compatibility", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_MODEL_SCHEMA_VERSION,
            "model_kind": "global_hbm_service_nonnegative_ridge",
            "calibration_id": self.calibration_id,
            "feature_names": list(SERVICE_FEATURE_NAMES),
            "coefficients": dict(self.coefficients),
            "compatibility": dict(self.compatibility),
            "metadata": dict(self.metadata),
        }

    def _domain_issues(self, work: PhysicalMemoryWork) -> tuple[str, ...]:
        domain = self.compatibility.get("domain", {})
        issues = []
        channels = set(int(value) for value in domain.get("channels", []))
        if channels and work.hbm_config.channels not in channels:
            issues.append(f"channels={work.hbm_config.channels}")
        dimensions = set(int(value) for value in domain.get("dimensions", []))
        used_dimensions = {item.dim for item in work.geometries}
        if dimensions and not used_dimensions <= dimensions:
            issues.append(f"dimensions={sorted(used_dimensions - dimensions)}")
        signatures = set(domain.get("request_signatures", []))
        used_signatures = {item.format_signature for item in work.geometries}
        if signatures and not used_signatures <= signatures:
            issues.append(f"request_signatures={sorted(used_signatures - signatures)}")
        root = Path(__file__).resolve().parents[2]
        current_hashes = {
            "dma_semantics_hash": _file_sha256(root / "transactional_emulator/src/dma.rs"),
            "request_geometry_hash": _file_sha256(Path(__file__)),
        }
        for name, current in current_hashes.items():
            calibrated = self.compatibility.get(name)
            if calibrated and current != calibrated:
                issues.append(f"{name}=mismatch")
        return tuple(issues)

    def predict(self, work: PhysicalMemoryWork) -> HbmServicePrediction:
        opcode_latency: Counter[str] = Counter()
        stage_latency: Counter[str] = Counter()
        total = 0.0
        total_floor = 0.0
        for geometry in work.geometries:
            features = geometry.features(work.hbm_config.channels)
            estimate = sum(
                self.coefficients.get(name, 0.0) * features[name]
                for name in SERVICE_FEATURE_NAMES
            )
            floor = (
                geometry.physical_read_bytes + geometry.physical_write_bytes
            ) / (
                work.hbm_config.channels
                * work.hbm_config.channel_bandwidth_bytes_per_ns
            )
            latency = max(estimate, floor)
            total += latency
            total_floor += floor
            opcode_latency[geometry.opcode] += latency
            stage_latency[geometry.stage] += latency
        issues = self._domain_issues(work)
        return HbmServicePrediction(
            latency_ns=total,
            theoretical_floor_ns=total_floor,
            opcode_latency_ns=dict(sorted(opcode_latency.items())),
            stage_latency_ns=dict(sorted(stage_latency.items())),
            calibration_in_domain=not issues,
            domain_issues=issues,
        )


def validate_holdout(
    model: HbmServiceModel,
    samples: Iterable[HbmServiceSample],
) -> dict[str, Any]:
    errors = []
    rows = []
    for sample in samples:
        if sample.split != "holdout":
            continue
        work = PhysicalMemoryWork(
            geometries=(sample.geometry,),
            precision_config=MemoryPrecisionConfig.active_mxint4(),
            hbm_config=HbmConfig(sample.channels),
            logical_object_count=1,
        )
        prediction = model.predict(work).latency_ns
        error = 100.0 * abs(prediction - sample.observed_latency_ns) / sample.observed_latency_ns
        errors.append(error)
        rows.append(
            {
                "family": sample.family,
                "channels": sample.channels,
                "observed_latency_ns": sample.observed_latency_ns,
                "predicted_latency_ns": prediction,
                "absolute_error_percent": error,
            }
        )
    ordered = sorted(errors)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "schema_version": SERVICE_MODEL_SCHEMA_VERSION,
        "calibration_id": model.calibration_id,
        "sample_count": len(rows),
        "median_absolute_error_percent": percentile(0.5),
        "p95_absolute_error_percent": percentile(0.95),
        "max_absolute_error_percent": max(errors, default=None),
        "samples": rows,
    }


__all__ = [
    "HbmConfig",
    "HbmServiceModel",
    "HbmServicePrediction",
    "HbmServiceSample",
    "MemoryFormat",
    "MemoryPrecisionConfig",
    "PhysicalDmaStream",
    "PhysicalGeometryWork",
    "PhysicalMemoryWork",
    "PhysicalRepeatAxis",
    "build_physical_dma_streams",
    "build_physical_memory_work",
    "clear_physical_memory_work_cache",
    "mapper_statistics",
    "mop4clxor_map",
    "sample_stream_requests",
    "summarize_physical_stream",
    "validate_holdout",
]
