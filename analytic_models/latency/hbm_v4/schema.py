"""Physical memory formats and exact production-DMA request planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np

from compiler.aten.isa_builder import DmaTransfer


REQUEST_BYTES = 64
PHYSICAL_BURST_BYTES = 16
DMA_SEMANTIC_VERSION = "production-dma-lines-v2"
FEATURE_SEMANTIC_VERSION = "production-dma-targeted-row-hit-v3"
MANIFEST_HASH_ALGORITHM = "fnv1a64-v1"
SUPPORTED_CHANNELS = (8, 32, 128)


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
        if family in {"mx", "mxint", "mxfp"}:
            if self.element_bits not in {4, 8}:
                raise ValueError(f"HBM V4 supports calibrated 4/8-bit MX formats, got {self.element_bits}")
            if self.scale_bits <= 0 or self.block <= 0:
                raise ValueError("MX formats require positive scale_bits and block")
        elif family == "plain":
            if self.element_bits <= 0 or self.scale_bits or self.block != 1:
                raise ValueError("plain formats require positive bits and no scale stream")
        else:
            raise ValueError(f"unsupported memory format family {self.family!r}")

    @property
    def is_mx(self) -> bool:
        return self.family in {"mx", "mxint", "mxfp"}

    def request_signature(self) -> str:
        if self.is_mx:
            return f"mx:e{self.element_bits}:s{self.scale_bits}:b{self.block}"
        return f"plain:e{self.element_bits}"

    @classmethod
    def from_settings(cls, value: Mapping[str, Any], *, name: str) -> MemoryFormat:
        format_name = str(value.get("format", "Plain")).lower()
        if format_name == "mx":
            element = value["ELEM"]
            scale = value["SCALE"]
            return cls(
                family="mxfp" if str(element.get("type", "Fp")).lower() == "fp" else "mxint",
                element_bits=(
                    1 + int(element["exponent"]) + int(element["mantissa"])
                    if str(element.get("type", "Fp")).lower() == "fp"
                    else int(element["width"])
                ),
                scale_bits=(
                    int(bool(scale.get("sign", False)))
                    + int(scale.get("exponent", 0))
                    + int(scale.get("mantissa", 0))
                    if str(scale.get("type", "Fp")).lower() == "fp"
                    else int(scale["width"])
                ),
                block=int(value["block"]),
                name=name,
            )
        data = value.get("DATA_TYPE", value)
        bits = (
            1 + int(data["exponent"]) + int(data["mantissa"])
            if str(data.get("type", "Int")).lower() == "fp"
            else int(data["width"])
        )
        return cls("plain", bits, name=name)


@dataclass(frozen=True)
class HbmPrecisionConfig:
    weight: MemoryFormat
    matrix_kv: MemoryFormat
    activation: MemoryFormat
    vector_kv: MemoryFormat
    integer: MemoryFormat

    @classmethod
    def from_settings(cls, precision: Mapping[str, Any]) -> HbmPrecisionConfig:
        keys = {
            "weight": "HBM_M_WEIGHT_TYPE",
            "matrix_kv": "HBM_M_KV_TYPE",
            "activation": "HBM_V_ACT_TYPE",
            "vector_kv": "HBM_V_KV_TYPE",
            "integer": "HBM_V_INT_TYPE",
        }
        return cls(
            **{
                role: MemoryFormat.from_settings(precision[key], name=key)
                for role, key in keys.items()
            }
        )

    def for_transfer(self, transfer: DmaTransfer) -> MemoryFormat:
        explicit = {
            "weight": "weight",
            "matrix_kv": "matrix_kv",
            "activation": "activation",
            "kv": "matrix_kv" if transfer.opcode == "H_PREFETCH_M" else "vector_kv",
            "vector_kv": "vector_kv",
            "integer": "integer",
            "output": "activation",
            "runtime": "matrix_kv" if transfer.role == "kv" and transfer.opcode == "H_PREFETCH_M" else transfer.role,
        }
        role = explicit.get(transfer.precision)
        if role is None:
            role = explicit.get(transfer.role)
        if role not in self.__dataclass_fields__:
            raise ValueError(
                f"cannot map DMA precision={transfer.precision!r}, role={transfer.role!r}, "
                f"opcode={transfer.opcode!r} to a physical format"
            )
        result = getattr(self, role)
        compiler_bits = transfer.element_bytes * 8
        if result.element_bits != compiler_bits:
            raise ValueError(
                f"compiler DMA addresses use {compiler_bits}-bit elements but {role} is "
                f"{result.element_bits}-bit; main lowering must describe repacked addresses explicitly"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {name: asdict(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class HbmV4Config:
    channels: int
    request_bytes: int = REQUEST_BYTES
    physical_burst_bytes: int = PHYSICAL_BURST_BYTES
    channel_bandwidth_bytes_per_ns: float = 16.0
    mapper: str = "MOP4CLXOR"
    preset: str = "HBM2_2Gbps"

    def __post_init__(self) -> None:
        if self.channels not in SUPPORTED_CHANNELS:
            raise ValueError(f"HBM V4 calibration supports channels={SUPPORTED_CHANNELS}, got {self.channels}")
        if self.request_bytes != REQUEST_BYTES or self.physical_burst_bytes != PHYSICAL_BURST_BYTES:
            raise ValueError("HBM V4 requires 64-B requests and 16-B physical bursts")
        if self.channel_bandwidth_bytes_per_ns != 16.0:
            raise ValueError("HBM V4 is calibrated only at 16 bytes/ns/channel")
        if self.mapper != "MOP4CLXOR" or self.preset != "HBM2_2Gbps":
            raise ValueError("HBM V4 is calibrated only for HBM2_2Gbps with MOP4CLXOR")


def _fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return value


def _manifest_hash(read_lines: Sequence[int], write_lines: Sequence[int]) -> str:
    text = [f"{DMA_SEMANTIC_VERSION}\n"]
    text.extend(f"R:{address:016x}\n" for address in read_lines)
    text.extend(f"W:{address:016x}\n" for address in write_lines)
    return f"fnv1a64:{_fnv1a64(''.join(text).encode()):016x}"


def _line_coverage(ranges: Sequence[tuple[int, int]]) -> dict[int, int]:
    if any(address < 0 or length < 0 for address, length in ranges):
        raise ValueError("DMA byte ranges cannot be negative")
    nonempty = tuple((address, length) for address, length in ranges if length)
    if not nonempty:
        return {}
    addresses = np.fromiter((address for address, _ in nonempty), dtype=np.uint64)
    lengths = np.fromiter((length for _, length in nonempty), dtype=np.uint64)
    first_lines = addresses // REQUEST_BYTES * REQUEST_BYTES
    final_lines = (addresses + lengths - np.uint64(1)) // REQUEST_BYTES * REQUEST_BYTES
    line_counts = ((final_lines - first_lines) // REQUEST_BYTES + 1).astype(np.int64)
    offsets = np.arange(int(line_counts.max()), dtype=np.uint64)
    candidates = first_lines[:, None] + offsets * REQUEST_BYTES
    valid = offsets[None, :] < line_counts[:, None]
    starts = np.maximum(candidates, addresses[:, None])
    ends = np.minimum(candidates + REQUEST_BYTES, addresses[:, None] + lengths[:, None])
    within = (starts - candidates)[valid]
    byte_counts = (ends - starts)[valid]
    touched = candidates[valid]
    full_mask = np.uint64((1 << REQUEST_BYTES) - 1)
    safe_counts = np.minimum(byte_counts, np.uint64(REQUEST_BYTES - 1))
    masks = (np.left_shift(np.uint64(1), safe_counts) - np.uint64(1)) << within
    masks = np.where(byte_counts == REQUEST_BYTES, full_mask, masks)
    unique, inverse = np.unique(touched, return_inverse=True)
    combined = np.zeros(len(unique), dtype=np.uint64)
    np.bitwise_or.at(combined, inverse, masks)
    return dict(zip(unique.tolist(), combined.tolist(), strict=True))


@dataclass(frozen=True)
class DmaRequestManifest:
    read_lines: tuple[int, ...]
    write_lines: tuple[int, ...]
    full_lines: int
    partial_lines: int
    payload_read_bytes: int
    payload_write_bytes: int

    @property
    def read_bytes(self) -> int:
        return len(self.read_lines) * REQUEST_BYTES

    @property
    def write_bytes(self) -> int:
        return len(self.write_lines) * REQUEST_BYTES

    @property
    def request_manifest_hash(self) -> str:
        return _manifest_hash(self.read_lines, self.write_lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_lines": len(self.read_lines),
            "write_lines": len(self.write_lines),
            "full_lines": self.full_lines,
            "partial_lines": self.partial_lines,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "payload_read_bytes": self.payload_read_bytes,
            "payload_write_bytes": self.payload_write_bytes,
            "request_manifest_hash": self.request_manifest_hash,
        }


def plan_dma_request_manifest(transfer: DmaTransfer, fmt: MemoryFormat) -> DmaRequestManifest:
    """Mirror main's packed MX layout and 64-B gather/scatter coalescing."""

    dim = int(transfer.dim)
    amount = int(transfer.amount)
    if dim <= 0 or amount <= 0:
        raise ValueError("DMA dim and amount must be positive")
    element_bits = fmt.element_bits * dim
    if element_bits % 8:
        raise ValueError("element row is not byte aligned")
    element_bytes = element_bits // 8
    scale_bytes = 0
    if fmt.is_mx:
        if dim % fmt.block:
            raise ValueError(f"dim={dim} is not divisible by MX block={fmt.block}")
        scale_bits = fmt.scale_bits * (dim // fmt.block)
        if scale_bits % 8:
            raise ValueError("scale row is not byte aligned")
        scale_bytes = scale_bits // 8
        if transfer.scale_base_bytes is None:
            raise ValueError("MX DMA transfer is missing scale_base_bytes")

    stride_bytes = element_bytes if int(transfer.rstride) != 1 else int(transfer.stride_bytes)
    stride_bits = stride_bytes * 8
    if stride_bits % fmt.element_bits:
        raise ValueError("packed element stride is not integral")
    stride_elements = stride_bits // fmt.element_bits
    scale_stride = 0
    if fmt.is_mx:
        if stride_elements % fmt.block:
            raise ValueError("packed scale stride is not block aligned")
        scale_stride_bits = stride_elements // fmt.block * fmt.scale_bits
        if scale_stride_bits % 8:
            raise ValueError("packed scale stride is not byte aligned")
        scale_stride = scale_stride_bits // 8

    ranges: list[tuple[int, int]] = []
    for row in range(amount):
        ranges.append((transfer.element_base_bytes + row * stride_bytes, element_bytes))
        if scale_bytes:
            assert transfer.scale_base_bytes is not None
            ranges.append((transfer.scale_base_bytes + row * scale_stride, scale_bytes))
    coverage = _line_coverage(ranges)
    full_mask = (1 << REQUEST_BYTES) - 1
    full_lines = sum(mask == full_mask for mask in coverage.values())
    partial_lines = len(coverage) - full_lines
    if transfer.direction == "read":
        read_lines = tuple(sorted(coverage))
        write_lines: tuple[int, ...] = ()
        payload_read = amount * (element_bytes + scale_bytes)
        payload_write = 0
    elif transfer.direction == "write":
        write_lines = tuple(sorted(coverage))
        read_lines = tuple(address for address in write_lines if coverage[address] != full_mask)
        payload_read = partial_lines * REQUEST_BYTES
        payload_write = amount * (element_bytes + scale_bytes)
    else:
        raise ValueError(f"unsupported DMA direction {transfer.direction!r}")
    return DmaRequestManifest(
        read_lines=read_lines,
        write_lines=write_lines,
        full_lines=full_lines,
        partial_lines=partial_lines,
        payload_read_bytes=payload_read,
        payload_write_bytes=payload_write,
    )


def request_manifest_fixture_hash() -> str:
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    fixtures = []
    values = (
        ("write", 0, 60, 128, 1, 64),
        ("write", 32, (1 << 20) + 32, 128, 1, 64),
        ("read", 0, 1 << 20, 128, 2, 64),
    )
    for direction, element, scale, dim, amount, stride in values:
        transfer = DmaTransfer(
            opcode="H_STORE_V" if direction == "write" else "H_PREFETCH_V",
            direction=direction,
            role="activation",
            element_base_bytes=element,
            scale_base_bytes=scale,
            dim=dim,
            amount=amount,
            stride_bytes=stride,
            rstride=1,
            write_amount=amount,
            element_bytes=1,
        )
        fixtures.append(plan_dma_request_manifest(transfer, fmt).to_dict())
    payload = json.dumps(fixtures, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mop4clxor_phase_statistics(
    lines: Sequence[int],
    config: HbmV4Config,
    open_rows: np.ndarray,
) -> tuple[int, int, int, int, int, int, int]:
    """Return exact critical-channel statistics and update open-row state."""

    if not lines:
        return (0, 0, 0, 0, 0, 0, 0)
    addresses = (
        np.asarray(lines, dtype=np.uint64)[:, None]
        + np.arange(0, REQUEST_BYTES, PHYSICAL_BURST_BYTES, dtype=np.uint64)[None, :]
    ).reshape(-1)
    channels = config.channels
    channel_bits = channels.bit_length() - 1
    channel_mask = channels - 1
    value = addresses >> np.uint64(4)
    column = value & np.uint64(0b11)
    value >>= np.uint64(2)
    channel = value & np.uint64(channel_mask)
    value >>= np.uint64(channel_bits)
    pseudo = value & np.uint64(1)
    value >>= np.uint64(1)
    bankgroup = value & np.uint64(0b11)
    value >>= np.uint64(2)
    bank = value & np.uint64(0b11)
    value >>= np.uint64(2)
    column |= (value & np.uint64(0b111)) << np.uint64(2)
    value >>= np.uint64(3)
    row = value.astype(np.int64, copy=False)
    channel ^= column & np.uint64(channel_mask)
    pseudo ^= (column >> np.uint64(channel_bits)) & np.uint64(1)
    bankgroup ^= (column >> np.uint64(channel_bits + 1)) & np.uint64(0b11)
    bank ^= (column >> np.uint64(channel_bits + 3)) & np.uint64(0b11)

    channel_i = channel.astype(np.int64, copy=False)
    pseudo_key = channel_i * 2 + pseudo.astype(np.int64, copy=False)
    group_key = pseudo_key * 4 + bankgroup.astype(np.int64, copy=False)
    bank_key = group_key * 4 + bank.astype(np.int64, copy=False)
    banks_per_channel = 32
    channel_load = np.bincount(channel_i, minlength=channels)
    pseudo_load = np.bincount(pseudo_key, minlength=channels * 2)
    group_load = np.bincount(group_key, minlength=channels * 8)
    bank_load = np.bincount(bank_key, minlength=channels * banks_per_channel)

    order = np.argsort(bank_key, kind="stable")
    sorted_bank = bank_key[order]
    sorted_row = row[order]
    starts = np.concatenate((np.asarray([0]), np.flatnonzero(sorted_bank[1:] != sorted_bank[:-1]) + 1))
    ends = np.concatenate((starts[1:] - 1, np.asarray([len(sorted_bank) - 1])))
    group_banks = sorted_bank[starts]
    group_channels = group_banks // banks_per_channel
    previous = open_rows[group_banks]
    first = sorted_row[starts]
    misses = np.bincount(group_channels[previous < 0], minlength=channels)
    initial = (previous >= 0) & (previous != first)
    initial_conflicts = np.bincount(group_channels[initial], minlength=channels)
    changed = (sorted_bank[1:] == sorted_bank[:-1]) & (sorted_row[1:] != sorted_row[:-1])
    internal_channels = sorted_bank[1:][changed] // banks_per_channel
    internal_conflicts = np.bincount(internal_channels, minlength=channels)
    open_rows[sorted_bank[ends]] = sorted_row[ends]
    return (
        int(channel_load.max(initial=0)),
        int(pseudo_load.max(initial=0)),
        int(group_load.max(initial=0)),
        int(bank_load.max(initial=0)),
        int(misses.max(initial=0)),
        int(initial_conflicts.max(initial=0)),
        int((initial_conflicts + internal_conflicts).max(initial=0)),
    )


def payload_row_bytes(transfer: DmaTransfer, fmt: MemoryFormat) -> tuple[int, int]:
    element = math.ceil(transfer.dim * fmt.element_bits / 8)
    scale = math.ceil((transfer.dim // fmt.block) * fmt.scale_bits / 8) if fmt.is_mx else 0
    return element, scale
