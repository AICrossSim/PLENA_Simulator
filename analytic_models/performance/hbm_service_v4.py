"""Production-DMA HBM service calibration and occurrence-level surrogate.

V4 intentionally coexists with the historical global V3 model.  Its training
target is one production DMA instruction completion interval, measured through
the same packed-layout and 64-byte ``gather``/``scatter`` path as the
transactional emulator.  The model therefore predicts one occurrence at a
time; callers must not fit an aggregate stream and divide by multiplicity.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .hbm_service_model import (
    HbmConfig,
    MemoryFormat,
    MemoryPrecisionConfig,
    PhysicalDmaStream,
    build_physical_dma_streams,
)
from .rtl_opcode_timing import OpcodeTimingEstimate


REQUEST_BYTES = 64
PHYSICAL_BURST_BYTES = 16
SCHEMA_VERSION = 4
DMA_SEMANTIC_VERSION = "production-dma-lines-v2"
FEATURE_SEMANTIC_VERSION = "production-dma-targeted-row-hit-v3"
LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION = "production-dma-targeted-row-hit-v2"
MANIFEST_HASH_ALGORITHM = "fnv1a64-v1"
DEFAULT_SEED = 20260716
DEFAULT_CHANNELS = (8, 32, 128)
DEFAULT_DIMENSIONS = (64, 128, 256, 512, 1024, 2048)
DEFAULT_LAYOUTS = (
    "aligned_contiguous",
    "offset32_contiguous",
    "aligned_stride2x",
    "offset32_large_stride",
)
FEATURE_NAMES = (
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
# Fully warmed rows have a different fixed command cost from cold/mixed-row
# traffic.  Fitting that low-latency regime with the complete cold feature set
# forces one set of nonnegative coefficients to compromise between two
# physically distinct cases.  Keep the warm model intentionally small: the
# transfer floor already accounts for data-bus work.  The strict warm regime
# has neither a closed-bank miss nor an open-row conflict; accesses that switch
# an already-open bank to another row stay in the cold/mixed model.
WARM_FEATURE_NAMES = (
    "read_phase_startup",
    "write_phase_startup",
    "read_row_conflict",
    "write_row_conflict",
)


def _fnv1a64(data: bytes) -> int:
    value = 0xCBF29CE484222325
    return _fnv1a64_update(value, data)


def _fnv1a64_update(value: int, data: bytes) -> int:
    """Extend an FNV-1a digest without materializing a full manifest string."""

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
    """Return a 64-bit byte-coverage mask for each touched physical line."""

    if not ranges:
        return {}
    if any(address < 0 or length < 0 for address, length in ranges):
        raise ValueError(f"negative DMA byte range in {ranges!r}")

    nonempty = tuple((address, length) for address, length in ranges if length)
    if not nonempty:
        return {}
    addresses = np.fromiter((address for address, _length in nonempty), dtype=np.uint64)
    lengths = np.fromiter((length for _address, length in nonempty), dtype=np.uint64)
    first_lines = addresses // REQUEST_BYTES * REQUEST_BYTES
    final_lines = (addresses + lengths - np.uint64(1)) // REQUEST_BYTES * REQUEST_BYTES
    line_counts = ((final_lines - first_lines) // REQUEST_BYTES + 1).astype(np.int64)
    offsets = np.arange(int(line_counts.max()), dtype=np.uint64)
    candidate_lines = first_lines[:, None] + offsets * REQUEST_BYTES
    valid = offsets[None, :] < line_counts[:, None]
    range_starts = addresses[:, None]
    range_ends = range_starts + lengths[:, None]
    segment_starts = np.maximum(candidate_lines, range_starts)
    segment_ends = np.minimum(candidate_lines + REQUEST_BYTES, range_ends)
    within = (segment_starts - candidate_lines)[valid]
    byte_counts = (segment_ends - segment_starts)[valid]
    touched_lines = candidate_lines[valid]

    full_mask = np.uint64((1 << REQUEST_BYTES) - 1)
    safe_counts = np.minimum(byte_counts, np.uint64(REQUEST_BYTES - 1))
    masks = (np.left_shift(np.uint64(1), safe_counts) - np.uint64(1)) << within
    masks = np.where(byte_counts == REQUEST_BYTES, full_mask, masks)
    unique_lines, inverse = np.unique(touched_lines, return_inverse=True)
    combined = np.zeros(len(unique_lines), dtype=np.uint64)
    np.bitwise_or.at(combined, inverse, masks)
    return dict(zip(unique_lines.tolist(), combined.tolist(), strict=True))


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


def combined_request_manifest_hash(
    manifests: Sequence[DmaRequestManifest],
) -> str:
    """Hash a production DMA sequence exactly like the Rust replay driver.

    The Rust calibration driver concatenates every occurrence's read lines,
    followed by every occurrence's write lines.  Updating the digest in two
    passes over manifest references preserves that order while avoiding one
    potentially very large canonical string or a duplicate address array.
    """

    value = _fnv1a64_update(0xCBF29CE484222325, f"{DMA_SEMANTIC_VERSION}\n".encode())
    for manifest in manifests:
        for address in manifest.read_lines:
            value = _fnv1a64_update(value, f"R:{address:016x}\n".encode())
    for manifest in manifests:
        for address in manifest.write_lines:
            value = _fnv1a64_update(value, f"W:{address:016x}\n".encode())
    return f"fnv1a64:{value:016x}"


def _format_from_mapping(value: Mapping[str, Any]) -> MemoryFormat:
    return MemoryFormat(
        family=str(value["family"]),
        element_bits=int(value["element_bits"]),
        scale_bits=int(value.get("scale_bits", 0)),
        block=int(value.get("block", 1)),
        name=str(value.get("name", "")),
    )


def plan_dma_request_manifest(
    transfer: Mapping[str, Any],
    format_value: MemoryFormat | Mapping[str, Any],
) -> DmaRequestManifest:
    """Mirror production ``MxLayout`` plus chunked line coalescing."""

    fmt = format_value if isinstance(format_value, MemoryFormat) else _format_from_mapping(format_value)
    dim = int(transfer["dim"])
    amount = int(transfer["amount"])
    if dim <= 0 or amount <= 0:
        raise ValueError("DMA dim and amount must be positive")
    element_bits = fmt.element_bits * dim
    if element_bits % 8:
        raise ValueError("element row is not byte aligned")
    element_bytes = element_bits // 8
    if fmt.is_mx:
        if dim % fmt.block:
            raise ValueError(f"dim={dim} is not divisible by MX block={fmt.block}")
        scale_row_bits = fmt.scale_bits * (dim // fmt.block)
        if scale_row_bits % 8:
            raise ValueError("scale row is not byte aligned")
        scale_bytes = scale_row_bits // 8
    else:
        scale_bytes = 0

    stride_bytes = int(transfer.get("stride_bytes", transfer.get("stride", 0)))
    if int(transfer.get("rstride", 1)) != 1:
        stride_bytes = element_bytes
    stride_bits = stride_bytes * 8
    if stride_bits % fmt.element_bits:
        raise ValueError("packed element stride is not integral")
    stride_elements = stride_bits // fmt.element_bits
    if fmt.is_mx:
        if stride_elements % fmt.block:
            raise ValueError("packed scale stride is not block aligned")
        scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
    else:
        scale_stride = 0

    element_base = int(transfer["element_base"])
    scale_base = int(transfer["scale_base"])
    ranges: list[tuple[int, int]] = []
    for row in range(amount):
        ranges.append((element_base + row * stride_bytes, element_bytes))
        if scale_bytes:
            ranges.append((scale_base + row * scale_stride, scale_bytes))
    coverage = _line_coverage(ranges)
    full_mask = (1 << REQUEST_BYTES) - 1
    full_lines = sum(mask == full_mask for mask in coverage.values())
    partial_lines = len(coverage) - full_lines
    direction = str(transfer["direction"])
    if direction == "read":
        read_lines = tuple(sorted(coverage))
        write_lines: tuple[int, ...] = ()
        payload_read = amount * (element_bytes + scale_bytes)
        payload_write = 0
    elif direction == "write":
        write_lines = tuple(sorted(coverage))
        read_lines = tuple(address for address in write_lines if coverage[address] != full_mask)
        payload_read = partial_lines * REQUEST_BYTES
        payload_write = amount * (element_bytes + scale_bytes)
    else:
        raise ValueError(f"unsupported DMA direction {direction!r}")
    return DmaRequestManifest(
        read_lines=read_lines,
        write_lines=write_lines,
        full_lines=full_lines,
        partial_lines=partial_lines,
        payload_read_bytes=payload_read,
        payload_write_bytes=payload_write,
    )


def request_manifest_fixture_hash() -> str:
    fixtures = []
    fmt = MemoryFormat("mxint", 4, 8, 64, "MXINT4")
    for transfer in (
        {
            "direction": "write",
            "element_base": 0,
            "scale_base": 60,
            "dim": 128,
            "amount": 1,
            "stride_bytes": 64,
            "rstride": 1,
        },
        {
            "direction": "write",
            "element_base": 32,
            "scale_base": (1 << 20) + 32,
            "dim": 128,
            "amount": 1,
            "stride_bytes": 64,
            "rstride": 1,
        },
        {
            "direction": "read",
            "element_base": 0,
            "scale_base": 1 << 20,
            "dim": 128,
            "amount": 2,
            "stride_bytes": 64,
            "rstride": 1,
        },
    ):
        fixtures.append(plan_dma_request_manifest(transfer, fmt).to_dict())
    payload = json.dumps(fixtures, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _conditioner_addresses(seed: int, channels: int, count: int = 16) -> list[int]:
    state = hashlib.sha256(f"v4:conditioner:{seed}:c{channels}".encode()).digest()
    result = []
    for index in range(count):
        state = hashlib.sha256(state + index.to_bytes(4, "little")).digest()
        result.append((int.from_bytes(state[:4], "little") % (1 << 20)) * 64)
    return result


def _address_base(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"v4:address:{seed}:{identity}".encode()).digest()
    return (int.from_bytes(digest[:5], "little") % (1 << 22)) * 64


def _split(seed: int, group: str) -> str:
    digest = hashlib.sha256(f"v4:split:{seed}:{group}".encode()).digest()
    return "holdout" if int.from_bytes(digest[:4], "little") % 5 == 0 else "train"


def _physical_formats() -> tuple[dict[str, Any], ...]:
    return (
        {
            "family": "mxint",
            "element_bits": 4,
            "scale_bits": 8,
            "block": 64,
            "name": "E4_S8_B64",
            "exponent_bits": 0,
            "mantissa_bits": 0,
            "equivalent_formats": ["MXINT4", "MXFP_E1M2", "MXFP_E2M1"],
        },
        {
            "family": "mxint",
            "element_bits": 8,
            "scale_bits": 8,
            "block": 64,
            "name": "E8_S8_B64",
            "exponent_bits": 0,
            "mantissa_bits": 0,
            "equivalent_formats": ["MXINT8", "MXFP_E4M3", "MXFP_E5M2"],
        },
        {
            "family": "mxfp",
            "element_bits": 8,
            "scale_bits": 8,
            "block": 8,
            "name": "E8_S8_B8",
            "exponent_bits": 4,
            "mantissa_bits": 3,
            "equivalent_formats": ["MXFP_E4M3", "MXFP_E5M2"],
        },
    )


def generate_hbm_service_v4_plan(
    *,
    seed: int = DEFAULT_SEED,
    repetitions: int = 3,
    channels: Sequence[int] = DEFAULT_CHANNELS,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    max_patterns: int | None = None,
    include_row_state_anchors: bool = False,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    patterns: list[dict[str, Any]] = []
    for channel_count in channels:
        for fmt_data in _physical_formats():
            fmt = _format_from_mapping(fmt_data)
            for dim in dimensions:
                if dim % fmt.block:
                    continue
                element_row_bytes = dim * fmt.element_bits // 8
                matrix_amounts = (dim // 4, dim // 2, dim)
                vector_amounts = tuple(sorted({value for value in (4, 16, 64, 256, 1024, dim) if value <= dim}))
                for opcode, direction, role, amounts in (
                    ("H_PREFETCH_M", "read", "weight", matrix_amounts),
                    ("H_PREFETCH_V", "read", "activation", vector_amounts),
                    ("H_STORE_V", "write", "activation", vector_amounts),
                ):
                    for amount in amounts:
                        for layout in DEFAULT_LAYOUTS:
                            alignment = 32 if layout.startswith("offset32") else 0
                            if layout.endswith("contiguous"):
                                stride_bytes = element_row_bytes
                            elif layout == "aligned_stride2x":
                                stride_bytes = 2 * element_row_bytes
                            else:
                                stride_bytes = max(8192, 4 * element_row_bytes)
                            identity = (
                                f"{opcode}:c{channel_count}:{fmt.request_signature()}:"
                                f"d{dim}:a{amount}:s{stride_bytes}:{layout}"
                            )
                            element_base = _address_base(seed, identity) + alignment
                            scale_base = (1 << 36) + _address_base(seed, f"scale:{identity}") + alignment
                            transfer = {
                                "opcode": opcode,
                                "direction": direction,
                                "precision": role,
                                "element_base": element_base,
                                "scale_base": scale_base,
                                "dim": int(dim),
                                "amount": int(amount),
                                "stride_bytes": int(stride_bytes),
                                "rstride": 1,
                                "write_amount": int(amount if opcode == "H_PREFETCH_M" else 1),
                            }
                            manifest = plan_dma_request_manifest(transfer, fmt)
                            group = f"{opcode}:c{channel_count}:{fmt.request_signature()}:d{dim}:{layout}:a{amount}"
                            pattern_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
                            format_payload = {
                                key: value for key, value in fmt_data.items() if key != "equivalent_formats"
                            }
                            patterns.append(
                                {
                                    "id": f"v4-{pattern_id}",
                                    "group": group,
                                    "split": _split(seed, group),
                                    "channels": int(channel_count),
                                    "repetitions": int(repetitions),
                                    "warmup": 0,
                                    "precision_role": role,
                                    "equivalent_formats": list(fmt_data["equivalent_formats"]),
                                    "format": format_payload,
                                    "transfer": transfer,
                                    "repeat_axes": [],
                                    "conditioner_addresses": _conditioner_addresses(seed, int(channel_count)),
                                    "run_transactional": True,
                                    "run_raw": False,
                                    "stream_family": layout,
                                    "expected_request_manifest": manifest.to_dict(),
                                }
                            )
    if include_row_state_anchors:
        # Small, targeted anchors separate fixed phase startup from row
        # miss/conflict cost.  Generic points use unrelated conditioner
        # addresses, so without these anchors both effects are collinear and
        # the nonnegative fit overestimates repeated Qwen vector prefetches.
        for channel_count in channels:
            for fmt_data in _physical_formats():
                fmt = _format_from_mapping(fmt_data)
                format_payload = {key: value for key, value in fmt_data.items() if key != "equivalent_formats"}
                for dim in (64, 128):
                    if dim % fmt.block:
                        continue
                    element_row_bytes = dim * fmt.element_bits // 8
                    for opcode, direction, role, amount in (
                        ("H_PREFETCH_V", "read", "activation", min(16, dim)),
                        ("H_STORE_V", "write", "activation", min(16, dim)),
                    ):
                        identity = f"row-hit:{opcode}:c{channel_count}:{fmt.request_signature()}:d{dim}:a{amount}"
                        element_base = _address_base(seed, identity)
                        scale_base = (1 << 36) + _address_base(seed, f"scale:{identity}")
                        transfer = {
                            "opcode": opcode,
                            "direction": direction,
                            "precision": role,
                            "element_base": element_base,
                            "scale_base": scale_base,
                            "dim": dim,
                            "amount": amount,
                            "stride_bytes": element_row_bytes,
                            "rstride": 1,
                            "write_amount": amount if opcode == "H_PREFETCH_M" else 1,
                        }
                        manifest = plan_dma_request_manifest(transfer, fmt)
                        # Conditioner reads are excluded from the timed
                        # interval but leave the target rows open in Ramulator.
                        conditioner = sorted(set(manifest.read_lines) | set(manifest.write_lines))
                        pattern_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
                        patterns.append(
                            {
                                "id": f"v4-{pattern_id}",
                                "group": identity,
                                "split": "train",
                                "channels": int(channel_count),
                                "repetitions": int(repetitions),
                                "warmup": 0,
                                "precision_role": role,
                                "equivalent_formats": list(fmt_data["equivalent_formats"]),
                                "format": format_payload,
                                "transfer": transfer,
                                "repeat_axes": [],
                                "conditioner_addresses": conditioner,
                                "run_transactional": True,
                                "run_raw": False,
                                "stream_family": "row_hit_anchor",
                                "expected_request_manifest": manifest.to_dict(),
                            }
                        )
    patterns.sort(key=lambda item: item["id"])
    if max_patterns is not None:
        if max_patterns <= 0:
            raise ValueError("max_patterns must be positive")
        patterns = patterns[:max_patterns]
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "ramulator_preset": "HBM2_2Gbps",
        "mapper": "MOP4CLXOR",
        "request_bytes": REQUEST_BYTES,
        "physical_burst_bytes": PHYSICAL_BURST_BYTES,
        "dma_semantic_version": DMA_SEMANTIC_VERSION,
        "feature_semantic_version": (FEATURE_SEMANTIC_VERSION if include_row_state_anchors else "cold-occurrence-v1"),
        "request_manifest_hash_algorithm": MANIFEST_HASH_ALGORITHM,
        "request_manifest_fixture_hash": request_manifest_fixture_hash(),
        "fit_target": "production_dma_completion_cycles",
        "patterns": patterns,
    }


def write_hbm_service_v4_plan(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    plan = generate_hbm_service_v4_plan(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


@dataclass(frozen=True)
class V4FeatureVector:
    theoretical_phase_floor_ns: float
    values: Mapping[str, float]


@dataclass
class Mop4clxorRowState:
    """Open-row state retained across production DMA occurrences.

    Ramulator is not reset between instructions.  Keeping this state explicit
    lets calibration reproduce its conditioner requests and lets validation
    distinguish repeated row hits from cold-bank misses.  Queue occupancy is
    deliberately not represented; V4 remains a post-hoc service surrogate.
    """

    channels: int
    open_rows: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.channels & (self.channels - 1):
            raise ValueError("MOP4CLXOR channels must be a positive power of two")
        self.open_rows = np.full(self.channels * 2 * 4 * 4, -1, dtype=np.int64)


def occurrence_features(
    manifest: DmaRequestManifest,
    transfer: Mapping[str, Any],
    channels: int,
    *,
    row_state: Mop4clxorRowState | None = None,
) -> V4FeatureVector:
    """Build critical-channel features for one production DMA occurrence.

    A channel-averaged byte floor is not a valid lower bound once MOP4CLXOR
    maps an irregular stride onto only a subset of channels.  Completion is
    controlled by the busiest channel, and a partial-line store executes its
    read and write phases sequentially.  The floor below therefore sums the
    busiest-channel burst work for each phase.  Row events are also measured
    on the critical channel instead of averaging them across all configured
    channels; averaging was the main source of the V3/V4-underfit on large
    strides with 32 or 128 channels.

    The simple row-state walk is intentionally a feature extractor, not a
    replacement DRAM simulator.  It preserves read-to-write open-row state for
    RMW stores and lets the nonnegative residual fit learn the remaining
    Ramulator command/drain cost.
    """

    if channels <= 0 or channels & (channels - 1):
        raise ValueError("MOP4CLXOR channels must be a positive power of two")
    if row_state is not None and row_state.channels != channels:
        raise ValueError(f"row-state channels={row_state.channels} do not match channels={channels}")

    # HBM2 has 2 pseudochannels, 4 bank groups and 4 banks.  A dense array is
    # both faster and less error-prone than millions of Python dictionary
    # operations, while retaining the exact open-row state across the read and
    # write phases of a partial-line scatter.
    banks_per_channel = 2 * 4 * 4
    open_rows = (
        row_state.open_rows if row_state is not None else np.full(channels * banks_per_channel, -1, dtype=np.int64)
    )

    def phase(
        lines: Sequence[int],
    ) -> tuple[int, int, int, int, int, int, int]:
        if not lines:
            return 0, 0, 0, 0, 0, 0, 0

        line_addresses = np.asarray(lines, dtype=np.uint64)
        offsets = np.arange(0, REQUEST_BYTES, PHYSICAL_BURST_BYTES, dtype=np.uint64)
        addresses = (line_addresses[:, None] + offsets).reshape(-1)

        # Vectorized form of ``mop4clxor_map``.  Keep this bit extraction next
        # to the scalar implementation in hbm_service_model.py conceptually:
        # calibration tests compare the resulting feature vectors against the
        # checked-in production-DMA data set.
        channel_bits = channels.bit_length() - 1
        channel_mask = channels - 1
        value = addresses >> np.uint64(4)
        column = value & np.uint64(0b11)
        value >>= np.uint64(2)
        channel = value & np.uint64(channel_mask)
        value >>= np.uint64(channel_bits)
        pseudochannel = value & np.uint64(0b1)
        value >>= np.uint64(1)
        bankgroup = value & np.uint64(0b11)
        value >>= np.uint64(2)
        bank = value & np.uint64(0b11)
        value >>= np.uint64(2)
        column |= (value & np.uint64(0b111)) << np.uint64(2)
        value >>= np.uint64(3)
        row = value.astype(np.int64, copy=False)

        channel ^= column & np.uint64(channel_mask)
        pseudochannel ^= (column >> np.uint64(channel_bits)) & np.uint64(0b1)
        bankgroup ^= (column >> np.uint64(channel_bits + 1)) & np.uint64(0b11)
        bank ^= (column >> np.uint64(channel_bits + 3)) & np.uint64(0b11)

        channel_i = channel.astype(np.int64, copy=False)
        pseudo_i = pseudochannel.astype(np.int64, copy=False)
        bankgroup_i = bankgroup.astype(np.int64, copy=False)
        bank_i = bank.astype(np.int64, copy=False)
        pseudo_key = channel_i * 2 + pseudo_i
        bankgroup_key = pseudo_key * 4 + bankgroup_i
        bank_key = bankgroup_key * 4 + bank_i

        channel_load = np.bincount(channel_i, minlength=channels)
        pseudochannel_load = np.bincount(pseudo_key, minlength=channels * 2)
        bankgroup_load = np.bincount(bankgroup_key, minlength=channels * 2 * 4)
        bank_load = np.bincount(bank_key, minlength=channels * banks_per_channel)

        # Stable grouping preserves the original request order within each
        # bank.  Row misses are first accesses to closed banks; conflicts are
        # row changes either from the previous phase or within this phase.
        order = np.argsort(bank_key, kind="stable")
        sorted_bank = bank_key[order]
        sorted_row = row[order]
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(sorted_bank[1:] != sorted_bank[:-1]) + 1,
            )
        )
        ends = np.concatenate((starts[1:] - 1, np.asarray([len(sorted_bank) - 1], dtype=np.int64)))
        group_banks = sorted_bank[starts]
        group_channels = group_banks // banks_per_channel
        previous_rows = open_rows[group_banks]
        first_rows = sorted_row[starts]

        miss_counts = np.bincount(group_channels[previous_rows < 0], minlength=channels)
        initial_conflicts = (previous_rows >= 0) & (previous_rows != first_rows)
        initial_conflict_counts = np.bincount(group_channels[initial_conflicts], minlength=channels)
        internal_conflict_channels = np.asarray([], dtype=np.int64)
        if len(sorted_bank) > 1:
            changed = (sorted_bank[1:] == sorted_bank[:-1]) & (sorted_row[1:] != sorted_row[:-1])
            internal_conflict_channels = sorted_bank[1:][changed] // banks_per_channel
        internal_conflict_counts = np.bincount(internal_conflict_channels, minlength=channels)
        total_conflict_counts = initial_conflict_counts + internal_conflict_counts
        open_rows[sorted_bank[ends]] = sorted_row[ends]

        return (
            int(channel_load.max(initial=0)),
            int(pseudochannel_load.max(initial=0)),
            int(bankgroup_load.max(initial=0)),
            int(bank_load.max(initial=0)),
            int(miss_counts.max(initial=0)),
            int(initial_conflict_counts.max(initial=0)),
            int(total_conflict_counts.max(initial=0)),
        )

    hbm = HbmConfig(channels)
    (
        read_maximum,
        read_pseudochannel_serial,
        read_bankgroup_serial,
        read_bank_serial,
        read_row_misses,
        read_initial_row_conflicts,
        read_row_conflicts,
    ) = phase(manifest.read_lines)
    # Scatter waits for every partial-line read before issuing its writes, so
    # the write walk deliberately inherits the rows opened by the read walk.
    (
        write_maximum,
        write_pseudochannel_serial,
        write_bankgroup_serial,
        write_bank_serial,
        write_row_misses,
        write_initial_row_conflicts,
        write_row_conflicts,
    ) = phase(manifest.write_lines)
    bursts_per_line = REQUEST_BYTES // PHYSICAL_BURST_BYTES
    read_average = len(manifest.read_lines) * bursts_per_line / channels
    write_average = len(manifest.write_lines) * bursts_per_line / channels
    burst_service_ns = PHYSICAL_BURST_BYTES / hbm.channel_bandwidth_bytes_per_ns
    # The local Ramulator2 HBM patch models data-bus occupancy at nBL/2 = 2
    # cycles per pseudo-channel command.  Channel and pseudo-channel limits are
    # both valid lower bounds, so each sequential DMA phase uses the larger.
    read_floor_bursts = max(read_maximum, 2 * read_pseudochannel_serial)
    write_floor_bursts = max(write_maximum, 2 * write_pseudochannel_serial)
    floor = burst_service_ns * (read_floor_bursts + write_floor_bursts)
    has_read = bool(manifest.read_lines)
    has_write = bool(manifest.write_lines)
    return V4FeatureVector(
        theoretical_phase_floor_ns=floor,
        values={
            "read_phase_startup": float(has_read),
            "write_phase_startup": float(has_write),
            "read_write_turnaround": float(has_read and has_write),
            "read_channel_tail": max(0.0, read_maximum - read_average),
            "write_channel_tail": max(0.0, write_maximum - write_average),
            # Same-bank-group CAS commands have nCCDL=2.  Keep the raw
            # critical-group depth as a residual feature; the stricter
            # pseudo-channel bus floor above has already removed unavoidable
            # transfer work from the fit target.
            "read_bankgroup_serial": float(read_bankgroup_serial),
            "write_bankgroup_serial": float(write_bankgroup_serial),
            # Four native bursts make one line-level request. Their fixed
            # first-line cost belongs in the phase-startup terms; only deeper
            # per-bank queues represent scalable serialization pressure.
            "read_bank_serial": float(max(0, read_bank_serial - 4)),
            "write_bank_serial": float(max(0, write_bank_serial - 4)),
            "read_row_miss": float(read_row_misses),
            "write_row_miss": float(write_row_misses),
            "read_row_conflict": float(read_row_conflicts),
            "write_row_conflict": float(write_row_conflicts),
            # Regime-only features.  Keep total conflict as the fitted
            # residual above, but distinguish a row switch inherited from a
            # previous DMA from switches intrinsic to this occurrence.
            "read_initial_row_conflict": float(read_initial_row_conflicts),
            "write_initial_row_conflict": float(write_initial_row_conflicts),
            "sram_dma_drain": (
                math.log2(int(transfer["amount"]) + 1)
                + math.sqrt((len(manifest.read_lines) + len(manifest.write_lines)) / max(1, channels))
            ),
        },
    )


def _nonnegative_ridge(
    matrix: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    *,
    iterations: int = 20_000,
) -> np.ndarray:
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
            denominator = float(values @ values + ridge)
            updated = max(0.0, float(values @ residual) / denominator)
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        if np.max(np.abs(coefficients - previous)) < 1e-10:
            break
    return coefficients / scales


def _conditioned_row_state(pattern: Mapping[str, Any], channels: int) -> Mop4clxorRowState | None:
    """Recreate the open rows used by a targeted calibration anchor."""

    if pattern.get("stream_family") != "row_hit_anchor":
        return None
    state = Mop4clxorRowState(channels)
    conditioner_lines = tuple(
        int(address) // REQUEST_BYTES * REQUEST_BYTES for address in pattern.get("conditioner_addresses", ())
    )
    if conditioner_lines:
        conditioner_manifest = DmaRequestManifest(
            read_lines=conditioner_lines,
            write_lines=(),
            full_lines=len(conditioner_lines),
            partial_lines=0,
            payload_read_bytes=len(conditioner_lines) * REQUEST_BYTES,
            payload_write_bytes=0,
        )
        occurrence_features(
            conditioner_manifest,
            {"amount": len(conditioner_lines)},
            channels,
            row_state=state,
        )
    return state


@dataclass(frozen=True)
class HbmServiceV4Prediction:
    latency_ns: float
    theoretical_phase_floor_ns: float
    calibration_in_domain: bool
    domain_issues: tuple[str, ...]
    extrapolation_ratio: float
    features: Mapping[str, float]
    row_state_regime: str = "cold_or_mixed"


@dataclass(frozen=True)
class HbmServiceV4WorkPrediction:
    """Aggregate of independently predicted production DMA occurrences."""

    latency_ns: float
    theoretical_floor_ns: float
    read_bytes: int
    write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    read_requests: int
    write_requests: int
    opcode_latency_ns: Mapping[str, float]
    stage_latency_ns: Mapping[str, float]
    traffic_breakdown: Mapping[str, Mapping[str, Mapping[str, int]]]
    calibration_in_domain: bool
    domain_issues: tuple[str, ...]
    max_extrapolation_ratio: float
    occurrence_count: int
    row_state_regime_counts: Mapping[str, int]
    stage_theoretical_floor_ns: Mapping[str, float] = field(default_factory=dict)
    aggregation: str = "per_occurrence"
    unique_geometry_count: int = 0
    unique_address_geometry_count: int = 0
    unique_feature_signature_count: int = 0
    scalar_fallback_count: int = 0
    exact_feature_equivalence: bool = False
    logical_occurrence_count: int = 0
    occurrences_elided: int = 0
    stage_opcode_latency_ns: Mapping[str, float] = field(default_factory=dict)
    stage_occurrence_count: Mapping[str, int] = field(default_factory=dict)
    stage_row_state_regime_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class HbmServiceModelV4:
    calibration_id: str
    coefficients: Mapping[str, Mapping[str, float]]
    domains: Mapping[str, Mapping[str, Any]]
    warm_coefficients: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def group_key(opcode: str, channels: int) -> str:
        return f"{opcode}:c{channels}"

    @classmethod
    def load(cls, path: str | Path) -> "HbmServiceModelV4":
        data = json.loads(Path(path).read_text())
        if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported HBM V4 schema {data.get('schema_version')!r}")
        if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("HBM V4 feature schema does not match evaluator")
        warm_coefficients = data.get("warm_coefficients", {})
        if warm_coefficients and tuple(data.get("warm_feature_names", ())) != WARM_FEATURE_NAMES:
            raise ValueError("HBM V4 warm feature schema does not match evaluator")
        compatibility = dict(data.get("compatibility", {}))
        expected = {
            "dma_semantic_version": DMA_SEMANTIC_VERSION,
            "request_manifest_hash_algorithm": MANIFEST_HASH_ALGORITHM,
            "request_manifest_fixture_hash": request_manifest_fixture_hash(),
            "physical_burst_bytes": PHYSICAL_BURST_BYTES,
        }
        mismatches = {
            name: {"artifact": compatibility.get(name), "runtime": value}
            for name, value in expected.items()
            if compatibility.get(name) != value
        }
        if mismatches:
            raise ValueError(f"HBM V4 artifact is incompatible with the production DMA planner: {mismatches}")
        feature_semantic = compatibility.get("feature_semantic_version", "cold-occurrence-v1")
        if feature_semantic not in {
            "cold-occurrence-v1",
            LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION,
            FEATURE_SEMANTIC_VERSION,
        }:
            raise ValueError(f"unsupported HBM V4 feature semantics {feature_semantic!r}")
        return cls(
            calibration_id=str(data["calibration_id"]),
            coefficients={
                str(group): {str(name): float(value) for name, value in values.items()}
                for group, values in data["coefficients"].items()
            },
            domains=dict(data.get("domains", {})),
            warm_coefficients={
                str(group): {str(name): float(value) for name, value in values.items()}
                for group, values in warm_coefficients.items()
            },
            compatibility=compatibility,
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_kind": "production_dma_occurrence_residual_v4",
            "calibration_id": self.calibration_id,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": {group: dict(values) for group, values in self.coefficients.items()},
            "warm_feature_names": list(WARM_FEATURE_NAMES),
            "warm_coefficients": {group: dict(values) for group, values in self.warm_coefficients.items()},
            "domains": dict(self.domains),
            "compatibility": dict(self.compatibility),
            "metadata": dict(self.metadata),
        }

    def predict_occurrence(
        self,
        opcode: str,
        transfer: Mapping[str, Any],
        format_value: MemoryFormat | Mapping[str, Any],
        channels: int,
        *,
        row_state: Mop4clxorRowState | None = None,
    ) -> HbmServiceV4Prediction:
        manifest = plan_dma_request_manifest(transfer, format_value)
        return self.predict_manifest(
            opcode,
            transfer,
            format_value,
            channels,
            manifest,
            row_state=row_state,
        )

    def predict_manifest(
        self,
        opcode: str,
        transfer: Mapping[str, Any],
        format_value: MemoryFormat | Mapping[str, Any],
        channels: int,
        manifest: DmaRequestManifest,
        *,
        row_state: Mop4clxorRowState | None = None,
    ) -> HbmServiceV4Prediction:
        """Predict an occurrence whose canonical request manifest is known.

        Production callers need the manifest both for byte accounting and for
        latency features.  Accepting it here avoids rebuilding the same large
        line set twice while preserving exactly the same prediction path used
        by the calibration and request-parity tests.
        """

        feature_vector = occurrence_features(
            manifest,
            transfer,
            channels,
            row_state=row_state,
        )
        return self.predict_feature_vector(
            opcode,
            transfer,
            format_value,
            channels,
            feature_vector,
        )

    def predict_feature_vector(
        self,
        opcode: str,
        transfer: Mapping[str, Any],
        format_value: MemoryFormat | Mapping[str, Any],
        channels: int,
        feature_vector: V4FeatureVector,
    ) -> HbmServiceV4Prediction:
        """Apply the calibrated residual model to exact sufficient statistics."""

        group = self.group_key(opcode, channels)
        if group not in self.coefficients:
            raise ValueError(f"HBM V4 has no calibrated group {group}")
        zero_miss = (
            bool(feature_vector.values["read_phase_startup"] or feature_vector.values["write_phase_startup"])
            and feature_vector.values["read_row_miss"] == 0.0
            and feature_vector.values["write_row_miss"] == 0.0
        )
        feature_semantics = self.compatibility.get("feature_semantic_version", FEATURE_SEMANTIC_VERSION)
        fully_warm = zero_miss and (
            feature_semantics == LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION
            or (
                feature_vector.values["read_initial_row_conflict"] == 0.0
                and feature_vector.values["write_initial_row_conflict"] == 0.0
            )
        )
        use_warm_model = fully_warm and group in self.warm_coefficients
        selected_coefficients = self.warm_coefficients[group] if use_warm_model else self.coefficients[group]
        selected_features = WARM_FEATURE_NAMES if use_warm_model else FEATURE_NAMES
        estimate = feature_vector.theoretical_phase_floor_ns + sum(
            selected_coefficients.get(name, 0.0) * feature_vector.values[name] for name in selected_features
        )
        regime = "fully_warm" if use_warm_model else "cold_or_mixed"
        group_domain = self.domains.get(group, {})
        domain = group_domain.get("row_state_regimes", {}).get(regime, group_domain)
        issues = []
        ratio = 1.0
        for name, value in feature_vector.values.items():
            limits = domain.get("features", {}).get(name)
            if not limits:
                continue
            lower = float(limits["min"])
            upper = float(limits["max"])
            scale = max(1.0, upper - lower, abs(lower), abs(upper))
            tolerance = 0.05 * scale
            if value < lower - tolerance:
                issues.append(f"{name}={value:g}<min={lower:g}")
                ratio = max(ratio, 1.0 + (lower - value) / scale)
            elif value > upper + tolerance:
                issues.append(f"{name}={value:g}>max={upper:g}")
                ratio = max(ratio, 1.0 + (value - upper) / scale)
        signature = (
            format_value.request_signature()
            if isinstance(format_value, MemoryFormat)
            else _format_from_mapping(format_value).request_signature()
        )
        signatures = set(domain.get("request_signatures", ()))
        if signatures and signature not in signatures:
            issues.append(f"request_signature={signature}")
        return HbmServiceV4Prediction(
            latency_ns=max(feature_vector.theoretical_phase_floor_ns, estimate),
            theoretical_phase_floor_ns=feature_vector.theoretical_phase_floor_ns,
            calibration_in_domain=not issues,
            domain_issues=tuple(issues),
            extrapolation_ratio=ratio,
            features=dict(feature_vector.values),
            row_state_regime=regime,
        )


def stream_occurrence_transfer(stream: PhysicalDmaStream, occurrence_index: int) -> dict[str, Any]:
    """Expand one mixed-radix repeat position without expanding the trace."""

    if occurrence_index < 0 or occurrence_index >= stream.multiplicity:
        raise IndexError(
            f"stream {stream.stream_index} occurrence {occurrence_index} is outside [0, {stream.multiplicity})"
        )
    element_delta = 0
    scale_delta = 0
    remainder = occurrence_index
    for axis in reversed(stream.axes):
        axis_index = remainder % axis.count
        remainder //= axis.count
        element_delta += axis_index * axis.element_byte_delta
        scale_delta += axis_index * axis.scale_byte_delta
    if remainder:
        raise ValueError(f"stream {stream.stream_index} axes do not cover occurrence {occurrence_index}")
    return {
        "opcode": stream.opcode,
        "direction": stream.direction,
        "precision": stream.precision_role,
        "element_base": stream.element_base + element_delta,
        "scale_base": stream.scale_base + scale_delta,
        "dim": stream.dim,
        "amount": stream.amount,
        "stride_bytes": stream.stride_bytes,
        "rstride": stream.rstride,
        "write_amount": stream.write_amount,
    }


@dataclass(frozen=True)
class _OccurrenceEstimate:
    latency_ns: float
    cycles: int
    prediction: HbmServiceV4Prediction
    read_bytes: int
    write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    read_requests: int
    write_requests: int


def _schedule_dma_count(node: Any, memo: dict[int, int] | None = None) -> int:
    """Count DMA instructions in a compressed schedule without expanding it.

    CostEmitter schedule nodes are intentionally consumed by duck typing here
    so the memory model does not acquire a hard import dependency on the
    compiler submodule.  An in-order scheduler can stall issue, but it cannot
    reorder instructions, so this compressed program order is also the exact
    order in which production DMA requests mutate Ramulator open-row state.
    """

    cache = {} if memo is None else memo
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]
    if hasattr(node, "opcode"):
        result = int(getattr(node, "memory_stream_index", None) is not None)
    elif hasattr(node, "children"):
        result = sum(_schedule_dma_count(child, cache) for child in node.children)
    elif hasattr(node, "body") and hasattr(node, "count"):
        result = int(node.count) * _schedule_dma_count(node.body, cache)
    else:
        result = 0
    cache[node_id] = result
    return result


def _iter_schedule_dma_stream_indices(node: Any):
    """Yield dynamic DMA stream indices in exact architectural issue order."""

    if hasattr(node, "opcode"):
        stream_index = getattr(node, "memory_stream_index", None)
        if stream_index is not None:
            yield int(stream_index)
        return
    if hasattr(node, "children"):
        for child in node.children:
            if _schedule_dma_count(child):
                yield from _iter_schedule_dma_stream_indices(child)
        return
    if hasattr(node, "body") and hasattr(node, "count"):
        if not _schedule_dma_count(node.body):
            return
        for _ in range(int(node.count)):
            yield from _iter_schedule_dma_stream_indices(node.body)


class V4DmaServiceProvider:
    """Serve and account for one V4 estimate per scheduled DMA occurrence.

    Positions are maintained independently per compiler memory stream.  This
    avoids V3's aggregate-stream division and makes missing or extra schedule
    consumption an explicit error.  Occurrence estimates are cached by exact
    physical geometry; repeated addresses therefore remain cheap without
    changing the request semantics.
    """

    def __init__(
        self,
        trace: Any,
        precision_config: MemoryPrecisionConfig | Mapping[str, Any],
        hbm_config: HbmConfig,
        model: HbmServiceModelV4,
        clock_period_ps: int,
        *,
        prepare_global_row_state: bool = True,
    ) -> None:
        if clock_period_ps <= 0:
            raise ValueError("clock_period_ps must be positive")
        self.model = model
        self.hbm = hbm_config
        self.clock_period_ps = int(clock_period_ps)
        self.streams = {
            stream.stream_index: (stream, fmt) for stream, fmt in build_physical_dma_streams(trace, precision_config)
        }
        self.positions: Counter[int] = Counter()
        self._cache: dict[tuple[Any, ...], _OccurrenceEstimate] = {}
        self._manifest_cache: dict[tuple[Any, ...], DmaRequestManifest] = {}
        self._stateful_estimates: dict[int, tuple[_OccurrenceEstimate, ...]] = {}
        self._consumed_latency_ns: Counter[str] = Counter()
        self._consumed_floor_ns: Counter[str] = Counter()
        self._domain_issues: set[str] = set()
        self._max_extrapolation_ratio = 1.0
        self._cycle_sequences: dict[int, tuple[int, ...]] = {}
        self._cycle_periods: dict[int, int] = {}
        self._ordered_stream_indices: tuple[int, ...] = ()
        self.row_state_semantics = "cold_geometry_cached_occurrence"
        if prepare_global_row_state and model.compatibility.get("feature_semantic_version") == FEATURE_SEMANTIC_VERSION:
            self._prepare_global_row_state_sequences(trace)

    @staticmethod
    def _fundamental_period(values: Sequence[int]) -> int:
        """Return the exact period of a finite sequence, or its full length."""

        if not values:
            return 1
        prefix = [0] * len(values)
        for index in range(1, len(values)):
            candidate = prefix[index - 1]
            while candidate and values[index] != values[candidate]:
                candidate = prefix[candidate - 1]
            if values[index] == values[candidate]:
                candidate += 1
            prefix[index] = candidate
        period = len(values) - prefix[-1]
        return period if len(values) % period == 0 else len(values)

    def _key(
        self,
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
        transfer: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        # Preserve every MOP4CLXOR field below the DRAM row.  The lower address
        # consists of the 16-byte native-transfer offset, five low/high column
        # bits, channel bits, pseudochannel, bank-group and bank.  Translating
        # both streams by this period only shifts row numbers.
        mapper_row_period = 16_384 * self.hbm.channels
        element_base = int(transfer["element_base"])
        scale_base = int(transfer["scale_base"])
        dim = int(transfer["dim"])
        amount = int(transfer["amount"])
        rstride = int(transfer["rstride"])
        element_row_bytes = fmt.element_bits * dim // 8
        stride_bytes = int(transfer["stride_bytes"])
        if rstride != 1:
            stride_bytes = element_row_bytes

        # Cold V4 starts with every bank closed.  Absolute row numbers and the
        # distance between two disjoint row intervals therefore cannot affect
        # its features: only lower MOP4CLXOR fields, within-region row changes,
        # and possible element/scale row equality matter.  Treat far-separated
        # MX regions as one exact equivalence class instead of retaining an
        # arbitrary tensor-allocation distance in the cache key.
        if not fmt.is_mx:
            scale_base_key = 0
            relative_row_relation: int | str = "no_scale"
        else:
            scale_base_key = scale_base % mapper_row_period
            stride_elements = stride_bytes * 8 // fmt.element_bits
            scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
            scale_row_bytes = fmt.scale_bits * (dim // fmt.block) // 8
            element_last = element_base + max(0, amount - 1) * stride_bytes + max(0, element_row_bytes - 1)
            scale_last = scale_base + max(0, amount - 1) * scale_stride + max(0, scale_row_bytes - 1)
            element_rows = (
                element_base // mapper_row_period,
                element_last // mapper_row_period,
            )
            scale_rows = (
                scale_base // mapper_row_period,
                scale_last // mapper_row_period,
            )
            if element_rows[1] < scale_rows[0] or scale_rows[1] < element_rows[0]:
                relative_row_relation = "disjoint"
            else:
                relative_row_relation = scale_base // mapper_row_period - element_base // mapper_row_period
        return (
            stream.opcode,
            stream.stage,
            fmt.request_signature(),
            element_base % mapper_row_period,
            scale_base_key,
            relative_row_relation,
            dim,
            amount,
            int(transfer["stride_bytes"]),
            rstride,
        )

    @staticmethod
    def _manifest_key(
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
        transfer: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        """Return an exact-address key for the canonical physical manifest.

        Cold feature prediction may normalize a geometry by whole DRAM rows,
        but stateful prediction must retain absolute row numbers.  Reusing the
        normalized feature key here would silently make translated tensors
        access the first occurrence's rows.
        """

        return (
            stream.opcode,
            fmt.request_signature(),
            int(transfer["element_base"]),
            int(transfer["scale_base"]),
            int(transfer["dim"]),
            int(transfer["amount"]),
            int(transfer["stride_bytes"]),
            int(transfer["rstride"]),
        )

    def _build_estimate(
        self,
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
        transfer: Mapping[str, Any],
        *,
        row_state: Mop4clxorRowState | None = None,
        retain_manifest: bool = True,
    ) -> _OccurrenceEstimate:
        key = self._manifest_key(stream, fmt, transfer)
        manifest = self._manifest_cache.get(key)
        if manifest is None:
            manifest = plan_dma_request_manifest(transfer, fmt)
            if retain_manifest:
                self._manifest_cache[key] = manifest
        prediction = self.model.predict_manifest(
            stream.opcode,
            transfer,
            fmt,
            self.hbm.channels,
            manifest,
            row_state=row_state,
        )
        latency_ns = prediction.latency_ns
        return _OccurrenceEstimate(
            latency_ns=latency_ns,
            cycles=max(1, math.ceil(latency_ns * 1000.0 / self.clock_period_ps)),
            prediction=prediction,
            read_bytes=manifest.read_bytes,
            write_bytes=manifest.write_bytes,
            payload_read_bytes=manifest.payload_read_bytes,
            payload_write_bytes=manifest.payload_write_bytes,
            read_requests=len(manifest.read_lines),
            write_requests=len(manifest.write_lines),
        )

    def _prepare_global_row_state_sequences(self, trace: Any) -> None:
        """Bake global MOP4CLXOR row state into per-stream occurrence arrays.

        The arrays are generated in dynamic program order, before timing
        simulation.  The scheduler later consumes them by stream position.
        This is equivalent because issue is in order, while allowing the
        scheduler to fast-forward exact repeats without reconstructing the
        inter-stream row-state transition from unordered occurrence counts.
        """

        unavailable = getattr(trace, "schedule_unavailable_reasons", {})
        if unavailable:
            raise ValueError(
                f"stateful HBM V4 requires an ordered CostTrace schedule; unavailable reasons: {dict(unavailable)}"
            )
        schedule = getattr(trace, "schedule", None)
        if schedule is None:
            raise ValueError("stateful HBM V4 requires CostTrace.schedule")
        expected = sum(stream.multiplicity for stream, _fmt in self.streams.values())
        scheduled = _schedule_dma_count(schedule)
        if scheduled != expected:
            raise ValueError(
                "stateful HBM V4 schedule coverage mismatch: "
                f"scheduled={scheduled}, physical_stream_occurrences={expected}"
            )

        positions: Counter[int] = Counter()
        estimates: dict[int, list[_OccurrenceEstimate]] = {stream_index: [] for stream_index in self.streams}
        row_state = Mop4clxorRowState(self.hbm.channels)
        ordered_stream_indices = tuple(_iter_schedule_dma_stream_indices(schedule))
        for stream_index in ordered_stream_indices:
            if stream_index not in self.streams:
                raise ValueError(f"ordered schedule references unknown V4 DMA stream {stream_index}")
            stream, fmt = self.streams[stream_index]
            position = positions[stream_index]
            if position >= stream.multiplicity:
                raise ValueError(f"ordered schedule over-consumes V4 DMA stream {stream_index}")
            transfer = stream_occurrence_transfer(stream, position)
            estimates[stream_index].append(
                self._build_estimate(
                    stream,
                    fmt,
                    transfer,
                    row_state=row_state,
                    # Stateful preparation visits every exact-address
                    # occurrence once and retains the resulting scalar
                    # estimate below.  Keeping every manifest as well would
                    # pin all of its 64-B line tuples for the provider's
                    # lifetime, which is several GiB for production MoE
                    # traces and provides no benefit to aggregate/scheduler
                    # evaluation.  ``ordered_manifests`` can still rebuild a
                    # manifest on demand for the dedicated parity tools.
                    retain_manifest=False,
                )
            )
            positions[stream_index] += 1

        missing = {
            stream_index: stream.multiplicity - positions[stream_index]
            for stream_index, (stream, _fmt) in self.streams.items()
            if positions[stream_index] != stream.multiplicity
        }
        if missing:
            raise ValueError(f"ordered schedule under-consumes V4 DMA streams: {missing}")
        self._stateful_estimates = {stream_index: tuple(values) for stream_index, values in estimates.items()}
        self._cycle_sequences = {
            stream_index: tuple(estimate.cycles for estimate in values)
            for stream_index, values in self._stateful_estimates.items()
        }
        self._cycle_periods = {
            stream_index: self._fundamental_period(values) for stream_index, values in self._cycle_sequences.items()
        }
        self._ordered_stream_indices = ordered_stream_indices
        self.row_state_semantics = "global_issue_order_precomputed"

    def ordered_estimates(self):
        """Yield stream metadata and estimates in dynamic DMA issue order.

        This diagnostic iterator does not consume scheduler positions.  It is
        used to attribute system-validation residuals to physical compiler
        streams while preserving the same occurrence estimates used by the
        scheduled shadow.
        """

        if not self._ordered_stream_indices:
            raise ValueError("ordered V4 estimates require global issue-order row semantics")
        positions: Counter[int] = Counter()
        for stream_index in self._ordered_stream_indices:
            position = positions[stream_index]
            stream, fmt = self.streams[stream_index]
            yield stream, fmt, position, self._estimate(stream_index, position)
            positions[stream_index] += 1

    def ordered_manifests(self):
        """Yield cached physical manifests in dynamic DMA issue order.

        Stateful V4 preparation already materializes each exact-address
        manifest to derive row-state features.  Returning references to those
        cached objects lets system validation compute a canonical sequence
        hash without copying their potentially large line arrays.
        """

        if not self._ordered_stream_indices:
            raise ValueError("ordered V4 manifests require global issue-order row semantics")
        positions: Counter[int] = Counter()
        for stream_index in self._ordered_stream_indices:
            position = positions[stream_index]
            stream, fmt = self.streams[stream_index]
            transfer = stream_occurrence_transfer(stream, position)
            key = self._manifest_key(stream, fmt, transfer)
            manifest = self._manifest_cache.get(key)
            if manifest is None:
                manifest = plan_dma_request_manifest(transfer, fmt)
                self._manifest_cache[key] = manifest
            yield manifest
            positions[stream_index] += 1

    def _estimate(self, stream_index: int, position: int) -> _OccurrenceEstimate:
        try:
            stream, fmt = self.streams[stream_index]
        except KeyError as exc:
            raise ValueError(f"no V4 DMA stream {stream_index}") from exc
        if self._stateful_estimates:
            try:
                return self._stateful_estimates[stream_index][position]
            except IndexError as exc:
                raise ValueError(f"no stateful V4 estimate for stream {stream_index} occurrence {position}") from exc
        transfer = stream_occurrence_transfer(stream, position)
        return self._estimate_transfer(stream, fmt, transfer)

    def _estimate_transfer(
        self,
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
        transfer: Mapping[str, Any],
    ) -> _OccurrenceEstimate:
        """Predict one cold transfer while sharing normalized geometry work."""

        key = self._key(stream, fmt, transfer)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        estimate = self._build_estimate(stream, fmt, transfer)
        self._cache[key] = estimate
        return estimate

    @staticmethod
    def _axis_residue_distribution(
        *,
        count: int,
        element_delta: int,
        scale_delta: int,
        modulus: int,
    ) -> dict[tuple[int, int], tuple[int, int, int]]:
        """Return exact affine residue counts and one representative delta."""

        element_period = modulus // math.gcd(modulus, abs(element_delta))
        scale_period = modulus // math.gcd(modulus, abs(scale_delta))
        period = math.lcm(element_period, scale_period)
        full, tail = divmod(count, period)
        result: dict[tuple[int, int], tuple[int, int, int]] = {}
        for index in range(min(count, period)):
            occurrences = full + int(index < tail)
            if not occurrences:
                continue
            raw_element = index * element_delta
            raw_scale = index * scale_delta
            key = (raw_element % modulus, raw_scale % modulus)
            previous = result.get(key)
            if previous is None:
                result[key] = (occurrences, raw_element, raw_scale)
            else:
                result[key] = (
                    previous[0] + occurrences,
                    previous[1],
                    previous[2],
                )
        return result

    def _cold_geometry_groups(
        self,
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
    ) -> tuple[
        dict[tuple[Any, ...], tuple[dict[str, Any], int]],
        bool,
    ]:
        """Group a cold affine DMA stream without visiting every occurrence."""

        mapper_row_period = 16_384 * self.hbm.channels
        if not stream.axes:
            transfer = stream_occurrence_transfer(stream, 0)
            return {self._key(stream, fmt, transfer): (transfer, 1)}, False

        # The normalized cold key may discard the relative row distance only
        # when the complete element and scale address envelopes cannot share a
        # DRAM row.  Otherwise retain the conservative per-occurrence path.
        if fmt.is_mx:
            element_min_delta = sum(min(0, (axis.count - 1) * axis.element_byte_delta) for axis in stream.axes)
            element_max_delta = sum(max(0, (axis.count - 1) * axis.element_byte_delta) for axis in stream.axes)
            scale_min_delta = sum(min(0, (axis.count - 1) * axis.scale_byte_delta) for axis in stream.axes)
            scale_max_delta = sum(max(0, (axis.count - 1) * axis.scale_byte_delta) for axis in stream.axes)
            element_row_bytes = fmt.element_bits * stream.dim // 8
            stride_bytes = element_row_bytes if stream.rstride != 1 else stream.stride_bytes
            stride_elements = stride_bytes * 8 // fmt.element_bits
            scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
            scale_row_bytes = fmt.scale_bits * (stream.dim // fmt.block) // 8
            element_range = (
                stream.element_base + element_min_delta,
                stream.element_base
                + element_max_delta
                + max(0, stream.amount - 1) * stride_bytes
                + max(0, element_row_bytes - 1),
            )
            scale_range = (
                stream.scale_base + scale_min_delta,
                stream.scale_base
                + scale_max_delta
                + max(0, stream.amount - 1) * scale_stride
                + max(0, scale_row_bytes - 1),
            )
            disjoint_rows = (
                element_range[1] // mapper_row_period < scale_range[0] // mapper_row_period
                or scale_range[1] // mapper_row_period < element_range[0] // mapper_row_period
            )
            if not disjoint_rows:
                groups: dict[tuple[Any, ...], tuple[dict[str, Any], int]] = {}
                for position in range(stream.multiplicity):
                    transfer = stream_occurrence_transfer(stream, position)
                    key = self._key(stream, fmt, transfer)
                    previous = groups.get(key)
                    groups[key] = (
                        transfer if previous is None else previous[0],
                        1 if previous is None else previous[1] + 1,
                    )
                return groups, True

        states: dict[tuple[int, int], tuple[int, int, int]] = {(0, 0): (1, 0, 0)}
        for axis in stream.axes:
            additions = self._axis_residue_distribution(
                count=axis.count,
                element_delta=axis.element_byte_delta,
                scale_delta=axis.scale_byte_delta if fmt.is_mx else 0,
                modulus=mapper_row_period,
            )
            combined: dict[tuple[int, int], tuple[int, int, int]] = {}
            for (element, scale), (
                state_count,
                state_element_rep,
                state_scale_rep,
            ) in states.items():
                for (element_add, scale_add), (
                    axis_count,
                    axis_element_rep,
                    axis_scale_rep,
                ) in additions.items():
                    key = (
                        (element + element_add) % mapper_row_period,
                        (scale + scale_add) % mapper_row_period,
                    )
                    previous = combined.get(key)
                    if previous is None:
                        combined[key] = (
                            state_count * axis_count,
                            state_element_rep + axis_element_rep,
                            state_scale_rep + axis_scale_rep,
                        )
                    else:
                        combined[key] = (
                            previous[0] + state_count * axis_count,
                            previous[1],
                            previous[2],
                        )
            states = combined

        groups = {}
        for _residue, (count, element_delta, scale_delta) in states.items():
            transfer = {
                "opcode": stream.opcode,
                "direction": stream.direction,
                "precision": stream.precision_role,
                "element_base": stream.element_base + element_delta,
                "scale_base": stream.scale_base + scale_delta,
                "dim": stream.dim,
                "amount": stream.amount,
                "stride_bytes": stream.stride_bytes,
                "rstride": stream.rstride,
                "write_amount": stream.write_amount,
            }
            key = self._key(stream, fmt, transfer)
            previous = groups.get(key)
            groups[key] = (
                transfer if previous is None else previous[0],
                count if previous is None else previous[1] + count,
            )
        if sum(count for _transfer, count in groups.values()) != stream.multiplicity:
            raise ValueError(f"affine V4 grouping lost multiplicity for stream {stream.stream_index}")
        return groups, False

    @staticmethod
    def _relative_region_lines(
        *,
        base_alignment: int,
        row_bytes: int,
        stride_bytes: int,
        amount: int,
    ) -> np.ndarray:
        coverage = _line_coverage(
            tuple(
                (
                    base_alignment + row * stride_bytes,
                    row_bytes,
                )
                for row in range(amount)
            )
        )
        return np.asarray(sorted(coverage), dtype=np.uint64)

    def _vectorized_cold_read_feature_groups(
        self,
        stream: PhysicalDmaStream,
        fmt: MemoryFormat,
        groups: Mapping[
            tuple[Any, ...],
            tuple[dict[str, Any], int],
        ],
        *,
        geometry_batch_size: int,
    ) -> (
        tuple[
            dict[tuple[Any, ...], tuple[_OccurrenceEstimate, int]],
            dict[tuple[Any, ...], _OccurrenceEstimate],
        ]
        | None
    ):
        """Collapse exact cold read geometries by batched MOP4CLXOR features.

        The scalar planner creates the same line-run shape repeatedly at
        different affine base addresses.  This path maps a batch of those
        translated line runs at once and reduces them to the exact feature
        vector consumed by V4.  It changes neither request coverage nor model
        coefficients.  Writes retain the scalar path because their partial
        line read-modify-write phase carries row state into the write phase.
        """

        if stream.direction != "read":
            return None
        channels = self.hbm.channels
        banks_per_channel = 2 * 4 * 4
        native_bursts_per_line = REQUEST_BYTES // PHYSICAL_BURST_BYTES
        channel_bits = channels.bit_length() - 1
        channel_mask = channels - 1

        dim = int(stream.dim)
        amount = int(stream.amount)
        element_row_bytes = fmt.element_bits * dim // 8
        stride_bytes = element_row_bytes if int(stream.rstride) != 1 else int(stream.stride_bytes)
        if fmt.is_mx:
            stride_elements = stride_bytes * 8 // fmt.element_bits
            scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
            scale_row_bytes = fmt.scale_bits * (dim // fmt.block) // 8
        else:
            scale_stride = 0
            scale_row_bytes = 0

        by_alignment: dict[
            tuple[int, int, int | str],
            list[tuple[dict[str, Any], int]],
        ] = defaultdict(list)
        for geometry_key, (transfer, count) in groups.items():
            by_alignment[
                (
                    int(transfer["element_base"]) % REQUEST_BYTES,
                    (int(transfer["scale_base"]) % REQUEST_BYTES if fmt.is_mx else 0),
                    geometry_key[5],
                )
            ].append((transfer, count))

        feature_groups: dict[
            tuple[Any, ...],
            tuple[_OccurrenceEstimate, int],
        ] = {}
        address_estimates: dict[tuple[Any, ...], _OccurrenceEstimate] = {}
        for (
            element_alignment,
            scale_alignment,
            _relative_row_relation,
        ), aligned in by_alignment.items():
            representative_transfer = aligned[0][0]
            element_origin = int(representative_transfer["element_base"]) - element_alignment
            element_lines = self._relative_region_lines(
                base_alignment=element_alignment,
                row_bytes=element_row_bytes,
                stride_bytes=stride_bytes,
                amount=amount,
            )
            scale_lines = (
                self._relative_region_lines(
                    # Express the scale line-run in the element origin's
                    # coordinate system.  For overlapping MX regions this
                    # lets np.unique perform exactly the same 64-B request
                    # coalescing as the scalar manifest planner.  For
                    # disjoint regions, whole mapper-row translations retain
                    # identical cold features.
                    base_alignment=(int(representative_transfer["scale_base"]) - element_origin),
                    row_bytes=scale_row_bytes,
                    stride_bytes=scale_stride,
                    amount=amount,
                )
                if scale_row_bytes
                else np.asarray([], dtype=np.uint64)
            )
            relative_lines = (
                np.unique(np.concatenate((element_lines, scale_lines))) if len(scale_lines) else element_lines
            )
            line_count = len(relative_lines)
            if line_count <= 0:
                return None
            burst_count = line_count * native_bursts_per_line
            # Keep all temporary mapping/load arrays comfortably below the
            # DSE's 2 GiB worker target even for wide matrix transfers.
            batch_size = min(
                geometry_batch_size,
                max(1, 2_000_000 // max(1, burst_count)),
                # One exact load census has 4096 physical banks for the
                # production 128-channel configuration.  Bounding this dense
                # tensor avoids trading the removed sort/scan work for a
                # large transient allocation on small DMA geometries.
                max(1, 4_000_000 // max(1, channels * banks_per_channel)),
            )
            read_average = burst_count / channels
            burst_service_ns = PHYSICAL_BURST_BYTES / self.hbm.channel_bandwidth_bytes_per_ns
            payload_read_bytes = amount * (element_row_bytes + scale_row_bytes)
            drain = math.log2(amount + 1) + math.sqrt(line_count / max(1, channels))

            for start in range(0, len(aligned), batch_size):
                batch = aligned[start : start + batch_size]
                transfers = [item[0] for item in batch]
                weights = np.asarray(
                    [item[1] for item in batch],
                    dtype=np.int64,
                )
                element_origins = np.asarray(
                    [int(transfer["element_base"]) - element_alignment for transfer in transfers],
                    dtype=np.uint64,
                )
                lines = element_origins[:, None] + relative_lines[None, :]

                offsets = np.arange(
                    0,
                    REQUEST_BYTES,
                    PHYSICAL_BURST_BYTES,
                    dtype=np.uint64,
                )
                addresses = (lines[:, :, None] + offsets[None, None, :]).reshape(len(batch), -1)
                value = addresses >> np.uint64(4)
                column = value & np.uint64(0b11)
                value >>= np.uint64(2)
                channel = value & np.uint64(channel_mask)
                value >>= np.uint64(channel_bits)
                pseudo = value & np.uint64(0b1)
                value >>= np.uint64(1)
                bankgroup = value & np.uint64(0b11)
                value >>= np.uint64(2)
                bank = value & np.uint64(0b11)
                value >>= np.uint64(2)
                column |= (value & np.uint64(0b111)) << np.uint64(2)
                value >>= np.uint64(3)
                row = value.astype(np.int64, copy=False)

                channel ^= column & np.uint64(channel_mask)
                pseudo ^= (column >> np.uint64(channel_bits)) & np.uint64(0b1)
                bankgroup ^= (column >> np.uint64(channel_bits + 1)) & np.uint64(0b11)
                bank ^= (column >> np.uint64(channel_bits + 3)) & np.uint64(0b11)

                bank_key = (
                    ((channel * np.uint64(2) + pseudo) * np.uint64(4) + bankgroup) * np.uint64(4) + bank
                ).astype(np.int64, copy=False)

                # Cold grouped prediction needs the maximum load at each DRAM
                # hierarchy and the number of initially closed banks per
                # channel.  A single exact bank census supplies all five
                # values.  The former implementation sorted and scanned the
                # full burst array four times; that was algebraically
                # equivalent but dominated long-context evaluation.
                batch_ids = np.arange(len(batch), dtype=np.int64)[:, None]
                flat_bank = (bank_key + batch_ids * (channels * banks_per_channel)).reshape(-1)
                bank_load = np.bincount(
                    flat_bank,
                    minlength=len(batch) * channels * banks_per_channel,
                ).reshape(len(batch), channels, 2, 4, 4)
                bank_max = bank_load.max(axis=(1, 2, 3, 4))
                bankgroup_max = bank_load.sum(axis=4).max(axis=(1, 2, 3))
                pseudo_max = bank_load.sum(axis=(3, 4)).max(axis=(1, 2))
                channel_max = bank_load.sum(axis=(2, 3, 4)).max(axis=1)
                row_miss_max = (bank_load > 0).sum(axis=(2, 3, 4)).max(axis=1)

                conflict_counts = np.zeros(
                    (len(batch), channels),
                    dtype=np.int32,
                )
                # ``row`` contains only address high bits and is monotonic in
                # the physical request walk.  Consequently, a bank's exact
                # internal conflict count is the number of distinct DRAM rows
                # it touches minus one.  Most compiler DMA geometries stay
                # within one or two aggregate DRAM rows, so a compact
                # row-by-bank census avoids sorting every burst.  Retain the
                # stable-sort implementation for unusually sparse, very wide
                # row spans where a dense census would waste memory.
                for batch_index in range(len(batch)):
                    row_values = row[batch_index]
                    row_min = int(row_values[0])
                    row_max = int(row_values[-1])
                    if row_min == row_max:
                        continue
                    row_span = row_max - row_min + 1
                    if row_span <= 64:
                        row_bank = (row_values - row_min) * (channels * banks_per_channel) + bank_key[batch_index]
                        row_bank_load = np.bincount(
                            row_bank,
                            minlength=(row_span * channels * banks_per_channel),
                        ).reshape(
                            row_span,
                            channels,
                            banks_per_channel,
                        )
                        rows_per_bank = (row_bank_load > 0).sum(axis=0)
                        conflict_counts[batch_index] = np.maximum(
                            rows_per_bank - 1,
                            0,
                        ).sum(axis=1)
                        continue

                    order = np.argsort(
                        bank_key[batch_index],
                        kind="stable",
                    )
                    sorted_bank = bank_key[batch_index, order]
                    sorted_row = row_values[order]
                    changed = (sorted_bank[1:] == sorted_bank[:-1]) & (sorted_row[1:] != sorted_row[:-1])
                    conflict_channels = sorted_bank[1:][changed] // banks_per_channel
                    conflict_counts[batch_index] = np.bincount(
                        conflict_channels,
                        minlength=channels,
                    )
                conflict_max = conflict_counts.max(axis=1)

                compact = np.stack(
                    (
                        channel_max,
                        pseudo_max,
                        bankgroup_max,
                        bank_max,
                        row_miss_max,
                        conflict_max,
                    ),
                    axis=1,
                )
                unique, inverse = np.unique(
                    compact,
                    axis=0,
                    return_inverse=True,
                )
                weighted_counts = np.zeros(len(unique), dtype=np.int64)
                np.add.at(weighted_counts, inverse, weights)
                representative = transfers[0]
                unique_estimates: list[_OccurrenceEstimate] = []
                for values, count in zip(
                    unique,
                    weighted_counts,
                    strict=True,
                ):
                    (
                        read_maximum,
                        read_pseudo_serial,
                        read_bankgroup_serial,
                        read_bank_serial,
                        read_row_miss,
                        read_row_conflict,
                    ) = (int(value) for value in values)
                    floor_bursts = max(
                        read_maximum,
                        2 * read_pseudo_serial,
                    )
                    feature_vector = V4FeatureVector(
                        theoretical_phase_floor_ns=(burst_service_ns * floor_bursts),
                        values={
                            "read_phase_startup": 1.0,
                            "write_phase_startup": 0.0,
                            "read_write_turnaround": 0.0,
                            "read_channel_tail": max(0.0, read_maximum - read_average),
                            "write_channel_tail": 0.0,
                            "read_bankgroup_serial": float(read_bankgroup_serial),
                            "write_bankgroup_serial": 0.0,
                            "read_bank_serial": float(max(0, read_bank_serial - 4)),
                            "write_bank_serial": 0.0,
                            "read_row_miss": float(read_row_miss),
                            "write_row_miss": 0.0,
                            "read_row_conflict": float(read_row_conflict),
                            "write_row_conflict": 0.0,
                            "read_initial_row_conflict": 0.0,
                            "write_initial_row_conflict": 0.0,
                            "sram_dma_drain": drain,
                        },
                    )
                    prediction = self.model.predict_feature_vector(
                        stream.opcode,
                        representative,
                        fmt,
                        channels,
                        feature_vector,
                    )
                    estimate = _OccurrenceEstimate(
                        latency_ns=prediction.latency_ns,
                        cycles=max(
                            1,
                            math.ceil(prediction.latency_ns * 1000.0 / self.clock_period_ps),
                        ),
                        prediction=prediction,
                        read_bytes=line_count * REQUEST_BYTES,
                        write_bytes=0,
                        payload_read_bytes=payload_read_bytes,
                        payload_write_bytes=0,
                        read_requests=line_count,
                        write_requests=0,
                    )
                    signature = (
                        line_count,
                        payload_read_bytes,
                        tuple(int(value) for value in values),
                    )
                    previous = feature_groups.get(signature)
                    feature_groups[signature] = (
                        estimate if previous is None else previous[0],
                        int(count) if previous is None else previous[1] + int(count),
                    )
                    unique_estimates.append(estimate)
                for index, transfer in enumerate(transfers):
                    key = self._key(stream, fmt, transfer)
                    # Stage attribution is applied by the current stream and
                    # is not part of the physical V4 prediction.
                    address_estimates[key[:1] + key[2:]] = unique_estimates[int(inverse[index])]
        return feature_groups, address_estimates

    def __call__(self, instruction: Any, _sequence: int) -> int:
        return self.timing_estimate(instruction, _sequence).resource_cycles

    def timing_estimate(self, instruction: Any, _sequence: int) -> OpcodeTimingEstimate:
        """Consume one occurrence and expose its V4 fidelity to the scheduler."""

        stream_index = instruction.memory_stream_index
        if stream_index is None or stream_index not in self.streams:
            raise ValueError(f"no V4 DMA estimate for stream {stream_index!r}")
        stream, _fmt = self.streams[stream_index]
        position = self.positions[stream_index]
        if position >= stream.multiplicity:
            raise ValueError(f"V4 DMA stream {stream_index} consumed more than {stream.multiplicity} occurrences")
        estimate = self._estimate(stream_index, position)
        self.positions[stream_index] += 1
        self._record(stream, estimate)
        prediction = estimate.prediction
        return OpcodeTimingEstimate(
            resource_cycles=estimate.cycles,
            result_ready_cycles=estimate.cycles,
            initiation_interval_cycles=estimate.cycles,
            calibration_status=("post_hoc_v4" if prediction.calibration_in_domain else "structural_extrapolation"),
            rtl_supported=True,
            calibration_in_domain=prediction.calibration_in_domain,
        )

    def _record(self, stream: PhysicalDmaStream, estimate: _OccurrenceEstimate) -> None:
        self._consumed_latency_ns[stream.opcode] += estimate.latency_ns
        self._consumed_floor_ns[stream.opcode] += estimate.prediction.theoretical_phase_floor_ns
        for issue in estimate.prediction.domain_issues:
            self._domain_issues.add(f"stream={stream.stream_index}:{issue}")
        self._max_extrapolation_ratio = max(
            self._max_extrapolation_ratio,
            estimate.prediction.extrapolation_ratio,
        )

    def assert_consumed(self) -> None:
        missing = {
            stream_index: stream.multiplicity - self.positions[stream_index]
            for stream_index, (stream, _fmt) in self.streams.items()
            if self.positions[stream_index] != stream.multiplicity
        }
        if missing:
            raise ValueError(f"V4 DMA occurrence predictions were not fully consumed: {missing}")

    @property
    def supports_exact_fast_forward(self) -> bool:
        return bool(self._cycle_sequences)

    def snapshot_state(self, stream_indices: Sequence[int] | None = None) -> tuple[tuple[int, int, int], ...]:
        """Return absolute positions plus exact periodic timing phases."""

        state = []
        selected = (
            self.streams.items()
            if stream_indices is None
            else ((int(stream_index), self.streams[int(stream_index)]) for stream_index in sorted(stream_indices))
        )
        for stream_index, (stream, _fmt) in sorted(selected):
            position = self.positions[stream_index]
            period = self._cycle_periods.get(stream_index, stream.multiplicity)
            state.append((stream_index, position, position % max(1, period)))
        return tuple(state)

    def advance_stream_counts(self, counts: Mapping[int, int]) -> None:
        """Advance occurrences skipped by an exact scheduler transition."""

        for stream_index, count in counts.items():
            if count < 0:
                raise ValueError("V4 DMA provider cannot move backwards")
            stream, _fmt = self.streams[int(stream_index)]
            next_position = self.positions[int(stream_index)] + int(count)
            if next_position > stream.multiplicity:
                raise ValueError(
                    f"V4 DMA stream {stream_index} fast-forwarded to {next_position}, beyond {stream.multiplicity}"
                )
            self.positions[int(stream_index)] = next_position

    def aggregate(
        self,
        *,
        stage_multipliers: Mapping[str, int] | None = None,
        group_cold_geometries: bool = True,
        aggregation_backend: str = "sufficient-statistics-v2",
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        geometry_batch_size: int = 4096,
    ) -> HbmServiceV4WorkPrediction:
        """Aggregate occurrence estimates, optionally scaling whole stages.

        ``stage_multipliers`` is used by the DSE-only representative-layer
        path.  The provider still predicts every occurrence in the retained
        schedule once, including its production DMA manifest and row-state
        transition, then scales decoder-layer work algebraically.  Full
        system validation never supplies multipliers and therefore retains
        exact global issue-order semantics.
        """

        if aggregation_backend not in {
            "scalar-v1",
            "sufficient-statistics-v2",
        }:
            raise ValueError(
                f"V4 aggregation_backend must be 'scalar-v1' or 'sufficient-statistics-v2', got {aggregation_backend!r}"
            )
        if geometry_batch_size <= 0:
            raise ValueError("geometry_batch_size must be positive")

        opcode_latency: Counter[str] = Counter()
        stage_opcode_latency: Counter[str] = Counter()
        stage_latency: Counter[str] = Counter()
        stage_floor: Counter[str] = Counter()
        latency_histogram: Counter[tuple[str, str, float]] = Counter()
        floor_histogram: Counter[tuple[str, float]] = Counter()
        total_floor = 0.0
        read_bytes = write_bytes = read_requests = write_requests = 0
        payload_read_bytes = payload_write_bytes = 0
        traffic: dict[str, dict[str, Counter[str]]] = {
            "by_role": {},
            "by_stage": {},
            "by_opcode": {},
            "by_stage_role": {},
            "by_opcode_role": {},
            "by_stage_opcode_role": {},
        }

        def add_traffic(
            group: str,
            key: str,
            estimate: _OccurrenceEstimate,
            multiplier: int,
        ) -> None:
            bucket = traffic[group].setdefault(key, Counter())
            bucket["physical_read_bytes"] += estimate.read_bytes * multiplier
            bucket["physical_write_bytes"] += estimate.write_bytes * multiplier
            bucket["payload_read_bytes"] += estimate.payload_read_bytes * multiplier
            bucket["payload_write_bytes"] += estimate.payload_write_bytes * multiplier
            bucket["read_requests"] += estimate.read_requests * multiplier
            bucket["write_requests"] += estimate.write_requests * multiplier

        issues: set[str] = set()
        max_ratio = 1.0
        occurrence_count = 0
        regime_counts: Counter[str] = Counter()
        stage_regime_counts: Counter[str] = Counter()
        stage_occurrences: Counter[str] = Counter()
        unique_address_geometry_count = 0
        unique_feature_signatures: set[tuple[Any, ...]] = set()
        scalar_fallback_count = 0
        logical_occurrence_count = 0
        processed_geometries = 0
        last_progress_time = 0.0

        def report_progress(
            *,
            stream: PhysicalDmaStream | None,
            force: bool = False,
        ) -> None:
            nonlocal last_progress_time
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force and processed_geometries % geometry_batch_size != 0 and now - last_progress_time < 2.0:
                return
            progress_callback(
                {
                    "phase": "v4_aggregation",
                    "progress_done": processed_geometries,
                    # The exact grouped total is not known until each affine
                    # stream has been reduced.  Physical occurrence count is
                    # an explicit upper bound and remains useful for liveness.
                    "progress_total": sum(physical.multiplicity for physical, _fmt in self.streams.values()),
                    "current_stream": (None if stream is None else stream.stream_index),
                }
            )
            last_progress_time = now

        def feature_signature(
            stream: PhysicalDmaStream,
            estimate: _OccurrenceEstimate,
        ) -> tuple[Any, ...]:
            prediction = estimate.prediction
            return (
                stream.opcode,
                stream.stage,
                estimate.read_bytes,
                estimate.write_bytes,
                estimate.payload_read_bytes,
                estimate.payload_write_bytes,
                estimate.read_requests,
                estimate.write_requests,
                prediction.row_state_regime,
                prediction.theoretical_phase_floor_ns,
                tuple(sorted(prediction.features.items())),
                prediction.calibration_in_domain,
                prediction.domain_issues,
                prediction.extrapolation_ratio,
            )

        def accumulate_estimate(
            *,
            stream_index: int,
            stream: PhysicalDmaStream,
            estimate: _OccurrenceEstimate,
            count: int,
        ) -> None:
            nonlocal total_floor, read_bytes, write_bytes
            nonlocal payload_read_bytes, payload_write_bytes
            nonlocal read_requests, write_requests, max_ratio, occurrence_count
            latency_histogram[(stream.opcode, stream.stage, estimate.latency_ns)] += count
            floor_histogram[
                (
                    stream.stage,
                    estimate.prediction.theoretical_phase_floor_ns,
                )
            ] += count
            read_bytes += estimate.read_bytes * count
            write_bytes += estimate.write_bytes * count
            payload_read_bytes += estimate.payload_read_bytes * count
            payload_write_bytes += estimate.payload_write_bytes * count
            read_requests += estimate.read_requests * count
            write_requests += estimate.write_requests * count
            add_traffic("by_role", stream.precision_role, estimate, count)
            add_traffic("by_stage", stream.stage, estimate, count)
            add_traffic("by_opcode", stream.opcode, estimate, count)
            add_traffic(
                "by_stage_role",
                f"{stream.stage}::{stream.precision_role}",
                estimate,
                count,
            )
            add_traffic(
                "by_opcode_role",
                f"{stream.opcode}::{stream.precision_role}",
                estimate,
                count,
            )
            add_traffic(
                "by_stage_opcode_role",
                f"{stream.stage}::{stream.opcode}::{stream.precision_role}",
                estimate,
                count,
            )
            max_ratio = max(max_ratio, estimate.prediction.extrapolation_ratio)
            issues.update(f"stream={stream_index}:{issue}" for issue in estimate.prediction.domain_issues)
            regime_counts[estimate.prediction.row_state_regime] += count
            stage_regime_counts[f"{stream.stage}::{estimate.prediction.row_state_regime}"] += count
            stage_occurrences[stream.stage] += count
            occurrence_count += count

        grouped_mode = bool(group_cold_geometries and not self._stateful_estimates)
        grouped_estimate_cache: dict[
            tuple[Any, ...],
            _OccurrenceEstimate,
        ] = {}
        prepared_groups: dict[
            int,
            tuple[
                dict[tuple[Any, ...], tuple[dict[str, Any], int]],
                bool,
            ],
        ] = {}
        if grouped_mode and aggregation_backend == "sufficient-statistics-v2":
            vector_templates: dict[
                tuple[Any, ...],
                tuple[
                    PhysicalDmaStream,
                    MemoryFormat,
                    dict[tuple[Any, ...], tuple[dict[str, Any], int]],
                ],
            ] = {}
            scalar_geometries: dict[
                tuple[Any, ...],
                tuple[PhysicalDmaStream, MemoryFormat, dict[str, Any]],
            ] = {}
            seen_geometries: set[tuple[Any, ...]] = set()
            for stream_index, (stream, fmt) in self.streams.items():
                groups, used_scalar_fallback = self._cold_geometry_groups(stream, fmt)
                prepared_groups[stream_index] = (
                    groups,
                    used_scalar_fallback,
                )
                template_key = (
                    stream.opcode,
                    stream.direction,
                    fmt.request_signature(),
                    stream.dim,
                    stream.amount,
                    stream.stride_bytes,
                    stream.rstride,
                )
                for key, (transfer, _count) in groups.items():
                    normalized_key = key[:1] + key[2:]
                    if normalized_key in seen_geometries:
                        continue
                    seen_geometries.add(normalized_key)
                    # A stream-wide element/scale envelope may overlap even
                    # when a particular affine occurrence is disjoint.  The
                    # normalized key carries that exact per-occurrence
                    # relation, so vectorize every proven-disjoint geometry
                    # and retain scalar planning only for true overlaps.
                    if stream.direction == "read":
                        template = vector_templates.get(template_key)
                        if template is None:
                            template = (stream, fmt, {})
                            vector_templates[template_key] = template
                        template[2][key] = (transfer, 1)
                    else:
                        scalar_geometries[normalized_key] = (
                            stream,
                            fmt,
                            transfer,
                        )

            for stream, fmt, template_groups in vector_templates.values():
                vectorized = self._vectorized_cold_read_feature_groups(
                    stream,
                    fmt,
                    template_groups,
                    geometry_batch_size=geometry_batch_size,
                )
                if vectorized is None:
                    for key, (transfer, _count) in template_groups.items():
                        scalar_geometries[key[:1] + key[2:]] = (
                            stream,
                            fmt,
                            transfer,
                        )
                    continue
                _feature_groups, address_estimates = vectorized
                grouped_estimate_cache.update(address_estimates)
            for normalized_key, (
                stream,
                fmt,
                transfer,
            ) in scalar_geometries.items():
                if normalized_key in grouped_estimate_cache:
                    continue
                grouped_estimate_cache[normalized_key] = self._build_estimate(
                    stream,
                    fmt,
                    transfer,
                    retain_manifest=False,
                )
                scalar_fallback_count += 1
            del vector_templates
            del scalar_geometries
            del seen_geometries

        for stream_index, (stream, _fmt) in self.streams.items():
            multiplier = int((stage_multipliers or {}).get(stream.stage, 1))
            if multiplier <= 0:
                raise ValueError(f"V4 stage multiplier for {stream.stage!r} must be positive")
            logical_occurrence_count += stream.multiplicity * multiplier
            if grouped_mode:
                prepared = prepared_groups.pop(stream_index, None)
                if prepared is None:
                    groups, used_scalar_fallback = self._cold_geometry_groups(stream, _fmt)
                else:
                    groups, used_scalar_fallback = prepared
                missing_groups: dict[
                    tuple[Any, ...],
                    tuple[dict[str, Any], int],
                ] = {}
                stream_feature_groups: dict[
                    tuple[Any, ...],
                    tuple[_OccurrenceEstimate, int],
                ] = {}

                def add_stream_feature(
                    estimate: _OccurrenceEstimate,
                    count: int,
                ) -> None:
                    signature = feature_signature(stream, estimate)
                    unique_feature_signatures.add(signature)
                    previous = stream_feature_groups.get(signature)
                    stream_feature_groups[signature] = (
                        estimate if previous is None else previous[0],
                        count if previous is None else previous[1] + count,
                    )

                for key, (transfer, count) in groups.items():
                    normalized_key = key[:1] + key[2:]
                    cached = grouped_estimate_cache.get(normalized_key)
                    if cached is None:
                        missing_groups[key] = (transfer, count)
                        continue
                    add_stream_feature(cached, count)
                vectorized = (
                    self._vectorized_cold_read_feature_groups(
                        stream,
                        _fmt,
                        missing_groups,
                        geometry_batch_size=geometry_batch_size,
                    )
                    if (aggregation_backend == "sufficient-statistics-v2" and missing_groups)
                    else None
                )
                if vectorized is not None:
                    feature_groups, address_estimates = vectorized
                    grouped_estimate_cache.update(address_estimates)
                    for estimate, count in feature_groups.values():
                        add_stream_feature(estimate, count)
                    del address_estimates
                    del feature_groups
                    missing_groups = {}
                for key, (transfer, count) in missing_groups.items():
                    # Grouped cold evaluation consumes each exact normalized
                    # address geometry once.  Retaining its manifest and
                    # estimate duplicates the groups dictionary and was the
                    # dominant long-context RSS source.
                    estimate = self._build_estimate(
                        stream,
                        _fmt,
                        transfer,
                        retain_manifest=False,
                    )
                    grouped_estimate_cache[key[:1] + key[2:]] = estimate
                    scalar_fallback_count += 1
                    add_stream_feature(estimate, count)
                    report_progress(stream=stream)
                for estimate, count in stream_feature_groups.values():
                    accumulate_estimate(
                        stream_index=stream_index,
                        stream=stream,
                        estimate=estimate,
                        count=count * multiplier,
                    )
                processed_geometries += len(groups)
                # Drop the per-stream geometry table before reducing the next
                # stream, so peak memory scales with the largest stream rather
                # than the complete decoder trace.
                del groups
                del stream_feature_groups
                report_progress(stream=stream, force=True)
                continue

            cycles = []
            for position in range(stream.multiplicity):
                estimate = self._estimate(stream_index, position)
                cycles.append(estimate.cycles)
                accumulate_estimate(
                    stream_index=stream_index,
                    stream=stream,
                    estimate=estimate,
                    count=multiplier,
                )
            cycle_sequence = tuple(cycles)
            self._cycle_sequences[stream_index] = cycle_sequence
            self._cycle_periods[stream_index] = self._fundamental_period(cycle_sequence)
        report_progress(stream=None, force=True)
        if grouped_mode:
            unique_address_geometry_count = len(grouped_estimate_cache)
        opcode_terms: dict[str, list[float]] = defaultdict(list)
        stage_opcode_terms: dict[str, list[float]] = defaultdict(list)
        stage_terms: dict[str, list[float]] = defaultdict(list)
        for (opcode, stage, latency), count in latency_histogram.items():
            term = latency * count
            opcode_terms[opcode].append(term)
            stage_opcode_terms[f"{stage}::{opcode}"].append(term)
            stage_terms[stage].append(term)
        opcode_latency.update({key: math.fsum(values) for key, values in opcode_terms.items()})
        stage_opcode_latency.update({key: math.fsum(values) for key, values in stage_opcode_terms.items()})
        stage_latency.update({key: math.fsum(values) for key, values in stage_terms.items()})
        floor_terms: dict[str, list[float]] = defaultdict(list)
        for (stage, floor), count in floor_histogram.items():
            floor_terms[stage].append(floor * count)
        stage_floor.update({key: math.fsum(values) for key, values in floor_terms.items()})
        total_floor = math.fsum(stage_floor.values())
        return HbmServiceV4WorkPrediction(
            latency_ns=sum(opcode_latency.values()),
            theoretical_floor_ns=total_floor,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            payload_read_bytes=payload_read_bytes,
            payload_write_bytes=payload_write_bytes,
            read_requests=read_requests,
            write_requests=write_requests,
            opcode_latency_ns=dict(sorted(opcode_latency.items())),
            stage_latency_ns=dict(sorted(stage_latency.items())),
            traffic_breakdown={
                group: {key: dict(sorted(values.items())) for key, values in sorted(buckets.items())}
                for group, buckets in traffic.items()
            },
            calibration_in_domain=not issues,
            domain_issues=tuple(sorted(issues)),
            max_extrapolation_ratio=max_ratio,
            occurrence_count=occurrence_count,
            row_state_regime_counts=dict(sorted(regime_counts.items())),
            stage_theoretical_floor_ns=dict(sorted(stage_floor.items())),
            aggregation=(
                (
                    "affine_feature_grouped_v2"
                    if aggregation_backend == "sufficient-statistics-v2"
                    else "affine_geometry_grouped_v1"
                )
                if grouped_mode
                else "per_occurrence"
            ),
            unique_geometry_count=(unique_address_geometry_count if grouped_mode else logical_occurrence_count),
            unique_address_geometry_count=(unique_address_geometry_count if grouped_mode else logical_occurrence_count),
            unique_feature_signature_count=(
                len(unique_feature_signatures) if grouped_mode else logical_occurrence_count
            ),
            scalar_fallback_count=scalar_fallback_count,
            exact_feature_equivalence=bool(grouped_mode and aggregation_backend == "sufficient-statistics-v2"),
            logical_occurrence_count=logical_occurrence_count,
            occurrences_elided=(logical_occurrence_count - unique_address_geometry_count if grouped_mode else 0),
            stage_opcode_latency_ns=dict(sorted(stage_opcode_latency.items())),
            stage_occurrence_count=dict(sorted(stage_occurrences.items())),
            stage_row_state_regime_counts=dict(sorted(stage_regime_counts.items())),
        )


def scale_hbm_service_v4_work_by_stage(
    work: HbmServiceV4WorkPrediction,
    stage_multipliers: Mapping[str, int],
) -> HbmServiceV4WorkPrediction:
    """Scale an already-predicted cold workload without re-planning DMA."""

    def multiplier(stage: str) -> int:
        value = int(stage_multipliers.get(stage, 1))
        if value <= 0:
            raise ValueError(f"V4 stage multiplier for {stage!r} must be positive")
        return value

    stage_latency = {stage: value * multiplier(stage) for stage, value in work.stage_latency_ns.items()}
    stage_floor = {stage: value * multiplier(stage) for stage, value in work.stage_theoretical_floor_ns.items()}
    stage_opcode = {
        key: value * multiplier(key.rsplit("::", 1)[0]) for key, value in work.stage_opcode_latency_ns.items()
    }
    opcode_latency: Counter[str] = Counter()
    for key, value in stage_opcode.items():
        _stage, opcode = key.rsplit("::", 1)
        opcode_latency[opcode] += value

    full_traffic = work.traffic_breakdown.get("by_stage_opcode_role", {})
    traffic: dict[str, dict[str, Counter[str]]] = {
        "by_role": {},
        "by_stage": {},
        "by_opcode": {},
        "by_stage_role": {},
        "by_opcode_role": {},
        "by_stage_opcode_role": {},
    }

    def add(group: str, key: str, values: Mapping[str, int]) -> None:
        traffic[group].setdefault(key, Counter()).update(values)

    for key, raw_values in full_traffic.items():
        stage, opcode, role = key.split("::", 2)
        values = {name: int(value) * multiplier(stage) for name, value in raw_values.items()}
        add("by_role", role, values)
        add("by_stage", stage, values)
        add("by_opcode", opcode, values)
        add("by_stage_role", f"{stage}::{role}", values)
        add("by_opcode_role", f"{opcode}::{role}", values)
        add("by_stage_opcode_role", key, values)

    totals = Counter()
    for values in traffic["by_stage_opcode_role"].values():
        totals.update(values)
    stage_occurrences = {stage: count * multiplier(stage) for stage, count in work.stage_occurrence_count.items()}
    stage_regimes = {
        key: count * multiplier(key.rsplit("::", 1)[0]) for key, count in work.stage_row_state_regime_counts.items()
    }
    regimes: Counter[str] = Counter()
    for key, count in stage_regimes.items():
        _stage, regime = key.rsplit("::", 1)
        regimes[regime] += count
    logical_occurrences = sum(stage_occurrences.values())
    return replace(
        work,
        latency_ns=sum(stage_latency.values()),
        theoretical_floor_ns=sum(stage_floor.values()),
        read_bytes=totals["physical_read_bytes"],
        write_bytes=totals["physical_write_bytes"],
        payload_read_bytes=totals["payload_read_bytes"],
        payload_write_bytes=totals["payload_write_bytes"],
        read_requests=totals["read_requests"],
        write_requests=totals["write_requests"],
        opcode_latency_ns=dict(sorted(opcode_latency.items())),
        stage_latency_ns=dict(sorted(stage_latency.items())),
        traffic_breakdown={
            group: {key: dict(sorted(values.items())) for key, values in sorted(buckets.items())}
            for group, buckets in traffic.items()
        },
        occurrence_count=sum(stage_occurrences.values()),
        row_state_regime_counts=dict(sorted(regimes.items())),
        stage_theoretical_floor_ns=dict(sorted(stage_floor.items())),
        logical_occurrence_count=logical_occurrences,
        occurrences_elided=logical_occurrences - work.unique_geometry_count,
        stage_opcode_latency_ns=dict(sorted(stage_opcode.items())),
        stage_occurrence_count=dict(sorted(stage_occurrences.items())),
        stage_row_state_regime_counts=dict(sorted(stage_regimes.items())),
    )


def _load_json(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text())


def _error_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = sorted(float(row["absolute_error_percent"]) for row in rows)

    def percentile(fraction: float) -> float | None:
        if not errors:
            return None
        position = (len(errors) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return errors[low]
        return errors[low] + (errors[high] - errors[low]) * (position - low)

    weighted_numerator = sum(
        abs(float(row["predicted_latency_ns"]) - float(row["observed_latency_ns"])) for row in rows
    )
    weighted_denominator = sum(float(row["observed_latency_ns"]) for row in rows)
    return {
        "sample_count": len(rows),
        "median_absolute_error_percent": percentile(0.5),
        "p95_absolute_error_percent": percentile(0.95),
        "max_absolute_error_percent": max(errors, default=None),
        "weighted_mape_percent": (
            None if weighted_denominator == 0 else 100 * weighted_numerator / weighted_denominator
        ),
    }


def fit_hbm_service_v4(
    plan_value: str | Path | Mapping[str, Any],
    results_value: str | Path | Mapping[str, Any],
    *,
    ridge: float = 1e-8,
    row_hit_anchor_weight: float = 1.0,
    row_hit_anchor_weights: Mapping[str, float] | None = None,
    relative_error_weight_power: float = 1.0,
) -> tuple[HbmServiceModelV4, dict[str, Any]]:
    plan = _load_json(plan_value)
    results = _load_json(results_value)
    if int(plan.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("HBM V4 fit requires a schema-4 plan")
    if int(results.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("HBM V4 fit requires schema-4 results")
    if row_hit_anchor_weight <= 0:
        raise ValueError("row_hit_anchor_weight must be positive")
    anchor_weights = {str(opcode): float(weight) for opcode, weight in (row_hit_anchor_weights or {}).items()}
    if any(weight <= 0 for weight in anchor_weights.values()):
        raise ValueError("all opcode-specific row-hit anchor weights must be positive")
    if not 0.0 <= relative_error_weight_power <= 1.0:
        raise ValueError("relative_error_weight_power must be in [0, 1]")
    if results.get("dma_semantic_version") != DMA_SEMANTIC_VERSION:
        raise ValueError("Rust DMA semantic version differs from Python V4")
    plan_burst_bytes = int(plan.get("physical_burst_bytes", PHYSICAL_BURST_BYTES))
    result_burst_bytes = int(results.get("physical_burst_bytes", PHYSICAL_BURST_BYTES))
    if plan_burst_bytes != PHYSICAL_BURST_BYTES or result_burst_bytes != PHYSICAL_BURST_BYTES:
        raise ValueError(
            "HBM V4 requires the production HBM2 16-byte native burst: "
            f"plan={plan_burst_bytes}, results={result_burst_bytes}"
        )
    by_id = {str(row["id"]): row for row in results["patterns"]}
    samples: list[dict[str, Any]] = []
    parity_errors = []
    for pattern in plan["patterns"]:
        result = by_id.get(str(pattern["id"]))
        if result is None:
            raise ValueError(f"V4 results are missing pattern {pattern['id']!r}")
        expected = pattern["expected_request_manifest"]
        observed = {
            "read_lines": int(result["read_lines"]),
            "write_lines": int(result["write_lines"]),
            "full_lines": int(result["full_lines"]),
            "partial_lines": int(result["partial_lines"]),
            "read_bytes": int(result["request_read_bytes"]),
            "write_bytes": int(result["request_write_bytes"]),
            "request_manifest_hash": str(result["request_manifest_hash"]),
        }
        comparable = {name: expected[name] for name in observed}
        if observed != comparable:
            parity_errors.append({"id": pattern["id"], "expected": comparable, "observed": observed})
        bursts_per_line = REQUEST_BYTES // PHYSICAL_BURST_BYTES
        physical_expected = {
            "physical_burst_bytes": PHYSICAL_BURST_BYTES,
            "read_physical_bursts": observed["read_lines"] * bursts_per_line,
            "write_physical_bursts": observed["write_lines"] * bursts_per_line,
            "read_physical_burst_bytes": observed["read_bytes"],
            "write_physical_burst_bytes": observed["write_bytes"],
        }
        physical_observed = {name: int(result.get(name, value)) for name, value in physical_expected.items()}
        if physical_observed != physical_expected:
            parity_errors.append(
                {
                    "id": pattern["id"],
                    "expected_physical": physical_expected,
                    "observed_physical": physical_observed,
                }
            )
        manifest = plan_dma_request_manifest(pattern["transfer"], pattern["format"])
        channels = int(pattern["channels"])
        row_state = _conditioned_row_state(pattern, channels)
        features = occurrence_features(
            manifest,
            pattern["transfer"],
            channels,
            row_state=row_state,
        )
        observed_ns = float(result["production_dma_median_completion_cycles"])
        samples.append(
            {
                "id": pattern["id"],
                "group": HbmServiceModelV4.group_key(str(pattern["transfer"]["opcode"]), int(pattern["channels"])),
                "opcode": str(pattern["transfer"]["opcode"]),
                "channels": channels,
                "split": str(pattern["split"]),
                "format_signature": _format_from_mapping(pattern["format"]).request_signature(),
                "stream_family": str(pattern.get("stream_family", "unknown")),
                "dim": int(pattern["transfer"]["dim"]),
                "amount": int(pattern["transfer"]["amount"]),
                "stride_bytes": int(pattern["transfer"]["stride_bytes"]),
                "observed_ns": observed_ns,
                "floor_ns": features.theoretical_phase_floor_ns,
                "features": dict(features.values),
            }
        )
    if parity_errors:
        raise ValueError(
            f"Rust/Python production DMA request parity failed for {len(parity_errors)} patterns: {parity_errors[:3]}"
        )

    coefficients: dict[str, dict[str, float]] = {}
    warm_coefficients: dict[str, dict[str, float]] = {}
    domains: dict[str, dict[str, Any]] = {}
    for group in sorted({sample["group"] for sample in samples}):
        group_samples = [sample for sample in samples if sample["group"] == group]
        all_training = [sample for sample in group_samples if sample["split"] == "train"]
        if not all_training:
            all_training = group_samples
        # Generic samples begin from a cold row state and train the complete
        # residual model. Targeted anchors train a separate low-dimensional
        # warm-row model, avoiding a coefficient compromise between cold and
        # repeatedly accessed production tensors.
        training = [sample for sample in all_training if sample["stream_family"] != "row_hit_anchor"]
        if not training:
            training = all_training
        warm_training = [sample for sample in all_training if sample["stream_family"] == "row_hit_anchor"]
        # Keep the one c8 anchor whose repeated request begins with an open-row
        # conflict.  It is the only low-latency observation that identifies
        # the warm model's total-conflict coefficient, which also applies to
        # conflicts intrinsic to an otherwise exact-row-hit occurrence.  The
        # runtime regime selector still routes inherited initial conflicts to
        # the cold/mixed model.
        matrix = np.asarray(
            [[sample["features"][name] for name in FEATURE_NAMES] for sample in training],
            dtype=float,
        )
        targets = np.asarray(
            [max(0.0, sample["observed_ns"] - sample["floor_ns"]) for sample in training],
            dtype=float,
        )
        # Fit relative residual error so small control-sized transfers and
        # production-sized DMA both influence the nonnegative model.
        weights = 1.0 / np.power(
            np.maximum(
                np.asarray([sample["observed_ns"] for sample in training], dtype=float),
                1.0,
            ),
            relative_error_weight_power,
        )
        fitted = _nonnegative_ridge(matrix * weights[:, np.newaxis], targets * weights, ridge)
        coefficients[group] = dict(zip(FEATURE_NAMES, fitted.tolist(), strict=True))

        cold_domain = {
            "features": {
                name: {
                    "min": min(float(sample["features"][name]) for sample in training),
                    "max": max(float(sample["features"][name]) for sample in training),
                }
                for name in FEATURE_NAMES
            },
            "request_signatures": sorted({str(sample["format_signature"]) for sample in training}),
            "training_samples": len(training),
        }
        domains[group] = dict(cold_domain)
        domains[group]["row_state_regimes"] = {"cold_or_mixed": cold_domain}

        if warm_training:
            warm_matrix = np.asarray(
                [[sample["features"][name] for name in WARM_FEATURE_NAMES] for sample in warm_training],
                dtype=float,
            )
            warm_targets = np.asarray(
                [max(0.0, sample["observed_ns"] - sample["floor_ns"]) for sample in warm_training],
                dtype=float,
            )
            warm_weights = 1.0 / np.power(
                np.maximum(
                    np.asarray(
                        [sample["observed_ns"] for sample in warm_training],
                        dtype=float,
                    ),
                    1.0,
                ),
                relative_error_weight_power,
            )
            warm_weights *= row_hit_anchor_weight * anchor_weights.get(warm_training[0]["opcode"], 1.0)
            warm_fitted = _nonnegative_ridge(
                warm_matrix * warm_weights[:, np.newaxis],
                warm_targets * warm_weights,
                ridge,
            )
            warm_coefficients[group] = dict(zip(WARM_FEATURE_NAMES, warm_fitted.tolist(), strict=True))
            domains[group]["row_state_regimes"]["fully_warm"] = {
                "features": {
                    name: {
                        "min": min(float(sample["features"][name]) for sample in warm_training),
                        "max": max(float(sample["features"][name]) for sample in warm_training),
                    }
                    for name in FEATURE_NAMES
                },
                "request_signatures": sorted({str(sample["format_signature"]) for sample in warm_training}),
                "training_samples": len(warm_training),
            }

    identity = {
        "coefficients": coefficients,
        "warm_coefficients": warm_coefficients,
        "domains": domains,
        "semantic": DMA_SEMANTIC_VERSION,
        "feature_semantic": plan.get("feature_semantic_version", "cold-occurrence-v1"),
        "fixture": plan["request_manifest_fixture_hash"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    model = HbmServiceModelV4(
        calibration_id=f"hbm-production-dma-v4-{digest}",
        coefficients=coefficients,
        domains=domains,
        warm_coefficients=warm_coefficients,
        compatibility={
            "ramulator_preset": plan["ramulator_preset"],
            "mapper": plan["mapper"],
            "request_bytes": int(plan["request_bytes"]),
            "physical_burst_bytes": PHYSICAL_BURST_BYTES,
            "dma_semantic_version": DMA_SEMANTIC_VERSION,
            "feature_semantic_version": plan.get("feature_semantic_version", "cold-occurrence-v1"),
            "request_manifest_hash_algorithm": MANIFEST_HASH_ALGORITHM,
            "request_manifest_fixture_hash": plan["request_manifest_fixture_hash"],
        },
        metadata={
            "fit_target": "production_dma_completion_cycles",
            "training_samples": sum(sample["split"] == "train" for sample in samples),
            "holdout_samples": sum(sample["split"] == "holdout" for sample in samples),
            "per_occurrence_prediction": True,
            "raw_v3_labels_used": False,
            "row_state_training": bool(plan.get("feature_semantic_version") == FEATURE_SEMANTIC_VERSION),
            "row_hit_anchor_weight": float(row_hit_anchor_weight),
            "row_hit_anchor_weights": dict(sorted(anchor_weights.items())),
            "row_state_regime_model": "strict_zero_conflict_warm_residual_v2",
            "relative_error_weight_power": float(relative_error_weight_power),
        },
    )

    rows = []
    for sample, pattern in zip(samples, plan["patterns"], strict=True):
        row_state = _conditioned_row_state(pattern, sample["channels"])
        prediction = model.predict_occurrence(
            sample["opcode"],
            pattern["transfer"],
            pattern["format"],
            sample["channels"],
            row_state=row_state,
        )
        error = 100 * abs(prediction.latency_ns - sample["observed_ns"]) / max(sample["observed_ns"], 1.0)
        rows.append(
            {
                "id": sample["id"],
                "group": sample["group"],
                "split": sample["split"],
                "opcode": sample["opcode"],
                "channels": sample["channels"],
                "format_signature": sample["format_signature"],
                "stream_family": sample["stream_family"],
                "dim": sample["dim"],
                "amount": sample["amount"],
                "stride_bytes": sample["stride_bytes"],
                "theoretical_phase_floor_ns": sample["floor_ns"],
                "features": dict(sample["features"]),
                "observed_latency_ns": sample["observed_ns"],
                "predicted_latency_ns": prediction.latency_ns,
                "row_state_regime": prediction.row_state_regime,
                "absolute_error_percent": error,
            }
        )
    holdout = [row for row in rows if row["split"] == "holdout"]
    row_state_anchors = [row for row in rows if row["stream_family"] == "row_hit_anchor"]
    exact_row_hit_anchors = [row for row in row_state_anchors if row["row_state_regime"] == "fully_warm"]
    initial_conflict_anchors = [row for row in row_state_anchors if row["row_state_regime"] == "cold_or_mixed"]
    by_group = {
        group: _error_summary([row for row in holdout if row["group"] == group])
        for group in sorted({row["group"] for row in holdout})
    }
    overall = _error_summary(holdout)
    store_p95 = _error_summary([row for row in holdout if row["group"].startswith("H_STORE_V:")])[
        "p95_absolute_error_percent"
    ]
    acceptance = {
        "median_le_10pct": overall["median_absolute_error_percent"] is not None
        and overall["median_absolute_error_percent"] <= 10,
        "p95_le_25pct": overall["p95_absolute_error_percent"] is not None
        and overall["p95_absolute_error_percent"] <= 25,
        "max_le_60pct": overall["max_absolute_error_percent"] is not None
        and overall["max_absolute_error_percent"] <= 60,
        "store_p95_le_20pct": store_p95 is not None and store_p95 <= 20,
    }
    validation = {
        "schema_version": SCHEMA_VERSION,
        "calibration_id": model.calibration_id,
        "request_manifest_mismatches": 0,
        "overall_holdout": overall,
        "store_holdout_p95_absolute_error_percent": store_p95,
        "by_opcode_channel": by_group,
        "row_state_anchor_fit": {
            "all": _error_summary(row_state_anchors),
            "exact_row_hit": _error_summary(exact_row_hit_anchors),
            "initial_conflict": _error_summary(initial_conflict_anchors),
        },
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "worst_cases": sorted(holdout, key=lambda row: row["absolute_error_percent"], reverse=True)[:20],
        "samples": rows,
    }
    return model, validation


__all__ = [
    "DMA_SEMANTIC_VERSION",
    "DmaRequestManifest",
    "FEATURE_SEMANTIC_VERSION",
    "HbmServiceModelV4",
    "HbmServiceV4Prediction",
    "HbmServiceV4WorkPrediction",
    "LEGACY_ROW_HIT_FEATURE_SEMANTIC_VERSION",
    "MANIFEST_HASH_ALGORITHM",
    "Mop4clxorRowState",
    "SCHEMA_VERSION",
    "V4DmaServiceProvider",
    "combined_request_manifest_hash",
    "fit_hbm_service_v4",
    "generate_hbm_service_v4_plan",
    "occurrence_features",
    "plan_dma_request_manifest",
    "request_manifest_fixture_hash",
    "scale_hbm_service_v4_work_by_stage",
    "stream_occurrence_transfer",
    "write_hbm_service_v4_plan",
]
