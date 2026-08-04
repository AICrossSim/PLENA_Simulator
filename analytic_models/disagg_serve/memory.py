"""Request-level HBM latency model with an aggregate compatibility path.

Peak bandwidth (`bytes / (HBM_WIDTH/8 x freq)`) overstates what decode achieves:
its transfers are small-to-medium bursts from a single-slot load engine, so they
use only part of the channels and pay a fixed per-transfer latency.

Instead we use bandwidth measured on the emulator (Ramulator DMA in
--blocking-prefetch mode, see the external calibration harness), split by
traffic class:

  weights_kv  — H_PREFETCH_M   HBM -> Matrix SRAM (weights + KV tiles)
  activations — H_PREFETCH_V   HBM -> Vector SRAM (small vector loads)
  writeback   — H_STORE_V      Vector SRAM -> HBM (read-modify-write)

Structured compiler DMA descriptors are expanded into element and MX-scale
64-byte requests using the exact MOP4CLXOR hierarchy of the pinned Ramulator2
HBM organization.  The prediction is the busiest-channel transfer floor plus
non-negative ridge coefficients for startup, channel tail, bank serialization,
row misses/conflicts, and partial-write read-modify-write traffic.  Fits are
separate by opcode, HBM generation, and channel count.
Ordered compiler traces retain the last open row in each channel-bank pair and
apply the fitted row-conflict coefficient only when the next descriptor's first
row differs; conflicts internal to one descriptor remain in its isolated fit.

The historical effective-bandwidth tables remain a compatibility path when a
caller has only aggregate byte counts.  Supplying request descriptors fails
closed unless the structured Ramulator calibration is present.  Regression is
weighted by inverse measured latency so every transfer size has comparable
influence on the relative-error validation metric.

CLI: ``python analytic_models/disagg_serve/memory.py --report`` prints both the
aggregate leave-one-channel-out result and the deterministic request holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .calibration_provenance import (
        CALIBRATION_EVIDENCE_GRADE,
        CALIBRATION_PUBLICATION_RECEIPT_COMPLETE,
        REQUEST_CALIBRATION_EVIDENCE_GRADE,
        REQUEST_CALIBRATION_PUBLICATION_RECEIPT_COMPLETE,
    )
except ImportError:
    from calibration_provenance import (
        CALIBRATION_EVIDENCE_GRADE,
        CALIBRATION_PUBLICATION_RECEIPT_COMPLETE,
        REQUEST_CALIBRATION_EVIDENCE_GRADE,
        REQUEST_CALIBRATION_PUBLICATION_RECEIPT_COMPLETE,
    )

_HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = _HERE / "calibration_bw.csv"
DEFAULT_DMA_CALIBRATION = _HERE / "calibration_dma.csv"
DEFAULT_REQUEST_CALIBRATION = _HERE / "calibration_dma_requests.csv"
DMA_LINE_BYTES = 64
HBM_CHANNEL_WIDTH_BITS = 64
HBM_PSEUDOCHANNELS = 2
HBM_BANK_GROUPS = 4
HBM_BANKS_PER_GROUP = 4
HBM_TRANSACTION_BYTES = 16
REQUEST_FEATURES = (
    "request_startup",
    "channel_tail_requests",
    "channel_arrival_burst",
    "bank_serialized_requests",
    "critical_bank_serialized_requests",
    "row_misses",
    "critical_bank_row_misses",
    "row_conflicts",
    "critical_bank_row_conflicts",
    "read_modify_write_requests",
)
REQUEST_FIT_OBJECTIVE = "relative_error_nonnegative_ridge"
REQUEST_MAPPING_SCHEMA = "ramulator2-mop4clxor-hbm-pseudochannel"
REQUEST_FEATURE_SCHEMA = "plena-dma-request-features"
REQUEST_STREAM_COMPOSITION_SCHEMA = "plena-sequential-open-row-composition-v1"

# op mnemonic (measurement) -> traffic class (model)
OP_TO_CLASS = {
    "H_PREFETCH_M": "weights_kv",
    "H_PREFETCH_V": "activations",
    "H_STORE_V": "writeback",
}
CLASSES = tuple(OP_TO_CLASS.values())
CALIBRATED_PIN_RATES_GBPS = {
    "HBM2": 2.0,
    "HBM3": 2.0,
}


def _calibration_identity(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"plena-bandwidth-calibration\0")
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "bandwidth-" + digest.hexdigest()


@dataclass(frozen=True)
class _Point:
    kv_size: int
    gen: str
    channels: int
    op: str
    bytes: int
    dt_ps: int

    @property
    def gbps(self) -> float:
        return self.bytes / (self.dt_ps / 1e12) / 1e9 if self.dt_ps else 0.0


@dataclass(frozen=True)
class DMARequestDescriptor:
    """Compiler-visible DMA shape expressed in physical HBM coordinates."""

    opcode: str
    hbm_generation: str
    channels: int
    address: int
    rows: int
    elements_per_row: int
    stride_bytes: int
    element_bits: int
    direction: str
    pin_rate_gbps: float
    tensor: str = "unspecified"
    scale_bits: int = 0
    block_size: int = 1
    scale_address: int | None = None
    scale_stride_bytes: int | None = None
    partial_write_rmw: bool = False

    def __post_init__(self) -> None:
        if self.opcode not in OP_TO_CLASS:
            raise ValueError(f"unsupported DMA opcode {self.opcode!r}")
        if self.hbm_generation not in CALIBRATED_PIN_RATES_GBPS:
            raise ValueError("DMA descriptor uses an unsupported HBM generation")
        for name in (
            "channels",
            "rows",
            "elements_per_row",
            "stride_bytes",
            "element_bits",
            "block_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels & (self.channels - 1):
            raise ValueError("HBM channel count must be a power of two")
        for name in ("address", "scale_bits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.scale_address is not None and self.scale_address < 0:
            raise ValueError("scale_address must be non-negative")
        if self.scale_stride_bytes is not None and self.scale_stride_bytes <= 0:
            raise ValueError("scale_stride_bytes must be positive")
        if self.direction not in {"read", "write"}:
            raise ValueError("DMA direction must be read or write")
        expected_direction = "write" if self.opcode == "H_STORE_V" else "read"
        if self.direction != expected_direction:
            raise ValueError("DMA direction disagrees with the opcode")
        if not math.isfinite(self.pin_rate_gbps) or self.pin_rate_gbps <= 0:
            raise ValueError("pin_rate_gbps must be finite and positive")
        if not self.tensor:
            raise ValueError("DMA tensor role must be explicit")

    @property
    def element_bytes_per_row(self) -> int:
        return math.ceil(self.elements_per_row * self.element_bits / 8)

    @property
    def scale_bytes_per_row(self) -> int:
        if self.scale_bits == 0:
            return 0
        blocks = math.ceil(self.elements_per_row / self.block_size)
        return math.ceil(blocks * self.scale_bits / 8)

    @property
    def resolved_scale_address(self) -> int | None:
        if self.scale_bytes_per_row == 0:
            return None
        if self.scale_address is not None:
            return self.scale_address
        end = (
            self.address
            + (self.rows - 1) * self.stride_bytes
            + self.element_bytes_per_row
        )
        return math.ceil(end / DMA_LINE_BYTES) * DMA_LINE_BYTES

    @property
    def resolved_scale_stride_bytes(self) -> int:
        return self.scale_stride_bytes or self.scale_bytes_per_row

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "hbm_generation": self.hbm_generation,
            "channels": self.channels,
            "address": self.address,
            "rows": self.rows,
            "elements_per_row": self.elements_per_row,
            "stride_bytes": self.stride_bytes,
            "element_bits": self.element_bits,
            "direction": self.direction,
            "pin_rate_gbps": self.pin_rate_gbps,
            "tensor": self.tensor,
            "scale_bits": self.scale_bits,
            "block_size": self.block_size,
            "scale_address": self.resolved_scale_address,
            "scale_stride_bytes": self.resolved_scale_stride_bytes,
            "partial_write_rmw": self.partial_write_rmw,
        }


@dataclass(frozen=True)
class PhysicalLineRequest:
    address: int
    direction: str
    tensor: str


@dataclass(frozen=True)
class DMARequestFeatures:
    critical_channel_floor_s: float
    values: tuple[float, ...]
    physical_read_bytes: int
    physical_write_bytes: int

    def __post_init__(self) -> None:
        if len(self.values) != len(REQUEST_FEATURES):
            raise ValueError("DMA request feature vector has the wrong width")
        if self.critical_channel_floor_s < 0 or any(value < 0 for value in self.values):
            raise ValueError("DMA request features must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_channel_floor_s": self.critical_channel_floor_s,
            "physical_read_bytes": self.physical_read_bytes,
            "physical_write_bytes": self.physical_write_bytes,
            **dict(zip(REQUEST_FEATURES, self.values)),
        }


def _line_addresses(address: int, byte_count: int) -> tuple[int, ...]:
    if byte_count <= 0:
        return ()
    first = address // DMA_LINE_BYTES * DMA_LINE_BYTES
    end = address + byte_count
    return tuple(range(first, end, DMA_LINE_BYTES))


def _mop4clxor_coordinates(
    address: int,
    channels: int,
) -> tuple[int, int, int]:
    """Map one transaction with Ramulator2's pinned HBM MOP4CLXOR rule.

    HBM2 and HBM3 use the same organization in the retained Ramulator2 commit:
    channel, two pseudochannels, four bank groups, four banks, 32K rows and 64
    columns.  The mapper removes the 16-byte pseudochannel transaction offset,
    places four transactions across the low column bits, and XORs the complete
    column value into the hierarchy levels.  A 64-byte physical line therefore
    contributes four independently mapped transactions.
    """

    if address < 0 or address % HBM_TRANSACTION_BYTES:
        raise ValueError("HBM transactions must use aligned physical addresses")
    if channels <= 0 or channels & (channels - 1):
        raise ValueError("channels must be a positive power of two")

    remaining = address // HBM_TRANSACTION_BYTES
    column = remaining & 0b11
    remaining >>= 2
    level_widths = (
        channels.bit_length() - 1,
        HBM_PSEUDOCHANNELS.bit_length() - 1,
        HBM_BANK_GROUPS.bit_length() - 1,
        HBM_BANKS_PER_GROUP.bit_length() - 1,
    )
    levels = []
    for width in level_widths:
        mask = (1 << width) - 1
        levels.append(remaining & mask)
        remaining >>= width
    column |= (remaining & 0b111) << 2
    remaining >>= 3
    row = remaining

    xor_offset = 0
    for index, width in enumerate(level_widths):
        mask = (1 << width) - 1
        levels[index] ^= (column >> xor_offset) & mask
        xor_offset += width
    channel, pseudochannel, bank_group, bank = levels
    combined_bank = (
        (pseudochannel * HBM_BANK_GROUPS + bank_group) * HBM_BANKS_PER_GROUP
        + bank
    )
    return channel, combined_bank, row


def expand_dma_requests(
    descriptor: DMARequestDescriptor,
) -> tuple[PhysicalLineRequest, ...]:
    """Expand element and scale planes, including store read-modify-write."""

    requests: list[PhysicalLineRequest] = []

    def append_plane(address: int, byte_count: int, tensor: str) -> None:
        for line_address in _line_addresses(address, byte_count):
            if descriptor.direction == "write" and descriptor.partial_write_rmw:
                requests.append(
                    PhysicalLineRequest(
                        line_address,
                        "read",
                        tensor,
                    )
                )
            requests.append(
                PhysicalLineRequest(
                    line_address,
                    descriptor.direction,
                    tensor,
                )
            )

    for row_index in range(descriptor.rows):
        append_plane(
            descriptor.address + row_index * descriptor.stride_bytes,
            descriptor.element_bytes_per_row,
            descriptor.tensor,
        )
        scale_address = descriptor.resolved_scale_address
        if scale_address is None:
            continue
        append_plane(
            scale_address + row_index * descriptor.resolved_scale_stride_bytes,
            descriptor.scale_bytes_per_row,
            f"{descriptor.tensor}_scale",
        )
    if not requests:
        raise ValueError("DMA descriptor expands to no physical requests")
    return tuple(requests)


def _maximum_channel_arrival_burst(
    channel_sequence: Sequence[int],
    channels: int,
) -> float:
    """Return the largest excess over uniform arrivals in any issue interval."""

    last_position: list[int | None] = [None] * channels
    running_excess = [0.0] * channels
    maximum = 0.0
    single_arrival_excess = 1.0 - 1.0 / channels
    for position, channel in enumerate(channel_sequence):
        previous = last_position[channel]
        if previous is None:
            running_excess[channel] = single_arrival_excess
        else:
            gap = position - previous
            running_excess[channel] = max(
                single_arrival_excess,
                running_excess[channel] + 1.0 - gap / channels,
            )
        last_position[channel] = position
        maximum = max(maximum, running_excess[channel])
    return maximum


def dma_request_features(descriptor: DMARequestDescriptor) -> DMARequestFeatures:
    requests = expand_dma_requests(descriptor)
    channel_counts = [0] * descriptor.channels
    bank_counts: dict[tuple[int, int], int] = defaultdict(int)
    rows_by_bank: dict[tuple[int, int], set[int]] = defaultdict(set)
    channel_sequence: list[int] = []
    read_bytes = 0
    write_bytes = 0
    for request in requests:
        if request.direction == "read":
            read_bytes += DMA_LINE_BYTES
        else:
            write_bytes += DMA_LINE_BYTES
        for transaction_address in range(
            request.address,
            request.address + DMA_LINE_BYTES,
            HBM_TRANSACTION_BYTES,
        ):
            channel, bank, row = _mop4clxor_coordinates(
                transaction_address,
                descriptor.channels,
            )
            channel_counts[channel] += 1
            channel_sequence.append(channel)
            bank_key = (channel, bank)
            bank_counts[bank_key] += 1
            rows_by_bank[bank_key].add(row)
    per_channel_bytes_per_s = (
        descriptor.pin_rate_gbps * 1e9 * HBM_CHANNEL_WIDTH_BITS / 8
    )
    busiest = max(channel_counts)
    transaction_count = len(requests) * DMA_LINE_BYTES // HBM_TRANSACTION_BYTES
    average = transaction_count / descriptor.channels
    # The DMA gather submits the descriptor concurrently and Ramulator uses
    # FRFCFS, so requests to one open row are grouped regardless of issue order.
    row_misses = {
        bank_key: len(rows)
        for bank_key, rows in rows_by_bank.items()
    }
    row_conflicts = {
        bank_key: max(0, len(rows) - 1)
        for bank_key, rows in rows_by_bank.items()
    }
    values = (
        1.0,
        max(0.0, busiest - average),
        _maximum_channel_arrival_burst(
            channel_sequence,
            descriptor.channels,
        ),
        float(sum(max(0, count - 1) for count in bank_counts.values())),
        float(max(max(0, count - 1) for count in bank_counts.values())),
        float(sum(row_misses.values())),
        float(max(row_misses.values())),
        float(sum(row_conflicts.values())),
        float(max(row_conflicts.values(), default=0)),
        float(
            sum(
                request.direction == "read"
                for request in requests
            )
            if descriptor.direction == "write"
            and descriptor.partial_write_rmw
            else 0
        ),
    )
    return DMARequestFeatures(
        critical_channel_floor_s=(
            busiest * HBM_TRANSACTION_BYTES / per_channel_bytes_per_s
        ),
        values=values,
        physical_read_bytes=read_bytes,
        physical_write_bytes=write_bytes,
    )


@dataclass(frozen=True)
class DMAObservation:
    descriptor: DMARequestDescriptor
    measured_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.measured_s) or self.measured_s <= 0:
            raise ValueError("measured DMA time must be finite and positive")


@dataclass(frozen=True)
class RequestModelFit:
    opcode: str
    hbm_generation: str
    channels: int
    coefficients_s: tuple[float, ...]
    training_points: int

    def __post_init__(self) -> None:
        if len(self.coefficients_s) != len(REQUEST_FEATURES):
            raise ValueError("request-model coefficient vector has the wrong width")
        if any(value < 0 or not math.isfinite(value) for value in self.coefficients_s):
            raise ValueError("request-model coefficients must be finite and non-negative")
        if self.training_points <= 0:
            raise ValueError("request-model fit requires training points")

    def residual_s(self, features: DMARequestFeatures) -> float:
        return sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients_s, features.values)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "hbm_generation": self.hbm_generation,
            "channels": self.channels,
            "training_points": self.training_points,
            "coefficients_s": dict(zip(REQUEST_FEATURES, self.coefficients_s)),
        }


@dataclass(frozen=True)
class RequestStreamRunPrediction:
    """Latency of one descriptor run with carried HBM open-row state."""

    seconds: float
    isolated_seconds: float
    carried_row_conflicts: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.seconds)
            or not math.isfinite(self.isolated_seconds)
            or self.seconds < 0
            or self.isolated_seconds < 0
            or self.carried_row_conflicts < 0
        ):
            raise ValueError("request-stream prediction values must be non-negative")


def _descriptor_bank_row_boundaries(
    descriptor: DMARequestDescriptor,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Return the first and last row touched in every channel-bank pair."""

    first_rows: dict[tuple[int, int], int] = {}
    last_rows: dict[tuple[int, int], int] = {}
    for request in expand_dma_requests(descriptor):
        for transaction_address in range(
            request.address,
            request.address + DMA_LINE_BYTES,
            HBM_TRANSACTION_BYTES,
        ):
            channel, bank, row = _mop4clxor_coordinates(
                transaction_address,
                descriptor.channels,
            )
            key = (channel, bank)
            first_rows.setdefault(key, row)
            last_rows[key] = row
    return first_rows, last_rows


def _nonnegative_ridge(
    matrix: Sequence[Sequence[float]],
    target: Sequence[float],
    *,
    ridge: float,
    iterations: int = 20_000,
    tolerance: float = 1e-15,
) -> tuple[float, ...]:
    if ridge < 0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if not matrix or len(matrix) != len(target):
        raise ValueError("request-model training matrix is empty or inconsistent")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("request-model training matrix has inconsistent rows")
    coefficients = [0.0] * width
    predictions = [0.0] * len(matrix)
    for _ in range(iterations):
        largest_change = 0.0
        for column in range(width):
            denominator = ridge
            numerator = 0.0
            old = coefficients[column]
            for row_index, row in enumerate(matrix):
                value = float(row[column])
                residual_without_column = (
                    float(target[row_index]) - predictions[row_index] + value * old
                )
                numerator += value * residual_without_column
                denominator += value * value
            updated = max(0.0, numerator / denominator) if denominator else 0.0
            delta = updated - old
            if delta:
                for row_index, row in enumerate(matrix):
                    predictions[row_index] += float(row[column]) * delta
            coefficients[column] = updated
            largest_change = max(largest_change, abs(delta))
        if largest_change <= tolerance:
            break
    return tuple(coefficients)


class RequestLatencyModel:
    """Opcode/channel request model with relative-error ridge residuals."""

    stream_composition_schema = REQUEST_STREAM_COMPOSITION_SCHEMA

    def __init__(self, fits: Sequence[RequestModelFit], calibration_id: str):
        if not fits or not calibration_id:
            raise ValueError("request latency model requires fits and an identity")
        prefix = "request-latency-"
        digest = calibration_id.removeprefix(prefix)
        if (
            not calibration_id.startswith(prefix)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "request latency calibration identity must be request-latency-<sha256>"
            )
        self._fits = {
            (fit.opcode, fit.hbm_generation, fit.channels): fit
            for fit in fits
        }
        if len(self._fits) != len(fits):
            raise ValueError("request latency model contains duplicate fits")
        self.calibration_id = calibration_id

    @classmethod
    def fit(
        cls,
        observations: Sequence[DMAObservation],
        *,
        ridge: float = 1e-12,
        calibration_id: str | None = None,
    ) -> "RequestLatencyModel":
        grouped: dict[tuple[str, str, int], list[DMAObservation]] = defaultdict(list)
        for observation in observations:
            descriptor = observation.descriptor
            grouped[
                (descriptor.opcode, descriptor.hbm_generation, descriptor.channels)
            ].append(observation)
        if not grouped:
            raise ValueError("request latency model has no observations")
        fits: list[RequestModelFit] = []
        identity_rows: list[dict[str, Any]] = []
        for key, group in sorted(grouped.items()):
            feature_rows: list[tuple[float, ...]] = []
            target: list[float] = []
            for observation in group:
                features = dma_request_features(observation.descriptor)
                sample_weight = 1.0 / observation.measured_s
                feature_rows.append(
                    tuple(value * sample_weight for value in features.values)
                )
                target.append(
                    max(
                        0.0,
                        observation.measured_s
                        - features.critical_channel_floor_s,
                    )
                    * sample_weight
                )
                identity_rows.append(
                    {
                        "descriptor": observation.descriptor.to_dict(),
                        "measured_s": observation.measured_s,
                    }
                )
            fits.append(
                RequestModelFit(
                    opcode=key[0],
                    hbm_generation=key[1],
                    channels=key[2],
                    coefficients_s=_nonnegative_ridge(
                        feature_rows,
                        target,
                        ridge=ridge,
                    ),
                    training_points=len(group),
                )
            )
        if calibration_id is None:
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "mapping_schema": REQUEST_MAPPING_SCHEMA,
                        "feature_schema": REQUEST_FEATURE_SCHEMA,
                        "feature_order": REQUEST_FEATURES,
                        "fit_objective": REQUEST_FIT_OBJECTIVE,
                        "ridge": ridge,
                        "geometry": {
                            "line_bytes": DMA_LINE_BYTES,
                            "channel_width_bits": HBM_CHANNEL_WIDTH_BITS,
                            "pseudochannels": HBM_PSEUDOCHANNELS,
                            "bank_groups": HBM_BANK_GROUPS,
                            "banks_per_group": HBM_BANKS_PER_GROUP,
                            "transaction_bytes": HBM_TRANSACTION_BYTES,
                            "calibrated_pin_rates_gbps": CALIBRATED_PIN_RATES_GBPS,
                        },
                        "observations": identity_rows,
                        "fits": [fit.to_dict() for fit in fits],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            calibration_id = f"request-latency-{digest}"
        return cls(fits, calibration_id)

    def predict(self, descriptor: DMARequestDescriptor) -> float:
        calibrated_rate = CALIBRATED_PIN_RATES_GBPS[descriptor.hbm_generation]
        if not math.isclose(
            descriptor.pin_rate_gbps,
            calibrated_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "request latency model cannot extrapolate beyond its calibrated HBM pin rate"
            )
        key = (descriptor.opcode, descriptor.hbm_generation, descriptor.channels)
        try:
            fit = self._fits[key]
        except KeyError as exc:
            raise KeyError(
                "request latency model has no fit for "
                f"opcode={key[0]} gen={key[1]} channels={key[2]}"
            ) from exc
        features = dma_request_features(descriptor)
        return features.critical_channel_floor_s + fit.residual_s(features)

    def predict_stream(
        self,
        runs: Sequence[tuple[DMARequestDescriptor, int]],
    ) -> tuple[RequestStreamRunPrediction, ...]:
        """Price an ordered request stream while retaining open HBM rows.

        The isolated fit already includes conflicts between distinct rows
        inside one descriptor. This method adds only the first row transition
        at each descriptor boundary, using the current opcode's fitted
        ``row_conflicts`` coefficient. Repetitions remain algebraic, so a long
        compiler loop does not need to be materialised.
        """

        open_rows: dict[tuple[int, int], int] = {}
        stream_geometry: tuple[str, int, float] | None = None
        row_conflict_index = REQUEST_FEATURES.index("row_conflicts")
        predictions = []
        for descriptor, repetitions in runs:
            if (
                isinstance(repetitions, bool)
                or not isinstance(repetitions, int)
                or repetitions <= 0
            ):
                raise ValueError(
                    "request-stream repetitions must be positive integers"
                )
            descriptor_geometry = (
                descriptor.hbm_generation,
                descriptor.channels,
                descriptor.pin_rate_gbps,
            )
            if stream_geometry is None:
                stream_geometry = descriptor_geometry
            elif descriptor_geometry != stream_geometry:
                raise ValueError(
                    "one request stream cannot mix HBM operating points"
                )
            key = (
                descriptor.opcode,
                descriptor.hbm_generation,
                descriptor.channels,
            )
            try:
                fit = self._fits[key]
            except KeyError as error:
                raise KeyError(
                    "request latency model has no fit for "
                    f"opcode={key[0]} gen={key[1]} channels={key[2]}"
                ) from error

            isolated_seconds = self.predict(descriptor) * repetitions
            first_rows, last_rows = _descriptor_bank_row_boundaries(descriptor)
            first_boundary_conflicts = sum(
                bank in open_rows and open_rows[bank] != row
                for bank, row in first_rows.items()
            )
            repeated_boundary_conflicts = sum(
                last_rows[bank] != row
                for bank, row in first_rows.items()
            )
            carried_row_conflicts = (
                first_boundary_conflicts
                + (repetitions - 1) * repeated_boundary_conflicts
            )
            correction_seconds = (
                carried_row_conflicts
                * fit.coefficients_s[row_conflict_index]
            )
            predictions.append(
                RequestStreamRunPrediction(
                    seconds=isolated_seconds + correction_seconds,
                    isolated_seconds=isolated_seconds,
                    carried_row_conflicts=carried_row_conflicts,
                )
            )
            open_rows.update(last_rows)
        return tuple(predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "plena-request-latency-model",
            "calibration_id": self.calibration_id,
            "mapping_schema": REQUEST_MAPPING_SCHEMA,
            "feature_schema": REQUEST_FEATURE_SCHEMA,
            "composition": "critical_channel_floor_plus_nonnegative_ridge",
            "fit_objective": REQUEST_FIT_OBJECTIVE,
            "feature_order": list(REQUEST_FEATURES),
            "fits": [
                self._fits[key].to_dict()
                for key in sorted(self._fits)
            ],
        }


def load_request_observations(
    csv_path: str | Path = DEFAULT_REQUEST_CALIBRATION,
) -> tuple[DMAObservation, ...]:
    observations = []
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            measured_raw = row.get("measured_s")
            measured_s = (
                float(measured_raw)
                if measured_raw not in (None, "")
                else float(row["dt_ps"]) / 1e12
            )

            def optional_int(name: str) -> int | None:
                value = row.get(name)
                return None if value in (None, "") else int(value)

            observation = DMAObservation(
                descriptor=DMARequestDescriptor(
                    opcode=str(row["opcode"]),
                    hbm_generation=str(row["hbm_generation"]),
                    channels=int(row["channels"]),
                    address=int(row["address"]),
                    rows=int(row["rows"]),
                    elements_per_row=int(row["elements_per_row"]),
                    stride_bytes=int(row["stride_bytes"]),
                    element_bits=int(row["element_bits"]),
                    direction=str(row["direction"]),
                    pin_rate_gbps=float(row["pin_rate_gbps"]),
                    tensor=str(row.get("tensor") or "unspecified"),
                    scale_bits=int(row.get("scale_bits") or 0),
                    block_size=int(row.get("block_size") or 1),
                    scale_address=optional_int("scale_address"),
                    scale_stride_bytes=optional_int("scale_stride_bytes"),
                    partial_write_rmw=str(
                        row.get("partial_write_rmw", "")
                    ).lower() in {"1", "true", "yes"},
                ),
                measured_s=measured_s,
            )
            features = dma_request_features(observation.descriptor)
            for column, expected in (
                ("physical_read_bytes", features.physical_read_bytes),
                ("physical_write_bytes", features.physical_write_bytes),
            ):
                recorded = row.get(column)
                if recorded not in (None, "") and int(recorded) != expected:
                    raise ValueError(
                        f"request calibration {column} disagrees with its descriptor"
                    )
            observations.append(observation)
    if not observations:
        raise ValueError("request calibration contains no observations")
    return tuple(observations)


def request_holdout_report(
    observations: Sequence[DMAObservation],
    *,
    holdout_fraction: float = 0.20,
    ridge: float = 1e-12,
) -> dict[str, Any]:
    """Fit on a deterministic split and report held-out latency error."""

    if not math.isfinite(holdout_fraction) or not 0.15 <= holdout_fraction < 1.0:
        raise ValueError("request-model holdout fraction must lie in [0.15, 1)")
    grouped: dict[tuple[str, str, int], list[DMAObservation]] = defaultdict(list)
    for observation in observations:
        descriptor = observation.descriptor
        grouped[(descriptor.opcode, descriptor.hbm_generation, descriptor.channels)].append(
            observation
        )
    training: list[DMAObservation] = []
    held_out: list[DMAObservation] = []
    for key, group in sorted(grouped.items()):
        if len(group) < 3:
            raise ValueError(f"request-model group {key!r} needs at least three points")
        by_descriptor: dict[str, list[DMAObservation]] = defaultdict(list)
        for observation in group:
            fingerprint = hashlib.sha256(
                json.dumps(
                    observation.descriptor.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            by_descriptor[fingerprint].append(observation)
        if len(by_descriptor) < 2:
            raise ValueError(
                f"request-model group {key!r} needs distinct descriptor shapes"
            )
        target_count = min(
            len(group) - 1,
            max(1, math.ceil(len(group) * holdout_fraction)),
        )
        selected_count = 0
        for fingerprint in sorted(by_descriptor):
            bucket = by_descriptor[fingerprint]
            if selected_count < target_count:
                held_out.extend(bucket)
                selected_count += len(bucket)
            else:
                training.extend(bucket)
        if selected_count == len(group):
            raise ValueError(f"request-model group {key!r} has no training points")
    model = RequestLatencyModel.fit(training, ridge=ridge)
    errors = []
    per_group: dict[str, list[float]] = defaultdict(list)
    for observation in held_out:
        predicted = model.predict(observation.descriptor)
        error = abs(predicted - observation.measured_s) / observation.measured_s * 100.0
        errors.append(error)
        descriptor = observation.descriptor
        per_group[
            f"{descriptor.opcode}/{descriptor.hbm_generation}/{descriptor.channels}"
        ].append(error)
    ordered_errors = sorted(errors)

    def percentile(values: Sequence[float], fraction: float) -> float:
        if not values:
            raise ValueError("request-model validation has no held-out errors")
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(values[lower])
        weight = position - lower
        return float(values[lower] * (1.0 - weight) + values[upper] * weight)

    return {
        "schema_version": "plena-request-latency-validation",
        "training_count": len(training),
        "holdout_count": len(held_out),
        "holdout_fraction": len(held_out) / len(observations),
        "split_unit": "descriptor_fingerprint",
        "fit_objective": REQUEST_FIT_OBJECTIVE,
        "feature_order": list(REQUEST_FEATURES),
        "mean_absolute_error_percent": sum(errors) / len(errors),
        "median_absolute_error_percent": percentile(ordered_errors, 0.5),
        "p95_absolute_error_percent": percentile(ordered_errors, 0.95),
        "p99_absolute_error_percent": percentile(ordered_errors, 0.99),
        "worst_absolute_error_percent": max(ordered_errors),
        "per_group": {
            key: {
                "count": len(values),
                "median_absolute_error_percent": percentile(sorted(values), 0.5),
                "p95_absolute_error_percent": percentile(sorted(values), 0.95),
            }
            for key, values in sorted(per_group.items())
        },
        "calibration_id": model.calibration_id,
    }


class TransferSizeModel:
    """Per-transfer DMA cost fitted from the dma_microbench sweep:
    time = t0 + bytes / bw_inf, one line per (gen, channels). Bandwidth is then
    bw(size) = size / (t0 + size/bw_inf), which rises with transfer size and
    saturates at bw_inf for large transfers (what large-MLEN chips issue),
    rather than being fixed at the decode testbench's transfer size."""

    def __init__(self, csv_path: str | Path = DEFAULT_DMA_CALIBRATION):
        pts: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (row["hbm_gen"], int(row["channels"]))
                pts[key].append(
                    (float(row["bytes_per_transfer"]), float(row["dt_ps_per_transfer"]) / 1e12)
                )
        # Fit a line t = t0 + slope * bytes for each (gen, channels).
        self._fit: dict[tuple[str, int], tuple[float, float]] = {}
        for key, xy in pts.items():
            n = len(xy)
            sx = sum(x for x, _ in xy)
            sy = sum(y for _, y in xy)
            sxx = sum(x * x for x, _ in xy)
            sxy = sum(x * y for x, y in xy)
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            t0 = (sy - slope * sx) / n
            self._fit[key] = (max(t0, 0.0), max(slope, 1e-15))

    def channel_counts(self, gen: str) -> list[int]:
        return sorted(ch for (g, ch) in self._fit if g == gen)

    def _params(self, gen: str, channels: int) -> tuple[float, float]:
        if (gen, channels) in self._fit:
            return self._fit[(gen, channels)]
        chs = self.channel_counts(gen)
        if not chs:
            raise KeyError(f"no DMA calibration for gen={gen}")
        # Channel scaling is only measured inside the calibrated range, so a
        # request outside it fails closed instead of being clamped to an edge.
        if channels < chs[0] or channels > chs[-1]:
            raise KeyError(
                f"DMA calibration for gen={gen} supports channels "
                f"{chs[0]}..{chs[-1]}, not {channels}"
            )
        lo = max(c for c in chs if c < channels)
        hi = min(c for c in chs if c > channels)
        f = (math.log(channels) - math.log(lo)) / (math.log(hi) - math.log(lo))
        t0l, sl = self._fit[(gen, lo)]
        t0h, sh = self._fit[(gen, hi)]
        interp = lambda a, b: math.exp(math.log(a) + f * (math.log(b) - math.log(a)))
        return interp(t0l, t0h), interp(sl, sh)

    def bw_gbps(self, gen: str, channels: int, transfer_bytes: float) -> float:
        t0, slope = self._params(gen, channels)
        t = t0 + slope * transfer_bytes
        return transfer_bytes / t / 1e9

    def holdout_report(self, holdout_amount_bytes: float = 32 * 1024) -> str:
        """For each (gen, ch), refit without the point nearest
        `holdout_amount_bytes` and report the error at that held-out point."""
        lines = [f"transfer-size holdout near {holdout_amount_bytes/1024:.0f} KiB"]
        errs = []
        with open(DEFAULT_DMA_CALIBRATION) as f:
            rows = list(csv.DictReader(f))
        keys = sorted({(r["hbm_gen"], int(r["channels"])) for r in rows})
        for gen, ch in keys:
            sub = [r for r in rows if r["hbm_gen"] == gen and int(r["channels"]) == ch]
            held = min(sub, key=lambda r: abs(float(r["bytes_per_transfer"]) - holdout_amount_bytes))
            rest = [r for r in sub if r is not held]
            n = len(rest)
            xs = [float(r["bytes_per_transfer"]) for r in rest]
            ys = [float(r["dt_ps_per_transfer"]) / 1e12 for r in rest]
            slope = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / (
                n * sum(x * x for x in xs) - sum(xs) ** 2)
            t0 = (sum(ys) - slope * sum(xs)) / n
            hb = float(held["bytes_per_transfer"])
            measured = hb / (float(held["dt_ps_per_transfer"]) / 1e12) / 1e9
            pred = hb / (t0 + slope * hb) / 1e9
            err = abs(pred - measured) / measured * 100
            errs.append(err)
            lines.append(f"  {gen:<6} ch={ch:<3} measured {measured:7.1f} GB/s  "
                         f"predicted {pred:7.1f}  err {err:4.1f}%")
        if errs:
            errs.sort()
            lines.append(f"  median {errs[len(errs)//2]:.1f}%   max {errs[-1]:.1f}%   n={len(errs)}")
        return "\n".join(lines)


class CalibratedBandwidth:
    """Effective-bandwidth lookup fitted from an aggregate calibration CSV."""

    def __init__(self, points: list[_Point], calibration_id: str | None = None):
        self._points = points
        self.calibration_id = calibration_id
        self.evidence_grade = CALIBRATION_EVIDENCE_GRADE
        self.publication_receipt_complete = (
            CALIBRATION_PUBLICATION_RECEIPT_COMPLETE
        )
        self.size_model: TransferSizeModel | None = None
        self.request_model: RequestLatencyModel | None = None
        # Bandwidth per (class, gen, channels), aggregated over kv sizes and
        # weighted by bytes so the large transfers that dominate decode also
        # dominate the estimate.
        acc: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
        for p in points:
            if p.op not in OP_TO_CLASS:
                continue
            key = (OP_TO_CLASS[p.op], p.gen, p.channels)
            acc[key][0] += p.bytes
            acc[key][1] += p.dt_ps
        self._bw: dict[tuple[str, str, int], float] = {
            key: (b / (t / 1e12) / 1e9) for key, (b, t) in acc.items() if t > 0
        }

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, csv_path: str | Path = DEFAULT_CALIBRATION) -> "CalibratedBandwidth":
        csv_path = Path(csv_path)
        points = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                points.append(
                    _Point(
                        kv_size=int(row["kv_size"]),
                        gen=row["hbm_gen"],
                        channels=int(row["channels"]),
                        op=row["op"],
                        bytes=int(row["bytes"]),
                        dt_ps=int(row["dt_ps"]),
                    )
                )
        identity_paths = [csv_path]
        if DEFAULT_DMA_CALIBRATION.exists():
            identity_paths.append(DEFAULT_DMA_CALIBRATION)
        if DEFAULT_REQUEST_CALIBRATION.exists():
            identity_paths.append(DEFAULT_REQUEST_CALIBRATION)
        model = cls(points, calibration_id=_calibration_identity(*identity_paths))
        # Use the size-aware curve for the weights/KV stream if the DMA sweep
        # exists; otherwise fall back to the per-class table.
        if DEFAULT_DMA_CALIBRATION.exists():
            model.size_model = TransferSizeModel()
        if DEFAULT_REQUEST_CALIBRATION.exists():
            model.request_model = RequestLatencyModel.fit(
                load_request_observations(DEFAULT_REQUEST_CALIBRATION)
            )
        return model

    # -- queries -------------------------------------------------------------

    def channel_counts(self, cls_name: str, gen: str) -> list[int]:
        return sorted(ch for (c, g, ch) in self._bw if c == cls_name and g == gen)

    def operating_point_calibration_id(
        self,
        gen: str,
        pin_rate_gbps: float,
    ) -> str | None:
        """Return an identity only when generation and measured rate match."""

        measured_rate = CALIBRATED_PIN_RATES_GBPS.get(gen)
        if (
            self.calibration_id is None
            or measured_rate is None
            or not math.isfinite(pin_rate_gbps)
            or abs(measured_rate - pin_rate_gbps) > 1e-12
        ):
            return None
        payload = (
            f"{self.calibration_id}\0{gen}\0{pin_rate_gbps:.12g}"
        ).encode("utf-8")
        return "bandwidth-operating-point-" + hashlib.sha256(payload).hexdigest()

    def evidence_status(self) -> dict[str, str | bool | None]:
        """Return the provenance grade independently of numerical fit quality."""

        return {
            "evidence_grade": self.evidence_grade,
            "publication_receipt_complete": self.publication_receipt_complete,
            "exact_historical_replay": False,
            "request_model_available": self.request_model is not None,
            "request_model_evidence_grade": (
                REQUEST_CALIBRATION_EVIDENCE_GRADE
                if self.request_model is not None
                else None
            ),
            "request_model_publication_receipt_complete": (
                REQUEST_CALIBRATION_PUBLICATION_RECEIPT_COMPLETE
                if self.request_model is not None
                else False
            ),
            "request_model_calibration_id": (
                self.request_model.calibration_id
                if self.request_model is not None
                else None
            ),
        }

    def bw_gbps(self, cls_name: str, gen: str, channels: int) -> float:
        """Effective bandwidth in GB/s; log-log interpolation across channels."""
        key = (cls_name, gen, channels)
        if key in self._bw:
            return self._bw[key]
        chs = self.channel_counts(cls_name, gen)
        if not chs:
            raise KeyError(f"no calibration for class={cls_name} gen={gen}")
        # Channel scaling saturates outside the calibrated range, so a request
        # beyond it fails closed instead of being clamped to an edge point.
        if channels < chs[0] or channels > chs[-1]:
            raise KeyError(
                f"bandwidth calibration for class={cls_name} gen={gen} supports "
                f"channels {chs[0]}..{chs[-1]}, not {channels}"
            )
        lo = max(c for c in chs if c < channels)
        hi = min(c for c in chs if c > channels)
        bw_lo = self._bw[(cls_name, gen, lo)]
        bw_hi = self._bw[(cls_name, gen, hi)]
        frac = (math.log(channels) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return math.exp(math.log(bw_lo) + frac * (math.log(bw_hi) - math.log(bw_lo)))

    def memory_time(
        self,
        bytes_by_class: dict[str, float],
        gen: str,
        channels: int,
        transfer_bytes: float | None = None,
        pin_rate_gbps: float | None = None,
        request_descriptors: Sequence[DMARequestDescriptor] | None = None,
    ) -> float:
        """Seconds to move the given per-class byte counts, added up serially.

        Each engine runs one DMA at a time and decode is dominated by the
        weights_kv stream, so summing the class times is a fair first-order
        estimate. `transfer_bytes` is the weights/KV per-DMA size; if given (and
        the DMA sweep exists) that stream is priced on the size-aware curve.
        """
        # A supplied pin rate must land on a calibrated operating point; there
        # is no measured basis for rescaling the tables to another rate.
        #
        # Structured descriptors replace the aggregate path entirely: each one
        # is priced against the Ramulator request fit at the same generation and
        # channel count, so the aggregate byte totals are never consulted.
        if pin_rate_gbps is not None and (
            self.operating_point_calibration_id(gen, pin_rate_gbps) is None
        ):
            raise ValueError(
                f"no calibrated bandwidth at {gen} {pin_rate_gbps:g} Gb/s"
            )
        if request_descriptors is not None:
            if not request_descriptors:
                return 0.0
            if self.request_model is None:
                raise RuntimeError(
                    "request descriptors require structured Ramulator calibration"
                )
            for descriptor in request_descriptors:
                if (
                    descriptor.hbm_generation != gen
                    or descriptor.channels != channels
                ):
                    raise ValueError(
                        "request descriptor operating point differs from memory_time"
                    )
            return sum(
                self.request_model.predict(descriptor)
                for descriptor in request_descriptors
            )
        total = 0.0
        for cls_name, nbytes in bytes_by_class.items():
            if nbytes <= 0:
                continue
            if cls_name == "weights_kv" and transfer_bytes and self.size_model is not None:
                bw = self.size_model.bw_gbps(gen, channels, transfer_bytes)
            else:
                bw = self.bw_gbps(cls_name, gen, channels)
            total += nbytes / (bw * 1e9)
        return total

    # -- reporting / validation ---------------------------------------------

    def table(self) -> str:
        lines = [f"{'class':<12}{'gen':<6}{'ch':>4}{'BW_eff GB/s':>14}"]
        for (cls_name, gen, ch), bw in sorted(self._bw.items()):
            lines.append(f"{cls_name:<12}{gen:<6}{ch:>4}{bw:>14.1f}")
        return "\n".join(lines)

    def holdout_report(self, holdout_channels: int) -> str:
        """Hold out one channel count and predict it from the rest by interpolation."""
        errs = []
        lines = [f"holdout: channels={holdout_channels}"]
        full = self._bw
        rest = CalibratedBandwidth(
            [p for p in self._points if p.channels != holdout_channels]
        )
        for (cls_name, gen, ch), measured in sorted(full.items()):
            if ch != holdout_channels:
                continue
            pred = rest.bw_gbps(cls_name, gen, ch)
            err = abs(pred - measured) / measured * 100
            errs.append(err)
            lines.append(
                f"  {cls_name:<12}{gen:<6} measured {measured:8.1f}  "
                f"predicted {pred:8.1f}  err {err:5.1f}%"
            )
        if errs:
            errs.sort()
            median = errs[len(errs) // 2]
            p95 = errs[min(len(errs) - 1, int(0.95 * len(errs)))]
            lines.append(f"  median {median:.1f}%   P95 {p95:.1f}%   n={len(errs)}")
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument(
        "--request-calibration",
        default=str(DEFAULT_REQUEST_CALIBRATION),
    )
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--holdout", type=int, default=16,
                    help="channel count to hold out for the fit-error report")
    ap.add_argument(
        "--request-holdout-fraction",
        type=float,
        default=0.2,
        help="deterministic structured-request holdout fraction (minimum 0.15)",
    )
    ap.add_argument(
        "--validation-out",
        type=Path,
        help="optional JSON path for the structured-request validation report",
    )
    args = ap.parse_args()

    model = CalibratedBandwidth.load(args.calibration)
    if args.report:
        print(model.table())
        print()
        print(model.holdout_report(args.holdout))
        request_path = Path(args.request_calibration)
        if request_path.is_file():
            request_report = request_holdout_report(
                load_request_observations(request_path),
                holdout_fraction=args.request_holdout_fraction,
            )
            print()
            print(json.dumps(request_report, indent=2, sort_keys=True))
            if args.validation_out is not None:
                args.validation_out.parent.mkdir(parents=True, exist_ok=True)
                args.validation_out.write_text(
                    json.dumps(
                        request_report,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        elif args.validation_out is not None:
            raise FileNotFoundError(request_path)


if __name__ == "__main__":
    main()
