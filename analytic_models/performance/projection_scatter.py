"""DSE/debug model for the executable Compiler L_SCATTER_M contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


CONTRACT_NAME = "plena.projection_scatter"
CONTRACT_VERSION = 1


class ProjectionFlow(StrEnum):
    BUFFERED = "buffered"
    FIFO_WITH_SPILL = "fifo_with_spill"


@dataclass(frozen=True)
class ScatterField:
    name: str
    producer: str
    consumer: str
    source_offset: int
    values_per_group: int
    physical_offset: int
    physical_span: int
    local_rows: int
    local_lanes: int
    group_shared: bool
    skew_kind: str
    skew_stride: int


@dataclass(frozen=True)
class ScatterPlan:
    algorithm: str
    phase: str
    context_id: int
    request_id: int
    layer_id: int
    token_offset: int
    valid_tokens: int
    activation_bytes: int
    source_input_features: int
    source_values_per_token: int
    source_projections: int
    layout: str
    banks: int
    ports_per_bank: int
    groups: int
    group_span_values: int
    physical_values_per_token: int
    physical_buffer_index: int
    physical_buffer_base_row: int
    physical_token_stride_rows: int
    physical_buffer_rows: int
    fallback_vram_addr: int
    fallback_token_stride: int
    flow: ProjectionFlow
    fifo_capacity_values: int
    producer_burst_values: int
    spill_write_values_per_cycle: int
    spill_policy: str
    head_lanes: int
    head_dim_lanes: int
    state_dim_lanes: int
    fields: tuple[ScatterField, ...]
    mapping_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScatterPlan:
        if raw.get("contract") != CONTRACT_NAME or raw.get("version") != CONTRACT_VERSION:
            raise ValueError("unsupported projection-scatter plan contract")
        fields = tuple(
            ScatterField(**{key: value for key, value in field.items() if key in ScatterField.__dataclass_fields__})
            for field in raw["fields"]
        )
        names = cls.__dataclass_fields__
        values = {key: value for key, value in raw.items() if key in names}
        values["flow"] = ProjectionFlow(raw["flow"])
        values["fields"] = fields
        plan = cls(**values)
        plan.validate()
        return plan

    @property
    def total_values(self) -> int:
        return self.valid_tokens * self.source_values_per_token

    def validate(self) -> None:
        if self.valid_tokens <= 0 or self.source_values_per_token <= 0:
            raise ValueError("projection token and value counts must be positive")
        if self.banks <= 0 or self.ports_per_bank <= 0:
            raise ValueError("projection bank geometry must be positive")
        if self.fifo_capacity_values < self.producer_burst_values:
            raise ValueError("projection FIFO cannot hold one producer burst")
        if self.fallback_token_stride < self.source_values_per_token:
            raise ValueError("projection fallback stride is too small")
        if self.physical_values_per_token != self.groups * self.group_span_values:
            raise ValueError("projection physical token size is inconsistent")
        if self.physical_token_stride_rows * self.banks != self.physical_values_per_token:
            raise ValueError("projection physical row stride is inconsistent")

        sources: set[int] = set()
        physical: set[tuple[int, int]] = set()
        for field in self.fields:
            if field.local_rows * field.local_lanes != field.values_per_group:
                raise ValueError(f"projection field {field.name} has an invalid shape")
            for group in range(self.groups):
                for local_row in range(field.local_rows):
                    for lane in range(field.local_lanes):
                        source, row, bank = self.address(field.name, group, local_row, lane)
                        expected_source = self.logical_source(field.name, group, local_row, lane)
                        if source != expected_source:
                            raise ValueError(f"projection mapping changes logical source {expected_source} to {source}")
                        sources.add(source)
                        if (row, bank) in physical:
                            raise ValueError(f"projection mapping aliases ({row}, {bank})")
                        physical.add((row, bank))
        if sources != set(range(self.source_values_per_token)):
            raise ValueError("projection fields do not cover the source packet")
        if len(physical) != self.source_values_per_token:
            raise ValueError("projection physical mapping is not bijective")
        if self.compute_mapping_sha256() != self.mapping_sha256:
            raise ValueError("projection mapping checksum does not match the physical plan")

    def field(self, name: str) -> ScatterField:
        try:
            return next(field for field in self.fields if field.name == name)
        except StopIteration as error:
            raise ValueError(f"projection field {name!r} is absent") from error

    def logical_source(self, field_name: str, group: int, local_row: int, lane: int) -> int:
        """Return a source index without consulting the physical mapping."""
        field = self.field(field_name)
        if not 0 <= group < self.groups:
            raise ValueError("projection group is out of range")
        if not 0 <= local_row < field.local_rows or not 0 <= lane < field.local_lanes:
            raise ValueError("projection field coordinate is out of range")
        local = local_row * field.local_lanes + lane
        return field.source_offset + group * field.values_per_group + local

    def address(self, field_name: str, group: int, local_row: int, lane: int) -> tuple[int, int, int]:
        field = self.field(field_name)
        local = local_row * field.local_lanes + lane
        source = self.logical_source(field_name, group, local_row, lane)
        if self.layout == "row_major":
            return (
                source,
                self.physical_buffer_base_row + source // self.banks,
                source % self.banks,
            )
        physical = group * self.group_span_values + field.physical_offset + local
        if field.skew_kind == "local_row_stride":
            skew = local_row * field.skew_stride
        elif field.skew_kind == "field_constant":
            skew = field.skew_stride
        elif field.skew_kind == "group_stride":
            skew = group * field.skew_stride
        elif field.skew_kind == "none":
            skew = 0
        else:
            raise ValueError(f"unknown projection skew {field.skew_kind!r}")
        return (
            source,
            self.physical_buffer_base_row + physical // self.banks,
            (local % self.banks + skew) % self.banks,
        )

    def compute_mapping_sha256(self) -> str:
        digest = hashlib.sha256()
        for field in self.fields:
            for group in range(self.groups):
                for local_row in range(field.local_rows):
                    for lane in range(field.local_lanes):
                        source, row, bank = self.address(field.name, group, local_row, lane)
                        digest.update(f"{field.name}:{group}:{local_row}:{lane}:{source}:{row}:{bank}\n".encode())
        return digest.hexdigest()

    def stream_runs(self) -> tuple[StreamRun, ...]:
        runs = []
        ordered = sorted(self.fields, key=lambda field: field.source_offset)
        for token in range(self.valid_tokens):
            token_base = token * self.source_values_per_token
            for field in ordered:
                for group in range(self.groups):
                    runs.append(
                        StreamRun(
                            start=token_base + field.source_offset + group * field.values_per_group,
                            values=field.values_per_group,
                            forced_spill=field.consumer != "state",
                        )
                    )
        return _coalesce_stream_runs(tuple(runs))


@dataclass(frozen=True)
class StreamRun:
    start: int
    values: int
    forced_spill: bool


@dataclass(frozen=True)
class ServiceRun:
    start: int
    values: int
    spilled: bool


@dataclass(frozen=True)
class FifoSpillStats:
    values: int
    bursts: int
    producer_cycles: int
    completion_cycles: int
    direct_values: int
    spill_values: int
    spill_bytes: int
    fifo_stall_cycles: int
    fifo_high_watermark: int
    service_runs: tuple[ServiceRun, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["service_runs"] = [asdict(run) for run in self.service_runs]
        return result


class ProjectionFifoSpillModel:
    """Conservative ordered FIFO with direct-state and Vector-SRAM spill sinks."""

    def __init__(
        self,
        *,
        flow: ProjectionFlow,
        fifo_capacity_values: int,
        producer_burst_values: int,
        spill_write_values_per_cycle: int,
        consumer_start_cycle: int,
        consumer_values_per_cycle: int,
        activation_bytes: int,
    ) -> None:
        for name, value in (
            ("fifo_capacity_values", fifo_capacity_values),
            ("producer_burst_values", producer_burst_values),
            ("spill_write_values_per_cycle", spill_write_values_per_cycle),
            ("consumer_values_per_cycle", consumer_values_per_cycle),
            ("activation_bytes", activation_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if consumer_start_cycle < 0:
            raise ValueError("consumer_start_cycle must be non-negative")
        if fifo_capacity_values < producer_burst_values:
            raise ValueError("FIFO capacity must hold one producer burst")
        self.flow = flow
        self.capacity = fifo_capacity_values
        self.burst = producer_burst_values
        self.spill_width = spill_write_values_per_cycle
        self.consumer_start = consumer_start_cycle
        self.consumer_width = consumer_values_per_cycle
        self.activation_bytes = activation_bytes

    def simulate(self, runs: tuple[StreamRun, ...], *, producer_cycles: int) -> FifoSpillStats:
        values = sum(run.values for run in runs)
        if values == 0:
            return FifoSpillStats(0, 0, producer_cycles, producer_cycles, 0, 0, 0, 0, 0, ())
        if producer_cycles <= 0:
            raise ValueError("producer_cycles must be positive for a non-empty projection")
        _validate_stream_runs(runs, values)

        bursts = math.ceil(values / self.burst)
        queue: deque[StreamRun] = deque()
        serviced: list[ServiceRun] = []
        occupancy = high_watermark = fifo_stall = clock = 0
        stream_index = 0
        stream_offset = 0

        for burst_index in range(bursts):
            arrival = math.floor(burst_index * producer_cycles / bursts) + fifo_stall
            if arrival > clock:
                occupancy -= self._drain(queue, serviced, clock, arrival - clock)
                clock = arrival
            burst_start = burst_index * self.burst
            burst_values = min(self.burst, values - burst_start)
            needed = max(0, occupancy + burst_values - self.capacity)
            while needed > 0:
                before = occupancy
                occupancy -= self._drain(queue, serviced, clock, 1)
                clock += 1
                fifo_stall += 1
                freed = before - occupancy
                if freed <= 0:
                    raise ValueError("projection FIFO has no active drain path")
                needed = max(0, occupancy + burst_values - self.capacity)
            remaining = burst_values
            while remaining:
                source_run = runs[stream_index]
                available = source_run.values - stream_offset
                take = min(remaining, available)
                queue.append(
                    StreamRun(
                        source_run.start + stream_offset,
                        take,
                        source_run.forced_spill,
                    )
                )
                remaining -= take
                stream_offset += take
                if stream_offset == source_run.values:
                    stream_index += 1
                    stream_offset = 0
            occupancy += burst_values
            high_watermark = max(high_watermark, occupancy)

        producer_done = producer_cycles + fifo_stall
        if clock < producer_done:
            occupancy -= self._drain(queue, serviced, clock, producer_done - clock)
            clock = producer_done
        while occupancy:
            occupancy -= self._drain(queue, serviced, clock, 1)
            clock += 1

        service_runs = _coalesce_service_runs(tuple(serviced))
        spilled = sum(run.values for run in service_runs if run.spilled)
        direct = values - spilled
        return FifoSpillStats(
            values=values,
            bursts=bursts,
            producer_cycles=producer_cycles,
            completion_cycles=clock,
            direct_values=direct,
            spill_values=spilled,
            spill_bytes=spilled * self.activation_bytes,
            fifo_stall_cycles=fifo_stall,
            fifo_high_watermark=high_watermark,
            service_runs=service_runs,
        )

    def _drain(
        self,
        queue: deque[StreamRun],
        serviced: list[ServiceRun],
        start_cycle: int,
        cycles: int,
    ) -> int:
        drained = 0
        clock = start_cycle
        remaining_cycles = cycles
        while queue and remaining_cycles > 0:
            front = queue[0]
            direct_ready = clock >= self.consumer_start
            spilled = self.flow == ProjectionFlow.BUFFERED or front.forced_spill or not direct_ready
            width = self.spill_width if spilled else self.consumer_width
            phase_cycles = remaining_cycles
            if not spilled and self.flow == ProjectionFlow.BUFFERED:
                raise AssertionError("buffered flow cannot select the direct sink")
            if not front.forced_spill and self.flow == ProjectionFlow.FIFO_WITH_SPILL and not direct_ready:
                phase_cycles = min(phase_cycles, self.consumer_start - clock)
            capacity = phase_cycles * width
            if capacity <= 0:
                # Unreachable today: width is validated positive and the
                # not-ready clamp above leaves phase_cycles >= 1. Fail loudly
                # rather than resetting the clock and retrying, which would spin
                # without consuming remaining_cycles and hang the whole sweep.
                raise ValueError(f"projection drain made no progress: phase_cycles={phase_cycles}, width={width}")
            take = min(front.values, capacity)
            used_cycles = math.ceil(take / width)
            serviced.append(ServiceRun(front.start, take, spilled))
            drained += take
            if take == front.values:
                queue.popleft()
            else:
                queue[0] = StreamRun(front.start + take, front.values - take, front.forced_spill)
            clock += used_cycles
            remaining_cycles -= used_cycles
        return drained


@dataclass(frozen=True)
class BankStats:
    packets: int = 0
    value_accesses: int = 0
    ideal_cycles: int = 0
    service_cycles: int = 0

    @property
    def stall_cycles(self) -> int:
        return self.service_cycles - self.ideal_cycles

    def to_dict(self) -> dict[str, int]:
        return {**asdict(self), "stall_cycles": self.stall_cycles}


@dataclass(frozen=True)
class ScatterEventCounters:
    event_index: int
    algorithm: str
    layer_id: int
    token_offset: int
    valid_tokens: int
    mapping_sha256: str
    fifo: FifoSpillStats
    scatter_writes: BankStats
    state_reads: BankStats
    gate_reads: BankStats
    bc_value_reads: int
    bc_broadcast_saved_reads: int
    roundtrip: ScatterRoundTrip | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            **asdict(self),
            "fifo": self.fifo.to_dict(),
            "scatter_writes": self.scatter_writes.to_dict(),
            "state_reads": self.state_reads.to_dict(),
            "gate_reads": self.gate_reads.to_dict(),
        }
        result["roundtrip"] = None if self.roundtrip is None else self.roundtrip.to_dict()
        return result


def simulate_scatter_event(
    event_index: int,
    plan: ScatterPlan,
    *,
    matrix_macs_per_cycle: int,
    consumer_start_cycle: int,
    consumer_values_per_cycle: int,
    bc_broadcast: bool,
    roundtrip_tokens: int | None = None,
) -> ScatterEventCounters:
    if matrix_macs_per_cycle <= 0:
        raise ValueError("matrix_macs_per_cycle must be positive")
    producer_cycles = math.ceil(plan.total_values * plan.source_input_features / matrix_macs_per_cycle)
    fifo = ProjectionFifoSpillModel(
        flow=plan.flow,
        fifo_capacity_values=plan.fifo_capacity_values,
        producer_burst_values=plan.producer_burst_values,
        spill_write_values_per_cycle=plan.spill_write_values_per_cycle,
        consumer_start_cycle=consumer_start_cycle,
        consumer_values_per_cycle=consumer_values_per_cycle,
        activation_bytes=plan.activation_bytes,
    ).simulate(plan.stream_runs(), producer_cycles=producer_cycles)
    spilled = _service_membership(fifo.service_runs, plan.source_values_per_token, spilled=True)
    writes = _scatter_write_stats(plan, spilled)
    state_reads, gate_reads, bc_reads, bc_saved = _consumer_bank_stats(plan, spilled, bc_broadcast)
    return ScatterEventCounters(
        event_index=event_index,
        algorithm=plan.algorithm,
        layer_id=plan.layer_id,
        token_offset=plan.token_offset,
        valid_tokens=plan.valid_tokens,
        mapping_sha256=plan.mapping_sha256,
        fifo=fifo,
        scatter_writes=writes,
        state_reads=state_reads,
        gate_reads=gate_reads,
        bc_value_reads=bc_reads,
        bc_broadcast_saved_reads=bc_saved,
        roundtrip=(
            None
            if roundtrip_tokens is None
            else verify_scatter_roundtrip(
                plan,
                tokens=roundtrip_tokens,
                service_runs=fifo.service_runs,
            )
        ),
    )


def simulate_lowered_trace(
    document: dict[str, Any],
    *,
    matrix_macs_per_cycle: int = 4096,
    consumer_start_cycle: int = 0,
    consumer_values_per_cycle: int = 16,
    bc_broadcast: bool = True,
    roundtrip_tokens: int | None = None,
) -> dict[str, Any]:
    _validate_contract_header(document)
    events = []
    for raw in document.get("projection_scatters", []):
        plan = ScatterPlan.from_dict(raw["plan"])
        events.append(
            simulate_scatter_event(
                int(raw["event_index"]),
                plan,
                matrix_macs_per_cycle=matrix_macs_per_cycle,
                consumer_start_cycle=consumer_start_cycle,
                consumer_values_per_cycle=consumer_values_per_cycle,
                bc_broadcast=bc_broadcast,
                roundtrip_tokens=roundtrip_tokens,
            )
        )
    if not events:
        raise ValueError("lowered trace contains no projection-scatter events")
    roundtrips = [event.roundtrip for event in events if event.roundtrip is not None]
    summary_roundtrip = None
    if roundtrips:
        summary_roundtrip = {
            "events_checked": len(roundtrips),
            "tokens_checked": sum(item.tokens_checked for item in roundtrips),
            "tokens_present": sum(event.valid_tokens for event in events),
            "values_round_tripped": sum(item.read_values for item in roundtrips),
            "direct_values": sum(item.direct_values for item in roundtrips),
            "banked_values": sum(item.banked_values for item in roundtrips),
            "max_bank_multiplicity": max(item.max_bank_multiplicity for item in roundtrips),
            "service_cycles": sum(item.service_cycles for item in roundtrips),
            "ideal_cycles": sum(item.ideal_cycles for item in roundtrips),
            "stall_cycles": sum(item.stall_cycles for item in roundtrips),
            "conflict_free": all(item.conflict_free for item in roundtrips),
        }
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "settings": {
            "matrix_macs_per_cycle": matrix_macs_per_cycle,
            "consumer_start_cycle": consumer_start_cycle,
            "consumer_values_per_cycle": consumer_values_per_cycle,
            "bc_broadcast": bc_broadcast,
            "roundtrip_tokens": roundtrip_tokens,
        },
        "roundtrip": summary_roundtrip,
        "summary": {
            "events": len(events),
            "tokens": sum(event.valid_tokens for event in events),
            "produced_values": sum(event.fifo.values for event in events),
            "direct_values": sum(event.fifo.direct_values for event in events),
            "spill_values": sum(event.fifo.spill_values for event in events),
            "spill_bytes": sum(event.fifo.spill_bytes for event in events),
            "fifo_stall_cycles": sum(event.fifo.fifo_stall_cycles for event in events),
            "fifo_high_watermark": max(event.fifo.fifo_high_watermark for event in events),
            "scatter_write_bank_stall_cycles": sum(event.scatter_writes.stall_cycles for event in events),
            "state_read_bank_stall_cycles": sum(event.state_reads.stall_cycles for event in events),
            "gate_read_bank_stall_cycles": sum(event.gate_reads.stall_cycles for event in events),
            "bc_value_reads": sum(event.bc_value_reads for event in events),
            "bc_broadcast_saved_reads": sum(event.bc_broadcast_saved_reads for event in events),
        },
        "events": [event.to_dict() for event in events],
    }


class BankedProjectionBuffer:
    """A real banks x rows store holding one word per bank per row.

    Bank-stall counting only proves a packet *could* be serviced in one cycle.
    It says nothing about whether the word the consumer receives is the word the
    projection wrote, so the skew has never been checked as an address mapping.
    This buffer stores actual values and refuses to write a cell twice or read a
    cell that was never written, which turns aliasing and drops into failures
    instead of silently plausible cycle counts.
    """

    def __init__(self, rows: int, banks: int) -> None:
        if rows <= 0 or banks <= 0:
            raise ValueError("banked buffer geometry must be positive")
        self.rows = rows
        self.banks = banks
        self._cells: list[list[int | None]] = [[None] * banks for _ in range(rows)]
        self.written = 0

    def write(self, row: int, bank: int, value: int) -> None:
        if not 0 <= row < self.rows or not 0 <= bank < self.banks:
            raise ValueError(f"projection write ({row}, {bank}) is outside the buffer")
        if self._cells[row][bank] is not None:
            raise ValueError(f"projection mapping writes ({row}, {bank}) twice: {self._cells[row][bank]} then {value}")
        self._cells[row][bank] = value
        self.written += 1

    def read_packet(self, coordinates: tuple[tuple[int, int], ...], ports_per_bank: int) -> tuple[list[int], int, int]:
        """Return the packet's values, its service cycles, and its worst bank load."""
        if ports_per_bank <= 0:
            raise ValueError("ports_per_bank must be positive")
        counts = [0] * self.banks
        values: list[int] = []
        for row, bank in coordinates:
            if not 0 <= row < self.rows or not 0 <= bank < self.banks:
                raise ValueError(f"projection read ({row}, {bank}) is outside the buffer")
            cell = self._cells[row][bank]
            if cell is None:
                raise ValueError(f"projection reads unwritten cell ({row}, {bank})")
            values.append(cell)
            counts[bank] += 1
        if not coordinates:
            return values, 0, 0
        worst = max(counts)
        return values, math.ceil(worst / ports_per_bank), worst


@dataclass(frozen=True)
class ScatterRoundTrip:
    algorithm: str
    layout: str
    tokens_checked: int
    written_values: int
    direct_values: int
    banked_values: int
    read_values: int
    packets: int
    max_bank_multiplicity: int
    service_cycles: int
    ideal_cycles: int

    @property
    def stall_cycles(self) -> int:
        return self.service_cycles - self.ideal_cycles

    @property
    def conflict_free(self) -> bool:
        return self.service_cycles == self.ideal_cycles

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stall_cycles": self.stall_cycles,
            "conflict_free": self.conflict_free,
        }


def verify_scatter_roundtrip(
    plan: ScatterPlan,
    *,
    tokens: int | None = None,
    service_runs: tuple[ServiceRun, ...] | None = None,
) -> ScatterRoundTrip:
    """Move independent logical values through direct and spill paths.

    Spilled values enter the banked buffer while direct values enter an ordered
    FIFO sink. Consumer packets reconstruct their lanes from both paths. The
    expected value is derived independently from the logical field coordinate,
    so a broken mapping or packet list cannot generate its own matching oracle.
    """
    checked = plan.valid_tokens if tokens is None else min(tokens, plan.valid_tokens)
    if checked <= 0:
        raise ValueError("projection round trip needs at least one token")
    token_rows = plan.physical_token_stride_rows
    buffer = BankedProjectionBuffer(plan.physical_buffer_base_row + checked * token_rows, plan.banks)
    if service_runs is None:
        service_runs = (ServiceRun(0, plan.total_values, True),)
    _validate_service_runs(service_runs, plan.total_values)
    service_assignment = {
        source: run.spilled for run in service_runs for source in range(run.start, run.start + run.values)
    }

    direct_store: dict[int, int] = {}
    for field in plan.fields:
        for group in range(plan.groups):
            for local_row in range(field.local_rows):
                for lane in range(field.local_lanes):
                    mapped_source, row, bank = plan.address(field.name, group, local_row, lane)
                    source = plan.logical_source(field.name, group, local_row, lane)
                    if mapped_source != source:
                        raise ValueError(f"projection mapping changes logical source {source} to {mapped_source}")
                    for token in range(checked):
                        value = token * plan.source_values_per_token + source
                        if service_assignment[value]:
                            buffer.write(row + token * token_rows, bank, value)
                        else:
                            if value in direct_store:
                                raise ValueError(f"projection sends direct value {value} twice")
                            direct_store[value] = value
    expected_written = checked * plan.source_values_per_token
    if buffer.written + len(direct_store) != expected_written:
        raise ValueError(
            f"projection service wrote {buffer.written + len(direct_store)} values, expected {expected_written}"
        )

    packets = consumer_packets(plan)
    source_counts = Counter(
        plan.logical_source(read.field, read.group, read.local_row, read.lane)
        for packet in packets
        for read in packet.reads
    )
    expected_sources = set(range(plan.source_values_per_token))
    missing = sorted(expected_sources - set(source_counts))
    extra = sorted(set(source_counts) - expected_sources)
    duplicated = sorted(source for source, count in source_counts.items() if count != 1)
    if missing or extra or duplicated:
        raise ValueError(
            "consumer packet coverage is invalid: "
            f"missing={missing[:8]}, extra={extra[:8]}, duplicated={duplicated[:8]}"
        )
    resolved = {
        packet: tuple(
            (
                plan.logical_source(read.field, read.group, read.local_row, read.lane),
                *plan.address(read.field, read.group, read.local_row, read.lane)[1:],
            )
            for read in packet.reads
        )
        for packet in packets
    }
    read_values = service_cycles = ideal_cycles = worst_bank = 0
    lanes = plan.banks * plan.ports_per_bank
    for token in range(checked):
        row_base = token * token_rows
        value_base = token * plan.source_values_per_token
        for packet in packets:
            entries = resolved[packet]
            values: list[int | None] = [None] * len(entries)
            positions: list[int] = []
            coordinates: list[tuple[int, int]] = []
            for position, (source, row, bank) in enumerate(entries):
                global_source = value_base + source
                if service_assignment[global_source]:
                    positions.append(position)
                    coordinates.append((row + row_base, bank))
                else:
                    try:
                        values[position] = direct_store[global_source]
                    except KeyError as error:
                        raise ValueError(f"consumer cannot reconstruct direct value {global_source}") from error
            banked, cycles, packet_worst = buffer.read_packet(tuple(coordinates), plan.ports_per_bank)
            for position, value in zip(positions, banked, strict=True):
                values[position] = value
            expected = [value_base + source for source, _, _ in entries]
            if values != expected:
                raise ValueError(
                    f"projection round trip returned the wrong values for a {packet.sink} packet on token {token}"
                )
            read_values += len(values)
            service_cycles += cycles
            ideal_cycles += math.ceil(len(coordinates) / lanes)
            worst_bank = max(worst_bank, packet_worst)
    if read_values != expected_written:
        raise ValueError(
            f"consumer packets read {read_values} values but the projection "
            f"produced {expected_written}: the packet set does not cover the packet"
        )
    return ScatterRoundTrip(
        algorithm=plan.algorithm,
        layout=plan.layout,
        tokens_checked=checked,
        written_values=buffer.written + len(direct_store),
        direct_values=len(direct_store),
        banked_values=buffer.written,
        read_values=read_values,
        packets=len(packets) * checked,
        max_bank_multiplicity=worst_bank,
        service_cycles=service_cycles,
        ideal_cycles=ideal_cycles,
    )


@dataclass(frozen=True)
class ConsumerRead:
    """One logical coordinate read inside a consumer packet."""

    field: str
    group: int
    local_row: int
    lane: int


@dataclass(frozen=True)
class ConsumerPacket:
    """Reads the consumer issues together in one buffer access.

    ``reads`` is a single copy. B/C records are shared by every head in a group,
    so a design without broadcast re-reads the same copy once per head; the
    caller applies that multiplier rather than the packet carrying policy.
    """

    sink: str
    reads: tuple[ConsumerRead, ...]
    broadcast: bool = False


def consumer_packets(plan: ScatterPlan) -> tuple[ConsumerPacket, ...]:
    """Packet shapes one token's consumer issues, before spill filtering.

    Bank counting and the physical round-trip check must agree on packet shape,
    otherwise a conflict-free stall count would not be evidence about the reads
    that actually happen. Both call this.
    """
    packets: list[ConsumerPacket] = []
    if plan.algorithm == "kda":
        key_dim = plan.field("q").local_lanes
        value_dim = plan.field("v").local_lanes
        for head in range(plan.groups):
            packets.append(ConsumerPacket("state", (ConsumerRead("beta", head, 0, 0),)))
            for key_start in range(0, key_dim, plan.state_dim_lanes):
                keys = range(key_start, min(key_start + plan.state_dim_lanes, key_dim))
                packets.append(
                    ConsumerPacket(
                        "state",
                        tuple(
                            ConsumerRead(field_name, head, 0, key) for field_name in ("q", "k", "decay") for key in keys
                        ),
                    )
                )
            for value_start in range(0, value_dim, plan.head_dim_lanes):
                values = range(value_start, min(value_start + plan.head_dim_lanes, value_dim))
                packets.append(
                    ConsumerPacket(
                        "state",
                        tuple(ConsumerRead("v", head, 0, value) for value in values),
                    )
                )
        return tuple(packets)
    if plan.algorithm != "mamba2":
        raise ValueError(f"unsupported projection consumer {plan.algorithm!r}")
    heads = plan.field("x").local_rows
    head_dim = plan.field("x").local_lanes
    state_dim = plan.field("b").local_lanes
    for group in range(plan.groups):
        for head_start in range(0, heads, plan.head_lanes):
            local_heads = range(head_start, min(head_start + plan.head_lanes, heads))
            packets.append(
                ConsumerPacket(
                    "state",
                    tuple(ConsumerRead("dt", group, head, 0) for head in local_heads),
                )
            )
            for dim_start in range(0, head_dim, plan.head_dim_lanes):
                dims = range(dim_start, min(dim_start + plan.head_dim_lanes, head_dim))
                coordinates = tuple((head, dim) for head in local_heads for dim in dims)
                packets.append(
                    ConsumerPacket(
                        "state",
                        tuple(ConsumerRead("x", group, head, dim) for head, dim in coordinates),
                    )
                )
                packets.append(
                    ConsumerPacket(
                        "gate",
                        tuple(ConsumerRead("gate", group, head, dim) for head, dim in coordinates),
                    )
                )
        for state_start in range(0, state_dim, plan.state_dim_lanes):
            states = range(state_start, min(state_start + plan.state_dim_lanes, state_dim))
            packets.append(
                ConsumerPacket(
                    "state",
                    tuple(ConsumerRead(field_name, group, 0, state) for field_name in ("b", "c") for state in states),
                    broadcast=True,
                )
            )
    return tuple(packets)


def _consumer_bank_stats(
    plan: ScatterPlan,
    spilled: dict[int, set[int]],
    bc_broadcast: bool,
) -> tuple[BankStats, BankStats, int, int]:
    packets = consumer_packets(plan)
    heads = plan.field("x").local_rows if plan.algorithm == "mamba2" else 1
    broadcast_copies = 1 if bc_broadcast else heads
    state_packets: list[list[int]] = []
    gate_packets: list[list[int]] = []
    bc_reads = bc_saved = 0
    resolved = {
        packet: tuple(plan.address(read.field, read.group, read.local_row, read.lane) for read in packet.reads)
        for packet in packets
    }
    for token in range(plan.valid_tokens):
        token_spilled = spilled.get(token, set())
        for packet in packets:
            banks = [bank for source, _, bank in resolved[packet] if source in token_spilled]
            if packet.broadcast:
                bc_reads += len(banks) * broadcast_copies
                bc_saved += len(banks) * (heads - broadcast_copies)
                state_packets.append(banks * broadcast_copies)
            elif packet.sink == "gate":
                gate_packets.append(banks)
            else:
                state_packets.append(banks)
    # KDA has no gate sink and no broadcast record, so its gate/broadcast
    # counters fall out as zero rather than needing a separate branch.
    return (
        _summarize_packets(state_packets, plan.banks, plan.ports_per_bank),
        _summarize_packets(gate_packets, plan.banks, plan.ports_per_bank),
        bc_reads,
        bc_saved,
    )


def _scatter_write_stats(plan: ScatterPlan, spilled: dict[int, set[int]]) -> BankStats:
    source_to_bank = {}
    for field in plan.fields:
        for group in range(plan.groups):
            for local_row in range(field.local_rows):
                for lane in range(field.local_lanes):
                    source, _, bank = plan.address(field.name, group, local_row, lane)
                    source_to_bank[source] = bank
    packets = []
    for token in range(plan.valid_tokens):
        token_spilled = spilled.get(token, set())
        for start in range(0, plan.source_values_per_token, plan.producer_burst_values):
            stop = min(start + plan.producer_burst_values, plan.source_values_per_token)
            packets.append([source_to_bank[source] for source in range(start, stop) if source in token_spilled])
    return _summarize_packets(packets, plan.banks, plan.ports_per_bank)


def _summarize_packets(packets: list[list[int]], banks: int, ports: int) -> BankStats:
    packet_count = values = ideal = service = 0
    for packet in packets:
        if not packet:
            continue
        counts = [0] * banks
        for bank in packet:
            counts[bank] += 1
        packet_count += 1
        values += len(packet)
        ideal += math.ceil(len(packet) / (banks * ports))
        service += max(math.ceil(count / ports) for count in counts)
    return BankStats(packet_count, values, ideal, service)


def _service_membership(
    runs: tuple[ServiceRun, ...],
    values_per_token: int,
    *,
    spilled: bool,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for run in runs:
        if run.spilled != spilled:
            continue
        remaining = run.values
        cursor = run.start
        while remaining:
            token = cursor // values_per_token
            source = cursor % values_per_token
            take = min(remaining, values_per_token - source)
            result.setdefault(token, set()).update(range(source, source + take))
            cursor += take
            remaining -= take
    return result


def _coalesce_stream_runs(runs: tuple[StreamRun, ...]) -> tuple[StreamRun, ...]:
    result: list[StreamRun] = []
    for run in sorted(runs, key=lambda item: item.start):
        if result and result[-1].start + result[-1].values == run.start and result[-1].forced_spill == run.forced_spill:
            previous = result[-1]
            result[-1] = StreamRun(previous.start, previous.values + run.values, previous.forced_spill)
        else:
            result.append(run)
    return tuple(result)


def _coalesce_service_runs(runs: tuple[ServiceRun, ...]) -> tuple[ServiceRun, ...]:
    result: list[ServiceRun] = []
    for run in runs:
        if result and result[-1].start + result[-1].values == run.start and result[-1].spilled == run.spilled:
            previous = result[-1]
            result[-1] = ServiceRun(previous.start, previous.values + run.values, previous.spilled)
        else:
            result.append(run)
    return tuple(result)


def _validate_stream_runs(runs: tuple[StreamRun, ...], values: int) -> None:
    cursor = 0
    for run in runs:
        if run.start != cursor or run.values <= 0:
            raise ValueError("projection stream runs must be positive, ordered, and contiguous")
        cursor += run.values
    if cursor != values:
        raise ValueError("projection stream runs do not cover all values")


def _validate_service_runs(runs: tuple[ServiceRun, ...], values: int) -> None:
    cursor = 0
    for run in runs:
        if run.start != cursor or run.values <= 0:
            raise ValueError("projection service runs must be positive, ordered, and contiguous")
        cursor += run.values
    if cursor != values:
        raise ValueError(f"projection service runs cover {cursor} values, expected {values}")


def _contract_document() -> dict[str, Any]:
    return {
        "contract": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "transport": "l-scatter-m-v1-plus-lowered-trace-debug-view",
        "isa_opcode": 0x3F,
        "semantics": {
            "fifo": "L_SCATTER_M applies the descriptor FIFO and buffering policy to Matrix writeback values.",
            "spill": "Materialized values retain fallback_vram_addr while the physical bank mapping is explicit.",
            "replay": "X_STATE consumes the staged physical layout only after the matching L_SCATTER_M command.",
            "layout": "The bank mapping is encoded by an L_SCATTER_M descriptor and remains absent from X_STATE descriptors.",
            "skew_kinds": "none, local_row_stride, field_constant, group_stride",
        },
    }


def _validate_contract_header(document: dict[str, Any]) -> None:
    header = document.get("projection_scatter_contract")
    if not isinstance(header, dict):
        raise ValueError("lowered trace has no projection-scatter contract header")
    if header.get("contract") != CONTRACT_NAME or header.get("version") != CONTRACT_VERSION:
        raise ValueError("lowered trace uses an unsupported projection-scatter contract")
    expected = hashlib.sha256(
        json.dumps(_contract_document(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if header.get("sha256") != expected:
        raise ValueError("projection-scatter contract checksum mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lowered_trace", type=Path)
    parser.add_argument("--matrix-macs-per-cycle", type=int, default=4096)
    parser.add_argument("--consumer-start-cycle", type=int, default=0)
    parser.add_argument("--consumer-values-per-cycle", type=int, default=16)
    parser.add_argument("--disable-bc-broadcast", action="store_true")
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help=(
            "Write real values through the physical bank mapping and read them "
            "back through the consumer packets. Proves the layout delivers the "
            "right value to the right lane, which the stall counters cannot."
        ),
    )
    parser.add_argument(
        "--roundtrip-tokens",
        type=int,
        default=2,
        help=(
            "Tokens per event to round trip. The mapping is token-invariant apart "
            "from the row stride, so two tokens prove the stride does not alias; "
            "the report records how many were actually checked."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.roundtrip_tokens <= 0:
        raise SystemExit("--roundtrip-tokens must be positive")
    document = json.loads(args.lowered_trace.read_text())
    report = simulate_lowered_trace(
        document,
        matrix_macs_per_cycle=args.matrix_macs_per_cycle,
        consumer_start_cycle=args.consumer_start_cycle,
        consumer_values_per_cycle=args.consumer_values_per_cycle,
        bc_broadcast=not args.disable_bc_broadcast,
        roundtrip_tokens=args.roundtrip_tokens if args.verify_roundtrip else None,
    )
    rendered = json.dumps(report, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
