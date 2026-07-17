"""Compressed CostTrace scheduling with transactional ``rtl-v1`` hazards.

This module mirrors ``transactional_emulator/src/scheduler.rs``.  It only
returns a makespan when instruction order and every required DMA service time
are available.  Counts-only compiler regions are rejected rather than being
silently serialized or assigned an invented overlap pattern.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from typing import Any, Callable

from compiler.asm_templates._imm import IMM2_BOUND
from compiler.aten.cost_emitter import (
    CostTrace,
    ScheduleAffineAdd,
    ScheduleAffineLoad,
    ScheduleInstruction,
    ScheduleNode,
    ScheduleRepeat,
    ScheduleSequence,
    ScheduleUnavailable,
)

from .rtl_opcode_timing import (
    ComputePrecisionConfig,
    OpcodeTimingEstimate,
    RtlOpcodeTimingCalibration,
    TimingHardware,
)


MATRIX_COMPUTE = {
    "M_MM",
    "M_TMM",
    "M_BMM",
    "M_BTMM",
    "M_MV",
    "M_TMV",
    "M_BMV",
    "M_BTMV",
}
MATRIX_WRITEOUT = {"M_MM_WO", "M_BMM_WO", "M_MV_WO", "M_BMV_WO"}
VECTOR_ELEMENT = {
    "V_ADD_VV",
    "V_ADD_VF",
    "V_SUB_VV",
    "V_SUB_VF",
    "V_MUL_VV",
    "V_MUL_VF",
    "V_EXP_V",
    "V_RECI_V",
    "V_SHIFT_V",
}
VECTOR_REDUCTION = {"V_RED_SUM", "V_RED_MAX"}
VECTOR_OPS = VECTOR_ELEMENT | VECTOR_REDUCTION
SCALAR_FP_COMPUTE = {
    "S_ADD_FP",
    "S_SUB_FP",
    "S_MAX_FP",
    "S_MUL_FP",
    "S_EXP_FP",
    "S_RECI_FP",
    "S_SQRT_FP",
}
SCALAR_SRAM = {"S_LD_FP", "S_ST_FP", "S_MAP_V_FP"}
SCALAR_OPS = SCALAR_FP_COMPUTE | SCALAR_SRAM | {
    "S_ADD_INT",
    "S_ADDI_INT",
    "S_SUB_INT",
    "S_MUL_INT",
    "S_LUI_INT",
    "S_LD_INT",
    "S_ST_INT",
}
CONTROL_OPS = {
    "C_SET_ADDR_REG",
    "C_SET_SCALE_REG",
    "C_SET_STRIDE_REG",
    "C_SET_V_MASK_REG",
    "C_LOOP_START",
    "C_LOOP_END",
    "C_BREAK",
}
MEMORY_OPS = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}
# Most compiler loops should normalize instead of being repeatedly expanded.
# S_MAP_V_FP is the exception: it can leave one pending Vector SRAM write per
# iteration for up to 2*VLEN cycles, so a finite loop may finish before that
# queue reaches steady state.  Give only map-containing loops a larger literal
# window; applying it globally expands every 128-row vector loop in Qwen.
MAX_DIRECT_REPEAT_INSTRUCTIONS = 256
MAX_DIRECT_MAP_REPEAT_INSTRUCTIONS = 1_024
MAX_REPEAT_PROBE_ITERATIONS = 512
# Large-immediate legalization is periodic in the low 12 address bits.  Qwen
# commonly advances row pointers by 128 elements, which produces a 32-iteration
# LUI/ADDI pattern.  The period search must cover that pattern before falling
# back to literal expansion.
MAX_REPEAT_PERIOD_ITERATIONS = 64
# A non-periodic affine stream may cross several immediate-legalization
# regimes.  Re-entering the repeat solver after each boundary is exact, but a
# single recursive tail containing thousands of iterations can exceed
# Python's recursion limit.  Split only that tail into bounded chunks; each
# chunk still uses the same exact transition proof and boundary handling.
MAX_REPEAT_TAIL_CHUNK_ITERATIONS = 64


class ScheduleUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class AddressRange:
    start: int
    end: int

    @classmethod
    def from_length(cls, start: int, length: int) -> AddressRange:
        return cls(start, start + length)

    def overlaps(self, other: AddressRange) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class InstructionAccesses:
    matrix_reads: list[AddressRange] = field(default_factory=list)
    matrix_writes: list[AddressRange] = field(default_factory=list)
    vector_reads: list[AddressRange] = field(default_factory=list)
    vector_writes: list[AddressRange] = field(default_factory=list)
    scalar_fp_reads: list[int] = field(default_factory=list)
    scalar_fp_writes: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Slot:
    until: int = 0
    sequence: int | None = None


@dataclass(frozen=True)
class PendingWrite:
    address_range: AddressRange
    slot: Slot


@dataclass(frozen=True)
class BusyInterval:
    start: int
    end: int
    sequence: int | None


@dataclass(frozen=True)
class ScheduledEvent:
    sequence: int
    opcode: str
    issue_cycle: int
    accepted_cycle: int
    recovery_cycles: int
    start_cycle: int
    result_ready_cycle: int
    completion_cycle: int
    resource: str
    calibration_status: str
    rtl_supported: bool
    calibration_in_domain: bool
    stall_reason: str | None = None
    dependency: int | None = None


@dataclass(frozen=True)
class ScheduledDmaOccurrence:
    """One dynamic DMA occurrence on the compressed scheduler timeline.

    Unlike ``events``, this stream is retained even when ordinary instruction
    events are disabled. Qwen traces contain only a few thousand DMA
    operations but tens of millions of compute/control instructions, so this
    compact timeline is practical for arrival-time Ramulator replay.
    """

    sequence: int
    stream_index: int
    opcode: str
    start_cycle: int
    completion_cycle: int


@dataclass(frozen=True)
class _RelativeDmaOccurrence:
    """DMA occurrence normalized to a repeat-transition boundary."""

    sequence_offset: int
    stream_index: int
    opcode: str
    start_cycle_offset: int
    completion_cycle_offset: int


@dataclass(frozen=True)
class ScheduledShadowResult:
    status: str
    fidelity: str
    makespan_cycles: int | None
    events: tuple[ScheduledEvent, ...]
    stall_cycles_by_reason: Mapping[str, int]
    resource_work_cycles: Mapping[str, int]
    validation: Mapping[str, Any]
    reason: str | None = None
    critical_path_cycles: Mapping[str, int] = field(default_factory=dict)
    dma_occurrences: tuple[ScheduledDmaOccurrence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "fidelity": self.fidelity,
            "makespan_cycles": self.makespan_cycles,
            "events": [asdict(event) for event in self.events],
            "stall_cycles_by_reason": dict(self.stall_cycles_by_reason),
            "resource_work_cycles": dict(self.resource_work_cycles),
            "critical_path_cycles": dict(self.critical_path_cycles),
            "dma_occurrences": [asdict(event) for event in self.dma_occurrences],
            "validation": dict(self.validation),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _SchedulerSnapshot:
    """Canonical scheduler state at a repeat boundary.

    Absolute cycle and dynamic sequence numbers are retained so a stable
    transition can be applied algebraically.  Expired scoreboard entries are
    canonicalized away; they cannot affect any future issue decision and
    otherwise prevent an idle resource from reaching a fixed point.
    """

    next_issue_cycle: int
    local_makespan_cycles: int
    sequence: int
    gp: tuple[int | None, ...]
    affine_positions: tuple[tuple[str, int], ...]
    affine_periods: tuple[tuple[str, int | None], ...]
    affine_phases: tuple[tuple[str, str, int, int], ...]
    slot_names: tuple[str, ...]
    slots: tuple[Slot, ...]
    scalar_fp_indices: tuple[int, ...]
    scalar_fp_results: tuple[Slot, ...]
    vector_element_latency: int | None
    include_matrix_writes: bool
    include_vector_writes: bool
    include_vector_port_a_writes: bool
    matrix_access_envelopes: tuple[AddressRange, ...] | None
    vector_access_envelopes: tuple[AddressRange, ...] | None
    matrix_writes: tuple[PendingWrite, ...]
    vector_writes: tuple[PendingWrite, ...]
    vector_port_a_writes: tuple[BusyInterval, ...]
    stall_cycles: tuple[tuple[str, int], ...]
    resource_work: tuple[tuple[str, int], ...]
    critical_path: tuple[tuple[str, int], ...]
    status_counts: tuple[tuple[str, int], ...]
    unsupported_counts: tuple[tuple[str, int], ...]
    out_of_domain_counts: tuple[tuple[str, int], ...]
    hbm_provider_state: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _SteadyTransition:
    """One affine repeat transition after time/sequence normalization."""

    cycle_delta: int
    sequence_delta: int
    gp_delta: tuple[int | None, ...]
    affine_position_delta: tuple[tuple[str, int], ...]
    affine_phases: tuple[tuple[str, str, int, int], ...]
    normalized_makespan: int
    slot_names: tuple[str, ...]
    normalized_slots: tuple[tuple[int, int | None], ...]
    scalar_fp_indices: tuple[int, ...]
    normalized_scalar_fp_results: tuple[tuple[int, int | None], ...]
    vector_element_latency: int | None
    include_matrix_writes: bool
    include_vector_writes: bool
    include_vector_port_a_writes: bool
    matrix_access_envelopes: tuple[AddressRange, ...] | None
    vector_access_envelopes: tuple[AddressRange, ...] | None
    matrix_write_shape: tuple[tuple[int, int, int, int | None], ...]
    matrix_write_address_delta: tuple[int, ...]
    vector_write_shape: tuple[tuple[int, int, int, int | None], ...]
    vector_write_address_delta: tuple[int, ...]
    vector_port_shape: tuple[tuple[int, int, int | None], ...]
    counter_delta: tuple[tuple[tuple[str, int], ...], ...]
    hbm_provider_phase_before: tuple[tuple[int, int], ...]
    hbm_provider_phase_after: tuple[tuple[int, int], ...]
    hbm_provider_count_delta: tuple[tuple[int, int], ...]
    dma_occurrences: tuple[_RelativeDmaOccurrence, ...] = ()


@dataclass(frozen=True)
class _CachedRepeatEffect:
    """Exact final local state for a previously replayed repeat.

    Cache entries are keyed by the complete repeat IR plus an absolute GP and
    normalized scoreboard entry signature.  Reuse therefore skips duplicate
    probing only when the scheduler is in the same observable state; it is not
    a throughput approximation.
    """

    cycle_delta: int
    sequence_delta: int
    gp_values: tuple[tuple[int, int | None], ...]
    affine_positions: tuple[tuple[str, int], ...]
    slot_names: tuple[str, ...]
    normalized_slots: tuple[tuple[int, int | None], ...]
    scalar_fp_indices: tuple[int, ...]
    normalized_scalar_fp_results: tuple[tuple[int, int | None], ...]
    vector_element_latency: int | None
    include_matrix_writes: bool
    include_vector_writes: bool
    include_vector_port_a_writes: bool
    matrix_access_envelopes: tuple[AddressRange, ...] | None
    vector_access_envelopes: tuple[AddressRange, ...] | None
    matrix_writes: tuple[tuple[int, int, int, int | None], ...]
    vector_writes: tuple[tuple[int, int, int, int | None], ...]
    vector_port_a_writes: tuple[tuple[int, int, int | None], ...]
    normalized_makespan: int
    counter_delta: tuple[tuple[tuple[str, int], ...], ...]
    hbm_provider_count_delta: tuple[tuple[int, int], ...]
    dma_occurrences: tuple[_RelativeDmaOccurrence, ...] = ()


@cache
def resource_for(opcode: str) -> str:
    if opcode == "H_PREFETCH_M":
        return "hbm_matrix_dma"
    if opcode == "H_PREFETCH_V":
        return "hbm_vector_dma"
    if opcode == "H_STORE_V":
        return "hbm_vector_store"
    if opcode in MATRIX_COMPUTE:
        return "matrix_compute"
    if opcode in MATRIX_WRITEOUT:
        return "matrix_writeout"
    if opcode in VECTOR_OPS:
        return "vector_pipeline"
    if opcode in SCALAR_OPS:
        return "scalar_pipeline"
    if opcode in CONTROL_OPS:
        return "control_frontend"
    raise ScheduleUnavailableError(f"no rtl-v1 resource for opcode {opcode!r}")


@cache
def _register_index(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise ScheduleUnavailableError(f"expected {prefix} register, got {value!r}")
    return int(value.removeprefix(prefix))


@cache
def _integer(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise ScheduleUnavailableError(f"non-constant ISA argument {value!r}") from exc


class RtlShadowScheduler:
    _SLOT_NAMES = (
        "hbm_shared",
        "hbm_matrix",
        "hbm_vector",
        "hbm_store",
        "matrix_compute",
        "matrix_writeout",
        "vector_pipeline",
        "scalar_fp_compute",
        "scalar_sram",
        "vector_reduction_result",
        "vector_element_result",
    )

    def __init__(
        self,
        *,
        hardware: TimingHardware,
        precision: ComputePrecisionConfig,
        calibration: RtlOpcodeTimingCalibration,
        memory_events: Mapping[int, Any],
        hbm_service_cycles: Callable[[ScheduleInstruction, int], int] | None,
        hbm_fidelity: str,
        retain_events: bool,
        max_expanded_instructions: int,
        initial_gp: Mapping[int, int] | None = None,
    ) -> None:
        self.hardware = hardware
        self.precision = precision
        self.calibration = calibration
        self.memory_events = memory_events
        self.hbm_service_cycles = hbm_service_cycles
        self.hbm_fidelity = hbm_fidelity
        self.retain_events = retain_events
        self.max_expanded_instructions = max_expanded_instructions
        self.next_issue_cycle = 0
        self.makespan_cycles = 0
        self.sequence = 0
        self.gp: list[int | None] = [0] * 32
        for register, value in (initial_gp or {}).items():
            self.gp[int(register)] = int(value) & 0xFFFFFFFF
        self.affine_positions: dict[str, int] = {}
        self.affine_specs: dict[
            str, ScheduleAffineLoad | ScheduleAffineAdd
        ] = {}
        self.hbm_shared = Slot()
        self.hbm_matrix = Slot()
        self.hbm_vector = Slot()
        self.hbm_store = Slot()
        self.matrix_compute = Slot()
        self.matrix_writeout = Slot()
        self.vector_pipeline = Slot()
        self.scalar_fp_compute = Slot()
        self.scalar_sram = Slot()
        self.scalar_fp_results = [Slot() for _ in range(32)]
        self.vector_reduction_result = Slot()
        self.vector_element_result = Slot()
        self.vector_element_latency: int | None = None
        self.matrix_writes: list[PendingWrite] = []
        self.vector_writes: list[PendingWrite] = []
        self.vector_port_a_writes: list[BusyInterval] = []
        self.events: list[ScheduledEvent] = []
        # Retain only DMA events independently of full event tracing. Its size
        # follows physical memory occurrences rather than total dynamic ISA.
        self.dma_occurrences: list[ScheduledDmaOccurrence] = []
        self.expanded_instruction_count = 0
        self.repeat_fast_forwards = 0
        self.fast_forwarded_iterations = 0
        self.fast_forwarded_dynamic_instructions = 0
        self.stall_cycles: Counter[str] = Counter()
        self.resource_work: Counter[str] = Counter()
        self.critical_path: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.unsupported_counts: Counter[str] = Counter()
        self.out_of_domain_counts: Counter[str] = Counter()
        self.repeat_stack: list[str] = []
        self.repeat_effect_cache: dict[tuple[Any, ...], _CachedRepeatEffect] = {}
        self.repeat_cache_hits = 0
        self.repeat_cache_hits_by_name: Counter[str] = Counter()
        self.repeat_cache_misses_by_name: Counter[str] = Counter()
        self.expanded_by_repeat: Counter[str] = Counter()
        # Dependency sequence numbers are needed only while their scoreboard
        # entries remain live.  The map is periodically compacted so a
        # multi-million-instruction validation replay does not retain one
        # Python object per retired instruction.
        self.sequence_resources: dict[int, str] = {}
        self.unresolved_dependency_resources = 0
        self.makespan_resource: str | None = None
        self.opcode_timing_cache: dict[str, OpcodeTimingEstimate] = {}

    @staticmethod
    def _counter_items(counter: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((str(key), int(value)) for key, value in counter.items()))

    def _register_active_sequence_resources(self) -> None:
        """Rebuild owners for live scoreboard dependencies after fast-forward.

        Repeat fast-forward shifts sequence numbers algebraically.  Registering
        the shifted live slots keeps later critical-path ownership exact even
        though the skipped instructions were not materialized individually.
        More specific producer slots are visited before generic result slots.
        """

        slot_resources = (
            (self.hbm_matrix, "hbm_matrix_dma"),
            (self.hbm_vector, "hbm_vector_dma"),
            (self.hbm_store, "hbm_vector_store"),
            (self.matrix_compute, "matrix_compute"),
            (self.matrix_writeout, "matrix_writeout"),
            (self.vector_pipeline, "vector_pipeline"),
            (self.vector_reduction_result, "vector_pipeline"),
            (self.vector_element_result, "vector_pipeline"),
            (self.scalar_fp_compute, "scalar_pipeline"),
            (self.scalar_sram, "scalar_pipeline"),
        )
        for slot, resource in slot_resources:
            if slot.sequence is not None:
                self.sequence_resources[slot.sequence] = resource
        for slot in self.scalar_fp_results:
            if slot.sequence is not None:
                self.sequence_resources.setdefault(slot.sequence, "scalar_pipeline")
        for write in self.matrix_writes:
            if write.slot.sequence is not None:
                self.sequence_resources.setdefault(
                    write.slot.sequence, "hbm_matrix_dma"
                )
        for write in self.vector_writes:
            if write.slot.sequence is not None:
                self.sequence_resources.setdefault(
                    write.slot.sequence, "vector_pipeline"
                )
        for interval in self.vector_port_a_writes:
            if interval.sequence is not None:
                self.sequence_resources.setdefault(
                    interval.sequence, "vector_pipeline"
                )

    def _compact_sequence_resources(self) -> None:
        """Drop retired dependency owners while retaining every live producer."""

        self._register_active_sequence_resources()
        live: set[int] = set()
        for name in self._SLOT_NAMES:
            sequence = getattr(self, name).sequence
            if sequence is not None:
                live.add(sequence)
        live.update(
            slot.sequence
            for slot in self.scalar_fp_results
            if slot.sequence is not None
        )
        live.update(
            write.slot.sequence
            for write in (*self.matrix_writes, *self.vector_writes)
            if write.slot.sequence is not None
        )
        live.update(
            interval.sequence
            for interval in self.vector_port_a_writes
            if interval.sequence is not None
        )
        self.sequence_resources = {
            sequence: self.sequence_resources[sequence]
            for sequence in live
            if sequence in self.sequence_resources
        }

    def _completion_owner(self, cycle: int) -> str | None:
        """Return the later-sequence resource that owns a completion tie."""

        sequences: list[int] = []
        for name in self._SLOT_NAMES:
            slot = getattr(self, name)
            if slot.until == cycle and slot.sequence is not None:
                sequences.append(slot.sequence)
        sequences.extend(
            slot.sequence
            for slot in self.scalar_fp_results
            if slot.until == cycle and slot.sequence is not None
        )
        sequences.extend(
            write.slot.sequence
            for write in (*self.matrix_writes, *self.vector_writes)
            if write.slot.until == cycle and write.slot.sequence is not None
        )
        if not sequences:
            return None
        return self.sequence_resources.get(max(sequences))

    @staticmethod
    def _active_slot(slot: Slot, boundary: int) -> Slot:
        return slot if slot.until > boundary else Slot()

    def _snapshot(
        self,
        affine_keys: frozenset[str] | None = None,
        slot_names: frozenset[str] | None = None,
        scalar_fp_indices: frozenset[int] | None = None,
        *,
        include_matrix_writes: bool = True,
        include_vector_writes: bool = True,
        include_vector_port_a_writes: bool = True,
        matrix_access_envelopes: tuple[AddressRange, ...] | None = None,
        vector_access_envelopes: tuple[AddressRange, ...] | None = None,
    ) -> _SchedulerSnapshot:
        boundary = self.next_issue_cycle
        active_affine_keys = (
            frozenset(self.affine_specs)
            if affine_keys is None
            else affine_keys & self.affine_specs.keys()
        )
        active_slot_names = tuple(
            name
            for name in self._SLOT_NAMES
            if slot_names is None or name in slot_names
        )
        active_scalar_indices = tuple(
            range(len(self.scalar_fp_results))
            if scalar_fp_indices is None
            else sorted(scalar_fp_indices)
        )
        slots = tuple(
            self._active_slot(getattr(self, name), boundary)
            for name in active_slot_names
        )
        scalar_results = tuple(
            self._active_slot(self.scalar_fp_results[index], boundary)
            for index in active_scalar_indices
        )
        matrix_writes = (
            tuple(
                item
                for item in self.matrix_writes
                if item.slot.until > boundary
                and (
                    matrix_access_envelopes is None
                    or any(
                        item.address_range.overlaps(envelope)
                        for envelope in matrix_access_envelopes
                    )
                )
            )
            if include_matrix_writes
            else ()
        )
        vector_writes = (
            tuple(
                item
                for item in self.vector_writes
                if item.slot.until > boundary
                and (
                    vector_access_envelopes is None
                    or any(
                        item.address_range.overlaps(envelope)
                        for envelope in vector_access_envelopes
                    )
                )
            )
            if include_vector_writes
            else ()
        )
        vector_port_writes = (
            tuple(item for item in self.vector_port_a_writes if item.end > boundary)
            if include_vector_port_a_writes
            else ()
        )
        local_makespan = max(
            (
                boundary,
                *(slot.until for slot in slots),
                *(slot.until for slot in scalar_results),
                *(item.slot.until for item in matrix_writes),
                *(item.slot.until for item in vector_writes),
            )
        )
        provider_snapshot = getattr(self.hbm_service_cycles, "snapshot_state", None)
        hbm_provider_state = (
            tuple(provider_snapshot()) if callable(provider_snapshot) else ()
        )
        return _SchedulerSnapshot(
            next_issue_cycle=boundary,
            local_makespan_cycles=local_makespan,
            sequence=self.sequence,
            gp=tuple(self.gp),
            affine_positions=tuple(
                sorted(
                    (key, self.affine_positions[key])
                    for key in active_affine_keys
                )
            ),
            affine_periods=tuple(
                sorted(
                    (key, self.affine_specs[key].period)
                    for key in active_affine_keys
                )
            ),
            affine_phases=tuple(
                sorted(
                    (
                        key,
                        *self._affine_phase(spec, self.affine_positions.get(key, 0)),
                    )
                    for key in active_affine_keys
                    for spec in (self.affine_specs[key],)
                )
            ),
            slot_names=active_slot_names,
            slots=slots,
            scalar_fp_indices=active_scalar_indices,
            scalar_fp_results=scalar_results,
            vector_element_latency=self.vector_element_latency,
            include_matrix_writes=include_matrix_writes,
            include_vector_writes=include_vector_writes,
            include_vector_port_a_writes=include_vector_port_a_writes,
            matrix_access_envelopes=matrix_access_envelopes,
            vector_access_envelopes=vector_access_envelopes,
            matrix_writes=matrix_writes,
            vector_writes=vector_writes,
            vector_port_a_writes=vector_port_writes,
            stall_cycles=self._counter_items(self.stall_cycles),
            resource_work=self._counter_items(self.resource_work),
            critical_path=self._counter_items(self.critical_path),
            status_counts=self._counter_items(self.status_counts),
            unsupported_counts=self._counter_items(self.unsupported_counts),
            out_of_domain_counts=self._counter_items(self.out_of_domain_counts),
            hbm_provider_state=hbm_provider_state,
        )

    def _repeat_access_envelopes(
        self, body: ScheduleSequence, count: int
    ) -> tuple[tuple[AddressRange, ...] | None, tuple[AddressRange, ...] | None]:
        """Conservatively bound SRAM addresses touched by a simple loop.

        The scheduler uses this only to discard pre-existing pending writes
        that cannot overlap any iteration of the repeat.  We therefore return
        ``None`` (meaning no filtering) whenever the body is nested, contains
        an affine IR node, wraps the 32-bit address space, or does not show the
        same address delta across three literal iterations.

        Most compiler hardware loops consist of ordinary instructions plus
        constant S_ADDI pointer bumps.  For those loops, three dry iterations
        prove a constant per-iteration address delta, and the first/last
        ranges form an exact conservative envelope for all iterations.
        """
        if count <= 0:
            return (), ()

        def is_simple(node: ScheduleNode) -> bool:
            if isinstance(node, ScheduleInstruction):
                return True
            if isinstance(node, ScheduleSequence):
                return all(is_simple(child) for child in node.children)
            return False

        if not is_simple(body):
            return None, None

        saved_gp = self.gp
        saved_affine_positions = self.affine_positions
        saved_affine_specs = self.affine_specs
        self.gp = list(saved_gp)
        self.affine_positions = dict(saved_affine_positions)
        self.affine_specs = dict(saved_affine_specs)

        def run_once(node: ScheduleNode) -> tuple[list[AddressRange], list[AddressRange]]:
            matrix: list[AddressRange] = []
            vector: list[AddressRange] = []

            def visit(current: ScheduleNode) -> None:
                if isinstance(current, ScheduleInstruction):
                    accesses = self._accesses(current)
                    matrix.extend(accesses.matrix_reads)
                    matrix.extend(accesses.matrix_writes)
                    vector.extend(accesses.vector_reads)
                    vector.extend(accesses.vector_writes)
                    self._execute_integer_side_effect(current)
                    return
                if isinstance(current, ScheduleSequence):
                    for child in current.children:
                        visit(child)
                    return
                raise ScheduleUnavailableError(
                    "non-simple node reached repeat access-envelope analysis"
                )

            visit(node)
            return matrix, vector

        try:
            iterations = [run_once(body) for _ in range(3)]
        except (ScheduleUnavailableError, IndexError, ValueError):
            return None, None
        finally:
            self.gp = saved_gp
            self.affine_positions = saved_affine_positions
            self.affine_specs = saved_affine_specs

        def envelopes(domain: int) -> tuple[AddressRange, ...] | None:
            first, second, third = (
                iterations[index][domain] for index in range(3)
            )
            if not (len(first) == len(second) == len(third)):
                return None
            result: list[AddressRange] = []
            for left, middle, right in zip(first, second, third, strict=True):
                length = left.end - left.start
                if (
                    middle.end - middle.start != length
                    or right.end - right.start != length
                ):
                    return None
                delta = middle.start - left.start
                if right.start - middle.start != delta:
                    return None
                final_start = left.start + delta * (count - 1)
                low = min(left.start, final_start)
                high = max(left.start, final_start) + length
                if low < 0 or high > 1 << 32:
                    return None
                result.append(AddressRange(low, high))
            return tuple(result)

        return envelopes(0), envelopes(1)

    @staticmethod
    def _affine_phase(
        spec: ScheduleAffineLoad | ScheduleAffineAdd, position: int
    ) -> tuple[str, int, int]:
        if spec.advance_every <= 0:
            raise ScheduleUnavailableError(
                f"affine load {spec.key!r} has invalid advance_every "
                f"{spec.advance_every}"
            )
        value = spec.start + spec.step * (position // spec.advance_every)
        low = value & 0xFFF
        return (
            "below" if value < IMM2_BOUND else "above",
            0 if value < IMM2_BOUND or low == 0 else 1,
            position % spec.advance_every,
        )

    @staticmethod
    def _normalized_slot(
        slot: Slot, snapshot: _SchedulerSnapshot
    ) -> tuple[int, int | None]:
        if slot.until <= snapshot.next_issue_cycle:
            return (0, None)
        sequence = (
            None if slot.sequence is None else slot.sequence - snapshot.sequence
        )
        return (slot.until - snapshot.next_issue_cycle, sequence)

    @staticmethod
    def _counter_delta(
        before: tuple[tuple[str, int], ...],
        after: tuple[tuple[str, int], ...],
    ) -> tuple[tuple[str, int], ...] | None:
        left = dict(before)
        right = dict(after)
        result = []
        for key in sorted(left.keys() | right.keys()):
            delta = right.get(key, 0) - left.get(key, 0)
            if delta < 0:
                return None
            if delta:
                result.append((key, delta))
        return tuple(result)

    @classmethod
    def _transition(
        cls,
        before: _SchedulerSnapshot,
        after: _SchedulerSnapshot,
        dma_occurrences: tuple[_RelativeDmaOccurrence, ...] = (),
    ) -> _SteadyTransition | None:
        cycle_delta = after.next_issue_cycle - before.next_issue_cycle
        sequence_delta = after.sequence - before.sequence
        if cycle_delta <= 0 or sequence_delta <= 0:
            return None

        before_provider = {
            stream: (position, phase)
            for stream, position, phase in before.hbm_provider_state
        }
        after_provider = {
            stream: (position, phase)
            for stream, position, phase in after.hbm_provider_state
        }
        if before_provider.keys() != after_provider.keys():
            return None
        provider_delta = []
        for stream in sorted(before_provider):
            delta = after_provider[stream][0] - before_provider[stream][0]
            if delta < 0:
                return None
            if delta:
                provider_delta.append((stream, delta))
        provider_phase_before = tuple(
            (stream, before_provider[stream][1]) for stream in sorted(before_provider)
        )
        provider_phase_after = tuple(
            (stream, after_provider[stream][1]) for stream in sorted(after_provider)
        )

        gp_delta: list[int | None] = []
        for left, right in zip(before.gp, after.gp, strict=True):
            if left is None or right is None:
                if left != right:
                    return None
                gp_delta.append(None)
            else:
                gp_delta.append((right - left) & 0xFFFFFFFF)

        if before.affine_periods != after.affine_periods:
            return None
        if (
            before.slot_names != after.slot_names
            or before.scalar_fp_indices != after.scalar_fp_indices
            or before.include_matrix_writes != after.include_matrix_writes
            or before.include_vector_writes != after.include_vector_writes
            or before.include_vector_port_a_writes
            != after.include_vector_port_a_writes
            or before.matrix_access_envelopes
            != after.matrix_access_envelopes
            or before.vector_access_envelopes
            != after.vector_access_envelopes
        ):
            return None
        before_positions = dict(before.affine_positions)
        after_positions = dict(after.affine_positions)
        if before_positions.keys() != after_positions.keys():
            return None
        affine_periods = dict(after.affine_periods)
        affine_delta = []
        for key in sorted(after_positions):
            delta = after_positions[key] - before_positions[key]
            period = affine_periods[key]
            if period is not None:
                delta %= period
            affine_delta.append((key, delta))

        def pending_transition(
            left: tuple[PendingWrite, ...],
            right: tuple[PendingWrite, ...],
        ) -> tuple[
            tuple[tuple[int, int, int, int | None], ...], tuple[int, ...]
        ] | None:
            if len(left) != len(right):
                return None
            shape = []
            address_delta = []
            for previous, current in zip(left, right, strict=True):
                previous_length = (
                    previous.address_range.end - previous.address_range.start
                )
                current_length = current.address_range.end - current.address_range.start
                if previous_length != current_length:
                    return None
                relative_until, relative_sequence = cls._normalized_slot(
                    current.slot, after
                )
                shape.append(
                    (
                        current_length,
                        relative_until,
                        current.slot.until - after.next_issue_cycle,
                        relative_sequence,
                    )
                )
                address_delta.append(
                    (current.address_range.start - previous.address_range.start)
                    & 0xFFFFFFFF
                )
            return tuple(shape), tuple(address_delta)

        matrix_pending = pending_transition(before.matrix_writes, after.matrix_writes)
        vector_pending = pending_transition(before.vector_writes, after.vector_writes)
        if matrix_pending is None or vector_pending is None:
            return None

        if len(before.vector_port_a_writes) != len(after.vector_port_a_writes):
            return None
        vector_port_shape = tuple(
            (
                item.start - after.next_issue_cycle,
                item.end - after.next_issue_cycle,
                None if item.sequence is None else item.sequence - after.sequence,
            )
            for item in after.vector_port_a_writes
        )

        counter_delta = []
        for left, right in (
            (before.stall_cycles, after.stall_cycles),
            (before.resource_work, after.resource_work),
            (before.critical_path, after.critical_path),
            (before.status_counts, after.status_counts),
            (before.unsupported_counts, after.unsupported_counts),
            (before.out_of_domain_counts, after.out_of_domain_counts),
        ):
            delta = cls._counter_delta(left, right)
            if delta is None:
                return None
            counter_delta.append(delta)

        return _SteadyTransition(
            cycle_delta=cycle_delta,
            sequence_delta=sequence_delta,
            gp_delta=tuple(gp_delta),
            affine_position_delta=tuple(affine_delta),
            affine_phases=after.affine_phases,
            normalized_makespan=max(
                0,
                after.local_makespan_cycles - after.next_issue_cycle,
            ),
            slot_names=after.slot_names,
            normalized_slots=tuple(
                cls._normalized_slot(slot, after) for slot in after.slots
            ),
            scalar_fp_indices=after.scalar_fp_indices,
            normalized_scalar_fp_results=tuple(
                cls._normalized_slot(slot, after)
                for slot in after.scalar_fp_results
            ),
            vector_element_latency=after.vector_element_latency,
            include_matrix_writes=after.include_matrix_writes,
            include_vector_writes=after.include_vector_writes,
            include_vector_port_a_writes=after.include_vector_port_a_writes,
            matrix_access_envelopes=after.matrix_access_envelopes,
            vector_access_envelopes=after.vector_access_envelopes,
            matrix_write_shape=matrix_pending[0],
            matrix_write_address_delta=matrix_pending[1],
            vector_write_shape=vector_pending[0],
            vector_write_address_delta=vector_pending[1],
            vector_port_shape=vector_port_shape,
            counter_delta=tuple(counter_delta),
            hbm_provider_phase_before=provider_phase_before,
            hbm_provider_phase_after=provider_phase_after,
            hbm_provider_count_delta=tuple(provider_delta),
            dma_occurrences=dma_occurrences,
        )

    def _relative_dma_occurrences(
        self,
        before: _SchedulerSnapshot,
        after: _SchedulerSnapshot,
    ) -> tuple[_RelativeDmaOccurrence, ...]:
        """Normalize DMA events belonging to ``[before, after)``."""

        return tuple(
            _RelativeDmaOccurrence(
                sequence_offset=event.sequence - before.sequence,
                stream_index=event.stream_index,
                opcode=event.opcode,
                start_cycle_offset=event.start_cycle - before.next_issue_cycle,
                completion_cycle_offset=(
                    event.completion_cycle - before.next_issue_cycle
                ),
            )
            for event in self.dma_occurrences
            if before.sequence <= event.sequence < after.sequence
        )

    def _append_relative_dma_occurrences(
        self,
        occurrences: tuple[_RelativeDmaOccurrence, ...],
        *,
        boundary: int,
        sequence: int,
        applications: int = 1,
        cycle_delta: int = 0,
        sequence_delta: int = 0,
    ) -> None:
        """Materialize DMA events skipped by an exact repeat transition."""

        for application in range(applications):
            cycle_base = boundary + application * cycle_delta
            sequence_base = sequence + application * sequence_delta
            self.dma_occurrences.extend(
                ScheduledDmaOccurrence(
                    sequence=sequence_base + event.sequence_offset,
                    stream_index=event.stream_index,
                    opcode=event.opcode,
                    start_cycle=cycle_base + event.start_cycle_offset,
                    completion_cycle=cycle_base + event.completion_cycle_offset,
                )
                for event in occurrences
            )

    @staticmethod
    def _shift_slot(
        slot: Slot,
        *,
        boundary: int,
        cycle_delta: int,
        sequence_delta: int,
    ) -> Slot:
        if slot.until <= boundary:
            return slot
        return Slot(
            slot.until + cycle_delta,
            None if slot.sequence is None else slot.sequence + sequence_delta,
        )

    def _apply_steady_transition(
        self,
        transition: _SteadyTransition,
        applications: int,
        *,
        loop_iterations_per_transition: int = 1,
    ) -> None:
        if applications <= 0:
            return
        boundary = self.next_issue_cycle
        cycle_delta = transition.cycle_delta * applications
        sequence_delta = transition.sequence_delta * applications
        self._append_relative_dma_occurrences(
            transition.dma_occurrences,
            boundary=boundary,
            sequence=self.sequence,
            applications=applications,
            cycle_delta=transition.cycle_delta,
            sequence_delta=transition.sequence_delta,
        )

        for name in transition.slot_names:
            setattr(
                self,
                name,
                self._shift_slot(
                    getattr(self, name),
                    boundary=boundary,
                    cycle_delta=cycle_delta,
                    sequence_delta=sequence_delta,
                ),
            )
        for index in transition.scalar_fp_indices:
            self.scalar_fp_results[index] = self._shift_slot(
                self.scalar_fp_results[index],
                boundary=boundary,
                cycle_delta=cycle_delta,
                sequence_delta=sequence_delta,
            )

        def shift_pending(
            items: list[PendingWrite],
            address_delta: tuple[int, ...],
            access_envelopes: tuple[AddressRange, ...] | None,
        ) -> list[PendingWrite]:
            active = [item for item in items if item.slot.until > boundary]
            shifted = []
            delta_index = 0
            for item in active:
                relevant = access_envelopes is None or any(
                    item.address_range.overlaps(envelope)
                    for envelope in access_envelopes
                )
                if not relevant:
                    # This completion belongs to an outer/earlier operation.
                    # It keeps absolute time and sequence while the local
                    # repeat is fast-forwarded.
                    shifted.append(item)
                    continue
                if delta_index >= len(address_delta):
                    raise ScheduleUnavailableError(
                        "scoreboard pending-write shape changed during fast-forward"
                    )
                per_iteration = address_delta[delta_index]
                delta_index += 1
                start = (
                    item.address_range.start + per_iteration * applications
                ) & 0xFFFFFFFF
                length = item.address_range.end - item.address_range.start
                shifted.append(
                    PendingWrite(
                        AddressRange.from_length(start, length),
                        self._shift_slot(
                            item.slot,
                            boundary=boundary,
                            cycle_delta=cycle_delta,
                            sequence_delta=sequence_delta,
                        ),
                    )
                )
            if delta_index != len(address_delta):
                raise ScheduleUnavailableError(
                    "scoreboard pending-write shape changed during fast-forward"
                )
            return shifted

        if transition.include_matrix_writes:
            self.matrix_writes = shift_pending(
                self.matrix_writes,
                transition.matrix_write_address_delta,
                transition.matrix_access_envelopes,
            )
        if transition.include_vector_writes:
            self.vector_writes = shift_pending(
                self.vector_writes,
                transition.vector_write_address_delta,
                transition.vector_access_envelopes,
            )
        if transition.include_vector_port_a_writes:
            self.vector_port_a_writes = [
                BusyInterval(
                    item.start + cycle_delta,
                    item.end + cycle_delta,
                    None if item.sequence is None else item.sequence + sequence_delta,
                )
                for item in self.vector_port_a_writes
                if item.end > boundary
            ]

        for index, delta in enumerate(transition.gp_delta):
            if delta is not None:
                assert self.gp[index] is not None
                self.gp[index] = (
                    int(self.gp[index]) + delta * applications
                ) & 0xFFFFFFFF
        for key, delta in transition.affine_position_delta:
            position = self.affine_positions[key] + delta * applications
            period = self.affine_specs[key].period
            self.affine_positions[key] = (
                position if period is None else position % period
            )

        counters = (
            self.stall_cycles,
            self.resource_work,
            self.critical_path,
            self.status_counts,
            self.unsupported_counts,
            self.out_of_domain_counts,
        )
        for counter, increments in zip(
            counters, transition.counter_delta, strict=True
        ):
            for key, increment in increments:
                counter[key] += increment * applications

        advance_provider = getattr(
            self.hbm_service_cycles, "advance_stream_counts", None
        )
        if transition.hbm_provider_count_delta:
            if not callable(advance_provider):
                raise ScheduleUnavailableError(
                    "DMA timing provider cannot follow an exact repeat fast-forward"
                )
            advance_provider(
                {
                    stream: count * applications
                    for stream, count in transition.hbm_provider_count_delta
                }
            )

        self.next_issue_cycle += cycle_delta
        self.sequence += sequence_delta
        self._register_active_sequence_resources()
        candidate_makespan = self.next_issue_cycle + transition.normalized_makespan
        if candidate_makespan >= self.makespan_cycles:
            self.makespan_cycles = candidate_makespan
            self.makespan_resource = (
                self._completion_owner(candidate_makespan)
                or self.makespan_resource
            )
        self.repeat_fast_forwards += 1
        self.fast_forwarded_iterations += (
            applications * loop_iterations_per_transition
        )
        self.fast_forwarded_dynamic_instructions += sequence_delta

    def _safe_affine_applications(
        self, transition: _SteadyTransition, requested: int
    ) -> int:
        """Cap a fast-forward before a non-periodic affine load changes regime.

        Below/above-``IMM2_BOUND`` loads legalize to different opcode
        sequences.  A transition observed entirely below the threshold must
        not be extrapolated through it.  Negative values are another hard
        boundary because the explicit visitor rejects them.  Periodic affine
        streams are safe: their complete period is part of the normalized
        state equality check.
        """
        safe = requested
        for key, position_delta in transition.affine_position_delta:
            spec = self.affine_specs[key]
            if position_delta <= 0:
                continue
            if position_delta % spec.advance_every:
                return 0
            logical_delta = position_delta // spec.advance_every
            if logical_delta <= 0:
                continue
            position = self.affine_positions[key]
            if position % spec.advance_every:
                return 0
            if spec.period is not None:
                positions_to_wrap = spec.period - position
                safe = min(safe, positions_to_wrap // position_delta)
            logical_position = position // spec.advance_every
            value = spec.start + spec.step * logical_position
            if spec.step > 0 and value < IMM2_BOUND:
                values_in_regime = (
                    IMM2_BOUND - value + spec.step - 1
                ) // spec.step
                safe = min(safe, values_in_regime // logical_delta)
            elif spec.step < 0:
                magnitude = -spec.step
                values_until_negative = value // magnitude + 1
                values_in_regime = values_until_negative
                if value >= IMM2_BOUND:
                    values_in_regime = min(
                        values_in_regime,
                        (value - IMM2_BOUND) // magnitude + 1,
                    )
                safe = min(safe, values_in_regime // logical_delta)
            if value >= IMM2_BOUND and spec.step:
                modulus = 1 << 12
                divisor = math.gcd(abs(spec.step), modulus)
                legalization_period = (
                    modulus // divisor
                ) * spec.advance_every
                # A measured transition spanning a complete low-immediate
                # period contains every zero/nonzero-low legalization shape.
                # Its normalized state can therefore be repeated without
                # stopping at each 4 KiB boundary.  Shorter transitions still
                # use the conservative boundary cap below.
                if position_delta % legalization_period == 0:
                    continue
                rhs = -value
                if rhs % divisor == 0:
                    reduced_modulus = modulus // divisor
                    if reduced_modulus == 1:
                        next_zero = 0
                    else:
                        reduced_step = (spec.step // divisor) % reduced_modulus
                        next_zero = (
                            (rhs // divisor)
                            * pow(reduced_step, -1, reduced_modulus)
                        ) % reduced_modulus
                    if value & 0xFFF:
                        # Nonzero-low loads remain in the same instruction
                        # shape until the first future zero-low value.
                        if next_zero:
                            safe = min(
                                safe, next_zero // logical_delta
                            )
                    elif reduced_modulus != 1:
                        # The current zero-low load is a one-value regime;
                        # the next value restores the low ADDI.
                        safe = min(safe, 1 // logical_delta)
        return safe

    def _complete_affine_period(
        self,
        transition: _SteadyTransition,
        observed_period: int,
    ) -> int | None:
        """Return a longer exact large-immediate period when one is available.

        A one-iteration transition above ``IMM2_BOUND`` often appears stable
        until the low 12 address bits become zero.  For example, a 512-byte
        stride repeats its LUI/ADDI legalization shape every eight loop
        iterations.  Accepting the shorter transition forces a restart at
        every 4 KiB boundary.  Probe through the full period when it fits in
        the bounded period search.

        Loads below ``IMM2_BOUND`` are deliberately excluded: crossing that
        threshold changes from one instruction to the large-immediate
        sequence and is not a periodic phase that can be extrapolated.
        """

        required_period = 1
        has_incomplete_period = False
        for key, position_delta in transition.affine_position_delta:
            if position_delta <= 0 or position_delta % observed_period:
                continue
            spec = self.affine_specs[key]
            if spec.step == 0 or spec.advance_every <= 0:
                continue
            position = self.affine_positions[key]
            if position % spec.advance_every:
                return None
            logical_position = position // spec.advance_every
            value = spec.start + spec.step * logical_position
            if value < IMM2_BOUND:
                return None

            per_iteration_delta = position_delta // observed_period
            legalization_positions = (
                (1 << 12) // math.gcd(abs(spec.step), 1 << 12)
            ) * spec.advance_every
            stream_period = legalization_positions // math.gcd(
                legalization_positions,
                per_iteration_delta,
            )
            if observed_period % stream_period:
                has_incomplete_period = True
                required_period = math.lcm(required_period, stream_period)
                if required_period > MAX_REPEAT_PERIOD_ITERATIONS:
                    return None
        return required_period if has_incomplete_period else None

    def _gp(self, arg: str) -> int:
        index = _register_index(arg, "gp")
        value = self.gp[index]
        if value is None:
            raise ScheduleUnavailableError(f"gp{index} has unknown value")
        return value

    def _vector_range(self, register: str, rows: int) -> AddressRange:
        return AddressRange.from_length(
            self._gp(register), rows * self.hardware.vlen
        )

    def _matrix_range(self, register: str, elements: int) -> AddressRange:
        return AddressRange.from_length(self._gp(register), elements)

    @staticmethod
    def _matrix_sources(args: tuple[str, ...]) -> tuple[str, str]:
        """Return decoded rs1/rs2 operands from compiler or hand-written IR.

        Compiler assembly retains the encoded ``rd`` field (usually literal
        zero) for matrix compute opcodes, while older differential tests use
        the already-decoded two-register form.  Rust ``Opcode`` drops that rd
        field for MM/TMM/BMM/BTMM, so accesses always use the final two args.
        """
        if len(args) < 2:
            raise ScheduleUnavailableError(
                f"matrix opcode requires rs1/rs2, got {args!r}"
            )
        return args[-2], args[-1]

    def _accesses(self, instruction: ScheduleInstruction) -> InstructionAccesses:
        opcode = instruction.opcode
        args = instruction.args
        mlen = self.hardware.mlen
        blen = self.hardware.blen
        access = InstructionAccesses()
        memory_event = (
            None
            if instruction.memory_stream_index is None
            else self.memory_events.get(instruction.memory_stream_index)
        )
        if opcode == "H_PREFETCH_M":
            amount = mlen if memory_event is None else int(memory_event.transfer.amount)
            access.matrix_writes.append(
                AddressRange.from_length(self._gp(args[0]), amount * mlen)
            )
        elif opcode == "H_PREFETCH_V":
            amount = 1 if memory_event is None else int(memory_event.transfer.amount)
            access.vector_writes.append(self._vector_range(args[0], amount))
        elif opcode == "H_STORE_V":
            amount = 1 if memory_event is None else int(memory_event.transfer.amount)
            access.vector_reads.append(self._vector_range(args[0], amount))
        elif opcode in {"M_MM", "M_BMM", "M_BTMM"}:
            rs1, rs2 = self._matrix_sources(args)
            access.matrix_reads.append(self._matrix_range(rs1, mlen * mlen))
            access.vector_reads.append(self._vector_range(rs2, mlen))
        elif opcode == "M_TMM":
            rs1, rs2 = self._matrix_sources(args)
            access.matrix_reads.append(self._matrix_range(rs2, mlen * mlen))
            access.vector_reads.append(self._vector_range(rs1, mlen))
        elif opcode in {"M_MV", "M_TMV", "M_BMV", "M_BTMV"}:
            rs1, rs2 = self._matrix_sources(args)
            access.matrix_reads.append(self._matrix_range(rs1, mlen * mlen))
            access.vector_reads.append(self._vector_range(rs2, 1))
        elif opcode == "M_MM_WO":
            stride = 1 if _register_index(args[1], "gp") == 0 else self._gp(args[1])
            output = self._gp(args[0]) + _integer(args[2])
            row_base = output // mlen * mlen
            for row in range(blen):
                access.vector_writes.append(
                    AddressRange.from_length(row_base + row * mlen * stride, mlen)
                )
        elif opcode == "M_BMM_WO":
            access.vector_writes.append(
                AddressRange.from_length(
                    self._gp(args[0]) + _integer(args[1]),
                    self.hardware.broadcast_amount * mlen * mlen,
                )
            )
        elif opcode in {"M_MV_WO", "M_BMV_WO"}:
            rows = self.hardware.broadcast_amount if opcode == "M_BMV_WO" else 1
            access.vector_writes.append(
                AddressRange.from_length(
                    self._gp(args[0]) + _integer(args[1]), rows * mlen
                )
            )
        elif opcode in {"V_ADD_VV", "V_SUB_VV", "V_MUL_VV"}:
            access.vector_reads.extend(
                (self._vector_range(args[1], 1), self._vector_range(args[2], 1))
            )
            access.vector_writes.append(self._vector_range(args[0], 1))
        elif opcode in {"V_ADD_VF", "V_SUB_VF", "V_MUL_VF"}:
            access.vector_reads.append(self._vector_range(args[1], 1))
            access.vector_writes.append(self._vector_range(args[0], 1))
            register = _register_index(args[2], "f")
            if register:
                access.scalar_fp_reads.append(register)
        elif opcode in {"V_EXP_V", "V_RECI_V", "V_SHIFT_V"}:
            access.vector_reads.append(self._vector_range(args[1], 1))
            access.vector_writes.append(self._vector_range(args[0], 1))
        elif opcode in VECTOR_REDUCTION:
            access.vector_reads.append(self._vector_range(args[1], 1))
            register = _register_index(args[0], "f")
            if register:
                access.scalar_fp_reads.append(register)
                access.scalar_fp_writes.append(register)
        elif opcode in {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP"}:
            rd, rs1, rs2 = (_register_index(arg, "f") for arg in args[:3])
            access.scalar_fp_reads.extend(register for register in (rs1, rs2) if register)
            if rd:
                access.scalar_fp_writes.append(rd)
        elif opcode in {"S_EXP_FP", "S_RECI_FP", "S_SQRT_FP"}:
            rd, rs1 = (_register_index(arg, "f") for arg in args[:2])
            if rs1:
                access.scalar_fp_reads.append(rs1)
            if rd:
                access.scalar_fp_writes.append(rd)
        elif opcode == "S_LD_FP":
            rd = _register_index(args[0], "f")
            if rd:
                access.scalar_fp_writes.append(rd)
        elif opcode == "S_ST_FP":
            rd = _register_index(args[0], "f")
            if rd:
                access.scalar_fp_reads.append(rd)
        elif opcode == "S_MAP_V_FP":
            access.vector_writes.append(self._vector_range(args[0], 1))
        return access

    @staticmethod
    def _include_slot(
        cycle: int,
        reason: str | None,
        dependency: int | None,
        slot: Slot,
        slot_reason: str,
    ) -> tuple[int, str | None, int | None]:
        if slot.until > cycle:
            return slot.until, slot_reason, slot.sequence
        return cycle, reason, dependency

    def _include_pending(
        self,
        cycle: int,
        reason: str | None,
        dependency: int | None,
        reads: list[AddressRange],
        pending: list[PendingWrite],
        pending_reason: str,
    ) -> tuple[int, str | None, int | None]:
        for write in pending:
            if any(read.overlaps(write.address_range) for read in reads):
                cycle, reason, dependency = self._include_slot(
                    cycle, reason, dependency, write.slot, pending_reason
                )
        return cycle, reason, dependency

    def _timing(
        self, instruction: ScheduleInstruction
    ) -> OpcodeTimingEstimate:
        timing = self.opcode_timing_cache.get(instruction.opcode)
        if timing is None and instruction.opcode not in MEMORY_OPS:
            timing = self.calibration.estimate(
                instruction.opcode, self.hardware, self.precision
            )
            if timing is not None:
                self.opcode_timing_cache[instruction.opcode] = timing
        if timing is not None:
            return timing
        if self.hbm_service_cycles is None:
            raise ScheduleUnavailableError(
                f"no DMA service timing for {instruction.opcode}"
            )
        provider_timing = getattr(self.hbm_service_cycles, "timing_estimate", None)
        if callable(provider_timing):
            timing = provider_timing(instruction, self.sequence)
            if not isinstance(timing, OpcodeTimingEstimate):
                raise ScheduleUnavailableError(
                    "DMA timing provider returned an invalid timing estimate"
                )
            return timing
        cycles = max(1, int(self.hbm_service_cycles(instruction, self.sequence)))
        return OpcodeTimingEstimate(
            resource_cycles=cycles,
            result_ready_cycles=cycles,
            initiation_interval_cycles=cycles,
            calibration_status=self.hbm_fidelity,
            rtl_supported=True,
            calibration_in_domain=self.hbm_fidelity == "ramulator_observed",
        )

    def schedule_instruction(self, instruction: ScheduleInstruction) -> None:
        if self.expanded_instruction_count >= self.max_expanded_instructions:
            raise ScheduleUnavailableError(
                "compressed repeat did not stabilize before expansion limit; "
                f"repeat_path={tuple(self.repeat_stack)}"
            )
        opcode = instruction.opcode
        accesses = self._accesses(instruction)
        timing = self._timing(instruction)
        resource = resource_for(opcode)
        issue = self.next_issue_cycle
        cycle = issue
        reason: str | None = None
        dependency: int | None = None

        self.matrix_writes = [item for item in self.matrix_writes if item.slot.until > issue]
        self.vector_writes = [item for item in self.vector_writes if item.slot.until > issue]
        self.vector_port_a_writes = [
            item for item in self.vector_port_a_writes if item.end > issue
        ]
        for reads, pending, pending_reason in (
            (accesses.matrix_reads, self.matrix_writes, "matrix_sram_operand_not_ready"),
            (accesses.vector_reads, self.vector_writes, "vector_sram_operand_not_ready"),
            (accesses.matrix_writes, self.matrix_writes, "matrix_sram_write_collision"),
            (accesses.vector_writes, self.vector_writes, "vector_sram_write_collision"),
        ):
            cycle, reason, dependency = self._include_pending(
                cycle, reason, dependency, reads, pending, pending_reason
            )
        if opcode in MATRIX_COMPUTE | MATRIX_WRITEOUT | VECTOR_OPS:
            while True:
                conflicts = [
                    interval
                    for interval in self.vector_port_a_writes
                    if interval.start <= cycle < interval.end
                ]
                if not conflicts:
                    break
                conflict = max(conflicts, key=lambda item: item.end)
                cycle, reason, dependency = (
                    conflict.end,
                    "vector_sram_port_a_write",
                    conflict.sequence,
                )

        def include(slot: Slot, slot_reason: str) -> None:
            nonlocal cycle, reason, dependency
            cycle, reason, dependency = self._include_slot(
                cycle, reason, dependency, slot, slot_reason
            )

        if resource == "hbm_matrix_dma":
            include(self.hbm_shared, "hbm_request_port_busy")
            include(self.hbm_matrix, "matrix_dma_busy")
        elif resource == "hbm_vector_dma":
            include(self.hbm_shared, "hbm_request_port_busy")
            include(self.hbm_vector, "vector_dma_busy")
            include(self.matrix_compute, "matrix_vector_sram_conflict")
            include(self.vector_pipeline, "vector_sram_port_busy")
        elif resource == "hbm_vector_store":
            include(self.hbm_shared, "hbm_request_port_busy")
            include(self.hbm_store, "vector_store_busy")
            include(self.matrix_compute, "matrix_vector_sram_conflict")
            include(self.vector_pipeline, "vector_sram_port_busy")
        elif resource == "matrix_compute":
            include(self.matrix_compute, "matrix_mcu_active")
            include(self.matrix_writeout, "matrix_writeout_active")
            include(self.hbm_vector, "vector_prefetch_in_progress")
            include(self.hbm_store, "vector_store_in_progress")
        elif resource == "matrix_writeout":
            include(self.matrix_writeout, "matrix_writeout_active")
        elif resource == "vector_pipeline":
            include(self.vector_pipeline, "vector_pipeline_busy")
            include(self.hbm_vector, "vector_prefetch_in_progress")
            include(self.hbm_store, "vector_store_in_progress")
            if opcode in {"V_ADD_VF", "V_SUB_VF", "V_MUL_VF"}:
                include(self.scalar_fp_compute, "scalar_fp_compute_in_progress")
            if (
                opcode in VECTOR_ELEMENT
                and self.vector_element_latency is not None
                and self.vector_element_latency != timing.result_ready_cycles
            ):
                include(self.vector_element_result, "vector_mixed_latency_in_order")
        elif resource == "scalar_pipeline":
            if opcode in SCALAR_FP_COMPUTE | SCALAR_SRAM:
                include(
                    self.vector_reduction_result,
                    "vector_reduction_result_not_ready",
                )
            if opcode in SCALAR_FP_COMPUTE:
                include(self.scalar_fp_compute, "scalar_fp_compute_in_progress")
            elif opcode in SCALAR_SRAM:
                include(self.scalar_sram, "scalar_fp_sram_busy")
        elif resource == "control_frontend" and opcode == "C_BREAK":
            include(self.hbm_vector, "vector_prefetch_in_progress")

        if resource in {"vector_pipeline", "scalar_pipeline"}:
            for register in accesses.scalar_fp_reads:
                include(self.scalar_fp_results[register], "scalar_fp_operand_not_ready")
            for register in accesses.scalar_fp_writes:
                include(self.scalar_fp_results[register], "scalar_fp_write_port_busy")

        recovery = int(cycle > issue)
        cycle += recovery
        accepted = cycle
        start = accepted
        if resource == "matrix_writeout" and self.matrix_compute.until > start:
            start = self.matrix_compute.until
            dependency = self.matrix_compute.sequence
            reason = "matrix_result_not_ready"
        completion = start + timing.resource_cycles
        ready = start + timing.result_ready_cycles

        blocked_cycles = accepted - issue
        if blocked_cycles > 0:
            dependency_resource = (
                None
                if dependency is None
                else self.sequence_resources.get(dependency)
            )
            if dependency is not None and dependency_resource is None:
                self.unresolved_dependency_resources += 1
            self.critical_path[
                dependency_resource or "control_frontend"
            ] += blocked_cycles
        # Rust's timeline profiler assigns one mutually exclusive frontend
        # cycle to every accepted instruction. Every calibrated opcode has at
        # least one service cycle, so acceptance necessarily precedes the
        # eventual final makespan.
        self.critical_path["control_frontend"] += 1

        resource_slot = Slot(completion, self.sequence)
        result_slot = Slot(ready, self.sequence)

        if resource == "hbm_matrix_dma":
            self.hbm_shared = self.hbm_matrix = resource_slot
        elif resource == "hbm_vector_dma":
            self.hbm_shared = self.hbm_vector = resource_slot
        elif resource == "hbm_vector_store":
            self.hbm_shared = self.hbm_store = resource_slot
        elif resource == "matrix_compute":
            self.matrix_compute = resource_slot
        elif resource == "matrix_writeout":
            self.matrix_writeout = resource_slot
        elif resource == "vector_pipeline":
            self.vector_pipeline = (
                resource_slot
                if opcode in VECTOR_REDUCTION
                else Slot(start + timing.initiation_interval_cycles, self.sequence)
            )
            if opcode in VECTOR_REDUCTION:
                self.vector_reduction_result = result_slot
            elif opcode in VECTOR_ELEMENT:
                self.vector_element_result = result_slot
                self.vector_element_latency = timing.result_ready_cycles
        elif resource == "scalar_pipeline":
            if opcode in SCALAR_FP_COMPUTE:
                self.scalar_fp_compute = resource_slot
            elif opcode in SCALAR_SRAM:
                self.scalar_sram = resource_slot

        self.matrix_writes.extend(
            PendingWrite(address_range, result_slot)
            for address_range in accesses.matrix_writes
        )
        for address_range in accesses.vector_writes:
            self.vector_writes.append(PendingWrite(address_range, result_slot))
        for register in accesses.scalar_fp_writes:
            self.scalar_fp_results[register] = result_slot
        if (
            resource == "vector_pipeline"
            and opcode not in VECTOR_REDUCTION
            and accesses.vector_writes
        ):
            self.vector_port_a_writes.append(
                BusyInterval(ready, ready + 1, self.sequence)
            )

        self.next_issue_cycle = accepted + 1
        if completion >= self.makespan_cycles:
            # Match Rust max_by_key((completion, sequence)): the later
            # instruction owns a tie at the final completion cycle.
            self.makespan_cycles = completion
            self.makespan_resource = resource
        if reason:
            # Match the transactional timeline profiler's accounting.  The
            # raw hazard owns only the cycles before it deasserts; the single
            # registered b1_pipeline_stall replay cycle is reported
            # separately so it is not hidden inside whichever hazard happened
            # to block this instruction.
            frontend_wait = accepted - issue - recovery
            if frontend_wait > 0:
                self.stall_cycles[reason] += frontend_wait
        if recovery:
            self.stall_cycles["pipeline_recovery"] += recovery
        self.resource_work[resource] += timing.resource_cycles
        self.sequence_resources[self.sequence] = resource
        self.status_counts[timing.calibration_status] += 1
        if not timing.rtl_supported:
            self.unsupported_counts[opcode] += 1
        elif not timing.calibration_in_domain:
            self.out_of_domain_counts[opcode] += 1
        event = ScheduledEvent(
            sequence=self.sequence,
            opcode=opcode,
            issue_cycle=issue,
            accepted_cycle=accepted,
            recovery_cycles=recovery,
            start_cycle=start,
            result_ready_cycle=ready,
            completion_cycle=completion,
            resource=resource,
            calibration_status=timing.calibration_status,
            rtl_supported=timing.rtl_supported,
            calibration_in_domain=timing.calibration_in_domain,
            stall_reason=reason,
            dependency=dependency,
        )
        if self.retain_events:
            self.events.append(event)
        if resource in {"hbm_matrix_dma", "hbm_vector_dma", "hbm_vector_store"}:
            if instruction.memory_stream_index is None:
                raise ScheduleUnavailableError(
                    f"DMA opcode {opcode} has no memory_stream_index"
                )
            self.dma_occurrences.append(
                ScheduledDmaOccurrence(
                    sequence=self.sequence,
                    stream_index=int(instruction.memory_stream_index),
                    opcode=opcode,
                    start_cycle=start,
                    completion_cycle=completion,
                )
            )
        self.sequence += 1
        self.expanded_instruction_count += 1
        if self.sequence % 4096 == 0:
            self._compact_sequence_resources()
        repeat_name = self.repeat_stack[-1] if self.repeat_stack else "<top-level>"
        self.expanded_by_repeat[repeat_name] += 1
        self._execute_integer_side_effect(instruction)

    def _execute_integer_side_effect(self, instruction: ScheduleInstruction) -> None:
        opcode = instruction.opcode
        args = instruction.args
        if opcode not in {
            "S_ADD_INT",
            "S_ADDI_INT",
            "S_SUB_INT",
            "S_MUL_INT",
            "S_LUI_INT",
            "S_LD_INT",
        }:
            return
        rd = _register_index(args[0], "gp")
        if rd == 0:
            return
        if opcode == "S_LD_INT":
            self.gp[rd] = None
            return
        if opcode == "S_LUI_INT":
            self.gp[rd] = (_integer(args[1]) << 12) & 0xFFFFFFFF
            return
        lhs = self._gp(args[1])
        rhs = _integer(args[2]) if opcode == "S_ADDI_INT" else self._gp(args[2])
        if opcode in {"S_ADD_INT", "S_ADDI_INT"}:
            value = lhs + rhs
        elif opcode == "S_SUB_INT":
            value = lhs - rhs
        else:
            value = lhs * rhs
        self.gp[rd] = value & 0xFFFFFFFF

    def _visit_affine_load(self, node: ScheduleAffineLoad) -> None:
        previous = self.affine_specs.setdefault(node.key, node)
        if replace(previous, stage=node.stage) != node:
            raise ScheduleUnavailableError(
                f"affine load key {node.key!r} has inconsistent definitions"
            )
        position = self.affine_positions.get(node.key, 0)
        if node.advance_every <= 0:
            raise ScheduleUnavailableError(
                f"affine load {node.key!r} has invalid advance_every "
                f"{node.advance_every}"
            )
        value = node.start + node.step * (position // node.advance_every)
        if value < 0:
            raise ScheduleUnavailableError(
                f"affine load {node.key!r} produced negative value {value}"
            )
        if value < IMM2_BOUND:
            self.schedule_instruction(
                ScheduleInstruction(
                    "S_ADDI_INT",
                    (node.register, "gp0", str(value)),
                    stage=node.stage,
                )
            )
        else:
            self.schedule_instruction(
                ScheduleInstruction(
                    "S_LUI_INT",
                    (node.register, str(value >> 12)),
                    stage=node.stage,
                )
            )
            lower = value & 0xFFF
            if lower:
                self.schedule_instruction(
                    ScheduleInstruction(
                        "S_ADDI_INT",
                        (node.register, node.register, str(lower)),
                        stage=node.stage,
                    )
                )
        next_position = position + 1
        if node.period is not None:
            if node.period <= 0:
                raise ScheduleUnavailableError(
                    f"affine load {node.key!r} has invalid period {node.period}"
                )
            next_position %= node.period
        self.affine_positions[node.key] = next_position

    def _visit_affine_add(self, node: ScheduleAffineAdd) -> None:
        previous = self.affine_specs.setdefault(node.key, node)
        if replace(previous, stage=node.stage) != node:
            raise ScheduleUnavailableError(
                f"affine add key {node.key!r} has inconsistent definitions"
            )
        if node.advance_every <= 0:
            raise ScheduleUnavailableError(
                f"affine add {node.key!r} has invalid advance_every "
                f"{node.advance_every}"
            )
        position = self.affine_positions.get(node.key, 0)
        value = node.start + node.step * (position // node.advance_every)
        if value < 0:
            raise ScheduleUnavailableError(
                f"affine add {node.key!r} produced negative value {value}"
            )
        if value < IMM2_BOUND:
            self.schedule_instruction(
                ScheduleInstruction(
                    "S_ADDI_INT",
                    (node.destination, node.source, str(value)),
                    stage=node.stage,
                )
            )
        else:
            self.schedule_instruction(
                ScheduleInstruction(
                    "S_LUI_INT",
                    (node.temp, str(value >> 12)),
                    stage=node.stage,
                )
            )
            lower = value & 0xFFF
            if lower:
                self.schedule_instruction(
                    ScheduleInstruction(
                        "S_ADDI_INT",
                        (node.temp, node.temp, str(lower)),
                        stage=node.stage,
                    )
                )
            self.schedule_instruction(
                ScheduleInstruction(
                    "S_ADD_INT",
                    (node.destination, node.source, node.temp),
                    stage=node.stage,
                )
            )
        next_position = position + 1
        if node.period is not None:
            if node.period <= 0:
                raise ScheduleUnavailableError(
                    f"affine add {node.key!r} has invalid period {node.period}"
                )
            next_position %= node.period
        self.affine_positions[node.key] = next_position

    def _visit_repeat_iteration(self, node: ScheduleRepeat) -> None:
        self.repeat_stack.append(f"{node.name}[{node.count}]")
        try:
            self.visit(node.body)
        finally:
            self.repeat_stack.pop()

    @classmethod
    def _repeat_cache_key(
        cls,
        node: ScheduleRepeat,
        snapshot: _SchedulerSnapshot,
        gp_indices: frozenset[int],
    ) -> tuple[Any, ...]:
        def pending(
            items: tuple[PendingWrite, ...],
        ) -> tuple[tuple[int, int, int, int | None], ...]:
            return tuple(
                (
                    item.address_range.start,
                    item.address_range.end,
                    *cls._normalized_slot(item.slot, snapshot),
                )
                for item in items
            )

        return (
            node,
            tuple((index, snapshot.gp[index]) for index in sorted(gp_indices)),
            snapshot.affine_positions,
            snapshot.affine_periods,
            snapshot.affine_phases,
            snapshot.slot_names,
            tuple(cls._normalized_slot(slot, snapshot) for slot in snapshot.slots),
            snapshot.scalar_fp_indices,
            tuple(
                cls._normalized_slot(slot, snapshot)
                for slot in snapshot.scalar_fp_results
            ),
            snapshot.vector_element_latency,
            snapshot.include_matrix_writes,
            snapshot.include_vector_writes,
            snapshot.include_vector_port_a_writes,
            snapshot.matrix_access_envelopes,
            snapshot.vector_access_envelopes,
            pending(snapshot.matrix_writes),
            pending(snapshot.vector_writes),
            tuple(
                (
                    item.start - snapshot.next_issue_cycle,
                    item.end - snapshot.next_issue_cycle,
                    None
                    if item.sequence is None
                    else item.sequence - snapshot.sequence,
                )
                for item in snapshot.vector_port_a_writes
            ),
            tuple(
                (stream, phase)
                for stream, _position, phase in snapshot.hbm_provider_state
            ),
        )

    @classmethod
    def _capture_repeat_effect(
        cls,
        before: _SchedulerSnapshot,
        after: _SchedulerSnapshot,
        gp_indices: frozenset[int],
        dma_occurrences: tuple[_RelativeDmaOccurrence, ...] = (),
    ) -> _CachedRepeatEffect:
        counter_delta = []
        for left, right in (
            (before.stall_cycles, after.stall_cycles),
            (before.resource_work, after.resource_work),
            (before.critical_path, after.critical_path),
            (before.status_counts, after.status_counts),
            (before.unsupported_counts, after.unsupported_counts),
            (before.out_of_domain_counts, after.out_of_domain_counts),
        ):
            delta = cls._counter_delta(left, right)
            if delta is None:
                raise ScheduleUnavailableError(
                    "repeat counters decreased while recording exact cache effect"
                )
            counter_delta.append(delta)

        def pending(
            items: tuple[PendingWrite, ...],
        ) -> tuple[tuple[int, int, int, int | None], ...]:
            return tuple(
                (
                    item.address_range.start,
                    item.address_range.end,
                    *cls._normalized_slot(item.slot, after),
                )
                for item in items
            )

        before_provider = {
            stream: position
            for stream, position, _phase in before.hbm_provider_state
        }
        after_provider = {
            stream: position
            for stream, position, _phase in after.hbm_provider_state
        }
        if before_provider.keys() != after_provider.keys():
            raise ScheduleUnavailableError(
                "DMA provider stream set changed while recording repeat effect"
            )
        provider_delta = tuple(
            (stream, after_provider[stream] - before_provider[stream])
            for stream in sorted(before_provider)
            if after_provider[stream] != before_provider[stream]
        )
        if any(delta < 0 for _stream, delta in provider_delta):
            raise ScheduleUnavailableError(
                "DMA provider moved backwards while recording repeat effect"
            )

        return _CachedRepeatEffect(
            cycle_delta=after.next_issue_cycle - before.next_issue_cycle,
            sequence_delta=after.sequence - before.sequence,
            gp_values=tuple(
                (index, after.gp[index]) for index in sorted(gp_indices)
            ),
            affine_positions=after.affine_positions,
            slot_names=after.slot_names,
            normalized_slots=tuple(
                cls._normalized_slot(slot, after) for slot in after.slots
            ),
            scalar_fp_indices=after.scalar_fp_indices,
            normalized_scalar_fp_results=tuple(
                cls._normalized_slot(slot, after)
                for slot in after.scalar_fp_results
            ),
            vector_element_latency=after.vector_element_latency,
            include_matrix_writes=after.include_matrix_writes,
            include_vector_writes=after.include_vector_writes,
            include_vector_port_a_writes=after.include_vector_port_a_writes,
            matrix_access_envelopes=after.matrix_access_envelopes,
            vector_access_envelopes=after.vector_access_envelopes,
            matrix_writes=pending(after.matrix_writes),
            vector_writes=pending(after.vector_writes),
            vector_port_a_writes=tuple(
                (
                    item.start - after.next_issue_cycle,
                    item.end - after.next_issue_cycle,
                    None
                    if item.sequence is None
                    else item.sequence - after.sequence,
                )
                for item in after.vector_port_a_writes
            ),
            normalized_makespan=max(
                0, after.local_makespan_cycles - after.next_issue_cycle
            ),
            counter_delta=tuple(counter_delta),
            hbm_provider_count_delta=provider_delta,
            dma_occurrences=dma_occurrences,
        )

    def _apply_cached_repeat_effect(
        self, effect: _CachedRepeatEffect, *, loop_iterations: int
    ) -> None:
        boundary = self.next_issue_cycle
        sequence = self.sequence
        final_boundary = boundary + effect.cycle_delta
        final_sequence = sequence + effect.sequence_delta
        self._append_relative_dma_occurrences(
            effect.dma_occurrences,
            boundary=boundary,
            sequence=sequence,
        )

        def slot(value: tuple[int, int | None]) -> Slot:
            relative_until, relative_sequence = value
            if relative_until <= 0:
                return Slot()
            return Slot(
                final_boundary + relative_until,
                None
                if relative_sequence is None
                else final_sequence + relative_sequence,
            )

        for name, value in zip(
            effect.slot_names, effect.normalized_slots, strict=True
        ):
            setattr(self, name, slot(value))
        for index, value in zip(
            effect.scalar_fp_indices,
            effect.normalized_scalar_fp_results,
            strict=True,
        ):
            self.scalar_fp_results[index] = slot(value)
        self.vector_element_latency = effect.vector_element_latency

        def replace_pending(
            items: list[PendingWrite],
            cached: tuple[tuple[int, int, int, int | None], ...],
            envelopes: tuple[AddressRange, ...] | None,
        ) -> list[PendingWrite]:
            unrelated = [
                item
                for item in items
                if item.slot.until > boundary
                and envelopes is not None
                and not any(
                    item.address_range.overlaps(envelope)
                    for envelope in envelopes
                )
            ]
            restored = [
                PendingWrite(AddressRange(start, end), slot((until, owner)))
                for start, end, until, owner in cached
            ]
            return unrelated + restored

        if effect.include_matrix_writes:
            self.matrix_writes = replace_pending(
                self.matrix_writes,
                effect.matrix_writes,
                effect.matrix_access_envelopes,
            )
        if effect.include_vector_writes:
            self.vector_writes = replace_pending(
                self.vector_writes,
                effect.vector_writes,
                effect.vector_access_envelopes,
            )
        if effect.include_vector_port_a_writes:
            self.vector_port_a_writes = [
                BusyInterval(
                    final_boundary + start,
                    final_boundary + end,
                    None if owner is None else final_sequence + owner,
                )
                for start, end, owner in effect.vector_port_a_writes
            ]

        for index, value in effect.gp_values:
            self.gp[index] = value
        for key, position in effect.affine_positions:
            self.affine_positions[key] = position
        counters = (
            self.stall_cycles,
            self.resource_work,
            self.critical_path,
            self.status_counts,
            self.unsupported_counts,
            self.out_of_domain_counts,
        )
        for counter, increments in zip(
            counters, effect.counter_delta, strict=True
        ):
            for name, increment in increments:
                counter[name] += increment

        if effect.hbm_provider_count_delta:
            advance_provider = getattr(
                self.hbm_service_cycles, "advance_stream_counts", None
            )
            if not callable(advance_provider):
                raise ScheduleUnavailableError(
                    "DMA timing provider cannot follow a cached repeat"
                )
            advance_provider(dict(effect.hbm_provider_count_delta))

        self.next_issue_cycle = final_boundary
        self.sequence = final_sequence
        self._register_active_sequence_resources()
        candidate_makespan = final_boundary + effect.normalized_makespan
        if candidate_makespan >= self.makespan_cycles:
            self.makespan_cycles = candidate_makespan
            self.makespan_resource = (
                self._completion_owner(candidate_makespan)
                or self.makespan_resource
            )
        self.repeat_fast_forwards += 1
        self.repeat_cache_hits += 1
        if self.repeat_stack:
            self.repeat_cache_hits_by_name[self.repeat_stack[-1]] += 1
        self.fast_forwarded_iterations += loop_iterations
        self.fast_forwarded_dynamic_instructions += effect.sequence_delta

    def visit(self, node: ScheduleNode) -> None:
        if not isinstance(node, ScheduleRepeat):
            self._visit_uncached(node)
            return
        provider_fast_forward = bool(
            getattr(self.hbm_service_cycles, "supports_exact_fast_forward", False)
        )
        if self.retain_events or (
            self.hbm_fidelity != "post_hoc_v3" and not provider_fast_forward
        ):
            self._visit_uncached(node)
            return

        repeat_affine_keys = _schedule_affine_keys(node.body)
        (
            repeat_slots,
            repeat_scalar_fp,
            include_matrix_writes,
            include_vector_writes,
            include_vector_port_a_writes,
        ) = _schedule_scoreboard_footprint(node.body)
        (
            matrix_access_envelopes,
            vector_access_envelopes,
        ) = self._repeat_access_envelopes(node.body, node.count)

        def snapshot() -> _SchedulerSnapshot:
            return self._snapshot(
                repeat_affine_keys,
                repeat_slots,
                repeat_scalar_fp,
                include_matrix_writes=include_matrix_writes,
                include_vector_writes=include_vector_writes,
                include_vector_port_a_writes=include_vector_port_a_writes,
                matrix_access_envelopes=matrix_access_envelopes,
                vector_access_envelopes=vector_access_envelopes,
            )

        gp_indices = _schedule_gp_registers(node.body)
        before = snapshot()
        # The first use of an affine key also registers its specification.
        # Record/reuse effects only after every key has been initialized so a
        # cache hit never skips that semantic side effect.
        affine_cache_ready = set(repeat_affine_keys) == {
            key for key, _period in before.affine_periods
        }
        key = self._repeat_cache_key(node, before, gp_indices)
        cached = self.repeat_effect_cache.get(key) if affine_cache_ready else None
        if cached is not None:
            self.repeat_stack.append(f"{node.name}[{node.count}]")
            try:
                self._apply_cached_repeat_effect(
                    cached, loop_iterations=node.count
                )
            finally:
                self.repeat_stack.pop()
            return

        self.repeat_cache_misses_by_name[f"{node.name}[{node.count}]"] += 1
        self._visit_uncached(node)
        after = snapshot()
        if affine_cache_ready:
            self.repeat_effect_cache[key] = self._capture_repeat_effect(
                before,
                after,
                gp_indices,
                self._relative_dma_occurrences(before, after),
            )

    def _visit_uncached(self, node: ScheduleNode) -> None:
        if isinstance(node, ScheduleInstruction):
            self.schedule_instruction(node)
            return
        if isinstance(node, ScheduleAffineLoad):
            self._visit_affine_load(node)
            return
        if isinstance(node, ScheduleAffineAdd):
            self._visit_affine_add(node)
            return
        if isinstance(node, ScheduleUnavailable):
            raise ScheduleUnavailableError(
                f"{node.reason} in stage {node.stage} "
                f"({node.dynamic_instruction_count} instructions)"
            )
        if isinstance(node, ScheduleSequence):
            for child in node.children:
                self.visit(child)
            return
        if isinstance(node, ScheduleRepeat):
            body_size = _schedule_instruction_count(node.body)
            remaining_budget = (
                self.max_expanded_instructions - self.expanded_instruction_count
            )
            literal_limit = (
                remaining_budget
                if self.retain_events
                or (
                    self.hbm_fidelity == "ramulator_observed"
                    and not bool(
                        getattr(
                            self.hbm_service_cycles,
                            "supports_exact_fast_forward",
                            False,
                        )
                    )
                )
                else 8
                if _schedule_contains_affine(node.body)
                else (
                    MAX_DIRECT_MAP_REPEAT_INSTRUCTIONS
                    if _schedule_contains_opcode(node.body, "S_MAP_V_FP")
                    else MAX_DIRECT_REPEAT_INSTRUCTIONS
                )
            )
            direct_budget = min(
                remaining_budget, literal_limit
            )
            if body_size * node.count <= direct_budget:
                for _ in range(node.count):
                    self._visit_repeat_iteration(node)
                return
            if self.retain_events:
                raise ScheduleUnavailableError(
                    f"repeat {node.name!r} requires {body_size * node.count} dynamic "
                    "instructions; event retention forbids repeat fast-forward"
                )

            if (
                self.hbm_fidelity != "post_hoc_v3"
                and _schedule_contains_memory(node.body)
                and not bool(
                    getattr(
                        self.hbm_service_cycles,
                        "supports_exact_fast_forward",
                        False,
                    )
                )
            ):
                raise ScheduleUnavailableError(
                    f"repeat {node.name!r} contains DMA with {self.hbm_fidelity!r} "
                    "service times; exact provider-state fast-forward is unavailable"
                )

            repeat_affine_keys = _schedule_affine_keys(node.body)
            (
                repeat_slots,
                repeat_scalar_fp,
                include_matrix_writes,
                include_vector_writes,
                include_vector_port_a_writes,
            ) = _schedule_scoreboard_footprint(node.body)
            (
                matrix_access_envelopes,
                vector_access_envelopes,
            ) = self._repeat_access_envelopes(node.body, node.count)
            snapshots = [
                self._snapshot(
                    repeat_affine_keys,
                    repeat_slots,
                    repeat_scalar_fp,
                    include_matrix_writes=include_matrix_writes,
                    include_vector_writes=include_vector_writes,
                    include_vector_port_a_writes=(
                        include_vector_port_a_writes
                    ),
                    matrix_access_envelopes=matrix_access_envelopes,
                    vector_access_envelopes=vector_access_envelopes,
                )
            ]
            # Immediate-legalization boundaries are capped analytically by
            # ``_safe_affine_applications``.  Two equal consecutive
            # transitions prove a normalized fixed point: each transition
            # includes its normalized final scoreboard state, pending-write
            # shape/address delta, GP delta, affine phase and counter delta.
            entry_snapshot = snapshots[0]
            pending_horizon_cycles = max(
                (
                    0,
                    *(
                        item.slot.until - entry_snapshot.next_issue_cycle
                        for item in entry_snapshot.matrix_writes
                    ),
                    *(
                        item.slot.until - entry_snapshot.next_issue_cycle
                        for item in entry_snapshot.vector_writes
                    ),
                    *(
                        item.end - entry_snapshot.next_issue_cycle
                        for item in entry_snapshot.vector_port_a_writes
                    ),
                )
            )
            pending_drain_iterations = (
                math.ceil(pending_horizon_cycles / max(1, body_size)) + 2
            )
            provider_fast_forward = bool(
                getattr(
                    self.hbm_service_cycles,
                    "supports_exact_fast_forward",
                    False,
                )
            )
            # A production-DMA stream can have a finite occurrence phase that
            # is longer than the scoreboard drain horizon.  For short loops,
            # consume the whole loop literally if no earlier normalized
            # transition appears.  This is exact (and bounded at 512
            # iterations), unlike replacing per-occurrence service times with
            # an average.  Stateless/V3 providers retain the smaller probe.
            provider_probe_iterations = (
                min(node.count, MAX_REPEAT_PROBE_ITERATIONS)
                if provider_fast_forward
                else 0
            )
            probe_limit = min(
                node.count,
                MAX_REPEAT_PROBE_ITERATIONS,
                max(
                    32,
                    pending_drain_iterations,
                    provider_probe_iterations,
                ),
            )
            for completed_iterations in range(1, probe_limit + 1):
                self._visit_repeat_iteration(node)
                snapshots.append(
                    self._snapshot(
                        repeat_affine_keys,
                        repeat_slots,
                        repeat_scalar_fp,
                        include_matrix_writes=include_matrix_writes,
                        include_vector_writes=include_vector_writes,
                        include_vector_port_a_writes=(
                            include_vector_port_a_writes
                        ),
                        matrix_access_envelopes=matrix_access_envelopes,
                        vector_access_envelopes=vector_access_envelopes,
                    )
                )
                # A pipelined scoreboard can settle into a short periodic
                # orbit rather than a one-iteration fixed point. Compare
                # two consecutive aggregate transitions for periods up to
                # eight loop iterations. Equality covers every normalized
                # resource slot, pending address delta, GP delta and counter
                # delta, so applying the period is exact rather than an
                # average-throughput approximation.
                max_period = min(
                    MAX_REPEAT_PERIOD_ITERATIONS,
                    completed_iterations // 2,
                )
                for period in range(1, max_period + 1):
                    boundaries = (
                        completed_iterations - 2 * period,
                        completed_iterations - period,
                        completed_iterations,
                    )
                    transitions = tuple(
                        self._transition(
                            snapshots[left],
                            snapshots[right],
                            self._relative_dma_occurrences(
                                snapshots[left], snapshots[right]
                            ),
                        )
                        for left, right in zip(
                            boundaries[:-1], boundaries[1:], strict=True
                        )
                    )
                    if transitions[0] is None or len(set(transitions)) != 1:
                        continue
                    transition = transitions[0]
                    assert transition is not None
                    remaining = node.count - completed_iterations
                    applications, tail = divmod(remaining, period)
                    safe_applications = self._safe_affine_applications(
                        transition, applications
                    )
                    if safe_applications < applications:
                        complete_period = self._complete_affine_period(
                            transition,
                            period,
                        )
                        additional_probe_iterations = (
                            0
                            if complete_period is None
                            else max(
                                0,
                                2 * complete_period - completed_iterations,
                            )
                        )
                        remaining_expansion_budget = (
                            self.max_expanded_instructions
                            - self.expanded_instruction_count
                        )
                        if (
                            complete_period is not None
                            and 2 * complete_period <= probe_limit
                            and additional_probe_iterations * body_size
                            <= remaining_expansion_budget
                        ):
                            continue
                    self._apply_steady_transition(
                        transition,
                        safe_applications,
                        loop_iterations_per_transition=period,
                    )
                    if safe_applications < applications:
                        remaining -= safe_applications * period
                        # Execute one real iteration to cross the affine
                        # legalization boundary, then discover the next exact
                        # normalized transition from the new regime.
                        self._visit_repeat_iteration(node)
                        remaining -= 1
                        while remaining:
                            chunk = min(
                                remaining,
                                MAX_REPEAT_TAIL_CHUNK_ITERATIONS,
                            )
                            self.visit(
                                ScheduleRepeat(
                                    chunk,
                                    node.body,
                                    name=node.name,
                                    repeat_kind=node.repeat_kind,
                                )
                            )
                            remaining -= chunk
                        return
                    for _ in range(tail):
                        self._visit_repeat_iteration(node)
                    return
            if probe_limit == node.count:
                return
            initial_sequence = snapshots[0].sequence
            final_snapshot = snapshots[-1]
            active_matrix_external = sum(
                item.slot.sequence is None
                or item.slot.sequence < initial_sequence
                for item in final_snapshot.matrix_writes
            )
            active_vector_external = sum(
                item.slot.sequence is None
                or item.slot.sequence < initial_sequence
                for item in final_snapshot.vector_writes
            )
            active_port_external = sum(
                item.sequence is None or item.sequence < initial_sequence
                for item in final_snapshot.vector_port_a_writes
            )
            vector_external_samples = [
                (
                    item.address_range.start,
                    item.address_range.end,
                    item.slot.until - final_snapshot.next_issue_cycle,
                    item.slot.sequence,
                )
                for item in final_snapshot.vector_writes
                if (
                    item.slot.sequence is None
                    or item.slot.sequence < initial_sequence
                )
            ][:3]
            raise ScheduleUnavailableError(
                f"repeat {node.name!r} did not reach an exact normalized "
                f"scoreboard fixed point within {probe_limit} iterations; "
                "active external scoreboard entries: "
                f"matrix_writes={active_matrix_external}, "
                f"vector_writes={active_vector_external}, "
                f"vector_port_a={active_port_external}; "
                f"gp={tuple(self.gp)}; "
                f"vector_external_samples={vector_external_samples}"
            )
        raise TypeError(type(node).__name__)

    def result(self) -> ScheduledShadowResult:
        if self.unsupported_counts:
            validation_status = "unsupported_opcodes"
        elif self.out_of_domain_counts:
            validation_status = "out_of_domain"
        else:
            validation_status = "validated"
        critical_path = self.critical_path.copy()
        drain_cycles = max(0, self.makespan_cycles - self.next_issue_cycle)
        if drain_cycles:
            critical_path[self.makespan_resource or "control_frontend"] += (
                drain_cycles
            )
        return ScheduledShadowResult(
            status="complete",
            fidelity=self.hbm_fidelity,
            makespan_cycles=self.makespan_cycles,
            events=tuple(self.events),
            stall_cycles_by_reason=dict(sorted(self.stall_cycles.items())),
            resource_work_cycles=dict(sorted(self.resource_work.items())),
            critical_path_cycles=dict(sorted(critical_path.items())),
            dma_occurrences=tuple(self.dma_occurrences),
            validation={
                "status": validation_status,
                "status_counts": dict(sorted(self.status_counts.items())),
                "unsupported_opcode_counts": dict(sorted(self.unsupported_counts.items())),
                "out_of_domain_opcode_counts": dict(
                    sorted(self.out_of_domain_counts.items())
                ),
                "expanded_instruction_count": self.expanded_instruction_count,
                "repeat_fast_forwards": self.repeat_fast_forwards,
                "repeat_cache_hits": self.repeat_cache_hits,
                "repeat_cache_hits_by_name": dict(
                    self.repeat_cache_hits_by_name.most_common()
                ),
                "repeat_cache_misses_by_name": dict(
                    self.repeat_cache_misses_by_name.most_common()
                ),
                "expanded_instructions_by_repeat": dict(
                    self.expanded_by_repeat.most_common()
                ),
                "fast_forwarded_iterations": self.fast_forwarded_iterations,
                "fast_forwarded_dynamic_instructions": (
                    self.fast_forwarded_dynamic_instructions
                ),
                "critical_path_accounting": (
                    "exact"
                    if self.unresolved_dependency_resources == 0
                    else "unresolved_dependency_owner"
                ),
                "unresolved_dependency_resources": (
                    self.unresolved_dependency_resources
                ),
            },
        )


@cache
def _schedule_instruction_count(node: ScheduleNode) -> int:
    if isinstance(node, ScheduleInstruction):
        return 1
    if isinstance(node, ScheduleAffineLoad):
        # A legalized absolute load contains at most LUI + low ADDI.
        return 2
    if isinstance(node, ScheduleAffineAdd):
        # Large relative adds materialize the immediate then issue S_ADD_INT.
        return 3
    if isinstance(node, ScheduleUnavailable):
        return node.dynamic_instruction_count
    if isinstance(node, ScheduleSequence):
        return sum(_schedule_instruction_count(child) for child in node.children)
    if isinstance(node, ScheduleRepeat):
        return node.count * _schedule_instruction_count(node.body)
    raise TypeError(type(node).__name__)


@cache
def _schedule_contains_memory(node: ScheduleNode) -> bool:
    if isinstance(node, ScheduleInstruction):
        return node.opcode in MEMORY_OPS
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return False
    if isinstance(node, ScheduleUnavailable):
        return False
    if isinstance(node, ScheduleSequence):
        return any(_schedule_contains_memory(child) for child in node.children)
    if isinstance(node, ScheduleRepeat):
        return _schedule_contains_memory(node.body)
    raise TypeError(type(node).__name__)


@cache
def _schedule_contains_opcode(node: ScheduleNode, opcode: str) -> bool:
    if isinstance(node, ScheduleInstruction):
        return node.opcode == opcode
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable)):
        return False
    if isinstance(node, ScheduleSequence):
        return any(_schedule_contains_opcode(child, opcode) for child in node.children)
    if isinstance(node, ScheduleRepeat):
        return _schedule_contains_opcode(node.body, opcode)
    raise TypeError(type(node).__name__)


@cache
def _schedule_gp_registers(node: ScheduleNode) -> frozenset[int]:
    registers: set[int] = set()

    def visit(current: ScheduleNode) -> None:
        if isinstance(current, ScheduleInstruction):
            for arg in current.args:
                if arg.startswith("gp") and arg[2:].isdigit():
                    registers.add(int(arg[2:]))
            return
        if isinstance(current, ScheduleAffineLoad):
            registers.add(_register_index(current.register, "gp"))
            return
        if isinstance(current, ScheduleAffineAdd):
            registers.update(
                (
                    _register_index(current.destination, "gp"),
                    _register_index(current.source, "gp"),
                    _register_index(current.temp, "gp"),
                )
            )
            return
        if isinstance(current, ScheduleUnavailable):
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body)
            return
        raise TypeError(type(current).__name__)

    visit(node)
    return frozenset(registers)


@cache
def _schedule_contains_affine(node: ScheduleNode) -> bool:
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return True
    if isinstance(node, (ScheduleInstruction, ScheduleUnavailable)):
        return False
    if isinstance(node, ScheduleSequence):
        return any(_schedule_contains_affine(child) for child in node.children)
    if isinstance(node, ScheduleRepeat):
        return _schedule_contains_affine(node.body)
    raise TypeError(type(node).__name__)


def _schedule_unavailable_counts(node: ScheduleNode) -> Counter[str]:
    if isinstance(
        node, (ScheduleInstruction, ScheduleAffineLoad, ScheduleAffineAdd)
    ):
        return Counter()
    if isinstance(node, ScheduleUnavailable):
        return Counter({node.reason: node.dynamic_instruction_count})
    if isinstance(node, ScheduleSequence):
        result: Counter[str] = Counter()
        for child in node.children:
            result.update(_schedule_unavailable_counts(child))
        return result
    if isinstance(node, ScheduleRepeat):
        body = _schedule_unavailable_counts(node.body)
        return Counter({reason: count * node.count for reason, count in body.items()})
    raise TypeError(type(node).__name__)


@cache
def _schedule_affine_keys(node: ScheduleNode) -> frozenset[str]:
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return frozenset((node.key,))
    if isinstance(node, (ScheduleInstruction, ScheduleUnavailable)):
        return frozenset()
    if isinstance(node, ScheduleSequence):
        keys: set[str] = set()
        for child in node.children:
            keys.update(_schedule_affine_keys(child))
        return frozenset(keys)
    if isinstance(node, ScheduleRepeat):
        return _schedule_affine_keys(node.body)
    raise TypeError(type(node).__name__)


@cache
def _schedule_scoreboard_footprint(
    node: ScheduleNode,
) -> tuple[frozenset[str], frozenset[int], bool, bool, bool]:
    """Return the exact scoreboard domains consulted or changed by ``node``.

    Pending Matrix SRAM writes are irrelevant to a Vector-only loop, and vice
    versa.  Keeping those unrelated, long-lived DMA completions in a repeat
    snapshot prevents an otherwise stable local transition from normalizing.
    The booleans therefore describe the pending-write/port domains separately
    from the named resource slots.
    """
    slots: set[str] = set()
    scalar_fp: set[int] = set()
    include_matrix_writes = False
    include_vector_writes = False
    include_vector_port_a_writes = False

    def fp_register(arg: str) -> int | None:
        if not arg.startswith("f"):
            return None
        index = int(arg[1:])
        return index if index else None

    def instruction(current: ScheduleInstruction) -> None:
        nonlocal include_matrix_writes
        nonlocal include_vector_writes
        nonlocal include_vector_port_a_writes
        opcode = current.opcode
        args = current.args
        resource = resource_for(opcode)
        if resource == "hbm_matrix_dma":
            slots.update(("hbm_shared", "hbm_matrix"))
            include_matrix_writes = True
        elif resource == "hbm_vector_dma":
            slots.update(
                (
                    "hbm_shared",
                    "hbm_vector",
                    "matrix_compute",
                    "vector_pipeline",
                )
            )
            include_vector_writes = True
        elif resource == "hbm_vector_store":
            slots.update(
                (
                    "hbm_shared",
                    "hbm_store",
                    "matrix_compute",
                    "vector_pipeline",
                )
            )
            include_vector_writes = True
        elif resource == "matrix_compute":
            slots.update(
                (
                    "matrix_compute",
                    "matrix_writeout",
                    "hbm_vector",
                    "hbm_store",
                )
            )
            include_matrix_writes = True
            include_vector_writes = True
            include_vector_port_a_writes = True
        elif resource == "matrix_writeout":
            slots.update(("matrix_compute", "matrix_writeout"))
            include_vector_writes = True
            include_vector_port_a_writes = True
        elif resource == "vector_pipeline":
            slots.update(
                (
                    "vector_pipeline",
                    "hbm_vector",
                    "hbm_store",
                    "vector_element_result",
                )
            )
            include_vector_writes = True
            include_vector_port_a_writes = True
            if opcode in VECTOR_REDUCTION:
                slots.add("vector_reduction_result")
                register = fp_register(args[0]) if args else None
                if register is not None:
                    scalar_fp.add(register)
            if opcode in {"V_ADD_VF", "V_SUB_VF", "V_MUL_VF"}:
                slots.add("scalar_fp_compute")
                register = fp_register(args[2]) if len(args) > 2 else None
                if register is not None:
                    scalar_fp.add(register)
        elif resource == "scalar_pipeline":
            if opcode in SCALAR_FP_COMPUTE | SCALAR_SRAM:
                slots.add("vector_reduction_result")
            if opcode in SCALAR_FP_COMPUTE:
                slots.add("scalar_fp_compute")
            elif opcode in SCALAR_SRAM:
                slots.add("scalar_sram")
            if opcode in {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP"}:
                for arg in args[:3]:
                    register = fp_register(arg)
                    if register is not None:
                        scalar_fp.add(register)
            elif opcode in {"S_EXP_FP", "S_RECI_FP", "S_SQRT_FP"}:
                for arg in args[:2]:
                    register = fp_register(arg)
                    if register is not None:
                        scalar_fp.add(register)
            elif opcode in {"S_LD_FP", "S_ST_FP"} and args:
                register = fp_register(args[0])
                if register is not None:
                    scalar_fp.add(register)
            if opcode == "S_MAP_V_FP":
                include_vector_writes = True
        elif resource == "control_frontend" and opcode == "C_BREAK":
            slots.add("hbm_vector")

    def visit(current: ScheduleNode) -> None:
        if isinstance(current, ScheduleInstruction):
            instruction(current)
            return
        if isinstance(current, (ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable)):
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body)
            return
        raise TypeError(type(current).__name__)

    visit(node)
    return (
        frozenset(slots),
        frozenset(scalar_fp),
        include_matrix_writes,
        include_vector_writes,
        include_vector_port_a_writes,
    )


def _clear_schedule_analysis_caches() -> None:
    """Release immutable IR nodes retained by module-level memoization.

    Optuna workers evaluate many independently allocated traces. Keeping a
    global cache across trials would retain every prior schedule tree, so the
    cache lifetime is explicitly one shadow evaluation.
    """

    for function in (
        _schedule_instruction_count,
        _schedule_contains_memory,
        _schedule_contains_opcode,
        _schedule_gp_registers,
        _schedule_contains_affine,
        _schedule_affine_keys,
        _schedule_scoreboard_footprint,
    ):
        function.cache_clear()


def evaluate_scheduled_shadow(
    trace: CostTrace,
    *,
    hardware: TimingHardware,
    precision: ComputePrecisionConfig,
    calibration: RtlOpcodeTimingCalibration,
    hbm_service_cycles: Callable[[ScheduleInstruction, int], int] | None = None,
    hbm_fidelity: str = "post_hoc_v3",
    retain_events: bool = False,
    max_expanded_instructions: int = 2_000_000,
    initial_gp: Mapping[int, int] | None = None,
) -> ScheduledShadowResult:
    _clear_schedule_analysis_caches()
    if trace.schedule_unavailable_reasons:
        reasons = ", ".join(
            f"{name}={count}"
            for name, count in sorted(trace.schedule_unavailable_reasons.items())
        )
        unavailable = _schedule_unavailable_counts(trace.schedule)
        unavailable_count = sum(unavailable.values())
        total = trace.dynamic_instruction_count
        result = ScheduledShadowResult(
            status="schedule_unavailable",
            fidelity=hbm_fidelity,
            makespan_cycles=None,
            events=(),
            stall_cycles_by_reason={},
            resource_work_cycles={},
            critical_path_cycles={},
            validation={
                "status": "schedule_unavailable",
                "ordered_dynamic_instructions": max(0, total - unavailable_count),
                "unavailable_dynamic_instructions": unavailable_count,
                "ordered_fraction": (
                    1.0 if total == 0 else max(0, total - unavailable_count) / total
                ),
                "unavailable_by_reason": dict(sorted(unavailable.items())),
            },
            reason=reasons,
        )
        _clear_schedule_analysis_caches()
        return result
    scheduler = RtlShadowScheduler(
        hardware=hardware,
        precision=precision,
        calibration=calibration,
        memory_events={event.stream_index: event for event in trace.memory_events},
        hbm_service_cycles=hbm_service_cycles,
        hbm_fidelity=hbm_fidelity,
        retain_events=retain_events,
        max_expanded_instructions=max_expanded_instructions,
        initial_gp=initial_gp,
    )
    try:
        scheduler.visit(trace.schedule)
    except ScheduleUnavailableError as exc:
        result = ScheduledShadowResult(
            status="schedule_unavailable",
            fidelity=hbm_fidelity,
            makespan_cycles=None,
            events=tuple(scheduler.events),
            stall_cycles_by_reason=dict(sorted(scheduler.stall_cycles.items())),
            resource_work_cycles=dict(sorted(scheduler.resource_work.items())),
            critical_path_cycles=dict(sorted(scheduler.critical_path.items())),
            validation={
                "status": "schedule_unavailable",
                "expanded_instruction_count": scheduler.expanded_instruction_count,
                "repeat_fast_forwards": scheduler.repeat_fast_forwards,
                "repeat_cache_hits": scheduler.repeat_cache_hits,
                "fast_forwarded_iterations": scheduler.fast_forwarded_iterations,
                "fast_forwarded_dynamic_instructions": (
                    scheduler.fast_forwarded_dynamic_instructions
                ),
            },
            reason=str(exc),
        )
        _clear_schedule_analysis_caches()
        return result
    except BaseException:
        _clear_schedule_analysis_caches()
        raise
    result = scheduler.result()
    _clear_schedule_analysis_caches()
    return result


__all__ = [
    "ScheduledDmaOccurrence",
    "ScheduledEvent",
    "ScheduledShadowResult",
    "ScheduleUnavailableError",
    "evaluate_scheduled_shadow",
    "resource_for",
]
