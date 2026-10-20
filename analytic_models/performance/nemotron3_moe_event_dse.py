"""Exact-routing, event-level Matrix-cluster DSE for Nemotron 3 MoE decode.

This model joins two pieces of evidence that were previously separate:

* the exact 127-step, 23-layer top-6 routing trace and routed-weight LRU; and
* the Shared-MoE event model's fixed-PE topology comparison, finite weight
  buffers, one shared HBM server, asynchronous completion, and one shared
  reduction resource.

It does not claim calibrated Nemotron PLENA latency.  The report keeps an ideal
geometry model and a transferred Shared-MoE calibration separate so topology
rankings that depend on the transferred constants are visible rather than
silently presented as RTL truth.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import (
    ModelArchConfig,
    load_model_config,
)

from .nemotron3_routing_dse import PINNED_TRACE, load_routing_trace
from .nemotron3_workload import Precision, storage_bytes


BASELINE_ROWS = 4
BASELINE_COLS = 1024
BASELINE_PES = BASELINE_ROWS * BASELINE_COLS
HBM_BURST_BYTES = 64
VECTOR_WIDTH = 64
PARTITION_LAUNCH_CYCLES = 128
MULTICORE_COMPLETION_CYCLES = 32
MIB = 1024 * 1024
GIB = 1024**3

# Existing Shared-MoE event model calibration.  It is useful as a sensitivity
# point, but it is not a direct Nemotron measurement.
TRANSFERRED_HBM_BYTES_PER_CYCLE = 39.717745090778166
QWEN_ROUTED_MATRIX_WAVE_CYCLES = 2_171_392
QWEN_HIDDEN = 2048
QWEN_INTERMEDIATE = 1408


class Mapping(StrEnum):
    EXPERT = "expert"
    M_SPLIT = "m_split"
    K_SPLIT = "k_split"
    N_TO_K = "n_to_k"
    DYNAMIC = "dynamic"
    M_BY_N = "m_by_n"
    M_BY_K = "m_by_k"


@dataclass(frozen=True)
class CoreGeometry:
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("matrix core dimensions must be positive")

    @property
    def pes(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class Candidate:
    name: str
    topology: str
    cores: tuple[CoreGeometry, ...]
    mapping: Mapping
    scheduler: str = "largest_first"
    hbm_policy: str = "critical_first"
    m_parts: int = 1
    secondary_parts: int = 1

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("candidate requires at least one core")
        if self.scheduler not in {"fcfs", "largest_first"}:
            raise ValueError("unsupported scheduler")
        if self.hbm_policy not in {"sequential", "round_robin", "critical_first"}:
            raise ValueError("unsupported HBM policy")
        if self.m_parts <= 0 or self.secondary_parts <= 0:
            raise ValueError("partition counts must be positive")

    @property
    def pe_count(self) -> int:
        return sum(core.pes for core in self.cores)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pe_count"] = self.pe_count
        result["core_count"] = len(self.cores)
        return result


@dataclass(frozen=True)
class CycleCalibration:
    name: str
    matrix_cycle_scale: float
    hbm_bytes_per_cycle: float
    source: str
    calibrated_for_nemotron: bool = False

    def __post_init__(self) -> None:
        if self.matrix_cycle_scale <= 0 or self.hbm_bytes_per_cycle <= 0:
            raise ValueError("cycle calibration values must be positive")


@dataclass(frozen=True)
class MoeShape:
    hidden: int
    routed_intermediate: int
    shared_intermediate: int
    routed_weight_bytes: int
    shared_weight_bytes: int

    @classmethod
    def from_arch(cls, arch: ModelArchConfig) -> MoeShape:
        if arch.moe is None:
            raise ValueError("Nemotron MoE DSE requires a MoE architecture")
        moe = arch.moe
        return cls(
            hidden=arch.hidden_size,
            routed_intermediate=moe.intermediate_size,
            shared_intermediate=moe.shared_intermediate_size,
            routed_weight_bytes=storage_bytes(
                2 * arch.hidden_size * moe.intermediate_size,
                Precision.NVFP4,
            ),
            shared_weight_bytes=storage_bytes(
                2 * arch.hidden_size * moe.shared_intermediate_size,
                Precision.NVFP4,
            ),
        )


@dataclass(frozen=True)
class RoutingPatterns:
    capacity_entries: int
    prefill_active_entries: int
    prefill_resident_entries: int
    accesses: int
    hits: int
    misses: int
    event_patterns: tuple[tuple[tuple[bool, ...], int], ...]

    @property
    def events(self) -> int:
        return sum(count for _, count in self.event_patterns)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["events"] = self.events
        result["hit_rate"] = self.hit_rate
        result["distinct_patterns"] = len(self.event_patterns)
        result["event_patterns"] = [
            {"hit_mask": [int(hit) for hit in pattern], "count": count} for pattern, count in self.event_patterns
        ]
        return result


@dataclass(frozen=True)
class ExpertJob:
    job_id: int
    kind: str
    expert_id: int | None
    tokens: int
    intermediate: int
    weight_bytes: int
    weight_cached: bool


@dataclass(frozen=True)
class PartPlan:
    part_id: int
    transfer_id: int | None
    tokens: int
    up_k: int
    up_n: int
    down_k: int
    down_n: int
    preferred_core_id: int | None


@dataclass(frozen=True)
class TransferPlan:
    transfer_id: int
    bytes: int
    unlock_parts: tuple[int, ...]


@dataclass(frozen=True)
class PlannedJob:
    job: ExpertJob
    mode: Mapping
    parts: tuple[PartPlan, ...]
    transfers: tuple[TransferPlan, ...]
    postprocess_cycles: int


@dataclass
class ActiveTransfer:
    sequence: int
    job_id: int
    transfer_id: int
    remaining_bytes: float
    priority: tuple[float, ...]


@dataclass(frozen=True)
class ReadyPart:
    sequence: int
    job_id: int
    part: PartPlan


@dataclass
class JobRuntime:
    plan: PlannedJob
    completed_parts: int = 0
    completion_time: float | None = None


@dataclass(frozen=True)
class EventResult:
    event_cycles: float
    hbm_bytes: int
    hbm_busy_cycles: float
    matrix_core_busy_cycles: float
    matrix_pe_busy_cycles: float
    matrix_pe_utilization: float
    hbm_starved_pe_cycles: float
    postprocess_cycles: float
    average_buffer_occupancy_bytes: float
    max_buffer_occupancy_bytes: int
    completion_skew_cycles: float
    completed_jobs: int

    def scaled_dict(self, count: int) -> dict[str, float | int]:
        return {
            "events": count,
            "cycles": self.event_cycles * count,
            "hbm_bytes": self.hbm_bytes * count,
            "hbm_busy_cycles": self.hbm_busy_cycles * count,
            "matrix_core_busy_cycles": self.matrix_core_busy_cycles * count,
            "matrix_pe_busy_cycles": self.matrix_pe_busy_cycles * count,
            "hbm_starved_pe_cycles": self.hbm_starved_pe_cycles * count,
            "postprocess_cycles": self.postprocess_cycles * count,
            "buffer_occupancy_byte_cycles": self.average_buffer_occupancy_bytes * self.event_cycles * count,
            "max_buffer_occupancy_bytes": self.max_buffer_occupancy_bytes,
            "completion_skew_cycles": self.completion_skew_cycles * count,
        }


def round_up(value: int, multiple: int) -> int:
    if value < 0 or multiple <= 0:
        raise ValueError("round_up requires a non-negative value and positive multiple")
    return ((value + multiple - 1) // multiple) * multiple


def split_integer(total: int, parts: int) -> list[int]:
    if total < 0 or parts <= 0:
        raise ValueError("split_integer requires non-negative work and positive parts")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def split_integer_weighted(total: int, weights: Sequence[float]) -> list[int]:
    if total < 0 or not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weighted split requires non-negative work and positive weights")
    exact = [total * weight / sum(weights) for weight in weights]
    result = [math.floor(value) for value in exact]
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - result[index], weights[index], -index),
        reverse=True,
    )
    for index in order[: total - sum(result)]:
        result[index] += 1
    return result


def candidate_space() -> tuple[Candidate, ...]:
    row_2_2 = (CoreGeometry(2, 1024), CoreGeometry(2, 1024))
    row_2_1_1 = (CoreGeometry(2, 1024), CoreGeometry(1, 1024), CoreGeometry(1, 1024))
    row_1_1_1_1 = tuple(CoreGeometry(1, 1024) for _ in range(4))
    grid_2x2 = tuple(CoreGeometry(2, 512) for _ in range(4))
    candidates = (
        Candidate(
            "baseline_4x1024__expert",
            "monolithic_4x1024",
            (CoreGeometry(4, 1024),),
            Mapping.EXPERT,
            scheduler="fcfs",
            hbm_policy="sequential",
        ),
        Candidate("row_2_2__expert", "row_2_2", row_2_2, Mapping.EXPERT),
        Candidate("row_2_2__k_split", "row_2_2", row_2_2, Mapping.K_SPLIT),
        Candidate("row_2_2__n_to_k", "row_2_2", row_2_2, Mapping.N_TO_K),
        Candidate("row_2_1_1__expert", "row_2_1_1", row_2_1_1, Mapping.EXPERT),
        Candidate("row_1_1_1_1__expert", "row_1_1_1_1", row_1_1_1_1, Mapping.EXPERT),
        Candidate("row_1_1_1_1__m_split", "row_1_1_1_1", row_1_1_1_1, Mapping.M_SPLIT),
        Candidate("row_1_1_1_1__k_split", "row_1_1_1_1", row_1_1_1_1, Mapping.K_SPLIT),
        Candidate("row_1_1_1_1__n_to_k", "row_1_1_1_1", row_1_1_1_1, Mapping.N_TO_K),
        Candidate("row_1_1_1_1__dynamic", "row_1_1_1_1", row_1_1_1_1, Mapping.DYNAMIC),
        Candidate(
            "grid_2x2__m_by_n",
            "grid_2x2",
            grid_2x2,
            Mapping.M_BY_N,
            m_parts=2,
            secondary_parts=2,
        ),
        Candidate(
            "grid_2x2__m_by_k",
            "grid_2x2",
            grid_2x2,
            Mapping.M_BY_K,
            m_parts=2,
            secondary_parts=2,
        ),
    )
    for candidate in candidates:
        if candidate.pe_count != BASELINE_PES:
            raise AssertionError(f"{candidate.name}: expected {BASELINE_PES} PEs, got {candidate.pe_count}")
    return candidates


def geometry_gemm_cycles(tokens: int, k: int, n: int, core: CoreGeometry) -> int:
    """Array occupancy proxy: K cycles for each active row/column tile."""

    if tokens <= 0 or k <= 0 or n <= 0:
        raise ValueError("GEMM dimensions must be positive")
    return math.ceil(tokens / core.rows) * math.ceil(n / core.cols) * k


def transferred_matrix_scale(shape: MoeShape) -> float:
    qwen_macs = 3 * QWEN_HIDDEN * QWEN_INTERMEDIATE
    nemotron_macs = 2 * shape.hidden * shape.routed_intermediate
    transferred_wave = QWEN_ROUTED_MATRIX_WAVE_CYCLES * nemotron_macs / qwen_macs
    geometry = geometry_gemm_cycles(
        1,
        shape.hidden,
        shape.routed_intermediate,
        CoreGeometry(BASELINE_ROWS, BASELINE_COLS),
    ) + geometry_gemm_cycles(
        1,
        shape.routed_intermediate,
        shape.hidden,
        CoreGeometry(BASELINE_ROWS, BASELINE_COLS),
    )
    return transferred_wave / geometry


def cycle_calibrations(shape: MoeShape) -> tuple[CycleCalibration, ...]:
    return (
        CycleCalibration(
            "ideal_geometry_hbm64",
            matrix_cycle_scale=1.0,
            hbm_bytes_per_cycle=64.0,
            source="ideal array-occupancy proxy; no PLENA timing calibration",
        ),
        CycleCalibration(
            "transferred_shared_moe",
            matrix_cycle_scale=transferred_matrix_scale(shape),
            hbm_bytes_per_cycle=TRANSFERRED_HBM_BYTES_PER_CYCLE,
            source=("Qwen/DeepSeek Shared-MoE event calibration transferred by expert MAC ratio; sensitivity only"),
        ),
    )


def routing_pattern_histogram(
    trace: dict[str, Any],
    *,
    capacity_entries: int,
    expert_order: str = "expert_id",
) -> RoutingPatterns:
    if capacity_entries < 0:
        raise ValueError("capacity_entries must be non-negative")
    if expert_order not in {"expert_id", "topk_rank"}:
        raise ValueError("expert_order must be expert_id or topk_rank")
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()

    def access(key: tuple[int, int]) -> bool:
        if capacity_entries == 0:
            return False
        if key in cache:
            cache.move_to_end(key)
            return True
        if len(cache) == capacity_entries:
            cache.popitem(last=False)
        cache[key] = None
        return False

    active_keys = [
        (layer, expert) for layer, experts in enumerate(trace["prefill_active_experts_by_layer"]) for expert in experts
    ]
    for key in active_keys:
        access(key)
    prefill_resident_entries = len(cache)

    patterns: Counter[tuple[bool, ...]] = Counter()
    hits = misses = accesses = 0
    for step in trace["decode_topk_by_step"]:
        for layer, experts in enumerate(step):
            ordered = experts if expert_order == "topk_rank" else sorted(experts)
            mask = tuple(access((layer, expert)) for expert in ordered)
            patterns[mask] += 1
            event_hits = sum(mask)
            hits += event_hits
            misses += len(mask) - event_hits
            accesses += len(mask)
    return RoutingPatterns(
        capacity_entries=capacity_entries,
        prefill_active_entries=len(active_keys),
        prefill_resident_entries=prefill_resident_entries,
        accesses=accesses,
        hits=hits,
        misses=misses,
        event_patterns=tuple(sorted(patterns.items())),
    )


def _split_bursts(total_bytes: int, weights: Sequence[float]) -> list[int]:
    total_bursts = round_up(total_bytes, HBM_BURST_BYTES) // HBM_BURST_BYTES
    return [bursts * HBM_BURST_BYTES for bursts in split_integer_weighted(total_bursts, weights)]


def _reduction_cycles(elements: int, parts: int) -> int:
    if parts <= 1:
        return 0
    levels = math.ceil(math.log2(parts))
    return math.ceil(elements / VECTOR_WIDTH) * levels + 32 * parts


def _postprocess_cycles(job: ExpertJob, mode: Mapping) -> int:
    activation = math.ceil(job.tokens * job.intermediate / VECTOR_WIDTH)
    if mode not in {Mapping.EXPERT, Mapping.M_SPLIT}:
        raise ValueError(f"postprocess helper does not support {mode}")
    return activation


def _effective_mode(job: ExpertJob, candidate: Candidate) -> Mapping:
    if candidate.mapping == Mapping.DYNAMIC:
        # Online, non-oracle heuristic: preserve independent routed-expert
        # parallelism and cooperatively execute the 2x-wide shared expert.
        return Mapping.N_TO_K if job.kind == "shared" else Mapping.EXPERT
    return candidate.mapping


def plan_job(job: ExpertJob, candidate: Candidate, shape: MoeShape) -> PlannedJob:
    mode = _effective_mode(job, candidate)
    core_ids = list(range(len(candidate.cores)))
    if mode in {Mapping.M_BY_N, Mapping.M_BY_K}:
        # B1 has one non-empty M partition.  The other grid row is idle.
        core_ids = core_ids[: candidate.secondary_parts]
    parts = len(core_ids)
    transfers: tuple[TransferPlan, ...]

    def transfer_id(index: int = 0) -> int | None:
        return None if job.weight_cached else index

    if mode == Mapping.EXPERT:
        part_plans = (
            PartPlan(
                0,
                transfer_id(),
                job.tokens,
                shape.hidden,
                job.intermediate,
                job.intermediate,
                shape.hidden,
                None,
            ),
        )
        transfers = () if job.weight_cached else (TransferPlan(0, round_up(job.weight_bytes, HBM_BURST_BYTES), (0,)),)
        postprocess = _postprocess_cycles(job, mode)
    elif mode == Mapping.M_SPLIT:
        token_counts = split_integer(job.tokens, len(core_ids))
        assignments = [(core_id, tokens) for core_id, tokens in zip(core_ids, token_counts, strict=True) if tokens]
        part_plans = tuple(
            PartPlan(
                index,
                transfer_id(),
                tokens,
                shape.hidden,
                job.intermediate,
                job.intermediate,
                shape.hidden,
                core_id,
            )
            for index, (core_id, tokens) in enumerate(assignments)
        )
        transfers = (
            ()
            if job.weight_cached
            else (
                TransferPlan(
                    0,
                    round_up(job.weight_bytes, HBM_BURST_BYTES),
                    tuple(range(len(part_plans))),
                ),
            )
        )
        postprocess = _postprocess_cycles(job, mode)
    elif mode in {Mapping.K_SPLIT, Mapping.M_BY_K}:
        capacities = [candidate.cores[core_id].cols for core_id in core_ids]
        up_ks = split_integer_weighted(shape.hidden, capacities)
        down_ks = split_integer_weighted(job.intermediate, capacities)
        transfer_weights = [
            up_k * job.intermediate + down_k * shape.hidden for up_k, down_k in zip(up_ks, down_ks, strict=True)
        ]
        transfer_bytes = _split_bursts(job.weight_bytes, transfer_weights)
        part_plans = tuple(
            PartPlan(
                index,
                transfer_id(index),
                job.tokens,
                up_ks[index],
                job.intermediate,
                down_ks[index],
                shape.hidden,
                core_id,
            )
            for index, core_id in enumerate(core_ids)
        )
        transfers = (
            ()
            if job.weight_cached
            else tuple(TransferPlan(index, size, (index,)) for index, size in enumerate(transfer_bytes))
        )
        activation = math.ceil(job.tokens * job.intermediate / VECTOR_WIDTH)
        postprocess = (
            _reduction_cycles(job.tokens * job.intermediate, parts)
            + activation
            + _reduction_cycles(job.tokens * shape.hidden, parts)
        )
    elif mode in {Mapping.N_TO_K, Mapping.M_BY_N}:
        capacities = [candidate.cores[core_id].cols for core_id in core_ids]
        intermediate_parts = split_integer_weighted(job.intermediate, capacities)
        transfer_weights = [2 * shape.hidden * width for width in intermediate_parts]
        transfer_bytes = _split_bursts(job.weight_bytes, transfer_weights)
        part_plans = tuple(
            PartPlan(
                index,
                transfer_id(index),
                job.tokens,
                shape.hidden,
                intermediate_parts[index],
                intermediate_parts[index],
                shape.hidden,
                core_id,
            )
            for index, core_id in enumerate(core_ids)
        )
        transfers = (
            ()
            if job.weight_cached
            else tuple(TransferPlan(index, size, (index,)) for index, size in enumerate(transfer_bytes))
        )
        postprocess = math.ceil(job.tokens * job.intermediate / VECTOR_WIDTH) + _reduction_cycles(
            job.tokens * shape.hidden,
            parts,
        )
    else:
        raise ValueError(f"unsupported mapping {mode}")

    if not part_plans:
        raise AssertionError("a planned job must contain at least one compute part")
    return PlannedJob(job, mode, part_plans, transfers, postprocess)


def compute_cycles(
    part: PartPlan,
    plan: PlannedJob,
    core: CoreGeometry,
    matrix_cycle_scale: float,
) -> float:
    raw = geometry_gemm_cycles(part.tokens, part.up_k, part.up_n, core)
    raw += geometry_gemm_cycles(part.tokens, part.down_k, part.down_n, core)
    cycles = raw * matrix_cycle_scale
    if len(plan.parts) > 1:
        cycles += PARTITION_LAUNCH_CYCLES
    return max(1.0, cycles)


def _job_priority(plan: PlannedJob, scheduler: str) -> tuple[float, ...]:
    if scheduler == "largest_first":
        return (-float(plan.job.weight_bytes), float(plan.job.job_id))
    return (float(plan.job.job_id),)


def simulate_event(
    jobs: Sequence[ExpertJob],
    candidate: Candidate,
    shape: MoeShape,
    calibration: CycleCalibration,
    *,
    weight_buffer_bytes: int | None = None,
) -> EventResult:
    if weight_buffer_bytes is None:
        weight_buffer_bytes = round_up(
            2 * shape.routed_weight_bytes,
            HBM_BURST_BYTES,
        )
    if weight_buffer_bytes <= 0:
        raise ValueError("weight_buffer_bytes must be positive")
    plans = [plan_job(job, candidate, shape) for job in jobs]
    largest_plan = max(
        (sum(transfer.bytes for transfer in plan.transfers) for plan in plans),
        default=0,
    )
    if largest_plan > weight_buffer_bytes:
        raise ValueError(f"weight buffer {weight_buffer_bytes} B cannot stage a {largest_plan} B job")
    pending = sorted(plans, key=lambda plan: _job_priority(plan, candidate.scheduler))
    runtime = {plan.job.job_id: JobRuntime(plan) for plan in plans}

    time = 0.0
    sequence = 0
    active_transfers: list[ActiveTransfer] = []
    ready_parts: list[ReadyPart] = []
    running: list[tuple[float, int, int, int, int]] = []
    finalizing: list[tuple[float, int, int]] = []
    free_cores = set(range(len(candidate.cores)))
    core_busy = [0.0 for _ in candidate.cores]
    transfer_parts_left: dict[tuple[int, int], int] = {}
    used_buffer_bytes = 0
    max_buffer_bytes = 0
    buffer_area = 0.0
    hbm_busy = 0.0
    hbm_starved_pe = 0.0
    postprocess_total = 0.0
    reducer_free_time = 0.0
    completed_jobs = 0

    def launch_pending() -> None:
        nonlocal sequence, used_buffer_bytes, max_buffer_bytes
        while pending:
            fit_index = next(
                (
                    index
                    for index, plan in enumerate(pending)
                    if not plan.transfers
                    or used_buffer_bytes + sum(transfer.bytes for transfer in plan.transfers) <= weight_buffer_bytes
                ),
                None,
            )
            if fit_index is None:
                return
            plan = pending.pop(fit_index)
            if not plan.transfers:
                for part in plan.parts:
                    sequence += 1
                    ready_parts.append(ReadyPart(sequence, plan.job.job_id, part))
                continue
            priority = _job_priority(plan, candidate.scheduler)
            for transfer in plan.transfers:
                sequence += 1
                transfer_parts_left[(plan.job.job_id, transfer.transfer_id)] = len(transfer.unlock_parts)
                active_transfers.append(
                    ActiveTransfer(
                        sequence,
                        plan.job.job_id,
                        transfer.transfer_id,
                        float(transfer.bytes),
                        (*priority, float(transfer.transfer_id)),
                    )
                )
                used_buffer_bytes += transfer.bytes
            max_buffer_bytes = max(max_buffer_bytes, used_buffer_bytes)

    def start_ready_parts() -> None:
        nonlocal sequence
        while free_cores and ready_parts:
            best: tuple[float, int, int, ReadyPart] | None = None
            for ready in ready_parts:
                plan = runtime[ready.job_id].plan
                eligible = (
                    free_cores if ready.part.preferred_core_id is None else free_cores & {ready.part.preferred_core_id}
                )
                for core_id in eligible:
                    finish = time + compute_cycles(
                        ready.part,
                        plan,
                        candidate.cores[core_id],
                        calibration.matrix_cycle_scale,
                    )
                    key = (finish, ready.sequence, core_id, ready)
                    if best is None or key[:3] < best[:3]:
                        best = key
            if best is None:
                return
            finish, _, core_id, ready = best
            ready_parts.remove(ready)
            free_cores.remove(core_id)
            core_busy[core_id] += finish - time
            sequence += 1
            heapq.heappush(
                running,
                (finish, sequence, core_id, ready.job_id, ready.part.part_id),
            )

    launch_pending()
    epsilon = 1e-7
    while completed_jobs < len(plans):
        start_ready_parts()
        rates: dict[int, float] = {}
        if active_transfers:
            if candidate.hbm_policy == "round_robin":
                share = calibration.hbm_bytes_per_cycle / len(active_transfers)
                rates = {index: share for index in range(len(active_transfers))}
            else:
                if candidate.hbm_policy == "critical_first":
                    chosen = min(
                        range(len(active_transfers)),
                        key=lambda index: (
                            active_transfers[index].priority,
                            active_transfers[index].sequence,
                        ),
                    )
                else:
                    chosen = min(
                        range(len(active_transfers)),
                        key=lambda index: active_transfers[index].sequence,
                    )
                rates = {chosen: calibration.hbm_bytes_per_cycle}

        hbm_delta = min(
            (active_transfers[index].remaining_bytes / rate for index, rate in rates.items()),
            default=math.inf,
        )
        compute_delta = running[0][0] - time if running else math.inf
        finalize_delta = finalizing[0][0] - time if finalizing else math.inf
        delta = min(hbm_delta, compute_delta, finalize_delta)
        if not math.isfinite(delta):
            raise RuntimeError(
                f"deadlock in {candidate.name}: pending={len(pending)} "
                f"active={len(active_transfers)} ready={len(ready_parts)} "
                f"running={len(running)} finalizing={len(finalizing)}"
            )
        if delta < -epsilon:
            raise AssertionError(f"time moved backwards by {delta}")
        delta = max(0.0, delta)

        buffer_area += used_buffer_bytes * delta
        if active_transfers:
            hbm_busy += delta
        if free_cores and (active_transfers or pending) and not ready_parts:
            hbm_starved_pe += sum(candidate.cores[index].pes for index in free_cores) * delta
        for index, rate in rates.items():
            active_transfers[index].remaining_bytes -= rate * delta
        time += delta

        completed_transfer_indices = [
            index for index, transfer in enumerate(active_transfers) if transfer.remaining_bytes <= epsilon
        ]
        for index in reversed(completed_transfer_indices):
            transfer = active_transfers.pop(index)
            plan = runtime[transfer.job_id].plan
            transfer_plan = next(item for item in plan.transfers if item.transfer_id == transfer.transfer_id)
            for part_id in transfer_plan.unlock_parts:
                sequence += 1
                ready_parts.append(ReadyPart(sequence, transfer.job_id, plan.parts[part_id]))

        while running and running[0][0] <= time + epsilon:
            _, _, core_id, job_id, part_id = heapq.heappop(running)
            free_cores.add(core_id)
            state = runtime[job_id]
            part = state.plan.parts[part_id]
            state.completed_parts += 1
            if part.transfer_id is not None:
                key = (job_id, part.transfer_id)
                transfer_parts_left[key] -= 1
                if transfer_parts_left[key] == 0:
                    transfer = next(item for item in state.plan.transfers if item.transfer_id == part.transfer_id)
                    used_buffer_bytes -= transfer.bytes
            if state.completed_parts == len(state.plan.parts):
                postprocess_total += state.plan.postprocess_cycles
                completion = MULTICORE_COMPLETION_CYCLES if len(state.plan.parts) > 1 else 0
                if state.plan.postprocess_cycles:
                    start = max(time, reducer_free_time)
                    reducer_free_time = start + state.plan.postprocess_cycles
                    finish = reducer_free_time + completion
                    sequence += 1
                    heapq.heappush(finalizing, (finish, sequence, job_id))
                elif completion:
                    sequence += 1
                    heapq.heappush(finalizing, (time + completion, sequence, job_id))
                else:
                    state.completion_time = time
                    completed_jobs += 1

        while finalizing and finalizing[0][0] <= time + epsilon:
            _, _, job_id = heapq.heappop(finalizing)
            runtime[job_id].completion_time = time
            completed_jobs += 1

        if used_buffer_bytes < 0:
            raise AssertionError("weight-buffer occupancy underflow")
        launch_pending()

    hbm_bytes = sum(transfer.bytes for plan in plans for transfer in plan.transfers)
    pe_busy = sum(busy * candidate.cores[index].pes for index, busy in enumerate(core_busy))
    pe_capacity = time * candidate.pe_count
    completion_times = [state.completion_time for state in runtime.values() if state.completion_time is not None]
    return EventResult(
        event_cycles=time,
        hbm_bytes=hbm_bytes,
        hbm_busy_cycles=hbm_busy,
        matrix_core_busy_cycles=sum(core_busy),
        matrix_pe_busy_cycles=pe_busy,
        matrix_pe_utilization=pe_busy / pe_capacity if pe_capacity else 0.0,
        hbm_starved_pe_cycles=hbm_starved_pe,
        postprocess_cycles=postprocess_total,
        average_buffer_occupancy_bytes=buffer_area / time if time else 0.0,
        max_buffer_occupancy_bytes=max_buffer_bytes,
        completion_skew_cycles=max(completion_times) - min(completion_times),
        completed_jobs=completed_jobs,
    )


def _jobs_for_pattern(
    pattern: tuple[bool, ...],
    shape: MoeShape,
    *,
    shared_resident: bool,
) -> tuple[ExpertJob, ...]:
    jobs = [
        ExpertJob(
            job_id=index,
            kind="routed",
            expert_id=index,
            tokens=1,
            intermediate=shape.routed_intermediate,
            weight_bytes=shape.routed_weight_bytes,
            weight_cached=hit,
        )
        for index, hit in enumerate(pattern)
    ]
    jobs.append(
        ExpertJob(
            job_id=len(jobs),
            kind="shared",
            expert_id=None,
            tokens=1,
            intermediate=shape.shared_intermediate,
            weight_bytes=shape.shared_weight_bytes,
            weight_cached=shared_resident,
        )
    )
    return tuple(jobs)


def evaluate_candidate(
    patterns: RoutingPatterns,
    candidate: Candidate,
    shape: MoeShape,
    calibration: CycleCalibration,
    *,
    decode_steps: int,
    shared_resident: bool,
    weight_buffer_bytes: int,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    max_buffer = 0
    for pattern, count in patterns.event_patterns:
        result = simulate_event(
            _jobs_for_pattern(pattern, shape, shared_resident=shared_resident),
            candidate,
            shape,
            calibration,
            weight_buffer_bytes=weight_buffer_bytes,
        )
        scaled = result.scaled_dict(count)
        for key, value in scaled.items():
            if key == "max_buffer_occupancy_bytes":
                max_buffer = max(max_buffer, int(value))
            else:
                totals[key] += value
    total_cycles = float(totals["cycles"])
    events = int(totals["events"])
    return {
        "candidate": candidate.to_dict(),
        "calibration": asdict(calibration),
        "capacity_entries": patterns.capacity_entries,
        "shared_resident": shared_resident,
        "events": events,
        "decode_steps": decode_steps,
        "total_cycles": total_cycles,
        "cycles_per_moe_layer_event": total_cycles / events,
        "moe_body_cycles_per_decode_token": total_cycles / decode_steps,
        "hbm_bytes": int(totals["hbm_bytes"]),
        "hbm_bytes_per_decode_token": totals["hbm_bytes"] / decode_steps,
        "hbm_gib_per_decode_token": totals["hbm_bytes"] / decode_steps / GIB,
        "hbm_busy_cycles": float(totals["hbm_busy_cycles"]),
        "matrix_pe_utilization": (
            totals["matrix_pe_busy_cycles"] / (total_cycles * candidate.pe_count) if total_cycles else 0.0
        ),
        "hbm_starved_pe_cycles": float(totals["hbm_starved_pe_cycles"]),
        "postprocess_cycles": float(totals["postprocess_cycles"]),
        "weight_buffer_capacity_bytes": weight_buffer_bytes,
        "average_buffer_occupancy_bytes": (
            totals["buffer_occupancy_byte_cycles"] / total_cycles if total_cycles else 0.0
        ),
        "max_buffer_occupancy_bytes": max_buffer,
        "mean_completion_skew_cycles": totals["completion_skew_cycles"] / events,
        "routing_cache_hit_rate": patterns.hit_rate,
    }


def _rank_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, bool], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["calibration"]["name"],
            record["capacity_entries"],
            record["shared_resident"],
        )
        grouped.setdefault(key, []).append(record)
    ranked: list[dict[str, Any]] = []
    for rows in grouped.values():
        baseline = next(row for row in rows if row["candidate"]["name"] == "baseline_4x1024__expert")
        for rank, row in enumerate(sorted(rows, key=lambda item: item["total_cycles"]), 1):
            row["rank"] = rank
            row["speedup_vs_baseline"] = baseline["total_cycles"] / row["total_cycles"]
            ranked.append(row)
    return ranked


def build_report(
    arch: ModelArchConfig,
    trace_path: Path = PINNED_TRACE,
    *,
    capacities: tuple[int, ...] = (0, 23, 46, 92, 137, 138, 256, 512),
    expert_order: str = "expert_id",
    weight_buffer_bytes: int | None = None,
) -> dict[str, Any]:
    trace = load_routing_trace(trace_path)
    shape = MoeShape.from_arch(arch)
    if weight_buffer_bytes is None:
        weight_buffer_bytes = round_up(
            2 * shape.routed_weight_bytes,
            HBM_BURST_BYTES,
        )
    calibrations = cycle_calibrations(shape)
    candidates = candidate_space()
    pattern_summaries = {
        capacity: routing_pattern_histogram(
            trace,
            capacity_entries=capacity,
            expert_order=expert_order,
        )
        for capacity in capacities
    }
    decode_steps = trace["shape"]["recurrent_decode_steps"]
    records = [
        evaluate_candidate(
            pattern_summaries[capacity],
            candidate,
            shape,
            calibration,
            decode_steps=decode_steps,
            shared_resident=shared_resident,
            weight_buffer_bytes=weight_buffer_bytes,
        )
        for calibration in calibrations
        for capacity in capacities
        for shared_resident in (False, True)
        for candidate in candidates
    ]
    records = _rank_records(records)
    shared_stream_sensitivity = [
        row
        for row in records
        if row["capacity_entries"] == 138
        and not row["shared_resident"]
        and row["calibration"]["name"] == "transferred_shared_moe"
    ]
    moe_layers = trace["shape"]["layers"]
    return {
        "schema_version": 1,
        "status": "exact_routing_and_event_scheduler_plena_cycles_uncalibrated",
        "trace_source": trace["source"],
        "trace_shape": trace["shape"],
        "architecture": {
            **asdict(shape),
            "routed_weight_mib": shape.routed_weight_bytes / MIB,
            "shared_weight_mib": shape.shared_weight_bytes / MIB,
            "routed_cache_138_mib": 138 * shape.routed_weight_bytes / MIB,
            "all_shared_resident_mib": moe_layers * shape.shared_weight_bytes / MIB,
            "routed138_plus_shared_mib": (138 * shape.routed_weight_bytes + moe_layers * shape.shared_weight_bytes)
            / MIB,
            "weight_precision": Precision.NVFP4,
            "hbm_burst_bytes": HBM_BURST_BYTES,
            "fixed_pe_budget": BASELINE_PES,
            "weight_staging_buffer_bytes": weight_buffer_bytes,
            "weight_staging_buffer_mib": weight_buffer_bytes / MIB,
        },
        "calibrations": [asdict(calibration) for calibration in calibrations],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "routing_patterns": {str(capacity): summary.to_dict() for capacity, summary in pattern_summaries.items()},
        "records": records,
        "shared_stream_sensitivity_capacity138": shared_stream_sensitivity,
        "assumptions": [
            "One event is one B1 decode token at one Nemotron MoE layer: six routed experts plus one shared expert.",
            "Exact routed-expert cache hits are replayed in expert-ID order; events are compressed only by identical six-bit hit masks.",
            "Both shared-streaming and an all-23-shared-resident upper bound are swept; full shared residency alone needs 246.22 MiB.",
            "A 138-entry routed cache plus all shared experts needs about 984.87 MiB before metadata and is not assumed FPGA-on-chip feasible.",
            "Every candidate has exactly 4096 Matrix PEs and the same aggregate HBM bandwidth; weight bytes are never duplicated by a split.",
            "M-split and the M dimension of 2-D mappings collapse at B1 because each routed expert receives one token.",
            "N-to-K partitions the up-projection output columns and uses the same slices as down-projection K partitions, requiring only the final hidden reduction.",
            "Every candidate uses the same byte-accurate staging buffer, sized by default for two routed experts (or one shared expert); split partitions do not increase capacity.",
            "The event scheduler models the finite staging buffer, one shared HBM server, asynchronous Matrix completion, and one shared postprocess/reduction resource.",
            "Layer-to-layer prefetch and router/dispatch/combine cycles are excluded, so reported cycles cover the expert body only.",
            "No cycle calibration is direct Nemotron RTL evidence; speedups are within-model topology comparisons only.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    records = report["records"]
    lines = [
        "# Nemotron 3 Exact-Routing MoE Event DSE",
        "",
        "## 证据边界",
        "",
        "- 输入是 B200 campaign 的 127-step × 23-layer 真实 top-6 trace。",
        "- 每个候选固定 4096 PE、同一条共享 HBM、64B burst；只改变 core 切法与 Expert/M/K 映射。",
        "- `ideal_geometry_hbm64` 是几何下限；`transferred_shared_moe` 是从旧 Qwen/DeepSeek PLENA 实验转移的敏感性点，不是 Nemotron RTL 标定。",
        "- 下表是 MoE expert body 内部比较，不是 PLENA 对 B200 的端到端加速。",
        "",
        "## Workload",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| Routed expert weight | {report['architecture']['routed_weight_mib']:.3f} MiB |",
        f"| Shared expert weight | {report['architecture']['shared_weight_mib']:.3f} MiB |",
        f"| Shared staging buffer for every candidate | {report['architecture']['weight_staging_buffer_mib']:.3f} MiB |",
        f"| Layer-step events | {report['trace_shape']['recurrent_decode_steps'] * report['trace_shape']['layers']:,} |",
        f"| Routed accesses | {report['trace_shape']['recurrent_decode_steps'] * report['trace_shape']['layers'] * report['trace_shape']['top_k']:,} |",
        "",
        "## Capacity 138, Shared Resident Upper Bound",
        "",
        f"This point needs {report['architecture']['routed138_plus_shared_mib']:.2f} MiB before metadata; it is a capacity upper bound, not an FPGA SRAM proposal.",
        "",
    ]
    for calibration in report["calibrations"]:
        rows = sorted(
            (
                row
                for row in records
                if row["calibration"]["name"] == calibration["name"]
                and row["capacity_entries"] == 138
                and row["shared_resident"]
            ),
            key=lambda row: row["rank"],
        )
        lines.extend(
            [
                f"### {calibration['name']}",
                "",
                "| Rank | Candidate | Mapping | Speedup | MoE-body cycles/token | PE util | HBM GiB/token |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['candidate']['name']} | {row['candidate']['mapping']} | "
                f"{row['speedup_vs_baseline']:.3f}x | {row['moe_body_cycles_per_decode_token']:,.0f} | "
                f"{100 * row['matrix_pe_utilization']:.1f}% | {row['hbm_gib_per_decode_token']:.3f} |"
            )
        lines.append("")

    cache_rows = report["routing_patterns"]
    lines.extend(
        [
            "## Exact Cache Replay",
            "",
            "| Routed slots | Capacity MiB | Hit rate | Misses | Distinct event masks |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for capacity in sorted(map(int, cache_rows)):
        row = cache_rows[str(capacity)]
        lines.append(
            f"| {capacity} | {capacity * report['architecture']['routed_weight_mib']:.1f} | "
            f"{100 * row['hit_rate']:.2f}% | {row['misses']:,} | {row['distinct_patterns']} |"
        )
    no_cache_stream = [
        row
        for row in records
        if row["capacity_entries"] == 0
        and not row["shared_resident"]
        and row["calibration"]["name"] == "ideal_geometry_hbm64"
    ]
    baseline_stream = next(row for row in no_cache_stream if row["candidate"]["name"] == "baseline_4x1024__expert")
    lines.extend(
        [
            "",
            "No-cache + shared-stream baseline reads "
            f"{baseline_stream['hbm_gib_per_decode_token']:.3f} GiB/token for expert weights. "
            "This intentionally matches the logical-workload model before router/activation traffic.",
            "",
            "## 判断",
            "",
            "1. M-split 在 B1 每个 expert 只有一个 token 时退化，不能作为 Nemotron decode 的主方案。",
            "2. Expert split、K-split 与 N-to-K 的排名必须同时通过两套 cycle sensitivity；只在旧标定下胜出的方案不能冻结进 RTL。",
            "3. 138-slot 的 cache 拐点现在已经真正接入 HBM 时间线，不再只是独立的 hit-rate 表。",
            "4. 下一步用 Rust/RTL 取得 Nemotron expert 的 Matrix wave、dequant 和 reduction 周期，再替换 transferred calibration。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=PINNED_TRACE)
    parser.add_argument(
        "--capacities",
        default=",".join(map(str, (0, 23, 46, 92, 137, 138, 256, 512))),
    )
    parser.add_argument("--expert-order", choices=("expert_id", "topk_rank"), default="expert_id")
    parser.add_argument(
        "--weight-buffer-mib",
        type=float,
        help="shared staging-buffer capacity; default is two routed expert weights",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)
    capacities = tuple(int(item) for item in args.capacities.split(",") if item)
    arch = load_model_config("nemotron3_nano_30b_a3b").arch
    report = build_report(
        arch,
        args.trace,
        capacities=capacities,
        expert_order=args.expert_order,
        weight_buffer_bytes=(math.ceil(args.weight_buffer_mib * MIB) if args.weight_buffer_mib is not None else None),
    )
    rendered = json.dumps(report, indent=2) + "\n"
    markdown = render_markdown(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
