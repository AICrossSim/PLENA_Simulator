"""Resource-constrained full-model timeline for Nemotron 3 and Kimi K3.

The workload modules count architecture-independent work. This module maps the
ordered stages onto one candidate PLENA device, where HBM, Matrix, Vector,
State, Conv, Exp, SRAM, and L-Compute are finite shared resources. It is a
pre-RTL model: cycle parameters are explicit DSE inputs and are never fitted
from GPU kernel time.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import load_model_config

from .b200_formal_campaign import build_report as build_b200_campaign
from .hybrid_routing_trace import KimiRoutingTrace, load_kimi_routing_trace
from .kimi_k3_workload import KimiK3Architecture, KimiK3HybridWorkloadModel
from .nemotron3_dse import (
    HardwareDesign,
    ProjectionBankModel,
    ProjectionLayout,
    ProjectionWriteBufferModel,
    StateCachePolicy,
)
from .nemotron3_routing_dse import PINNED_TRACE, load_routing_trace
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    StageWork,
    Traffic,
    WorkloadReport,
    WorkloadScenario,
    formal_nemotron_nvfp4_weight_policy,
    storage_bytes,
)


MIB = 1024 * 1024


class ModelFamily(StrEnum):
    NEMOTRON3 = "nemotron3"
    KIMI_K3 = "kimi_k3"


class Resource(StrEnum):
    HBM = "hbm"
    SRAM = "sram"
    MATRIX = "matrix"
    VECTOR = "vector"
    STATE = "state"
    CONV = "conv"
    EXP = "exp"
    LAYOUT = "layout"


@dataclass(frozen=True)
class SystemDesign:
    name: str = "plena_hybrid_candidate"
    frequency_hz: int = 1_000_000_000
    matrix_macs_per_cycle: int = 4096
    vector_ops_per_cycle: int = 256
    conv_macs_per_cycle: int = 256
    exp_ops_per_cycle: int = 16
    scan_compositions_per_cycle: int = 1
    hbm_bytes_per_cycle: int = 64
    hbm_burst_bytes: int = 64
    sram_bytes_per_cycle: int = 128
    activation_sram_bytes: int = 4 * MIB
    activation_tile_tokens: int = 64
    fused_layer_dataflow: bool = True
    projection_buffer_banks: int = 16
    projection_buffer_ports_per_bank: int = 1
    projection_fifo_values: int = 64
    matrix_result_burst_values: int = 64
    projection_buffer_write_values_per_cycle: int = 16
    projection_consume_values_per_cycle: int = 16
    projection_layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    projection_direct_bypass: bool = True
    bc_broadcast: bool = True
    head_lanes: int = 8
    head_dim_lanes: int = 4
    state_dim_lanes: int = 8
    state_cache_bytes: int = 0
    state_cache_policy: StateCachePolicy = StateCachePolicy.NONE
    kv_cache_bytes: int = 0
    moe_weight_cache_bytes: int = 0
    calibrated: bool = False

    def __post_init__(self) -> None:
        positive = (
            "frequency_hz",
            "matrix_macs_per_cycle",
            "vector_ops_per_cycle",
            "conv_macs_per_cycle",
            "exp_ops_per_cycle",
            "scan_compositions_per_cycle",
            "hbm_bytes_per_cycle",
            "hbm_burst_bytes",
            "sram_bytes_per_cycle",
            "activation_tile_tokens",
            "projection_buffer_banks",
            "projection_buffer_ports_per_bank",
            "projection_fifo_values",
            "matrix_result_burst_values",
            "projection_buffer_write_values_per_cycle",
            "projection_consume_values_per_cycle",
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "activation_sram_bytes",
            "state_cache_bytes",
            "kv_cache_bytes",
            "moe_weight_cache_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.projection_fifo_values < self.matrix_result_burst_values:
            raise ValueError("projection FIFO must hold one Matrix-result burst")
        if self.state_cache_bytes == 0 and self.state_cache_policy != StateCachePolicy.NONE:
            raise ValueError("zero-sized state cache requires policy=none")
        if self.state_cache_bytes > 0 and self.state_cache_policy == StateCachePolicy.NONE:
            raise ValueError("non-zero state cache requires lru or pinned policy")

    @property
    def state_macs_per_cycle(self) -> int:
        return self.head_lanes * self.head_dim_lanes * self.state_dim_lanes

    def projection_design(self) -> HardwareDesign:
        return HardwareDesign(
            name=self.name,
            frequency_hz=self.frequency_hz,
            matrix_macs_per_cycle=self.matrix_macs_per_cycle,
            vector_ops_per_cycle=self.vector_ops_per_cycle,
            conv_macs_per_cycle=self.conv_macs_per_cycle,
            exp_ops_per_cycle=self.exp_ops_per_cycle,
            hbm_bytes_per_cycle=self.hbm_bytes_per_cycle,
            projection_buffer_banks=self.projection_buffer_banks,
            projection_buffer_ports_per_bank=self.projection_buffer_ports_per_bank,
            matrix_result_burst_values=self.matrix_result_burst_values,
            projection_buffer_write_values_per_cycle=self.projection_buffer_write_values_per_cycle,
            projection_fifo_values=self.projection_fifo_values,
            projection_direct_bypass=self.projection_direct_bypass,
            projection_consume_values_per_cycle=self.projection_consume_values_per_cycle,
            head_lanes=self.head_lanes,
            head_dim_lanes=self.head_dim_lanes,
            state_dim_lanes=self.state_dim_lanes,
            projection_layout=self.projection_layout,
            bc_broadcast=self.bc_broadcast,
            state_cache_bytes=self.state_cache_bytes,
            state_cache_policy=self.state_cache_policy,
            calibrated=self.calibrated,
        )


@dataclass(frozen=True)
class PrecisionConfig:
    activation: Precision = Precision.BF16
    weight: Precision = Precision.BF16
    state: Precision = Precision.FP32
    conv_state: Precision | None = None
    use_formal_nemotron_weight_map: bool = True

    def resolved_conv_state(self, model: ModelFamily) -> Precision:
        if self.conv_state is not None:
            return self.conv_state
        return Precision.FP32 if model == ModelFamily.NEMOTRON3 else Precision.BF16


@dataclass(frozen=True)
class ResourceSpan:
    resource: Resource
    kind: str
    release_cycle: int
    start_cycle: int
    end_cycle: int
    service_cycles: int
    queue_wait_cycles: int
    logical_bytes: int = 0
    physical_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resource"] = self.resource.value
        return result


@dataclass(frozen=True)
class TimelineStage:
    index: int
    token_index: int
    context_length: int
    layer_id: int
    layer_type: str
    name: str
    release_cycle: int
    compute_end_cycle: int
    data_ready_cycle: int
    retire_cycle: int
    effective_macs: int
    logical_hbm_read_bytes: int
    logical_hbm_write_bytes: int
    physical_hbm_read_bytes: int
    physical_hbm_write_bytes: int
    bank_service_cycles: int
    bank_stall_cycles: int
    spans: tuple[ResourceSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "spans"},
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class TokenTiming:
    token_index: int
    context_length: int
    start_cycle: int
    ready_cycle: int

    @property
    def cycles(self) -> int:
        return self.ready_cycle - self.start_cycle

    def to_dict(self, frequency_hz: int) -> dict[str, int | float]:
        return {
            **asdict(self),
            "cycles": self.cycles,
            "latency_us": self.cycles / frequency_hz * 1e6,
        }


@dataclass(frozen=True)
class StateCacheSummary:
    policy: StateCachePolicy
    capacity_bytes: int
    entry_bytes: int
    total_entries: int
    resident_entries: int
    accesses: int
    hits: int
    misses: int
    warm_start: bool
    preload_bytes: int
    final_flush_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "policy": self.policy.value, "hit_rate": self.hit_rate}


@dataclass(frozen=True)
class KvCacheSummary:
    capacity_bytes: int
    read_bytes: int
    hit_bytes: int
    hbm_read_bytes: int
    write_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hit_bytes / self.read_bytes if self.read_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hit_rate": self.hit_rate}


@dataclass(frozen=True)
class MoeCacheSummary:
    capacity_bytes: int
    entry_bytes: int
    capacity_entries: int
    accesses: int
    hits: int
    misses: int
    hbm_read_bytes: int
    routing_source: str

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "hit_rate": self.hit_rate}


@dataclass(frozen=True)
class SystemTimelineReport:
    model: ModelFamily
    phase: InferencePhase
    design: SystemDesign
    precision: PrecisionConfig
    system_spans: tuple[ResourceSpan, ...]
    stages: tuple[TimelineStage, ...]
    tokens: tuple[TokenTiming, ...]
    state_cache: StateCacheSummary
    kv_cache: KvCacheSummary
    moe_cache: MoeCacheSummary
    resource_busy_cycles: dict[Resource, int]
    resource_queue_wait_cycles: dict[Resource, int]
    model_ready_cycle: int
    final_cycle: int
    gpu_evidence: dict[str, Any]
    limits: tuple[str, ...]

    @property
    def finalization_cycles(self) -> int:
        return self.final_cycle - self.model_ready_cycle

    @property
    def average_cycles_per_token(self) -> float:
        return sum(token.cycles for token in self.tokens) / len(self.tokens)

    @property
    def prompt_tokens(self) -> int:
        if self.phase == InferencePhase.PREFILL:
            return self.tokens[0].context_length
        return 0

    @property
    def ttft_cycles(self) -> int | None:
        return self.model_ready_cycle if self.phase == InferencePhase.PREFILL else None

    @property
    def tpot_cycles(self) -> float | None:
        return self.average_cycles_per_token if self.phase == InferencePhase.DECODE else None

    def to_dict(self, *, include_stages: bool = True) -> dict[str, Any]:
        total_hbm_read = sum(stage.logical_hbm_read_bytes for stage in self.stages)
        total_hbm_write = sum(stage.logical_hbm_write_bytes for stage in self.stages)
        physical_read = sum(stage.physical_hbm_read_bytes for stage in self.stages)
        physical_write = sum(stage.physical_hbm_write_bytes for stage in self.stages)
        physical_system = sum(span.physical_bytes for span in self.system_spans)
        layer_service: dict[str, int] = {}
        layer_hbm: dict[str, int] = {}
        for stage in self.stages:
            layer_service[stage.layer_type] = layer_service.get(stage.layer_type, 0) + sum(
                span.service_cycles for span in stage.spans
            )
            layer_hbm[stage.layer_type] = layer_hbm.get(stage.layer_type, 0) + (
                stage.logical_hbm_read_bytes + stage.logical_hbm_write_bytes
            )
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "pre_rtl_resource_timeline_uncalibrated",
            "model": self.model.value,
            "phase": self.phase.value,
            "design": {
                **asdict(self.design),
                "projection_layout": self.design.projection_layout.value,
                "state_cache_policy": self.design.state_cache_policy.value,
                "state_macs_per_cycle": self.design.state_macs_per_cycle,
            },
            "precision": {
                **asdict(self.precision),
                "activation": self.precision.activation.value,
                "weight": self.precision.weight.value,
                "state": self.precision.state.value,
                "conv_state": self.precision.resolved_conv_state(self.model).value,
            },
            "metrics": {
                "model_ready_cycles": self.model_ready_cycle,
                "final_cycle": self.final_cycle,
                "finalization_cycles": self.finalization_cycles,
                "average_cycles_per_token": self.average_cycles_per_token,
                "average_latency_us_per_token": self.average_cycles_per_token / self.design.frequency_hz * 1e6,
                "prompt_tokens": self.prompt_tokens,
                "prefill_cycles_per_prompt_token": (
                    self.model_ready_cycle / self.prompt_tokens if self.prompt_tokens else None
                ),
                "ttft_cycles": self.ttft_cycles,
                "ttft_us": (
                    self.ttft_cycles / self.design.frequency_hz * 1e6 if self.ttft_cycles is not None else None
                ),
                "tpot_cycles": self.tpot_cycles,
                "tpot_us": (
                    self.tpot_cycles / self.design.frequency_hz * 1e6 if self.tpot_cycles is not None else None
                ),
                "logical_hbm_read_bytes": total_hbm_read,
                "logical_hbm_write_bytes": total_hbm_write,
                "physical_burst_hbm_read_bytes": physical_read,
                "physical_burst_hbm_write_bytes": physical_write,
                "physical_system_state_transfer_bytes": physical_system,
                "bank_service_cycles": sum(stage.bank_service_cycles for stage in self.stages),
                "bank_stall_cycles": sum(stage.bank_stall_cycles for stage in self.stages),
                "resource_busy_cycles": {
                    resource.value: cycles for resource, cycles in self.resource_busy_cycles.items()
                },
                "resource_queue_wait_cycles": {
                    resource.value: cycles for resource, cycles in self.resource_queue_wait_cycles.items()
                },
                "resource_utilization": {
                    resource.value: cycles / self.final_cycle if self.final_cycle else 0.0
                    for resource, cycles in self.resource_busy_cycles.items()
                },
                "service_cycle_breakdown": layer_service,
                "logical_hbm_byte_breakdown": layer_hbm,
            },
            "tokens": [token.to_dict(self.design.frequency_hz) for token in self.tokens],
            "state_cache": self.state_cache.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "moe_weight_cache": self.moe_cache.to_dict(),
            "gpu_evidence": self.gpu_evidence,
            "limits": list(self.limits),
            "system_spans": [span.to_dict() for span in self.system_spans],
        }
        if include_stages:
            result["stages"] = [stage.to_dict() for stage in self.stages]
        return result


class _ResourceScheduler:
    def __init__(self, design: SystemDesign) -> None:
        self.design = design
        self.available = {resource: 0 for resource in Resource}
        self.busy = {resource: 0 for resource in Resource}
        self.wait = {resource: 0 for resource in Resource}

    def reserve(
        self,
        resource: Resource,
        release: int,
        cycles: int,
        kind: str,
        *,
        logical_bytes: int = 0,
        physical_bytes: int = 0,
    ) -> ResourceSpan | None:
        if cycles <= 0:
            return None
        start = max(release, self.available[resource])
        end = start + cycles
        wait = start - release
        self.available[resource] = end
        self.busy[resource] += cycles
        self.wait[resource] += wait
        return ResourceSpan(
            resource,
            kind,
            release,
            start,
            end,
            cycles,
            wait,
            logical_bytes,
            physical_bytes,
        )

    def transfer(self, release: int, byte_count: int, kind: str) -> ResourceSpan | None:
        if byte_count <= 0:
            return None
        physical = _round_up(byte_count, self.design.hbm_burst_bytes)
        cycles = math.ceil(physical / self.design.hbm_bytes_per_cycle)
        return self.reserve(
            Resource.HBM,
            release,
            cycles,
            kind,
            logical_bytes=byte_count,
            physical_bytes=physical,
        )


class _StateResidency:
    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        entry_bytes: int,
        design: SystemDesign,
        warm_start: bool,
    ) -> None:
        self.layer_ids = layer_ids
        self.entry_bytes = entry_bytes
        capacity_entries = min(len(layer_ids), design.state_cache_bytes // entry_bytes if entry_bytes else 0)
        if design.state_cache_policy == StateCachePolicy.PINNED:
            self.resident = set(layer_ids[:capacity_entries])
        elif design.state_cache_policy == StateCachePolicy.LRU and capacity_entries == len(layer_ids):
            self.resident = set(layer_ids)
        else:
            self.resident = set()
        self.policy = design.state_cache_policy
        self.capacity = design.state_cache_bytes
        self.warm_start = warm_start
        self.accesses = 0
        self.hits = 0
        self.seen: set[tuple[int, int]] = set()

    def is_resident(self, token_index: int, layer_id: int) -> bool:
        key = (token_index, layer_id)
        if key not in self.seen:
            self.seen.add(key)
            self.accesses += 1
            if layer_id in self.resident:
                self.hits += 1
        return layer_id in self.resident

    def summary(self, phase: InferencePhase) -> StateCacheSummary:
        preload = 0 if self.warm_start or phase == InferencePhase.PREFILL else len(self.resident) * self.entry_bytes
        flush = len(self.resident) * self.entry_bytes if phase == InferencePhase.DECODE else 0
        return StateCacheSummary(
            policy=self.policy,
            capacity_bytes=self.capacity,
            entry_bytes=self.entry_bytes,
            total_entries=len(self.layer_ids),
            resident_entries=len(self.resident),
            accesses=self.accesses,
            hits=self.hits,
            misses=self.accesses - self.hits,
            warm_start=self.warm_start,
            preload_bytes=preload,
            final_flush_bytes=flush,
        )


class _MoeWeightCache:
    def __init__(self, capacity_bytes: int, entry_bytes: int, routing_source: str) -> None:
        self.capacity_bytes = capacity_bytes
        self.entry_bytes = entry_bytes
        self.capacity_entries = capacity_bytes // entry_bytes if entry_bytes else 0
        self.routing_source = routing_source
        self.cache: OrderedDict[tuple[str, int, int], int] = OrderedDict()
        self.used_bytes = 0
        self.accesses = 0
        self.hits = 0
        self.hbm_read_bytes = 0

    def seed(self, entries: list[tuple[tuple[str, int, int], int]]) -> None:
        for key, size in entries:
            self._access(key, size, charge=False)

    def access(self, entries: tuple[tuple[tuple[str, int, int], int], ...]) -> int:
        transferred = 0
        for key, size in entries:
            if not self._access(key, size, charge=True):
                transferred += size
        self.hbm_read_bytes += transferred
        return transferred

    def _access(self, key: tuple[str, int, int], size: int, *, charge: bool) -> bool:
        if charge:
            self.accesses += 1
        if self.capacity_bytes == 0 or size > self.capacity_bytes:
            return False
        if key in self.cache:
            self.cache.move_to_end(key)
            if charge:
                self.hits += 1
            return True
        while self.cache and self.used_bytes + size > self.capacity_bytes:
            _, evicted_size = self.cache.popitem(last=False)
            self.used_bytes -= evicted_size
        self.cache[key] = size
        self.used_bytes += size
        return False

    def summary(self) -> MoeCacheSummary:
        return MoeCacheSummary(
            capacity_bytes=self.capacity_bytes,
            entry_bytes=self.entry_bytes,
            capacity_entries=self.capacity_entries,
            accesses=self.accesses,
            hits=self.hits,
            misses=self.accesses - self.hits,
            hbm_read_bytes=self.hbm_read_bytes,
            routing_source=self.routing_source,
        )


@dataclass
class _KvTracker:
    capacity_bytes: int
    read_bytes: int = 0
    hit_bytes: int = 0
    write_bytes: int = 0

    def apply(self, traffic: Traffic, *, total_history_bytes: int) -> Traffic:
        self.read_bytes += traffic.kv_read_bytes
        self.write_bytes += traffic.kv_write_bytes
        if traffic.kv_read_bytes == 0 or total_history_bytes <= 0:
            return traffic
        hit_rate = min(1.0, self.capacity_bytes / total_history_bytes)
        hits = min(traffic.kv_read_bytes, math.floor(traffic.kv_read_bytes * hit_rate))
        self.hit_bytes += hits
        return replace(traffic, kv_read_bytes=traffic.kv_read_bytes - hits)

    def summary(self) -> KvCacheSummary:
        return KvCacheSummary(
            capacity_bytes=self.capacity_bytes,
            read_bytes=self.read_bytes,
            hit_bytes=self.hit_bytes,
            hbm_read_bytes=self.read_bytes - self.hit_bytes,
            write_bytes=self.write_bytes,
        )


@dataclass(frozen=True)
class _StageInstance:
    token_index: int
    context_length: int
    phase: InferencePhase
    workload_tokens: int
    stage: StageWork
    total_kv_history_bytes: int


class HybridSystemTimelineModel:
    """Build and schedule full text-backbone prefill or decode workloads."""

    def __init__(
        self,
        model: ModelFamily,
        design: SystemDesign,
        precision: PrecisionConfig = PrecisionConfig(),
        *,
        routing_trace_path: Path = PINNED_TRACE,
        kimi_routing_trace_path: Path | None = None,
    ) -> None:
        self.model = model
        self.design = design
        self.precision = precision
        self.routing_trace = load_routing_trace(routing_trace_path) if model == ModelFamily.NEMOTRON3 else None
        self.kimi_routing_trace: KimiRoutingTrace | None = (
            load_kimi_routing_trace(kimi_routing_trace_path)
            if model == ModelFamily.KIMI_K3 and kimi_routing_trace_path is not None
            else None
        )
        self._nemotron_config = load_model_config("nemotron3_nano_30b_a3b") if model == ModelFamily.NEMOTRON3 else None
        self._kimi_arch = KimiK3Architecture() if model == ModelFamily.KIMI_K3 else None
        self._campaign = _formal_campaign()
        self._mamba_bank_stats: tuple[int, int, int, int] | None = None
        self._mamba_projection_fifo_stall: dict[int, int] = {}
        self._kda_bank_cache: dict[tuple[bool, int, int, int], tuple[int, int]] = {}

    def simulate(
        self,
        phase: InferencePhase,
        *,
        batch_size: int = 1,
        sequence_length: int = 2048,
        context_length: int = 2048,
        decode_tokens: int = 4,
        include_embedding: bool = True,
        include_lm_head: bool = True,
        warm_state_cache: bool = True,
    ) -> SystemTimelineReport:
        if batch_size != 1:
            raise ValueError("the first full-system timeline supports batch_size=1 only")
        if phase == InferencePhase.DECODE and decode_tokens <= 0:
            raise ValueError("decode_tokens must be positive")
        if phase == InferencePhase.PREFILL and sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        instances, reports = self._workload_instances(
            phase,
            sequence_length=sequence_length,
            context_length=context_length,
            decode_tokens=decode_tokens,
            include_embedding=include_embedding,
            include_lm_head=include_lm_head,
        )
        state = self._state_residency(warm_state_cache)
        kv = _KvTracker(self.design.kv_cache_bytes)
        moe = self._moe_cache(phase)
        scheduler = _ResourceScheduler(self.design)
        stages: list[TimelineStage] = []
        tokens: list[TokenTiming] = []
        system_spans: list[ResourceSpan] = []
        dependency = 0

        state_summary_before = state.summary(phase)
        preload = scheduler.transfer(0, state_summary_before.preload_bytes, "state_cache_preload")
        if preload is not None:
            system_spans.append(preload)
            dependency = preload.end_cycle

        current_token = -1
        token_start = dependency
        token_context = 0
        for instance in instances:
            if instance.token_index != current_token:
                if current_token >= 0:
                    tokens.append(TokenTiming(current_token, token_context, token_start, dependency))
                current_token = instance.token_index
                token_start = dependency
                token_context = instance.context_length
            stage = self._prepare_stage(instance, state, kv, moe)
            timed = self._schedule_stage(len(stages), instance, stage, scheduler, dependency)
            stages.append(timed)
            dependency = timed.data_ready_cycle
        if current_token >= 0:
            tokens.append(TokenTiming(current_token, instances[-1].context_length, token_start, dependency))

        model_ready = dependency
        state_summary = state.summary(phase)
        flush = scheduler.transfer(model_ready, state_summary.final_flush_bytes, "state_cache_final_flush")
        if flush is not None:
            system_spans.append(flush)
        final_cycle = max(
            model_ready,
            flush.end_cycle if flush is not None else 0,
            *scheduler.available.values(),
        )
        routing_limit = (
            "Kimi expert traffic is driven by the validated empirical top-16 trace."
            if self.model == ModelFamily.KIMI_K3 and self.kimi_routing_trace is not None
            else "Kimi routing uses deterministic sensitivity traffic until an empirical Kimi top-16 trace is supplied."
        )
        return SystemTimelineReport(
            model=self.model,
            phase=phase,
            design=self.design,
            precision=self.precision,
            system_spans=tuple(system_spans),
            stages=tuple(stages),
            tokens=tuple(tokens),
            state_cache=state_summary,
            kv_cache=kv.summary(),
            moe_cache=moe.summary(),
            resource_busy_cycles=dict(scheduler.busy),
            resource_queue_wait_cycles=dict(scheduler.wait),
            model_ready_cycle=model_ready,
            final_cycle=final_cycle,
            gpu_evidence=self._gpu_evidence(phase, reports),
            limits=(
                "Cycle parameters are DSE inputs, not RTL-calibrated throughput or frequency.",
                "B200/RTX profiling validates shapes, precision, routing, and bottleneck direction; GPU time is not converted to PLENA cycles.",
                "HBM uses one non-preemptive writeback-first server with burst rounding; future RTL arbitration may change queue order.",
                "The timeline executes the full logical text backbone but does not bind or numerically replay complete checkpoint weights.",
                routing_limit,
            ),
        )

    def _workload_instances(
        self,
        phase: InferencePhase,
        *,
        sequence_length: int,
        context_length: int,
        decode_tokens: int,
        include_embedding: bool,
        include_lm_head: bool,
    ) -> tuple[list[_StageInstance], list[WorkloadReport]]:
        instances: list[_StageInstance] = []
        reports: list[WorkloadReport] = []
        steps = decode_tokens if phase == InferencePhase.DECODE else 1
        for token_index in range(steps):
            step_context = context_length + token_index if phase == InferencePhase.DECODE else sequence_length
            scenario = WorkloadScenario(
                phase=phase,
                batch_size=1,
                sequence_length=1 if phase == InferencePhase.DECODE else sequence_length,
                context_length=step_context,
                decode_tokens=1,
                scan_strategy=ScanStrategy.CHUNKED_AFFINE,
                include_embedding=include_embedding,
                include_lm_head=include_lm_head,
                moe_unique_experts=(
                    6 if self.model == ModelFamily.NEMOTRON3 and phase == InferencePhase.DECODE else None
                ),
            )
            report = self._build_workload(scenario)
            reports.append(report)
            total_kv = report.total_traffic.kv_read_bytes
            for stage in report.stages:
                instances.append(
                    _StageInstance(
                        token_index,
                        step_context,
                        phase,
                        scenario.tokens,
                        stage,
                        total_kv,
                    )
                )
        return instances, reports

    def _build_workload(self, scenario: WorkloadScenario) -> WorkloadReport:
        if self.model == ModelFamily.KIMI_K3:
            assert self._kimi_arch is not None
            return KimiK3HybridWorkloadModel(
                self._kimi_arch,
                activation_precision=self.precision.activation,
                weight_precision=self.precision.weight,
                state_precision=self.precision.state,
                conv_state_precision=self.precision.resolved_conv_state(self.model),
            ).build(scenario)
        assert self._nemotron_config is not None
        policy = None
        if self.precision.weight == Precision.NVFP4 and self.precision.use_formal_nemotron_weight_map:
            policy = formal_nemotron_nvfp4_weight_policy(
                self._nemotron_config.arch,
                self._campaign["nemotron"]["checkpoint_quantization"],
            )
        return Nemotron3WorkloadModel(
            self._nemotron_config.arch,
            activation_precision=self.precision.activation,
            weight_precision=self.precision.weight,
            state_precision=self.precision.state,
            weight_precision_policy=policy,
        ).build(scenario)

    def _state_residency(self, warm_start: bool) -> _StateResidency:
        if self.model == ModelFamily.NEMOTRON3:
            assert self._nemotron_config is not None and self._nemotron_config.arch.mamba is not None
            arch = self._nemotron_config.arch
            mamba = arch.mamba
            layers = tuple(index for index, kind in enumerate(arch.layer_types) if kind == "mamba")
            entry = storage_bytes(mamba.state_elements, self.precision.state) + storage_bytes(
                mamba.conv_channels * mamba.conv_kernel,
                self.precision.resolved_conv_state(self.model),
            )
        else:
            assert self._kimi_arch is not None
            layers = tuple(layer - 1 for layer in self._kimi_arch.kda_layer_numbers)
            entry = self._kimi_arch.recurrent_state_bytes(self.precision.state) // len(layers)
            entry += self._kimi_arch.conv_state_bytes(self.precision.resolved_conv_state(self.model)) // len(layers)
        return _StateResidency(
            layer_ids=layers,
            entry_bytes=entry,
            design=self.design,
            warm_start=warm_start,
        )

    def _moe_cache(self, phase: InferencePhase) -> _MoeWeightCache:
        if self.model == ModelFamily.NEMOTRON3:
            assert self._nemotron_config is not None and self._nemotron_config.arch.moe is not None
            arch = self._nemotron_config.arch
            precision = self.precision.weight
            entry = storage_bytes(2 * arch.hidden_size * arch.moe.intermediate_size, precision)
            source = "exact_nemotron_127_step_top6" if phase == InferencePhase.DECODE else "nemotron_prefill_active_set"
            cache = _MoeWeightCache(self.design.moe_weight_cache_bytes, entry, source)
            if phase == InferencePhase.DECODE and cache.capacity_entries and self.routing_trace is not None:
                moe_layers = [index for index, kind in enumerate(arch.layer_types) if kind == "moe"]
                entries = [
                    (("routed", layer, expert), entry)
                    for layer, experts in zip(
                        moe_layers,
                        self.routing_trace["prefill_active_experts_by_layer"],
                        strict=True,
                    )
                    for expert in sorted(experts)
                ]
                cache.seed(entries)
            return cache
        assert self._kimi_arch is not None
        entry = storage_bytes(
            3 * self._kimi_arch.routed_expert_hidden_size * self._kimi_arch.moe_intermediate_size,
            self.precision.weight,
        )
        return _MoeWeightCache(
            self.design.moe_weight_cache_bytes,
            entry,
            (
                "empirical_kimi_top16"
                if self.kimi_routing_trace is not None
                else "deterministic_kimi_top16_sensitivity_not_empirical"
            ),
        )

    def _prepare_stage(
        self,
        instance: _StageInstance,
        state: _StateResidency,
        kv: _KvTracker,
        moe: _MoeWeightCache,
    ) -> StageWork:
        stage = instance.stage
        traffic = self._apply_activation_residency(instance, stage)
        if traffic.state_read_bytes or traffic.state_write_bytes:
            if state.is_resident(instance.token_index, stage.layer_id):
                traffic = replace(traffic, state_read_bytes=0, state_write_bytes=0)
        traffic = kv.apply(traffic, total_history_bytes=instance.total_kv_history_bytes)
        traffic = self._apply_moe_cache(instance, stage, traffic, moe)
        return replace(stage, traffic=traffic)

    def _apply_activation_residency(
        self,
        instance: _StageInstance,
        stage: StageWork,
    ) -> Traffic:
        traffic = stage.traffic
        if not self.design.fused_layer_dataflow:
            return traffic
        hidden_size = (
            self._nemotron_config.arch.hidden_size
            if self.model == ModelFamily.NEMOTRON3 and self._nemotron_config is not None
            else self._kimi_arch.hidden_size
            if self._kimi_arch is not None
            else 0
        )
        tile_tokens = min(instance.workload_tokens, self.design.activation_tile_tokens)
        double_buffer_bytes = 2 * storage_bytes(
            tile_tokens * hidden_size,
            self.precision.activation,
        )
        if double_buffer_bytes > self.design.activation_sram_bytes:
            return traffic
        # The full-model schedule keeps hidden/residual tiles between adjacent
        # stages. Embedding writes into this SRAM and layer outputs stay there;
        # only final logits are materialized back to HBM.
        activation_write = traffic.activation_write_bytes if stage.name == "lm_head" else 0
        return replace(
            traffic,
            activation_read_bytes=0,
            activation_write_bytes=activation_write,
        )

    def _apply_moe_cache(
        self,
        instance: _StageInstance,
        stage: StageWork,
        traffic: Traffic,
        cache: _MoeWeightCache,
    ) -> Traffic:
        routed_names = {"moe_routed_experts", "latent_moe_routed_experts"}
        shared_names = {"moe_shared_expert", "latent_moe_shared_experts"}
        if stage.name in routed_names:
            experts = self._experts(instance, stage.layer_id)
            if not experts:
                return traffic
            entries = tuple((("routed", stage.layer_id, expert), cache.entry_bytes) for expert in experts)
            return replace(traffic, weight_read_bytes=cache.access(entries))
        if stage.name in shared_names:
            count = 1
            if self.model == ModelFamily.KIMI_K3 and self._kimi_arch is not None:
                count = self._kimi_arch.shared_experts
            entry_bytes = traffic.weight_read_bytes // count
            entries = tuple((("shared", stage.layer_id, expert), entry_bytes) for expert in range(count))
            return replace(traffic, weight_read_bytes=cache.access(entries))
        return traffic

    def _experts(self, instance: _StageInstance, layer_id: int) -> tuple[int, ...]:
        if self.model == ModelFamily.NEMOTRON3:
            assert self._nemotron_config is not None and self.routing_trace is not None
            moe_layers = [index for index, kind in enumerate(self._nemotron_config.arch.layer_types) if kind == "moe"]
            layer_index = moe_layers.index(layer_id)
            if instance.phase == InferencePhase.PREFILL:
                return tuple(sorted(self.routing_trace["prefill_active_experts_by_layer"][layer_index]))
            if instance.token_index >= len(self.routing_trace["decode_topk_by_step"]):
                raise ValueError("exact Nemotron routing trace contains only 127 recurrent decode steps")
            return tuple(self.routing_trace["decode_topk_by_step"][instance.token_index][layer_index])
        assert self._kimi_arch is not None
        if self.kimi_routing_trace is not None:
            return self.kimi_routing_trace.experts(
                instance.phase,
                instance.token_index,
                layer_id,
            )
        unique_experts = min(
            self._kimi_arch.num_experts,
            instance.workload_tokens * self._kimi_arch.experts_per_token,
        )
        return tuple(
            (layer_id * 131 + instance.token_index * 17 + rank * 53) % self._kimi_arch.num_experts
            for rank in range(unique_experts)
        )

    def _schedule_stage(
        self,
        index: int,
        instance: _StageInstance,
        stage: StageWork,
        scheduler: _ResourceScheduler,
        dependency: int,
    ) -> TimelineStage:
        spans: list[ResourceSpan] = []
        read = scheduler.transfer(dependency, stage.traffic.logical_hbm_read_bytes, "read")
        if read is not None:
            spans.append(read)
        sram_read = self._reserve_bytes(
            scheduler,
            Resource.SRAM,
            dependency,
            stage.traffic.on_chip_read_bytes,
            "read",
        )
        if sram_read is not None:
            spans.append(sram_read)
        bank_service, bank_stall = self._bank_cycles(instance, stage)
        layout = scheduler.reserve(Resource.LAYOUT, dependency, bank_service, "consumer_bank_service")
        if layout is not None:
            spans.append(layout)
        ready = max(
            dependency,
            read.end_cycle if read is not None else 0,
            sram_read.end_cycle if sram_read is not None else 0,
            layout.end_cycle if layout is not None else 0,
        )
        effective_macs = self._effective_macs(stage)
        for resource, kind, cycles in self._compute_services(stage, effective_macs):
            span = scheduler.reserve(resource, ready, cycles, kind)
            if span is not None:
                spans.append(span)
                ready = span.end_cycle
        compute_end = ready
        sram_write = self._reserve_bytes(
            scheduler,
            Resource.SRAM,
            compute_end,
            stage.traffic.on_chip_write_bytes,
            "write",
        )
        if sram_write is not None:
            spans.append(sram_write)
        write = scheduler.transfer(compute_end, stage.traffic.logical_hbm_write_bytes, "writeback")
        if write is not None:
            spans.append(write)
        data_ready = compute_end
        if sram_write is not None:
            data_ready = max(data_ready, sram_write.end_cycle)
        if stage.traffic.activation_write_bytes and write is not None:
            data_ready = max(data_ready, write.end_cycle)
        retire = max(
            compute_end,
            sram_write.end_cycle if sram_write is not None else 0,
            write.end_cycle if write is not None else 0,
        )
        return TimelineStage(
            index=index,
            token_index=instance.token_index,
            context_length=instance.context_length,
            layer_id=stage.layer_id,
            layer_type=stage.layer_type,
            name=stage.name,
            release_cycle=dependency,
            compute_end_cycle=compute_end,
            data_ready_cycle=data_ready,
            retire_cycle=retire,
            effective_macs=effective_macs,
            logical_hbm_read_bytes=stage.traffic.logical_hbm_read_bytes,
            logical_hbm_write_bytes=stage.traffic.logical_hbm_write_bytes,
            physical_hbm_read_bytes=read.physical_bytes if read is not None else 0,
            physical_hbm_write_bytes=write.physical_bytes if write is not None else 0,
            bank_service_cycles=bank_service,
            bank_stall_cycles=bank_stall,
            spans=tuple(spans),
        )

    def _reserve_bytes(
        self,
        scheduler: _ResourceScheduler,
        resource: Resource,
        release: int,
        byte_count: int,
        kind: str,
    ) -> ResourceSpan | None:
        cycles = math.ceil(byte_count / self.design.sram_bytes_per_cycle)
        return scheduler.reserve(resource, release, cycles, kind, logical_bytes=byte_count)

    def _effective_macs(self, stage: StageWork) -> int:
        if stage.name == "mamba_chunk_intra_cb" and self.design.bc_broadcast:
            assert self._nemotron_config is not None and self._nemotron_config.arch.mamba is not None
            return stage.macs // self._nemotron_config.arch.mamba.heads_per_group
        return stage.macs

    def _compute_services(self, stage: StageWork, effective_macs: int) -> tuple[tuple[Resource, str, int], ...]:
        services: list[tuple[Resource, str, int]] = []
        if stage.exp_ops:
            services.append((Resource.EXP, "exp", math.ceil(stage.exp_ops / self.design.exp_ops_per_cycle)))
        if effective_macs:
            if stage.resource == "state":
                resource = Resource.STATE
                throughput = self.design.state_macs_per_cycle
            elif stage.resource == "conv":
                resource = Resource.CONV
                throughput = self.design.conv_macs_per_cycle
            else:
                resource = Resource.MATRIX
                throughput = self.design.matrix_macs_per_cycle
            services.append((resource, "mac", math.ceil(effective_macs / throughput)))
        if stage.elementwise_ops:
            services.append(
                (
                    Resource.VECTOR,
                    "elementwise",
                    math.ceil(stage.elementwise_ops / self.design.vector_ops_per_cycle),
                )
            )
        if stage.scan_compositions:
            services.append(
                (
                    Resource.STATE,
                    "scan_compose",
                    math.ceil(stage.scan_compositions / self.design.scan_compositions_per_cycle),
                )
            )
        return tuple(services)

    def _bank_cycles(self, instance: _StageInstance, stage: StageWork) -> tuple[int, int]:
        tokens = instance.workload_tokens
        if self.model == ModelFamily.NEMOTRON3 and stage.layer_type == "mamba":
            if self._mamba_bank_stats is None:
                assert self._nemotron_config is not None
                stats = ProjectionBankModel(
                    self._nemotron_config.arch,
                    self.design.projection_design(),
                ).simulate_one_token_request_layer()
                self._mamba_bank_stats = (
                    stats.state_input.service_cycles,
                    stats.state_input.stall_cycles,
                    stats.gate.service_cycles,
                    stats.gate.stall_cycles,
                )
            state_service, state_stall, gate_service, gate_stall = self._mamba_bank_stats
            if stage.name in {"mamba_state_update", "mamba_chunk_state_build"}:
                return state_service * tokens, state_stall * tokens
            if stage.name == "mamba_gate_group_rms_norm":
                return gate_service * tokens, gate_stall * tokens
            if stage.name == "mamba_in_projection":
                mamba = self._nemotron_config.arch.mamba
                assert mamba is not None
                if tokens not in self._mamba_projection_fifo_stall:
                    producer = math.ceil(
                        tokens
                        * self._nemotron_config.arch.hidden_size
                        * mamba.projection_size
                        / self.design.matrix_macs_per_cycle
                    )
                    write = ProjectionWriteBufferModel(self.design.projection_design()).simulate(
                        values=tokens * mamba.projection_size,
                        producer_cycles=producer,
                        values_per_token=mamba.projection_size,
                        forced_spill_values_per_token=mamba.d_inner,
                        activation_bytes=storage_bytes(1, self.precision.activation),
                    )
                    self._mamba_projection_fifo_stall[tokens] = write.fifo_stall_cycles
                stall = self._mamba_projection_fifo_stall[tokens]
                return stall, stall
            return 0, 0
        if self.model == ModelFamily.KIMI_K3 and stage.name in {
            "kda_state_decay_prediction",
            "kda_chunk_prepare",
        }:
            optimized = self.design.projection_layout == ProjectionLayout.GROUP_MAJOR_SKEWED
            service, stall = self._kda_bank_cycles(optimized)
            return service * tokens, stall * tokens
        return 0, 0

    def _kda_bank_cycles(self, optimized: bool) -> tuple[int, int]:
        key = (
            optimized,
            self.design.projection_buffer_banks,
            self.design.projection_buffer_ports_per_bank,
            self.design.state_dim_lanes,
        )
        if key in self._kda_bank_cache:
            return self._kda_bank_cache[key]
        assert self._kimi_arch is not None
        best = _kda_bank_stats(
            optimized,
            self.design.projection_buffer_banks,
            self.design.projection_buffer_ports_per_bank,
            self.design.state_dim_lanes,
            self.design.head_dim_lanes,
            self._kimi_arch.kda.num_heads,
            self._kimi_arch.kda.key_dim,
            self._kimi_arch.kda.value_dim,
        )
        self._kda_bank_cache[key] = best
        return best

    def _gpu_evidence(self, phase: InferencePhase, reports: list[WorkloadReport]) -> dict[str, Any]:
        if self.model == ModelFamily.KIMI_K3:
            cases = {record["case"]: record for record in self._campaign["kda"]["cases"]}
            case = "prefill_b1_s2048" if phase == InferencePhase.PREFILL else "decode_b1"
            return {
                "status": "workload_validation_only_not_cycle_fit",
                "source": "formal B200 KDA campaign",
                "case": case,
                "matrix_path_time_fraction": cases[case]["matrix_path_time_fraction"],
                "state_core_time_fraction": cases[case]["state_core_time_fraction"],
                "observed_recurrent_state": "fp32",
            }
        nemotron = self._campaign["nemotron"]
        if phase == InferencePhase.DECODE:
            gpu = nemotron["latency"]["decode_s2048_128"]
            case = "decode"
        else:
            gpu = nemotron["latency"].get("prefill_s2048", nemotron["latency"])
            case = "prefill"
        logical = reports[0].total_traffic
        return {
            "status": "workload_validation_only_not_cycle_fit",
            "source": "formal B200 full Nemotron campaign",
            "case": case,
            "gpu_latency": gpu,
            "logical_hbm_read_bytes_per_modeled_pass": logical.logical_hbm_read_bytes,
            "checkpoint_weight_default": nemotron["checkpoint_quantization"]["default_linear_weight"],
            "observed_mamba_state": "fp32",
        }


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple if value else 0


@cache
def _formal_campaign() -> dict[str, Any]:
    return build_b200_campaign()


@cache
def _kda_bank_stats(
    optimized: bool,
    banks: int,
    ports: int,
    key_tile: int,
    value_tile: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[int, int]:
    """Return one-layer KDA packet service without repeating identical heads."""

    def packet_cycles(packet: list[int]) -> tuple[int, int]:
        counts = [0] * banks
        for bank in packet:
            counts[bank] += 1
        ideal = math.ceil(len(packet) / (banks * ports))
        service = max(math.ceil(count / ports) for count in counts)
        return ideal, service

    value_ideal = value_service = 0
    for start in range(0, value_dim, value_tile):
        packet = [lane % banks for lane in range(start, min(start + value_tile, value_dim))]
        ideal, service = packet_cycles(packet)
        value_ideal += ideal
        value_service += service

    rotations = product(range(banks), repeat=2) if optimized else ((0, 0),)
    best: tuple[int, int] | None = None
    for k_rotation, decay_rotation in rotations:
        key_ideal = key_service = 0
        for start in range(0, key_dim, key_tile):
            lanes = range(start, min(start + key_tile, key_dim))
            packet = [
                *(lane % banks for lane in lanes),
                *((lane + k_rotation) % banks for lane in lanes),
                *((lane + decay_rotation) % banks for lane in lanes),
            ]
            ideal, service = packet_cycles(packet)
            key_ideal += ideal
            key_service += service
        ideal = heads * (1 + key_ideal + value_ideal)
        service = heads * (1 + key_service + value_service)
        candidate = (service, service - ideal)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


__all__ = [
    "HybridSystemTimelineModel",
    "ModelFamily",
    "PrecisionConfig",
    "Resource",
    "SystemDesign",
    "SystemTimelineReport",
]
