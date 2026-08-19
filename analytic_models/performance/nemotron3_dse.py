"""Hardware timing, bank-layout, and persistent-state DSE for Nemotron 3."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product

from transactional_emulator.testbench.model_configs.loader import ModelArchConfig

from .nemotron3_layout import GroupMajorSkewedLayout, ProjectionGeometry
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    StageWork,
    WorkloadReport,
    WorkloadScenario,
    WeightPrecisionPolicy,
    storage_bytes,
)
from .projection_scatter import (
    ProjectionFifoSpillModel,
    ProjectionFlow,
    StreamRun,
)


class ProjectionLayout(StrEnum):
    ROW_MAJOR = "row_major"
    TRANSPOSED = "transposed"
    GROUP_MAJOR = "group_major"
    GROUP_MAJOR_SKEWED = "group_major_skewed"


class StateCachePolicy(StrEnum):
    NONE = "none"
    LRU = "lru"
    PINNED = "pinned"


@dataclass(frozen=True)
class HardwareDesign:
    name: str = "plena_mamba_candidate"
    frequency_hz: int = 1_000_000_000
    matrix_macs_per_cycle: int = 4096
    vector_ops_per_cycle: int = 256
    conv_macs_per_cycle: int = 256
    exp_ops_per_cycle: int = 16
    hbm_bytes_per_cycle: int = 64
    projection_buffer_banks: int = 16
    projection_buffer_ports_per_bank: int = 1
    matrix_result_burst_values: int = 64
    projection_buffer_write_values_per_cycle: int = 16
    projection_fifo_values: int = 64
    projection_direct_bypass: bool = True
    projection_consumer_start_cycles: int = 0
    projection_consume_values_per_cycle: int = 16
    head_lanes: int = 8
    head_dim_lanes: int = 4
    state_dim_lanes: int = 8
    projection_layout: ProjectionLayout = ProjectionLayout.ROW_MAJOR
    bc_broadcast: bool = False
    state_cache_bytes: int = 0
    state_cache_policy: StateCachePolicy = StateCachePolicy.NONE
    calibrated: bool = False

    def __post_init__(self) -> None:
        positive = (
            "frequency_hz",
            "matrix_macs_per_cycle",
            "vector_ops_per_cycle",
            "conv_macs_per_cycle",
            "exp_ops_per_cycle",
            "hbm_bytes_per_cycle",
            "projection_buffer_banks",
            "projection_buffer_ports_per_bank",
            "matrix_result_burst_values",
            "projection_buffer_write_values_per_cycle",
            "projection_fifo_values",
            "projection_consume_values_per_cycle",
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
        )
        for field_name in positive:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.state_cache_bytes < 0:
            raise ValueError("state_cache_bytes must be non-negative")
        if self.projection_consumer_start_cycles < 0:
            raise ValueError("projection_consumer_start_cycles must be non-negative")
        if self.state_cache_bytes == 0 and self.state_cache_policy != StateCachePolicy.NONE:
            raise ValueError("zero-sized state cache requires policy=none")
        if self.state_cache_bytes > 0 and self.state_cache_policy == StateCachePolicy.NONE:
            raise ValueError("non-zero state cache requires lru or pinned policy")

    @property
    def state_macs_per_cycle(self) -> int:
        return self.head_lanes * self.head_dim_lanes * self.state_dim_lanes


@dataclass(frozen=True)
class StateCacheStats:
    policy: StateCachePolicy
    capacity_bytes: int
    entry_bytes: int
    resident_entries: int
    accesses: int
    hits: int
    misses: int
    evictions: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    final_flush_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hit_rate"] = self.hit_rate
        return result


class PersistentStateCacheModel:
    """Simulate layer-major decode state ownership at (request, layer) granularity."""

    def __init__(
        self,
        arch: ModelArchConfig,
        precision: Precision,
        capacity_bytes: int,
        policy: StateCachePolicy,
    ) -> None:
        mamba = arch.mamba
        if mamba is None:
            raise ValueError("state cache requires Mamba architecture")
        self.mamba_layer_ids = [idx for idx, layer_type in enumerate(arch.layer_types) if layer_type == "mamba"]
        self.entry_bytes = storage_bytes(mamba.state_elements, precision) + storage_bytes(
            mamba.conv_channels * mamba.conv_kernel, precision
        )
        self.capacity_bytes = capacity_bytes
        self.policy = policy

    def simulate(self, *, batch_size: int, decode_tokens: int, flush_at_end: bool = True) -> StateCacheStats:
        one_step = [(request_id, layer_id) for layer_id in self.mamba_layer_ids for request_id in range(batch_size)]
        accesses = len(one_step) * decode_tokens
        capacity_entries = self.capacity_bytes // self.entry_bytes if self.entry_bytes else 0

        if self.policy == StateCachePolicy.NONE or capacity_entries == 0:
            traffic = accesses * self.entry_bytes
            return StateCacheStats(
                policy=StateCachePolicy.NONE,
                capacity_bytes=self.capacity_bytes,
                entry_bytes=self.entry_bytes,
                resident_entries=0,
                accesses=accesses,
                hits=0,
                misses=accesses,
                evictions=0,
                hbm_read_bytes=traffic,
                hbm_write_bytes=traffic,
                final_flush_bytes=0,
            )

        capacity_entries = min(capacity_entries, len(one_step))
        if self.policy == StateCachePolicy.PINNED:
            return self._simulate_pinned(one_step, decode_tokens, capacity_entries, flush_at_end)
        return self._simulate_lru(one_step, decode_tokens, capacity_entries, flush_at_end)

    def _simulate_pinned(
        self,
        one_step: list[tuple[int, int]],
        decode_tokens: int,
        capacity_entries: int,
        flush_at_end: bool,
    ) -> StateCacheStats:
        # The compiler/runtime pins a deterministic subset during prefill. All
        # non-pinned states use direct read-update-write streaming.
        pinned = set(one_step[:capacity_entries])
        hits = 0
        misses = 0
        reads = 0
        writes = 0
        for _ in range(decode_tokens):
            for key in one_step:
                if key in pinned:
                    hits += 1
                else:
                    misses += 1
                    reads += self.entry_bytes
                    writes += self.entry_bytes
        flush = len(pinned) * self.entry_bytes if flush_at_end and decode_tokens else 0
        writes += flush
        return StateCacheStats(
            policy=self.policy,
            capacity_bytes=self.capacity_bytes,
            entry_bytes=self.entry_bytes,
            resident_entries=len(pinned),
            accesses=len(one_step) * decode_tokens,
            hits=hits,
            misses=misses,
            evictions=0,
            hbm_read_bytes=reads,
            hbm_write_bytes=writes,
            final_flush_bytes=flush,
        )

    def _simulate_lru(
        self,
        one_step: list[tuple[int, int]],
        decode_tokens: int,
        capacity_entries: int,
        flush_at_end: bool,
    ) -> StateCacheStats:
        # Prefill leaves the most recently visited tail of the hybrid network in
        # cache. Entries are initially clean because prefill has committed state.
        cache: OrderedDict[tuple[int, int], bool] = OrderedDict((key, False) for key in one_step[-capacity_entries:])
        hits = misses = evictions = reads = writes = 0
        for _ in range(decode_tokens):
            for key in one_step:
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                    cache[key] = True
                    continue
                misses += 1
                reads += self.entry_bytes
                if len(cache) == capacity_entries:
                    _, dirty = cache.popitem(last=False)
                    evictions += 1
                    if dirty:
                        writes += self.entry_bytes
                cache[key] = True

        dirty_entries = sum(cache.values())
        flush = dirty_entries * self.entry_bytes if flush_at_end else 0
        writes += flush
        return StateCacheStats(
            policy=self.policy,
            capacity_bytes=self.capacity_bytes,
            entry_bytes=self.entry_bytes,
            resident_entries=len(cache),
            accesses=len(one_step) * decode_tokens,
            hits=hits,
            misses=misses,
            evictions=evictions,
            hbm_read_bytes=reads,
            hbm_write_bytes=writes,
            final_flush_bytes=flush,
        )


@dataclass(frozen=True)
class BankStageStats:
    packets: int
    value_reads: int
    ideal_cycles: int
    service_cycles: int

    @property
    def stall_cycles(self) -> int:
        return self.service_cycles - self.ideal_cycles

    def __add__(self, other: BankStageStats) -> BankStageStats:
        return BankStageStats(
            packets=self.packets + other.packets,
            value_reads=self.value_reads + other.value_reads,
            ideal_cycles=self.ideal_cycles + other.ideal_cycles,
            service_cycles=self.service_cycles + other.service_cycles,
        )

    def scaled(self, factor: int) -> BankStageStats:
        return BankStageStats(
            packets=self.packets * factor,
            value_reads=self.value_reads * factor,
            ideal_cycles=self.ideal_cycles * factor,
            service_cycles=self.service_cycles * factor,
        )

    def scaled_fraction(self, factor: float) -> BankStageStats:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("bank traffic fraction must be between zero and one")
        return BankStageStats(
            packets=round(self.packets * factor),
            value_reads=round(self.value_reads * factor),
            ideal_cycles=round(self.ideal_cycles * factor),
            service_cycles=round(self.service_cycles * factor),
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["stall_cycles"] = self.stall_cycles
        return result


@dataclass(frozen=True)
class MatrixBankStats:
    state_input: BankStageStats
    gate: BankStageStats
    bc_value_reads: int

    @property
    def total(self) -> BankStageStats:
        return self.state_input + self.gate

    def scaled(self, factor: int) -> MatrixBankStats:
        return MatrixBankStats(
            state_input=self.state_input.scaled(factor),
            gate=self.gate.scaled(factor),
            bc_value_reads=self.bc_value_reads * factor,
        )

    def to_dict(self) -> dict:
        return {
            "state_input": self.state_input.to_dict(),
            "gate": self.gate.to_dict(),
            "total": self.total.to_dict(),
            "bc_value_reads": self.bc_value_reads,
        }


@dataclass(frozen=True)
class ProjectionWriteStats:
    values: int
    bursts: int
    producer_cycles: int
    minimum_write_cycles: int
    completion_cycles: int
    fifo_stall_cycles: int
    fifo_high_watermark: int
    direct_values: int
    spill_values: int
    spill_bytes: int


class ProjectionWriteBufferModel:
    """Model Matrix-result bursts draining through the L-compute scatter FIFO."""

    def __init__(self, design: HardwareDesign) -> None:
        self.design = design

    def simulate(
        self,
        *,
        values: int,
        producer_cycles: int,
        values_per_token: int | None = None,
        forced_spill_values_per_token: int = 0,
        activation_bytes: int = 2,
    ) -> ProjectionWriteStats:
        if values < 0 or producer_cycles < 0:
            raise ValueError("values and producer_cycles must be non-negative")
        if values == 0:
            return ProjectionWriteStats(0, 0, producer_cycles, 0, producer_cycles, 0, 0, 0, 0, 0)
        if producer_cycles <= 0:
            raise ValueError("producer_cycles must be positive for a non-empty projection")
        values_per_token = values if values_per_token is None else values_per_token
        if values_per_token <= 0 or values % values_per_token:
            raise ValueError("values must contain a whole number of projection tokens")
        if not 0 <= forced_spill_values_per_token <= values_per_token:
            raise ValueError("forced spill values must fit within one projection token")
        stream_runs = []
        for token in range(values // values_per_token):
            start = token * values_per_token
            if forced_spill_values_per_token:
                stream_runs.append(StreamRun(start, forced_spill_values_per_token, True))
            direct_values = values_per_token - forced_spill_values_per_token
            if direct_values:
                stream_runs.append(
                    StreamRun(start + forced_spill_values_per_token, direct_values, False)
                )
        flow = (
            ProjectionFlow.FIFO_WITH_SPILL
            if self.design.projection_direct_bypass
            else ProjectionFlow.BUFFERED
        )
        stats = ProjectionFifoSpillModel(
            flow=flow,
            fifo_capacity_values=self.design.projection_fifo_values,
            producer_burst_values=self.design.matrix_result_burst_values,
            spill_write_values_per_cycle=self.design.projection_buffer_write_values_per_cycle,
            consumer_start_cycle=self.design.projection_consumer_start_cycles,
            consumer_values_per_cycle=self.design.projection_consume_values_per_cycle,
            activation_bytes=activation_bytes,
        ).simulate(tuple(stream_runs), producer_cycles=producer_cycles)
        return ProjectionWriteStats(
            values=values,
            bursts=stats.bursts,
            producer_cycles=producer_cycles,
            minimum_write_cycles=math.ceil(
                stats.spill_values / self.design.projection_buffer_write_values_per_cycle
            ),
            completion_cycles=stats.completion_cycles,
            fifo_stall_cycles=stats.fifo_stall_cycles,
            fifo_high_watermark=stats.fifo_high_watermark,
            direct_values=stats.direct_values,
            spill_values=stats.spill_values,
            spill_bytes=stats.spill_bytes,
        )


class ProjectionBankModel:
    """Count bank-port service cycles for projected Mamba intermediates."""

    def __init__(self, arch: ModelArchConfig, design: HardwareDesign) -> None:
        if arch.mamba is None:
            raise ValueError("bank model requires Mamba architecture")
        self.mamba = arch.mamba
        self.design = design
        self.skewed_layout = GroupMajorSkewedLayout(
            ProjectionGeometry(
                num_heads=self.mamba.num_heads,
                head_dim=self.mamba.head_dim,
                state_dim=self.mamba.state_dim,
                groups=self.mamba.groups,
                banks=design.projection_buffer_banks,
                head_lanes=design.head_lanes,
                head_dim_lanes=design.head_dim_lanes,
                state_dim_lanes=design.state_dim_lanes,
            )
        )

    def simulate_one_token_request_layer(self) -> MatrixBankStats:
        input_packets: list[list[int]] = []
        gate_packets: list[list[int]] = []
        bc_reads = 0
        mamba = self.mamba
        heads_per_group = mamba.heads_per_group

        for group in range(mamba.groups):
            for local_head_start in range(0, heads_per_group, self.design.head_lanes):
                local_heads = range(local_head_start, min(local_head_start + self.design.head_lanes, heads_per_group))
                dt_packet = [self._bank("dt", group, head, 0) for head in local_heads]
                input_packets.append(dt_packet)
                for dim_start in range(0, mamba.head_dim, self.design.head_dim_lanes):
                    dims = range(dim_start, min(dim_start + self.design.head_dim_lanes, mamba.head_dim))
                    input_packets.append([self._bank("x", group, head, dim) for head in local_heads for dim in dims])
                    gate_packets.append([self._bank("gate", group, head, dim) for head in local_heads for dim in dims])

            for state_start in range(0, mamba.state_dim, self.design.state_dim_lanes):
                states = range(state_start, min(state_start + self.design.state_dim_lanes, mamba.state_dim))
                repetitions = 1 if self.design.bc_broadcast else heads_per_group
                packet = []
                for _ in range(repetitions):
                    packet.extend(self._bank("b", group, 0, state) for state in states)
                    packet.extend(self._bank("c", group, 0, state) for state in states)
                bc_reads += len(packet)
                input_packets.append(packet)

        return MatrixBankStats(
            state_input=self._summarize(input_packets),
            gate=self._summarize(gate_packets),
            bc_value_reads=bc_reads,
        )

    def _summarize(self, packets: list[list[int]]) -> BankStageStats:
        ideal = service = value_reads = 0
        banks = self.design.projection_buffer_banks
        ports = self.design.projection_buffer_ports_per_bank
        for packet in packets:
            if not packet:
                continue
            counts = [0] * banks
            for bank in packet:
                counts[bank] += 1
            value_reads += len(packet)
            ideal += math.ceil(len(packet) / (banks * ports))
            service += max(math.ceil(count / ports) for count in counts)
        return BankStageStats(len(packets), value_reads, ideal, service)

    def _bank(self, field: str, group: int, local_head: int, lane: int) -> int:
        banks = self.design.projection_buffer_banks
        mamba = self.mamba
        global_head = group * mamba.heads_per_group + local_head
        row_bases = {
            "gate": 0,
            "x": mamba.d_inner,
            "b": 2 * mamba.d_inner,
            "c": 2 * mamba.d_inner + mamba.groups * mamba.state_dim,
            "dt": 2 * mamba.d_inner + 2 * mamba.groups * mamba.state_dim,
        }

        if self.design.projection_layout == ProjectionLayout.ROW_MAJOR:
            if field in {"gate", "x"}:
                address = row_bases[field] + global_head * mamba.head_dim + lane
            elif field in {"b", "c"}:
                address = row_bases[field] + group * mamba.state_dim + lane
            else:
                address = row_bases[field] + global_head
            return address % banks

        if self.design.projection_layout == ProjectionLayout.TRANSPOSED:
            if field in {"gate", "x"}:
                address = row_bases[field] + lane * mamba.num_heads + global_head
            elif field in {"b", "c"}:
                address = row_bases[field] + lane * mamba.groups + group
            else:
                address = row_bases[field] + global_head
            return address % banks

        group_span = 2 * mamba.heads_per_group * mamba.head_dim + 2 * mamba.state_dim + mamba.heads_per_group
        group_base = group * group_span
        group_offsets = {
            "x": 0,
            "gate": mamba.heads_per_group * mamba.head_dim,
            "b": 2 * mamba.heads_per_group * mamba.head_dim,
            "c": 2 * mamba.heads_per_group * mamba.head_dim + mamba.state_dim,
            "dt": 2 * mamba.heads_per_group * mamba.head_dim + 2 * mamba.state_dim,
        }
        if self.design.projection_layout == ProjectionLayout.GROUP_MAJOR:
            if field in {"gate", "x"}:
                address = group_base + group_offsets[field] + local_head * mamba.head_dim + lane
            elif field in {"b", "c"}:
                address = group_base + group_offsets[field] + lane
            else:
                address = group_base + group_offsets[field] + local_head
            return address % banks

        # The executable layout model also checks row ownership and bijection;
        # the DSE only needs the resulting bank number here.
        return self.skewed_layout.address(field, group, local_head, lane).bank


@dataclass(frozen=True)
class StageTiming:
    layer_id: int
    layer_type: str
    name: str
    effective_macs: int
    compute_cycles: int
    hbm_cycles: int
    projection_buffer_cycles: int
    projection_fifo_stall_cycles: int
    total_cycles: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    bank_stall_cycles: int
    projection_direct_values: int = 0
    projection_spill_values: int = 0
    projection_spill_bytes: int = 0
    projection_fifo_high_watermark: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DseResult:
    design: HardwareDesign
    workload: WorkloadReport
    state_cache: StateCacheStats
    matrix_banks: MatrixBankStats
    stages: tuple[StageTiming, ...]

    @property
    def total_cycles(self) -> int:
        return sum(stage.total_cycles for stage in self.stages)

    @property
    def hbm_read_bytes(self) -> int:
        return sum(stage.hbm_read_bytes for stage in self.stages)

    @property
    def hbm_write_bytes(self) -> int:
        return sum(stage.hbm_write_bytes for stage in self.stages)

    @property
    def bank_stall_cycles(self) -> int:
        return sum(stage.bank_stall_cycles for stage in self.stages)

    @property
    def projection_fifo_stall_cycles(self) -> int:
        return sum(stage.projection_fifo_stall_cycles for stage in self.stages)

    @property
    def projection_direct_values(self) -> int:
        return sum(stage.projection_direct_values for stage in self.stages)

    @property
    def projection_spill_values(self) -> int:
        return sum(stage.projection_spill_values for stage in self.stages)

    @property
    def projection_spill_bytes(self) -> int:
        return sum(stage.projection_spill_bytes for stage in self.stages)

    @property
    def projection_fifo_high_watermark(self) -> int:
        return max((stage.projection_fifo_high_watermark for stage in self.stages), default=0)

    @property
    def measured_steps(self) -> int:
        if self.workload.scenario.phase == InferencePhase.DECODE:
            return self.workload.scenario.decode_tokens
        return 1

    @property
    def cycles_per_step(self) -> float:
        return self.total_cycles / self.measured_steps

    @property
    def latency_us_per_step(self) -> float:
        return self.cycles_per_step / self.design.frequency_hz * 1e6

    @property
    def tokens_per_second(self) -> float:
        scenario = self.workload.scenario
        generated = scenario.batch_size * self.measured_steps
        seconds = self.total_cycles / self.design.frequency_hz
        return generated / seconds if seconds else 0.0

    def to_dict(self, include_stages: bool = True) -> dict:
        cycle_breakdown: dict[str, int] = {}
        hbm_breakdown: dict[str, int] = {}
        for stage in self.stages:
            cycle_breakdown[stage.layer_type] = cycle_breakdown.get(stage.layer_type, 0) + stage.total_cycles
            hbm_breakdown[stage.layer_type] = hbm_breakdown.get(stage.layer_type, 0) + (
                stage.hbm_read_bytes + stage.hbm_write_bytes
            )
        design_dict = asdict(self.design)
        design_dict["state_macs_per_cycle"] = self.design.state_macs_per_cycle
        result = {
            "design": design_dict,
            "metrics": {
                "total_cycles": self.total_cycles,
                "cycles_per_step": self.cycles_per_step,
                "latency_us_per_step": self.latency_us_per_step,
                "tokens_per_second": self.tokens_per_second,
                "hbm_read_bytes": self.hbm_read_bytes,
                "hbm_write_bytes": self.hbm_write_bytes,
                "hbm_bytes_per_step": (self.hbm_read_bytes + self.hbm_write_bytes) / self.measured_steps,
                "bank_stall_cycles": self.bank_stall_cycles,
                "bank_stall_cycles_per_step": self.bank_stall_cycles / self.measured_steps,
                "projection_fifo_stall_cycles": self.projection_fifo_stall_cycles,
                "projection_fifo_high_watermark": self.projection_fifo_high_watermark,
                "projection_direct_values": self.projection_direct_values,
                "projection_spill_values": self.projection_spill_values,
                "projection_spill_bytes": self.projection_spill_bytes,
                "state_cache_hit_rate": self.state_cache.hit_rate,
                "calibrated": self.design.calibrated,
                "cycle_breakdown": cycle_breakdown,
                "hbm_byte_breakdown": hbm_breakdown,
            },
            "state_cache": self.state_cache.to_dict(),
            "matrix_banks": self.matrix_banks.to_dict(),
            "workload_totals": self.workload.to_dict()["totals"],
        }
        if include_stages:
            result["stages"] = [stage.to_dict() for stage in self.stages]
        return result


class Nemotron3DseModel:
    def __init__(self, arch: ModelArchConfig) -> None:
        if arch.mamba is None:
            raise ValueError("Nemotron 3 DSE requires Mamba architecture")
        self.arch = arch

    def evaluate(
        self,
        scenario: WorkloadScenario,
        design: HardwareDesign,
        *,
        activation_precision: Precision = Precision.BF16,
        weight_precision: Precision = Precision.BF16,
        state_precision: Precision = Precision.FP32,
        weight_precision_policy: WeightPrecisionPolicy | None = None,
    ) -> DseResult:
        workload = Nemotron3WorkloadModel(
            self.arch,
            activation_precision=activation_precision,
            weight_precision=weight_precision,
            state_precision=state_precision,
            weight_precision_policy=weight_precision_policy,
        ).build(scenario)
        cache_model = PersistentStateCacheModel(
            self.arch,
            state_precision,
            design.state_cache_bytes,
            design.state_cache_policy,
        )
        if scenario.phase == InferencePhase.DECODE:
            cache = cache_model.simulate(
                batch_size=scenario.batch_size,
                decode_tokens=scenario.decode_tokens,
            )
        else:
            state_traffic = workload.total_traffic
            accesses = scenario.batch_size * self.arch.count_layers("mamba")
            resident = min(
                accesses,
                design.state_cache_bytes // cache_model.entry_bytes if cache_model.entry_bytes else 0,
            )
            cache = StateCacheStats(
                policy=design.state_cache_policy,
                capacity_bytes=design.state_cache_bytes,
                entry_bytes=cache_model.entry_bytes,
                resident_entries=resident,
                accesses=accesses,
                hits=0,
                misses=accesses,
                evictions=0,
                hbm_read_bytes=state_traffic.state_read_bytes,
                hbm_write_bytes=state_traffic.state_write_bytes,
                final_flush_bytes=state_traffic.state_write_bytes,
            )
        repeat = scenario.decode_tokens if scenario.phase == InferencePhase.DECODE else 1
        bank_per_layer = ProjectionBankModel(self.arch, design).simulate_one_token_request_layer()
        bank_all = bank_per_layer.scaled(repeat * scenario.tokens * self.arch.count_layers("mamba"))
        stages = self._time_stages(workload, design, cache, bank_per_layer, repeat)
        return DseResult(design, workload, cache, bank_all, tuple(stages))

    def _time_stages(
        self,
        workload: WorkloadReport,
        design: HardwareDesign,
        cache: StateCacheStats,
        bank_per_layer: MatrixBankStats,
        repeat: int,
    ) -> list[StageTiming]:
        baseline_state_read = workload.total_traffic.state_read_bytes
        baseline_state_write = workload.total_traffic.state_write_bytes
        read_scale = cache.hbm_read_bytes / baseline_state_read if baseline_state_read else 0.0
        write_scale = cache.hbm_write_bytes / baseline_state_write if baseline_state_write else 0.0
        stages = []
        scenario = workload.scenario
        state_buffer_fraction_by_layer: dict[int, float] = {}

        for stage in workload.stages:
            effective_macs = self._effective_macs(stage, design)
            compute_cycles = self._compute_cycles(stage, design, effective_macs) * repeat
            non_state_read = stage.traffic.logical_hbm_read_bytes - stage.traffic.state_read_bytes
            non_state_write = stage.traffic.logical_hbm_write_bytes - stage.traffic.state_write_bytes
            hbm_read = non_state_read * repeat + round(stage.traffic.state_read_bytes * read_scale)
            hbm_write = non_state_write * repeat + round(stage.traffic.state_write_bytes * write_scale)
            hbm_cycles = math.ceil((hbm_read + hbm_write) / design.hbm_bytes_per_cycle)

            bank_stats = None
            if stage.name in {"mamba_state_update", "mamba_chunk_state_build"}:
                # State input includes x/dt/B/C. Charge it once per token for
                # either the decode update or the prefill chunk-state build.
                bank_stats = bank_per_layer.state_input.scaled(scenario.tokens * repeat)
                bank_stats = bank_stats.scaled_fraction(
                    state_buffer_fraction_by_layer.get(stage.layer_id, 1.0)
                )
            elif stage.name == "mamba_gate_group_rms_norm":
                bank_stats = bank_per_layer.gate.scaled(scenario.tokens * repeat)
            projection_buffer_cycles = bank_stats.service_cycles if bank_stats is not None else 0
            bank_stall = bank_stats.stall_cycles if bank_stats is not None else 0
            projection_fifo_stall = 0
            projection_direct_values = 0
            projection_spill_values = 0
            projection_spill_bytes = 0
            projection_fifo_high_watermark = 0
            if stage.name == "mamba_in_projection":
                mamba = self.arch.mamba
                assert mamba is not None
                projection_values = scenario.tokens * repeat * mamba.projection_size
                write_stats = ProjectionWriteBufferModel(design).simulate(
                    values=projection_values,
                    producer_cycles=max(compute_cycles, hbm_cycles),
                    values_per_token=mamba.projection_size,
                    forced_spill_values_per_token=mamba.d_inner,
                    activation_bytes=storage_bytes(1, workload.activation_precision),
                )
                projection_buffer_cycles = max(projection_buffer_cycles, write_stats.completion_cycles)
                projection_fifo_stall = write_stats.fifo_stall_cycles
                projection_direct_values = write_stats.direct_values
                projection_spill_values = write_stats.spill_values
                projection_spill_bytes = write_stats.spill_bytes
                projection_fifo_high_watermark = write_stats.fifo_high_watermark
                token_count = scenario.tokens * repeat
                state_values = token_count * (mamba.projection_size - mamba.d_inner)
                spilled_state = max(0, write_stats.spill_values - token_count * mamba.d_inner)
                state_buffer_fraction_by_layer[stage.layer_id] = (
                    spilled_state / state_values if state_values else 0.0
                )
            total_cycles = max(compute_cycles, hbm_cycles, projection_buffer_cycles)
            stages.append(
                StageTiming(
                    layer_id=stage.layer_id,
                    layer_type=stage.layer_type,
                    name=stage.name,
                    effective_macs=effective_macs * repeat,
                    compute_cycles=compute_cycles,
                    hbm_cycles=hbm_cycles,
                    projection_buffer_cycles=projection_buffer_cycles,
                    projection_fifo_stall_cycles=projection_fifo_stall,
                    total_cycles=total_cycles,
                    hbm_read_bytes=hbm_read,
                    hbm_write_bytes=hbm_write,
                    bank_stall_cycles=bank_stall,
                    projection_direct_values=projection_direct_values,
                    projection_spill_values=projection_spill_values,
                    projection_spill_bytes=projection_spill_bytes,
                    projection_fifo_high_watermark=projection_fifo_high_watermark,
                )
            )
        return stages

    def _effective_macs(self, stage: StageWork, design: HardwareDesign) -> int:
        if stage.name == "mamba_chunk_intra_cb" and design.bc_broadcast:
            mamba = self.arch.mamba
            assert mamba is not None
            return stage.macs // mamba.heads_per_group
        return stage.macs

    @staticmethod
    def _compute_cycles(stage: StageWork, design: HardwareDesign, effective_macs: int) -> int:
        if stage.resource == "state":
            mac_throughput = design.state_macs_per_cycle
        elif stage.resource == "conv":
            mac_throughput = design.conv_macs_per_cycle
        else:
            mac_throughput = design.matrix_macs_per_cycle
        mac_cycles = math.ceil(effective_macs / mac_throughput)
        vector_cycles = math.ceil(stage.elementwise_ops / design.vector_ops_per_cycle)
        exp_cycles = math.ceil(stage.exp_ops / design.exp_ops_per_cycle)
        # These terms are kept separate in the report and summed conservatively
        # until RTL counters establish which operations are truly fused.
        return mac_cycles + vector_cycles + exp_cycles


def sweep_designs(
    base: HardwareDesign,
    *,
    layouts: tuple[ProjectionLayout, ...],
    broadcasts: tuple[bool, ...],
    cache_sizes: tuple[int, ...],
    cache_policies: tuple[StateCachePolicy, ...],
    state_dim_lanes: tuple[int, ...],
    bypasses: tuple[bool, ...] = (True,),
    fifo_values: tuple[int, ...] | None = None,
) -> list[HardwareDesign]:
    designs = []
    fifo_values = fifo_values or (base.projection_fifo_values,)
    for layout, broadcast, cache_size, policy, n_lanes, bypass, fifo_capacity in product(
        layouts,
        broadcasts,
        cache_sizes,
        cache_policies,
        state_dim_lanes,
        bypasses,
        fifo_values,
    ):
        if cache_size == 0 and policy != StateCachePolicy.NONE:
            continue
        if cache_size > 0 and policy == StateCachePolicy.NONE:
            continue
        designs.append(
            HardwareDesign(
                **{
                    **asdict(base),
                    "name": (
                        f"{layout}-bc{int(broadcast)}-bypass{int(bypass)}-fifo{fifo_capacity}"
                        f"-cache{cache_size // (1024 * 1024)}m-{policy}-n{n_lanes}"
                    ),
                    "projection_layout": layout,
                    "bc_broadcast": broadcast,
                    "state_cache_bytes": cache_size,
                    "state_cache_policy": policy,
                    "state_dim_lanes": n_lanes,
                    "projection_direct_bypass": bypass,
                    "projection_fifo_values": fifo_capacity,
                }
            )
        )
    return designs
