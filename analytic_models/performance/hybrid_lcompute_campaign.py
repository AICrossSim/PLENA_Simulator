"""Unified pre-RTL A-G campaign for hybrid L-Compute.

This module joins four kinds of evidence without conflating them:

* official Nemotron/Kimi architecture work and logical traffic;
* Compiler-emitted dynamic issue streams;
* executable bank/FIFO service models;
* measured GPU evidence used only for shape and bottleneck cross-checks.

The result is a deterministic Compiler/Simulator estimate. It is not RTL PPA,
silicon frequency, or a real-checkpoint numerical execution.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .gpu_evidence import build_report as build_gpu_report
from .hybrid_arch import load_nemotron3_arch
from .kimi_k3_workload import (
    KimiK3Architecture,
    KimiK3HybridWorkloadModel,
    formal_kimi_k3_mxfp4_weight_policy,
)
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    StageWork,
    WorkloadReport,
    WorkloadScenario,
    formal_nemotron_nvfp4_weight_policy,
    storage_bytes,
)


@functools.lru_cache(maxsize=1)
def _cached_gpu_report() -> dict[str, Any]:
    return build_gpu_report()


class Variant(StrEnum):
    A_ROW_GATHER = "A_row_gather_static"
    B_ARLO_POSTINC = "B_arlo_stride_postincrement"
    C_CONSUMER_MAJOR = "C_consumer_major"
    D_AFFINE_LAYOUT = "D_affine_layout"
    E_STREAM_ADDRESSING = "E_stream_addressing"
    F_AFFINE_STREAM = "F_affine_plus_stream"
    G_OVERLAP = "G_affine_stream_overlap"


@dataclass(frozen=True)
class VariantPolicy:
    issue_mode: str
    layout_mode: str
    producer_consumer_overlap: bool = False


VARIANT_POLICIES = {
    Variant.A_ROW_GATHER: VariantPolicy("baseline", "row_major"),
    # Arlo's tall/column views remove an explicit gather but do not change the
    # physical bank placement.  Keep that separate from C's producer-side
    # consumer-major writeback so software and architecture gains cannot be
    # accidentally credited to each other.
    Variant.B_ARLO_POSTINC: VariantPolicy("postincrement", "row_stride"),
    # C/D retain B's post-increment lowering.  They isolate only the producer
    # layout decision; reverting to the unoptimised issue stream here would
    # incorrectly charge layout variants for losing Arlo's compiler work.
    Variant.C_CONSUMER_MAJOR: VariantPolicy("postincrement", "consumer_major"),
    Variant.D_AFFINE_LAYOUT: VariantPolicy("postincrement", "selected"),
    Variant.E_STREAM_ADDRESSING: VariantPolicy("stream", "row_stride"),
    Variant.F_AFFINE_STREAM: VariantPolicy("stream", "selected"),
    Variant.G_OVERLAP: VariantPolicy("stream", "selected", True),
}


@dataclass(frozen=True)
class HardwarePoint:
    name: str = "transactional_64_candidate"
    mlen: int = 64
    blen: int = 4
    vector_lanes: int = 64
    banks: int = 16
    bank_width: int = 4
    read_ports_per_bank: int = 2
    write_ports_per_bank: int = 1
    fifo_values: int = 64
    layout_slots: int = 4
    hbm_bytes_per_cycle: int = 512
    hbm_burst_bytes: int = 64
    clock_period_ps: int = 1000
    exp_latency: int = 2
    reduction_latency: int = 8
    mamba_parallel_heads: int = 8
    kda_parallel_heads: int = 4
    explicit_state_resident_bytes: int = 0
    activation_bits: int = 16

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if field in {"explicit_state_resident_bytes", "fifo_values"}:
                if value < 0:
                    raise ValueError(f"{field} must be non-negative")
            elif field != "name" and value <= 0:
                raise ValueError(f"{field} must be positive")
        if self.banks * self.bank_width != self.vector_lanes:
            raise ValueError(
                "banked output SRAM must preserve one full Vector row: banks*bank_width must equal vector_lanes"
            )

    @property
    def matrix_macs_per_cycle(self) -> int:
        return self.mlen * self.blen

    @property
    def ordinary_row_read_cycles(self) -> int:
        return math.ceil(self.vector_lanes / (self.banks * self.bank_width))

    @property
    def binary_row_operand_cycles(self) -> int:
        return math.ceil(2 / self.read_ports_per_bank)

    @property
    def regular_vector_regression(self) -> bool:
        return self.ordinary_row_read_cycles > 1 or self.binary_row_operand_cycles > 1

    @property
    def stream_slots_sufficient(self) -> bool:
        # The widest existing lowering binds destination, source and scalar
        # streams concurrently.  Fewer slots can still execute the fallback,
        # but cannot claim the stream-addressing result.
        return self.layout_slots >= 3

    def resource_proxies(self) -> dict[str, Any]:
        return {
            "additional_sram_payload_bytes": 0,
            "bank_count": self.banks,
            "bank_word_bits": self.bank_width * self.activation_bits,
            "read_ports_per_bank": self.read_ports_per_bank,
            "write_ports_per_bank": self.write_ports_per_bank,
            "affine_address_adders": self.banks,
            "cyclic_lane_restore_lanes": self.vector_lanes,
            "layout_config_slots": self.layout_slots,
            "layout_config_bits_upper_bound": self.layout_slots * 15 * 32,
            "fifo_bits": self.fifo_values * self.activation_bits,
            "compiler_managed_state_tile_bytes": self.explicit_state_resident_bytes,
            "ordinary_row_read_cycles": self.ordinary_row_read_cycles,
            "binary_row_operand_cycles": self.binary_row_operand_cycles,
            "regular_vector_regression": self.regular_vector_regression,
            "required_concurrent_stream_slots": 3,
            "stream_slots_sufficient": self.stream_slots_sufficient,
            "scope": "structural proxies only; no area, power, timing, or PPA claim",
        }


@dataclass(frozen=True)
class ResidencyPlan:
    capacity_bytes: int
    resident_bytes: int
    resident_layers: tuple[int, ...]
    explicit_preload_bytes: int
    explicit_final_store_bytes: int
    streamed_state_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceTotals:
    busy: dict[str, int]
    queue_wait: dict[str, int]

    @classmethod
    def empty(cls) -> ResourceTotals:
        return cls(defaultdict(int), defaultdict(int))


@dataclass
class ResourceTimeline:
    """Deterministic single-request schedule over shared accelerator resources.

    Every resource has capacity one.  This is intentionally conservative: it
    models resource queueing and explicitly requested producer/consumer
    overlap, but it does not assume an unimplemented inter-layer lookahead
    prefetcher or multiple requests in flight.
    """

    available: dict[str, int]
    totals: ResourceTotals
    event_count: int = 0

    @classmethod
    def empty(cls) -> ResourceTimeline:
        return cls(defaultdict(int), ResourceTotals.empty())

    def schedule(self, resource: str, duration: int, ready: int) -> tuple[int, int]:
        if duration <= 0:
            return ready, ready
        start = max(ready, self.available[resource])
        end = start + duration
        self.totals.busy[resource] += duration
        self.totals.queue_wait[resource] += start - ready
        self.available[resource] = end
        self.event_count += 1
        return start, end

    @property
    def makespan(self) -> int:
        return max(self.available.values(), default=0)


def _round_bursts(value: int, burst: int) -> int:
    return math.ceil(value / burst) * burst if value else 0


def _sha256_json(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _compiler_modules(compiler_root: Path):
    root = str(compiler_root.resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        from compiler.aten.plena.affine_layout import BankGeometry
        from compiler.aten.plena.hybrid_compile_report import build_report
        from compiler.aten.plena.hybrid_workloads import (
            kimi_k3_manifest,
            kimi_k3_projection_layout_request,
            nemotron3_manifest,
            nemotron_projection_layout_request,
            state_multirow_layout_request,
        )
        from compiler.aten.plena.layout_planner import AffineLayoutPlanner

        return {
            "BankGeometry": BankGeometry,
            "build_report": build_report,
            "kimi_manifest": kimi_k3_manifest,
            "kimi_projection": kimi_k3_projection_layout_request,
            "nemotron_manifest": nemotron3_manifest,
            "nemotron_projection": nemotron_projection_layout_request,
            "state_request": state_multirow_layout_request,
            "planner": AffineLayoutPlanner,
        }
    finally:
        if inserted:
            sys.path.pop(0)


def load_compiler_evidence(compiler_root: Path) -> dict[str, Any]:
    modules = _compiler_modules(compiler_root)
    report = modules["build_report"](compiler_root / "doc/Model_Lib")
    if report.get("schema_version") != 2:
        raise ValueError("pinned Compiler predates the hybrid L-Compute report")
    if report["isa"] != {
        "new_opcode": "L_STREAM_CFG",
        "math_opcodes": "existing Matrix/Vector ISA",
        "loop_opcode": "existing C_LOOP_START/C_LOOP_END",
        "model_specific_opcode": False,
        "cache": False,
    }:
        raise ValueError("Compiler ISA boundary drifted from the no-cache contract")
    return report


def _layout_summary(plan) -> dict[str, Any]:
    keep = {"row_major", "consumer_major", "transpose", plan.selected.name}
    scores = {score.name: score.to_dict() for score in plan.candidates if score.name in keep}
    return {
        "selected": plan.selected.name,
        "scores": scores,
        "candidate_count": len(plan.candidates),
    }


def build_layout_evidence(
    hardware: HardwarePoint,
    compiler_root: Path,
) -> dict[str, Any]:
    modules = _compiler_modules(compiler_root)
    geometry = modules["BankGeometry"](
        banks=hardware.banks,
        bank_width=hardware.bank_width,
        read_ports=hardware.read_ports_per_bank,
        write_ports=hardware.write_ports_per_bank,
    )
    planner = modules["planner"](geometry)
    model_lib = compiler_root / "doc/Model_Lib"
    nemotron = modules["nemotron_manifest"](model_lib / "nemotron-3-nano-30b-a3b.json")
    kimi = modules["kimi_manifest"](model_lib / "kimi-k3-text.json")

    def state(name: str, groups: int, rows: int, width: int, parallel: int):
        return modules["state_request"](
            name=name,
            groups=groups,
            rows_per_group=rows,
            row_elements=width,
            geometry=geometry,
            parallel_rows=min(parallel, rows),
            repeats=(groups * math.ceil(rows / min(parallel, rows)) * math.ceil(width / hardware.bank_width)),
        )

    plans = {
        "nemotron_mamba_projection": planner.plan(
            modules["nemotron_projection"](
                nemotron,
                geometry,
                parallel_heads=min(hardware.mamba_parallel_heads, 8),
            )
        ),
        "kimi_k3_kda_projection": planner.plan(
            modules["kimi_projection"](
                kimi,
                geometry,
                parallel_heads=min(hardware.kda_parallel_heads, 96),
            )
        ),
        "nemotron_mamba_state": planner.plan(state("nemotron_mamba_state", 64, 128, 64, hardware.mamba_parallel_heads)),
        "kimi_k3_kda_state": planner.plan(state("kimi_k3_kda_state", 96, 128, 128, hardware.kda_parallel_heads)),
    }
    return {name: _layout_summary(plan) for name, plan in plans.items()}


def _opcode_latency(opcode: str, hardware: HardwarePoint) -> int:
    if opcode in {"V_RED_SUM", "V_RED_MAX"}:
        return hardware.reduction_latency
    if opcode in {"V_EXP_V", "V_SOFTPLUS_V", "S_EXP_FP"}:
        return hardware.exp_latency
    if opcode == "M_MM":
        return hardware.blen
    return 1


def _weighted_issue(metrics: dict[str, Any], hardware: HardwarePoint) -> int:
    return sum(int(count) * _opcode_latency(opcode, hardware) for opcode, count in metrics["opcode_census"].items())


def _compiler_recurrence_cycles(
    pair: dict[str, Any],
    issue_mode: str,
    hardware: HardwarePoint,
) -> int:
    if issue_mode == "stream":
        return _weighted_issue(pair["stream"], hardware)
    baseline = _weighted_issue(pair["baseline"], hardware)
    if issue_mode == "postincrement":
        return baseline - int(pair["postincrement_only"]["removed_foldable_self_advances"])
    return baseline


def _state_bytes_per_layer(report: WorkloadReport) -> dict[int, int]:
    per_layer: dict[int, int] = defaultdict(int)
    for stage in report.stages:
        per_layer[stage.layer_id] += stage.traffic.state_write_bytes
    return {layer: value for layer, value in per_layer.items() if layer >= 0 and value}


def build_residency_plan(report: WorkloadReport, capacity_bytes: int, decode_tokens: int) -> ResidencyPlan:
    per_layer = _state_bytes_per_layer(report)
    resident = []
    used = 0
    for layer_id, size in sorted(per_layer.items()):
        if used + size <= capacity_bytes:
            resident.append(layer_id)
            used += size
    total = sum(per_layer.values())
    streamed_per_token = total - used
    return ResidencyPlan(
        capacity_bytes=capacity_bytes,
        resident_bytes=used,
        resident_layers=tuple(resident),
        explicit_preload_bytes=used,
        explicit_final_store_bytes=used,
        streamed_state_bytes=streamed_per_token * decode_tokens * 2,
    )


def _generic_compute(stage: StageWork, hardware: HardwarePoint) -> tuple[int, int]:
    matrix = math.ceil(stage.macs / hardware.matrix_macs_per_cycle) if stage.macs else 0
    vector = math.ceil(stage.elementwise_ops / hardware.vector_lanes) if stage.elementwise_ops else 0
    if stage.exp_ops:
        vector += math.ceil(stage.exp_ops / hardware.vector_lanes) * hardware.exp_latency
    if stage.scan_compositions:
        vector += math.ceil(stage.scan_compositions / hardware.vector_lanes)
    if stage.resource in {"conv", "state", "exp"}:
        vector += matrix
        matrix = 0
    return matrix, vector


def _layout_score(
    evidence: dict[str, Any],
    key: str,
    mode: str,
) -> dict[str, Any]:
    plan = evidence[key]
    if mode == "selected":
        name = plan["selected"]
    elif mode == "consumer_major":
        name = "consumer_major"
    else:
        name = "row_major"
    if name not in plan["scores"]:
        name = "row_major"
    score = dict(plan["scores"][name])
    if mode == "row_stride":
        # A strided logical view avoids the explicit gather/reorder but leaves
        # the row-major bank service unchanged.
        reorder = int(score["reorder_cycles"])
        score["reorder_cycles"] = 0
        score["total_cycles"] = int(score["total_cycles"]) - reorder
    return score


def _projection_layout_key(stage: StageWork) -> str | None:
    if stage.name == "mamba_in_projection":
        return "nemotron_mamba_projection"
    if stage.name == "kda_qkv_projection":
        return "kimi_k3_kda_projection"
    return None


def _aggregate_stage(
    totals: dict[str, Any],
    stage: StageWork,
    duration: int,
    matrix_cycles: int,
    vector_cycles: int,
    hbm_cycles: int,
    layout: dict[str, int],
) -> None:
    layer = totals["by_layer_type"].setdefault(
        stage.layer_type,
        {
            "cycles": 0,
            "matrix_cycles": 0,
            "vector_cycles": 0,
            "hbm_cycles": 0,
            "layout_service_cycles": 0,
            "bank_conflict_stall_cycles": 0,
        },
    )
    layer["cycles"] += duration
    layer["matrix_cycles"] += matrix_cycles
    layer["vector_cycles"] += vector_cycles
    layer["hbm_cycles"] += hbm_cycles
    layer["layout_service_cycles"] += layout.get("service", 0)
    layer["bank_conflict_stall_cycles"] += layout.get("conflict", 0)


def simulate_workload(
    report: WorkloadReport,
    *,
    model_name: str,
    variant: Variant,
    hardware: HardwarePoint,
    compiler_evidence: dict[str, Any],
    layout_evidence: dict[str, Any],
    residency: ResidencyPlan | None = None,
) -> dict[str, Any]:
    policy = VARIANT_POLICIES[variant]
    timeline = ResourceTimeline.empty()
    totals: dict[str, Any] = {
        "cycles": 0,
        "logical_hbm_read_bytes": 0,
        "logical_hbm_write_bytes": 0,
        "physical_hbm_read_bytes": 0,
        "physical_hbm_write_bytes": 0,
        "layout_service_cycles": 0,
        "bank_conflict_stall_cycles": 0,
        "lane_restore_cycles": 0,
        "fifo_stall_cycles": 0,
        "by_layer_type": {},
    }
    resident_layers = set(residency.resident_layers if residency else ())
    recurrence_consumed: set[tuple[int, str]] = set()
    dependency_ready = 0

    if residency and residency.explicit_preload_bytes:
        logical = residency.explicit_preload_bytes
        physical = _round_bursts(logical, hardware.hbm_burst_bytes)
        _, dependency_ready = timeline.schedule("hbm", math.ceil(physical / hardware.hbm_bytes_per_cycle), 0)
        totals["logical_hbm_read_bytes"] += logical
        totals["physical_hbm_read_bytes"] += physical
        totals["explicit_state_transfer_bytes"] = logical

    for stage in report.stages:
        stage_ready = dependency_ready
        matrix_cycles, vector_cycles = _generic_compute(stage, hardware)
        recurrence_key: str | None = None
        recurrence_names: set[str] = set()
        if report.scenario.phase == InferencePhase.DECODE and stage.layer_type == "mamba":
            recurrence_key = "nemotron_mamba_decode_recurrence"
            recurrence_names = {"mamba_state_update", "mamba_state_output"}
        elif report.scenario.phase == InferencePhase.DECODE and stage.layer_type == "kda":
            recurrence_key = "kimi_k3_decode_recurrent_mixer"
            recurrence_names = {
                "kda_qk_l2norm",
                "kda_state_decay_prediction",
                "kda_delta_update_output",
                "kda_output_gate_rmsnorm",
            }
        if recurrence_key and stage.name in recurrence_names:
            marker = (stage.layer_id, recurrence_key)
            if marker in recurrence_consumed:
                matrix_cycles = vector_cycles = 0
            else:
                recurrence_consumed.add(marker)
                matrix_cycles = 0
                vector_cycles = _compiler_recurrence_cycles(
                    compiler_evidence["assembly"][recurrence_key],
                    policy.issue_mode,
                    hardware,
                )

        traffic = stage.traffic
        weight_read_bytes = traffic.weight_read_bytes
        dynamic_read_bytes = traffic.activation_read_bytes + traffic.kv_read_bytes + traffic.state_read_bytes
        write_bytes = traffic.activation_write_bytes + traffic.kv_write_bytes + traffic.state_write_bytes
        if stage.layer_id in resident_layers:
            dynamic_read_bytes -= traffic.state_read_bytes
            write_bytes -= traffic.state_write_bytes
        read_bytes = weight_read_bytes + dynamic_read_bytes
        physical_weight_read = _round_bursts(weight_read_bytes, hardware.hbm_burst_bytes)
        physical_dynamic_read = _round_bursts(dynamic_read_bytes, hardware.hbm_burst_bytes)
        physical_read = physical_weight_read + physical_dynamic_read
        physical_write = _round_bursts(write_bytes, hardware.hbm_burst_bytes)

        _, weight_ready = timeline.schedule(
            "hbm",
            math.ceil(physical_weight_read / hardware.hbm_bytes_per_cycle),
            stage_ready,
        )
        _, dynamic_ready = timeline.schedule(
            "hbm",
            math.ceil(physical_dynamic_read / hardware.hbm_bytes_per_cycle),
            stage_ready,
        )
        compute_ready = max(stage_ready, weight_ready, dynamic_ready)

        matrix_start, matrix_end = timeline.schedule("matrix", matrix_cycles, compute_ready)
        vector_start_ready = matrix_end if matrix_cycles else compute_ready
        _, vector_end = timeline.schedule("vector", vector_cycles, vector_start_ready)
        compute_end = max(matrix_end, vector_end)

        layout_stats = {"service": 0, "conflict": 0, "restore": 0}
        layout_key = _projection_layout_key(stage)
        if layout_key:
            repeats = report.scenario.tokens
            if variant == Variant.A_ROW_GATHER:
                row = _layout_score(layout_evidence, layout_key, "row_major")
                # Official Kimi uses eight independent projection tensors, so
                # it has no packed-QKV gather to charge.  Nemotron's packed
                # in-projection still pays the explicit baseline reorder.
                service = int(row["reorder_cycles"]) * repeats if layout_key == "nemotron_mamba_projection" else 0
                layout_stats["service"] = service
            elif variant in {
                Variant.C_CONSUMER_MAJOR,
                Variant.D_AFFINE_LAYOUT,
                Variant.F_AFFINE_STREAM,
                Variant.G_OVERLAP,
            }:
                score = _layout_score(layout_evidence, layout_key, policy.layout_mode)
                # Existing serial Vector instructions already pay their normal
                # one-row SRAM access in their opcode latency.  Do not charge a
                # counterfactual multirow packet read on top.  Only incremental
                # affine write conflicts and cyclic lane restore are exposed in
                # the executable timeline; the full packet service remains a
                # separately labelled architecture upper bound.
                write_overhead = max(0, int(score["write_cycles"]) - int(score["write_floor_cycles"]))
                conflict = write_overhead * repeats
                restore = int(score["lane_restore_cycles"]) * repeats
                service = conflict + restore
                if policy.producer_consumer_overlap:
                    if hardware.fifo_values < hardware.vector_lanes:
                        layout_stats["fifo_stall"] = service
                    # The incremental mapper/restore path streams alongside
                    # Matrix writeback; duration below takes the max with the
                    # producer rather than adding it.
                layout_stats.update(
                    service=service,
                    conflict=conflict,
                    restore=restore,
                )

        layout_ready = compute_end
        if policy.producer_consumer_overlap and layout_key and hardware.fifo_values >= hardware.vector_lanes:
            # A full packet FIFO allows the affine write mapper to drain while
            # Matrix output rows are being produced.  No smaller FIFO receives
            # this overlap credit.
            layout_ready = matrix_start if matrix_cycles else compute_ready
        _, layout_end = timeline.schedule("banked_output_sram", layout_stats["service"], layout_ready)
        output_ready = max(compute_end, layout_end)

        _, write_end = timeline.schedule(
            "hbm",
            math.ceil(physical_write / hardware.hbm_bytes_per_cycle),
            output_ready,
        )
        # An activation explicitly written to HBM is the next stage's source;
        # state/KV commits may continue in parallel, but still contend for the
        # single shared HBM resource when the next stage requests data.
        dependency_ready = max(output_ready, write_end) if traffic.activation_write_bytes else output_ready
        duration = dependency_ready - stage_ready
        hbm_cycles = (
            math.ceil(physical_weight_read / hardware.hbm_bytes_per_cycle)
            + math.ceil(physical_dynamic_read / hardware.hbm_bytes_per_cycle)
            + math.ceil(physical_write / hardware.hbm_bytes_per_cycle)
        )

        totals["logical_hbm_read_bytes"] += read_bytes
        totals["logical_hbm_write_bytes"] += write_bytes
        totals["physical_hbm_read_bytes"] += physical_read
        totals["physical_hbm_write_bytes"] += physical_write
        totals["layout_service_cycles"] += layout_stats["service"]
        totals["bank_conflict_stall_cycles"] += layout_stats["conflict"]
        totals["lane_restore_cycles"] += layout_stats["restore"]
        totals["fifo_stall_cycles"] += layout_stats.get("fifo_stall", 0)
        _aggregate_stage(
            totals,
            stage,
            duration,
            matrix_cycles,
            vector_cycles,
            hbm_cycles,
            layout_stats,
        )

    if residency and residency.explicit_final_store_bytes:
        logical = residency.explicit_final_store_bytes
        physical = _round_bursts(logical, hardware.hbm_burst_bytes)
        _, dependency_ready = timeline.schedule(
            "hbm",
            math.ceil(physical / hardware.hbm_bytes_per_cycle),
            dependency_ready,
        )
        totals["logical_hbm_write_bytes"] += logical
        totals["physical_hbm_write_bytes"] += physical
        totals["explicit_state_transfer_bytes"] = totals.get("explicit_state_transfer_bytes", 0) + logical

    cycles = max(1, dependency_ready, timeline.makespan)
    totals["cycles"] = cycles
    totals["timeline_event_count"] = timeline.event_count
    totals["resource_busy_cycles"] = dict(timeline.totals.busy)
    totals["resource_queue_wait_cycles"] = dict(timeline.totals.queue_wait)
    totals["resource_utilization"] = {name: value / cycles for name, value in timeline.totals.busy.items()}
    totals["variant"] = variant
    totals["model"] = model_name
    totals["phase"] = report.scenario.phase
    totals["clock_period_ps_assumption"] = hardware.clock_period_ps
    totals["latency_us_proxy"] = totals["cycles"] * hardware.clock_period_ps / 1_000_000
    totals["scope"] = "Compiler/Simulator cycle estimate; not RTL or silicon"
    return totals


def _scenario(
    phase: InferencePhase,
    *,
    sequence_length: int,
    context_length: int,
    include_embedding: bool = True,
    include_lm_head: bool = True,
    moe_unique_experts: int | None = None,
) -> WorkloadScenario:
    return WorkloadScenario(
        phase=phase,
        batch_size=1,
        sequence_length=sequence_length,
        context_length=context_length,
        decode_tokens=1,
        scan_strategy=(ScanStrategy.CHUNKED_AFFINE if phase == InferencePhase.PREFILL else ScanStrategy.SEQUENTIAL),
        include_embedding=include_embedding,
        include_lm_head=include_lm_head,
        moe_unique_experts=moe_unique_experts,
    )


def _model(
    model_name: str,
    compiler_root: Path,
    *,
    activation_precision: Precision,
    weight_precision: Precision | None,
    state_precision: Precision,
):
    if model_name == "nemotron3":
        arch = load_nemotron3_arch(compiler_root / "doc/Model_Lib/nemotron-3-nano-30b-a3b.json")
        policy = None
        if weight_precision is None:
            policy = formal_nemotron_nvfp4_weight_policy(
                arch,
                _cached_gpu_report()["b200_formal"]["nemotron"]["checkpoint_quantization"],
            )
        return Nemotron3WorkloadModel(
            arch,
            activation_precision=activation_precision,
            weight_precision=weight_precision or Precision.NVFP4,
            state_precision=state_precision,
            weight_precision_policy=policy,
        )
    if model_name == "kimi_k3":
        policy = None
        if weight_precision is None:
            contract = json.loads((compiler_root / "doc/Model_Lib/kimi-k3-text.json").read_text())["precision_contract"]
            policy = formal_kimi_k3_mxfp4_weight_policy(contract)
        return KimiK3HybridWorkloadModel(
            KimiK3Architecture(),
            activation_precision=activation_precision,
            weight_precision=weight_precision or Precision.MXFP4,
            state_precision=state_precision,
            conv_state_precision=Precision.BF16,
            weight_precision_policy=policy,
        )
    raise ValueError(f"unknown model {model_name}")


def validate_workload_schedule(
    report: WorkloadReport,
    model_name: str,
    compiler_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove that the Simulator timeline matches the pinned Compiler manifest."""

    manifest = compiler_evidence["workloads"][model_name]
    layers = manifest["layers"]
    observed_ids = [stage.layer_id for stage in report.stages if stage.layer_id >= 0]
    if observed_ids != sorted(observed_ids):
        raise ValueError(f"{model_name} workload stages are not in layer order")
    if set(observed_ids) != set(range(len(layers))):
        raise ValueError(f"{model_name} workload does not cover every manifest layer")

    by_layer: dict[int, list[StageWork]] = defaultdict(list)
    for stage in report.stages:
        if stage.layer_id >= 0:
            by_layer[stage.layer_id].append(stage)

    def require_subsequence(layer_id: int, names: tuple[str, ...]) -> None:
        actual = [stage.name for stage in by_layer[layer_id]]
        cursor = 0
        for name in names:
            try:
                cursor = actual.index(name, cursor) + 1
            except ValueError as error:
                raise ValueError(f"{model_name} layer {layer_id + 1} misses ordered stage {name}") from error

    for layer_id, expected in enumerate(layers):
        actual_types = {stage.layer_type for stage in by_layer[layer_id]}
        mixer = expected["mixer"]
        ffn = expected["ffn"]
        if model_name == "nemotron3":
            expected_type = {"gqa": "attention"}.get(mixer or ffn, mixer or ffn)
            if actual_types != {expected_type}:
                raise ValueError(f"Nemotron layer {layer_id + 1}: {actual_types} != {expected_type}")
            required = {
                "mamba": (
                    "block_rms_norm",
                    "mamba_in_projection",
                    "mamba_conv1d",
                    "mamba_gate_group_rms_norm",
                    "mamba_out_projection",
                    "block_residual",
                ),
                "attention": (
                    "block_rms_norm",
                    "attention_qkv_projection",
                    "attention_qk_softmax_pv",
                    "attention_out_projection",
                    "block_residual",
                ),
                "moe": (
                    "block_rms_norm",
                    "moe_router_topk",
                    "moe_routed_experts",
                    "moe_shared_expert",
                    "moe_combine",
                    "block_residual",
                ),
            }[expected_type]
            require_subsequence(layer_id, required)
            continue

        expected_mixer = str(mixer)
        expected_ffn = "dense" if ffn == "dense_ffn" else str(ffn)
        required_types = {"attn_res", expected_mixer, expected_ffn}
        if not required_types.issubset(actual_types):
            raise ValueError(f"Kimi layer {layer_id + 1}: {actual_types} misses {required_types}")
        mixer_names = (
            (
                "input_rms_norm",
                "kda_qkv_projection",
                "kda_short_conv",
                "kda_state_decay_prediction",
                "kda_delta_update_output",
                "kda_out_projection",
                "prefix_sum_after_mixer",
            )
            if expected_mixer == "kda"
            else (
                "input_rms_norm",
                "mla_q_low_rank_projection",
                "mla_kv_latent_projection",
                "mla_compressed_kv_attention",
                "mla_out_projection",
                "prefix_sum_after_mixer",
            )
        )
        ffn_names = (
            ("post_attention_rms_norm", "dense_situ_ffn", "prefix_sum_after_ffn")
            if expected_ffn == "dense"
            else (
                "post_attention_rms_norm",
                "latent_moe_router_top16",
                "latent_moe_routed_experts",
                "latent_moe_shared_experts",
                "latent_moe_combine",
                "prefix_sum_after_ffn",
            )
        )
        require_subsequence(layer_id, (*mixer_names, *ffn_names))

    compressed_mla = None
    if model_name == "kimi_k3":
        cache_elements = manifest["dimensions"]["mla_cache_elements_per_token"]
        if cache_elements != 576:
            raise ValueError("Kimi MLA manifest no longer uses compressed 512+64 cache")
        bytes_per_token = storage_bytes(cache_elements, report.activation_precision)
        expected_write = storage_bytes(
            report.scenario.tokens * cache_elements,
            report.activation_precision,
        )
        expected_read = (
            storage_bytes(
                report.scenario.batch_size * report.scenario.context_length * cache_elements,
                report.activation_precision,
            )
            if report.scenario.phase == InferencePhase.DECODE
            else 0
        )
        for stage in report.stages:
            if stage.name == "mla_kv_latent_projection":
                if stage.traffic.kv_write_bytes != expected_write:
                    raise ValueError("Kimi MLA cache write expanded beyond 576 elements/token")
            elif stage.name == "mla_compressed_kv_attention":
                if stage.traffic.kv_read_bytes != expected_read:
                    raise ValueError("Kimi MLA cache read is not compressed")
        compressed_mla = {
            "elements_per_token": cache_elements,
            "bytes_per_token": bytes_per_token,
            "expanded_96_head_kv_materialized": False,
        }

    return {
        "validated": True,
        "manifest_layers": len(layers),
        "stage_count": len(report.stages),
        "phase": report.scenario.phase,
        "compressed_mla": compressed_mla,
    }


def run_ablation(
    model_name: str,
    *,
    phase: InferencePhase,
    tokens: int,
    context_length: int,
    decode_tokens: int,
    hardware: HardwarePoint,
    compiler_root: Path,
    compiler_evidence: dict[str, Any],
    layout_evidence: dict[str, Any],
    activation_precision: Precision = Precision.BF16,
    weight_precision: Precision | None = None,
    state_precision: Precision = Precision.FP32,
) -> dict[str, Any]:
    workload_model = _model(
        model_name,
        compiler_root,
        activation_precision=activation_precision,
        weight_precision=weight_precision,
        state_precision=state_precision,
    )
    reports = []
    if phase == InferencePhase.PREFILL:
        assignments_per_token = 6 if model_name == "nemotron3" else 16
        expert_count = 128 if model_name == "nemotron3" else 896
        # No Kimi routing trace was collected.  This is a deterministic
        # worst-case distinct-expert bound, not a measured routing claim.
        unique = min(expert_count, tokens * assignments_per_token)
        reports.append(
            workload_model.build(
                _scenario(
                    phase,
                    sequence_length=tokens,
                    context_length=context_length,
                    moe_unique_experts=unique,
                )
            )
        )
    else:
        for offset in range(decode_tokens):
            reports.append(
                workload_model.build(
                    _scenario(
                        phase,
                        sequence_length=1,
                        context_length=context_length + offset,
                        moe_unique_experts=6 if model_name == "nemotron3" else 16,
                    )
                )
            )

    schedule_validations = [validate_workload_schedule(report, model_name, compiler_evidence) for report in reports]

    residency = None
    if phase == InferencePhase.DECODE:
        residency = build_residency_plan(reports[0], hardware.explicit_state_resident_bytes, decode_tokens)
    records = []
    for variant in Variant:
        pieces = []
        for index, report in enumerate(reports):
            per_token_residency = None
            if residency:
                per_token_residency = replace(
                    residency,
                    explicit_preload_bytes=(residency.explicit_preload_bytes if index == 0 else 0),
                    explicit_final_store_bytes=(
                        residency.explicit_final_store_bytes if index == len(reports) - 1 else 0
                    ),
                )
            pieces.append(
                simulate_workload(
                    report,
                    model_name=model_name,
                    variant=variant,
                    hardware=hardware,
                    compiler_evidence=compiler_evidence,
                    layout_evidence=layout_evidence,
                    residency=per_token_residency,
                )
            )
        record = {
            "variant": variant,
            "cycles": sum(piece["cycles"] for piece in pieces),
            "timeline_event_count": sum(piece["timeline_event_count"] for piece in pieces),
            "logical_hbm_read_bytes": sum(piece["logical_hbm_read_bytes"] for piece in pieces),
            "logical_hbm_write_bytes": sum(piece["logical_hbm_write_bytes"] for piece in pieces),
            "physical_hbm_read_bytes": sum(piece["physical_hbm_read_bytes"] for piece in pieces),
            "physical_hbm_write_bytes": sum(piece["physical_hbm_write_bytes"] for piece in pieces),
            "explicit_state_transfer_bytes": sum(piece.get("explicit_state_transfer_bytes", 0) for piece in pieces),
            "layout_service_cycles": sum(piece["layout_service_cycles"] for piece in pieces),
            "bank_conflict_stall_cycles": sum(piece["bank_conflict_stall_cycles"] for piece in pieces),
            "lane_restore_cycles": sum(piece["lane_restore_cycles"] for piece in pieces),
            "fifo_stall_cycles": sum(piece["fifo_stall_cycles"] for piece in pieces),
            "latency_us_proxy": sum(piece["latency_us_proxy"] for piece in pieces),
            "by_layer_type": {},
            "resource_busy_cycles": {},
            "resource_queue_wait_cycles": {},
        }
        resources = {
            name
            for piece in pieces
            for field in ("resource_busy_cycles", "resource_queue_wait_cycles")
            for name in piece[field]
        }
        for resource in sorted(resources):
            record["resource_busy_cycles"][resource] = sum(
                piece["resource_busy_cycles"].get(resource, 0) for piece in pieces
            )
            record["resource_queue_wait_cycles"][resource] = sum(
                piece["resource_queue_wait_cycles"].get(resource, 0) for piece in pieces
            )
        record["resource_utilization"] = {
            resource: busy / max(1, record["cycles"]) for resource, busy in record["resource_busy_cycles"].items()
        }
        layer_types = {name for piece in pieces for name in piece["by_layer_type"]}
        for layer_type in sorted(layer_types):
            record["by_layer_type"][layer_type] = {
                key: sum(piece["by_layer_type"].get(layer_type, {}).get(key, 0) for piece in pieces)
                for key in (
                    "cycles",
                    "matrix_cycles",
                    "vector_cycles",
                    "hbm_cycles",
                    "layout_service_cycles",
                    "bank_conflict_stall_cycles",
                )
            }
        records.append(record)

    baseline = next(record for record in records if record["variant"] == Variant.B_ARLO_POSTINC)
    for record in records:
        record["speedup_vs_arlo_B"] = baseline["cycles"] / max(1, record["cycles"])
    return {
        "model": model_name,
        "phase": phase,
        "tokens": tokens if phase == InferencePhase.PREFILL else decode_tokens,
        "context_length": context_length,
        "records": records,
        "schedule_validation": {
            **schedule_validations[0],
            "validated_reports": len(schedule_validations),
        },
        "residency": residency.to_dict() if residency else None,
        "prefill_lowering": (
            "Nemotron chunked affine SSD; Kimi architecture work is chunk-linear and does not reuse decode issue counts"
            if phase == InferencePhase.PREFILL
            else None
        ),
        "routing_scope": (
            "Nemotron/Kimi prefill uses a deterministic distinct-expert upper bound; "
            "Nemotron measured routing is reported separately and Kimi routing was not profiled"
            if phase == InferencePhase.PREFILL
            else "one decode token can select at most top-k distinct experts per layer"
        ),
    }


def _gpu_summary(gpu: dict[str, Any]) -> dict[str, Any]:
    formal = gpu["b200_formal"]
    supplemental = gpu["b200_supplemental"]
    rtx5090 = gpu["rtx5090_mamba"]
    decode_b1 = next(case for case in formal["kda"]["cases"] if case["case"] == "decode_b1")
    component_latency = {
        f"{row['component']}_{row['case']}": {
            "median_ms": row["median_ms"],
            "p95_ms": row["p95_ms"],
        }
        for row in supplemental["kimi_component_latency"]
    }
    return {
        "gpu": formal["gpu"],
        "kda_shape": formal["kda"]["shape"],
        "kda_projection_storage": formal["kda"]["projection_storage"],
        "kda_decode_b1_matrix_time_fraction": decode_b1["matrix_path_time_fraction"],
        "kda_decode_b1_state_core_time_fraction": decode_b1["state_core_time_fraction"],
        "nemotron_model": formal["nemotron"]["model"],
        "nemotron_revision": formal["nemotron"]["revision"],
        "nemotron_decode_itl_median_ms": formal["nemotron"]["latency"]["decode_s2048_128"]["itl_median_ms"],
        "nemotron_moe_to_mamba_prefill_dram_read_ratio": formal["nemotron"]["moe_to_mamba_prefill_dram_read_ratio"],
        "nemotron_decode_max_hotspot_to_mean": formal["nemotron"]["routing"]["decode_max_hotspot_to_mean"],
        "rtx5090_mamba_decode_b1_state_core_time_fraction": rtx5090["nsys_stages"]["decode_b1"][
            "state_core_time_fraction"
        ],
        "kimi_component_latency": component_latency,
        "kimi_component_parity": supplemental["kimi_component_parity"],
        "mamba_precision_s32768": supplemental["mamba_precision_s32768"],
        "evidence_use": "shape/bottleneck/baseline only; GPU time is not a PLENA cycle constant",
    }


def _dse_points(base: HardwarePoint) -> list[HardwarePoint]:
    points = [base]
    for banks in (8, 16, 32):
        points.append(replace(base, name=f"banks_{banks}", banks=banks, bank_width=base.vector_lanes // banks))
    for ports in (1, 2):
        points.append(replace(base, name=f"read_ports_{ports}", read_ports_per_bank=ports))
        points.append(replace(base, name=f"write_ports_{ports}", write_ports_per_bank=ports))
    for parallel in (1, 2, 4, 8, 16):
        points.append(
            replace(
                base,
                name=f"parallel_{parallel}",
                mamba_parallel_heads=parallel,
                kda_parallel_heads=parallel,
            )
        )
    for fifo in (0, 16, 32, 64, 128):
        points.append(replace(base, name=f"fifo_{fifo}", fifo_values=fifo))
    for slots in (1, 2, 4, 8):
        points.append(replace(base, name=f"slots_{slots}", layout_slots=slots))
    for bandwidth in (64, 128, 256, 512, 1024, 2048, 4096):
        points.append(replace(base, name=f"hbm_{bandwidth}", hbm_bytes_per_cycle=bandwidth))
    for state_tile_mib in (0, 24, 32, 48, 64):
        points.append(
            replace(
                base,
                name=f"explicit_state_tile_{state_tile_mib}mib",
                explicit_state_resident_bytes=state_tile_mib * 1024 * 1024,
            )
        )
    seen = set()
    unique = []
    for point in points:
        key = tuple((name, value) for name, value in asdict(point).items() if name != "name")
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def run_dse(
    base: HardwarePoint,
    *,
    compiler_root: Path,
    compiler_evidence: dict[str, Any],
) -> dict[str, Any]:
    records = []
    layout_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    for point in _dse_points(base):
        layout_key = (
            point.banks,
            point.bank_width,
            point.read_ports_per_bank,
            point.write_ports_per_bank,
            point.mamba_parallel_heads,
            point.kda_parallel_heads,
        )
        layout = layout_cache.get(layout_key)
        if layout is None:
            layout = build_layout_evidence(point, compiler_root)
            layout_cache[layout_key] = layout
        models = {}
        for model_name in ("nemotron3", "kimi_k3"):
            result = run_ablation(
                model_name,
                phase=InferencePhase.DECODE,
                tokens=1,
                context_length=2048,
                # Four tokens make explicit state residency observable while
                # keeping the one-factor sweep inexpensive.  A one-token sweep
                # charges the same preload/store as streaming and cannot answer
                # the residency question.
                decode_tokens=4,
                hardware=point,
                compiler_root=compiler_root,
                compiler_evidence=compiler_evidence,
                layout_evidence=layout,
            )
            f = next(record for record in result["records"] if record["variant"] == Variant.F_AFFINE_STREAM)
            e = next(record for record in result["records"] if record["variant"] == Variant.E_STREAM_ADDRESSING)
            g = next(record for record in result["records"] if record["variant"] == Variant.G_OVERLAP)
            b = next(record for record in result["records"] if record["variant"] == Variant.B_ARLO_POSTINC)
            projection_key = "nemotron_mamba_projection" if model_name == "nemotron3" else "kimi_k3_kda_projection"
            state_key = "nemotron_mamba_state" if model_name == "nemotron3" else "kimi_k3_kda_state"
            projection = layout[projection_key]
            projection_row = projection["scores"]["row_major"]
            projection_selected = projection["scores"][projection["selected"]]
            state = layout[state_key]
            state_row = state["scores"]["row_major"]
            state_selected = state["scores"][state["selected"]]
            models[model_name] = {
                "B_cycles": b["cycles"],
                "E_cycles": e["cycles"],
                "F_cycles": f["cycles"],
                "G_cycles": g["cycles"],
                "E_stream_speedup_vs_B": b["cycles"] / max(1, e["cycles"]),
                "F_speedup_vs_B": b["cycles"] / max(1, f["cycles"]),
                "G_speedup_vs_B": b["cycles"] / max(1, g["cycles"]),
                "G_affine_incremental_speedup_vs_E": e["cycles"] / max(1, g["cycles"]),
                "F_bank_conflict_stalls": f["bank_conflict_stall_cycles"],
                "local_projection_packet_speedup_upper_bound": (
                    projection_row["total_cycles"] / max(1, projection_selected["total_cycles"])
                ),
                "local_state_packet_speedup_upper_bound": (
                    state_row["total_cycles"] / max(1, state_selected["total_cycles"])
                ),
            }
        records.append(
            {
                "hardware": asdict(point),
                "resource_proxies": point.resource_proxies(),
                "models": models,
                "eligible_for_freeze": (not point.regular_vector_regression and point.stream_slots_sufficient),
            }
        )
    base_record = next(record for record in records if record["hardware"]["name"] == base.name)
    stream_pass = all(model["E_stream_speedup_vs_B"] >= 1.005 for model in base_record["models"].values())
    affine_pass = all(model["G_affine_incremental_speedup_vs_E"] >= 1.01 for model in base_record["models"].values())
    return {
        "method": "one-factor sweep around the transactional-64 base point",
        "records": records,
        "freeze_rule": (
            "reject any point that regresses ordinary full-row/binary Vector access "
            "or cannot bind the three concurrent streams used by the widest lowering"
        ),
        "base_decision": {
            "stream_addressing_earns_isa": stream_pass,
            "affine_colayout_earns_isa_on_current_serial_consumer": affine_pass,
            "models": base_record["models"],
            "interpretation": (
                "Local packet speedups are architecture upper bounds. Affine layout earns an "
                "architectural mode only when G improves over executable stream-only E, not "
                "merely when a counterfactual packet has fewer bank conflicts."
            ),
        },
    }


def run_precision_dse(
    hardware: HardwarePoint,
    *,
    compiler_root: Path,
    compiler_evidence: dict[str, Any],
    layout_evidence: dict[str, Any],
    gpu_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Keep activation, state, and weight storage as independent knobs.

    This sweep reports traffic/performance only.  It deliberately does not
    manufacture an accuracy prediction; a precision point is numerically legal
    only after a long-sequence golden comparison exists for that exact policy.
    """

    defaults = {
        "nemotron3": (Precision.BF16, Precision.NVFP4, Precision.FP32),
        "kimi_k3": (Precision.BF16, Precision.MXFP4, Precision.FP32),
    }
    records = []
    measured_mamba = gpu_evidence["b200_supplemental"]["mamba_precision_s32768"]
    measured_state_names = {
        Precision.FP32: "fp32",
        Precision.BF16: "bf16_chunk128",
        Precision.FP16: "fp16_chunk128",
        Precision.MX8: "mx8_chunk128",
    }
    for model_name, default in defaults.items():
        activation_default, weight_default, state_default = default
        points = []
        for state in (Precision.FP32, Precision.BF16, Precision.FP16, Precision.MX8):
            points.append(("state", state, activation_default, weight_default, state))
        for activation in (Precision.BF16, Precision.MX8):
            points.append(("activation", activation, activation, weight_default, state_default))
        weight_options = (
            (Precision.BF16, Precision.NVFP4) if model_name == "nemotron3" else (Precision.BF16, Precision.MXFP4)
        )
        for weight in weight_options:
            points.append(("weight", weight, activation_default, weight, state_default))

        seen = set()
        for axis, value, activation, weight, state in points:
            key = (activation, weight, state)
            if key in seen:
                continue
            seen.add(key)
            result = run_ablation(
                model_name,
                phase=InferencePhase.DECODE,
                tokens=1,
                context_length=2048,
                decode_tokens=1,
                hardware=hardware,
                compiler_root=compiler_root,
                compiler_evidence=compiler_evidence,
                layout_evidence=layout_evidence,
                activation_precision=activation,
                weight_precision=weight,
                state_precision=state,
            )
            b = next(record for record in result["records"] if record["variant"] == Variant.B_ARLO_POSTINC)
            f = next(record for record in result["records"] if record["variant"] == Variant.F_AFFINE_STREAM)
            accuracy: dict[str, Any] = {"status": "not_measured"}
            if model_name == "nemotron3" and axis == "state":
                measured = measured_mamba[measured_state_names[state]]
                accuracy = {
                    "status": "measured_real_shape_random_weights",
                    "sequence_length": 32_768,
                    "schedule": "chunk128",
                    "output_relative_l2_mean": measured["output_relative_l2_mean"],
                    "state_relative_l2_mean": measured["state_relative_l2_mean"],
                    "state_total_bytes": measured["total_bytes"],
                    "hbm_reduction_vs_fp32": measured["hbm_reduction_vs_fp32"],
                }
            records.append(
                {
                    "model": model_name,
                    "swept_axis": axis,
                    "swept_value": value,
                    "activation_precision": activation,
                    "weight_precision": weight,
                    "state_precision": state,
                    "B_cycles": b["cycles"],
                    "F_cycles": f["cycles"],
                    "F_speedup_vs_B": b["cycles"] / max(1, f["cycles"]),
                    "F_logical_hbm_bytes": (f["logical_hbm_read_bytes"] + f["logical_hbm_write_bytes"]),
                    "accuracy": accuracy,
                }
            )
    return {
        "records": records,
        "legality_gate": (
            "Nemotron Mamba state points use the existing S32768 numerical sweep. "
            "KDA state, activation and weight points remain illegal to freeze until "
            "an exact-policy numerical comparison exists."
        ),
    }


def summarize_ablation(experiments: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_name, model_experiments in experiments.items():
        summary[model_name] = {}
        for experiment_name, experiment in model_experiments.items():
            records = {str(record["variant"]): record for record in experiment["records"]}
            baseline = records[str(Variant.B_ARLO_POSTINC)]
            summary[model_name][experiment_name] = {
                "B_arlo_cycles": baseline["cycles"],
                "variant_cycles": {name: record["cycles"] for name, record in records.items()},
                "speedup_vs_arlo_B": {
                    name: baseline["cycles"] / max(1, record["cycles"]) for name, record in records.items()
                },
                "bank_conflict_stall_cycles": {
                    name: record["bank_conflict_stall_cycles"] for name, record in records.items()
                },
            }
    return summary


def build_campaign(
    *,
    compiler_root: Path,
    hardware: HardwarePoint = HardwarePoint(),
    run_long: bool = False,
) -> dict[str, Any]:
    compiler = load_compiler_evidence(compiler_root)
    layout = build_layout_evidence(hardware, compiler_root)
    gpu = _cached_gpu_report()
    experiments = {}
    for model_name in ("nemotron3", "kimi_k3"):
        experiments[model_name] = {
            "prefill_s16": run_ablation(
                model_name,
                phase=InferencePhase.PREFILL,
                tokens=16,
                context_length=16,
                decode_tokens=1,
                hardware=hardware,
                compiler_root=compiler_root,
                compiler_evidence=compiler,
                layout_evidence=layout,
            ),
            "decode_4": run_ablation(
                model_name,
                phase=InferencePhase.DECODE,
                tokens=1,
                context_length=2048,
                decode_tokens=4,
                hardware=hardware,
                compiler_root=compiler_root,
                compiler_evidence=compiler,
                layout_evidence=layout,
            ),
        }
        if run_long:
            experiments[model_name]["prefill_s128"] = run_ablation(
                model_name,
                phase=InferencePhase.PREFILL,
                tokens=128,
                context_length=128,
                decode_tokens=1,
                hardware=hardware,
                compiler_root=compiler_root,
                compiler_evidence=compiler,
                layout_evidence=layout,
            )
            experiments[model_name]["decode_32"] = run_ablation(
                model_name,
                phase=InferencePhase.DECODE,
                tokens=1,
                context_length=2048,
                decode_tokens=32,
                hardware=hardware,
                compiler_root=compiler_root,
                compiler_evidence=compiler,
                layout_evidence=layout,
            )

    dse = run_dse(hardware, compiler_root=compiler_root, compiler_evidence=compiler)
    precision_dse = run_precision_dse(
        hardware,
        compiler_root=compiler_root,
        compiler_evidence=compiler,
        layout_evidence=layout,
        gpu_evidence=gpu,
    )
    decision = dse["base_decision"]
    report = {
        "schema_version": 2,
        "status": "compiler_simulator_pre_rtl",
        "claim_boundaries": {
            "dimensions": "official pinned real shapes and full 52/93-layer schedules",
            "weights": "symbolic performance execution; not full-checkpoint numerical execution",
            "cycles": "Compiler/Simulator estimate at an assumed clock, not RTL timing",
            "state": "ordinary tensors with explicit transfers; no cache/hit/miss/replacement",
            "state_multirow_layout": ("architecture upper bound only until a packetized consumer lowering executes it"),
            "projection_colayout_execution": (
                "Compiler placement + Python/Rust physical roundtrip and service model; "
                "candidate banked output SRAM is not current RTL"
            ),
            "shared_timeline": (
                "single-request dependency-ordered timeline with one shared Matrix, Vector, "
                "HBM and banked-output resource; no multi-request queueing claim"
            ),
        },
        "hardware": asdict(hardware),
        "resource_proxies": hardware.resource_proxies(),
        "compiler_report_sha256": _sha256_json(compiler),
        "compiler_summary": {
            "workloads": compiler["workloads"],
            "assembly": compiler["assembly"],
            "isa": compiler["isa"],
        },
        "layout_evidence": layout,
        "gpu_evidence": _gpu_summary(gpu),
        "experiments": experiments,
        "ablation_summary": summarize_ablation(experiments),
        "dse": dse,
        "precision_dse": precision_dse,
        "isa_freeze_status": {
            "stream_addressing": (
                "passes the pre-RTL functionality/performance gate on Mamba, KDA and generic "
                "SAXPY; RTL area/timing remains outside this work"
                if decision["stream_addressing_earns_isa"]
                else "fails the pre-RTL performance gate; retain ordinary post-increment"
            ),
            "affine_colayout": (
                "passes the executable incremental gate over stream-only addressing"
                if decision["affine_colayout_earns_isa_on_current_serial_consumer"]
                else "does not earn an architectural ISA mode on the current serial one-row "
                "consumer; keep the physical mapper as a packetized-candidate experiment only"
            ),
        },
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty campaign table: {path.name}")
    columns = sorted({column for row in rows for column in row})
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
                    for column, value in row.items()
                }
            )


def write_campaign_tables(report: dict[str, Any], output_dir: Path) -> None:
    """Write stable, flat tables for plots and independent review."""

    ablation_rows = []
    for model, experiments in report["experiments"].items():
        for experiment, result in experiments.items():
            for record in result["records"]:
                ablation_rows.append(
                    {
                        "model": model,
                        "experiment": experiment,
                        "phase": result["phase"],
                        "tokens": result["tokens"],
                        "variant": record["variant"],
                        "cycles": record["cycles"],
                        "speedup_vs_arlo_B": record["speedup_vs_arlo_B"],
                        "logical_hbm_read_bytes": record["logical_hbm_read_bytes"],
                        "logical_hbm_write_bytes": record["logical_hbm_write_bytes"],
                        "physical_hbm_read_bytes": record["physical_hbm_read_bytes"],
                        "physical_hbm_write_bytes": record["physical_hbm_write_bytes"],
                        "layout_service_cycles": record["layout_service_cycles"],
                        "bank_conflict_stall_cycles": record["bank_conflict_stall_cycles"],
                        "lane_restore_cycles": record["lane_restore_cycles"],
                        "fifo_stall_cycles": record["fifo_stall_cycles"],
                        "timeline_event_count": record["timeline_event_count"],
                        "resource_busy_cycles": record["resource_busy_cycles"],
                        "resource_queue_wait_cycles": record["resource_queue_wait_cycles"],
                    }
                )

    dse_rows = []
    for point in report["dse"]["records"]:
        for model, result in point["models"].items():
            dse_rows.append(
                {
                    "point": point["hardware"]["name"],
                    "model": model,
                    "eligible_for_freeze": point["eligible_for_freeze"],
                    **{f"hw_{name}": value for name, value in point["hardware"].items() if name != "name"},
                    **result,
                }
            )

    precision_rows = []
    for record in report["precision_dse"]["records"]:
        accuracy = record["accuracy"]
        precision_rows.append(
            {
                **{name: value for name, value in record.items() if name != "accuracy"},
                "accuracy_status": accuracy["status"],
                "accuracy": accuracy,
            }
        )

    schedule_rows = []
    for model, experiments in report["experiments"].items():
        for experiment, result in experiments.items():
            validation = result["schedule_validation"]
            schedule_rows.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "phase": result["phase"],
                    **validation,
                }
            )

    _write_csv(output_dir / "ablation.csv", ablation_rows)
    _write_csv(output_dir / "dse.csv", dse_rows)
    _write_csv(output_dir / "precision.csv", precision_rows)
    _write_csv(output_dir / "schedule_validation.csv", schedule_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=Path("PLENA_Compiler"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--long", action="store_true", help="also run S128 prefill and 32-token decode")
    args = parser.parse_args(argv)
    report = build_campaign(compiler_root=args.compiler_root, run_long=args.long)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    if args.csv_dir:
        write_campaign_tables(report, args.csv_dir)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
