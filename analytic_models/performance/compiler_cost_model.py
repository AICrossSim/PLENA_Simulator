"""Evaluate compiler-emitted symbolic costs with transactional semantics."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from compiler.aten.cost_emitter import (
    CostTrace,
    MemoryEvent,
    ScheduleRepeat,
    ScheduleSequence,
    ScheduleUnavailable,
    opcode_category,
)
from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    compile_native_decoder_cost_trace,
    load_cost_model_config,
)
from compiler.aten.isa_builder import DmaTransfer, RepeatAxis

from .hbm_cost import (
    CalibrationSample,
    HbmCalibration,
    dma_geometry,
    dma_geometry_classes,
    dma_stream_geometry,
    file_sha256,
    fit_hbm_calibration,
    hbm_formats_from_settings,
    load_transactional_toml,
)
from .ramulator_calibration import (
    fit_hbm_calibration_from_ramulator,
    write_dma_calibration_plan,
)
from .hbm_service_model import (
    HbmConfig,
    HbmServiceModel,
    MemoryFormat,
    MemoryPrecisionConfig,
    build_physical_memory_work,
)
from .hbm_service_calibration import (
    fit_hbm_service_model_from_ramulator,
    write_hbm_service_calibration_plan,
)
from .hbm_service_v4 import (
    HbmServiceModelV4,
    V4DmaServiceProvider,
    fit_hbm_service_v4,
    write_hbm_service_v4_plan,
)
from .rtl_opcode_timing import (
    DEFAULT_RTL_TIMING_CALIBRATION,
    ComputePrecisionConfig,
    ComputeWork,
    RtlOpcodeTimingCalibration,
    TimingHardware,
    aggregate_compute_work,
)
from .scheduled_shadow import ScheduledShadowResult, evaluate_scheduled_shadow


MATRIX_TILE_OPS = {"M_MM", "M_TMM", "M_BMM", "M_BTMM"}
MATRIX_VECTOR_OPS = {"M_MV", "M_TMV"}
MATRIX_BROADCAST_VECTOR_OPS = {"M_BMV", "M_BTMV"}
MATRIX_WRITE_OPS = {"M_MM_WO", "M_BMM_WO", "M_MV_WO", "M_BMV_WO"}
VECTOR_ADD_OPS = {"V_ADD_VV", "V_ADD_VF", "V_SUB_VV", "V_SUB_VF"}
VECTOR_MUL_OPS = {"V_MUL_VV", "V_MUL_VF", "V_SHIFT_V"}
SCALAR_FP_BASIC_OPS = {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP"}
SCALAR_INT_OPS = {
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
V4_MEMORY_EVALUATION_MODES = {
    "auto",
    "full-cached-occurrence",
    "full-global-stateful",
    "one-layer-cached-occurrence-scaled",
    "one-layer-stateful-scaled",
}
_V4_WORK_CACHE_LIMIT = 64
_V4_WORK_CACHE: OrderedDict[
    tuple[Any, ...], tuple[Any, Any]
] = OrderedDict()


def clear_v4_work_cache() -> None:
    """Clear cached aggregate V4 work used by warm DSE evaluations."""

    _V4_WORK_CACHE.clear()


def _used_memory_precision_cache_payload(
    trace: CostTrace,
    precision: MemoryPrecisionConfig,
) -> tuple[tuple[str, str, str], ...]:
    """Return only memory formats that can affect this trace's DMA geometry.

    ``MemoryPrecisionConfig`` also carries formats such as scalar integer data
    that may not occur in a given program. Including those unused formats in
    the persistent V4 key causes identical physical DMA manifests to be
    planned repeatedly during a hardware DSE. The event opcode is part of the
    identity because the generic ``kv`` alias resolves to matrix or vector KV
    storage according to the HBM opcode.
    """

    used: set[tuple[str, str, str]] = set()
    for event in trace.memory_events:
        transfer = event.transfer
        role = str(transfer.precision_role or transfer.precision)
        opcode = str(transfer.opcode)
        used.add((role, opcode, precision.for_role(role, opcode).request_signature()))
    return tuple(sorted(used))


def _v4_work_cache_key(
    trace: CostTrace,
    precision: MemoryPrecisionConfig,
    hbm: HbmConfig,
    service_model: HbmServiceModelV4,
    clock_period_ps: int,
    memory_mode: str,
) -> tuple[Any, ...] | None:
    config_hash = trace.metadata.get("config_hash")
    if not config_hash:
        return None
    return (
        "v4_work_v2_used_precision_roles",
        str(config_hash),
        int(trace.metadata.get("num_layers", 1)),
        _used_memory_precision_cache_payload(trace, precision),
        hbm,
        service_model.calibration_id,
        int(clock_period_ps),
        memory_mode,
    )


def _load_or_compute_persistent_v4_work(
    cache_dir: Path,
    cache_key: tuple[Any, ...],
    compute_work: Callable[[], tuple[Any, Any]],
) -> tuple[tuple[Any, Any], bool, str]:
    """Share deterministic aggregate V4 work across DSE worker processes.

    The existing in-process LRU is ineffective when workers are deliberately
    recycled to bound RSS. A per-key advisory lock ensures that exactly one
    process plans a missing entry. Completed entries are immutable and
    installed with ``os.replace``, so readers use an optimistic lock-free path
    instead of serializing all DSE workers behind an exclusive lock.
    """

    encoded_key = pickle.dumps(cache_key, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(encoded_key).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{digest}.pickle"
    lock_path = cache_dir / f"{digest}.lock"

    def load_cached() -> tuple[Any, Any]:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("schema") != "v4_work_v1":
            raise ValueError(f"invalid persistent V4 work cache {cache_path}")
        work = payload.get("work")
        if not isinstance(work, tuple) or len(work) != 2:
            raise ValueError(
                f"persistent V4 work cache {cache_path} has invalid payload"
            )
        return work

    # Writers publish only complete pickle files through atomic rename. An
    # existing path is therefore safe to read without taking the writer lock.
    if cache_path.exists():
        return load_cached(), True, digest

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.exists():
                return load_cached(), True, digest

            work = compute_work()
            temporary = cache_path.with_suffix(f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                pickle.dump(
                    {"schema": "v4_work_v1", "work": work},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary, cache_path)
            return work, False, digest
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _config_value(config: Mapping[str, Any], name: str) -> int:
    return int(config[name]["value"])


def _optional_config_value(
    config: Mapping[str, Any], name: str, default: int
) -> int:
    entry = config.get(name)
    return default if entry is None else int(entry["value"])


def _latency_value(settings: Mapping[str, Any], name: str, dc_en: int) -> int:
    key = "dc_lib_en" if dc_en else "dc_lib_dis"
    return int(settings["LATENCY"][name][key])


@dataclass(frozen=True)
class TransactionalCycleModel:
    settings_path: Path
    raw_settings: Mapping[str, Any]
    mlen: int
    blen: int
    vlen: int
    hlen: int
    broadcast_amount: int
    hbm_channels: int
    hbm_m_prefetch_amount: int
    hbm_v_prefetch_amount: int
    hbm_v_writeback_amount: int
    matrix_sram_size: int
    dc_en: int
    systolic_processing_overhead: int
    vector_add_cycles: int
    vector_mul_cycles: int
    vector_exp_cycles: int
    vector_reci_cycles: int
    vector_max_cycles: int
    vector_sum_cycles: int
    scalar_fp_basic_cycles: int
    scalar_fp_exp_cycles: int
    scalar_fp_sqrt_cycles: int
    scalar_fp_reci_cycles: int
    scalar_int_basic_cycles: int
    clock_period_ps: int = 1000

    @classmethod
    def load(cls, path: str | Path) -> TransactionalCycleModel:
        source = Path(path).resolve()
        settings = load_transactional_toml(source)
        config = settings["CONFIG"]
        dc_en = _config_value(config, "DC_EN")
        return cls(
            settings_path=source,
            raw_settings=settings,
            mlen=_config_value(config, "MLEN"),
            blen=_config_value(config, "BLEN"),
            vlen=_config_value(config, "VLEN"),
            hlen=_config_value(config, "HLEN"),
            broadcast_amount=_config_value(config, "BROADCAST_AMOUNT"),
            # Match the transactional Rust loader's historical default so an
            # older settings file remains usable without silently changing
            # any explicitly configured experiment.
            hbm_channels=_optional_config_value(config, "HBM_CHANNELS", 8),
            hbm_m_prefetch_amount=_config_value(config, "HBM_M_Prefetch_Amount"),
            hbm_v_prefetch_amount=_config_value(config, "HBM_V_Prefetch_Amount"),
            hbm_v_writeback_amount=_config_value(config, "HBM_V_Writeback_Amount"),
            matrix_sram_size=_config_value(config, "MATRIX_SRAM_SIZE"),
            dc_en=dc_en,
            systolic_processing_overhead=_latency_value(
                settings, "SYSTOLIC_PROCESSING_OVERHEAD", dc_en
            ),
            vector_add_cycles=_latency_value(settings, "VECTOR_ADD_CYCLES", dc_en),
            vector_mul_cycles=_latency_value(settings, "VECTOR_MUL_CYCLES", dc_en),
            vector_exp_cycles=_latency_value(settings, "VECTOR_EXP_CYCLES", dc_en),
            vector_reci_cycles=_latency_value(settings, "VECTOR_RECI_CYCLES", dc_en),
            vector_max_cycles=_latency_value(settings, "VECTOR_MAX_CYCLES", dc_en),
            vector_sum_cycles=_latency_value(settings, "VECTOR_SUM_CYCLES", dc_en),
            scalar_fp_basic_cycles=_latency_value(settings, "SCALAR_FP_BASIC_CYCLES", dc_en),
            scalar_fp_exp_cycles=_latency_value(settings, "SCALAR_FP_EXP_CYCLES", dc_en),
            scalar_fp_sqrt_cycles=_latency_value(settings, "SCALAR_FP_SQRT_CYCLES", dc_en),
            scalar_fp_reci_cycles=_latency_value(settings, "SCALAR_FP_RECI_CYCLES", dc_en),
            scalar_int_basic_cycles=_latency_value(settings, "SCALAR_INT_BASIC_CYCLES", dc_en),
            clock_period_ps=_optional_config_value(config, "CLOCK_PERIOD_PS", 1000),
        )

    @property
    def formats(self):
        return hbm_formats_from_settings(self.raw_settings)

    def instruction_cycles(self, opcode: str) -> int:
        """Return the historical functional-emulator delay for ``legacy`` mode."""
        if opcode in MATRIX_TILE_OPS:
            return self.systolic_processing_overhead + self.mlen
        if opcode in MATRIX_VECTOR_OPS:
            return self.mlen
        if opcode in MATRIX_BROADCAST_VECTOR_OPS:
            return self.systolic_processing_overhead + 1
        if opcode in MATRIX_WRITE_OPS:
            return 1
        if opcode in VECTOR_ADD_OPS:
            return self.vector_add_cycles
        if opcode in VECTOR_MUL_OPS:
            return self.vector_mul_cycles
        if opcode == "V_EXP_V":
            return self.vector_exp_cycles
        if opcode == "V_RECI_V":
            return self.vector_reci_cycles
        if opcode == "V_RED_MAX":
            return self.vector_max_cycles
        if opcode == "V_RED_SUM":
            return self.vector_sum_cycles
        if opcode in SCALAR_FP_BASIC_OPS:
            return self.scalar_fp_basic_cycles
        if opcode == "S_EXP_FP":
            return self.scalar_fp_exp_cycles
        if opcode == "S_RECI_FP":
            return self.scalar_fp_reci_cycles
        if opcode == "S_SQRT_FP":
            return self.scalar_fp_sqrt_cycles
        if opcode in {"S_LD_FP", "S_ST_FP"}:
            return 1
        if opcode == "S_MAP_V_FP":
            # vector_transfer_fp and the dispatch arm each call cycle!(VLEN).
            return 2 * self.vlen
        if opcode in SCALAR_INT_OPS:
            return self.scalar_int_basic_cycles
        if opcode in CONTROL_OPS:
            return 1
        if opcode in MEMORY_OPS:
            return 0
        raise ValueError(f"transactional cycle model has no semantics for opcode {opcode!r}")

    def assert_trace_compatible(self, trace: CostTrace) -> None:
        hardware = trace.metadata.get("hardware", {})
        expected = {
            "mlen": self.mlen,
            "blen": self.blen,
            "vlen": self.vlen,
            "hlen": self.hlen,
            "hbm_m_prefetch_amount": self.hbm_m_prefetch_amount,
            "hbm_v_prefetch_amount": self.hbm_v_prefetch_amount,
            "hbm_v_writeback_amount": self.hbm_v_writeback_amount,
            "hbm_channels": self.hbm_channels,
        }
        mismatches = [
            f"{name}: trace={hardware.get(name)!r}, settings={value!r}"
            for name, value in expected.items()
            if int(hardware.get(name, -1)) != value
        ]
        schedule = trace.metadata.get("attention_schedule", {})
        hardware_broadcast = schedule.get(
            "hardware_broadcast", schedule.get("physical_broadcast")
        )
        if hardware_broadcast is not None and int(hardware_broadcast) != self.broadcast_amount:
            mismatches.append(
                "hardware broadcast: "
                f"trace={hardware_broadcast!r}, settings={self.broadcast_amount!r}"
            )
        if mismatches:
            raise ValueError("cost trace and transactional settings are incompatible: " + "; ".join(mismatches))


@dataclass(frozen=True)
class MemoryCost:
    latency_ns: float
    read_requests: int
    write_requests: int
    read_bytes: int
    write_bytes: int
    payload_read_bytes: int
    payload_write_bytes: int
    opcode_latency_ns: Mapping[str, float]
    stage_latency_ns: Mapping[str, float]
    source_latency_ns: Mapping[str, float]


@dataclass(frozen=True)
class CompilerCostReport:
    compute_latency_ns: float
    hbm_latency_ns: float
    transactional_serial_latency_ns: float
    one_layer_latency_ns: float
    legacy_layer_x64_latency_ns: float
    true_full_model_latency_ns: float
    hbm_read_bytes: int
    hbm_write_bytes: int
    hbm_read_requests: int
    hbm_write_requests: int
    one_layer_hbm_read_bytes: int
    one_layer_hbm_write_bytes: int
    one_layer_hbm_read_requests: int
    one_layer_hbm_write_requests: int
    hbm_opcode_latency_ns: Mapping[str, float]
    one_layer_hbm_opcode_latency_ns: Mapping[str, float]
    hbm_stage_latency_ns: Mapping[str, float]
    one_layer_hbm_stage_latency_ns: Mapping[str, float]
    hbm_source_latency_ns: Mapping[str, float]
    one_layer_hbm_source_latency_ns: Mapping[str, float]
    category_latency_ns: Mapping[str, float]
    one_layer_category_latency_ns: Mapping[str, float]
    stage_latency_ns: Mapping[str, float]
    stage_compute_latency_ns: Mapping[str, float]
    stage_roofline_latency_ns: Mapping[str, float]
    stage_bound: Mapping[str, str]
    roofline_latency_ns: float
    memory_latency_ns: float
    serial_latency_ns: float
    calibration_in_domain: bool
    memory_model_version: str
    combination: str
    calibration_id: str
    compatibility: Mapping[str, Any]
    compute_resource_work_cycles: int = 0
    one_layer_compute_resource_work_cycles: int = 0
    compute_timing_mode: str = "legacy"
    compute_timing_semantics: str = "legacy_serial_opcode_delay"
    compute_timing_artifact: Mapping[str, Any] = field(default_factory=dict)
    compute_opcode_work_cycles: Mapping[str, int] = field(default_factory=dict)
    one_layer_compute_opcode_work_cycles: Mapping[str, int] = field(default_factory=dict)
    compute_validation: Mapping[str, Any] = field(default_factory=dict)
    one_layer_compute_validation: Mapping[str, Any] = field(default_factory=dict)
    compute_calibration_in_domain: bool = False
    legacy_compute_latency_ns: float = 0.0
    one_layer_legacy_compute_latency_ns: float = 0.0
    scheduled_shadow_makespan_cycles: int | None = None
    scheduled_shadow_latency_ns: float | None = None
    scheduled_shadow: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "one_layer_latency_ms": self.one_layer_latency_ns / 1e6,
                "legacy_layer_x64_latency_s": self.legacy_layer_x64_latency_ns / 1e9,
                "true_full_model_latency_s": self.true_full_model_latency_ns / 1e9,
                "transactional_serial_latency_s": self.transactional_serial_latency_ns / 1e9,
            }
        )
        return result


def _format_for_event(event: MemoryEvent, model: TransactionalCycleModel):
    formats = model.formats
    precision = event.transfer.precision
    aliases = {
        "kv": "matrix_kv" if event.transfer.opcode == "H_PREFETCH_M" else "vector_kv",
        "key_value": "matrix_kv" if event.transfer.opcode == "H_PREFETCH_M" else "vector_kv",
    }
    precision = aliases.get(precision, precision)
    try:
        return formats[precision]
    except KeyError as exc:
        raise ValueError(f"unknown DMA precision {event.transfer.precision!r}") from exc


def _dma_source_group(source: str) -> str:
    if not source:
        return "unspecified"
    domain, separator, detail = source.partition(":")
    if not separator:
        return domain
    if domain == "linear_projection":
        for name in ("W_q", "W_k", "W_v", "W_o", "R_rope"):
            if detail.startswith(name):
                return f"{domain}:{name}"
    if domain in {"packed_kv", "store_to_hbm"} and detail[:1] in {"K", "V"}:
        return f"{domain}:{detail[0]}"
    if domain in {"ffn", "load_batch", "packed_q_rope"}:
        return f"{domain}:{detail.split(':', 1)[0]}"
    return domain


def _memory_cost(
    events: Iterable[MemoryEvent],
    model: TransactionalCycleModel,
    calibration: HbmCalibration,
) -> MemoryCost:
    totals = Counter()
    opcode_latency: Counter[str] = Counter()
    stage_latency: Counter[str] = Counter()
    source_latency: Counter[str] = Counter()
    used_signatures = set()
    materialized = list(events)
    for event in materialized:
        used_signatures.add(_format_for_event(event, model).signature())
    calibration.assert_compatible(
        channels=model.hbm_channels,
        format_signatures=used_signatures,
        dma_semantics_hash=_current_dma_semantics_hash(),
        trace_schema_version=2,
    )
    previous_address = None
    for event in materialized:
        event_format = _format_for_event(event, model)
        if calibration.stream_models:
            axes = event.enclosing_axes
            if not axes and event.multiplicity > 1:
                axes = (RepeatAxis("stream_instruction", event.multiplicity),)
            stream = dma_stream_geometry(
                event.transfer,
                axes,
                event_format,
                previous_address=previous_address,
            )
            if stream.instruction_count != event.multiplicity:
                raise ValueError(
                    f"DMA stream {event.stream_index} geometry covers "
                    f"{stream.instruction_count} instructions, expected {event.multiplicity}"
                )
            latency = calibration.predict_stream_ns(
                event.transfer.opcode,
                stream,
                model.hbm_channels,
            )
            opcode_latency[event.transfer.opcode] += latency
            stage_latency[event.stage] += latency
            source_latency[_dma_source_group(event.transfer.source)] += latency
            totals["latency_ns"] += latency
            for name in (
                "read_requests",
                "write_requests",
                "read_bytes",
                "write_bytes",
                "payload_read_bytes",
                "payload_write_bytes",
            ):
                totals[name] += getattr(stream, name)
            previous_address = stream.final_element_base
            continue
        classes = (
            dma_geometry_classes(
                event.transfer,
                event.enclosing_axes,
                event_format,
            )
            if event.enclosing_axes
            else [(dma_geometry(event.transfer, event_format), event.multiplicity)]
        )
        class_count = sum(count for _, count in classes)
        if class_count != event.multiplicity:
            raise ValueError(
                f"DMA stream {event.stream_index} geometry classes cover {class_count} "
                f"instructions, expected {event.multiplicity}"
            )
        for geometry, multiplicity in classes:
            latency = (
                calibration.predict_ns(
                    event.transfer.opcode,
                    geometry,
                    model.hbm_channels,
                )
                * multiplicity
            )
            opcode_latency[event.transfer.opcode] += latency
            stage_latency[event.stage] += latency
            source_latency[_dma_source_group(event.transfer.source)] += latency
            totals["latency_ns"] += latency
            totals["read_requests"] += geometry.read_requests * multiplicity
            totals["write_requests"] += geometry.write_requests * multiplicity
            totals["read_bytes"] += geometry.read_bytes * multiplicity
            totals["write_bytes"] += geometry.write_bytes * multiplicity
            totals["payload_read_bytes"] += geometry.payload_read_bytes * multiplicity
            totals["payload_write_bytes"] += geometry.payload_write_bytes * multiplicity
        previous_address = int(event.transfer.element_base)
    return MemoryCost(
        latency_ns=float(totals["latency_ns"]),
        read_requests=int(totals["read_requests"]),
        write_requests=int(totals["write_requests"]),
        read_bytes=int(totals["read_bytes"]),
        write_bytes=int(totals["write_bytes"]),
        payload_read_bytes=int(totals["payload_read_bytes"]),
        payload_write_bytes=int(totals["payload_write_bytes"]),
        opcode_latency_ns=dict(opcode_latency),
        stage_latency_ns=dict(stage_latency),
        source_latency_ns=dict(source_latency),
    )


def _legacy_compute_work(
    counts: Mapping[str, int], model: TransactionalCycleModel
) -> ComputeWork:
    categories: Counter[str] = Counter()
    opcode_cycles: dict[str, int] = {}
    total_cycles = 0
    for opcode, raw_count in counts.items():
        count = int(raw_count)
        if opcode in MEMORY_OPS or count == 0:
            continue
        cycles = count * model.instruction_cycles(opcode)
        opcode_cycles[opcode] = cycles
        categories[opcode_category(opcode)] += cycles
        total_cycles += cycles
    cycle_to_ns = model.clock_period_ps / 1000.0
    return ComputeWork(
        resource_work_cycles=total_cycles,
        latency_ns=total_cycles * cycle_to_ns,
        category_cycles=dict(sorted(categories.items())),
        category_latency_ns={
            name: cycles * cycle_to_ns for name, cycles in sorted(categories.items())
        },
        opcode_cycles=dict(sorted(opcode_cycles.items())),
        validation={
            "status": "legacy",
            "total_opcodes": sum(
                int(count)
                for opcode, count in counts.items()
                if opcode not in MEMORY_OPS
            ),
            "calibration_in_domain": False,
        },
    )


@dataclass(frozen=True)
class ComputeTimingContext:
    mode: str
    precision: ComputePrecisionConfig | None
    calibration: RtlOpcodeTimingCalibration | None

    @property
    def semantics(self) -> str:
        if self.mode == "rtl-v1":
            return "serial_resource_work"
        return "legacy_serial_opcode_delay"

    @property
    def artifact(self) -> Mapping[str, Any]:
        return {} if self.calibration is None else self.calibration.metadata()

    def evaluate(
        self, counts: Mapping[str, int], model: TransactionalCycleModel
    ) -> ComputeWork:
        if self.mode == "legacy":
            return _legacy_compute_work(counts, model)
        assert self.calibration is not None
        assert self.precision is not None
        return aggregate_compute_work(
            counts,
            calibration=self.calibration,
            hardware=TimingHardware(
                model.mlen,
                model.blen,
                model.vlen,
                model.hlen,
                model.broadcast_amount,
            ),
            precision=self.precision,
            clock_period_ps=model.clock_period_ps,
            opcode_category=opcode_category,
        )


def _memory_precision_matches_settings(
    memory: MemoryPrecisionConfig,
    compute: ComputePrecisionConfig,
) -> list[str]:
    checks = {
        "weight": (memory.weight, compute.weight),
        "activation": (memory.activation, compute.activation),
        "matrix_kv": (memory.matrix_kv, compute.kv),
        "vector_kv": (memory.vector_kv, compute.kv),
    }
    mismatches = []
    for name, (memory_format, compute_format) in checks.items():
        if (
            memory_format.family != compute_format.family
            or memory_format.element_bits != compute_format.element_bits
            or memory_format.scale_bits != compute_format.scale_bits
            or memory_format.block != compute_format.block
        ):
            mismatches.append(
                f"{name}: memory={memory_format!r}, settings={compute_format!r}"
            )
    if memory.integer.element_bits != compute.integer_bits:
        mismatches.append(
            "integer: "
            f"memory={memory.integer.element_bits}, settings={compute.integer_bits}"
        )
    return mismatches


def _memory_precision_from_compute(
    precision: ComputePrecisionConfig,
) -> MemoryPrecisionConfig:
    def convert(value, name: str) -> MemoryFormat:
        return MemoryFormat(
            family=value.family,
            element_bits=value.element_bits,
            scale_bits=value.scale_bits,
            block=value.block,
            name=name,
        )

    kv = convert(precision.kv, "transactional_kv")
    return MemoryPrecisionConfig(
        weight=convert(precision.weight, "transactional_weight"),
        activation=convert(precision.activation, "transactional_activation"),
        matrix_kv=kv,
        vector_kv=kv,
        integer=MemoryFormat(
            "plain",
            precision.integer_bits,
            scale_bits=0,
            block=1,
            name=f"INT{precision.integer_bits}",
        ),
    )


def _build_compute_timing_context(
    model: TransactionalCycleModel,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any] | None,
    *,
    compute_timing_mode: str,
    rtl_timing_calibration: RtlOpcodeTimingCalibration | str | Path,
) -> ComputeTimingContext:
    if compute_timing_mode not in {"legacy", "rtl-v1"}:
        raise ValueError(
            "compute_timing_mode must be 'legacy' or 'rtl-v1', got "
            f"{compute_timing_mode!r}"
        )
    try:
        settings_precision = ComputePrecisionConfig.from_settings(model.raw_settings)
    except (KeyError, TypeError, ValueError):
        if compute_timing_mode == "rtl-v1" or precision_config is not None:
            raise
        return ComputeTimingContext(
            mode="legacy",
            precision=None,
            calibration=None,
        )
    explicit_compute = settings_precision
    if isinstance(precision_config, Mapping):
        explicit_compute = ComputePrecisionConfig.from_mapping(
            precision_config, fallback=settings_precision
        )
        mismatches = settings_precision.mismatch_messages(explicit_compute)
        if mismatches:
            raise ValueError(
                "explicit precision_config and transactional TOML disagree: "
                + "; ".join(mismatches)
            )
    if precision_config is not None:
        memory_precision = (
            precision_config
            if isinstance(precision_config, MemoryPrecisionConfig)
            else MemoryPrecisionConfig.from_mapping(precision_config)
        )
        mismatches = _memory_precision_matches_settings(
            memory_precision, settings_precision
        )
        if mismatches:
            raise ValueError(
                "memory precision_config and transactional TOML disagree: "
                + "; ".join(mismatches)
            )
    calibration = None
    if compute_timing_mode == "rtl-v1":
        calibration = (
            rtl_timing_calibration
            if isinstance(rtl_timing_calibration, RtlOpcodeTimingCalibration)
            else RtlOpcodeTimingCalibration.load(rtl_timing_calibration)
        )
    return ComputeTimingContext(
        mode=compute_timing_mode,
        precision=explicit_compute,
        calibration=calibration,
    )


def _one_layer_events(trace: CostTrace) -> list[MemoryEvent]:
    num_layers = int(trace.metadata.get("num_layers", 1))
    result = []
    for event in trace.memory_events:
        multiplicity = event.multiplicity
        if event.stage.startswith("layer/"):
            if multiplicity % num_layers:
                raise ValueError(
                    f"layer event multiplicity {multiplicity} is not divisible by num_layers={num_layers}"
                )
            multiplicity //= num_layers
        axes = tuple(
            axis for axis in event.enclosing_axes if axis.name != "decoder_layer"
        )
        result.append(
            MemoryEvent(
                stage=event.stage,
                transfer=event.transfer,
                multiplicity=multiplicity,
                enclosing_axes=axes,
                stream_index=event.stream_index,
            )
        )
    return result


def _one_layer_schedule(node: Any) -> Any:
    """Reduce model-layer repeats to one while preserving all other loops."""

    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(_one_layer_schedule(child) for child in node.children),
        )
    if isinstance(node, ScheduleRepeat):
        count = 1 if node.repeat_kind == "model_layer" else node.count
        return replace(
            node,
            count=count,
            body=_one_layer_schedule(node.body),
        )
    return node


def _one_layer_v4_trace(trace: CostTrace) -> CostTrace:
    """Build the exact one-layer DMA view used by the scalable V4 shadow.

    The compiler emits one ``model_layer`` repeat around the decoder body.
    Reducing that repeat and the matching event multiplicities keeps stream
    indices and affine transfer axes aligned, which the V4 provider checks
    before evaluating any occurrence.
    """

    metadata = dict(trace.metadata)
    metadata["num_layers"] = 1
    return CostTrace(
        memory_events=_one_layer_events(trace),
        schedule=_one_layer_schedule(trace.schedule),
        schedule_unavailable_reasons=Counter(trace.schedule_unavailable_reasons),
        metadata=metadata,
    )


@dataclass(frozen=True)
class ComputeReportData:
    total: ComputeWork
    one_layer: ComputeWork
    legacy_total: ComputeWork
    legacy_one_layer: ComputeWork
    stage_latency_ns: Mapping[str, float]


def _one_layer_counts(trace: CostTrace) -> Mapping[str, int]:
    counts = trace.metadata.get("one_layer_dynamic_opcodes")
    if counts is None:
        if int(trace.metadata.get("num_layers", 1)) != 1:
            raise ValueError(
                "multi-layer trace does not contain one_layer_dynamic_opcodes metadata"
            )
        return trace.dynamic_opcodes
    return counts


def _evaluate_compute(
    trace: CostTrace,
    model: TransactionalCycleModel,
    timing: ComputeTimingContext,
) -> ComputeReportData:
    one_layer_counts = _one_layer_counts(trace)
    return ComputeReportData(
        total=timing.evaluate(trace.dynamic_opcodes, model),
        one_layer=timing.evaluate(one_layer_counts, model),
        legacy_total=_legacy_compute_work(trace.dynamic_opcodes, model),
        legacy_one_layer=_legacy_compute_work(one_layer_counts, model),
        stage_latency_ns=_stage_compute_costs(trace, model, timing),
    )


def _compute_report_fields(
    compute: ComputeReportData,
    timing: ComputeTimingContext,
) -> dict[str, Any]:
    return {
        "compute_resource_work_cycles": compute.total.resource_work_cycles,
        "one_layer_compute_resource_work_cycles": (
            compute.one_layer.resource_work_cycles
        ),
        "compute_timing_mode": timing.mode,
        "compute_timing_semantics": timing.semantics,
        "compute_timing_artifact": dict(timing.artifact),
        "compute_opcode_work_cycles": dict(compute.total.opcode_cycles),
        "one_layer_compute_opcode_work_cycles": dict(
            compute.one_layer.opcode_cycles
        ),
        "compute_validation": dict(compute.total.validation),
        "one_layer_compute_validation": dict(compute.one_layer.validation),
        "compute_calibration_in_domain": bool(
            compute.total.validation.get("calibration_in_domain", False)
        ),
        "legacy_compute_latency_ns": compute.legacy_total.latency_ns,
        "one_layer_legacy_compute_latency_ns": (
            compute.legacy_one_layer.latency_ns
        ),
    }


def _scheduled_report_fields(
    result: ScheduledShadowResult,
    model: TransactionalCycleModel,
) -> dict[str, Any]:
    cycles = result.makespan_cycles
    return {
        "scheduled_shadow_makespan_cycles": cycles,
        "scheduled_shadow_latency_ns": (
            None if cycles is None else cycles * model.clock_period_ps / 1000.0
        ),
        "scheduled_shadow": result.to_dict(),
    }


class _ObservedDmaServiceProvider:
    """Serve either stream-indexed cycles or an exact Rust DMA event replay."""

    def __init__(
        self,
        values: Mapping[int, int | Sequence[int]]
        | Sequence[Mapping[str, Any]]
        | str
        | Path,
    ) -> None:
        if isinstance(values, (str, Path)):
            payload = json.loads(Path(values).read_text())
            values = payload.get("events", [])
        self.stream_values = (
            {
                int(stream): (
                    tuple(int(item) for item in value)
                    if isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    else int(value)
                )
                for stream, value in values.items()
            }
            if isinstance(values, Mapping)
            else None
        )
        self.events = None if isinstance(values, Mapping) else tuple(values)
        self.positions: Counter[int] = Counter()
        self.event_position = 0
        self.periods = {
            stream: self._fundamental_period(value)
            for stream, value in (self.stream_values or {}).items()
            if isinstance(value, tuple)
        }

    @staticmethod
    def _fundamental_period(values: Sequence[int]) -> int:
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

    @property
    def supports_exact_fast_forward(self) -> bool:
        return self.stream_values is not None and all(
            isinstance(value, tuple) for value in self.stream_values.values()
        )

    def snapshot_state(
        self, stream_indices: Sequence[int] | None = None
    ) -> tuple[tuple[int, int, int], ...]:
        if not self.supports_exact_fast_forward:
            return ()
        assert self.stream_values is not None
        selected = (
            sorted(self.stream_values)
            if stream_indices is None
            else sorted(int(stream) for stream in stream_indices)
        )
        return tuple(
            (
                stream,
                self.positions[stream],
                self.positions[stream] % max(1, self.periods[stream]),
            )
            for stream in selected
        )

    def advance_stream_counts(self, counts: Mapping[int, int]) -> None:
        if not self.supports_exact_fast_forward:
            raise ValueError("observed DMA provider cannot fast-forward event-list input")
        assert self.stream_values is not None
        for stream, count in counts.items():
            stream = int(stream)
            value = self.stream_values[stream]
            assert isinstance(value, tuple)
            next_position = self.positions[stream] + int(count)
            if next_position > len(value):
                raise ValueError(
                    f"observed DMA stream {stream} fast-forwarded to "
                    f"{next_position}/{len(value)}"
                )
            self.positions[stream] = next_position

    def __call__(self, instruction, _sequence: int) -> int:
        if self.stream_values is not None:
            stream_index = instruction.memory_stream_index
            if stream_index is None or stream_index not in self.stream_values:
                raise ValueError(
                    f"no observed DMA completion interval for stream {stream_index!r}"
                )
            value = self.stream_values[stream_index]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                position = self.positions[stream_index]
                if position >= len(value):
                    raise ValueError(
                        f"observed DMA stream {stream_index} has only "
                        f"{len(value)} intervals"
                    )
                self.positions[stream_index] += 1
                return int(value[position])
            return int(value)

        assert self.events is not None
        if self.event_position >= len(self.events):
            raise ValueError(
                "CostEmitter schedule contains more DMA instructions than the "
                f"observed trace ({len(self.events)})"
            )
        event = self.events[self.event_position]
        self.event_position += 1
        expected_opcode = str(event["opcode"])
        if instruction.opcode != expected_opcode:
            raise ValueError(
                "observed DMA opcode order differs from CostEmitter schedule at "
                f"position {self.event_position - 1}: expected {expected_opcode}, "
                f"got {instruction.opcode}"
            )
        start = int(event["start_cycle"])
        completion = int(event["completion_cycle"])
        if completion < start:
            raise ValueError(
                f"observed DMA event completes before it starts: {event!r}"
            )
        return max(1, completion - start)

    def assert_consumed(self) -> None:
        if self.events is not None and self.event_position != len(self.events):
            raise ValueError(
                "CostEmitter schedule consumed "
                f"{self.event_position}/{len(self.events)} observed DMA events"
            )
        if self.stream_values is not None:
            missing = {
                stream: len(value) - self.positions[stream]
                for stream, value in self.stream_values.items()
                if isinstance(value, tuple) and self.positions[stream] != len(value)
            }
            if missing:
                raise ValueError(
                    f"observed DMA stream intervals were not fully consumed: {missing}"
                )


def _actual_dma_service_provider(
    values: Mapping[int, int | Sequence[int]]
    | Sequence[Mapping[str, Any]]
    | str
    | Path,
) -> _ObservedDmaServiceProvider:
    return _ObservedDmaServiceProvider(values)


def _v3_dma_service_provider(
    trace: CostTrace,
    precision: MemoryPrecisionConfig,
    hbm: HbmConfig,
    service_model: HbmServiceModel,
    clock_period_ps: int,
):
    per_stream_cycles: dict[int, int] = {}
    for event in trace.memory_events:
        event_trace = CostTrace(
            memory_events=[event],
            metadata=dict(trace.metadata),
        )
        event_work = build_physical_memory_work(event_trace, precision, hbm)
        event_latency_ns = service_model.predict(event_work).latency_ns
        per_instruction_ns = event_latency_ns / max(1, event.multiplicity)
        per_stream_cycles[event.stream_index] = max(
            1, math.ceil(per_instruction_ns * 1000.0 / clock_period_ps)
        )

    def provider(instruction, _sequence: int) -> int:
        stream_index = instruction.memory_stream_index
        if stream_index is None or stream_index not in per_stream_cycles:
            raise ValueError(f"no V3 DMA estimate for stream {stream_index!r}")
        return per_stream_cycles[stream_index]

    return provider


def _evaluate_scheduled_shadow(
    trace: CostTrace,
    model: TransactionalCycleModel,
    timing: ComputeTimingContext,
    *,
    enabled: bool,
    hbm_service_cycles=None,
    hbm_fidelity: str = "unavailable",
) -> ScheduledShadowResult:
    if not enabled:
        return ScheduledShadowResult(
            status="disabled",
            fidelity=hbm_fidelity,
            makespan_cycles=None,
            events=(),
            stall_cycles_by_reason={},
            resource_work_cycles={},
            validation={},
            reason="scheduled shadow was not requested",
        )
    if timing.mode != "rtl-v1" or timing.calibration is None or timing.precision is None:
        return ScheduledShadowResult(
            status="schedule_unavailable",
            fidelity=hbm_fidelity,
            makespan_cycles=None,
            events=(),
            stall_cycles_by_reason={},
            resource_work_cycles={},
            validation={},
            reason="scheduled shadow requires compute_timing_mode='rtl-v1'",
        )
    expansion_limit = int(
        os.environ.get("PLENA_SCHEDULE_MAX_EXPANDED_INSTRUCTIONS", "2000000")
    )
    if expansion_limit <= 0:
        raise ValueError(
            "PLENA_SCHEDULE_MAX_EXPANDED_INSTRUCTIONS must be positive"
        )
    return evaluate_scheduled_shadow(
        trace,
        hardware=TimingHardware(
            model.mlen,
            model.blen,
            model.vlen,
            model.hlen,
            model.broadcast_amount,
        ),
        precision=timing.precision,
        calibration=timing.calibration,
        hbm_service_cycles=hbm_service_cycles,
        hbm_fidelity=hbm_fidelity,
        # Observed service times are instruction-specific, so validation must
        # replay the complete ordered schedule rather than fast-forwarding a
        # repeat with an averaged/provider-external state. Events are not
        # retained, keeping memory usage bounded by scoreboard state.
        max_expanded_instructions=(
            max(2_000_000, trace.dynamic_instruction_count + 1)
            if hbm_fidelity == "ramulator_observed"
            else expansion_limit
        ),
    )


def _stage_compute_costs(
    trace: CostTrace,
    model: TransactionalCycleModel,
    timing: ComputeTimingContext,
) -> dict[str, float]:
    return {
        stage_name: timing.evaluate(stage.dynamic_opcodes, model).latency_ns
        for stage_name, stage in trace.stages.items()
    }


def _stage_roofline(
    stage_compute: Mapping[str, float],
    stage_memory: Mapping[str, float],
    *,
    bound_ratio: float = 1.2,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    serial = {}
    roofline = {}
    bounds = {}
    for stage in sorted(set(stage_compute) | set(stage_memory)):
        compute = float(stage_compute.get(stage, 0.0))
        memory = float(stage_memory.get(stage, 0.0))
        serial[stage] = compute + memory
        roofline[stage] = max(compute, memory)
        if compute == 0.0 and memory == 0.0:
            bounds[stage] = "balanced"
        elif memory >= bound_ratio * compute:
            bounds[stage] = "memory_bound"
        elif compute >= bound_ratio * memory:
            bounds[stage] = "compute_bound"
        else:
            bounds[stage] = "balanced"
    return serial, roofline, bounds


def _load_memory_backend(
    value: HbmCalibration | HbmServiceModel | HbmServiceModelV4 | str | Path,
) -> HbmCalibration | HbmServiceModel | HbmServiceModelV4:
    if isinstance(value, (HbmCalibration, HbmServiceModel, HbmServiceModelV4)):
        return value
    path = Path(value)
    data = json.loads(path.read_text())
    if int(data.get("schema_version", -1)) == 3:
        return HbmServiceModel.load(path)
    if int(data.get("schema_version", -1)) == 4:
        return HbmServiceModelV4.load(path)
    return HbmCalibration.load(path)


def _evaluate_v3(
    trace: CostTrace,
    model: TransactionalCycleModel,
    service_model: HbmServiceModel,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any],
    timing: ComputeTimingContext,
    *,
    combination: str,
    scheduled_shadow_enabled: bool,
    scheduled_dma_completion_cycles: Mapping[int, int | Sequence[int]]
    | Sequence[Mapping[str, Any]]
    | str
    | Path
    | None,
) -> CompilerCostReport:
    if combination != "serial":
        raise ValueError(f"V3 primary latency currently requires combination='serial', got {combination!r}")
    precision = (
        precision_config
        if isinstance(precision_config, MemoryPrecisionConfig)
        else MemoryPrecisionConfig.from_mapping(precision_config)
    )
    hbm = HbmConfig(channels=model.hbm_channels)
    work = build_physical_memory_work(trace, precision, hbm)
    memory = service_model.predict(work)
    compute = _evaluate_compute(trace, model, timing)
    compute_ns = compute.total.latency_ns
    categories = dict(compute.total.category_latency_ns)
    categories["memory"] = memory.latency_ns

    one_compute_ns = compute.one_layer.latency_ns
    one_categories = dict(compute.one_layer.category_latency_ns)
    one_trace_metadata = dict(trace.metadata)
    one_trace_metadata["num_layers"] = 1
    one_trace = CostTrace(memory_events=_one_layer_events(trace), metadata=one_trace_metadata)
    one_work = build_physical_memory_work(one_trace, precision, hbm)
    one_memory = service_model.predict(one_work)
    one_categories["memory"] = one_memory.latency_ns

    stage_compute = dict(compute.stage_latency_ns)
    stage_serial, stage_roofline, stage_bound = _stage_roofline(
        stage_compute, memory.stage_latency_ns
    )
    schedule_has_order = not trace.schedule_unavailable_reasons
    if scheduled_dma_completion_cycles is not None:
        dma_provider = _actual_dma_service_provider(
            scheduled_dma_completion_cycles
        )
        hbm_fidelity = "ramulator_observed"
    elif scheduled_shadow_enabled and schedule_has_order:
        dma_provider = _v3_dma_service_provider(
            trace,
            precision,
            hbm,
            service_model,
            model.clock_period_ps,
        )
        hbm_fidelity = "post_hoc_v3"
    else:
        dma_provider = None
        hbm_fidelity = "post_hoc_v3"
    scheduled = _evaluate_scheduled_shadow(
        trace,
        model,
        timing,
        enabled=scheduled_shadow_enabled,
        hbm_service_cycles=dma_provider,
        hbm_fidelity=hbm_fidelity,
    )
    if (
        isinstance(dma_provider, _ObservedDmaServiceProvider)
        and scheduled.status == "complete"
    ):
        dma_provider.assert_consumed()
    serial_ns = compute_ns + memory.latency_ns
    one_layer_ns = one_compute_ns + one_memory.latency_ns
    return CompilerCostReport(
        compute_latency_ns=compute_ns,
        hbm_latency_ns=memory.latency_ns,
        transactional_serial_latency_ns=serial_ns,
        one_layer_latency_ns=one_layer_ns,
        legacy_layer_x64_latency_ns=one_layer_ns * 64,
        true_full_model_latency_ns=serial_ns,
        hbm_read_bytes=work.read_bytes,
        hbm_write_bytes=work.write_bytes,
        hbm_read_requests=work.read_requests,
        hbm_write_requests=work.write_requests,
        one_layer_hbm_read_bytes=one_work.read_bytes,
        one_layer_hbm_write_bytes=one_work.write_bytes,
        one_layer_hbm_read_requests=one_work.read_requests,
        one_layer_hbm_write_requests=one_work.write_requests,
        hbm_opcode_latency_ns=dict(memory.opcode_latency_ns),
        one_layer_hbm_opcode_latency_ns=dict(one_memory.opcode_latency_ns),
        hbm_stage_latency_ns=dict(memory.stage_latency_ns),
        one_layer_hbm_stage_latency_ns=dict(one_memory.stage_latency_ns),
        hbm_source_latency_ns={},
        one_layer_hbm_source_latency_ns={},
        category_latency_ns=dict(sorted(categories.items())),
        one_layer_category_latency_ns=dict(sorted(one_categories.items())),
        stage_latency_ns=stage_serial,
        stage_compute_latency_ns=dict(sorted(stage_compute.items())),
        stage_roofline_latency_ns=stage_roofline,
        stage_bound=stage_bound,
        roofline_latency_ns=sum(stage_roofline.values()),
        memory_latency_ns=memory.latency_ns,
        serial_latency_ns=serial_ns,
        calibration_in_domain=memory.calibration_in_domain,
        memory_model_version="global_v3",
        combination=combination,
        calibration_id=service_model.calibration_id,
        compatibility={
            "settings": str(model.settings_path),
            "config_hash": trace.metadata.get("config_hash"),
            "compiler_revision": trace.metadata.get("compiler_revision"),
            "trace_schema_version": 4,
            "hbm_memory_trace_schema_version": 3,
            "precision_config": precision.to_dict(),
            "hbm_service_model": dict(service_model.compatibility),
            "calibration_in_domain": memory.calibration_in_domain,
            "domain_issues": list(memory.domain_issues),
            "theoretical_floor_ns": memory.theoretical_floor_ns,
            "compute_precision_config": (
                {} if timing.precision is None else timing.precision.to_dict()
            ),
            "clock_period_ps": model.clock_period_ps,
        },
        **_compute_report_fields(compute, timing),
        **_scheduled_report_fields(scheduled, model),
    )


def _evaluate_v4(
    trace: CostTrace,
    model: TransactionalCycleModel,
    service_model: HbmServiceModelV4,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any],
    timing: ComputeTimingContext,
    *,
    combination: str,
    scheduled_shadow_enabled: bool,
    scheduled_dma_completion_cycles: Mapping[int, int | Sequence[int]]
    | Sequence[Mapping[str, Any]]
    | str
    | Path
    | None,
    memory_evaluation_mode: str,
    use_work_cache: bool,
    persistent_work_cache_dir: Path | None,
) -> CompilerCostReport:
    """Evaluate V4 as serial occurrence work plus an optional overlap shadow."""

    if combination != "serial":
        raise ValueError(
            "V4 primary latency currently requires combination='serial', "
            f"got {combination!r}"
        )
    precision = (
        precision_config
        if isinstance(precision_config, MemoryPrecisionConfig)
        else MemoryPrecisionConfig.from_mapping(precision_config)
    )
    if memory_evaluation_mode not in V4_MEMORY_EVALUATION_MODES:
        raise ValueError(
            "V4 memory_evaluation_mode must be one of "
            f"{sorted(V4_MEMORY_EVALUATION_MODES)}, got {memory_evaluation_mode!r}"
        )
    num_layers = int(trace.metadata.get("num_layers", 1))
    if num_layers <= 0:
        raise ValueError(f"trace num_layers must be positive, got {num_layers}")
    if memory_evaluation_mode == "auto":
        if scheduled_shadow_enabled or scheduled_dma_completion_cycles is not None:
            effective_memory_mode = "full-global-stateful"
        elif bool(trace.metadata.get("latency_only", False)):
            effective_memory_mode = "one-layer-cached-occurrence-scaled"
        elif num_layers == 1:
            effective_memory_mode = "full-global-stateful"
        else:
            effective_memory_mode = "one-layer-stateful-scaled"
    else:
        effective_memory_mode = memory_evaluation_mode
    if (
        effective_memory_mode
        in {"one-layer-stateful-scaled", "one-layer-cached-occurrence-scaled"}
        and (scheduled_shadow_enabled or scheduled_dma_completion_cycles is not None)
    ):
        raise ValueError(
            "one-layer-stateful-scaled V4 cannot drive scheduled replay; "
            "use full-global-stateful"
        )

    hbm = HbmConfig(channels=model.hbm_channels)
    provider_trace = (
        _one_layer_v4_trace(trace)
        if effective_memory_mode
        in {"one-layer-stateful-scaled", "one-layer-cached-occurrence-scaled"}
        else trace
    )
    cache_allowed = bool(
        use_work_cache
        and not scheduled_shadow_enabled
        and scheduled_dma_completion_cycles is None
    )
    cache_key = (
        _v4_work_cache_key(
            trace,
            precision,
            hbm,
            service_model,
            model.clock_period_ps,
            effective_memory_mode,
        )
        if cache_allowed
        else None
    )
    cached_work = _V4_WORK_CACHE.get(cache_key) if cache_key is not None else None
    work_cache_hit = cached_work is not None
    persistent_work_cache_hit = False
    persistent_work_cache_key: str | None = None
    work_provider: V4DmaServiceProvider | None = None
    if cached_work is not None:
        _V4_WORK_CACHE.move_to_end(cache_key)  # type: ignore[arg-type]
        memory, one_memory = cached_work
    else:
        def compute_memory_work() -> tuple[Any, Any]:
            nonlocal work_provider
            work_provider = V4DmaServiceProvider(
                provider_trace,
                precision,
                hbm,
                service_model,
                model.clock_period_ps,
                prepare_global_row_state=(
                    effective_memory_mode
                    not in {
                        "full-cached-occurrence",
                        "one-layer-cached-occurrence-scaled",
                    }
                ),
            )

            if effective_memory_mode in {
                "one-layer-stateful-scaled",
                "one-layer-cached-occurrence-scaled",
            }:
                stage_multipliers = {
                    event.stage: num_layers
                    for event in provider_trace.memory_events
                    if event.stage.startswith("layer/")
                }
                one = work_provider.aggregate()
                total = work_provider.aggregate(stage_multipliers=stage_multipliers)
            else:
                total = work_provider.aggregate()
                one = total

            if num_layers == 1:
                # A one-layer validation trace already is its own one-layer
                # view. Replanning every physical line would double runtime.
                one = total
            elif effective_memory_mode == "full-global-stateful":
                one_trace = _one_layer_v4_trace(trace)
                one = V4DmaServiceProvider(
                    one_trace,
                    precision,
                    hbm,
                    service_model,
                    model.clock_period_ps,
                ).aggregate()
            return total, one

        if (
            persistent_work_cache_dir is not None
            and cache_key is not None
            and cache_allowed
        ):
            (
                (memory, one_memory),
                persistent_work_cache_hit,
                persistent_work_cache_key,
            ) = _load_or_compute_persistent_v4_work(
                persistent_work_cache_dir,
                cache_key,
                compute_memory_work,
            )
            work_cache_hit = persistent_work_cache_hit
        else:
            memory, one_memory = compute_memory_work()
        if cache_key is not None:
            _V4_WORK_CACHE[cache_key] = (memory, one_memory)
            _V4_WORK_CACHE.move_to_end(cache_key)
            while len(_V4_WORK_CACHE) > _V4_WORK_CACHE_LIMIT:
                _V4_WORK_CACHE.popitem(last=False)

    compute = _evaluate_compute(trace, model, timing)
    compute_ns = compute.total.latency_ns
    one_compute_ns = compute.one_layer.latency_ns
    categories = dict(compute.total.category_latency_ns)
    categories["memory"] = memory.latency_ns
    one_categories = dict(compute.one_layer.category_latency_ns)
    one_categories["memory"] = one_memory.latency_ns
    stage_compute = dict(compute.stage_latency_ns)
    stage_serial, stage_roofline, stage_bound = _stage_roofline(
        stage_compute, memory.stage_latency_ns
    )

    schedule_has_order = not trace.schedule_unavailable_reasons
    if scheduled_dma_completion_cycles is not None:
        dma_provider = _actual_dma_service_provider(
            scheduled_dma_completion_cycles
        )
        hbm_fidelity = "ramulator_observed"
    elif scheduled_shadow_enabled and schedule_has_order:
        # ``aggregate`` populated the exact per-stream occurrence sequences;
        # reuse this provider so the scheduler can fast-forward only across a
        # proven repeating timing phase.
        if work_provider is None:
            raise AssertionError("scheduled V4 evaluation unexpectedly reused work cache")
        dma_provider = work_provider
        hbm_fidelity = (
            "post_hoc_v4_cached_occurrence"
            if effective_memory_mode == "full-cached-occurrence"
            else "post_hoc_v4"
        )
    else:
        dma_provider = None
        hbm_fidelity = "post_hoc_v4"
    scheduled = _evaluate_scheduled_shadow(
        trace,
        model,
        timing,
        enabled=scheduled_shadow_enabled,
        hbm_service_cycles=dma_provider,
        hbm_fidelity=hbm_fidelity,
    )
    if scheduled.status == "complete" and isinstance(
        dma_provider, (_ObservedDmaServiceProvider, V4DmaServiceProvider)
    ):
        dma_provider.assert_consumed()

    serial_ns = compute_ns + memory.latency_ns
    one_layer_ns = one_compute_ns + one_memory.latency_ns
    return CompilerCostReport(
        compute_latency_ns=compute_ns,
        hbm_latency_ns=memory.latency_ns,
        transactional_serial_latency_ns=serial_ns,
        one_layer_latency_ns=one_layer_ns,
        legacy_layer_x64_latency_ns=one_layer_ns * 64,
        true_full_model_latency_ns=serial_ns,
        hbm_read_bytes=memory.read_bytes,
        hbm_write_bytes=memory.write_bytes,
        hbm_read_requests=memory.read_requests,
        hbm_write_requests=memory.write_requests,
        one_layer_hbm_read_bytes=one_memory.read_bytes,
        one_layer_hbm_write_bytes=one_memory.write_bytes,
        one_layer_hbm_read_requests=one_memory.read_requests,
        one_layer_hbm_write_requests=one_memory.write_requests,
        hbm_opcode_latency_ns=dict(memory.opcode_latency_ns),
        one_layer_hbm_opcode_latency_ns=dict(one_memory.opcode_latency_ns),
        hbm_stage_latency_ns=dict(memory.stage_latency_ns),
        one_layer_hbm_stage_latency_ns=dict(one_memory.stage_latency_ns),
        hbm_source_latency_ns={},
        one_layer_hbm_source_latency_ns={},
        category_latency_ns=dict(sorted(categories.items())),
        one_layer_category_latency_ns=dict(sorted(one_categories.items())),
        stage_latency_ns=stage_serial,
        stage_compute_latency_ns=dict(sorted(stage_compute.items())),
        stage_roofline_latency_ns=stage_roofline,
        stage_bound=stage_bound,
        roofline_latency_ns=sum(stage_roofline.values()),
        memory_latency_ns=memory.latency_ns,
        serial_latency_ns=serial_ns,
        calibration_in_domain=memory.calibration_in_domain,
        memory_model_version=(
            "production_dma_v4"
            if service_model.metadata.get("promotion_status") == "accepted"
            else "production_dma_v4_candidate"
        ),
        combination=combination,
        calibration_id=service_model.calibration_id,
        compatibility={
            "settings": str(model.settings_path),
            "config_hash": trace.metadata.get("config_hash"),
            "compiler_revision": trace.metadata.get("compiler_revision"),
            "trace_schema_version": 4,
            "hbm_memory_trace_schema_version": 4,
            "precision_config": precision.to_dict(),
            "hbm_service_model": dict(service_model.compatibility),
            "calibration_in_domain": memory.calibration_in_domain,
            "domain_issues": list(memory.domain_issues),
            "max_extrapolation_ratio": memory.max_extrapolation_ratio,
            "theoretical_floor_ns": memory.theoretical_floor_ns,
            "occurrence_count": memory.occurrence_count,
            "row_state_regime_counts": dict(memory.row_state_regime_counts),
            "per_occurrence_prediction": (
                effective_memory_mode
                in {"full-global-stateful", "full-cached-occurrence"}
            ),
            "memory_evaluation_mode": effective_memory_mode,
            "memory_evaluation_requested": memory_evaluation_mode,
            "memory_layer_scale": (
                num_layers
                if effective_memory_mode
                in {
                    "one-layer-stateful-scaled",
                    "one-layer-cached-occurrence-scaled",
                }
                else 1
            ),
            "dma_row_state_runtime": (
                work_provider.row_state_semantics
                if work_provider is not None
                else (
                    "cold_geometry_cached_occurrence"
                    if effective_memory_mode
                    in {
                        "full-cached-occurrence",
                        "one-layer-cached-occurrence-scaled",
                    }
                    else "cached_aggregate_work"
                )
            ),
            "v4_work_cache_hit": work_cache_hit,
            "v4_work_cache_enabled": cache_allowed,
            "v4_work_cache_key_version": "v4_work_v2_used_precision_roles",
            "v4_persistent_work_cache_enabled": bool(
                persistent_work_cache_dir is not None and cache_allowed
            ),
            "v4_persistent_work_cache_hit": persistent_work_cache_hit,
            "v4_persistent_work_cache_key": persistent_work_cache_key,
            "runtime_geometry_cache": (
                "exact_manifest_plus_translation_invariant_cold_feature_cache"
            ),
            "compute_precision_config": (
                {} if timing.precision is None else timing.precision.to_dict()
            ),
            "clock_period_ps": model.clock_period_ps,
        },
        **_compute_report_fields(compute, timing),
        **_scheduled_report_fields(scheduled, model),
    )


def evaluate_compiler_cost(
    trace: CostTrace,
    transactional_settings: TransactionalCycleModel | str | Path,
    hbm_calibration: HbmCalibration | HbmServiceModel | HbmServiceModelV4 | str | Path,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any] | None = None,
    *,
    combination: str = "serial",
    compute_timing_mode: str = "rtl-v1",
    rtl_timing_calibration: RtlOpcodeTimingCalibration
    | str
    | Path = DEFAULT_RTL_TIMING_CALIBRATION,
    scheduled_shadow: bool = False,
    scheduled_dma_completion_cycles: Mapping[int, int | Sequence[int]]
    | Sequence[Mapping[str, Any]]
    | str
    | Path
    | None = None,
    v4_memory_evaluation: str = "auto",
    use_v4_work_cache: bool = True,
    persistent_v4_work_cache_dir: str | Path | None = None,
) -> CompilerCostReport:
    """Evaluate a compressed trace without rendering or executing ISA.

    The calibrated serial resource-work estimate is the production objective.
    Ordered schedule replay is intentionally opt-in because exact observed-DMA
    validation expands the dynamic trace.  Accepted V4 artifacts use a clearly
    labelled one-layer stateful scaling mode for multi-layer aggregate DSE
    shadows unless full replay is explicitly requested.
    """
    model = (
        transactional_settings
        if isinstance(transactional_settings, TransactionalCycleModel)
        else TransactionalCycleModel.load(transactional_settings)
    )
    calibration = _load_memory_backend(hbm_calibration)
    model.assert_trace_compatible(trace)
    timing = _build_compute_timing_context(
        model,
        precision_config,
        compute_timing_mode=compute_timing_mode,
        rtl_timing_calibration=rtl_timing_calibration,
    )

    if isinstance(calibration, HbmServiceModel):
        if precision_config is None and timing.precision is None:
            raise ValueError(
                "V3 memory evaluation requires complete transactional precision settings"
            )
        memory_precision = (
            _memory_precision_from_compute(timing.precision)  # type: ignore[arg-type]
            if precision_config is None
            else precision_config
        )
        return _evaluate_v3(
            trace,
            model,
            calibration,
            memory_precision,
            timing,
            combination=combination,
            scheduled_shadow_enabled=scheduled_shadow,
            scheduled_dma_completion_cycles=scheduled_dma_completion_cycles,
        )
    if isinstance(calibration, HbmServiceModelV4):
        if precision_config is None and timing.precision is None:
            raise ValueError(
                "V4 memory evaluation requires complete transactional precision settings"
            )
        memory_precision = (
            _memory_precision_from_compute(timing.precision)  # type: ignore[arg-type]
            if precision_config is None
            else precision_config
        )
        return _evaluate_v4(
            trace,
            model,
            calibration,
            memory_precision,
            timing,
            combination=combination,
            scheduled_shadow_enabled=scheduled_shadow,
            scheduled_dma_completion_cycles=scheduled_dma_completion_cycles,
            memory_evaluation_mode=v4_memory_evaluation,
            use_work_cache=use_v4_work_cache,
            persistent_work_cache_dir=(
                None
                if persistent_v4_work_cache_dir is None
                else Path(persistent_v4_work_cache_dir)
            ),
        )
    if combination != "serial":
        raise ValueError("legacy V1/V2 HBM models support only serial combination")

    compute = _evaluate_compute(trace, model, timing)
    compute_ns = compute.total.latency_ns
    categories = dict(compute.total.category_latency_ns)
    memory = _memory_cost(trace.memory_events, model, calibration)
    categories["memory"] = memory.latency_ns

    one_compute_ns = compute.one_layer.latency_ns
    one_categories = dict(compute.one_layer.category_latency_ns)
    one_memory = _memory_cost(_one_layer_events(trace), model, calibration)
    one_categories["memory"] = one_memory.latency_ns
    one_layer_ns = one_compute_ns + one_memory.latency_ns

    stage_compute = dict(compute.stage_latency_ns)
    stage_latency, stage_roofline, stage_bound = _stage_roofline(
        stage_compute, memory.stage_latency_ns
    )

    total_ns = compute_ns + memory.latency_ns
    dma_provider = (
        None
        if scheduled_dma_completion_cycles is None
        else _actual_dma_service_provider(scheduled_dma_completion_cycles)
    )
    scheduled = _evaluate_scheduled_shadow(
        trace,
        model,
        timing,
        enabled=scheduled_shadow,
        hbm_service_cycles=dma_provider,
        hbm_fidelity=(
            "ramulator_observed"
            if scheduled_dma_completion_cycles is not None
            else "unavailable"
        ),
    )
    if (
        isinstance(dma_provider, _ObservedDmaServiceProvider)
        and scheduled.status == "complete"
    ):
        dma_provider.assert_consumed()
    return CompilerCostReport(
        compute_latency_ns=compute_ns,
        hbm_latency_ns=memory.latency_ns,
        transactional_serial_latency_ns=total_ns,
        one_layer_latency_ns=one_layer_ns,
        legacy_layer_x64_latency_ns=one_layer_ns * 64,
        true_full_model_latency_ns=total_ns,
        hbm_read_bytes=memory.read_bytes,
        hbm_write_bytes=memory.write_bytes,
        hbm_read_requests=memory.read_requests,
        hbm_write_requests=memory.write_requests,
        one_layer_hbm_read_bytes=one_memory.read_bytes,
        one_layer_hbm_write_bytes=one_memory.write_bytes,
        one_layer_hbm_read_requests=one_memory.read_requests,
        one_layer_hbm_write_requests=one_memory.write_requests,
        hbm_opcode_latency_ns=dict(sorted(memory.opcode_latency_ns.items())),
        one_layer_hbm_opcode_latency_ns=dict(
            sorted(one_memory.opcode_latency_ns.items())
        ),
        hbm_stage_latency_ns=dict(sorted(memory.stage_latency_ns.items())),
        one_layer_hbm_stage_latency_ns=dict(sorted(one_memory.stage_latency_ns.items())),
        hbm_source_latency_ns=dict(sorted(memory.source_latency_ns.items())),
        one_layer_hbm_source_latency_ns=dict(sorted(one_memory.source_latency_ns.items())),
        category_latency_ns=dict(sorted(categories.items())),
        one_layer_category_latency_ns=dict(sorted(one_categories.items())),
        stage_latency_ns=dict(sorted(stage_latency.items())),
        stage_compute_latency_ns=dict(sorted(stage_compute.items())),
        stage_roofline_latency_ns=dict(sorted(stage_roofline.items())),
        stage_bound=dict(sorted(stage_bound.items())),
        roofline_latency_ns=sum(stage_roofline.values()),
        memory_latency_ns=memory.latency_ns,
        serial_latency_ns=total_ns,
        calibration_in_domain=True,
        memory_model_version="legacy_v2" if calibration.stream_models else "legacy_v1",
        combination=combination,
        calibration_id=calibration.calibration_id,
        compatibility={
            "settings": str(model.settings_path),
            "config_hash": trace.metadata.get("config_hash"),
            "compiler_revision": trace.metadata.get("compiler_revision"),
            "calibration": dict(calibration.compatibility),
            "compute_precision_config": (
                {} if timing.precision is None else timing.precision.to_dict()
            ),
            "clock_period_ps": model.clock_period_ps,
        },
        **_compute_report_fields(compute, timing),
        **_scheduled_report_fields(scheduled, model),
    )


def _current_dma_semantics_hash() -> str | None:
    root = Path(__file__).resolve().parents[2]
    source = root / "transactional_emulator" / "src" / "dma.rs"
    return file_sha256(source) if source.exists() else None


def _portable_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _profile_transfer(opcode: str, model: TransactionalCycleModel) -> DmaTransfer:
    if opcode == "H_PREFETCH_M":
        return DmaTransfer(
            opcode=opcode,
            direction="read",
            precision="matrix",
            element_base=0,
            scale_base=0,
            dim=model.mlen,
            amount=model.hbm_m_prefetch_amount,
            stride=model.mlen,
            write_amount=model.mlen,
        )
    if opcode == "H_PREFETCH_V":
        return DmaTransfer(
            opcode=opcode,
            direction="read",
            precision="activation",
            element_base=0,
            scale_base=0,
            dim=model.vlen,
            amount=model.hbm_v_prefetch_amount,
            stride=model.vlen,
        )
    if opcode == "H_STORE_V":
        return DmaTransfer(
            opcode=opcode,
            direction="write",
            precision="activation",
            element_base=0,
            scale_base=0,
            dim=model.vlen,
            amount=model.hbm_v_writeback_amount,
            stride=model.vlen,
        )
    raise ValueError(f"unsupported profile DMA opcode {opcode}")


def calibration_samples_from_emulator_runs(
    run_directories: Sequence[str | Path],
) -> tuple[list[CalibrationSample], dict[str, Any], dict[str, Any]]:
    samples = []
    channels = set()
    signatures = set()
    sources = []
    for directory_value in run_directories:
        directory = Path(directory_value).resolve()
        profile_path = directory / "memory_profile.json"
        stats_path = directory / "rust_emulator_run_stats.json"
        if not profile_path.exists() or not stats_path.exists():
            raise FileNotFoundError(f"{directory} must contain memory_profile.json and rust_emulator_run_stats.json")
        profile = json.loads(profile_path.read_text())
        stats = json.loads(stats_path.read_text())
        settings_path = Path(stats["config_path"])
        if not settings_path.exists():
            settings_path = directory / "plena_settings.toml"
        model = TransactionalCycleModel.load(settings_path)
        channels.add(model.hbm_channels)
        signatures.update(fmt.signature() for fmt in model.formats.values())
        expected_read = 0
        expected_write = 0
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
            opcode_profile = profile["opcodes"][opcode]
            count = int(opcode_profile["count"])
            transfer = _profile_transfer(opcode, model)
            precision = "matrix" if opcode == "H_PREFETCH_M" else "activation"
            geometry = dma_geometry(transfer, model.formats[precision])
            expected_read += geometry.read_bytes * count
            expected_write += geometry.write_bytes * count
            samples.append(
                CalibrationSample(
                    opcode=opcode,
                    geometry=geometry,
                    channels=model.hbm_channels,
                    observed_ns=float(opcode_profile["total_ns"]) / count,
                    source=_portable_path(directory),
                )
            )
        actual_read = int(profile["hbm_bytes_read"])
        actual_write = int(profile["hbm_bytes_written"])
        if (expected_read, expected_write) != (actual_read, actual_write):
            raise ValueError(
                f"DMA geometry mismatch for {directory}: predicted "
                f"read/write={expected_read}/{expected_write}, profile={actual_read}/{actual_write}"
            )
        sources.append(
            {
                "run_directory": _portable_path(directory),
                "profile_sha256": file_sha256(profile_path),
                "settings_sha256": file_sha256(settings_path),
            }
        )
    compatibility = {
        "ramulator_preset": "HBM2_2Gbps",
        "request_bytes": 64,
        "channels": sorted(channels),
        "precision_signatures": sorted(signatures),
        "dma_semantics_hash": _current_dma_semantics_hash() or "unknown",
    }
    metadata = {"seed": 20260711, "sources": sources, "source_kind": "transactional_emulator_ramulator"}
    return samples, compatibility, metadata


def fit_hbm_calibration_from_runs(
    run_directories: Sequence[str | Path], *, ridge: float = 1e-8
) -> HbmCalibration:
    samples, compatibility, metadata = calibration_samples_from_emulator_runs(run_directories)
    return fit_hbm_calibration(
        samples,
        ridge=ridge,
        compatibility=compatibility,
        metadata=metadata,
    )


def _fit_nonnegative_affine(inputs: Sequence[float], targets: Sequence[float]) -> tuple[float, float]:
    if len(inputs) != len(targets) or not inputs:
        raise ValueError("nonnegative affine fit requires equally sized nonempty inputs and targets")
    candidates = [(0.0, 0.0)]
    candidates.append((sum(targets) / len(targets), 0.0))
    squared_input = sum(value * value for value in inputs)
    if squared_input:
        candidates.append((0.0, sum(x * y for x, y in zip(inputs, targets, strict=True)) / squared_input))
    count = len(inputs)
    sum_input = sum(inputs)
    sum_target = sum(targets)
    denominator = count * squared_input - sum_input * sum_input
    if denominator:
        scale = (
            count * sum(x * y for x, y in zip(inputs, targets, strict=True))
            - sum_input * sum_target
        ) / denominator
        bias = (sum_target - scale * sum_input) / count
        if bias >= 0 and scale >= 0:
            candidates.append((bias, scale))
    return min(
        candidates,
        key=lambda pair: sum(
            (pair[0] + pair[1] * value - target) ** 2
            for value, target in zip(inputs, targets, strict=True)
        ),
    )


def calibrate_hbm_integration(
    base_calibration: HbmCalibration | str | Path,
    run_directories: Sequence[str | Path],
) -> HbmCalibration:
    """Fit full-emulator integration overhead on top of standalone Ramulator timing."""
    base = (
        base_calibration
        if isinstance(base_calibration, HbmCalibration)
        else HbmCalibration.load(base_calibration)
    )
    samples, _, run_metadata = calibration_samples_from_emulator_runs(run_directories)
    grouped: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.opcode, []).append(sample)
    corrections = {}
    models = {}
    for opcode, model in base.models.items():
        opcode_samples = grouped.get(opcode, [])
        if not opcode_samples:
            raise ValueError(f"integration calibration has no samples for {opcode}")
        inputs = [
            model.predict_ns(sample.geometry, sample.channels) for sample in opcode_samples
        ]
        targets = [sample.observed_ns for sample in opcode_samples]
        bias, scale = _fit_nonnegative_affine(inputs, targets)
        models[opcode] = replace(model, integration_bias_ns=bias, integration_scale=scale)
        corrections[opcode] = {
            "bias_ns": bias,
            "scale": scale,
            "sample_count": len(opcode_samples),
        }
    identity = {
        "base_calibration_id": base.calibration_id,
        "corrections": corrections,
        "sources": run_metadata["sources"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return HbmCalibration(
        calibration_id=f"hbm-hybrid-{digest}",
        models=models,
        compatibility=dict(base.compatibility),
        metadata={
            **dict(base.metadata),
            "source_kind": "ramulator_dma_plus_transactional_integration",
            "base_calibration_id": base.calibration_id,
            "integration_corrections": corrections,
            "integration_sources": run_metadata["sources"],
        },
    )


def compare_report_to_profile(
    report: CompilerCostReport,
    profile_path: str | Path,
    trace: CostTrace | None = None,
) -> dict[str, Any]:
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    actual = float(profile["program_total_ns"])
    predicted = report.one_layer_latency_ns
    category_error = {}
    for category, values in profile["categories"].items():
        actual_category = float(values["total_ns"])
        predicted_category = float(report.one_layer_category_latency_ns.get(category, 0.0))
        category_error[category] = {
            "actual_ns": actual_category,
            "predicted_ns": predicted_category,
            "error_percent": 100.0 * (predicted_category - actual_category) / actual_category,
        }
    comparison = {
        "actual_ns": actual,
        "predicted_ns": predicted,
        "absolute_error_ns": predicted - actual,
        "error_percent": 100.0 * (predicted - actual) / actual,
        "hbm_read_bytes_match": report.one_layer_hbm_read_bytes == int(profile["hbm_bytes_read"]),
        "hbm_write_bytes_match": report.one_layer_hbm_write_bytes
        == int(profile["hbm_bytes_written"]),
        "categories": category_error,
    }
    if trace is not None:
        predicted_opcodes = trace.metadata.get("one_layer_dynamic_opcodes", trace.dynamic_opcodes)
        actual_opcodes = {opcode: int(values["count"]) for opcode, values in profile["opcodes"].items()}
        all_opcodes = sorted(set(predicted_opcodes) | set(actual_opcodes))
        mismatches = {
            opcode: {
                "predicted": int(predicted_opcodes.get(opcode, 0)),
                "actual": int(actual_opcodes.get(opcode, 0)),
            }
            for opcode in all_opcodes
            if int(predicted_opcodes.get(opcode, 0)) != int(actual_opcodes.get(opcode, 0))
        }
        comparison["opcode_counts_match"] = not mismatches
        comparison["opcode_count_mismatches"] = mismatches
        stats_path = profile_path.with_name("rust_emulator_run_stats.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            actual_static = int(stats["artifacts"]["machine_code_lines"])
            predicted_static = sum(trace.metadata.get("one_layer_static_opcodes", {}).values())
            comparison["static_machine_instructions"] = {
                "predicted": predicted_static,
                "actual": actual_static,
                "match": predicted_static == actual_static,
            }
    return comparison


def _hardware_from_settings(model_config: Any, settings: TransactionalCycleModel) -> CompilerCostHardware:
    model, _ = load_cost_model_config(model_config)
    logical_broadcast = model.num_heads // model.num_kv_heads
    return CompilerCostHardware(
        mlen=settings.mlen,
        blen=settings.blen,
        vlen=settings.vlen,
        hlen=settings.hlen,
        broadcast_amount=logical_broadcast,
        mram_tile_capacity=settings.matrix_sram_size // settings.mlen,
        hbm_m_prefetch_amount=settings.hbm_m_prefetch_amount,
        hbm_v_prefetch_amount=settings.hbm_v_prefetch_amount,
        hbm_v_writeback_amount=settings.hbm_v_writeback_amount,
        hbm_channels=settings.hbm_channels,
    )


@lru_cache(maxsize=1)
def _compiler_trace_source_fingerprint() -> str:
    """Hash compiler Python sources that can change a symbolic cost trace."""

    compiler_root = Path(__file__).resolve().parents[2] / "PLENA_Compiler" / "aten"
    digest = hashlib.sha256()
    for path in sorted(compiler_root.rglob("*.py")):
        digest.update(str(path.relative_to(compiler_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _routing_cache_fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else str(value)
    if isinstance(value, Mapping):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    else:
        payload = repr(value)
    return hashlib.sha256(payload.encode()).hexdigest()


def _persistent_trace_cache_key(
    model_config: Any,
    hardware: CompilerCostHardware,
    *,
    seq_len: int,
    batch_size: int,
    num_layers: int | None,
    layer_idx: int,
    moe_routing_mode: str,
    moe_routing_plan: Any,
    max_static_routes: int,
    moe_layer_scaling: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    vector_scalar_schedule: str,
) -> str:
    model, configured_layers = load_cost_model_config(model_config)
    payload = {
        "schema": "persistent_unscheduled_cost_trace_v1",
        "compiler_source": _compiler_trace_source_fingerprint(),
        "model": asdict(model),
        "configured_layers": configured_layers,
        "hardware": asdict(hardware),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "layer_idx": layer_idx,
        "moe_routing_mode": moe_routing_mode,
        "moe_routing_plan": _routing_cache_fingerprint(moe_routing_plan),
        "max_static_routes": max_static_routes,
        "moe_layer_scaling": moe_layer_scaling,
        "native_layout_mode": native_layout_mode,
        "packed_attention_schedule": packed_attention_schedule,
        "vector_scalar_schedule": vector_scalar_schedule,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strip_ordered_schedule_for_persistent_cache(trace: CostTrace) -> CostTrace:
    """Drop schedule replay state while preserving counts, stages, and DMA."""

    trace.schedule = ScheduleSequence(
        (
            ScheduleUnavailable(
                reason="persistent_unscheduled_trace_cache",
                stage="global",
                dynamic_instruction_count=trace.dynamic_instruction_count,
            ),
        )
    )
    trace.schedule_unavailable_reasons = Counter(
        {"persistent_unscheduled_trace_cache": 1}
    )
    trace.metadata["persistent_trace_schedule"] = "counts_and_dma_only"
    return trace


def _load_or_compile_persistent_trace(
    cache_dir: Path,
    cache_key: str,
    compile_trace: Callable[[], CostTrace],
) -> CostTrace:
    """Load one shape trace or compile it once across all DSE processes.

    Cache files are immutable after an atomic rename. Existing traces can be
    read concurrently; only the cache-miss compilation path needs the per-key
    exclusive lock.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pickle"
    lock_path = cache_dir / f"{cache_key}.lock"

    def load_cached() -> CostTrace:
        with cache_path.open("rb") as handle:
            trace = pickle.load(handle)
        if not isinstance(trace, CostTrace):
            raise TypeError(
                f"persistent trace cache {cache_path} did not contain CostTrace"
            )
        trace.metadata["persistent_trace_cache_hit"] = True
        trace.metadata["persistent_trace_cache_key"] = cache_key
        return trace

    if cache_path.exists():
        return load_cached()

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.exists():
                return load_cached()

            trace = _strip_ordered_schedule_for_persistent_cache(compile_trace())
            trace.metadata["persistent_trace_cache_hit"] = False
            trace.metadata["persistent_trace_cache_key"] = cache_key
            temporary = cache_path.with_suffix(f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_path)
            return trace
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def compile_and_evaluate_compiler_cost(
    model_config: Any,
    transactional_settings: TransactionalCycleModel | str | Path,
    hbm_calibration: HbmCalibration | HbmServiceModel | HbmServiceModelV4 | str | Path,
    *,
    seq_len: int,
    batch_size: int,
    num_layers: int | None = None,
    layer_idx: int = 0,
    moe_routing_mode: str = "static-indices",
    moe_routing_plan: Any = None,
    max_static_routes: int = 1024,
    moe_layer_scaling: str = "single-layer",
    native_layout_mode: str = "compact",
    packed_attention_schedule: str = "direct-first-block-v1",
    vector_scalar_schedule: str = "compiler-v1",
    precision_config: MemoryPrecisionConfig | Mapping[str, Any] | None = None,
    compute_timing_mode: str = "rtl-v1",
    rtl_timing_calibration: RtlOpcodeTimingCalibration
    | str
    | Path = DEFAULT_RTL_TIMING_CALIBRATION,
    scheduled_shadow: bool = False,
    scheduled_dma_completion_cycles: Mapping[int, int | Sequence[int]]
    | Sequence[Mapping[str, Any]]
    | str
    | Path
    | None = None,
    v4_memory_evaluation: str = "auto",
    use_trace_cache: bool = True,
    use_v4_work_cache: bool = True,
    persistent_trace_cache_dir: str | Path | None = None,
    persistent_v4_work_cache_dir: str | Path | None = None,
) -> tuple[CostTrace, CompilerCostReport]:
    """Compile and evaluate a dense, static-index, or fixed-balanced Qwen3 point."""
    settings = (
        transactional_settings
        if isinstance(transactional_settings, TransactionalCycleModel)
        else TransactionalCycleModel.load(transactional_settings)
    )
    hardware = _hardware_from_settings(model_config, settings)
    def compile_trace() -> CostTrace:
        return compile_native_decoder_cost_trace(
            model_config,
            hardware,
            seq_len=seq_len,
            batch_size=batch_size,
            num_layers=num_layers,
            layer_idx=layer_idx,
            moe_routing_mode=moe_routing_mode,
            moe_routing_plan=moe_routing_plan,
            max_static_routes=max_static_routes,
            moe_layer_scaling=moe_layer_scaling,
            native_layout_mode=native_layout_mode,
            packed_attention_schedule=packed_attention_schedule,
            vector_scalar_schedule=vector_scalar_schedule,
            use_cache=use_trace_cache,
        )

    if persistent_trace_cache_dir is not None and not scheduled_shadow:
        persistent_key = _persistent_trace_cache_key(
            model_config,
            hardware,
            seq_len=seq_len,
            batch_size=batch_size,
            num_layers=num_layers,
            layer_idx=layer_idx,
            moe_routing_mode=moe_routing_mode,
            moe_routing_plan=moe_routing_plan,
            max_static_routes=max_static_routes,
            moe_layer_scaling=moe_layer_scaling,
            native_layout_mode=native_layout_mode,
            packed_attention_schedule=packed_attention_schedule,
            vector_scalar_schedule=vector_scalar_schedule,
        )
        trace = _load_or_compile_persistent_trace(
            Path(persistent_trace_cache_dir), persistent_key, compile_trace
        )
    else:
        trace = compile_trace()
    return trace, evaluate_compiler_cost(
        trace,
        settings,
        hbm_calibration,
        precision_config,
        compute_timing_mode=compute_timing_mode,
        rtl_timing_calibration=rtl_timing_calibration,
        scheduled_shadow=scheduled_shadow,
        scheduled_dma_completion_cycles=scheduled_dma_completion_cycles,
        v4_memory_evaluation=v4_memory_evaluation,
        use_v4_work_cache=use_v4_work_cache,
        persistent_v4_work_cache_dir=persistent_v4_work_cache_dir,
    )


def validate_hbm_service_v4_system_case(
    trace: CostTrace,
    transactional_settings: TransactionalCycleModel | str | Path,
    service_model: HbmServiceModelV4 | str | Path,
    precision_config: MemoryPrecisionConfig | Mapping[str, Any],
    observed_dma_trace: str | Path | Mapping[str, Any],
    *,
    compute_timing_mode: str = "rtl-v1",
    rtl_timing_calibration: RtlOpcodeTimingCalibration
    | str
    | Path = DEFAULT_RTL_TIMING_CALIBRATION,
) -> dict[str, Any]:
    """Compare V4 occurrence work and scheduled shadow to observed DMA replay."""

    settings = (
        transactional_settings
        if isinstance(transactional_settings, TransactionalCycleModel)
        else TransactionalCycleModel.load(transactional_settings)
    )
    calibration = (
        service_model
        if isinstance(service_model, HbmServiceModelV4)
        else HbmServiceModelV4.load(service_model)
    )
    observed_payload = (
        dict(observed_dma_trace)
        if isinstance(observed_dma_trace, Mapping)
        else json.loads(Path(observed_dma_trace).read_text())
    )
    observed_events = tuple(observed_payload.get("events", ()))
    if not observed_events:
        raise ValueError("observed DMA trace contains no events")

    predicted = evaluate_compiler_cost(
        trace,
        settings,
        calibration,
        precision_config,
        compute_timing_mode=compute_timing_mode,
        rtl_timing_calibration=rtl_timing_calibration,
        scheduled_shadow=True,
    )
    # The observed comparison changes only DMA service intervals.  Re-running
    # the complete V4 occurrence aggregation and compute resource-work pass
    # would produce byte-for-byte identical primary fields, so replay the
    # ordered schedule directly with the measured completion intervals.
    timing = _build_compute_timing_context(
        settings,
        precision_config,
        compute_timing_mode=compute_timing_mode,
        rtl_timing_calibration=rtl_timing_calibration,
    )
    # Re-index the globally ordered replay intervals by physical compiler
    # stream. This preserves exact per-occurrence values while allowing the
    # compressed scheduler to fast-forward only proven periodic phases.
    ordering_provider = V4DmaServiceProvider(
        trace,
        precision_config,
        HbmConfig(channels=settings.hbm_channels),
        calibration,
        settings.clock_period_ps,
    )
    ordered_occurrences = tuple(ordering_provider.ordered_estimates())
    if len(ordered_occurrences) != len(observed_events):
        raise ValueError(
            "observed DMA trace occurrence count differs from CostTrace: "
            f"observed={len(observed_events)}, expected={len(ordered_occurrences)}"
        )
    observed_stream_cycles: dict[int, list[int]] = defaultdict(list)
    for index, (event, occurrence) in enumerate(
        zip(observed_events, ordered_occurrences, strict=True)
    ):
        stream = occurrence[0]
        opcode = str(event["opcode"])
        if opcode != stream.opcode:
            raise ValueError(
                "observed DMA opcode order differs from CostTrace at "
                f"{index}: observed={opcode}, expected={stream.opcode}"
            )
        start = int(event["start_cycle"])
        completion = int(event["completion_cycle"])
        if completion < start:
            raise ValueError(f"observed DMA completes before start: {event!r}")
        observed_stream_cycles[stream.stream_index].append(
            max(1, completion - start)
        )
    observed_provider = _actual_dma_service_provider(
        {stream: tuple(values) for stream, values in observed_stream_cycles.items()}
    )
    observed_scheduled = _evaluate_scheduled_shadow(
        trace,
        settings,
        timing,
        enabled=True,
        hbm_service_cycles=observed_provider,
        hbm_fidelity="ramulator_observed",
    )
    if observed_scheduled.status == "complete":
        observed_provider.assert_consumed()

    observed_opcode_cycles: Counter[str] = Counter()
    for event in observed_events:
        start = int(event["start_cycle"])
        completion = int(event["completion_cycle"])
        if completion < start:
            raise ValueError(f"observed DMA completes before start: {event!r}")
        observed_opcode_cycles[str(event["opcode"])] += max(1, completion - start)
    observed_opcode_ns = {
        opcode: cycles * settings.clock_period_ps / 1000.0
        for opcode, cycles in sorted(observed_opcode_cycles.items())
    }

    def error_percent(prediction: float, reference: float) -> float:
        return 100.0 * abs(prediction - reference) / max(reference, 1.0)

    opcode_rows = {}
    for opcode in sorted(
        set(predicted.hbm_opcode_latency_ns) | set(observed_opcode_ns)
    ):
        predicted_ns = float(predicted.hbm_opcode_latency_ns.get(opcode, 0.0))
        observed_ns = float(observed_opcode_ns.get(opcode, 0.0))
        opcode_rows[opcode] = {
            "predicted_work_ns": predicted_ns,
            "observed_work_ns": observed_ns,
            "absolute_error_percent": error_percent(predicted_ns, observed_ns),
        }
    predicted_total = sum(predicted.hbm_opcode_latency_ns.values())
    observed_total = sum(observed_opcode_ns.values())
    predicted_makespan = predicted.scheduled_shadow_makespan_cycles
    observed_makespan = observed_scheduled.makespan_cycles
    makespan_error = (
        None
        if predicted_makespan is None or observed_makespan is None
        else error_percent(float(predicted_makespan), float(observed_makespan))
    )
    acceptance = {
        "per_opcode_hbm_work_le_25pct": bool(opcode_rows)
        and all(row["absolute_error_percent"] <= 25.0 for row in opcode_rows.values()),
        "total_hbm_work_le_20pct": error_percent(predicted_total, observed_total)
        <= 20.0,
        "scheduled_makespan_le_10pct": makespan_error is not None
        and makespan_error <= 10.0,
    }
    return {
        "schema_version": 4,
        "calibration_id": calibration.calibration_id,
        "dma_semantic_version": calibration.compatibility.get(
            "dma_semantic_version"
        ),
        "observed_dma_timing_semantics": observed_payload.get(
            "dma_timing_semantics",
            "functional-executor-service-interval-replayed-on-rtl-v1",
        ),
        "opcode_work": opcode_rows,
        "total_hbm_work": {
            "predicted_ns": predicted_total,
            "observed_ns": observed_total,
            "absolute_error_percent": error_percent(predicted_total, observed_total),
        },
        "scheduled_makespan": {
            "predicted_cycles": predicted_makespan,
            "observed_cycles": observed_makespan,
            "absolute_error_percent": makespan_error,
            "predicted_status": predicted.scheduled_shadow.get("status"),
            "observed_status": observed_scheduled.status,
            "predicted_reason": predicted.scheduled_shadow.get("reason"),
            "observed_reason": observed_scheduled.reason,
            "predicted_validation": predicted.scheduled_shadow.get("validation", {}),
            "observed_validation": dict(observed_scheduled.validation),
        },
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "correctness_gate_modified": False,
        "numerical_execution_modified": False,
        "model_domain": predicted.compatibility.get("domain_issues", []),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="compile and evaluate a Qwen3 cost trace")
    evaluate.add_argument("--model-config", type=Path, required=True)
    evaluate.add_argument("--settings", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--seq-len", type=int, required=True)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--num-layers", type=int)
    evaluate.add_argument("--layer-idx", type=int, default=0)
    evaluate.add_argument(
        "--moe-routing-mode",
        choices=("static-indices", "fixed-balanced"),
        default="static-indices",
        help="MoE route source; fixed-balanced is a latency-only aggregate",
    )
    evaluate.add_argument(
        "--moe-routing-plan",
        type=Path,
        help="Static-index MoeRoutingPlan JSON required for a selected MoE layer",
    )
    evaluate.add_argument("--max-static-routes", type=int, default=1024)
    evaluate.add_argument(
        "--moe-layer-scaling",
        choices=("single-layer", "repeat-static-plan", "repeat-fixed-balanced"),
        default="single-layer",
    )
    evaluate.add_argument(
        "--native-layout-mode",
        choices=("compact", "legacy"),
        default="compact",
        help="native decoder row/head storage layout (default: compact)",
    )
    evaluate.add_argument(
        "--packed-attention-schedule",
        choices=("direct-first-block-v1", "legacy"),
        default="direct-first-block-v1",
        help=(
            "packed-GQA online-softmax/output schedule; the optimized default "
            "specializes the first K block and accumulates directly into packed O"
        ),
    )
    evaluate.add_argument(
        "--vector-scalar-schedule",
        choices=("compiler-v1", "legacy"),
        default="compiler-v1",
        help="native Vector/Scalar lowering schedule (default: compiler-v1)",
    )
    evaluate.add_argument(
        "--precision-config",
        type=Path,
        help="MemoryPrecisionConfig JSON; defaults to active transactional precision",
    )
    evaluate.add_argument(
        "--compute-timing",
        choices=("rtl-v1", "legacy"),
        default="rtl-v1",
        help="compute timing source; rtl-v1 reports calibrated serial resource work",
    )
    evaluate.add_argument(
        "--rtl-timing-calibration",
        type=Path,
        default=DEFAULT_RTL_TIMING_CALIBRATION,
    )
    evaluate.add_argument(
        "--scheduled-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "evaluate the ordered hazard/overlap shadow; disabled by default "
            "because production Qwen replay is a validation path, not the DSE objective"
        ),
    )
    evaluate.add_argument(
        "--v4-memory-evaluation",
        choices=tuple(sorted(V4_MEMORY_EVALUATION_MODES)),
        default="auto",
        help=(
            "V4 aggregate fidelity: auto scales one stateful decoder layer for "
            "multi-layer non-scheduled runs and retains full global state for "
            "system/scheduled validation"
        ),
    )
    evaluate.add_argument(
        "--scheduled-dma-completion-trace",
        type=Path,
        help=(
            "compact transactional rtl-v1 DMA event JSON for exact observed "
            "schedule validation; requires --scheduled-shadow"
        ),
    )
    evaluate.add_argument("--profile", type=Path)
    evaluate.add_argument("--trace-output", type=Path)
    evaluate.add_argument("--output", type=Path)
    calibrate = subparsers.add_parser("calibrate", help="fit a surrogate from emulator run directories")
    calibrate.add_argument("--run-dir", type=Path, action="append", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ridge", type=float, default=1e-8)
    plan = subparsers.add_parser(
        "generate-calibration-plan", help="write compressed DMA patterns for the Rust Ramulator driver"
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--repetitions", type=int, default=3)
    plan.add_argument("--warmup", type=int, default=1)
    plan.add_argument("--seed", type=int, default=20260711)
    fit_ramulator = subparsers.add_parser(
        "fit-ramulator", help="fit and validate a surrogate from standalone Ramulator results"
    )
    fit_ramulator.add_argument("--plan", type=Path, required=True)
    fit_ramulator.add_argument("--results", type=Path, required=True)
    fit_ramulator.add_argument("--output", type=Path, required=True)
    fit_ramulator.add_argument("--validation-output", type=Path)
    fit_ramulator.add_argument("--ridge", type=float, default=1e-8)
    integrate = subparsers.add_parser(
        "calibrate-integration", help="fit emulator integration overhead over a Ramulator artifact"
    )
    integrate.add_argument("--base-calibration", type=Path, required=True)
    integrate.add_argument("--run-dir", type=Path, action="append", required=True)
    integrate.add_argument("--output", type=Path, required=True)
    service_plan = subparsers.add_parser(
        "generate-service-calibration-plan",
        help="write the format-generic global HBM V3 request plan",
    )
    service_plan.add_argument("--output", type=Path, required=True)
    service_plan.add_argument("--repetitions", type=int, default=1)
    service_plan.add_argument("--warmup", type=int, default=0)
    service_plan.add_argument("--seed", type=int, default=20260711)
    service_plan.add_argument("--max-patterns", type=int, default=1536)
    fit_service = subparsers.add_parser(
        "fit-service-model",
        help="fit and validate the global nonnegative-ridge HBM V3 model",
    )
    fit_service.add_argument("--plan", type=Path, required=True)
    fit_service.add_argument("--results", type=Path, required=True)
    fit_service.add_argument("--output", type=Path, required=True)
    fit_service.add_argument("--validation-output", type=Path)
    fit_service.add_argument("--ridge", type=float, default=1e-8)
    service_v4_plan = subparsers.add_parser(
        "generate-service-v4-plan",
        help="write the production-DMA occurrence calibration plan",
    )
    service_v4_plan.add_argument("--output", type=Path, required=True)
    service_v4_plan.add_argument("--repetitions", type=int, default=3)
    service_v4_plan.add_argument("--seed", type=int, default=20260716)
    service_v4_plan.add_argument("--max-patterns", type=int)
    service_v4_plan.add_argument(
        "--row-state-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append small row-hit anchors that identify startup versus row-state cost",
    )
    fit_service_v4 = subparsers.add_parser(
        "fit-service-v4",
        help="fit and validate the production-DMA HBM V4 candidate",
    )
    fit_service_v4.add_argument("--plan", type=Path, required=True)
    fit_service_v4.add_argument("--results", type=Path, required=True)
    fit_service_v4.add_argument("--output", type=Path, required=True)
    fit_service_v4.add_argument("--validation-output", type=Path, required=True)
    fit_service_v4.add_argument("--ridge", type=float, default=1e-8)
    fit_service_v4.add_argument("--row-hit-anchor-weight", type=float, default=1.0)
    fit_service_v4.add_argument("--vector-row-hit-anchor-weight", type=float)
    fit_service_v4.add_argument("--store-row-hit-anchor-weight", type=float)
    fit_service_v4.add_argument(
        "--relative-error-weight-power",
        type=float,
        default=1.0,
        help=(
            "fit weight exponent in 1/latency**power; 1 is strict relative "
            "error and 0 is unweighted residual error"
        ),
    )
    validate_service_v4 = subparsers.add_parser(
        "validate-service-v4-system",
        help="compare a V4 scheduled shadow against an observed DMA trace",
    )
    validate_service_v4.add_argument("--model-config", type=Path, required=True)
    validate_service_v4.add_argument("--settings", type=Path, required=True)
    validate_service_v4.add_argument("--calibration", type=Path, required=True)
    validate_service_v4.add_argument("--precision-config", type=Path, required=True)
    validate_service_v4.add_argument("--observed-dma-trace", type=Path, required=True)
    validate_service_v4.add_argument("--seq-len", type=int, required=True)
    validate_service_v4.add_argument("--batch-size", type=int, default=1)
    validate_service_v4.add_argument("--num-layers", type=int, default=1)
    validate_service_v4.add_argument("--layer-idx", type=int, default=0)
    validate_service_v4.add_argument(
        "--moe-routing-mode",
        choices=("static-indices", "fixed-balanced"),
        default="static-indices",
    )
    validate_service_v4.add_argument("--moe-routing-plan", type=Path)
    validate_service_v4.add_argument("--max-static-routes", type=int, default=1024)
    validate_service_v4.add_argument(
        "--moe-layer-scaling",
        choices=("single-layer", "repeat-static-plan", "repeat-fixed-balanced"),
        default="single-layer",
    )
    validate_service_v4.add_argument(
        "--rtl-timing-calibration",
        type=Path,
        default=DEFAULT_RTL_TIMING_CALIBRATION,
    )
    validate_service_v4.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "calibrate":
        calibration = fit_hbm_calibration_from_runs(args.run_dir, ridge=args.ridge)
        calibration.save(args.output)
        print(json.dumps(calibration.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "generate-calibration-plan":
        plan = write_dma_calibration_plan(
            args.output,
            seed=args.seed,
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
        print(json.dumps({"output": str(args.output), "patterns": len(plan["patterns"])}, indent=2))
        return 0
    if args.command == "fit-ramulator":
        calibration, validation = fit_hbm_calibration_from_ramulator(
            args.plan, args.results, ridge=args.ridge
        )
        calibration.save(args.output)
        if args.validation_output:
            args.validation_output.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"calibration": calibration.to_dict(), "validation": validation},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "calibrate-integration":
        calibration = calibrate_hbm_integration(args.base_calibration, args.run_dir)
        calibration.save(args.output)
        print(json.dumps(calibration.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "generate-service-calibration-plan":
        plan = write_hbm_service_calibration_plan(
            args.output,
            seed=args.seed,
            repetitions=args.repetitions,
            warmup=args.warmup,
            max_patterns=args.max_patterns,
        )
        print(json.dumps({"output": str(args.output), "patterns": len(plan["patterns"])}, indent=2))
        return 0
    if args.command == "fit-service-model":
        service_model, validation = fit_hbm_service_model_from_ramulator(
            args.plan,
            args.results,
            ridge=args.ridge,
        )
        service_model.save(args.output)
        if args.validation_output:
            args.validation_output.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"calibration": service_model.to_dict(), "validation": validation},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-service-v4-plan":
        plan = write_hbm_service_v4_plan(
            args.output,
            seed=args.seed,
            repetitions=args.repetitions,
            max_patterns=args.max_patterns,
            include_row_state_anchors=args.row_state_anchors,
        )
        print(
            json.dumps(
                {"output": str(args.output), "patterns": len(plan["patterns"])},
                indent=2,
            )
        )
        return 0
    if args.command == "fit-service-v4":
        service_model, validation = fit_hbm_service_v4(
            args.plan,
            args.results,
            ridge=args.ridge,
            row_hit_anchor_weight=args.row_hit_anchor_weight,
            row_hit_anchor_weights={
                opcode: weight
                for opcode, weight in (
                    ("H_PREFETCH_V", args.vector_row_hit_anchor_weight),
                    ("H_STORE_V", args.store_row_hit_anchor_weight),
                )
                if weight is not None
            },
            relative_error_weight_power=args.relative_error_weight_power,
        )
        service_model.save(args.output)
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        args.validation_output.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {
                    "calibration": service_model.to_dict(),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-service-v4-system":
        settings = TransactionalCycleModel.load(args.settings)
        hardware = _hardware_from_settings(args.model_config, settings)
        trace = compile_native_decoder_cost_trace(
            args.model_config,
            hardware,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            layer_idx=args.layer_idx,
            moe_routing_mode=args.moe_routing_mode,
            moe_routing_plan=args.moe_routing_plan,
            max_static_routes=args.max_static_routes,
            moe_layer_scaling=args.moe_layer_scaling,
        )
        precision = json.loads(args.precision_config.read_text())
        validation = validate_hbm_service_v4_system_case(
            trace,
            settings,
            args.calibration,
            precision,
            args.observed_dma_trace,
            rtl_timing_calibration=args.rtl_timing_calibration,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    if args.scheduled_dma_completion_trace is not None and not args.scheduled_shadow:
        raise ValueError(
            "--scheduled-dma-completion-trace requires --scheduled-shadow"
        )
    settings = TransactionalCycleModel.load(args.settings)
    hardware = _hardware_from_settings(args.model_config, settings)
    trace = compile_native_decoder_cost_trace(
        args.model_config,
        hardware,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        layer_idx=args.layer_idx,
        moe_routing_mode=args.moe_routing_mode,
        moe_routing_plan=args.moe_routing_plan,
        max_static_routes=args.max_static_routes,
        moe_layer_scaling=args.moe_layer_scaling,
        native_layout_mode=args.native_layout_mode,
        packed_attention_schedule=args.packed_attention_schedule,
        vector_scalar_schedule=args.vector_scalar_schedule,
    )
    precision_config = json.loads(args.precision_config.read_text()) if args.precision_config else None
    report = evaluate_compiler_cost(
        trace,
        settings,
        args.calibration,
        precision_config,
        compute_timing_mode=args.compute_timing,
        rtl_timing_calibration=args.rtl_timing_calibration,
        scheduled_shadow=args.scheduled_shadow,
        scheduled_dma_completion_cycles=args.scheduled_dma_completion_trace,
        v4_memory_evaluation=args.v4_memory_evaluation,
    )
    output = {"report": report.to_dict(), "trace_summary": trace.to_dict()}
    if args.profile:
        output["profile_comparison"] = compare_report_to_profile(report, args.profile, trace)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n")
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompilerCostReport",
    "TransactionalCycleModel",
    "build_physical_memory_work",
    "calibrate_hbm_integration",
    "calibration_samples_from_emulator_runs",
    "clear_v4_work_cache",
    "compare_report_to_profile",
    "compile_and_evaluate_compiler_cost",
    "evaluate_compiler_cost",
    "fit_hbm_calibration_from_runs",
    "validate_hbm_service_v4_system_case",
]
