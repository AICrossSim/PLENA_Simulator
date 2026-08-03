"""CostTrace adapter for the production-DMA HBM V4 model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from compiler.aten.isa_builder import DmaTransfer, RepeatAxis
from compiler.aten.program_sink import CostTrace, TraceDma

from ..schemas import MemoryLatencyReport
from .model import HbmServiceModelV4
from .schema import HbmPrecisionConfig, HbmV4Config, MemoryFormat, plan_dma_request_manifest


def _axis_delta(axis: RepeatAxis, field: str) -> int:
    return dict(axis.deltas).get(field, 0)


def _axis_product(event: TraceDma) -> int:
    represented = math.prod(axis.count for axis in event.repeat_axes) if event.repeat_axes else 1
    if represented <= 0 or event.multiplicity % represented:
        raise ValueError(
            f"DMA {event.transfer.opcode} at {event.stage} has multiplicity={event.multiplicity} "
            f"but repeat axes represent an incompatible {represented} occurrences"
        )
    return represented


def _iter_occurrence_transfers(event: TraceDma) -> Iterator[DmaTransfer]:
    represented = _axis_product(event)
    duplicates = event.multiplicity // represented
    if not event.repeat_axes:
        for _ in range(duplicates):
            yield event.transfer
        return
    for linear_index in range(represented):
        remainder = linear_index
        element_delta = 0
        scale_delta = 0
        for axis in reversed(event.repeat_axes):
            index = remainder % axis.count
            remainder //= axis.count
            element_delta += index * _axis_delta(axis, "element_base_bytes")
            scale_delta += index * _axis_delta(axis, "scale_base_bytes")
        if remainder:
            raise ValueError("DMA repeat axes did not consume the occurrence index")
        transfer = replace(
            event.transfer,
            element_base_bytes=event.transfer.element_base_bytes + element_delta,
            scale_base_bytes=(
                None
                if event.transfer.scale_base_bytes is None
                else event.transfer.scale_base_bytes + scale_delta
            ),
            axes=(),
        )
        for _ in range(duplicates):
            yield transfer


@dataclass(frozen=True)
class _WeightedGeometry:
    stage: str
    transfer: DmaTransfer
    fmt: MemoryFormat
    count: int


def _axis_residue_distribution(
    axis: RepeatAxis,
    *,
    modulus: int,
    include_relation: bool,
) -> dict[tuple[int, ...], tuple[int, int, int]]:
    element_delta = _axis_delta(axis, "element_base_bytes")
    scale_delta = _axis_delta(axis, "scale_base_bytes")
    element_period = modulus // math.gcd(modulus, abs(element_delta))
    scale_period = modulus // math.gcd(modulus, abs(scale_delta))
    period = math.lcm(element_period, scale_period)
    relation_cycle_delta = period * (scale_delta - element_delta)
    sample_count = axis.count if include_relation and relation_cycle_delta else min(axis.count, period)
    preserve_relation = include_relation and bool(relation_cycle_delta)
    full, tail = (0, axis.count) if preserve_relation else divmod(axis.count, period)
    result: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for index in range(sample_count):
        occurrences = 1 if preserve_relation else full + int(index < tail)
        if not occurrences:
            continue
        element = index * element_delta
        scale = index * scale_delta
        key: tuple[int, ...] = (element % modulus, scale % modulus)
        if include_relation:
            key += (scale - element,)
        old = result.get(key)
        result[key] = (
            occurrences if old is None else old[0] + occurrences,
            element if old is None else old[1],
            scale if old is None else old[2],
        )
    return result


def _merge_distributions(
    left: Mapping[tuple[int, ...], tuple[int, int, int]],
    right: Mapping[tuple[int, ...], tuple[int, int, int]],
    *,
    modulus: int,
) -> dict[tuple[int, ...], tuple[int, int, int]]:
    if len(right) == 1:
        right_key, (right_count, right_element, right_scale) = next(iter(right.items()))
        result = {}
        for left_key, (left_count, left_element, left_scale) in left.items():
            key = (
                (left_key[0] + right_key[0]) % modulus,
                (left_key[1] + right_key[1]) % modulus,
            )
            if len(left_key) == 3:
                key += (left_key[2] + right_key[2],)
            result[key] = (
                left_count * right_count,
                left_element + right_element,
                left_scale + right_scale,
            )
        return result
    result: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for left_key, (left_count, left_element, left_scale) in left.items():
        for right_key, (right_count, right_element, right_scale) in right.items():
            key = (
                (left_key[0] + right_key[0]) % modulus,
                (left_key[1] + right_key[1]) % modulus,
            )
            if len(left_key) == 3:
                key += (left_key[2] + right_key[2],)
            old = result.get(key)
            result[key] = (
                left_count * right_count if old is None else old[0] + left_count * right_count,
                left_element + right_element if old is None else old[1],
                left_scale + right_scale if old is None else old[2],
            )
    return result


def _regions_are_disjoint(event: TraceDma, fmt: MemoryFormat, modulus: int) -> bool:
    if not fmt.is_mx:
        return True
    transfer = event.transfer
    element_row_bytes = fmt.element_bits * transfer.dim // 8
    stride_bytes = element_row_bytes if transfer.rstride != 1 else transfer.stride_bytes
    stride_elements = stride_bytes * 8 // fmt.element_bits
    scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
    scale_row_bytes = fmt.scale_bits * (transfer.dim // fmt.block) // 8
    element_span = max(0, transfer.amount - 1) * stride_bytes + max(0, element_row_bytes - 1)
    scale_span = max(0, transfer.amount - 1) * scale_stride + max(0, scale_row_bytes - 1)
    assert transfer.scale_base_bytes is not None
    scale_after_element = transfer.scale_base_bytes - transfer.element_base_bytes - element_span
    element_after_scale = transfer.element_base_bytes - transfer.scale_base_bytes - scale_span
    for axis in event.repeat_axes:
        relative = _axis_delta(axis, "scale_base_bytes") - _axis_delta(axis, "element_base_bytes")
        scale_after_element += min(0, (axis.count - 1) * relative)
        element_after_scale += min(0, (axis.count - 1) * -relative)
    return scale_after_element >= modulus or element_after_scale >= modulus


def _normalized_geometry_key(
    stage: str,
    transfer: DmaTransfer,
    fmt: MemoryFormat,
    *,
    modulus: int,
) -> tuple[Any, ...]:
    element_row_bytes = fmt.element_bits * transfer.dim // 8
    stride = element_row_bytes if transfer.rstride != 1 else transfer.stride_bytes
    if not fmt.is_mx:
        scale_residue = 0
        relation: int | str = "no_scale"
    else:
        assert transfer.scale_base_bytes is not None
        scale_residue = transfer.scale_base_bytes % modulus
        stride_elements = stride * 8 // fmt.element_bits
        scale_stride = stride_elements // fmt.block * fmt.scale_bits // 8
        scale_row_bytes = fmt.scale_bits * (transfer.dim // fmt.block) // 8
        element_last = transfer.element_base_bytes + max(0, transfer.amount - 1) * stride + element_row_bytes - 1
        scale_last = transfer.scale_base_bytes + max(0, transfer.amount - 1) * scale_stride + scale_row_bytes - 1
        element_rows = (transfer.element_base_bytes // modulus, element_last // modulus)
        scale_rows = (transfer.scale_base_bytes // modulus, scale_last // modulus)
        relation = (
            "disjoint"
            if element_rows[1] < scale_rows[0] or scale_rows[1] < element_rows[0]
            else scale_rows[0] - element_rows[0]
        )
    return (
        stage,
        transfer.opcode,
        transfer.direction,
        transfer.role,
        fmt.request_signature(),
        transfer.element_base_bytes % modulus,
        scale_residue,
        relation,
        transfer.dim,
        transfer.amount,
        transfer.stride_bytes,
        transfer.rstride,
        transfer.write_amount,
    )


def _event_geometry_groups(
    event: TraceDma,
    fmt: MemoryFormat,
    *,
    modulus: int,
) -> list[_WeightedGeometry]:
    represented = _axis_product(event)
    duplicates = event.multiplicity // represented
    include_relation = not _regions_are_disjoint(event, fmt, modulus)
    states: dict[tuple[int, ...], tuple[int, int, int]] = {
        (0, 0, 0) if include_relation else (0, 0): (1, 0, 0)
    }
    for axis in event.repeat_axes:
        states = _merge_distributions(
            states,
            _axis_residue_distribution(axis, modulus=modulus, include_relation=include_relation),
            modulus=modulus,
        )
    result = []
    for count, element_delta, scale_delta in states.values():
        result.append(
            _WeightedGeometry(
                stage=event.stage,
                transfer=replace(
                    event.transfer,
                    element_base_bytes=event.transfer.element_base_bytes + element_delta,
                    scale_base_bytes=(
                        None
                        if event.transfer.scale_base_bytes is None
                        else event.transfer.scale_base_bytes + scale_delta
                    ),
                    axes=(),
                ),
                fmt=fmt,
                count=count * duplicates,
            )
        )
    if sum(item.count for item in result) != event.multiplicity:
        raise ValueError(
            "affine HBM grouping lost DMA multiplicity: "
            f"{event.stage}/{event.transfer.opcode} expected={event.multiplicity} "
            f"actual={sum(item.count for item in result)} axes={event.repeat_axes}"
        )
    return result


_PREFIX_AXES = {"visible_k_block", "visible_v_block", "streaming_kv_block"}


def _prefix_axis(event: TraceDma) -> RepeatAxis | None:
    found = tuple(axis for axis in event.repeat_axes if axis.name in _PREFIX_AXES)
    return found[0] if len(found) == 1 else None


def _prefix_family_key(event: TraceDma) -> tuple[Any, ...] | None:
    prefix = _prefix_axis(event)
    if prefix is None:
        return None
    fixed = tuple(axis for axis in event.repeat_axes if axis != prefix)
    transfer = event.transfer
    return (
        event.stage,
        transfer,
        fixed,
        prefix.name,
        _axis_delta(prefix, "element_base_bytes"),
        _axis_delta(prefix, "scale_base_bytes"),
    )


def _prefix_family_groups(
    events: Sequence[TraceDma],
    fmt: MemoryFormat,
    *,
    modulus: int,
) -> list[_WeightedGeometry] | None:
    if len(events) < 2 or any(not _regions_are_disjoint(event, fmt, modulus) for event in events):
        return None
    representative = max(events, key=lambda item: _prefix_axis(item).count)  # type: ignore[union-attr]
    prefix = _prefix_axis(representative)
    assert prefix is not None
    fixed_axes = tuple(axis for axis in representative.repeat_axes if axis != prefix)
    prefix_lengths: Counter[int] = Counter()
    duplicate_factor: int | None = None
    for event in events:
        axis = _prefix_axis(event)
        assert axis is not None
        represented = _axis_product(event)
        duplicates = event.multiplicity // represented
        if duplicate_factor is None:
            duplicate_factor = duplicates
        if duplicates != duplicate_factor:
            return None
        prefix_lengths[axis.count] += duplicates
    active = sum(prefix_lengths.values())
    prefix_states: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for index in range(max(prefix_lengths)):
        if index:
            active -= prefix_lengths[index]
        if active <= 0:
            continue
        element = index * _axis_delta(prefix, "element_base_bytes")
        scale = index * _axis_delta(prefix, "scale_base_bytes")
        key = (element % modulus, scale % modulus)
        old = prefix_states.get(key)
        prefix_states[key] = (
            active if old is None else old[0] + active,
            element if old is None else old[1],
            scale if old is None else old[2],
        )
    states = prefix_states
    for axis in fixed_axes:
        states = _merge_distributions(
            states,
            _axis_residue_distribution(axis, modulus=modulus, include_relation=False),
            modulus=modulus,
        )
    result = [
        _WeightedGeometry(
            stage=representative.stage,
            transfer=replace(
                representative.transfer,
                element_base_bytes=representative.transfer.element_base_bytes + element,
                scale_base_bytes=(
                    None
                    if representative.transfer.scale_base_bytes is None
                    else representative.transfer.scale_base_bytes + scale
                ),
                axes=(),
            ),
            fmt=fmt,
            count=count,
        )
        for count, element, scale in states.values()
    ]
    expected = sum(event.multiplicity for event in events)
    if sum(item.count for item in result) != expected:
        raise ValueError(f"causal-prefix grouping lost multiplicity: expected {expected}")
    return result


class HbmV4MemoryProvider:
    name = "hbm-v4-production-dma"

    def __init__(
        self,
        model: HbmServiceModelV4,
        precision: HbmPrecisionConfig,
        config: HbmV4Config,
        *,
        aggregation: str = "sufficient-statistics",
        fail_on_extrapolation: bool = False,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        geometry_batch_size: int = 4096,
    ) -> None:
        if aggregation not in {"scalar", "stateful", "sufficient-statistics"}:
            raise ValueError(f"unsupported HBM V4 aggregation {aggregation!r}")
        if geometry_batch_size <= 0:
            raise ValueError("geometry_batch_size must be positive")
        self.model = model
        self.precision = precision
        self.config = config
        self.aggregation = aggregation
        self.fail_on_extrapolation = fail_on_extrapolation
        self.progress_callback = progress_callback
        self.geometry_batch_size = geometry_batch_size

    def estimate(self, trace: CostTrace) -> MemoryLatencyReport:
        if self.aggregation == "stateful" and not trace.metadata.get("ordered_dma_events_available", False):
            raise ValueError("stateful HBM V4 requires an ordered DMA trace; CostTrace summary is unordered")
        if self.aggregation == "sufficient-statistics":
            return self._estimate_grouped(trace)
        return self._estimate_scalar(trace)

    def _new_accumulator(self) -> dict[str, Any]:
        return {
            "by_stage_ns": Counter(),
            "by_stage_floor_ns": Counter(),
            "by_opcode_ns": Counter(),
            "traffic": defaultdict(lambda: defaultdict(Counter)),
            "read_bytes": 0,
            "write_bytes": 0,
            "payload_read": 0,
            "payload_write": 0,
            "read_requests": 0,
            "write_requests": 0,
            "issues": Counter(),
            "regimes": Counter(),
            "occurrence_count": 0,
        }

    def _add_prediction(
        self,
        accumulator: dict[str, Any],
        *,
        stage: str,
        transfer: DmaTransfer,
        manifest: Any,
        prediction: Any,
        count: int,
    ) -> None:
        if count <= 0:
            return
        if self.fail_on_extrapolation and not prediction.calibration_in_domain:
            raise ValueError(f"HBM V4 occurrence is outside calibration domain: {prediction.domain_issues}")
        accumulator["by_stage_ns"][stage] += prediction.latency_ns * count
        accumulator["by_stage_floor_ns"][stage] += prediction.theoretical_phase_floor_ns * count
        accumulator["by_opcode_ns"][transfer.opcode] += prediction.latency_ns * count
        accumulator["read_bytes"] += manifest.read_bytes * count
        accumulator["write_bytes"] += manifest.write_bytes * count
        accumulator["payload_read"] += manifest.payload_read_bytes * count
        accumulator["payload_write"] += manifest.payload_write_bytes * count
        accumulator["read_requests"] += len(manifest.read_lines) * count
        accumulator["write_requests"] += len(manifest.write_lines) * count
        accumulator["traffic"][stage][transfer.role].update(
            physical_read_bytes=manifest.read_bytes * count,
            physical_write_bytes=manifest.write_bytes * count,
            payload_read_bytes=manifest.payload_read_bytes * count,
            payload_write_bytes=manifest.payload_write_bytes * count,
            read_requests=len(manifest.read_lines) * count,
            write_requests=len(manifest.write_lines) * count,
        )
        accumulator["issues"].update({issue: count for issue in prediction.domain_issues})
        accumulator["regimes"][prediction.row_state_regime] += count
        accumulator["occurrence_count"] += count

    def _estimate_scalar(self, trace: CostTrace) -> MemoryLatencyReport:
        accumulator = self._new_accumulator()
        open_rows = np.full(self.config.channels * 32, -1, dtype=np.int64) if self.aggregation == "stateful" else None
        for event in trace.dma_events:
            if not event.stage:
                raise ValueError("DMA event has no stage ownership")
            fmt = self.precision.for_transfer(event.transfer)
            for transfer in _iter_occurrence_transfers(event):
                manifest = plan_dma_request_manifest(transfer, fmt)
                prediction = self.model.predict(
                    transfer.opcode,
                    transfer,
                    fmt,
                    self.config,
                    manifest,
                    open_rows=open_rows,
                )
                self._add_prediction(
                    accumulator,
                    stage=event.stage,
                    transfer=transfer,
                    manifest=manifest,
                    prediction=prediction,
                    count=1,
                )
        return self._finish_report(accumulator, unique_geometry_count=accumulator["occurrence_count"])

    def _estimate_grouped(self, trace: CostTrace) -> MemoryLatencyReport:
        modulus = 16_384 * self.config.channels
        family_events: defaultdict[tuple[Any, ...], list[TraceDma]] = defaultdict(list)
        ordinary: list[TraceDma] = []
        for event in trace.dma_events:
            family_key = _prefix_family_key(event)
            if family_key is None:
                ordinary.append(event)
            else:
                family_events[family_key].append(event)

        weighted: list[_WeightedGeometry] = []
        prefix_stream_count_folded = 0
        scalar_fallback_count = 0
        for family in family_events.values():
            fmt = self.precision.for_transfer(family[0].transfer)
            folded = _prefix_family_groups(family, fmt, modulus=modulus)
            if folded is None:
                ordinary.extend(family)
                scalar_fallback_count += len(family)
            else:
                weighted.extend(folded)
                prefix_stream_count_folded += len(family)
        for event in ordinary:
            weighted.extend(
                _event_geometry_groups(
                    event,
                    self.precision.for_transfer(event.transfer),
                    modulus=modulus,
                )
            )

        grouped: dict[tuple[Any, ...], _WeightedGeometry] = {}
        address_geometry_count = len(weighted)
        for item in weighted:
            key = _normalized_geometry_key(item.stage, item.transfer, item.fmt, modulus=modulus)
            old = grouped.get(key)
            grouped[key] = (
                item
                if old is None
                else _WeightedGeometry(old.stage, old.transfer, old.fmt, old.count + item.count)
            )
        logical_occurrences = sum(event.multiplicity for event in trace.dma_events)
        if sum(item.count for item in grouped.values()) != logical_occurrences:
            raise ValueError("grouped HBM V4 lost logical DMA occurrences")

        accumulator = self._new_accumulator()
        total = len(grouped)
        for index, item in enumerate(grouped.values(), start=1):
            manifest = plan_dma_request_manifest(item.transfer, item.fmt)
            prediction = self.model.predict(
                item.transfer.opcode,
                item.transfer,
                item.fmt,
                self.config,
                manifest,
            )
            self._add_prediction(
                accumulator,
                stage=item.stage,
                transfer=item.transfer,
                manifest=manifest,
                prediction=prediction,
                count=item.count,
            )
            if self.progress_callback and (index % self.geometry_batch_size == 0 or index == total):
                self.progress_callback(
                    {
                        "phase": "hbm-v4-sufficient-statistics",
                        "progress_done": index,
                        "progress_total": total,
                    }
                )
        return self._finish_report(
            accumulator,
            unique_geometry_count=len(grouped),
            extra_provenance={
                "v4_aggregation": "affine-feature-grouped-v1",
                "exact_feature_equivalence": True,
                "unique_address_geometry_count": address_geometry_count,
                "unique_feature_signature_count": len(grouped),
                "logical_occurrence_count": logical_occurrences,
                "occurrences_elided": logical_occurrences - len(grouped),
                "scalar_fallback_count": scalar_fallback_count,
                "prefix_stream_count_folded": prefix_stream_count_folded,
            },
        )

    def _finish_report(
        self,
        accumulator: dict[str, Any],
        *,
        unique_geometry_count: int,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> MemoryLatencyReport:
        by_stage_ns: Counter[str] = accumulator["by_stage_ns"]
        by_stage_floor_ns: Counter[str] = accumulator["by_stage_floor_ns"]
        by_opcode_ns: Counter[str] = accumulator["by_opcode_ns"]
        traffic: defaultdict[str, defaultdict[str, Counter[str]]] = accumulator["traffic"]
        issues: Counter[str] = accumulator["issues"]
        regimes: Counter[str] = accumulator["regimes"]
        by_stage_picos = {
            stage: round(value * 1_000) for stage, value in sorted(by_stage_ns.items())
        }
        floor_picos = {
            stage: round(value * 1_000) for stage, value in sorted(by_stage_floor_ns.items())
        }
        warnings = tuple(
            [f"HBM V4 extrapolated {sum(issues.values())} feature observations across {len(issues)} issues"]
            if issues
            else []
        )
        return MemoryLatencyReport(
            total_picos=sum(by_stage_picos.values()),
            by_stage_picos=by_stage_picos,
            physical_read_bytes=accumulator["read_bytes"],
            physical_write_bytes=accumulator["write_bytes"],
            payload_read_bytes=accumulator["payload_read"],
            payload_write_bytes=accumulator["payload_write"],
            read_requests=accumulator["read_requests"],
            write_requests=accumulator["write_requests"],
            by_opcode_picos={
                opcode: round(value * 1_000) for opcode, value in sorted(by_opcode_ns.items())
            },
            by_stage_floor_picos=floor_picos,
            traffic_breakdown={
                stage: {role: dict(values) for role, values in sorted(roles.items())}
                for stage, roles in sorted(traffic.items())
            },
            provider=self.name,
            provenance={
                "provider": self.name,
                "calibration_id": self.model.calibration_id,
                "aggregation": self.aggregation,
                "memory_config": {
                    "channels": self.config.channels,
                    "request_bytes": self.config.request_bytes,
                    "physical_burst_bytes": self.config.physical_burst_bytes,
                    "channel_bandwidth_bytes_per_ns": self.config.channel_bandwidth_bytes_per_ns,
                    "mapper": self.config.mapper,
                    "preset": self.config.preset,
                },
                "precision": self.precision.to_dict(),
                "occurrence_count": accumulator["occurrence_count"],
                "unique_geometry_count": unique_geometry_count,
                "row_state_regime_counts": dict(regimes),
                "calibration_in_domain": not issues,
                "domain_issues": dict(issues),
                "latency_semantics": "calibrated-production-dma-occurrence-service",
                "queue_overlap_modeled": False,
                **(dict(extra_provenance) if extra_provenance else {}),
            },
            warnings=warnings,
        )


def estimate_hbm_v4(
    trace: CostTrace,
    memory_config: HbmV4Config,
    precision: HbmPrecisionConfig,
    calibration: str | Path | HbmServiceModelV4,
    *,
    aggregation: str = "sufficient-statistics",
    fail_on_extrapolation: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    geometry_batch_size: int = 4096,
) -> MemoryLatencyReport:
    model = calibration if isinstance(calibration, HbmServiceModelV4) else HbmServiceModelV4.load(calibration)
    return HbmV4MemoryProvider(
        model,
        precision,
        memory_config,
        aggregation=aggregation,
        fail_on_extrapolation=fail_on_extrapolation,
        progress_callback=progress_callback,
        geometry_batch_size=geometry_batch_size,
    ).estimate(trace)


__all__ = ["HbmV4MemoryProvider", "estimate_hbm_v4"]
