"""Routed-expert placement and cache analysis for Qwen3-30B-A3B.

The network ledger counts logical point-to-point transfers.  It does not model
an all-to-all collective, switch contention, packetisation, or overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "plena-qwen3-moe-expert-placement-input/v1"
REPORT_SCHEMA = "plena-qwen3-moe-expert-placement-report/v1"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
POLICIES = frozenset(
    {
        "sharded_dispatch",
        "hot_expert_replication",
        "local_lru_weight_cache",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_REVISION = re.compile(r"[0-9a-f]{7,64}")

ExpertInstance = tuple[int, int]


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class EvidenceReceipt:
    """Auditable receipt for one measured artifact and its claimed subject."""

    artifact_path: str
    artifact_sha256: str
    subject_sha256: str
    command: tuple[str, ...]
    tool_revision: str
    recorded_at_utc: str
    sample_count: int

    def __post_init__(self) -> None:
        for field in (
            "artifact_path",
            "artifact_sha256",
            "subject_sha256",
            "tool_revision",
            "recorded_at_utc",
        ):
            object.__setattr__(self, field, str(getattr(self, field)))
        command = (
            (str(self.command),)
            if isinstance(self.command, (str, bytes))
            else tuple(str(part) for part in self.command)
        )
        object.__setattr__(self, "command", command)


@dataclass(frozen=True)
class RouteRecord:
    """One token's ordered top-k decision at one MoE layer."""

    token_id: str
    layer: int
    source_chip: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_ids", tuple(self.expert_ids))
        if not isinstance(self.token_id, str) or not self.token_id:
            raise ValueError("token_id is required")
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or not 0 <= self.layer < NUM_LAYERS
        ):
            raise ValueError(f"layer must be in [0, {NUM_LAYERS})")
        _positive_integer(self.source_chip, "source_chip", allow_zero=True)
        if len(self.expert_ids) != TOP_K:
            raise ValueError(f"each route record must contain exactly top-{TOP_K}")
        if len(set(self.expert_ids)) != TOP_K:
            raise ValueError("expert_ids must be unique within a top-k decision")
        for expert_id in self.expert_ids:
            if (
                isinstance(expert_id, bool)
                or not isinstance(expert_id, int)
                or not 0 <= expert_id < NUM_EXPERTS
            ):
                raise ValueError(f"expert_id must be in [0, {NUM_EXPERTS})")


@dataclass(frozen=True)
class RoutingStep:
    """An ordered group of route records observed at one decode step."""

    step_index: int
    records: tuple[RouteRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        _positive_integer(self.step_index, "step_index", allow_zero=True)
        if not self.records:
            raise ValueError("routing steps must contain at least one record")
        keys = [(record.token_id, record.layer) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("a token may have at most one route record per layer and step")


@dataclass(frozen=True)
class RoutingTrace:
    """Measured or synthetic ordered routing evidence."""

    source_kind: str
    steps: tuple[RoutingStep, ...]
    receipt: EvidenceReceipt | None = None
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if self.source_kind not in {"measured", "synthetic"}:
            raise ValueError("source_kind must be 'measured' or 'synthetic'")
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("routing trace must identify the sealed Qwen3 checkpoint")
        if not self.steps:
            raise ValueError("routing trace must contain at least one step")
        indices = [step.step_index for step in self.steps]
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ValueError("routing step indices must be strictly increasing")


@dataclass(frozen=True)
class PlacementConfig:
    """Placement policy and byte-calibrated network inputs."""

    policy: str
    chip_count: int
    expert_weight_bytes: int
    activation_bytes_per_assignment: int
    link_bandwidth_bytes_per_s_per_chip: float
    link_energy_j_per_byte: float
    cache_capacity_bytes_per_chip: int = 0
    hot_expert_count: int = 0
    hot_replica_count: int = 1
    hot_expert_instances: tuple[ExpertInstance, ...] | None = None
    model_bytes_receipt: EvidenceReceipt | None = None
    link_calibration_receipt: EvidenceReceipt | None = None
    hot_placement_receipt: EvidenceReceipt | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unsupported placement policy {self.policy!r}")
        _positive_integer(self.chip_count, "chip_count")
        if self.chip_count > NUM_EXPERTS:
            raise ValueError(f"chip_count cannot exceed {NUM_EXPERTS}")
        _positive_integer(self.expert_weight_bytes, "expert_weight_bytes")
        _positive_integer(
            self.activation_bytes_per_assignment,
            "activation_bytes_per_assignment",
        )
        object.__setattr__(
            self,
            "link_bandwidth_bytes_per_s_per_chip",
            _finite(
                self.link_bandwidth_bytes_per_s_per_chip,
                "link_bandwidth_bytes_per_s_per_chip",
            ),
        )
        object.__setattr__(
            self,
            "link_energy_j_per_byte",
            _finite(
                self.link_energy_j_per_byte,
                "link_energy_j_per_byte",
                allow_zero=True,
            ),
        )
        _positive_integer(
            self.cache_capacity_bytes_per_chip,
            "cache_capacity_bytes_per_chip",
            allow_zero=True,
        )
        _positive_integer(self.hot_expert_count, "hot_expert_count", allow_zero=True)
        _positive_integer(self.hot_replica_count, "hot_replica_count")
        if self.hot_expert_instances is not None:
            values: list[ExpertInstance] = []
            for value in self.hot_expert_instances:
                if (
                    not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    or len(value) != 2
                ):
                    raise ValueError("each hot expert instance must be [layer, expert]")
                layer, expert = value
                if (
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or isinstance(expert, bool)
                    or not isinstance(expert, int)
                    or not 0 <= layer < NUM_LAYERS
                    or not 0 <= expert < NUM_EXPERTS
                ):
                    raise ValueError("hot expert instance is outside the model contract")
                values.append((layer, expert))
            instances = tuple(sorted(values))
            if len(instances) != len(set(instances)):
                raise ValueError("hot_expert_instances must not contain duplicates")
            object.__setattr__(self, "hot_expert_instances", instances)

        if self.policy == "local_lru_weight_cache":
            if self.hot_expert_count or self.hot_expert_instances is not None:
                raise ValueError("LRU policy does not accept hot expert settings")
            if self.hot_replica_count != 1:
                raise ValueError("LRU policy requires hot_replica_count=1")
        elif self.cache_capacity_bytes_per_chip != 0:
            raise ValueError("cache capacity is only valid for the LRU policy")

        if self.policy == "hot_expert_replication":
            effective_count = (
                len(self.hot_expert_instances)
                if self.hot_expert_instances is not None
                else self.hot_expert_count
            )
            if effective_count <= 0:
                raise ValueError("hot replication requires at least one hot expert")
            if self.hot_expert_count not in {0, effective_count}:
                raise ValueError("hot_expert_count differs from the explicit hot list")
            if not 2 <= self.hot_replica_count <= self.chip_count:
                raise ValueError("hot_replica_count must be in [2, chip_count]")
        elif (
            self.hot_expert_count
            or self.hot_expert_instances is not None
            or self.hot_replica_count != 1
        ):
            raise ValueError("hot expert settings are only valid for hot replication")
        if self.policy != "hot_expert_replication" and self.hot_placement_receipt:
            raise ValueError("hot placement receipt is only valid for hot replication")


def _record_json(record: RouteRecord) -> dict[str, Any]:
    return {
        "token_id": record.token_id,
        "layer": record.layer,
        "source_chip": record.source_chip,
        "expert_ids": list(record.expert_ids),
    }


def trace_content(trace: RoutingTrace) -> dict[str, Any]:
    """Return the receipt-independent routing payload."""

    return {
        "model_id": trace.model_id,
        "model_revision": trace.model_revision,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "steps": [
            {
                "step_index": step.step_index,
                "records": [_record_json(record) for record in step.records],
            }
            for step in trace.steps
        ],
    }


def trace_content_hash(trace: RoutingTrace) -> str:
    return _canonical_hash(trace_content(trace))


def model_bytes_subject(config: PlacementConfig) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "expert_weight_bytes": config.expert_weight_bytes,
        "activation_bytes_per_assignment": config.activation_bytes_per_assignment,
    }


def link_calibration_subject(config: PlacementConfig) -> dict[str, Any]:
    return {
        "traffic_model": "logical_unicast_per_chip_full_duplex",
        "chip_count": config.chip_count,
        "link_bandwidth_bytes_per_s_per_chip": (
            config.link_bandwidth_bytes_per_s_per_chip
        ),
        "link_energy_j_per_byte": config.link_energy_j_per_byte,
    }


def hot_placement_subject(
    config: PlacementConfig, selected: Sequence[ExpertInstance]
) -> dict[str, Any]:
    return {
        "policy": "hot_expert_replication",
        "chip_count": config.chip_count,
        "hot_replica_count": config.hot_replica_count,
        "replica_placement_rule": "home_then_cyclic_successors",
        "hot_expert_instances": [list(instance) for instance in selected],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_receipt(
    receipt: EvidenceReceipt | None,
    *,
    expected_subject_sha256: str,
    base_dir: Path,
) -> dict[str, Any]:
    reasons: list[str] = []
    if receipt is None:
        return {
            "present": False,
            "valid": False,
            "expected_subject_sha256": expected_subject_sha256,
            "reasons": ["receipt_missing"],
        }

    artifact = Path(receipt.artifact_path)
    resolved = artifact if artifact.is_absolute() else base_dir / artifact
    if not receipt.artifact_path:
        reasons.append("artifact_path_missing")
    elif not resolved.is_file():
        reasons.append("artifact_missing")
    elif not _HEX_SHA256.fullmatch(receipt.artifact_sha256):
        reasons.append("artifact_sha256_malformed")
    elif _sha256_file(resolved) != receipt.artifact_sha256:
        reasons.append("artifact_sha256_mismatch")

    if receipt.subject_sha256 != expected_subject_sha256:
        reasons.append("subject_sha256_mismatch")
    if not receipt.command or any(not part for part in receipt.command):
        reasons.append("command_missing")
    if not _HEX_REVISION.fullmatch(receipt.tool_revision):
        reasons.append("tool_revision_malformed")
    try:
        timestamp = datetime.fromisoformat(receipt.recorded_at_utc.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            reasons.append("recorded_at_utc_has_no_timezone")
        elif timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            reasons.append("recorded_at_utc_is_not_utc")
    except ValueError:
        reasons.append("recorded_at_utc_malformed")
    if (
        isinstance(receipt.sample_count, bool)
        or not isinstance(receipt.sample_count, int)
        or receipt.sample_count <= 0
    ):
        reasons.append("sample_count_not_positive")

    return {
        "present": True,
        "valid": not reasons,
        "artifact_path": receipt.artifact_path,
        "artifact_sha256": receipt.artifact_sha256,
        "subject_sha256": receipt.subject_sha256,
        "expected_subject_sha256": expected_subject_sha256,
        "command": list(receipt.command),
        "tool_revision": receipt.tool_revision,
        "recorded_at_utc": receipt.recorded_at_utc,
        "sample_count": receipt.sample_count,
        "reasons": reasons,
    }


def _base_residents(chip_count: int) -> list[set[ExpertInstance]]:
    residents = [set() for _ in range(chip_count)]
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            residents[expert % chip_count].add((layer, expert))
    return residents


def _coverage(trace: RoutingTrace) -> dict[str, Any]:
    token_layers: dict[tuple[int, str], set[int]] = {}
    layers = set()
    records = 0
    for step in trace.steps:
        for record in step.records:
            records += 1
            layers.add(record.layer)
            token_layers.setdefault((step.step_index, record.token_id), set()).add(
                record.layer
            )
    complete = sum(layer_set == set(range(NUM_LAYERS)) for layer_set in token_layers.values())
    return {
        "step_count": len(trace.steps),
        "token_step_count": len(token_layers),
        "route_record_count": records,
        "observed_layers": sorted(layers),
        "complete_48_layer_token_steps": complete,
        "complete_48_layer_fraction": complete / len(token_layers),
        "publication_complete": complete == len(token_layers),
    }


def _select_hot_instances(
    trace: RoutingTrace, config: PlacementConfig
) -> tuple[tuple[ExpertInstance, ...], str, Counter[ExpertInstance], dict[ExpertInstance, Counter[int]]]:
    frequency: Counter[ExpertInstance] = Counter()
    source_frequency: dict[ExpertInstance, Counter[int]] = {}
    for step in trace.steps:
        for record in step.records:
            for expert in record.expert_ids:
                instance = (record.layer, expert)
                frequency[instance] += 1
                source_frequency.setdefault(instance, Counter())[record.source_chip] += 1

    if config.hot_expert_instances is not None:
        return config.hot_expert_instances, "explicit", frequency, source_frequency
    selected = sorted(
        (
            (layer, expert)
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
        key=lambda instance: (-frequency[instance], instance[0], instance[1]),
    )[: config.hot_expert_count]
    return tuple(selected), "evaluation_trace_frequency", frequency, source_frequency


def _load_summary(loads: Sequence[int]) -> dict[str, Any]:
    total = sum(loads)
    mean = total / len(loads)
    coefficient = (
        math.sqrt(sum((load - mean) ** 2 for load in loads) / len(loads)) / mean
        if mean
        else 0.0
    )
    return {
        "assignments": total,
        "mean_assignments_per_chip": mean,
        "max_to_mean_ratio": max(loads) / mean if mean else 0.0,
        "min_to_mean_ratio": min(loads) / mean if mean else 0.0,
        "coefficient_of_variation": coefficient,
        "max_minus_min_assignments": max(loads) - min(loads),
        "per_chip": [
            {
                "chip": chip,
                "assignments": load,
                "fraction": load / total if total else 0.0,
            }
            for chip, load in enumerate(loads)
        ],
    }


def analyze_expert_placement(
    trace: RoutingTrace,
    config: PlacementConfig,
    *,
    receipt_base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Evaluate one policy without inventing routing or calibration evidence."""

    base_dir = Path(receipt_base_dir)
    for step in trace.steps:
        for record in step.records:
            if record.source_chip >= config.chip_count:
                raise ValueError("route source_chip is outside configured chip_count")

    trace_hash = trace_content_hash(trace)
    coverage = _coverage(trace)
    expected_assignments = coverage["route_record_count"] * TOP_K
    base_residents = _base_residents(config.chip_count)
    residents = [set(values) for values in base_residents]
    base_counts = [len(values) for values in base_residents]
    peak_counts = list(base_counts)
    loads = [0 for _ in range(config.chip_count)]
    tx_bytes = [0 for _ in range(config.chip_count)]
    rx_bytes = [0 for _ in range(config.chip_count)]
    remote_by_chip = [0 for _ in range(config.chip_count)]
    dispatch_bytes = 0
    combine_bytes = 0
    weight_fetch_bytes = 0
    remote_assignments = 0

    cache_hits = [0 for _ in range(config.chip_count)]
    cache_misses = [0 for _ in range(config.chip_count)]
    cache_evictions = [0 for _ in range(config.chip_count)]
    uncacheable_misses = [0 for _ in range(config.chip_count)]
    caches: list[OrderedDict[ExpertInstance, None]] = [
        OrderedDict() for _ in range(config.chip_count)
    ]

    hot_instances: tuple[ExpertInstance, ...] = ()
    hot_source = "not_applicable"
    hot_replica_placement_rule = "not_applicable"
    hot_rows: list[dict[str, Any]] = []
    replicas: dict[ExpertInstance, tuple[int, ...]] = {}
    if config.policy == "hot_expert_replication":
        hot_instances, hot_source, frequency, source_frequency = _select_hot_instances(
            trace, config
        )
        for instance in hot_instances:
            home = instance[1] % config.chip_count
            sites = {home}
            if hot_source == "explicit":
                hot_replica_placement_rule = "home_then_cyclic_successors"
                candidates = sorted(
                    (chip for chip in range(config.chip_count) if chip != home),
                    key=lambda chip: ((chip - home) % config.chip_count, chip),
                )
            else:
                hot_replica_placement_rule = (
                    "evaluation_trace_source_demand_then_resident_balance"
                )
                candidates = sorted(
                    (chip for chip in range(config.chip_count) if chip != home),
                    key=lambda chip: (
                        -source_frequency.get(instance, Counter())[chip],
                        len(residents[chip]),
                        chip,
                    ),
                )
            for chip in candidates[: config.hot_replica_count - 1]:
                residents[chip].add(instance)
                sites.add(chip)
            replicas[instance] = tuple(sorted(sites))
            hot_rows.append(
                {
                    "layer": instance[0],
                    "expert": instance[1],
                    "observed_assignments": frequency[instance],
                    "resident_chips": list(replicas[instance]),
                }
            )
        peak_counts = [len(values) for values in residents]

    def transfer(source: int, target: int, amount: int) -> None:
        if source == target:
            return
        tx_bytes[source] += amount
        rx_bytes[target] += amount

    observed_assignments = 0
    for step in trace.steps:
        for record in step.records:
            source = record.source_chip
            for expert in record.expert_ids:
                observed_assignments += 1
                instance = (record.layer, expert)
                home = expert % config.chip_count

                if config.policy == "sharded_dispatch":
                    target = home
                    if target != source:
                        remote_assignments += 1
                        remote_by_chip[source] += 1
                        dispatch_bytes += config.activation_bytes_per_assignment
                        combine_bytes += config.activation_bytes_per_assignment
                        transfer(source, target, config.activation_bytes_per_assignment)
                        transfer(target, source, config.activation_bytes_per_assignment)
                    loads[target] += 1

                elif config.policy == "hot_expert_replication":
                    sites = replicas.get(instance, (home,))
                    target = source if source in sites else min(
                        sites, key=lambda chip: (loads[chip], chip)
                    )
                    if target != source:
                        remote_assignments += 1
                        remote_by_chip[source] += 1
                        dispatch_bytes += config.activation_bytes_per_assignment
                        combine_bytes += config.activation_bytes_per_assignment
                        transfer(source, target, config.activation_bytes_per_assignment)
                        transfer(target, source, config.activation_bytes_per_assignment)
                    loads[target] += 1

                else:
                    loads[source] += 1
                    if home == source:
                        continue
                    cache = caches[source]
                    if instance in cache:
                        cache_hits[source] += 1
                        cache.move_to_end(instance)
                        continue

                    cache_misses[source] += 1
                    remote_assignments += 1
                    remote_by_chip[source] += 1
                    weight_fetch_bytes += config.expert_weight_bytes
                    transfer(home, source, config.expert_weight_bytes)
                    if config.cache_capacity_bytes_per_chip < config.expert_weight_bytes:
                        uncacheable_misses[source] += 1
                        continue
                    while (
                        (len(cache) + 1) * config.expert_weight_bytes
                        > config.cache_capacity_bytes_per_chip
                    ):
                        cache.popitem(last=False)
                        cache_evictions[source] += 1
                    cache[instance] = None
                    peak_counts[source] = max(
                        peak_counts[source], base_counts[source] + len(cache)
                    )

    if config.policy == "local_lru_weight_cache":
        for chip, cache in enumerate(caches):
            residents[chip].update(cache)

    load = _load_summary(loads)
    conserved = (
        expected_assignments == observed_assignments == load["assignments"]
    )
    if not conserved:
        raise RuntimeError("internal routed-assignment conservation failure")

    final_counts = [len(values) for values in residents]
    resident_rows = [
        {
            "chip": chip,
            "base_expert_instances": base_counts[chip],
            "additional_expert_instances": final_counts[chip] - base_counts[chip],
            "final_expert_instances": final_counts[chip],
            "resident_bytes": final_counts[chip] * config.expert_weight_bytes,
            "peak_resident_bytes": peak_counts[chip] * config.expert_weight_bytes,
        }
        for chip in range(config.chip_count)
    ]

    cache_lookups = sum(cache_hits) + sum(cache_misses)
    link_bytes = dispatch_bytes + combine_bytes + weight_fetch_bytes
    endpoint_bottleneck_bytes = max(
        max(tx_bytes, default=0), max(rx_bytes, default=0)
    )
    link_time_s = (
        endpoint_bottleneck_bytes / config.link_bandwidth_bytes_per_s_per_chip
    )

    trace_receipt = _audit_receipt(
        trace.receipt,
        expected_subject_sha256=trace_hash,
        base_dir=base_dir,
    )
    if (
        trace_receipt["valid"]
        and trace_receipt["sample_count"] != coverage["token_step_count"]
    ):
        trace_receipt["valid"] = False
        trace_receipt["reasons"].append("sample_count_mismatch")
    model_receipt = _audit_receipt(
        config.model_bytes_receipt,
        expected_subject_sha256=_canonical_hash(model_bytes_subject(config)),
        base_dir=base_dir,
    )
    link_receipt = _audit_receipt(
        config.link_calibration_receipt,
        expected_subject_sha256=_canonical_hash(link_calibration_subject(config)),
        base_dir=base_dir,
    )
    placement_receipt: dict[str, Any] | None = None
    if config.policy == "hot_expert_replication" and hot_source == "explicit":
        placement_receipt = _audit_receipt(
            config.hot_placement_receipt,
            expected_subject_sha256=_canonical_hash(
                hot_placement_subject(config, hot_instances)
            ),
            base_dir=base_dir,
        )

    reasons: list[str] = [
        "selected_hardware_row_binding_missing",
        "producer_specific_model_link_artifact_schema_unvalidated",
    ]
    if trace.source_kind != "measured":
        reasons.append("routing_trace_is_not_measured")
    if not coverage["publication_complete"]:
        reasons.append("routing_trace_lacks_complete_48_layer_token_steps")
    for label, audit in (
        ("routing_trace_receipt", trace_receipt),
        ("model_bytes_receipt", model_receipt),
        ("link_calibration_receipt", link_receipt),
    ):
        if not audit["valid"]:
            reasons.extend(f"{label}:{reason}" for reason in audit["reasons"])
    if config.link_energy_j_per_byte <= 0:
        reasons.append("link_energy_j_per_byte_not_positive")
    if config.policy == "hot_expert_replication":
        if hot_source != "explicit":
            reasons.append("hot_experts_derived_from_evaluation_trace")
        elif placement_receipt is not None and not placement_receipt["valid"]:
            reasons.extend(
                f"hot_placement_receipt:{reason}"
                for reason in placement_receipt["reasons"]
            )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "model_contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
        },
        "policy": {
            "name": config.policy,
            "chip_count": config.chip_count,
            "expert_weight_bytes": config.expert_weight_bytes,
            "activation_bytes_per_assignment": (
                config.activation_bytes_per_assignment
            ),
            "cache_capacity_bytes_per_chip": config.cache_capacity_bytes_per_chip,
            "hot_expert_count": len(hot_instances),
            "hot_replica_count": config.hot_replica_count,
            "hot_selection_source": hot_source,
            "hot_replica_placement_rule": hot_replica_placement_rule,
            "hot_expert_placement": hot_rows,
            "base_shard_rule": "expert_id_mod_chip_count_across_all_layers",
        },
        "routing": {
            "trace_source_kind": trace.source_kind,
            "trace_content_hash": trace_hash,
            "coverage": coverage,
            "assignment_conservation": {
                "expected": expected_assignments,
                "observed": observed_assignments,
                "per_chip_sum": load["assignments"],
                "conserved": conserved,
            },
            "remote_assignments": remote_assignments,
            "remote_route_fraction": remote_assignments / expected_assignments,
            "remote_assignment_definition": (
                "off-chip token dispatch"
                if config.policy != "local_lru_weight_cache"
                else "non-owner assignment requiring an expert-weight fetch"
            ),
            "remote_assignments_by_source_chip": remote_by_chip,
        },
        "resident_weights": {
            "scope": "routed_expert_weights_only",
            "expert_instances_in_model": NUM_LAYERS * NUM_EXPERTS,
            "final_resident_bytes": sum(
                row["resident_bytes"] for row in resident_rows
            ),
            "sum_of_per_chip_peak_resident_bytes": sum(
                row["peak_resident_bytes"] for row in resident_rows
            ),
            "per_chip": resident_rows,
        },
        "load": load,
        "traffic": {
            "dispatch_activation_bytes": dispatch_bytes,
            "combine_activation_bytes": combine_bytes,
            "weight_fetch_bytes": weight_fetch_bytes,
            "logical_unicast_bytes": link_bytes,
            "per_chip": [
                {
                    "chip": chip,
                    "tx_bytes": tx_bytes[chip],
                    "rx_bytes": rx_bytes[chip],
                }
                for chip in range(config.chip_count)
            ],
        },
        "link": {
            "traffic_model": "logical_point_to_point_unicast",
            "all_to_all_collective_modelled": False,
            "contention_modelled": False,
            "overlap_modelled": False,
            "bandwidth_scope": "per_chip_full_duplex",
            "bandwidth_bytes_per_s_per_chip": (
                config.link_bandwidth_bytes_per_s_per_chip
            ),
            "endpoint_bottleneck_bytes": endpoint_bottleneck_bytes,
            "time_s": link_time_s,
            "time_interpretation": "calibrated_endpoint_lower_bound",
            "aggregate_serial_equivalent_time_s": (
                link_bytes / config.link_bandwidth_bytes_per_s_per_chip
            ),
            "energy_j_per_byte": config.link_energy_j_per_byte,
            "energy_j": link_bytes * config.link_energy_j_per_byte,
        },
        "cache": {
            "lookup_scope": "non-owner expert assignments",
            "lookups": cache_lookups,
            "hits": sum(cache_hits),
            "misses": sum(cache_misses),
            "evictions": sum(cache_evictions),
            "uncacheable_misses": sum(uncacheable_misses),
            "hit_fraction": sum(cache_hits) / cache_lookups if cache_lookups else None,
            "per_chip": [
                {
                    "chip": chip,
                    "hits": cache_hits[chip],
                    "misses": cache_misses[chip],
                    "evictions": cache_evictions[chip],
                    "uncacheable_misses": uncacheable_misses[chip],
                    "final_cached_expert_instances": len(caches[chip]),
                }
                for chip in range(config.chip_count)
            ],
        },
        "provenance": {
            "evidence_grade": (
                "measured_and_receipt_audited" if not reasons else "projected"
            ),
            "publication_rankable": not reasons,
            "publication_scope": (
                "routed_expert_placement_and_logical_unicast_ledger_only"
            ),
            "all_to_all_publication_rankable": False,
            "unrankable_reasons": reasons,
            "routing_trace_receipt": trace_receipt,
            "model_bytes_receipt": model_receipt,
            "link_calibration_receipt": link_receipt,
            "hot_placement_receipt": placement_receipt,
            "scope_limits": [
                "logical unicast bytes only",
                "no all-to-all collective or topology model",
                "no packet, switch-contention, or communication-overlap model",
                "link time is a calibrated per-chip endpoint lower bound",
            ],
        },
    }
    report["content_hash"] = _canonical_hash(report)
    return report


def _receipt_from_json(value: Any) -> EvidenceReceipt | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("receipt must be an object or null")
    command = value.get("command", ())
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("receipt command must be an argv array")
    return EvidenceReceipt(
        artifact_path=str(value.get("artifact_path", "")),
        artifact_sha256=str(value.get("artifact_sha256", "")),
        subject_sha256=str(value.get("subject_sha256", "")),
        command=tuple(str(part) for part in command),
        tool_revision=str(value.get("tool_revision", "")),
        recorded_at_utc=str(value.get("recorded_at_utc", "")),
        sample_count=value.get("sample_count", 0),
    )


def parse_analysis_input(payload: Mapping[str, Any]) -> tuple[RoutingTrace, PlacementConfig]:
    """Parse the strict JSON input contract used by the CLI."""

    if not isinstance(payload, Mapping):
        raise ValueError("placement input must be an object")
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"input schema must be {INPUT_SCHEMA!r}")
    trace_value = payload.get("trace")
    config_value = payload.get("config")
    if not isinstance(trace_value, Mapping) or not isinstance(config_value, Mapping):
        raise ValueError("input requires trace and config objects")

    steps = []
    raw_steps = trace_value.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("trace.steps must be an array")
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise ValueError("each routing step must be an object")
        raw_records = raw_step.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes)
        ):
            raise ValueError("routing step records must be an array")
        records_list = []
        for record_value in raw_records:
            if not isinstance(record_value, Mapping):
                raise ValueError("each route record must be an object")
            expert_ids = record_value.get("expert_ids")
            if not isinstance(expert_ids, Sequence) or isinstance(
                expert_ids, (str, bytes)
            ):
                raise ValueError("expert_ids must be an array")
            records_list.append(
                RouteRecord(
                    token_id=str(record_value["token_id"]),
                    layer=record_value["layer"],
                    source_chip=record_value["source_chip"],
                    expert_ids=tuple(expert_ids),
                )
            )
        records = tuple(records_list)
        steps.append(RoutingStep(step_index=raw_step["step_index"], records=records))

    trace = RoutingTrace(
        source_kind=str(trace_value["source_kind"]),
        steps=tuple(steps),
        receipt=_receipt_from_json(trace_value.get("receipt")),
        model_id=str(trace_value["model_id"]),
        model_revision=str(trace_value["model_revision"]),
    )
    hot_instances_value = config_value.get("hot_expert_instances")
    if hot_instances_value is not None and (
        not isinstance(hot_instances_value, Sequence)
        or isinstance(hot_instances_value, (str, bytes))
    ):
        raise ValueError("hot_expert_instances must be an array or null")
    hot_instances = (
        None
        if hot_instances_value is None
        else tuple((value[0], value[1]) for value in hot_instances_value)
    )
    config = PlacementConfig(
        policy=str(config_value["policy"]),
        chip_count=config_value["chip_count"],
        expert_weight_bytes=config_value["expert_weight_bytes"],
        activation_bytes_per_assignment=config_value[
            "activation_bytes_per_assignment"
        ],
        link_bandwidth_bytes_per_s_per_chip=config_value[
            "link_bandwidth_bytes_per_s_per_chip"
        ],
        link_energy_j_per_byte=config_value["link_energy_j_per_byte"],
        cache_capacity_bytes_per_chip=config_value.get(
            "cache_capacity_bytes_per_chip", 0
        ),
        hot_expert_count=config_value.get("hot_expert_count", 0),
        hot_replica_count=config_value.get("hot_replica_count", 1),
        hot_expert_instances=hot_instances,
        model_bytes_receipt=_receipt_from_json(
            config_value.get("model_bytes_receipt")
        ),
        link_calibration_receipt=_receipt_from_json(
            config_value.get("link_calibration_receipt")
        ),
        hot_placement_receipt=_receipt_from_json(
            config_value.get("hot_placement_receipt")
        ),
    )
    return trace, config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="placement input JSON")
    parser.add_argument("--output", type=Path, help="report path; default is stdout")
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    trace, config = parse_analysis_input(payload)
    report = analyze_expert_placement(
        trace, config, receipt_base_dir=args.input.resolve().parent
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
