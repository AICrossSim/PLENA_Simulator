"""Held-out hot-expert replication and ordered weight-cache sensitivity.

Permanent replica sites are learned from the sealed training prefix.  The
held-out suffix is used only to execute the already-fixed placement and the
deterministic runtime scheduler.  Replicated hidden/router state makes every
selected site rank-local, so hidden-state dispatch remains zero.  Dynamic
whole-expert cache fills are point-to-point weight transfers; the link ledger
is an endpoint-bandwidth lower bound and is not an all-to-all model.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .body_weight_layout import (
    EXPERT_ID_PARALLEL,
    BodyWeightPhysicalLayout,
)
from .expert_id_full_decode_projection import (
    CONTROL_CYCLIC,
    CONTROL_FREQUENCY,
    CONTROL_NAMES,
    CONTROL_TENSOR_EXACT,
    CONTROL_TENSOR_NATIVE,
    FullDecodeProjectionConfig,
    _comparison,
    _expert_id_full_body_layout,
    _run_control,
    _validate_target,
    build_expert_id_full_decode_projection,
    validate_expert_id_full_decode_projection,
)
from .expert_id_trace_balancing import (
    EXPERT_ID_MAPPING,
    ExpertIdBalanceConfig,
    _audit_trace,
    audit_expert_id_balancing_report,
    canonical_hash,
    file_hash,
)
from .expert_placement import (
    MODEL_ID,
    MODEL_REVISION,
    NUM_EXPERTS,
    NUM_LAYERS,
    TOP_K,
    RoutingTrace,
    trace_content_hash,
)
from .handoff import LINK_GENS
from .physical_ledger import PlaneBytes, WeightLedger, matrix_planes
from ..performance import disagg_decode as decode
from ..performance.perf_model import PerfModel


SENSITIVITY_SCHEMA = "plena-qwen3-moe-expert-id-replication-cache-sensitivity/v1"
SENSITIVITY_RECEIPT_SCHEMA = "plena-qwen3-moe-expert-id-replication-cache-sensitivity-receipt/v1"
ROUTE_PROJECTION_SCHEMA = decode.LAYER_EXACT_MOE_ROUTE_PROJECTION_SCHEMA
CACHE_POLICY = "rank_local_hbm_whole_expert_lru_cold_per_projected_step/v1"
PLACEMENT_POLICY = "training_frequency_hot_replica_greedy_with_exact_training_fallback/v1"
RUNTIME_POLICY = "least_resulting_rank_cycles_resident_then_rank_tiebreak/v1"
CANDIDATE_PREFIX = "frequency_expert_id_hot_replica_cache"


@dataclass(frozen=True)
class ReplicationCacheSensitivityConfig:
    """Sealed byte budgets searched for one held-out window."""

    replica_budget_bytes_per_rank: tuple[int, ...] = (0,)
    cache_budget_bytes_per_rank: tuple[int, ...] = (0,)
    cache_policy: str = CACHE_POLICY
    placement_policy: str = PLACEMENT_POLICY
    runtime_policy: str = RUNTIME_POLICY

    def __post_init__(self) -> None:
        for field in (
            "replica_budget_bytes_per_rank",
            "cache_budget_bytes_per_rank",
        ):
            raw = getattr(self, field)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ValueError(f"{field} must be an integer sequence")
            values = tuple(raw)
            if (
                not values
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
                or len(values) != len(set(values))
                or tuple(sorted(values)) != values
                or values[0] != 0
            ):
                raise ValueError(f"{field} must be sorted unique non-negative integers starting at zero")
            object.__setattr__(self, field, values)
        if self.cache_policy != CACHE_POLICY:
            raise ValueError("unsupported expert-weight cache policy")
        if self.placement_policy != PLACEMENT_POLICY:
            raise ValueError("unsupported hot-expert placement policy")
        if self.runtime_policy != RUNTIME_POLICY:
            raise ValueError("unsupported replica runtime policy")


def _hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_hash", None)
    return body | {"content_hash": canonical_hash(body)}


def _plane(value: PlaneBytes) -> dict[str, int]:
    return {
        "element_raw": int(value.element_raw),
        "element_aligned": int(value.element_aligned),
        "scale_raw": int(value.scale_raw),
        "scale_aligned": int(value.scale_aligned),
        "total_aligned": int(value.total_aligned),
    }


def _scale_plane(value: PlaneBytes, copies: int) -> PlaneBytes:
    if copies < 0:
        raise ValueError("plane copies must be non-negative")
    return PlaneBytes(
        element_raw=value.element_raw * copies,
        element_aligned=value.element_aligned * copies,
        scale_raw=value.scale_raw * copies,
        scale_aligned=value.scale_aligned * copies,
    )


def _sum_planes(values: Sequence[PlaneBytes]) -> PlaneBytes:
    result = PlaneBytes()
    for value in values:
        result += value
    return result


def _one_expert_plane(
    balance: ExpertIdBalanceConfig,
    precision: Mapping[str, Any],
) -> PlaneBytes:
    hidden = math.ceil(balance.hidden_size / balance.mlen) * balance.mlen
    intermediate = math.ceil(balance.expert_intermediate_size / balance.mlen) * balance.mlen
    common = {
        "element_bits": int(precision["ffn_elem"]),
        "effective_bits": float(precision["ffn_bits"]),
        "block_size": int(precision["block_size"]),
        "alignment_bytes": balance.weight_alignment_bytes,
    }
    return matrix_planes(intermediate, hidden, 2, **common) + matrix_planes(hidden, intermediate, 1, **common)


def _histogram(counts: Mapping[int, int]) -> dict[str, int]:
    values = Counter(int(value) for value in counts.values() if int(value) > 0)
    return {str(count): values[count] for count in sorted(values)}


def _timing(
    counts: Mapping[int, int],
    *,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
    resident_site_count: int,
    source: str,
) -> dict[str, Any]:
    return perf.moe_decode_expert_timing_from_histogram(
        balance.hidden_size,
        balance.expert_intermediate_size,
        _histogram(counts),
        owned_experts=max(1, resident_site_count),
        expected_route_assignments=sum(counts.values()),
        source=source,
    )


def _frequency_loads(frequencies: Sequence[int], sites_by_expert: Sequence[set[int]], tp: int) -> tuple[int, ...]:
    loads = [0] * tp
    for expert in sorted(range(NUM_EXPERTS), key=lambda value: (-int(frequencies[value]), value)):
        for _ in range(int(frequencies[expert])):
            target = min(sites_by_expert[expert], key=lambda rank: (loads[rank], rank))
            loads[target] += 1
    return tuple(loads)


def _training_window_cycles(
    ordered_steps: Sequence[Sequence[Any]],
    layer: int,
    sites_by_expert: Sequence[set[int]],
    *,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
) -> dict[str, int]:
    values: list[int] = []
    for start in range(0, len(ordered_steps), balance.batch_size):
        records = [step[layer] for step in ordered_steps[start : start + balance.batch_size]]
        counts, _ = _schedule_layer(
            records,
            sites_by_expert,
            caches=None,
            cache_slots_per_rank=0,
            one_expert=PlaneBytes(),
            perf=perf,
            balance=balance,
        )
        timings = [
            _timing(
                values_by_expert,
                perf=perf,
                balance=balance,
                resident_site_count=sum(rank in sites for sites in sites_by_expert),
                source="training_only_hot_replica_validation",
            )
            for rank, values_by_expert in enumerate(counts)
        ]
        values.append(max(int(value["expert_stage_cycles"]) for value in timings))
    return {
        "window_count": len(values),
        "total_slowest_rank_expert_stage_cycles": sum(values),
        "maximum_slowest_rank_expert_stage_cycles": max(values),
    }


def _learn_replica_sites(
    report: Mapping[str, Any],
    ordered_steps: Sequence[Sequence[Any]],
    *,
    replica_slots_per_rank: int,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
) -> tuple[list[list[set[int]]], dict[str, Any]]:
    """Learn permanent sites without consulting the held-out suffix."""

    placement_rows = report["placement"]["layers"]
    sites: list[list[set[int]]] = []
    frequencies: list[tuple[int, ...]] = []
    for layer in range(NUM_LAYERS):
        owners = tuple(int(value) for value in placement_rows[layer]["selected_expert_owner_by_id"])
        sites.append([{owners[expert]} for expert in range(NUM_EXPERTS)])
        frequencies.append(tuple(int(value) for value in placement_rows[layer]["training_expert_assignment_frequency"]))

    replica_counts = [0] * balance.tensor_parallel_degree
    candidate_order: list[tuple[float, int, int, int, int]] = []
    for layer in range(NUM_LAYERS):
        for expert, frequency in enumerate(frequencies[layer]):
            for replica_ordinal in range(1, balance.tensor_parallel_degree):
                marginal = frequency / (replica_ordinal + 1)
                candidate_order.append((-marginal, -frequency, layer, expert, replica_ordinal))
    candidate_order.sort()

    accepted: list[dict[str, Any]] = []
    for _, negative_frequency, layer, expert, replica_ordinal in candidate_order:
        current_sites = sites[layer][expert]
        if len(current_sites) != replica_ordinal:
            continue
        current_loads = _frequency_loads(frequencies[layer], sites[layer], balance.tensor_parallel_degree)
        current_objective = (
            max(current_loads),
            sum(value * value for value in current_loads),
        )
        choices: list[tuple[tuple[int, int], int, tuple[int, ...]]] = []
        for rank in range(balance.tensor_parallel_degree):
            if rank in current_sites or replica_counts[rank] >= replica_slots_per_rank:
                continue
            trial = [set(values) for values in sites[layer]]
            trial[expert].add(rank)
            trial_loads = _frequency_loads(frequencies[layer], trial, balance.tensor_parallel_degree)
            objective = (
                max(trial_loads),
                sum(value * value for value in trial_loads),
            )
            choices.append((objective, rank, trial_loads))
        if not choices:
            continue
        objective, rank, loads = min(choices)
        if objective > current_objective:
            continue
        sites[layer][expert].add(rank)
        replica_counts[rank] += 1
        accepted.append(
            {
                "layer": layer,
                "expert_id": expert,
                "replica_rank": rank,
                "training_frequency": -negative_frequency,
                "training_loads_before": list(current_loads),
                "training_loads_after": list(loads),
            }
        )

    layer_validation: list[dict[str, Any]] = []
    training_steps = ordered_steps[: balance.train_step_count]
    for layer in range(NUM_LAYERS):
        primary = [{int(owner)} for owner in placement_rows[layer]["selected_expert_owner_by_id"]]
        baseline = _training_window_cycles(training_steps, layer, primary, perf=perf, balance=balance)
        candidate = _training_window_cycles(training_steps, layer, sites[layer], perf=perf, balance=balance)
        nonregression = (
            candidate["total_slowest_rank_expert_stage_cycles"] <= baseline["total_slowest_rank_expert_stage_cycles"]
            and candidate["maximum_slowest_rank_expert_stage_cycles"]
            <= baseline["maximum_slowest_rank_expert_stage_cycles"]
        )
        if not nonregression:
            for expert in range(NUM_EXPERTS):
                for rank in sites[layer][expert] - primary[expert]:
                    replica_counts[rank] -= 1
                sites[layer][expert] = set(primary[expert])
            candidate = baseline
        layer_validation.append(
            {
                "layer": layer,
                "baseline_training_timing": baseline,
                "selected_training_timing": candidate,
                "training_nonregression": True,
                "fallback_to_no_replication": not nonregression,
                "held_out_steps_used": False,
            }
        )

    accepted_sites = [
        {
            "layer": layer,
            "expert_id": expert,
            "primary_rank": int(placement_rows[layer]["selected_expert_owner_by_id"][expert]),
            "resident_ranks": sorted(sites[layer][expert]),
        }
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
        if len(sites[layer][expert]) > 1
    ]
    final_counts = [
        sum(
            rank in sites[layer][expert] and rank != int(placement_rows[layer]["selected_expert_owner_by_id"][expert])
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        )
        for rank in range(balance.tensor_parallel_degree)
    ]
    if final_counts != replica_counts or any(value > replica_slots_per_rank for value in final_counts):
        raise AssertionError("replica placement exceeded its per-rank capacity")
    plan = {
        "policy": PLACEMENT_POLICY,
        "training_step_count": balance.train_step_count,
        "training_step_indices_content_hash": report["split"]["training"]["step_indices_content_hash"],
        "held_out_steps_used": False,
        "replica_slot_limit_per_rank": replica_slots_per_rank,
        "replica_site_count_by_rank": final_counts,
        "replica_sites": accepted_sites,
        "all_128_experts_retain_one_primary_site_per_layer": True,
        "all_layers_training_nonregression": True,
        "layer_training_validation": layer_validation,
        "greedy_acceptance_log_before_exact_layer_fallback": accepted,
    }
    plan["content_hash"] = canonical_hash(plan)
    return sites, plan


def _schedule_layer(
    records: Sequence[Any],
    permanent_sites: Sequence[set[int]],
    *,
    primary_owner_by_id: Sequence[int] | None = None,
    caches: list[OrderedDict[tuple[int, int], None]] | None,
    cache_slots_per_rank: int,
    one_expert: PlaneBytes,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
) -> tuple[list[Counter[int]], dict[str, Any]]:
    """Schedule each route once; cache state is mutated in deterministic order."""

    tp = balance.tensor_parallel_degree
    counts: list[Counter[int]] = [Counter() for _ in range(tp)]
    cache_rows = {
        "hits_by_rank": [0] * tp,
        "misses_by_rank": [0] * tp,
        "evictions_by_rank": [0] * tp,
        "fills_by_rank": [0] * tp,
        "source_reads_by_rank": [PlaneBytes() for _ in range(tp)],
        "destination_writes_by_rank": [PlaneBytes() for _ in range(tp)],
        "link_tx_by_rank": [PlaneBytes() for _ in range(tp)],
        "link_rx_by_rank": [PlaneBytes() for _ in range(tp)],
        "peak_cache_entries_by_rank": [len(cache) for cache in caches] if caches is not None else [0] * tp,
    }
    layer = int(records[0].layer) if records else -1
    for record in records:
        if int(record.layer) != layer:
            raise ValueError("layer scheduler received mixed layers")
        for expert_object in record.expert_ids:
            expert = int(expert_object)
            choices: list[tuple[int, int, int, int]] = []
            for rank in range(tp):
                key = (layer, expert)
                permanent = rank in permanent_sites[expert]
                cached = caches is not None and key in caches[rank]
                fill = not permanent and not cached
                if fill and cache_slots_per_rank <= 0:
                    continue
                trial = counts[rank].copy()
                trial[expert] += 1
                site_count = sum(rank in sites for sites in permanent_sites)
                if caches is not None:
                    site_count += len(caches[rank]) + int(fill)
                cycles = int(
                    _timing(
                        trial,
                        perf=perf,
                        balance=balance,
                        resident_site_count=max(1, site_count),
                        source="held_out_replica_cache_runtime_scheduler",
                    )["expert_stage_cycles"]
                )
                choices.append((cycles, int(fill), sum(trial.values()), rank))
            if not choices:
                raise AssertionError("every expert must retain an executable site")
            _, fill_value, _, target = min(choices)
            key = (layer, expert)
            permanent = target in permanent_sites[expert]
            if caches is not None and not permanent:
                cache = caches[target]
                if key in cache:
                    cache_rows["hits_by_rank"][target] += 1
                    cache.move_to_end(key)
                else:
                    if fill_value != 1:
                        raise AssertionError("cache miss target was not charged")
                    cache_rows["misses_by_rank"][target] += 1
                    cache_rows["fills_by_rank"][target] += 1
                    while len(cache) >= cache_slots_per_rank:
                        cache.popitem(last=False)
                        cache_rows["evictions_by_rank"][target] += 1
                    cache[key] = None
                    source = (
                        int(primary_owner_by_id[expert])
                        if primary_owner_by_id is not None
                        else min(permanent_sites[expert])
                    )
                    if source not in permanent_sites[expert]:
                        raise AssertionError("cache source is not a permanent site")
                    if source == target:
                        raise AssertionError("cache fill source equals resident target")
                    cache_rows["source_reads_by_rank"][source] = cache_rows["source_reads_by_rank"][source] + one_expert
                    cache_rows["destination_writes_by_rank"][target] = (
                        cache_rows["destination_writes_by_rank"][target] + one_expert
                    )
                    cache_rows["link_tx_by_rank"][source] = cache_rows["link_tx_by_rank"][source] + one_expert
                    cache_rows["link_rx_by_rank"][target] = cache_rows["link_rx_by_rank"][target] + one_expert
                    cache_rows["peak_cache_entries_by_rank"][target] = max(
                        cache_rows["peak_cache_entries_by_rank"][target],
                        len(cache),
                    )
            counts[target][expert] += 1
    expected = len(records) * TOP_K
    if sum(sum(values.values()) for values in counts) != expected:
        raise AssertionError("runtime route scheduling does not conserve")
    return counts, cache_rows


def _cache_accumulator(tp: int) -> dict[str, Any]:
    return {
        "hits_by_rank": [0] * tp,
        "misses_by_rank": [0] * tp,
        "evictions_by_rank": [0] * tp,
        "fills_by_rank": [0] * tp,
        "peak_cache_entries_by_rank": [0] * tp,
        "source_reads_by_rank": [PlaneBytes() for _ in range(tp)],
        "destination_writes_by_rank": [PlaneBytes() for _ in range(tp)],
        "link_tx_by_rank": [PlaneBytes() for _ in range(tp)],
        "link_rx_by_rank": [PlaneBytes() for _ in range(tp)],
    }


def _merge_cache_rows(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for field in ("hits_by_rank", "misses_by_rank", "evictions_by_rank", "fills_by_rank"):
        for rank, value in enumerate(source[field]):
            target[field][rank] += int(value)
    for rank, value in enumerate(source["peak_cache_entries_by_rank"]):
        target["peak_cache_entries_by_rank"][rank] = max(target["peak_cache_entries_by_rank"][rank], int(value))
    for field in (
        "source_reads_by_rank",
        "destination_writes_by_rank",
        "link_tx_by_rank",
        "link_rx_by_rank",
    ):
        for rank, value in enumerate(source[field]):
            target[field][rank] = target[field][rank] + value


def _evaluate_held_out_route(
    report: Mapping[str, Any],
    overlay: Mapping[str, Any],
    held_out_steps: Sequence[Sequence[Any]],
    permanent_sites: Sequence[Sequence[set[int]]],
    placement_plan: Mapping[str, Any],
    *,
    cache_slots_per_rank: int,
    one_expert: PlaneBytes,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the 48-layer route hook and the exact cache endpoint ledger."""

    tp = balance.tensor_parallel_degree
    kvp = balance.kv_parallel_degree
    caches = [OrderedDict() for _ in range(tp)]
    cache_totals = _cache_accumulator(tp)
    layers: list[dict[str, Any]] = []
    primary_owners = [
        tuple(int(value) for value in report["placement"]["layers"][layer]["selected_expert_owner_by_id"])
        for layer in range(NUM_LAYERS)
    ]
    for layer in range(NUM_LAYERS):
        records = [step[layer] for step in held_out_steps]
        counts_by_rank, cache_row = _schedule_layer(
            records,
            permanent_sites[layer],
            primary_owner_by_id=primary_owners[layer],
            caches=caches,
            cache_slots_per_rank=cache_slots_per_rank,
            one_expert=one_expert,
            perf=perf,
            balance=balance,
        )
        _merge_cache_rows(cache_totals, cache_row)
        site_counts = [sum(rank in sites for sites in permanent_sites[layer]) for rank in range(tp)]
        timings = [
            _timing(
                counts,
                perf=perf,
                balance=balance,
                resident_site_count=site_counts[rank] + cache_slots_per_rank,
                source="held_out_hot_replica_cache_exact_rank_histogram",
            )
            for rank, counts in enumerate(counts_by_rank)
        ]
        cycles = [int(value["expert_stage_cycles"]) for value in timings]
        active_counts = [len(values) for values in counts_by_rank]
        rank_streamed = [_scale_plane(one_expert, value) for value in active_counts]
        rank_resident = [_scale_plane(one_expert, value) for value in site_counts]
        assignments = [sum(values.values()) for values in counts_by_rank]
        logical = balance.batch_size * TOP_K
        if sum(assignments) != logical:
            raise AssertionError("held-out route assignment was duplicated or lost")
        unique_ids = {expert for counts in counts_by_rank for expert in counts}
        layer_cache = {
            field: list(cache_row[field])
            for field in (
                "hits_by_rank",
                "misses_by_rank",
                "evictions_by_rank",
                "fills_by_rank",
                "peak_cache_entries_by_rank",
            )
        } | {
            field: [_plane(value) for value in cache_row[field]]
            for field in (
                "source_reads_by_rank",
                "destination_writes_by_rank",
                "link_tx_by_rank",
                "link_rx_by_rank",
            )
        }
        layers.append(
            {
                "layer": layer,
                "primary_expert_owner_by_id": list(primary_owners[layer]),
                "permanent_resident_ranks_by_expert_id": [sorted(values) for values in permanent_sites[layer]],
                "all_128_experts_have_a_primary_site": all(
                    primary_owners[layer][expert] in permanent_sites[layer][expert] for expert in range(NUM_EXPERTS)
                ),
                "permanent_expert_site_count_by_rank": site_counts,
                "active_experts_per_rank": active_counts,
                "active_expert_ids_by_rank": [sorted(values) for values in counts_by_rank],
                "unique_active_experts": len(unique_ids),
                "expert_token_counts_by_rank": [
                    {str(key): int(value) for key, value in sorted(counts.items())} for counts in counts_by_rank
                ],
                "expert_token_count_histogram_by_rank": [_histogram(values) for values in counts_by_rank],
                "assignment_count_by_rank": assignments,
                "physical_assignment_count_by_rank_across_kvp": [value * kvp for value in assignments],
                "expert_stage_cycles_by_rank": cycles,
                "matrix_instruction_histogram_by_rank": [
                    dict(value["matrix_instruction_histogram"]) for value in timings
                ],
                "auxiliary_instruction_histogram_by_rank": [
                    dict(value["auxiliary_instruction_histogram"]) for value in timings
                ],
                "matrix_cycles_by_rank": [int(value["matrix_cycles"]) for value in timings],
                "activation_cycles_by_rank": [int(value["activation_cycles"]) for value in timings],
                "auxiliary_cycles_by_rank": [int(value["auxiliary_cycles"]) for value in timings],
                "slowest_rank_expert_stage_cycles": max(cycles),
                "expert_streamed_by_rank": [_plane(value) for value in rank_streamed],
                "slowest_rank_expert_streamed": _plane(max(rank_streamed, key=lambda value: value.total_aligned)),
                "system_expert_streamed": _plane(_scale_plane(_sum_planes(rank_streamed), kvp)),
                "permanent_expert_resident_by_rank": [_plane(value) for value in rank_resident],
                "slowest_rank_expert_resident": _plane(max(rank_resident, key=lambda value: value.total_aligned)),
                "system_expert_resident": _plane(_scale_plane(_sum_planes(rank_resident), kvp)),
                "cache": layer_cache,
                "logical_route_assignments": logical,
                "physical_whole_expert_assignments_across_kvp": logical * kvp,
                "source_hidden_dispatch_bytes": 0,
                "expert_output_collective_count": int(tp > 1),
            }
        )

    permanent_counts_by_rank = [
        sum(rank in permanent_sites[layer][expert] for layer in range(NUM_LAYERS) for expert in range(NUM_EXPERTS))
        for rank in range(tp)
    ]
    cache_reserved_by_rank = [cache_slots_per_rank] * tp
    total_resident_counts = [
        permanent + cache for permanent, cache in zip(permanent_counts_by_rank, cache_reserved_by_rank)
    ]
    rank_resident = [_scale_plane(one_expert, value) for value in total_resident_counts]
    rank_streamed_totals = [
        _sum_planes(
            [
                PlaneBytes(
                    element_raw=int(layer["expert_streamed_by_rank"][rank]["element_raw"]),
                    element_aligned=int(layer["expert_streamed_by_rank"][rank]["element_aligned"]),
                    scale_raw=int(layer["expert_streamed_by_rank"][rank]["scale_raw"]),
                    scale_aligned=int(layer["expert_streamed_by_rank"][rank]["scale_aligned"]),
                )
                for layer in layers
            ]
        )
        for rank in range(tp)
    ]
    logical = NUM_LAYERS * balance.batch_size * TOP_K
    physical = logical * kvp
    route_body = {
        "schema": ROUTE_PROJECTION_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "report_content_hash": report["content_hash"],
        "source_content_hash": placement_plan["content_hash"],
        "window_index": int(overlay["window_index"]),
        "first_step_index": int(overlay["first_step_index"]),
        "last_step_index": int(overlay["last_step_index"]),
        "study_config_content_hash": report["study_config_content_hash"],
        "policy": CANDIDATE_PREFIX,
        "mapping": EXPERT_ID_MAPPING,
        "expert_parallel_mode": EXPERT_ID_PARALLEL,
        "batch_size": balance.batch_size,
        "tensor_parallel_degree": tp,
        "kv_parallel_degree": kvp,
        "layer_count": NUM_LAYERS,
        "global_layer_collapse_allowed": False,
        "layers": layers,
        "totals": {
            "sum_of_per_layer_slowest_rank_expert_stage_cycles": sum(
                int(value["slowest_rank_expert_stage_cycles"]) for value in layers
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_element_bytes": sum(
                int(value["slowest_rank_expert_streamed"]["element_aligned"]) for value in layers
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_scale_bytes": sum(
                int(value["slowest_rank_expert_streamed"]["scale_aligned"]) for value in layers
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_bytes": sum(
                int(value["slowest_rank_expert_streamed"]["total_aligned"]) for value in layers
            ),
            "system_expert_streamed_bytes": sum(
                int(value["system_expert_streamed"]["total_aligned"]) for value in layers
            ),
            "per_rank_expert_streamed": [_plane(value) for value in rank_streamed_totals],
            "per_rank_expert_resident_including_cache_reservation": [_plane(value) for value in rank_resident],
            "rank_aggregated_then_slowest_expert_resident_bytes": max(value.total_aligned for value in rank_resident),
            "system_expert_resident_bytes": (sum(value.total_aligned for value in rank_resident) * kvp),
            "sum_of_per_layer_slowest_rank_expert_resident_bytes": sum(
                int(value["slowest_rank_expert_resident"]["total_aligned"]) for value in layers
            ),
            "resident_capacity_uses_rank_aggregate_not_layer_max_sum": True,
            "logical_route_assignments": logical,
            "expected_logical_route_assignments": logical,
            "physical_whole_expert_assignments_across_kvp": physical,
            "expected_physical_whole_expert_assignments_across_kvp": physical,
            "source_hidden_dispatch_bytes": 0,
            "expert_output_collective_count": NUM_LAYERS * int(tp > 1),
            "all_128_experts_have_a_primary_site_per_layer": True,
        },
        "classification": {
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
            "compiler_validated": False,
            "emulator_validated": False,
            "rtl_validated": False,
            "power_calibrated": False,
            "headline_eligible": False,
            "headline_win_claimed": False,
        },
    }
    route_projection = _hashed(route_body)

    link_tx = cache_totals["link_tx_by_rank"]
    link_rx = cache_totals["link_rx_by_rank"]
    source_reads = cache_totals["source_reads_by_rank"]
    destination_writes = cache_totals["destination_writes_by_rank"]
    logical_fetch = _sum_planes(link_tx)
    if logical_fetch.total_aligned != _sum_planes(link_rx).total_aligned:
        raise AssertionError("cache link transmit and receive planes differ")
    lookups = sum(cache_totals["hits_by_rank"]) + sum(cache_totals["misses_by_rank"])
    cache_ledger = {
        "policy": CACHE_POLICY,
        "storage_tier": "rank_local_hbm_whole_expert_slots",
        "onchip_sram_cache_modelled": False,
        "cache_reset_scope": "empty_at_each_stationary_projected_batch_step",
        "ordered_traversal": "decoder_layer_then_trace_step_then_topk_order",
        "capacity_slots_per_rank": cache_slots_per_rank,
        "capacity_bytes_per_rank": cache_slots_per_rank * one_expert.total_aligned,
        "lookups": lookups,
        "hits": sum(cache_totals["hits_by_rank"]),
        "misses": sum(cache_totals["misses_by_rank"]),
        "fills": sum(cache_totals["fills_by_rank"]),
        "evictions": sum(cache_totals["evictions_by_rank"]),
        "hit_fraction": (sum(cache_totals["hits_by_rank"]) / lookups if lookups else None),
        "per_rank": [
            {
                "rank": rank,
                "hits": cache_totals["hits_by_rank"][rank],
                "misses": cache_totals["misses_by_rank"][rank],
                "fills": cache_totals["fills_by_rank"][rank],
                "evictions": cache_totals["evictions_by_rank"][rank],
                "peak_entries": cache_totals["peak_cache_entries_by_rank"][rank],
                "final_entries": len(caches[rank]),
                "source_hbm_reads": _plane(source_reads[rank]),
                "destination_hbm_writes": _plane(destination_writes[rank]),
                "link_tx": _plane(link_tx[rank]),
                "link_rx": _plane(link_rx[rank]),
            }
            for rank in range(tp)
        ],
        "logical_fetch_plane_per_tp_group": _plane(logical_fetch),
        "physical_fetch_plane_across_kvp": _plane(_scale_plane(logical_fetch, kvp)),
        "system_hbm_fill_read_plane_across_kvp": _plane(_scale_plane(_sum_planes(source_reads), kvp)),
        "system_hbm_fill_write_plane_across_kvp": _plane(_scale_plane(_sum_planes(destination_writes), kvp)),
        "slowest_rank_fill_hbm_bytes": max(
            (source_reads[rank].total_aligned + destination_writes[rank].total_aligned for rank in range(tp)),
            default=0,
        ),
        "endpoint_bottleneck_bytes_per_tp_group": max(
            max((value.total_aligned for value in link_tx), default=0),
            max((value.total_aligned for value in link_rx), default=0),
        ),
        "source_hidden_dispatch_bytes": 0,
        "all_to_all_collective_modelled": False,
        "switch_contention_modelled": False,
        "communication_overlap_modelled": False,
        "energy_modelled": False,
    }
    cache_ledger["content_hash"] = canonical_hash(cache_ledger)
    return route_projection, cache_ledger


def _replace_weight_ledger_residence(value: WeightLedger, extra: PlaneBytes) -> WeightLedger:
    return replace(value, ffn_resident=value.ffn_resident + extra)


def _replicated_body_layout(
    base: BodyWeightPhysicalLayout,
    *,
    replica_site_count_by_rank: Sequence[int],
    cache_slots_per_rank: int,
    one_expert: PlaneBytes,
    balance: ExpertIdBalanceConfig,
) -> BodyWeightPhysicalLayout:
    extra_counts = [int(value) + cache_slots_per_rank for value in replica_site_count_by_rank]
    slowest_extra = _scale_plane(one_expert, max(extra_counts, default=0))
    group_extra = _scale_plane(one_expert, sum(extra_counts))
    system_extra = _scale_plane(one_expert, sum(extra_counts) * balance.kv_parallel_degree)
    provenance = dict(base.provenance) | {
        "replication_cache_sensitivity": True,
        "per_rank_additional_permanent_replica_and_cache_slots": extra_counts,
        "per_rank_permanent_replica_sites": [int(value) for value in replica_site_count_by_rank],
        "per_rank_reserved_cache_slots": [cache_slots_per_rank] * balance.tensor_parallel_degree,
        "resident_capacity_aggregation": ("sum_sites_per_rank_across_48_layers_then_select_slowest_rank"),
        "cache_storage_tier": "hbm",
        "whole_expert_sram_cache_assumed": False,
        "publication_rankable": False,
        "selection_eligible": False,
    }
    return BodyWeightPhysicalLayout(
        slowest_rank=_replace_weight_ledger_residence(base.slowest_rank, slowest_extra),
        tensor_parallel_group=_replace_weight_ledger_residence(base.tensor_parallel_group, group_extra),
        system=_replace_weight_ledger_residence(base.system, system_extra),
        provenance=provenance,
    )


def _apply_cache_endpoint_adjustment(
    control: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    perf: PerfModel,
    balance: ExpertIdBalanceConfig,
    serving: FullDecodeProjectionConfig,
) -> dict[str, Any]:
    """Charge cold cache fills without changing any decode-loop component."""

    result = copy.deepcopy(dict(control))
    loop = result["loop"]
    proof = result["component_and_hbm_step_proof"]
    system_read = int(cache["system_hbm_fill_read_plane_across_kvp"]["total_aligned"])
    system_write = int(cache["system_hbm_fill_write_plane_across_kvp"]["total_aligned"])
    physical_link = int(cache["physical_fetch_plane_across_kvp"]["total_aligned"])
    rank_hbm = int(cache["slowest_rank_fill_hbm_bytes"])
    endpoint_bytes = int(cache["endpoint_bottleneck_bytes_per_tp_group"])
    endpoint_time = (
        endpoint_bytes / (LINK_GENS[balance.link_generation] * balance.tp_link_ports) if endpoint_bytes else 0.0
    )
    peak_bw = decode.peak_hbm_bw_bytes(perf.config)
    totals = {
        "time_s": 0.0,
        "compute_s": 0.0,
        "memory_s": 0.0,
        "collective_s": 0.0,
        "expert_weight_fetch_endpoint_s": 0.0,
        "system_bytes": 0.0,
    }
    memory_bound_steps = 0
    adjusted_samples: list[dict[str, Any]] = []
    for sample_value in proof["samples"]:
        sample = copy.deepcopy(dict(sample_value))
        span = int(sample["represented_output_steps"])
        compute_s = float(sample["compute_time_s"])
        collective_s = float(sample["collective_time_s"])
        adjusted_rank_bytes = float(sample["slowest_rank_hbm_bytes_after_overfetch"]) + rank_hbm
        adjusted_system_bytes = float(sample["system_hbm_bytes_after_overfetch"]) + system_read + system_write
        memory_s = adjusted_rank_bytes / peak_bw
        step_s = max(compute_s, memory_s) + collective_s + endpoint_time
        sample.update(
            {
                "cache_fill_slowest_rank_hbm_bytes": rank_hbm,
                "cache_fill_system_hbm_read_bytes": system_read,
                "cache_fill_system_hbm_write_bytes": system_write,
                "cache_fill_physical_link_bytes_across_kvp": physical_link,
                "cache_fill_endpoint_lower_bound_time_s": endpoint_time,
                "slowest_rank_hbm_bytes_after_overfetch_and_cache_fill": (adjusted_rank_bytes),
                "system_hbm_bytes_after_overfetch_and_cache_fill": (adjusted_system_bytes),
                "slowest_rank_hbm_time_s_after_cache_fill": memory_s,
                "projected_step_time_s_after_cache_fill": step_s,
                "all_nonroute_component_cycles_unchanged": True,
                "expert_output_collective_unchanged": True,
            }
        )
        adjusted_samples.append(sample)
        totals["time_s"] += step_s * span
        totals["compute_s"] += compute_s * span
        totals["memory_s"] += memory_s * span
        totals["collective_s"] += collective_s * span
        totals["expert_weight_fetch_endpoint_s"] += endpoint_time * span
        totals["system_bytes"] += adjusted_system_bytes * span
        memory_bound_steps += int(memory_s >= compute_s) * span

    output_steps = serving.output_sequence_tokens
    total_time = totals["time_s"]
    base_traffic = loop["traffic_breakdown_per_batch_step"]
    base_read = sum(float(value) for name, value in base_traffic.items() if name.endswith("_read_bytes"))
    base_write = sum(float(value) for name, value in base_traffic.items() if name.endswith("_write_bytes"))
    loop["traffic_breakdown_per_batch_step"] = dict(base_traffic) | {
        "expert_weight_cache_fill_read_bytes": float(system_read),
        "expert_weight_cache_fill_write_bytes": float(system_write),
    }
    loop["traffic_breakdown_per_generated_token"] = {
        name: float(value) / balance.batch_size for name, value in loop["traffic_breakdown_per_batch_step"].items()
    }
    loop.update(
        {
            "total_time": total_time,
            "tpot": total_time / output_steps,
            "tps": balance.batch_size * output_steps / total_time,
            "first_step": adjusted_samples[0]["projected_step_time_s_after_cache_fill"],
            "read_bytes_per_second": ((base_read + system_read) * output_steps / total_time),
            "write_bytes_per_second": ((base_write + system_write) * output_steps / total_time),
            "array_active_fraction": min(1.0, totals["compute_s"] / total_time),
            "avg_bytes_per_batch_step": totals["system_bytes"] / output_steps,
            "avg_bytes_per_generated_token": (totals["system_bytes"] / output_steps / balance.batch_size),
            "avg_bytes_per_token": totals["system_bytes"] / output_steps,
            "avg_memory_seconds": totals["memory_s"] / output_steps,
            "avg_expert_weight_fetch_endpoint_seconds": (totals["expert_weight_fetch_endpoint_s"] / output_steps),
            "expert_weight_fetch_link_bytes_per_batch_step": physical_link,
            "expert_weight_fetch_link_bytes_per_generated_token": (physical_link / balance.batch_size),
            "link_bytes_per_second": (
                (float(loop["collective_bytes_per_batch_step"]) + physical_link) * output_steps / total_time
            ),
            "frac_mem_bound": memory_bound_steps / output_steps,
            "frac_communication_bound": (totals["collective_s"] + totals["expert_weight_fetch_endpoint_s"])
            / total_time,
        }
    )
    proof["base_native_hook_weighted_totals"] = copy.deepcopy(proof["weighted_totals"])
    proof["samples"] = adjusted_samples
    proof["weighted_totals"] = totals
    proof["checks"] = {
        "total_time_matches_step_proof": math.isclose(
            float(loop["total_time"]), totals["time_s"], rel_tol=1e-12, abs_tol=1e-15
        ),
        "compute_time_matches_step_proof": math.isclose(
            float(loop["avg_realized_compute_seconds"]),
            totals["compute_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "memory_time_matches_step_proof": math.isclose(
            float(loop["avg_memory_seconds"]),
            totals["memory_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "collective_time_matches_step_proof": math.isclose(
            float(loop["avg_collective_seconds"]),
            totals["collective_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "endpoint_time_matches_step_proof": math.isclose(
            float(loop["avg_expert_weight_fetch_endpoint_seconds"]),
            totals["expert_weight_fetch_endpoint_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "system_hbm_bytes_match_step_proof": math.isclose(
            float(loop["avg_bytes_per_batch_step"]),
            totals["system_bytes"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ),
    }
    if not all(proof["checks"].values()):
        raise AssertionError("cache-adjusted loop differs from its component proof")
    proof["cache_fill_checks"] = {
        "route_cycles_unchanged": True,
        "attention_router_topk_combine_head_kv_unchanged": True,
        "expert_output_collective_charged_exactly_once": True,
        "system_fill_hbm_bytes_equal_read_plus_write": (system_read + system_write == 2 * physical_link),
        "endpoint_time_uses_existing_tp_link_rate": True,
    }
    result["cache_endpoint_adjustment"] = {
        "cache_ledger_content_hash": cache["content_hash"],
        "link_scope": "per_rank_full_duplex_endpoint_bandwidth_lower_bound",
        "dependency_composition": "additive_required_weight_fetch_before_use",
        "bandwidth_bytes_per_s_per_port": LINK_GENS[balance.link_generation],
        "tp_link_ports": balance.tp_link_ports,
        "endpoint_bottleneck_bytes_per_tp_group": endpoint_bytes,
        "endpoint_lower_bound_time_s_per_projected_step": endpoint_time,
        "physical_link_bytes_across_kvp_per_projected_step": physical_link,
        "system_hbm_fill_read_bytes_per_projected_step": system_read,
        "system_hbm_fill_write_bytes_per_projected_step": system_write,
        "all_to_all_modelled": False,
        "contention_modelled": False,
        "overlap_modelled": False,
        "energy_modelled": False,
        "other_decode_components_changed": False,
    }
    return result


def _candidate_metrics(candidate: Mapping[str, Any]) -> dict[str, float | bool]:
    control = candidate["full_decode_control"]
    loop = control["loop"]
    capacity = control["capacity"]
    cache = candidate["cache_ledger"]
    return {
        "tpot_s": float(loop["tpot"]),
        "throughput_tokens_per_s": float(loop["tps"]),
        "system_hbm_bytes_per_batch_step": float(loop["avg_bytes_per_batch_step"]),
        "remote_fetch_link_bytes_per_batch_step": float(cache["physical_fetch_plane_across_kvp"]["total_aligned"]),
        "total_accounted_hbm_plus_link_bytes_per_batch_step": float(loop["avg_bytes_per_batch_step"])
        + float(cache["physical_fetch_plane_across_kvp"]["total_aligned"]),
        "slowest_rank_hbm_required_bytes": float(capacity["slowest_rank_hbm_required_bytes"]),
        "slowest_rank_capacity_margin_bytes": float(capacity["slowest_rank_capacity_margin_bytes"]),
        "fits_hbm": bool(capacity["fits_hbm"]),
        "fits_runtime": bool(capacity["fits_runtime"]),
    }


def _cold_vs_ideal_warm_bounds(
    control: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    output_steps: int,
) -> dict[str, Any]:
    """Expose the charged cold point and an explicitly optimistic zero-fill bound."""

    cold_loop = control["loop"]
    base_totals = control["component_and_hbm_step_proof"]["base_native_hook_weighted_totals"]
    ideal_total_time = float(base_totals["time_s"])
    ideal_tpot = ideal_total_time / output_steps
    ideal_hbm = float(base_totals["system_bytes"]) / output_steps
    cold_tpot = float(cold_loop["tpot"])
    cold_hbm = float(cold_loop["avg_bytes_per_batch_step"])
    misses = int(cache["misses"])
    return {
        "cold_reset_projection": {
            "cache_state": "empty_at_each_stationary_projected_batch_step",
            "tpot_s": cold_tpot,
            "throughput_tokens_per_s": float(cold_loop["tps"]),
            "system_hbm_bytes_per_batch_step": cold_hbm,
            "remote_fetch_link_bytes_per_batch_step": int(cache["physical_fetch_plane_across_kvp"]["total_aligned"]),
            "all_fills_charged": True,
        },
        "ideal_zero_fill_warm_lower_bound": {
            "assumption": (
                "retain_the_same_cache_enabled_rank_execution_schedule_but_set_"
                "all_cache_fill_hbm_and_link_costs_to_zero"
            ),
            "tpot_s": ideal_tpot,
            "throughput_tokens_per_s": (batch_size * output_steps / ideal_total_time),
            "system_hbm_bytes_per_batch_step": ideal_hbm,
            "remote_fetch_link_bytes_per_batch_step": 0,
            "capacity_enforceable_for_observed_schedule": misses == 0,
            "not_a_simulated_persistent_lru_result": True,
            "optimistic_lower_bound_only": True,
        },
        "cold_minus_ideal": {
            "tpot_s": cold_tpot - ideal_tpot,
            "tpot_percent_of_ideal": (100.0 * (cold_tpot - ideal_tpot) / ideal_tpot),
            "system_hbm_bytes_per_batch_step": cold_hbm - ideal_hbm,
        },
        "persistent_across_decode_steps_lru_modelled": False,
        "persistent_lru_deferred_reason": (
            "persistent_cache_state_can_change_each_layer_rank_histogram_and_"
            "therefore_requires_a_per_decode_step_48_layer_route_hook;the_"
            "current_native_hook_is_intentionally_stationary"
        ),
        "included_in_pareto_or_best_selection": False,
        "headline_win_claimed": False,
    }


def _pareto_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        value
        for value in candidates
        if value.get("status") == "evaluated"
        and value["metrics"]["fits_hbm"] is True
        and value["metrics"]["fits_runtime"] is True
    ]
    dimensions = (
        "tpot_s",
        "system_hbm_bytes_per_batch_step",
        "remote_fetch_link_bytes_per_batch_step",
        "slowest_rank_hbm_required_bytes",
    )

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_metrics = left["metrics"]
        right_metrics = right["metrics"]
        return all(float(left_metrics[field]) <= float(right_metrics[field]) for field in dimensions) and any(
            float(left_metrics[field]) < float(right_metrics[field]) for field in dimensions
        )

    pareto = [
        value
        for value in eligible
        if not any(other["candidate_id"] != value["candidate_id"] and dominates(other, value) for other in eligible)
    ]
    pareto.sort(
        key=lambda value: (
            float(value["metrics"]["tpot_s"]),
            float(value["metrics"]["system_hbm_bytes_per_batch_step"]),
            float(value["metrics"]["remote_fetch_link_bytes_per_batch_step"]),
            float(value["metrics"]["slowest_rank_hbm_required_bytes"]),
            str(value["candidate_id"]),
        )
    )

    def best(key: Any) -> str | None:
        return min(eligible, key=key)["candidate_id"] if eligible else None

    return {
        "eligible_candidate_count": len(eligible),
        "objective_directions": {field: "minimize" for field in dimensions},
        "pareto_candidate_ids": [value["candidate_id"] for value in pareto],
        "best_by_latency_candidate_id": best(
            lambda value: (float(value["metrics"]["tpot_s"]), str(value["candidate_id"]))
        ),
        "best_by_traffic_candidate_id": best(
            lambda value: (
                float(value["metrics"]["system_hbm_bytes_per_batch_step"]),
                float(value["metrics"]["remote_fetch_link_bytes_per_batch_step"]),
                str(value["candidate_id"]),
            )
        ),
        "best_by_capacity_candidate_id": best(
            lambda value: (
                float(value["metrics"]["slowest_rank_hbm_required_bytes"]),
                str(value["candidate_id"]),
            )
        ),
        "headline_candidate_id": None,
        "selection_applied": False,
    }


def build_expert_id_replication_cache_sensitivity(
    report: Mapping[str, Any],
    balanced_overlay: Mapping[str, Any],
    cyclic_overlay: Mapping[str, Any],
    trace: RoutingTrace,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    serving: FullDecodeProjectionConfig,
    sensitivity: ReplicationCacheSensitivityConfig,
) -> dict[str, Any]:
    """Evaluate a sealed replica/cache byte grid on one held-out window."""

    balance = _validate_target(report, balanced_overlay, cyclic_overlay, perf, dims, precision)
    ordered_steps = _audit_trace(trace)
    if trace_content_hash(trace) != report.get("trace_content_hash"):
        raise ValueError("replica/cache trace differs from the held-out report")
    audit_expert_id_balancing_report(
        report,
        trace,
        perf,
        balance,
        source_binding=report["source_binding"],
        hardware_binding=report["hardware_binding"],
    )
    first = int(balanced_overlay["first_step_index"])
    last = int(balanced_overlay["last_step_index"])
    if first <= balance.train_step_count - 1 or last - first + 1 != balance.batch_size:
        raise ValueError("replica/cache evaluation must use one disjoint held-out window")
    held_out_steps = ordered_steps[first : last + 1]
    if len(held_out_steps) != balance.batch_size:
        raise ValueError("held-out trace no longer covers the requested window")

    base_projection = build_expert_id_full_decode_projection(
        report,
        balanced_overlay,
        cyclic_overlay,
        perf,
        dims,
        precision,
        serving,
    )
    validate_expert_id_full_decode_projection(
        base_projection,
        report=report,
        balanced_overlay=balanced_overlay,
        cyclic_overlay=cyclic_overlay,
    )
    base_frequency = base_projection["controls"][CONTROL_FREQUENCY]
    base_layout = _expert_id_full_body_layout(balanced_overlay, dims, precision, balance)
    one_expert = _one_expert_plane(balance, precision)
    if one_expert.total_aligned <= 0:
        raise AssertionError("whole expert has no physical storage")
    base_margin = int(base_frequency["capacity"]["slowest_rank_capacity_margin_bytes"])
    headroom_slots = max(0, base_margin // one_expert.total_aligned)

    grid_binding = {
        "replica_budget_bytes_per_rank": list(sensitivity.replica_budget_bytes_per_rank),
        "cache_budget_bytes_per_rank": list(sensitivity.cache_budget_bytes_per_rank),
        "cross_product": True,
        "candidate_count": (
            len(sensitivity.replica_budget_bytes_per_rank) * len(sensitivity.cache_budget_bytes_per_rank)
        ),
        "one_padded_expert_plane": _plane(one_expert),
        "base_frequency_slowest_rank_capacity_margin_bytes": base_margin,
        "whole_expert_headroom_slots_per_rank": headroom_slots,
        "replica_storage_tier": "hbm",
        "cache_storage_tier": "hbm",
        "sram_fraction": None,
        "sram_exclusion_reason": ("whole_expert_cache_is_not_assumed_to_fit_onchip_matrix_sram"),
        "grid_learned_from_held_out": False,
    }
    grid_binding["content_hash"] = canonical_hash(grid_binding)

    candidates: list[dict[str, Any]] = []
    for replica_budget in sensitivity.replica_budget_bytes_per_rank:
        for cache_budget in sensitivity.cache_budget_bytes_per_rank:
            candidate_id = f"{CANDIDATE_PREFIX}_replica_{replica_budget}_cache_{cache_budget}"
            requested_replica_slots = replica_budget // one_expert.total_aligned
            cache_slots = cache_budget // one_expert.total_aligned
            budget = {
                "replica_budget_bytes_per_rank": replica_budget,
                "cache_budget_bytes_per_rank": cache_budget,
                "replica_fraction_of_per_chip_hbm": (replica_budget / balance.hbm_capacity_bytes_per_chip),
                "cache_fraction_of_per_chip_hbm": (cache_budget / balance.hbm_capacity_bytes_per_chip),
                "requested_replica_whole_expert_slots_per_rank": (requested_replica_slots),
                "requested_cache_whole_expert_slots_per_rank": cache_slots,
                "replica_unusable_tail_bytes_per_rank": (
                    replica_budget - requested_replica_slots * one_expert.total_aligned
                ),
                "cache_unusable_tail_bytes_per_rank": (cache_budget - cache_slots * one_expert.total_aligned),
            }
            budget["content_hash"] = canonical_hash(budget)
            if cache_slots > headroom_slots:
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "rejected_cache_reservation_exceeds_hbm_headroom",
                        "budget": budget,
                        "classification": {
                            "publication_rankable": False,
                            "hardware_rankable": False,
                            "selection_eligible": False,
                            "timing_selection_allowed": False,
                            "compiler_validated": False,
                            "emulator_validated": False,
                            "rtl_validated": False,
                            "power_calibrated": False,
                            "headline_eligible": False,
                            "headline_win_claimed": False,
                        },
                    }
                )
                continue
            replica_slots = min(requested_replica_slots, headroom_slots - cache_slots)
            sites, plan = _learn_replica_sites(
                report,
                ordered_steps,
                replica_slots_per_rank=replica_slots,
                perf=perf,
                balance=balance,
            )
            route, cache_ledger = _evaluate_held_out_route(
                report,
                balanced_overlay,
                held_out_steps,
                sites,
                plan,
                cache_slots_per_rank=cache_slots,
                one_expert=one_expert,
                perf=perf,
                balance=balance,
            )
            body_layout = _replicated_body_layout(
                base_layout,
                replica_site_count_by_rank=plan["replica_site_count_by_rank"],
                cache_slots_per_rank=cache_slots,
                one_expert=one_expert,
                balance=balance,
            )
            control = _run_control(
                perf=perf,
                dims=dims,
                precision=precision,
                balance=balance,
                config=serving,
                body_layout=body_layout,
                expert_parallel_mode=EXPERT_ID_PARALLEL,
                route_projection=route,
            )
            control = _apply_cache_endpoint_adjustment(
                control,
                cache_ledger,
                perf=perf,
                balance=balance,
                serving=serving,
            )
            cache_bounds = _cold_vs_ideal_warm_bounds(
                control,
                cache_ledger,
                batch_size=balance.batch_size,
                output_steps=serving.output_sequence_tokens,
            )
            budget.update(
                {
                    "effective_capacity_constrained_replica_slots_per_rank": (replica_slots),
                    "replica_slot_limit_clipped_by_hbm_headroom": (replica_slots < requested_replica_slots),
                    "actual_replica_site_count_by_rank": list(plan["replica_site_count_by_rank"]),
                    "actual_replica_bytes_by_rank": [
                        int(value) * one_expert.total_aligned for value in plan["replica_site_count_by_rank"]
                    ],
                    "reserved_cache_bytes_per_rank": (cache_slots * one_expert.total_aligned),
                }
            )
            budget["content_hash"] = canonical_hash(
                {key: value for key, value in budget.items() if key != "content_hash"}
            )
            candidate = {
                "candidate_id": candidate_id,
                "status": "evaluated",
                "budget": budget,
                "placement_plan": plan,
                "route_projection": route,
                "cache_ledger": cache_ledger,
                "cold_vs_ideal_warm_bounds": cache_bounds,
                "full_decode_control": control,
                "resident_capacity_conservation": {
                    "expected_slowest_rank_added_expert_bytes": (
                        max(int(value) + cache_slots for value in plan["replica_site_count_by_rank"])
                        * one_expert.total_aligned
                    ),
                    "observed_slowest_rank_added_resident_weight_bytes": (
                        int(control["capacity"]["slowest_rank_resident_weight_bytes"])
                        - int(base_frequency["capacity"]["slowest_rank_resident_weight_bytes"])
                    ),
                    "expected_system_added_expert_bytes": (
                        (
                            sum(int(value) for value in plan["replica_site_count_by_rank"])
                            + cache_slots * balance.tensor_parallel_degree
                        )
                        * one_expert.total_aligned
                        * balance.kv_parallel_degree
                    ),
                    "observed_system_added_resident_weight_bytes": (
                        int(control["capacity"]["system_resident_weight_bytes"])
                        - int(base_frequency["capacity"]["system_resident_weight_bytes"])
                    ),
                    "all_expert_replica_and_cache_bytes_charged": True,
                },
                "comparisons": {
                    "vs_no_replication_no_cache_frequency_expert_id": _comparison(
                        control, base_projection["controls"][CONTROL_FREQUENCY]
                    ),
                    "vs_cyclic_expert_id": _comparison(control, base_projection["controls"][CONTROL_CYCLIC]),
                    "vs_trace_exact_tensor": _comparison(control, base_projection["controls"][CONTROL_TENSOR_EXACT]),
                    "vs_native_tensor": _comparison(control, base_projection["controls"][CONTROL_TENSOR_NATIVE]),
                    "headline_win_claimed": False,
                },
                "classification": {
                    "publication_rankable": False,
                    "hardware_rankable": False,
                    "selection_eligible": False,
                    "timing_selection_allowed": False,
                    "compiler_validated": False,
                    "emulator_validated": False,
                    "rtl_validated": False,
                    "power_calibrated": False,
                    "headline_eligible": False,
                    "headline_win_claimed": False,
                },
            }
            candidate["metrics"] = _candidate_metrics(candidate)
            candidate["held_out_observation"] = {
                "tpot_nonregression_vs_no_replication_no_cache_observed": (
                    candidate["comparisons"]["vs_no_replication_no_cache_frequency_expert_id"]["tpot"][
                        "nonregression_observed"
                    ]
                ),
                "held_out_win_guaranteed": False,
                "headline_win_claimed": False,
            }
            candidates.append(candidate)

    evaluated = [value for value in candidates if value["status"] == "evaluated"]
    zero_id = f"{CANDIDATE_PREFIX}_replica_0_cache_0"
    zero = next((value for value in evaluated if value["candidate_id"] == zero_id), None)
    if zero is None:
        raise AssertionError("sealed grid omitted its no-replication/no-cache control")
    parity_fields = (
        "total_time",
        "tpot",
        "tps",
        "avg_bytes_per_batch_step",
        "avg_realized_compute_seconds",
        "avg_memory_seconds",
        "avg_collective_seconds",
    )
    zero_parity = {
        field: math.isclose(
            float(zero["full_decode_control"]["loop"][field]),
            float(base_frequency["loop"][field]),
            rel_tol=1e-12,
            abs_tol=1e-15 if "bytes" not in field else 1e-6,
        )
        for field in parity_fields
    }
    if not all(zero_parity.values()):
        raise AssertionError("zero-budget candidate differs from frequency control")

    pareto = _pareto_summary(candidates)
    body = {
        "schema": SENSITIVITY_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": report["trace_content_hash"],
        "report_content_hash": report["content_hash"],
        "balanced_overlay_content_hash": balanced_overlay["content_hash"],
        "cyclic_overlay_content_hash": cyclic_overlay["content_hash"],
        "base_full_decode_projection_content_hash": base_projection["content_hash"],
        "study_config_content_hash": report["study_config_content_hash"],
        "precision_content_hash": canonical_hash(dict(precision)),
        "dims_content_hash": canonical_hash(dict(dims)),
        "serving_config": asdict(serving),
        "serving_config_content_hash": canonical_hash(asdict(serving)),
        "sensitivity_config": asdict(sensitivity),
        "sensitivity_config_content_hash": canonical_hash(asdict(sensitivity)),
        "window_index": int(balanced_overlay["window_index"]),
        "first_step_index": first,
        "last_step_index": last,
        "split": {
            "training_step_count": balance.train_step_count,
            "training_step_indices_content_hash": report["split"]["training"]["step_indices_content_hash"],
            "held_out_step_indices_content_hash": canonical_hash({"indices": list(range(first, last + 1))}),
            "train_eval_overlap_count": 0,
            "placement_or_budget_learned_from_held_out": False,
            "held_out_used_only_for_fixed_policy_runtime_evaluation": True,
        },
        "grid_binding": grid_binding,
        "base_full_decode_projection": base_projection,
        "retained_controls": list(CONTROL_NAMES),
        "no_replication_no_cache_parity": {
            "candidate_id": zero_id,
            "reference_control": CONTROL_FREQUENCY,
            "checks": zero_parity,
            "all_checks_passed": True,
        },
        "candidates": candidates,
        "pareto_and_best": pareto,
        "conservation": {
            "logical_route_assignments_per_candidate_batch_step": (NUM_LAYERS * balance.batch_size * TOP_K),
            "physical_whole_expert_assignments_across_kvp_per_candidate_batch_step": (
                NUM_LAYERS * balance.batch_size * TOP_K * balance.kv_parallel_degree
            ),
            "kvp_routes_are_identical_not_new_training_samples": True,
            "source_hidden_dispatch_bytes": 0,
            "tp_expert_output_allreduce_count_per_batch_step": (NUM_LAYERS * int(balance.tensor_parallel_degree > 1)),
            "mapping": EXPERT_ID_MAPPING,
        },
        "evidence_scope": {
            "route_evidence": "measured_held_out_trace_exact",
            "replica_learning": "measured_training_trace_only",
            "decode_timing": "analytic_full_loop_projection",
            "cache_fill_link_time": ("existing_per_chip_endpoint_bandwidth_lower_bound"),
            "cache_state_projection": ("cold_reset_charged_plus_ideal_zero_fill_warm_lower_bound"),
            "persistent_across_decode_steps_lru_modelled": False,
            "ideal_warm_bound_included_in_pareto": False,
            "all_to_all_modelled": False,
            "link_contention_or_overlap_modelled": False,
            "calibrated_energy_modelled": False,
        },
        "classification": {
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
            "compiler_validated": False,
            "emulator_validated": False,
            "rtl_validated": False,
            "power_calibrated": False,
            "headline_eligible": False,
            "headline_win_claimed": False,
            "blockers": [
                "held_out_window_is_a_stationary_post_hoc_route_proxy",
                "replica_and_cache_timing_is_an_unvalidated_analytic_projection",
                "cache_fetch_uses_an_endpoint_lower_bound_without_contention_or_overlap",
                "persistent_lru_requires_a_per_decode_step_layer_exact_route_hook",
                "matched_compiler_emulator_rtl_power_and_accuracy_receipts_missing",
            ],
        },
    }
    return _hashed(body)


def validate_expert_id_replication_cache_sensitivity(
    artifact: Mapping[str, Any],
) -> None:
    """Reject changed ledgers, lost routes, or promoted analytic candidates."""

    if not isinstance(artifact, Mapping):
        raise ValueError("replica/cache sensitivity must be an object")
    body = dict(artifact)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("replica/cache sensitivity content hash mismatch")
    if artifact.get("schema") != SENSITIVITY_SCHEMA:
        raise ValueError("unsupported replica/cache sensitivity schema")
    if artifact.get("model_id") != MODEL_ID or artifact.get("model_revision") != MODEL_REVISION:
        raise ValueError("replica/cache sensitivity model identity differs")
    classification = artifact.get("classification")
    false_fields = (
        "publication_rankable",
        "hardware_rankable",
        "selection_eligible",
        "timing_selection_allowed",
        "compiler_validated",
        "emulator_validated",
        "rtl_validated",
        "power_calibrated",
        "headline_eligible",
        "headline_win_claimed",
    )
    if not isinstance(classification, Mapping) or any(classification.get(field) is not False for field in false_fields):
        raise ValueError("replica/cache sensitivity was promoted")
    split = artifact.get("split")
    if (
        not isinstance(split, Mapping)
        or int(split.get("train_eval_overlap_count", -1)) != 0
        or split.get("placement_or_budget_learned_from_held_out") is not False
        or split.get("held_out_used_only_for_fixed_policy_runtime_evaluation") is not True
    ):
        raise ValueError("replica/cache training and held-out split is unsafe")
    grid = artifact.get("grid_binding")
    if not isinstance(grid, Mapping):
        raise ValueError("replica/cache grid binding is missing")
    grid_body = dict(grid)
    grid_hash = grid_body.pop("content_hash", None)
    if grid_hash != canonical_hash(grid_body):
        raise ValueError("replica/cache grid binding changed")
    base = artifact.get("base_full_decode_projection")
    if not isinstance(base, Mapping):
        raise ValueError("base full-decode controls are missing")
    validate_expert_id_full_decode_projection(base)
    if (
        base.get("content_hash") != artifact.get("base_full_decode_projection_content_hash")
        or tuple(artifact.get("retained_controls", ())) != CONTROL_NAMES
    ):
        raise ValueError("base full-decode control binding differs")

    conservation = artifact.get("conservation")
    if (
        not isinstance(conservation, Mapping)
        or conservation.get("mapping") != EXPERT_ID_MAPPING
        or int(conservation.get("source_hidden_dispatch_bytes", -1)) != 0
    ):
        raise ValueError("replica/cache topology mapping differs")
    expected_logical = int(conservation["logical_route_assignments_per_candidate_batch_step"])
    expected_physical = int(conservation["physical_whole_expert_assignments_across_kvp_per_candidate_batch_step"])
    expected_collectives = int(conservation["tp_expert_output_allreduce_count_per_batch_step"])
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != int(grid["candidate_count"]):
        raise ValueError("replica/cache candidate grid is incomplete")
    ids = [str(value.get("candidate_id", "")) for value in candidates]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("replica/cache candidate IDs are invalid")
    evaluated_ids: set[str] = set()
    feasible_ids: set[str] = set()
    for candidate in candidates:
        status = candidate.get("status")
        budget = candidate.get("budget")
        if not isinstance(budget, Mapping):
            raise ValueError("replica/cache candidate budget is missing")
        budget_body = dict(budget)
        budget_hash = budget_body.pop("content_hash", None)
        if budget_hash != canonical_hash(budget_body):
            raise ValueError("replica/cache candidate budget changed")
        candidate_classification = candidate.get("classification")
        if not isinstance(candidate_classification, Mapping) or any(
            candidate_classification.get(field) is not False for field in false_fields
        ):
            raise ValueError("replica/cache candidate was promoted")
        if status == "rejected_cache_reservation_exceeds_hbm_headroom":
            if set(candidate) - {
                "candidate_id",
                "status",
                "budget",
                "classification",
            }:
                raise ValueError("rejected cache point carries fabricated metrics")
            continue
        if status != "evaluated":
            raise ValueError("replica/cache candidate has an unknown status")
        candidate_id = str(candidate["candidate_id"])
        evaluated_ids.add(candidate_id)
        for nested_name in ("budget", "placement_plan", "route_projection", "cache_ledger"):
            nested = candidate.get(nested_name)
            if not isinstance(nested, Mapping):
                raise ValueError(f"candidate {nested_name} is missing")
            nested_body = dict(nested)
            nested_hash = nested_body.pop("content_hash", None)
            if nested_hash != canonical_hash(nested_body):
                raise ValueError(f"candidate {nested_name} changed")
        plan = candidate["placement_plan"]
        if (
            plan.get("held_out_steps_used") is not False
            or plan.get("all_layers_training_nonregression") is not True
            or any(
                row.get("training_nonregression") is not True or row.get("held_out_steps_used") is not False
                for row in plan.get("layer_training_validation", ())
            )
        ):
            raise ValueError("replica placement leaks or regresses training")
        route = candidate["route_projection"]
        if (
            route.get("schema") != ROUTE_PROJECTION_SCHEMA
            or route.get("mapping") != EXPERT_ID_MAPPING
            or route.get("expert_parallel_mode") != EXPERT_ID_PARALLEL
            or route.get("global_layer_collapse_allowed") is not False
            or int(route["totals"].get("logical_route_assignments", -1)) != expected_logical
            or int(route["totals"].get("physical_whole_expert_assignments_across_kvp", -1)) != expected_physical
            or int(route["totals"].get("source_hidden_dispatch_bytes", -1)) != 0
            or int(route["totals"].get("expert_output_collective_count", -1)) != expected_collectives
            or route["totals"].get("resident_capacity_uses_rank_aggregate_not_layer_max_sum") is not True
        ):
            raise ValueError("replica/cache route projection does not conserve")
        layers = route.get("layers")
        if (
            not isinstance(layers, list)
            or len(layers) != NUM_LAYERS
            or [int(value.get("layer", -1)) for value in layers] != list(range(NUM_LAYERS))
        ):
            raise ValueError("replica/cache route projection lacks 48 layers")
        tp = int(route["tensor_parallel_degree"])
        kvp = int(route["kv_parallel_degree"])
        per_layer_logical = expected_logical // NUM_LAYERS
        for layer in layers:
            sites = layer.get("permanent_resident_ranks_by_expert_id")
            owners = layer.get("primary_expert_owner_by_id")
            if (
                not isinstance(sites, list)
                or len(sites) != NUM_EXPERTS
                or not isinstance(owners, list)
                or len(owners) != NUM_EXPERTS
                or any(int(owners[expert]) not in sites[expert] for expert in range(NUM_EXPERTS))
            ):
                raise ValueError("one or more experts lost their primary site")
            assignments = [int(value) for value in layer["assignment_count_by_rank"]]
            physical = [int(value) for value in layer["physical_assignment_count_by_rank_across_kvp"]]
            if (
                len(assignments) != tp
                or sum(assignments) != per_layer_logical
                or physical != [value * kvp for value in assignments]
                or int(layer.get("logical_route_assignments", -1)) != per_layer_logical
                or int(layer.get("physical_whole_expert_assignments_across_kvp", -1)) != per_layer_logical * kvp
                or int(layer.get("source_hidden_dispatch_bytes", -1)) != 0
                or int(layer.get("expert_output_collective_count", -1)) != int(tp > 1)
            ):
                raise ValueError("replica/cache layer route work does not conserve")
            for rank, counts in enumerate(layer["expert_token_counts_by_rank"]):
                if sum(int(value) for value in counts.values()) != assignments[rank]:
                    raise ValueError("rank-local expert counts do not conserve")
        cache = candidate["cache_ledger"]
        logical_fetch = int(cache["logical_fetch_plane_per_tp_group"]["total_aligned"])
        physical_fetch = int(cache["physical_fetch_plane_across_kvp"]["total_aligned"])
        read = int(cache["system_hbm_fill_read_plane_across_kvp"]["total_aligned"])
        write = int(cache["system_hbm_fill_write_plane_across_kvp"]["total_aligned"])
        if (
            int(cache["fills"]) != int(cache["misses"])
            or physical_fetch != logical_fetch * kvp
            or read != physical_fetch
            or write != physical_fetch
            or int(cache.get("source_hidden_dispatch_bytes", -1)) != 0
            or cache.get("all_to_all_collective_modelled") is not False
            or cache.get("energy_modelled") is not False
            or any(int(value["peak_entries"]) > int(cache["capacity_slots_per_rank"]) for value in cache["per_rank"])
            or any(
                int(value["fills"]) != int(value["evictions"]) + int(value["final_entries"])
                for value in cache["per_rank"]
            )
        ):
            raise ValueError("expert-weight cache ledger does not conserve")
        control = candidate.get("full_decode_control")
        if (
            not isinstance(control, Mapping)
            or control.get("route_hook_applied") is not True
            or control.get("route_projection_content_hash") != route.get("content_hash")
            or control["loop"]["layer_exact_moe_route_projection"]["content_hash"] != route.get("content_hash")
            or control["loop"]["collective_breakdown_per_batch_step"].get("expert_routing_bytes") != 0
            or control["cache_endpoint_adjustment"].get("other_decode_components_changed") is not False
            or not math.isclose(
                sum(float(value) for value in control["loop"]["traffic_breakdown_per_batch_step"].values()),
                float(control["loop"]["avg_bytes_per_batch_step"]),
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("full decode loop did not bind the exact cache route")
        expected_cache_bounds = _cold_vs_ideal_warm_bounds(
            control,
            cache,
            batch_size=int(route["batch_size"]),
            output_steps=int(artifact["serving_config"]["output_sequence_tokens"]),
        )
        if candidate.get("cold_vs_ideal_warm_bounds") != expected_cache_bounds:
            raise ValueError("cold/ideal-warm cache bounds changed")
        resident = candidate.get("resident_capacity_conservation")
        if (
            not isinstance(resident, Mapping)
            or resident.get("all_expert_replica_and_cache_bytes_charged") is not True
            or int(resident["expected_slowest_rank_added_expert_bytes"])
            != int(resident["observed_slowest_rank_added_resident_weight_bytes"])
            or int(resident["expected_system_added_expert_bytes"])
            != int(resident["observed_system_added_resident_weight_bytes"])
        ):
            raise ValueError("replica/cache resident capacity does not conserve")
        for sample in control["component_and_hbm_step_proof"]["samples"]:
            if not math.isclose(
                float(sample["component_cycle_sum"]),
                math.fsum(float(value) for value in sample["component_cycles"].values()),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError("full decode component cycles do not sum")
        if candidate.get("metrics") != _candidate_metrics(candidate):
            raise ValueError("replica/cache candidate metrics changed")
        expected_comparisons = {
            "vs_no_replication_no_cache_frequency_expert_id": _comparison(control, base["controls"][CONTROL_FREQUENCY]),
            "vs_cyclic_expert_id": _comparison(control, base["controls"][CONTROL_CYCLIC]),
            "vs_trace_exact_tensor": _comparison(control, base["controls"][CONTROL_TENSOR_EXACT]),
            "vs_native_tensor": _comparison(control, base["controls"][CONTROL_TENSOR_NATIVE]),
            "headline_win_claimed": False,
        }
        if candidate.get("comparisons") != expected_comparisons:
            raise ValueError("replica/cache candidate comparisons changed")
        expected_held_out = {
            "tpot_nonregression_vs_no_replication_no_cache_observed": (
                expected_comparisons["vs_no_replication_no_cache_frequency_expert_id"]["tpot"]["nonregression_observed"]
            ),
            "held_out_win_guaranteed": False,
            "headline_win_claimed": False,
        }
        if candidate.get("held_out_observation") != expected_held_out:
            raise ValueError("replica/cache held-out observation changed")
        if candidate["metrics"].get("fits_hbm") is True and candidate["metrics"].get("fits_runtime") is True:
            feasible_ids.add(candidate_id)

    parity = artifact.get("no_replication_no_cache_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("all_checks_passed") is not True
        or not all(parity.get("checks", {}).values())
        or parity.get("candidate_id") not in evaluated_ids
    ):
        raise ValueError("zero-budget control parity is missing")
    pareto = artifact.get("pareto_and_best")
    expected_pareto = _pareto_summary(candidates)
    if (
        not isinstance(pareto, Mapping)
        or dict(pareto) != expected_pareto
        or pareto.get("selection_applied") is not False
        or pareto.get("headline_candidate_id") is not None
        or not set(pareto.get("pareto_candidate_ids", ())).issubset(feasible_ids)
        or any(
            value is not None and value not in feasible_ids
            for value in (
                pareto.get("best_by_latency_candidate_id"),
                pareto.get("best_by_traffic_candidate_id"),
                pareto.get("best_by_capacity_candidate_id"),
            )
        )
    ):
        raise ValueError("replica/cache Pareto summary is unsafe")


def audit_expert_id_replication_cache_sensitivity(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    balanced_overlay: Mapping[str, Any],
    cyclic_overlay: Mapping[str, Any],
    trace: RoutingTrace,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    serving: FullDecodeProjectionConfig,
    sensitivity: ReplicationCacheSensitivityConfig,
) -> Mapping[str, Any]:
    """Recompute the complete receipt from its bound inputs."""

    validate_expert_id_replication_cache_sensitivity(artifact)
    rebuilt = build_expert_id_replication_cache_sensitivity(
        report,
        balanced_overlay,
        cyclic_overlay,
        trace,
        perf,
        dims,
        precision,
        serving,
        sensitivity,
    )
    if dict(artifact) != rebuilt:
        raise ValueError("replica/cache sensitivity differs from exact recomputation")
    return artifact


def _encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_install(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"content-addressed artifact differs: {path}")
        os.replace(temporary, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def materialize_expert_id_replication_cache_sensitivity(
    artifact: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Write one immutable, content-addressed sensitivity receipt."""

    validate_expert_id_replication_cache_sensitivity(artifact)
    output = Path(output_dir)
    content_hash = str(artifact["content_hash"])
    path = output / (f"expert_id_replication_cache_window_{int(artifact['window_index'])}.{content_hash[:16]}.json")
    payload = _encode(artifact)
    _atomic_install(path, payload)
    return {
        "schema": SENSITIVITY_RECEIPT_SCHEMA,
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "content_hash": content_hash,
        "window_index": int(artifact["window_index"]),
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
        "timing_selection_allowed": False,
        "compiler_validated": False,
        "emulator_validated": False,
        "rtl_validated": False,
        "power_calibrated": False,
        "headline_eligible": False,
        "headline_win_claimed": False,
    }


__all__ = [
    "CACHE_POLICY",
    "PLACEMENT_POLICY",
    "RUNTIME_POLICY",
    "SENSITIVITY_RECEIPT_SCHEMA",
    "SENSITIVITY_SCHEMA",
    "ReplicationCacheSensitivityConfig",
    "audit_expert_id_replication_cache_sensitivity",
    "build_expert_id_replication_cache_sensitivity",
    "materialize_expert_id_replication_cache_sensitivity",
    "validate_expert_id_replication_cache_sensitivity",
]
