"""Explicit system metrics for aggregated and disaggregated serving results."""

from __future__ import annotations

import math
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


TTFT_SEMANTICS = frozenset(
    {
        "request_visible_arrival_to_first_token",
        "scheduler_admitted_first_schedule_to_first_token",
        "inferred_serial_staircase",
        "static_fixed_batch_barrier",
    }
)
GOODPUT_TTFT_SEMANTICS = "request_visible_arrival_to_first_token"


def aggregated_system_metrics(
    *,
    batch_size: int,
    full_batch_e2e_s: float,
    full_batch_energy_j: float,
    input_tokens_per_request: int | None = None,
    output_tokens_per_request: int | None = None,
    total_tokens_per_request: int | None = None,
    request_visible_ttft_s: float | None = None,
    scheduler_admitted_ttft_exact_or_proxy_s: float | None = None,
    scheduler_admitted_ttft_source: str | None = None,
    batch_first_token_barrier_s: float | None = None,
) -> dict[str, Any]:
    """Summarize a measured fixed-batch aggregated serving run."""

    _validate_positive(batch_size=batch_size, duration=full_batch_e2e_s, energy=full_batch_energy_j)
    throughput = batch_size / full_batch_e2e_s
    energy_per_request = full_batch_energy_j / batch_size
    output_tokens = _batch_tokens(batch_size, output_tokens_per_request)
    input_tokens = _batch_tokens(batch_size, input_tokens_per_request)
    total_tokens = (
        _batch_tokens(batch_size, total_tokens_per_request)
        if total_tokens_per_request is not None
        else _optional_sum(input_tokens, output_tokens)
    )
    output_tps = output_tokens / full_batch_e2e_s if output_tokens is not None else None
    output_tokens_per_j = output_tokens / full_batch_energy_j if output_tokens is not None else None
    if scheduler_admitted_ttft_exact_or_proxy_s is not None and not scheduler_admitted_ttft_source:
        raise ValueError("scheduler-admitted TTFT requires an exact-or-proxy source")
    return {
        "metric_schema": "fixed-batch-serving-metrics-v3",
        "fidelity": "measured_aggregated_full_batch",
        "full_batch_e2e_s": full_batch_e2e_s,
        "service_interval_s": full_batch_e2e_s,
        "throughput_requests_per_s": throughput,
        "output_tokens_per_request": output_tokens_per_request,
        "global_output_tokens": output_tokens,
        "output_tokens_per_s": output_tps,
        "output_tokens_per_j": output_tokens_per_j,
        "energy_per_output_token_j": (1.0 / output_tokens_per_j if output_tokens_per_j else None),
        "input_tokens_per_j": (input_tokens / full_batch_energy_j if input_tokens is not None else None),
        "total_tokens_per_j": (total_tokens / full_batch_energy_j if total_tokens is not None else None),
        "system_energy_j": full_batch_energy_j,
        "energy_per_request_j": energy_per_request,
        "average_system_power_w": full_batch_energy_j / full_batch_e2e_s,
        "throughput_per_watt_requests_per_j": batch_size / full_batch_energy_j,
        "energy_per_token_j": (full_batch_energy_j / total_tokens if total_tokens else None),
        "legacy_metric_alias": True,
        "legacy_metric_semantics": {
            "throughput_requests_per_s": "auxiliary_requests_per_second",
            "throughput_per_watt_requests_per_j": "auxiliary_requests_per_joule",
            "energy_per_token_j": "energy_per_input_plus_output_token",
        },
        "a100_request_visible_ttft_s": request_visible_ttft_s,
        "a100_scheduler_admitted_ttft_exact_or_proxy_s": (scheduler_admitted_ttft_exact_or_proxy_s),
        "a100_scheduler_admitted_ttft_source": scheduler_admitted_ttft_source,
        "a100_batch_first_token_barrier_s": batch_first_token_barrier_s,
        "a100_throughput_equivalent_prefill_interval_s": (
            batch_first_token_barrier_s / batch_size if batch_first_token_barrier_s is not None else None
        ),
        "a100_throughput_equivalent_prefill_interval_semantics": (
            "batch_first_token_barrier_divided_by_batch_not_ttft"
        ),
    }


def disaggregated_pipeline_metrics(
    *,
    batch_size: int,
    prefill_interval_s: float,
    kv_handoff_interval_s: float,
    decode_interval_s: float,
    prefill_energy_j: float,
    kv_handoff_energy_j: float,
    decode_energy_j: float,
    prefill_idle_power_w: float = 0.0,
    kv_handoff_idle_power_w: float = 0.0,
    decode_idle_power_w: float = 0.0,
    e2e_latency_s: float | None = None,
    input_tokens_per_request: int | None = None,
    output_tokens_per_request: int | None = None,
    total_tokens_per_request: int | None = None,
    request_ttft_s: float | None = None,
    request_ttft_semantics: str | None = None,
    request_tpot_s: float | None = None,
    ttft_slo_s: float | None = None,
    tpot_slo_s: float | None = None,
    first_decode_step_s: float | None = None,
) -> dict[str, Any]:
    """Evaluate the deterministic three-stage pipeline envelope.

    Stages are serialized within one batch and may overlap only across
    different batches. This is not a queueing or continuous-batching model.
    """

    _validate_positive(
        batch_size=batch_size,
        duration=max(prefill_interval_s, kv_handoff_interval_s, decode_interval_s),
        energy=prefill_energy_j + kv_handoff_energy_j + decode_energy_j,
    )
    if min(prefill_interval_s, kv_handoff_interval_s, decode_interval_s) < 0:
        raise ValueError("pipeline stage intervals must be nonnegative")
    if min(prefill_energy_j, kv_handoff_energy_j, decode_energy_j) < 0:
        raise ValueError("pipeline stage energies must be nonnegative")
    if min(prefill_idle_power_w, kv_handoff_idle_power_w, decode_idle_power_w) < 0:
        raise ValueError("pipeline stage idle powers must be nonnegative")
    if request_ttft_s is not None:
        if request_ttft_semantics not in TTFT_SEMANTICS:
            raise ValueError("request TTFT requires an explicit supported semantic")
    elif request_ttft_semantics is not None:
        raise ValueError("request_ttft_semantics was supplied without request_ttft_s")
    if first_decode_step_s is not None and first_decode_step_s < 0:
        raise ValueError("first decode step must be nonnegative")

    service_interval = max(
        prefill_interval_s,
        kv_handoff_interval_s,
        decode_interval_s,
    )
    active_stage_energy = math.fsum((prefill_energy_j, kv_handoff_energy_j, decode_energy_j))
    stage_idle_durations = {
        "prefill": service_interval - prefill_interval_s,
        "kv_handoff": service_interval - kv_handoff_interval_s,
        "decode": service_interval - decode_interval_s,
    }
    stage_idle_energies = {
        "prefill": stage_idle_durations["prefill"] * prefill_idle_power_w,
        "kv_handoff": stage_idle_durations["kv_handoff"] * kv_handoff_idle_power_w,
        "decode": stage_idle_durations["decode"] * decode_idle_power_w,
    }
    cross_stage_idle_energy = math.fsum(stage_idle_energies.values())
    system_energy = active_stage_energy + cross_stage_idle_energy
    throughput = batch_size / service_interval
    energy_per_request = system_energy / batch_size
    output_tokens = _batch_tokens(batch_size, output_tokens_per_request)
    input_tokens = _batch_tokens(batch_size, input_tokens_per_request)
    total_tokens = (
        _batch_tokens(batch_size, total_tokens_per_request)
        if total_tokens_per_request is not None
        else _optional_sum(input_tokens, output_tokens)
    )
    output_tps = output_tokens / service_interval if output_tokens is not None else None
    output_tokens_per_j = output_tokens / system_energy if output_tokens is not None else None
    slo_complete = ttft_slo_s is not None and tpot_slo_s is not None
    slo_observed = request_ttft_s is not None and request_tpot_s is not None
    if slo_complete and slo_observed and request_ttft_semantics != GOODPUT_TTFT_SEMANTICS:
        raise ValueError("goodput TTFT SLO accepts only request-visible arrival-to-first-token latency")
    slo_pass = (
        bool(request_ttft_s <= ttft_slo_s and request_tpot_s <= tpot_slo_s) if slo_complete and slo_observed else None
    )
    return {
        "metric_schema": "fixed-batch-serving-metrics-v4",
        "fidelity": "analytical_fixed_batch_pipeline_envelope",
        "prefill_interval_s": prefill_interval_s,
        "plena_batch_admitted_prefill_ttft_s": prefill_interval_s,
        "plena_batch_first_token_barrier_s": prefill_interval_s,
        "plena_throughput_equivalent_prefill_interval_s": (prefill_interval_s / batch_size),
        "plena_throughput_equivalent_prefill_interval_semantics": (
            "fixed_batch_prefill_makespan_divided_by_batch_not_ttft"
        ),
        "plena_disaggregated_batch_admitted_ttft_s": (
            prefill_interval_s + kv_handoff_interval_s + first_decode_step_s
            if first_decode_step_s is not None
            else None
        ),
        "kv_handoff_interval_s": kv_handoff_interval_s,
        "decode_interval_s": decode_interval_s,
        "service_interval_s": service_interval,
        "bottleneck_stage": max(
            (
                ("prefill", prefill_interval_s),
                ("kv_handoff", kv_handoff_interval_s),
                ("decode", decode_interval_s),
            ),
            key=lambda item: item[1],
        )[0],
        "projected_pipeline_throughput_requests_per_s": throughput,
        "output_tokens_per_request": output_tokens_per_request,
        "global_output_tokens": output_tokens,
        "projected_pipeline_output_tokens_per_s": output_tps,
        "projected_output_tokens_per_j": output_tokens_per_j,
        "energy_per_output_token_j": (1.0 / output_tokens_per_j if output_tokens_per_j else None),
        "input_tokens_per_j": (input_tokens / system_energy if input_tokens is not None else None),
        "total_tokens_per_j": (total_tokens / system_energy if total_tokens is not None else None),
        "e2e_latency_s": e2e_latency_s,
        "system_energy_j": system_energy,
        "steady_state_system_energy_per_batch_j": system_energy,
        "active_stage_energy_j": active_stage_energy,
        "cross_stage_idle_energy_j": cross_stage_idle_energy,
        "prefill_cross_stage_idle_energy_j": stage_idle_energies["prefill"],
        "kv_handoff_cross_stage_idle_energy_j": stage_idle_energies["kv_handoff"],
        "decode_cross_stage_idle_energy_j": stage_idle_energies["decode"],
        "prefill_cross_stage_idle_duration_s": stage_idle_durations["prefill"],
        "kv_handoff_cross_stage_idle_duration_s": stage_idle_durations["kv_handoff"],
        "decode_cross_stage_idle_duration_s": stage_idle_durations["decode"],
        "prefill_idle_power_w": prefill_idle_power_w,
        "kv_handoff_idle_power_w": kv_handoff_idle_power_w,
        "decode_idle_power_w": decode_idle_power_w,
        "cross_stage_idle_energy_accounted": True,
        "energy_per_request_j": energy_per_request,
        "steady_state_average_system_power_w": system_energy / service_interval,
        "average_pipeline_power_w": system_energy / service_interval,
        "single_batch_average_system_power_w": (
            active_stage_energy / e2e_latency_s
            if e2e_latency_s is not None and e2e_latency_s > 0
            else None
        ),
        "throughput_per_watt_requests_per_j": batch_size / system_energy,
        "energy_per_token_j": system_energy / total_tokens if total_tokens else None,
        "legacy_metric_alias": True,
        "legacy_metric_semantics": {
            "projected_pipeline_throughput_requests_per_s": "auxiliary_requests_per_second",
            "throughput_per_watt_requests_per_j": "auxiliary_requests_per_joule",
            "average_pipeline_power_w": "steady_state_average_system_power_w",
            "energy_per_token_j": "energy_per_input_plus_output_token",
        },
        "goodput_requests_per_s": throughput if slo_pass is True else (0.0 if slo_pass is False else None),
        "goodput_status": (
            "defined_and_satisfied"
            if slo_pass is True
            else "defined_and_violated"
            if slo_pass is False
            else "undefined_without_explicit_ttft_and_tpot_slo"
        ),
        "ttft_slo_s": ttft_slo_s,
        "tpot_slo_s": tpot_slo_s,
        "request_ttft_s": request_ttft_s,
        "request_ttft_semantics": request_ttft_semantics,
        "no_queueing_or_continuous_batching_simulation": True,
        "imported_kv_decode_proxy": True,
        "real_plena_to_a100_kv_import": False,
    }


def select_max_output_tokens_per_j(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
) -> Mapping[str, Any] | None:
    """Select the most output-token-efficient feasible system."""

    if aggregated_e2e_s <= 0 or max_e2e_ratio <= 0:
        raise ValueError("latency reference and ratio must be positive")
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate.get("accuracy", -math.inf)) > minimum_accuracy
        and bool(candidate.get("area_constraint_satisfied", False))
        and bool(candidate.get("hbm_constraint_satisfied", False))
        and float(candidate.get("e2e_latency_s", math.inf)) <= max_e2e_ratio * aggregated_e2e_s
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda candidate: (
            float(candidate["projected_output_tokens_per_j"]),
            float(candidate.get("projected_pipeline_output_tokens_per_s", 0.0)),
            -float(candidate.get("energy_per_request_j", math.inf)),
            -float(candidate.get("aggregate_area_mm2", math.inf)),
        ),
    )


def select_max_throughput_per_watt(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
) -> Mapping[str, Any] | None:
    """Compatibility alias for :func:`select_max_output_tokens_per_j`."""

    return select_max_output_tokens_per_j(
        (_with_output_metric_aliases(candidate) for candidate in candidates),
        aggregated_e2e_s=aggregated_e2e_s,
        max_e2e_ratio=max_e2e_ratio,
        minimum_accuracy=minimum_accuracy,
    )


def system_output_tps_efficiency_pareto(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Return the feasible output-TPS/output-tokens-per-joule Pareto.

    When ``prefill_pareto_trial_ids`` is supplied, candidates that do not
    originate from that prefill latency/energy Pareto are deliberately
    excluded. This makes the accepted search limitation explicit.
    """

    if aggregated_e2e_s <= 0 or max_e2e_ratio <= 0:
        raise ValueError("latency reference and ratio must be positive")
    feasible: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        trial = int(candidate.get("prefill_trial", candidate.get("trial", -1)))
        if prefill_pareto_trial_ids is not None and trial not in prefill_pareto_trial_ids:
            continue
        accuracy = float(candidate.get("accuracy", candidate.get("accuracy_score", -math.inf)))
        if accuracy <= minimum_accuracy:
            continue
        if not _constraint_satisfied(candidate, "area"):
            continue
        if not _constraint_satisfied(candidate, "hbm"):
            continue
        if float(candidate.get("e2e_latency_s", math.inf)) > max_e2e_ratio * aggregated_e2e_s:
            continue
        output_tps = float(candidate.get("projected_pipeline_output_tokens_per_s", -math.inf))
        efficiency = float(candidate.get("projected_output_tokens_per_j", -math.inf))
        if not math.isfinite(output_tps) or not math.isfinite(efficiency):
            continue
        candidate["system_selector_latency_ratio"] = max_e2e_ratio
        candidate["system_selector_accuracy_threshold"] = minimum_accuracy
        candidate["system_selector_prefill_pareto_only"] = prefill_pareto_trial_ids is not None
        feasible.append(candidate)

    # Collapse exact objective duplicates before the Pareto scan.
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for candidate in feasible:
        key = (
            float(candidate["projected_pipeline_output_tokens_per_s"]),
            float(candidate["projected_output_tokens_per_j"]),
        )
        incumbent = unique.get(key)
        if incumbent is None or _system_tie_key(candidate) < _system_tie_key(incumbent):
            unique[key] = candidate

    # Sorting throughput high-to-low reduces the two-objective Pareto test to
    # a linear scan over the best efficiency seen at a higher throughput.
    ordered = sorted(
        unique.values(),
        key=lambda candidate: (
            -float(candidate["projected_pipeline_output_tokens_per_s"]),
            -float(candidate["projected_output_tokens_per_j"]),
            _system_tie_key(candidate),
        ),
    )
    front: list[dict[str, Any]] = []
    best_efficiency = -math.inf
    index = 0
    while index < len(ordered):
        output_tps = float(ordered[index]["projected_pipeline_output_tokens_per_s"])
        group_end = index + 1
        while (
            group_end < len(ordered)
            and float(ordered[group_end]["projected_pipeline_output_tokens_per_s"]) == output_tps
        ):
            group_end += 1
        candidate = ordered[index]
        efficiency = float(candidate["projected_output_tokens_per_j"])
        if efficiency > best_efficiency:
            front.append(candidate)
        best_efficiency = max(best_efficiency, efficiency)
        index = group_end
    return front


def system_throughput_efficiency_pareto(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility alias for the output-token Pareto implementation."""

    return system_output_tps_efficiency_pareto(
        (_with_output_metric_aliases(candidate) for candidate in candidates),
        aggregated_e2e_s=aggregated_e2e_s,
        max_e2e_ratio=max_e2e_ratio,
        minimum_accuracy=minimum_accuracy,
        prefill_pareto_trial_ids=prefill_pareto_trial_ids,
    )


def select_system_output_endpoints(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, Any]:
    """Build the formal two-objective system front and its named endpoints."""

    front = system_output_tps_efficiency_pareto(
        candidates,
        aggregated_e2e_s=aggregated_e2e_s,
        max_e2e_ratio=max_e2e_ratio,
        minimum_accuracy=minimum_accuracy,
        prefill_pareto_trial_ids=prefill_pareto_trial_ids,
    )
    maximum_output_tps = (
        min(
            front,
            key=lambda candidate: (
                -float(candidate["projected_pipeline_output_tokens_per_s"]),
                -float(candidate["projected_output_tokens_per_j"]),
                _system_tie_key(candidate),
            ),
        )
        if front
        else None
    )
    maximum_output_tokens_per_j = (
        max(
            front,
            key=lambda candidate: (
                float(candidate["projected_output_tokens_per_j"]),
                float(candidate["projected_pipeline_output_tokens_per_s"]),
                -float(candidate.get("energy_per_request_j", math.inf)),
                -float(candidate.get("aggregate_area_mm2", math.inf)),
            ),
        )
        if front
        else None
    )
    return {
        "schema": "output-tps-energy-efficiency-v2",
        "objective_directions": {
            "projected_pipeline_output_tokens_per_s": "maximize",
            "projected_output_tokens_per_j": "maximize",
        },
        "max_e2e_ratio": max_e2e_ratio,
        "minimum_accuracy": minimum_accuracy,
        "prefill_pareto_only": prefill_pareto_trial_ids is not None,
        "pareto": front,
        "maximum_output_tps": maximum_output_tps,
        "maximum_output_tokens_per_j": maximum_output_tokens_per_j,
        "maximum_throughput": maximum_output_tps,
        "maximum_throughput_per_watt": maximum_output_tokens_per_j,
        "legacy_endpoint_aliases": {
            "maximum_throughput": "maximum_output_tps",
            "maximum_throughput_per_watt": "maximum_output_tokens_per_j",
        },
    }


def select_system_pareto_endpoints(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, Any]:
    """Compatibility alias for :func:`select_system_output_endpoints`."""

    return select_system_output_endpoints(
        (_with_output_metric_aliases(candidate) for candidate in candidates),
        aggregated_e2e_s=aggregated_e2e_s,
        max_e2e_ratio=max_e2e_ratio,
        minimum_accuracy=minimum_accuracy,
        prefill_pareto_trial_ids=prefill_pareto_trial_ids,
    )


def write_system_selector_artifacts(
    output_dir: Path,
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    prefill_pareto_trial_ids: set[int] | frozenset[int],
    minimum_accuracy: float = 0.9,
    e2e_ratios: tuple[float, ...] = (1.0, 1.25, 1.5),
    ungated_candidates: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the formal system Pareto, endpoints, and latency sensitivity.

    Ungated candidates, when supplied, are ranked in an independent shadow
    front. They can never alter the ideal-gating endpoint selection.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [_with_output_metric_aliases(candidate) for candidate in candidates]
    by_ratio = {
        str(ratio): select_system_output_endpoints(
            candidate_rows,
            aggregated_e2e_s=aggregated_e2e_s,
            max_e2e_ratio=ratio,
            minimum_accuracy=minimum_accuracy,
            prefill_pareto_trial_ids=prefill_pareto_trial_ids,
        )
        for ratio in e2e_ratios
    }
    nominal = by_ratio["1.25"]
    _write_mapping_csv(
        output_dir / "system_output_tps_efficiency_pareto.csv",
        nominal["pareto"],
    )
    ungated_by_ratio = None
    if ungated_candidates is not None:
        ungated_rows = [_with_output_metric_aliases(candidate) for candidate in ungated_candidates]
        ungated_by_ratio = {
            str(ratio): select_system_output_endpoints(
                ungated_rows,
                aggregated_e2e_s=aggregated_e2e_s,
                max_e2e_ratio=ratio,
                minimum_accuracy=minimum_accuracy,
                prefill_pareto_trial_ids=prefill_pareto_trial_ids,
            )
            for ratio in e2e_ratios
        }
        _write_mapping_csv(
            output_dir / "system_output_tps_efficiency_pareto_ungated_shadow.csv",
            ungated_by_ratio["1.25"]["pareto"],
        )
    payload = {
        "schema": "system_selector_endpoints_v2",
        "main_energy_semantics": "ideal_hierarchical_gating",
        "ungated_semantics": "shadow_only_not_used_for_ranking",
        "sensitivity_by_e2e_ratio": by_ratio,
        "ungated_shadow_sensitivity_by_e2e_ratio": ungated_by_ratio,
    }
    (output_dir / "system_selector_endpoints_v2.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _constraint_satisfied(candidate: Mapping[str, Any], kind: str) -> bool:
    explicit = candidate.get(f"{kind}_constraint_satisfied")
    if explicit is not None:
        return bool(explicit)
    if kind == "hbm":
        for key in (
            "per_chip_hbm_capacity_feasible",
            "hbm_capacity_feasible",
        ):
            if candidate.get(key) is not None:
                return bool(candidate[key])
        for required_key, capacity_key in (
            ("per_chip_hbm_required_bytes", "per_chip_hbm_capacity_bytes"),
            ("aggregate_hbm_required_bytes", "aggregate_hbm_capacity_bytes"),
        ):
            if candidate.get(required_key) is not None and candidate.get(capacity_key) is not None:
                return float(candidate[required_key]) <= float(candidate[capacity_key])
    if kind == "area":
        budget = candidate.get("area_budget_constraint_mm2")
        actual = candidate.get(
            "total_silicon_area_mm2",
            candidate.get("aggregate_area_mm2"),
        )
        if budget is not None and actual is not None:
            return float(actual) <= float(budget)
    violation = candidate.get(f"{kind}_budget_violation")
    if violation is None:
        violation = candidate.get(f"{kind}_budget_constraint")
    if violation is not None:
        return float(violation) <= 0.0
    return False


def _system_tie_key(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        -float(candidate.get("accuracy", candidate.get("accuracy_score", 0.0))),
        float(candidate.get("aggregate_area_mm2", math.inf)),
        int(candidate.get("prefill_trial", candidate.get("trial", -1))),
    )


def _with_output_metric_aliases(candidate: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(candidate)
    if normalized.get("projected_pipeline_output_tokens_per_s") is None:
        normalized["projected_pipeline_output_tokens_per_s"] = normalized.get(
            "projected_pipeline_throughput_requests_per_s"
        )
    if normalized.get("projected_output_tokens_per_j") is None:
        normalized["projected_output_tokens_per_j"] = normalized.get("throughput_per_watt_requests_per_j")
    return normalized


def _write_mapping_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    scalar_rows = [
        {key: value for key, value in row.items() if value is None or isinstance(value, (bool, int, float, str))}
        for row in rows
    ]
    fields = sorted({key for row in scalar_rows for key in row}) or ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _validate_positive(*, batch_size: int, duration: float, energy: float) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if energy <= 0:
        raise ValueError("energy must be positive")


def _batch_tokens(batch_size: int, tokens_per_request: int | None) -> int | None:
    if tokens_per_request is None:
        return None
    if tokens_per_request <= 0:
        raise ValueError("tokens per request must be positive")
    return batch_size * tokens_per_request


def _optional_sum(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left + right


__all__ = [
    "aggregated_system_metrics",
    "disaggregated_pipeline_metrics",
    "select_max_output_tokens_per_j",
    "select_max_throughput_per_watt",
    "select_system_output_endpoints",
    "select_system_pareto_endpoints",
    "system_output_tps_efficiency_pareto",
    "system_throughput_efficiency_pareto",
    "write_system_selector_artifacts",
]
