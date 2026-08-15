"""Explicit system metrics for aggregated and disaggregated serving results."""

from __future__ import annotations

import math
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def aggregated_system_metrics(
    *,
    batch_size: int,
    full_batch_e2e_s: float,
    full_batch_energy_j: float,
    total_tokens_per_request: int | None = None,
) -> dict[str, float | str | None]:
    """Summarize a measured fixed-batch aggregated serving run."""

    _validate_positive(batch_size=batch_size, duration=full_batch_e2e_s, energy=full_batch_energy_j)
    throughput = batch_size / full_batch_e2e_s
    energy_per_request = full_batch_energy_j / batch_size
    return {
        "metric_schema": "fixed-batch-throughput-v1",
        "fidelity": "measured_aggregated_full_batch",
        "full_batch_e2e_s": full_batch_e2e_s,
        "service_interval_s": full_batch_e2e_s,
        "throughput_requests_per_s": throughput,
        "system_energy_j": full_batch_energy_j,
        "energy_per_request_j": energy_per_request,
        "average_system_power_w": full_batch_energy_j / full_batch_e2e_s,
        "throughput_per_watt_requests_per_j": batch_size / full_batch_energy_j,
        "energy_per_token_j": (
            energy_per_request / total_tokens_per_request
            if total_tokens_per_request is not None and total_tokens_per_request > 0
            else None
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
    e2e_latency_s: float | None = None,
    total_tokens_per_request: int | None = None,
    request_ttft_s: float | None = None,
    request_tpot_s: float | None = None,
    ttft_slo_s: float | None = None,
    tpot_slo_s: float | None = None,
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

    service_interval = max(
        prefill_interval_s,
        kv_handoff_interval_s,
        decode_interval_s,
    )
    system_energy = math.fsum(
        (prefill_energy_j, kv_handoff_energy_j, decode_energy_j)
    )
    throughput = batch_size / service_interval
    energy_per_request = system_energy / batch_size
    slo_complete = ttft_slo_s is not None and tpot_slo_s is not None
    slo_observed = request_ttft_s is not None and request_tpot_s is not None
    slo_pass = (
        bool(request_ttft_s <= ttft_slo_s and request_tpot_s <= tpot_slo_s)
        if slo_complete and slo_observed
        else None
    )
    return {
        "metric_schema": "fixed-batch-throughput-v1",
        "fidelity": "analytical_fixed_batch_pipeline_envelope",
        "prefill_interval_s": prefill_interval_s,
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
        "e2e_latency_s": e2e_latency_s,
        "system_energy_j": system_energy,
        "energy_per_request_j": energy_per_request,
        "average_pipeline_power_w": system_energy / service_interval,
        "throughput_per_watt_requests_per_j": batch_size / system_energy,
        "energy_per_token_j": (
            energy_per_request / total_tokens_per_request
            if total_tokens_per_request is not None and total_tokens_per_request > 0
            else None
        ),
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
        "no_queueing_or_continuous_batching_simulation": True,
        "imported_kv_decode_proxy": True,
        "real_plena_to_a100_kv_import": False,
    }


def select_max_throughput_per_watt(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
) -> Mapping[str, Any] | None:
    """Select the efficient feasible system without conflating throughput and goodput."""

    if aggregated_e2e_s <= 0 or max_e2e_ratio <= 0:
        raise ValueError("latency reference and ratio must be positive")
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate.get("accuracy", -math.inf)) > minimum_accuracy
        and bool(candidate.get("area_constraint_satisfied", False))
        and bool(candidate.get("hbm_constraint_satisfied", False))
        and float(candidate.get("e2e_latency_s", math.inf))
        <= max_e2e_ratio * aggregated_e2e_s
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda candidate: (
            float(candidate["throughput_per_watt_requests_per_j"]),
            float(candidate.get("projected_pipeline_throughput_requests_per_s", 0.0)),
            -float(candidate.get("energy_per_request_j", math.inf)),
            -float(candidate.get("aggregate_area_mm2", math.inf)),
        ),
    )


def system_throughput_efficiency_pareto(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Return the feasible throughput/throughput-per-watt maximum Pareto.

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
        if (
            prefill_pareto_trial_ids is not None
            and trial not in prefill_pareto_trial_ids
        ):
            continue
        accuracy = float(
            candidate.get("accuracy", candidate.get("accuracy_score", -math.inf))
        )
        if accuracy <= minimum_accuracy:
            continue
        if not _constraint_satisfied(candidate, "area"):
            continue
        if not _constraint_satisfied(candidate, "hbm"):
            continue
        if (
            float(candidate.get("e2e_latency_s", math.inf))
            > max_e2e_ratio * aggregated_e2e_s
        ):
            continue
        throughput = float(
            candidate.get("projected_pipeline_throughput_requests_per_s", -math.inf)
        )
        efficiency = float(
            candidate.get("throughput_per_watt_requests_per_j", -math.inf)
        )
        if not math.isfinite(throughput) or not math.isfinite(efficiency):
            continue
        candidate["system_selector_latency_ratio"] = max_e2e_ratio
        candidate["system_selector_accuracy_threshold"] = minimum_accuracy
        candidate["system_selector_prefill_pareto_only"] = (
            prefill_pareto_trial_ids is not None
        )
        feasible.append(candidate)

    front: list[dict[str, Any]] = []
    for index, candidate in enumerate(feasible):
        throughput = float(candidate["projected_pipeline_throughput_requests_per_s"])
        efficiency = float(candidate["throughput_per_watt_requests_per_j"])
        dominated = False
        for other_index, other in enumerate(feasible):
            if index == other_index:
                continue
            other_throughput = float(
                other["projected_pipeline_throughput_requests_per_s"]
            )
            other_efficiency = float(other["throughput_per_watt_requests_per_j"])
            if (
                other_throughput >= throughput
                and other_efficiency >= efficiency
                and (other_throughput > throughput or other_efficiency > efficiency)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    # Collapse exact objective duplicates deterministically.
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for candidate in front:
        key = (
            float(candidate["projected_pipeline_throughput_requests_per_s"]),
            float(candidate["throughput_per_watt_requests_per_j"]),
        )
        incumbent = unique.get(key)
        if incumbent is None or _system_tie_key(candidate) < _system_tie_key(
            incumbent
        ):
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            -float(candidate["projected_pipeline_throughput_requests_per_s"]),
            -float(candidate["throughput_per_watt_requests_per_j"]),
            _system_tie_key(candidate),
        ),
    )


def select_system_pareto_endpoints(
    candidates: Iterable[Mapping[str, Any]],
    *,
    aggregated_e2e_s: float,
    max_e2e_ratio: float = 1.25,
    minimum_accuracy: float = 0.9,
    prefill_pareto_trial_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, Any]:
    """Build the formal two-objective system front and its named endpoints."""

    front = system_throughput_efficiency_pareto(
        candidates,
        aggregated_e2e_s=aggregated_e2e_s,
        max_e2e_ratio=max_e2e_ratio,
        minimum_accuracy=minimum_accuracy,
        prefill_pareto_trial_ids=prefill_pareto_trial_ids,
    )
    maximum_throughput = (
        min(
            front,
            key=lambda candidate: (
                -float(candidate["projected_pipeline_throughput_requests_per_s"]),
                -float(candidate["throughput_per_watt_requests_per_j"]),
                _system_tie_key(candidate),
            ),
        )
        if front
        else None
    )
    maximum_efficiency = (
        max(
            front,
            key=lambda candidate: (
                float(candidate["throughput_per_watt_requests_per_j"]),
                float(candidate["projected_pipeline_throughput_requests_per_s"]),
                -float(candidate.get("energy_per_request_j", math.inf)),
                -float(candidate.get("aggregate_area_mm2", math.inf)),
            ),
        )
        if front
        else None
    )
    return {
        "schema": "system_throughput_efficiency_pareto_v1",
        "objective_directions": {
            "projected_pipeline_throughput_requests_per_s": "maximize",
            "throughput_per_watt_requests_per_j": "maximize",
        },
        "max_e2e_ratio": max_e2e_ratio,
        "minimum_accuracy": minimum_accuracy,
        "prefill_pareto_only": prefill_pareto_trial_ids is not None,
        "pareto": front,
        "maximum_throughput": maximum_throughput,
        "maximum_throughput_per_watt": maximum_efficiency,
    }


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
    candidate_rows = [dict(candidate) for candidate in candidates]
    by_ratio = {
        str(ratio): select_system_pareto_endpoints(
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
        output_dir / "system_throughput_efficiency_pareto.csv",
        nominal["pareto"],
    )
    ungated_by_ratio = None
    if ungated_candidates is not None:
        ungated_rows = [dict(candidate) for candidate in ungated_candidates]
        ungated_by_ratio = {
            str(ratio): select_system_pareto_endpoints(
                ungated_rows,
                aggregated_e2e_s=aggregated_e2e_s,
                max_e2e_ratio=ratio,
                minimum_accuracy=minimum_accuracy,
                prefill_pareto_trial_ids=prefill_pareto_trial_ids,
            )
            for ratio in e2e_ratios
        }
        _write_mapping_csv(
            output_dir
            / "system_throughput_efficiency_pareto_ungated_shadow.csv",
            ungated_by_ratio["1.25"]["pareto"],
        )
    payload = {
        "schema": "system_selector_endpoints_v1",
        "main_energy_semantics": "ideal_hierarchical_gating",
        "ungated_semantics": "shadow_only_not_used_for_ranking",
        "sensitivity_by_e2e_ratio": by_ratio,
        "ungated_shadow_sensitivity_by_e2e_ratio": ungated_by_ratio,
    }
    (output_dir / "system_selector_endpoints.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
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
            if (
                candidate.get(required_key) is not None
                and candidate.get(capacity_key) is not None
            ):
                return float(candidate[required_key]) <= float(
                    candidate[capacity_key]
                )
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


def _write_mapping_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    scalar_rows = [
        {
            key: value
            for key, value in row.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
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


__all__ = [
    "aggregated_system_metrics",
    "disaggregated_pipeline_metrics",
    "select_max_throughput_per_watt",
    "system_throughput_efficiency_pareto",
    "select_system_pareto_endpoints",
    "write_system_selector_artifacts",
]
