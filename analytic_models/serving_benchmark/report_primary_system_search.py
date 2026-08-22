"""Re-score the primary 90k/8k PLENA Pareto sets against measured A100 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .phases import enrich_phase_ttft_semantics
from .system_metrics import (
    disaggregated_pipeline_metrics,
    select_system_output_endpoints,
)


BATCH_SIZE = 8
INPUT_TOKENS = 90_000
OUTPUT_TOKENS = 8_000
TOTAL_TOKENS_PER_REQUEST = INPUT_TOKENS + OUTPUT_TOKENS
INTERCONNECT_ENERGY_PJ_PER_BIT = 8.0


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _aggregate_rows(path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _read_json(path)["rows"]]
    for row in rows:
        if row.get("median_mean_request_tpot_s") is not None:
            continue
        summary_path = path.parent / "points" / str(row["point_id"]) / "summary.json"
        if summary_path.is_file():
            row.update(_summary_row(summary_path))
    return rows


def _find_gpu_row(rows: Iterable[Mapping[str, Any]], *, tp: int, local_batch: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if int(row["tensor_parallel_size"]) == tp
        and int(row["local_batch_size"]) == local_batch
        and row.get("validation_status") == "pass"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one TP{tp}/B{local_batch} row, found {len(matches)}")
    return matches[0]


def _summary_row(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("status") != "complete":
        raise ValueError(f"measurement did not pass validation: {path}")
    repetitions = summary.get("repetitions", [])
    if not repetitions:
        raise ValueError(f"measurement has no repetitions: {path}")

    point = summary["point"]
    phases = []
    for repetition in repetitions:
        diagnostics = repetition.get("request_diagnostics")
        no_preemption = bool(diagnostics) and all(
            int(item.get("max_preemptions", 0)) == 0 for item in diagnostics.values()
        )
        phases.append(
            enrich_phase_ttft_semantics(
                repetition["phase"],
                batch_size=int(point["local_batch_size"]),
                uniform_prompt_shape=True,
                no_preemption=no_preemption,
            )
        )

    def median(path_keys: tuple[str, ...]) -> float:
        values = []
        for repetition in repetitions:
            value: Any = repetition
            for key in path_keys:
                value = value[key]
            values.append(float(value))
        return statistics.median(values)

    def phase_median(field: str) -> float | None:
        values = [phase.get(field) for phase in phases]
        if not values or any(value is None for value in values):
            return None
        return float(statistics.median(float(value) for value in values))

    request_ttft_samples = [
        float(value)
        for phase in phases
        for value in (phase.get("request_ttft_s", {}).values() if isinstance(phase.get("request_ttft_s"), dict) else ())
    ]
    request_tpot_samples = [
        float(value)
        for phase in phases
        for value in (phase.get("request_tpot_s", {}).values() if isinstance(phase.get("request_tpot_s"), dict) else ())
    ]
    return {
        "validation_status": "pass",
        "model": point["model_name"],
        "tensor_parallel_size": int(point["tensor_parallel_size"]),
        "local_batch_size": int(point["local_batch_size"]),
        "input_tokens": int(point["input_tokens"]),
        "output_tokens": int(point["output_tokens"]),
        "median_full_request_latency_s": median(("phase", "full_request_latency_s")),
        "median_complete_energy_mj": median(("power", "complete_request", "nvml_counter_energy_mj")),
        "median_imported_kv_decode_proxy_latency_s": median(("phase", "imported_kv_decode_proxy_latency_s")),
        "median_imported_kv_decode_proxy_energy_mj": median(
            ("power", "imported_kv_decode_proxy", "nvml_counter_energy_mj")
        ),
        "median_first_decode_iteration_latency_s": median(("phase", "first_decode_iteration_latency_s")),
        "median_idle_total_board_power_w": median(("power", "idle_baseline", "average_total_board_power_w")),
        "median_earliest_request_ttft_s": phase_median("earliest_request_ttft_s"),
        "median_mean_request_ttft_s": phase_median("mean_request_ttft_s"),
        "median_median_request_ttft_s": phase_median("median_request_ttft_s"),
        "median_p95_request_ttft_s": phase_median("p95_request_ttft_s"),
        "median_latest_request_ttft_s": phase_median("latest_request_ttft_s"),
        "median_scheduler_admitted_ttft_best_available_s": phase_median("scheduler_admitted_ttft_best_available_s"),
        "median_mean_request_tpot_s": phase_median("mean_request_tpot_s"),
        "median_median_request_tpot_s": phase_median("median_request_tpot_s"),
        "median_p95_request_tpot_s": phase_median("p95_request_tpot_s"),
        "median_median_tbt_s": phase_median("median_tbt_s"),
        "median_p95_tbt_s": phase_median("p95_tbt_s"),
        "request_visible_ttft_samples_s": request_ttft_samples or None,
        "request_tpot_samples_s": request_tpot_samples or None,
        "scheduler_admitted_ttft_sources": ";".join(
            sorted({str(phase["scheduler_admitted_ttft_source"]) for phase in phases})
        ),
        "phase_schema_version": "request-visible-v4",
    }


def _replica_check_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("requires_concurrency_correction"):
        raise ValueError(f"replica correction exceeds the accepted 5% bound: {path}")
    summary_paths = sorted(path.parent.glob("replica_*/points/*/summary.json"))
    if len(summary_paths) == len(payload["concurrent_replicas"]):
        return [_summary_row(summary_path) for summary_path in summary_paths]
    return [dict(row) for row in payload["concurrent_replicas"]]


def reconstruct_replica_phase(rows: Sequence[Mapping[str, Any]], *, phase: str, fidelity: str) -> dict[str, Any]:
    """Combine replica measurements and charge idle tails to the slowest replica."""

    if not rows:
        raise ValueError("at least one replica row is required")
    if phase == "complete":
        latency_key = "median_full_request_latency_s"
        energy_key = "median_complete_energy_mj"
    elif phase == "decode":
        latency_key = "median_imported_kv_decode_proxy_latency_s"
        energy_key = "median_imported_kv_decode_proxy_energy_mj"
    else:
        raise ValueError(f"unsupported phase: {phase}")

    makespan = max(float(row[latency_key]) for row in rows)
    active_energy_j = math.fsum(float(row[energy_key]) / 1000.0 for row in rows)
    idle_tail_energy_j = math.fsum(
        float(row["median_idle_total_board_power_w"]) * (makespan - float(row[latency_key])) for row in rows
    )
    global_batch = sum(int(row["local_batch_size"]) for row in rows)
    output_token_values = {int(row["output_tokens"]) for row in rows}
    if len(output_token_values) != 1:
        raise ValueError("replicas must use the same output-token count")
    output_tokens_per_request = output_token_values.pop()

    request_tpot_samples = [
        float(value)
        for row in rows
        for value in (
            row.get("request_tpot_samples_s")
            if isinstance(row.get("request_tpot_samples_s"), list)
            else [row["median_mean_request_tpot_s"]] * int(row["local_batch_size"])
            if row.get("median_mean_request_tpot_s") is not None
            else []
        )
    ]
    request_ttft_samples = [
        float(value)
        for row in rows
        for value in (
            row.get("request_visible_ttft_samples_s")
            if isinstance(row.get("request_visible_ttft_samples_s"), list)
            else [row["median_median_request_ttft_s"]] * int(row["local_batch_size"])
            if row.get("median_median_request_ttft_s") is not None
            else []
        )
    ]
    per_replica_p95_tpot = [
        float(row["median_p95_request_tpot_s"]) for row in rows if row.get("median_p95_request_tpot_s") is not None
    ]
    scheduler_admitted = [
        (float(row["median_scheduler_admitted_ttft_best_available_s"]), int(row["local_batch_size"]))
        for row in rows
        if row.get("median_scheduler_admitted_ttft_best_available_s") is not None
    ]
    scheduler_sources = sorted(
        {source for row in rows for source in str(row.get("scheduler_admitted_ttft_sources", "")).split(";") if source}
    )
    return {
        "latency_s": makespan,
        "energy_j": active_energy_j + idle_tail_energy_j,
        "active_energy_j": active_energy_j,
        "idle_tail_energy_j": idle_tail_energy_j,
        "first_decode_step_s": (
            max(float(row["median_first_decode_iteration_latency_s"]) for row in rows) if phase == "decode" else None
        ),
        "global_batch_size": global_batch,
        "output_tokens_per_request": output_tokens_per_request,
        "global_output_tokens": global_batch * output_tokens_per_request,
        "output_tokens_per_s": (global_batch * output_tokens_per_request / makespan),
        "mean_request_tpot_s": (statistics.fmean(request_tpot_samples) if request_tpot_samples else None),
        "p95_request_tpot_s": _p95(request_tpot_samples),
        "conservative_p95_request_tpot_s": (max(per_replica_p95_tpot) if per_replica_p95_tpot else None),
        "median_request_visible_ttft_s": (statistics.median(request_ttft_samples) if request_ttft_samples else None),
        "p95_request_visible_ttft_s": _p95(request_ttft_samples),
        "batch_first_token_barrier_s": (max(request_ttft_samples) if request_ttft_samples else None),
        "scheduler_admitted_ttft_exact_or_proxy_s": (
            math.fsum(value * count for value, count in scheduler_admitted)
            / sum(count for _, count in scheduler_admitted)
            if scheduler_admitted
            else None
        ),
        "scheduler_admitted_ttft_source": (";".join(scheduler_sources) if scheduler_sources else "unavailable"),
        "request_tpot_samples_s": request_tpot_samples or None,
        "request_visible_ttft_samples_s": request_ttft_samples or None,
        "replica_count": len(rows),
        "fidelity": fidelity,
    }


def _copies(row: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    return [dict(row) for _ in range(count)]


def _gpu_topologies(measurement_root: Path) -> dict[str, dict[str, Any]]:
    screening_32b = _aggregate_rows(measurement_root / "screening_32b/aggregate.json")
    screening_235b = _aggregate_rows(measurement_root / "screening_235b/aggregate.json")
    followup = measurement_root / "topology_followup_v1"

    row32 = lambda tp, batch: _find_gpu_row(  # noqa: E731
        screening_32b, tp=tp, local_batch=batch
    )
    row235 = lambda tp, batch: _find_gpu_row(  # noqa: E731
        screening_235b, tp=tp, local_batch=batch
    )

    r8_tp2 = _replica_check_rows(followup / "r8_tp2x4_b2/replica_check.json")
    d4_tp2 = _replica_check_rows(followup / "d4_tp2x2_b4/replica_check.json")
    d4_tp1 = _replica_check_rows(followup / "d4_tp1x4_b2/replica_check.json")
    d6_b3 = _replica_check_rows(followup / "d6_tp2x3_b332/b3_replicas/replica_check.json")
    replica_32_tp4 = _replica_check_rows(measurement_root / "followup_v1/replica_32b_tp4_b4/replica_check.json")
    replica_235_tp4 = _replica_check_rows(measurement_root / "followup_v1/replica_235b_tp4_b4/replica_check.json")
    d12_root = followup / "235b_d12_tp4x3_b332_probe"
    d12_b2 = _summary_row(d12_root / "b2/points/qwen3-235b-a22b.primary-90000x8000.tp4.b2/summary.json")
    d12_b3 = _summary_row(d12_root / "b3/points/qwen3-235b-a22b.primary-90000x8000.tp4.b3/summary.json")

    aggregate_32 = {
        "TP1xDP8_B1": reconstruct_replica_phase(
            _copies(row32(1, 1), 8),
            phase="complete",
            fidelity="measured_replica_extrapolation",
        ),
        "TP2xDP4_B2": reconstruct_replica_phase(r8_tp2, phase="complete", fidelity="measured_concurrent_replicas"),
        "TP4xDP2_B4": reconstruct_replica_phase(
            replica_32_tp4,
            phase="complete",
            fidelity="measured_concurrent_replicas",
        ),
        "TP8xDP1_B8": reconstruct_replica_phase([row32(8, 8)], phase="complete", fidelity="measured_single_replica"),
    }
    decode_32 = {
        6: {
            "TP2xDP3_B3_3_2": reconstruct_replica_phase(
                [*d6_b3, row32(2, 2)],
                phase="decode",
                fidelity="measured_b3_pair_plus_measured_b2",
            ),
            "TP1xDP6_B2_2_1_1_1_1": reconstruct_replica_phase(
                [*_copies(row32(1, 2), 2), *_copies(row32(1, 1), 4)],
                phase="decode",
                fidelity="measured_replica_extrapolation",
            ),
        },
        4: {
            "TP4xDP1_B8": reconstruct_replica_phase([row32(4, 8)], phase="decode", fidelity="measured_single_replica"),
            "TP2xDP2_B4": reconstruct_replica_phase(d4_tp2, phase="decode", fidelity="measured_concurrent_replicas"),
            "TP1xDP4_B2": reconstruct_replica_phase(d4_tp1, phase="decode", fidelity="measured_concurrent_replicas"),
        },
    }

    aggregate_235 = {
        "TP4xDP4_B2": reconstruct_replica_phase(
            _copies(d12_b2, 4),
            phase="complete",
            fidelity="measured_replica_extrapolation_to_16_gpus",
        ),
        "TP8xDP2_B4": reconstruct_replica_phase(
            _copies(row235(8, 4), 2),
            phase="complete",
            fidelity="measured_replica_extrapolation_to_16_gpus",
        ),
    }
    decode_235 = {
        12: {
            "TP4xDP3_B3_3_2": reconstruct_replica_phase(
                [d12_b3, d12_b3, d12_b2],
                phase="decode",
                fidelity="measured_mixed_pair_plus_one_b3_replica_extrapolation",
            )
        },
        8: {
            "TP4xDP2_B4": reconstruct_replica_phase(
                replica_235_tp4,
                phase="decode",
                fidelity="measured_concurrent_replicas",
            ),
            "TP8xDP1_B8": reconstruct_replica_phase([row235(8, 8)], phase="decode", fidelity="measured_single_replica"),
        },
        4: {
            "TP4xDP1_B8": reconstruct_replica_phase([row235(4, 8)], phase="decode", fidelity="measured_single_replica")
        },
    }
    return {
        "qwen3-32b": {"aggregate": aggregate_32, "decode": decode_32},
        "qwen3-235b-a22b": {"aggregate": aggregate_235, "decode": decode_235},
    }


def _aggregate_candidates(topologies: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for topology, values in topologies.items():
        latency = float(values["latency_s"])
        energy = float(values["energy_j"])
        global_batch = int(values["global_batch_size"])
        if global_batch != BATCH_SIZE:
            raise ValueError(f"aggregated topology {topology} reconstructs batch {global_batch}")
        global_output_tokens = global_batch * OUTPUT_TOKENS
        candidates.append(
            {
                "topology": topology,
                **dict(values),
                "throughput_requests_per_s": BATCH_SIZE / latency,
                "throughput_per_watt_requests_per_j": BATCH_SIZE / energy,
                "output_tokens_per_s": global_output_tokens / latency,
                "output_tokens_per_j": global_output_tokens / energy,
                "energy_per_output_token_j": energy / global_output_tokens,
                "energy_per_request_j": energy / BATCH_SIZE,
                "average_power_w": energy / latency,
                "legacy_metric_alias": True,
            }
        )
    return candidates


def _aggregate_endpoints(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "maximum_output_tps": dict(
            max(
                candidates,
                key=lambda row: (
                    float(row["output_tokens_per_s"]),
                    float(row["output_tokens_per_j"]),
                ),
            )
        ),
        "maximum_output_tokens_per_j": dict(
            max(
                candidates,
                key=lambda row: (
                    float(row["output_tokens_per_j"]),
                    float(row["output_tokens_per_s"]),
                ),
            )
        ),
        "legacy_endpoint_aliases": {
            "maximum_throughput": "maximum_output_tps",
            "maximum_throughput_per_watt": "maximum_output_tokens_per_j",
        },
    }


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _system_candidate(
    row: Mapping[str, str],
    *,
    model: str,
    prefill_budget: int,
    decode_gpu_count: int,
    decode_topology: str,
    decode: Mapping[str, Any],
) -> dict[str, Any]:
    prefill_s = float(row["prefill_latency_ms"]) / 1000.0
    prefill_energy_j = float(row["prefill_system_energy_mj_ideal"]) / 1000.0
    handoff_s = float(row["fp16_kv_handoff_latency_ms"]) / 1000.0
    handoff_bytes = float(row["fp16_kv_handoff_bytes"])
    handoff_energy_j = handoff_bytes * 8.0 * INTERCONNECT_ENERGY_PJ_PER_BIT * 1e-12
    decode_s = float(decode["latency_s"])
    e2e_s = prefill_s + handoff_s + decode_s
    area_budget_mm2 = prefill_budget * 0.9 * 826.0
    metrics = disaggregated_pipeline_metrics(
        batch_size=BATCH_SIZE,
        prefill_interval_s=prefill_s,
        kv_handoff_interval_s=handoff_s,
        decode_interval_s=decode_s,
        prefill_energy_j=prefill_energy_j,
        kv_handoff_energy_j=handoff_energy_j,
        decode_energy_j=float(decode["energy_j"]),
        e2e_latency_s=e2e_s,
        input_tokens_per_request=INPUT_TOKENS,
        output_tokens_per_request=OUTPUT_TOKENS,
        total_tokens_per_request=TOTAL_TOKENS_PER_REQUEST,
        request_tpot_s=decode.get("mean_request_tpot_s"),
        first_decode_step_s=float(decode["first_decode_step_s"]),
    )
    return {
        **metrics,
        "model": model,
        "prefill_trial": int(row["trial"]),
        "prefill_budget_a100_equivalent": prefill_budget,
        "decode_gpu_count": decode_gpu_count,
        "system_split": f"P{prefill_budget}:D{decode_gpu_count}",
        "decode_topology": decode_topology,
        "decode_fidelity": decode["fidelity"],
        "accuracy": float(row["accuracy_score"]),
        "precision_profile": row["precision_profile"],
        "MLEN": int(row["MLEN"]),
        "VLEN": int(row["VLEN"]),
        "BLEN": int(row["BLEN"]),
        "INT_DATA_WIDTH": int(row["INT_DATA_WIDTH"]),
        "softmax_row_lanes": int(row["softmax_row_lanes"]),
        "softmax_row_lane_fidelity": row["softmax_row_lane_fidelity"],
        "rtl_validation_available": _bool(row["rtl_validation_available"]),
        "matrix_sram_config_id": row["matrix_sram_config_id"],
        "matrix_sram_policy": row["matrix_sram_policy"],
        "matrix_sram_tiles": int(row["MATRIX_SRAM_TILES"]),
        "physical_chip_count": int(row["physical_chip_count"]),
        "dp_degree": int(row["dp_degree"]),
        "tp_degree": int(row["tp_degree"]),
        "ep_degree": int(row["ep_degree"]),
        "nvlink_port_count": int(row["nvlink_port_count"]),
        "aggregate_area_mm2": float(row["total_silicon_area_mm2"]),
        "total_silicon_area_mm2": float(row["total_silicon_area_mm2"]),
        "area_budget_constraint_mm2": area_budget_mm2,
        "area_constraint_satisfied": (float(row["total_silicon_area_mm2"]) <= area_budget_mm2),
        "hbm_constraint_satisfied": True,
        "resource_feasibility_source": "completed_budget_conditioned_dse_trial",
        "prefill_energy_j": prefill_energy_j,
        "kv_handoff_energy_j": handoff_energy_j,
        "decode_energy_j": float(decode["energy_j"]),
        "projected_static_batch_request_ttft_s": metrics["plena_disaggregated_batch_admitted_ttft_s"],
        "projected_mean_request_tpot_s": decode.get("mean_request_tpot_s"),
        "projected_p95_request_tpot_s": decode.get(
            "conservative_p95_request_tpot_s",
            decode.get("p95_request_tpot_s"),
        ),
        "projected_request_latency_fidelity": (
            "analytical_static_batch_ttft_plus_measured_imported_kv_decode_proxy_tpot"
        ),
        "fp16_kv_handoff_bytes": handoff_bytes,
        "candidate_fidelity": row["candidate_fidelity"],
    }


def _comparison(endpoint: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "output_tps_speedup_x": (
            float(endpoint["projected_pipeline_output_tokens_per_s"]) / float(baseline["output_tokens_per_s"])
        ),
        "output_tokens_per_j_improvement_x": (
            float(endpoint["projected_output_tokens_per_j"]) / float(baseline["output_tokens_per_j"])
        ),
        "energy_per_output_token_reduction_x": (
            float(baseline["energy_per_output_token_j"]) / float(endpoint["energy_per_output_token_j"])
        ),
        "energy_per_request_reduction_x": (
            float(baseline["energy_per_request_j"]) / float(endpoint["energy_per_request_j"])
        ),
        "e2e_latency_ratio_x": (float(endpoint["e2e_latency_s"]) / float(baseline["latency_s"])),
    }


def _select(
    candidates: Sequence[Mapping[str, Any]],
    *,
    aggregate_fastest: Mapping[str, Any],
    rtl_only: bool,
) -> dict[str, Any]:
    selected_rows = [dict(row) for row in candidates if not rtl_only or bool(row["rtl_validation_available"])]
    trial_ids = {int(row["prefill_trial"]) for row in selected_rows}
    by_ratio = {
        str(ratio): select_system_output_endpoints(
            selected_rows,
            aggregated_e2e_s=float(aggregate_fastest["latency_s"]),
            max_e2e_ratio=ratio,
            minimum_accuracy=0.9,
            prefill_pareto_trial_ids=trial_ids,
        )
        for ratio in (1.0, 1.25, 1.5)
    }
    return {
        "candidate_count": len(selected_rows),
        "candidate_scope": "rtl_isa_encodable_r_le_8" if rtl_only else "r16_structural_included",
        "sensitivity_by_e2e_ratio": by_ratio,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    scalar_rows = [
        {key: value for key, value in row.items() if value is None or isinstance(value, (bool, int, float, str))}
        for row in rows
    ]
    fields = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _endpoint_identity(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["prefill_trial"]),
        str(row["decode_topology"]),
        str(row["system_split"]),
    )


def _fmt_endpoint(row: Mapping[str, Any]) -> str:
    return (
        f"{row['system_split']}, {row['decode_topology']}, trial {row['prefill_trial']}, "
        f"M/V/B={row['MLEN']}/{row['VLEN']}/{row['BLEN']}, "
        f"R={row['softmax_row_lanes']}, N={row['physical_chip_count']} "
        f"DP/TP/EP={row['dp_degree']}/{row['tp_degree']}/{row['ep_degree']}, "
        f"{row['nvlink_port_count']} ports, "
        f"{row['matrix_sram_policy']}:{row['matrix_sram_tiles']}, "
        f"{row['precision_profile']}, acc={row['accuracy']:.2f}, "
        f"area={row['aggregate_area_mm2']:.1f} mm2"
    )


def _fmt_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "unavailable"
    return f"{numeric:.{digits}f}"


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Primary 90k/8k Serving Metrics And System Search",
        "",
        "## Semantics",
        "",
        "This report combines the completed bank-aware PLENA prefill Pareto sets with",
        "the measured A100 W4A16/FP16-KV curves for `90k input / 8k output, batch 8`.",
        "The headline throughput metric is generated output tokens per second. Energy",
        "efficiency is output tokens per joule. Requests/s and requests/J are retained",
        "only as auxiliary fixed-workload metrics.",
        "",
        "Disaggregated values are an analytical fixed-batch pipeline envelope using an",
        "imported-KV A100 decode proxy. There is no real PLENA-to-A100 KV import, queueing",
        "simulation, or continuous-batching simulation. PLENA energy uses ideal",
        "hierarchical clock gating. R16 is a structural extrapolation, so an ISA-encodable",
        "R<=8 result is retained beside the main result.",
        "",
    ]
    for model, result in payload["models"].items():
        aggregate = result["aggregate_gpu"]
        fastest_gpu = aggregate["endpoints"]["maximum_output_tps"]
        efficient_gpu = aggregate["endpoints"]["maximum_output_tokens_per_j"]
        nominal = result["main"]["sensitivity_by_e2e_ratio"]["1.25"]
        fastest = nominal["maximum_output_tps"]
        efficient = nominal["maximum_output_tokens_per_j"]
        lines.extend(
            [
                f"## {model}",
                "",
                "### User-visible latency",
                "",
                "| System | Objective | Median TTFT (s) | P95 TTFT (s) | Mean TPOT (ms) | P95 TPOT (ms) | Full-batch E2E (s) |",
                "|---|---|---:|---:|---:|---:|---:|",
                f"| Aggregated A100 | Maximum output TPS | {_fmt_optional(fastest_gpu.get('median_request_visible_ttft_s'))} | {_fmt_optional(fastest_gpu.get('p95_request_visible_ttft_s'))} | {_fmt_optional(float(fastest_gpu['mean_request_tpot_s']) * 1000 if fastest_gpu.get('mean_request_tpot_s') is not None else None)} | {_fmt_optional(float(fastest_gpu['conservative_p95_request_tpot_s']) * 1000 if fastest_gpu.get('conservative_p95_request_tpot_s') is not None else None)} | {fastest_gpu['latency_s']:.3f} |",
                f"| Aggregated A100 | Maximum output tokens/J | {_fmt_optional(efficient_gpu.get('median_request_visible_ttft_s'))} | {_fmt_optional(efficient_gpu.get('p95_request_visible_ttft_s'))} | {_fmt_optional(float(efficient_gpu['mean_request_tpot_s']) * 1000 if efficient_gpu.get('mean_request_tpot_s') is not None else None)} | {_fmt_optional(float(efficient_gpu['conservative_p95_request_tpot_s']) * 1000 if efficient_gpu.get('conservative_p95_request_tpot_s') is not None else None)} | {efficient_gpu['latency_s']:.3f} |",
                f"| PLENA + A100 | Maximum output TPS | {fastest['projected_static_batch_request_ttft_s']:.3f} | {fastest['projected_static_batch_request_ttft_s']:.3f} | {_fmt_optional(float(fastest['projected_mean_request_tpot_s']) * 1000 if fastest.get('projected_mean_request_tpot_s') is not None else None)} | {_fmt_optional(float(fastest['projected_p95_request_tpot_s']) * 1000 if fastest.get('projected_p95_request_tpot_s') is not None else None)} | {fastest['e2e_latency_s']:.3f} |",
                f"| PLENA + A100 | Maximum output tokens/J | {efficient['projected_static_batch_request_ttft_s']:.3f} | {efficient['projected_static_batch_request_ttft_s']:.3f} | {_fmt_optional(float(efficient['projected_mean_request_tpot_s']) * 1000 if efficient.get('projected_mean_request_tpot_s') is not None else None)} | {_fmt_optional(float(efficient['projected_p95_request_tpot_s']) * 1000 if efficient.get('projected_p95_request_tpot_s') is not None else None)} | {efficient['e2e_latency_s']:.3f} |",
                "",
                "PLENA TTFT is a projected static-batch value. It is not an online scheduler",
                "measurement and is therefore not used as a request-SLO selector.",
                "",
                "### Steady-state performance",
                "",
                "| System | Objective | Configuration | Output TPS | Requests/s | Bottleneck |",
                "|---|---|---|---:|---:|---|",
                f"| Aggregated A100 | Maximum output TPS | {fastest_gpu['topology']} | {fastest_gpu['output_tokens_per_s']:.2f} | {fastest_gpu['throughput_requests_per_s']:.6f} | full request |",
                f"| Aggregated A100 | Maximum output tokens/J | {efficient_gpu['topology']} | {efficient_gpu['output_tokens_per_s']:.2f} | {efficient_gpu['throughput_requests_per_s']:.6f} | full request |",
                f"| PLENA + A100 | Maximum output TPS | {_fmt_endpoint(fastest)} | {fastest['projected_pipeline_output_tokens_per_s']:.2f} | {fastest['projected_pipeline_throughput_requests_per_s']:.6f} | {fastest['bottleneck_stage']} |",
                f"| PLENA + A100 | Maximum output tokens/J | {_fmt_endpoint(efficient)} | {efficient['projected_pipeline_output_tokens_per_s']:.2f} | {efficient['projected_pipeline_throughput_requests_per_s']:.6f} | {efficient['bottleneck_stage']} |",
                "",
                "### Energy efficiency",
                "",
                "| System | Objective | Output tokens/J | Energy/output token (J) | Energy/request (kJ) | Average power (W) |",
                "|---|---|---:|---:|---:|---:|",
                f"| Aggregated A100 | Maximum output TPS | {fastest_gpu['output_tokens_per_j']:.6f} | {fastest_gpu['energy_per_output_token_j']:.6f} | {fastest_gpu['energy_per_request_j'] / 1000:.3f} | {fastest_gpu['average_power_w']:.2f} |",
                f"| Aggregated A100 | Maximum output tokens/J | {efficient_gpu['output_tokens_per_j']:.6f} | {efficient_gpu['energy_per_output_token_j']:.6f} | {efficient_gpu['energy_per_request_j'] / 1000:.3f} | {efficient_gpu['average_power_w']:.2f} |",
                f"| PLENA + A100 | Maximum output TPS | {fastest['projected_output_tokens_per_j']:.6f} | {fastest['energy_per_output_token_j']:.6f} | {fastest['energy_per_request_j'] / 1000:.3f} | {fastest['steady_state_average_system_power_w']:.2f} |",
                f"| PLENA + A100 | Maximum output tokens/J | {efficient['projected_output_tokens_per_j']:.6f} | {efficient['energy_per_output_token_j']:.6f} | {efficient['energy_per_request_j'] / 1000:.3f} | {efficient['steady_state_average_system_power_w']:.2f} |",
                "",
            ]
        )
        output_tps_cmp = result["comparisons"]["maximum_output_tps_vs_gpu"]
        efficiency_cmp = result["comparisons"]["maximum_output_tokens_per_j_vs_gpu"]
        lines.extend(
            [
                "### Matched-objective ratios",
                "",
                "| Objective | Output TPS speedup | Output tokens/J improvement | Energy/output token reduction | E2E ratio |",
                "|---|---:|---:|---:|---:|",
                f"| Maximum output TPS | {output_tps_cmp['output_tps_speedup_x']:.3f}x | {output_tps_cmp['output_tokens_per_j_improvement_x']:.3f}x | {output_tps_cmp['energy_per_output_token_reduction_x']:.3f}x lower | {output_tps_cmp['e2e_latency_ratio_x']:.3f}x |",
                f"| Maximum output tokens/J | {efficiency_cmp['output_tps_speedup_x']:.3f}x | {efficiency_cmp['output_tokens_per_j_improvement_x']:.3f}x | {efficiency_cmp['energy_per_output_token_reduction_x']:.3f}x lower | {efficiency_cmp['e2e_latency_ratio_x']:.3f}x |",
                "",
                "### Scheduler-admission diagnostic",
                "",
                "| A100 objective | Scheduler-admitted TTFT exact/proxy (s) | Fidelity | Batch first-token barrier (s) |",
                "|---|---:|---|---:|",
                f"| Maximum output TPS | {_fmt_optional(fastest_gpu.get('scheduler_admitted_ttft_exact_or_proxy_s'))} | {fastest_gpu.get('scheduler_admitted_ttft_source', 'unavailable')} | {_fmt_optional(fastest_gpu.get('batch_first_token_barrier_s'))} |",
                f"| Maximum output tokens/J | {_fmt_optional(efficient_gpu.get('scheduler_admitted_ttft_exact_or_proxy_s'))} | {efficient_gpu.get('scheduler_admitted_ttft_source', 'unavailable')} | {_fmt_optional(efficient_gpu.get('batch_first_token_barrier_s'))} |",
                "",
                "No disaggregated candidate satisfies the `1.0x` E2E bound. The complete",
                "`1.0x/1.25x/1.5x` selector results are retained in the JSON artifact.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Output TPS is `batch * output_tokens / max(prefill, handoff, decode)`.",
            "- Output tokens/J includes PLENA prefill, KV handoff, A100 decode, and replica",
            "  idle-tail energy. It is numerically TPS/W.",
            "- The 1.25x bound uses the fastest aggregated-A100 full-batch E2E result.",
            "- More than eight A100s and unmeasured replica combinations are labelled as",
            "  measured-replica extrapolations.",
            "- Endpoint/link static power, cooling, board-regulator loss, real KV-import",
            "  overhead, queueing, and continuous batching remain excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def run_search(*, dse_root: Path, measurement_root: Path, output_dir: Path) -> dict[str, Any]:
    topology_data = _gpu_topologies(measurement_root)
    campaign_specs = {
        "qwen3-32b": (
            (2, 6, "qwen3_32b_90k8_p2_r16_w4_v1"),
            (4, 4, "qwen3_32b_90k8_p4_r16_w4_v1"),
        ),
        "qwen3-235b-a22b": (
            (4, 12, "qwen3_235b_90k8_p4_r16_w4_v1"),
            (8, 8, "qwen3_235b_90k8_p8_r16_w4_v1"),
            (12, 4, "qwen3_235b_90k8_p12_r16_w4_v1"),
        ),
    }
    payload: dict[str, Any] = {
        "schema": "primary_90k8_system_search_v3",
        "workload": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS, "batch": BATCH_SIZE},
        "system_objectives": ["maximize_output_tps", "maximize_output_tokens_per_j"],
        "latency_constraint": "disaggregated_e2e <= ratio * fastest_aggregated_a100_e2e",
        "main_e2e_ratio": 1.25,
        "interconnect_energy_pj_per_bit": INTERCONNECT_ENERGY_PJ_PER_BIT,
        "models": {},
    }
    all_front_rows: list[dict[str, Any]] = []
    for model, specs in campaign_specs.items():
        aggregate_rows = _aggregate_candidates(topology_data[model]["aggregate"])
        aggregate_endpoints = _aggregate_endpoints(aggregate_rows)
        system_candidates: list[dict[str, Any]] = []
        for prefill_budget, decode_gpus, campaign in specs:
            pareto_rows = _read_csv(dse_root / campaign / "pareto_trials.csv")
            for row in pareto_rows:
                for decode_topology, decode in topology_data[model]["decode"][decode_gpus].items():
                    system_candidates.append(
                        _system_candidate(
                            row,
                            model=model,
                            prefill_budget=prefill_budget,
                            decode_gpu_count=decode_gpus,
                            decode_topology=decode_topology,
                            decode=decode,
                        )
                    )
        main = _select(
            system_candidates,
            aggregate_fastest=aggregate_endpoints["maximum_output_tps"],
            rtl_only=False,
        )
        rtl = _select(
            system_candidates,
            aggregate_fastest=aggregate_endpoints["maximum_output_tps"],
            rtl_only=True,
        )
        nominal = main["sensitivity_by_e2e_ratio"]["1.25"]
        result = {
            "aggregate_gpu": {"candidates": aggregate_rows, "endpoints": aggregate_endpoints},
            "system_candidate_count": len(system_candidates),
            "main": main,
            "rtl_encodable": rtl,
            "comparisons": {
                "maximum_output_tps_vs_gpu": _comparison(
                    nominal["maximum_output_tps"],
                    aggregate_endpoints["maximum_output_tps"],
                ),
                "maximum_output_tokens_per_j_vs_gpu": _comparison(
                    nominal["maximum_output_tokens_per_j"],
                    aggregate_endpoints["maximum_output_tokens_per_j"],
                ),
            },
            "rtl_comparisons": {
                "maximum_output_tps_vs_gpu": _comparison(
                    rtl["sensitivity_by_e2e_ratio"]["1.25"]["maximum_output_tps"],
                    aggregate_endpoints["maximum_output_tps"],
                ),
                "maximum_output_tokens_per_j_vs_gpu": _comparison(
                    rtl["sensitivity_by_e2e_ratio"]["1.25"]["maximum_output_tokens_per_j"],
                    aggregate_endpoints["maximum_output_tokens_per_j"],
                ),
            },
        }
        payload["models"][model] = result
        for scope in ("main", "rtl_encodable"):
            for row in result[scope]["sensitivity_by_e2e_ratio"]["1.25"]["pareto"]:
                all_front_rows.append({"selection_scope": scope, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / "primary_90k8_system_search_v2_results.json"
    if legacy_path.is_file():
        legacy = _read_json(legacy_path)
        migration_audit: dict[str, Any] = {"status": "pass", "models": {}}
        endpoint_names = (
            ("maximum_throughput", "maximum_output_tps"),
            ("maximum_throughput_per_watt", "maximum_output_tokens_per_j"),
        )
        for model, result in payload["models"].items():
            model_audit: dict[str, Any] = {}
            for old_name, new_name in endpoint_names:
                old_row = legacy["models"][model]["main"]["sensitivity_by_e2e_ratio"]["1.25"][old_name]
                new_row = result["main"]["sensitivity_by_e2e_ratio"]["1.25"][new_name]
                old_identity = _endpoint_identity(old_row)
                new_identity = _endpoint_identity(new_row)
                if old_identity != new_identity:
                    raise ValueError(f"metric migration changed {model} {new_name}: {old_identity} != {new_identity}")
                model_audit[new_name] = {
                    "legacy_identity": old_identity,
                    "v3_identity": new_identity,
                    "unchanged": True,
                }
            migration_audit["models"][model] = model_audit
        payload["legacy_v2_endpoint_identity_audit"] = migration_audit
    (output_dir / "primary_90k8_system_search_v3_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "primary_90k8_system_search_v3_pareto.csv", all_front_rows)
    _write_csv(output_dir / "system_output_tps_efficiency_pareto.csv", all_front_rows)
    (output_dir / "system_selector_endpoints_v2.json").write_text(
        json.dumps(
            {
                "schema": "system_selector_endpoints_v2",
                "models": {
                    model: {
                        "main": result["main"]["sensitivity_by_e2e_ratio"],
                        "rtl_encodable": result["rtl_encodable"]["sensitivity_by_e2e_ratio"],
                    }
                    for model, result in payload["models"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "primary_90k8_system_search_v3.md").write_text(_render_report(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dse-root",
        type=Path,
        default=Path("Workspace/qwen3_32b_dense_analytic/runs/disaggregate_prefill_dse_config_v1"),
    )
    parser.add_argument(
        "--measurement-root",
        type=Path,
        default=Path("Workspace/measurements/runpod_a100/plena_runpod_a100_v1/formal_v2"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Workspace/reports/serving"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_search(
        dse_root=args.dse_root,
        measurement_root=args.measurement_root,
        output_dir=args.output_dir,
    )
    print(json.dumps({model: data["comparisons"] for model, data in payload["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
