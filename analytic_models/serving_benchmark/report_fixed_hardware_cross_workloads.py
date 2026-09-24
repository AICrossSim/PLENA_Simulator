"""Evaluate primary-selected PLENA systems on the two fixed-workload holdouts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .prefill_idle_power import PrefillIdlePowerEstimator
from .report_primary_system_search import (
    INTERCONNECT_ENERGY_PJ_PER_BIT,
    REPO_ROOT,
    _aggregate_candidates,
    _aggregate_endpoints,
    _aggregate_rows,
    _annotate_fixed_batch_ttft,
    _comparison,
    _copies,
    reconstruct_replica_phase,
)
from .system_metrics import disaggregated_pipeline_metrics


BATCH_SIZE = 8
PRIMARY_RATIO = "1.25"
OBJECTIVES = ("maximum_output_tps", "maximum_output_tokens_per_j")
WORKLOADS = {
    "short-1400x200": {"input_tokens": 1_400, "output_tokens": 200},
    "primary-90000x8000": {"input_tokens": 90_000, "output_tokens": 8_000},
    "holdout-114000x5000": {"input_tokens": 114_000, "output_tokens": 5_000},
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_complete_trial(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("state") == "complete"]
    if not rows:
        raise ValueError(f"fixed-hardware run has no complete trial: {path}")
    reference = rows[0]
    invariant_fields = (
        "prefill_latency_ms",
        "prefill_system_energy_mj_ideal",
        "precision_profile",
        "MLEN",
        "VLEN",
        "BLEN",
        "INT_DATA_WIDTH",
        "softmax_row_lanes",
        "MATRIX_SRAM_TILES",
        "physical_chip_count",
        "dp_degree",
        "tp_degree",
        "ep_degree",
        "nvlink_port_count",
        "fp16_kv_handoff_latency_ms",
        "fp16_kv_handoff_bytes",
    )
    for row in rows[1:]:
        if any(row.get(field) != reference.get(field) for field in invariant_fields):
            raise ValueError(f"fixed-hardware run contains non-identical complete trials: {path}")
    return reference


def _find_measurement_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    tp: int,
    local_batch: int,
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if str(row["model"]) == model
        and int(row["tensor_parallel_size"]) == tp
        and int(row["local_batch_size"]) == local_batch
        and str(row.get("validation_status")) in {"pass", "warning"}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {model} TP{tp}/B{local_batch} measurement, found {len(matches)}"
        )
    return matches[0]


def _measurement_fidelity(rows: Sequence[Mapping[str, Any]], *, scope: str) -> str:
    statuses = sorted({str(row.get("validation_status")) for row in rows})
    suffix = "_with_measurement_warning" if "warning" in statuses else ""
    return f"{scope}{suffix}"


def _a100_topologies(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    input_tokens: int,
    fixed_batch_ttft_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = lambda tp, batch: _find_measurement_row(  # noqa: E731
        rows,
        model=model,
        tp=tp,
        local_batch=batch,
    )
    if model == "qwen3-32b":
        aggregate_specs = {
            "TP1xDP8_B1": _copies(row(1, 1), 8),
            "TP2xDP4_B2": _copies(row(2, 2), 4),
            "TP4xDP2_B4": _copies(row(4, 4), 2),
            "TP8xDP1_B8": [row(8, 8)],
        }
        decode_specs = {
            "maximum_output_tps": (
                "TP2xDP3_B3_3_2",
                [row(2, 3), row(2, 3), row(2, 2)],
            ),
            "maximum_output_tokens_per_j": ("TP4xDP1_B8", [row(4, 8)]),
        }
        topology_shapes = {
            "TP1xDP8_B1": (1, 1),
            "TP2xDP4_B2": (2, 2),
            "TP4xDP2_B4": (4, 4),
            "TP8xDP1_B8": (8, 8),
        }
    elif model == "qwen3-235b-a22b":
        aggregate_specs = {
            "TP4xDP4_B2": _copies(row(4, 2), 4),
            "TP8xDP2_B4": _copies(row(8, 4), 2),
        }
        decode_specs = {
            "maximum_output_tps": (
                "TP4xDP3_B3_3_2",
                [row(4, 3), row(4, 3), row(4, 2)],
            ),
            "maximum_output_tokens_per_j": (
                "TP4xDP2_B4",
                [row(4, 4), row(4, 4)],
            ),
        }
        topology_shapes = {
            "TP4xDP4_B2": (4, 2),
            "TP8xDP2_B4": (8, 4),
        }
    else:
        raise ValueError(f"unsupported model: {model}")

    aggregate = {
        topology: reconstruct_replica_phase(
            replicas,
            phase="complete",
            fidelity=_measurement_fidelity(
                replicas,
                scope=(
                    "measured_single_replica"
                    if len(replicas) == 1
                    else "measured_replica_extrapolation"
                ),
            ),
        )
        for topology, replicas in aggregate_specs.items()
    }
    _annotate_fixed_batch_ttft(
        aggregate,
        audit_rows=fixed_batch_ttft_rows,
        model=model,
        input_tokens=input_tokens,
        topology_shapes=topology_shapes,
    )
    decode = {}
    for objective, (topology, replicas) in decode_specs.items():
        decode[objective] = {
            "topology": topology,
            **reconstruct_replica_phase(
                replicas,
                phase="decode",
                fidelity=_measurement_fidelity(
                    replicas,
                    scope=(
                        "measured_single_replica"
                        if len(replicas) == 1
                        else "measured_replica_extrapolation"
                    ),
                ),
            ),
        }
    output_tokens = {int(item["output_tokens"]) for item in rows if str(item["model"]) == model}
    if len(output_tokens) != 1:
        raise ValueError(f"expected one output-token count for {model}, found {output_tokens}")
    output_tokens_per_request = output_tokens.pop()
    return {
        "aggregate": aggregate,
        "aggregate_endpoints": _aggregate_endpoints(
            _aggregate_candidates(
                aggregate,
                batch_size=BATCH_SIZE,
                output_tokens_per_request=output_tokens_per_request,
            )
        ),
        "decode": decode,
    }


def _fixed_hardware_mismatches(
    primary: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    integer_fields = (
        "MLEN",
        "VLEN",
        "BLEN",
        "INT_DATA_WIDTH",
        "softmax_row_lanes",
        "physical_chip_count",
        "dp_degree",
        "tp_degree",
        "ep_degree",
        "nvlink_port_count",
    )
    mismatches: dict[str, tuple[Any, Any]] = {}
    if str(primary["precision_profile"]) != str(holdout["precision_profile"]):
        mismatches["precision_profile"] = (
            primary["precision_profile"],
            holdout["precision_profile"],
        )
    for field in integer_fields:
        if int(primary[field]) != int(holdout[field]):
            mismatches[field] = (int(primary[field]), int(holdout[field]))
    if int(primary["matrix_sram_tiles"]) != int(holdout["MATRIX_SRAM_TILES"]):
        mismatches["matrix_sram_tiles"] = (
            int(primary["matrix_sram_tiles"]),
            int(holdout["MATRIX_SRAM_TILES"]),
        )
    if not math.isclose(
        float(primary["total_silicon_area_mm2"]),
        float(holdout["total_silicon_area_mm2"]),
        rel_tol=1e-10,
        abs_tol=1e-7,
    ):
        mismatches["total_silicon_area_mm2"] = (
            float(primary["total_silicon_area_mm2"]),
            float(holdout["total_silicon_area_mm2"]),
        )
    return mismatches


def _compose_holdout_system(
    row: Mapping[str, Any],
    *,
    primary: Mapping[str, Any],
    workload: Mapping[str, int],
    decode: Mapping[str, Any],
    prefill_static: Mapping[str, Any],
) -> dict[str, Any]:
    prefill_s = float(row["prefill_latency_ms"]) / 1000.0
    prefill_energy_j = float(row["prefill_system_energy_mj_ideal"]) / 1000.0
    handoff_s = float(row["fp16_kv_handoff_latency_ms"]) / 1000.0
    handoff_bytes = float(row["fp16_kv_handoff_bytes"])
    handoff_energy_j = handoff_bytes * 8.0 * INTERCONNECT_ENERGY_PJ_PER_BIT * 1e-12
    decode_s = float(decode["latency_s"])
    metrics = disaggregated_pipeline_metrics(
        batch_size=BATCH_SIZE,
        prefill_interval_s=prefill_s,
        kv_handoff_interval_s=handoff_s,
        decode_interval_s=decode_s,
        prefill_energy_j=prefill_energy_j,
        kv_handoff_energy_j=handoff_energy_j,
        decode_energy_j=float(decode["energy_j"]),
        prefill_idle_power_w=float(prefill_static["idle_power_w"]),
        decode_idle_power_w=float(decode["idle_power_w"]),
        e2e_latency_s=prefill_s + handoff_s + decode_s,
        input_tokens_per_request=int(workload["input_tokens"]),
        output_tokens_per_request=int(workload["output_tokens"]),
        total_tokens_per_request=int(workload["input_tokens"] + workload["output_tokens"]),
        request_tpot_s=decode.get("mean_request_tpot_s"),
        first_decode_step_s=float(decode["first_decode_step_s"]),
    )
    return {
        **metrics,
        "primary_selection_trial": int(primary["prefill_trial"]),
        "system_split": str(primary["system_split"]),
        "decode_topology": str(decode["topology"]),
        "decode_fidelity": str(decode["fidelity"]),
        "precision_profile": str(row["precision_profile"]),
        "MLEN": int(row["MLEN"]),
        "VLEN": int(row["VLEN"]),
        "BLEN": int(row["BLEN"]),
        "INT_DATA_WIDTH": int(row["INT_DATA_WIDTH"]),
        "softmax_row_lanes": int(row["softmax_row_lanes"]),
        "matrix_sram_tiles": int(row["MATRIX_SRAM_TILES"]),
        "physical_chip_count": int(row["physical_chip_count"]),
        "dp_degree": int(row["dp_degree"]),
        "tp_degree": int(row["tp_degree"]),
        "ep_degree": int(row["ep_degree"]),
        "nvlink_port_count": int(row["nvlink_port_count"]),
        "total_silicon_area_mm2": float(row["total_silicon_area_mm2"]),
        "prefill_energy_j": prefill_energy_j,
        "kv_handoff_energy_j": handoff_energy_j,
        "decode_energy_j": float(decode["energy_j"]),
        "prefill_idle_power_w": float(prefill_static["idle_power_w"]),
        "projected_static_batch_request_ttft_s": metrics[
            "plena_disaggregated_batch_admitted_ttft_s"
        ],
        "projected_mean_request_tpot_s": decode.get("mean_request_tpot_s"),
        "projected_p95_request_tpot_s": decode.get(
            "conservative_p95_request_tpot_s",
            decode.get("p95_request_tpot_s"),
        ),
        "fixed_hardware_validation": "exact_match_to_primary_selected_system",
    }


def _primary_row(primary: Mapping[str, Any]) -> dict[str, Any]:
    return dict(primary)


def _flat_row(
    *,
    model: str,
    workload_name: str,
    objective: str,
    system: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = _comparison(system, baseline)
    admitted_ttft = baseline.get("scheduler_admitted_ttft_exact_or_proxy_s")
    measured_ttft = baseline.get("median_request_visible_ttft_s")
    if admitted_ttft is not None:
        report_ttft = admitted_ttft
        report_ttft_source = "one_token_fixed_batch_measurement"
    else:
        report_ttft = measured_ttft
        report_ttft_source = "topology_matched_measured_median"
    return {
        "model": model,
        "workload": workload_name,
        "input_tokens": WORKLOADS[workload_name]["input_tokens"],
        "output_tokens": WORKLOADS[workload_name]["output_tokens"],
        "batch_size": BATCH_SIZE,
        "primary_selected_objective": objective,
        "system_split": system["system_split"],
        "decode_topology": system["decode_topology"],
        "a100_topology": baseline["topology"],
        "plena_static_ttft_s": system["projected_static_batch_request_ttft_s"],
        "a100_scheduler_admitted_ttft_s": baseline.get(
            "scheduler_admitted_ttft_exact_or_proxy_s"
        ),
        "a100_scheduler_admitted_ttft_source": baseline.get(
            "scheduler_admitted_ttft_source"
        ),
        "a100_scheduler_admitted_ttft_fidelity": baseline.get(
            "scheduler_admitted_ttft_fidelity"
        ),
        "a100_fixed_batch_ttft_point_id": baseline.get("fixed_batch_ttft_point_id"),
        "a100_request_visible_median_ttft_s": baseline.get(
            "median_request_visible_ttft_s"
        ),
        "a100_request_visible_p95_ttft_s": baseline.get("p95_request_visible_ttft_s"),
        "a100_ttft_s": report_ttft,
        "a100_ttft_source": report_ttft_source,
        "plena_e2e_s": system["e2e_latency_s"],
        "a100_e2e_s": baseline["latency_s"],
        "plena_output_tps": system["projected_pipeline_output_tokens_per_s"],
        "a100_output_tps": baseline["output_tokens_per_s"],
        "output_tps_ratio_x": comparison["output_tps_speedup_x"],
        "plena_output_tokens_per_j": system["projected_output_tokens_per_j"],
        "a100_output_tokens_per_j": baseline["output_tokens_per_j"],
        "output_tokens_per_j_ratio_x": comparison[
            "output_tokens_per_j_improvement_x"
        ],
        "e2e_ratio_x": comparison["e2e_latency_ratio_x"],
        "plena_bottleneck": system["bottleneck_stage"],
        "plena_prefill_s": system["prefill_interval_s"],
        "plena_handoff_s": system["kv_handoff_interval_s"],
        "a100_decode_s": system["decode_interval_s"],
        "plena_cross_stage_idle_energy_j": system["cross_stage_idle_energy_j"],
        "a100_measurement_fidelity": baseline["fidelity"],
        "decode_measurement_fidelity": system.get("decode_fidelity"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Fixed-Hardware Cross-Workload System Evaluation",
        "",
        "The two PLENA-A100 endpoint systems selected on `90k input / 8k output, batch 8`",
        "are re-evaluated without changing their numerical format, datapath geometry, SRAM,",
        "row width, chip count, DP/TP/EP mapping, link ports, or decode allocation. Only the",
        "workload token counts change. A100 baselines use the measured curve for each workload",
        "at the same total GPU budget.",
        "",
    ]
    for model, model_payload in payload["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                "| Workload | Primary-selected endpoint | PLENA TTFT (s) | A100 TTFT (s) | PLENA output TPS | A100 output TPS | TPS ratio | PLENA tokens/J | A100 tokens/J | Efficiency ratio | Bottleneck |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for workload_name in WORKLOADS:
            for objective in OBJECTIVES:
                row = model_payload["workloads"][workload_name][objective]["flat"]
                objective_label = "maximum TPS" if objective == "maximum_output_tps" else "maximum tokens/J"
                a100_ttft = row["a100_ttft_s"]
                a100_ttft_text = "unavailable" if a100_ttft is None else f"{float(a100_ttft):.3f}"
                lines.append(
                    f"| {workload_name} | {objective_label} | {row['plena_static_ttft_s']:.3f} | "
                    f"{a100_ttft_text} | "
                    f"{row['plena_output_tps']:.3f} | {row['a100_output_tps']:.3f} | "
                    f"{row['output_tps_ratio_x']:.3f}x | {row['plena_output_tokens_per_j']:.6f} | "
                    f"{row['a100_output_tokens_per_j']:.6f} | {row['output_tokens_per_j_ratio_x']:.3f}x | "
                    f"{row['plena_bottleneck']} |"
                )
        lines.extend(
            [
                "",
                "Short-workload energy rows retain the A100 NVML sampling warning recorded by the",
                "100 Hz measurement campaign.",
                "",
            ]
        )
    return "\n".join(lines)


def run_report(
    *,
    primary_results: Path,
    fixed_run_root: Path,
    measurement_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primary_payload = _read_json(primary_results)
    short_rows = _aggregate_rows(measurement_root / "short_energy_100hz_v2/aggregate.json")
    fixed_batch_ttft_rows = [
        dict(row)
        for row in _read_json(
            measurement_root / "fixed_batch_prefill_audit_v1/aggregate.json"
        )["rows"]
    ]
    holdout_rows = {
        "qwen3-32b": _aggregate_rows(measurement_root / "full_114k_32b/aggregate.json"),
        "qwen3-235b-a22b": _aggregate_rows(measurement_root / "full_114k_235b/aggregate.json"),
    }
    run_names = {
        ("qwen3-32b", "maximum_output_tps", "short-1400x200"): "qwen3_32b_tps_short_b8",
        ("qwen3-32b", "maximum_output_tps", "holdout-114000x5000"): "qwen3_32b_tps_holdout_b8",
        ("qwen3-32b", "maximum_output_tokens_per_j", "short-1400x200"): "qwen3_32b_eff_short_b8",
        ("qwen3-32b", "maximum_output_tokens_per_j", "holdout-114000x5000"): "qwen3_32b_eff_holdout_b8",
        ("qwen3-235b-a22b", "maximum_output_tps", "short-1400x200"): "qwen3_235b_tps_short_b8",
        ("qwen3-235b-a22b", "maximum_output_tps", "holdout-114000x5000"): "qwen3_235b_tps_holdout_b8",
        ("qwen3-235b-a22b", "maximum_output_tokens_per_j", "short-1400x200"): "qwen3_235b_eff_short_b8",
        ("qwen3-235b-a22b", "maximum_output_tokens_per_j", "holdout-114000x5000"): "qwen3_235b_eff_holdout_b8",
    }
    payload: dict[str, Any] = {
        "schema": "fixed_hardware_cross_workload_system_evaluation_v2",
        "selection_workload": "primary-90000x8000",
        "batch_size": BATCH_SIZE,
        "fixed_fields": [
            "precision_profile",
            "MLEN",
            "VLEN",
            "BLEN",
            "INT_DATA_WIDTH",
            "softmax_row_lanes",
            "matrix_sram_tiles",
            "physical_chip_count",
            "dp_degree",
            "tp_degree",
            "ep_degree",
            "nvlink_port_count",
            "decode_gpu_count",
            "decode_topology",
        ],
        "models": {},
    }
    flat_rows: list[dict[str, Any]] = []
    for model in ("qwen3-32b", "qwen3-235b-a22b"):
        primary_model = primary_payload["models"][model]
        primary_selected = primary_model["main"]["sensitivity_by_e2e_ratio"][PRIMARY_RATIO]
        model_payload: dict[str, Any] = {"workloads": {}}
        for workload_name, workload in WORKLOADS.items():
            workload_payload: dict[str, Any] = {}
            if workload_name == "primary-90000x8000":
                a100 = primary_model["aggregate_gpu"]["endpoints"]
                for objective in OBJECTIVES:
                    system = _primary_row(primary_selected[objective])
                    baseline = dict(a100[objective])
                    flat = _flat_row(
                        model=model,
                        workload_name=workload_name,
                        objective=objective,
                        system=system,
                        baseline=baseline,
                    )
                    workload_payload[objective] = {
                        "system": system,
                        "a100_baseline": baseline,
                        "comparison": _comparison(system, baseline),
                        "flat": flat,
                    }
                    flat_rows.append(flat)
            else:
                rows = short_rows if workload_name == "short-1400x200" else holdout_rows[model]
                a100 = _a100_topologies(
                    rows,
                    model=model,
                    input_tokens=int(workload["input_tokens"]),
                    fixed_batch_ttft_rows=fixed_batch_ttft_rows,
                )
                for objective in OBJECTIVES:
                    primary = primary_selected[objective]
                    run_dir = fixed_run_root / run_names[(model, objective, workload_name)]
                    fixed_row = _read_complete_trial(run_dir / "all_trials.csv.gz")
                    mismatches = _fixed_hardware_mismatches(primary, fixed_row)
                    if mismatches:
                        raise ValueError(
                            f"{model}/{objective}/{workload_name} changed fixed hardware: {mismatches}"
                        )
                    expected_decode_topology = str(primary["decode_topology"])
                    decode = a100["decode"][objective]
                    if str(decode["topology"]) != expected_decode_topology:
                        raise ValueError(
                            f"decode topology changed for {model}/{objective}: "
                            f"{decode['topology']} != {expected_decode_topology}"
                        )
                    static = PrefillIdlePowerEstimator.from_campaign(
                        run_dir,
                        repo_root=REPO_ROOT,
                    ).estimate(fixed_row)
                    system = _compose_holdout_system(
                        fixed_row,
                        primary=primary,
                        workload=workload,
                        decode=decode,
                        prefill_static=static,
                    )
                    baseline = dict(a100["aggregate_endpoints"][objective])
                    flat = _flat_row(
                        model=model,
                        workload_name=workload_name,
                        objective=objective,
                        system=system,
                        baseline=baseline,
                    )
                    workload_payload[objective] = {
                        "fixed_run_dir": str(run_dir),
                        "system": system,
                        "a100_baseline": baseline,
                        "comparison": _comparison(system, baseline),
                        "flat": flat,
                    }
                    flat_rows.append(flat)
            model_payload["workloads"][workload_name] = workload_payload
        payload["models"][model] = model_payload

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fixed_hardware_cross_workload_v2_results.json"
    csv_path = output_dir / "fixed_hardware_cross_workload_v2.csv"
    report_path = output_dir / "fixed_hardware_cross_workload_v2.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(csv_path, flat_rows)
    report_path.write_text(_render_report(payload) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=Path("Workspace/reports/serving/primary_90k8_system_search_v5_results.json"),
    )
    parser.add_argument(
        "--fixed-run-root",
        type=Path,
        default=Path("Workspace/reports/serving/fixed_hardware_holdout_runs"),
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
    payload = run_report(
        primary_results=args.primary_results,
        fixed_run_root=args.fixed_run_root,
        measurement_root=args.measurement_root,
        output_dir=args.output_dir,
    )
    summary = {
        model: {
            workload: {
                objective: data["flat"]["output_tps_ratio_x"]
                for objective, data in workload_data.items()
            }
            for workload, workload_data in model_data["workloads"].items()
        }
        for model, model_data in payload["models"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
