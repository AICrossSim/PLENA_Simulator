"""Strict aggregation and repeatability checks for RunPod benchmark points."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from .io import read_json, write_json_atomic
from .manifest import BenchmarkManifest


LATENCY_FIELDS = (
    "prefill_latency_s",
    "first_decode_iteration_latency_s",
    "measured_generation_latency_s",
    "full_request_latency_s",
    "imported_kv_decode_proxy_latency_s",
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _coefficient_of_variation_pct(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return 0.0 if mean == 0 else statistics.pstdev(values) / mean * 100.0


def aggregate_point(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete":
        raise ValueError(f"cannot aggregate incomplete point {summary.get('point_id')}")
    repetitions = summary.get("repetitions", [])
    expected = int(summary["point"]["repetitions"])
    if len(repetitions) != expected:
        raise ValueError(f"{summary['point_id']}: expected {expected} repetitions, found {len(repetitions)}")
    row: dict[str, Any] = {
        "point_id": summary["point_id"],
        "model": summary["point"]["model_name"],
        "workload": summary["point"]["workload_name"],
        "tensor_parallel_size": summary["point"]["tensor_parallel_size"],
        "local_batch_size": summary["point"]["local_batch_size"],
        "input_tokens": summary["point"]["input_tokens"],
        "output_tokens": summary["point"]["output_tokens"],
        "quantization": summary["quantization"],
        "revision": summary["resolved_revision"],
        "decode_fidelity": summary["decode_fidelity"],
    }
    warnings: list[str] = []
    reference_output_hashes = repetitions[0].get("output_token_hashes")
    if reference_output_hashes is None:
        raise ValueError(f"{summary['point_id']}: output token hashes are missing")
    if len(reference_output_hashes) != int(summary["point"]["local_batch_size"]):
        raise ValueError(f"{summary['point_id']}: output token hash count does not match local batch")
    if any(repetition.get("output_token_hashes") != reference_output_hashes for repetition in repetitions[1:]):
        raise ValueError(f"{summary['point_id']}: greedy outputs differ across repetitions")
    max_cv = 0.0
    for field in LATENCY_FIELDS:
        values = [float(repetition["phase"][field]) for repetition in repetitions]
        row[f"median_{field}"] = _median(values)
        cv = _coefficient_of_variation_pct(values)
        row[f"cv_{field}_pct"] = cv
        max_cv = max(max_cv, cv)
    energy_values: list[float] = []
    decode_energy_values: list[float] = []
    dynamic_energy_values: list[float] = []
    idle_power_values: list[float] = []
    sampling_errors: list[float] = []
    nvlink_tx_values: list[float] = []
    nvlink_rx_values: list[float] = []
    for repetition in repetitions:
        expected_global_tokens = int(summary["point"]["output_tokens"]) * int(
            summary["point"]["local_batch_size"]
        )
        if int(repetition["phase"].get("global_output_tokens", expected_global_tokens)) != expected_global_tokens:
            raise ValueError(f"{summary['point_id']}: global output token count mismatch")
        if float(repetition["phase"].get("stage_reconstruction_error_pct", 0.0)) > 1.0:
            warnings.append("phase_latency_reconstruction_error_exceeds_1pct")
        complete = repetition["power"]["complete_request"]
        direct = complete.get("nvml_counter_energy_mj")
        if direct is None:
            direct = complete["sampled_energy_mj"]
            warnings.append("nvml_total_energy_counter_unavailable")
        energy_values.append(float(direct))
        dynamic_energy_values.append(float(complete["idle_subtracted_dynamic_energy_mj"]))
        idle_power_values.append(float(repetition["power"]["idle_baseline"]["average_total_board_power_w"]))
        decode = repetition["power"].get("imported_kv_decode_proxy", {})
        decode_direct = decode.get("nvml_counter_energy_mj")
        if decode_direct is None:
            decode_direct = decode.get("sampled_energy_mj")
        if decode_direct is None:
            raise ValueError(f"{summary['point_id']}: imported-KV proxy energy is unavailable")
        decode_energy_values.append(float(decode_direct))
        for phase_name in ("prefill", "first_decode_iteration", "measured_generation", "complete_request"):
            error = repetition["power"][phase_name].get("counter_sampling_error_pct")
            if error is not None:
                sampling_errors.append(float(error))
        energy_reconstruction_error = repetition["power"].get("counter_energy_reconstruction_error_pct")
        if energy_reconstruction_error is not None and float(energy_reconstruction_error) > 0.1:
            warnings.append("phase_energy_reconstruction_error_exceeds_0.1pct")
        nvlink = repetition.get("nvlink", {})
        if nvlink.get("status") == "available":
            nvlink_tx_values.append(float(nvlink["aggregate_gpu_tx_bytes"]))
            nvlink_rx_values.append(float(nvlink["aggregate_gpu_rx_bytes"]))
    energy_cv = _coefficient_of_variation_pct(energy_values)
    max_cv = max(max_cv, energy_cv)
    row.update(
        {
            "median_complete_energy_mj": _median(energy_values),
            "median_complete_idle_subtracted_dynamic_energy_mj": _median(dynamic_energy_values),
            "median_idle_total_board_power_w": _median(idle_power_values),
            "median_imported_kv_decode_proxy_energy_mj": _median(decode_energy_values),
            "cv_complete_energy_pct": energy_cv,
            "max_latency_or_energy_cv_pct": max_cv,
            "max_counter_sampling_error_pct": max(sampling_errors, default=math.nan),
            "median_nvlink_tx_bytes": _median(nvlink_tx_values) if nvlink_tx_values else math.nan,
            "median_nvlink_rx_bytes": _median(nvlink_rx_values) if nvlink_rx_values else math.nan,
        }
    )
    if max_cv > 5.0:
        warnings.append("repeat_cv_exceeds_5pct")
    if sampling_errors and max(sampling_errors) > 3.0:
        warnings.append("nvml_counter_sampling_error_exceeds_3pct")
    if int(summary["point"]["tensor_parallel_size"]) > 1 and not nvlink_tx_values:
        warnings.append("nvlink_traffic_measurement_unavailable")
    if any(repetition["phase"].get("multi_token_step_observed") for repetition in repetitions):
        raise ValueError(f"{summary['point_id']}: multi-token engine step invalidates phase timing")
    row["validation_status"] = "pass" if not warnings else "warning"
    row["warnings"] = ";".join(sorted(set(warnings)))
    return row


def aggregate_campaign(
    *,
    manifest: BenchmarkManifest,
    output_root: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []
    capacity_infeasible: list[str] = []
    selected_points = manifest.formal_points
    campaign_path = output_root / "campaign.json"
    if campaign_path.is_file():
        selected_ids = set(read_json(campaign_path).get("selected_point_ids", ()))
        selected_points = tuple(point for point in manifest.formal_points if point.point_id in selected_ids)
    for point in selected_points:
        path = output_root / "points" / point.point_id / "summary.json"
        if not path.exists():
            missing.append(point.point_id)
            continue
        summary = read_json(path)
        if summary.get("status") == "capacity_infeasible":
            capacity_infeasible.append(point.point_id)
            continue
        if summary.get("status") != "complete":
            failed.append(point.point_id)
            continue
        rows.append(aggregate_point(summary))
    if (missing or failed) and not allow_missing:
        raise RuntimeError(f"campaign is incomplete: missing={missing}, failed={failed}")
    csv_path = output_root / "aggregate.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema_version": "runpod-serving-aggregate-v1",
        "campaign": manifest.campaign,
        "manifest_hash": manifest.fingerprint,
        "complete_points": len(rows),
        "expected_points": len(selected_points),
        "missing_points": missing,
        "failed_points": failed,
        "capacity_infeasible_points": capacity_infeasible,
        "warning_points": [row["point_id"] for row in rows if row["validation_status"] != "pass"],
        "aggregate_csv": str(csv_path),
        "rows": rows,
    }
    write_json_atomic(output_root / "aggregate.json", report)
    return report
