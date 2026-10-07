"""Validate the checked-in GPU evidence used by static Mamba/KDA analysis.

These measurements pin real workload shapes, numerical formats, routing skew,
physical traffic, and GPU baselines.  They do not turn GPU time into PLENA
cycles and do not establish an RTL frequency, area, power, or speedup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .b200_formal_campaign import build_report as build_b200_report


PROFILE_DIR = Path(__file__).with_name("profiles")
SOURCE_MANIFEST = PROFILE_DIR / "gpu_sources.json"
RTX5090_DIR = PROFILE_DIR / "rtx5090_nemotron_mamba"
SUPPLEMENTAL_DIR = PROFILE_DIR / "b200_supplemental"
NEMOTRON_REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"
KIMI_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


class GpuEvidenceError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GpuEvidenceError(f"cannot read {path}: {error}") from error


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as source:
            return list(csv.DictReader(source))
    except OSError as error:
        raise GpuEvidenceError(f"cannot read {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise GpuEvidenceError(f"{label}: {actual} != {expected}")


def _true(value: str) -> bool:
    return value.strip().lower() == "true"


def _validate_manifest() -> dict[str, Any]:
    manifest = _load_json(SOURCE_MANIFEST)
    if manifest.get("schema_version") != 1:
        raise GpuEvidenceError("unexpected GPU source-manifest schema")
    expected_archives = {
        "rtx5090_mamba",
        "b200_formal",
        "b200_kda_stage2",
        "b200_supplemental",
    }
    if set(manifest.get("archives", {})) != expected_archives:
        raise GpuEvidenceError("GPU source manifest does not cover all four archives")
    for relative, metadata in manifest.get("imported_files", {}).items():
        path = PROFILE_DIR / relative
        if not path.is_file():
            raise GpuEvidenceError(f"missing imported GPU evidence: {relative}")
        if _sha256(path) != metadata.get("sha256"):
            raise GpuEvidenceError(f"GPU evidence hash mismatch: {relative}")
    return manifest


def _validate_5090() -> dict[str, Any]:
    metadata = _load_json(RTX5090_DIR / "mamba_layer_latency_meta.json")
    if metadata.get("model_revision") != NEMOTRON_REVISION:
        raise GpuEvidenceError("RTX 5090 Nemotron revision is not pinned")
    expected_shape = {
        "hidden_size": 2688,
        "projection_size": 10304,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "chunk_size": 128,
    }
    if metadata.get("shape") != expected_shape:
        raise GpuEvidenceError("RTX 5090 Mamba shape is not Nemotron 3 Nano 30B")

    rows = _load_csv(RTX5090_DIR / "mamba_layer_latency.csv")
    expected_cases = {
        ("prefill", 1, 128),
        ("prefill", 1, 512),
        ("prefill", 1, 2048),
        ("prefill", 1, 8192),
        ("decode", 1, 1),
        ("decode", 4, 1),
        ("decode", 8, 1),
        ("decode", 16, 1),
    }
    observed_cases = {(row["phase"], int(row["batch"]), int(row["sequence_length"])) for row in rows}
    if observed_cases != expected_cases:
        raise GpuEvidenceError("RTX 5090 latency matrix is incomplete")
    normalized_latency = []
    for row in rows:
        if row["model_revision"] != NEMOTRON_REVISION:
            raise GpuEvidenceError("RTX 5090 latency rows mix model revisions")
        if int(row["warmup_iterations"]) != 20 or int(row["measured_iterations"]) != 100:
            raise GpuEvidenceError("RTX 5090 latency protocol changed")
        if not (_true(row["output_all_finite"]) and _true(row["state_storage_stable"])):
            raise GpuEvidenceError("RTX 5090 latency correctness checks failed")
        normalized_latency.append(
            {
                "phase": row["phase"],
                "batch": int(row["batch"]),
                "sequence_length": int(row["sequence_length"]),
                "median_ms": float(row["median_ms"]),
                "p95_ms": float(row["p95_ms"]),
                "peak_vram_mib": float(row["peak_vram_mib"]),
                "ssm_state_dtype": row["ssm_state_dtype"],
            }
        )

    stages = _load_json(RTX5090_DIR / "nvtx_stage_cuda_kernel_summary.json")
    if stages.get("model_rerun") is not False:
        raise GpuEvidenceError("RTX 5090 NSYS summary must be an offline export")
    expected_stage_names = set(stages.get("stage_order", []))
    if expected_stage_names != {
        "mamba_in_projection",
        "mamba_conv1d",
        "mamba_dt_exp",
        "mamba_state_update_output_fused",
        "mamba_gate_group_rms_norm",
        "mamba_out_projection",
    }:
        raise GpuEvidenceError("RTX 5090 NSYS stage contract changed")
    if set(stages.get("cases", {})) != {"prefill_b1_s2048", "decode_b1", "decode_b8"}:
        raise GpuEvidenceError("RTX 5090 NSYS stage cases are incomplete")

    normalized_stages = {}
    for case_name, case in stages["cases"].items():
        if set(case["stages"]) != expected_stage_names:
            raise GpuEvidenceError(f"{case_name}: incomplete stage attribution")
        attributed = sum(stage["total_gpu_time_us"] for stage in case["stages"].values())
        _close(
            attributed + case["unassigned_gpu_time_us"],
            case["full_mixer_gpu_time_us"],
            f"{case_name} attributed time",
            tolerance=1e-6,
        )
        projections = sum(
            case["stages"][name]["total_gpu_time_us"] for name in ("mamba_in_projection", "mamba_out_projection")
        )
        state = case["stages"]["mamba_state_update_output_fused"]["total_gpu_time_us"]
        normalized_stages[case_name] = {
            "full_mixer_gpu_time_us": case["full_mixer_gpu_time_us"],
            "projection_time_fraction": projections / case["full_mixer_gpu_time_us"],
            "state_core_time_fraction": state / case["full_mixer_gpu_time_us"],
            "stages": case["stages"],
        }

    ncu_rows = []
    for name in ("ncu_mamba_prefill.csv", "ncu_mamba_decode_b1.csv", "ncu_mamba_decode_b8.csv"):
        ncu_rows.extend(_load_csv(RTX5090_DIR / name))
    if len(ncu_rows) != 7:
        raise GpuEvidenceError("RTX 5090 NCU kernel set is incomplete")
    if any(row["dram__bytes_op_read.sum unit"] != "byte" for row in ncu_rows):
        raise GpuEvidenceError("RTX 5090 NCU DRAM metric is not in bytes")

    return {
        "scope": metadata["scope"],
        "weights": metadata["weights"],
        "shape": expected_shape,
        "latency": sorted(
            normalized_latency,
            key=lambda row: (row["phase"], row["batch"], row["sequence_length"]),
        ),
        "nsys_stages": normalized_stages,
        "ncu_kernel_rows": len(ncu_rows),
        "ncu_scope": (
            "Concurrency-qualified counters; retained for direction only. "
            "Clean CUDA-event latency and NSYS runs predate the other process."
        ),
    }


def _validate_supplemental(manifest: dict[str, Any]) -> dict[str, Any]:
    validation = _load_json(SUPPLEMENTAL_DIR / "source_validation.json")
    if validation.get("status") != "COMPLETE" or not all(validation.get("validations", {}).values()):
        raise GpuEvidenceError("B200 supplemental campaign is incomplete")

    latency = _load_csv(SUPPLEMENTAL_DIR / "kimi_component_latency.csv")
    if len(latency) != 8:
        raise GpuEvidenceError("B200 Kimi component latency matrix is incomplete")
    expected_component_cases = {
        (component, case)
        for component in ("mla", "moe")
        for case in ("decode_b1_s2048", "decode_b8_s2048", "prefill_s128", "prefill_s2048")
    }
    if {(row["component"], row["case"]) for row in latency} != expected_component_cases:
        raise GpuEvidenceError("B200 Kimi component case set changed")
    for row in latency:
        if int(row["warmup"]) != 20 or int(row["measurements"]) != 100:
            raise GpuEvidenceError("B200 Kimi component latency protocol changed")

    parity = _load_json(SUPPLEMENTAL_DIR / "kimi_component_parity.json").get("records", [])
    if len(parity) != 4:
        raise GpuEvidenceError("B200 Kimi component parity coverage is incomplete")
    for record in parity:
        if any(record[field] != 0 for field in ("max_abs", "relative_l2", "nan_count", "inf_count")):
            raise GpuEvidenceError("B200 Kimi component parity failed")

    ncu = _load_csv(SUPPLEMENTAL_DIR / "kimi_component_ncu.csv")
    nsys = _load_csv(SUPPLEMENTAL_DIR / "kimi_component_nsys.csv")
    source_counts = manifest["supplemental_source"]["counts"]
    if len(ncu) != source_counts["ncu_stage_rows"] or len(nsys) != 8 * 7:
        raise GpuEvidenceError("B200 Kimi stage summaries are incomplete")

    precision = _load_csv(SUPPLEMENTAL_DIR / "mamba_precision.csv")
    if len(precision) != 25:
        raise GpuEvidenceError("Mamba precision aggregate must cover 25 variants")
    if any(int(row["nan_count"]) or int(row["inf_count"]) for row in precision):
        raise GpuEvidenceError("Mamba precision sweep contains NaN or Inf")
    selected_precision = {}
    for row in precision:
        if row["case"] == "prefill_s32768" and row["variant"] in {
            "fp32",
            "bf16_chunk128",
            "fp16_chunk128",
            "mx8_chunk128",
        }:
            selected_precision[row["variant"]] = {
                "output_relative_l2_mean": float(row["output_relative_l2_mean"]),
                "state_relative_l2_mean": float(row["state_relative_l2_mean"]),
                "total_bytes": int(row["total_bytes"]),
                "hbm_reduction_vs_fp32": float(row["hbm_read_reduction_vs_fp32"]),
            }
    if set(selected_precision) != {"fp32", "bf16_chunk128", "fp16_chunk128", "mx8_chunk128"}:
        raise GpuEvidenceError("long-sequence Mamba precision candidates are incomplete")
    if selected_precision["fp32"]["total_bytes"] != 2_097_152:
        raise GpuEvidenceError("unexpected FP32 Mamba state size")
    if selected_precision["bf16_chunk128"]["total_bytes"] != 1_048_576:
        raise GpuEvidenceError("unexpected BF16 Mamba state size")
    if selected_precision["mx8_chunk128"]["total_bytes"] != 528_384:
        raise GpuEvidenceError("unexpected MX8 Mamba state size")

    traffic = _load_csv(SUPPLEMENTAL_DIR / "nemotron_prefill_mamba_stages.csv")
    if sum(int(row["kernel_calls"]) for row in traffic) != 563:
        raise GpuEvidenceError("Nemotron prefill Mamba stage attribution is incomplete")

    return {
        "source": manifest["supplemental_source"],
        "kimi_component_latency": [
            {
                "component": row["component"],
                "case": row["case"],
                "median_ms": float(row["median_ms"]),
                "p95_ms": float(row["p95_ms"]),
                "peak_memory_bytes": int(row["peak_memory_bytes"]),
            }
            for row in latency
        ],
        "kimi_component_parity": "four small/real-shape MLA and LatentMoE cases are exact",
        "kimi_ncu_stage_rows": len(ncu),
        "kimi_nsys_stage_rows": len(nsys),
        "mamba_precision_s32768": selected_precision,
        "nemotron_prefill_mamba_stages": {
            row["stage"]: {
                "kernel_calls": int(row["kernel_calls"]),
                "duration_ns": int(row["duration_ns"]),
                "dram_read_bytes": int(row["dram_read_bytes"]),
            }
            for row in traffic
        },
    }


def build_report() -> dict[str, Any]:
    manifest = _validate_manifest()
    b200 = build_b200_report()
    rtx5090 = _validate_5090()
    supplemental = _validate_supplemental(manifest)
    if b200["nemotron"]["revision"] != NEMOTRON_REVISION:
        raise GpuEvidenceError("B200 and RTX 5090 Nemotron revisions differ")
    if manifest["supplemental_source"]["revisions"]["kimi_hf"] != KIMI_REVISION:
        raise GpuEvidenceError("Kimi component revision differs from the formal campaign")
    return {
        "schema_version": 1,
        "sources": manifest["archives"],
        "b200_formal": b200,
        "rtx5090_mamba": rtx5090,
        "b200_supplemental": supplemental,
        "design_implications": [
            "Official KDA uses eight independent contiguous projection tensors, not packed QKV.",
            "KDA full-layer DSE must include Matrix and weight traffic; the recurrent core is 5-15% of measured GPU kernel time.",
            "Full-checkpoint Nemotron system DSE must include MoE routing skew and MoE traffic, not Mamba alone.",
            "FP16/BF16 halve Mamba state bytes with much lower long-sequence state error than the measured MX8 candidate.",
        ],
        "evidence_boundaries": manifest["evidence_boundaries"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(build_report(), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
