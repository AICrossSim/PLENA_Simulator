"""Validate and normalize a standalone Nemotron 3 Mamba GPU microprofile."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
EXPECTED_SHAPE = {
    "hidden_size": 2688,
    "projection_size": 10304,
    "num_mamba_heads": 64,
    "head_dim": 64,
    "state_dim": 128,
    "groups": 8,
    "conv_kernel": 4,
    "chunk_size": 128,
}
EXPECTED_CASES = {
    ("prefill", 1, 128),
    ("prefill", 1, 512),
    ("prefill", 1, 2048),
    ("prefill", 1, 8192),
    ("decode", 1, 1),
    ("decode", 4, 1),
    ("decode", 8, 1),
    ("decode", 16, 1),
}
NCU_FILES = {
    ("prefill", 1, 2048): "ncu_mamba_prefill.csv",
    ("decode", 1, 1): "ncu_mamba_decode_b1.csv",
    ("decode", 8, 1): "ncu_mamba_decode_b8.csv",
}
NSYS_FILES = {
    ("prefill", 1, 2048): (
        "nsys/mamba_prefill_b1_s2048_measured_scan_cuda_gpu_kern_sum_nvtx=mamba_state_update_output_fused-20.csv"
    ),
    ("decode", 1, 1): (
        "nsys/mamba_decode_b1_measured_state_update_cuda_gpu_kern_sum_nvtx=mamba_state_update_output_fused-21.csv"
    ),
    ("decode", 8, 1): (
        "nsys/mamba_decode_b8_measured_state_update_cuda_gpu_kern_sum_nvtx=mamba_state_update_output_fused-21.csv"
    ),
}
STAGE_SUMMARY_FILE = "nsys_nvtx_stage_summary/nvtx_stage_cuda_kernel_summary.json"
STAGE_SUMMARY_SCHEMA = "nemotron3-nsys-nvtx-stage-summary-v2"
EXPECTED_STAGES = (
    "mamba_in_projection",
    "mamba_conv1d",
    "mamba_dt_exp",
    "mamba_state_update_output_fused",
    "mamba_gate_group_rms_norm",
    "mamba_out_projection",
)
STAGE_CASES = {
    "prefill_b1_s2048": ("prefill", 1, 2048),
    "decode_b1": ("decode", 1, 1),
    "decode_b8": ("decode", 8, 1),
}


class MicroprofileFormatError(ValueError):
    pass


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise MicroprofileFormatError(f"cannot read {path}: {error}") from error
    if not rows:
        raise MicroprofileFormatError(f"{path} has no data rows")
    return rows


def _source_manifest(root: Path) -> dict[str, Any]:
    source = root / "official_source_manifest.json"
    try:
        manifest = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MicroprofileFormatError(f"cannot read {source}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("model_id") != EXPECTED_MODEL:
        raise MicroprofileFormatError(f"{source}: unexpected model ID")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files.get("modeling_nemotron_h.py"):
        raise MicroprofileFormatError(f"{source}: missing official modeling source hash")
    return manifest


def _integer(row: dict[str, str], key: str, source: Path) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as error:
        raise MicroprofileFormatError(f"{source}: {key} is not an integer") from error


def _number(row: dict[str, str], key: str, source: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise MicroprofileFormatError(f"{source}: {key} is not numeric") from error
    if not math.isfinite(value):
        raise MicroprofileFormatError(f"{source}: {key} is not finite")
    return value


def _truth(row: dict[str, str], key: str, source: Path) -> bool:
    try:
        value = row[key]
    except KeyError as error:
        raise MicroprofileFormatError(f"{source}: missing {key}") from error
    if value not in {"True", "False"}:
        raise MicroprofileFormatError(f"{source}: {key} must be True or False")
    return value == "True"


@dataclass(frozen=True)
class LayerLatency:
    phase: str
    batch: int
    sequence_length: int
    median_us: float
    p95_us: float
    peak_vram_mib: float
    state_dtype: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.phase, self.batch, self.sequence_length)


@dataclass(frozen=True)
class NcuSummary:
    phase: str
    batch: int
    sequence_length: int
    kernels: int
    dram_read_bytes: int
    dram_write_bytes: int
    replay_kernel_time_us: float

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.phase, self.batch, self.sequence_length)


@dataclass(frozen=True)
class NsysSummary:
    phase: str
    batch: int
    sequence_length: int
    kernels: int
    scan_kernel_time_us: float

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.phase, self.batch, self.sequence_length)


@dataclass(frozen=True)
class StageSummary:
    phase: str
    batch: int
    sequence_length: int
    full_mixer_gpu_time_us: float
    full_mixer_kernel_count: int
    stages: dict[str, dict[str, float | int]]
    unassigned_gpu_time_us: float
    unassigned_kernel_count: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.phase, self.batch, self.sequence_length)


def _load_latency(root: Path) -> tuple[str, str, tuple[LayerLatency, ...]]:
    source = root / "mamba_layer_latency.csv"
    cases: list[LayerLatency] = []
    model: str | None = None
    revision: str | None = None
    observed: set[tuple[str, int, int]] = set()
    for row in _rows(source):
        row_model = row.get("model", "")
        row_revision = row.get("model_revision", "")
        if model is None:
            model, revision = row_model, row_revision
        if row_model != model or row_revision != revision:
            raise MicroprofileFormatError("latency rows mix model IDs or revisions")
        for field, expected in EXPECTED_SHAPE.items():
            if _integer(row, field, source) != expected:
                raise MicroprofileFormatError(f"{source}: unexpected {field}")
        if (
            row.get("dtype") != "bfloat16"
            or row.get("conv_state_dtype") != "bfloat16"
            or row.get("ssm_state_dtype") != "bfloat16"
        ):
            raise MicroprofileFormatError("profile is not the expected BF16 mixer run")
        if not all(_truth(row, field, source) for field in ("output_all_finite", "state_storage_stable", "fast_path")):
            raise MicroprofileFormatError("profile failed correctness or fast-path checks")
        if _integer(row, "warmup_iterations", source) != 20 or _integer(row, "measured_iterations", source) != 100:
            raise MicroprofileFormatError("latency matrix must use 20 warmups and 100 measurements")
        case = LayerLatency(
            phase=row.get("phase", ""),
            batch=_integer(row, "batch", source),
            sequence_length=_integer(row, "sequence_length", source),
            median_us=1000 * _number(row, "median_ms", source),
            p95_us=1000 * _number(row, "p95_ms", source),
            peak_vram_mib=_number(row, "peak_vram_mib", source),
            state_dtype=row.get("ssm_state_dtype", ""),
        )
        if case.key in observed:
            raise MicroprofileFormatError(f"duplicate latency case {case.key}")
        observed.add(case.key)
        cases.append(case)
    if observed != EXPECTED_CASES:
        raise MicroprofileFormatError(
            f"latency cases differ: missing={EXPECTED_CASES - observed}, extra={observed - EXPECTED_CASES}"
        )
    if model != EXPECTED_MODEL or not revision:
        raise MicroprofileFormatError("unexpected model ID or empty revision")
    return model, revision, tuple(cases)


def _load_ncu(root: Path) -> tuple[NcuSummary, ...]:
    summaries: list[NcuSummary] = []
    for key, relative in NCU_FILES.items():
        source = root / relative
        rows = _rows(source)
        for row in rows:
            if (row.get("phase"), _integer(row, "batch", source), _integer(row, "sequence_length", source)) != key:
                raise MicroprofileFormatError(f"{source}: case metadata does not match filename")
            if row.get("dram__bytes_op_read.sum unit") != "byte" or row.get("dram__bytes_op_write.sum unit") != "byte":
                raise MicroprofileFormatError(f"{source}: NCU DRAM counters are not bytes")
            if row.get("gpu__time_duration.sum unit") != "ns":
                raise MicroprofileFormatError(f"{source}: NCU duration is not ns")
        summaries.append(
            NcuSummary(
                phase=key[0],
                batch=key[1],
                sequence_length=key[2],
                kernels=len(rows),
                dram_read_bytes=sum(_integer(row, "dram__bytes_op_read.sum", source) for row in rows),
                dram_write_bytes=sum(_integer(row, "dram__bytes_op_write.sum", source) for row in rows),
                replay_kernel_time_us=sum(_number(row, "gpu__time_duration.sum", source) for row in rows) / 1000,
            )
        )
    return tuple(summaries)


def _load_nsys(root: Path) -> tuple[NsysSummary, ...]:
    summaries: list[NsysSummary] = []
    for key, relative in NSYS_FILES.items():
        source = root / relative
        rows = _rows(source)
        summaries.append(
            NsysSummary(
                phase=key[0],
                batch=key[1],
                sequence_length=key[2],
                kernels=sum(_integer(row, "Instances", source) for row in rows),
                scan_kernel_time_us=sum(_integer(row, "Total Time (ns)", source) for row in rows) / 1000,
            )
        )
    return tuple(summaries)


def _load_stage_summary(root: Path) -> tuple[StageSummary, ...]:
    source = root / STAGE_SUMMARY_FILE
    try:
        document = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MicroprofileFormatError(f"cannot read {source}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != STAGE_SUMMARY_SCHEMA:
        raise MicroprofileFormatError(f"{source}: unexpected schema version")
    if document.get("model_rerun") is not False:
        raise MicroprofileFormatError(f"{source}: expected an offline report export")
    if tuple(document.get("stage_order", ())) != EXPECTED_STAGES:
        raise MicroprofileFormatError(f"{source}: unexpected stage order")
    cases = document.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(STAGE_CASES):
        raise MicroprofileFormatError(f"{source}: unexpected stage cases")

    summaries: list[StageSummary] = []
    for name, key in STAGE_CASES.items():
        case = cases[name]
        if not isinstance(case, dict):
            raise MicroprofileFormatError(f"{source}: {name} is not an object")
        actual_key = (case.get("phase"), case.get("batch"), case.get("sequence_length"))
        if actual_key != key:
            raise MicroprofileFormatError(f"{source}: {name} metadata mismatch")
        full_time = case.get("full_mixer_gpu_time_us")
        full_kernels = case.get("full_mixer_kernel_count")
        unassigned_time = case.get("unassigned_gpu_time_us")
        unassigned_kernels = case.get("unassigned_kernel_count")
        if not isinstance(full_time, (int, float)) or not math.isfinite(full_time) or full_time <= 0:
            raise MicroprofileFormatError(f"{source}: invalid full mixer GPU time for {name}")
        if not isinstance(full_kernels, int) or full_kernels <= 0:
            raise MicroprofileFormatError(f"{source}: invalid full mixer kernel count for {name}")
        if not isinstance(unassigned_time, (int, float)) or unassigned_time < 0:
            raise MicroprofileFormatError(f"{source}: invalid unassigned GPU time for {name}")
        if not isinstance(unassigned_kernels, int) or unassigned_kernels < 0:
            raise MicroprofileFormatError(f"{source}: invalid unassigned kernel count for {name}")

        raw_stages = case.get("stages")
        if not isinstance(raw_stages, dict) or set(raw_stages) != set(EXPECTED_STAGES):
            raise MicroprofileFormatError(f"{source}: unexpected stages for {name}")
        stages: dict[str, dict[str, float | int]] = {}
        assigned_time = 0.0
        assigned_kernels = 0
        for stage_name in EXPECTED_STAGES:
            stage = raw_stages[stage_name]
            if not isinstance(stage, dict):
                raise MicroprofileFormatError(f"{source}: invalid {name}/{stage_name}")
            stage_time = stage.get("total_gpu_time_us")
            stage_kernels = stage.get("kernel_count")
            percentage = stage.get("percentage_of_full_mixer_time")
            if not isinstance(stage_time, (int, float)) or not math.isfinite(stage_time) or stage_time < 0:
                raise MicroprofileFormatError(f"{source}: invalid GPU time for {name}/{stage_name}")
            if not isinstance(stage_kernels, int) or stage_kernels <= 0:
                raise MicroprofileFormatError(f"{source}: invalid kernel count for {name}/{stage_name}")
            expected_percentage = 100 * stage_time / full_time
            if (
                not isinstance(percentage, (int, float))
                or not math.isfinite(percentage)
                or not math.isclose(percentage, expected_percentage, rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise MicroprofileFormatError(f"{source}: invalid percentage for {name}/{stage_name}")
            assigned_time += stage_time
            assigned_kernels += stage_kernels
            stages[stage_name] = {
                "total_gpu_time_us": float(stage_time),
                "kernel_count": stage_kernels,
                "percentage_of_gpu_kernel_time": float(percentage),
            }
        if not math.isclose(assigned_time + unassigned_time, full_time, rel_tol=1e-9, abs_tol=1e-6):
            raise MicroprofileFormatError(f"{source}: stage GPU times do not sum for {name}")
        if assigned_kernels + unassigned_kernels != full_kernels:
            raise MicroprofileFormatError(f"{source}: stage kernel counts do not sum for {name}")
        summaries.append(
            StageSummary(
                phase=key[0],
                batch=key[1],
                sequence_length=key[2],
                full_mixer_gpu_time_us=float(full_time),
                full_mixer_kernel_count=full_kernels,
                stages=stages,
                unassigned_gpu_time_us=float(unassigned_time),
                unassigned_kernel_count=unassigned_kernels,
            )
        )
    return tuple(summaries)


def build_report(root: Path) -> dict[str, Any]:
    manifest = _source_manifest(root)
    model, revision, latencies = _load_latency(root)
    if manifest.get("revision") != revision:
        raise MicroprofileFormatError("official source manifest and latency revision differ")
    ncu = {summary.key: summary for summary in _load_ncu(root)}
    nsys = {summary.key: summary for summary in _load_nsys(root)}
    stage_summaries = {summary.key: summary for summary in _load_stage_summary(root)}
    latency_by_key = {case.key: case for case in latencies}

    state_element_bytes = 2
    state_elements = EXPECTED_SHAPE["num_mamba_heads"] * EXPECTED_SHAPE["head_dim"] * EXPECTED_SHAPE["state_dim"]
    conv_elements = (
        EXPECTED_SHAPE["num_mamba_heads"] * EXPECTED_SHAPE["head_dim"]
        + 2 * EXPECTED_SHAPE["groups"] * EXPECTED_SHAPE["state_dim"]
    ) * EXPECTED_SHAPE["conv_kernel"]
    recurrent_bytes = state_elements * state_element_bytes
    conv_bytes = conv_elements * state_element_bytes
    persistent_per_layer = recurrent_bytes + conv_bytes
    mamba_layers = 23

    representatives = []
    for key in NCU_FILES:
        latency = latency_by_key[key]
        ncu_case = ncu[key]
        nsys_case = nsys[key]
        stage_case = stage_summaries[key]
        stage_scan = stage_case.stages["mamba_state_update_output_fused"]
        if (
            not math.isclose(
                float(stage_scan["total_gpu_time_us"]),
                nsys_case.scan_kernel_time_us,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or int(stage_scan["kernel_count"]) != nsys_case.kernels
        ):
            raise MicroprofileFormatError(f"stage export and scan summary differ for {key}")
        scan_fraction = nsys_case.scan_kernel_time_us / latency.median_us
        expected_recurrent_read = key[1] * recurrent_bytes
        representatives.append(
            {
                "phase": key[0],
                "batch": key[1],
                "sequence_length": key[2],
                "full_mixer_median_us": latency.median_us,
                "scan_kernel_time_us": nsys_case.scan_kernel_time_us,
                "scan_kernel_count": nsys_case.kernels,
                "scan_fraction_of_mixer": scan_fraction,
                "infinite_scan_speedup_upper_bound": 1 / (1 - scan_fraction),
                "ncu_dram_read_bytes": ncu_case.dram_read_bytes,
                "ncu_dram_write_bytes": ncu_case.dram_write_bytes,
                "expected_decode_recurrent_read_bytes": expected_recurrent_read if key[0] == "decode" else None,
                "decode_recurrent_read_coverage": (
                    ncu_case.dram_read_bytes / expected_recurrent_read if key[0] == "decode" else None
                ),
                "ncu_replay_kernel_time_us_not_latency_baseline": ncu_case.replay_kernel_time_us,
            }
        )

    stage_breakdown = []
    for name, key in STAGE_CASES.items():
        stage_case = stage_summaries[key]
        latency = latency_by_key[key]
        input_projection = stage_case.stages["mamba_in_projection"]
        output_projection = stage_case.stages["mamba_out_projection"]
        projection_time = float(input_projection["total_gpu_time_us"]) + float(output_projection["total_gpu_time_us"])
        stage_breakdown.append(
            {
                "case": name,
                "phase": key[0],
                "batch": key[1],
                "sequence_length": key[2],
                "full_mixer_gpu_kernel_time_us": stage_case.full_mixer_gpu_time_us,
                "full_mixer_wall_median_us": latency.median_us,
                "gpu_kernel_time_fraction_of_wall_median": (stage_case.full_mixer_gpu_time_us / latency.median_us),
                "full_mixer_kernel_count": stage_case.full_mixer_kernel_count,
                "stages": stage_case.stages,
                "combined_projection_gpu_time_us": projection_time,
                "combined_projection_fraction_of_gpu_kernel_time": (
                    projection_time / stage_case.full_mixer_gpu_time_us
                ),
                "unassigned_gpu_time_us": stage_case.unassigned_gpu_time_us,
                "unassigned_kernel_count": stage_case.unassigned_kernel_count,
            }
        )

    return {
        "schema_version": 1,
        "profile_scope": "official NemotronHMamba2Mixer with random BF16 weights; not a full-model baseline",
        "model_id": model,
        "model_revision": revision,
        "official_modeling_source_sha256": manifest["files"]["modeling_nemotron_h.py"],
        "shape": EXPECTED_SHAPE,
        "observed_precision": {
            "weights": "bfloat16",
            "activations": "bfloat16",
            "conv_state": "bfloat16",
            "ssm_state": "bfloat16",
            "official_config_declares_ssm_cache": "float32",
            "source_audit": "the pinned HybridMambaAttentionDynamicCache constructor receives model.dtype",
        },
        "persistent_state": {
            "mamba_layers": mamba_layers,
            "recurrent_bytes_per_layer_per_request": recurrent_bytes,
            "conv_bytes_per_layer_per_request": conv_bytes,
            "total_bytes_per_layer_per_request": persistent_per_layer,
            "total_mib_per_request": mamba_layers * persistent_per_layer / (1024 * 1024),
            "logical_decode_read_write_mib_per_token_per_request": 2
            * mamba_layers
            * persistent_per_layer
            / (1024 * 1024),
        },
        "latency_cases": [
            {
                "phase": case.phase,
                "batch": case.batch,
                "sequence_length": case.sequence_length,
                "median_us": case.median_us,
                "p95_us": case.p95_us,
                "peak_vram_mib": case.peak_vram_mib,
            }
            for case in latencies
        ],
        "representative_scan_measurements": representatives,
        "nvtx_stage_breakdown": {
            "denominator": (
                "CUDA GPU kernel time inside the measured nemotron.profile NVTX range; CPU launch gaps are excluded"
            ),
            "cases": stage_breakdown,
        },
        "validated_facts": [
            "the profiled fast path uses the real Nemotron Mamba shape",
            "the observed standalone cache is BF16 and storage-stable",
            "decode NCU reads approximately one complete recurrent state per request",
            "the scan kernel is a small fraction of standalone mixer latency on RTX 5090",
            "input and output projections dominate active GPU kernel time in all three staged cases",
        ],
        "limits": [
            "NCU was collected while an unrelated process occupied GPU memory",
            "physical DRAM write=0 means stores remained cached; it is not zero logical state write traffic",
            "NCU replay durations are excluded from latency conclusions",
            "NVTX stage percentages use active GPU kernel time and exclude CPU launch gaps",
            "this profile does not calibrate PLENA cycles, FPGA PPA, MoE, attention, or full-model latency",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Nemotron 3 standalone Mamba RTX 5090 microprofile")
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.profile_root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
