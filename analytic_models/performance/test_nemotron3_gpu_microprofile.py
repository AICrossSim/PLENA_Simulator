from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from .nemotron3_gpu_microprofile import (
    EXPECTED_CASES,
    EXPECTED_MODEL,
    EXPECTED_SHAPE,
    NCU_FILES,
    NSYS_FILES,
    STAGE_SUMMARY_FILE,
    STAGE_SUMMARY_SCHEMA,
    MicroprofileFormatError,
    build_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile_fixture(root: Path, *, state_dtype: str = "bfloat16") -> None:
    (root / "official_source_manifest.json").write_text(
        json.dumps(
            {
                "model_id": EXPECTED_MODEL,
                "revision": "test-revision",
                "files": {"modeling_nemotron_h.py": "test-sha256"},
            }
        )
    )
    latency_rows = []
    medians_ms = {
        ("prefill", 1, 128): 0.498,
        ("prefill", 1, 512): 0.752,
        ("prefill", 1, 2048): 1.213,
        ("prefill", 1, 8192): 4.046,
        ("decode", 1, 1): 0.23448,
        ("decode", 4, 1): 0.249,
        ("decode", 8, 1): 0.245264,
        ("decode", 16, 1): 0.253,
    }
    for phase, batch, sequence_length in sorted(EXPECTED_CASES):
        latency_rows.append(
            {
                "model": EXPECTED_MODEL,
                "model_revision": "test-revision",
                "phase": phase,
                "batch": batch,
                "sequence_length": sequence_length,
                "dtype": "bfloat16",
                "conv_state_dtype": "bfloat16",
                "ssm_state_dtype": state_dtype,
                "warmup_iterations": 20,
                "measured_iterations": 100,
                "median_ms": medians_ms[(phase, batch, sequence_length)],
                "p95_ms": medians_ms[(phase, batch, sequence_length)] * 1.1,
                "peak_vram_mib": 100,
                "output_all_finite": True,
                "state_storage_stable": True,
                "fast_path": True,
                **EXPECTED_SHAPE,
            }
        )
    _write_csv(root / "mamba_layer_latency.csv", latency_rows)

    ncu_values = {
        ("prefill", 1, 2048): (108_371_200, 307_712, 136_512),
        ("decode", 1, 1): (1_067_008, 0, 3_808),
        ("decode", 8, 1): (8_493_824, 0, 9_088),
    }
    nsys_values = {
        ("prefill", 1, 2048): 89_278,
        ("decode", 1, 1): 1_792,
        ("decode", 8, 1): 6_272,
    }
    for key, relative in NCU_FILES.items():
        read_bytes, write_bytes, duration_ns = ncu_values[key]
        _write_csv(
            root / relative,
            [
                {
                    "phase": key[0],
                    "batch": key[1],
                    "sequence_length": key[2],
                    "dram__bytes_op_read.sum": read_bytes,
                    "dram__bytes_op_read.sum unit": "byte",
                    "dram__bytes_op_write.sum": write_bytes,
                    "dram__bytes_op_write.sum unit": "byte",
                    "gpu__time_duration.sum": duration_ns,
                    "gpu__time_duration.sum unit": "ns",
                }
            ],
        )
    for key, relative in NSYS_FILES.items():
        instances = 5 if key == ("prefill", 1, 2048) else 1
        _write_csv(
            root / relative,
            [
                {
                    "Instances": instances,
                    "Total Time (ns)": nsys_values[key],
                    "Name": "kernel",
                }
            ],
        )

    stage_times = {
        "prefill_b1_s2048": (838.005, [509.754, 12.672, 0.864, 89.278, 16.479, 201.790], 7.168),
        "decode_b1": (62.944, [39.936, 1.216, 0.832, 1.792, 1.024, 15.712], 2.432),
        "decode_b8": (45.856, [16.608, 1.600, 0.768, 6.272, 1.056, 17.440], 2.112),
    }
    stage_names = (
        "mamba_in_projection",
        "mamba_conv1d",
        "mamba_dt_exp",
        "mamba_state_update_output_fused",
        "mamba_gate_group_rms_norm",
        "mamba_out_projection",
    )
    stage_cases = {}
    case_metadata = {
        "prefill_b1_s2048": ("prefill", 1, 2048, 16, 6),
        "decode_b1": ("decode", 1, 1, 9, 3),
        "decode_b8": ("decode", 8, 1, 9, 3),
    }
    for name, (full_time, times, unassigned) in stage_times.items():
        phase, batch, sequence_length, full_kernels, unassigned_kernels = case_metadata[name]
        stage_cases[name] = {
            "phase": phase,
            "batch": batch,
            "sequence_length": sequence_length,
            "full_mixer_gpu_time_us": full_time,
            "full_mixer_kernel_count": full_kernels,
            "stages": {
                stage_name: {
                    "total_gpu_time_us": time,
                    "kernel_count": 5 if stage_name == "mamba_state_update_output_fused" and phase == "prefill" else 1,
                    "percentage_of_full_mixer_time": 100 * time / full_time,
                }
                for stage_name, time in zip(stage_names, times, strict=True)
            },
            "unassigned_gpu_time_us": unassigned,
            "unassigned_kernel_count": unassigned_kernels,
        }
    stage_path = root / STAGE_SUMMARY_FILE
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        json.dumps(
            {
                "schema_version": STAGE_SUMMARY_SCHEMA,
                "model_rerun": False,
                "stage_order": list(stage_names),
                "cases": stage_cases,
            }
        )
    )


def test_microprofile_validates_state_traffic_and_scan_fraction(tmp_path: Path) -> None:
    _profile_fixture(tmp_path)
    report = build_report(tmp_path)

    assert report["persistent_state"]["total_mib_per_request"] == pytest.approx(24.078125)
    assert report["persistent_state"]["logical_decode_read_write_mib_per_token_per_request"] == pytest.approx(48.15625)
    decode_b1 = next(
        result
        for result in report["representative_scan_measurements"]
        if result["phase"] == "decode" and result["batch"] == 1
    )
    assert decode_b1["decode_recurrent_read_coverage"] == pytest.approx(1_067_008 / 1_048_576)
    assert decode_b1["scan_fraction_of_mixer"] == pytest.approx(1.792 / 234.48)
    assert decode_b1["ncu_dram_write_bytes"] == 0
    decode_stages = next(result for result in report["nvtx_stage_breakdown"]["cases"] if result["case"] == "decode_b1")
    assert decode_stages["combined_projection_fraction_of_gpu_kernel_time"] == pytest.approx((39.936 + 15.712) / 62.944)
    assert decode_stages["gpu_kernel_time_fraction_of_wall_median"] == pytest.approx(62.944 / 234.48)


def test_microprofile_rejects_a_state_dtype_mismatch(tmp_path: Path) -> None:
    _profile_fixture(tmp_path, state_dtype="float32")
    with pytest.raises(MicroprofileFormatError, match="BF16"):
        build_report(tmp_path)


def test_microprofile_rejects_stage_and_scan_disagreement(tmp_path: Path) -> None:
    _profile_fixture(tmp_path)
    source = tmp_path / STAGE_SUMMARY_FILE
    document = json.loads(source.read_text())
    document["cases"]["decode_b1"]["stages"]["mamba_state_update_output_fused"]["total_gpu_time_us"] = 2.0
    document["cases"]["decode_b1"]["stages"]["mamba_state_update_output_fused"]["percentage_of_full_mixer_time"] = (
        100 * 2.0 / 62.944
    )
    document["cases"]["decode_b1"]["unassigned_gpu_time_us"] = 2.224
    source.write_text(json.dumps(document))
    with pytest.raises(MicroprofileFormatError, match="stage export and scan summary differ"):
        build_report(tmp_path)
