import json

import pytest

from analytic_models.performance.nemotron3_profile import GpuProfile, ProfileFormatError, load_gpu_profile


def _profile_document() -> dict:
    return {
        "schema_version": 1,
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "model_revision": "test-revision",
        "environment": {
            "gpu_model": "test-gpu",
            "gpu_count": 1,
            "driver_version": "test-driver",
            "cuda_version": "test-cuda",
            "framework_version": "test-framework",
            "dtype": "bf16",
            "state_dtype": "fp32",
        },
        "scenarios": [
            {
                "phase": "decode",
                "batch_size": 1,
                "input_sequence_length": 2048,
                "context_length": 2048,
                "generated_tokens": 16,
                "warmup_iterations": 3,
                "measured_iterations": 10,
                "ttft_us": 1000.0,
                "mean_token_latency_us": 100.0,
                "kernels": [
                    {
                        "canonical_stage": "mamba_state_update_output_fused",
                        "layer_type": "mamba",
                        "kernel_name": "fused_kernel",
                        "calls": 368,
                        "total_time_us": 50.0,
                        "dram_read_bytes": 1000,
                        "dram_write_bytes": 500,
                    },
                    {
                        "canonical_stage": "mamba_state_update_output_fused",
                        "layer_type": "mamba",
                        "kernel_name": "fused_kernel_tail",
                        "calls": 368,
                        "total_time_us": 25.0,
                        "dram_read_bytes": 250,
                        "dram_write_bytes": 125,
                    },
                ],
            }
        ],
    }


def test_profile_contract_aggregates_kernels_by_canonical_stage() -> None:
    profile = GpuProfile.from_dict(_profile_document())
    aggregate = profile.scenarios[0].aggregate_stages()["mamba_state_update_output_fused"]

    assert aggregate == {
        "calls": 736,
        "total_time_us": 75.0,
        "dram_read_bytes": 1250,
        "dram_write_bytes": 625,
    }


def test_profile_contract_rejects_unmapped_stage() -> None:
    document = _profile_document()
    document["scenarios"][0]["kernels"][0]["canonical_stage"] = "unknown_fused_thing"

    with pytest.raises(ProfileFormatError, match="not recognized"):
        GpuProfile.from_dict(document)


def test_profile_contract_rejects_zero_measured_iterations() -> None:
    document = _profile_document()
    document["scenarios"][0]["measured_iterations"] = 0

    with pytest.raises(ProfileFormatError, match="measured_iterations"):
        GpuProfile.from_dict(document)


def test_profile_loader_reads_json(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile_document()))

    assert load_gpu_profile(path).environment.gpu_model == "test-gpu"
