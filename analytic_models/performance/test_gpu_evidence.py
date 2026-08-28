from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest

from .gpu_evidence import build_report
from .gpu_evidence_import import import_evidence
from .hybrid_model import layer_plan


def test_gpu_evidence_pins_real_shapes_bottlenecks_and_precision() -> None:
    report = build_report()

    formal = report["b200_formal"]
    assert formal["kda"]["projection_storage"] == ("eight_independent_contiguous_tensors_not_packed_qkv")
    formal_kda = {case["case"]: case for case in formal["kda"]["cases"]}
    assert formal_kda["decode_b1"]["matrix_path_time_fraction"] == pytest.approx(0.7445, abs=5e-4)
    assert formal_kda["decode_b1"]["state_core_time_fraction"] == pytest.approx(0.0502, abs=5e-4)
    assert formal_kda["prefill_b1_s2048"]["matrix_path_time_fraction"] == pytest.approx(0.7433, abs=5e-4)

    nemotron = formal["nemotron"]
    assert nemotron["latency"]["decode_s2048_128"]["itl_median_ms"] == pytest.approx(4.047566)
    assert nemotron["routing"]["decode_max_hotspot_count"] == 2139
    assert nemotron["moe_to_mamba_prefill_dram_read_ratio"] == pytest.approx(8.919, abs=1e-3)

    rtx5090 = report["rtx5090_mamba"]
    latency = {(row["phase"], row["batch"], row["sequence_length"]): row for row in rtx5090["latency"]}
    assert latency[("prefill", 1, 2048)]["median_ms"] == pytest.approx(1.2127519845962524)
    assert latency[("decode", 1, 1)]["median_ms"] == pytest.approx(0.23448000103235245)
    assert rtx5090["nsys_stages"]["decode_b1"]["state_core_time_fraction"] == pytest.approx(1.792 / 62.944)
    assert "Concurrency-qualified" in rtx5090["ncu_scope"]

    precision = report["b200_supplemental"]["mamba_precision_s32768"]
    assert precision["bf16_chunk128"]["total_bytes"] == 1_048_576
    assert precision["bf16_chunk128"]["state_relative_l2_mean"] == pytest.approx(0.001667918069863536)
    assert precision["mx8_chunk128"]["total_bytes"] == 528_384
    assert precision["mx8_chunk128"]["state_relative_l2_mean"] == pytest.approx(0.026866347683043326)
    assert report["evidence_boundaries"]["plena_cycles"] == ("not calibrated by these files")


def test_gpu_evidence_has_no_collection_machine_paths() -> None:
    profile_root = Path(__file__).with_name("profiles")
    forbidden = (b"/home/mcl123", b"/dev/shm", b"/tmp/")
    for path in profile_root.rglob("*"):
        if path.is_file():
            payload = path.read_bytes()
            assert not any(fragment in payload for fragment in forbidden), path


def test_profiled_nemotron_config_matches_the_full_checkpoint_campaign() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "PLENA_Compiler" / "doc" / "Model_Lib" / "nemotron-3-nano-30b-a3b.json"
    config = json.loads(config_path.read_text())
    plan = layer_plan(config)
    report = build_report()["b200_formal"]["nemotron"]

    assert len(plan) == 52
    assert Counter(layer.mixer for layer in plan) == {
        "mamba": 23,
        "attention": 6,
        None: 23,
    }
    assert Counter(layer.ffn for layer in plan) == {"moe": 23, None: 29}
    assert config["hidden_size"] == 2688
    assert config["head_dim"] == 128
    assert config["mamba_ssm_cache_dtype"] == "float32"
    assert report["model"].endswith("30B-A3B-NVFP4")


RAW_GPU_ARTIFACTS = Path(os.environ["PLENA_GPU_ARTIFACT_ROOT"]) if os.environ.get("PLENA_GPU_ARTIFACT_ROOT") else None


@pytest.mark.skipif(
    RAW_GPU_ARTIFACTS is None or not RAW_GPU_ARTIFACTS.is_dir(),
    reason="raw GPU archives are not part of a fresh clone",
)
def test_local_archives_reproduce_every_imported_file(tmp_path: Path) -> None:
    source_profiles = Path(__file__).with_name("profiles")
    for name in (
        "b200_kda_nemotron_campaign_complete.json",
        "nemotron3_decode_routing_trace.json",
    ):
        shutil.copy2(source_profiles / name, tmp_path / name)

    assert RAW_GPU_ARTIFACTS is not None
    rebuilt = import_evidence(RAW_GPU_ARTIFACTS, tmp_path)
    pinned = build_report()["sources"]
    assert rebuilt["archives"] == pinned
    for relative in rebuilt["imported_files"]:
        assert (tmp_path / relative).read_bytes() == (source_profiles / relative).read_bytes()
