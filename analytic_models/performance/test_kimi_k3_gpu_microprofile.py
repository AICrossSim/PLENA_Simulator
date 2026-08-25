from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .kimi_k3_gpu_microprofile import (
    EXPECTED_FIELDS,
    EXPECTED_FLASHKDA_COMMIT,
    EXPECTED_HF_REVISION,
    EXPECTED_KIMI_COMMIT,
    KdaMicroprofileFormatError,
    STATE_BYTES_PER_REQUEST,
    build_report,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> None:
    comparison = {
        "schema_version": 1,
        "official_source": {"huggingface_model_revision": EXPECTED_HF_REVISION},
        "weight_mapping_bit_identical": True,
        "custom_mode": {"note": "same operator; wrapper equivalence"},
        "comparisons": [],
    }
    clean = {
        "max_abs": 0.0,
        "finite_custom": True,
        "finite_official": True,
    }
    for sequence in (1, 16, 256, 2048):
        comparison["comparisons"].append(
            {
                "sequence_length": sequence,
                "output": clean,
                "recurrent_state": clean,
                "conv_state": {"q": clean, "k": clean, "v": clean},
            }
        )
    comparison_path = root / "validation/kda_custom_vs_official.json"
    _write(comparison_path, comparison)

    layout = {
        "derivation": "official source plus runtime hooks; no PLENA Simulator mapping consulted",
        "fields": [
            {
                "field_name": name,
                "storage_relation": "independent_tensor; test",
                "runtime_hook": {"output": {"contiguous": True}},
            }
            for name in EXPECTED_FIELDS
        ],
        "runtime_hook_case": {"batch": 1, "sequence_length": 1},
        "non_projection_consumer_order": [],
    }
    layout_path = root / "kda/kda_projection_layout.json"
    _write(layout_path, layout)

    flashkda = {
        "passed": True,
        "criteria": {"cosine_min": 0.999, "mean_abs_max": 0.02},
        "results": [
            {
                "sequence_length": sequence,
                "output": {
                    "max_abs": 0.001,
                    "mean_abs": 0.001,
                    "cosine": 0.9999,
                },
                "final_state": {
                    "max_abs": 0.01,
                    "mean_abs": 0.001,
                    "cosine": 0.9999,
                },
            }
            for sequence in (16, 256, 2048)
        ],
    }
    flashkda_path = root / "validation/flashkda_vs_fla.json"
    _write(flashkda_path, flashkda)

    cases = {}
    for name in ("decode_b1", "decode_b8", "prefill_b1_s2048"):
        l2_read = 10 if name == "decode_b1" else 20
        l2_write = 12 if name == "decode_b1" else 24
        cases[name] = {
            "totals": {
                "dram_read_bytes": 100,
                "dram_write_bytes": 0 if name == "decode_b1" else 50,
                "l2_read_sectors": l2_read,
                "l2_write_sectors": l2_write,
                "l2_read_bytes_derived": l2_read * 32,
                "l2_write_bytes_derived": l2_write * 32,
            },
            "kernels": [],
            "decode_ncu_details": "N/A",
        }
    traffic = {
        "raw_metric_support_on_b200": {
            "dram__bytes_read.sum": "supported, unit byte",
            "dram__bytes_write.sum": "supported, unit byte",
        },
        "cases": cases,
    }
    traffic_path = root / "ncu/kda_memory_traffic_summary.json"
    _write(traffic_path, traffic)

    files = (comparison_path, flashkda_path, layout_path, traffic_path)
    hashes = {f"/home/fixture-user/plena-profiles/{path.relative_to(root)}": _hash(path) for path in files}
    manifest = {
        "source": {
            "kimi_k3": {"commit": EXPECTED_KIMI_COMMIT},
            "flashkda": {"commit": EXPECTED_FLASHKDA_COMMIT},
            "huggingface_kimi_k3": {"revision": EXPECTED_HF_REVISION},
        },
        "environments": {"isolation_verified": True},
        "artifact_sha256": hashes,
    }
    _write(root / "manifests/stage2-environment-source-manifest.json", manifest)


def test_profile_keeps_logical_state_write_separate_from_zero_dram_write(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = build_report(tmp_path)
    case = next(item for item in report["memory_traffic"] if item["case"] == "decode_b1")
    assert case["physical_b200_traffic"]["dram_write_bytes"] == 0
    assert case["logical_core_traffic"]["recurrent_state_write_bytes"] == STATE_BYTES_PER_REQUEST
    assert report["official_equivalence"]["wrapper_projection_state_bit_exact"] is True
    assert report["backend_equivalence"]["passed"] is True


def test_profile_rejects_a_changed_hashed_artifact(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "kda/kda_projection_layout.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(KdaMicroprofileFormatError, match="hash mismatch"):
        build_report(tmp_path)


def test_profile_rejects_invalid_l2_sector_conversion(tmp_path: Path) -> None:
    _fixture(tmp_path)
    traffic_path = tmp_path / "ncu/kda_memory_traffic_summary.json"
    traffic = json.loads(traffic_path.read_text())
    traffic["cases"]["decode_b8"]["totals"]["l2_read_bytes_derived"] += 1
    _write(traffic_path, traffic)
    manifest_path = tmp_path / "manifests/stage2-environment-source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    remote = "/home/fixture-user/plena-profiles/ncu/kda_memory_traffic_summary.json"
    manifest["artifact_sha256"][remote] = _hash(traffic_path)
    _write(manifest_path, manifest)
    with pytest.raises(KdaMicroprofileFormatError, match="sector conversion"):
        build_report(tmp_path)
