"""Input provenance and request-disjointness checks for compute ceiling DSE."""
import hashlib
import json
from pathlib import Path

import pytest

from research.moe_dispatch.compute_ceiling.trace_inputs import (
    load_primary_inputs, load_qwen_inputs, write_input_manifest,
)


def _primary_fixture(tmp_path: Path) -> Path:
    manifest = {"input_directory": str(tmp_path), "input_sha256": {},
                "model": {"H": 2048, "routed_F": 1408, "shared_F": 2816}}
    for split in ("development", "heldout"):
        ids = []
        for prefix in ("", "mixed_"):
            identity = f"{prefix}{split}_b2"
            window = {"id": identity, "batch": 2, "top_k": 6, "experts": [
                {"id": -1, "Me": 2, "H": 2048, "F": 2816, "is_shared": True},
                *[{"id": i, "Me": 2, "H": 2048, "F": 1408, "is_shared": False} for i in range(6)]
            ]}
            filename = f"{prefix}{split}.json"
            data = json.dumps({"workloads": [window]}).encode()
            (tmp_path / filename).write_bytes(data)
            manifest["input_sha256"][filename] = hashlib.sha256(data).hexdigest()
            ids.append(identity)
        manifest[f"{split}_window_ids"] = ids
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _qwen_fixture(root: Path, requests=30, layers=(0, 12, 23)) -> None:
    root.mkdir(parents=True)
    for request in range(requests):
        for layer in layers:
            experts = [(request * 7 + offset) % 128 for offset in range(8)]
            record = {
                "model": {"name": "Qwen3-30B-A3B", "hidden_size": 2048,
                          "intermediate_size": 768, "num_experts": 128,
                          "top_k": 8, "layer_index": layer},
                "workload": {"phase": "decode", "token_count": 1,
                             "benchmark": "test_capture", "sample_id": f"request{request}",
                             "sample_index": request},
                "routing": {
                    "source": "true_hidden_state_router_logits_topk_softmax_greedy_decode",
                    "topk_indices": [experts], "topk_weights": [[.125] * 8],
                    "expert_counts": [int(index in experts) for index in range(128)],
                },
            }
            (root / f"r{request}_l{layer}.json").write_text(json.dumps(record))


def test_development_only_never_reads_heldout_payload(tmp_path):
    manifest = _primary_fixture(tmp_path)
    (tmp_path / "heldout.json").unlink()
    (tmp_path / "mixed_heldout.json").unlink()
    result = load_primary_inputs("development", manifest_path=manifest)
    assert len(result["development"]) == 2
    assert "heldout" not in result
    assert set(result["manifest"]["loaded_input_sha256"]) == {"development.json", "mixed_development.json"}


def test_primary_frozen_hash_and_route_conservation(tmp_path):
    manifest = _primary_fixture(tmp_path)
    assert len(load_primary_inputs(None, manifest_path=manifest)["heldout"]) == 2
    with (tmp_path / "development.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_primary_inputs("development", manifest_path=manifest)


def test_primary_shared_shape_and_me_are_enforced(tmp_path):
    manifest = _primary_fixture(tmp_path)
    source = tmp_path / "development.json"
    data = json.loads(source.read_text())
    data["workloads"][0]["experts"][0]["Me"] = 1
    source.write_text(json.dumps(data))
    metadata = json.loads(manifest.read_text())
    metadata["input_sha256"]["development.json"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Me equal to batch"):
        load_primary_inputs("development", manifest_path=manifest)


def test_qwen_microbatches_keep_all_layers_of_request_in_one_partition(tmp_path):
    root = tmp_path / "routes"
    _qwen_fixture(root)
    result = load_qwen_inputs(None, route_root=root, max_windows_per_group=None)
    request_sets = {}
    for split in ("development", "heldout"):
        assert result[split]
        request_sets[split] = {
            tuple(key) for window in result[split]
            for key in window["provenance"]["request_keys"]
        }
        for window in result[split]:
            assert sum(expert["Me"] for expert in window["experts"]) == 8 * window["batch"]
            assert all(not expert["is_shared"] for expert in window["experts"])
            assert not window["provenance"]["new_batched_model_forward_executed"]
            assert len(window["provenance"]["request_keys"]) == window["batch"]
    assert not request_sets["development"] & request_sets["heldout"]
    assert result["manifest"]["valid_source_records"] == 90
    assert not result["manifest"]["rejected_sources"]


def test_qwen_corrupted_histogram_is_rejected_and_never_used(tmp_path):
    root = tmp_path / "routes"
    _qwen_fixture(root)
    path = root / "r0_l0.json"
    data = json.loads(path.read_text())
    data["routing"]["expert_counts"][0] = 99
    path.write_text(json.dumps(data))
    result = load_qwen_inputs(None, route_root=root)
    assert result["manifest"]["valid_source_records"] == 89
    assert result["manifest"]["rejected_sources"] == [{
        "path": str(path), "reason": "captured expert histogram disagrees with token routes"
    }]
    assert all(str(path) not in window["provenance"]["source_paths"]
               for split in ("development", "heldout") for window in result[split])


def test_qwen_group_cap_and_manifest_are_reproducible(tmp_path):
    root = tmp_path / "routes"
    _qwen_fixture(root)
    first = load_qwen_inputs("heldout", route_root=root, max_windows_per_group=1)
    second = load_qwen_inputs("heldout", route_root=root, max_windows_per_group=1)
    assert first == second
    groups = [(window["dataset"], window["layer"], window["batch"]) for window in first["heldout"]]
    assert len(groups) == len(set(groups))
    primary = load_primary_inputs("development", manifest_path=_primary_fixture(tmp_path))
    output = tmp_path / "export/input_manifest.json"
    write_input_manifest(output, primary, qwen=first)
    saved = json.loads(output.read_text())
    assert saved["supplements"]["qwen"]["request_disjoint"]
    assert saved["primary"]["loaded_splits"] == ["development"]
