from __future__ import annotations

import csv
import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from .aggregate import aggregate_campaign, aggregate_point
from .cli import DEFAULT_MANIFEST, main
from .environment import create_environment_lock, load_environment_lock, validate_environment_lock
from .inventory import (
    EPHEMERAL_MODEL_CACHE_STORAGE,
    _parse_gpu_query,
    _parse_topology,
    validate_a100_sxm_inventory,
)
from .io import write_json_atomic
from .manifest import load_manifest
from .nvlink import DcgmNvlinkMonitor, _GPU_ROW, _parse_nvidia_smi_counters
from .phases import PhaseTracker
from .power import PowerMark, direct_energy_delta_mj, integrate_power_csv_mj
from .runner import (
    _allocate_gpu_group,
    _execution_groups,
    _formal_execution_policy,
    _worker_command,
    group_points,
    pending_points,
)
from .runtime import runtime_point_fingerprint
from . import vllm_worker


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(DEFAULT_MANIFEST)


def test_manifest_expands_formal_matrix_and_context_profiles(manifest) -> None:
    assert len(manifest.formal_points) == 42
    assert len(manifest.preflight_points) == 7
    short = manifest.point_by_id("qwen3-32b.short-1400x200.tp1.b1")
    long = manifest.point_by_id("qwen3-32b.primary-90000x8000.tp1.b1")
    assert short.max_model_len == 32768
    assert short.rope_scaling is None
    assert long.max_model_len == 131072
    assert long.rope_scaling == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    assert all(point.input_tokens + point.output_tokens <= point.max_model_len for point in manifest.points())
    assert sum(not point.required_success for point in manifest.preflight_points) == 3


def test_rope_scaling_supports_direct_and_hf_override_apis() -> None:
    class LegacyEngineArgs:
        def __init__(self, *, rope_scaling=None):
            self.rope_scaling = rope_scaling

    class CurrentEngineArgs:
        def __init__(self, *, hf_overrides=None):
            self.hf_overrides = hf_overrides

    rope = {"rope_type": "yarn", "factor": 4.0}
    legacy_values: dict[str, object] = {}
    current_values: dict[str, object] = {}
    assert vllm_worker._set_rope_scaling(LegacyEngineArgs, legacy_values, rope) == "rope_scaling"
    assert legacy_values == {"rope_scaling": rope}
    assert vllm_worker._set_rope_scaling(CurrentEngineArgs, current_values, rope) == "hf_overrides"
    assert current_values == {"hf_overrides": {"rope_parameters": rope}}


def test_formal_points_share_ten_engine_configurations(manifest) -> None:
    revisions = {name: f"revision-{name}" for name in manifest.models}
    groups = group_points(manifest.formal_points, revisions=revisions, quantization="awq_marlin")
    assert len(groups) == 10
    assert sum(len(group) for group in groups) == 42
    assert all(len({point.gpu_ids for point in group}) == 1 for group in groups)


def test_parallel_scheduler_allocates_disjoint_tp_groups() -> None:
    available = set(range(8))
    allocations = []
    for required in (1, 1, 2, 4):
        allocation = _allocate_gpu_group(available, required)
        assert allocation is not None
        allocations.append(allocation)
        available.difference_update(allocation)
    assert allocations == [(0,), (1,), (2, 3), (4, 5, 6, 7)]
    assert available == set()
    assert _allocate_gpu_group(available, 1) is None


def test_execution_policy_parallelizes_screening_but_isolates_confirmation() -> None:
    assert _formal_execution_policy("screening", "auto") == ("sharded-engine", True)
    assert _formal_execution_policy("short-sweep", "auto") == ("engine", True)
    assert _formal_execution_policy("confirmation", "auto") == ("engine", False)
    assert _formal_execution_policy("holdout", "auto") == ("engine", False)
    assert _formal_execution_policy("confirmation", "gpu-parallel") == ("point", True)


def test_point_parallel_groups_and_physical_assignment_do_not_change_fingerprint(manifest) -> None:
    points = tuple(
        point
        for point in manifest.formal_points
        if point.workload_name == "primary-90000x8000"
    )
    revisions = {name: f"revision-{name}" for name in manifest.models}
    groups = _execution_groups(
        points,
        revisions=revisions,
        quantization="awq",
        granularity="point",
    )
    assert len(groups) == 14
    assert all(len(group) == 1 for group in groups)
    assert groups[0][0].tensor_parallel_size == 8
    assert groups[-1][0].tensor_parallel_size == 1
    point = next(point for point in points if point.tensor_parallel_size == 4)
    fingerprint = runtime_point_fingerprint(
        point,
        revision=revisions[point.model_name],
        quantization="awq",
        environment_hash="environment",
    )
    command = _worker_command(
        manifest=manifest,
        points=(dataclasses.replace(point, repetitions=1, measurement_stage="screening"),),
        revision=revisions[point.model_name],
        quantization="awq",
        environment_hash="environment",
        output_root=Path("/tmp/output"),
        physical_gpu_ids=(4, 5, 6, 7),
    )
    assert command[-2:] == ["--physical-gpu-ids", "4,5,6,7"]
    assert runtime_point_fingerprint(
        point,
        revision=revisions[point.model_name],
        quantization="awq",
        environment_hash="environment",
    ) == fingerprint


def test_screening_shards_reuse_engines_within_gpu_capacity(manifest) -> None:
    points = tuple(
        point
        for point in manifest.formal_points
        if point.workload_name == "primary-90000x8000"
    )
    revisions = {name: f"revision-{name}" for name in manifest.models}
    groups = _execution_groups(
        points,
        revisions=revisions,
        quantization="awq",
        granularity="sharded-engine",
        gpu_capacity=8,
        max_parallel_groups=8,
    )
    assert len(groups) == 11
    assert sum(len(group) for group in groups) == 14
    assert all(len({point.tensor_parallel_size for point in group}) == 1 for group in groups)
    tp8_groups = [group for group in groups if group[0].tensor_parallel_size == 8]
    assert len(tp8_groups) == 2
    assert sorted(len(group) for group in tp8_groups) == [1, 2]


def test_worker_physical_gpu_assignment_does_not_replace_logical_point(
    manifest,
    monkeypatch,
    tmp_path: Path,
) -> None:
    point = manifest.point_by_id("preflight.32b.tp1.short")
    captured = {}

    monkeypatch.setattr(vllm_worker, "create_engine", lambda *args, **kwargs: object())
    monkeypatch.setattr(vllm_worker, "_token_pool", lambda engine: (1, 2, 3))

    def fake_execute_point(engine, worker_point, **kwargs):
        captured["logical_gpu_ids"] = worker_point.gpu_ids
        captured["physical_gpu_ids"] = kwargs["physical_gpu_ids"]

    monkeypatch.setattr(vllm_worker, "execute_point", fake_execute_point)
    vllm_worker.run_worker(
        manifest=manifest,
        point_ids=(point.point_id,),
        revision="revision",
        quantization="awq",
        environment_hash="environment",
        output_root=tmp_path,
        physical_gpu_ids=(7,),
    )
    assert captured == {"logical_gpu_ids": (0,), "physical_gpu_ids": (7,)}


def test_phase_tracker_uses_per_request_makespan() -> None:
    tracker = PhaseTracker(("a", "b"), expected_output_tokens=3)
    tracker.observe("a", cumulative_tokens=1, finished=False, timestamp_s=2.0)
    tracker.observe("a", cumulative_tokens=2, finished=False, timestamp_s=2.2)
    tracker.observe("b", cumulative_tokens=1, finished=False, timestamp_s=3.0)
    tracker.observe("b", cumulative_tokens=2, finished=False, timestamp_s=3.5)
    tracker.observe("a", cumulative_tokens=3, finished=True, timestamp_s=4.0)
    tracker.observe("b", cumulative_tokens=3, finished=True, timestamp_s=5.0)
    summary = tracker.summary(request_start_s=1.0)
    assert summary["prefill_latency_s"] == pytest.approx(2.0)
    assert summary["first_decode_iteration_latency_s"] == pytest.approx(0.5)
    assert summary["measured_generation_latency_s"] == pytest.approx(2.0)
    assert summary["post_global_prefill_generated_tokens"] == 3
    assert summary["post_global_prefill_output_tokens_per_s"] == pytest.approx(1.5)
    assert summary["imported_kv_decode_proxy_latency_s"] == pytest.approx(0.5 + 4.0 / 1.5)
    assert summary["prefill_decode_overlap_s"] == pytest.approx(0.0)
    assert summary["overlap_adjusted_reconstruction_error_pct"] == pytest.approx(0.0)
    assert summary["global_output_tokens"] == 6


def test_phase_tracker_rejects_multi_token_steps() -> None:
    tracker = PhaseTracker(("request",), expected_output_tokens=2)
    tracker.observe("request", cumulative_tokens=2, finished=True, timestamp_s=2.0)
    assert tracker.summary(request_start_s=1.0)["multi_token_step_observed"] is True


def test_inventory_parsers_and_topology_validation() -> None:
    query = "\n".join(
        f"{index}, GPU-{index}, NVIDIA A100-SXM4-80GB, 81920, 400, 1410, 550.54"
        for index in range(8)
    )
    gpus = _parse_gpu_query(query)
    header = "        " + " ".join(f"GPU{index}" for index in range(8)) + " CPU Affinity"
    rows = [header]
    for source in range(8):
        links = ["X" if source == target else "NV12" for target in range(8)]
        rows.append(f"GPU{source}   " + " ".join(links) + " 0-63")
    topology_raw = "\n".join(rows)
    inventory = {
        "gpus": gpus,
        "topology_links": _parse_topology(topology_raw, 8),
        "nvidia_smi_query_raw": "MIG Mode\n Current : Disabled\n" * 8,
        "storage": {"workspace_exists": True, "workspace_total_bytes": 500_000_000_000},
    }
    assert validate_a100_sxm_inventory(inventory) == []

    styled_topology = topology_raw.replace(
        "GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7",
        "\x1b[4mGPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7\x1b[0m",
        1,
    )
    assert _parse_topology(styled_topology, 8) == inventory["topology_links"]

    inventory["storage_mode"] = EPHEMERAL_MODEL_CACHE_STORAGE
    inventory["storage"] = {
        "workspace_exists": True,
        "workspace_total_bytes": 20_000_000_000,
        "workspace_free_bytes": 19_000_000_000,
        "root_total_bytes": 1_000_000_000_000,
        "root_free_bytes": 900_000_000_000,
    }
    assert validate_a100_sxm_inventory(inventory) == []


def test_environment_lock_is_hashed_and_manifest_bound(manifest, tmp_path: Path) -> None:
    revisions = {name: f"sha-{name}" for name in manifest.models}
    inventory = {"inventory_hash": "inventory"}
    lock = create_environment_lock(
        manifest=manifest,
        inventory=inventory,
        resolved_revisions=revisions,
        quantization="awq",
        image_digest="sha256:image",
        preflight_artifacts=[],
    )
    path = tmp_path / "environment.json"
    write_json_atomic(path, lock)
    loaded = load_environment_lock(path)
    assert loaded["environment_hash"] == lock["environment_hash"]
    assert validate_environment_lock(loaded, manifest=manifest, image_digest="sha256:image") == []
    loaded["campaign"] = "tampered"
    write_json_atomic(path, loaded)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_environment_lock(path)


def test_environment_lock_can_use_runtime_fingerprint(manifest) -> None:
    revisions = {name: f"sha-{name}" for name in manifest.models}
    lock = create_environment_lock(
        manifest=manifest,
        inventory={"inventory_hash": "inventory"},
        resolved_revisions=revisions,
        quantization="awq_marlin",
        image_digest=None,
        preflight_artifacts=[],
    )
    assert lock["image_identity"]["kind"] == "runtime-environment-fingerprint"
    assert lock["image_identity"]["fidelity"] == "container_digest_unavailable"
    assert validate_environment_lock(lock, manifest=manifest, image_digest=None) == []


def test_resume_requires_runtime_fingerprint(manifest, tmp_path: Path) -> None:
    point = manifest.formal_points[0]
    revisions = {name: f"sha-{name}" for name in manifest.models}
    fingerprint = runtime_point_fingerprint(
        point,
        revision=revisions[point.model_name],
        quantization="awq",
        environment_hash="environment",
    )
    summary = tmp_path / "points" / point.point_id / "summary.json"
    write_json_atomic(summary, {"status": "complete", "point_fingerprint": fingerprint})
    assert pending_points(
        (point,),
        output_root=tmp_path,
        revisions=revisions,
        quantization="awq",
        environment_hash="environment",
    ) == ()
    with pytest.raises(RuntimeError, match="stale fingerprint"):
        pending_points(
            (point,),
            output_root=tmp_path,
            revisions=revisions,
            quantization="awq_marlin",
            environment_hash="environment",
        )
    write_json_atomic(
        summary,
        {"status": "capacity_infeasible", "point_fingerprint": fingerprint},
    )
    assert pending_points(
        (point,),
        output_root=tmp_path,
        revisions=revisions,
        quantization="awq",
        environment_hash="environment",
    ) == ()


def test_power_counter_and_trapezoid_integration(tmp_path: Path) -> None:
    start = PowerMark("start", 1.0, 1.0, ({"gpu_index": 0, "energy_mj": 100.0},))
    end = PowerMark("end", 3.0, 3.0, ({"gpu_index": 0, "energy_mj": 500.0},))
    assert direct_energy_delta_mj(start, end) == pytest.approx(400.0)

    path = tmp_path / "power.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=("monotonic_s", "gpu_index", "power_w"))
        writer.writeheader()
        writer.writerow({"monotonic_s": 1.0, "gpu_index": 0, "power_w": 200.0})
        writer.writerow({"monotonic_s": 2.0, "gpu_index": 0, "power_w": 200.0})
        writer.writerow({"monotonic_s": 3.0, "gpu_index": 0, "power_w": 200.0})
    assert integrate_power_csv_mj(path, start_s=1.0, end_s=3.0) == pytest.approx(400_000.0)


def test_dcgm_nvlink_row_and_rate_integration() -> None:
    match = _GPU_ROW.match(" GPU 3  1.5e9  2.5e9")
    assert match is not None
    assert match.group("gpu") == "3"
    assert DcgmNvlinkMonitor._integrate([(1.0, 100.0), (2.0, 100.0), (3.0, 100.0)]) == 200.0


def test_nvidia_smi_nvlink_counter_parser_filters_gpus() -> None:
    raw = """GPU 0: NVIDIA A100-SXM4-80GB
 Link 0: Data Tx: 100 KiB
 Link 0: Data Rx: 80 KiB
GPU 1: NVIDIA A100-SXM4-80GB
 Link 0: Data Tx: 70 KiB
 Link 0: Data Rx: 60 KiB
"""
    assert _parse_nvidia_smi_counters(raw, {1}) == (70 * 1024, 60 * 1024)


def test_vllm_worker_phase_loop_without_gpu(manifest, monkeypatch) -> None:
    point = manifest.point_by_id("preflight.32b.tp1.short")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMetrics:
        num_cached_tokens = 0
        num_preemptions = 0

    class FakeCompletion:
        def __init__(self, count):
            self.token_ids = list(range(count))

    class FakeOutput:
        def __init__(self, request_id, count, finished):
            self.request_id = request_id
            self.outputs = [FakeCompletion(count)]
            self.finished = finished
            self.metrics = FakeMetrics()

    class FakeEngine:
        def __init__(self):
            self.requests = []
            self.step_index = 0

        def add_request(self, request_id, prompt, params):
            assert len(prompt["prompt_token_ids"]) == point.input_tokens
            self.requests.append(request_id)

        def has_unfinished_requests(self):
            return self.step_index < 2

        def step(self):
            self.step_index += 1
            return [
                FakeOutput(request_id, self.step_index, self.step_index == 2)
                for request_id in self.requests
            ]

    monkeypatch.setattr(vllm_worker, "_vllm_classes", lambda: (object, object, FakeSamplingParams))
    result = vllm_worker.execute_batch(
        FakeEngine(),
        point,
        output_tokens=2,
        repetition_label="test",
        token_pool=(1, 2, 3),
        output_dir=None,
    )
    assert result["phase"]["global_output_tokens"] == 2
    assert result["phase"]["multi_token_step_observed"] is False


def test_only_capacity_errors_are_optional() -> None:
    assert vllm_worker._failure_status(RuntimeError("CUDA out of memory")) == "capacity_infeasible"
    assert vllm_worker._failure_status(RuntimeError("unsupported quantization")) == "failed"


def _synthetic_point_summary(point) -> dict:
    point = dataclasses.replace(point, repetitions=3, measurement_stage="confirmation")
    repetitions = []
    for scale in (0.99, 1.0, 1.01):
        phase = {
            "prefill_latency_s": 10.0 * scale,
            "first_decode_iteration_latency_s": 0.01 * scale,
            "measured_generation_latency_s": 20.0 * scale,
            "full_request_latency_s": 30.0 * scale,
            "imported_kv_decode_proxy_latency_s": 20.01 * scale,
            "multi_token_step_observed": False,
            "global_output_tokens": point.output_tokens * point.local_batch_size,
            "stage_reconstruction_error_pct": 0.0,
        }
        complete_power = {
            "nvml_counter_energy_mj": 1000.0 * scale,
            "sampled_energy_mj": 990.0 * scale,
            "counter_sampling_error_pct": 1.0,
            "idle_subtracted_dynamic_energy_mj": 500.0 * scale,
        }
        repetitions.append(
            {
                "phase": phase,
                "output_token_hashes": ["stable-hash"],
                "power": {
                    "prefill": {**complete_power, "nvml_counter_energy_mj": 400.0 * scale},
                    "first_decode_iteration": {**complete_power, "nvml_counter_energy_mj": 10.0 * scale},
                    "measured_generation": {**complete_power, "nvml_counter_energy_mj": 600.0 * scale},
                    "complete_request": complete_power,
                    "counter_energy_reconstruction_error_pct": 0.0,
                    "idle_baseline": {"average_total_board_power_w": 50.0 * scale},
                    "imported_kv_decode_proxy": {
                        "nvml_counter_energy_mj": 700.0 * scale,
                        "sampled_energy_mj": 693.0 * scale,
                    },
                },
            }
        )
    return {
        "status": "complete",
        "point_id": point.point_id,
        "point": point.as_dict(),
        "quantization": "awq",
        "resolved_revision": "revision",
        "decode_fidelity": "imported_kv_decode_proxy",
        "repetitions": repetitions,
    }


def test_aggregate_checks_repeatability_and_supports_partial_campaign(manifest, tmp_path: Path) -> None:
    point = manifest.formal_points[0]
    summary = _synthetic_point_summary(point)
    row = aggregate_point(summary)
    assert row["validation_status"] == "pass"
    assert row["greedy_outputs_repeatable"] is True
    assert row["unique_greedy_output_sets"] == 1
    assert row["median_full_request_latency_s"] == pytest.approx(30.0)
    path = tmp_path / "points" / point.point_id / "summary.json"
    write_json_atomic(path, summary)
    report = aggregate_campaign(manifest=manifest, output_root=tmp_path, allow_missing=True)
    assert report["complete_points"] == 1
    assert len(report["missing_points"]) == 41
    assert (tmp_path / "aggregate.csv").is_file()


def test_aggregate_warns_for_nonrepeatable_greedy_outputs(manifest) -> None:
    point = manifest.formal_points[0]
    summary = _synthetic_point_summary(point)
    summary["repetitions"][1]["output_token_hashes"] = ["different-hash"]

    row = aggregate_point(summary)

    assert row["validation_status"] == "warning"
    assert row["greedy_outputs_repeatable"] is False
    assert row["unique_greedy_output_sets"] == 2
    assert "greedy_outputs_differ_across_repetitions" in row["warnings"]


def test_validate_manifest_cli(capsys) -> None:
    assert main(["validate-manifest", "--manifest", str(DEFAULT_MANIFEST)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_points"] == 42
    assert payload["engine_configurations"] == 10
