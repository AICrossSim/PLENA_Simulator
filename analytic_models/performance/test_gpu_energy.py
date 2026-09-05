from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from .gpu_energy import PowerTrace, reanalyze_campaign
from . import gpu_timing_campaign as recorder
from .gpu_timing_campaign import NvmlSampler, _bind_cuda_device, _verify_cuda_device


def test_energy_sorts_samples_and_clips_both_window_edges() -> None:
    # P(t)=100+100t W; integrate [0.25, 0.75] s = 75 J, not the full 150 J.
    trace = PowerTrace(
        [
            {"timestamp_ns": 1_000_000_000, "power_w": 200},
            {"timestamp_ns": 0, "power_w": 100},
            {"timestamp_ns": 500_000_000, "power_w": 150},
        ]
    )
    assert trace.integrate(250_000_000, 750_000_000) == pytest.approx(75)
    assert trace.integrate(0, 1_000_000_000) == pytest.approx(150)


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (0, 20), (5, 30)])
def test_energy_rejects_invalid_or_unbracketed_window(start: int, end: int) -> None:
    trace = PowerTrace([{"timestamp_ns": 1, "power_w": 10}, {"timestamp_ns": 20, "power_w": 10}])
    with pytest.raises(ValueError):
        trace.integrate(start, end)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1])
def test_energy_rejects_invalid_power(bad: float) -> None:
    with pytest.raises(ValueError):
        PowerTrace([{"timestamp_ns": 0, "power_w": bad}, {"timestamp_ns": 10, "power_w": 1}])


def test_energy_handles_duplicate_identical_readings_without_extrapolation() -> None:
    rows = [{"timestamp_ns": 0, "power_w": 10}, {"timestamp_ns": 10**9, "power_w": 10}]
    assert PowerTrace(rows + rows).integrate(0, 10**9) == 10
    with pytest.raises(ValueError, match="conflicting"):
        PowerTrace([*rows, {"timestamp_ns": 0, "power_w": 20}])


class FakeNvml:
    class NVMLError(Exception):
        pass

    def __init__(self) -> None:
        self.call_threads: set[int] = set()
        self.shutdown = False

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return "fake-gpu"

    def nvmlDeviceGetPowerUsage(self, handle):
        self.call_threads.add(threading.get_ident())
        return 100_000

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=50, memory=40)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=1024)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return []

    def nvmlShutdown(self):
        self.shutdown = True


def test_sampler_has_one_power_reader_and_integrates_recorded_batch_window(tmp_path: Path) -> None:
    nvml = FakeNvml()
    sampler = NvmlSampler(0, tmp_path / "power.csv", 0.001, nvml=nvml)
    try:
        sampler.start()
        sampler.set_context(trial_id="trial", phase="measurement")
        start = time.perf_counter_ns()
        sampler._wait_for_sample(start)
        end = time.perf_counter_ns()
        # Changing the diagnostic context does not change the integration window.
        sampler.set_context(trial_id="following_trial", phase="idle")
        result = sampler.trial_stats("trial", start, end)
        assert result["energy_joules"] == pytest.approx(100 * (end - start) / 1e9)
        assert result["energy_method"] == "sorted_interpolated_recorded_batch_window_v2"
        assert nvml.call_threads == {sampler.thread.ident}
        assert threading.get_ident() not in nvml.call_threads
    finally:
        sampler.stop()
    assert nvml.shutdown
    assert all(r["power_read_start_ns"] <= r["timestamp_ns"] <= r["power_read_end_ns"] for r in sampler.rows)
    assert [r["timestamp_ns"] for r in sampler.rows] == sorted(r["timestamp_ns"] for r in sampler.rows)


def test_sampler_propagates_background_errors(tmp_path: Path) -> None:
    class FailingNvml(FakeNvml):
        def nvmlDeviceGetMemoryInfo(self, handle):
            raise RuntimeError("failed memory query")

    sampler = NvmlSampler(0, tmp_path / "power.csv", 0.001, nvml=FailingNvml())
    try:
        with pytest.raises(RuntimeError, match="sampling worker failed"):
            sampler.start()
    finally:
        sampler.stop()


def test_sampler_closes_nvml_when_device_lookup_fails(tmp_path: Path) -> None:
    class BadDeviceNvml(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index):
            raise RuntimeError("bad physical GPU")

    nvml = BadDeviceNvml()
    with pytest.raises(RuntimeError, match="bad physical GPU"):
        NvmlSampler(4, tmp_path / "power.csv", 0.001, nvml=nvml)
    assert nvml.shutdown


def test_sampler_preserves_missing_power_as_unavailable(tmp_path: Path) -> None:
    class UnsupportedPowerNvml(FakeNvml):
        def nvmlDeviceGetPowerUsage(self, handle):
            raise self.NVMLError("power counter unavailable")

    nvml = UnsupportedPowerNvml()
    sampler = NvmlSampler(0, tmp_path / "power.csv", 0.001, nvml=nvml)
    try:
        sampler.start()
        start, end = sampler.rows[0]["timestamp_ns"], time.perf_counter_ns()
        result = sampler.trial_stats("unsupported", start, end)
        assert result["energy_joules"] == "N/A"
        assert not sampler.power_supported
        assert "power counter unavailable" in sampler.power_error
    finally:
        sampler.stop()
    with (tmp_path / "power.csv").open() as source:
        assert all(row["power_w"] == "N/A" for row in csv.DictReader(source))
    assert nvml.shutdown


def test_sampler_short_window_has_energy_but_no_observed_memory_peak(tmp_path: Path) -> None:
    sampler = NvmlSampler(0, tmp_path / "power.csv", 0.001, nvml=FakeNvml())
    sampler.rows = [
        {
            "timestamp_ns": 0,
            "power_w": 100,
            "memory_used_bytes": 1024,
            "compute_pids": [],
            "unexpected_compute_pids": [],
        },
        {
            "timestamp_ns": 1_000_000_000,
            "power_w": 100,
            "memory_used_bytes": 1024,
            "compute_pids": [],
            "unexpected_compute_pids": [],
        },
    ]
    try:
        result = sampler.trial_stats("short", 250_000_000, 750_000_000)
        assert result["energy_joules"] == 50
        assert result["power_sample_count"] == 0
        assert result["gpu_memory_used_peak_bytes"] == "N/A"
    finally:
        sampler.stop()  # Also covers cleanup before the worker was started.


def _fake_torch(uuid: str = "fake-gpu", *, initialized: bool = False, count: int = 1):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: initialized,
            device_count=lambda: count,
            get_device_properties=lambda index: SimpleNamespace(uuid=uuid),
            current_device=lambda: 0,
            synchronize=lambda: None,
        )
    )


def test_cuda_binds_by_nvml_uuid_before_initialization(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _bind_cuda_device("GPU-1234")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-1234"
    assert _verify_cuda_device(_fake_torch("1234"), "GPU-1234") == "1234"


@pytest.mark.parametrize("count,actual", [(2, "1234"), (1, "other-gpu")])
def test_cuda_refuses_unsafe_rebinding_after_initialization(monkeypatch, count: int, actual: str) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(actual, initialized=True, count=count))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old-selection")
    with pytest.raises(RuntimeError, match="already initialized"):
        _bind_cuda_device("GPU-1234")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "old-selection"


def test_cuda_accepts_an_initialized_matching_single_device(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("1234", initialized=True))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    _bind_cuda_device("GPU-1234")


def _prepare_mock_capture(tmp_path: Path, monkeypatch, nvml: FakeNvml):
    sample = {
        "sample_id": "sample",
        "benchmark": "synthetic",
        "prompt_length": 1,
        "prompt_token_ids": [1],
        "prompt_sha256": "test",
    }
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps({"samples": [sample]}))
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recorder",
            "--model",
            str(tmp_path / "model"),
            "--samples",
            str(samples),
            "--output-dir",
            str(output),
            "--physical-gpu",
            "1",
            "--warmup",
            "0",
            "--measurements",
            "1",
            "--power-interval-ms",
            "1",
        ],
    )
    monkeypatch.setattr(
        recorder,
        "build_groups",
        lambda _: [
            {
                "benchmark": "synthetic",
                "mode": "batch_sweep",
                "batch_size": 1,
                "group_index": 0,
                "samples": [sample],
                "generation_limit": 2,
            }
        ],
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong-prior-device")
    model_selections = []

    class FakeLLM:
        def __init__(self, **kwargs):
            model_selections.append(os.environ["CUDA_VISIBLE_DEVICES"])

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    samplers = []

    def create_sampler(gpu, path, interval):
        assert gpu == 1
        sampler = NvmlSampler(gpu, path, interval, nvml=nvml)
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(recorder, "NvmlSampler", create_sampler)

    def run_batch(*args):
        start = time.perf_counter_ns()
        end = time.perf_counter_ns()
        return (
            [
                {
                    **sample,
                    "generation_limit": 2,
                    "generated_token_ids": [2, 3],
                    "generated_tokens": 2,
                    "ttft_ms": 0.01,
                    "itl_ms": [0.01],
                    "tpot_ms": 0.01,
                    "e2e_ms": 0.02,
                    "padded_prompt_length": 1,
                    "arrival_ns": start,
                    "finish_ns": end,
                }
            ],
            start,
            end,
        )

    monkeypatch.setattr(recorder, "run_batch", run_batch)
    return output, samplers, model_selections


def test_recorder_cleans_up_a_sampler_start_failure(tmp_path: Path, monkeypatch) -> None:
    class FailingNvml(FakeNvml):
        def nvmlDeviceGetMemoryInfo(self, handle):
            raise RuntimeError("first sample failed")

    nvml = FailingNvml()
    output, samplers, _ = _prepare_mock_capture(tmp_path, monkeypatch, nvml)
    with pytest.raises(RuntimeError, match="sampling worker failed"):
        recorder.main()
    assert nvml.shutdown
    assert not samplers[0].thread.is_alive()
    assert (output / "power_raw.csv").exists()
    assert not (output / "timing_metadata.json").exists()


def test_recorder_verifies_device_before_loading_the_model(tmp_path: Path, monkeypatch) -> None:
    nvml = FakeNvml()
    output, samplers, model_selections = _prepare_mock_capture(tmp_path, monkeypatch, nvml)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("wrong-device"))
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        recorder.main()
    assert model_selections == []
    assert samplers[0].thread.ident is None
    assert nvml.shutdown
    assert not (output / "timing_metadata.json").exists()


def test_recorder_finishes_when_memory_or_power_samples_are_unavailable(tmp_path: Path, monkeypatch) -> None:
    nvml = FakeNvml()
    output, _, model_selections = _prepare_mock_capture(tmp_path, monkeypatch, nvml)
    # Deterministically model a window with no memory observation and an
    # unsupported power counter; never invent zero-valued measurements.
    monkeypatch.setattr(
        NvmlSampler,
        "trial_stats",
        lambda *args: {
            "power_sample_count": 0,
            "energy_joules": "N/A",
            "gpu_memory_used_peak_bytes": "N/A",
            "energy_method": "sorted_interpolated_recorded_batch_window_v2",
        },
    )
    recorder.main()
    assert model_selections == ["fake-gpu"]
    assert nvml.shutdown
    metadata = json.loads((output / "timing_metadata.json").read_text())
    assert metadata["cuda_nvml_uuid_verified"]
    assert metadata["cuda_gpu_uuid"] == "fake-gpu"
    with (output / "latency_summary.csv").open() as f:
        summary = next(csv.DictReader(f))
    assert summary["gpu_memory_used_peak_bytes"] == "N/A"
    assert summary["batch_energy_joules_median"] == "N/A"


def _legacy_campaign(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    power = [
        {"trial_id": "trial", "timestamp_ns": 2_000_000_000, "power_w": 300},
        {"trial_id": "trial", "timestamp_ns": 0, "power_w": 100},
        {"trial_id": "trial", "timestamp_ns": 1_000_000_000, "power_w": 200},
    ]
    with (root / "power_raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(power[0]))
        writer.writeheader()
        writer.writerows(power)
    row = {
        "trial_id": "trial",
        "batch_size": 1,
        "benchmark": "synthetic",
        "group_index": 0,
        "mode": "batch_sweep",
        "phase": "measurement",
        "arrival_ns": 500_000_000,
        "finish_ns": 1_500_000_000,
        "batch_energy_joules": 400,
        "power_sample_count": 3,
    }
    (root / "latency_raw.jsonl").write_text(json.dumps(row) + "\n")
    _hash_campaign(root)
    return root


def _hash_campaign(root: Path) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
            for name in ("latency_raw.jsonl", "power_raw.csv")
        )
    )


def test_legacy_reanalysis_is_immutable_and_explicitly_approximate(tmp_path: Path) -> None:
    root = _legacy_campaign(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = reanalyze_campaign(root)
    summary = next(r for r in result["summary"] if r["benchmark"] == "all")
    assert summary["archived_batch_energy_joules_median"] == 400
    assert summary["window_reanalysis_joules_median"] == pytest.approx(200)
    assert summary["nonmonotonic_archived_trials"] == 1
    assert summary["window_kinds"] == ["observed_request_window_approximation"]
    assert "not a new GPU capture" in result["claim_boundary"]
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    (root / "power_raw.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        reanalyze_campaign(root)


def test_reanalysis_uses_recorded_batch_boundaries_when_available(tmp_path: Path) -> None:
    root = _legacy_campaign(tmp_path)
    path = root / "latency_raw.jsonl"
    row = json.loads(path.read_text())
    row.update(group_start_ns=0, group_end_ns=2_000_000_000)
    path.write_text(json.dumps(row) + "\n")
    _hash_campaign(root)
    summary = reanalyze_campaign(root)["summary"][0]
    assert summary["window_reanalysis_joules_median"] == pytest.approx(400)
    assert summary["window_kinds"] == ["recorded_batch_window"]
