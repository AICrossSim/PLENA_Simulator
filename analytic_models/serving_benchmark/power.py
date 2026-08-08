"""NVML board-energy and fixed-rate power sampling."""

from __future__ import annotations

import csv
import gzip
import itertools
import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import write_json_atomic


class PowerBackend(Protocol):
    def sample(self, gpu_indices: Iterable[int]) -> list[dict[str, Any]]: ...


class NvmlBackend:
    def __init__(self) -> None:
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("nvidia-ml-py is required on the RunPod host") from exc
        self._nvml = pynvml
        pynvml.nvmlInit()

    def close(self) -> None:
        self._nvml.nvmlShutdown()

    def sample(self, gpu_indices: Iterable[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        nvml = self._nvml
        for index in gpu_indices:
            handle = nvml.nvmlDeviceGetHandleByIndex(int(index))
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                energy_mj: float | None = float(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            except nvml.NVMLError:
                energy_mj = None
            rows.append(
                {
                    "gpu_index": int(index),
                    "uuid": str(nvml.nvmlDeviceGetUUID(handle)),
                    "power_w": float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                    "energy_mj": energy_mj,
                    "sm_clock_mhz": int(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)),
                    "temperature_c": int(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
                    "memory_used_mib": float(memory.used) / (1024.0**2),
                    "memory_total_mib": float(memory.total) / (1024.0**2),
                    "gpu_utilization_pct": int(utilization.gpu),
                }
            )
        return rows


POWER_COLUMNS = (
    "monotonic_s",
    "epoch_s",
    "gpu_index",
    "uuid",
    "power_w",
    "energy_mj",
    "sm_clock_mhz",
    "temperature_c",
    "memory_used_mib",
    "memory_total_mib",
    "gpu_utilization_pct",
)


@dataclass(frozen=True)
class PowerMark:
    name: str
    monotonic_s: float
    epoch_s: float
    samples: tuple[dict[str, Any], ...]


class PowerMonitor:
    def __init__(
        self,
        *,
        gpu_indices: Iterable[int],
        output_path: Path,
        sampling_hz: float = 20.0,
        backend: PowerBackend | None = None,
    ) -> None:
        if sampling_hz <= 0:
            raise ValueError("sampling_hz must be positive")
        self.gpu_indices = tuple(int(index) for index in gpu_indices)
        self.output_path = output_path
        self.sampling_hz = float(sampling_hz)
        self.backend = backend or NvmlBackend()
        self._owns_backend = backend is None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._marks: list[PowerMark] = []
        self._writer_lock = threading.Lock()
        self._file: Any = None
        self._writer: csv.DictWriter | None = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.output_path, "wt", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=POWER_COLUMNS)
        self._writer.writeheader()
        self._thread = threading.Thread(target=self._sample_loop, name="nvml-power-sampler", daemon=True)
        self._thread.start()

    def _record(self, monotonic_s: float, epoch_s: float, samples: list[dict[str, Any]]) -> None:
        if self._writer is None:
            return
        with self._writer_lock:
            for sample in samples:
                self._writer.writerow({"monotonic_s": monotonic_s, "epoch_s": epoch_s, **sample})

    def _sample_loop(self) -> None:
        period = 1.0 / self.sampling_hz
        deadline = time.monotonic()
        while not self._stop.is_set():
            monotonic_s = time.monotonic()
            epoch_s = time.time()
            samples = self.backend.sample(self.gpu_indices)
            self._record(monotonic_s, epoch_s, samples)
            deadline += period
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def mark(self, name: str) -> PowerMark:
        monotonic_s = time.monotonic()
        epoch_s = time.time()
        samples = tuple(self.backend.sample(self.gpu_indices))
        self._record(monotonic_s, epoch_s, list(samples))
        mark = PowerMark(name=name, monotonic_s=monotonic_s, epoch_s=epoch_s, samples=samples)
        self._marks.append(mark)
        return mark

    def stop(self) -> tuple[PowerMark, ...]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2.0 / self.sampling_hz))
        if self._file is not None:
            self._file.flush()
            self._file.close()
        if self._owns_backend and hasattr(self.backend, "close"):
            self.backend.close()  # type: ignore[attr-defined]
        return tuple(self._marks)


def wait_for_gpu_idle(
    gpu_indices: Iterable[int],
    *,
    max_power_w_per_gpu: float = 120.0,
    poll_interval_s: float = 0.5,
    consecutive_samples: int = 3,
    timeout_s: float = 60.0,
    backend: PowerBackend | None = None,
) -> dict[str, Any]:
    """Wait for post-warmup GPU activity to settle before measuring idle power."""
    if max_power_w_per_gpu <= 0 or poll_interval_s <= 0 or consecutive_samples <= 0 or timeout_s <= 0:
        raise ValueError("idle-settle parameters must be positive")
    indices = tuple(int(index) for index in gpu_indices)
    sampler = backend or NvmlBackend()
    owns_backend = backend is None
    start_s = time.monotonic()
    stable = 0
    last_samples: list[dict[str, Any]] = []
    try:
        while time.monotonic() - start_s <= timeout_s:
            last_samples = sampler.sample(indices)
            powers = [float(sample["power_w"]) for sample in last_samples]
            if powers and max(powers) <= max_power_w_per_gpu:
                stable += 1
                if stable >= consecutive_samples:
                    return {
                        "status": "settled",
                        "duration_s": time.monotonic() - start_s,
                        "max_power_w_per_gpu": max_power_w_per_gpu,
                        "observed_power_w_by_gpu": {
                            str(sample["gpu_index"]): float(sample["power_w"])
                            for sample in last_samples
                        },
                        "observed_utilization_pct_by_gpu": {
                            str(sample["gpu_index"]): int(sample["gpu_utilization_pct"])
                            for sample in last_samples
                        },
                    }
            else:
                stable = 0
            time.sleep(poll_interval_s)
    finally:
        if owns_backend and hasattr(sampler, "close"):
            sampler.close()  # type: ignore[attr-defined]
    powers = {str(sample["gpu_index"]): float(sample["power_w"]) for sample in last_samples}
    raise RuntimeError(
        f"GPUs did not settle below {max_power_w_per_gpu:.1f} W within {timeout_s:.1f} s: {powers}"
    )


def _energy_by_gpu(mark: PowerMark) -> dict[int, float]:
    return {
        int(sample["gpu_index"]): float(sample["energy_mj"])
        for sample in mark.samples
        if sample.get("energy_mj") is not None
    }


def direct_energy_delta_mj(start: PowerMark, end: PowerMark) -> float | None:
    start_values = _energy_by_gpu(start)
    end_values = _energy_by_gpu(end)
    if not start_values or start_values.keys() != end_values.keys():
        return None
    deltas = [end_values[index] - start_values[index] for index in start_values]
    if any(delta < 0 for delta in deltas):
        return None
    return math.fsum(deltas)


def integrate_power_csv_mj(path: Path, *, start_s: float, end_s: float) -> float:
    if end_s < start_s:
        raise ValueError("end_s must not precede start_s")
    per_gpu: dict[int, list[tuple[float, float]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            timestamp = float(row["monotonic_s"])
            if start_s <= timestamp <= end_s:
                per_gpu.setdefault(int(row["gpu_index"]), []).append((timestamp, float(row["power_w"])))
    energy_j = 0.0
    for samples in per_gpu.values():
        samples.sort()
        energy_j += math.fsum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in itertools.pairwise(samples)
        )
    return energy_j * 1000.0


def power_summary(
    path: Path,
    marks: Iterable[PowerMark],
    *,
    phase_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    marks_by_name = {mark.name: mark for mark in marks}
    required = ("request_start", "prefill_complete", "first_decode_complete", "request_complete")
    missing = [name for name in required if name not in marks_by_name]
    if missing:
        raise ValueError(f"missing power marks: {missing}")

    def phase(start_name: str, end_name: str) -> dict[str, Any]:
        start = marks_by_name[start_name]
        end = marks_by_name[end_name]
        direct = direct_energy_delta_mj(start, end)
        integrated = integrate_power_csv_mj(path, start_s=start.monotonic_s, end_s=end.monotonic_s)
        error_pct = None
        if direct not in {None, 0.0}:
            error_pct = abs(integrated - direct) / direct * 100.0
        return {
            "duration_s": end.monotonic_s - start.monotonic_s,
            "nvml_counter_energy_mj": direct,
            "sampled_energy_mj": integrated,
            "counter_sampling_error_pct": error_pct,
        }

    summary = {
        "prefill": phase("request_start", "prefill_complete"),
        "first_decode_iteration": phase("prefill_complete", "first_decode_complete"),
        "measured_generation": phase("prefill_complete", "request_complete"),
        "complete_request": phase("request_start", "request_complete"),
    }
    counter_parts = (summary["prefill"]["nvml_counter_energy_mj"], summary["measured_generation"]["nvml_counter_energy_mj"])
    if all(value is not None for value in counter_parts) and summary["complete_request"]["nvml_counter_energy_mj"]:
        reconstructed = math.fsum(float(value) for value in counter_parts)
        complete = float(summary["complete_request"]["nvml_counter_energy_mj"])
        summary["counter_energy_reconstruction_error_pct"] = abs(reconstructed - complete) / complete * 100.0
    else:
        summary["counter_energy_reconstruction_error_pct"] = None
    if "idle_start" in marks_by_name and "idle_end" in marks_by_name:
        idle = phase("idle_start", "idle_end")
        idle_energy = idle["nvml_counter_energy_mj"]
        if idle_energy is None:
            idle_energy = idle["sampled_energy_mj"]
        idle["average_total_board_power_w"] = idle_energy / idle["duration_s"] / 1000.0
        summary["idle_baseline"] = idle
    generation = summary["measured_generation"]
    proxy_latency = phase_summary.get("imported_kv_decode_proxy_latency_s") if phase_summary else None
    proxy_duration_s = (
        float(proxy_latency) if proxy_latency is not None else float(generation["duration_s"])
    )
    proxy = summary.setdefault("imported_kv_decode_proxy", {})
    proxy["duration_s"] = proxy_duration_s
    proxy["fidelity"] = (
        phase_summary.get("decode_proxy_fidelity")
        if phase_summary is not None
        else "measured_post_global_prefill_tail"
    )
    tail_duration_s = float(generation["duration_s"])
    if tail_duration_s <= 0:
        raise ValueError("post-global-prefill decode tail must have positive duration")
    for key in ("nvml_counter_energy_mj", "sampled_energy_mj"):
        tail_energy = generation.get(key)
        if tail_energy is not None:
            proxy[key] = float(tail_energy) / tail_duration_s * proxy_duration_s
    direct = proxy.get("nvml_counter_energy_mj")
    sampled = proxy.get("sampled_energy_mj")
    proxy["counter_sampling_error_pct"] = (
        abs(float(sampled) - float(direct)) / float(direct) * 100.0
        if direct not in {None, 0.0} and sampled is not None
        else None
    )
    if "idle_baseline" in summary:
        idle_power_mw = summary["idle_baseline"]["average_total_board_power_w"] * 1000.0
        for phase_name in (
            "prefill",
            "first_decode_iteration",
            "measured_generation",
            "complete_request",
            "imported_kv_decode_proxy",
        ):
            item = summary[phase_name]
            measured = item.get("nvml_counter_energy_mj")
            if measured is None:
                measured = item.get("sampled_energy_mj")
            item["idle_subtracted_dynamic_energy_mj"] = measured - idle_power_mw * item["duration_s"]
    summary["marks"] = [
        {"name": mark.name, "monotonic_s": mark.monotonic_s, "epoch_s": mark.epoch_s}
        for mark in marks_by_name.values()
    ]
    all_samples: list[dict[str, float]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            all_samples.append(
                {
                    "monotonic_s": float(row["monotonic_s"]),
                    "gpu_index": float(row["gpu_index"]),
                    "power_w": float(row["power_w"]),
                    "memory_used_mib": float(row["memory_used_mib"]),
                }
            )
    start_s = marks_by_name["request_start"].monotonic_s
    end_s = marks_by_name["request_complete"].monotonic_s
    in_window = [sample for sample in all_samples if start_s <= sample["monotonic_s"] <= end_s]
    power_by_timestamp: dict[float, float] = {}
    for sample in in_window:
        power_by_timestamp[sample["monotonic_s"]] = (
            power_by_timestamp.get(sample["monotonic_s"], 0.0) + sample["power_w"]
        )
    complete_energy = summary["complete_request"]["nvml_counter_energy_mj"]
    if complete_energy is None:
        complete_energy = summary["complete_request"]["sampled_energy_mj"]
    summary["telemetry"] = {
        "average_total_board_power_w": complete_energy / summary["complete_request"]["duration_s"] / 1000.0,
        "peak_total_board_power_w": max(power_by_timestamp.values(), default=None),
        "peak_memory_used_mib_per_gpu": {
            str(gpu): max(
                sample["memory_used_mib"] for sample in in_window if int(sample["gpu_index"]) == gpu
            )
            for gpu in sorted({int(sample["gpu_index"]) for sample in in_window})
        },
    }
    return summary


def write_power_marks(path: Path, marks: Iterable[PowerMark]) -> None:
    write_json_atomic(
        path,
        [
            {
                "name": mark.name,
                "monotonic_s": mark.monotonic_s,
                "epoch_s": mark.epoch_s,
                "samples": list(mark.samples),
            }
            for mark in marks
        ],
    )
