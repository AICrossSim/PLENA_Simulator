"""Best-effort DCGM NVLink bandwidth capture for A100 tensor parallel runs."""

from __future__ import annotations

import csv
import itertools
import math
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from .io import write_json_atomic


_GPU_ROW = re.compile(
    r"^\s*(?:GPU\s+)?(?P<gpu>\d+)\s+(?P<tx>N/A|[-+0-9.eE]+)\s+(?P<rx>N/A|[-+0-9.eE]+)\s*$"
)
_GPU_HEADER = re.compile(r"^\s*GPU\s+(?P<gpu>\d+):")
_LINK_COUNTER = re.compile(
    r"^\s*Link\s+\d+:\s+Data\s+(?P<direction>Tx|Rx):\s+(?P<value>\d+)\s+KiB\s*$"
)


def _parse_nvidia_smi_counters(raw: str, gpu_indices: set[int]) -> tuple[int, int]:
    current_gpu: int | None = None
    tx_kib = 0
    rx_kib = 0
    for line in raw.splitlines():
        header = _GPU_HEADER.match(line)
        if header is not None:
            current_gpu = int(header.group("gpu"))
            continue
        counter = _LINK_COUNTER.match(line)
        if counter is None or current_gpu not in gpu_indices:
            continue
        value = int(counter.group("value"))
        if counter.group("direction") == "Tx":
            tx_kib += value
        else:
            rx_kib += value
    return tx_kib * 1024, rx_kib * 1024


class DcgmNvlinkMonitor:
    """Stream DCGM fields 1011/1012 and integrate their byte/s rates.

    DCGM profiling requires host support and often administrator privileges.
    Absence is reported rather than replaced with an estimate.
    """

    def __init__(self, *, gpu_indices: tuple[int, ...], output_dir: Path, sampling_ms: int = 50) -> None:
        self.gpu_indices = set(gpu_indices)
        self.output_dir = output_dir
        self.sampling_ms = sampling_ms
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.raw_file: TextIO | None = None
        self.samples: list[tuple[float, int, float, float]] = []
        self.status = "not_started"
        self.error: str | None = None
        self.measurement_start_s: float | None = None
        self.measurement_end_s: float | None = None
        self.counter_executable: str | None = None
        self.counter_start: tuple[int, int] | None = None
        self.counter_end: tuple[int, int] | None = None
        self.counter_raw_start: str | None = None
        self.counter_raw_end: str | None = None

    def start(self) -> None:
        executable = shutil.which("dcgmi")
        if executable is None:
            self.counter_executable = shutil.which("nvidia-smi")
            if self.counter_executable is None:
                self.status = "unavailable"
                self.error = "dcgmi_and_nvidia_smi_not_installed"
            else:
                self.status = "counter_ready"
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_file = (self.output_dir / "nvlink_dcgm_raw.log").open("w", encoding="utf-8")
        command = [executable, "dmon", "-e", "1011,1012", "-d", str(self.sampling_ms)]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.status = "unavailable"
            self.error = str(exc)
            self.raw_file.close()
            return
        self.status = "running"
        self.thread = threading.Thread(target=self._read, name="dcgm-nvlink-reader", daemon=True)
        self.thread.start()

    def mark_start(self) -> None:
        self.measurement_start_s = time.monotonic()
        if self.status == "counter_ready":
            self.counter_start, self.counter_raw_start = self._read_counters()

    def mark_end(self) -> None:
        self.measurement_end_s = time.monotonic()
        if self.status == "counter_ready":
            self.counter_end, self.counter_raw_end = self._read_counters()

    def _read_counters(self) -> tuple[tuple[int, int], str]:
        assert self.counter_executable is not None
        completed = subprocess.run(
            [self.counter_executable, "nvlink", "-gt", "d"],
            check=True,
            text=True,
            capture_output=True,
        )
        return _parse_nvidia_smi_counters(completed.stdout, self.gpu_indices), completed.stdout

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None and self.raw_file is not None
        for line in self.process.stdout:
            timestamp = time.monotonic()
            self.raw_file.write(line)
            self.raw_file.flush()
            match = _GPU_ROW.match(line)
            if match is None or "N/A" in (match.group("tx"), match.group("rx")):
                continue
            gpu = int(match.group("gpu"))
            if gpu in self.gpu_indices:
                self.samples.append((timestamp, gpu, float(match.group("tx")), float(match.group("rx"))))

    @staticmethod
    def _integrate(samples: list[tuple[float, float]]) -> float:
        samples.sort()
        return math.fsum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in itertools.pairwise(samples)
        )

    def stop(self) -> dict[str, Any]:
        if self.status == "counter_ready":
            return self._stop_counters()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.raw_file is not None:
            self.raw_file.close()
        measured_samples = [
            sample
            for sample in self.samples
            if (self.measurement_start_s is None or sample[0] >= self.measurement_start_s)
            and (self.measurement_end_s is None or sample[0] <= self.measurement_end_s)
        ]
        if self.status == "running":
            self.status = "available" if measured_samples else "unavailable"
            if not measured_samples:
                self.error = "dcgm_returned_no_samples_in_request_window"
        per_gpu: dict[int, dict[str, list[tuple[float, float]]]] = {}
        for timestamp, gpu, tx, rx in measured_samples:
            item = per_gpu.setdefault(gpu, {"tx": [], "rx": []})
            item["tx"].append((timestamp, tx))
            item["rx"].append((timestamp, rx))
        tx_bytes = math.fsum(self._integrate(item["tx"]) for item in per_gpu.values())
        rx_bytes = math.fsum(self._integrate(item["rx"]) for item in per_gpu.values())
        csv_path = self.output_dir / "nvlink_samples.csv"
        if self.samples:
            with csv_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(("monotonic_s", "gpu_index", "tx_bytes_per_s", "rx_bytes_per_s"))
                writer.writerows(self.samples)
        summary = {
            "measurement_backend": "dcgm_prof_nvlink_bytes_1011_1012",
            "status": self.status,
            "error": self.error,
            "sample_count": len(measured_samples),
            "aggregate_gpu_tx_bytes": tx_bytes if measured_samples else None,
            "aggregate_gpu_rx_bytes": rx_bytes if measured_samples else None,
            "accounting_note": "TX and RX are reported separately; summing them double-counts peer payload",
            "raw_path": str(self.output_dir / "nvlink_dcgm_raw.log") if self.raw_file is not None else None,
        }
        write_json_atomic(self.output_dir / "nvlink_summary.json", summary)
        return summary

    def _stop_counters(self) -> dict[str, Any]:
        if self.counter_end is None:
            self.counter_end, self.counter_raw_end = self._read_counters()
        if self.counter_start is None:
            self.status = "unavailable"
            self.error = "measurement_start_counter_missing"
            tx_bytes = None
            rx_bytes = None
        else:
            assert self.counter_end is not None
            tx_delta = self.counter_end[0] - self.counter_start[0]
            rx_delta = self.counter_end[1] - self.counter_start[1]
            if tx_delta < 0 or rx_delta < 0:
                self.status = "unavailable"
                self.error = "nvlink_counter_decreased_or_wrapped"
                tx_bytes = None
                rx_bytes = None
            else:
                self.status = "available"
                tx_bytes = tx_delta
                rx_bytes = rx_delta
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.counter_raw_start is not None:
            (self.output_dir / "nvlink_counter_start.log").write_text(
                self.counter_raw_start, encoding="utf-8"
            )
        if self.counter_raw_end is not None:
            (self.output_dir / "nvlink_counter_end.log").write_text(
                self.counter_raw_end, encoding="utf-8"
            )
        summary = {
            "measurement_backend": "nvidia_smi_nvlink_cumulative_counter_delta",
            "status": self.status,
            "error": self.error,
            "sample_count": 2 if self.counter_start is not None and self.counter_end is not None else 0,
            "aggregate_gpu_tx_bytes": tx_bytes,
            "aggregate_gpu_rx_bytes": rx_bytes,
            "accounting_note": "TX and RX are reported separately; summing them double-counts peer payload",
            "raw_path": str(self.output_dir / "nvlink_counter_end.log"),
        }
        write_json_atomic(self.output_dir / "nvlink_summary.json", summary)
        return summary
