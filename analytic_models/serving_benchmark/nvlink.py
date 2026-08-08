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

    def start(self) -> None:
        executable = shutil.which("dcgmi")
        if executable is None:
            self.status = "unavailable"
            self.error = "dcgmi_not_installed"
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

    def mark_end(self) -> None:
        self.measurement_end_s = time.monotonic()

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
