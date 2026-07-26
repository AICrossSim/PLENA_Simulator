#!/usr/bin/env python3
# ruff: noqa: E402
"""Resource-safe mapped-DC plus RTL-activity power calibration pipeline.

The three stages use independent executors. Mapping and VCD replay share one
license semaphore; Verilator never consumes a DC license. Every scenario is
append-only and resumable, so interruption cannot invalidate completed work.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.area_new.scripts.calibration_csv import (
    append_row,
    read_rows,
    write_rows,
)
from analytic_models.area_new.scripts.license_utils import (
    is_dc_license_unavailable_text,
    resolve_dc_worker_count,
)
from analytic_models.area_new.scripts.run_matrix_machine_calibration import create_worker_copy
from analytic_models.power.scripts.rtl_activity import (
    ActivityArtifact,
    generate_activity_scenarios,
    install_harness,
    qwen_mix_semantic_hash,
)
from analytic_models.power.scripts.run_power_calibration import (
    ACTIVITY_FIELDS,
    DEFAULT_RTL_ROOT,
    PowerPoint,
    SCENARIOS,
    _optional_power_value,
    _power_value,
    _prepare_point,
    _write_plan,
    build_agu_plan,
    build_plan,
    build_plan_v2,
    scenarios_for_point_v2,
)
from analytic_models.power.sram_energy import DEFAULT_CATALOG, build_sram_energy_catalog


GIB = 1024**3
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_power_workers_activity_v2")
MAPPING_FLOW_VERSION = "rtl_saif_map_v1"
MAPPING_FIELDS = [
    "point_id", "point_key", "component", "module", "top_module", "holdout",
    "status", "worker_id", "start_time", "end_time", "elapsed_sec",
    "peak_rss_kib", "mapped_ddc", "mapped_netlist", "sdf", "sdc", "wns_ns",
    "timing_status", "logic_area_um2", "report_dir", "saif_name_map",
    "saif_map_report", "mapping_flow_version", "failure_reason",
]
ACTIVITY_EXTRA_FIELDS = [
    "activity_elapsed_sec", "activity_peak_rss_kib", "activity_fingerprint",
    "measurement_start_ns", "measurement_end_ns", "measurement_cycles",
    "accepted_actions", "completed_actions", "vcd_retention",
    "replay_start_time", "replay_end_time", "replay_elapsed_sec",
    "dc_power_peak_rss_kib", "saif_annotated_objects",
    "saif_mapped_seq_cells", "saif_total_seq_cells",
    "saif_seq_coverage_pct", "saif_coverage_status",
    "saif_map_report", "power_coverage_report", "saif_name_map",
    "saif_name_map_converted_objects",
    "saif_packed_port_overrides",
    "switching_activity_report",
    "clock_network_dynamic_power_mw", "register_dynamic_power_mw",
    "combinational_dynamic_power_mw", "nonclock_dynamic_power_mw",
    "clock_network_energy_pj", "register_dynamic_energy_pj",
    "combinational_dynamic_energy_pj", "nonclock_dynamic_energy_pj",
    "power_group_semantics",
]
POWER_FIELDS = list(dict.fromkeys(ACTIVITY_FIELDS + ACTIVITY_EXTRA_FIELDS))
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


_POWER_UNIT_TO_MW = {
    "W": 1_000.0,
    "mW": 1.0,
    "uW": 1e-3,
    "nW": 1e-6,
    "pW": 1e-9,
}


def _parse_power_group_table(text: str) -> dict[str, dict[str, float]]:
    """Parse DC's summary power-group table and normalize every value to mW.

    DC prints units only on the ``Total`` row.  The preceding group rows use
    those same four column units, so parsing the numeric values without the
    footer silently treats leakage pW as mW.  The returned dynamic field is
    internal plus switching power; leakage is deliberately excluded.
    """

    total = re.search(
        r"^\s*Total\s+"
        r"([-+\d.eE]+)\s+(W|mW|uW|nW|pW)\s+"
        r"([-+\d.eE]+)\s+(W|mW|uW|nW|pW)\s+"
        r"([-+\d.eE]+)\s+(W|mW|uW|nW|pW)\s+"
        r"([-+\d.eE]+)\s+(W|mW|uW|nW|pW)\s*$",
        text,
        re.MULTILINE,
    )
    if total is None:
        raise ValueError("DC report has no parseable Power Group Total row")
    units = [total.group(index) for index in (2, 4, 6, 8)]
    scales = [_POWER_UNIT_TO_MW[unit] for unit in units]
    groups: dict[str, dict[str, float]] = {}
    row_pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+"
        r"([-+\d.eE]+)\s+([-+\d.eE]+)\s+"
        r"([-+\d.eE]+)\s+([-+\d.eE]+)\s+\(\s*[\d.]+%\)",
        re.MULTILINE,
    )
    for match in row_pattern.finditer(text):
        values = [float(match.group(index)) * scale for index, scale in zip(range(2, 6), scales, strict=True)]
        internal, switching, leakage, total_power = values
        groups[match.group(1)] = {
            "internal_power_mw": internal,
            "switching_power_mw": switching,
            "leakage_power_mw": leakage,
            "total_power_mw": total_power,
            "dynamic_power_mw": internal + switching,
        }
    if "clock_network" not in groups:
        raise ValueError("DC report has no clock_network power group")
    return groups


def _power_group_fields(report_text: str, window_ns: float) -> dict[str, Any]:
    """Return auditable clock/non-clock dynamic decomposition for one window."""

    groups = _parse_power_group_table(report_text)
    clock = groups["clock_network"]["dynamic_power_mw"]
    register = groups.get("register", {}).get("dynamic_power_mw", 0.0)
    combinational = groups.get("combinational", {}).get("dynamic_power_mw", 0.0)
    nonclock = sum(
        values["dynamic_power_mw"]
        for name, values in groups.items()
        if name != "clock_network"
    )
    return {
        "clock_network_dynamic_power_mw": clock,
        "register_dynamic_power_mw": register,
        "combinational_dynamic_power_mw": combinational,
        "nonclock_dynamic_power_mw": nonclock,
        "clock_network_energy_pj": clock * window_ns,
        "register_dynamic_energy_pj": register * window_ns,
        "combinational_dynamic_energy_pj": combinational * window_ns,
        "nonclock_dynamic_energy_pj": nonclock * window_ns,
        "power_group_semantics": "dc_pre_cts_clock_baseline_plus_nonclock_dynamic_v2",
    }


def _backfill_power_group_fields(path: Path) -> int:
    """Populate v2 decomposition fields from archived reports without rerunning DC."""

    rows = read_rows(path)
    changed = 0
    for row in rows:
        if row.get("status") != "complete":
            continue
        if row.get("clock_network_energy_pj") and row.get("nonclock_dynamic_energy_pj"):
            continue
        report = Path(row.get("power_report", ""))
        if not report.exists():
            continue
        try:
            window_ns = float(row.get("window_ns") or 0.0)
            row.update(_power_group_fields(report.read_text(errors="ignore"), window_ns))
        except (OSError, ValueError):
            continue
        changed += 1
    if changed:
        write_rows(path, rows, POWER_FIELDS)
    return changed


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GIB
    return 0.0


def _release_runner_locks(handles: list[Any]) -> None:
    """Release process-owned locks without treating stale lock files as state."""

    for handle in reversed(handles):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        except (OSError, ValueError):
            pass


def _acquire_runner_locks(run_dir: Path, worker_root: Path) -> list[Any]:
    """Exclusively own both the append-only run and disposable worker root.

    A second process writing the same CSV or cleaning the same worker root can
    corrupt an otherwise resumable calibration.  The files are intentionally
    persistent: the kernel lock, not file existence, represents ownership, so
    a SIGKILL leaves an auditable owner record without blocking the next run.
    """

    worker_digest = hashlib.sha256(str(worker_root).encode()).hexdigest()[:16]
    lock_paths = (
        run_dir / ".runner.lock",
        Path("/tmp") / f".plena_power_worker_{worker_digest}.lock",
    )
    handles: list[Any] = []
    try:
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip() or "owner metadata unavailable"
                handle.close()
                raise RuntimeError(
                    f"power calibration lock is already held: {path}; {owner}"
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at": _utc_now(),
                        "run_dir": str(run_dir),
                        "worker_root": str(worker_root),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            handles.append(handle)
    except Exception:
        _release_runner_locks(handles)
        raise
    return handles


def _run_tracked_process(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one external job in a process group that Ctrl-C can terminate."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)
    try:
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        with _ACTIVE_PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)


def _terminate_active_processes() -> None:
    """Best-effort stop of DC process trees before executor shutdown waits."""

    with _ACTIVE_PROCESS_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def _install_interrupt_handler() -> None:
    """Ensure a single Ctrl-C stops child process groups before unwinding."""

    def handle_interrupt(_signum: int, _frame: Any) -> None:
        _terminate_active_processes()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)


def _peak_rss(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(errors="ignore"))
    return int(match.group(1)) if match else None


class ResourceGate:
    """Admission tokens with live low-memory and low-disk backpressure."""

    def __init__(self, *, memory_reserve_gib: float, tmp_reserve_gib: float) -> None:
        self.memory_reserve_gib = memory_reserve_gib
        self.tmp_reserve_gib = tmp_reserve_gib
        self.initial_memory_capacity = max(
            0.0, _mem_available_gib() - memory_reserve_gib
        )
        self.initial_tmp_capacity = max(
            0.0, shutil.disk_usage("/tmp").free / GIB - tmp_reserve_gib
        )
        self.memory_used = 0.0
        self.tmp_used = 0.0
        self.estimates: dict[str, tuple[float, float]] = {
            # Initial tokens are 1.35x the peaks observed in the completed v1
            # campaign.  ``release`` raises a class estimate when a new peak
            # is observed, so admission follows this machine rather than a
            # permanently conservative hand-written bound.
            "map_light": (1.50, 0.75),
            "map_medium": (2.00, 1.00),
            "map_heavy": (10.50, 3.50),
            "activity_light": (2.50, 1.00),
            "activity_heavy": (5.00, 3.00),
            "power": (3.25, 0.25),
        }
        self.condition = threading.Condition()

    def acquire(self, kind: str) -> tuple[float, float]:
        with self.condition:
            while True:
                memory, disk = self.estimates[kind]
                # MemAvailable and free disk already include the footprint of
                # running jobs. Require enough *additional* live headroom for
                # this job so admission cannot consume the configured reserve
                # when another user increases pressure after startup.
                live_memory_capacity = max(
                    0.0, _mem_available_gib() - self.memory_reserve_gib
                )
                live_tmp_capacity = max(
                    0.0,
                    shutil.disk_usage("/tmp").free / GIB - self.tmp_reserve_gib,
                )
                # Include all admitted tokens, even though a running job's
                # current RSS is already reflected in MemAvailable.  The
                # deliberate double-count is conservative and also covers the
                # gap between process launch and its eventual peak RSS.  Unlike
                # a startup-only capacity, this can expand when another user
                # releases memory or disk space.
                token_ok = (
                    self.memory_used + memory <= live_memory_capacity
                    and self.tmp_used + disk <= live_tmp_capacity
                )
                if token_ok:
                    self.memory_used += memory
                    self.tmp_used += disk
                    return memory, disk
                self.condition.wait(timeout=30.0)

    def release(self, kind: str, token: tuple[float, float], *, peak_rss_kib: int | None = None) -> None:
        with self.condition:
            memory, disk = token
            self.memory_used = max(0.0, self.memory_used - memory)
            self.tmp_used = max(0.0, self.tmp_used - disk)
            if peak_rss_kib:
                measured = peak_rss_kib / 1024**2 * 1.35
                current_memory, current_disk = self.estimates[kind]
                if measured > current_memory:
                    self.estimates[kind] = (measured, current_disk)
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "memory_reserve_gib": self.memory_reserve_gib,
                "tmp_reserve_gib": self.tmp_reserve_gib,
                "initial_memory_capacity_gib": self.initial_memory_capacity,
                "initial_tmp_capacity_gib": self.initial_tmp_capacity,
                "learned_estimates": {key: {"memory_gib": value[0], "tmp_gib": value[1]} for key, value in self.estimates.items()},
            }


class WeightedSemaphore:
    """A condition-based token pool for weighted CPU admission."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.available = self.capacity
        self.condition = threading.Condition()

    def acquire(self, amount: int) -> int:
        amount = max(1, min(amount, self.capacity))
        with self.condition:
            while self.available < amount:
                self.condition.wait(timeout=10.0)
            self.available -= amount
        return amount

    def release(self, amount: int) -> None:
        with self.condition:
            self.available = min(self.capacity, self.available + amount)
            self.condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self.condition:
            return {"capacity": self.capacity, "available": self.available}


def _point_weight(point: PowerPoint, stage: str) -> str:
    if stage == "activity":
        return "activity_heavy" if point.component in {"vector", "hbm"} else "activity_light"
    if point.component in {"vector", "hbm"} or int(point.params.get("BLOCK_DIM", 0)) >= 8:
        return "map_heavy"
    if point.component == "matrix":
        return "map_medium"
    return "map_light"


def _read_latest(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        return {}
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest[tuple(row.get(field, "") for field in key_fields)] = row
    return latest


def _resume_row_is_current(
    row: dict[str, str],
    scenario: tuple[str, str, int, str],
    expected_activity_fingerprint: str | None = None,
) -> bool:
    """Require semantic provenance, not only success, for resume skipping."""

    if row.get("status") != "complete":
        return False
    if (
        expected_activity_fingerprint is not None
        and row.get("activity_fingerprint") != expected_activity_fingerprint
    ):
        return False
    _, pattern, _, microkernel = scenario
    if pattern != "representative-qwen" or microkernel != "mixed":
        return True
    try:
        sidecar = json.loads(row.get("features_json", ""))
    except json.JSONDecodeError:
        return False
    return sidecar.get("qwen_mix_semantic_hash") == qwen_mix_semantic_hash()


def _export_latest_complete(source: Path, destination: Path) -> int:
    """Write one latest successful row per activity job, preserving raw history."""

    latest = _read_latest(source, ("point_key", "scenario"))
    rows = [row for row in latest.values() if row.get("status") == "complete"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POWER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row.get("component", ""), row.get("point_id", ""), row.get("scenario", ""))))
    os.replace(temporary, destination)
    return len(rows)


def _run_dc_retry(
    command: list[str], *, cwd: Path, log: Path, wait_sec: float,
    max_retries: int, license_sem: threading.Semaphore,
) -> subprocess.CompletedProcess[str]:
    """Run one DC command, releasing the local slot between busy retries.

    A FlexNet checkout can fail because another host or user consumed a token
    after this runner sized its pool.  Keeping the local semaphore while
    sleeping would prevent unrelated queued jobs from attempting checkout, so
    only the actual DC invocation owns a slot.
    """

    attempt = 0
    while True:
        attempt += 1
        with license_sem:
            proc = _run_tracked_process(command, cwd=cwd)
        log.parent.mkdir(parents=True, exist_ok=True)
        (log.parent / f"{log.stem}.attempt_{attempt}.stdout.log").write_text(proc.stdout)
        (log.parent / f"{log.stem}.attempt_{attempt}.stderr.log").write_text(proc.stderr)
        log.write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode == 0 or not is_dc_license_unavailable_text(proc.stdout + proc.stderr):
            return proc
        if max_retries and attempt >= max_retries:
            return proc
        time.sleep(wait_sec)


def _ensure_worker(point: PowerPoint, index: int, worker_root: Path, rtl_root: Path) -> Path:
    destination = worker_root / f"worker_{index}" / "PLENA_RTL"
    if not destination.exists():
        destination = create_worker_copy(index, worker_root, rtl_root)
    _prepare_point(point, destination)
    _patch_power_synthesis_flow(destination)
    return destination


def _patch_power_synthesis_flow(worker: Path) -> None:
    """Retain RTL names through compile for reliable post-map SAIF replay."""

    synth = worker / "tools/synopsys/synth.tcl"
    text = synth.read_text()
    start_marker = "POWER_ACTIVITY_SAIF_MAP_START_V1"
    if start_marker not in text:
        anchor = 'puts "\\n>>> Reading RTL files..."\n'
        if anchor not in text:
            raise ValueError(f"cannot locate RTL read anchor in {synth}")
        text = text.replace(
            anchor,
            f"# {start_marker}\n"
            "# Initialize before analyze/elaborate so synthesis-invariant RTL\n"
            "# names survive optimization in the mapped DDC.\n"
            "saif_map -start\n"
            + anchor,
            1,
        )
    report_marker = "POWER_ACTIVITY_SAIF_MAP_REPORT_V1"
    if report_marker not in text:
        anchor = '#------------------------------\n# Save mapped design\n'
        if anchor not in text:
            raise ValueError(f"cannot locate mapped-design anchor in {synth}")
        text = text.replace(
            anchor,
            f"# {report_marker}\n"
            "saif_map -write_map ${out_dir}/${MODULE}_saif.namemap\n"
            "saif_map -report -rtl_summary -missing_rtl > "
            "${rpt_dir}/${MODULE}_saif_map.rpt\n\n"
            + anchor,
            1,
        )
    synth.write_text(text)


def _archive_mapping(point: PowerPoint, worker: Path, run_dir: Path) -> dict[str, Any]:
    latest = worker / "build/synth" / point.top_module / "latest"
    archive = run_dir / "mapped" / point.point_key
    archive.mkdir(parents=True, exist_ok=True)
    candidates = {
        "mapped_ddc": latest / "out" / f"{point.top_module}_mapped.ddc",
        "mapped_netlist": latest / "out" / f"{point.top_module}_mapped.v",
        "sdf": latest / "out" / f"{point.top_module}.sdf",
        "sdc": latest / "out" / f"{point.top_module}.sdc",
        "timing": latest / "reports" / f"{point.top_module}_timing.rpt",
        "qor": latest / "reports" / f"{point.top_module}_qor.rpt",
        "area": latest / "reports" / f"{point.top_module}_area.rpt",
        "reference": latest / "reports" / f"{point.top_module}_reference.rpt",
        "saif_name_map": latest / "out" / f"{point.top_module}_saif.namemap",
        "saif_map_report": latest / "reports" / f"{point.top_module}_saif_map.rpt",
    }
    copied: dict[str, str] = {}
    for name, source in candidates.items():
        if source.exists():
            target = archive / source.name
            shutil.copy2(source, target)
            copied[name] = str(target)
    if "mapped_ddc" not in copied:
        raise FileNotFoundError(f"normal synthesis produced no mapped DDC for {point.point_id}")
    if "saif_name_map" not in copied or "saif_map_report" not in copied:
        raise FileNotFoundError(f"power mapping produced no SAIF name map for {point.point_id}")
    area = None
    if "area" in copied:
        match = re.search(r"Total cell area:\s*([0-9.eE+-]+)", Path(copied["area"]).read_text(errors="ignore"))
        area = float(match.group(1)) if match else None
    wns = None
    if "timing" in copied:
        matches = re.findall(r"slack\s*\([^)]*\)\s*(-?[0-9.]+)", Path(copied["timing"]).read_text(errors="ignore"), re.I)
        wns = min(map(float, matches)) if matches else None
    manifest = {
        **copied,
        "logic_area_um2": "" if area is None else area,
        "wns_ns": "" if wns is None else wns,
        "timing_status": "timing_unknown" if wns is None else "timing_unclosed" if wns < 0 else "timing_closed",
        "report_dir": str(archive),
        "mapping_flow_version": MAPPING_FLOW_VERSION,
    }
    (archive / "mapping_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _mapping_manifest(point: PowerPoint, run_dir: Path) -> dict[str, Any] | None:
    archive = run_dir / "mapped" / point.point_key
    metadata = archive / "mapping_manifest.json"
    if not metadata.exists():
        return None
    try:
        persisted = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if persisted.get("mapping_flow_version") != MAPPING_FLOW_VERSION:
        return None
    ddc = archive / f"{point.top_module}_mapped.ddc"
    if not ddc.exists():
        return None
    result: dict[str, Any] = {
        "mapped_ddc": str(ddc), "report_dir": str(archive),
        "mapping_flow_version": MAPPING_FLOW_VERSION,
    }
    for field, suffix in (("mapped_netlist", "_mapped.v"), ("sdf", ".sdf"), ("sdc", ".sdc"), ("timing", "_timing.rpt"), ("area", "_area.rpt")):
        path = archive / f"{point.top_module}{suffix}"
        if path.exists():
            result[field] = str(path)
    if "area" in result:
        match = re.search(r"Total cell area:\s*([0-9.eE+-]+)", Path(result["area"]).read_text(errors="ignore"))
        result["logic_area_um2"] = float(match.group(1)) if match else ""
    else:
        result["logic_area_um2"] = ""
    for field, filename in (
        ("saif_name_map", f"{point.top_module}_saif.namemap"),
        ("saif_map_report", f"{point.top_module}_saif_map.rpt"),
    ):
        path = archive / filename
        if not path.exists():
            return None
        result[field] = str(path)
    return result


def _reuse_mapping_manifest(
    point: PowerPoint,
    source_run: Path | None,
    destination_run: Path,
) -> dict[str, Any] | None:
    """Reference a validated archived mapping from an earlier calibration run.

    Mapped DDCs dominate persistent storage.  A relative symlink keeps the v2
    run self-describing while leaving the immutable v1 archive as the single
    copy.  Worker cleanup never traverses ``run_dir/mapped``, so reuse cannot
    delete or mutate the source campaign.
    """

    if source_run is None:
        return None
    source_run = source_run.resolve()
    manifest = _mapping_manifest(point, source_run)
    if manifest is None:
        return None
    source_archive = source_run / "mapped" / point.point_key
    destination_archive = destination_run / "mapped" / point.point_key
    destination_archive.parent.mkdir(parents=True, exist_ok=True)
    if destination_archive.exists() or destination_archive.is_symlink():
        if destination_archive.is_symlink() and destination_archive.resolve() == source_archive.resolve():
            return _mapping_manifest(point, destination_run)
        return None
    destination_archive.symlink_to(
        os.path.relpath(source_archive, destination_archive.parent),
        target_is_directory=True,
    )
    return _mapping_manifest(point, destination_run)


def _verilator_saif_name(original_line: str, saif_line: str) -> str | None:
    """Translate a DC packed-object name to Verilator's nested SAIF path."""

    original_match = re.search(r"\sL:(\S+)\s*$", original_line)
    saif_match = re.match(r"sname\s+(-?)\s*(\S+)\s*$", saif_line)
    if not original_match or not saif_match:
        return None
    original = original_match.group(1)
    current = saif_match.group(2).split("[", 1)[0]
    tokens = re.findall(r"\[([^]]+)\]", original)
    if not tokens or not any(not token.isdigit() for token in tokens):
        return None
    for token in tokens:
        current += f"[{token}]" if token.isdigit() else f"/{token}"
    return f"sname - {current}"


def _write_verilator_saif_name_map(source: Path, destination: Path) -> int:
    """Write a replay map whose source names match Verilator's SAIF hierarchy."""

    lines = source.read_text().splitlines()
    converted = 0
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("oname ") or not lines[index + 1].startswith("sname "):
            continue
        replacement = _verilator_saif_name(line, lines[index + 1])
        if replacement is not None:
            lines[index + 1] = replacement
            converted += 1
    destination.write_text("\n".join(lines) + "\n")
    return converted


def _saif_signal_activity(
    saif: Path, *, instance_path: str
) -> dict[str, tuple[int, int, int]]:
    """Return ``relative/path -> (T0, T1, TC)`` from a line-oriented SAIF.

    Verilator represents a packed SystemVerilog struct as nested SAIF
    ``INSTANCE`` scopes.  Power Compiler's RTL name map can identify those
    scopes, but DC 2024.09 does not attach their activity to flattened primary
    input bits.  Parsing the measured SAIF values lets replay apply an exact
    per-bit override rather than replacing the missing inputs with an assumed
    toggle rate.
    """

    opener = gzip.open if saif.suffix == ".gz" else Path.open
    with opener(saif, "rt", errors="ignore") as handle:
        lines = handle.readlines()
    root = tuple(part for part in instance_path.split("/") if part)
    instances: list[tuple[int, str]] = []
    result: dict[str, tuple[int, int, int]] = {}
    signal_re = re.compile(r"^(\s*)\(([^()\s]+)\s*$")
    value_re = re.compile(r"\((T0|T1|TC)\s+(\d+)\)")

    for index, line in enumerate(lines):
        instance = re.match(r"^(\s*)\(INSTANCE\s+([^()\s]+)", line)
        if instance:
            indent = len(instance.group(1))
            while instances and instances[-1][0] >= indent:
                instances.pop()
            instances.append((indent, instance.group(2)))
            continue
        match = signal_re.match(line)
        if not match or match.group(2) in {"NET", "PORT", "INSTANCE", "SAIFILE"}:
            continue
        indent = len(match.group(1))
        if not instances or indent <= instances[-1][0]:
            continue
        values: dict[str, int] = {}
        for detail in lines[index + 1 : index + 6]:
            for key, value in value_re.findall(detail):
                values[key] = int(value)
            if len(values) == 3:
                break
        if not {"T0", "T1", "TC"}.issubset(values):
            continue
        path = tuple(name.replace("\\[", "[").replace("\\]", "]") for _, name in instances)
        if path[: len(root)] != root:
            continue
        signal = match.group(2).replace("\\[", "[").replace("\\]", "]")
        relative = "/".join((*path[len(root) :], signal))
        # Keep the first occurrence at the requested DUT root.  Verilator may
        # repeat a struct in deeper generated scopes, which is not a port.
        result.setdefault(relative, (values["T0"], values["T1"], values["TC"]))
    return result


def _packed_port_activity_tcl(
    *, name_map: Path, saif: Path, instance_path: str, window_ns: float
) -> tuple[str, int]:
    """Build exact DC overrides for flattened packed-struct input bits."""

    activity = _saif_signal_activity(saif, instance_path=instance_path)
    lines = name_map.read_text().splitlines()
    commands: list[str] = []
    for index, line in enumerate(lines[:-2]):
        port_match = re.match(r"port\s+(\S+)\s*$", line)
        if not port_match or not lines[index + 2].startswith("sname "):
            continue
        source_match = re.match(r"sname\s+-?\s*(\S+)\s*$", lines[index + 2])
        if not source_match or "/" not in source_match.group(1):
            continue
        values = activity.get(source_match.group(1))
        if values is None:
            continue
        t0, t1, transitions = values
        known_time = t0 + t1
        probability = 0.0 if known_time == 0 else t1 / known_time
        toggle_rate = transitions / window_ns
        commands.append(
            "set_switching_activity "
            f"-static_probability {probability:.12g} "
            f"-toggle_rate {toggle_rate:.12g} "
            f"[get_ports {{{port_match.group(1)}}}]"
        )
    header = (
        "# DC 2024.09 does not propagate nested packed-struct SAIF members to\n"
        "# flattened primary-input bits. Apply exact T1/(T0+T1) and TC/window\n"
        "# values parsed from this same measured SAIF window.\n"
    )
    return header + "\n".join(commands), len(commands)


def _activity_tcl(
    *, ddc: Path, top: str, saif: Path, name_map: Path,
    instance_path: str, report: Path, packed_port_activity: str = ""
) -> str:
    return f"""read_ddc {{{ddc}}}
current_design {{{top}}}
link
# Mapped DDCs retain vectorless activity from synthesis.  It must be removed or
# read_saif leaves some of those old annotations in place (PWR-200), corrupting
# active-minus-idle energy slopes.
reset_switching_activity
saif_map -read_map {{{name_map}}}
set annotated [read_saif -input {{{saif}}} -instance_name {{{instance_path}}} -auto_map_names -verbose]
if {{$annotated == 0}} {{
  puts "ERROR: SAIF did not annotate any object"
  exit 2
}}
saif_map -report -rtl_summary -missing_rtl > {{{report.with_name('power.saif_map.rpt')}}}
{packed_port_activity}
# Verilator emits SystemVerilog struct members as nested scopes, while DC
# flattens constant/unused members into primary-input bits.  Preserve every
# mapped SAIF value and seed only genuinely unannotated input bits as constants.
set saif_inputs [filter_collection [all_inputs] "saif_toggle_rate_flag == true"]
set missing_inputs [remove_from_collection [all_inputs] $saif_inputs]
if {{[sizeof_collection $missing_inputs] > 0}} {{
  set_switching_activity -static_probability 0.0 -toggle_rate 0.0 $missing_inputs
}}
report_saif -hierarchy -missing > {{{report.with_suffix('.coverage.rpt')}}}
report_power > {{{report}}}
report_power -hierarchy > {{{report.with_name('power.hierarchy.rpt')}}}
exit
"""


def _parse_saif_map_seq_coverage(path: Path) -> tuple[int, int, float]:
    """Return mapped/total synthesis-invariant sequential cells.

    ``saif_map -rtl_summary`` has four mutually exclusive mapped columns
    (automatic/user, direct/connected) followed by the total object count.
    This is the meaningful RTL-activity coverage metric; the generic
    ``report_saif`` net percentage also includes optimized combinational nets
    and substantially understates useful register coverage.
    """

    text = path.read_text(errors="ignore")
    match = re.search(r"^\s*Seq Cells\s+(.+?)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"SAIF map report has no Seq Cells summary: {path}")
    mapped_columns = [int(value) for value in re.findall(r"(\d+)\([^)]*\)", match.group(1))]
    total_match = re.search(r"(\d+)\s*$", match.group(1))
    if len(mapped_columns) < 4 or not total_match:
        raise ValueError(f"cannot parse Seq Cells summary in {path}")
    mapped = sum(mapped_columns[:4])
    total = int(total_match.group(1))
    coverage = 100.0 if total == 0 else 100.0 * mapped / total
    return mapped, total, coverage


def _vcd_timescale_ps(vcd: Path) -> float:
    """Return the VCD timestamp quantum in picoseconds."""
    with vcd.open(errors="ignore") as handle:
        header = handle.read(16_384)
    match = re.search(r"\$timescale\s+(\d+(?:\.\d+)?)\s*(fs|ps|ns|us)\s+\$end", header, re.I)
    if not match:
        raise ValueError(f"VCD has no supported $timescale: {vcd}")
    multiplier = {"fs": 1e-3, "ps": 1.0, "ns": 1e3, "us": 1e6}[match.group(2).lower()]
    return float(match.group(1)) * multiplier


def _convert_vcd_to_saif(
    *, vcd: Path, saif: Path, instance_path: str, start_ns: float, end_ns: float, log: Path
) -> None:
    quantum_ps = _vcd_timescale_ps(vcd)
    start_tick = round(start_ns * 1_000.0 / quantum_ps)
    end_tick = round(end_ns * 1_000.0 / quantum_ps)
    if end_tick <= start_tick:
        raise ValueError(f"invalid VCD measurement window: {start_tick}..{end_tick}")
    command = [
        "/mnt/applications/synopsys/2024-25/RHELx86/SYN_2024.09-SP2/bin/vcd2saif",
        "-input", str(vcd), "-output", str(saif), "-instance", instance_path,
        "-time", str(start_tick), str(end_tick),
    ]
    saif.unlink(missing_ok=True)
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not saif.exists():
        raise RuntimeError(f"vcd2saif failed with exit {proc.returncode}; see {log}")


def _power_replay(
    *,
    point: PowerPoint,
    artifact: ActivityArtifact,
    mapping: dict[str, Any],
    run_dir: Path,
    rtl_root: Path,
    gate: ResourceGate,
    cpu_gate: WeightedSemaphore,
    license_sem: threading.Semaphore,
    csv_lock: threading.Lock,
    retry_wait: float,
    max_retries: int,
    min_seq_coverage_pct: float,
) -> dict[str, Any]:
    token = gate.acquire("power")
    cpu_token = cpu_gate.acquire(1)
    started = time.monotonic()
    sidecar = json.loads(artifact.sidecar.read_text())
    report_dir = run_dir / "reports" / point.point_key / artifact.scenario
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "power.rpt"
    tcl = report_dir / "power.tcl"
    saif = report_dir / "activity.saif.gz"
    replay_name_map = report_dir / "verilator_saif.namemap"
    converted_name_count = _write_verilator_saif_name_map(
        Path(mapping["saif_name_map"]), replay_name_map
    )
    instance_path = "power_activity_tb" if point.component == "matrix" else "power_activity_tb/dut"
    row: dict[str, Any] = {
        "point_id": point.point_id, "point_key": point.point_key,
        "component": point.component, "scenario": artifact.scenario,
        "pattern": artifact.pattern, "repeat_count": artifact.repeat_count,
        "microkernel": artifact.microkernel,
        "holdout": int(point.holdout), "status": "failed",
        "activity_elapsed_sec": round(artifact.elapsed_sec, 3),
        "activity_peak_rss_kib": artifact.peak_rss_kib or "",
        "activity_fingerprint": artifact.fingerprint,
        "saif_name_map": str(replay_name_map),
        "saif_name_map_converted_objects": converted_name_count,
        "measurement_start_ns": sidecar["measurement_start_ns"],
        "measurement_end_ns": sidecar["measurement_end_ns"],
        "measurement_cycles": sidecar["measurement_cycles"],
        "accepted_actions": sidecar["accepted_actions"],
        "completed_actions": sidecar["completed_actions"],
        "window_ns": float(sidecar["measurement_end_ns"]) - float(sidecar["measurement_start_ns"]),
        "features_json": json.dumps(sidecar, sort_keys=True),
        "activity_level": "rtl_vcd_mapped_dc",
        "logic_area_um2": mapping.get("logic_area_um2", ""),
        "vcd_path": str(artifact.vcd), "power_report": str(report),
        "activity_log": str(report_dir / "dc_power.log"),
        "replay_start_time": _utc_now(),
        "saif_map_report": str(report_dir / "power.saif_map.rpt"),
        "power_coverage_report": str(report.with_suffix(".coverage.rpt")),
        # Design Compiler 2024.09 does not expose report_switching_activity;
        # quantitative coverage comes from report_saif and saif_map instead.
        "switching_activity_report": "unsupported_by_dc_2024_09",
    }
    peak = None
    pwr_414 = False
    pwr_415 = False
    try:
        _convert_vcd_to_saif(
            vcd=artifact.vcd, saif=saif, instance_path=instance_path,
            start_ns=float(sidecar["measurement_start_ns"]),
            end_ns=float(sidecar["measurement_end_ns"]),
            log=report_dir / "vcd2saif.log",
        )
        packed_activity_tcl, packed_override_count = _packed_port_activity_tcl(
            name_map=replay_name_map,
            saif=saif,
            instance_path=instance_path,
            window_ns=float(row["window_ns"]),
        )
        row["saif_packed_port_overrides"] = packed_override_count
        tcl.write_text(
            _activity_tcl(
                ddc=Path(mapping["mapped_ddc"]), top=point.top_module,
                saif=saif, name_map=replay_name_map,
                instance_path=instance_path, report=report,
                packed_port_activity=packed_activity_tcl,
            )
        )
        command = (
            "source /mnt/applications/synopsys/2024-25/scripts/SYN_2024.09-SP2_RHELx86.sh && "
            "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV _PYTHON_SYSCONFIGDATA_NAME "
            "_PYTHON_HOST_PLATFORM PYTHONNOUSERSITE PYTHONHASHSEED && "
            "export PATH=$(printf '%s' \"$PATH\" | tr ':' '\\n' | "
            "grep -v '/nix/store.*python' | tr '\\n' ':' | sed 's/:$//') && "
            "/usr/bin/time -v -o " + shlex.quote(str(report_dir / "dc_power.time"))
            + " dc_shell -f " + shlex.quote(str(tcl))
        )
        proc = _run_dc_retry(
            ["bash", "-lc", command], cwd=rtl_root / "tools/synopsys",
            log=report_dir / "dc_power.log", wait_sec=retry_wait,
            max_retries=max_retries, license_sem=license_sem,
        )
        peak = _peak_rss(report_dir / "dc_power.time")
        if proc.returncode != 0 or not report.exists():
            raise RuntimeError(f"DC power replay failed with exit {proc.returncode}")
        report_text = report.read_text(errors="ignore")
        coverage_text = report.with_suffix(".coverage.rpt").read_text(errors="ignore")
        combined = proc.stdout + proc.stderr + report_text + coverage_text
        pwr_414 = "PWR-414" in combined
        pwr_415 = "PWR-415" in combined
        mapped_seq, total_seq, seq_coverage = _parse_saif_map_seq_coverage(
            report_dir / "power.saif_map.rpt"
        )
        annotated_match = re.search(r"Annotated\s*=\s*(\d+)", combined, re.IGNORECASE)
        row.update({
            "saif_annotated_objects": annotated_match.group(1) if annotated_match else "",
            "saif_mapped_seq_cells": mapped_seq,
            "saif_total_seq_cells": total_seq,
            "saif_seq_coverage_pct": round(seq_coverage, 4),
            "saif_coverage_status": (
                "complete" if not pwr_415 else
                "partial_noncritical" if seq_coverage >= min_seq_coverage_pct else
                "critical_missing_sequential_activity"
            ),
        })
        if pwr_414:
            raise RuntimeError("critical primary-input activity warning PWR-414")
        if pwr_415 and seq_coverage < min_seq_coverage_pct:
            raise RuntimeError(
                "critical sequential activity coverage: "
                f"{seq_coverage:.2f}% < {min_seq_coverage_pct:.2f}% (PWR-415)"
            )
        dynamic = _power_value(report_text, "Total Dynamic Power")
        leakage = _power_value(report_text, "Cell Leakage Power")
        switching = _optional_power_value(report_text, "Net Switching Power")
        internal = _optional_power_value(report_text, "Cell Internal Power")
        window = float(row["window_ns"])
        row.update({
            "status": "complete", "dynamic_power_mw": dynamic,
            "leakage_power_mw": leakage,
            "switching_power_mw": "" if switching is None else switching,
            "internal_power_mw": "" if internal is None else internal,
            "window_dynamic_energy_pj": dynamic * window,
            "incremental_energy_pj": "", "pwr_414": int(pwr_414),
            "pwr_415": int(pwr_415),
            "failure_reason": "",
            **_power_group_fields(report_text, window),
        })
        keep = point.holdout or artifact.pattern in {"representative-qwen", "mixed-kernel-holdout"}
        if keep:
            gz = artifact.vcd.with_suffix(artifact.vcd.suffix + ".gz")
            with artifact.vcd.open("rb") as source, gzip.open(gz, "wb", compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            artifact.vcd.unlink()
            row["vcd_path"] = str(gz)
            row["vcd_retention"] = "gzip_validation"
        else:
            artifact.vcd.unlink(missing_ok=True)
            row["vcd_retention"] = "hash_sidecar_only"
    except Exception as exc:
        row["failure_reason"] = repr(exc)
        row["pwr_414"] = int(pwr_414 or "PWR-414" in str(exc))
        row["pwr_415"] = int(pwr_415 or "PWR-415" in str(exc))
        row["vcd_retention"] = "raw_failed"
    finally:
        row["replay_end_time"] = _utc_now()
        row["replay_elapsed_sec"] = round(time.monotonic() - started, 3)
        row["dc_power_peak_rss_kib"] = "" if peak is None else peak
        with csv_lock:
            append_row(run_dir / "power_calibration_points.csv", row, POWER_FIELDS)
        cpu_gate.release(cpu_token)
        gate.release("power", token, peak_rss_kib=peak)
    return row


def _resolve_count(value: str, default: int) -> int:
    return default if value == "auto" else max(1, int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--plan-version", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--reuse-mapping-run",
        type=Path,
        help="Read-only archived run providing compatible mapped DDCs.",
    )
    parser.add_argument("--map-workers", default="auto")
    parser.add_argument("--activity-workers", default="auto")
    parser.add_argument("--heavy-activity-workers", type=int, default=4)
    parser.add_argument("--power-workers", default="auto")
    parser.add_argument("--reserve-licenses", type=int, default=1)
    parser.add_argument("--memory-reserve-gib", type=float, default=16.0)
    parser.add_argument("--tmp-reserve-gib", type=float, default=8.0)
    parser.add_argument("--cpu-capacity", type=int, default=60)
    parser.add_argument("--verilator-jobs", type=int, default=4)
    parser.add_argument(
        "--component",
        choices=("matrix", "vector", "scalar", "control", "hbm", "agu", "all"),
        default="all",
    )
    parser.add_argument(
        "--point-id-regex",
        help="Run only mapped points whose point_id matches this regular expression.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scenario-regex",
        help="Run only activity scenario names matching this regular expression (debug/smoke only).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-power-replay", action="store_true",
        help="Regenerate activity and append a new replay attempt even when the latest row is complete.",
    )
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument(
        "--copy-to-calibration", action=argparse.BooleanOptionalAction, default=False,
        help="Publish the latest-complete compact CSV to analytic_models/power/calibration.",
    )
    parser.add_argument("--license-retry-wait-sec", type=float, default=60.0)
    parser.add_argument("--license-max-retries", type=int, default=0)
    parser.add_argument(
        "--min-sequential-saif-coverage-pct", type=float, default=90.0,
        help=(
            "Treat PWR-415 as critical below this mapped sequential-cell "
            "coverage. The warning remains recorded above the threshold."
        ),
    )
    args = parser.parse_args()

    # Stage commands change cwd into disposable RTL copies. Normalize every
    # persisted root once so logs, reports, and resume manifests never become
    # relative to a worker directory.
    args.run_dir = args.run_dir.resolve()
    args.rtl_root = args.rtl_root.resolve()
    args.worker_root = args.worker_root.resolve()
    if args.reuse_mapping_run is not None:
        args.reuse_mapping_run = args.reuse_mapping_run.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    runner_locks = _acquire_runner_locks(args.run_dir, args.worker_root)
    atexit.register(_release_runner_locks, runner_locks)
    _install_interrupt_handler()
    if not args.keep_workers:
        # Covers normal completion and uncaught KeyboardInterrupt/exception.
        # Per-point build cleanup remains immediate; this is only the final
        # best-effort guard for copied RTL workers.
        atexit.register(shutil.rmtree, args.worker_root, ignore_errors=True)

    if args.component == "agu":
        points = build_agu_plan()
    else:
        plan_builder = build_plan_v2 if args.plan_version == "v2" else build_plan
        points = [
            point
            for point in plan_builder()
            if args.component == "all" or point.component == args.component
        ]
    if args.point_id_regex:
        point_matcher = re.compile(args.point_id_regex)
        points = [
            point for point in points if point_matcher.search(point.point_id)
        ]
        if not points:
            raise ValueError(
                f"--point-id-regex matched no points: {args.point_id_regex!r}"
            )
    if args.limit is not None:
        points = points[: args.limit]
    scenarios_by_point: dict[str, list[tuple[str, str, int, str]]] = {}
    for point in points:
        scenarios = (
            scenarios_for_point_v2(point)
            if args.plan_version == "v2"
            else [(name, pattern, repeats, "mixed") for name, pattern, repeats in SCENARIOS]
        )
        scenarios_by_point[point.point_key] = scenarios
    if args.scenario_regex:
        matcher = re.compile(args.scenario_regex)
        scenarios_by_point = {
            point_key: [scenario for scenario in scenarios if matcher.search(scenario[0])]
            for point_key, scenarios in scenarios_by_point.items()
        }
        if not any(scenarios_by_point.values()):
            raise ValueError(f"--scenario-regex matched no scenarios: {args.scenario_regex!r}")
    _write_plan(points, args.run_dir, scenarios_by_point=scenarios_by_point)
    build_sram_energy_catalog(output=DEFAULT_CATALOG)

    available_mem = _mem_available_gib()
    available_tmp = shutil.disk_usage("/tmp").free / GIB
    usable_memory = max(0.0, available_mem - args.memory_reserve_gib)
    default_map = max(1, min(8, int(usable_memory // 10.5)))
    default_activity = max(1, min(6, int(usable_memory // 5.0)))
    dc_free = resolve_dc_worker_count("auto", repo_root=REPO_ROOT)
    dc_capacity = max(1, dc_free - args.reserve_licenses)
    default_power = max(1, min(11, dc_capacity, int(max(0.0, available_mem - args.memory_reserve_gib) // 2.5)))
    map_workers = _resolve_count(args.map_workers, default_map)
    activity_workers = _resolve_count(args.activity_workers, default_activity)
    power_workers = _resolve_count(args.power_workers, default_power)
    power_workers = min(power_workers, dc_capacity)
    expected_jobs = sum(len(scenarios) for scenarios in scenarios_by_point.values())
    summary: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "running",
        "plan_version": args.plan_version,
        "points": len(points), "activity_jobs": expected_jobs,
        "scenario_counts_by_point": {
            point.point_id: len(scenarios_by_point[point.point_key]) for point in points
        },
        "map_workers": map_workers, "activity_workers": activity_workers,
        "heavy_activity_workers": args.heavy_activity_workers,
        "power_workers": power_workers, "dc_capacity_after_reserve": dc_capacity,
        "cpu_capacity": args.cpu_capacity,
        "reuse_mapping_run": "" if args.reuse_mapping_run is None else str(args.reuse_mapping_run),
        "mem_available_gib": available_mem, "tmp_available_gib": available_tmp,
        "gate_level_validation": "not_run_by_scope",
    }
    (args.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    if available_tmp < args.tmp_reserve_gib:
        raise RuntimeError(f"/tmp free space {available_tmp:.1f} GiB is below reserve {args.tmp_reserve_gib:.1f} GiB")
    if available_mem < args.memory_reserve_gib:
        raise RuntimeError(f"MemAvailable {available_mem:.1f} GiB is below reserve {args.memory_reserve_gib:.1f} GiB")

    gate = ResourceGate(memory_reserve_gib=args.memory_reserve_gib, tmp_reserve_gib=args.tmp_reserve_gib)
    cpu_gate = WeightedSemaphore(args.cpu_capacity)
    license_sem = threading.Semaphore(dc_capacity)
    csv_lock = threading.Lock()
    heavy_activity_sem = threading.Semaphore(max(1, args.heavy_activity_workers))
    power_futures: list[Future[dict[str, Any]]] = []
    power_futures_lock = threading.Lock()
    backfilled_power_rows = _backfill_power_group_fields(
        args.run_dir / "power_calibration_points.csv"
    )
    summary["power_group_rows_backfilled"] = backfilled_power_rows
    (args.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mapping_latest = _read_latest(args.run_dir / "mapping_points.csv", ("point_key",))
    power_latest = _read_latest(args.run_dir / "power_calibration_points.csv", ("point_key", "scenario"))
    failures: list[str] = []

    def map_point(index_point: tuple[int, PowerPoint]) -> tuple[int, PowerPoint, Path, dict[str, Any]]:
        index, point = index_point
        manifest = _mapping_manifest(point, args.run_dir) if args.resume else None
        if manifest is None and args.resume:
            manifest = _reuse_mapping_manifest(
                point, args.reuse_mapping_run, args.run_dir
            )
        worker = _ensure_worker(point, index, args.worker_root, args.rtl_root)
        if manifest is not None:
            if mapping_latest.get((point.point_key,), {}).get("status") != "complete":
                recovered = {
                    "point_id": point.point_id,
                    "point_key": point.point_key,
                    "component": point.component,
                    "module": point.module,
                    "top_module": point.top_module,
                    "holdout": int(point.holdout),
                    "status": "complete",
                    "worker_id": index,
                    "start_time": _utc_now(),
                    "end_time": _utc_now(),
                    "elapsed_sec": 0.0,
                    "peak_rss_kib": "",
                    "failure_reason": "",
                    **manifest,
                }
                with csv_lock:
                    append_row(
                        args.run_dir / "mapping_points.csv", recovered, MAPPING_FIELDS
                    )
            return index, point, worker, manifest
        kind = _point_weight(point, "map")
        token = gate.acquire(kind)
        cpu_token = cpu_gate.acquire(4)
        start = _utc_now()
        started = time.monotonic()
        log_dir = args.run_dir / "command_logs" / point.point_key / "mapping"
        log_dir.mkdir(parents=True, exist_ok=True)
        time_report = log_dir / "synth.time"
        peak = None
        row = {
            "point_id": point.point_id, "point_key": point.point_key,
            "component": point.component, "module": point.module,
            "top_module": point.top_module, "holdout": int(point.holdout),
            "worker_id": index, "start_time": start, "status": "failed",
        }
        try:
            shell = (
                f"cd {shlex.quote(str(worker))} && SYNTH_MAX_CORES=4 "
                f"/usr/bin/time -v -o {shlex.quote(str(time_report))} "
                f"just synth {shlex.quote(point.top_module)} 1000 normal"
            )
            mapping_command = (
                ["bash", "-lc", shell]
                if os.environ.get("IN_NIX_SHELL")
                else ["nix", "develop", str(args.rtl_root), "--command", "bash", "-lc", shell]
            )
            proc = _run_dc_retry(
                mapping_command,
                cwd=args.rtl_root, log=log_dir / "synth.log",
                wait_sec=args.license_retry_wait_sec,
                max_retries=args.license_max_retries,
                license_sem=license_sem,
            )
            peak = _peak_rss(time_report)
            if proc.returncode != 0:
                raise RuntimeError(f"mapping failed with exit {proc.returncode}")
            manifest = _archive_mapping(point, worker, args.run_dir)
            row.update(manifest)
            row.update({"status": "complete", "failure_reason": ""})
        except Exception as exc:
            row["failure_reason"] = repr(exc)
            raise
        finally:
            row.update({
                "end_time": _utc_now(), "elapsed_sec": round(time.monotonic() - started, 3),
                "peak_rss_kib": peak or "",
            })
            with csv_lock:
                append_row(args.run_dir / "mapping_points.csv", row, MAPPING_FIELDS)
            shutil.rmtree(worker / "build/synth" / point.top_module, ignore_errors=True)
            cpu_gate.release(cpu_token)
            gate.release(kind, token, peak_rss_kib=peak)
        assert manifest is not None
        return index, point, worker, manifest

    with ThreadPoolExecutor(max_workers=power_workers, thread_name_prefix="power") as power_pool, ThreadPoolExecutor(
        max_workers=activity_workers, thread_name_prefix="activity"
    ) as activity_pool, ThreadPoolExecutor(max_workers=map_workers, thread_name_prefix="map") as map_pool:
        activity_futures: list[Future[list[ActivityArtifact]]] = []

        def submit_activity(mapped: tuple[int, PowerPoint, Path, dict[str, Any]]) -> Future[list[ActivityArtifact]]:
            index, point, worker, mapping = mapped
            selected_scenarios = scenarios_by_point[point.point_key]
            # Resume is valid only for the exact RTL/config/harness semantics.
            # Computing the same fingerprint used by the activity generator
            # lets a stimulus fix invalidate stale successful rows without a
            # global --force-power-replay.
            _, expected_activity_fingerprint = install_harness(
                point, worker, REPO_ROOT
            )
            pending = [
                scenario for scenario in selected_scenarios
                if not (
                    args.resume
                    and not args.force_power_replay
                    and _resume_row_is_current(
                        power_latest.get((point.point_key, scenario[0]), {}),
                        scenario,
                        expected_activity_fingerprint,
                    )
                )
            ]

            def run_activity() -> list[ActivityArtifact]:
                if not pending:
                    if not args.keep_workers:
                        shutil.rmtree(args.worker_root / f"worker_{index}", ignore_errors=True)
                    return []
                kind = _point_weight(point, "activity")
                token = gate.acquire(kind)
                cpu_token = cpu_gate.acquire(args.verilator_jobs)

                def submit_power(artifact: ActivityArtifact) -> None:
                    future = power_pool.submit(
                        _power_replay, point=point, artifact=artifact, mapping=mapping,
                        run_dir=args.run_dir, rtl_root=args.rtl_root, gate=gate,
                        cpu_gate=cpu_gate,
                        license_sem=license_sem, csv_lock=csv_lock,
                        retry_wait=args.license_retry_wait_sec,
                        max_retries=args.license_max_retries,
                        min_seq_coverage_pct=args.min_sequential_saif_coverage_pct,
                    )
                    with power_futures_lock:
                        power_futures.append(future)

                artifacts: list[ActivityArtifact] = []
                peak = None
                try:
                    if point.component in {"vector", "hbm"}:
                        heavy_activity_sem.acquire()
                    try:
                        artifacts = generate_activity_scenarios(
                            point=point, worker_rtl=worker, source_rtl=args.rtl_root,
                            repo_root=REPO_ROOT, run_dir=args.run_dir,
                            scenarios=pending, verilator_jobs=args.verilator_jobs,
                            on_artifact=submit_power,
                        )
                    finally:
                        if point.component in {"vector", "hbm"}:
                            heavy_activity_sem.release()
                    peaks = [artifact.peak_rss_kib for artifact in artifacts if artifact.peak_rss_kib]
                    peak = max(peaks) if peaks else None
                    return artifacts
                finally:
                    cpu_gate.release(cpu_token)
                    gate.release(kind, token, peak_rss_kib=peak)
                    if not args.keep_workers:
                        shutil.rmtree(args.worker_root / f"worker_{index}", ignore_errors=True)

            return activity_pool.submit(run_activity)

        map_futures = {map_pool.submit(map_point, item): item[1] for item in enumerate(points)}
        for future in as_completed(map_futures):
            point = map_futures[future]
            try:
                activity_futures.append(submit_activity(future.result()))
            except Exception as exc:
                failures.append(f"mapping:{point.point_id}:{exc!r}")

        for future in as_completed(activity_futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(f"activity:{exc!r}")

        with power_futures_lock:
            pending_power = list(power_futures)
        for future in as_completed(pending_power):
            try:
                row = future.result()
                if row.get("status") != "complete":
                    failures.append(f"power:{row.get('point_id')}/{row.get('scenario')}:{row.get('failure_reason')}")
            except Exception as exc:
                failures.append(f"power-exception:{exc!r}")

    if not args.keep_workers:
        shutil.rmtree(args.worker_root, ignore_errors=True)
    complete_rows = _read_latest(args.run_dir / "power_calibration_points.csv", ("point_key", "scenario"))
    selected_point_keys = {point.point_key for point in points}
    selected_jobs = {
        (point_key, scenario[0])
        for point_key, scenarios in scenarios_by_point.items()
        for scenario in scenarios
    }
    completed = sum(
        row.get("status") == "complete"
        for (point_key, scenario), row in complete_rows.items()
        if point_key in selected_point_keys and (point_key, scenario) in selected_jobs
    )
    summary.update({
        "status": "complete" if not failures and completed == expected_jobs else "incomplete",
        "completed_activity_jobs": completed,
        "completed_power_jobs": completed,
        "expected_activity_jobs": expected_jobs,
        "expected_power_jobs": expected_jobs,
        "failures": failures,
        "resource_gate": gate.snapshot(),
        "cpu_gate": cpu_gate.snapshot(),
        "final_mem_available_gib": _mem_available_gib(),
        "final_tmp_available_gib": shutil.disk_usage("/tmp").free / GIB,
    })
    (args.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source_csv = args.run_dir / "power_calibration_points.csv"
    if source_csv.exists():
        compact = args.run_dir / "compact/power_calibration_points.csv"
        summary["compact_rows"] = _export_latest_complete(source_csv, compact)
        summary["compact_csv"] = str(compact)
        if args.copy_to_calibration and summary["status"] == "complete":
            destination_name = (
                "power_calibration_points_v2.csv"
                if args.plan_version == "v2"
                else "power_calibration_points.csv"
            )
            destination = REPO_ROOT / "analytic_models/power/calibration" / destination_name
            _export_latest_complete(source_csv, destination)
            summary["published_calibration_csv"] = str(destination)
        elif args.copy_to_calibration:
            summary["calibration_csv_publication"] = (
                "skipped because the selected run is incomplete"
            )
        (args.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "completed_activity_jobs", "expected_activity_jobs", "failures")}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
