"""Shared execution engine for parallel Synopsys DC area calibration.

Adapters own module-specific RTL patching and report parsing. This runtime owns
the cross-cutting guarantees: deterministic job identity, isolated /tmp worker
copies, license-aware concurrency, append-only result logging, resumability,
artifact retention, and best-effort cleanup after success or interruption.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from calibration_csv import COMMON_FIELDS, append_row, read_rows, union_fields, write_latest_jobs, write_rows
    from license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
    from run_matrix_machine_calibration import cleanup_workers, create_worker_copy, json_safe
except ModuleNotFoundError:
    from .calibration_csv import COMMON_FIELDS, append_row, read_rows, union_fields, write_latest_jobs, write_rows
    from .license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
    from .run_matrix_machine_calibration import cleanup_workers, create_worker_copy, json_safe


class CalibrationAdapter(Protocol):
    """Interface implemented by each module-specific calibration adapter."""
    name: str
    row_fields: list[str]

    def run_point(
        self,
        point: Any,
        worker_id: int,
        worker_rtl: Path,
        rtl_root: Path,
        run_dir: Path,
        cleanup_builds: bool,
        license_retry_wait_sec: float,
        license_max_retries: int,
    ) -> dict[str, Any]:
        ...

    def compact_exports(self) -> list["CompactExport"]:
        ...


@dataclass(frozen=True)
class CalibrationJob:
    """One immutable point plus the adapter and plan provenance that created it."""
    job_key: str
    runner: str
    point: Any
    adapter: CalibrationAdapter
    source_plan: str


@dataclass(frozen=True)
class CompactExport:
    """Rules for producing a module-local successful-point CSV view."""
    name: str
    fields: list[str]
    predicate: Callable[[dict[str, Any]], bool]
    key_fn: Callable[[dict[str, Any]], Any]


@dataclass
class RuntimeConfig:
    """Execution, retry, license reservation, and cleanup policy for one run."""
    run_dir: Path
    rtl_root: Path
    worker_root: Path
    workers: str | int = "auto"
    reserve: int = 1
    cleanup_worker_builds: bool = True
    keep_workers: bool = False
    resume: bool = True
    skip_failed: bool = False
    retry_failed: bool = False
    license_retry_wait_sec: float = 60.0
    license_max_retries: int = 0


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for portable attempt records."""
    return datetime.now(timezone.utc).isoformat()


def stable_job_key(runner: str, point: Any) -> str:
    """Hash all behavior-affecting point fields into a restart-stable key."""
    payload = {
        "runner": runner,
        "point_id": getattr(point, "point_id", ""),
        "module": getattr(point, "module", ""),
        "top_module": getattr(point, "top_module", ""),
        "level": getattr(point, "level", ""),
        "mode": getattr(point, "mode", ""),
        "params": getattr(point, "params", {}),
    }
    import hashlib

    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    safe_runner = re.sub(r"[^A-Za-z0-9_]+", "_", runner).strip("_")
    return f"{safe_runner}_{digest}"


def classify_failure(row: dict[str, Any], stdout: str = "", stderr: str = "") -> str:
    """Map heterogeneous DC/RTL/runtime errors into a small diagnostic taxonomy."""
    reason = str(row.get("failure_reason", ""))
    text = "\n".join([reason, stdout, stderr])
    if str(row.get("status")) == "complete":
        return ""
    if is_dc_license_unavailable_text(text):
        return "license_busy"
    if re.search(r"Received Signal\s+15|Process terminated by kill|SIGTERM", text, re.IGNORECASE):
        return "killed_by_signal"
    if re.search(r"Array index out of bounds|Presto compilation terminated|ERROR:\s+link failed|unresolved references", text, re.IGNORECASE):
        return "rtl_elaboration_error"
    if "report" in reason.lower() and "missing" in reason.lower():
        return "report_missing"
    if "ValueError" in reason or "unsupported" in reason.lower():
        return "config_error"
    if "exception" in reason.lower() or "Traceback" in text:
        return "python_exception"
    if "interrupted" in reason.lower():
        return "interrupted"
    return "synth_failed"


def _read_log_tail_text(log_dir: Path | None, *, max_bytes_per_file: int = 256_000) -> str:
    if log_dir is None or not log_dir.exists():
        return ""
    chunks: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                if size > max_bytes_per_file:
                    f.seek(max(0, size - max_bytes_per_file))
                chunks.append(f.read(max_bytes_per_file).decode(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _copy_job_artifacts(run_dir: Path, source_key: str, job_key: str, row: dict[str, Any]) -> dict[str, Any]:
    if source_key and source_key != job_key:
        report_dir = row.get("report_dir")
        if report_dir:
            src = Path(str(report_dir))
            if src.exists():
                dest = run_dir / "reports" / job_key
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
                row["report_dir"] = str(dest)
                summary = dest / "summary.log"
                if summary.exists():
                    row["summary_log"] = str(summary)

        log_dir = run_dir / "command_logs"
        dest_log_dir = log_dir / job_key
        copied = False
        for path in log_dir.glob(f"{source_key}.*.log"):
            dest_log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_log_dir / path.name.replace(source_key, job_key, 1))
            copied = True
        if copied:
            row["command_log_dir"] = str(dest_log_dir)
    return row


def normalize_result(job: CalibrationJob, row: dict[str, Any], *, start_time: str, end_time: str, attempt: int, run_dir: Path) -> dict[str, Any]:
    """Attach scheduler identity, timestamps, parameters, logs, and failure class."""
    source_key = str(row.get("point_key") or getattr(job.point, "point_key", ""))
    normalized = dict(row)
    normalized["job_key"] = job.job_key
    normalized["point_key"] = source_key
    normalized["runner"] = job.runner
    normalized["attempt"] = attempt
    normalized["start_time"] = start_time
    normalized["end_time"] = end_time
    normalized["source_plan"] = job.source_plan
    normalized = _copy_job_artifacts(run_dir, source_key, job.job_key, normalized)
    command_log_dir = normalized.get("command_log_dir")
    log_text = _read_log_tail_text(Path(str(command_log_dir)) if command_log_dir else None)
    normalized["failure_class"] = classify_failure(normalized, stdout=log_text)
    params = getattr(job.point, "params", {})
    for key, value in params.items():
        normalized.setdefault(key, value)
    return normalized


def should_skip_job(job: CalibrationJob, latest: dict[str, dict[str, Any]], *, skip_failed: bool, retry_failed: bool) -> bool:
    """Apply resume policy to one job using its most recent attempt."""
    row = latest.get(job.job_key)
    if not row:
        return False
    if row.get("status") == "complete":
        return True
    if row.get("status") == "failed" and skip_failed and not retry_failed:
        return True
    return False


def write_compact_exports(run_dir: Path, rows: list[dict[str, Any]], adapters: dict[str, CalibrationAdapter]) -> None:
    """Write latest-success module CSVs without leaking the unified raw schema."""
    compact_dir = run_dir / "compact"
    compact_dir.mkdir(parents=True, exist_ok=True)
    for adapter in adapters.values():
        for export in adapter.compact_exports():
            selected: dict[Any, dict[str, Any]] = {}
            for row in rows:
                if row.get("runner") != adapter.name or row.get("status") != "complete":
                    continue
                if not export.predicate(row):
                    continue
                selected[export.key_fn(row)] = row
            if selected:
                projected = [
                    {field: row.get(field, "") for field in export.fields}
                    for row in selected.values()
                ]
                write_rows(compact_dir / export.name, projected, export.fields)


def run_calibration_jobs(
    jobs: list[CalibrationJob],
    adapters: dict[str, CalibrationAdapter],
    config: RuntimeConfig,
) -> int:
    """Execute jobs concurrently and persist every completed attempt immediately.

    Worker copies are reused serially by each worker thread, while different
    threads synthesize in independent RTL trees. Results are written as futures
    finish, so Ctrl-C or host failure loses at most currently running points.
    """
    config.run_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = config.run_dir / "calibration_points.csv"
    latest_csv = config.run_dir / "latest_jobs.csv"
    fields = union_fields(COMMON_FIELDS, *(adapter.row_fields for adapter in adapters.values()))

    latest = {}
    if config.resume:
        latest = {str(row.get("job_key")): row for row in read_rows(latest_csv)}
        if not latest:
            latest = {str(row.get("job_key")): row for row in read_rows(raw_csv) if row.get("job_key")}
    pending = [
        job
        for job in jobs
        if not should_skip_job(job, latest, skip_failed=config.skip_failed, retry_failed=config.retry_failed)
    ]
    if not pending:
        rows = read_rows(raw_csv)
        write_latest_jobs(raw_csv, latest_csv, fields)
        write_compact_exports(config.run_dir, rows, adapters)
        _write_summary(config, jobs, rows, pending, cleanup_status="not_needed")
        print("No pending calibration jobs.")
        return 0

    if config.worker_root.exists() and not config.keep_workers:
        shutil.rmtree(config.worker_root)
    config.worker_root.mkdir(parents=True, exist_ok=True)
    os.environ["PLENA_DC_LICENSE_RESERVE"] = str(max(0, int(config.reserve)))
    worker_count = resolve_dc_worker_count(config.workers, repo_root=Path(__file__).resolve().parents[3])
    worker_paths = [create_worker_copy(idx, config.worker_root, config.rtl_root) for idx in range(worker_count)]
    worker_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in enumerate(worker_paths):
        worker_queue.put(item)
    csv_lock = threading.Lock()
    attempt_by_job: dict[str, int] = {}

    def wrapped(job: CalibrationJob) -> dict[str, Any]:
        worker_id, worker_path = worker_queue.get()
        start_time = utc_now()
        attempt = attempt_by_job.get(job.job_key, 0) + 1
        attempt_by_job[job.job_key] = attempt
        try:
            row = job.adapter.run_point(
                job.point,
                worker_id,
                worker_path,
                config.rtl_root,
                config.run_dir,
                config.cleanup_worker_builds,
                config.license_retry_wait_sec,
                config.license_max_retries,
            )
            return normalize_result(job, row, start_time=start_time, end_time=utc_now(), attempt=attempt, run_dir=config.run_dir)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            row = {
                "point_id": getattr(job.point, "point_id", ""),
                "module": getattr(job.point, "module", ""),
                "top_module": getattr(job.point, "top_module", ""),
                "status": "failed",
                "worker_id": worker_id,
                "failure_reason": repr(exc),
            }
            return normalize_result(job, row, start_time=start_time, end_time=utc_now(), attempt=attempt, run_dir=config.run_dir)
        finally:
            worker_queue.put((worker_id, worker_path))

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(wrapped, job): job for job in pending}
            for future in as_completed(futures):
                row = future.result()
                with csv_lock:
                    append_row(raw_csv, row, fields)
                    write_latest_jobs(raw_csv, latest_csv, fields)
                print(f"[{row['status']}] {row['runner']} {row['point_id']} area={row.get('area_um2', '')} reason={row.get('failure_reason', '')}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed rows already written remain resumable.", file=sys.stderr)
    finally:
        rows = read_rows(raw_csv)
        write_latest_jobs(raw_csv, latest_csv, fields)
        write_compact_exports(config.run_dir, rows, adapters)
        cleanup_status = "kept"
        if not config.keep_workers:
            cleanup_workers(config.worker_root)
            cleanup_status = "removed"
        _write_summary(config, jobs, rows, pending, cleanup_status=cleanup_status)
    return 130 if interrupted else 0


def _write_summary(config: RuntimeConfig, jobs: list[CalibrationJob], rows: list[dict[str, Any]], pending: list[CalibrationJob], *, cleanup_status: str) -> None:
    latest = list({str(row.get("job_key")): row for row in rows if row.get("job_key")}.values())
    counts: dict[str, int] = {}
    for row in latest:
        counts[str(row.get("status", ""))] = counts.get(str(row.get("status", "")), 0) + 1
    summary = {
        "run_dir": str(config.run_dir),
        "rtl_root": str(config.rtl_root),
        "worker_root": str(config.worker_root),
        "worker_cleanup": cleanup_status,
        "planned_jobs": len(jobs),
        "pending_jobs_at_start": len(pending),
        "latest_status_counts": counts,
        "unfinished_job_keys": [
            job.job_key
            for job in jobs
            if not any(row.get("job_key") == job.job_key and row.get("status") == "complete" for row in latest)
        ],
    }
    (config.run_dir / "run_summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True))
