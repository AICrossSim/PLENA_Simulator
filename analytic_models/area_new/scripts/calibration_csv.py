"""Durable CSV operations for resumable, multi-hour calibration campaigns.

Raw calibration CSVs are append-only attempt logs. A job may therefore have a
failed row followed by a successful retry. ``latest_jobs.csv`` and compact
exports are derived views; they must never destroy the raw history needed for
debugging license, RTL, or resource failures.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

COMMON_FIELDS = [
    "job_key",
    "point_key",
    "point_id",
    "runner",
    "module",
    "top_module",
    "status",
    "attempt",
    "worker_id",
    "start_time",
    "end_time",
    "elapsed_sec",
    "area_um2",
    "dynamic_power",
    "leakage_power",
    "total_power",
    "report_dir",
    "summary_log",
    "command_log_dir",
    "failure_class",
    "failure_reason",
    "source_plan",
]


def union_fields(*field_groups: Iterable[str]) -> list[str]:
    """Return a stable-order union of multiple CSV schemas."""
    fields: list[str] = []
    for group in field_groups:
        for field in group:
            if field not in fields:
                fields.append(field)
    return fields


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows, treating a missing file as an empty resumable run."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Atomically rewrite a derived CSV view with all encountered columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = union_fields(fields, *(row.keys() for row in rows))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_row(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    """Append one attempt while allowing schema evolution across old runs.

    When a new field appears, the existing file is rewritten once with the
    expanded header; otherwise the normal path is a cheap append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        existing_fields = list(csv.DictReader(path.open(newline="")).fieldnames or [])
        if any(field not in existing_fields for field in row):
            rows = read_rows(path)
            rows.append({key: str(value) for key, value in row.items()})
            write_rows(path, rows, union_fields(existing_fields, fields, row.keys()))
            return
        fieldnames = existing_fields
    else:
        fieldnames = union_fields(fields, row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def latest_by_key(rows: Iterable[dict[str, Any]], key_field: str = "job_key") -> dict[str, dict[str, Any]]:
    """Select the last observed attempt for each stable job key."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_field, ""))
        if not key:
            continue
        latest[key] = dict(row)
    return latest


def latest_complete_rows(
    rows: Iterable[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Select the latest successful row for each semantic configuration."""
    latest: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = key_fn(row)
        if key is None:
            continue
        latest[key] = dict(row)
    return list(latest.values())


def write_latest_jobs(raw_csv: Path, latest_csv: Path, fields: list[str]) -> list[dict[str, Any]]:
    """Regenerate ``latest_jobs.csv`` from the append-only attempt log."""
    latest = list(latest_by_key(read_rows(raw_csv)).values())
    write_rows(latest_csv, latest, fields)
    return latest
