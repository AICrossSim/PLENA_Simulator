#!/usr/bin/env python3
"""Merge completed attempts from an auxiliary run into a resumable run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from calibration_csv import read_rows, union_fields, write_latest_jobs, write_rows
except ModuleNotFoundError:
    from .calibration_csv import read_rows, union_fields, write_latest_jobs, write_rows


def stable_key(row: dict[str, str]) -> str:
    """Return the stable identifier used by either calibration runner."""

    return str(row.get("job_key") or row.get("point_key") or "")


def merge_completed(source: Path, target: Path) -> int:
    """Append successful source jobs that are not already successful in target."""

    source_rows = read_rows(source)
    target_rows = read_rows(target)
    completed = {
        stable_key(row)
        for row in target_rows
        if stable_key(row) and row.get("status") == "complete"
    }
    additions = [
        row
        for row in source_rows
        if row.get("status") == "complete"
        and stable_key(row)
        and stable_key(row) not in completed
    ]
    if not additions:
        return 0
    fields = union_fields(
        list(csv.DictReader(target.open(newline="")).fieldnames or [])
        if target.exists()
        else [],
        *(row.keys() for row in source_rows),
    )
    write_rows(target, target_rows + additions, fields)
    write_latest_jobs(target, target.parent / "latest_jobs.csv", fields)
    return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    count = merge_completed(args.source, args.target)
    print(f"merged {count} completed calibration attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
