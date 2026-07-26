#!/usr/bin/env python3
"""Audit and remove stale PLENA-generated temporary directories.

The cleanup is deliberately allow-list based.  It never removes model caches,
opaque temporary directories, paths owned by another user, or paths referenced
by a process visible through ``/proc``.  A JSON manifest is always written so a
calibration run has an auditable record of the disk-space preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = (
    "plena_rtl_power_workers*",
    "plena_rtl_area_workers*",
    "area_new_power_*",
    "plena_other_*",
    "plena_txn_codex_target*",
    "qwen3_vs_sharded_*",
    "native_compact_layout_*",
    "qwen3_moe_vs_*",
    "roofline_v4_grid_*",
    "plena_mixed_precision_*",
    "plena_kv_*",
    "dse_*",
)

PRESERVED_NAMES = {"moe_models_e13", "bc625"}


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _tree_size(path: Path) -> int:
    total = 0
    try:
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _active_paths(tmp_root: Path) -> set[Path]:
    active: set[Path] = set()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        for candidate in (entry / "cwd", entry / "root"):
            try:
                resolved = candidate.resolve(strict=True)
                if resolved != tmp_root and tmp_root in resolved.parents:
                    active.add(resolved)
            except (OSError, PermissionError):
                pass
        fd_dir = entry / "fd"
        try:
            fds = tuple(fd_dir.iterdir())
        except (OSError, PermissionError):
            continue
        for fd in fds:
            try:
                resolved = fd.resolve(strict=True)
                if resolved != tmp_root and tmp_root in resolved.parents:
                    active.add(resolved)
            except (OSError, PermissionError):
                pass
    return active


def _is_active(path: Path, active: set[Path]) -> bool:
    resolved = path.resolve()
    return any(item == resolved or resolved in item.parents or item in resolved.parents for item in active)


def cleanup(
    *,
    tmp_root: Path,
    manifest: Path,
    apply: bool,
    min_age_hours: float,
) -> dict[str, Any]:
    now = time.time()
    uid = os.getuid()
    disk_before = _disk(tmp_root)
    active = _active_paths(tmp_root.resolve())
    candidates = {path for pattern in DEFAULT_PATTERNS for path in tmp_root.glob(pattern)}
    records: list[dict[str, Any]] = []
    for path in sorted(candidates):
        record: dict[str, Any] = {"path": str(path), "action": "skipped"}
        try:
            stat = path.lstat()
            record.update(
                owner_uid=stat.st_uid,
                mtime=stat.st_mtime,
                age_hours=(now - stat.st_mtime) / 3600.0,
                size_bytes=_tree_size(path) if path.is_dir() else stat.st_size,
            )
            referenced = _is_active(path, active)
            record["open_file_state"] = (
                "referenced_by_active_process" if referenced else "not_referenced"
            )
            if path.name in PRESERVED_NAMES:
                record["reason"] = "explicitly_preserved"
            elif stat.st_uid != uid:
                record["reason"] = "different_owner"
            elif record["age_hours"] < min_age_hours:
                record["reason"] = "too_recent"
            elif referenced:
                record["reason"] = "active_process_reference"
            elif not apply:
                record.update(action="would_delete", reason="dry_run")
            else:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                record.update(action="deleted", reason="allowlisted_stale_path")
        except OSError as exc:
            record["reason"] = f"filesystem_error: {exc}"
        records.append(record)

    payload = {
        "schema_version": 1,
        "timestamp": time.time(),
        "tmp_root": str(tmp_root),
        "apply": apply,
        "min_age_hours": min_age_hours,
        "preserved_names": sorted(PRESERVED_NAMES),
        "patterns": list(DEFAULT_PATTERNS),
        "disk_before": disk_before,
        "records": records,
    }
    # disk_after captures the actual release before the manifest itself is
    # written. The manifest therefore reports only cleanup effects.
    payload["disk_after"] = _disk(tmp_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-age-hours", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload = cleanup(
        tmp_root=args.tmp_root,
        manifest=args.manifest,
        apply=args.apply,
        min_age_hours=args.min_age_hours,
    )
    deleted = [row for row in payload["records"] if row["action"] == "deleted"]
    would_delete = [row for row in payload["records"] if row["action"] == "would_delete"]
    print(
        json.dumps(
            {
                "deleted_paths": len(deleted),
                "deleted_bytes": sum(int(row.get("size_bytes", 0)) for row in deleted),
                "would_delete_paths": len(would_delete),
                "free_bytes": payload["disk_after"]["free_bytes"],
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
