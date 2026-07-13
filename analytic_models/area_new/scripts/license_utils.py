"""Helpers for sizing Synopsys Design Compiler worker pools."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_OBSERVED_DC_LICENSES = 12


def is_dc_license_unavailable_text(text: str) -> bool:
    """Return true when Synopsys reports temporary DC license exhaustion."""
    patterns = [
        "SEC-50",
        "Design-Compiler licenses are in use",
        "Unable to obtain license",
        "Unable to checkout license",
        "Cannot obtain license",
    ]
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _parse_lmstat(text: str) -> tuple[int, int] | None:
    patterns = [
        r"Users of Design-Compiler:\s*\(Total of\s+(\d+)\s+licenses? issued;\s*Total of\s+(\d+)\s+licenses? in use\)",
        r"Users of Design-Compiler\s*:.*?Total of\s+(\d+)\s+licenses? issued.*?Total of\s+(\d+)\s+licenses? in use",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _query_lmstat() -> tuple[int, int] | None:
    license_file = os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")
    candidates: list[list[str]] = []
    lmutil = shutil.which("lmutil")
    lmstat = shutil.which("lmstat")
    if lmutil:
        cmd = [lmutil, "lmstat", "-a"]
        if license_file:
            cmd += ["-c", license_file]
        candidates.append(cmd)
    if lmstat:
        cmd = [lmstat, "-a"]
        if license_file:
            cmd += ["-c", license_file]
        candidates.append(cmd)

    for cmd in candidates:
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=20, check=False)
        except Exception:
            continue
        parsed = _parse_lmstat(f"{result.stdout}\n{result.stderr}")
        if parsed:
            return parsed
    return None


def _parse_recent_sec50_logs(search_root: Path) -> int | None:
    if not search_root.exists():
        return None
    newest: tuple[float, Path] | None = None
    for path in search_root.rglob("*.stdout.log"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if newest is None or stat.st_mtime > newest[0]:
            newest = (stat.st_mtime, path)
    for path in search_root.rglob("*.stderr.log"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if newest is None or stat.st_mtime > newest[0]:
            newest = (stat.st_mtime, path)
    if newest is None:
        return None
    try:
        text = newest[1].read_text(errors="ignore")
    except OSError:
        return None
    if "SEC-50" not in text and "Design-Compiler" not in text:
        return None
    users = re.findall(r"^\S+\s+at\s+\S+,\s+started on\s+", text, flags=re.MULTILINE)
    return len(users) or None


def resolve_dc_worker_count(workers: str | int, *, repo_root: Path, default: int = DEFAULT_OBSERVED_DC_LICENSES) -> int:
    """Resolve a worker count from an integer or the string ``auto``.

    Auto mode tries live FlexNet status first. If license tooling is unavailable,
    it falls back to PLENA_DC_LICENSE_TOTAL, recent SEC-50 logs, then the observed
    local default. PLENA_DC_LICENSE_RESERVE can keep licenses free for others.
    PLENA_DC_LICENSE_MAX_WORKERS can cap the final value.
    """
    if isinstance(workers, int) or str(workers).strip().lower() != "auto":
        return max(1, int(workers))

    reserve = max(0, int(os.environ.get("PLENA_DC_LICENSE_RESERVE", "0")))
    max_workers_env = os.environ.get("PLENA_DC_LICENSE_MAX_WORKERS")
    live = _query_lmstat()
    if live:
        total, used = live
        count = max(1, total - used - reserve)
        source = f"lmstat total={total} used={used}"
    else:
        total_env = os.environ.get("PLENA_DC_LICENSE_TOTAL")
        if total_env:
            total = int(total_env)
            source = "PLENA_DC_LICENSE_TOTAL"
        else:
            total = _parse_recent_sec50_logs(repo_root / "Workspace") or default
            source = "recent SEC-50 logs" if total != default else "observed local default"
        count = max(1, total - reserve)

    if max_workers_env:
        count = min(count, max(1, int(max_workers_env)))
    print(f"Resolved --workers auto to {count} Design Compiler workers ({source}, reserve={reserve})")
    return count
