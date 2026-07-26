"""Linux resource telemetry used by parallel DSE workers."""

from __future__ import annotations

import math
import os
import resource
from pathlib import Path


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024**2)


def current_process_rss_gib() -> float:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return peak_rss_gib()
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024**3)


def system_cpu_jiffies() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total - idle, total


def mem_available_gib() -> float:
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024**2)
    raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def _process_tree_pids(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    visited: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            children = Path(
                f"/proc/{pid}/task/{pid}/children"
            ).read_text().split()
        except (FileNotFoundError, OSError):
            children = []
        pending.extend(int(child) for child in children)
    return visited


def process_tree_rss_gib(root_pid: int) -> float:
    resident_pages = 0
    for pid in _process_tree_pids(root_pid):
        try:
            resident_pages += int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024**3)


def process_tree_cpu_seconds(root_pid: int) -> float:
    ticks = 0
    for pid in _process_tree_pids(root_pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            ticks += int(fields[13]) + int(fields[14])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
