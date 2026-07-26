"""Resource policy for the resumable multi-process DSE worker pool."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_RESERVED_LOGICAL_CPUS = 0
DEFAULT_WORKER_CAP = max(
    1,
    min(64, (os.cpu_count() or 1) - DEFAULT_RESERVED_LOGICAL_CPUS),
)


@dataclass(frozen=True)
class WorkerResourcePolicy:
    worker_cap: int = DEFAULT_WORKER_CAP
    max_trials_per_process: int = 0
    rss_recycle_gib: float = 2.5
    initial_rss_gib: float = 1.5
    launch_reserve_gib: float = 22.0
    resume_gib: float = 26.0
    emergency_gib: float = 18.0
    process_tree_limit_gib: float = 6.0
    stall_timeout_seconds: float = 900.0

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_WORKER_POLICY = WorkerResourcePolicy()
