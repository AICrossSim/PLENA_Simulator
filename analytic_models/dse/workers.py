"""Resource policy for the resumable multi-process DSE worker pool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .resources import logical_cpu_capacity, mem_total_gib


@dataclass(frozen=True)
class WorkerResourcePolicy:
    profile: str
    detected_logical_cpus: int
    detected_memory_gib: float
    reserved_logical_cpus: int
    worker_cap: int
    max_trials_per_process: int = 0
    rss_recycle_gib: float = 2.5
    initial_rss_gib: float = 1.5
    launch_reserve_gib: float = 22.0
    resume_gib: float = 26.0
    emergency_gib: float = 18.0
    process_tree_limit_gib: float = 6.0
    stall_timeout_seconds: float = 900.0
    launch_burst: int = 8
    launch_interval_seconds: float = 2.0
    monitor_interval_seconds: float = 0.2

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def auto_worker_resource_policy(
    *,
    logical_cpus: int | None = None,
    memory_gib: float | None = None,
) -> WorkerResourcePolicy:
    cpus = logical_cpu_capacity() if logical_cpus is None else logical_cpus
    memory = mem_total_gib() if memory_gib is None else memory_gib
    if cpus <= 0 or memory <= 0:
        raise ValueError("worker resource detection requires positive resources")

    if cpus >= 128 and memory >= 512.0:
        reserved_cpus = 0
        launch_reserve = max(96.0, min(256.0, memory * 0.10))
        resume = min(memory * 0.20, launch_reserve + max(32.0, memory * 0.02))
        emergency = max(64.0, min(128.0, memory * 0.05))
        return WorkerResourcePolicy(
            profile="large-shared-server-full-cpu-v2",
            detected_logical_cpus=cpus,
            detected_memory_gib=memory,
            reserved_logical_cpus=reserved_cpus,
            worker_cap=max(1, cpus - reserved_cpus),
            launch_reserve_gib=launch_reserve,
            resume_gib=max(launch_reserve, resume),
            emergency_gib=min(emergency, launch_reserve - 1.0),
            launch_burst=32,
            launch_interval_seconds=0.5,
            monitor_interval_seconds=1.0,
        )

    reserved_cpus = 0
    return WorkerResourcePolicy(
        profile="workstation-v1",
        detected_logical_cpus=cpus,
        detected_memory_gib=memory,
        reserved_logical_cpus=reserved_cpus,
        worker_cap=max(1, min(64, cpus - reserved_cpus)),
    )


def tpe_startup_worker_wave_floor(
    workers: str,
    trial_budget: int,
    *,
    worker_cap: int | None = None,
    logical_cpus: int | None = None,
) -> int:
    """Return the largest worker wave that can start against this budget."""

    if trial_budget <= 0:
        raise ValueError("TPE startup trial budget must be positive")
    cap = DEFAULT_WORKER_CAP if worker_cap is None else worker_cap
    cpus = logical_cpu_capacity() if logical_cpus is None else logical_cpus
    if cap <= 0 or cpus <= 0:
        raise ValueError("TPE startup worker capacity must be positive")
    requested_workers = min(cap, cpus) if workers == "auto" else max(1, int(workers))
    return max(1, min(requested_workers, trial_budget))


DEFAULT_WORKER_POLICY = auto_worker_resource_policy()
DEFAULT_WORKER_CAP = DEFAULT_WORKER_POLICY.worker_cap
DEFAULT_RESERVED_LOGICAL_CPUS = DEFAULT_WORKER_POLICY.reserved_logical_cpus
