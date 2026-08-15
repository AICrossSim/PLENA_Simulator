from __future__ import annotations

import pytest

from analytic_models.dse.workers import auto_worker_resource_policy


def test_workstation_policy_preserves_legacy_limits() -> None:
    policy = auto_worker_resource_policy(logical_cpus=64, memory_gib=128.0)

    assert policy.profile == "workstation-v1"
    assert policy.worker_cap == 64
    assert policy.launch_burst == 8
    assert policy.launch_interval_seconds == 2.0
    assert policy.monitor_interval_seconds == 0.2


def test_large_server_policy_uses_all_logical_cpus() -> None:
    policy = auto_worker_resource_policy(logical_cpus=288, memory_gib=2300.0)

    assert policy.profile == "large-shared-server-full-cpu-v2"
    assert policy.reserved_logical_cpus == 0
    assert policy.worker_cap == 288
    assert policy.launch_burst == 32
    assert policy.launch_interval_seconds == 0.5
    assert policy.monitor_interval_seconds == 1.0
    assert policy.emergency_gib < policy.launch_reserve_gib <= policy.resume_gib


@pytest.mark.parametrize(
    ("logical_cpus", "memory_gib"),
    [(0, 128.0), (64, 0.0), (-1, 128.0)],
)
def test_policy_rejects_invalid_resource_detection(
    logical_cpus: int,
    memory_gib: float,
) -> None:
    with pytest.raises(ValueError):
        auto_worker_resource_policy(
            logical_cpus=logical_cpus,
            memory_gib=memory_gib,
        )
