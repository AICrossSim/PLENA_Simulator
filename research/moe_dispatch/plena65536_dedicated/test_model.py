"""Independent timing, true-route conservation and hard-affinity checks."""
from __future__ import annotations

import numpy as np

from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .geometry import Core
from .independent_audit import (integer_job_cycles, integer_window_cycles,
                                independent_score, validate_conservation,
                                validate_schedule)
from .model import job_cycles, schedule, cost_table


def toy_window(batch=4):
    return {"id": f"toy_{batch}", "batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": 2048, "F": 2816, "is_shared": True},
        *[{"id": i, "Me": 2, "H": 2048, "F": 1408, "is_shared": False}
          for i in range(batch * 3)],
    ]}


def test_scalar_and_vector_integer_timing_including_proposed_pm1():
    cores = [Core(1, 128, 1, 128), Core(1, 256, 1, 128),
             Core(2, 128, 1, 128), Core(2, 128, 1, 192),
             Core(4, 512, 1, 16), Core(32, 2048)]
    types = [(1, 9, 13), (2, 2048, 1408), (3, 2048, 1408),
             (8, 2048, 2816), (128, 2048, 2816)]
    for profile in PROFILES:
        table = cost_table(cores, types, profile)
        assert table.dtype.kind in "iu"
        for ci, core in enumerate(cores):
            for ti, (m, h, f) in enumerate(types):
                expert = dict(Me=m, H=h, F=f)
                expected = integer_job_cycles(core, expert,
                                              tree_stage=profile.tree_stage,
                                              serialized=profile.serialized)
                assert job_cycles(core, expert, profile) == expected
                assert int(table[ci, ti]) == expected


def test_hard_affinity_stays_fixed_when_one_core_has_long_idle_tail():
    # The larger, faster Shared core must idle while the small Routed core runs.
    cores = (Core(2, 128, 1, 192), Core(1, 128, 1, 128))
    window = toy_window(16)
    for profile in PROFILES:
        result = schedule(cores, window, profile)
        assert result["core_cycles"][1] > result["core_cycles"][0]
        validate_schedule(cores, window, result,
                          tree_stage=profile.tree_stage, serialized=profile.serialized)
        assert len([j for j in result["jobs"] if int(j["core_id"]) == 0]) == 1
        assert {int(j["expert_id"]) for j in result["jobs"] if int(j["core_id"]) == 1} == set(range(48))
        assert result == schedule(cores, window, profile)


def test_single_core_processes_all_shared_and_routed_experts():
    cores = (Core(1, 256, 1, 256),)
    window = toy_window(8)
    for profile in PROFILES:
        result = schedule(cores, window, profile)
        validate_schedule(cores, window, result,
                          tree_stage=profile.tree_stage, serialized=profile.serialized)
        assert len(result["jobs"]) == 25
        assert all(int(j["core_id"]) == 0 for j in result["jobs"])


def test_all_135_true_route_windows_conserve_work_and_enforce_owners():
    windows = load_workloads("heldout")["heldout"]
    assert len(windows) == 135
    configs = [(Core(32, 2048),),
               (Core(2, 256, 1, 128),),
               (Core(2, 128, 1, 128), Core(2, 128, 1, 128)),
               (Core(2, 128, 1, 192), Core(1, 128, 1, 128))]
    for window in windows:
        counted = validate_conservation(window)
        for cores in configs:
            result = schedule(cores, window)
            validate_schedule(cores, window, result)
            assert int(result["useful_MACs"]) == counted["total_useful_MACs"]
            assert int(result["unique_weight_bytes"]) == counted["unique_weight_bytes"]
            # 1 GHz: peak is 65,536 useful MACs per 1 ns, or 131,072 GFLOP/s.
            utilization = counted["total_useful_MACs"] / (65536 * int(result["cycles"]))
            assert 0 < utilization <= 1
            if len(cores) == 2:
                # C1 <= half the chip but owns exactly 75% of all useful work.
                assert int(result["cycles"]) * cores[1].macs >= counted["routed_useful_MACs"]
                assert utilization <= 2 / 3


def test_equal_batch_weight_does_not_favor_repeated_small_batch_windows():
    windows = [toy_window(2), toy_window(2), toy_window(4), toy_window(8)]
    assert abs(independent_score([4, 16, 8, 64], windows) - 16) < 1e-12
    duplicate = windows + [toy_window(2), toy_window(2)]
    assert abs(independent_score([4, 16, 8, 64, 4, 16], duplicate) - 16) < 1e-12


def test_invalid_total_multiplier_budget_cannot_run():
    import pytest
    with pytest.raises(ValueError):
        schedule((Core(2, 128, 1, 128),), toy_window())
