"""Abstract two-stage model of the dispatch bound."""

import pytest

from complementary_arrays.dse.oracle import jobs, list_schedule


def _tables():
    # (estimate, compute cycles, HBM bytes) per (array, job kind); HBM delivers 1000 bytes per cycle
    return {
        (0, "shared"): (100.0, 100.0, 5000.0),
        (1, "shared"): (150.0, 150.0, 5000.0),
        (0, 1): (40.0, 40.0, 30000.0),
        (1, 1): (30.0, 30.0, 30000.0),
        (0, 3): (60.0, 60.0, 30000.0),
        (1, 3): (45.0, 45.0, 30000.0),
    }


@pytest.mark.parametrize("policy", [9, 11, 12, 13])
def test_single_job_runs_on_the_faster_array(policy):
    assert list_schedule(policy, [("shared", -1)], _tables(), 2, 1000, 8, macs=[2, 1]) == 100


@pytest.mark.parametrize("policy", [9, 11, 12, 13])
def test_makespan_respects_the_hbm_floor(policy):
    J = [("shared", -1), (3, 7), (1, 4), (1, 9)]
    floor = 5 + 3 * 30
    for buffer in (1, 2):
        assert list_schedule(policy, J, _tables(), 2, 1000, 8, buffer, macs=[2, 1]) >= floor


def test_jobs_count_tokens_per_expert():
    import numpy as np

    window = dict(ids=np.array([[1, 4], [4, 9], [4, 2]]))
    assert jobs(window) == [("shared", -1), (1, 1), (1, 2), (3, 4), (1, 9)]
