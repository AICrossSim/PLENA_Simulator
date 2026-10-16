"""Check diagnostic exactness against independent tiny exhaustive ownership."""
from itertools import product

import pytest

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_dedicated.model import job_cycles
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .assignment_gap import exact_plan, _summary


@pytest.mark.parametrize("cores", [
    (Core(1, 256, 1, 128), Core(1, 256, 1, 128)),
    (Core(1, 64, 1, 704), Core(1, 128, 1, 160)),
])
def test_exact_oracle_equals_exhaustive_individual_assignments(cores):
    window = {"id": "tiny", "batch": 4, "experts": [
        {"id": -1, "is_shared": True, "Me": 4, "H": 2048, "F": 2816},
        {"id": 0, "Me": 1, "H": 2048, "F": 1408},
        {"id": 1, "Me": 1, "H": 2048, "F": 1408},
        {"id": 2, "Me": 2, "H": 2048, "F": 1408},
        {"id": 3, "Me": 3, "H": 2048, "F": 1408},
    ]}
    for profile in PROFILES:
        values = []
        for owners in product((0, 1), repeat=len(window["experts"])):
            loads = [sum(job_cycles(cores[c], e, profile) for e, o in zip(window["experts"], owners)
                         if o == c) for c in (0, 1)]
            values.append(max(loads))
        result = exact_plan(cores, window, profile)
        assert result["cycles"] == min(values)
        assert len(result["jobs"]) == 5
        assert sum(result["group_counts"]) == 5
        assert result["cycles"] == max(result["core_cycles"])
        assert result == exact_plan(cores, window, profile)


def test_single_core_or_unbalanced_resource_total_is_rejected():
    with pytest.raises(ValueError, match="two frozen cores"):
        exact_plan((Core(1, 512, 1, 128),), {"experts": []})


def test_batch_balanced_oracle_gap_does_not_overweight_small_batch_windows():
    rows = []
    for batch, ratio, count in ((2, 4, 9), (4, 1, 1), (8, 1, 1), (16, 1, 1),
                                (64, 1, 1), (96, 1, 1), (128, 1, 1)):
        for _ in range(count):
            rows.append({"design": "fixture", "split": "heldout", "batch": batch,
                         "policy_ms": ratio, "oracle_ms": 1, "single_ms": 1,
                         "policy_vs_oracle": ratio, "oracle_vs_single": 1,
                         "policy_vs_single": ratio, "policy_cycles": 100 * ratio,
                         "oracle_cycles": 100})
    got = _summary(rows, "7-Batch GM")
    assert got["policy_vs_oracle"] == pytest.approx(4 ** (1 / 7))
    assert got["windows_policy_equals_oracle"] == 6
