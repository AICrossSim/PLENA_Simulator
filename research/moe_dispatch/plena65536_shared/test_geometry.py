from dataclasses import FrozenInstanceError
from itertools import combinations_with_replacement

import pytest

from research.moe_dispatch.plena65536_shared.geometry import (
    BUDGET,
    Core,
    count_dual_pairs,
    count_heterogeneous_pairs,
    enumerate_cores,
    enumerate_native_designs,
    geometry_counts,
    homogeneous_cores,
    iter_dual_pairs,
    iter_heterogeneous_pairs,
    single_cores,
)


def test_native_paper_multiplier_count_is_not_logical_gemm_volume():
    core = Core(32, 2048)
    assert core.shape == (32, 32, 64)
    assert core.macs == 65_536
    assert core.native
    assert "BLEN=32, MLEN=2048" in str(core)
    with pytest.raises(FrozenInstanceError):
        core.b = 16


def test_replication_has_physical_aliases_with_different_frames():
    first = Core(2, 128, 2, 4)
    second = Core(4, 256, 1, 2)
    assert first.shape == second.shape == (4, 8, 64)
    assert first.macs == second.macs == 2048
    assert first != second
    assert not first.native and not second.native
    assert first in enumerate_cores(2048) and second in enumerate_cores(2048)


@pytest.mark.parametrize(
    "args",
    [
        (1, 128, 1, 4),
        (4, 2, 1, 1),
        (4, 96, 1, 1),
        (4, 128, 3, 1),
        (2, 128, 1, 1),
        (4, 128, 1, 0),
        (64, 2048, 1, 1),
        (4.0, 128, 1, 1),
    ],
)
def test_invalid_frames_or_resource_limits_rejected(args):
    with pytest.raises(ValueError):
        Core(*args)


def test_small_domain_pair_counts_match_brute_force_without_duplicates():
    budget = 128
    cores = enumerate_cores(budget)
    brute = {
        frozenset((a, b))
        for a, b in combinations_with_replacement(cores, 2)
        if a.macs + b.macs == budget
    }
    actual = list(iter_dual_pairs(budget))
    assert len(actual) == len(brute) == count_dual_pairs(budget)
    assert {frozenset(pair) for pair in actual} == brute
    hetero = list(iter_heterogeneous_pairs(budget))
    assert all(a != b and a.macs + b.macs == budget for a, b in hetero)
    assert len(hetero) == count_heterogeneous_pairs(budget)
    assert len(actual) - len(hetero) == len(homogeneous_cores(budget))


def test_full_domain_count_certificate_and_no_artificial_pn_cap():
    assert geometry_counts() == {
        "budget": 65_536,
        "individual_cores": 53_040,
        "distinct_physical_shapes": 32_239,
        "distinct_core_budgets": 8192,
        "single": 188,
        "homogeneous": 176,
        "heterogeneous": 784_402,
        "dual_including_homogeneous": 784_578,
        "native_single": 2,
        "native_homogeneous": 3,
        "native_heterogeneous": 3,
    }
    assert max(core.pn for core in enumerate_cores()) == 32_768
    assert all(core.macs == BUDGET for core in single_cores())
    assert all(core.macs == BUDGET // 2 for core in homogeneous_cores())
    assert len(enumerate_cores()) == len(set(enumerate_cores()))


def test_strict_native_points_and_unequal_replication_budget_available():
    native = enumerate_native_designs()
    assert {(c.b, c.r) for c in native["single"]} == {(32, 2048), (64, 1024)}
    assert {(a.b, a.r) for a, _ in native["homogeneous"]} == {
        (16, 2048), (32, 1024), (64, 512)
    }
    assert all(a.macs == b.macs == 32_768 for a, b in native["heterogeneous"])
    # Native-sized array replication supports a 3:1 resource split without
    # modifying the total multiplier budget.  A 2:1 exact integer split does not.
    small = Core(4, 1024, 1, 4)
    large = Core(4, 1024, 1, 12)
    assert small.macs + large.macs == BUDGET
    assert large.macs == 3 * small.macs
    assert BUDGET % 3 != 0
