"""Paired two-way bootstrap."""

import numpy as np
import pytest

from complementary_arrays.dse.bootstrap import stats

FAM = ("single2", "hom11", "het11")


def _per_batch(scale):
    rng = np.random.default_rng(1)
    out = {}
    for B in (32, 64, 96, 128):
        n = 24
        base = rng.integers(10_000, 20_000, n).astype(float)
        out[B] = dict(
            cycles={o: (base * scale[o]).tolist() for o in FAM},
            weights=[1 / n] * n,
            layers=[1 + 2 * (i % 3) for i in range(n)],
            groups=[f"g{i % 4}" for i in range(n)],
        )
    return out


def test_equal_chips_give_unit_ratios():
    r = stats(_per_batch(dict(single2=1.0, hom11=1.0, het11=1.0)), FAM)
    for metric in ("B32-128", "B64"):
        for k in ("vs_hom", "vs_single", "vs_best"):
            assert r[metric][k]["point"] == pytest.approx(1.0)
            assert r[metric][k]["ci_two_way"] == pytest.approx([1.0, 1.0])
    assert "B2-128" not in r


def test_constant_speedup_is_recovered():
    r = stats(_per_batch(dict(single2=1.3, hom11=1.1, het11=1.0)), FAM)["B32-128"]
    assert r["vs_hom"]["point"] == pytest.approx(1.1) and r["vs_single"]["point"] == pytest.approx(1.3)
    assert r["vs_best"]["ci_layer"] == pytest.approx([1.1, 1.1])
