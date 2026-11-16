"""Two-stage bootstrap: replicate statistics and intervals."""

import numpy as np
import pytest

from complementary_arrays.dse.bootstrap import merge


def _reps(scale, n=200, batches=(32, 64, 96, 128), infeasible=()):
    rng = np.random.default_rng(1)
    reps = []
    for _ in range(n):
        base = rng.uniform(0.9, 1.1)
        reps.append(dict(batches={
            f"B{B}": {o: (dict(infeasible=True) if (o, B) in infeasible else dict(step_cycles=1000.0 * B * base * scale[o]))
                      for o in scale}
            for B in batches
        }))
    return reps


def test_constant_speedup_is_recovered():
    r = merge(_reps(dict(single2=1.3, hom11=1.1, het11=1.0)), (32, 64, 96, 128))
    assert r["geomean"]["vs_hom"]["ci95"] == pytest.approx([1.1, 1.1])
    assert r["geomean"]["vs_single"]["ci95"] == pytest.approx([1.3, 1.3])
    assert r["geomean"]["vs_best"]["ci95"] == pytest.approx([1.1, 1.1])
    assert r["batches"]["B64"]["vs_hom"]["replicates"] == 200


def test_infeasible_batches_are_counted():
    r = merge(_reps(dict(single2=1.3, hom11=1.1, het11=1.0), batches=(256, 512), infeasible={("single2", 256)}), (256, 512))
    assert r["batches"]["B256"]["vs_single"]["infeasible"] == 200 and "ci95" not in r["batches"]["B256"]["vs_single"]
    assert r["batches"]["B256"]["vs_hom"]["ci95"] == pytest.approx([1.1, 1.1])
