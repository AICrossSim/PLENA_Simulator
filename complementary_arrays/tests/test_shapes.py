"""Shape vocabulary, legality and the analytic step-time model."""

import numpy as np
import pytest

from complementary_arrays.dse import shapes as sh


@pytest.mark.parametrize("model,fmt", [("qwen", "bf16"), ("deepseek", "w8g128"), ("qwen3next", "w4g32")])
def test_vocabulary_is_record_aligned(model, fmt):
    v = sh.vocabulary(model, fmt)
    group = {"bf16": 32, "w8g128": 128, "w4g32": 32}[fmt]
    rows = [tuple(r) for r in v.tolist()]
    assert rows == sorted(set(rows))
    assert set(v[:, 0].tolist()) <= set(sh.HEIGHTS)
    for m, n, k in v.tolist():
        assert n % sh.lcm(m, 16) == 0 and k % sh.lcm(m, group) == 0 and n <= sh.MAX_N and n * k <= sh.MAX_MACS


def test_state_bytes_and_legality():
    banks = (3 * 2 - 2 + 5 + 2 + 2) // 3 + 2
    assert (
        sh.state_bytes(2, 64, 64)
        == 8 * 64 * 64 + 4 * 31 * 2 * 64 + 4 * (2 * 64 + 64 * 64) + 2 * 64 * 64 // 2 + 4 * banks * 2 * 64
    )
    with pytest.raises(ValueError):
        sh.state_bytes(4, 66, 64)
    assert sh.legal([[8, 192, 256], [8, 192, 256]], 98304)
    assert not sh.legal([[8, 192, 256], [8, 192, 256]], 98303)
    assert not sh.legal([[8, 600, 128], [8, 600, 128]], 196608)
    assert not sh.legal([[3, 64, 64]], 98304)


def test_analytic_times_respect_the_hbm_floor(tmp_path):
    m = sh.AnalyticModel(tmp_path, "qwen", "bf16")
    rows = m.shapes.tolist()
    i, j = rows.index([2, 64, 64]), rows.index([8, 256, 256])
    hh = np.zeros((2, 130), np.int32)
    hh[0, 1], hh[0, 129] = 8, 1
    hh[1, 1], hh[1, 3], hh[1, 129] = 100, 40, 128
    floor = m.floor(hh)
    for t in (m.single(i, hh, 1.0), m.homogeneous(j, hh, 1.0), m.pair(i, j, hh, 1.0), m.pair(i, j, hh, 0.33)):
        assert np.all(t >= floor)
    plain = m.single(j, hh, float("inf"))
    m.drain, m.expert_drains = (lambda n: n - 1), 2
    assert np.all(m.single(j, hh, float("inf")) >= plain)
