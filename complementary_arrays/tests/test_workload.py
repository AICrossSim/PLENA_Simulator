"""Workload tables and window weights."""

import numpy as np
import pytest

from complementary_arrays.dse.shapes import vocabulary
from complementary_arrays.sim import workload as wl


def _ceil(a, b):
    return -(-a // b)


def _matrix(m, n, k, M, N, K):
    tiles = _ceil(M, m) * _ceil(N, n)
    return tiles * _ceil(K, k) * m + tiles - 1 + 3 * m + (k // m - 1).bit_length()


def _vector(M, kind, w, v):
    H, F, S, E, gate, topk = (
        w["hidden"],
        w["routed_width"],
        w["shared_width"],
        w["experts"],
        int(w["shared_gate"]),
        w["topk"],
    )
    if kind == 2:
        return 8 + _ceil(M * E * (4 + topk), v) + 4
    t = 8 + _ceil(2 * M * (S if kind == 1 else F), v) + 4
    if kind == 0 or gate:
        t += _ceil(M * H, v) + 4
    if kind == 1 and gate:
        t += _ceil(M, v) + 4
    return t + M * (_ceil(H, v) + 4)


@pytest.mark.parametrize("model", sorted(wl.MODELS))
def test_matrix_table_matches_the_tile_formula(model):
    w = wl.MODELS[model]
    H, F, S, E, gate = w["hidden"], w["routed_width"], w["shared_width"], w["experts"], int(w["shared_gate"])
    shapes = vocabulary(model, "w8g128")[::97]
    table = wl.matrix_table(model, shapes)
    for row, (m, n, k) in zip(table, shapes.tolist()):
        for M in (1, 2, 7, 64, 128):
            assert row[M - 1] == _matrix(m, n, k, M, 2 * F, H) + _matrix(m, n, k, M, H, F)
            assert row[127 + M] == _matrix(m, n, k, M, 2 * S + gate, H) + _matrix(m, n, k, M, H, S)
            assert row[255 + M] == _matrix(m, n, k, M, E, H)


@pytest.mark.parametrize("model", sorted(wl.MODELS))
def test_vector_table_matches_the_lane_formula(model):
    table = wl.vector_table(model)
    assert table.shape == (wl.VECTOR_LANES + 1, wl.TASK_COLUMNS) and not table[0].any()
    for v in (1, 3, 512, 1024, 2048):
        for kind in range(3):
            for M in (1, 5, 128):
                assert table[v, kind * 128 + M - 1] == _vector(M, kind, wl.MODELS[model], v)


def test_window_weights_balance_sources_and_groups():
    rows = [dict(source="a", group="a0")] * 3 + [dict(source="a", group="a1")] + [dict(source="b", group="b0")] * 4
    w = wl.window_weights(rows)
    assert w.sum() == pytest.approx(1.0)
    assert w[:3].sum() == pytest.approx(w[3]) == pytest.approx(0.25)
    assert np.allclose(w[4:], 0.125)


def test_geomean():
    assert wl.geomean([2.0, 8.0]) == pytest.approx(4.0)
