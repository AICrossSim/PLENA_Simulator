"""Snapshot draws and windows."""

import numpy as np
import pytest

from complementary_arrays.dse import snapshots as ss


def test_inclusion_caps_long_requests():
    w = np.array([10.0, 1, 1, 1, 1, 1, 1, 1])
    pi, capped = ss.inclusion(w, 4)
    assert capped == 1 and pi[0] == 1.0 and pi.sum() == pytest.approx(4) and np.all(pi <= 1 + 1e-12)
    with pytest.raises(ValueError):
        ss.inclusion(w, 9)


def test_draws_are_distinct_and_proportional():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 500, 300)
    sources = np.array(["a", "b", "c"])[rng.integers(0, 3, 300)]
    w = ss.s1_weights(sources, lengths)
    req, pos = ss.draw_snapshots(w, lengths, 32, 400, np.random.default_rng(ss.seed_of("t", 32)))
    assert all(len(set(r)) == 32 for r in req.tolist()) and np.all(pos < lengths[req])
    share = np.bincount(req.ravel(), minlength=300) / req.size
    assert np.abs(share - w).max() < 0.01
    again, _ = ss.draw_snapshots(w, lengths, 32, 400, np.random.default_rng(ss.seed_of("t", 32)))
    assert np.array_equal(req, again)


def test_weights_split_sources_equally():
    w = ss.s1_weights(["a", "a", "b"], [1, 3, 2])
    assert w.tolist() == pytest.approx([0.125, 0.375, 0.5])


def test_files_and_windows(tmp_path):
    layers, topk, experts, B = list(range(8)), 2, 16, 4
    rng = np.random.default_rng(3)
    routes = [np.stack([np.stack([rng.choice(experts, topk, replace=False) for _ in layers]) for _ in range(20)])
              for _ in range(12)]
    pool = ss.Pool([f"r{i}" for i in range(12)], ["a", "b"] * 6, ["dev"] * 12, [20] * 12, layers, topk, experts, routes)
    req, pos = ss.draw_snapshots(ss.s1_weights(pool.sources, pool.lengths), pool.lengths, B, 20, rng)
    ids = ss.materialize(pool, req, pos)
    assert np.array_equal(ids[3, 5, 1], routes[req[3, 1]][pos[3, 1], 5])
    path = ss.snapshot_path(tmp_path, "toy", "ref", "dev", B)
    ss.save_snapshots(path, "toy", pool, req, pos, ids, "dev", B, ss.seed_of("toy"))
    rows, slots, info = ss.load_snapshots(path)
    assert len(rows) == 20 * len(layers) and np.array_equal(slots["slot_req"], req) and info["batch"] == B
    screen = ss.windows(tmp_path, "toy", "ref", B, "screen")
    assert {r["layer"] for r in screen} == {2, 4, 6} and {r["snapshot"] for r in screen} == set(range(16))
    assert ss.digest(path) == ss.digest(path)
