"""Scalar-vs-vector independent consistency over exact cuts and full domains."""
import random

import numpy as np

from .geometry import cores_by_macs, enumerate_cores, iter_heterogeneous_pairs
from .timing import BATCHES, PROFILES, partition
from .vector_search import evaluate_domain, exact_pairs


def test_all_small_domain_pair_plans_match_scalar_exactly():
    cores = enumerate_cores(128)
    indices = {c: i for i, c in enumerate(cores)}
    ids = np.asarray([(indices[a], indices[b]) for a, b in iter_heterogeneous_pairs(128)], dtype=np.int32)
    for profile in PROFILES:
        got = exact_pairs(cores, ids, profile)
        for row, (i, j) in enumerate(ids):
            for col, batch in enumerate(BATCHES):
                expected = partition((cores[i], cores[j]), batch, profile)
                assert got['cycles'][row, col] == expected['cycles']
                assert got['rows'][row, col] == (expected['kind'] == 'rows')
                if expected['kind'] == 'rows':
                    assert got['row_split'][row, col] == expected['row_split']
                else:
                    assert got['gu_split'][row, col] == expected['gu_split']
                    assert got['down_split'][row, col] == expected['down_split']


def test_random_65536_budget_pair_plans_match_scalar_exactly():
    cores = enumerate_cores()
    indices = {c: i for i, c in enumerate(cores)}
    groups = cores_by_macs()
    budgets = sorted(b for b in groups if 65536-b in groups)
    rng = random.Random(112026)
    pairs = []
    for _ in range(150):
        b = rng.choice(budgets)
        pairs.append((rng.choice(groups[b]), rng.choice(groups[65536-b])))
    ids = np.asarray([(indices[a], indices[b]) for a, b in pairs], dtype=np.int32)
    for profile in PROFILES:
        got = exact_pairs(cores, ids, profile)
        for row, cs in enumerate(pairs):
            for col, batch in enumerate(BATCHES):
                expected = partition(cs, batch, profile)
                assert got['cycles'][row, col] == expected['cycles']
                assert got['rows'][row, col] == (expected['kind'] == 'rows')
                if expected['kind'] == 'rows':
                    assert got['row_split'][row, col] == expected['row_split']
                else:
                    assert got['gu_split'][row, col] == expected['gu_split']
                    assert got['down_split'][row, col] == expected['down_split']


def test_domain_scores_are_chunk_independent_and_match_direct_loop():
    cores = enumerate_cores(64)
    indices = {c: i for i, c in enumerate(cores)}
    ids = np.asarray([(indices[a], indices[b]) for a, b in iter_heterogeneous_pairs(64)], dtype=np.int32)
    selection1, scores1 = evaluate_domain(cores, ids, chunk=1)
    selection2, scores2 = evaluate_domain(cores, ids, chunk=8192)
    assert np.array_equal(scores1, scores2)
    assert selection1 == selection2
    expected = np.asarray([[np.exp(np.mean(np.log([
        partition((cores[i], cores[j]), batch, profile)['cycles'] for batch in BATCHES])))
                            for i, j in ids] for profile in PROFILES])
    assert np.array_equal(scores1, expected)
