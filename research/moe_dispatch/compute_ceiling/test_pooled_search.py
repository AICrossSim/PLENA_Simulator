"""Ownership and cache checks independent of the large campaign."""
from itertools import product
import unittest

from .pooled_search import (cached_core_replay, exact_issue_assignment,
                            issue_costs, pooled_layer, workload_key)
from .search import Shape


def issue_count(e, c):
    # Count the physical invocations of the three GEMMs independently.
    total = 0
    for n, k in ((e['F'], e['H']), (e['F'], e['H']), (e['H'], e['F'])):
        total += (len(range(0, e['Me'], c.pm)) * len(range(0, n, c.pn))
                  * len(range(0, k, c.pk)))
    return total


class PooledSearchTests(unittest.TestCase):
    def test_issue_assignment_matches_exhaustive_ownership(self):
        cores = (Shape(1, 4, 16), Shape(2, 4, 16))
        experts = [dict(id=i, Me=m, H=65, F=49, is_shared=i == 0)
                   for i, m in enumerate((3, 1, 2, 3, 1))]
        w = dict(id='manual', batch=3, experts=experts)
        expected = min(max(sum(issue_count(e, cores[ci]) for e, owner in zip(experts, owners)
                               if owner == ci) for ci in range(2))
                       for owners in product(range(2), repeat=len(experts)))
        floor, owned, rows = exact_issue_assignment(w, cores)
        self.assertEqual(floor, expected)
        self.assertEqual({r['expert_id'] for r in rows}, set(range(len(experts))))
        self.assertEqual(sum(map(len, owned)), len(experts))
        result = pooled_layer(w, cores)
        self.assertGreaterEqual(result['cycles'], expected + 20)
        self.assertEqual(result['cycles'], max(result['core_cycles']))

    def test_vector_issue_counts_include_non_power_two_K_and_tails(self):
        cores = (Shape(1, 8, 48), Shape(3, 4, 16))
        signatures = ((1, 65, 49), (5, 96, 80))
        counts = issue_costs(cores, signatures)
        for ci, c in enumerate(cores):
            for si, (m, h, f) in enumerate(signatures):
                self.assertEqual(counts[ci, si], issue_count(dict(Me=m, H=h, F=f), c))

    def test_cache_preserves_distinct_shapes_on_equal_logical_work(self):
        # Shared width2*M1 and routed width1*M2 tie in useful work, but their
        # GU/Down record counts and dependency graphs differ.
        jobs = [dict(id=-1, Me=1, H=16, F=32, is_shared=True),
                dict(id=2, Me=2, H=16, F=16, is_shared=False)]
        w = dict(id='equal-work', batch=1, experts=jobs)
        core = Shape(1, 4, 16)
        uncached = pooled_layer(w, (core,), cached=False)
        cached = pooled_layer(w, (core,), cached=True)
        self.assertEqual(uncached, cached)
        a = cached_core_replay(workload_key(jobs), core)
        self.assertEqual(a['issues'], sum(issue_count(e, core) for e in jobs))


if __name__ == '__main__':
    unittest.main()
