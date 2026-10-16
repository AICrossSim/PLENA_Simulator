"""Independent LP corner cases and exhaustive integer-assignment checks."""
from fractions import Fraction
from itertools import product
from random import Random
import unittest

from research.moe_dispatch.plena65536_moe.bounds import (
    fractional_two_machine, fractional_two_machine_exact, full_window_feed_lower_bound)
from research.moe_dispatch.plena65536_shared.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES, phase


class FractionalBoundsTests(unittest.TestCase):
    def test_hand_computed_frontier(self):
        self.assertEqual(fractional_two_machine_exact((7,), (11,)), Fraction(77, 18))
        self.assertEqual(fractional_two_machine_exact((2,), (3,), (5,)), 6)
        self.assertEqual(fractional_two_machine_exact((2, 4), (4, 2), (3, 3)), 6)
        self.assertEqual(fractional_two_machine_exact((), ()), 0)
        self.assertEqual(fractional_two_machine((), ()), 0.0)

    def test_lower_than_every_integer_assignment(self):
        rng = Random(65536)
        for _ in range(200):
            types = rng.randint(1, 5)
            a = tuple(rng.randint(1, 1000) for _ in range(types))
            b = tuple(rng.randint(1, 1000) for _ in range(types))
            counts = tuple(rng.randint(0, 3) for _ in range(types))
            exact = fractional_two_machine_exact(a, b, counts)
            bound = fractional_two_machine(a, b, counts)
            optimum = min(max(sum(v*x for v,x in zip(a, selected)),
                              sum(v*(n-x) for v,n,x in zip(b, counts, selected)))
                          for selected in product(*(range(n+1) for n in counts)))
            self.assertLessEqual(exact, optimum)
            self.assertLessEqual(Fraction.from_float(bound), exact)
            self.assertEqual(exact, fractional_two_machine_exact(b, a, counts))

    def test_grouping_retains_fractional_feasible_set(self):
        a, b, counts = (11, 71, 17), (61, 13, 27), (3, 5, 2)
        expanded_a = tuple(v for v,n in zip(a,counts) for _ in range(n))
        expanded_b = tuple(v for v,n in zip(b,counts) for _ in range(n))
        self.assertEqual(fractional_two_machine_exact(a,b,counts),
                         fractional_two_machine_exact(expanded_a,expanded_b))

    def test_complete_mixed_window_feed_bound(self):
        experts = [dict(id="shared", Me=8, H=2048, F=2816, is_shared=True),
                   dict(id=0, Me=1, H=2048, F=1408),
                   dict(id=1, Me=7, H=2048, F=1408)]
        cores = (Core(2, 128, 1, 128), Core(2, 512, 1, 32))
        self.assertEqual(sum(c.macs for c in cores), 65536)
        for profile in PROFILES:
            costs = [[phase(c,e["Me"],e["F"],e["H"],2,profile) +
                      phase(c,e["Me"],e["H"],e["F"],1,profile)
                      for e in experts] for c in cores]
            for selected in product((0,1), repeat=len(experts)):
                feasible = max(sum(costs[c][j] for j,owner in enumerate(selected)
                                   if owner == c) for c in (0,1))
                self.assertLessEqual(full_window_feed_lower_bound(experts,cores,profile), feasible)
                self.assertLessEqual(full_window_feed_lower_bound(experts,cores,profile,
                                                                  preserve_m_padding=True), feasible)


if __name__ == "__main__":
    unittest.main()
