import unittest
from .domain import domains, validate


class DomainTests(unittest.TestCase):
    def test_every_design_is_exact_budget_and_aligned(self):
        for ds in domains().values():
            for cs in ds:
                validate(cs)

    def test_controls_retain_restricted_and_stream_capabilities(self):
        d = domains()
        self.assertTrue(set(d['compact_single']) <= set(d['fair_single']))
        self.assertTrue(set(d['compact_homo']) <= set(d['fair_homo']))
        self.assertTrue(any(cs[0].pm == 1 for cs in d['fair_single']))
        self.assertTrue(any(cs[0].pm == 1 for cs in d['fair_homo']))
        self.assertTrue(all(cs[0].b >= 2 for cs in d['fair_native_single']))
        self.assertEqual({(a.macs+b.macs) for a,b in d['hetero']}, {65536})
        self.assertTrue(any(a.macs < b.macs for a,b in d['hetero']))

    def test_restricted_domain_has_all_allocations_and_no_decimal_split(self):
        d = domains()
        self.assertEqual(len(d['hetero']), 4531)
        self.assertEqual({a.macs for a,b in d['hetero']}, set(range(4096,65536,4096)))
        self.assertFalse(any(a.macs == 43690 for a,b in d['hetero']))


if __name__ == '__main__':
    unittest.main()
