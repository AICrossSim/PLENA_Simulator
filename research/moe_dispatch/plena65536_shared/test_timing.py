import unittest
from .geometry import Core
from .timing import phase, shared, partition, _split_monotone, resource_ledger, Profile

class TimingTests(unittest.TestCase):
    def test_original_resource_and_feed(self):
        c=Core(32,2048)
        self.assertEqual(c.macs,65536)
        self.assertEqual(phase(c,2,2816,2048,2),5632+49)
        self.assertEqual(phase(c,2,2048,2816),4096+49)
        self.assertEqual(shared(c,2),9826)
        self.assertEqual(resource_ledger([c])['M_SUM_adders'],64512)
    def test_splits_match_exhaustive(self):
        for a,b in [(Core(2,16,1,4),Core(4,16)),(Core(8,128),Core(4,64,2,4))]:
            for n in range(1,37):
                f=lambda x: phase(a,3,x,29)
                g=lambda x: phase(b,3,x,29)
                best=min(max(f(x),g(n-x)) for x in range(n+1))
                self.assertEqual(_split_monotone(n,f,g)[0],best)
    def test_coverage_and_useful_peak(self):
        for batch in [2,4,8,16,64,96,128]:
            cs=[Core(16,2048),Core(16,2048)]
            p=partition(cs,batch)
            self.assertGreaterEqual(p['cycles'],264*batch)
    def test_tree_not_free(self):
        c=Core(2,2048,1,16)
        self.assertGreater(shared(c,8,Profile('tree2',2)),shared(c,8))
    def test_zero_region_is_not_charged(self):
        self.assertEqual(phase(Core(32,2048),2,0,2048),0)
