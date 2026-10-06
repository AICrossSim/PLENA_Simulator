"""Cross-check accelerated full-space DSE against scalar/event contracts."""
import unittest
import numpy as np

from .compute import ContextLimits, projection, paired_gate_up
from .robust_compute import (BUDGET, Shape, balanced_score, costs, domain, gid,
                            minimax_selection, scalar_layer, timing, vector_schedule)
from .test_compute import event_projection, event_paired_gate_up


def workload(batch, tokens):
    return {"id": f"test_b{batch}", "batch": batch,
            "experts": [{"id": i, "Me": m, "H": 129, "F": 77, "is_shared": i == 0}
                        for i, m in enumerate(tokens)]}


class RobustComputeTests(unittest.TestCase):
    def test_expanded_domain_equal_budget_canonical_and_all_axes_vary(self):
        shapes, indices, fam = domain()
        self.assertEqual(len(shapes), 9417)
        self.assertEqual(len(indices), 114034)
        self.assertEqual([int(np.count_nonzero(fam == i)) for i in range(3)], [88,72,113874])
        self.assertEqual(len(set(map(tuple,indices.tolist()))), len(indices))
        self.assertTrue(any(c.pm > 16 for c in shapes))
        self.assertTrue(any(c.pn > 192 for c in shapes))
        for a,b in indices:
            self.assertEqual(shapes[a].macs + (shapes[b].macs if b >= 0 else 0), BUDGET)
            if b >= 0:
                self.assertLessEqual(a,b)
        self.assertTrue(any(b >= 0 and shapes[a].pk != shapes[b].pk for a,b in indices))

    def test_vectorized_cost_matches_scalar_and_independent_event_scoreboard(self):
        shapes = (Shape(1,768,16), Shape(3,17,128), Shape(6,4,512), Shape(2,3,2048))
        signatures = [(m,h,f) for m in (1,4,17) for h,f in ((1,1),(129,77),(2048,1408))]
        for name in ("flat20","log_stage1","log_stage2"):
            for q in (0,1,4,8,16,32):
                cycles, issued, useful, issues = costs(shapes, signatures, name, q)
                for ci,c in enumerate(shapes):
                    for si,(m,h,f) in enumerate(signatures):
                        gate_records = 2*((m+c.pm-1)//c.pm)*((f+c.pn-1)//c.pn)
                        down_records = ((m+c.pm-1)//c.pm)*((h+c.pn-1)//c.pn)
                        limits = ContextLimits(down_records if q == 0 else q,1 << 40)
                        gu_limits = ContextLimits(gate_records if q == 0 else q,1 << 40)
                        pair_n = 2*((f+c.pn-1)//c.pn)*c.pn
                        gu = projection(m,pair_n,h,c,timing(name),gu_limits)
                        dn = projection(m,h,f,c,timing(name),limits)
                        self.assertEqual(int(cycles[ci,si]),gu.cycles+dn.cycles)
                        self.assertEqual(int(issued[ci,si]),gu.issued_macs+dn.issued_macs)
                        self.assertEqual(int(useful[ci,si]),3*m*h*f)
                        self.assertEqual(int(issues[ci,si]),gu.issues+dn.issues)
                        # Independent event scheduler; no closed-form reuse.
                        self.assertEqual(dn.cycles,event_projection(m,h,f,c,timing(name),limits)[0])
                        self.assertEqual(gu.cycles,event_projection(m,pair_n,h,c,timing(name),gu_limits)[0])

    def test_vector_dispatch_matches_scalar_with_unequal_shapes_and_context_budget(self):
        shapes = (Shape(6,4,512),Shape(3,4,512),Shape(4,4,512),Shape(2,4,512))
        indices = np.array([(0,-1),(1,1),(2,3)])
        windows = [workload(4,[4,1,2,3]),workload(16,[16,12,3,1,1,1])]
        signatures = sorted({(e['Me'],e['H'],e['F']) for w in windows for e in w['experts']})
        for q in (0,8,16,32):
            for name in ("flat20","log_stage1","log_stage2"):
                vectors = vector_schedule(windows,signatures,costs(shapes,signatures,name,q)[0],
                                          costs(shapes,signatures,name,q)[0],indices)
                for gi,(a,b) in enumerate(indices):
                    cores = (shapes[a],) if b < 0 else (shapes[a],shapes[b])
                    for wi,w in enumerate(windows):
                        self.assertEqual(int(vectors[gi,wi]),scalar_layer(w,cores,name,q)['cycles'])

    def test_batch_balancing_avoids_more_windows_or_large_batch_dominance(self):
        windows = [{"batch":2},{"batch":128},{"batch":128},{"batch":128}]
        ratios = np.array([[2.,.5,.5,.5]])
        self.assertAlmostEqual(float(balanced_score(ratios,np.ones(4),windows)[0]),1.)
        self.assertNotAlmostEqual(float(np.exp(np.log(ratios).mean())),1.)

    def test_minimax_is_within_family_and_not_fastest_single_assumption(self):
        scores = np.array([[1.,1.1,1.6, 2.,2.2,3.2, 3.,3.3,4.8],
                           [1.6,1.1,1., 3.2,2.2,2., 4.8,3.3,3.]])
        fam = np.array([0,0,0,1,1,1,2,2,2])
        chosen,regret = minimax_selection(scores,fam)
        self.assertEqual(chosen,{"single":1,"homogeneous":4,"heterogeneous":7})
        for i in chosen.values():
            self.assertAlmostEqual(float(regret[:,i].max()),1.1)

    def test_padding_is_not_full_device_idle_capacity(self):
        cores = (Shape(2,4,512),Shape(4,4,512))
        result = scalar_layer(workload(4,[4,2]),cores)
        self.assertGreaterEqual(result['padding_macs'],0)
        self.assertGreater(result['cycles'],0)
        self.assertLessEqual(result['wall_mac_utilization'],result['spatial_utilization'])
        self.assertEqual(result['geometry'],gid(cores))

    def test_gate_up_independence_and_separate_logical_tail_padding(self):
        c = Shape(1,32,128)
        e = {"Me":1,"H":513,"F":33}
        from .robust_compute import scalar_expert
        cost = scalar_expert(e,c,"flat20",0)
        prior_serial = paired_gate_up(1,33,513,c,timing("flat20"),ContextLimits(64,1<<40))
        self.assertLess(cost['gate_up_cycles'],prior_serial.cycles)
        self.assertEqual(cost['useful_macs'],3*33*513)
        # Each F=33 has two 32-wide tiles, rather than packing Gate/Up into three.
        self.assertEqual(cost['issued_macs'],(2*2*5+17*1)*c.macs)


if __name__ == '__main__':
    unittest.main()
