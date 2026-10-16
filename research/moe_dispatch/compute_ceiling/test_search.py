"""Check relaxed bounds and exact assignments against independent enumeration."""
import itertools
import unittest
import numpy as np

from .search import fractional_bounds, optimal_counts, exact_layer, Shape,scalar_expert,costs


class SearchTests(unittest.TestCase):
    def test_exact_assignment_and_valid_fractional_bound(self):
        rng=np.random.default_rng(31)
        for _ in range(100):
            c0=tuple(map(int,rng.integers(1,50,3)))
            c1=tuple(map(int,rng.integers(1,50,3)))
            n=tuple(map(int,rng.integers(1,4,3)))
            true=min(max(sum(a*x for a,x in zip(c0,t)),sum(a*(m-x) for a,m,x in zip(c1,n,t)))
                     for t in itertools.product(*(range(m+1) for m in n)))
            result,x=optimal_counts(c0,c1,n)
            self.assertEqual(result,true)
            lower=fractional_bounds(np.array([c0]),np.array([c1]),np.array(n))[0]
            self.assertLessEqual(lower,true+1e-8)

    def test_base_loads_shared_down_overlap(self):
        c0,c1,n=(17,11),(5,23),(2,3)
        expected=min(max(23+sum(a*x for a,x in zip(c0,t)),11+sum(a*(m-x) for a,m,x in zip(c1,n,t)))
                     for t in itertools.product(*(range(m+1) for m in n)))
        self.assertEqual(optimal_counts(c0,c1,n,23,11)[0],expected)

    def test_equal_machine_bitset_solver_and_unequal_initial_loads(self):
        rng=np.random.default_rng(77)
        for _ in range(40):
            c=tuple(map(int,rng.integers(2,20,3)))
            n=tuple(map(int,rng.integers(1,4,3)))
            b0,b1=map(int,rng.integers(0,200,2))
            expected=min(max(b0+sum(a*x for a,x in zip(c,t)),b1+sum(a*(m-x) for a,m,x in zip(c,n,t)))
                         for t in itertools.product(*(range(m+1) for m in n)))
            self.assertEqual(optimal_counts(c,c,n,b0,b1)[0],expected)

    def test_non_power2_K_is_part_of_cost_domain(self):
        c=Shape(1,128,96)
        scalar=scalar_expert({'Me':4,'H':2048,'F':1408},c,'flat20',0)
        self.assertEqual(int(costs([c],[(4,2048,1408)],'flat20')[0,0]),scalar['cycles'])

    def test_projection_timing_matches_independent_event_dependencies(self):
        def events(q,s,l):
            previous=[0]*q
            now=0
            for _ in range(s):
                for i in range(q):
                    now=max(now,previous[i])
                    previous[i]=now+l
                    now+=1
            return max(previous)
        from .search import timing
        for core in (Shape(1,128,96),Shape(3,8,384),Shape(6,4,512)):
            for m,h,f in ((1,129,77),(4,2048,1408),(17,256,192)):
                for profile in ('flat20','log_stage1','log_stage2'):
                    nm=(m+core.pm-1)//core.pm
                    qgu=2*nm*((f+core.pn-1)//core.pn)
                    qdn=nm*((h+core.pn-1)//core.pn)
                    l=timing(profile).completion_latency(core)
                    expected=events(qgu,(h+core.pk-1)//core.pk,l)+events(qdn,(f+core.pk-1)//core.pk,l)
                    self.assertEqual(scalar_expert({'Me':m,'H':h,'F':f},core,profile,0)['cycles'],expected)

    def test_shared_not_omitted_and_no_implicit_big_core_pin(self):
        w={'id':'test','batch':4,'experts':[
             {'id':-1,'Me':4,'H':128,'F':64,'is_shared':True},
             {'id':1,'Me':1,'H':128,'F':64,'is_shared':False}]}
        r=exact_layer(w,(Shape(1,16,512),Shape(1,8,512)))
        self.assertEqual(r['useful_macs'],3*5*128*64)
        self.assertEqual(len(r['assignments']),2)
        self.assertEqual(r['cycles'],max(r['core_cycles']))

    def test_PM1_not_given_artificial_hot_penalty(self):
        e={'Me':128,'H':2048,'F':2816}
        one=scalar_expert(e,Shape(1,96,128),'flat20',0)['cycles']
        six=scalar_expert(e,Shape(6,16,128),'flat20',0)['cycles']
        self.assertLess(one,six)


if __name__=='__main__':
    unittest.main()
