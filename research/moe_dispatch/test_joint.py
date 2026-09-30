"""Numerical and protocol verification for bounded joint dispatch.

Uses actual Rust DMA/issue/commit traces with seeded payloads. Real-route
performance windows are a separate analytical experiment, not this proof.
"""
import copy
import unittest

from test_engine import AnalyticalEngineIntegrationTests
from test_runtime import workload
from trace_payload import replay


SHAPES=((6,),(3,3),(4,2),(8,),(4,4),(5,3))


def resources(lanes):
    n=len(lanes);total=sum(lanes)
    # Equal arena split is a legal test point; performance uses frozen layouts.
    return dict(weight_slots=[10] if n==1 else [5,5],weight_banks=[64//n]*n,
        x_banks=[4*m for m in lanes],acc_banks=[2*m for m in lanes],
        acc_bytes=[2*1024**2//n]*n,control_bytes=[4352//n]*n,
        feedback_state_bytes=96,joint_state_bytes=256)


def with_shared(ms,h=33,f=19):
    w=workload(ms,h=h,f=f)
    # Replace the last routed expert by an always-active Shared expert. Shapes
    # and IDs stay consistent with an actual source descriptor stream.
    shared=w['experts'][-1]
    prior=shared['id']
    shared.update(id=-1,is_shared=True,Me=w['batch'],token_indices=list(range(w['batch'])),
                  route_slots=[-1]*w['batch'],route_scores=[1.0]*w['batch'])
    w['top_k']-=1
    for t in w['tokens']:t['routes']=[r for r in t['routes'] if r['expert_id']!=prior]
    w['experts']=[shared]+w['experts'][:-1]
    return w


class JointRuntimeTests(unittest.TestCase):
    run_engine=AnalyticalEngineIntegrationTests.run_engine

    def run_case(self,w,lanes,**changes):
        cfg=dict(resources=resources(lanes),dispatch='joint',runtime_fsm=True,window=8,
                 arbiter='stock',group=4,next_prefetch=True,tail_partition=False)
        cfg.update(changes)
        r,layout=self.run_engine(w,list(lanes),**cfg)
        self.assertEqual(r['weight_bytes'],32*r['dma_transactions_accepted'])
        self.assertEqual(r['dma_transactions_accepted'],r['dma_transactions_landed'])
        self.assertEqual(r['useful_macs'],sum(3*e['Me']*e['H']*e['F'] for e in w['experts']))
        self.assertLessEqual(r['pending_window_peak'],8)
        self.assertEqual(r['dispatch_decisions'],len(w['experts']))
        owners=[t for t in r['trace'] if t['event']=='commit_owner']
        self.assertEqual(len(owners),len(w['experts']))
        self.assertEqual(len({t['task'] for t in owners}),len(w['experts']))
        for c,slots,m in zip(r['cores'],resources(lanes)['weight_slots'],lanes):
            self.assertLessEqual(c['stats']['weight_peak_bytes'],4096*slots)
            self.assertLessEqual(c['stats']['x_peak_bytes'],2*m*512*2)
            self.assertLessEqual(c['stats']['workspace_peak_bytes'],c['capacity'])
        for a in r['dispatch_audit']:
            self.assertEqual(a['actual_minus_predicted_cycles'],a['actual_finish_cycle']-a['predicted_finish_cycle'])
        return r,layout

    def test_odd_dimensions_and_all_organizations(self):
        w=with_shared([7,2,1,3],h=513,f=35)
        hashes=set();bytes_seen=set()
        for lanes in SHAPES:
            with self.subTest(lanes=lanes):
                r,layout=self.run_case(w,lanes)
                hashes.add(replay(w,r)['sha256_bf16'])
                bytes_seen.add(r['weight_bytes'])
                self.assertEqual(sum(resources(lanes)['control_bytes']),4352)
        self.assertEqual(len(hashes),1)
        self.assertEqual(len(bytes_seen),1)

    def test_old_policies_and_joint_ablations_same_math(self):
        w=with_shared([4,2,1,7],h=33,f=19)
        variants=[dict(dispatch=p) for p in ('fifo','dynamic','feedback','joint')]
        variants += [{flag:False} for flag in ('joint_late_bind','joint_pairing','joint_feedback','next_prefetch')]
        for lanes in ((3,3),(4,2),(5,3)):
            hashes=set();bytes_seen=set()
            for variant in variants:
                with self.subTest(lanes=lanes,variant=variant):
                    r,_=self.run_case(w,lanes,**variant)
                    hashes.add(replay(w,r)['sha256_bf16'])
                    bytes_seen.add(r['weight_bytes'])
            self.assertEqual(len(hashes),1)
            self.assertEqual(len(bytes_seen),1)

    def test_more_than_window_tasks_no_loss_with_dma_backpressure(self):
        w=with_shared([1,2,1,8,1,3,2,1,4,1,2,1,7],h=33,f=19)
        for lanes in ((6,),(3,3),(4,2)):
            with self.subTest(lanes=lanes):
                r,_=self.run_case(w,lanes,credits=2,dma_ready_period=7,dma_ready_cycles=2)
                self.assertGreater(r['input_backpressure_cycles'],0)
                self.assertEqual(r['pending_window_peak'],8)
                self.assertTrue(replay(w,r)['all_bit_exact'])

    def test_full_json_repeat_and_no_prefetch(self):
        w=with_shared([3,1,2,7,1],h=17,f=9)
        for lanes in ((6,),(4,2)):
            for prefetch in (False,True):
                with self.subTest(lanes=lanes,prefetch=prefetch):
                    a,_=self.run_case(w,lanes,next_prefetch=prefetch)
                    b,_=self.run_case(w,lanes,next_prefetch=prefetch)
                    self.assertEqual(a,b)
                    self.assertTrue(replay(w,a)['all_bit_exact'])

    def test_shared_last_stream_still_completes_every_expert(self):
        first=with_shared([7,2,1,3,2,1,2,4,1,2],h=33,f=19)
        last=copy.deepcopy(first)
        last['experts']=last['experts'][1:]+last['experts'][:1]
        # Replay seeds are task-order dependent, so compare each order with its
        # own reference, not hashes across different payload generation orders.
        for w in (first,last):
            r,_=self.run_case(w,(4,2))
            self.assertTrue(replay(w,r)['all_bit_exact'])


if __name__=='__main__':unittest.main()
