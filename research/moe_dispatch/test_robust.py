"""Resource/feedback tests replay the actual finite timed DMA/issue stream."""
import unittest
from test_engine import AnalyticalEngineIntegrationTests
from test_runtime import workload
from trace_payload import replay
from frontend import compiler

class RobustRuntimeTests(unittest.TestCase):
    run_engine=AnalyticalEngineIntegrationTests.run_engine

    def resources(self, lanes, reverse=False):
        n=len(lanes);total=sum(lanes)
        return dict(weight_slots=[10] if n==1 else ([3,7] if reverse else [7,3]),
                    weight_banks=[64//n]*n,x_banks=[4*m for m in lanes],
                    acc_banks=[2*total//n]*n,acc_bytes=[2*1024**2//n]*n,
                    control_bytes=[4096//n]*n,feedback_state_bytes=96)

    def test_all_shapes_tails_and_nonuniform_slots_bit_exact(self):
        w=workload([7,2,1,3],h=513,f=19);hashes=set()
        for lanes in compiler.ROBUST_ORGANIZATIONS:
            for reverse in (False,True):
                with self.subTest(lanes=lanes,reverse=reverse):
                    r,_=self.run_engine(w,list(lanes),resources=self.resources(lanes,reverse),group=2,
                                        runtime_fsm=True,dispatch='feedback',arbiter='stock')
                    hashes.add(replay(w,r)['sha256_bf16'])
                    self.assertEqual(r['feedback_updates'],len(w['experts']))
                    for c,slots in zip(r['cores'],self.resources(lanes,reverse)['weight_slots']):
                        self.assertLessEqual(c['stats']['weight_peak_bytes'],4096*slots)
        self.assertEqual(len(hashes),1)

    def test_fixed_assignment_and_feedback_have_same_math(self):
        w=workload([2,3,7,1,2],h=33,f=19);hashes=set()
        for mode in ('fifo','dynamic','feedback','fixed'):
            kw={'fixed_assignment':[1,0,1,0,1]} if mode=='fixed' else {}
            r,_=self.run_engine(w,[5,3],resources=self.resources([5,3]),group=2,dispatch=mode,
                                runtime_fsm=True,arbiter='stock',**kw)
            hashes.add(replay(w,r)['sha256_bf16'])
            if mode=='fixed':self.assertEqual([a['core'] for a in r['dispatch_audit']],kw['fixed_assignment'])
            if mode=='feedback':
                completions=[e for e in r['trace'] if e['event']=='expert_drained']
                updates=[e for e in r['trace'] if e['event']=='service_feedback']
                self.assertEqual(len(updates),len(completions))
                self.assertTrue(all(any(x['core']==e['core'] and x['cycle']<=e['cycle'] for x in completions) for e in updates))
        self.assertEqual(len(hashes),1)

    def test_every_dse_resource_point_replays_tails(self):
        import sys
        from frontend import COMPILER_DIR
        sys.path.insert(0,str(COMPILER_DIR))
        from robust_space import designs
        w=workload([4,2],h=33,f=19);hashes=set()
        for d in designs():
            with self.subTest(design=d['id']):
                r,_=self.run_engine(w,d['lanes'],resources=d['resources'],group=d['group'],
                                    runtime_fsm=True,dispatch='feedback',arbiter='stock')
                hashes.add(replay(w,r)['sha256_bf16'])
        self.assertEqual(len(hashes),1)

    def test_tail_partition_preserves_full_z_and_weight_bytes(self):
        w=workload([1,2,7],h=513,f=35);hashes=set()
        for lanes in compiler.ROBUST_ORGANIZATIONS:
            for policy in ('fifo','dynamic','feedback'):
                for split in (False,True):
                    r,_=self.run_engine(w,list(lanes),resources=self.resources(lanes),group=2,
                        runtime_fsm=True,dispatch=policy,tail_partition=split,arbiter='stock')
                    hashes.add(replay(w,r)['sha256_bf16'])
                    self.assertEqual(r['tail_partition_count'],int(split and len(lanes)==2))
                    if split and len(lanes)==2:
                        self.assertEqual(sum(c['stats']['z_exchange_bytes'] for c in r['cores']),2*7*35)
                    self.assertEqual(r['useful_macs'],sum(3*e['Me']*e['H']*e['F'] for e in w['experts']))
        self.assertEqual(len(hashes),1)
