"""Native callbacks, conserved credits/bytes and numerical replay of their schedule."""
import copy, json, os, subprocess, tempfile, unittest
from pathlib import Path
from test_runtime import workload
from trace_payload import replay
from research.moe_dispatch.frontend import compiler, default_binary
from research.moe_dispatch.native_profile import config

class NativeProfileTests(unittest.TestCase):
    def run_case(self,w,lanes,**changes):
        w=copy.deepcopy(w);w['engine_layout']=compiler.engine_layout(w,lanes,4)
        cfg=config(lanes,trace=True);cfg.update(changes)
        with tempfile.TemporaryDirectory(prefix='plena-native-profile-') as d:
            d=Path(d);(d/'w.json').write_text(json.dumps(w));(d/'c.json').write_text(json.dumps(cfg))
            reports=[]
            for repeat in (1,2):
                proc=subprocess.run([str(default_binary()),str(d/'w.json'),str(d/'c.json'),str(d/'r.json')],capture_output=True,text=True,timeout=90)
                self.assertEqual(proc.returncode,0,proc.stderr[-5000:])
                reports.append(json.loads((d/'r.json').read_text()))
            self.assertEqual(reports[0],reports[1]);r=reports[0]
        n=r['native_hbm'];p=r['live_timing_profile']
        self.assertEqual(n['pending'],0)
        self.assertEqual(n['accepted'],n['completed'])
        self.assertEqual(n['accepted']*32,r['weight_bytes'])
        self.assertEqual(n['completed'],r['dma_transactions_landed'])
        self.assertEqual(p['weight_callbacks_returned'],n['completed'])
        self.assertLessEqual(r['credit_peak'],cfg['credits'])
        self.assertLessEqual(p['mac_active_union_cycles'],r['cycles'])
        self.assertLessEqual(p['fetch_mac_overlap_cycles'],p['mac_active_union_cycles'])
        self.assertGreater(p['request_latency_mean_cycles'],0)
        self.assertTrue(replay(w,r)['all_bit_exact'])
        return r

    def test_tails_and_shared_ffn_on_all_three_organizations(self):
        for h,f,ms in ((33,19,[4,2]),(513,19,[7,2,1]),(9,513,[3,2])):
            w=workload(ms,h,f)
            w['experts'][0].update(is_shared=True,id=-1)
            for lanes in ([6],[3,3],[4,2]):
                with self.subTest(h=h,f=f,lanes=lanes):self.run_case(w,lanes)

    def test_native_with_ready_backpressure_and_tiny_credit_window(self):
        r=self.run_case(workload([4,2,1],513,19),[4,2],credits=2,
            dma_ready_period=7,dma_ready_cycles=2)
        self.assertEqual(r['credit_peak'],2)
        self.assertGreater(sum(c['stats']['dma_backpressure_cycles'] for c in r['cores']),0)

    def test_native_reject_retry_preserves_address_and_cursor(self):
        cfg=config([6]);
        # Force a native read-queue rejection independently of the core DMA-ready gate.
        # HBM12's buffer capacity is a native controller param, never a campaign setting.
        for c in cfg['native_hbm_config']['memory_system']['controllers']:
            c['read_buffer_size']=1
        w=workload([4,2],513,19)
        r=self.run_case(w,[6],native_hbm_config=cfg['native_hbm_config'])
        self.assertGreater(r['native_hbm']['rejected_attempts'],0)

if __name__=='__main__':unittest.main()
