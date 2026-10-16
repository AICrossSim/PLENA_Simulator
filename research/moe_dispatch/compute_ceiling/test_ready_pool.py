"""Manual and independent dependency checks for ready-record scheduling."""
from types import SimpleNamespace
import unittest

from .ready_pool import schedule_ready_pool
from .search import Shape


TINY = SimpleNamespace(name='manual_L3',initiation_interval=1,
                       completion_latency=lambda core:3)


def expert(eid,m=1,h=16,f=16):
    return {'id':eid,'Me':m,'H':h,'F':f,'is_shared':False}


class ReadyPoolTests(unittest.TestCase):
    def test_two_single_segment_experts_manual_cycles(self):
        # GUI issues 0/1/2/3; Down at4/6; last commits7/9.
        result = schedule_ready_pool([expert(0),expert(1)],Shape(1,16,16),TINY,record_trace=True)
        self.assertEqual(result['cycles'],9)
        self.assertEqual(result['whole_expert_baseline_cycles'],14)
        self.assertEqual(result['issues'],6)
        self.assertEqual([t['issue_cycle'] for t in result['issue_trace']],[0,1,2,3,4,6])

    def test_single_expert_two_K_segments_manual_cycles(self):
        # GUI0k0/GUI1k0=0/1; next segments3/4 finish6/7;
        # Down independent output records7/8 finish10/11.
        result = schedule_ready_pool([expert(0,h=32)],Shape(1,16,16),TINY)
        self.assertEqual(result['cycles'],11)
        self.assertEqual(result['whole_expert_baseline_cycles'],11)
        self.assertEqual(result['issues'],6)

    def test_two_experts_with_K_dependencies_manual_cycles(self):
        result = schedule_ready_pool([expert(0,h=32),expert(1,h=32)],Shape(1,16,16),TINY)
        self.assertEqual(result['cycles'],14)
        self.assertEqual(result['whole_expert_baseline_cycles'],22)

    def test_record_dependencies_and_expert_barriers_from_trace(self):
        jobs = [expert(2,m=5,h=65,f=49),expert(1,m=2,h=65,f=49)]
        core = Shape(3,4,16)
        result = schedule_ready_pool(jobs,core,record_trace=True)
        last_issue = -1
        prior = {}
        gu_end = {}
        for item in result['issue_trace']:
            self.assertGreaterEqual(item['issue_cycle'],last_issue+1)
            last_issue = item['issue_cycle']
            r = item['record']
            if r in prior:
                self.assertEqual(item['k_segment'],prior[r]['k_segment']+1)
                self.assertGreaterEqual(item['issue_cycle'],prior[r]['commit_cycle'])
            else:
                self.assertEqual(item['k_segment'],0)
            prior[r] = item
            if item['phase'] != 'Down':
                gu_end[item['expert_id']] = max(gu_end.get(item['expert_id'],0),item['commit_cycle'])
        for item in result['issue_trace']:
            if item['phase'] == 'Down':
                self.assertGreaterEqual(item['issue_cycle'],gu_end[item['expert_id']])
        self.assertEqual(result['cycles'],max(t['commit_cycle'] for t in result['issue_trace']))
        self.assertEqual(result['useful_macs'],sum(3*e['Me']*e['H']*e['F'] for e in jobs))
        self.assertEqual(result['issued_macs'],result['issues']*core.macs)
        self.assertEqual(result['padding_macs'],result['issued_macs']-result['useful_macs'])
        self.assertLessEqual(result['peak_inflight_dot_results'],result['completion_latency_cycles'])

    def test_repeated_output_and_fallback_are_deterministic(self):
        jobs = [expert(1,m=4,h=128,f=64),expert(2,m=1,h=128,f=64)]
        core = Shape(2,8,32)
        a = schedule_ready_pool(jobs,core)
        b = schedule_ready_pool(jobs,core)
        self.assertEqual(a,b)
        self.assertEqual(a['best_feasible_cycles'],min(a['cycles'],a['whole_expert_baseline_cycles']))
        self.assertGreater(a['peak_live_partial_bytes'],0)

    def test_empty_core_and_invalid_latency_contract(self):
        self.assertEqual(schedule_ready_pool([],Shape(1,16,16))['cycles'],0)
        slow = SimpleNamespace(name='II2',initiation_interval=2,completion_latency=lambda core:3)
        with self.assertRaises(ValueError):
            schedule_ready_pool([expert(0)],Shape(1,16,16),slow)

    def test_non_power_of_two_PK_uses_current_campaign_timing(self):
        jobs = [expert(1,m=2,h=96,f=80),expert(2,m=1,h=96,f=80)]
        for profile in ('flat20','log_stage1','log_stage2'):
            result = schedule_ready_pool(jobs,Shape(1,8,48),profile,record_trace=True)
            self.assertGreaterEqual(result['cycles'],result['issues'])
            self.assertEqual(result['issued_macs'],result['issues']*48*8)
            self.assertTrue(all(t['issue_cycle'] >= t['previous_commit_cycle']
                                for t in result['issue_trace']))


if __name__ == '__main__':
    unittest.main()
