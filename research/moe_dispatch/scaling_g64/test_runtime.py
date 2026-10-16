"""Regression and pooled FIFO/worker/control invariants for scaling."""
import itertools
import unittest
from collections import Counter

from research.moe_dispatch.mdpu_compute.runtime import schedule as old_schedule
from research.moe_dispatch.mdpu_compute.test_runtime import make_window
from research.moe_dispatch.plena65536_dedicated.model import job_cycles
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .geometry import Core
from .runtime import schedule, schedule_from_costs, estimate_cycles


class GeometryTests(unittest.TestCase):
    def test_g_is_group_count_not_reduction_width(self):
        core = Core.from_shape(4,256,64)
        self.assertEqual((core.b,core.r,core.groups,core.macs),(4,256,64,65536))
        self.assertNotEqual(1408%core.r,0)
        self.assertEqual(1408%64,0)

    def test_requested_budgets(self):
        for pm,budget in ((4,65536),(8,131072),(16,262144)):
            self.assertEqual(Core.from_shape(pm,256).macs,budget)
        with self.assertRaises(ValueError): Core.from_shape(4,255)


class RuntimeTests(unittest.TestCase):
    def test_64k_legal_legacy_result_regression(self):
        designs = [(Core.from_shape(4,256),),
                   (Core.from_shape(2,256),)*2,
                   (Core.from_shape(4,128),Core.from_shape(1,512))]
        window = make_window(16,(1,2,4,8,16,1,1,2,1,4,8))
        for cores,policy,prefetch,guard,context in itertools.product(
                designs, ("mdpu","greedy","hard_threshold","random"),
                (False,True),(False,True),(0,4)):
            args=dict(policy=policy,prefetch=prefetch,wait_best=guard,context_cycles=context,queue_depth=2)
            actual,old=schedule(cores,window,**args),old_schedule(cores,window,**args)
            old_claims=Counter((j["queue_owner"],j["claim"]) for j in old["jobs"])
            new_claims=Counter((j["queue_owner"],j["claim"]) for j in actual["jobs"])
            self.assertEqual(max(new_claims.values()),1)
            if max(old_claims.values())==1:
                self.assertEqual(actual,old)

    def test_own_claim_and_head_steal_cannot_pop_same_fifo_twice(self):
        from research.moe_dispatch.plena65536_moe.workloads import load_workloads
        windows=load_workloads("heldout")["heldout"]
        window=next(w for w in windows if w["id"]=="heldout_swe_b8_l2_s11")
        cores=(Core.from_shape(2,256),)*2
        old=old_schedule(cores,window,wait_best=False)
        collided=[j for j in old["jobs"] if j["queue_owner"]==0 and j["claim"]==8339]
        self.assertEqual({j["expert_id"] for j in collided},{60,63})
        actual=schedule(cores,window,wait_best=False)
        claims=Counter((j["queue_owner"],j["claim"]) for j in actual["jobs"])
        self.assertEqual(max(claims.values()),1)
        by_id={j["expert_id"]:j for j in actual["jobs"]}
        self.assertGreater(by_id[63]["claim"],by_id[60]["claim"])

    def test_estimator_matches_charged_arithmetic(self):
        cores=[Core.from_shape(pm,256) for pm in (4,8,16)]
        cores += [Core.from_shape(pm,512) for pm in (1,2,4)]
        for core,profile,m,f in itertools.product(cores,PROFILES,(1,2,7,16,128),(1408,2816)):
            e={"Me":m,"H":2048,"F":f}
            self.assertEqual(estimate_cycles(core,e,profile),job_cycles(core,e,profile))

    def test_scaled_utilization_uses_total_mac_budget(self):
        for pm,budget in ((4,65536),(8,131072),(16,262144)):
            r=schedule((Core.from_shape(pm,256),),make_window())
            self.assertEqual(r["wall_peak_utilization"],r["useful_MACs"]/(budget*r["cycles"]))

    def test_reject_duplicate_or_missing_worker(self):
        cs=(Core.from_shape(8,128),Core.from_shape(1,512),Core.from_shape(1,512))
        for groups in (((0,),(1,)),((0,),(1,1)),((0,),(),(1,2))):
            with self.assertRaises(ValueError):schedule(cs,make_window(),groups=groups)

    def test_pool_capacity_port_and_worker_conservation(self):
        cs=(Core.from_shape(16,128),Core.from_shape(2,512),Core.from_shape(2,512))
        w=make_window(128,tuple([1,2,4,8,16,32,64,128]*4))
        for policy,prefetch,context,depth in itertools.product(
                ("mdpu","hard_threshold","random"),(False,True),(0,4),(1,8)):
            r=schedule(cs,w,groups=((0,),(1,2)),policy=policy,prefetch=prefetch,
                       context_cycles=context,queue_depth=depth,wait_best=True)
            self.assertEqual(len(r["queue_max_occupancy"]),2)
            self.assertTrue(all(n<=depth for n in r["queue_max_occupancy"]))
            claims=Counter((x["queue_group"],x["cycle"]) for x in r["dequeue_events"])
            self.assertEqual(max(claims.values()),1)
            self.assertEqual(len(r["jobs"]),len(w["experts"]))
            for c in range(3):
                jobs=sorted((j for j in r["jobs"] if j["core_id"]==c),key=lambda j:j["start"])
                self.assertTrue(all(a["end"]<=b["start"] for a,b in zip(jobs,jobs[1:])))
                self.assertEqual(sum(j["cycles"] for j in jobs),r["core_cycles"][c])
            if policy=="hard_threshold":
                self.assertEqual(r["steal_count"],0)
                self.assertTrue(all(j["group_id"]==j["preferred_group"] for j in r["jobs"]))

    def test_four_workers_still_only_two_fifos(self):
        cs=(Core.from_shape(1,512),)*4
        r=schedule(cs,make_window(8,tuple([1]*24)),groups=((0,1),(2,3)),wait_best=True)
        self.assertEqual(r["group_MACs"],[65536,65536])
        self.assertEqual(len(r["queue_max_occupancy"]),2)
        self.assertTrue(all(b>0 for b in r["core_cycles"]))

    def test_two_lanes_have_independent_compute_overlap(self):
        cs=(Core.from_shape(8,128),Core.from_shape(1,512),Core.from_shape(1,512))
        experts=make_window(2,(1,1,1,1))["experts"]
        r=schedule_from_costs(cs,experts,[[100]*5]*3,groups=((0,),(1,2)),
                              policy="hard_threshold",context_cycles=0,details=False)
        # Shared starts at1, routed lane1 at2 and lane2 at3; each E lane
        # performs two jobs. One E FIFO can launch different lanes concurrently.
        self.assertEqual(r["core_cycles"],[100,200,200])
        self.assertEqual(r["cycles"],203)

    def test_prefetch_zero_context_cannot_change_service(self):
        cs=(Core.from_shape(8,128),Core.from_shape(1,512),Core.from_shape(1,512))
        # With fixed affinity, preconfig can change early lane claims but must
        # neither create arithmetic work nor remove a descriptor.
        a=schedule(cs,make_window(),groups=((0,),(1,2)),policy="hard_threshold",prefetch=True,context_cycles=0)
        b=schedule(cs,make_window(),groups=((0,),(1,2)),policy="hard_threshold",prefetch=False,context_cycles=0)
        self.assertEqual(sum(a["core_cycles"]),sum(b["core_cycles"]))
        self.assertEqual(a["context_service_cycles"],0)
        self.assertEqual(a["useful_MACs"],b["useful_MACs"])


if __name__ == "__main__":unittest.main()
