"""Current/Next protocol integration and numerical replay of the timed schedule."""
import copy
import unittest
import test_engine
from test_engine import fixture
from trace_payload import replay


def workload(ms, h=33, f=19):
    w = fixture(max(ms), h, f)
    w["id"] = "runtime_protocol_" + "_".join(map(str, ms))
    w["experts"] = []
    w["top_k"] = len(ms)
    for t, m in enumerate(ms):
        e = copy.deepcopy(fixture(m, h, f)["experts"][0])
        e["id"] = t
        e["route_slots"] = [t]*m
        e["route_scores"] = [1/len(ms)]*m
        for weight in e["weights"].values():
            weight["hbm_base"] += t*3*16*1024**2
        w["experts"].append(e)
    w["tokens"] = [{"token_index": i, "sample_id": "synthetic_"+str(i),
                    "routes": [{"expert_id": t, "slot": t, "score": 1/len(ms)}
                               for t, m in enumerate(ms) if i<m]} for i in range(max(ms))]
    return w


class RuntimeTests(unittest.TestCase):
    run_engine = test_engine.AnalyticalEngineIntegrationTests.run_engine

    def run_case(self, w, lanes=(4, 2), **kw):
        config = dict(dispatch="dynamic", window=8, runtime_fsm=True, next_prefetch=True)
        config.update(kw)
        r, _ = self.run_engine(w, list(lanes), **config)
        self.assertEqual(r["dma_transactions_accepted"]*32, r["weight_bytes"])
        self.assertEqual(r["dma_transactions_accepted"], r["dma_transactions_landed"])
        self.assertLessEqual(r["pending_window_peak"], 8)
        for c in r["cores"]:
            self.assertLessEqual(c["stats"]["next_peak_bytes"], 4096)
            self.assertLessEqual(c["stats"]["weight_peak_bytes"], (10 if len(lanes)==1 else 5)*4096)
            self.assertLessEqual(c["stats"]["workspace_peak_bytes"], c["capacity"])
        return r

    def test_more_than_eight_tasks_unique_owner_no_next_overwrite(self):
        w = workload([4, 2, 7, 1, 3, 2, 1, 6, 2, 4, 3, 1, 2])
        r = self.run_case(w)
        self.assertEqual(r["dispatch_decisions"], 13)
        self.assertEqual(r["pending_window_peak"], 8)
        self.assertGreater(r["input_backpressure_cycles"], 0)
        self.assertTrue(replay(w, r)["all_bit_exact"])

    def test_dma_exhaustion_continued_fetch_and_all_phases(self):
        w = workload([4, 2], h=513, f=19)
        r = self.run_case(w, credits=128, dma_ready_period=7, dma_ready_cycles=2)
        self.assertGreater(r["dma_transactions_accepted"], 129)
        self.assertEqual(r["credit_peak"], 128)
        self.assertEqual({e["request"]["phase"] for e in r["trace"] if e["event"]=="dma_fire"}, {0, 1, 4})
        self.assertTrue(replay(w, r)["all_bit_exact"])

    def test_startup_late_response_and_early_prefetch(self):
        w = workload([1, 1, 1, 1], h=17, f=9)
        late = self.run_case(w, lanes=(6,), hbm_latency_ns=500, dma_ready_after=1000, credits=2)
        first_issue = min(e["cycle"] for e in late["trace"] if e["event"]=="mac_issue")
        self.assertGreaterEqual(first_issue, 1500)
        self.assertTrue(replay(w, late)["all_bit_exact"])
        early = self.run_case(w, lanes=(6,))
        self.assertGreater(sum(c["stats"]["next_ready_at_promotion"] for c in early["cores"]), 0)
        self.assertTrue(replay(w, early)["all_bit_exact"])

    def test_m_n_k_tails_all_organizations_and_four_modes_bit_exact(self):
        w = workload([7, 2, 1], h=513, f=19)
        hashes=set()
        for lanes in ((6,), (3, 3), (4, 2)):
            for dispatch in ("fifo", "dynamic"):
                for prefetch in (False, True):
                    with self.subTest(lanes=lanes, dispatch=dispatch, prefetch=prefetch):
                        r = self.run_case(w, lanes, dispatch=dispatch, next_prefetch=prefetch)
                        hashes.add(replay(w, r)["sha256_bf16"])
        self.assertEqual(len(hashes), 1)

    def test_down_k_tail_and_shared_combine(self):
        w = workload([3, 2], h=9, f=513)
        w["experts"][0]["is_shared"] = True
        w["experts"][0]["id"] = -1
        r = self.run_case(w)
        self.assertTrue(replay(w, r)["all_bit_exact"])

    def test_full_report_repeated_exactly(self):
        w = workload([4, 2, 3], h=33, f=19)
        self.assertEqual(self.run_case(w), self.run_case(w))

    def test_policy_matrix_same_bytes_bit_exact_and_binding_error(self):
        w = workload([7, 2, 1, 3, 7], h=33, f=19)
        w["experts"][-1]["is_shared"] = True
        w["experts"][-1]["id"] = -1
        variants = [{}, {"shared_large": True}]
        variants += [{"next_prefetch_ready_threshold": n} for n in (1, 2, 4)]
        variants += [{"late_bind_cycles": n} for n in (128, 512, 2048)]
        hashes, weights = set(), set()
        for lanes in ((6,), (3, 3), (4, 2)):
            for arbiter in ("rr", "stock"):
                for policy in variants:
                    with self.subTest(lanes=lanes, arbiter=arbiter, policy=policy):
                        r = self.run_case(w, lanes, arbiter=arbiter, **policy)
                        hashes.add(replay(w, r)["sha256_bf16"])
                        weights.add(r["weight_bytes"])
                        for a in r["dispatch_audit"]:
                            self.assertGreater(len(a["eligible_cores"]), 0)
                            self.assertEqual(a["actual_minus_predicted_cycles"],
                                             a["actual_finish_cycle"]-a["predicted_finish_cycle"])
                            if "late_bind_cycles" in policy:
                                self.assertLess(a["remaining_estimate"], policy["late_bind_cycles"])
                            if policy.get("shared_large") and a["expert"] == -1:
                                self.assertEqual(a["core"], 0)
        self.assertEqual(len(hashes), 1)
        self.assertEqual(len(weights), 1)

    def test_prefetch_permission_requires_all_three_projections_sent(self):
        w = workload([7, 2, 3, 1], h=513, f=19)
        for n in (1, 2, 4):
            r = self.run_case(w, (6,), next_prefetch_ready_threshold=n,
                              dma_ready_period=7, dma_ready_cycles=3)
            fired = {}
            checked = 0
            for ev in r["trace"]:
                if ev["event"] == "dma_fire":
                    t = ev["request"]["task"]
                    fired[t] = fired.get(t, 0) + 32
                if ev["event"] == "next_slot_reserved" and ev["current_task"] is not None:
                    t = ev["current_task"]
                    expected = sum(x["physical_bytes"] for x in w["experts"][t]["weights"].values())
                    self.assertEqual(fired[t], expected)
                    self.assertTrue(ev["current_all_requests_sent"])
                    self.assertLess(ev["current_ready_tiles"], n)
                    checked += 1
            self.assertGreater(checked, 0)
            self.assertTrue(replay(w, r)["all_bit_exact"])

    def test_late_binding_can_leave_fifo_waiting(self):
        r = self.run_case(workload([7, 7, 3, 2], h=513, f=19), late_bind_cycles=128)
        self.assertGreater(r["late_bind_wait_cycles"], 0)
        self.assertEqual(len(r["dispatch_audit"]), 4)

    def test_surplus_rules_tails_and_phase_ahead_payload(self):
        w = workload([7,2,1,3],h=513,f=19)
        hashes=set()
        for lanes in ((6,), (3,3), (4,2)):
            for rules in (0,1,2,3,4):
                with self.subTest(lanes=lanes,rules=rules):
                    r=self.run_case(w,lanes,surplus_rules=rules,surplus_margin_cycles=64,
                                    arbiter="stock",dma_ready_period=7,dma_ready_cycles=3)
                    hashes.add(replay(w,r)["sha256_bf16"])
                    self.assertEqual(r["supply_diagnostics"]["credit_completions"],r["dma_transactions_landed"])
                    if rules==4:
                        self.assertGreater(sum(c["phase_ahead_tiles"] for c in r["supply_diagnostics"]["cores"]),0)
                        self.assertTrue(any(t["event"]=="phase_ahead_reserved" for t in r["trace"]))
        self.assertEqual(len(hashes),1)

    def test_ahead_down_with_multiple_k_segments_and_shared_output(self):
        w=workload([3,2,1],h=9,f=513)
        w["experts"][0]["is_shared"]=True
        w["experts"][0]["id"]=-1
        hashes=set()
        for lanes in ((6,), (3,3), (4,2)):
            for margin in (32,128):
                r=self.run_case(w,lanes,surplus_rules=4,surplus_margin_cycles=margin,arbiter="stock")
                hashes.add(replay(w,r)["sha256_bf16"])
        self.assertEqual(len(hashes),1)

    def test_expanded_credit_diagnostic_keeps_data_path_correct(self):
        w=workload([7,2,1,3],h=513,f=19)
        hashes=set()
        for lanes in ((6,), (3,3), (4,2)):
            for credits in (512,1024):
                r=self.run_case(w,lanes,surplus_rules=4,credits=credits,
                                diagnostic_credit_expansion=True,arbiter="stock")
                self.assertTrue(r["resource_conditions"]["not_eligible_for_equal_budget_claim"])
                self.assertEqual(r["dma_return_capacity_bytes"],32*credits)
                self.assertEqual(r["resource_conditions"]["extra_credit_tag_bytes"],2*(credits-256))
                hashes.add(replay(w,r)["sha256_bf16"])
        self.assertEqual(len(hashes),1)


if __name__ == "__main__":
    unittest.main()
