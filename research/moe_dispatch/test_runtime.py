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


if __name__ == "__main__":
    unittest.main()
