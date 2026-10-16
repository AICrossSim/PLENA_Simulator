"""Shared-row cooperation: exhaustive ownership and real-route conservation."""
from __future__ import annotations

from itertools import product
import unittest

from research.moe_dispatch.plena65536_shared.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES, phase
from research.moe_dispatch.plena65536_moe.model import job_cycles, schedule
from research.moe_dispatch.plena65536_moe.split_diagnostic import schedule_shared_rows, sequential_cooperative_columns
from research.moe_dispatch.plena65536_moe.workloads import load_workloads


class SharedRowsDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows = load_workloads("development")["development"]
        cls.cores = (Core(2, 128, 1, 128), Core(2, 256, 1, 64))

    def test_exhaustive_small_real_window_and_all_profiles(self):
        window = self.windows[0]
        shared = next(e for e in window["experts"] if e["is_shared"])
        routed = [e for e in window["experts"] if not e["is_shared"]]
        for profile in PROFILES:
            best = None
            for cut in range(window["batch"] + 1):
                bases = [0 if m == 0 else job_cycles(c, dict(shared, Me=m), profile)
                         for c, m in zip(self.cores, (cut, window["batch"] - cut))]
                for owners in product((0, 1), repeat=len(routed)):
                    loads = list(bases)
                    for owner, expert in zip(owners, routed):
                        loads[owner] += job_cycles(self.cores[owner], expert, profile)
                    best = max(loads) if best is None else min(best, max(loads))
            actual = schedule_shared_rows(self.cores, window, profile)
            self.assertEqual(actual["cycles"], best)
            self.assertLessEqual(actual["cycles"], schedule(self.cores, window, profile)["cycles"])

    def test_real_large_window_slices_and_whole_routed_conservation(self):
        window = next(w for w in self.windows if w["batch"] == 128)
        plan = schedule_shared_rows(self.cores, window)
        self.assertEqual(plan["row_cuts_evaluated"], 129)
        shared_jobs = [j for j in plan["jobs"] if j["is_shared"]]
        self.assertEqual(sorted(t for j in shared_jobs for t in j["token_indices"]), list(range(128)))
        self.assertEqual(sum(j["M"] for j in shared_jobs), 128)
        routed_jobs = [j for j in plan["jobs"] if not j["is_shared"]]
        self.assertEqual(sorted(j["expert_id"] for j in routed_jobs),
                         sorted(e["id"] for e in window["experts"] if not e["is_shared"]))
        for owner, total in enumerate(plan["core_cycles"]):
            queue = [j for j in plan["jobs"] if j["core_id"] == owner]
            self.assertEqual(sum(j["cycles"] for j in queue), total)
            if any(j["is_shared"] for j in queue):
                self.assertTrue(queue[0]["is_shared"])
            self.assertEqual([j["start"] for j in queue][1:], [j["end"] for j in queue][:-1])

    def test_single_core_owns_all_shared_rows_and_exact_budget_is_required(self):
        core = Core(2, 512, 1, 64)
        window = self.windows[0]
        plan = schedule_shared_rows((core,), window)
        self.assertEqual(plan["row_cut"], window["batch"])
        self.assertEqual(plan["cycles"], schedule((core,), window)["cycles"])
        with self.assertRaisesRegex(ValueError, "65,536"):
            schedule_shared_rows((self.cores[0],), window)

    def test_column_construction_matches_exhaustive_output_cuts(self):
        window = dict(self.windows[0], experts=[self.windows[0]["experts"][0]])
        expert = window["experts"][0]
        m, h, f = expert["Me"], expert["H"], expert["F"]
        left, right = self.cores
        for profile in PROFILES:
            gu = min(max(phase(left, m, cut, h, 2, profile), phase(right, m, f - cut, h, 2, profile))
                     for cut in range(f + 1))
            down = min(max(phase(left, m, cut, f, 1, profile), phase(right, m, h - cut, f, 1, profile))
                       for cut in range(h + 1))
            self.assertEqual(sequential_cooperative_columns(self.cores, window, profile)["cycles"], gu + down)


if __name__ == "__main__":
    unittest.main()
