"""Independent complete-window timing and assignment replay checks."""
from fractions import Fraction
from itertools import product
from random import Random
import unittest

import numpy as np

from research.moe_dispatch.plena65536_moe.bounds import fractional_two_machine_exact
from research.moe_dispatch.plena65536_moe.model import (
    balanced_score, cost_table, expert_key, fractional_pair_bound, group_window,
    job_cycles, schedule, useful_macs, window_histograms)
from research.moe_dispatch.plena65536_shared.geometry import Core, enumerate_cores
from research.moe_dispatch.plena65536_shared.timing import PROFILES


def independently_priced_job(core, expert, profile):
    """Unshared integer arithmetic, independent of timing.phase/cost_table."""
    m, h, f = (int(expert[name]) for name in ("Me", "H", "F"))
    rows = (m + core.pm - 1) // core.pm
    gu_records = 2 * rows * ((f + core.pn - 1) // core.pn)
    down_records = rows * ((h + core.pn - 1) // core.pn)
    h_feed = ((h + core.r - 1) // core.r) * core.b
    f_feed = ((f + core.r - 1) // core.r) * core.b
    drain = core.b + 11 + (core.groups.bit_length() - 1) * profile.tree_stage
    if profile.serialized:
        return gu_records * (h_feed + drain) + down_records * (f_feed + drain)
    return gu_records * h_feed + down_records * f_feed + 2 * drain


def toy_window(batch=4):
    # Equal shapes retain distinct expert identities and must not disappear.
    return {"id": "toy", "batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": 2048, "F": 2816, "is_shared": True},
        {"id": 0, "Me": 1, "H": 2048, "F": 1408},
        {"id": 1, "Me": 1, "H": 2048, "F": 1408},
        {"id": 2, "Me": 2, "H": 2048, "F": 1408},
        {"id": 3, "Me": 3, "H": 2048, "F": 1408},
        {"id": 4, "Me": 4, "H": 2048, "F": 1408},
    ]}


class CompleteModelTests(unittest.TestCase):
    pairs = ((Core(2, 128, 1, 128), Core(4, 512, 1, 16)),
             (Core(2, 128, 1, 96), Core(2, 256, 1, 80)),
             (Core(2, 128, 1, 128), Core(2, 128, 1, 128)))

    def test_scalar_and_vector_prices_against_independent_arithmetic(self):
        all_cores = enumerate_cores()
        stride = max(1, len(all_cores) // 29)
        cores = all_cores[::stride]
        types = [(1, 9, 13), (2, 2048, 1408), (3, 2048, 1408),
                 (8, 2048, 2816), (128, 2048, 2816)]
        for profile in PROFILES:
            table = cost_table(cores, types, profile)
            for ci, core in enumerate(cores):
                for ti, (m, h, f) in enumerate(types):
                    expert = dict(Me=m, H=h, F=f)
                    expected = independently_priced_job(core, expert, profile)
                    self.assertEqual(job_cycles(core, expert, profile), expected)
                    self.assertEqual(int(table[ci, ti]), expected)

    def test_every_expert_in_exact_whole_window_schedule(self):
        window = toy_window()
        experts = window["experts"]
        for cores in self.pairs:
            for profile in PROFILES:
                got = schedule(cores, window, profile)
                optimum = min(max(sum(independently_priced_job(cores[c], e, profile)
                                      for e,owner in zip(experts, owners) if owner == c)
                                  for c in (0,1))
                              for owners in product((0,1), repeat=len(experts)))
                self.assertEqual(got["cycles"], optimum)
                self.assertEqual(got["contract"], "whole_expert_exact")
                self.assertEqual(sorted(j["expert_id"] for j in got["jobs"]), [-1,0,1,2,3,4])
                self.assertEqual(got["useful_MACs"], useful_macs(window))
                by_id = {e["id"]: e for e in experts}
                for side in (0,1):
                    queue = sorted((j for j in got["jobs"] if j["core_id"] == side),
                                   key=lambda j: j["start"])
                    now = 0
                    for job in queue:
                        self.assertEqual(job["start"], now)
                        duration = independently_priced_job(cores[side], by_id[job["expert_id"]], profile)
                        self.assertEqual(job["end"] - job["start"], duration)
                        self.assertEqual(job["cycles"], duration)
                        now = job["end"]
                    self.assertEqual(now, got["core_cycles"][side])
                self.assertEqual(max(got["core_cycles"]), optimum)
                self.assertEqual(got, schedule(cores, window, profile))

    def test_single_core_contains_shared_and_all_routed_work(self):
        core = Core(2, 2048, 1, 16)
        window = toy_window()
        for profile in PROFILES:
            got = schedule((core,), window, profile)
            expected = sum(independently_priced_job(core,e,profile) for e in window["experts"])
            self.assertEqual(got["cycles"], expected)
            self.assertEqual(len(got["jobs"]), len(window["experts"]))

    def test_vector_lp_against_exact_rational_relaxation(self):
        rng = Random(65536)
        # General integer tables exercise sorting/frontier logic separately
        # from architecture costs, including repeated costs and zero counts.
        for _ in range(25):
            costs = np.asarray([[rng.randint(1,100000) for _ in range(9)]
                                for _ in range(6)], dtype=np.int64)
            ids = np.asarray([(a,b) for a in range(6) for b in range(a,6)], dtype=np.int32)
            hists = np.asarray([[rng.randint(0,7) for _ in range(9)] for _ in range(4)], dtype=np.int64)
            got = fractional_pair_bound(costs,ids,hists)
            for pi,(a,b) in enumerate(ids):
                for wi,hist in enumerate(hists):
                    expected = fractional_two_machine_exact(costs[a].tolist(),costs[b].tolist(),hist.tolist())
                    self.assertLessEqual(Fraction.from_float(float(got[pi,wi])), expected)
                    self.assertLessEqual(abs(float(got[pi,wi])-float(expected)),1e-6)

    def test_full_window_vector_bounds_and_batch_balancing(self):
        windows = [toy_window(2),toy_window(2),toy_window(4),toy_window(8)]
        types,hists,weights = window_histograms(windows)
        self.assertAlmostEqual(float(weights.sum()),1.0)
        self.assertEqual(weights.tolist(),[1/6,1/6,1/3,1/3])
        cores = [core for pair in self.pairs for core in pair]
        ids = np.asarray([(0,1),(2,3),(4,5)],dtype=np.int32)
        for profile in PROFILES:
            costs = cost_table(cores,types,profile)
            bounds = fractional_pair_bound(costs,ids,hists)
            for pi,(a,b) in enumerate(ids):
                for wi,hist in enumerate(hists):
                    expected = fractional_two_machine_exact(costs[a].tolist(),costs[b].tolist(),hist.tolist())
                    self.assertLessEqual(bounds[pi,wi],float(expected)+1e-6)
                    self.assertLessEqual(abs(bounds[pi,wi]-float(expected)),1e-6)
        self.assertAlmostEqual(balanced_score([4,16,8,64],windows),16.0)
        self.assertEqual(sum(len(es) for _,es in group_window(windows[0])),len(windows[0]["experts"]))
        self.assertEqual(len({expert_key(e) for e in windows[0]["experts"]}),len(group_window(windows[0])))

    def test_uncertified_overflow_never_returns_a_pruning_bound(self):
        costs = np.asarray([[2**40,2**40+1],[2**40+3,2**40+5]],dtype=np.int64)
        ids = np.asarray([(0,1)],dtype=np.int32)
        with self.assertRaisesRegex(OverflowError, "no pruning bound returned"):
            fractional_pair_bound(costs,ids,np.asarray([[1,2]],dtype=np.int64))
        # An empty workload needs no risky products and has an exact zero LB.
        np.testing.assert_equal(fractional_pair_bound(costs,ids,np.zeros((1,2),dtype=np.int64)), [[0.0]])


if __name__ == "__main__":
    unittest.main()
