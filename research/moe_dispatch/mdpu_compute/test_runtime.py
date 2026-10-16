"""Control conservation, backpressure, timing and compiler/array agreement."""
import itertools
import unittest

from research.moe_dispatch.mdpu_compute.runtime import (
    choose_owner, compiler_coefficients, estimate_cycles, prepare_experts,
    schedule, schedule_from_costs,
)
from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_dedicated.model import job_cycles
from research.moe_dispatch.plena65536_shared.timing import PROFILES


SINGLE = (Core(1, 512, 1, 128),)
HOMO = (Core(1, 256, 1, 128),) * 2
HETERO = (Core(1, 64, 1, 704), Core(1, 128, 1, 160))


def make_window(batch=8, counts=(1, 2, 4, 8, 1, 2)):
    return {"batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": 2048, "F": 2816, "is_shared": True},
        *[{"id": i, "Me": m, "H": 2048, "F": 1408, "is_shared": False}
          for i, m in enumerate(counts)],
    ]}


class RuntimeTests(unittest.TestCase):
    def test_selector_ready_valid_and_tie(self):
        for preferred, fd, fs, sd, ss, bypass in itertools.product(
                (0, 1), (5, 9), (5, 9), (False, True), (False, True), (False, True)):
            got = choose_owner(preferred, fd, fs, sd, ss, bypass)
            legal = ([c for c, space in enumerate((sd, ss)) if space]
                     if bypass else [preferred] if (sd, ss)[preferred] else [])
            expected = (min(legal, key=lambda c: ((fd, fs)[c], c != preferred))
                        if legal else None)
            self.assertEqual(got, expected)

    def test_guarded_admission_waits_for_better_full_queue(self):
        self.assertEqual(choose_owner(1, 4000, 300, True, False), 0)
        self.assertIsNone(choose_owner(1, 4000, 300, True, False, wait_best=True))
        self.assertEqual(choose_owner(1, 200, 300, True, False, wait_best=True), 0)
        self.assertEqual(choose_owner(1, 4000, 300, True, True, wait_best=True), 1)
        # Disabling bypass still enforces hard affinity regardless of wait flag.
        self.assertEqual(choose_owner(0, 4000, 300, True, False, bypass=False, wait_best=True), 0)

    def test_guarded_admission_terminates_after_backpressure(self):
        experts = make_window(2, (1,) * 11)["experts"]
        costs = [[1000] * len(experts), [100] * len(experts)]
        guarded = schedule_from_costs(HOMO, experts, costs, queue_depth=2, wait_best=True)
        forced = schedule_from_costs(HOMO, experts, costs, queue_depth=2)
        self.assertGreater(guarded["frontend_backpressure_cycles"], 0)
        self.assertTrue(all(x <= 2 for x in guarded["queue_max_occupancy"]))
        self.assertLess(guarded["cycles"], forced["cycles"])

    def test_default_preserves_legacy_result_and_policy_controls(self):
        w = make_window()
        self.assertEqual(schedule(HETERO, w), schedule(HETERO, w, wait_best=False))
        for p in ("greedy", "hard_threshold", "random"):
            a, b = schedule(HETERO, w, policy=p), schedule(HETERO, w, policy=p, wait_best=True)
            b.pop("wait_best")
            self.assertEqual(a, b)

    def test_compiler_coefficient_exactly_matches_array_timing(self):
        cores = SINGLE + HOMO + HETERO + (Core(32, 2048), Core(8, 256, 2, 4))
        for core, profile, m, f in itertools.product(cores, PROFILES, (1, 2, 3, 8, 31, 96, 128), (1408, 2816)):
            e = {"Me": m, "H": 2048, "F": f}
            self.assertEqual(estimate_cycles(core, e, profile), job_cycles(core, e, profile))
            self.assertGreater(compiler_coefficients(core, 2048, f, profile)["coefficient"], 0)

    def test_hand_calculated_shadow_overlap(self):
        experts = make_window(2, (1, 1))["experts"]
        prefetched = schedule_from_costs(SINGLE, experts, [[20, 20, 20]], prefetch=True)
        demand = schedule_from_costs(SINGLE, experts, [[20, 20, 20]], prefetch=False)
        self.assertEqual(prefetched["cycles"], 65)  # one ingress + four setup + 60 compute
        self.assertEqual(demand["cycles"], 73)  # one ingress + three four-cycle setup
        self.assertEqual(prefetched["context_overlap_cycles"], 8)
        self.assertEqual(demand["context_overlap_cycles"], 0)
        self.assertEqual(prefetched["core_cycles"], demand["core_cycles"])

    def test_zero_context_does_not_manufacture_prefetch_gain(self):
        w = make_window()
        for cores in (SINGLE, HOMO, HETERO):
            a = schedule(cores, w, context_cycles=0, prefetch=True)
            b = schedule(cores, w, context_cycles=0, prefetch=False)
            self.assertEqual(a["cycles"], b["cycles"])
            self.assertEqual(a["context_service_cycles"], 0)

    def test_queue_depth_and_input_backpressure(self):
        experts = make_window(8, (1,) * 12)["experts"]
        r = schedule_from_costs(SINGLE, experts, [[100] * 13], queue_depth=1)
        self.assertEqual(r["cycles"], 1305)
        self.assertEqual(r["queue_max_occupancy"], [1])
        self.assertGreater(r["frontend_backpressure_cycles"], 0)

    def test_arithmetic_never_overlaps_with_itself(self):
        for cores, policy, depth, context, prefetch in itertools.product(
                (SINGLE, HOMO, HETERO), ("mdpu", "greedy", "hard_threshold", "random"),
                (1, 8), (0, 4, 16), (False, True)):
            w = make_window()
            r = schedule(cores, w, policy=policy, queue_depth=depth,
                         context_cycles=context, prefetch=prefetch)
            self.assertEqual(len(r["jobs"]), len(w["experts"]))
            self.assertEqual(len({j["expert_id"] for j in r["jobs"]}), len(w["experts"]))
            self.assertEqual(sum(r["core_cycles"]), sum(j["cycles"] for j in r["jobs"]))
            self.assertTrue(all(x <= depth for x in r["queue_max_occupancy"]))
            self.assertEqual(r["context_overlap_cycles"] + r["context_exposed_cycles"],
                             r["context_service_cycles"])
            self.assertEqual(r["supply_delay_cycles"], 0)
            for c in range(len(cores)):
                jobs = sorted((j for j in r["jobs"] if j["core_id"] == c), key=lambda j: j["start"])
                for j in jobs:
                    self.assertLess(j["queue_entry"], j["claim"])
                    self.assertLessEqual(j["context_ready"], j["start"])
                    self.assertEqual(j["context_ready"], j["claim"] + context)
                    self.assertEqual(j["end"] - j["start"], j["cycles"])
                self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(jobs, jobs[1:])))

    def test_hard_affinity_and_no_bypass(self):
        w = make_window()
        for threshold in (1, 2, 4, 8):
            hard = schedule(HETERO, w, policy="hard_threshold", threshold=threshold)
            without = schedule(HETERO, w, policy="mdpu", threshold=threshold, bypass=False)
            self.assertEqual(hard["cycles"], without["cycles"])
            self.assertEqual(hard["steal_count"], 0)
            self.assertEqual(hard["affinity_deviation_count"], 0)
            self.assertIn(threshold, hard["boundary_stats"]["realizable_thresholds"])

    def test_single_baselines_share_same_controller_cost(self):
        w = make_window()
        results = [schedule(SINGLE, w, policy=p) for p in ("mdpu", "greedy", "random", "hard_threshold")]
        self.assertEqual(len({r["cycles"] for r in results}), 1)
        self.assertEqual(len({tuple(r["core_cycles"]) for r in results}), 1)

    def test_determinism_and_cached_cost_api(self):
        w = make_window()
        experts = prepare_experts(w)
        costs = [[job_cycles(c, e) for e in experts] for c in HETERO]
        a = schedule(HETERO, w, policy="random", seed=123)
        b = schedule_from_costs(HETERO, experts, costs, policy="random", seed=123, details=True)
        self.assertEqual(a, b)
        self.assertEqual(a, schedule(HETERO, w, policy="random", seed=123))

    def test_invalid_budget_and_queues_rejected(self):
        w = make_window()
        with self.assertRaises(ValueError):
            schedule((HOMO[0],), w)
        with self.assertRaises(ValueError):
            schedule(SINGLE, w, queue_depth=0)
        with self.assertRaises(ValueError):
            schedule(SINGLE, w, context_cycles=-1)


if __name__ == "__main__":
    unittest.main()
