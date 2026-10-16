"""Correctness and falsification tests for the finite-cache/DMA sensitivity."""
from __future__ import annotations

from math import inf
import unittest

from research.moe_dispatch.cache_route_a.model import (
    Block, FROZEN_CORES, MIB, WeightCache, blocks_for_job, run_window,
)
from research.moe_dispatch.plena65536_affinity.model import schedule


def window(batch=4, routed=(1, 2, 4), *, h=256, f=256):
    return {"window_id": "test_layer", "batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": h, "F": 2 * f, "is_shared": True},
        *[{"id": i, "Me": m, "H": h, "F": f, "is_shared": False}
          for i, m in enumerate(routed)],
    ]}


def block(key, size, shared=False):
    return Block((key,), -1 if shared else 0, "gate", 0, 1, size, 1, shared)


class CacheTests(unittest.TestCase):
    def test_lru_evicts_only_unprotected(self):
        c = WeightCache(30)
        a, b, d = block("a", 10), block("b", 10), block("d", 20)
        c.seed([a, b])
        c.acquire(a)
        self.assertIsNotNone(c.reserve(d, ready=True))
        self.assertIn(a.key, c.entries)
        self.assertNotIn(b.key, c.entries)
        self.assertEqual(c.used_bytes, 30)
        self.assertEqual(c.evicted_bytes, 10)

    def test_rejected_reservation_has_no_partial_evictions(self):
        c = WeightCache(30)
        a, b, d = block("a", 20), block("b", 10), block("d", 20)
        c.seed([a, b])
        c.acquire(a)
        self.assertIsNone(c.reserve(d))
        self.assertEqual(c.used_bytes, 30)
        self.assertIn(b.key, c.entries)
        self.assertEqual(c.evictions, 0)

    def test_pinned_reservation_is_not_extra_capacity(self):
        c = WeightCache(30, "pin_shared", 20)
        a, s, b = block("a", 10), block("shared", 20, True), block("b", 10)
        c.seed([a])
        c.seed([s])
        c.seed([b])
        self.assertEqual(c.used_bytes, 30)
        self.assertEqual(c.nonpinned_bytes, 10)
        self.assertIn(s.key, c.entries)
        self.assertNotIn(a.key, c.entries)

    def test_oversized_strip_and_different_pinned_layer_fail(self):
        c = WeightCache(30, "pin_shared", 20)
        with self.assertRaises(ValueError):
            c.reserve(block("oversize", 11))
        c.seed([block("shared", 20, True)])
        with self.assertRaises(ValueError):
            c.reserve(block("other_shared", 20, True))


class ModelTests(unittest.TestCase):
    def test_block_compute_and_weight_conservation(self):
        w = window()
        for cores in FROZEN_CORES.values():
            plan = schedule(cores, w)
            total_bytes = 0
            loads = [0] * len(cores)
            for job in plan["jobs"]:
                owner = job["core_id"]
                blocks = blocks_for_job(cores[owner], job, "test_layer")
                self.assertEqual(sum(b.cycles for b in blocks), job["cycles"])
                self.assertEqual(sum(b.nbytes for b in blocks), 6 * job["H"] * job["F"])
                total_bytes += sum(b.nbytes for b in blocks)
                loads[owner] += sum(b.cycles for b in blocks)
            self.assertEqual(total_bytes, plan["unique_weight_bytes"])
            self.assertEqual(loads, plan["core_cycles"])

    def test_zero_supply_delay_exactly_preserves_inherited_compute(self):
        for cores in FROZEN_CORES.values():
            for policy in ("workload_aware", "greedy_eft", "hard_threshold"):
                for prefetch in (False, True):
                    r = run_window(cores, window(), policy=policy, prefetch=prefetch,
                                   bandwidth_gbs=inf, request_latency_ns=0)
                    self.assertEqual(r["total_cycles"], r["ideal_compute_cycles"])
                    self.assertEqual(r["stall_cycles"], 0)

    def test_single_core_can_overlap_dma_and_compute(self):
        w = window(128, (128, 127), h=2048, f=1408)
        a = run_window(FROZEN_CORES["single"], w, prefetch=False, debug=True)
        b = run_window(FROZEN_CORES["single"], w, prefetch=True, debug=True)
        self.assertAlmostEqual(a["total_cycles"], a["hbm_busy_cycles"] + a["compute_cycles"])
        self.assertLess(b["total_cycles"], a["total_cycles"])
        dma = [e for e in b["events"] if e["event"] == "dma_start"]
        comp = [e for e in b["events"] if e["event"] == "compute_start"]
        self.assertTrue(any(max(d["time_ns"], c["time_ns"]) < min(d["end_ns"], c["end_ns"])
                            for d in dma for c in comp))

    def test_pin_cold_first_touch_is_charged(self):
        w = window(h=2048, f=1408)
        for cores in FROZEN_CORES.values():
            cold = run_window(cores, w, cache_policy="pin_shared")
            warm = run_window(cores, w, cache_policy="pin_shared", initial="shared_resident")
            self.assertEqual(cold["hbm_bytes"], cold["unique_weight_bytes"])
            self.assertEqual(cold["shared_hit_fraction"], 0)
            self.assertEqual(cold["pinned_reservation_bytes"], 33 * MIB)
            self.assertEqual(cold["weight_capacity_bytes"], 56 * MIB)
            self.assertEqual(warm["shared_hit_fraction"], 1)
            self.assertEqual(warm["explicitly_seeded_bytes"], 33 * MIB)
            self.assertEqual(cold["hbm_bytes"] - warm["hbm_bytes"], 33 * MIB)
            self.assertEqual(cold["refetch_bytes"], 0)
            self.assertLessEqual(cold["cache_peak_bytes"], 56 * MIB)

    def test_all_resident_repeat_is_exact_compute_and_has_no_fetch(self):
        w = window()
        for cores in FROZEN_CORES.values():
            c = WeightCache()
            cold = run_window(cores, w, cache=c)
            repeat = run_window(cores, w, cache=c)
            self.assertGreater(cold["hbm_bytes"], 0)
            self.assertEqual(repeat["hbm_bytes"], 0)
            self.assertEqual(repeat["shared_hit_fraction"], 1)
            self.assertEqual(repeat["total_cycles"], repeat["compute_cycles"])

    def test_shared_server_and_phase_precedence(self):
        r = run_window(FROZEN_CORES["heterogeneous"], window(8, (1, 2, 8)), debug=True)
        starts = [e for e in r["events"] if e["event"] == "dma_start"]
        for a, b in zip(starts, starts[1:]):
            self.assertLessEqual(a["end_ns"], b["time_ns"])
        for owner in (0, 1):
            comps = [e for e in r["events"] if e["event"] == "compute_start" and e["core"] == owner]
            for a, b in zip(comps, comps[1:]):
                self.assertLessEqual(a["end_ns"], b["time_ns"])
            for eid in r["core_expert_ids"][owner]:
                jobs = [e for e in comps if e["key"][1] == eid]
                phase_order = {"gate": 0, "up": 1, "down": 2}
                values = [phase_order[e["key"][2]] for e in jobs]
                self.assertEqual(values, sorted(values))
        self.assertLessEqual(r["hbm_busy_cycles"], r["total_cycles"])
        self.assertEqual(sum(r["core_hbm_bytes"]), r["hbm_bytes"])

    def test_capacity_contention_with_tiny_cache_completes(self):
        w = window()
        r = run_window(FROZEN_CORES["homogeneous"], w,
                       total_sram_bytes=8 * MIB + 256 * 1024,
                       reserved_sram_bytes=8 * MIB)
        self.assertGreater(r["cache_evictions"], 0)
        self.assertLessEqual(r["cache_peak_bytes"], 256 * 1024)
        self.assertEqual(r["refetch_bytes"], 0)
        self.assertEqual(r["hbm_bytes"], r["unique_weight_bytes"])

    def test_exact_repeat_determinism_and_latency_knob(self):
        kwargs = {"cores": FROZEN_CORES["heterogeneous"], "window": window(), "debug": True}
        self.assertEqual(run_window(**kwargs), run_window(**kwargs))
        normal = run_window(**kwargs)
        ideal_req = run_window(**kwargs, request_latency_ns=0)
        self.assertLessEqual(ideal_req["total_cycles"], normal["total_cycles"])
        self.assertEqual(ideal_req["hbm_bytes"], normal["hbm_bytes"])

    def test_invalid_arguments_rejected(self):
        for bad in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                run_window(FROZEN_CORES["single"], window(), bandwidth_gbs=bad)
        with self.assertRaises(ValueError):
            run_window(FROZEN_CORES["single"], window(), request_latency_ns=-1)
        with self.assertRaises(ValueError):
            run_window(FROZEN_CORES["single"], window(), initial="magic_warm")


if __name__ == "__main__":
    unittest.main()
