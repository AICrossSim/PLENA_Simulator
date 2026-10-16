import math
import unittest

from research.moe_dispatch.mdpu_compute.report import BATCHES, balanced_gm, bootstrap_speedup, paired_speedup, summaries, presentation_boundary


class ReportTests(unittest.TestCase):
    def rows(self):
        # Deliberately imbalanced counts: averaging all windows is incorrect.
        return [{"window_id": f"b{batch}_{index}", "batch": batch,
                 "cycles": 16 if batch == 2 else 1}
                for batch in BATCHES for index in range(27 if batch == 2 else 9)]

    def test_equal_batch_weights(self):
        rows = self.rows()
        self.assertAlmostEqual(balanced_gm(rows), 16 ** (1 / 7))
        self.assertNotAlmostEqual(balanced_gm(rows), math.exp(sum(math.log(r["cycles"]) for r in rows) / len(rows)))

    def test_missing_batch_rejected(self):
        with self.assertRaises(ValueError):
            balanced_gm(self.rows()[:-9])

    def test_paired_ratios_ignore_input_order(self):
        ref = self.rows()
        cand = [dict(r, cycles=r["cycles"] / 2) for r in reversed(ref)]
        self.assertTrue(all(r["ratio"] == 2 for r in paired_speedup(ref, cand)))
        ci = bootstrap_speedup(ref, cand)
        self.assertAlmostEqual(ci["speedup"], 2)
        self.assertAlmostEqual(ci["ci95_low"], 2)
        self.assertAlmostEqual(ci["ci95_high"], 2)
        self.assertEqual(ci, bootstrap_speedup(ref, cand))

    def test_mismatched_or_duplicate_windows_rejected(self):
        ref = self.rows()
        with self.assertRaises(ValueError):
            paired_speedup(ref, ref[:-1])
        with self.assertRaises(ValueError):
            paired_speedup(ref + ref[:1], ref + ref[:1])

    def test_zero_load_is_not_silently_a_finite_imbalance(self):
        base = {"threshold": 1, "cycles": 100, "compute_ms": .0001,
                "threshold_realizable": 0, "zero_load_core_window": 1,
                "imbalance_ratio": None, "core0_useful_FLOPs_pct": 0,
                "core1_useful_FLOPs_pct": 100, "core0_busy_ms": 0,
                "core1_busy_ms": .0001, "core0_idle_pct": 100,
                "core1_idle_pct": 0, "padding_of_issued_pct": 0,
                "affinity_deviation_count": 1, "steal_count": 0,
                "shared_on_core0": 0}
        rows = [dict(base, batch=b, window_id=str(b)) for b in BATCHES]
        aggregate = summaries(rows)
        self.assertEqual(aggregate[-1]["zero_load_core_windows"], 7)
        self.assertTrue(presentation_boundary(aggregate)[-1]["失配比 max/min"].startswith("∞"))


if __name__ == "__main__":
    unittest.main()
