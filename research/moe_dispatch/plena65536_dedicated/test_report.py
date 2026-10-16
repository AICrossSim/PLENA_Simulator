"""Aggregation regressions that can change the reported dedicated winner."""
import math
import unittest

from .report import aggregate, paired_bootstrap


class ReportTests(unittest.TestCase):
    def test_aggregate_window_max_before_component_geometric_means(self):
        rows = [
            {"batch": 2, "window_count": 1, "window_id": "a", "row_type": "window",
             "shared_core_ms": 10.0, "routed_core_ms": 1.0, "compute_ms": 10.0},
            {"batch": 2, "window_count": 1, "window_id": "b", "row_type": "window",
             "shared_core_ms": 1.0, "routed_core_ms": 10.0, "compute_ms": 10.0},
        ]
        result = aggregate(rows, 2, "batch_GM")
        self.assertAlmostEqual(result["compute_ms"], 10.0)
        self.assertAlmostEqual(result["shared_core_ms"], math.sqrt(10))
        self.assertGreater(result["compute_ms"], max(result["shared_core_ms"], result["routed_core_ms"]))

    def test_bootstrap_preserves_paired_constant_speedup(self):
        rows = []
        for batch in (2, 4, 8, 16, 64, 96, 128):
            for index in range(3):
                for design, cycles in (("extended_single", 100), ("extended_homogeneous", 200),
                                       ("extended_dedicated_largeShared", 50)):
                    rows.append({
                        "design": design, "split": "heldout", "row_type": "window",
                        "window_id": f"B{batch}w{index}", "batch": batch,
                        "compute_ms": cycles / 1e6, "compute_cycles": cycles,
                    })
        got = paired_bootstrap(rows, repetitions=50)
        self.assertEqual(len(got), 2)
        for result, expected in zip(got, (2.0, 4.0)):
            self.assertAlmostEqual(result["batch_balanced_GM_speedup"], expected)
            self.assertAlmostEqual(result["bootstrap_95pct_lower"], expected)
            self.assertAlmostEqual(result["bootstrap_95pct_upper"], expected)
            self.assertEqual(result["candidate_win_windows"], 21)


if __name__ == "__main__":
    unittest.main()
