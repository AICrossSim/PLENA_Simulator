"""Real-route conservation and balanced evaluation regression checks."""
from __future__ import annotations

import copy
import unittest
from collections import Counter

from research.moe_dispatch.plena65536_moe.workloads import (
    BATCHES,
    batch_balanced_metrics,
    load_workloads,
    profile_window,
    validate_window,
)


class FrozenSharedMoeWorkloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = load_workloads()

    def test_actual_frozen_split_coverage_and_both_mac_populations(self):
        self.assertEqual(len(self.inputs["development"]), 18)
        self.assertEqual(len(self.inputs["heldout"]), 135)
        self.assertEqual(Counter(w["batch"] for w in self.inputs["heldout"]),
                         {2: 27, 4: 27, 8: 27, 16: 27, 64: 9, 96: 9, 128: 9})
        for split in ("development", "heldout"):
            self.assertEqual(set(w["batch"] for w in self.inputs[split]), set(BATCHES))
            for window in self.inputs[split]:
                profile = profile_window(window)
                self.assertEqual(profile["routed_token_assignments"], 6 * window["batch"])
                self.assertEqual(profile["shared_token_assignments"], window["batch"])
                self.assertEqual(profile["shared_useful_macs"], 3 * window["batch"] * 2048 * 2816)
                self.assertEqual(profile["routed_useful_macs"], 3 * profile["shared_useful_macs"])
                self.assertEqual(profile["total_useful_macs"], 4 * profile["shared_useful_macs"])

    def test_real_per_token_duplicate_is_rejected(self):
        window = copy.deepcopy(self.inputs["development"][0])
        window["tokens"][0]["routes"][1]["expert_id"] = window["tokens"][0]["routes"][0]["expert_id"]
        with self.assertRaisesRegex(ValueError, "Top-6 expert IDs"):
            validate_window(window)

    def test_token_descriptor_disagreement_is_rejected(self):
        window = copy.deepcopy(self.inputs["development"][0])
        routed = next(expert for expert in window["experts"] if not expert["is_shared"])
        routed["route_scores"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "token/descriptor route mismatch"):
            validate_window(window)

    def test_missing_shared_is_rejected(self):
        window = copy.deepcopy(self.inputs["development"][0])
        window["experts"] = [expert for expert in window["experts"] if not expert["is_shared"]]
        with self.assertRaisesRegex(ValueError, "Shared descriptor"):
            validate_window(window)

    def test_mixed_address_aliases_do_not_remove_distinct_expert_weights(self):
        window = next(w for w in self.inputs["heldout"] if w["batch"] == 128)
        profile = profile_window(window)
        self.assertEqual(profile["total_unique_weight_bytes"],
                         34603008 + profile["active_routed_experts"] * 17301504)
        addresses = [e["weights"]["gate"]["hbm_base"] for e in window["experts"] if not e["is_shared"]]
        self.assertLess(len(set(addresses)), len(addresses))

    def test_batch_balanced_metric_is_not_sample_weighted(self):
        windows = [{"id": "a", "batch": 2}, {"id": "b", "batch": 2},
                   {"id": "c", "batch": 128}]
        result = batch_balanced_metrics(windows, {"a": 1, "b": 1, "c": 4})
        self.assertAlmostEqual(result["batch_balanced_gm"], 2)
        self.assertAlmostEqual(result["sample_weighted_gm"], 4 ** (1 / 3))

    def test_development_request_manifest_does_not_read_heldout(self):
        result = load_workloads("development")
        self.assertEqual(set(result), {"development", "manifest"})
        self.assertEqual(set(result["manifest"]["loaded_input_sha256"]),
                         {"development.json", "mixed_development.json"})


if __name__ == "__main__":
    unittest.main()
