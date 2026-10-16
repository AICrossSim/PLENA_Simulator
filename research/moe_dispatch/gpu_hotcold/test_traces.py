"""Stdlib trace-contract tests. Synthetic fixtures are unit tests, not workloads."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import traces


def fixture() -> dict:
    tokens = [{"token_index": i, "routes": [
        {"expert_id": j + i * 3, "slot": j, "score": (j + 1) / 21}
        for j in range(6)]} for i in range(2)]
    counts = [sum(r["expert_id"] == e for t in tokens for r in t["routes"]) for e in range(64)]
    return {"id": "unit_test_fixture_not_a_capture", "batch": 2, "split": "heldout",
            "tokens": tokens, "route_ids": [[r["expert_id"] for r in t["routes"]] for t in tokens],
            "route_weights": [[r["score"] for r in t["routes"]] for t in tokens],
            "counts_64": counts, "token_routes_sha256": traces.canonical_sha256(tokens),
            "provenance": "captured_decode", "source": {"sha256": "a" * 64}}


class TraceContractTests(unittest.TestCase):
    def setUp(self):
        self.window = fixture()

    def test_valid_fixture_and_preserved_unordered_tokens(self):
        traces.validate_window(self.window)
        self.window["tokens"].reverse()
        self.window["token_routes_sha256"] = traces.canonical_sha256(self.window["tokens"])
        traces.validate_window(self.window)

    def test_route_slots_define_order(self):
        self.window["tokens"][0]["routes"].reverse()
        self.window["token_routes_sha256"] = traces.canonical_sha256(self.window["tokens"])
        traces.validate_window(self.window)

    def test_histogram_mismatch_rejected(self):
        self.window["counts_64"][0] += 1
        with self.assertRaisesRegex(ValueError, "conservation"):
            traces.validate_window(self.window)

    def test_duplicate_topk_rejected(self):
        self.window["tokens"][0]["routes"][0]["expert_id"] = 1
        with self.assertRaisesRegex(ValueError, "distinct"):
            traces.validate_window(self.window)

    def test_duplicate_slot_rejected(self):
        self.window["tokens"][0]["routes"][0]["slot"] = 1
        with self.assertRaisesRegex(ValueError, "slots"):
            traces.validate_window(self.window)

    def test_duplicate_token_rejected(self):
        self.window["tokens"][1]["token_index"] = 0
        with self.assertRaisesRegex(ValueError, "index conservation"):
            traces.validate_window(self.window)

    def test_score_change_rejected(self):
        self.window["route_weights"][0][0] += 0.125
        with self.assertRaisesRegex(ValueError, "arrays disagree"):
            traces.validate_window(self.window)

    def test_nonfinite_or_negative_score_rejected(self):
        for invalid in (math.nan, math.inf, -0.1):
            with self.subTest(invalid=invalid):
                window = fixture()
                window["tokens"][0]["routes"][0]["score"] = invalid
                with self.assertRaisesRegex(ValueError, "score"):
                    traces.validate_window(window)

    def test_no_score_renormalization(self):
        for token in self.window["tokens"]:
            for route in token["routes"]:
                route["score"] *= 2.5
        self.window["route_weights"] = [[r["score"] for r in t["routes"]] for t in self.window["tokens"]]
        self.window["token_routes_sha256"] = traces.canonical_sha256(self.window["tokens"])
        traces.validate_window(self.window)
        self.assertAlmostEqual(sum(self.window["route_weights"][0]), 2.5)

    def test_hash_change_rejected(self):
        self.window["token_routes_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "payload hash"):
            traces.validate_window(self.window)

    def test_generated_provenance_rejected(self):
        self.window["provenance"] = "synthetic_zipf"
        with self.assertRaisesRegex(ValueError, "provenance"):
            traces.validate_window(self.window)

    def test_float_expert_id_rejected(self):
        self.window["tokens"][0]["routes"][0]["expert_id"] = 0.0
        with self.assertRaisesRegex(ValueError, "integer"):
            traces.validate_window(self.window)

    def test_reference_flops_not_timing(self):
        metrics = traces.population_metrics(self.window)
        self.assertEqual(metrics["shared_gemm_flops"] * 4, metrics["total_gemm_flops"])
        self.assertEqual(metrics["cold_token_assignments"] + metrics["hot_token_assignments"], 12)
        self.assertEqual(metrics["hot_experts"], 0)
        self.assertEqual(metrics["cold_experts"], 9)
        self.assertIn("not a time", metrics["arithmetic_scope"])

    def test_zero_experts_not_classified_cold(self):
        metrics = traces.population_metrics(self.window)
        self.assertEqual(metrics["inactive_experts"], 55)
        self.assertEqual(metrics["inactive_experts"] + metrics["hot_experts"] + metrics["cold_experts"], 64)

    def test_actual_export_if_available(self):
        path = traces.DEFAULT_OUTPUT
        if not path.exists():
            self.skipTest("Full captured export not present in this environment")
        data = traces.load_routes(path)
        self.assertEqual(len(data["windows"]), 135)
        for window in data["windows"]:
            metrics = traces.population_metrics(window)
            self.assertEqual(metrics["shared_gemm_flops"] * 4, metrics["total_gemm_flops"])
            self.assertEqual(sum(window["counts_64"]), 6 * window["batch"])


if __name__ == "__main__":
    unittest.main()
