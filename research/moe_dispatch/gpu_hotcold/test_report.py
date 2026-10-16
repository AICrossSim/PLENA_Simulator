"""Tests prevent misleading additive timing and incomplete-campaign claims."""
from __future__ import annotations

import math
import unittest

from .report import additive_summary, component_accounting, geometric_mean, sample_statistics


class AccountingTests(unittest.TestCase):
    def test_event_intervals_close_and_preserve_other(self):
        series = {"shared": [1., 2.], "dispatch": [.1, .2], "hot": [0., 0.],
                  "cold": [2., 3.], "combine": [.2, .1]}
        closed = component_accounting([4., 6.], series)
        self.assertAlmostEqual(closed["other"][0], .7)
        self.assertAlmostEqual(closed["other"][1], .7)
        self.assertEqual(closed["hot"], [0., 0.])
        rows = [{f"{k}_ms": sample_statistics(v)["mean_ms"] for k, v in closed.items()}]
        summary = additive_summary(rows)
        self.assertAlmostEqual(sum(summary[f"{k}_pct"] for k in (*series, "other")), 100.)
        self.assertEqual(summary["hot_pct"], 0.)
        self.assertAlmostEqual(summary["closure_error_ms"], 0.)

    def test_components_cannot_exceed_total(self):
        series = {k: [1.] for k in ("shared", "dispatch", "hot", "cold", "combine")}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            component_accounting([4.], series)

    def test_component_samples_must_be_paired(self):
        series = {k: [0.] for k in ("shared", "dispatch", "hot", "cold", "combine")}
        with self.assertRaisesRegex(ValueError, "align"):
            component_accounting([1., 2.], series)

    def test_event_rounding_tolerance_is_audited_not_negative_other(self):
        series = {"shared": [1.00001], "dispatch": [0.], "hot": [0.], "cold": [0.], "combine": [0.]}
        closed = component_accounting([1.], series)
        self.assertEqual(closed["other"], [0.])
        self.assertAlmostEqual(closed["closure_roundoff"][0], -.00001)

    def test_mean_accounting_is_not_sum_of_medians(self):
        self.assertEqual(sample_statistics([0, 0, 9])["mean_ms"], 3.)
        self.assertEqual(sample_statistics([0, 0, 9])["median_ms"], 0.)

    def test_gm_rejects_zero_hot_category_and_nonfinite(self):
        for values in ([0, 1], [-1, 1], [math.nan, 1], [math.inf, 1], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                geometric_mean(values)
        self.assertAlmostEqual(geometric_mean([2., 8.]), 4.)

    def test_empty_category_has_exact_zero_statistics(self):
        row = sample_statistics([0.] * 50)
        self.assertEqual(row["mean_ms"], 0.)
        self.assertEqual(row["median_ms"], 0.)
        self.assertEqual(row["stdev_ms"], 0.)

    def test_additive_ratios_use_ratio_of_means(self):
        rows = []
        for shared, total in [(1., 2.), (1., 10.)]:
            row = {f"{k}_ms": 0. for k in ("shared", "dispatch", "hot", "cold", "combine", "other", "closure_roundoff", "total")}
            row.update(shared_ms=shared, cold_ms=total - shared, total_ms=total)
            rows.append(row)
        out = additive_summary(rows)
        self.assertAlmostEqual(out["shared_pct"], 100/6)
        self.assertNotAlmostEqual(out["shared_pct"], 30.)


class CampaignAggregationTests(unittest.TestCase):
    @staticmethod
    def row(batch=2, repeat=1, path="triton_split", scale=1.0):
        stages = {"shared": [scale] * 50, "combine": [0.2 * scale] * 50}
        if path.endswith("split"):
            stages.update(hot=[0.] * 50, cold=[2 * scale] * 50)
        else:
            stages["routed"] = [2 * scale] * 50
        return {"window_id": f"fixture_B{batch}", "batch": batch, "repeat": repeat,
                "mode": "graph", "path": path, "gpu": "unit test, no GPU measurement",
                "e2e_ms": [3.0 * scale] * 50, "instrumented_total_ms": [3.2 * scale] * 50,
                "stage_samples_ms": stages}

    def test_all_seven_batches_empty_hot_are_valid_without_gm_zero(self):
        from .report import aggregate, BATCHES
        rows = [self.row(batch=b, repeat=r) for b in BATCHES for r in (1, 2)]
        result = aggregate(rows)
        self.assertEqual(len(result["main_latency"]), 8)
        self.assertEqual(len(result["split_breakdown"]), 8)
        for row in result["split_breakdown"]:
            self.assertEqual(row["hot_ms"], 0.)
            self.assertEqual(row["hot_pct"], 0.)
            self.assertAlmostEqual(row["shared_pct"] + row["cold_pct"] + row["combine_pct"], 100.)
        gm = result["main_latency"][-1]
        self.assertEqual(gm["batch"], "7-Batch GM")
        self.assertAlmostEqual(gm["mean_ms"], 3.)
        self.assertIsNone(gm["pooled_median_ms"])

    def test_instrumented_time_never_used_for_split_slowdown(self):
        from .report import aggregate
        normal = self.row(path="triton_grouped")
        split = self.row(path="triton_split")
        split["e2e_ms"] = [6.] * 50
        result = aggregate([normal, split])
        ratio = result["controlled_ratios"][0]
        self.assertEqual(ratio["numerator"], "triton_split")
        self.assertAlmostEqual(ratio["paired_ratio_gm"], 2.)

    def test_missing_batch_does_not_generate_seven_batch_gm(self):
        from .report import aggregate
        result = aggregate([self.row()])
        self.assertEqual(len(result["main_latency"]), 1)
        self.assertEqual(len(result["split_breakdown"]), 1)

    def test_equal_batch_gm_is_not_pooled_window_gm(self):
        from .report import aggregate, BATCHES
        rows = [self.row(batch=b, scale=float(b)) for b in BATCHES]
        extra = self.row(batch=2, scale=2.)
        extra["window_id"] = "another_B2_fixture"
        rows.append(extra)
        result = aggregate(rows)
        gm = result["main_latency"][-1]["mean_ms"]
        self.assertAlmostEqual(gm, 3 * geometric_mean(BATCHES))
        self.assertNotAlmostEqual(gm, geometric_mean([3*b for b in BATCHES] + [6]))

    def test_paired_repeats_are_exposed_not_forced_bitwise_equal(self):
        from .report import aggregate
        result = aggregate([self.row(repeat=1), self.row(repeat=2, scale=1.1)])
        stability = result["repeat_stability"][0]
        self.assertAlmostEqual(stability["repeat2_over_repeat1_gm"], 1.1)
        self.assertAlmostEqual(stability["mean_absolute_repeat_delta_pct"], 10.)


class CoverageTests(unittest.TestCase):
    def make_campaigns(self, base):
        import json
        from pathlib import Path
        from .report import sha256
        from .test_traces import fixture
        trace = {"windows": [fixture()]}
        wid = trace["windows"][0]["id"]
        source = base / "trace.json"
        source.write_text(json.dumps(trace))
        paths = ("vllm_grouped", "triton_grouped", "triton_split", "vllm_split")
        folders = []
        for repeat in (1, 2):
            folder = base / f"repeat{repeat}"
            folder.mkdir()
            records = []
            for mode in ("eager", "graph"):
                for path in paths:
                    row = CampaignAggregationTests.row(path=path)
                    row.update(window_id=wid, mode=mode, warmup=10, samples=50,
                               hot_experts=0, cold_experts=9, hot_token_assignments=0,
                               cold_token_assignments=12, raw_output_correct=True, instrumented_output_correct=True)
                    records.append(row)
            (folder / "measurements.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            (folder / "metadata.json").write_text(json.dumps({"routes_sha256": sha256(source),
                "gpu": "unit fixture", "software": {"torch": "unit fixture"}}))
            (folder / "COMPLETE.json").write_text(json.dumps({"status": "PASS", "points": len(records), "windows": 1}))
            (folder / "correctness.json").write_text(json.dumps([
                {"window_id": wid, "path": p, "passed": True, "max_abs_error": 0., "relative_l2_error": 0.}
                for p in paths]))
            folders.append(folder)
        return folders, trace, source

    def test_complete_campaign_provenance_and_numerics(self):
        import tempfile
        from pathlib import Path
        from .report import load_campaigns
        with tempfile.TemporaryDirectory() as d:
            folders, trace, source = self.make_campaigns(Path(d))
            rows, audit = load_campaigns(folders, trace, source)
            self.assertEqual(len(rows), 16)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(len(audit["numeric_checks"]), 8)

    def test_missing_timing_point_rejected(self):
        import tempfile
        from pathlib import Path
        from .report import load_campaigns
        with tempfile.TemporaryDirectory() as d:
            folders, trace, source = self.make_campaigns(Path(d))
            p = folders[0] / "measurements.jsonl"
            p.write_text("\n".join(p.read_text().splitlines()[1:]))
            with self.assertRaisesRegex(ValueError, "Incomplete campaign"):
                load_campaigns(folders, trace, source)

    def test_missing_numerical_check_rejected(self):
        import tempfile
        from pathlib import Path
        from .report import load_campaigns
        with tempfile.TemporaryDirectory() as d:
            folders, trace, source = self.make_campaigns(Path(d))
            (folders[0] / "correctness.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Numerical checks"):
                load_campaigns(folders, trace, source)

    def test_source_digest_change_rejected(self):
        import tempfile
        from pathlib import Path
        from .report import load_campaigns
        with tempfile.TemporaryDirectory() as d:
            folders, trace, source = self.make_campaigns(Path(d))
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_campaigns(folders, trace, source)

    def test_nonzero_empty_hot_rejected(self):
        import tempfile
        import json
        from pathlib import Path
        from .report import load_campaigns
        with tempfile.TemporaryDirectory() as d:
            folders, trace, source = self.make_campaigns(Path(d))
            p = folders[0] / "measurements.jsonl"
            records = [json.loads(line) for line in p.read_text().splitlines()]
            next(r for r in records if r["path"] == "vllm_split")["stage_samples_ms"]["hot"][0] = .001
            p.write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "Empty category"):
                load_campaigns(folders, trace, source)


class CounterWorkbookTests(unittest.TestCase):
    def test_existing_counter_csv_loaded_without_modification(self):
        import tempfile
        from pathlib import Path
        from .report import load_counter_tables, sha256
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            p = folder / "counter_summary.csv"
            p.write_text("batch,path,dram_util_pct,scope\n2,vllm_grouped,23.75,replay\n")
            original = sha256(p)
            tables, evidence = load_counter_tables(folder)
            self.assertEqual(tables["counter_summary"][0]["batch"], 2)
            self.assertEqual(tables["counter_summary"][0]["dram_util_pct"], 23.75)
            self.assertEqual(evidence["counter_summary"]["sha256"], original)
            self.assertEqual(sha256(p), original)

    def test_optional_absent_counter_csv_is_not_fabricated(self):
        import tempfile
        from pathlib import Path
        from .report import load_counter_tables
        with tempfile.TemporaryDirectory() as d:
            tables, evidence = load_counter_tables(Path(d))
            self.assertEqual(tables, {})
            self.assertEqual(evidence, {})


if __name__ == "__main__":
    unittest.main()
