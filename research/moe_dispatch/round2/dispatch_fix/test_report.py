"""Check pairing, byte-baseline semantics, exact regression and saved reports."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from . import report


def fixture():
    rows = []
    for mode in report.MODES:
        for design in report.DESIGNS:
            for dispatch in (*report.DISPATCHES, report.CONTROL):
                for batch in report.BATCHES:
                    window = report.CASE_ID if batch == 128 else f"heldout_t{batch}"
                    single = design in ("B0", "B1")
                    latency = float(batch) * (1 if single else 0.9)
                    if not single and dispatch == "eft_old":
                        latency *= 1.2
                    elif not single and dispatch == report.CONTROL:
                        latency *= 1.1
                    elif not single and dispatch == "fixed":
                        latency *= 0.99
                    digest = f"{design}-{mode}-{window}" + ("" if single else dispatch)
                    hbm = 100 * 2**20
                    details = []
                    if not single and dispatch in ("fixed", "eft_old", "milp") and batch == 128:
                        hbm = (148 if dispatch != "milp" else 100) * 2**20
                        details = [{"expert_index": 0, "expert_id": "Shared", "Me": 128,
                                    "is_shared": True, "core": 0, "refetch_factor": 26 if dispatch != "milp" else 2,
                                    "best_refetch_factor": 2, "best_refetch_core": 1,
                                    "chosen_hbm_bytes": (52 if dispatch != "milp" else 4) * 2**20,
                                    "minimum_hbm_bytes": 4 * 2**20,
                                    "excess_B": (48 if dispatch != "milp" else 0) * 2**20, "z_chunks": 26,
                                    "decision_kind": "all_cores_refetch", "candidate_comparisons": []}]
                    rows.append({"design": design, "onchip_mode": mode, "dispatch": dispatch,
                                 "window_id": window, "batch": batch, "cycles": latency * 1e6,
                                 "latency_ms": latency, "hbm_bytes": hbm,
                                 "native_unique_bytes": 90 * 2**20,
                                 "refetch_tasks": len(details), "refetch_details": details,
                                 "shared_tasks": [{"expert_id": "Shared", "core": 0,
                                                   "start_cycles": 1000, "start_ms": 0.001,
                                                   "refetch_factor": 2, "z_chunks": 2}],
                                 "result_digest": digest, "repeat_digest": digest,
                                 "solver_status": "OPTIMAL" if dispatch == "milp" else None})
    frozen = {"modes": {m: {d: {"cores": [{"pm": 3, "pn": 16, "pk": 128}] *
                                          (1 if d in ("B0", "B1") else 2)}
                            for d in report.DESIGNS} for m in report.MODES}}
    selection = {"chosen": {"t_big": 2, "large_first": False}, "bootstrap_draws": 200,
                 "bootstrap_selection_counts": [{"t_big": 2, "large_first": False,
                                                  "count": 200, "fraction": 1}]}
    return rows, selection, frozen


class ReportTests(unittest.TestCase):
    def test_hbm_exception_uses_milp_and_keeps_all_core_ownership_evidence(self):
        rows, _, _ = fixture()
        traffic = report.hbm_rows(rows)
        single = next(r for r in traffic if r["design"] == "B1")
        self.assertEqual(single["extra_pct"], 0)  # Native unique is smaller than offline wire traffic.
        exceptions = report.excess_rows(rows, traffic)
        self.assertEqual(len(exceptions), 12)
        self.assertTrue(all(abs(r["extra_pct"] - 48) < 1e-10 for r in exceptions))
        self.assertTrue(all(r["all_cores_refetch_but_lower_traffic_alternative_tasks"] == 1
                            for r in exceptions))
        self.assertTrue(all(r["chosen_above_minimum_MiB"] == 48 for r in exceptions))
        deltas = report.task_hbm_deltas(rows)
        self.assertEqual(len(deltas), 6)
        self.assertTrue(all(r["delta_MiB"] == 48 for r in deltas))

    def test_unpaired_windows_are_rejected(self):
        rows, _, _ = fixture()
        rows = [r for r in rows if not (r["design"] == "B2" and r["dispatch"] == "milp" and
                                       r["onchip_mode"] == "pipelined" and r["batch"] == 2)]
        with self.assertRaisesRegex(ValueError, "unpaired windows"):
            report.comparison_rows(rows)

    def test_matching_cycles_without_matching_full_result_is_not_bit_exact(self):
        rows, _, _ = fixture()
        changed = next(r for r in rows if r["design"] == "B1" and r["dispatch"] == "fixed")
        changed["result_digest"] = "different predicted task metadata"
        regressions = report.regression_rows(rows)
        self.assertEqual(len(regressions), 28)
        self.assertEqual(sum(r["bit_exact"] for r in regressions), 27)

    def test_task_deltas_reconstruct_clean_side_and_preserve_reductions(self):
        rows, _, _ = fixture()
        for batch, dispatch in ((2, "fixed"), (4, "milp")):
            changed = next(r for r in rows if r["design"] == "B2" and r["onchip_mode"] == "pipelined"
                           and r["batch"] == batch and r["dispatch"] == dispatch)
            changed["hbm_bytes"] += 2 * 2**20
            changed["refetch_details"] = [{"expert_index": 7, "expert_id": "routed", "Me": 2,
                                           "is_shared": False, "core": 0, "refetch_factor": 2,
                                           "best_refetch_factor": 1, "best_refetch_core": 1,
                                           "chosen_hbm_bytes": 4 * 2**20, "minimum_hbm_bytes": 2 * 2**20,
                                           "excess_B": 2 * 2**20,
                                           "decision_kind": "immediate_refetch_faster_than_wait",
                                           "candidate_comparisons": [{"core": 0, "predicted_finish": 10},
                                                                     {"core": 1, "predicted_finish": 11}]}]
        deltas = report.task_hbm_deltas(rows)
        reconstructed = [r for r in deltas if r["expert_index"] == 7]
        self.assertEqual([r["delta_MiB"] for r in reconstructed], [2, -2])
        self.assertTrue(all(r["unique_hbm_bytes"] == 2 * 2**20 for r in reconstructed))
        self.assertIn('"predicted_finish":11', reconstructed[0]["candidate_comparisons"])

    def test_task_layer_byte_mismatch_blocks_report(self):
        rows, _, _ = fixture()
        changed = next(r for r in rows if r["design"] == "B2" and r["dispatch"] == "fixed")
        changed["hbm_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "task/layer HBM delta mismatch"):
            report.task_hbm_deltas(rows)

    def test_case_two_percent_gate_is_distinct_from_aggregate_one_percent_gate(self):
        rows, _, _ = fixture()
        by_key = {report._key(r): r for r in rows}
        for fixed in rows:
            if fixed["onchip_mode"] != "pipelined" or fixed["dispatch"] != "fixed":
                continue
            if fixed["design"] == "B2" or (fixed["design"] == "best_hetero" and fixed["batch"] == 128):
                milp = by_key[fixed["design"], fixed["onchip_mode"], "milp", fixed["window_id"]]
                fixed["latency_ms"] = milp["latency_ms"] * 1.015
                fixed["cycles"] = fixed["latency_ms"] * 1e6
        deltas = report.task_hbm_deltas(rows)
        acceptance = report.acceptance_details(rows, report.comparison_rows(rows), report.hbm_rows(rows),
                                               report.regression_rows(rows), deltas)
        aggregate = next(r for r in acceptance["B2_geomean_latency_gates"] if r["onchip_mode"] == "pipelined")
        case = next(r for r in acceptance["gpqa_t128_cases"] if r["onchip_mode"] == "pipelined" and
                    r["design"] == "best_hetero")
        self.assertAlmostEqual(aggregate["fixed_vs_milp"], 1.015)
        self.assertFalse(aggregate["latency_within_1pct"])
        self.assertAlmostEqual(case["fixed_vs_milp"], 1.015)
        self.assertTrue(case["latency_within_2pct"])
        self.assertNotIn("latency_within_1pct", case)
        self.assertIn("2% 延迟门槛（≤1.02）通过", report.case_markdown(rows, deltas))

    def test_all_outputs_and_compressed_fresh_checkout(self):
        rows, selection, frozen = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = json.dumps(rows).encode()
            with gzip.open(directory / "per_window.json.gz", "wb") as stream:
                stream.write(raw)
            for name, value in (("selection.json", selection), ("frozen_designs.json", frozen)):
                (directory / name).write_text(json.dumps(value))
            evidence = report.generate(directory)
            self.assertEqual(evidence["row_counts"]["compare.csv"], 30)
            self.assertEqual(evidence["row_counts"]["compare_predictor_control.csv"], 10)
            self.assertEqual(evidence["row_counts"]["hbm_bytes.csv"], 210)
            self.assertEqual(evidence["row_counts"]["hbm_predictor_control.csv"], 70)
            with (directory / "compare.csv").open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), report.COMPARE_FIELDS)
                comparison = list(reader)
            b1 = next(r for r in comparison if r["design"] == "B1" and r["dispatch"] == "fixed")
            self.assertEqual(float(b1["ratio_vs_B1_fixed"]), 1)
            with (directory / "paired_improvement.csv").open() as stream:
                cis = list(csv.DictReader(stream))
            b2 = next(r for r in cis if r["design"] == "B2" and r["baseline"] == "B1_fixed")
            self.assertGreater(float(b2["ci95_low_improvement_pct"]), 0)
            summary = (directory / "SUMMARY.md").read_text()
            self.assertIn("不能解释为阈值稳健性", summary)
            self.assertIn("3x16x128+3x16x128", summary)
            self.assertIn("回归28/28项", summary)
            self.assertIn("额外24次仍取决于归属", summary)
            case = (directory / "gpqa_t128_case.md").read_text()
            self.assertIn("2% HBM 门槛未通过", case)
            self.assertIn("已安装核最小流量 4.000 MiB", case)


if __name__ == "__main__":
    unittest.main()
