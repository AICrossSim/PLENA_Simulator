"""Integration checks for the analytical timing binary, not numerical execution.

Fixtures are explicitly synthetic boundary tests. Every run receives the real
compiler address-layout format; no test relies on the engine's dummy fallback.
The SRAM payload/numerical suite is independent in test_numerical.py.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from frontend import compiler, default_binary

align, engine_layout = compiler.align, compiler.engine_layout


BINARY = default_binary()


def fixture(m=7, h=512, f=32):
    """One expert, one route per token: exactly controlled causal timing input."""
    weights = {}
    for i, (phase, n, k) in enumerate((("gate", f, h), ("up", f, h), ("down", h, f))):
        weights[phase] = {"shape_nk": [n, k], "hbm_base": i * 16 * 1024**2,
                          "row_stride_bytes": align(k * 2), "physical_bytes": n * align(k * 2)}
    return {"id": "synthetic_engine_boundary_m%d_h%d_f%d" % (m, h, f),
            "batch": m, "hidden": h, "top_k": 1,
            "tokens": [{"token_index": t, "sample_id": "synthetic_%d" % t,
                        "routes": [{"expert_id": 0, "slot": 0, "score": 1.0}]} for t in range(m)],
            "experts": [{"id": 0, "is_shared": False, "Me": m, "H": h, "F": f,
                         "token_indices": list(range(m)), "route_slots": [0] * m,
                         "route_scores": [1.0] * m, "weights": weights}]}


class AnalyticalEngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError("Build the release analytical binary first: %s" % BINARY)

    def run_engine(self, workload, lanes, **changes):
        cfg = {"lanes": lanes, "group": 4, "dispatch": "fixed", "split": "none",
               "hbm_bytes_per_ns": 256, "hbm_latency_ns": 64, "credits": 256,
               "arbiter": "rr", "ideal_hbm": False, "ideal_onchip": False,
               "control_cost": True, "window": 4, "onchip_bytes_per_ns": 384,
               "vector_elements_per_ns": 32, "dot_tail_ns": 20, "record_trace": True}
        cfg.update(changes)
        if cfg["split"] != "none":
            cfg["runtime_fsm"] = False  # Frozen split diagnostic, outside Current/Next.
        workload = {**workload, "engine_layout": engine_layout(workload, lanes, cfg["group"])}
        if cfg.get("surplus_rules",0):
            workload["engine_layout"]["surplus_lut"] = compiler.surplus_policy_lut(lanes,cfg)
        with tempfile.TemporaryDirectory(prefix="plena-dispatch-engine-test-") as tmp:
            tmp = Path(tmp)
            wp, cp, out = tmp / "workload.json", tmp / "config.json", tmp / "report.json"
            wp.write_text(json.dumps(workload))
            cp.write_text(json.dumps(cfg))
            completed = subprocess.run([str(BINARY), str(wp), str(cp), str(out)],
                                       capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr[-6000:])
            report = json.loads(out.read_text())
        self.assertTrue(report["drained"])
        self.assertTrue(report["ownership_k_order_capacity_checks"])
        self.assertIn("does not execute numerical tensors", report["numerical_validation"])
        self.assertGreater(report["cycles"], 0)
        self.assertLessEqual(report["credit_peak"], cfg["credits"])
        return report, workload["engine_layout"]

    def test_forced_split_exchanges_exact_z_and_reads_weights_once(self):
        w = fixture(m=7, h=512, f=32)
        whole, _ = self.run_engine(w, [4, 2])
        split, layout = self.run_engine(w, [4, 2], split="forced")
        expected_weights = 2 * 32 * align(512 * 2) + 512 * align(32 * 2)
        expected_macs = 3 * 7 * 512 * 32
        for report in (whole, split):
            self.assertEqual(report["weight_bytes"], expected_weights)
            self.assertEqual(report["useful_macs"], expected_macs)
        self.assertEqual(sum(c["stats"]["z_exchange_bytes"] for c in whole["cores"]), 0)
        self.assertEqual(sum(c["stats"]["z_exchange_bytes"] for c in split["cores"]), 2 * 7 * 32)
        owners = [t for t in split["trace"] if t["event"] == "commit_owner"]
        self.assertEqual({t["core"] for t in owners}, {0, 1})
        self.assertTrue(all(t["split"] for t in owners))
        self.assertEqual(len([t for t in split["trace"] if t["event"] == "expert_drained"]), 2)
        for c, plan in zip(split["cores"], layout["cores"]):
            self.assertEqual(c["reserved_input_result_control"], plan["reserved"])

    def test_m_n_k_tails_preserve_finite_slots_contexts_and_private_addresses(self):
        w = fixture(m=7, h=513, f=19)
        expected_weights = 2 * 19 * align(513 * 2) + 513 * align(19 * 2)
        for lanes, mode in (([6], "none"), ([3, 3], "forced"), ([4, 2], "forced")):
            with self.subTest(lanes=lanes):
                report, layout = self.run_engine(w, lanes, split=mode)
                self.assertEqual(report["weight_bytes"], expected_weights)
                self.assertEqual(report["useful_macs"], 3 * 7 * 513 * 19)
                self.assertGreater(report["issued_macs"], report["useful_macs"])
                plans = layout["whole" if mode == "none" else "paired_n"][0]
                for index, (c, plan) in enumerate(zip(report["cores"], plans)):
                    stats = c["stats"]
                    self.assertLessEqual(stats["weight_peak_bytes"], (10 if len(lanes) == 1 else 5) * 4096)
                    self.assertLessEqual(stats["x_peak_bytes"], 2 * lanes[index] * 512 * 2)
                    self.assertGreater(stats["contexts_peak"], 0)
                    self.assertLessEqual(stats["contexts_peak"], 8)
                    self.assertLessEqual(stats["workspace_peak_bytes"], c["capacity"])
                    self.assertEqual(c["capacity"], layout["cores"][index]["capacity"])
                    self.assertLessEqual(plan["peak"], c["capacity"])
                    for name, a in plan["allocations"].items():
                        if "aliases" not in a:
                            self.assertLessEqual(a["base"] + a["bytes"], plan["peak"], name)
                        else:
                            element_bytes = 4 if name == "y" else 2
                            end = a["base"] + (a["shape"][0] - 1) * a["stride"] + a["shape"][1] * element_bytes
                            self.assertLessEqual(end, plan["peak"], name)
                    self.assertNotEqual(plan["allocations"]["x"]["base"], plan["allocations"]["y"]["base"])

    def test_single_fixed_hbm_relaxation_preserves_bytes_and_does_not_slow(self):
        # One expert and one core eliminate owner/dispatch changes as a cause.
        w = fixture(m=4, h=1024, f=64)
        constrained, _ = self.run_engine(w, [6], hbm_bytes_per_ns=32, credits=8)
        wider, _ = self.run_engine(w, [6], hbm_bytes_per_ns=256, credits=8)
        more_credits, _ = self.run_engine(w, [6], hbm_bytes_per_ns=256, credits=256)
        self.assertLessEqual(wider["cycles"], constrained["cycles"])
        self.assertLessEqual(more_credits["cycles"], wider["cycles"])
        self.assertLess(more_credits["cycles"], constrained["cycles"])
        for field in ("weight_bytes", "useful_macs", "issued_macs"):
            self.assertEqual(constrained[field], wider[field])
            self.assertEqual(wider[field], more_credits[field])
        for report in (constrained, wider, more_credits):
            owners = [t for t in report["trace"] if t["event"] == "commit_owner"]
            self.assertEqual([(t["expert"], t["core"], t["split"]) for t in owners], [(0, 0, False)])


if __name__ == "__main__":
    unittest.main()
