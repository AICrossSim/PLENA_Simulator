"""Independent arithmetic tests for reverse supply requirements."""
from __future__ import annotations

from dataclasses import dataclass
import json
import unittest

from .supply import _shape_traffic, latency_hiding_requirements, reverse_requirements


class TrafficTests(unittest.TestCase):
    def test_bandwidth_delay_product_is_a_lower_bound(self):
        got = latency_hiding_requirements(256, 64, 32)
        self.assertEqual(got["inflight_payload_bytes_lower_bound"], 16_384)
        self.assertEqual(got["payload_credit_count_lower_bound"], 512)
        self.assertTrue(got["not_a_stall_free_guarantee"])

    def test_tail_masked_traffic_matches_explicit_tile_iteration(self):
        m, n, k = 5, 7, 9
        pm, pn, pk = 3, 4, 4
        got = _shape_traffic(m, n, k, (pm, pn, pk))
        weight = activation = partial_reads = partial_writes = useful = issues = 0
        for c in range(0, k, pk):
            actual_k = min(pk, k - c)
            for j in range(0, n, pn):
                actual_n = min(pn, n - j)
                for i in range(0, m, pm):
                    actual_m = min(pm, m - i)
                    issues += 1
                    useful += actual_m * actual_n * actual_k
                    weight += actual_n * actual_k * 2
                    activation += actual_m * actual_k * 2
                    partial_writes += actual_m * actual_n * 4
                    if c:
                        partial_reads += actual_m * actual_n * 4
        array = got["array_operand_and_partial_traffic"]
        self.assertEqual(got["issues"], issues)
        self.assertEqual(got["useful_macs"], useful)
        self.assertEqual(array["weight_masked_bytes"], weight)
        self.assertEqual(array["x_masked_bytes"], activation)
        self.assertEqual(array["partial_read_masked_bytes"], partial_reads)
        self.assertEqual(array["partial_write_masked_bytes"], partial_writes)
        self.assertEqual(array["weight_full_slice_bytes"], issues * pn * pk * 2)
        self.assertEqual(array["x_full_slice_bytes"], issues * pm * pk * 2)
        self.assertGreater(array["weight_full_slice_bytes"], weight)

    def test_first_k_segment_does_not_read_old_partial(self):
        got = _shape_traffic(2, 4, 3, (2, 4, 4))
        self.assertEqual(got["array_operand_and_partial_traffic"]["partial_read_full_record_bytes"], 0)
        self.assertEqual(got["array_operand_and_partial_traffic"]["partial_write_full_record_bytes"], 32)

    def test_stationary_loops_have_distinct_refills_and_capacity(self):
        got = _shape_traffic(8, 12, 16, (2, 4, 4))
        os = got["os_kmajor_shared_pool_to_local"]
        alternatives = got["loop_alternatives"]
        self.assertEqual(os["weight_masked_bytes"], 12 * 16 * 2)
        self.assertEqual(os["x_masked_bytes"], 8 * 16 * 2)
        self.assertEqual(alternatives["IS"]["weight_masked_bytes"], 4 * 12 * 16 * 2)
        ws_one = alternatives["WS"][0]
        ws_all = alternatives["WS"][-1]
        self.assertEqual(ws_one["x_masked_bytes"], 3 * 8 * 16 * 2)
        self.assertEqual(ws_all["x_masked_bytes"], os["x_masked_bytes"])
        self.assertEqual(ws_all["partial_record_bytes"], got["capacity"]["all_partial_records_full_bytes"])
        self.assertEqual(got["capacity"]["x_k_panel_double_buffer_bytes"], 2 * 8 * 4 * 2)


@dataclass
class DummyCore:
    pm: int
    pn: int
    pk: int


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.window = {"id": "small", "batch": 4, "routed_expert_count": 2,
                       "experts": [
                           {"id": "cold", "Me": 1, "H": 8, "F": 6, "is_shared": False},
                           {"id": "inactive", "Me": 0, "H": 8, "F": 6, "is_shared": False},
                           {"id": "shared", "Me": 4, "H": 8, "F": 12, "is_shared": True},
                       ]}
        self.assignments = [
            {"expert_id": "cold", "owner": 1, "start_cycle": 0, "finish_cycle": 50},
            {"expert_id": "shared", "owner": 0, "start_cycle": 0, "finish_cycle": 100},
        ]

    def run_model(self, **kwargs):
        return reverse_requirements(self.window, [DummyCore(4, 4, 4), (1, 4, 4)],
                                    self.assignments, 100, [5, 5], **kwargs)

    def test_unique_weights_and_global_activations_no_double_counting(self):
        got = self.run_model()
        selected = 3 * 8 * (6 + 12) * 2
        all_weights = 3 * 8 * (2 * 6 + 12) * 2
        self.assertEqual(got["external"]["selected_unique_bf16_weight_bytes"], selected)
        self.assertEqual(got["capacity"]["all_routed_and_shared_weight_bytes"], all_weights)
        self.assertEqual(got["external"]["cold_weight_average_required_GB_s"], selected / 100)
        self.assertEqual(got["external"]["compulsory_global_input_bf16_bytes"], 4 * 8 * 2)
        self.assertEqual(got["onchip"]["logical_expert_input_gather_bytes"], (1 + 4) * 8 * 2)
        self.assertEqual(got["external"]["fully_resident_weight_HBM_bytes_per_replay"], 0)
        self.assertEqual(got["capacity"]["selected_layer_weights_bytes"], selected)

    def test_joint_gate_up_panel_is_loaded_once_but_array_receives_both(self):
        got = self.run_model()
        pool = got["onchip"]["os_kmajor_shared_pool_to_local"]
        self.assertEqual(pool["x_masked_bytes"], sum(e["Me"] * (e["H"] + e["F"]) * 2
                                                   for e in self.window["experts"]))
        for expert in got["per_expert"]:
            gate, up, down = expert["projections"]
            self.assertEqual(gate["os_kmajor_shared_pool_to_local"]["x_masked_bytes"], expert["Me"] * expert["H"] * 2)
            self.assertEqual(up["os_kmajor_shared_pool_to_local"]["x_masked_bytes"], 0)
            self.assertEqual(up["os_kmajor_shared_pool_to_local"]["x_full_slice_bytes"], 0)
            self.assertEqual(gate["array_operand_and_partial_traffic"]["x_full_slice_bytes"],
                             up["array_operand_and_partial_traffic"]["x_full_slice_bytes"])
            self.assertGreater(up["array_operand_and_partial_traffic"]["x_full_slice_bytes"], 0)
            self.assertEqual(gate["x_panel_refill_record_count"], gate["records"] + up["records"])
            self.assertEqual(down["x_panel_refill_record_count"], down["records"])

    def test_joint_gate_up_peak_refill_uses_joint_record_period(self):
        got = self.run_model()
        big = got["per_core"][0]
        expert = next(e for e in got["per_expert"] if e["owner"] == 0)
        gate, _, down = expert["projections"]
        expected = max(gate["capacity"]["x_k_panel_full_bytes"] / max(2 * gate["records"], 5),
                       down["capacity"]["x_k_panel_full_bytes"] / max(down["records"], 5))
        self.assertEqual(big["worst_case_per_issue_ports"]["X_panel_steady_state_refill_GB_s"], expected)

    def test_full_model_fallback_names_assumption(self):
        self.window.pop("routed_expert_count")
        self.window["experts"].pop(1)
        got = self.run_model()
        self.assertEqual(got["capacity"]["routed_expert_count"], 64)
        self.assertEqual(got["capacity"]["all_routed_and_shared_weight_bytes"], 3 * 8 * (64 * 6 + 12) * 2)
        self.assertIn("defaulted to 64", got["capacity"]["provenance"])

    def test_frequency_and_ii_affect_rates_not_traffic(self):
        base = self.run_model()
        self.window["frequency_hz"] = 500_000_000
        changed = self.run_model(ii=2)
        self.assertEqual(base["external"]["selected_unique_bf16_weight_bytes"], changed["external"]["selected_unique_bf16_weight_bytes"])
        self.assertEqual(changed["external"]["cold_weight_average_required_GB_s"], base["external"]["cold_weight_average_required_GB_s"] / 2)
        self.assertEqual(changed["compute_latency_ms"], base["compute_latency_ms"] * 2)
        old_peak = base["peak_refill"]["simultaneous_array_weight_full_GB_s_upper_envelope"]
        new_peak = changed["peak_refill"]["simultaneous_array_weight_full_GB_s_upper_envelope"]
        self.assertEqual(new_peak, old_peak / 4)

    def test_latched_weight_refill_can_be_lower_than_array_delivery(self):
        self.window["experts"][-1]["Me"] = self.window["batch"] = 8
        got = self.run_model()
        big = got["per_core"][0]["worst_case_per_issue_ports"]
        self.assertEqual(big["weight_pool_refill_GB_s_with_M_tile_latch_reuse"],
                         big["weight_full_operand_GB_s_at_II"] / 2)

    def test_output_is_json_safe_and_dimension_issues_match(self):
        got = self.run_model()
        json.dumps(got, allow_nan=False)
        for expert in got["per_expert"]:
            shape = got["per_core"][expert["owner"]]["shape"]
            total_useful = 0
            for p in expert["projections"]:
                self.assertEqual(p["issues"], p["m_tiles"] * p["n_tiles"] * p["k_tiles"])
                self.assertEqual(p["issued_macs"], p["issues"] * shape["pm"] * shape["pn"] * shape["pk"])
                total_useful += p["useful_macs"]
            self.assertEqual(total_useful, 3 * expert["Me"] * expert["H"] * expert["F"])

    def test_capacity_retains_gate_until_corresponding_up_and_has_no_quant_U(self):
        got = self.run_model()
        shared = next(e for e in got["per_expert"] if e["is_shared"])
        self.assertEqual(shared["gate_up_paired_partial_full_bytes"], 2 * 4 * 12 * 4)
        self.assertEqual(got["capacity"]["low_rank_U_bytes"], 0)
        self.assertTrue(got["capacity"]["capacity_does_not_imply_bandwidth"])
        self.assertTrue(got["peak_refill"]["latency_hiding"]["stall_free_not_guaranteed_by_double_buffering"])

    def test_invalid_assignment_rejected(self):
        self.assignments.append(dict(self.assignments[0]))
        with self.assertRaisesRegex(ValueError, "multiple cores"):
            self.run_model()

    def test_shared_batch_mismatch_rejected(self):
        self.window["experts"][-1]["Me"] = 3
        with self.assertRaisesRegex(ValueError, "Shared"):
            self.run_model()


if __name__ == "__main__":
    unittest.main()
