"""Independent full-workload conservation and supply-boundary checks."""
from __future__ import annotations

import unittest

from .dataflow import BF16_BYTES, FP32_BYTES, plena_mixed_traffic, resource_ledger


def window():
    return {"id": "fixture", "batch": 2, "hidden": 2048, "top_k": 2,
            "experts": [
                {"id": -1, "is_shared": True, "Me": 2, "H": 2048, "F": 2816},
                {"id": 7, "is_shared": False, "Me": 1, "H": 2048, "F": 1408},
                {"id": 9, "is_shared": False, "Me": 3, "H": 2048, "F": 1408},
                {"id": 11, "is_shared": False, "Me": 0, "H": 2048, "F": 1408},
            ]}


def plan():
    return {"cycles": 50, "jobs": [
        {"expert_id": -1, "core_id": 0, "start": 0, "end": 50},
        {"expert_id": 7, "core_id": 1, "start": 0, "end": 10},
        {"expert_id": 9, "core_id": 1, "start": 10, "end": 40},
    ]}


CORES = ({"b": 2, "r": 256, "rm": 1, "rn": 64},
         {"b": 2, "r": 256, "rm": 1, "rn": 64})


class FullMoETrafficTests(unittest.TestCase):
    def test_actual_shared_and_routed_f_are_conserved(self):
        got = plena_mixed_traffic(window(), CORES, "OS", plan())
        self.assertEqual(got["active_routed_experts"], 2)
        self.assertEqual(got["shared_tokens"], 2)
        self.assertEqual(got["routed_token_assignments"], 4)
        self.assertEqual(got["shared_useful_macs"], 3 * 2 * 2048 * 2816)
        self.assertEqual(got["routed_useful_macs"], 3 * 4 * 2048 * 1408)
        self.assertEqual(got["useful_macs"], got["shared_useful_macs"] + got["routed_useful_macs"])
        self.assertEqual(got["external"]["cold_weight_only_unique_bytes"],
                         (33 + 2 * 16.5) * 1024**2)
        self.assertGreaterEqual(got["issued_macs"], got["useful_macs"])

    def test_inactive_expert_weight_is_not_counted_and_address_alias_not_identity(self):
        source = window()
        for expert in source["experts"]:
            expert["weights"] = {name: {"hbm_base": 0} for name in ("gate", "up", "down")}
        got = plena_mixed_traffic(source, CORES, "WS", plan())
        self.assertEqual(len(got["experts"]), 3)
        self.assertEqual(got["external"]["cold_weight_only_unique_bytes"], 66 * 1024**2)
        self.assertEqual(got["pool_to_local"]["weight_masked_bytes"], 66 * 1024**2)

    def test_unique_external_x_once_and_combined_y_once(self):
        got = plena_mixed_traffic(window(), CORES, "OS", plan())
        ext = got["external"]
        self.assertEqual(ext["cold_weights_inputs_final_output_bytes"],
                         66 * 1024**2 + 2 * 2048 * (BF16_BYTES + FP32_BYTES))
        self.assertEqual(got["postprocess"]["expert_Y_combine_read_fp32_bytes"],
                         (2 + 1 + 3) * 2048 * FP32_BYTES)
        self.assertEqual(got["postprocess"]["combined_Y_write_fp32_bytes"], 2 * 2048 * FP32_BYTES)
        self.assertEqual(ext["no_pool_stream_total_bytes"],
                         sum(ext[key] for key in ("no_pool_stream_weight_bytes",
                                                  "no_pool_stream_activation_bytes",
                                                  "no_pool_stream_partial_bytes",
                                                  "no_pool_stream_intermediate_and_final_bytes")))

    def test_cross_flow_arithmetic_and_external_unique_are_identical(self):
        points = [plena_mixed_traffic(window(), CORES, flows, plan())
                  for flows in ((a, b) for a in ("OS", "WS", "IS") for b in ("OS", "WS", "IS"))]
        for key in ("useful_macs", "issued_macs", "issues", "shared_useful_macs", "routed_useful_macs"):
            self.assertEqual(len({point[key] for point in points}), 1)
        self.assertEqual(len({point["external"]["cold_weight_only_unique_bytes"] for point in points}), 1)
        self.assertGreater(len({point["pool_to_local"]["weight_masked_bytes"] for point in points}), 1)
        self.assertGreater(len({point["pool_to_local"]["partial_read_masked_bytes"] for point in points}), 1)

    def test_static_hardware_once_and_scheduled_scratch_uses_actual_overlap(self):
        got = plena_mixed_traffic(window(), CORES, "WS", plan())
        ledger = resource_ledger(CORES)
        self.assertEqual(ledger["multipliers"], 65536)
        self.assertEqual(got["capacity"]["fixed_hardware_storage_bytes"], ledger["static_storage_bytes"])
        scratch = {entry["expert_id"]: entry["scratch_bytes"] for entry in got["scratch_execution_intervals"]}
        self.assertEqual(got["capacity"]["scheduled_scratch_upper_bytes"],
                         scratch[-1] + max(scratch[7], scratch[9]))
        self.assertTrue(got["capacity"]["scheduled_scratch_uses_job_intervals"])
        self.assertEqual(got["capacity"]["unified_resident_payload_upper_bytes"],
                         got["capacity"]["active_unique_weights_bytes"]
                         + got["capacity"]["activation_liveness_upper_bytes"]
                         + ledger["static_storage_bytes"]
                         + got["capacity"]["scheduled_scratch_upper_bytes"])

    def test_schedule_missing_or_duplicate_job_rejected(self):
        for jobs in (plan()["jobs"][:-1], plan()["jobs"] + [plan()["jobs"][-1]]):
            with self.assertRaisesRegex(ValueError, "every active expert"):
                plena_mixed_traffic(window(), CORES, "OS", {"jobs": jobs})

    def test_reused_physical_id_is_rejected(self):
        source = window()
        source["experts"][2]["id"] = 7
        with self.assertRaisesRegex(ValueError, "identities"):
            plena_mixed_traffic(source, CORES, "OS", plan())


if __name__ == "__main__":
    unittest.main()
