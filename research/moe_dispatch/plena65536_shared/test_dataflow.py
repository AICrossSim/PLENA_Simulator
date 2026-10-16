"""Independent arithmetic/ownership tests for supply-only OS/WS/IS loops."""
from __future__ import annotations

from collections import defaultdict
from math import ceil
import unittest

from .dataflow import (BF16_BYTES, FP32_BYTES, Region, add_rates, gemm_traffic,
                       plena_gemm_traffic, plena_shared_traffic, shared_traffic)


def enumerate_refills(m, n, k, pm, pn, pk, flow):
    """Independent literal loop order; refill only when a retained tile changes."""
    ranges = {"m": range(0, m, pm), "n": range(0, n, pn), "k": range(0, k, pk)}
    order = {"OS": "mnk", "WS": "knm", "IS": "kmn"}[flow]
    visits = []
    for a in ranges[order[0]]:
        for b in ranges[order[1]]:
            for c in ranges[order[2]]:
                pos = dict(zip(order, (a, b, c)))
                visits.append((pos["m"], pos["n"], pos["k"]))
    last_w = last_x = None
    weight = activation = macs = 0
    for i, j, s in visits:
        am, an, ak = min(pm, m - i), min(pn, n - j), min(pk, k - s)
        w_key, x_key = (j, s), (i, s)
        if flow != "WS" or w_key != last_w:
            weight += an * ak * BF16_BYTES
        if flow != "IS" or x_key != last_x:
            activation += am * ak * BF16_BYTES
        last_w, last_x = w_key, x_key
        macs += am * an * ak
    return weight, activation, macs, len(visits)


class GemmAccountingTests(unittest.TestCase):
    def test_literal_loops_match_masked_refills_and_tails(self):
        for dims in ((5, 7, 9, 3, 4, 4), (8, 12, 16, 2, 4, 4), (1, 8, 16, 4, 4, 8)):
            for flow in ("OS", "WS", "IS"):
                with self.subTest(dims=dims, flow=flow):
                    got = gemm_traffic(*dims, flow)
                    w, x, macs, issues = enumerate_refills(*dims, flow)
                    self.assertEqual(got["pool_to_local"]["weight_masked_bytes"], w)
                    self.assertEqual(got["pool_to_local"]["x_masked_bytes"], x)
                    self.assertEqual(got["useful_macs"], macs)
                    self.assertEqual(got["issues"], issues)
                    self.assertGreaterEqual(got["issued_macs"], macs)

    def test_dataflow_changes_refills_not_arithmetic(self):
        got = [gemm_traffic(8, 12, 16, 2, 4, 4, flow) for flow in ("OS", "WS", "IS")]
        self.assertEqual(len({item["issues"] for item in got}), 1)
        self.assertEqual(len({item["useful_macs"] for item in got}), 1)
        self.assertEqual(len({item["issued_macs"] for item in got}), 1)
        self.assertEqual(got[0]["array_delivery"], got[1]["array_delivery"])
        self.assertEqual(got[0]["array_delivery"], got[2]["array_delivery"])
        self.assertEqual([item["weight_refill_factor"] for item in got], [4, 1, 4])
        self.assertEqual([item["x_refill_factor"] for item in got], [3, 3, 1])

    def test_os_partial_retention_and_ws_is_spill_are_charged(self):
        os, ws, ins = [gemm_traffic(8, 12, 16, 2, 4, 4, f) for f in ("OS", "WS", "IS")]
        self.assertEqual(os["pool_to_local"]["partial_read_masked_bytes"], 0)
        self.assertEqual(os["capacity"]["partial_scratch_full_bytes"], 0)
        for item in (ws, ins):
            self.assertEqual(item["pool_to_local"]["partial_read_masked_bytes"], 8 * 12 * 4 * 3)
            self.assertEqual(item["pool_to_local"]["partial_write_masked_bytes"], 8 * 12 * 4 * 3)
            self.assertEqual(item["capacity"]["partial_scratch_full_bytes"], 8 * 12 * 4)

    def test_one_k_segment_needs_no_partial_scratch(self):
        for flow in ("OS", "WS", "IS"):
            item = gemm_traffic(2, 4, 3, 2, 4, 4, flow)
            self.assertEqual(item["pool_to_local"]["partial_read_masked_bytes"], 0)
            self.assertEqual(item["capacity"]["partial_scratch_full_bytes"], 0)

    def test_empty_assignment_has_zero_traffic_and_capacity(self):
        for flow in ("OS", "WS", "IS"):
            item = gemm_traffic(0, 128, 2048, 4, 4, 512, flow)
            self.assertEqual(item["issues"], 0)
            self.assertFalse(any(item["pool_to_local"].values()))
            self.assertFalse(any(item["array_delivery"].values()))
            self.assertEqual(item["capacity"]["local_plus_partial_peak_bytes"], 0)

    def test_bad_inputs_rejected(self):
        for dims in ((-1, 4, 16, 2, 4, 8), (2, 4, 0, 2, 4, 8), (2, 4, 16, 0, 4, 8)):
            with self.assertRaises(ValueError):
                gemm_traffic(*dims)
        with self.assertRaises(ValueError):
            gemm_traffic(2, 4, 16, 2, 4, 8, "RS")


class SharedAccountingTests(unittest.TestCase):
    @staticmethod
    def row_parts(batch, split):
        return {name: [Region(0, 0, split, 0, n), Region(1, split, batch, 0, n)]
                for name, n in (("gate", 2816), ("up", 2816), ("down", 2048))}

    def test_unique_33mib_weight_fixed_all_batches_all_flows(self):
        for batch in (2, 4, 8, 16, 64, 96, 128):
            for flow in ("OS", "WS", "IS"):
                item = shared_traffic(batch, [(2, 32, 1024)], flow)
                self.assertEqual(item["external"]["cold_weight_only_unique_bytes"], 33 * 1024**2)
                self.assertEqual(item["capacity"]["all_shared_weights_bytes"], 33 * 1024**2)
                self.assertEqual(item["useful_macs"], 3 * batch * 2048 * 2816)
                self.assertEqual(item["external"]["fully_resident_weight_replay_bytes"], 0)

    def test_row_split_conserves_macs_and_global_unique_but_duplicates_local_w(self):
        batch = 8
        single = shared_traffic(batch, [(2, 16, 2048)], "WS")
        pair = shared_traffic(batch, [(2, 16, 1024), (2, 16, 1024)], "WS", self.row_parts(batch, 4))
        self.assertEqual(pair["useful_macs"], single["useful_macs"])
        self.assertEqual(pair["external"]["cold_weight_only_unique_bytes"], single["external"]["cold_weight_only_unique_bytes"])
        self.assertEqual(pair["pool_to_local"]["weight_masked_bytes"], 2 * 33 * 1024**2)
        self.assertEqual(pair["external"]["no_pool_stream_weight_refetch_factor"], 2)
        self.assertEqual(pair["external"]["unified_pool_external_weight_refetch_factor"], 1)

    def test_column_split_conserves_weights_with_ws(self):
        parts = {name: [Region(0, 0, 8, 0, n // 2), Region(1, 0, 8, n // 2, n)]
                 for name, n in (("gate", 2816), ("up", 2816), ("down", 2048))}
        item = shared_traffic(8, [(2, 16, 1024), (2, 16, 1024)], "WS", parts)
        self.assertEqual(item["pool_to_local"]["weight_masked_bytes"], 33 * 1024**2)
        self.assertEqual(item["external"]["no_pool_stream_weight_refetch_factor"], 1)

    def test_mixed_flows_apply_to_the_correct_core(self):
        item = shared_traffic(8, [(4, 32, 512), (2, 32, 512)], ["WS", "OS"], self.row_parts(8, 4))
        self.assertEqual(item["per_core"][0]["dataflow"], "WS")
        self.assertEqual(item["per_core"][1]["dataflow"], "OS")
        self.assertEqual(item["per_core"][0]["pool_to_local"]["weight_masked_bytes"], 33 * 1024**2)
        self.assertEqual(item["per_core"][1]["pool_to_local"]["weight_masked_bytes"], 2 * 33 * 1024**2)

    def test_ownership_cover_and_overlap_are_validated(self):
        parts = self.row_parts(8, 4)
        parts["gate"] = [Region(0, 0, 8, 0, 2816), Region(1, 4, 8, 0, 2816)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            shared_traffic(8, [(2, 16, 1024)] * 2, "OS", parts)
        parts["gate"] = [Region(0, 0, 4, 0, 2816)]
        with self.assertRaisesRegex(ValueError, "cover"):
            shared_traffic(8, [(2, 16, 1024)] * 2, "OS", parts)

    def test_stream_external_bytes_are_not_just_weight_bytes(self):
        item = shared_traffic(8, [(2, 32, 1024)], "IS")
        ext = item["external"]
        self.assertEqual(ext["no_pool_stream_total_bytes"],
                         ext["no_pool_stream_weight_bytes"] + ext["no_pool_stream_activation_bytes"]
                         + ext["no_pool_stream_partial_bytes"] + ext["no_pool_stream_intermediate_and_final_bytes"])
        self.assertGreater(ext["no_pool_stream_total_bytes"], ext["no_pool_stream_weight_bytes"])
        self.assertFalse(item["contract"]["native_WS_IS_support_claimed"])

    def test_rate_and_fetch_units_are_decimal_not_gib(self):
        item = shared_traffic(2, [(2, 32, 1024)], "OS")
        rates = add_rates(item, 1000)
        weight = 33 * 1024**2
        self.assertAlmostEqual(rates["unique_weight_required_GB_s"], weight / 1000)
        self.assertAlmostEqual(rates["unique_weight_fetch_ms"], weight / 256e6)
        self.assertEqual(rates["compute_ms"], 0.001)
        self.assertAlmostEqual(rates["useful_GFLOP_s"], 2 * item["useful_macs"] / 1000)


class PlenaTopologyTests(unittest.TestCase):
    def test_original_physical_count_is_65536_not_full_logical_product(self):
        item = plena_gemm_traffic(32, 2816, 2048, 32, 2048, 1, 1, "OS")
        geometry = item["native_geometry"]
        self.assertEqual(geometry["physical_multipliers"], 65536)
        self.assertEqual(geometry["G"], 64)
        self.assertEqual(geometry["temporal_steps_per_segment"], 32)
        self.assertEqual(item["capacity"]["local_accumulator_bytes"], 256 * 1024)
        self.assertEqual(item["capacity"]["shadow_partial_context_bytes"], 256 * 1024)

    def test_os_keeps_grouped_partials_until_single_final_m_sum(self):
        item = plena_gemm_traffic(8, 2048, 2816, 8, 512, 1, 2, "OS")
        self.assertEqual(item["k_tiles"], ceil(2816 / 512))
        self.assertEqual(item["M_SUM"]["invocations"], item["output_records"])
        self.assertTrue(item["M_SUM"]["only_after_complete_K"])
        self.assertEqual(item["pool_to_local"]["partial_read_full_bytes"], 0)
        self.assertEqual(item["capacity"]["partial_scratch_full_bytes"], 0)

    def test_ws_is_preserve_g_contexts_without_early_reduction(self):
        for flow in ("WS", "IS"):
            item = plena_gemm_traffic(5, 7, 9, 2, 8, 1, 2, flow)
            groups = 4
            expected_full_scratch = ceil(5 / 2) * ceil(7 / 4) * 2 * 4 * groups * 4
            self.assertEqual(item["capacity"]["partial_scratch_full_bytes"], expected_full_scratch)
            self.assertEqual(item["pool_to_local"]["partial_read_full_bytes"], expected_full_scratch)
            self.assertEqual(item["pool_to_local"]["partial_write_full_bytes"], expected_full_scratch)
            self.assertFalse(item["native_support_claimed"])

    def test_native_wrapper_accepts_root_lowercase_geometry_dict(self):
        got = plena_shared_traffic(2, [{"b": 32, "r": 2048, "rm": 1, "rn": 1}], "OS")
        self.assertEqual(got["external"]["cold_weight_only_unique_bytes"], 33 * 1024**2)
        self.assertEqual(got["geometry_backend"], "native_plena_explicit_replication")
        self.assertTrue(all(item["physical_multipliers"] == 65536 for item in got["projections"]))

    def test_replica_extension_does_not_claim_original_native_support(self):
        item = plena_gemm_traffic(2, 2816, 2048, 2, 2048, 1, 16, "OS")
        self.assertEqual(item["physical_multipliers"], 65536)
        self.assertFalse(item["native_support_claimed"])


if __name__ == "__main__":
    unittest.main()
