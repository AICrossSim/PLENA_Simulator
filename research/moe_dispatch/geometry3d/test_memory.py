"""Independent address, resource and finite-request micro checks."""
from dataclasses import replace
import unittest

from .memory import (FabricProfile, HBMEndpoint, NativeWeightLayout, SharedHBM,
                     SRAM_BYTES, allocate_banks, memory_budget, projection_traffic,
                     record_group_traffic)


def transaction_oracle(n, k, n0, cols, k0, kv, base=0):
    """Enumerate every BF16 element; deliberately does not use layout helpers."""
    stride = ((2 * k + 31) // 32) * 32
    return tuple(sorted({(base + col * stride + 2 * red) // 32 * 32
                         for col in range(n0, n0 + cols)
                         for red in range(k0, k0 + kv)}))


def request_cycle_oracle(transactions, credits, latency, grant, landing):
    """Single transfer, explicit per-request response/commit timestamps."""
    now = accepted = committed = 0
    response_times = []
    commit_times = []
    returned = 0
    while committed < transactions:
        due_commits = sum(x == now for x in commit_times)
        committed += due_commits
        returned += sum(x == now for x in response_times)
        land = min(returned, landing)
        returned -= land
        commit_times.extend([now + 1] * land)
        free = credits - (accepted - committed)
        issue = min(grant, free, transactions - accepted)
        accepted += issue
        response_times.extend([now + latency] * issue)
        if committed == transactions:
            return now
        now += 1


class NativeAddressTests(unittest.TestCase):
    def test_original_native_bf16_valid_rows_and_short_k(self):
        layout = NativeWeightLayout(10, 1025)
        self.assertEqual(layout.storage_bytes, 10 * 2080)
        self.assertEqual(layout.tile(0, 10, 0, 1025).wire_bytes, 10 * 2080)
        self.assertEqual(layout.tile(9, 1, 1024, 1).useful_bytes, 2)
        self.assertEqual(layout.tile(9, 1, 1024, 1).wire_bytes, 32)

    def test_sector_addresses_against_element_oracle(self):
        for n, k, n0, cols, k0, kv in ((7, 515, 1, 5, 31, 130),
                                       (10, 1025, 9, 1, 1024, 1),
                                       (3, 1025, 0, 3, 480, 545),
                                       (8, 1408, 3, 4, 1024, 384)):
            layout = NativeWeightLayout(n, k, base=64)
            got = layout.tile(n0, cols, k0, kv)
            self.assertEqual(got.transactions, transaction_oracle(n, k, n0, cols, k0, kv, 64))
            self.assertEqual(sum(b for _, b in got.spans), got.wire_bytes)
            self.assertTrue(all(a % 32 == 0 for a in got.transactions))
            self.assertGreaterEqual(got.splice_read_bytes, got.useful_bytes)
            self.assertGreaterEqual(got.splice_write_bytes, got.useful_bytes)

    def test_coalesced_sector_fetched_once_no_capacity_alias(self):
        layout = NativeWeightLayout(1, 33)
        a, b = layout.tile(0, 1, 0, 3), layout.tile(0, 1, 3, 4)
        self.assertEqual(a.wire_bytes + b.wire_bytes, 64)
        joined = layout.coalesce((a, b))
        self.assertEqual(joined.wire_bytes, 32)
        self.assertEqual(joined.useful_bytes, 14)
        self.assertEqual(joined.splice_write_bytes, 16)

    def test_pk_does_not_change_unique_native_weights(self):
        for pk in (32, 64, 128, 256, 512, 1024):
            traffic = projection_traffic(2, 10, 1025, (1, 4, pk))
            self.assertEqual(traffic.unique_weight_storage_bytes, 20800)
            self.assertEqual(traffic.weight_hbm_bytes, 20800)
        layout = NativeWeightLayout(10, 1025, layout="v3_tiles")
        self.assertGreater(layout.storage_bytes, NativeWeightLayout(10, 1025).storage_bytes)


class FrozenBudgetTests(unittest.TestCase):
    def test_fixed_installed_cost_and_fixed_aggregate_ports(self):
        shapes = (((6, 4, 512),), ((3, 4, 512), (3, 4, 512)),
                  ((4, 4, 512), (1, 32, 128)))
        for cores in shapes:
            for batch in (2, 16, 64, 128):
                for allocation in ("equal", "proportional"):
                    b = memory_budget(cores, batch, 2048, 2816, allocation=allocation)
                    self.assertEqual(b.total_bytes, SRAM_BYTES)
                    self.assertEqual(b.slack_bytes, 0)
                    self.assertTrue(b.fits)
                    self.assertEqual(sum(c.w_bytes for c in b.cores), 40 * 1024)
                    self.assertEqual(sum(c.x_register_bytes for c in b.cores), 12 * 1024)
                    self.assertEqual(sum(c.accumulator_bytes for c in b.cores), 96 * 1024)
                    self.assertEqual(sum(c.z_bytes for c in b.cores), 384 * 1024)
                    self.assertEqual(sum(c.w_bandwidth for c in b.cores), 1024)
                    self.assertEqual(sum(c.x_bandwidth for c in b.cores), 384)
                    self.assertEqual(sum(c.accumulator_bandwidth for c in b.cores), 192)

    def test_single_w_buffer_remains_legal(self):
        b = memory_budget(((4, 4, 512), (1, 32, 128)), 16, 2048, 2816)
        self.assertTrue(b.fits)
        self.assertEqual(b.cores[1].w_slots, 1)
        self.assertEqual(b.cores[1].x_buffers, 2)

    def test_capacity_failures_do_not_resize_installed_budget(self):
        b = memory_budget(((6, 4, 512),), 256, 2048, 2816)
        self.assertFalse(b.fits)
        self.assertEqual(b.total_bytes, SRAM_BYTES)
        b = memory_budget(((12, 1, 1024),), 128, 2048, 2816)
        self.assertFalse(b.fits)
        self.assertEqual(b.cores[0].x_buffers, 0)

    def test_bank_partition_exact(self):
        for total in (12, 24, 64):
            for weights in ((1,), (1, 1), (1, 191), (2, 5)):
                partition = allocate_banks(total, weights)
                self.assertEqual(sum(partition), total)
                self.assertGreaterEqual(min(partition), 1)


class TrafficTests(unittest.TestCase):
    def test_bounded_record_group_against_issue_element_oracle(self):
        m, n, k, core = 5, 11, 65, (2, 3, 32)
        first, records = 1, 6
        nm = (m + core[0] - 1) // core[0]
        nt = {r // nm for r in range(first, first + records)}
        mt = {r % nm for r in range(first, first + records)}
        wire = xread = 0
        for k0 in range(0, k, core[2]):
            kv = min(core[2], k - k0)
            for ti in nt:
                cols = min(core[1], n - ti * core[1])
                wire += len(transaction_oracle(n, k, ti * core[1], cols, k0, kv)) * 32
            for ti in mt:
                rows = min(core[0], m - ti * core[0])
                xread += rows * ((kv * 2 + 15) // 16 * 16)
        out = record_group_traffic(m, n, k, core, first, records)
        self.assertEqual(out.weight_hbm_bytes, wire)
        self.assertEqual(out.x_sram_read_bytes, xread)
        self.assertEqual(out.issues, records * 3)
        self.assertEqual(out.accumulator_write_bytes, records * 3 * 2 * 3 * 4)
        self.assertEqual(out.accumulator_read_bytes, records * 2 * 2 * 3 * 4)
        no_reuse = record_group_traffic(m, n, k, core, first, records, False, False)
        self.assertGreaterEqual(no_reuse.weight_hbm_bytes, wire)
        self.assertGreaterEqual(no_reuse.x_sram_read_bytes, xread)

    def test_ws_is_os_reload_counts_and_finite_spill_backing(self):
        core = (2, 4, 32)
        ws = projection_traffic(8, 16, 65, core, "ws", 2, acc_capacity_bytes=8 * 8 * 4)
        os = projection_traffic(8, 16, 65, core, "os", 2, acc_capacity_bytes=2 * 8 * 4)
        ins = projection_traffic(8, 16, 65, core, "is", 2, x_capacity_bytes=2 * 32 * 2,
                                 acc_capacity_bytes=2 * 16 * 4)
        self.assertEqual(os.weight_hbm_bytes, 4 * ws.weight_hbm_bytes)
        self.assertEqual(ins.weight_hbm_bytes, 4 * ws.weight_hbm_bytes)
        self.assertEqual(ins.x_sram_read_bytes, 8 * 65 * 2)
        with self.assertRaisesRegex(ValueError, "finite spill backing"):
            projection_traffic(8, 16, 65, core, "is", 2, x_capacity_bytes=2 * 32 * 2,
                               acc_capacity_bytes=32)


class SharedHBMTests(unittest.TestCase):
    def test_finite_requests_against_independent_cycle_oracle(self):
        for count, credit, latency, bandwidth, land_bw in (
                (1, 256, 64, 256, 256), (11, 3, 7, 64, 32),
                (100, 16, 5, 128, 64), (1024, 256, 64, 256, 256)):
            profile = replace(FabricProfile(), hbm_credits=credit,
                              hbm_latency_cycles=latency, hbm_bytes_per_cycle=bandwidth,
                              landing_bytes_per_cycle=land_bw)
            server = SharedHBM(profile)
            t = server.submit(0, 0, count)
            server.advance(10000)
            expected = request_cycle_oracle(count, credit, latency, bandwidth // 32, land_bw // 32)
            self.assertEqual(t.ready_cycle, expected)
            self.assertEqual(server.credit_used, 0)
            self.assertLessEqual(server.credit_peak, credit)
            self.assertEqual(server.accepted_transactions, server.landed_transactions)
            self.assertEqual(server.pool_used, count * 32)
            server.consume(t)
            self.assertEqual(server.pool_used, 0)

    def test_two_cores_share_grants_without_doubling_bandwidth(self):
        server = SharedHBM()
        a = server.submit(0, 0, 4)
        b = server.submit(1, 0, 4)
        server.advance(65)
        self.assertEqual((a.ready_cycle, b.ready_cycle), (65, 65))
        self.assertEqual(server.accepted_transactions, 8)
        server.consume(a)
        server.consume(b)
        c = server.submit(0, 65, 1)
        server.advance(131)
        self.assertEqual(c.ready_cycle, 131)

    def test_landing_pool_lease_survives_until_future_last_use(self):
        profile = replace(FabricProfile(), landing_pool_bytes=128)
        server = SharedHBM(profile, per_core_capacity_bytes=(128,))
        a = server.submit(0, 0, 4)
        b = server.submit(0, 0, 4)
        server.advance(65)
        self.assertEqual(a.ready_cycle, 65)
        self.assertEqual(b.accepted, 0)
        server.consume(a, at=100)
        server.advance(99)
        self.assertEqual(b.accepted, 0)
        self.assertEqual(server.pool_used, 128)
        server.advance(165)
        self.assertEqual(b.ready_cycle, 165)
        self.assertLessEqual(server.pool_peak, 128)

    def test_private_pool_partitions_do_not_borrow(self):
        profile = replace(FabricProfile(), landing_pool_bytes=256)
        server = SharedHBM(profile, per_core_capacity_bytes=(128, 128))
        a = server.submit(0, 0, 4)
        b = server.submit(0, 0, 4)
        c = server.submit(1, 0, 4)
        server.advance(65)
        self.assertEqual(a.ready_cycle, c.ready_cycle)
        self.assertEqual(b.accepted, 0)

    def test_ingress_backpressure_with_large_credit_window(self):
        profile = replace(FabricProfile(), hbm_credits=1024, landing_bytes_per_cycle=32,
                          hbm_latency_cycles=1, ingress_bytes=64)
        server = SharedHBM(profile)
        t = server.submit(0, 0, 128)
        server.advance(1000)
        self.assertIsNotNone(t.ready_cycle)
        self.assertLessEqual(server.ingress_peak_bytes, 64)
        self.assertGreater(server.response_backpressure_cycles, 0)
        self.assertEqual(server.credit_used, 0)

    def test_continuous_endpoint_screen_is_explicitly_labeled(self):
        profile = FabricProfile()
        self.assertEqual(profile.credit_bandwidth_upper_bound, 128)
        self.assertAlmostEqual(profile.landing_credit_bandwidth_upper_bound, 8192 / 65)
        server = HBMEndpoint(profile)
        first = server.reserve(0, 8192, buffer_bytes=8192)
        self.assertEqual(first, 129)
        second = server.reserve(0, 8192, buffer_bytes=4096)
        self.assertEqual(second, 259)
        self.assertIn("approximation", server.report()["scope"])


if __name__ == "__main__":
    unittest.main()
