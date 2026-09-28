"""Small correctness/negative tests; execute with the plena-py311 interpreter."""
from collections import Counter
import unittest

import numpy as np

from numerical import (
    Contribution, Core, SRAM, activation, aligned_split, bf16, bf16_bits,
    combine_rank_order, copy_payload, execute_expert,
    reference_expert, reference_projection, validation_report,
)


class NumericalContractTests(unittest.TestCase):
    def test_bf16_rounding_ties_even_and_special_values(self):
        x = np.array([1.00390625, 1.01171875, np.inf, -np.inf, np.nan, -0.0], dtype=np.float32)
        bits = bf16_bits(x)
        self.assertEqual(int(bits[0]), 0x3F80)
        self.assertEqual(int(bits[1]), 0x3F82)
        self.assertEqual(int(bits[2]), 0x7F80)
        self.assertEqual(int(bits[3]), 0xFF80)
        self.assertNotEqual(int(bits[4]) & 0x007F, 0)
        self.assertEqual(int(bits[5]), 0x8000)

    def test_private_owner_lifetime_and_missing_payload(self):
        mem = SRAM(0, "audit", 128, Counter())
        t = mem.alloc("test", (2, 4), "bf16")
        with self.assertRaises(RuntimeError):
            mem.read(0, t)
        mem.write(0, t, np.ones((1, 4), dtype=np.float32), (slice(0, 1), slice(None)))
        with self.assertRaises(RuntimeError):
            mem.read(0, t)
        with self.assertRaises(PermissionError):
            mem.read(1, t, (slice(0, 1), slice(None)))
        mem.free(0, t)
        replacement = mem.alloc("replacement", (2, 4), "bf16")
        self.assertEqual(t.base, replacement.base)
        with self.assertRaises(RuntimeError):
            mem.write(0, t, np.zeros((2, 4), dtype=np.float32))

    def test_cross_core_copy_is_explicit_and_counts_both_endpoints(self):
        count = Counter()
        a, b = SRAM(0, "src", 64, count), SRAM(1, "dst", 64, count)
        src, dst = a.alloc("a", (2, 4), "bf16"), b.alloc("b", (2, 4), "bf16")
        a.write(0, src, np.arange(8, dtype=np.float32).reshape(2, 4))
        before = dict(count)
        copy_payload(a, src, b, dst, counters=count, kind="test")
        self.assertEqual(count["test_payload_bytes"], 16)
        self.assertEqual(count["cross_core_payload_bytes"], 16)
        self.assertEqual(count["src_read_bytes"] - before.get("src_read_bytes", 0), 16)
        self.assertEqual(count["dst_write_bytes"] - before.get("dst_write_bytes", 0), 16)
        np.testing.assert_array_equal(b.read(1, dst), np.arange(8).reshape(2, 4))

    def test_capacity_is_per_private_memory_and_copy_cannot_change_dtype(self):
        count = Counter()
        a, b = SRAM(0, "a", 16, count), SRAM(1, "b", 1024, count)
        src = a.alloc("full", (2, 4), "bf16")
        with self.assertRaises(MemoryError):
            a.alloc("one_more", (1,), "bf16")
        dest = b.alloc("fp32", (2, 4), "f32")
        with self.assertRaises(ValueError):
            copy_payload(a, src, b, dest, counters=count, kind="illegal_cast")

    def test_whole_and_paired_split_exact_all_tail_shapes(self):
        report = validation_report()
        self.assertTrue(report["all_bit_exact"])
        self.assertEqual(report["execution_count"], 25)
        for case in report["cases"]:
            self.assertEqual(len({e["sha256_bf16"] for e in case["executions"]}), 1)
            for execution in case["executions"]:
                for core in execution["core_peaks"]:
                    self.assertLessEqual(core["workspace_bytes"], core["workspace_capacity"])

    def test_small_core_m_tail_reads_only_actual_rows(self):
        rng = np.random.default_rng(44)
        x = rng.normal(0, 0.2, (3, 9)).astype(np.float32)
        weights = [rng.normal(0, 0.1, shape).astype(np.float32) for shape in ((7, 9), (7, 9), (9, 7))]
        execution = execute_expert(x, *weights, core_m=(2,), mode="whole")
        np.testing.assert_array_equal(bf16_bits(execution.output()), bf16_bits(reference_expert(x, *weights)))
        self.assertGreater(execution.counters["issued_mac_slots"], execution.counters["useful_macs"])
        self.assertEqual(execution.counters["useful_macs"], 3 * 3 * 9 * 7)

    def test_split_does_not_get_free_workspace(self):
        x = np.ones((4, 13), dtype=np.float32)
        weights = [np.ones(shape, dtype=np.float32) for shape in ((9, 13), (9, 13), (13, 9))]
        with self.assertRaises(MemoryError):
            execute_expert(x, *weights, mode="split_n", workspace_bytes=(100, 1024 * 1024))

    def test_gate_z_alias_fits_without_allocating_third_tensor(self):
        counters = Counter()
        core = Core(0, 2, 64, counters)
        gate = core.workspace.alloc("gate_z", (2, 8), "bf16")
        up = core.workspace.alloc("up", (2, 8), "bf16")
        g = bf16(np.linspace(-1, 1, 16, dtype=np.float32).reshape(2, 8))
        u = bf16(np.linspace(0.1, 2, 16, dtype=np.float32).reshape(2, 8))
        core.workspace.write(0, gate, g)
        core.workspace.write(0, up, u)
        with self.assertRaises(MemoryError):
            core.workspace.alloc("separate_z_would_overflow", (2, 8), "bf16")
        before_reads = counters["workspace_read_bytes"]
        result = core.activate(gate, up)
        self.assertEqual(result, gate)
        self.assertEqual(counters["workspace_read_bytes"] - before_reads, 64)
        self.assertEqual(core.workspace.peak_bytes, 64)
        self.assertEqual(core.workspace.live_bytes, 32)
        np.testing.assert_array_equal(bf16_bits(core.workspace.read(0, result)), bf16_bits(activation(g, u)))
        with self.assertRaises(RuntimeError):
            core.workspace.read(0, up)

    def test_partial_gate_z_alias_does_not_initialize_remote_columns(self):
        counters = Counter()
        core = Core(1, 2, 64, counters)
        gate = core.workspace.alloc("full_gate_z", (2, 8), "bf16")
        up = core.workspace.alloc("local_up", (2, 4), "bf16")
        core.workspace.write(1, gate, np.ones((2, 4), dtype=np.float32), (slice(None), slice(4, 8)))
        core.workspace.write(1, up, np.ones((2, 4), dtype=np.float32))
        result = core.activate(gate, up, columns=(4, 8))
        with self.assertRaises(RuntimeError):
            core.workspace.read(1, result)
        with self.assertRaises(PermissionError):
            core.workspace.read(0, result, (slice(None), slice(4, 8)))
        observed = core.workspace.read(1, result, (slice(None), slice(4, 8)))
        expected = bf16(activation(np.ones((2, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32)))
        np.testing.assert_array_equal(observed, expected)

    def test_split_must_have_two_nonempty_aligned_shards(self):
        self.assertEqual(aligned_split(5), 4)
        with self.assertRaises(ValueError):
            aligned_split(4)

    def test_compute_uses_staged_payload_not_host_original(self):
        count = Counter()
        c = Core(0, 2, 4096, count)
        original = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        staged = c.stage_input(original)
        original[:] = 12345
        # Mutating the actual SRAM payload changes the result, unlike the host array.
        c.workspace.write(0, staged, np.array([[1, 0, 0, 0, 0]], dtype=np.float32))
        y = c.project(staged, np.ones((3, 5), dtype=np.float32), 0, 3, "result")
        np.testing.assert_array_equal(c.workspace.read(0, y), np.ones((1, 3)))

    def test_dot_uses_balanced_512_tree_not_serial_sum(self):
        counters = Counter()
        core = Core(0, 2, 4096, counters)
        x = core.stage_input(np.ones((1, 4), dtype=np.float32))
        w = np.array([[2.0 ** 25, 1, -(2.0 ** 25), 1]], dtype=np.float32)
        y = core.project(x, w, 0, 1, "tree_result")
        observed = core.workspace.read(0, y)
        # Pairwise cancellation gives 0; serial FP32 addition would give 1.
        np.testing.assert_array_equal(observed, np.zeros((1, 1)))
        np.testing.assert_array_equal(observed, reference_projection(np.ones((1, 4)), w))

    def test_route_sum_rounds_before_shared_addition(self):
        counters = Counter()
        destination = Core(99, 2, 4096, counters)
        routed_mem, shared_mem = SRAM(0, "routed", 64, counters), SRAM(1, "shared", 64, counters)
        routed = routed_mem.alloc("routed_y", (1, 1), "f32")
        shared = shared_mem.alloc("shared_y", (1, 1), "f32")
        routed_mem.write(0, routed, np.array([[2]], dtype=np.float32))
        shared_mem.write(1, shared, np.array([[0.00390625]], dtype=np.float32))
        parts = [Contribution(0, routed_mem, routed, (0,), np.array([0.501953125], dtype=np.float32)),
                 Contribution(-1, shared_mem, shared, (0,), np.ones(1, dtype=np.float32), is_shared=True)]
        result = combine_rank_order(destination, parts[::-1], 1, 1)
        np.testing.assert_array_equal(destination.workspace.read(99, result), np.array([[1.0]], dtype=np.float32))
        self.assertEqual(float(bf16(np.array([[2 * 0.501953125 + 0.00390625]]))[0, 0]), 1.0078125)

    def test_router_rank_combine_not_completion_order(self):
        counters = Counter()
        destination = Core(99, 4, 4096, counters)
        contributions = []
        # Explicit FP32 cancellation makes completion-order reduction observably wrong.
        for rank, value in enumerate((2.0 ** 25, -(2.0 ** 25), 1.0)):
            source = SRAM(rank, "source", 256, counters)
            t = source.alloc("expert_output", (2, 5), "bf16")
            source.write(rank, t, np.full((2, 5), value, dtype=np.float32))
            contributions.append(Contribution(rank, source, t, (0, 1), np.ones(2, dtype=np.float32)))
        combined = combine_rank_order(destination, [contributions[2], contributions[0], contributions[1]], 2, 5)
        np.testing.assert_array_equal(destination.workspace.read(99, combined), np.ones((2, 5)))
        self.assertEqual(counters["combine_transfer_payload_bytes"], 3 * 2 * 5 * 2)
        self.assertEqual(counters["cross_core_payload_bytes"], 3 * 2 * 5 * 2)

    def test_complete_small_moe_routes_shared_expert_and_ordered_combine(self):
        rng = np.random.default_rng(91)
        x = rng.normal(0, 0.2, (4, 13)).astype(np.float32)
        routing = [((0, 1, 2, 3), (0, 0, 1, 0)), ((0, 2), (1, 0)),
                   ((1, 3), (1, 1)), ((0, 1, 2, 3), (2, 2, 2, 2))]
        expert_weights = [[rng.normal(0, 0.1, shape).astype(np.float32)
                           for shape in ((7, 13), (7, 13), (13, 7))] for _ in routing]
        expected = np.zeros_like(x)
        reference_rows = []
        for (ids, ranks), weights in zip(routing, expert_weights):
            reference = reference_expert(x[list(ids)], *weights)
            for row, (token, rank) in enumerate(zip(ids, ranks)):
                score = np.float32((0.7, 0.3, 1.0)[rank])
                reference_rows.append((rank, token, reference[row], score))
        for rank, token, value, score in sorted(reference_rows, key=lambda r: (r[0], r[1])):
            if rank < 2:
                expected[token] = np.add(expected[token], np.multiply(value, score, dtype=np.float32), dtype=np.float32)
        expected = bf16(expected)
        for rank, token, value, _ in reference_rows:
            if rank == 2:
                expected[token] = bf16(np.add(expected[token], value, dtype=np.float32))
        outputs = []
        for mode in ("whole", "split_n"):
            destination = Core(99, 4, 128 * 1024, Counter())
            contributions = []
            for (ids, ranks), weights in zip(routing, expert_weights):
                execution = execute_expert(x[list(ids)], *weights, mode=mode)
                contributions.append(Contribution(ranks, execution.result_core.workspace,
                                                  execution.result, ids,
                                                  np.array([(0.7, 0.3, 1.0)[r] for r in ranks], dtype=np.float32),
                                                  is_shared=all(r == 2 for r in ranks)))
            result = combine_rank_order(destination, contributions[::-1], 4, 13)
            observed = destination.workspace.read(99, result)
            np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
            outputs.append(observed)
            self.assertEqual(destination.counters["combine_transfer_payload_bytes"], (4 + 2 + 2 + 4) * 13 * 4)
        np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))

    def test_router_rank_varies_per_token_and_expert(self):
        counters = Counter()
        destination = Core(99, 2, 4096, counters)
        contributions = []
        # For token0, expert0 precedes expert1. For token1, the order reverses.
        for owner, ranks, values in ((0, (0, 1), (2, 3)), (1, (1, 0), (5, 7))):
            source = SRAM(owner, "source", 256, counters)
            t = source.alloc("expert_output", (2, 1), "bf16")
            source.write(owner, t, np.array(values, dtype=np.float32).reshape(2, 1))
            contributions.append(Contribution(ranks, source, t, (0, 1), np.array([0.5, 0.25], dtype=np.float32)))
        result = combine_rank_order(destination, contributions[::-1], 2, 1)
        np.testing.assert_array_equal(destination.workspace.read(99, result), np.array([[3.5], [2.5]], dtype=np.float32))
        contributions[1].rank = (0, 0)
        with self.assertRaises(ValueError):
            combine_rank_order(destination, contributions, 2, 1)


if __name__ == "__main__":
    unittest.main()
