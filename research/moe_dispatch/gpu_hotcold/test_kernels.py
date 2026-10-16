"""GPU numerical checks for the grouped prototype; never benchmark fixtures.

Tests skip explicitly when CUDA/BF16/Triton are unavailable. CPU trace tests
remain independently executable without PyTorch.
"""
from __future__ import annotations

import importlib.util
import unittest

try:
    import torch
except ImportError:
    torch = None

CUDA_BF16 = bool(torch is not None and torch.cuda.is_available()
                 and torch.cuda.is_bf16_supported()
                 and importlib.util.find_spec("triton") is not None)


@unittest.skipUnless(CUDA_BF16, "Numerical kernel checks require CUDA, BF16, and Triton")
class GroupedKernelNumerics(unittest.TestCase):
    def setUp(self):
        from . import kernels
        self.kernels = kernels
        self.device = "cuda"
        torch.manual_seed(417)

    def ints(self, values):
        return torch.tensor(values, dtype=torch.int32, device=self.device)

    def values(self, *shape):
        return (torch.randn(*shape, device=self.device, dtype=torch.float32) * 0.1).bfloat16()

    def assert_close(self, actual, expected, *, atol=0.003, rtol=0.025):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)

    def plan(self, route_ids, experts):
        rows = sorted((expert, token, slot) for token, ids in enumerate(route_ids)
                      for slot, expert in enumerate(ids))
        counts = [sum(row[0] == expert for row in rows) for expert in range(experts)]
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + count)
        packed = [[0] * len(route_ids[0]) for _ in route_ids]
        for row, (_, token, slot) in enumerate(rows):
            packed[token][slot] = row
        return rows, counts, starts, packed

    def tile_map(self, selected, counts):
        tiles = [(expert, offset) for expert in selected for offset in range(0, counts[expert], 16)]
        return self.ints([t[0] for t in tiles]), self.ints([t[1] for t in tiles])

    def test_grouped_matmul_masks_multitile_and_empty_expert(self):
        counts = [1, 2, 0, 19]
        starts = [0, 1, 3, 3, 22]
        a, w = self.values(22, 33), self.values(4, 33, 70)
        out = torch.full((22, 70), torch.nan, device=self.device, dtype=torch.bfloat16)
        tile_e, tile_m = self.tile_map(range(4), counts)
        self.kernels.grouped_matmul(a, w, out, self.ints(starts), tile_e, tile_m)
        expected = torch.empty_like(out)
        for expert in range(4):
            lo, hi = starts[expert:expert + 2]
            expected[lo:hi] = (a[lo:hi].float() @ w[expert].float()).bfloat16()
        self.assert_close(out, expected)

    def test_empty_population_is_noop(self):
        a, w = self.values(2, 32), self.values(3, 32, 48)
        output = self.values(2, 48)
        before = output.clone()
        empty = self.ints([])
        self.kernels.grouped_matmul(a, w, output, self.ints([0, 0, 2, 2]), empty, empty)
        self.kernels.gather_rows(a, self.ints([0, 1]), empty, a)
        torch.testing.assert_close(output, before, atol=0, rtol=0)

    def test_shared_hot_cold_pipeline_matches_perexpert_reference(self):
        # Five hot experts shared by all six tokens; expert 6 cold (Me=2),
        # expert 5 hot (Me=4), expert 7 inactive. All six routes remain distinct.
        route_ids = [[0, 1, 2, 3, 4, 6 if token < 2 else 5] for token in range(6)]
        self.full_pipeline_check(route_ids)

    def test_zero_hot_b2_full_pipeline(self):
        self.full_pipeline_check([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 6, 7]])

    def full_pipeline_check(self, route_ids):
        batch, hidden, width, experts = len(route_ids), 35, 48, 8
        rows, counts, starts, pack = self.plan(route_ids, experts)
        x = self.values(batch, hidden)
        wgu, wd = self.values(experts, hidden, 2 * width), self.values(experts, width, hidden)
        shared_gu, shared_down = self.values(hidden, 2 * width), self.values(width, hidden)
        shared_gu_result = (x.float() @ shared_gu.float()).bfloat16()
        sg, su = shared_gu_result[:, :width].float(), shared_gu_result[:, width:].float()
        shared_z = (torch.nn.functional.silu(sg) * su).bfloat16()
        shared = (shared_z.float() @ shared_down.float()).bfloat16()
        scores = torch.tensor([[float(slot + 1) / 17 for slot in range(6)] for _ in range(batch)],
                              device=self.device, dtype=torch.float32)
        packed_x = torch.full((batch * 6, hidden), torch.nan, device=self.device, dtype=torch.bfloat16)
        gu = torch.full((batch * 6, 2 * width), torch.nan, device=self.device, dtype=torch.bfloat16)
        z = torch.full((batch * 6, width), torch.nan, device=self.device, dtype=torch.bfloat16)
        routed = torch.full((batch * 6, hidden), torch.nan, device=self.device, dtype=torch.bfloat16)
        token_ids = self.ints([token for _, token, _ in rows])
        for selected in ([e for e in range(experts) if counts[e] > 2],
                         [e for e in range(experts) if 0 < counts[e] <= 2]):
            row_ids = self.ints([row for row, (e, _, _) in enumerate(rows) if e in selected])
            te, tm = self.tile_map(selected, counts)
            self.kernels.gather_rows(x, token_ids, row_ids, packed_x)
            self.kernels.grouped_matmul(packed_x, wgu, gu, self.ints(starts), te, tm)
            self.kernels.silu_multiply_rows(gu, row_ids, z)
            self.kernels.grouped_matmul(z, wd, routed, self.ints(starts), te, tm)
        result = torch.empty_like(x)
        self.kernels.combine_routed(routed, self.ints(pack), scores, shared, result)

        reference = shared.float().clone()
        for expert in range(experts):
            lo, hi = starts[expert:expert + 2]
            if lo == hi:
                continue
            toks = [rows[r][1] for r in range(lo, hi)]
            local_x = x[toks]
            local_gu = (local_x.float() @ wgu[expert].float()).bfloat16()
            g, u = local_gu[:, :width].float(), local_gu[:, width:].float()
            local_z = (torch.nn.functional.silu(g) * u).bfloat16()
            local_y = (local_z.float() @ wd[expert].float()).bfloat16()
            self.assert_close(routed[lo:hi], local_y)
            for relative, packed_row in enumerate(range(lo, hi)):
                _, token, slot = rows[packed_row]
                reference[token] += local_y[relative].float() * scores[token, slot]
        self.assert_close(result, reference.bfloat16())
        self.assertTrue(bool(torch.isfinite(gu).all()))
        self.assertTrue(bool(torch.isfinite(z).all()))
        self.assertTrue(bool(torch.isfinite(packed_x).all()))


if __name__ == "__main__":
    unittest.main()
