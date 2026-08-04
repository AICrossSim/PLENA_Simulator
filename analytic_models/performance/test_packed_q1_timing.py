"""Checks for GQA broadcast chunking in the decode attention timing model."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from packed_q1_timing import packed_q1_matrix_histogram  # noqa: E402


def _ops(histogram):
    return dict(histogram)


class GqaBroadcastChunkingTest(unittest.TestCase):
    BASE = dict(
        cache_tokens=8448, batch=32, blen=32, hlen=128,
        query_heads=64, kv_heads=8, head_dim=128, batch_packed=True,
    )

    def test_ratio_wider_than_the_array_is_chunked_not_rejected(self) -> None:
        # Qwen3-32B has a GQA ratio of 8; at MLEN=HLEN=128 the array broadcasts
        # to one query head, so the group issues as 8 chunks over the same K/V.
        narrow = _ops(packed_q1_matrix_histogram(mlen=128, **self.BASE))
        wide = _ops(packed_q1_matrix_histogram(mlen=1024, **self.BASE))
        self.assertEqual(narrow["M_BTMM"], wide["M_BTMM"] * 8)

    def test_chunking_leaves_the_pv_stage_untouched(self) -> None:
        # Chunking repeats query-side work only; K/V stays resident.
        narrow = _ops(packed_q1_matrix_histogram(mlen=1024, **self.BASE))
        base = dict(self.BASE)
        base["kv_heads"] = 64  # ratio 1, no chunking
        plain = _ops(packed_q1_matrix_histogram(mlen=1024, **base))
        self.assertEqual(narrow["M_MM"], plain["M_MM"])

    def test_every_legal_array_width_is_priced(self) -> None:
        for mlen in (128, 256, 512, 1024, 2048):
            with self.subTest(mlen=mlen):
                ops = _ops(packed_q1_matrix_histogram(mlen=mlen, **self.BASE))
                self.assertGreater(ops["M_BTMM"], 0)
                self.assertGreater(ops["M_MM"], 0)

    def test_chunk_count_follows_the_ratio_over_the_broadcast_width(self) -> None:
        for mlen, expected in ((128, 8), (256, 4), (512, 2), (1024, 1), (2048, 1)):
            with self.subTest(mlen=mlen):
                ops = _ops(packed_q1_matrix_histogram(mlen=mlen, **self.BASE))
                per_chunk = (
                    math.ceil(self.BASE["batch"] / self.BASE["blen"])
                    * self.BASE["kv_heads"]
                    * math.ceil(self.BASE["cache_tokens"] / self.BASE["blen"])
                )
                self.assertEqual(ops["M_BTMM"], per_chunk * expected)


if __name__ == "__main__":
    unittest.main()
