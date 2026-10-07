from __future__ import annotations

import numpy as np

from transactional_emulator.testbench.moe_timing.replay.analyze_full_swe_grouping import (
    Aggregate,
    MODEL_SPECS,
    aggregate_row,
    counts_from_routes,
    iter_decode_blocks,
)


def test_counts_from_routes_ignores_padding_sentinel() -> None:
    routes = np.asarray([[0, 1, 1, 4], [2, 2, 2, 4]], dtype=np.int64)
    counts = counts_from_routes(routes, num_experts=4)
    assert counts.tolist() == [[1, 2, 0, 0], [0, 0, 3, 0]]


def test_decode_cohorts_keep_partial_final_batch_and_valid_mask() -> None:
    indices = np.asarray(
        [
            [[[[0, 1]]]],
            [[[[1, 2]]]],
            [[[[2, 3]]]],
        ],
        dtype=np.int64,
    ).reshape(3, 1, 1, 2)
    valid = np.asarray([[True], [False], [True]])
    blocks = list(iter_decode_blocks(indices, valid, nominal_batch=2, num_experts=4))
    routes = np.concatenate([block[0] for block in blocks])
    active = np.concatenate([block[1] for block in blocks])
    assert routes.tolist() == [[0, 1, 4, 4], [2, 3, 4, 4]]
    assert active.tolist() == [1, 1]


def test_aggregate_reports_grouped_blen_padding() -> None:
    aggregate = Aggregate()
    aggregate.update(
        np.asarray([[7, 1, 0], [0, 2, 2]], dtype=np.int64),
        np.asarray([8, 4], dtype=np.int64),
    )
    assert aggregate.windows == 2
    assert aggregate.route_pairs == 12
    assert aggregate.active_expert_loads == 4
    assert aggregate.group_size_histogram == {1: 1, 2: 2, 7: 1}
    assert aggregate.grouped_issued_rows == 20


def test_aggregate_row_reports_each_cold_expert_bucket() -> None:
    aggregate = Aggregate()
    aggregate.update(
        np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
    )
    row = aggregate_row(
        aggregate,
        spec=MODEL_SPECS["qwen"],
        phase="decode",
        nominal_batch=8,
        layer="all",
        truth_scope="test",
    )
    assert [row[f"m{size}_expert_load_pct"] for size in range(1, 8)] == [12.5] * 7
    assert row["m8plus_expert_load_pct"] == 12.5
