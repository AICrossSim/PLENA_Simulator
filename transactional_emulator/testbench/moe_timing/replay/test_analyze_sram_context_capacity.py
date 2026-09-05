import numpy as np

from transactional_emulator.testbench.moe_timing.replay.analyze_sram_context_capacity import (
    ORGANIZATIONS,
    TOTAL_CELLS,
    organization_for_jobs,
)


def _org(key: str):
    return next(org for org in ORGANIZATIONS if org.key == key)


def test_static_eight_context_pingpong_capacity_and_waves():
    jobs = np.array([1, 8, 9, 17], dtype=np.int64)
    contexts, depth, waves = organization_for_jobs(_org("static_8ctx_d2"), jobs)
    assert contexts.tolist() == [1, 8, 8, 8]
    assert depth.tolist() == [2, 2, 2, 2]
    assert waves.tolist() == [1, 1, 2, 3]
    assert np.all(contexts * depth * 4 <= TOTAL_CELLS)


def test_elastic_contexts_reassign_spare_cells_to_depth():
    jobs = np.array([1, 2, 4, 8, 9], dtype=np.int64)
    contexts, depth, waves = organization_for_jobs(_org("elastic_tagged_1to8ctx"), jobs)
    assert contexts.tolist() == [1, 2, 4, 8, 8]
    assert depth.tolist() == [16, 8, 4, 2, 2]
    assert waves.tolist() == [1, 1, 1, 1, 2]
    assert np.all(contexts * depth * 4 <= TOTAL_CELLS)


def test_blocking_baseline_serializes_all_jobs():
    jobs = np.array([1, 4, 11], dtype=np.int64)
    contexts, depth, waves = organization_for_jobs(_org("blocking_1ctx_d1"), jobs)
    assert contexts.tolist() == [1, 1, 1]
    assert depth.tolist() == [1, 1, 1]
    assert waves.tolist() == [1, 4, 11]
