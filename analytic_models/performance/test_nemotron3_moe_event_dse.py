from __future__ import annotations

import pytest

from transactional_emulator.testbench.model_configs.loader import load_model_config

from .nemotron3_moe_event_dse import (
    BASELINE_PES,
    HBM_BURST_BYTES,
    Candidate,
    CoreGeometry,
    CycleCalibration,
    ExpertJob,
    Mapping,
    MoeShape,
    candidate_space,
    plan_job,
    round_up,
    routing_pattern_histogram,
    simulate_event,
)
from .nemotron3_routing_dse import (
    load_routing_trace,
    simulate_routed_expert_lru,
)


def _shape() -> MoeShape:
    return MoeShape.from_arch(load_model_config("nemotron3_nano_30b_a3b").arch)


def _candidate(mapping: Mapping, cores: int = 4) -> Candidate:
    return Candidate(
        name=f"test_{mapping}",
        topology="test",
        cores=tuple(CoreGeometry(1, 1024) for _ in range(cores)),
        mapping=mapping,
    )


def _job(shape: MoeShape, *, cached: bool = False, kind: str = "routed") -> ExpertJob:
    shared = kind == "shared"
    return ExpertJob(
        job_id=0,
        kind=kind,
        expert_id=None if shared else 7,
        tokens=1,
        intermediate=(
            shape.shared_intermediate if shared else shape.routed_intermediate
        ),
        weight_bytes=shape.shared_weight_bytes if shared else shape.routed_weight_bytes,
        weight_cached=cached,
    )


def test_routing_patterns_match_the_independent_lru_replay() -> None:
    trace = load_routing_trace()
    shape = _shape()
    for capacity in (0, 92, 137, 138, 256, 512):
        patterns = routing_pattern_histogram(
            trace,
            capacity_entries=capacity,
            expert_order="expert_id",
        )
        replay = simulate_routed_expert_lru(
            trace,
            entry_bytes=shape.routed_weight_bytes,
            capacity_entries=capacity,
            expert_order="expert_id",
        )
        assert patterns.events == 127 * 23
        assert patterns.accesses == 127 * 23 * 6
        assert patterns.hits == replay.hits
        assert patterns.misses == replay.misses
        assert patterns.prefill_resident_entries == replay.prefill_resident_entries


def test_candidate_space_preserves_the_4096_pe_budget() -> None:
    candidates = candidate_space()
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert all(candidate.pe_count == BASELINE_PES for candidate in candidates)


def test_split_mappings_preserve_physical_weight_bytes() -> None:
    shape = _shape()
    job = _job(shape)
    expected = round_up(shape.routed_weight_bytes, HBM_BURST_BYTES)
    for mapping in (Mapping.K_SPLIT, Mapping.N_TO_K):
        plan = plan_job(job, _candidate(mapping), shape)
        assert sum(transfer.bytes for transfer in plan.transfers) == expected
        assert all(transfer.bytes % HBM_BURST_BYTES == 0 for transfer in plan.transfers)
        assert len(plan.parts) == 4


def test_cache_hit_skips_every_weight_transfer() -> None:
    shape = _shape()
    for mapping in (Mapping.EXPERT, Mapping.M_SPLIT, Mapping.K_SPLIT, Mapping.N_TO_K):
        plan = plan_job(_job(shape, cached=True), _candidate(mapping), shape)
        assert plan.transfers == ()
        assert all(part.transfer_id is None for part in plan.parts)


def test_b1_m_split_collapses_to_one_active_core() -> None:
    shape = _shape()
    plan = plan_job(_job(shape), _candidate(Mapping.M_SPLIT), shape)
    assert len(plan.parts) == 1
    assert plan.parts[0].tokens == 1
    assert plan.parts[0].preferred_core_id == 0


def test_k_split_pays_both_partial_sum_reductions() -> None:
    shape = _shape()
    k_split = plan_job(_job(shape, cached=True), _candidate(Mapping.K_SPLIT), shape)
    n_to_k = plan_job(_job(shape, cached=True), _candidate(Mapping.N_TO_K), shape)
    assert k_split.postprocess_cycles > n_to_k.postprocess_cycles
    assert sum(part.up_k for part in k_split.parts) == shape.hidden
    assert sum(part.down_k for part in k_split.parts) == shape.routed_intermediate
    assert sum(part.up_n for part in n_to_k.parts) == shape.routed_intermediate
    assert sum(part.down_k for part in n_to_k.parts) == shape.routed_intermediate


def test_event_scheduler_conserves_miss_traffic_and_completes_all_jobs() -> None:
    shape = _shape()
    jobs = (
        *(
            ExpertJob(
                job_id=index,
                kind="routed",
                expert_id=index,
                tokens=1,
                intermediate=shape.routed_intermediate,
                weight_bytes=shape.routed_weight_bytes,
                weight_cached=index % 2 == 0,
            )
            for index in range(6)
        ),
        ExpertJob(
            job_id=6,
            kind="shared",
            expert_id=None,
            tokens=1,
            intermediate=shape.shared_intermediate,
            weight_bytes=shape.shared_weight_bytes,
            weight_cached=True,
        ),
    )
    calibration = CycleCalibration("test", 1.0, 64.0, "unit test")
    result = simulate_event(
        jobs,
        _candidate(Mapping.K_SPLIT),
        shape,
        calibration,
    )
    assert result.completed_jobs == 7
    assert result.event_cycles > 0
    assert result.hbm_bytes == 3 * round_up(
        shape.routed_weight_bytes,
        HBM_BURST_BYTES,
    )
    assert 0 < result.matrix_pe_utilization <= 1
    assert result.max_buffer_occupancy_bytes <= 2 * shape.routed_weight_bytes


def test_event_scheduler_rejects_a_buffer_smaller_than_one_job() -> None:
    shape = _shape()
    calibration = CycleCalibration("test", 1.0, 64.0, "unit test")
    with pytest.raises(ValueError, match="cannot stage"):
        simulate_event(
            (_job(shape),),
            _candidate(Mapping.EXPERT),
            shape,
            calibration,
            weight_buffer_bytes=shape.routed_weight_bytes - HBM_BURST_BYTES,
        )


def test_dynamic_mapping_only_gangs_the_shared_expert() -> None:
    shape = _shape()
    candidate = _candidate(Mapping.DYNAMIC)
    routed = plan_job(_job(shape, cached=True), candidate, shape)
    shared = plan_job(_job(shape, cached=True, kind="shared"), candidate, shape)
    assert routed.mode == Mapping.EXPERT
    assert len(routed.parts) == 1
    assert shared.mode == Mapping.N_TO_K
    assert len(shared.parts) == 4
