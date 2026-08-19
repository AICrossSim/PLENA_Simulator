from dataclasses import replace

from analytic_models.performance.nemotron3_dse import (
    HardwareDesign,
    Nemotron3DseModel,
    PersistentStateCacheModel,
    ProjectionBankModel,
    ProjectionLayout,
    ProjectionWriteBufferModel,
    StateCachePolicy,
    sweep_designs,
)
from analytic_models.performance.nemotron3_workload import (
    InferencePhase,
    Precision,
    ScanStrategy,
    WorkloadScenario,
)
from transactional_emulator.testbench.model_configs.loader import load_model_config


def _arch():
    return load_model_config("nemotron3_nano_30b_a3b").arch


def test_bc_broadcast_removes_eight_redundant_group_reads() -> None:
    base = HardwareDesign(projection_layout=ProjectionLayout.ROW_MAJOR)
    without_broadcast = ProjectionBankModel(_arch(), base).simulate_one_token_request_layer()
    with_broadcast = ProjectionBankModel(_arch(), replace(base, bc_broadcast=True)).simulate_one_token_request_layer()

    assert without_broadcast.bc_value_reads == 8 * with_broadcast.bc_value_reads
    assert with_broadcast.bc_value_reads == 2 * 8 * 128


def test_skewed_layout_is_conflict_free_for_the_candidate_packet_shape() -> None:
    base = HardwareDesign(bc_broadcast=True)
    row = ProjectionBankModel(_arch(), base).simulate_one_token_request_layer()
    skewed = ProjectionBankModel(
        _arch(), replace(base, projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED)
    ).simulate_one_token_request_layer()

    assert row.total.stall_cycles > 0
    assert skewed.total.stall_cycles == 0
    assert skewed.total.service_cycles < row.total.service_cycles


def test_projection_fifo_models_matrix_result_burst_backpressure() -> None:
    assert HardwareDesign().projection_fifo_values == 64
    normal = ProjectionWriteBufferModel(HardwareDesign()).simulate(values=10304, producer_cycles=7000)
    constrained = ProjectionWriteBufferModel(
        HardwareDesign(
            matrix_result_burst_values=64,
            projection_buffer_write_values_per_cycle=8,
            projection_fifo_values=64,
        )
    ).simulate(values=10304, producer_cycles=100)

    assert normal.fifo_stall_cycles == 0
    assert normal.completion_cycles == 7000
    assert constrained.fifo_stall_cycles > 0
    assert constrained.completion_cycles > constrained.producer_cycles


def test_projection_bypass_spills_only_gate_when_consumer_is_ready() -> None:
    ready = ProjectionWriteBufferModel(HardwareDesign()).simulate(
        values=10304,
        producer_cycles=7000,
        values_per_token=10304,
        forced_spill_values_per_token=4096,
    )
    delayed = ProjectionWriteBufferModel(
        HardwareDesign(projection_consumer_start_cycles=10000)
    ).simulate(
        values=10304,
        producer_cycles=7000,
        values_per_token=10304,
        forced_spill_values_per_token=4096,
    )
    buffered = ProjectionWriteBufferModel(
        HardwareDesign(projection_direct_bypass=False)
    ).simulate(
        values=10304,
        producer_cycles=7000,
        values_per_token=10304,
        forced_spill_values_per_token=4096,
    )

    assert (ready.direct_values, ready.spill_values) == (6208, 4096)
    assert (delayed.direct_values, delayed.spill_values) == (0, 10304)
    assert (buffered.direct_values, buffered.spill_values) == (0, 10304)


def test_partial_lru_thrashes_but_capacity_aware_pinning_retains_hits() -> None:
    capacity = 16 * 1024 * 1024
    lru = PersistentStateCacheModel(_arch(), Precision.FP32, capacity, StateCachePolicy.LRU).simulate(
        batch_size=1, decode_tokens=4
    )
    pinned = PersistentStateCacheModel(_arch(), Precision.FP32, capacity, StateCachePolicy.PINNED).simulate(
        batch_size=1, decode_tokens=4
    )

    assert lru.hit_rate == 0.0
    assert pinned.hit_rate > 0.0
    assert pinned.hbm_read_bytes < lru.hbm_read_bytes


def test_full_state_cache_hits_and_only_flushes_once() -> None:
    cache = PersistentStateCacheModel(_arch(), Precision.FP32, 64 * 1024 * 1024, StateCachePolicy.LRU).simulate(
        batch_size=1, decode_tokens=8
    )

    assert cache.hit_rate == 1.0
    assert cache.hbm_read_bytes == 0
    assert cache.final_flush_bytes == 23 * (2 * 1024 * 1024 + 96 * 1024)
    assert cache.hbm_write_bytes == cache.final_flush_bytes


def test_dse_reports_real_hybrid_breakdown_and_ablation_counters() -> None:
    scenario = WorkloadScenario(
        phase=InferencePhase.DECODE,
        batch_size=1,
        context_length=2048,
        decode_tokens=4,
        include_embedding=False,
        include_lm_head=False,
    )
    model = Nemotron3DseModel(_arch())
    baseline = model.evaluate(scenario, HardwareDesign())
    optimized = model.evaluate(
        scenario,
        HardwareDesign(
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            bc_broadcast=True,
            state_cache_bytes=16 * 1024 * 1024,
            state_cache_policy=StateCachePolicy.PINNED,
        ),
    )

    breakdown = baseline.to_dict(include_stages=False)["metrics"]["cycle_breakdown"]
    assert set(breakdown) >= {"mamba", "moe", "attention"}
    assert optimized.bank_stall_cycles < baseline.bank_stall_cycles
    assert optimized.state_cache.hit_rate > baseline.state_cache.hit_rate
    assert optimized.hbm_read_bytes + optimized.hbm_write_bytes < baseline.hbm_read_bytes + baseline.hbm_write_bytes


def test_prefill_bc_reuse_computes_chunk_cb_once_per_group() -> None:
    scenario = WorkloadScenario(
        phase=InferencePhase.PREFILL,
        sequence_length=128,
        context_length=128,
        scan_strategy=ScanStrategy.CHUNKED_AFFINE,
        include_embedding=False,
        include_lm_head=False,
    )
    model = Nemotron3DseModel(_arch())
    expanded = model.evaluate(scenario, HardwareDesign(bc_broadcast=False))
    reused = model.evaluate(scenario, HardwareDesign(bc_broadcast=True))
    expanded_cb = next(stage for stage in expanded.stages if stage.name == "mamba_chunk_intra_cb")
    reused_cb = next(stage for stage in reused.stages if stage.name == "mamba_chunk_intra_cb")

    assert expanded_cb.effective_macs == 8 * reused_cb.effective_macs


def test_sweep_filters_invalid_cache_policy_combinations() -> None:
    designs = sweep_designs(
        HardwareDesign(),
        layouts=(ProjectionLayout.ROW_MAJOR,),
        broadcasts=(False,),
        cache_sizes=(0, 16 * 1024 * 1024),
        cache_policies=(StateCachePolicy.NONE, StateCachePolicy.LRU, StateCachePolicy.PINNED),
        state_dim_lanes=(8,),
    )
    assert len(designs) == 3
    assert all(design.state_macs_per_cycle == 256 for design in designs)
    assert {(design.state_cache_bytes, design.state_cache_policy) for design in designs} == {
        (0, StateCachePolicy.NONE),
        (16 * 1024 * 1024, StateCachePolicy.LRU),
        (16 * 1024 * 1024, StateCachePolicy.PINNED),
    }


def test_sweep_can_ablate_bypass_and_fifo_capacity() -> None:
    designs = sweep_designs(
        HardwareDesign(),
        layouts=(ProjectionLayout.GROUP_MAJOR_SKEWED,),
        broadcasts=(True,),
        cache_sizes=(0,),
        cache_policies=(StateCachePolicy.NONE,),
        state_dim_lanes=(8,),
        bypasses=(False, True),
        fifo_values=(64, 256),
    )
    assert len(designs) == 4
    assert {(design.projection_direct_bypass, design.projection_fifo_values) for design in designs} == {
        (False, 64),
        (False, 256),
        (True, 64),
        (True, 256),
    }
