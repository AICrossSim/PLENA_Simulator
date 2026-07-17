from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from compiler.aten.cost_emitter import (
    CostTrace,
    MemoryEvent,
    ScheduleAffineLoad,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
)
from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    compile_native_decoder_cost_trace,
)
from compiler.aten.isa_builder import DmaTransfer
from compiler.aten.model_extract import ModelConfig

from analytic_models.performance.rtl_opcode_timing import (
    ComputeFormat,
    ComputePrecisionConfig,
    FpFormat,
    RtlOpcodeTimingCalibration,
    TimingHardware,
    aggregate_compute_work,
)
from analytic_models.performance.compiler_cost_model import (
    _actual_dma_service_provider,
)
from analytic_models.performance.scheduled_shadow import evaluate_scheduled_shadow


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = RtlOpcodeTimingCalibration.load(
    ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v1.json"
)
QWEN3_32B_REFERENCE = json.loads(
    (
        ROOT
        / "analytic_models/performance/calibration/rtl_v1_qwen3_32b_resource_work_reference.json"
    ).read_text()
)
HARDWARE = TimingHardware(mlen=16, blen=4, vlen=16, hlen=8, broadcast_amount=1)
MXFP = ComputePrecisionConfig(
    weight=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    activation=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    kv=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    matrix_internal_fp=FpFormat(8, 7),
    vector_internal_fp=FpFormat(8, 7),
    scalar_fp=FpFormat(8, 7),
    integer_bits=32,
)
MXINT = ComputePrecisionConfig(
    weight=ComputeFormat("mxint", 4, block=64),
    activation=ComputeFormat("mxint", 4, block=64),
    kv=ComputeFormat("mxint", 4, block=64),
    matrix_internal_fp=FpFormat(5, 6),
    vector_internal_fp=FpFormat(5, 6),
    scalar_fp=FpFormat(5, 6),
    integer_bits=32,
)


def _trace(*instructions: ScheduleInstruction, memory_events=()) -> CostTrace:
    counts = Counter(instruction.opcode for instruction in instructions)
    return CostTrace(
        dynamic_opcodes=counts,
        memory_events=list(memory_events),
        schedule=ScheduleSequence(tuple(instructions)),
    )


def _schedule(trace: CostTrace, *, dma_cycles=None):
    return evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=dma_cycles,
        hbm_fidelity="ramulator_observed",
        retain_events=True,
    )


def _set_gp(index: int, value: int) -> ScheduleInstruction:
    return ScheduleInstruction("S_ADDI_INT", (f"gp{index}", "gp0", str(value)))


def test_python_timing_matches_artifact_formulas_for_mxint_and_mxfp() -> None:
    mxfp_mm = CALIBRATION.estimate("M_MM", HARDWARE, MXFP)
    mxfp_wo = CALIBRATION.estimate("M_MM_WO", HARDWARE, MXFP)
    mxint_mm = CALIBRATION.estimate("M_MM", HARDWARE, MXINT)
    mxint_wo = CALIBRATION.estimate("M_MM_WO", HARDWARE, MXINT)

    assert mxfp_mm and mxfp_mm.resource_cycles == HARDWARE.blen + 4
    assert mxfp_wo and mxfp_wo.resource_cycles == 2 * HARDWARE.blen + 13
    assert mxint_mm and mxint_mm.resource_cycles == 2 * HARDWARE.blen + 8
    assert mxint_wo and mxint_wo.resource_cycles == HARDWARE.blen + 2
    assert CALIBRATION.estimate("V_RED_SUM", HARDWARE, MXFP).resource_cycles == 40
    assert CALIBRATION.estimate("V_RED_MAX", HARDWARE, MXFP).resource_cycles == 15


def test_resource_work_is_count_times_backend_occupancy() -> None:
    counts = {"M_MM": 3, "M_MM_WO": 2, "V_ADD_VV": 5, "H_PREFETCH_M": 7}
    work = aggregate_compute_work(
        counts,
        calibration=CALIBRATION,
        hardware=HARDWARE,
        precision=MXFP,
        clock_period_ps=1000,
        opcode_category=lambda opcode: "matrix" if opcode.startswith("M_") else "vector",
    )

    assert work.resource_work_cycles == 3 * 8 + 2 * 21 + 5 * 12
    assert work.latency_ns == work.resource_work_cycles
    assert work.validation["unsupported_opcode_counts"] == {"M_MM_WO": 2}


def test_observed_dma_trace_replay_validates_order_and_consumption() -> None:
    events = [
        {
            "opcode": "H_PREFETCH_M",
            "start_cycle": 7,
            "completion_cycle": 19,
        },
        {
            "opcode": "H_STORE_V",
            "start_cycle": 25,
            "completion_cycle": 31,
        },
    ]
    provider = _actual_dma_service_provider(events)
    assert provider(ScheduleInstruction("H_PREFETCH_M"), 0) == 12
    assert provider(ScheduleInstruction("H_STORE_V"), 1) == 6
    provider.assert_consumed()

    mismatch = _actual_dma_service_provider(events)
    with pytest.raises(ValueError, match="opcode order differs"):
        mismatch(ScheduleInstruction("H_PREFETCH_V"), 0)

    incomplete = _actual_dma_service_provider(events)
    incomplete(ScheduleInstruction("H_PREFETCH_M"), 0)
    with pytest.raises(ValueError, match="consumed 1/2"):
        incomplete.assert_consumed()


def test_observed_dma_repeat_is_replayed_without_provider_fast_forward() -> None:
    count = 20
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        element_base=0,
        scale_base=4096,
        dim=16,
        amount=1,
        stride=16,
    )
    instruction = ScheduleInstruction(
        "H_PREFETCH_V",
        ("gp1", "a0", "gp0", "gp0", "0"),
        memory_stream_index=0,
    )
    trace = CostTrace(
        dynamic_opcodes=Counter({"H_PREFETCH_V": count}),
        memory_events=[MemoryEvent("test", transfer, count, stream_index=0)],
        schedule=ScheduleSequence(
            (ScheduleRepeat(count, ScheduleSequence((instruction,)), name="dma"),)
        ),
    )
    services = iter(range(1, count + 1))
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=lambda _instruction, _sequence: next(services),
        hbm_fidelity="ramulator_observed",
        max_expanded_instructions=count + 1,
    )

    assert result.status == "complete"
    assert result.validation["expanded_instruction_count"] == count
    assert result.validation["repeat_fast_forwards"] == 0


def test_stateful_v4_dma_provider_fast_forward_matches_literal_replay() -> None:
    class PeriodicProvider:
        supports_exact_fast_forward = True

        def __init__(self, count: int) -> None:
            self.count = count
            self.position = 0

        def __call__(self, _instruction, _sequence: int) -> int:
            if self.position >= self.count:
                raise ValueError("provider over-consumed")
            value = (3, 5)[self.position % 2]
            self.position += 1
            return value

        def snapshot_state(self):
            return ((0, self.position, self.position % 2),)

        def advance_stream_counts(self, counts):
            self.position += int(counts.get(0, 0))
            if self.position > self.count:
                raise ValueError("provider fast-forwarded beyond stream")

        def assert_consumed(self):
            assert self.position == self.count

    count = 1_000
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        element_base=0,
        scale_base=4096,
        dim=16,
        amount=1,
        stride=16,
    )
    instruction = ScheduleInstruction(
        "H_PREFETCH_V",
        ("gp1", "a0", "gp0", "gp0", "0"),
        memory_stream_index=0,
    )
    trace = CostTrace(
        dynamic_opcodes=Counter({"H_PREFETCH_V": count}),
        memory_events=[MemoryEvent("test", transfer, count, stream_index=0)],
        schedule=ScheduleSequence(
            (ScheduleRepeat(count, ScheduleSequence((instruction,)), name="dma"),)
        ),
    )
    literal_provider = PeriodicProvider(count)
    literal = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=literal_provider,
        hbm_fidelity="post_hoc_v4",
        retain_events=True,
        max_expanded_instructions=count + 1,
    )
    literal_provider.assert_consumed()

    compressed_provider = PeriodicProvider(count)
    compressed = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=compressed_provider,
        hbm_fidelity="post_hoc_v4",
        max_expanded_instructions=32,
    )
    compressed_provider.assert_consumed()

    assert compressed.status == literal.status == "complete"
    assert compressed.makespan_cycles == literal.makespan_cycles
    assert compressed.resource_work_cycles == literal.resource_work_cycles
    assert compressed.stall_cycles_by_reason == literal.stall_cycles_by_reason
    assert compressed.dma_occurrences == literal.dma_occurrences
    assert len(compressed.dma_occurrences) == count
    assert compressed.validation["repeat_fast_forwards"] >= 1


def test_observed_stream_intervals_support_exact_repeat_fast_forward() -> None:
    count = 1_000
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        element_base=0,
        scale_base=4096,
        dim=16,
        amount=1,
        stride=16,
    )
    instruction = ScheduleInstruction(
        "H_PREFETCH_V",
        ("gp1", "a0", "gp0", "gp0", "0"),
        memory_stream_index=0,
    )
    trace = CostTrace(
        dynamic_opcodes=Counter({"H_PREFETCH_V": count}),
        memory_events=[MemoryEvent("test", transfer, count, stream_index=0)],
        schedule=ScheduleSequence(
            (ScheduleRepeat(count, ScheduleSequence((instruction,)), name="dma"),)
        ),
    )
    intervals = [3, 5] * (count // 2)
    provider = _actual_dma_service_provider({0: intervals})
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=provider,
        hbm_fidelity="ramulator_observed",
        max_expanded_instructions=32,
    )
    provider.assert_consumed()

    assert result.status == "complete"
    assert result.validation["repeat_fast_forwards"] >= 1
    assert len(result.dma_occurrences) == count
    assert [item.completion_cycle - item.start_cycle for item in result.dma_occurrences] == intervals


def test_qwen3_32b_resource_work_matches_transactional_reference() -> None:
    """Keep CostEmitter instruction counts and rtl-v1 work in lockstep with Rust."""
    model_values = QWEN3_32B_REFERENCE["model"]
    hardware_values = QWEN3_32B_REFERENCE["hardware"]
    model = ModelConfig(
        hidden_size=model_values["hidden_size"],
        inter_dim=model_values["inter_dim"],
        num_heads=model_values["num_heads"],
        num_kv_heads=model_values["num_kv_heads"],
        head_dim=model_values["head_dim"],
        eps=1e-6,
        rope_theta=model_values["rope_theta"],
        vocab_size=151_936,
        model_type="qwen3",
    )
    compiler_hardware = CompilerCostHardware(
        mlen=hardware_values["mlen"],
        blen=hardware_values["blen"],
        vlen=hardware_values["vlen"],
        hlen=hardware_values["hlen"],
        broadcast_amount=hardware_values["logical_broadcast_amount"],
        mram_tile_capacity=hardware_values["mram_tile_capacity"],
        hbm_m_prefetch_amount=hardware_values["hbm_m_prefetch_amount"],
        hbm_v_prefetch_amount=hardware_values["hbm_v_prefetch_amount"],
        hbm_v_writeback_amount=hardware_values["hbm_v_writeback_amount"],
        hbm_channels=hardware_values["hbm_channels"],
    )
    trace = compile_native_decoder_cost_trace(
        model,
        compiler_hardware,
        seq_len=model_values["seq_len"],
        batch_size=model_values["batch_size"],
        num_layers=model_values["num_layers"],
        use_cache=False,
    )
    actual_non_hbm = {
        opcode: int(count)
        for opcode, count in trace.dynamic_opcodes.items()
        if not opcode.startswith("H_")
    }

    assert trace.dynamic_instruction_count == QWEN3_32B_REFERENCE[
        "dynamic_instruction_count"
    ]
    assert actual_non_hbm == QWEN3_32B_REFERENCE["non_hbm_opcode_counts"]

    work = aggregate_compute_work(
        trace.dynamic_opcodes,
        calibration=CALIBRATION,
        hardware=TimingHardware(
            hardware_values["mlen"],
            hardware_values["blen"],
            hardware_values["vlen"],
            hardware_values["hlen"],
            hardware_values["physical_broadcast_amount"],
        ),
        precision=MXFP,
        clock_period_ps=hardware_values["clock_period_ps"],
        opcode_category=lambda opcode: (
            "control"
            if opcode.startswith("C_")
            else "matrix_compute"
            if opcode.startswith("M_")
            else "vector_compute"
            if opcode.startswith("V_")
            else "scalar_compute"
        ),
    )
    expected_resources = QWEN3_32B_REFERENCE["resource_work_cycles"]
    assert work.resource_work_cycles == QWEN3_32B_REFERENCE[
        "non_hbm_resource_work_cycles"
    ]
    assert work.category_cycles == {
        "control": expected_resources["control_frontend"],
        "matrix_compute": (
            expected_resources["matrix_compute"]
            + expected_resources["matrix_writeout"]
        ),
        "scalar_compute": expected_resources["scalar_pipeline"],
        "vector_compute": expected_resources["vector_pipeline"],
    }
    assert work.validation["status"] == QWEN3_32B_REFERENCE[
        "rtl_validation_status"
    ]
    assert work.validation["unsupported_opcode_counts"] == QWEN3_32B_REFERENCE[
        "unsupported_opcode_counts"
    ]


def test_matrix_compute_writeout_and_consumer_follow_dependencies() -> None:
    result = _schedule(
        _trace(
            _set_gp(1, 0),
            _set_gp(2, 4096),
            _set_gp(3, 8192),
            _set_gp(4, 12288),
            ScheduleInstruction("M_MM", ("gp1", "gp2")),
            ScheduleInstruction("M_MM_WO", ("gp3", "gp0", "0")),
            ScheduleInstruction("V_ADD_VV", ("gp4", "gp3", "gp3", "0")),
        )
    )
    mm, writeout, consumer = result.events[-3:]
    assert writeout.start_cycle == mm.completion_cycle
    assert writeout.stall_reason == "matrix_result_not_ready"
    assert consumer.start_cycle == writeout.result_ready_cycle + 1
    assert consumer.stall_reason == "vector_sram_operand_not_ready"


def test_vector_reduction_blocks_all_scalar_fp_operations() -> None:
    result = _schedule(
        _trace(
            _set_gp(1, 1024),
            _set_gp(2, 2048),
            ScheduleInstruction("V_RED_SUM", ("f1", "gp1", "0")),
            ScheduleInstruction("S_LD_FP", ("f2", "gp2", "0")),
        )
    )
    reduction, scalar_load = result.events[-2:]
    assert scalar_load.start_cycle == reduction.result_ready_cycle + 1
    assert scalar_load.recovery_cycles == 1
    assert scalar_load.stall_reason == "vector_reduction_result_not_ready"


def test_scalar_sfu_blocks_vector_scalar_broadcast() -> None:
    result = _schedule(
        _trace(
            _set_gp(1, 1024),
            _set_gp(2, 2048),
            ScheduleInstruction("S_EXP_FP", ("f1", "f2")),
            ScheduleInstruction("V_ADD_VF", ("gp1", "gp2", "f3", "0")),
        )
    )
    scalar, vector = result.events[-2:]
    assert vector.start_cycle == scalar.completion_cycle + 1
    assert vector.stall_reason == "scalar_fp_compute_in_progress"


def test_hbm_vector_prefetch_blocks_vector_pipeline_with_recovery_cycle() -> None:
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        element_base=0,
        scale_base=4096,
        dim=16,
        amount=1,
        stride=16,
    )
    memory_event = MemoryEvent("test", transfer, 1, stream_index=0)
    result = _schedule(
        _trace(
            _set_gp(1, 1024),
            _set_gp(2, 2048),
            ScheduleInstruction(
                "H_PREFETCH_V",
                ("gp1", "a0", "gp0", "gp0", "0"),
                memory_stream_index=0,
            ),
            ScheduleInstruction("V_ADD_VV", ("gp2", "gp1", "gp1", "0")),
            memory_events=(memory_event,),
        ),
        dma_cycles=lambda _instruction, _sequence: 40,
    )
    dma, vector = result.events[-2:]
    assert vector.start_cycle == dma.completion_cycle + 1
    assert vector.recovery_cycles == 1
    assert vector.stall_reason in {
        "vector_prefetch_in_progress",
        "vector_sram_operand_not_ready",
    }


def test_mixed_vector_latencies_retire_in_issue_order() -> None:
    result = _schedule(
        _trace(
            _set_gp(1, 1024),
            _set_gp(2, 2048),
            _set_gp(3, 3072),
            _set_gp(4, 4096),
            _set_gp(5, 5120),
            _set_gp(6, 6144),
            ScheduleInstruction("V_ADD_VV", ("gp1", "gp2", "gp3", "0")),
            ScheduleInstruction("V_MUL_VV", ("gp4", "gp5", "gp6", "0")),
        )
    )
    add, multiply = result.events[-2:]
    assert multiply.start_cycle == add.result_ready_cycle + 1
    assert multiply.stall_reason == "vector_mixed_latency_in_order"


def test_python_scheduler_matches_rust_differential_cycles() -> None:
    hardware = TimingHardware(64, 4, 64, 8, 1)
    artifact = json.loads(
        (ROOT / "Workspace/rtl_v1_latency_validation/scheduler_differential_traces.json").read_text()
    )
    expected = {case["name"]: case["events"] for case in artifact["cases"]}
    cases = {
        "matrix_compute_writeout_row_consumer": (
            {
                1: 0,
                2: 4096,
                3: 8192,
                4: 12288,
                5: 16384,
            },
            (
                ScheduleInstruction("M_MM", ("gp1", "gp2")),
                ScheduleInstruction("M_MM_WO", ("gp3", "gp0", "0")),
                ScheduleInstruction("V_ADD_VV", ("gp4", "gp3", "gp5", "0")),
            ),
            None,
            (),
        ),
        "vector_reduction_to_scalar_fp": (
            {1: 0, 2: 4096},
            (
                ScheduleInstruction("V_RED_SUM", ("f1", "gp1", "0")),
                ScheduleInstruction("S_LD_FP", ("f2", "gp2", "0")),
            ),
            None,
            (),
        ),
        "scalar_sfu_to_vector_broadcast": (
            {3: 64, 4: 128},
            (
                ScheduleInstruction("S_EXP_FP", ("f1", "f2")),
                ScheduleInstruction("V_MUL_VF", ("gp3", "gp4", "f5", "0")),
            ),
            None,
            (),
        ),
        "hbm_vector_prefetch_to_vector_compute": (
            {1: 0, 2: 4096, 3: 8192, 4: 12288},
            (
                ScheduleInstruction(
                    "H_PREFETCH_V",
                    ("gp1", "a0", "gp0", "gp0", "0"),
                    memory_stream_index=0,
                ),
                ScheduleInstruction("V_ADD_VV", ("gp4", "gp2", "gp3", "0")),
            ),
            lambda _instruction, _sequence: 20,
            (
                MemoryEvent(
                    "test",
                    DmaTransfer(
                        opcode="H_PREFETCH_V",
                        direction="read",
                        precision="activation",
                        element_base=0,
                        scale_base=4096,
                        dim=64,
                        amount=1,
                        stride=64,
                    ),
                    1,
                    stream_index=0,
                ),
            ),
        ),
        "stall_recovery_reissue": (
            {1: 0, 2: 128, 3: 4096, 4: 8192},
            (
                ScheduleInstruction("V_ADD_VV", ("gp1", "gp2", "gp3", "0")),
                ScheduleInstruction("V_ADD_VV", ("gp2", "gp1", "gp4", "0")),
            ),
            None,
            (),
        ),
        "mixed_vector_latency_in_order": (
            {1: 0, 2: 4096, 3: 8192, 4: 128, 5: 12288, 6: 16384},
            (
                ScheduleInstruction("V_ADD_VV", ("gp1", "gp2", "gp3", "0")),
                ScheduleInstruction("V_MUL_VV", ("gp4", "gp5", "gp6", "0")),
            ),
            None,
            (),
        ),
    }

    fields = (
        "issue_cycle",
        "accepted_cycle",
        "recovery_cycles",
        "start_cycle",
        "result_ready_cycle",
        "completion_cycle",
        "resource",
        "stall_reason",
        "dependency",
    )
    for name, (initial_gp, instructions, dma_provider, memory_events) in cases.items():
        trace = _trace(*instructions, memory_events=memory_events)
        result = evaluate_scheduled_shadow(
            trace,
            hardware=hardware,
            precision=MXFP,
            calibration=CALIBRATION,
            hbm_service_cycles=dma_provider,
            hbm_fidelity="ramulator_observed",
            retain_events=True,
            initial_gp=initial_gp,
        )
        assert result.status == "complete", (name, result.reason)
        assert len(result.events) == len(expected[name])
        for actual_event, expected_event in zip(result.events, expected[name], strict=True):
            for field in fields:
                assert getattr(actual_event, field) == expected_event.get(field), (
                    name,
                    field,
                    actual_event,
                    expected_event,
                )

        # Reconstruct the transactional profiler's mutually exclusive
        # critical-path partition from the Rust differential events.
        expected_critical = Counter()
        by_sequence = {event["sequence"]: event for event in expected[name]}
        issue_cursor = 0
        for event in sorted(
            expected[name], key=lambda item: (item["accepted_cycle"], item["sequence"])
        ):
            blocked = event["accepted_cycle"] - issue_cursor
            if blocked:
                dependency = by_sequence.get(event.get("dependency"))
                expected_critical[
                    "control_frontend"
                    if dependency is None
                    else dependency["resource"]
                ] += blocked
            expected_critical["control_frontend"] += 1
            issue_cursor = event["accepted_cycle"] + 1
        makespan = max(event["completion_cycle"] for event in expected[name])
        drain = makespan - issue_cursor
        if drain:
            owner = max(
                expected[name],
                key=lambda item: (item["completion_cycle"], item["sequence"]),
            )
            expected_critical[owner["resource"]] += drain
        assert result.critical_path_cycles == dict(expected_critical), name
        assert sum(result.critical_path_cycles.values()) == result.makespan_cycles
        assert result.validation["critical_path_accounting"] == "exact"


def test_large_repeat_uses_exact_normalized_state_fast_forward() -> None:
    count = 10_000_000
    trace = CostTrace(
        dynamic_opcodes=Counter({"S_ADDI_INT": count}),
        schedule=ScheduleSequence(
            (
                ScheduleRepeat(
                    count=count,
                    body=ScheduleSequence(
                        (ScheduleInstruction("S_ADDI_INT", ("gp1", "gp1", "4")),)
                    ),
                    name="affine_integer_loop",
                ),
            )
        ),
    )
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
        max_expanded_instructions=16,
    )

    assert result.status == "complete"
    assert result.makespan_cycles == count
    assert result.resource_work_cycles == {"scalar_pipeline": count}
    assert result.critical_path_cycles == {"control_frontend": count}
    assert result.validation["expanded_instruction_count"] == 2
    assert result.validation["repeat_fast_forwards"] == 1
    assert result.validation["fast_forwarded_iterations"] == count - 2
    assert result.validation["fast_forwarded_dynamic_instructions"] == count - 2


def test_fast_forward_matches_fully_expanded_vector_address_loop() -> None:
    count = 100
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp1", "gp2", "gp3", "0")),
            ScheduleInstruction("S_ADDI_INT", ("gp1", "gp1", str(HARDWARE.vlen))),
            ScheduleInstruction("S_ADDI_INT", ("gp2", "gp2", str(HARDWARE.vlen))),
            ScheduleInstruction("S_ADDI_INT", ("gp3", "gp3", str(HARDWARE.vlen))),
        )
    )
    trace = CostTrace(
        dynamic_opcodes=Counter({"V_ADD_VV": count, "S_ADDI_INT": 3 * count}),
        schedule=ScheduleSequence((ScheduleRepeat(count, body, name="vector_rows"),)),
    )
    kwargs = dict(
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
        initial_gp={1: 0, 2: 4096, 3: 8192},
    )
    expanded = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=1_000, **kwargs
    )
    compressed = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=128, **kwargs
    )

    assert expanded.status == compressed.status == "complete"
    assert compressed.makespan_cycles == expanded.makespan_cycles
    assert compressed.stall_cycles_by_reason == expanded.stall_cycles_by_reason
    assert compressed.resource_work_cycles == expanded.resource_work_cycles
    assert compressed.validation["status"] == expanded.validation["status"]
    assert compressed.validation["repeat_fast_forwards"] == 1


def test_affine_fast_forward_does_not_cross_legalization_boundary() -> None:
    count = 100
    affine = ScheduleAffineLoad(
        key="threshold_stream",
        register="gp1",
        start=0,
        step=128,
        period=count,
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (
                ScheduleRepeat(
                    count,
                    ScheduleSequence((affine,)),
                    name="threshold_crossing",
                ),
            )
        )
    )
    kwargs = dict(
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
    )
    expanded = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=1_000, **kwargs
    )
    compressed = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=8, **kwargs
    )

    assert expanded.status == compressed.status == "complete"
    assert compressed.makespan_cycles == expanded.makespan_cycles
    assert compressed.resource_work_cycles == expanded.resource_work_cycles
    assert compressed.validation["repeat_fast_forwards"] >= 1


def test_affine_fast_forward_recognizes_complete_lui_addi_period() -> None:
    count = 1_024
    affine = ScheduleAffineLoad(
        key="periodic_large_address",
        register="gp1",
        start=8_192,
        step=128,
        period=count,
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (
                ScheduleRepeat(
                    count,
                    ScheduleSequence((affine,)),
                    name="periodic_large_address",
                ),
            )
        )
    )
    kwargs = dict(
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
    )
    literal = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=4_096, **kwargs
    )
    compressed = evaluate_scheduled_shadow(
        trace, max_expanded_instructions=8, **kwargs
    )

    assert literal.status == compressed.status == "complete"
    assert compressed.makespan_cycles == literal.makespan_cycles
    assert compressed.resource_work_cycles == literal.resource_work_cycles
    assert compressed.validation["expanded_instruction_count"] < 200
    assert compressed.validation["fast_forwarded_iterations"] > 900


def test_large_affine_repeat_fast_forwards_complete_4k_period() -> None:
    """The production 512-byte row stride must not recurse per 4 KiB page."""

    count = 8_192
    affine = ScheduleAffineLoad(
        key="production_row_address",
        register="gp1",
        start=8_650_752,
        step=512,
        period=count,
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (
                ScheduleRepeat(
                    count,
                    ScheduleSequence((affine,)),
                    name="production_rows",
                ),
            )
        )
    )
    kwargs = dict(
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
    )
    literal = evaluate_scheduled_shadow(
        trace,
        retain_events=True,
        max_expanded_instructions=30_000,
        **kwargs,
    )
    compressed = evaluate_scheduled_shadow(
        trace,
        max_expanded_instructions=64,
        **kwargs,
    )

    assert literal.status == compressed.status == "complete"
    assert compressed.makespan_cycles == literal.makespan_cycles
    assert compressed.resource_work_cycles == literal.resource_work_cycles
    assert compressed.stall_cycles_by_reason == literal.stall_cycles_by_reason
    assert compressed.validation["expanded_instruction_count"] < 64
    assert compressed.validation["fast_forwarded_iterations"] > 8_000


def test_compressed_qwen_kernels_match_fully_expanded_schedule() -> None:
    """Projection, FFN, RMSNorm, and attention compression is cycle exact."""
    model = ModelConfig(
        hidden_size=192,
        inter_dim=384,
        num_heads=12,
        num_kv_heads=2,
        head_dim=16,
        eps=1e-6,
        rope_theta=10_000.0,
        vocab_size=1_024,
        model_type="qwen3",
    )
    compiler_hardware = CompilerCostHardware(
        mlen=128,
        blen=64,
        vlen=128,
        hlen=16,
        broadcast_amount=6,
        # One resident tile forces the ordered unrolled FFN schedule, which
        # exercises all compressed kernel builders in a tractable trace.
        mram_tile_capacity=1,
        hbm_m_prefetch_amount=128,
        hbm_v_prefetch_amount=128,
        hbm_v_writeback_amount=128,
        hbm_channels=32,
    )
    trace = compile_native_decoder_cost_trace(
        model,
        compiler_hardware,
        seq_len=8,
        batch_size=1,
        num_layers=1,
        use_cache=False,
    )
    hardware = TimingHardware(128, 64, 128, 16, 6)
    dma_cycles = lambda _instruction, _sequence: 7
    kwargs = dict(
        hardware=hardware,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=dma_cycles,
        hbm_fidelity="post_hoc_v3",
    )
    literal = evaluate_scheduled_shadow(
        trace,
        retain_events=True,
        max_expanded_instructions=1_000_000,
        **kwargs,
    )
    compressed = evaluate_scheduled_shadow(
        trace,
        retain_events=False,
        max_expanded_instructions=10_000,
        **kwargs,
    )

    assert trace.dynamic_instruction_count > 50_000
    assert literal.status == compressed.status == "complete"
    assert compressed.makespan_cycles == literal.makespan_cycles
    assert compressed.resource_work_cycles == literal.resource_work_cycles
    assert compressed.stall_cycles_by_reason == literal.stall_cycles_by_reason
    assert compressed.validation["status"] == literal.validation["status"]
    assert len(literal.events) == trace.dynamic_instruction_count
    assert compressed.validation["expanded_instruction_count"] < 10_000


def test_unrelated_matrix_dma_does_not_pollute_vector_repeat_state() -> None:
    count = 100
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        precision="matrix",
        element_base=0,
        scale_base=4096,
        dim=HARDWARE.mlen,
        amount=HARDWARE.mlen,
        stride=HARDWARE.mlen,
    )
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp2", "gp2", "gp3", "0")),
            ScheduleInstruction(
                "S_ADDI_INT", ("gp2", "gp2", str(HARDWARE.vlen))
            ),
            ScheduleInstruction(
                "S_ADDI_INT", ("gp3", "gp3", str(HARDWARE.vlen))
            ),
        )
    )
    trace = CostTrace(
        dynamic_opcodes=Counter(
            {
                "H_PREFETCH_M": 1,
                "V_ADD_VV": count,
                "S_ADDI_INT": 2 * count,
            }
        ),
        memory_events=[MemoryEvent("test", transfer, 1, stream_index=0)],
        schedule=ScheduleSequence(
            (
                ScheduleInstruction(
                    "H_PREFETCH_M",
                    ("gp1", "a0", "gp0", "gp0", "0"),
                    memory_stream_index=0,
                ),
                ScheduleRepeat(count, body, name="vector_after_matrix_dma"),
            )
        ),
    )
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_service_cycles=lambda _instruction, _sequence: 10_000,
        hbm_fidelity="post_hoc_v3",
        max_expanded_instructions=128,
        initial_gp={1: 0, 2: 1 << 20, 3: 2 << 20},
    )

    assert result.status == "complete"
    assert result.makespan_cycles == 10_000
    assert result.validation["expanded_instruction_count"] < 50


def test_exact_repeat_effect_cache_matches_literal_replay() -> None:
    count = 20
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp1", "gp2", "gp3", "0")),
            ScheduleInstruction(
                "S_ADDI_INT", ("gp1", "gp1", str(HARDWARE.vlen))
            ),
            ScheduleInstruction(
                "S_ADDI_INT", ("gp2", "gp2", str(HARDWARE.vlen))
            ),
            ScheduleInstruction(
                "S_ADDI_INT", ("gp3", "gp3", str(HARDWARE.vlen))
            ),
        )
    )
    repeated = ScheduleRepeat(count, body, name="cacheable_rows")
    children = []
    for _ in range(3):
        children.extend(
            (
                _set_gp(1, 1 << 20),
                _set_gp(2, 2 << 20),
                _set_gp(3, 3 << 20),
                repeated,
            )
        )
    trace = CostTrace(schedule=ScheduleSequence(tuple(children)))
    kwargs = dict(
        hardware=HARDWARE,
        precision=MXFP,
        calibration=CALIBRATION,
        hbm_fidelity="post_hoc_v3",
        max_expanded_instructions=1_000,
    )
    literal = evaluate_scheduled_shadow(trace, retain_events=True, **kwargs)
    cached = evaluate_scheduled_shadow(trace, retain_events=False, **kwargs)

    assert literal.status == cached.status == "complete"
    assert cached.makespan_cycles == literal.makespan_cycles
    assert cached.stall_cycles_by_reason == literal.stall_cycles_by_reason
    assert cached.resource_work_cycles == literal.resource_work_cycles
    assert cached.validation["status_counts"] == literal.validation["status_counts"]
    assert cached.validation["repeat_cache_hits"] >= 1
