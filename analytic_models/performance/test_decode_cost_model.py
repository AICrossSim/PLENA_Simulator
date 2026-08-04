"""Focused tests for compiler-derived decode tracing and stage composition."""

from __future__ import annotations

import hashlib

import pytest

from analytic_models.performance.decode_cost_model import (
    DecodeCostModel,
    validate_packed_q1_execution_trace,
)
from analytic_models.performance.decode_stage_validation import (
    summarize_stage_validation,
)
from analytic_models.roofline.asm_profiler import profile_execution_trace
from compiler.aten.execution_trace import (
    ExecutionTrace,
    TensorTraceMetadata,
    build_execution_trace,
    build_request_memory_trace,
)
from compiler.aten.plena import PlenaCompiler


def test_nested_loops_are_counted_algebraically_with_dma_metadata() -> None:
    assembly = """\
; Normalize (rms) X
S_ADDI_INT gp1, gp0, 1024
C_SET_ADDR_REG a0, gp0, gp1
C_LOOP_START gp2, 3
V_ADD_VV gp3, gp4, gp5, 0
C_LOOP_START gp6, 2
H_PREFETCH_M gp7, gp8, a0, 1, 1
C_LOOP_END gp6
C_LOOP_END gp2
"""
    trace = build_execution_trace(
        assembly,
        mlen=64,
        blen=4,
        hlen=16,
        tensors=(
            TensorTraceMetadata(
                name="K_cache",
                hbm_address=1024,
                precision_mode="key",
                element_bits=4,
                block_size=8,
                scale_bits=8,
            ),
        ),
    )

    assert trace.opcode_histogram == {
        "C_LOOP_END": 9,
        "C_LOOP_START": 4,
        "C_SET_ADDR_REG": 1,
        "H_PREFETCH_M": 6,
        "S_ADDI_INT": 1,
        "V_ADD_VV": 3,
    }
    dma = next(entry for entry in trace.entries if entry.opcode == "H_PREFETCH_M")
    assert dma.dynamic_count == 6
    assert dma.precision_mode == "key"
    assert dma.tensor == "K_cache"
    assert dma.tile_shape == dma.dma_shape == (64, 64)
    assert dma.dma_bytes == 2560
    assert dma.total_dma_bytes == 15360
    assert ExecutionTrace.from_dict(trace.to_dict()) == trace


def test_compiler_emits_trace_bound_to_final_assembly_and_tensor_layout() -> None:
    compiler = PlenaCompiler(
        mlen=64,
        blen=4,
        hbm_element_width=4,
        hbm_block_size=8,
        hbm_scale_width=8,
    )
    source = compiler.input(
        "activation",
        (1, 64),
        physical_shape=(1, 64),
        precision_role="activation",
    )
    compiler.load_batch(source)
    compiler.free_input(source)

    artifact = compiler.compile_with_trace()
    assert artifact.trace.assembly_sha256 == hashlib.sha256(
        artifact.assembly.encode("utf-8")
    ).hexdigest()
    dma = next(
        entry
        for entry in artifact.execution_trace.entries
        if entry.opcode == "H_PREFETCH_V"
    )
    assert dma.tensor == "activation"
    assert dma.precision_mode == "activation"
    assert dma.dma_shape == (4, 64)
    assert dma.dma_bytes == 160


def test_request_memory_trace_executes_loop_carried_dma_addresses() -> None:
    assembly = """\
; Load_Batch activation -> VRAM
S_ADDI_INT gp1, gp0, 4096
C_SET_ADDR_REG a0, gp0, gp1
S_ADDI_INT gp2, gp0, 1024
C_SET_SCALE_REG gp2
S_ADDI_INT gp3, gp0, 64
C_SET_STRIDE_REG gp3
S_ADDI_INT gp4, gp0, 0
C_LOOP_START gp5, 3
H_PREFETCH_V gp0, gp4, a0, 1, 0
S_ADDI_INT gp4, gp4, 64
C_LOOP_END gp5
"""
    tensor = TensorTraceMetadata(
        name="activation",
        hbm_address=4096,
        precision_mode="activation",
        element_bits=8,
        block_size=8,
        scale_bits=8,
        physical_shape=(16, 64),
        element_plane_bytes=1024,
        hbm_size=1280,
    )
    trace = build_execution_trace(
        assembly,
        mlen=64,
        blen=4,
        hlen=16,
        vector_prefetch_amount=4,
        tensors=(tensor,),
    )
    sidecar = build_request_memory_trace(
        assembly,
        trace,
        vector_prefetch_amount=4,
        vector_store_amount=4,
        tensors=(tensor,),
    )

    assert len(sidecar.bindings) == 1
    binding = sidecar.bindings[0]
    # The loop's three iterations compress into one affine run whose ordinals
    # still resolve to the exact addresses the loop issues.
    assert [run.repetitions for run in binding.runs] == [3]
    assert [run.address_step_bytes for run in binding.runs] == [64]
    requests = list(binding.iter_requests())
    assert [request.address for request in requests] == [4096, 4160, 4224]
    assert [request.scale_address for request in requests] == [5120, 5128, 5136]
    assert all(request.stride_bytes == 64 for request in requests)
    assert all(request.scale_stride_bytes == 8 for request in requests)
    assert binding.trace_entry_sha256


def test_cost_model_overlaps_within_stage_and_serializes_between_stages() -> None:
    assembly = """\
; Normalize (rms) X
C_LOOP_START gp1, 4
V_ADD_VV gp2, gp3, gp4, 0
H_PREFETCH_V gp5, gp6, a0, 0, 0
C_LOOP_END gp1
; VRAM Matrix Add
V_ADD_VV gp2, gp3, gp4, 0
"""
    trace = build_execution_trace(
        assembly,
        mlen=64,
        blen=4,
        hlen=16,
    )
    model = DecodeCostModel(
        {
            "C_LOOP_START": 1,
            "C_LOOP_END": 1,
            "V_ADD_VV": 3,
            "H_PREFETCH_V": 1,
        },
        memory_bandwidth_bytes_per_cycle=32,
    )
    result = model.evaluate(trace)

    rms, residual = result.stages
    assert (rms.compute_cycles, rms.memory_cycles, rms.cycles) == (21, 36, 36)
    assert (residual.compute_cycles, residual.memory_cycles, residual.cycles) == (3, 0, 3)
    assert result.total_cycles == 39
    assert result.step_composition == "max_compute_memory"

    profile = profile_execution_trace(trace)
    assert profile[3] == trace.dynamic_instruction_count


def test_cost_model_fails_closed_on_geometry_or_opcode_drift() -> None:
    trace = build_execution_trace(
        "V_ADD_VV gp1, gp2, gp3, 0\n",
        mlen=64,
        blen=4,
        hlen=16,
    )
    geometry_model = DecodeCostModel(
        {"V_ADD_VV": 1},
        memory_bandwidth_bytes_per_cycle=32,
        expected_geometry=(128, 4, 128, 16),
    )
    with pytest.raises(ValueError, match="geometry differs"):
        geometry_model.evaluate(trace)

    opcode_model = DecodeCostModel(
        {"S_ADDI_INT": 1},
        memory_bandwidth_bytes_per_cycle=32,
    )
    with pytest.raises(ValueError, match="unpriced opcodes"):
        opcode_model.evaluate(trace)


def test_stage_validation_reports_worst_component_and_coverage() -> None:
    modelled = {
        "Activation load + RMSNorm": 79,
        "Q/K/V + W_O projection + RoPE": 103,
        "KV store": 20,
        "Flash attention": 200,
        "Residual add": 50,
        "FFN (gate/up/down)": 300,
        "LM head": 100,
    }
    measured = {
        stage: {"scalar": cycles}
        for stage, cycles in {
            "Activation load + RMSNorm": 80,
            "Q/K/V + W_O projection + RoPE": 100,
            "KV store": 20,
            "Flash attention": 200,
            "Residual add": 50,
            "FFN (gate/up/down)": 300,
            "LM head": 100,
            "Setup": 5,
        }.items()
    }
    summary = summarize_stage_validation(modelled, measured)
    assert summary.worst_stage == "Q/K/V + W_O projection + RoPE"
    assert summary.worst_stage_error == pytest.approx(0.03)
    assert summary.coverage == pytest.approx(850 / 855)
    assert summary.meets_target()


def test_packed_q1_contract_uses_dynamic_trace_histogram_and_assembly_hash() -> None:
    trace = build_execution_trace(
        "M_BTMM 0, gp1, gp2\nM_BMM_WO gp3, gp0, 0\n",
        mlen=64,
        blen=4,
        hlen=16,
    )

    class Point:
        opcode_histogram = tuple(trace.opcode_histogram.items())
        assembly_sha256 = trace.assembly_sha256

    class Contract:
        @staticmethod
        def point(cache_tokens: int):
            if cache_tokens != 128:
                raise KeyError(cache_tokens)
            return Point()

    assert validate_packed_q1_execution_trace(
        trace,
        Contract(),
        cache_tokens=128,
    ) == (True, "packed_q1_execution_trace_validated")
    assert validate_packed_q1_execution_trace(
        trace,
        Contract(),
        cache_tokens=256,
    ) == (False, "packed_q1_cache_point_missing")
