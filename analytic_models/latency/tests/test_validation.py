import pytest

from compiler.aten.cost_frontend import (
    CompilerHardwareSpec,
    DecoderModelSpec,
    RoutingHistogram,
    compile_dense_decoder_trace,
    compile_routed_moe_trace,
)
from compiler.aten.isa_builder import Instr
from compiler.aten.program_sink import SymbolicCostSink

from analytic_models.latency.compute import estimate_compute_latency
from analytic_models.latency.timing import MainTimingConfig
from analytic_models.latency.validation import (
    validate_compute_against_emulator_profile,
    validate_detailed_trace_against_assembly,
)


def _hardware():
    return CompilerHardwareSpec(mlen=8, blen=4, mram_tile_capacity=4)


def _dense_model():
    return DecoderModelSpec(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )


def _moe_model():
    return DecoderModelSpec(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=32,
        experts_per_token=4,
        moe_intermediate_size=16,
        model_type="gpt_oss",
    )


@pytest.mark.parametrize("workload", ["dense", "moe"])
def test_detailed_trace_matches_rendered_emulator_assembly(workload):
    if workload == "dense":
        result = compile_dense_decoder_trace(
            _dense_model(), _hardware(), seq_len=3, include_assembly=True
        )
    else:
        result = compile_routed_moe_trace(
            _moe_model(),
            _hardware(),
            RoutingHistogram.balanced(token_count=1, top_k=4, num_experts=32),
            include_assembly=True,
        )
    assert result.assembly is not None
    parity = validate_detailed_trace_against_assembly(result.trace, result.assembly)
    assert parity.exact
    assert parity.trace_counts == parity.assembly_counts


def test_emulator_profile_comparison_uses_rust_resource_partition():
    sink = SymbolicCostSink(default_stage="decoder/test")
    for instruction in (
        Instr("M_MM"),
        Instr("V_ADD_VV"),
        Instr("S_ADD_FP"),
        Instr("C_SET_ADDR_REG"),
    ):
        sink.emit_instruction(instruction)
    trace = sink.finish()
    config = MainTimingConfig(mlen=8, blen=4, vlen=8, hlen=4, broadcast_amount=2)
    compute = estimate_compute_latency(trace, config)
    profile = {
        "total_instructions_executed": 4,
        "total_resource_proxy_picos": {
            "matrix": compute.by_resource_picos["matrix"],
            "vector": compute.by_resource_picos["vector"],
            "scalar": (
                compute.by_resource_picos["scalar"]
                + compute.by_resource_picos["control"]
            ),
            "dma": 0,
            "other": 0,
        },
    }
    assert validate_compute_against_emulator_profile(trace, compute, profile).exact

    profile["total_resource_proxy_picos"]["vector"] += 1
    with pytest.raises(ValueError, match="vector_picos"):
        validate_compute_against_emulator_profile(trace, compute, profile)


def test_assembly_parity_reports_opcode_drift():
    sink = SymbolicCostSink(default_stage="decoder/test")
    sink.emit_instruction(Instr("V_ADD_VV"))
    trace = sink.finish()
    report = validate_detailed_trace_against_assembly(
        trace, "V_MUL_VV gp1, gp2, gp3, 0\n", raise_on_mismatch=False
    )
    assert not report.exact
    assert report.mismatches == {
        "V_ADD_VV": (1, 0),
        "V_MUL_VV": (0, 1),
    }
