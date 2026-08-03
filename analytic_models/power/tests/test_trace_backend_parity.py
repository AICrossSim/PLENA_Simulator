from compiler.aten.cost_frontend import (
    CompilerHardwareSpec,
    DecoderModelSpec,
    RoutingHistogram,
    compile_dense_decoder_trace,
    compile_routed_moe_trace,
)
from compiler.aten.program_sink import (
    COST_TRACE_GRANULARITY_DETAILED,
    COST_TRACE_GRANULARITY_SUMMARY,
)

from analytic_models.latency import MainTimingConfig, estimate_compute_latency
from analytic_models.latency.hbm_v4 import (
    DEFAULT_HBM_V4_CALIBRATION,
    HbmPrecisionConfig,
    HbmServiceModelV4,
    HbmV4Config,
    HbmV4MemoryProvider,
    MemoryFormat,
)
from analytic_models.power import (
    DEFAULT_LOGIC_ENERGY,
    ActionHardwareConfig,
    estimate_action_energy,
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
        model_type="qwen3_moe",
    )


def _precision(*, router_bf16: bool) -> HbmPrecisionConfig:
    plain8 = MemoryFormat("plain", 8, name="plain8")
    return HbmPrecisionConfig(
        weight=plain8,
        matrix_kv=(
            MemoryFormat("plain", 16, name="router-bf16")
            if router_bf16
            else plain8
        ),
        activation=plain8,
        vector_kv=plain8,
        integer=MemoryFormat("plain", 32, name="int32"),
    )


def _assert_backend_parity(detailed, summary, *, router_bf16: bool) -> None:
    assert detailed.trace.dynamic_opcode_counts == summary.trace.dynamic_opcode_counts
    detailed_dma = sum(event.multiplicity for event in detailed.trace.dma_events)
    summary_dma = sum(event.multiplicity for event in summary.trace.dma_events)
    assert detailed_dma == summary_dma

    timing = MainTimingConfig(mlen=8, blen=4, vlen=8, hlen=4, broadcast_amount=2)
    detailed_compute = estimate_compute_latency(detailed.trace, timing)
    summary_compute = estimate_compute_latency(summary.trace, timing)
    assert detailed_compute.total_picos == summary_compute.total_picos
    assert detailed_compute.by_stage_picos == summary_compute.by_stage_picos
    assert detailed_compute.by_resource_picos == summary_compute.by_resource_picos

    def provider():
        return HbmV4MemoryProvider(
            HbmServiceModelV4.load(DEFAULT_HBM_V4_CALIBRATION),
            _precision(router_bf16=router_bf16),
            HbmV4Config(8),
            aggregation="sufficient-statistics",
        )
    detailed_memory = provider().estimate(detailed.trace)
    summary_memory = provider().estimate(summary.trace)
    assert detailed_memory.total_picos == summary_memory.total_picos
    assert detailed_memory.physical_read_bytes == summary_memory.physical_read_bytes
    assert detailed_memory.physical_write_bytes == summary_memory.physical_write_bytes

    action_hardware = ActionHardwareConfig(mlen=8, blen=4, vlen=8)
    detailed_energy = estimate_action_energy(
        detailed.trace, action_hardware, DEFAULT_LOGIC_ENERGY
    )
    summary_energy = estimate_action_energy(
        summary.trace, action_hardware, DEFAULT_LOGIC_ENERGY
    )
    assert detailed_energy.nominal_energy_pj == summary_energy.nominal_energy_pj
    assert detailed_energy.by_component_pj == summary_energy.by_component_pj


def test_dense_detailed_and_summary_backends_are_exactly_equivalent():
    detailed = compile_dense_decoder_trace(
        _dense_model(),
        _hardware(),
        seq_len=3,
        cost_trace_granularity=COST_TRACE_GRANULARITY_DETAILED,
    )
    summary = compile_dense_decoder_trace(
        _dense_model(),
        _hardware(),
        seq_len=3,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )
    _assert_backend_parity(detailed, summary, router_bf16=False)


def test_moe_detailed_and_summary_backends_are_exactly_equivalent():
    routing = RoutingHistogram.balanced(token_count=1, top_k=4, num_experts=32)
    detailed = compile_routed_moe_trace(
        _moe_model(),
        _hardware(),
        routing,
        cost_trace_granularity=COST_TRACE_GRANULARITY_DETAILED,
    )
    summary = compile_routed_moe_trace(
        _moe_model(),
        _hardware(),
        routing,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )
    _assert_backend_parity(detailed, summary, router_bf16=True)
