from compiler.aten.isa_builder import DmaTransfer
from compiler.aten.program_sink import CostTrace, TraceDma, TraceInstruction

from analytic_models.latency import (
    ConfiguredBandwidthMemoryProvider,
    MainTimingConfig,
    estimate_compute_latency,
    estimate_latency,
)


def _trace():
    return CostTrace(
        schema_version="test-v1",
        isa_hash="isa",
        compiler_hash="compiler",
        instructions=(
            TraceInstruction("attention", "M_MM", (), (), None, (), 2),
            TraceInstruction("ffn", "S_ADD_INT", (), (), None, (), 10),
            TraceInstruction("attention", "H_PREFETCH_M", (), (), None, (), 1),
        ),
        dma_events=(
            TraceDma(
                stage="attention",
                transfer=DmaTransfer(
                    opcode="H_PREFETCH_M",
                    direction="read",
                    role="weight",
                    element_base_bytes=0,
                    scale_base_bytes=None,
                    dim=64,
                    amount=1,
                    stride_bytes=64,
                ),
                multiplicity=1,
                repeat_axes=(),
            ),
        ),
        metadata={},
    )


def _config():
    return MainTimingConfig(mlen=16, blen=4, vlen=16, hlen=4, broadcast_amount=4)


def test_configured_bandwidth_uses_line_rounded_dma_and_stage_roofline():
    trace = _trace()
    compute = estimate_compute_latency(trace, _config())
    memory = ConfiguredBandwidthMemoryProvider(1.0).estimate(trace)
    report = estimate_latency(trace, compute, memory)

    assert memory.physical_read_bytes == 64
    assert memory.by_stage_picos == {"attention": 64_000}
    assert report.total_picos == 64_000 + 10_000
    assert report.serial_total_picos == compute.total_picos + memory.total_picos
    assert [stage.stage for stage in report.stages] == ["attention", "ffn"]


def test_serial_overlap_policy_is_reported_separately():
    trace = _trace()
    report = estimate_latency(
        trace,
        "main",
        ConfiguredBandwidthMemoryProvider(1.0),
        overlap_policy="serial",
        hardware_config=_config(),
    )
    assert report.total_picos == report.serial_total_picos
