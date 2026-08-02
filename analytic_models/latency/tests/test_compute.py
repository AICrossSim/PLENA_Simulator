from compiler.aten.program_sink import CostTrace, TraceInstruction

from analytic_models.latency import MainTimingConfig, estimate_compute_latency


def _instruction(stage, opcode, count, variant=()):
    return TraceInstruction(
        stage=stage,
        opcode=opcode,
        operands=(),
        variant=variant,
        active=None,
        sram=(),
        multiplicity=count,
    )


def _trace(*instructions, metadata=None):
    return CostTrace(
        schema_version="test-v1",
        isa_hash="isa",
        compiler_hash="compiler",
        instructions=tuple(instructions),
        dma_events=(),
        metadata=metadata or {},
    )


def _config():
    return MainTimingConfig(
        mlen=16,
        blen=4,
        vlen=16,
        hlen=4,
        broadcast_amount=4,
        period_picos=1_000,
        vector_sum_cycles=3,
    )


def test_compute_uses_symbolic_multiplicity_and_stage_ownership():
    trace = _trace(
        _instruction("decoder/attention", "M_MM", 3),
        _instruction("decoder/attention", "V_RED_SUM", 5),
        _instruction("decoder/ffn", "S_ADD_INT", 7),
        _instruction("decoder/ffn", "H_PREFETCH_M", 11),
    )
    report = estimate_compute_latency(trace, _config())
    assert report.total_picos == (3 * 16 + 5 * 3 + 7) * 1_000
    assert report.by_stage_picos == {
        "decoder/attention": 63_000,
        "decoder/ffn": 7_000,
    }
    assert report.by_resource_picos == {
        "matrix": 48_000,
        "scalar": 7_000,
        "vector": 15_000,
    }
    assert report.instruction_coverage == 1.0


def test_experimental_ii1_only_changes_vector_scalar_control():
    trace = _trace(
        _instruction("layer", "M_MM", 2),
        _instruction("layer", "V_RED_SUM", 5),
        _instruction("layer", "S_MAP_V_FP", 3),
    )
    main = estimate_compute_latency(trace, _config(), "main")
    ideal = estimate_compute_latency(trace, _config(), "ideal-ii1")
    assert main.by_resource_picos["matrix"] == ideal.by_resource_picos["matrix"]
    assert ideal.total_picos == (2 * 16 + 5 + 3) * 1_000
    assert ideal.timing_provenance["hazards_included"] is False


def test_topk_uses_explicit_routing_histogram():
    trace = _trace(
        _instruction("router", "V_TOPK", 2),
        metadata={"routing_histogram": [1] * 128},
    )
    report = estimate_compute_latency(trace, _config())
    assert report.total_picos == 2 * 128 * _config().vector_max_cycles * 1_000
