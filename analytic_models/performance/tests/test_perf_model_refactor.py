from pathlib import Path

import pytest

from analytic_models.performance.perf_model import CycleBreakdown, HardwareConfig, PerfModel

_CUSTOM_ISA_PATH = Path(__file__).resolve().parents[1] / "customISA_lib.json"


def _hardware_config() -> HardwareConfig:
    return HardwareConfig(
        MLEN=64,
        BLEN=4,
        VLEN=64,
        HLEN=16,
        VECTOR_SRAM_SIZE=1024,
        HBM_V_Prefetch_Amount=4,
        VECTOR_BASIC_CYCLES=1,
        VECTOR_ADD_CYCLES=1,
        VECTOR_MUL_CYCLES=1,
        VECTOR_EXP_CYCLES=3,
        VECTOR_MAX_CYCLES=2,
        VECTOR_SUM_CYCLES=2,
        SCALAR_FP_BASIC_CYCLES=1,
        SCALAR_FP_EXP_CYCLES=3,
        SCALAR_FP_SQRT_CYCLES=4,
        SCALAR_FP_RECI_CYCLES=3,
        SCALAR_INT_BASIC_CYCLES=1,
    )


def _perf_model() -> PerfModel:
    return PerfModel(_hardware_config(), str(_CUSTOM_ISA_PATH))


def test_latency_expression_evaluator_supports_custom_isa_arithmetic():
    from analytic_models.performance.latency import build_latency_context, evaluate_latency_expression

    context = build_latency_context(_hardware_config())

    assert context["SA_ACC_CYCLES"] == 5
    assert evaluate_latency_expression("(MLEN // BLEN)**2 * BLEN", context) == 1024
    assert evaluate_latency_expression("MLEN / BLEN", context) == 16
    assert evaluate_latency_expression("-BLEN + MLEN", context) == 60


def test_latency_expression_evaluator_rejects_large_power_exponents():
    from analytic_models.performance.latency import build_latency_context, evaluate_latency_expression

    context = build_latency_context(_hardware_config())

    with pytest.raises(ValueError, match="Power exponent"):
        evaluate_latency_expression("2 ** 9", context)


def test_latency_expression_evaluator_rejects_function_calls():
    from analytic_models.performance.latency import build_latency_context, evaluate_latency_expression

    context = build_latency_context(_hardware_config())

    with pytest.raises(ValueError, match="Unsupported latency expression"):
        evaluate_latency_expression("__import__('os').system('echo unsafe')", context)


def test_cycle_breakdown_supports_explicit_total_and_immutable_components():
    breakdown = CycleBreakdown({"compute": 10, "overlapped_transfer": 7}, total_cycles=10)

    assert breakdown.total == 10
    with pytest.raises(TypeError):
        breakdown.components["compute"] = 20


@pytest.mark.parametrize(
    ("mode", "expected_components", "expected_total"),
    [
        ("prefill", {"up_gate": 8192, "activation": 768, "down": 4096}, 13056),
        ("decode", {"up_gate": 1024, "activation": 48, "down": 512}, 1584),
    ],
)
def test_feed_forward_breakdown_matches_fixed_cycle_baselines(mode, expected_components, expected_total):
    perf = _perf_model()

    breakdown = perf.feed_forward_breakdown(
        hidden_size=128,
        intermediate_size=256,
        seq_len=16,
        batch_size=2,
        mode=mode,
    )

    assert dict(breakdown.components) == expected_components
    assert breakdown.total == expected_total
    assert perf.feed_forward(128, 256, 16, 2, mode) == expected_total


@pytest.mark.parametrize(
    ("mode", "expected_components", "expected_total"),
    [
        (
            "prefill",
            {
                "norm": 256,
                "router": 208,
                "topk": 224,
                "routing_softmax": 256,
                "expert_up_gate": 26624,
                "expert_activation": 1536,
                "expert_down": 10752,
                "combine": 256,
            },
            40112,
        ),
        (
            "decode",
            {
                "norm": 16,
                "router": 60,
                "topk": 14,
                "routing_softmax": 16,
                "expert_up_gate": 7680,
                "expert_activation": 96,
                "expert_down": 3200,
                "combine": 16,
            },
            11098,
        ),
    ],
)
def test_mlp_moe_breakdown_matches_fixed_cycle_baselines(mode, expected_components, expected_total):
    perf = _perf_model()

    breakdown = perf.mlp_moe_breakdown(
        hidden_size=128,
        seq_len=16,
        batch_size=2,
        num_experts=8,
        expert_per_token=2,
        intermediate_size=256,
        mode=mode,
    )

    assert dict(breakdown.components) == expected_components
    assert breakdown.total == expected_total
    assert perf.mlp_moe(128, 16, 2, 8, 2, 256, mode) == expected_total
