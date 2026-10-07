import torch

from transactional_emulator.testbench.gpt_oss_testkit import (
    MxfpWeightCache,
    _comparison_diagnostics,
    _linear_projection_golden,
    _machine_code_line_count,
    _scan_cache_append_tokens,
)


def test_cached_linear_golden_is_exact_and_invalidates_on_mutation() -> None:
    torch.manual_seed(7)
    value = torch.randn(2, 320, dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(320, 128, dtype=torch.float32).to(torch.bfloat16)
    cache = MxfpWeightCache()

    expected = _linear_projection_golden(value, weight, mlen=64, hbm_input=False)
    first = _linear_projection_golden(
        value,
        weight,
        mlen=64,
        hbm_input=False,
        weight_cache=cache,
    )
    second = _linear_projection_golden(
        value,
        weight,
        mlen=64,
        hbm_input=False,
        weight_cache=cache,
    )
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)

    weight[0, 0] += 1
    mutated_expected = _linear_projection_golden(value, weight, mlen=64, hbm_input=False)
    mutated_actual = _linear_projection_golden(
        value,
        weight,
        mlen=64,
        hbm_input=False,
        weight_cache=cache,
    )
    assert torch.equal(mutated_actual, mutated_expected)


def test_comparison_diagnostics_separates_prefill_decode_and_final_layer() -> None:
    golden = torch.ones(2, 3, 2)
    simulated = golden.clone()
    simulated[0, 2, 0] = 2.0
    results = {
        "golden_values": golden.flatten(),
        "simulated_values": simulated.flatten(),
        "atol": 0.0,
        "rtol": 0.0,
    }

    diagnostics = _comparison_diagnostics(
        results,
        checkpoint_stages=2,
        total_tokens=3,
        hidden=2,
        prefill_tokens=2,
    )

    assert diagnostics["compared_values"] == 12
    assert diagnostics["mismatch_values"] == 1
    assert diagnostics["prefill_allclose_match_rate"] == 100.0
    assert diagnostics["decode_allclose_match_rate"] == 75.0
    assert diagnostics["final_layer_allclose_match_rate"] == 100.0


def test_cache_append_scan_does_not_need_an_assembly_line_list() -> None:
    assembly = "\n".join(
        (
            "; DECODE_CACHE_APPEND cache_a token=0 row=0",
            "S_ADDI_INT gp1, gp0, 1",
            "; DECODE_CACHE_APPEND ignored token=9 row=9",
            "; DECODE_CACHE_APPEND cache_b token=4 row=4",
            "; DECODE_CACHE_APPEND cache_a token=1 row=1",
        )
    )

    assert _scan_cache_append_tokens(assembly, ("cache_a", "cache_b")) == {
        "cache_a": [0, 1],
        "cache_b": [4],
    }


def test_machine_code_line_count_uses_runner_artifact_summary() -> None:
    assert _machine_code_line_count({"artifacts": {"machine_code_lines": 17}}) == 17
