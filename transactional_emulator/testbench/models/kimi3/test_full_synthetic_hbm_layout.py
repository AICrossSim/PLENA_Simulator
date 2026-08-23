from itertools import pairwise

import pytest
import torch

from transactional_emulator.testbench.models.kimi3 import (
    full_synthetic_connected_test as full_model,
)


@pytest.fixture(autouse=True)
def _restore_default_lengths():
    original = (
        full_model.PREFILL_TOKENS,
        full_model.DECODE_TOKENS,
    )
    try:
        yield
    finally:
        full_model._configure_lengths(*original)


@pytest.mark.parametrize(
    ("prefill_tokens", "decode_tokens"),
    ((128, 4), (16, 128)),
)
def test_long_runs_keep_kda_hbm_regions_disjoint(prefill_tokens: int, decode_tokens: int) -> None:
    full_model._configure_lengths(prefill_tokens, decode_tokens)
    phases = full_model._lower_kda_phases()
    regions = full_model._validate_kda_hbm_regions(*phases)

    ordered = [
        regions["prefill_descriptors"],
        regions["decode_descriptors"],
        regions["state_arena"],
        regions["projection_weights"],
    ]
    assert all(left[1] <= right[0] for left, right in pairwise(ordered))


def test_hbm_region_guard_rejects_state_weight_overlap(monkeypatch) -> None:
    full_model._configure_lengths(16, 4)
    phases = full_model._lower_kda_phases()
    state_end = phases[0].hbm_layout().realized_arena_bytes(len(phases[1].events))
    monkeypatch.setattr(full_model, "KDA_WEIGHT_BASE", state_end - 64)

    with pytest.raises(ValueError, match=r"state_arena.*projection_weights"):
        full_model._validate_kda_hbm_regions(*phases)


def test_cache_precision_diagnostics_report_one_bf16_ulp() -> None:
    expected = torch.full((1024,), 4.0, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0] = 4.03125

    diagnostics = full_model._cache_precision_diagnostics(actual, expected)

    assert diagnostics["max_abs_error"] == 0.03125
    assert diagnostics["relative_l2_error"] < 0.001
    assert diagnostics["mismatch_values"] == 0
