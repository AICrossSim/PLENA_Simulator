import pytest

from transactional_emulator.testbench.models.nemotron3 import (
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


def test_long_supported_run_updates_all_length_fields() -> None:
    full_model._configure_lengths(128, 128)

    assert full_model.PREFILL_TOKENS == 128
    assert full_model.DECODE_TOKENS == 128
    assert full_model.TOTAL_TOKENS == 256


@pytest.mark.parametrize("prefill_tokens,decode_tokens", ((0, 4), (16, 0), (512, 1)))
def test_invalid_lengths_are_rejected(prefill_tokens: int, decode_tokens: int) -> None:
    with pytest.raises(ValueError):
        full_model._configure_lengths(prefill_tokens, decode_tokens)
