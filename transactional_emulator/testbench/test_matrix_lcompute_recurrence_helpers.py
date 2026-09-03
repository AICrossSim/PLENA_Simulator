from __future__ import annotations

import pytest
import torch

from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (
    _assert_close,
)


def test_numerical_guard_rejects_a_lane_permutation() -> None:
    expected = torch.arange(256, dtype=torch.float32).reshape(4, 64)
    permuted = expected.roll(1, dims=1)
    with pytest.raises(AssertionError, match="values mismatch"):
        _assert_close("permuted lanes", permuted, expected)


def test_numerical_guard_rejects_a_head_permutation() -> None:
    expected = torch.arange(256, dtype=torch.float32).reshape(4, 64)
    permuted = expected.roll(1, dims=0)
    with pytest.raises(AssertionError, match="values mismatch"):
        _assert_close("permuted heads", permuted, expected)
