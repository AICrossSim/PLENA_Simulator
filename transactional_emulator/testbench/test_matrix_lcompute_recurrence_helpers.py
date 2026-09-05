from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Honor the same Compiler checkout as the campaign tests before importing the
# package.  Otherwise pytest collection silently loads the in-tree submodule
# even when PLENA_COMPILER_ROOT names an external worktree.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPILER_ROOT = Path(os.environ.get("PLENA_COMPILER_ROOT", _REPO_ROOT / "PLENA_Compiler")).resolve()
for _path in (_REPO_ROOT, _COMPILER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compiler.aten.plena.matrix_recurrence_lowering import (  # noqa: E402
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    RecurrenceKind,
    RecurrenceLayout,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
)
from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (  # noqa: E402
    RECURRENCE_RELATIVE_L2_LIMIT,
    _assert_close,
    _bf16,
    _kda_inputs,
    _kda_reference,
    _mamba_inputs,
    _mamba_packet_values,
    _mamba_reference,
    _state_seed,
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


@pytest.mark.parametrize(
    ("spec", "inputs", "reference"),
    [(NEMOTRON_MAMBA, _mamba_inputs, _mamba_reference), (KIMI_KDA, _kda_inputs, _kda_reference)],
    ids=["nemotron", "kimi"],
)
def test_numerical_guard_rejects_ten_percent_loss_on_official_outputs(spec, inputs, reference) -> None:
    # These low-magnitude outputs used to pass the 0.01 absolute tolerance
    # even with a 10% gain error. Exercise the actual seeded four-token data.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = _state_seed(spec)
        for token in range(4):
            output, state = reference(state, inputs(token))
            with pytest.raises(AssertionError, match="l2_budget"):
                _assert_close(f"token {token} gain loss", _bf16(output * 0.9), output)
            _assert_close(f"token {token} identity", output, output)
    finally:
        torch.set_num_threads(previous_threads)


def test_numerical_guard_enforces_the_declared_norm_budget() -> None:
    expected = torch.linspace(-0.03, 0.03, 256)
    _assert_close("within BF16 budget", expected * (1 - 0.8 * RECURRENCE_RELATIVE_L2_LIMIT), expected)
    with pytest.raises(AssertionError, match="l2_budget"):
        _assert_close("outside BF16 budget", expected * (1 - 1.2 * RECURRENCE_RELATIVE_L2_LIMIT), expected)


@pytest.mark.parametrize("reference_value", [0.0, 1e-9])
def test_numerical_guard_uses_an_absolute_rms_floor_near_zero(reference_value: float) -> None:
    expected = torch.full((256,), reference_value)
    _assert_close("near zero roundoff", expected + 5e-8, expected)
    with pytest.raises(AssertionError, match="l2_budget"):
        _assert_close("near zero corruption", expected + 2e-7, expected)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_numerical_guard_rejects_non_finite_values(bad_value: float) -> None:
    bad = torch.tensor([bad_value])
    with pytest.raises(AssertionError, match="non-finite"):
        _assert_close("bad result", bad, torch.zeros(1))
    with pytest.raises(AssertionError, match="non-finite"):
        _assert_close("bad reference", torch.zeros(1), bad)


def test_partial_mamba_head_group_zero_pads_every_scalar_packet() -> None:
    """A 24-head checkpoint must fill a 32-head physical packet canonically."""

    spec = MatrixRecurrenceSpec(
        name="mamba2_130m",
        kind=RecurrenceKind.MAMBA,
        heads=24,
        row_elements=64,
        recurrence_rows=128,
        primitives=NEMOTRON_MAMBA.primitives,
    )
    working_set = build_recurrence_working_set(
        spec,
        layout=RecurrenceLayout.AFFINE,
    )
    manifest = build_recurrence_field_manifest(
        working_set,
        field_hbm_base=0,
    )
    operands = {
        "x": torch.zeros(24, 64),
        "dt": torch.arange(24, dtype=torch.float32),
        "a": torch.ones(24, 128),
        "b": torch.ones(24, 128),
        "c": torch.ones(24, 128),
        "d": torch.arange(24, dtype=torch.float32),
    }

    for field in ("dt", "d", "update", "c"):
        packet = manifest.packet(field, group=0, chunk=0 if field in {"update", "c"} else None)
        shaped = _mamba_packet_values(packet, operands, working_set)
        values = shaped.reshape(-1)
        assert values.numel() == packet.logical_values
        if field in {"dt", "d"}:
            assert torch.count_nonzero(values[48:]) == 0
        elif field == "update":
            assert torch.count_nonzero(shaped[:, 48:]) == 0
        else:
            assert torch.count_nonzero(shaped[:, 24:]) == 0
