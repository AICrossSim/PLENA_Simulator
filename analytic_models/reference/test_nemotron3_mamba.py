from __future__ import annotations

import torch

from .nemotron3_mamba import (
    Mamba2Shape,
    Mamba2State,
    Mamba2Weights,
    affine_scan_chunked,
    gated_group_rms_norm,
    mamba_prefill_sequential,
    mamba_state_engine_prefill,
    mamba_state_engine_step,
    mamba_step,
    selective_state_step,
)
from .state_precision import StateStorage, quantize_state, storage_bytes


def _tiny_shape() -> Mamba2Shape:
    return Mamba2Shape(hidden_size=6, num_heads=4, head_dim=2, state_dim=3, groups=2, conv_kernel=3)


def _tiny_weights(shape: Mamba2Shape) -> Mamba2Weights:
    generator = torch.Generator().manual_seed(7)

    def randn(*dims: int, scale: float = 0.1) -> torch.Tensor:
        return torch.randn(*dims, generator=generator) * scale

    return Mamba2Weights(
        in_proj_weight=randn(shape.projection_size, shape.hidden_size),
        in_proj_bias=randn(shape.projection_size),
        conv_weight=randn(shape.conv_channels, shape.conv_kernel),
        conv_bias=randn(shape.conv_channels),
        a_log=randn(shape.num_heads),
        dt_bias=randn(shape.num_heads),
        d_skip=randn(shape.num_heads),
        norm_weight=torch.ones(shape.d_inner),
        out_proj_weight=randn(shape.hidden_size, shape.d_inner),
        out_proj_bias=randn(shape.hidden_size),
    )


def test_nemotron_shape_facts() -> None:
    shape = Mamba2Shape.nemotron3()
    assert shape.d_inner == 4096
    assert shape.conv_channels == 6144
    assert shape.projection_size == 10304
    assert shape.heads_per_group == 8
    assert shape.state_elements * 4 == 2 * 1024 * 1024


def test_selective_step_matches_explicit_equations() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, 2, generator=generator)
    b = torch.randn(1, 2, 3, generator=generator)
    c = torch.randn(1, 2, 3, generator=generator)
    state = torch.randn(1, 4, 2, 3, generator=generator)
    dt_raw = torch.randn(1, 4, generator=generator)
    a_log = torch.randn(4, generator=generator)
    dt_bias = torch.randn(4, generator=generator)
    d_skip = torch.randn(4, generator=generator)

    output, new_state = selective_state_step(x, b, c, dt_raw, state, a_log, dt_bias, d_skip)
    dt = torch.nn.functional.softplus(dt_raw + dt_bias)
    a = -torch.exp(a_log)
    b_heads = b.repeat_interleave(2, dim=1)
    c_heads = c.repeat_interleave(2, dim=1)
    expected_state = state * torch.exp(dt[:, :, None, None] * a[None, :, None, None])
    expected_state += dt[:, :, None, None] * b_heads[:, :, None, :] * x[:, :, :, None]
    expected_output = (expected_state * c_heads[:, :, None, :]).sum(-1)
    expected_output += x * d_skip[None, :, None]
    torch.testing.assert_close(new_state, expected_state)
    torch.testing.assert_close(output, expected_output)


def test_gated_group_rms_norm_gates_before_normalization() -> None:
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    gate = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    actual = gated_group_rms_norm(value, gate, weight, groups=2, eps=1e-5)
    gated = value * torch.nn.functional.silu(gate)
    grouped = gated.reshape(1, 2, 2)
    expected = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)).reshape(1, 4)
    torch.testing.assert_close(actual, expected * weight)


def test_sequential_prefill_matches_repeated_step() -> None:
    shape = _tiny_shape()
    weights = _tiny_weights(shape)
    hidden = torch.randn(2, 5, shape.hidden_size, generator=torch.Generator().manual_seed(11))
    initial = Mamba2State.zeros(shape, 2)
    output, state = mamba_prefill_sequential(hidden, initial, weights, shape)

    expected_outputs = []
    expected_state = initial
    for token in hidden.unbind(1):
        token_output, expected_state = mamba_step(token, expected_state, weights, shape)
        expected_outputs.append(token_output)
    torch.testing.assert_close(output, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(state.ssm, expected_state.ssm)
    torch.testing.assert_close(state.conv, expected_state.conv)


def test_x_state_prefill_matches_repeated_step_before_gate() -> None:
    shape = _tiny_shape()
    weights = _tiny_weights(shape)
    projected = torch.randn(
        2,
        5,
        shape.projection_size,
        generator=torch.Generator().manual_seed(13),
    )
    initial = Mamba2State.zeros(shape, 2)
    output, state = mamba_state_engine_prefill(
        projected,
        initial,
        weights.conv_weight,
        weights.a_log,
        weights.dt_bias,
        weights.d_skip,
        shape,
        conv_bias=weights.conv_bias,
        state_storage=StateStorage.BF16,
    )
    expected_outputs = []
    expected_state = initial
    for token in projected.unbind(1):
        token_output, expected_state = mamba_state_engine_step(
            token,
            expected_state,
            weights.conv_weight,
            weights.a_log,
            weights.dt_bias,
            weights.d_skip,
            shape,
            conv_bias=weights.conv_bias,
            state_storage=StateStorage.BF16,
        )
        expected_outputs.append(token_output)
    torch.testing.assert_close(output, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(state.ssm, expected_state.ssm)
    torch.testing.assert_close(state.conv, expected_state.conv)


def test_chunked_affine_scan_matches_recurrence() -> None:
    generator = torch.Generator().manual_seed(19)
    decay = torch.sigmoid(torch.randn(7, 2, 3, generator=generator))
    update = torch.randn(7, 2, 3, generator=generator)
    initial = torch.randn(2, 3, generator=generator)
    actual, boundary = affine_scan_chunked(decay, update, initial, chunk_size=3)
    expected = []
    state = initial
    for a_t, b_t in zip(decay, update, strict=True):
        state = a_t * state + b_t
        expected.append(state)
    torch.testing.assert_close(actual, torch.stack(expected))
    torch.testing.assert_close(boundary, expected[-1])


def test_state_storage_round_trips_and_byte_counts() -> None:
    value = torch.randn(257, generator=torch.Generator().manual_seed(23)) * 3.7
    torch.testing.assert_close(quantize_state(value, StateStorage.FP32), value)
    for storage in (StateStorage.BF16, StateStorage.FP16, StateStorage.MX8_B128):
        rounded = quantize_state(value, storage)
        assert rounded.shape == value.shape
        assert torch.isfinite(rounded).all()
        assert not torch.equal(rounded, value)
    assert storage_bytes(256, StateStorage.FP32) == 1024
    assert storage_bytes(256, StateStorage.BF16) == 512
    assert storage_bytes(256, StateStorage.MX8_B128) == 258


def test_mx8_e4m3fn_rounding_carries_across_exponents() -> None:
    value = torch.zeros(128)
    value[:5] = torch.tensor([1.9375, 248.0, 432.0, 448.0, -448.0])
    restored = quantize_state(value, StateStorage.MX8_B128)
    torch.testing.assert_close(
        restored[:5],
        torch.tensor([2.0, 256.0, 448.0, 448.0, -448.0]),
        rtol=0,
        atol=0,
    )
