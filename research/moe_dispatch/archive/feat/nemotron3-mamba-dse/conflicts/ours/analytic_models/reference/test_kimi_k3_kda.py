from __future__ import annotations

import math

import pytest
import torch

from .kimi_k3_kda import (
    KdaConvWeights,
    KdaShape,
    KdaState,
    KdaXState,
    activate_log_decay,
    kda_recurrent_sequence,
    kda_state_engine_prefill,
    kda_state_engine_step,
    kda_step,
)
from .state_precision import StateStorage


def _small_shape() -> KdaShape:
    return KdaShape(hidden_size=8, num_heads=2, key_dim=4, value_dim=3, conv_kernel=2)


def test_official_kimi_k3_state_geometry() -> None:
    shape = KdaShape.kimi_k3()
    assert shape.projection_size == 12_288
    assert shape.state_elements == 1_572_864
    assert shape.state_elements * 2 == 3 * 1024 * 1024
    assert shape.conv_state_elements * 2 == 288 * 1024


def test_log_decay_is_channelwise_and_bounded() -> None:
    shape = _small_shape()
    gate = torch.tensor(
        [[[-100.0, -1.0, 0.0, 1.0], [100.0, 1.0, 0.0, -1.0]]],
        dtype=torch.float32,
    )
    decay = activate_log_decay(
        gate,
        torch.zeros(shape.num_heads),
        torch.zeros(shape.num_heads, shape.key_dim),
        lower_bound=shape.gate_lower_bound,
    )
    assert torch.all(decay <= 0)
    assert torch.all(decay >= shape.gate_lower_bound)
    assert decay[0, 0, 0] > decay[0, 0, -1]


def test_step_matches_delta_rule_matrix_form() -> None:
    torch.manual_seed(7)
    shape = _small_shape()
    batch = 1
    q = torch.randn(batch, shape.num_heads, shape.key_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch, shape.num_heads, shape.value_dim)
    gate = torch.randn_like(q)
    beta_logit = torch.randn(batch, shape.num_heads)
    initial = KdaState(torch.randn(batch, shape.num_heads, shape.value_dim, shape.key_dim))
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    scale = 0.25

    output, final = kda_step(
        q,
        k,
        v,
        gate,
        beta_logit,
        initial,
        a_log,
        dt_bias,
        shape,
        scale=scale,
        state_storage=StateStorage.FP32,
    )

    qn = q.float() * torch.rsqrt(q.float().square().sum(dim=-1, keepdim=True) + 1.0e-6)
    kn = k.float() * torch.rsqrt(k.float().square().sum(dim=-1, keepdim=True) + 1.0e-6)
    log_decay = activate_log_decay(gate, a_log, dt_bias, lower_bound=shape.gate_lower_bound)
    decayed = initial.recurrent * torch.exp(log_decay)[:, :, None, :]
    beta = torch.sigmoid(beta_logit)
    expected_states = []
    expected_outputs = []
    for head in range(shape.num_heads):
        identity = torch.eye(shape.key_dim)
        transition = identity - beta[0, head] * torch.outer(kn[0, head], kn[0, head])
        expected = decayed[0, head] @ transition
        expected = expected + beta[0, head] * torch.outer(v[0, head], kn[0, head])
        expected_states.append(expected)
        expected_outputs.append(scale * (expected @ qn[0, head]))
    expected_state = torch.stack(expected_states).unsqueeze(0)
    expected_output = torch.stack(expected_outputs).unsqueeze(0)

    torch.testing.assert_close(final.recurrent, expected_state)
    torch.testing.assert_close(output, expected_output)


def test_sequence_matches_repeated_steps() -> None:
    torch.manual_seed(11)
    shape = _small_shape()
    batch, tokens = 2, 5
    q = torch.randn(batch, tokens, shape.num_heads, shape.key_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, shape.num_heads, shape.value_dim)
    gate = torch.randn_like(q)
    beta = torch.randn(batch, tokens, shape.num_heads)
    a_log = torch.zeros(shape.num_heads)
    dt_bias = torch.zeros(shape.num_heads, shape.key_dim)
    initial = KdaState.zeros(shape, batch)

    outputs, final = kda_recurrent_sequence(
        q,
        k,
        v,
        gate,
        beta,
        initial,
        a_log,
        dt_bias,
        shape,
        state_storage=StateStorage.FP32,
    )
    assert outputs.shape == (batch, tokens, shape.num_heads, shape.value_dim)
    assert final.recurrent.shape == (batch, shape.num_heads, shape.value_dim, shape.key_dim)
    assert torch.isfinite(outputs).all()


def test_x_state_prefill_includes_conv_and_matches_repeated_step() -> None:
    torch.manual_seed(17)
    shape = _small_shape()
    batch, tokens = 2, 4
    key_width = shape.num_heads * shape.key_dim
    value_width = shape.num_heads * shape.value_dim
    projection_width = 3 * key_width + value_width + shape.num_heads
    projected = torch.randn(batch, tokens, projection_width)
    conv_weights = KdaConvWeights(
        q=torch.randn(key_width, shape.conv_kernel),
        k=torch.randn(key_width, shape.conv_kernel),
        v=torch.randn(value_width, shape.conv_kernel),
    )
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    initial = KdaXState.zeros(shape, batch)
    output, state = kda_state_engine_prefill(
        projected,
        initial,
        conv_weights,
        a_log,
        dt_bias,
        shape,
        state_storage=StateStorage.BF16,
    )
    expected_outputs = []
    expected_state = initial
    for token in projected.unbind(1):
        token_output, expected_state = kda_state_engine_step(
            token,
            expected_state,
            conv_weights,
            a_log,
            dt_bias,
            shape,
            state_storage=StateStorage.BF16,
        )
        expected_outputs.append(token_output)
    torch.testing.assert_close(output, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(state.recurrent, expected_state.recurrent)
    torch.testing.assert_close(state.conv, expected_state.conv)


def test_default_scale_uses_key_dimension() -> None:
    shape = _small_shape()
    q = torch.ones(1, shape.num_heads, shape.key_dim)
    k = torch.ones_like(q)
    v = torch.ones(1, shape.num_heads, shape.value_dim)
    gate = torch.zeros_like(q)
    beta = torch.zeros(1, shape.num_heads)
    state = KdaState.zeros(shape, 1)
    a_log = torch.zeros(shape.num_heads)
    dt_bias = torch.zeros(shape.num_heads, shape.key_dim)

    default, _ = kda_step(q, k, v, gate, beta, state, a_log, dt_bias, shape)
    explicit, _ = kda_step(
        q,
        k,
        v,
        gate,
        beta,
        state,
        a_log,
        dt_bias,
        shape,
        scale=1.0 / math.sqrt(shape.key_dim),
    )
    torch.testing.assert_close(default, explicit)


def test_shape_validation_rejects_wrong_state_layout() -> None:
    shape = _small_shape()
    q = torch.zeros(1, shape.num_heads, shape.key_dim)
    v = torch.zeros(1, shape.num_heads, shape.value_dim)
    wrong = KdaState(torch.zeros(1, shape.num_heads, shape.key_dim, shape.value_dim))
    with pytest.raises(ValueError, match="state has shape"):
        kda_step(
            q,
            q,
            v,
            q,
            torch.zeros(1, shape.num_heads),
            wrong,
            torch.zeros(shape.num_heads),
            torch.zeros(shape.num_heads, shape.key_dim),
            shape,
        )


def test_conv_state_precision_is_independent_of_recurrent_state_precision() -> None:
    """FP32 recurrent state with BF16 conv state is what actually ships.

    The descriptor carries `state_precision` and `conv_state_precision`
    separately, but this reference used to fold both onto one parameter, so the
    shipped Kimi combination had no CPU reference at all. Two tokens are needed:
    the first token's conv rounding only becomes observable once it is read back
    as history.
    """
    torch.manual_seed(23)
    shape = _small_shape()
    batch, tokens = 1, 2
    key_width = shape.num_heads * shape.key_dim
    value_width = shape.num_heads * shape.value_dim
    projected = torch.randn(batch, tokens, 3 * key_width + value_width + shape.num_heads)
    conv_weights = KdaConvWeights(
        q=torch.randn(key_width, shape.conv_kernel),
        k=torch.randn(key_width, shape.conv_kernel),
        v=torch.randn(value_width, shape.conv_kernel),
    )
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    initial = KdaXState.zeros(shape, batch)

    common = dict(state_storage=StateStorage.FP32)
    uniform, uniform_state = kda_state_engine_prefill(
        projected, initial, conv_weights, a_log, dt_bias, shape, **common
    )
    mixed, mixed_state = kda_state_engine_prefill(
        projected, initial, conv_weights, a_log, dt_bias, shape,
        conv_state_storage=StateStorage.BF16, **common
    )

    # The recurrent state stays FP32 in both runs, so any divergence can only
    # have entered through the conv history.
    assert not torch.equal(uniform_state.conv, mixed_state.conv)
    assert not torch.equal(uniform, mixed)
    torch.testing.assert_close(uniform, mixed, rtol=2e-2, atol=2e-2)
