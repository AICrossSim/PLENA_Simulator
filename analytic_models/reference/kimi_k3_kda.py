"""Readable FP32 CPU reference for the recurrent core of Kimi K3 KDA.

The state layout follows FlashKDA's ``transpose_state_layout=True`` contract:
``[batch, head, value_dim, key_dim]``. This module models the recurrent KDA
core after q/k/v/decay/beta projection and short convolution. Output gating,
RMSNorm, and the output projection remain ordinary PLENA Vector/Matrix work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .state_precision import StateStorage, quantize_state


@dataclass(frozen=True)
class KdaShape:
    hidden_size: int
    num_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    chunk_size: int = 16
    gate_lower_bound: float = -5.0

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "key_dim", "value_dim", "conv_kernel", "chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound must be negative")

    @classmethod
    def kimi_k3(cls) -> KdaShape:
        return cls(7168, 96, 128, 128, 4)

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.key_dim

    @property
    def state_elements(self) -> int:
        return self.num_heads * self.value_dim * self.key_dim

    @property
    def conv_state_elements(self) -> int:
        return 3 * self.projection_size * self.conv_kernel


@dataclass
class KdaState:
    recurrent: Tensor

    @classmethod
    def zeros(
        cls,
        shape: KdaShape,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
    ) -> KdaState:
        return cls(
            torch.zeros(
                batch_size,
                shape.num_heads,
                shape.value_dim,
                shape.key_dim,
                dtype=torch.float32,
                device=device,
            )
        )

    def clone(self) -> KdaState:
        return KdaState(self.recurrent.clone())

    def reset_(self) -> None:
        self.recurrent.zero_()


def _require_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")


def activate_log_decay(
    gate: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    *,
    lower_bound: float,
) -> Tensor:
    """Return Kimi K3's channel-wise log decay in ``[lower_bound, 0]``."""
    if gate.ndim != 3:
        raise ValueError("gate must have shape [batch, heads, key_dim]")
    _, heads, key_dim = gate.shape
    _require_shape("a_log", a_log, (heads,))
    _require_shape("dt_bias", dt_bias, (heads, key_dim))
    if lower_bound >= 0:
        raise ValueError("lower_bound must be negative")
    rate = torch.exp(a_log.float())[None, :, None]
    return lower_bound * torch.sigmoid(rate * (gate.float() + dt_bias.float()[None, :, :]))


def kda_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta_logit: Tensor,
    state: KdaState,
    a_log: Tensor,
    dt_bias: Tensor,
    shape: KdaShape,
    *,
    scale: float | None = None,
    state_storage: StateStorage | str = StateStorage.BF16,
) -> tuple[Tensor, KdaState]:
    """Execute one recurrent KDA token with FP32 update and reduction.

    With state ``S`` stored as ``[value, key]`` the operation is::

        D = S * exp(log_decay)
        error = sigmoid(beta) * (v - D @ k)
        S_new = D + outer(error, k)
        output = scale * S_new @ q

    This is equivalent to the delta-rule matrix form
    ``S_new = D @ (I - beta*k*k^T) + beta*v*k^T``.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [batch, heads, dimension]")
    batch = q.shape[0]
    key_shape = (batch, shape.num_heads, shape.key_dim)
    value_shape = (batch, shape.num_heads, shape.value_dim)
    _require_shape("q", q, key_shape)
    _require_shape("k", k, key_shape)
    _require_shape("v", v, value_shape)
    _require_shape("gate", gate, key_shape)
    _require_shape("beta_logit", beta_logit, (batch, shape.num_heads))
    _require_shape(
        "state",
        state.recurrent,
        (batch, shape.num_heads, shape.value_dim, shape.key_dim),
    )

    q_normalized = functional.normalize(q.float(), p=2.0, dim=-1)
    k_normalized = functional.normalize(k.float(), p=2.0, dim=-1)
    log_decay = activate_log_decay(
        gate,
        a_log,
        dt_bias,
        lower_bound=shape.gate_lower_bound,
    )
    decayed = state.recurrent.float() * torch.exp(log_decay)[:, :, None, :]
    prediction = torch.einsum("bhvk,bhk->bhv", decayed, k_normalized)
    beta = torch.sigmoid(beta_logit.float())[:, :, None]
    error = beta * (v.float() - prediction)
    updated = decayed + error[:, :, :, None] * k_normalized[:, :, None, :]
    output_scale = scale if scale is not None else 1.0 / math.sqrt(shape.key_dim)
    output = output_scale * torch.einsum("bhvk,bhk->bhv", updated, q_normalized)
    return output, KdaState(quantize_state(updated, state_storage))


def kda_recurrent_sequence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta_logit: Tensor,
    state: KdaState,
    a_log: Tensor,
    dt_bias: Tensor,
    shape: KdaShape,
    *,
    scale: float | None = None,
    state_storage: StateStorage | str = StateStorage.BF16,
) -> tuple[Tensor, KdaState]:
    """Sequential golden reference for a prefill or multi-token decode trace."""
    if q.ndim != 4:
        raise ValueError("sequence q must have shape [batch, tokens, heads, key_dim]")
    batch, tokens, heads, key_dim = q.shape
    _require_shape("q", q, (batch, tokens, shape.num_heads, shape.key_dim))
    _require_shape("k", k, (batch, tokens, heads, key_dim))
    _require_shape("v", v, (batch, tokens, shape.num_heads, shape.value_dim))
    _require_shape("gate", gate, (batch, tokens, heads, key_dim))
    _require_shape("beta_logit", beta_logit, (batch, tokens, shape.num_heads))

    outputs = []
    current = state
    for token in range(tokens):
        output, current = kda_step(
            q[:, token],
            k[:, token],
            v[:, token],
            gate[:, token],
            beta_logit[:, token],
            current,
            a_log,
            dt_bias,
            shape,
            scale=scale,
            state_storage=state_storage,
        )
        outputs.append(output)
    return torch.stack(outputs, dim=1), current
