"""Readable FP32 CPU reference for the Nemotron 3 Mamba-2 layer.

The decode equations mirror NVIDIA's public ``modeling_nemotron_h.py``. The
implementation favors explicit shapes and checkable stages over performance.
It is intended as the golden answer for the simulator, compiler traces, and
future RTL tests; it is not an optimized inference kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .state_precision import StateStorage, quantize_state


@dataclass(frozen=True)
class Mamba2Shape:
    hidden_size: int
    num_heads: int
    head_dim: int
    state_dim: int
    groups: int
    conv_kernel: int
    norm_eps: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "head_dim", "state_dim", "groups", "conv_kernel"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_heads % self.groups:
            raise ValueError("num_heads must be divisible by groups")

    @classmethod
    def nemotron3(cls) -> Mamba2Shape:
        return cls(2688, 64, 64, 128, 8, 4)

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def heads_per_group(self) -> int:
        return self.num_heads // self.groups

    @property
    def conv_channels(self) -> int:
        return self.d_inner + 2 * self.groups * self.state_dim

    @property
    def projection_size(self) -> int:
        return self.d_inner + self.conv_channels + self.num_heads

    @property
    def state_elements(self) -> int:
        return self.num_heads * self.head_dim * self.state_dim


@dataclass
class Mamba2State:
    ssm: Tensor
    conv: Tensor

    @classmethod
    def zeros(
        cls,
        shape: Mamba2Shape,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Mamba2State:
        return cls(
            ssm=torch.zeros(
                batch_size,
                shape.num_heads,
                shape.head_dim,
                shape.state_dim,
                dtype=torch.float32,
                device=device,
            ),
            conv=torch.zeros(
                batch_size,
                shape.conv_channels,
                shape.conv_kernel,
                dtype=torch.float32,
                device=device,
            ),
        )

    def clone(self) -> Mamba2State:
        return Mamba2State(self.ssm.clone(), self.conv.clone())

    def reset_(self) -> None:
        self.ssm.zero_()
        self.conv.zero_()


@dataclass(frozen=True)
class Mamba2Weights:
    in_proj_weight: Tensor
    conv_weight: Tensor
    a_log: Tensor
    dt_bias: Tensor
    d_skip: Tensor
    norm_weight: Tensor
    out_proj_weight: Tensor
    in_proj_bias: Tensor | None = None
    conv_bias: Tensor | None = None
    out_proj_bias: Tensor | None = None
    dt_limit: tuple[float, float] = (0.0, float("inf"))


def _require_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")


def expand_group_parameter(value: Tensor, num_heads: int) -> Tensor:
    """Broadcast ``[batch, groups, state]`` to ``[batch, heads, state]``."""
    if value.ndim != 3:
        raise ValueError("group parameter must have shape [batch, groups, state]")
    groups = value.shape[1]
    if num_heads % groups:
        raise ValueError("num_heads must be divisible by groups")
    return value.repeat_interleave(num_heads // groups, dim=1)


def causal_conv_step(
    xbc: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Append one token, run depthwise causal convolution, and apply SiLU."""
    if xbc.ndim != 2 or conv_state.ndim != 3:
        raise ValueError("xbc and conv_state must be [batch, channels] and [batch, channels, kernel]")
    batch, channels = xbc.shape
    if conv_state.shape[:2] != (batch, channels):
        raise ValueError("conv_state batch/channel shape does not match xbc")
    kernel = conv_state.shape[-1]
    if weight.shape == (channels, 1, kernel):
        weight = weight[:, 0, :]
    _require_shape("conv_weight", weight, (channels, kernel))

    new_state = torch.roll(conv_state.float(), shifts=-1, dims=-1)
    new_state[..., -1] = xbc.float()
    output = (new_state * weight.float().unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        _require_shape("conv_bias", bias, (channels,))
        output = output + bias.float()
    return functional.silu(output), new_state


def selective_state_step(
    x: Tensor,
    b_group: Tensor,
    c_group: Tensor,
    dt_raw: Tensor,
    state: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    d_skip: Tensor,
    *,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Tensor]:
    """One Mamba-2 selective recurrence with FP32 update and reduction.

    The output uses the newly computed FP32 state. Quantization is applied only
    when that state is persisted for the next token.
    """
    if x.ndim != 3 or state.ndim != 4:
        raise ValueError("x and state must be [batch, heads, p] and [batch, heads, p, n]")
    batch, heads, head_dim = x.shape
    state_dim = state.shape[-1]
    _require_shape("state", state, (batch, heads, head_dim, state_dim))
    _require_shape("dt_raw", dt_raw, (batch, heads))
    _require_shape("a_log", a_log, (heads,))
    _require_shape("dt_bias", dt_bias, (heads,))
    _require_shape("d_skip", d_skip, (heads,))

    b = expand_group_parameter(b_group.float(), heads)
    c = expand_group_parameter(c_group.float(), heads)
    _require_shape("B", b, (batch, heads, state_dim))
    _require_shape("C", c, (batch, heads, state_dim))

    dt = functional.softplus(dt_raw.float() + dt_bias.float())
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    a = -torch.exp(a_log.float())
    decay = torch.exp(dt[:, :, None, None] * a[None, :, None, None])
    update = dt[:, :, None, None] * b[:, :, None, :] * x.float()[:, :, :, None]
    state_fp32 = state.float() * decay + update
    y = (state_fp32 * c[:, :, None, :]).sum(dim=-1)
    y = y + x.float() * d_skip.float()[None, :, None]
    return y, quantize_state(state_fp32, state_storage)


def mamba_state_engine_step(
    projected: Tensor,
    state: Mamba2State,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    d_skip: Tensor,
    shape: Mamba2Shape,
    *,
    conv_bias: Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Mamba2State]:
    """Execute exactly the X_STATE Mamba boundary for one projected token.

    The projection's gate segment is deliberately ignored here. Gating,
    group RMSNorm, and output projection remain Vector/Matrix operations.
    """
    if projected.ndim != 2:
        raise ValueError("projected token must have shape [batch, projection_size]")
    batch = projected.shape[0]
    _require_shape("projected", projected, (batch, shape.projection_size))
    _, xbc, dt = projected.float().split([shape.d_inner, shape.conv_channels, shape.num_heads], dim=-1)
    xbc, conv_state = causal_conv_step(xbc, state.conv, conv_weight, conv_bias)
    x, b, c = xbc.split(
        [
            shape.d_inner,
            shape.groups * shape.state_dim,
            shape.groups * shape.state_dim,
        ],
        dim=-1,
    )
    y, ssm_state = selective_state_step(
        x.reshape(batch, shape.num_heads, shape.head_dim),
        b.reshape(batch, shape.groups, shape.state_dim),
        c.reshape(batch, shape.groups, shape.state_dim),
        dt,
        state.ssm,
        a_log,
        dt_bias,
        d_skip,
        dt_limit=dt_limit,
        state_storage=state_storage,
    )
    persisted_conv = quantize_state(conv_state, state_storage)
    return y.reshape(batch, shape.d_inner), Mamba2State(ssm_state, persisted_conv)


def mamba_state_engine_prefill(
    projected: Tensor,
    state: Mamba2State,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    d_skip: Tensor,
    shape: Mamba2Shape,
    *,
    conv_bias: Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Mamba2State]:
    """Golden X_STATE PREFILL, defined as sequential STEP operations."""
    if projected.ndim != 3 or projected.shape[-1] != shape.projection_size:
        raise ValueError("projected prefill input must be [batch, sequence, projection_size]")
    outputs = []
    current = state
    for token in projected.unbind(dim=1):
        output, current = mamba_state_engine_step(
            token,
            current,
            conv_weight,
            a_log,
            dt_bias,
            d_skip,
            shape,
            conv_bias=conv_bias,
            dt_limit=dt_limit,
            state_storage=state_storage,
        )
        outputs.append(output)
    return torch.stack(outputs, dim=1), current


def gated_group_rms_norm(
    value: Tensor,
    gate: Tensor,
    weight: Tensor,
    *,
    groups: int,
    eps: float,
) -> Tensor:
    """Nemotron's ``norm_before_gate=False`` gated group RMSNorm."""
    if value.shape != gate.shape:
        raise ValueError("value and gate must have the same shape")
    if value.shape[-1] % groups:
        raise ValueError("last dimension must be divisible by groups")
    _require_shape("norm_weight", weight, (value.shape[-1],))
    gated = value.float() * functional.silu(gate.float())
    group_size = value.shape[-1] // groups
    grouped = gated.reshape(*gated.shape[:-1], groups, group_size)
    inverse_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    return (grouped * inverse_rms).reshape_as(gated) * weight.float()


def mamba_step_from_projection(
    projected: Tensor,
    state: Mamba2State,
    weights: Mamba2Weights,
    shape: Mamba2Shape,
    *,
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Mamba2State]:
    """Execute one token after the input projection."""
    if projected.ndim != 2:
        raise ValueError("projected token must have shape [batch, projection_size]")
    batch = projected.shape[0]
    _require_shape("projected", projected, (batch, shape.projection_size))
    gate, xbc, dt = projected.float().split([shape.d_inner, shape.conv_channels, shape.num_heads], dim=-1)
    xbc, conv_state = causal_conv_step(xbc, state.conv, weights.conv_weight, weights.conv_bias)
    x, b, c = xbc.split([shape.d_inner, shape.groups * shape.state_dim, shape.groups * shape.state_dim], dim=-1)
    y, ssm_state = selective_state_step(
        x.reshape(batch, shape.num_heads, shape.head_dim),
        b.reshape(batch, shape.groups, shape.state_dim),
        c.reshape(batch, shape.groups, shape.state_dim),
        dt,
        state.ssm,
        weights.a_log,
        weights.dt_bias,
        weights.d_skip,
        dt_limit=weights.dt_limit,
        state_storage=state_storage,
    )
    normalized = gated_group_rms_norm(
        y.reshape(batch, shape.d_inner),
        gate,
        weights.norm_weight,
        groups=shape.groups,
        eps=shape.norm_eps,
    )
    output = functional.linear(
        normalized,
        weights.out_proj_weight.float(),
        weights.out_proj_bias.float() if weights.out_proj_bias is not None else None,
    )
    return output, Mamba2State(ssm_state, conv_state)


def mamba_step(
    hidden: Tensor,
    state: Mamba2State,
    weights: Mamba2Weights,
    shape: Mamba2Shape,
    *,
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Mamba2State]:
    """Execute the complete one-token Nemotron Mamba layer."""
    if hidden.ndim != 2:
        raise ValueError("hidden token must have shape [batch, hidden_size]")
    projected = functional.linear(
        hidden.float(),
        weights.in_proj_weight.float(),
        weights.in_proj_bias.float() if weights.in_proj_bias is not None else None,
    )
    return mamba_step_from_projection(projected, state, weights, shape, state_storage=state_storage)


def mamba_prefill_sequential(
    hidden: Tensor,
    state: Mamba2State,
    weights: Mamba2Weights,
    shape: Mamba2Shape,
    *,
    state_storage: StateStorage | str = StateStorage.FP32,
) -> tuple[Tensor, Mamba2State]:
    """Golden prefill implemented as repeated decode steps."""
    if hidden.ndim != 3 or hidden.shape[-1] != shape.hidden_size:
        raise ValueError("hidden prefill input must be [batch, sequence, hidden_size]")
    outputs = []
    current = state
    for token in hidden.unbind(dim=1):
        output, current = mamba_step(token, current, weights, shape, state_storage=state_storage)
        outputs.append(output)
    return torch.stack(outputs, dim=1), current


def affine_scan_chunked(
    decay: Tensor,
    update: Tensor,
    initial_state: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Inclusive chunked scan for ``s[t] = decay[t] * s[t-1] + update[t]``.

    This deliberately simple implementation exposes the same affine pair
    composition used by an RTL prefill engine while remaining a CPU oracle.
    """
    if decay.shape != update.shape or decay.ndim < 1:
        raise ValueError("decay and update must have the same [sequence, ...] shape")
    if tuple(initial_state.shape) != tuple(decay.shape[1:]):
        raise ValueError("initial_state must match decay.shape[1:]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    outputs: list[Tensor] = []
    boundary = initial_state.float()
    for start in range(0, decay.shape[0], chunk_size):
        chunk_decay = decay[start : start + chunk_size].float()
        chunk_update = update[start : start + chunk_size].float()
        prefix_a = torch.ones_like(boundary)
        prefix_b = torch.zeros_like(boundary)
        for a_t, b_t in zip(chunk_decay, chunk_update, strict=True):
            prefix_a = a_t * prefix_a
            prefix_b = a_t * prefix_b + b_t
            outputs.append(prefix_a * boundary + prefix_b)
        boundary = outputs[-1]
    stacked = torch.stack(outputs, dim=0) if outputs else decay.clone()
    return stacked, boundary
