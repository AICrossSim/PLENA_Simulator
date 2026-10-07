"""State-storage quantizers for the Nemotron 3 Mamba CPU reference.

All arithmetic outside :func:`quantize_state` remains FP32. ``MX8_B128`` is a
PLENA design candidate with one power-of-two scale per 128 state values. It is
deliberately named differently from OCP MXFP8, whose standard block size is 32.
"""

from __future__ import annotations

import math
from enum import StrEnum

import torch
from torch import Tensor


class StateStorage(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    MX8_B128 = "mx8_b128"


def _round_e4m3fn(value: Tensor) -> Tensor:
    """Round finite FP32 values to the E4M3FN value grid and return FP32."""
    sign = torch.sign(value)
    magnitude = value.abs().clamp(max=448.0)
    minimum_normal = 2.0**-6
    subnormal_step = 2.0**-9

    subnormal = torch.round(magnitude / subnormal_step) * subnormal_step
    safe = magnitude.clamp_min(minimum_normal)
    exponent = torch.floor(torch.log2(safe)).clamp(min=-6, max=8)
    normal_step = torch.pow(torch.tensor(2.0, device=value.device), exponent - 3)
    normal = torch.round(magnitude / normal_step) * normal_step
    rounded = torch.where(magnitude < minimum_normal, subnormal, normal)
    return sign * rounded.clamp(max=448.0)


def quantize_mx8_b128(value: Tensor, *, block_size: int = 128) -> Tensor:
    """Round-trip through E4M3FN with one E8M0-like scale per last-axis block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    original_shape = value.shape
    if not original_shape:
        raise ValueError("MX8 block quantization requires at least one dimension")

    source = value.float()
    last = original_shape[-1]
    blocks = math.ceil(last / block_size)
    padded_last = blocks * block_size
    if padded_last != last:
        source = torch.nn.functional.pad(source, (0, padded_last - last))
    blocked = source.reshape(*source.shape[:-1], blocks, block_size)
    maximum = blocked.abs().amax(dim=-1, keepdim=True)

    # Ceil prevents finite E4M3 values from overflowing 448. E8M0 can encode
    # negative powers as well, so small state blocks still use the mantissa.
    scale_exponent = torch.ceil(torch.log2((maximum / 448.0).clamp_min(2.0**-126)))
    scale_exponent = scale_exponent.clamp(min=-126, max=127)
    scale_exponent = torch.where(maximum == 0, torch.zeros_like(scale_exponent), scale_exponent)
    scale = torch.pow(torch.tensor(2.0, device=value.device), scale_exponent)
    restored = _round_e4m3fn(blocked / scale) * scale
    return restored.reshape(*source.shape)[..., :last].reshape(original_shape)


def quantize_state(
    value: Tensor,
    storage: StateStorage | str,
    *,
    block_size: int = 128,
) -> Tensor:
    """Return the FP32 value seen after writing and reading the state store."""
    storage = StateStorage(storage)
    value = value.float()
    if storage == StateStorage.FP32:
        return value.clone()
    if storage == StateStorage.BF16:
        return value.to(torch.bfloat16).float()
    if storage == StateStorage.FP16:
        return value.to(torch.float16).float()
    if storage == StateStorage.MX8_B128:
        return quantize_mx8_b128(value, block_size=block_size)
    raise ValueError(f"unsupported state storage: {storage}")


def storage_bytes(elements: int, storage: StateStorage | str, *, block_size: int = 128) -> int:
    if elements < 0:
        raise ValueError("elements must be non-negative")
    storage = StateStorage(storage)
    if storage == StateStorage.FP32:
        return 4 * elements
    if storage in {StateStorage.BF16, StateStorage.FP16}:
        return 2 * elements
    return elements + math.ceil(elements / block_size)
