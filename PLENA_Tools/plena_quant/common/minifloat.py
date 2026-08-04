"""Minifloat quantization functions for hardware simulation."""

import torch
from torch import Tensor

from .utils import ste_clamp, ste_round


def _minifloat_denorm_quantize_hardware(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias: int | None = None,
):
    """
    - Converts IEEE FP32/64 to minifloat without the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has no implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example:
    a minifloat(bits=8, exponent_bits=4, mantissa_bits=3) number,
    1 0111 011, is equal to (-1)**1 * 2**(7-15) * (3/8) = -0.00146484375

    ---
    Tested extreme values: large values to saturate, small values close to zero (precision), and 0
    """
    mantissa_bits = width - exponent_width - 1

    # default bias value
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias
    # if the mantissa is an integer, the max mantissa value will be (2**mantissa_bits -1)
    shifted_mantissa_max = 2**mantissa_bits - 1
    shifted_mantissa_min = 0

    sign = torch.sign(x + 1e-9)

    value = torch.abs(x)
    # ceiling ensures mantissa in the range of [0, 1)
    exponent = torch.ceil(torch.log2(value + 1e-9))
    exponent = ste_clamp(exponent, exponent_min, exponent_max)

    # divide value by clipped exponent. this ensures the simulated minifloat value is correct
    # when x is too large (minifloat will saturate) or too close to 0.
    mantissa = value / 2**exponent
    shift = 2**mantissa_bits
    shifted_mantissa = ste_round(mantissa * shift)
    # clip the integer mantissa.
    shifted_mantissa = ste_clamp(shifted_mantissa, shifted_mantissa_min, shifted_mantissa_max)
    mantissa = shifted_mantissa / shift
    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(value, torch.tensor([0.0], dtype=value.dtype, device=value.device))
    minifloat_denorm_x = (~is_close_to_0)*(sign*(2**exponent)*mantissa) + is_close_to_0*x
    # fmt: on
    return minifloat_denorm_x, exponent, sign * mantissa


def _minifloat_ieee_quantize_hardware(x: Tensor, width: int, exponent_width: int, exponent_bias: int | None = None):
    """
    - Converts IEEE FP32/64 to minifloat with the implicit leading bit in mantissas.
    - No representation for +/-inf or NaN. Large IEEE FP32/64 values will saturate.

    ---
    - forward: convert IEEE FP32/64 to minifloat (mantissa has an implicit leading bit)
    - backward: STE

    ---
    width: the bit width of minifloat
    exponent_width: the number of exponent bits in the minifloat
    exponent_bias: the value of the exponent bias. If None, the default bias will be (2**exponent_bits - 1) >> 1.

    ---
    For example, E4M3 bits ``1 0111 011`` represent ``-1.375``.

    ---

    Tested extreme cases: large values to saturate, small normal values, small subnormal values, normal precision, subnormal precision, and 0
    """
    mantissa_bits = width - exponent_width - 1

    # set default bias
    if exponent_bias in (None, "none", "None"):
        exponent_bias = (
            1 if exponent_width == 1 else 2 ** (exponent_width - 1) - 1
        )
    max_exponent_code = (
        2**exponent_width - 1
        if exponent_width == 1
        else 2**exponent_width - 2
    )
    exponent_max = max_exponent_code - exponent_bias
    exponent_min = 1 - exponent_bias
    fraction_levels = 2**mantissa_bits
    fraction_max = fraction_levels - 1

    value = torch.abs(x)
    finite_nonzero = torch.isfinite(value) & (value != 0)
    safe_value = torch.where(finite_nonzero, value, torch.ones_like(value))
    raw_exponent = torch.floor(torch.log2(safe_value))
    min_normal = float(2**exponent_min)
    is_subnormal = finite_nonzero & (value < min_normal)
    exponent = ste_clamp(raw_exponent, exponent_min, exponent_max)

    normal_fraction = ste_round(
        (safe_value / 2**exponent - 1.0) * fraction_levels
    )
    normal_carry = normal_fraction >= fraction_levels
    normal_fraction = torch.where(
        normal_carry,
        torch.zeros_like(normal_fraction),
        normal_fraction,
    )
    normal_fraction = ste_clamp(normal_fraction, 0, fraction_max)
    normal_exponent = exponent + normal_carry.to(exponent.dtype)

    subnormal_fraction = ste_round(value / min_normal * fraction_levels)
    subnormal_carry = subnormal_fraction >= fraction_levels
    subnormal_fraction = torch.where(
        subnormal_carry,
        torch.zeros_like(subnormal_fraction),
        subnormal_fraction,
    )
    subnormal_fraction = ste_clamp(subnormal_fraction, 0, fraction_max)

    overflow = (~torch.isfinite(value)) | (
        finite_nonzero
        & ((raw_exponent > exponent_max) | (normal_exponent > exponent_max))
    )
    quantized_exponent = torch.where(
        is_subnormal,
        torch.full_like(exponent, exponent_min),
        normal_exponent,
    )
    encoded_exponent = torch.where(
        is_subnormal & ~subnormal_carry,
        torch.full_like(exponent, -exponent_bias),
        quantized_exponent,
    )
    fraction = torch.where(
        is_subnormal,
        subnormal_fraction,
        normal_fraction,
    )
    mantissa = torch.where(
        is_subnormal & ~subnormal_carry,
        fraction / fraction_levels,
        1.0 + fraction / fraction_levels,
    )

    encoded_exponent = torch.where(
        overflow,
        torch.full_like(encoded_exponent, exponent_max),
        encoded_exponent,
    )
    mantissa = torch.where(
        overflow,
        torch.full_like(mantissa, 1.0 + fraction_max / fraction_levels),
        mantissa,
    )
    encoded_exponent = torch.where(
        finite_nonzero | overflow,
        encoded_exponent,
        torch.full_like(encoded_exponent, -exponent_bias),
    )
    mantissa = torch.where(
        finite_nonzero | overflow,
        mantissa,
        torch.zeros_like(mantissa),
    )

    negative = torch.signbit(x)
    signed_mantissa = torch.where(negative, -mantissa, mantissa)
    minifloat_ieee_x = signed_mantissa * 2**torch.where(
        overflow,
        torch.full_like(quantized_exponent, exponent_max),
        quantized_exponent,
    )
    minifloat_ieee_x = torch.where(value == 0, x, minifloat_ieee_x)
    return minifloat_ieee_x, encoded_exponent, signed_mantissa
