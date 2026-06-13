"""Floating point conversion utilities."""

import torch


def pack_fp_to_bin(signed_exponent, signed_mantissa, exp_width, man_width):
    exp_shape = signed_exponent.shape
    signed_exponent = signed_exponent.reshape(-1)
    signed_mantissa = signed_mantissa.reshape(-1)

    sign = signed_mantissa.sign()
    sign_bit = torch.where(sign < 0, torch.tensor(1), torch.tensor(0))

    exponent_bias = (2 ** (exp_width - 1)) - 1
    exponent_bit = signed_exponent + exponent_bias

    if not torch.all((exponent_bit >= 0) & (exponent_bit <= (2**exp_width - 1))):
        raise AssertionError("Exponent out of range!")

    mantissa = torch.where(signed_mantissa < 0, -signed_mantissa, signed_mantissa)
    mantissa_bit = torch.where(exponent_bit == 0, mantissa, mantissa - 1)

    mantissa_bit = mantissa_bit * 2 ** (man_width)

    result = ((sign_bit * 2 ** (exp_width + man_width)) + exponent_bit * 2 ** (man_width) + mantissa_bit).int()

    result = result.reshape(exp_shape)

    return result
