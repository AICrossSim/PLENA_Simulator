"""MXINT-specific utility functions."""

import torch
from torch import Tensor


def pack_mxint_to_bin(signed_mantissa: Tensor, mantissa_width: int) -> Tensor:
    """
    Pack MXINT mantissa values to binary representation.

    Args:
        signed_mantissa: Signed mantissa values (normalized to [-1, 1))
        mantissa_width: Number of mantissa bits (excluding sign bit)

    Returns:
        Integer tensor with packed binary representation
    """
    sign = signed_mantissa.sign()
    sign_bit = torch.where(sign < 0, torch.tensor(1), torch.tensor(0))

    mantissa = torch.abs(signed_mantissa)
    mantissa_int = (mantissa * 2**mantissa_width).int()

    result = (sign_bit << mantissa_width) | mantissa_int
    return result.int()


def unpack_bin_to_mxint(packed: Tensor, mantissa_width: int) -> Tensor:
    """
    Unpack binary representation to MXINT mantissa values.

    Args:
        packed: Integer tensor with packed binary representation
        mantissa_width: Number of mantissa bits (excluding sign bit)

    Returns:
        Signed mantissa values (normalized to [-1, 1))
    """
    sign_bit = (packed >> mantissa_width) & 1
    mantissa_int = packed & ((1 << mantissa_width) - 1)

    mantissa = mantissa_int.float() / 2**mantissa_width
    sign = torch.where(sign_bit == 1, torch.tensor(-1.0), torch.tensor(1.0))

    return sign * mantissa
