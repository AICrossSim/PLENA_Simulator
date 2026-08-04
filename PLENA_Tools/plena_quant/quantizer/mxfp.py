"""Compatibility wrapper for the canonical MXFP quantizer."""

import torch
from torch import Tensor

from ..mxfp.quantizer import _mx_fp_quantize_hardware


def _mx_fp_quantize(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: list[int] | int = [16],
    skip_first_dim: bool = False,
):
    """Return the canonical OCP MXFP quantize-dequantize result."""
    normalized_block_size = (
        [block_size] if isinstance(block_size, int) else block_size
    )
    quantized, _, _, _ = _mx_fp_quantize_hardware(
        x,
        width=width,
        exponent_width=exponent_width,
        exponent_bias_width=exponent_bias_width,
        block_size=normalized_block_size,
        skip_first_dim=skip_first_dim,
    )
    return quantized


class MXFPQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        width: int,
        exponent_width: int,
        exponent_bias_width: int,
        block_size: list[int] | int = [16],
        skip_first_dim: bool = False,
    ):
        return _mx_fp_quantize(
            x,
            width=width,
            exponent_width=exponent_width,
            exponent_bias_width=exponent_bias_width,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None, None, None, None, None


def mxfp_quantizer(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: list[int] | int = [16],
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM), where an exponent bias is shared over all elements in a block
    - `2**-bias_shared x [(-1)^s1 x 2^exponent1 x mantissa1, (-1)^s2 x 2^exponent2 x mantissa2, ...]`
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `width`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_width`: the number of exponent_bits
    - `exponent_bias_width`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    return MXFPQuantize.apply(
        x,
        width,
        exponent_width,
        exponent_bias_width,
        block_size,
        skip_first_dim,
    )
