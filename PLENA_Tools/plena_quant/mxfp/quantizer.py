"""MXFP quantization functions."""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..common.utils import ste_clamp
from ..common.minifloat import _minifloat_ieee_quantize_hardware


def _mx_fp_quantize_hardware(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: list[int] | int = [16],
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to MXFP with one shared exponent per block.
    - Dequantized values are ``2**shared_exponent * element``.
    - See https://openreview.net/forum?id=6zaTwpNSsQ2

    ---
    - forward: convert IEEE FP32/64 to BM
    - backward: STE

    ---
    - `width`: the number of bits (1 sign bit + exponent_bits + mantissa_bits)
    - `exponent_width`: the number of exponent_bits
    - `exponent_bias_width`: the number of bits of the shared exponent bias
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    The shared exponent follows the OCP MX convention: the block maximum is
    placed at the top of the element format's finite exponent range. This is
    a software accuracy convention; the datapath consumes the resulting E8M0
    scale code without selecting a scale-placement rule.

    """
    if isinstance(block_size, int):
        block_size = [block_size]
    if len(block_size) == 1:
        block_size = [1, block_size[0]]
    else:
        assert len(block_size) == 2, "block_size must be a list of two integers"

    x_shape = x.shape
    # Pre-compute padding requirements
    x_pad_size_0 = (block_size[0] - (x_shape[-2] % block_size[0])) % block_size[0]
    x_pad_size_1 = (block_size[1] - (x_shape[-1] % block_size[1])) % block_size[1]

    # Pad x if needed
    px = F.pad(x, (0, x_pad_size_1, 0, x_pad_size_0), "constant", 0)
    px_shape = px.shape

    # in order to follow the law of torch.mm
    # px will be reshaped to (-1, number_of_blocks_0, block_size[0], number_of_blocks_1, block_size[1])
    # and be view as (-1, number_of_blocks_0, number_of_blocks_1, block_size[0], block_size[1])
    px = px.view(
        -1, px_shape[-2] // block_size[0], block_size[0], px_shape[-1] // block_size[1], block_size[1]
    ).permute(0, 1, 3, 2, 4)
    px = px.reshape(-1, block_size[0] * block_size[1])

    per_block_max = px.abs().max(dim=-1, keepdim=True).values
    nonzero_block = per_block_max > 0
    safe_block_max = torch.where(
        nonzero_block,
        per_block_max,
        torch.ones_like(per_block_max),
    )
    scale_bias = 2 ** (exponent_bias_width - 1) - 1
    scale_max = 2**exponent_bias_width - 1 - scale_bias
    element_exponent_bias = (
        1 if exponent_width == 1 else 2 ** (exponent_width - 1) - 1
    )
    element_max_exponent_code = (
        2**exponent_width - 1
        if exponent_width == 1
        else 2**exponent_width - 2
    )
    element_max_exponent = element_max_exponent_code - element_exponent_bias
    per_block_exponent_bias = ste_clamp(
        torch.floor(torch.log2(safe_block_max)) - element_max_exponent,
        -scale_bias,
        scale_max,
    )
    per_block_exponent_bias = torch.where(
        nonzero_block,
        per_block_exponent_bias,
        torch.zeros_like(per_block_exponent_bias),
    )

    px = px / 2**per_block_exponent_bias
    per_block_bm_x, per_block_fp_exp, per_block_fp_mant = _minifloat_ieee_quantize_hardware(
        px,
        width=width,
        exponent_width=exponent_width,
    )

    per_block_bm_x = per_block_bm_x * 2**per_block_exponent_bias

    bm_x = per_block_bm_x.reshape(
        -1,
        px_shape[-2] // block_size[0],
        px_shape[-1] // block_size[1],
        block_size[0],
        block_size[1],
    )
    bm_x = bm_x.permute(0, 1, 3, 2, 4)
    bm_x = bm_x.reshape(-1, px_shape[-2], px_shape[-1])
    bm_x = bm_x[:, : x_shape[-2], : x_shape[-1]]

    per_block_exponent_bias = per_block_exponent_bias + scale_bias

    return bm_x, per_block_fp_exp, per_block_fp_mant, per_block_exponent_bias
