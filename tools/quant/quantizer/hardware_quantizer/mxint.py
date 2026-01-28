import torch
from torch import Tensor

from quant.quantizer.utils import block, my_clamp, unblock, my_round
from quant.quantizer.hardware_quantizer.minifloat import _minifloat_ieee_quantize_hardware
from cfl_tools.debugger import set_excepthook
from torch.nn import functional as F

set_excepthook()

def _mx_int_quantize_hardware(
    x: Tensor,
    width: int = 12,
    exponent_width: int = 8,
    exponent_bias: int = None,
    block_size: list[int] = [16],
    skip_first_dim: bool = True,
):
    """
    - Convert IEEE FP32/64 to Microscaling Integer (MXINT) which is also called block_floating_point or MSFP, where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    if isinstance(block_size, int):
        block_size = [block_size]
    # separate x into blocks
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        # all elements in zero-initialized bias can be 0 thus per_block_max is 0
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()
    # minifloat_denorm_quantizer on each block over which a exponent is shared
    mantissa_bits = width - 1
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    mantissa_integer_max = 2**mantissa_bits - 1
    # sign
    per_block_sign = torch.sign(blocked_x + 1e-9)
    # exponent
    per_block_value = torch.abs(blocked_x) + 1e-9
    per_block_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_exponent = my_clamp(per_block_exponent, exponent_min, exponent_max)
    # mantissa
    per_block_mantissa = per_block_value / 2**per_block_exponent
    shift = 2**mantissa_bits
    per_block_mantissa_integer = my_clamp(
        my_round(per_block_mantissa * shift), 0, mantissa_integer_max
    )
    per_block_mantissa = per_block_mantissa_integer / shift

    per_block_msfp = per_block_sign * (2**per_block_exponent) * per_block_mantissa
    scaling = per_block_exponent
    msfp_x = unblock(
        per_block_msfp,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    msfp_x = (~is_close_to_0) * msfp_x + (is_close_to_0) * x
    # fmt: on
    return msfp_x, per_block_mantissa, scaling

def test_bin_mxint():
    x = torch.randn([4, 16,8])
    exp_bias_width = 4
    exp_width = 4
    mant_width = 3
    width = exp_width + mant_width + 1
    bm_x, per_block_fp_exp, per_block_fp_mant, per_block_exponent_bias = _mx_fp_quantize_hardware(
        x, width, exp_width, exp_bias_width, [4]
    )
    print(bm_x)
    print(per_block_fp_exp.shape)
    print(per_block_fp_mant.shape)
    print(per_block_exponent_bias.shape)

    from quant.quantizer.hardware_quantizer.utils import pack_fp_to_bin
    fp_bin = pack_fp_to_bin(per_block_fp_exp, per_block_fp_mant, exp_width, mant_width)
    print(fp_bin.shape)
    

def test_functionality():
    x = torch.randn([4, 16,8]) * 100 - 50
    exp_bias_width = 4
    exp_width = 1
    mant_width = 3
    width = exp_width + mant_width + 1
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        x, width, exp_width, exp_bias_width, [4]
    )
    from quant.quantizer.minifloat import _minifloat_ieee_quantize
    minifloat_x = _minifloat_ieee_quantize(
        x,
        width=width,
        exponent_width=exp_width,
    )

    from cfl_tools.debugger import _get_similarity
    print(_get_similarity(x, bm_x, metric="cosine").mean())
    print(_get_similarity(x, minifloat_x, metric="cosine").mean())


if __name__ == "__main__":
    test_functionality()