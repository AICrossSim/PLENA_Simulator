import torch
from torch import Tensor

from quant.quantizer.utils import block, my_clamp, unblock, my_round
from quant.quantizer.hardware_quantizer.minifloat import _minifloat_ieee_quantize_hardware
from utils.debugger import set_excepthook
from torch.nn import functional as F

set_excepthook()

def _mx_fp_quantize_hardware(
    x: Tensor,
    width: int,
    exponent_width: int,
    exponent_bias_width: int,
    block_size: list[int] | int = [16],
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Block Minifloat (BM) which is also called as MXFP, where an exponent bias is shared over all elements in a block
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
    if len(block_size) == 1:
        block_size = [1, block_size[0]]
    else:
        assert len(block_size) == 2, "block_size must be a list of two integers"

    x_shape = x.shape
    # Pre-compute padding requirements
    x_pad_size_0 = (block_size[0] - (x_shape[-2] % block_size[0])) % block_size[0]
    x_pad_size_1 = (block_size[1] - (x_shape[-1] % block_size[1])) % block_size[1]
    
    # Pad x if needed 
    px = F.pad(x, (0, x_pad_size_1, 0, x_pad_size_0), 'constant', 0)
    px_shape = px.shape

    # in order to follow the law of torch.mm
    # px will be reshaped to (-1, number_of_blocks_0, block_size[0], number_of_blocks_1, block_size[1])
    # and be view as (-1, number_of_blocks_0, number_of_blocks_1, block_size[0], block_size[1])
    px = px.view(-1, px_shape[-2]//block_size[0], block_size[0], px_shape[-1]//block_size[1], block_size[1]).permute(0, 1, 3, 2, 4)
    px = px.reshape(-1, block_size[0] * block_size[1])

    per_block_max = px.abs().max(dim=-1, keepdim=True).values + 1e-9
    per_block_exponent_bias = my_clamp(
        torch.floor(torch.log2(per_block_max)), -2**(exponent_bias_width - 1), 2**(exponent_bias_width - 1) - 1
    )

    px = px / 2**per_block_exponent_bias
    per_block_bm_x, per_block_fp_exp, per_block_fp_mant = _minifloat_ieee_quantize_hardware(
        px,
        width=width,
        exponent_width=exponent_width,
    )

    per_block_bm_x = per_block_bm_x * 2**per_block_exponent_bias

    bm_x = per_block_bm_x.reshape(-1, px_shape[0]//block_size[0], px_shape[1]//block_size[1], block_size[0], block_size[1])
    bm_x = bm_x.permute(0, 1, 3, 2, 4)
    bm_x = bm_x.reshape(-1, px_shape[-2], px_shape[-1])
    bm_x = bm_x[:, :x_shape[-2], :x_shape[-1]]
    
    bias_bias = 2**(exponent_bias_width - 1) - 1
    per_block_exponent_bias = per_block_exponent_bias + bias_bias


    return bm_x, per_block_fp_exp, per_block_fp_mant, per_block_exponent_bias

def test_bin_mxfp():
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

    from utils.debugger import _get_similarity
    print(_get_similarity(x, bm_x, metric="cosine").mean())
    print(_get_similarity(x, minifloat_x, metric="cosine").mean())


if __name__ == "__main__":
    test_functionality()