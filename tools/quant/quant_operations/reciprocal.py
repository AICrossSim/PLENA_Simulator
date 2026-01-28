
from quant.quantizer import integer
from sympy import Q
import torch
from torch import Tensor

from quant.quantizer.hardware_quantizer.utils import fixed_point_cast
from torch._refs import to

def fp_reciprocal(
        signed_exponent_in: torch.Tensor,
        signed_mantissa_in: torch.Tensor,
        config: dict
    ):
    in_fix_width = config["in_fix_width"]
    in_fix_frac_width = config["in_fix_frac_width"]
    in_exp_width = config["in_exp_width"]

    integer_mantissa_in = signed_mantissa_in * 2**(in_fix_frac_width)
    integer_exp = signed_exponent_in - in_fix_frac_width 

    out_fix_width = config["out_fix_width"]
    out_frac_width = config["out_fix_frac_width"]
    out_exp_width = config["out_exp_width"]

    reciprocal_mantissa = torch.zeros_like(integer_mantissa_in)
    reciprocal_mantissa[torch.where(integer_mantissa_in == 0)] = (1 * 2 ** (out_fix_width + in_fix_width - 1) - 1) / 2 ** (out_fix_width + in_fix_width - 1)
    reciprocal_mantissa[torch.where(integer_mantissa_in != 0)] = 1 / integer_mantissa_in[torch.where(integer_mantissa_in != 0)]

    ## currently the reciprocal is a 2**(out_width + in_fix_width - 1) - 1
    leading_zeros = (- reciprocal_mantissa.abs().log2()).ceil()
    extend_exp =  - integer_exp - leading_zeros
    # note here: the min max in ieee floating point is different with else where
    out_exp_min = - (2**(out_exp_width - 1) - 1)
    out_exp_max = 2**(out_exp_width-1) - 1
    out_exp = torch.clamp(extend_exp, min=out_exp_min, max=out_exp_max)
    exp_difference = extend_exp - out_exp

    output_mantissa_lossless = reciprocal_mantissa * 2 ** (leading_zeros) * 2**exp_difference 

    output_mantissa_integer = (output_mantissa_lossless * 2**(out_frac_width)).round()
    mantissa_max = 2**(out_frac_width+1) - 1
    # mantissa_min = -2**(out_fix_width-1)
    clamped_output_mantissa_integer = torch.clamp(output_mantissa_integer, min=-mantissa_max, max=mantissa_max)
    output_mantissa = clamped_output_mantissa_integer / 2**(out_frac_width)
    return out_exp, output_mantissa

def test_reciprocal():
    config = {
        "in_exp_width": 4,
        "in_fix_width": 5,
        "in_fix_frac_width": 3,
        "out_exp_width": 4,
        "out_fix_width": 6,
        "out_fix_frac_width": 4,
    }
    torch.manual_seed(0)
    # data_in = torch.tensor([0.001])
    data_in = torch.rand(100) * 100 - 50
    from quant.quantizer.hardware_quantizer import _minifloat_ieee_quantize_hardware
    qdata_in, exp_in, mant_in = _minifloat_ieee_quantize_hardware(
        data_in, 
        config["in_fix_frac_width"] + config["in_exp_width"]+ 1, 
        config["in_exp_width"])

    out = 1/qdata_in
    qout, out_exp, out_mant = _minifloat_ieee_quantize_hardware(
        out, 
        config["out_fix_frac_width"] + config["out_exp_width"]+ 1, 
        config["out_exp_width"])

    hardware_out_exp, hardware_out_mant = fp_reciprocal(exp_in, mant_in, config)

    golden_out = qout
    hardware_out = hardware_out_mant * 2**(hardware_out_exp)
    assert torch.allclose(golden_out, hardware_out), f"golden_out: {golden_out[torch.where(golden_out != hardware_out)]}, hardware_out: {hardware_out[torch.where(hardware_out != golden_out)]}"

if __name__ == "__main__":
    test_reciprocal()