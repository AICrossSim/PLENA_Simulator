
import torch
from torch import Tensor

from quant.quantizer.hardware_quantizer.utils import fixed_point_cast

def fp_add_hardware(
        a_exp: torch.Tensor, 
        a_mant: torch.Tensor, 
        b_exp: torch.Tensor, 
        b_mant: torch.Tensor, 
        config: dict
    ):
    out_fix_width = config["OUT_FIX_WIDTH"]
    out_fix_frac_width = config["OUT_FIX_FRAC_WIDTH"]
    out_exp_width = config["OUT_EXP_WIDTH"]
    floor = config["FLOOR"]

    a_greater = a_exp > b_exp
    # Calculate aligned exponent
    exp_sum = torch.where(a_greater, a_exp, b_exp)
    a_mant_shifted = a_mant / 2** (exp_sum - a_exp)
    b_mant_shifted = b_mant / 2** (exp_sum - b_exp)

    ## avoid loss here
    data_fix_width = out_fix_width - 1
    data_fix_frac_width = out_fix_frac_width
    a_mant_casted = fixed_point_cast(a_mant_shifted, data_fix_width, data_fix_frac_width, floor=floor)
    b_mant_casted = fixed_point_cast(b_mant_shifted, data_fix_width, data_fix_frac_width, floor=floor)

    mant_sum = a_mant_casted + b_mant_casted
    return exp_sum, mant_sum