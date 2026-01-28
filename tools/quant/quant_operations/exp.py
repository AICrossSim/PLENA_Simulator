from quant.quantizer import integer
from sympy import Q
import torch
from torch import Tensor
import math

from quant.quantizer.hardware_quantizer.utils import fixed_point_cast
from torch._refs import to
from quant.quantizer.hardware_quantizer import _minifloat_ieee_quantize_hardware
import logging
from cfl_tools.logger import get_logger
logger = get_logger(__name__)
logger_level = logging.DEBUG
logger.setLevel(logger_level)


def hardware_round(x: torch.Tensor):
    """Round to nearest with hardware-like behavior"""
    x_sign = x.sign()
    x_abs = x.abs()
    x_abs_rounded = ((x_abs * 4).floor() / 4).round()
    return x_sign * x_abs_rounded

def hardware_dynamic_shift(x: torch.Tensor, shift_amt: torch.Tensor, out_width):
    """
    Dynamic shift of x by shift_amt
    """
    x_sign = x.sign()
    x_abs = x.abs()
    max_data = 2**(out_width - 1) - 1
    x_abs_shifted = (x_abs * (2** shift_amt)).floor()
    x_abs_shifted = torch.clamp(x_abs_shifted, 0, max_data)
    return x_sign * x_abs_shifted

def fp_exp_hardware(signed_exp_in: torch.Tensor, signed_mant_in: torch.Tensor, config: dict):
    """
    Hardware model of the fp_exp.sv module
    
    Algorithm:
    1. Multiply mantissa by log2(e) constant
    2. Apply exponent scaling  
    3. Extract integer and fractional parts
    4. Apply Taylor series to fractional part: 2^f ≈ 1 + ln(2)*f + ln²(2)*f²/2! + ln³(2)*f³/3!
    5. Return integer part as exponent and Taylor result as mantissa
    """
    in_exp_width = config["in_exp_width"]
    in_fix_width = config["in_fix_width"] 
    in_fix_frac_width = config["in_fix_frac_width"]
    extend_width = config.get("extend_width", 0)
    out_exp_width = config["out_exp_width"]
    out_fix_width = config["out_fix_width"]
    out_fix_frac_width = config["out_fix_frac_width"]

    # Step 1: Multiply mantissa by MLOG2_E (log2(e) coefficient)
    # MLOG2_E = 92 in hardware (this is log2(e) * 2^6 for Q1.6 format)
    MLOG2_E = 92/2**7
    ELOG2_E = 1
    
    # Convert signed mantissa to unsigned for multiplication
    # Multiply by log2(e) coefficient (fixed point multiplication)
    signed_mant_log2_e = signed_mant_in * MLOG2_E * (2**(in_fix_frac_width + extend_width))
    signed_mant_log2_e = hardware_round(signed_mant_log2_e) / (2**(in_fix_frac_width + extend_width))
    logger.debug(f"signed_mant_in: {signed_mant_in}")
    logger.debug(f"signed_mant_log2_e: {signed_mant_log2_e}")
    # Adjust exponent
    signed_exp_log2_e = signed_exp_in + ELOG2_E

    
    # Step 2: Apply exponent scaling (shift mantissa by exponent)
    # This creates fixed point data with integer and fractional parts
    FIXED_POINT_WIDTH = in_fix_width + 10
    FIX_POINT_MAX = (2**(FIXED_POINT_WIDTH - 1)- 1)/2**(in_fix_frac_width)
    FIX_POINT_MIN = -2**(FIXED_POINT_WIDTH - 1)/2**(in_fix_frac_width)

    fixed_point_data = signed_mant_log2_e * (2 ** signed_exp_log2_e)
    logger.debug(f"fixed_point_data: {fixed_point_data}")
    taylor_frac_width = in_fix_frac_width + extend_width
    fixed_point_data = hardware_dynamic_shift(signed_mant_log2_e*2**(taylor_frac_width), signed_exp_log2_e, FIXED_POINT_WIDTH)/2**(taylor_frac_width)
    fixed_point_data = torch.clamp(fixed_point_data, FIX_POINT_MIN, FIX_POINT_MAX)
    logger.debug(f"fixed_point_data: {fixed_point_data}")
    
    # Step 3: Extract integer and fractional parts
    fixed_point_int_part = fixed_point_data.floor()
    fixed_point_frac_part = fixed_point_data - fixed_point_int_part
    
    # Step 4: Apply Taylor series to fractional part
    logger.debug(f"fixed_point_int_part: {fixed_point_int_part}, fixed_point_frac_part: {fixed_point_frac_part}")
    taylor_result = taylor_series_hardware(fixed_point_frac_part, taylor_frac_width)

    # Step 5: Return results
    # The RTL assigns signed_exp_out = signed_exp_in (pass through)
    # The RTL assigns signed_mant_out = taylor_output
    output_exp = fixed_point_int_part
    output_mant = (taylor_result * 2**(in_fix_frac_width)).floor()/2**(in_fix_frac_width)
    logger.debug(f"fixed_point_frac_part: {fixed_point_frac_part}, taylor_result: {taylor_result}")
    
    return output_exp, output_mant

def taylor_series_hardware(x: torch.Tensor, frac_width: int):
    """
    Taylor series expansion of 2^x for testing
    """
    # ln(2) coefficient in fixed point (Q7.5 format)
    ln2 = torch.tensor(22) / 2**5
    
    # Calculate Taylor series terms: 2^x ≈ 1 + ln(2)*x + ln²(2)*x²/2! + ln³(2)*x³/3!
    term0 = 1.0

    term1 = x * ln2 * (2**(frac_width))
    term1 = hardware_round(term1) / (2**(frac_width))
    
    term2 = (term1 * term1 * 2**(frac_width)) 
    term2 = hardware_round(term2) //2 / (2**(frac_width))
    term3_inter = ((term2 * term1 )*2**(frac_width)) 
    term3_inter = hardware_round(term3_inter) / 2**(frac_width)
    term3 = term3_inter / 3 * 2**(frac_width)
    term3 = hardware_round(term3) / (2**(frac_width))

    logger.debug(f"term0: {term0 * 2**(frac_width)}, term1: {term1 * 2**(frac_width)}, term2: {term2 * 2**(frac_width)}, term3: {term3 * 2**(frac_width)}")
    return term0 + term1 + term2 + term3



def tayor_exp(x: torch.Tensor):
    """
    Taylor series expansion of 2^x for testing
    """
    def range_reduction(x: torch.Tensor):
        """
        Range reduction of x
        """
        MLOG2_E = 92/2**7
        ELOG2_E = 1
        new_mx = x * MLOG2_E * 2
        logger.debug(f"new_mx: {new_mx}")
        integ = new_mx.floor()
        frac = new_mx - integ
        return frac, integ

    def taylor_series(frac: torch.Tensor):
        """
        Taylor series expansion of 2^x for testing
        """
        ln2 = 22/2**5
        term0 = 1.0
        term1 = frac * ln2
        term2 = term1 * term1 / 2
        term3 = term2 * term1 / 3
        return term0 + term1 + term2 + term3

    frac, integ = range_reduction(x)
    logger.debug(f"frac: {frac}, integ: {integ}")
    taylor_result = taylor_series(frac)
    return taylor_result * 2**integ

    

def test_exp():
    """Test exponential function with simple cases."""
    config = {
        "in_exp_width": 4,
        "in_fix_width": 5,
        "in_fix_frac_width": 3,
        "out_exp_width": 4,
        "out_fix_width": 6,
        "out_fix_frac_width": 4,
    }
    
    # Test cases: [0, 1, -1, 2, 0.5, -0.5]
    inputs = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5, -0.5])
    
    # Quantize inputs
    qdata_in, exp_in, mant_in = _minifloat_ieee_quantize_hardware(
        inputs, 
        config["in_fix_frac_width"] + config["in_exp_width"] + 1, 
        config["in_exp_width"]
    )
    golden_exp = torch.exp(qdata_in)
    logger.debug(f"---taylor_exp---")
    taylor_result = tayor_exp(qdata_in)
    logger.debug(f"---fp_exp_hardware---")
    logger.debug(f"""
    taylor_result: {taylor_result}, 
    golden_exp: {golden_exp}
    """)
    qtaylor_result, taylor_exp, taylor_mant = _minifloat_ieee_quantize_hardware(
        taylor_result, 
        config["out_fix_frac_width"] + config["out_exp_width"] + 1, 
        config["out_exp_width"]
    )
    hardware_exp, hardware_mant = fp_exp_hardware(exp_in, mant_in, config)
    hardware_result = hardware_mant * (2 ** hardware_exp)
    logger.debug(f"---hardware_result---")
    logger.debug(f"""
    taylor_result: {qtaylor_result}, 
    hardware_result: {hardware_result}
    """)


if __name__ == "__main__":
    test_exp()