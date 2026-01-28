import torch
from cfl_tools.logger import get_logger
from quant.quantizer.hardware_quantizer import _minifloat_ieee_quantize_hardware

logger = get_logger(__name__)

def fp_mult_hardware(
        exp_a: torch.Tensor,
        mant_a: torch.Tensor,
        exp_b: torch.Tensor,
        mant_b: torch.Tensor,
        IN_FIX_FRAC_WIDTH: int,
        OUT_FIX_FRAC_WIDTH: int,
        log,
):
    exp_out = exp_a + exp_b
    intermediate_mant = mant_a * mant_b
    mant_out = (intermediate_mant * 2**(OUT_FIX_FRAC_WIDTH)).floor()
    log.debug(f"software mant_out: {mant_out}")
    mant_out = mant_out / 2**(OUT_FIX_FRAC_WIDTH)

    return exp_out, mant_out

def fp_mult_software(a, b, config):
    a_exp_width = config["a_exp_width"]
    a_man_width = config["a_man_width"]
    b_exp_width = config["b_exp_width"]
    b_man_width = config["b_man_width"]
    out_exp_width = config["out_exp_width"]
    out_man_width = config["out_man_width"]

    qa, _, _ = _minifloat_ieee_quantize_hardware(a, a_man_width + a_exp_width + 1, a_exp_width)
    qb, _, _ = _minifloat_ieee_quantize_hardware(b, b_man_width + b_exp_width + 1, b_exp_width)

    out = qa * qb
    qout, out_exp, out_mant = _minifloat_ieee_quantize_hardware(out, out_man_width + out_exp_width + 1, out_exp_width)

    return qout
    
    

def test_fp_mult_hardware():
    exp_a = torch.tensor([1, 2, 3, 4])
    mant_a = torch.tensor([1, 2, 3, 4])
    exp_b = torch.tensor([1, 2, 3, 4])
    mant_b = torch.tensor([1, 2, 3, 4])
    IN_FIX_FRAC_WIDTH = 4
    OUT_FIX_FRAC_WIDTH = 4
    log = get_logger(__name__)
    exp_out, mant_out = fp_mult_hardware(exp_a, mant_a, exp_b, mant_b, IN_FIX_FRAC_WIDTH, OUT_FIX_FRAC_WIDTH, log)
    print(exp_out, mant_out)

def test_fp_mult_software():
    torch.manual_seed(0)
    a = torch.rand(100)
    b = torch.rand(100)
    config = {
        "a_exp_width": 3,
        "a_man_width": 4,
        "b_exp_width": 3,
        "b_man_width": 4,
        "out_exp_width": 6,
        "out_man_width": 4,
    }
    out = fp_mult_software(a, b, config)
    print(out)

if __name__ == "__main__":
    test_fp_mult_software()