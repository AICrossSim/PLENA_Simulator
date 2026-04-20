import torch
from quant.quantizer.hardware_quantizer import _minifloat_ieee_quantize_hardware


def fp_cast_hardware(x: torch.Tensor, config: dict):
    qx, _x_exp, _ = _minifloat_ieee_quantize_hardware(x, config["in_mant_width"] + 8 + 1, 8)
    q_out, _, _ = _minifloat_ieee_quantize_hardware(
        qx, config["out_mant_width"] + config["out_exp_width"] + 1, config["out_exp_width"]
    )
    return q_out
