import torch
from torch import Tensor


def hardware_round(x: Tensor, round_bits: int = 2):
    x = x * 2**round_bits
    x = torch.floor(x)
    x = x / 2**round_bits
    return x.round()


def fixed_point_cast(
    x: Tensor,
    out_width: int,
    out_frac_width: int,
    floor: bool = True,
):
    min_val = -(2 ** (out_width - 1))
    max_val = 2 ** (out_width) - 1
    if floor:
        x = torch.clamp((x * 2 ** (out_frac_width)).floor(), min_val, max_val)
    else:
        x = torch.clamp((x * 2 ** (out_frac_width)).round(), min_val, max_val)

    x = x / 2 ** (out_frac_width)
    return x
