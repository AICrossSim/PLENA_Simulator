import torch
from torch import Tensor

def fixed_point_cast(
        x: torch.Tensor,
        OUT_WIDTH: int,
        OUT_FRAC_WIDTH: int,
        floor: bool = True,
):
    min_val = -2**(OUT_WIDTH - 1)
    max_val = 2**(OUT_WIDTH) - 1
    if floor:
        x = torch.clamp((x * 2**(OUT_FRAC_WIDTH)).floor(), min_val, max_val)
    else:
        x = torch.clamp((x * 2**(OUT_FRAC_WIDTH)).round(), min_val, max_val)

    x = x / 2**(OUT_FRAC_WIDTH)
    return x

def hardware_round(x: Tensor, round_bits: int = 2):
    x = x * 2**round_bits
    x = torch.floor(x)
    x = x / 2**round_bits
    return x.round()


def fixed_point_cast(
        x: Tensor,
        OUT_WIDTH: int,
        OUT_FRAC_WIDTH: int,
        floor: bool = True,
):
    min_val = -2**(OUT_WIDTH - 1)
    max_val = 2**(OUT_WIDTH) - 1
    if floor:
        x = torch.clamp((x * 2**(OUT_FRAC_WIDTH)).floor(), min_val, max_val)
    else:
        x = torch.clamp((x * 2**(OUT_FRAC_WIDTH)).round(), min_val, max_val)

    x = x / 2**(OUT_FRAC_WIDTH)
    return x