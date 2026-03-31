"""CPU (PyTorch) reference implementation for conv2d via im2col + matmul."""

import torch


def conv2d_cpu(
    input_col: torch.Tensor,
    weight_2d: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: conv2d expressed as im2col matmul.

    Args:
        input_col: im2col-transformed input, shape (B*OH*OW, C_in*K*K)
        weight_2d: reshaped weight, shape (C_in*K*K, C_out)

    Returns:
        Output tensor, shape (B*OH*OW, C_out)
    """
    return torch.matmul(input_col.float(), weight_2d.float())
