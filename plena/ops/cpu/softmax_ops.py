"""CPU (PyTorch) reference implementation for softmax."""

import torch


def softmax_cpu(
    input: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    CPU reference: row-wise softmax with optional pre-scaling.

    Computes:  softmax(input * scale, dim=-1)

    This is the golden value against which the PLENA ISA simulation is verified.
    Uses float32 accumulation to match hardware precision.

    Args:
        input: Input tensor of shape (rows, cols), any dtype.
        scale: Multiplicative scale applied before softmax (default 1.0).

    Returns:
        Softmax output, same shape as input, in float32.
    """
    x = input.float() * scale
    return torch.softmax(x, dim=-1)
