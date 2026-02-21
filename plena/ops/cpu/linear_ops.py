"""CPU (PyTorch) reference implementations for linear projection."""

import torch


def linear_cpu(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: input @ weight (float32 accumulation)."""
    return torch.matmul(input.float(), weight.float())
