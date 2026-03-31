"""CPU (PyTorch) reference implementations for normalization."""

import torch


def rms_norm_cpu(
    input: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU reference: RMS normalization."""
    x = input.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


def layer_norm_cpu(
    input: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CPU reference: Layer normalization (zero-mean, unit-variance per row)."""
    x = input.float()
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
