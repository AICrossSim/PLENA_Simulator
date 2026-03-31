"""CPU (PyTorch) reference implementations for attention."""

import math
import torch


def flash_attention_cpu(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """CPU reference: Flash Attention = softmax(Q @ K.T * scale) @ V."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q.float(), K.float().T) * scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V.float())
