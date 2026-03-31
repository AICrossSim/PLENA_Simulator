"""CPU (PyTorch) reference implementation for FFN operator."""

import torch
import torch.nn.functional as F


def ffn_cpu(
    input: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: FFN with SiLU gate matching PLENA hardware.

    Computes: w_down @ (silu(w_up @ x) * (w_gate @ x))
    Hardware applies SiLU to the up projection (w_up), not the gate projection.
    All intermediate accumulations in float32.
    """
    x = input.float()
    gate = torch.matmul(x, w_gate.float())
    up = torch.matmul(x, w_up.float())
    hidden = F.silu(up) * gate
    return torch.matmul(hidden, w_down.float())
