"""CPU (PyTorch) reference implementations for positional encoding operators."""

import torch


def embedding_add_cpu(input: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """CPU reference: add learned position embeddings to patch embeddings.

    Implements: output = input + pos_weight  (element-wise, same shape)

    This is the SigLIP vision encoder's learned PE step:
        embeddings = patch_embeds + position_embedding(position_ids)

    Both tensors have shape (seq_len, hidden_size).
    """
    return input + pos_weight


def rope_cpu(
    x: torch.Tensor,
    x_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """CPU reference: apply Rotary Position Embedding (RoPE).

    Implements: output = x * cos + rotate_half(x) * sin

    The caller must precompute:
        x_rot = rotate_half(x)  where  rotate_half(x) = cat(-x[d//2:], x[:d//2])
        cos, sin = positional frequency tables, shape (seq_len, head_dim)

    All tensors have shape (seq_len, head_dim).
    """
    return x * cos + x_rot * sin
