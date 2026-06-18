"""Embedding-stage helpers for SigLIP full-model harness.

This module keeps embedding ASM emission and embedding-related VRAM preload
logic separate from the main full-model harness.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.utils.core import align_up
from transactional_emulator.testbench.siglip.utils.vram import pack_seq_to_chunk_major


def fill_embedding_inputs_for_asm(
    *,
    config: dict,
    embedding_weights: dict,
    seq_len: int,
    hidden_size: int,
    mlen: int,
    vram_preload: np.ndarray,
    patch_input_base: int,
    patch_bias_base: int,
    position_base: int,
) -> torch.Tensor:
    """Fill VRAM preload for Stage 0 embedding ASM with chunk-major patch/position tensors."""
    patch_size = int(config["patch_size"])
    num_channels = int(config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, mlen)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, num_channels, config["image_size"], config["image_size"], dtype=torch.float32)
    patches = F.unfold(pixel_values, kernel_size=patch_size, stride=patch_size).transpose(1, 2).contiguous()[0]
    patches_raw = patches.clone()
    if aligned_in_features != in_features:
        patches = F.pad(patches, (0, aligned_in_features - in_features))

    patches_padded = torch.zeros(seq_len, aligned_in_features, dtype=torch.float32)
    valid_rows = min(seq_len, patches.shape[0])
    patches_padded[:valid_rows, :aligned_in_features] = patches[:valid_rows, :aligned_in_features]
    patch_chunk = pack_seq_to_chunk_major(patches_padded.to(torch.bfloat16).float(), mlen=mlen)
    patch_u16 = patch_chunk.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    vram_preload[patch_input_base : patch_input_base + patch_u16.size] = patch_u16

    patch_bias_src = embedding_weights.get("patch_bias")
    if patch_bias_src is not None:
        patch_bias_vec = patch_bias_src.float().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        patch_bias_u16 = patch_bias_vec
        vram_preload[patch_bias_base : patch_bias_base + patch_bias_u16.size] = patch_bias_u16

    pos_src = embedding_weights["position_table"].float()
    pos_padded = torch.zeros(seq_len, hidden_size, dtype=torch.float32)
    valid_rows = min(seq_len, pos_src.shape[0])
    valid_hidden = min(hidden_size, pos_src.shape[1])
    pos_padded[:valid_rows, :valid_hidden] = pos_src[:valid_rows, :valid_hidden]
    pos_chunk = pack_seq_to_chunk_major(pos_padded.to(torch.bfloat16).float(), mlen=mlen)
    pos_u16 = pos_chunk.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    vram_preload[position_base : position_base + pos_u16.size] = pos_u16

    return patches_raw


def prepare_vram_preload_from_embedding(
    embedding_out: torch.Tensor,
    seq_len_kernel: int,
    hidden_runtime: int,
    hidden_visible: int,
    mlen: int,
) -> np.ndarray:
    """Prepare VRAM preload from a precomputed embedding output tensor."""
    embed_padded = torch.zeros(seq_len_kernel, hidden_runtime, dtype=torch.float32)
    valid_seq = min(seq_len_kernel, embedding_out.shape[0])
    embed_padded[:valid_seq, :hidden_visible] = embedding_out[:valid_seq, :hidden_visible].float()
    embed_chunk_major = pack_seq_to_chunk_major(embed_padded.to(torch.bfloat16).float(), mlen=mlen)
    return embed_chunk_major.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
