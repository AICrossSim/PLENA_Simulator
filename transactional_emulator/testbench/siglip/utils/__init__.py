"""Utility namespace for shared SigLIP testbench helpers."""

from .core import (
    ENCODER_HBM_DATA_ORDER,
    json_default,
    resolve_position_embedding,
    resolve_vision_encoder_layer,
    tensor_metrics,
    write_golden_values_file,
)
from .math import mha_sdpa, matmul_bf16_visible, projection_matmul_k_split_visible, quantize_flattened_like_hbm
from .vram import (
    load_vram_bf16,
    load_vram_chunk_major_to_seq,
    load_vram_head_major_q_to_seq,
    load_vram_seq_major_to_seq,
    pack_seq_to_chunk_major,
    unpack_chunk_major_to_seq,
)

__all__ = [
    "ENCODER_HBM_DATA_ORDER",
    "mha_sdpa",
    "json_default",
    "load_vram_bf16",
    "load_vram_chunk_major_to_seq",
    "load_vram_head_major_q_to_seq",
    "load_vram_seq_major_to_seq",
    "matmul_bf16_visible",
    "pack_seq_to_chunk_major",
    "projection_matmul_k_split_visible",
    "quantize_flattened_like_hbm",
    "resolve_position_embedding",
    "resolve_vision_encoder_layer",
    "tensor_metrics",
    "unpack_chunk_major_to_seq",
    "write_golden_values_file",
]
