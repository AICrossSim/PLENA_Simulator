"""Compatibility shim for SigLIP encoder-layer ASM emitters.

Canonical implementations now live under compiler.asm_templates.siglip.
This module is kept to preserve existing testbench import paths.
"""

from compiler.asm_templates.siglip.encoder_layer import (
    build_encoder_layer_asm,
    build_mlp_block,
)

__all__ = [
    "build_encoder_layer_asm",
    "build_mlp_block",
]
