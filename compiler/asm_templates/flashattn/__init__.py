"""Flash Attention assembly code generation package.

This package provides modular components for generating Flash Attention
assembly code for the coprocessor.

Modules:
    qkt: QKT multiplication (Q @ K.T)
    online_softmax: Online softmax computation
    pv: PV multiplication (P @ V)
    output: Output computation and row-wise scaling
    reset: Memory initialization/reset utilities
    overall: Main flash_attn_asm function that orchestrates all components
"""

from .online_softmax import online_softmax_code
from .output import computing_o_code, computing_row_wise_scaling_code
from .overall import flash_attn_asm
from .pv import computing_pv_code
from .qkt import qkt_multiply
from .reset import reset_fpsram_code, reset_kv_prefetch, reset_vssram_code

__all__ = [
    "computing_o_code",
    "computing_pv_code",
    "computing_row_wise_scaling_code",
    "flash_attn_asm",
    "online_softmax_code",
    "qkt_multiply",
    "reset_fpsram_code",
    "reset_kv_prefetch",
    "reset_vssram_code",
]
