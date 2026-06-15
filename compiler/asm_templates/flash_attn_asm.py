"""Flash Attention assembly code generation.

This module has been refactored into the flashattn package.
This file re-exports all functions for backward compatibility.

For new code, prefer importing from compiler.asm_templates.flashattn directly.
"""

# Re-export all functions from the flashattn package for backward compatibility
from .flashattn import (
    computing_o_code as _computing_o_code,
)
from .flashattn import (
    computing_pv_code as _computing_pv_code,
)
from .flashattn import (
    computing_row_wise_scaling_code as _computing_row_wise_scaling_code,
)
from .flashattn import (
    flash_attn_asm,
    qkt_multiply,
)
from .flashattn import (
    online_softmax_code as _online_softmax_code,
)
from .flashattn import (
    reset_fpsram_code as _reset_fpsram_code,
)
from .flashattn import (
    reset_kv_prefetch as _reset_kv_prefetch,
)
from .flashattn import (
    reset_vssram_code as _reset_vssram_code,
)

# Also export IMM2_BOUND for backward compatibility
IMM2_BOUND = 2**18 - 1

__all__ = [
    "IMM2_BOUND",
    "_computing_o_code",
    "_computing_pv_code",
    "_computing_row_wise_scaling_code",
    "_online_softmax_code",
    "_reset_fpsram_code",
    "_reset_kv_prefetch",
    "_reset_vssram_code",
    "flash_attn_asm",
    "qkt_multiply",
]
