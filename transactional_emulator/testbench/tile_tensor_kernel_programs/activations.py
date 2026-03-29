"""Status notes for `tilelang_kernels.activations`.

The current `TileTensorProgram` runtime exposes:

- tile matmul
- tile binary ops: add/sub/mul
- row ops: exp/add/sub/mul/reduce_sum/reduce_max
- FP scalar ops: copy/add/sub/mul/max/exp/reci/sqrt

It does not currently expose a tensor-domain or FP-domain `sigmoid` or `tanh`
primitive. Because the TileLang kernels are:

- GELU(tanh approximation)
- SiLU = x * sigmoid(x)

we keep these builders explicit about being unsupported instead of writing an
implementation that only looks plausible.
"""

from __future__ import annotations


def build_gelu_program(*args, **kwargs):
    raise NotImplementedError(
        "TileTensorProgram does not currently expose the tanh/sigmoid-style primitives "
        "needed for the TileLang GELU kernel."
    )


def build_silu_program(*args, **kwargs):
    raise NotImplementedError(
        "TileTensorProgram does not currently expose the sigmoid primitive "
        "needed for the TileLang SiLU kernel."
    )
