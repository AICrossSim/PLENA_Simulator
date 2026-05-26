"""TVM minimal Linear testbench — column-offset output variant.

Exercises ``make_linear_min``'s ``c_wide_n`` / ``c_col_offset``
parameters: the kernel computes an (M, N) GEMM but writes its result
into a column-slice of a WIDER ``c_wide_n``-column HBM tensor, starting
at ``c_col_offset``.

This is the mechanism the single-stream-block chain uses to drop the
MLP-in projection straight into the left part of
``concat([mlp, attn])`` with no separate concat kernel — the linear
counterpart to flash_attention_min's ``o_head_offset``.

This testbench sets c_wide_n = 2*N, c_col_offset = N (the "linear
writes right half" case — a non-zero offset, since offset 0 is
indistinguishable from the plain kernel). The rest of the wide C_hbm
stays whatever it was preloaded as; the golden zero-fills it so
view_mem's compare passes only when the kernel wrote exactly its
column slice and touched nothing else.

Golden: ``torch.nn.functional.linear(a, b, bias) == a @ b.T + bias``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _THIS_FILE.parents if (p / ".venv").is_dir() and (p / "compiler").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(f"could not locate repo root above {_THIS_FILE}")
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

import torch  # noqa: E402
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402

from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec,
    run,
    resolve_output_layout,
)


_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

MLEN = _HW.mlen  # from plena_settings.toml
M_BLOCKS = 2
N_BLOCKS = 2
K_BLOCKS = 2
WITH_BIAS = True

M = M_BLOCKS * MLEN
N = N_BLOCKS * MLEN
K = K_BLOCKS * MLEN

# Output tensor is 2x wide. The kernel writes its (M, N) result into
# columns [C_COL_OFFSET : C_COL_OFFSET + N]; the rest stays at the
# preloaded value. C_COL_OFFSET = N selects the right half (the
# non-zero offset case — offset 0 is indistinguishable from the plain
# kernel). C_COL_OFFSET must be a multiple of MLEN.
C_WIDE_N = 2 * N
C_COL_OFFSET = N


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Linear golden, placed into a column-slice of a 2x-wide output."""
    torch.manual_seed(seed)
    # Small magnitudes so the fp16 K-reduction stays in range. Same idea
    # as the GPU tilelang gemm test that runs at 0.5 scale.
    a_2d = torch.randn(M, K, dtype=torch.float32) * 0.25
    # Weight is (N, K) — nn.Linear convention. The kernel issues
    # T.gemm(..., transpose_B=True), which lowers to M_TMM so the host
    # does NOT need to transpose.
    b_2d = torch.randn(N, K, dtype=torch.float32) * 0.25

    # PLENA HBM tensors are 4D BSHD; linear has no head axis, so lay
    # (rows, cols) along (seq, hlen) with batch=head=1.
    a = a_2d.view(1, M, 1, K).contiguous()
    b = b_2d.view(1, N, 1, K).contiguous()

    if WITH_BIAS:
        bias_1d = torch.randn(N, dtype=torch.float32) * 0.1
        c_2d = a_2d @ b_2d.T + bias_1d  # bias broadcasts (N,) across M rows
        bias_tile = bias_1d.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()
    else:
        c_2d = a_2d @ b_2d.T
        bias_tile = None

    # Place the (M, N) result into a 2x-wide tensor's column-slice
    # [C_COL_OFFSET : C_COL_OFFSET + N]. The rest is zero — the C_hbm
    # scratch is staged as zeros, the kernel only writes its slice, so a
    # correct kernel leaves the other columns at 0.
    c_wide_golden = torch.zeros(M, C_WIDE_N, dtype=torch.float32)
    c_wide_golden[:, C_COL_OFFSET : C_COL_OFFSET + N] = c_2d

    if WITH_BIAS:
        hbm_inputs = {
            "A_hbm": a,
            "B_hbm": b,
            "BIAS_hbm": bias_tile,
            "C_hbm": torch.zeros(1, M, 1, C_WIDE_N, dtype=torch.float32),
        }
    else:
        hbm_inputs = {
            "A_hbm": a,
            "B_hbm": b,
            "C_hbm": torch.zeros(1, M, 1, C_WIDE_N, dtype=torch.float32),
        }

    return {
        "hbm_inputs": hbm_inputs,
        "golden_flat": c_wide_golden,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    # Geometry (num_rows / num_batches / elements_per_batch /
    # row_dim / use_stride_mode) from the canonical OutputLayout
    # so it agrees with golden_flat by construction.
    layout = resolve_output_layout(
        num_batches=M,
        elements_per_batch=C_WIDE_N,
        mlen=MLEN,
    )
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **layout.comparison_params(),
    }


SPEC = TvmTestbenchSpec(
    asm_name="linear_min_offset",
    kernel="tilelang_tvm_compiler.kernels.linear_min:make_linear_min",
    kernel_kwargs={
        "m_blocks": M_BLOCKS,
        "n_blocks": N_BLOCKS,
        "k_blocks": K_BLOCKS,
        "with_bias": WITH_BIAS,
        "c_wide_n": C_WIDE_N,
        "c_col_offset": C_COL_OFFSET,
    },
    mlen=MLEN,
    stage_output="C_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
