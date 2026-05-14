"""TVM minimal Linear testbench.

Drives ``tilelang_tvm_compiler.kernels.linear_min`` — multi-tile GEMM
with M/N/K all multiples of MLEN. Tile counts (M_BLOCKS / N_BLOCKS /
K_BLOCKS) and bias mode are env-controlled so the same testbench
exercises (1, 1, 1) up through bigger tile grids.

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

from tilelang_tvm_compiler.test_helper import TvmTestbenchSpec, run  # noqa: E402


MLEN = 64
M_BLOCKS = 2
N_BLOCKS = 2
K_BLOCKS = 2
WITH_BIAS = True

M = M_BLOCKS * MLEN
N = N_BLOCKS * MLEN
K = K_BLOCKS * MLEN


def build_inputs_and_golden(seed: int = 0) -> dict:
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
        c_2d_golden = a_2d @ b_2d.T + bias_1d  # bias broadcasts (N,) across M rows
        bias_tile = bias_1d.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()
        hbm_inputs = {
            "A_hbm":    a,
            "B_hbm":    b,
            "BIAS_hbm": bias_tile,
            "C_hbm":    torch.zeros(1, M, 1, N, dtype=torch.float32),
        }
    else:
        c_2d_golden = a_2d @ b_2d.T
        hbm_inputs = {
            "A_hbm": a,
            "B_hbm": b,
            "C_hbm": torch.zeros(1, M, 1, N, dtype=torch.float32),
        }

    return {
        "hbm_inputs": hbm_inputs,
        "golden_flat": c_2d_golden,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    # Output is (M, N) row-major. view_mem reads VRAM as MLEN-wide
    # chunks; one logical row of N elements spans (N // MLEN) chunks.
    chunks_per_row = N // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": M * chunks_per_row,
        "num_batches": M,
        "elements_per_batch": N,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="linear_min",
    kernel="tilelang_tvm_compiler.kernels.linear_min:make_linear_min",
    kernel_kwargs={
        "m_blocks": M_BLOCKS,
        "n_blocks": N_BLOCKS,
        "k_blocks": K_BLOCKS,
        "with_bias": WITH_BIAS,
    },
    mlen=MLEN,
    stage_output="C_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
