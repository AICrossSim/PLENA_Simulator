"""TVM minimal Linear testbench (no-transpose-B variant).

Drives ``tilelang_tvm_compiler.kernels.linear_min_no_transpose`` — same multi-tile
GEMM as ``tvm_linear_min_test`` but feeds the weight already transposed
into ``(K, N)`` row-major. The kernel issues plain ``M_MM`` (no
``M_TMM``) since ``transpose_B`` is False.

Golden uses the same untransposed weight ``a @ w.T`` so numerical
results match the transpose-B variant byte-for-byte (modulo fp16
quantisation order).
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
    a_2d = torch.randn(M, K, dtype=torch.float32) * 0.25
    # Reference weight in nn.Linear convention (N, K). Host-transpose
    # into (K, N) for the kernel — this is the variant where the host
    # pays the transpose cost.
    w_2d_nk = torch.randn(N, K, dtype=torch.float32) * 0.25
    b_2d_kn = w_2d_nk.T.contiguous()

    a = a_2d.view(1, M, 1, K).contiguous()
    b = b_2d_kn.view(1, K, 1, N).contiguous()

    if WITH_BIAS:
        bias_1d = torch.randn(N, dtype=torch.float32) * 0.1
        c_2d_golden = a_2d @ w_2d_nk.T + bias_1d
        bias_tile = bias_1d.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()
        hbm_inputs = {
            "A_hbm":    a,
            "B_hbm":    b,
            "BIAS_hbm": bias_tile,
            "C_hbm":    torch.zeros(1, M, 1, N, dtype=torch.float32),
        }
    else:
        c_2d_golden = a_2d @ w_2d_nk.T
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
    asm_name="linear_min_no_transpose",
    kernel="tilelang_tvm_compiler.kernels.linear_min_no_transpose:make_linear_min_no_transpose",
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
