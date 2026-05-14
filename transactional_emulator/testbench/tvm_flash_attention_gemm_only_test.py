"""TVM minimal Q K V gemm-only testbench (no softmax).

Drives ``tilelang_tvm_compiler.kernels.flash_attention_gemm_only``.
Golden: ``out = (Q @ K^T) @ V`` per head (no softmax, no scale).
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


BATCH = 1
ROWS = 64
HLEN = 16
MLEN = 64
HEAD_COUNT = 4              # one packed-head group (= MLEN/HLEN)
NUM_KV_BLOCKS = 1
NUM_Q_BLOCKS = 1
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25

    # Per-head: S = Q @ K^T, out = S @ V.
    score = torch.einsum("bihd,bjhd->bihj", q, k)             # (B, Q, H, KV)
    out   = torch.einsum("bihj,bjhd->bihd", score, v)         # (B, Q, H, D)

    golden_flat = out.reshape(BATCH * Q_SEQ, HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "Q_hbm": q,
            "K_hbm": k,
            "V_hbm": v,
            "O_hbm": torch.zeros_like(q),
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    chunks_per_batch = (HEAD_COUNT * HLEN + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * Q_SEQ * chunks_per_batch,
        "num_batches": BATCH * Q_SEQ,
        "elements_per_batch": HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_attention_gemm_only",
    kernel=(
        "tilelang_tvm_compiler.kernels.flash_attention_gemm_only:"
        "make_flash_attention_gemm_only"
    ),
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_kv_blocks": NUM_KV_BLOCKS, "num_q_blocks": NUM_Q_BLOCKS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="O_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
