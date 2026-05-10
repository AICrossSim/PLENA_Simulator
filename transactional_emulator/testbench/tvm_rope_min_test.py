"""TVM minimal RoPE testbench (Q-side only).

What the kernel computes (Q-side RoPE, branchless pair-swap via loop fission):
    pair_d = d ^ 1
    if d is even:  Q_OUT[d] = XQ[d]      * COS[d] + XQ[pair_d] * NEG_SIN[d]
    if d is odd:   Q_OUT[d] = XQ[pair_d] * SIN[d] + XQ[d]      * COS[d]

COS / SIN / NEG_SIN are pre-expanded by the testbench to full BSHD layout
so the kernel doesn't need any half-index repeat patterns at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

# venv probing + sys.path setup.
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
ROWS = 64           # rows per (s_block, head) tile == mlen
HLEN = 16
MLEN = 64
HEAD_COUNT = 8
HALF_DIM = 8
FULL_DIM = HALF_DIM * 2
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    xq = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, FULL_DIM, dtype=torch.float32) * 0.5

    # Standard RoPE frequency table: theta_{p,k} = p * 10000^(-2k/D).
    pos = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, SEQ_LEN, 1, 1)
    dim = torch.arange(HALF_DIM, dtype=torch.float32).view(1, 1, 1, HALF_DIM)
    theta = pos * torch.pow(10000.0, -2.0 * dim / FULL_DIM)
    cos_half = torch.cos(theta)
    sin_half = torch.sin(theta)
    cos_full = torch.repeat_interleave(cos_half, repeats=2, dim=-1)
    sin_full = torch.repeat_interleave(sin_half, repeats=2, dim=-1)
    cos_full = cos_full.expand(BATCH, SEQ_LEN, HEAD_COUNT, FULL_DIM).contiguous()
    sin_full = sin_full.expand(BATCH, SEQ_LEN, HEAD_COUNT, FULL_DIM).contiguous()
    neg_sin_full = -sin_full

    # Golden via the same pair-swap formula the kernel uses.
    pair_index = torch.arange(FULL_DIM) ^ 1
    xq_pair = xq.index_select(-1, pair_index)
    even_mask = (torch.arange(FULL_DIM) % 2 == 0).view(1, 1, 1, FULL_DIM)
    q_even = xq * cos_full + xq_pair * neg_sin_full
    q_odd  = xq_pair * sin_full + xq * cos_full
    q_golden = torch.where(even_mask, q_even, q_odd)
    golden_flat = q_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * FULL_DIM)

    return {
        "hbm_inputs": {
            "XQ_hbm":      xq,
            "COS_hbm":     cos_full,
            "SIN_hbm":     sin_full,
            "NEG_SIN_hbm": neg_sin_full,
            "Q_OUT_hbm":   torch.zeros_like(xq),
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    chunks_per_batch = (HEAD_COUNT * FULL_DIM + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * SEQ_LEN * chunks_per_batch,
        "num_batches": BATCH * SEQ_LEN,
        "elements_per_batch": HEAD_COUNT * FULL_DIM,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="rope_min",
    kernel="tilelang_tvm_compiler.kernels.rope_min:make_rope_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "half_dim": HALF_DIM, "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="Q_OUT_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
