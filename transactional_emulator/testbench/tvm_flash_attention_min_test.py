"""TVM minimal FlashAttention testbench (multi-block flow).

Drives the ``tilelang_tvm_compiler.kernels.flash_attention_min`` kernel
through the test_helper. Builds Q/K/V, FP-preload (scale / m_init /
l_init from the compiler's --dump-buffer-addrs JSON), and per-head
softmax-attention golden.

What the kernel computes (all heads):
    score = Q @ K^T                       (BTMM #1, all lanes get raw score)
    P     = softmax(scale * score)        (online softmax over kv blocks)
    O     = P @ V                         (BTMM #2)
    -> O_hbm
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# venv probing + sys.path setup (must run before torch / helper imports).
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
ROWS = 64           # q-tile length == mlen
HLEN = 16
MLEN = 64
HEAD_COUNT = 8
HARDWARE_LANE_COUNT = MLEN // HLEN
ACTIVE_LANE = 2
NUM_KV_BLOCKS = 2   # full multi-block online softmax
NUM_Q_BLOCKS = 2    # multi-Q outer loop
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS

# Finite "negative-infinity" surrogate compatible with float16 / FP-scalar
# arithmetic. Mirrors attention.py's choice.
NEG_INF = -1.0e4


def parse_buffer_addrs(raw: dict) -> dict:
    """Single source of truth for FP-slot addresses — read from the
    compiler's --dump-buffer-addrs JSON instead of mirroring
    AddressAllocationPass by hand."""
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])
    last = addr_of("L_INIT")
    return {
        "SCALE":  addr_of("SCALE"),
        "M_INIT": addr_of("M_INIT"),
        "L_INIT": last,
        # Last byte the FPRAM preload tensor needs to cover. ``L_INIT``
        # is one of the (lane_count, rows) scalar slots, so its end
        # is ``addr + lane_count * rows``.
        "fp_preload_end": last + HARDWARE_LANE_COUNT * ROWS,
    }


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Full softmax-attention golden over the entire kv sequence."""
    torch.manual_seed(seed)
    q = torch.randn(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)
    score = torch.einsum("bihd,bjhd->bihj", q, k)        # (B, Q_SEQ, H, KV_SEQ)

    out = torch.empty(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32)
    for h in range(HEAD_COUNT):
        score_h = score[:, :, h, :]
        v_h     = v[:, :, h, :]
        scaled  = score_h * scale
        p       = torch.softmax(scaled, dim=-1)
        out[:, :, h, :] = torch.einsum("bij,bjd->bid", p, v_h)

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


def build_fp_preload(io: dict, addrs: dict):  # noqa: ARG001
    """Full FP preload: SCALE = 1/sqrt(d_k), M_INIT = -inf surrogate,
    L_INIT = 0. The kernel resets M_OLD / L_OLD from M_INIT / L_INIT
    at the start of every q_block."""
    del io
    fp = torch.zeros(addrs["fp_preload_end"], dtype=torch.float16)
    scale_val = 1.0 / math.sqrt(HLEN)
    for h in range(HARDWARE_LANE_COUNT):
        row_base = h * ROWS
        scale_start = addrs["SCALE"] + row_base
        fp[scale_start : scale_start + ROWS] = scale_val
        m_init_start = addrs["M_INIT"] + row_base
        fp[m_init_start : m_init_start + ROWS] = float(NEG_INF)
        l_init_start = addrs["L_INIT"] + row_base
        fp[l_init_start : l_init_start + ROWS] = 0.0
    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    chunks_per_batch = (HEAD_COUNT * HLEN + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        # view_mem reads VRAM as row_dim-wide chunks. When one logical
        # output row spans multiple MLEN chunks (e.g. head_count=8,
        # hlen=16 -> 128 elements), read all chunks before stride-mode
        # reordering.
        "num_rows": BATCH * Q_SEQ * chunks_per_batch,
        "num_batches": BATCH * Q_SEQ,
        "elements_per_batch": HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_attention_min",
    kernel="tilelang_tvm_compiler.kernels.flash_attention_min:make_flash_attention_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "active_lane": ACTIVE_LANE,
        "num_kv_blocks": NUM_KV_BLOCKS, "num_q_blocks": NUM_Q_BLOCKS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="O_hbm",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
