"""TVM FlashAttention testbench — head-offset output variant.

Exercises ``make_flash_attention_min``'s ``o_head_count`` /
``o_head_offset`` parameters: the kernel computes attention over
``head_count`` heads but writes its output into a head-slice of a
WIDER ``o_head_count``-head HBM tensor, starting at ``o_head_offset``.

This is the mechanism the single-stream-block chain uses to build
``concat([attn_out, mlp_out])`` on-device: attention drops its result
straight into the left half of a 2x-wide output tensor (no separate
concat kernel).

This testbench sets o_head_count = 2*head_count, o_head_offset = 0
(the "attn writes left half" case). The right half stays whatever it
was preloaded as — the golden zero-fills it so view_mem's compare
passes only when attention wrote exactly the left-half heads and
touched nothing else.
"""

from __future__ import annotations

import math
import os
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
    TvmTestbenchSpec, run, resolve_output_layout,
)


BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

HLEN = _HW.hlen  # from plena_settings.toml
MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per tile == mlen
HEAD_COUNT = 8
ACTIVE_LANE = 2
NUM_KV_BLOCKS = 2
NUM_Q_BLOCKS = 2
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS

# Output tensor is 2x wide. ATTN_OFFSET env var selects which head
# half attention writes into; default 8 = right half (the non-zero
# offset case — offset 0 is indistinguishable from the plain kernel).
O_HEAD_COUNT = 2 * HEAD_COUNT
O_HEAD_OFFSET = 8


def parse_buffer_addrs(raw: dict) -> dict:
    del raw
    return {}


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Attention golden, placed into the LEFT half of a 2x-wide output."""
    torch.manual_seed(seed)
    q = torch.randn(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)
    score = torch.einsum("bihd,bjhd->bihj", q, k)

    out = torch.empty(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32)
    for h in range(HEAD_COUNT):
        scaled = score[:, :, h, :] * scale
        p = torch.softmax(scaled, dim=-1)
        out[:, :, h, :] = torch.einsum("bij,bjd->bid", p, v[:, :, h, :])

    # Place attention output into a 2x-wide tensor's head-slice
    # [O_HEAD_OFFSET : O_HEAD_OFFSET + HEAD_COUNT]. The rest is zero —
    # the O_hbm scratch is staged as zeros, attention only writes its
    # slice, so a correct kernel leaves the other heads at 0.
    out_wide = torch.zeros(BATCH, Q_SEQ, O_HEAD_COUNT, HLEN, dtype=torch.float32)
    out_wide[:, :, O_HEAD_OFFSET : O_HEAD_OFFSET + HEAD_COUNT, :] = out

    golden_flat = out_wide.reshape(BATCH * Q_SEQ, O_HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "Q_hbm": q,
            "K_hbm": k,
            "V_hbm": v,
            "O_hbm": torch.zeros(BATCH, Q_SEQ, O_HEAD_COUNT, HLEN, dtype=torch.float32),
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    # Geometry (num_rows / num_batches / elements_per_batch /
    # row_dim / use_stride_mode) from the canonical OutputLayout
    # so it agrees with golden_flat by construction.
    layout = resolve_output_layout(
        num_batches=BATCH * Q_SEQ,
        elements_per_batch=O_HEAD_COUNT * HLEN,
        mlen=MLEN,
    )
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **layout.comparison_params(),
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_attention_offset",
    kernel="tilelang_tvm_compiler.kernels.flash_attention_min:make_flash_attention_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "active_lane": ACTIVE_LANE,
        "num_kv_blocks": NUM_KV_BLOCKS, "num_q_blocks": NUM_Q_BLOCKS,
        "o_head_count": O_HEAD_COUNT, "o_head_offset": O_HEAD_OFFSET,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="O_hbm",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
