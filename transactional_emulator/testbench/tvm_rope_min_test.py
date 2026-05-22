"""TVM minimal RoPE testbench (Q-side only).

What the kernel computes (Q-side RoPE, branchless pair-swap via loop fission):
    pair_d = d ^ 1
    if d is even:  Q_OUT[d] = XQ[d]      * COS[d] + XQ[pair_d] * NEG_SIN[d]
    if d is odd:   Q_OUT[d] = XQ[pair_d] * SIN[d] + XQ[d]      * COS[d]

COS / SIN / NEG_SIN are pre-expanded by the testbench to full BSHD layout
so the kernel doesn't need any half-index repeat patterns at runtime.

Verification mode (matches linear / flash_attention / layernorm):
MX-E4M3 round-trip on BOTH golden sides + HBM-direct compare
(``check_hbm=True``) + fingerprint cache (``TB_CACHE=1``, default).
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
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402

from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec, resolve_output_layout,
)

# Shared MX-E4M3 round-trip + fingerprint cache.
sys.path.insert(0, str(_THIS_FILE.parent))
from _tb_cache import mx_roundtrip, run_cached  # noqa: E402


BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

HLEN = _HW.hlen  # from plena_settings.toml
MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per (s_block, head) tile == mlen
HEAD_COUNT = 8
# rope_min requires full_dim == hlen (2*half_dim == hlen), matching the
# single_stream_block chain which passes half_dim = HLEN // 2. The old
# hard-coded HALF_DIM = 8 (full_dim 16 != hlen 128) fails that check.
HALF_DIM = HLEN // 2
FULL_DIM = HALF_DIM * 2
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS


_OUT_LAYOUT = resolve_output_layout(
    num_batches=BATCH * SEQ_LEN,
    elements_per_batch=HEAD_COUNT * FULL_DIM,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull Q_OUT_hbm's MXFP-packed byte base for the HBM-direct compare."""
    return {"result_hbm_start_byte": int(raw["Q_OUT_hbm"]["address"])}


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

    # The kernel reads XQ / COS / SIN / NEG_SIN back from HBM already
    # MX-E4M3 quantized, so the golden computes on the MX-round-tripped
    # inputs (NEG_SIN is its own HBM tensor — quantize it directly).
    xq_eff      = mx_roundtrip(xq)
    cos_eff     = mx_roundtrip(cos_full)
    sin_eff     = mx_roundtrip(sin_full)
    neg_sin_eff = mx_roundtrip(neg_sin_full)

    # Golden via the same pair-swap formula the kernel uses.
    pair_index = torch.arange(FULL_DIM) ^ 1
    xq_pair = xq_eff.index_select(-1, pair_index)
    even_mask = (torch.arange(FULL_DIM) % 2 == 0).view(1, 1, 1, FULL_DIM)
    q_even = xq_eff * cos_eff + xq_pair * neg_sin_eff
    q_odd  = xq_pair * sin_eff + xq_eff * cos_eff
    q_golden = torch.where(even_mask, q_even, q_odd)

    # The kernel writes Q_OUT to HBM as MX-E4M3; HBM-direct reads those
    # bytes back, so the golden is MX-round-tripped too.
    q_golden = mx_roundtrip(q_golden)

    return {
        "hbm_inputs": {
            "XQ_hbm":      xq,
            "COS_hbm":     cos_full,
            "SIN_hbm":     sin_full,
            "NEG_SIN_hbm": neg_sin_full,
            "Q_OUT_hbm":   torch.zeros_like(xq),
        },
        "golden_flat": _OUT_LAYOUT.flatten_golden(
            q_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * FULL_DIM)
        ),
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io
    params = _OUT_LAYOUT.comparison_params()
    return {
        "check_hbm": True,
        "start_row_idx": 0,
        "compare_fpsram": False,
        "result_hbm_start_byte": int(addrs["result_hbm_start_byte"]),
        "scale_offset": params["num_batches"] * params["elements_per_batch"],
        **params,
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
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def _fingerprint() -> dict:
    return {
        "kernel": "rope_min",
        "batch": BATCH, "mlen": MLEN, "hlen": HLEN, "head_count": HEAD_COUNT,
        "half_dim": HALF_DIM, "full_dim": FULL_DIM,
        "num_s_blocks": NUM_S_BLOCKS, "seq_len": SEQ_LEN, "seed": 0, "schema": 1,
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
