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


HD = HEAD_COUNT * HLEN  # collapsed head*dim axis (BSHD -> BS·1·(H*D))

_OUT_LAYOUT = resolve_output_layout(
    num_batches=BATCH * SEQ_LEN,
    elements_per_batch=HD,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull Q_OUT_hbm's MXFP-packed byte base for the HBM-direct compare."""
    return {"result_hbm_start_byte": int(raw["Q_OUT_hbm"]["address"])}


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    # BSHD collapsed to (1, SEQ, 1, H*D). Build per-head then flatten the
    # (head, dim) pair into the H*D axis.
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

    # SGN_SIN[d] = -sin at even d, +sin at odd d (pre-combine NEG_SIN/SIN).
    even_mask_d = (torch.arange(FULL_DIM) % 2 == 0).view(1, 1, 1, FULL_DIM)
    sgn_sin_full = torch.where(even_mask_d, -sin_full, sin_full)

    # Collapse (head, dim) -> H*D.
    def _to_hd(t):
        return t.reshape(BATCH, SEQ_LEN, 1, HD).contiguous()

    xq_hd      = _to_hd(xq)
    cos_hd     = _to_hd(cos_full)
    sgn_sin_hd = _to_hd(sgn_sin_full)

    # Pair-swap permutation matrix P (MLEN x MLEN): block-diagonal 2x2
    # swaps, P[2i, 2i+1] = P[2i+1, 2i] = 1. Because the full-H*D pair-swap
    # is block-diagonal and never crosses an MLEN boundary (MLEN even), one
    # shared MLEN×MLEN diagonal block suffices — the kernel applies it to
    # each MLEN-wide column block independently (no K accumulation).
    # 0/1 are exact in MX-E4M3 so P round-trips losslessly.
    p_mat = torch.zeros(MLEN, MLEN, dtype=torch.float32)
    idx = torch.arange(MLEN)
    p_mat[idx, idx ^ 1] = 1.0
    p_hbm = p_mat.view(1, MLEN, 1, MLEN).contiguous()

    # Kernel reads inputs back from HBM already MX-E4M3 quantized, so the
    # golden computes on the MX-round-tripped inputs.
    xq_eff      = mx_roundtrip(xq_hd)
    cos_eff     = mx_roundtrip(cos_hd)
    sgn_sin_eff = mx_roundtrip(sgn_sin_hd)
    p_eff       = mx_roundtrip(p_hbm)

    # Golden: OUT = X ⊙ COS + shuffle(X) ⊙ SGN_SIN. shuffle(X) applies the
    # MLEN×MLEN pair-swap P to each MLEN-wide column block independently
    # (block-diagonal — equivalent to the full-H*D pair-swap).
    x2d  = xq_eff.reshape(BATCH * SEQ_LEN, HD)
    p2d  = p_eff.reshape(MLEN, MLEN)
    n_blocks = HD // MLEN
    xs_blocks = [x2d[:, b * MLEN:(b + 1) * MLEN] @ p2d for b in range(n_blocks)]
    xs2d = torch.cat(xs_blocks, dim=-1)      # shuffle(X)
    out2d = (x2d * cos_eff.reshape(BATCH * SEQ_LEN, HD)
             + xs2d * sgn_sin_eff.reshape(BATCH * SEQ_LEN, HD))

    # Kernel writes Q_OUT to HBM as MX-E4M3; HBM-direct reads those bytes
    # back, so the golden is MX-round-tripped too.
    q_golden = mx_roundtrip(out2d)

    return {
        "hbm_inputs": {
            "XQ_hbm":      xq_hd,
            "COS_hbm":     cos_hd,
            "SGN_SIN_hbm": sgn_sin_hd,
            "P_hbm":       p_hbm,
            "Q_OUT_hbm":   torch.zeros_like(xq_hd),
        },
        "golden_flat": _OUT_LAYOUT.flatten_golden(q_golden),
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
        "num_s_blocks": NUM_S_BLOCKS, "seq_len": SEQ_LEN, "hd": HD,
        "seed": 0, "schema": 3,  # schema 3 = shuffle-matrix, MLEN×MLEN P (no K-acc)
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
