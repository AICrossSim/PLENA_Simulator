"""TVM flash-decode gemm-only debug testbench.

Drives ``tilelang_tvm_compiler.kernels.flash_decode_min_gemm_only``:

    score = Q @ K^T                 (BTMV, packed-head)
    out   = score @ V               (MV,   per-head)

No scale, no softmax, no online state. Same Q_cache / O_cache cache
layout as flash_decode_min (head-major (HEAD_COUNT, HLEN)). Bisects the
new region+dim_roles gemm schema across multi by_number.
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
HEAD_COUNT = 8
HARDWARE_LANE_COUNT = MLEN // HLEN
NUM_KV_BLOCKS = 2
KV_SEQ = NUM_KV_BLOCKS * ROWS

CACHE_MLEN_ROWS = (HEAD_COUNT * HLEN) // MLEN


def parse_buffer_addrs(raw: dict) -> dict:
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])
    # No kernel scalars in this kernel — Q_FP_STAGE goes at FPRAM 0.
    return {
        "Q_CACHE":    addr_of("Q_cache"),
        "O_CACHE":    addr_of("O_cache"),
        "Q_FP_STAGE": 0,
    }


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(BATCH, 1,      HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    # out[b, 0, h, :] = (q[b, 0, h, :] @ K_h^T) @ V_h
    score = torch.einsum("bihd,bjhd->bihj", q, k)         # (B, 1, H, KV)
    out = torch.einsum("bihj,bjhd->bihd", score, v)       # (B, 1, H, HLEN)

    golden_flat = out.reshape(BATCH * 1, HEAD_COUNT * HLEN)

    return {
        "hbm_inputs":  {"K_hbm": k, "V_hbm": v},
        "golden_flat": golden_flat,
        "q_token":     q,
    }


def build_pre_kernel_stub(addrs: dict) -> str:
    lines: list[str] = [
        "; pre-kernel cache init: FPRAM[Q_FP_STAGE] -> VRAM[Q_CACHE]",
        f"S_ADDI_INT gp1, gp0, {addrs['Q_CACHE']}",
        f"S_ADDI_INT gp2, gp0, {addrs['Q_FP_STAGE']}",
    ]
    for i in range(CACHE_MLEN_ROWS):
        lines.append("S_MAP_V_FP gp1, gp2, 0")
        if i < CACHE_MLEN_ROWS - 1:
            lines.append(f"S_ADDI_INT gp1, gp1, {MLEN}")
            lines.append(f"S_ADDI_INT gp2, gp2, {MLEN}")
    return "\n".join(lines) + "\n"


def build_fp_preload(io: dict, addrs: dict):
    q_token = io["q_token"]
    total = addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN
    fp = torch.zeros(total, dtype=torch.float16)
    q_flat = q_token[0, 0].reshape(HEAD_COUNT * HLEN).to(torch.float16)
    fp[addrs["Q_FP_STAGE"] : addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN] = q_flat
    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    return {
        "check_hbm": False,
        "start_row_idx": addrs["O_CACHE"] // MLEN,
        "num_rows": CACHE_MLEN_ROWS,
        "num_batches": BATCH * 1,
        "elements_per_batch": HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_decode_min_gemm_only",
    kernel=(
        "tilelang_tvm_compiler.kernels.flash_decode_min_gemm_only:"
        "make_flash_decode_min_gemm_only"
    ),
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_kv_blocks": NUM_KV_BLOCKS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_pre_kernel_stub=build_pre_kernel_stub,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
