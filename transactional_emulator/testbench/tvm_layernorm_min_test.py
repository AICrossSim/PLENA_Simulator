"""TVM minimal LayerNorm testbench.

Verifies the chain: row_reduce_sum_at -> fp_mul -> row_sub_fp_at ->
tile_mul -> row_reduce_sum_at -> fp_mul / fp_add / fp_sqrt / fp_reci
-> tile_mul -> row_mul_fp_at -> tile_add against a PyTorch
LayerNorm reference (no head-packing — D = hidden_size > MLEN).

FP preload: ``INV_N = 1/hidden_size``, ``EPS = 1e-6``, ``SS_INIT = 0``.
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
MLEN = 64
HIDDEN_SIZE = 128         # > MLEN -> exercises the multi-d_tile path
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS
EPS = 1e-6


_FP_CONSTS: dict[str, float] = {
    "INV_N":   1.0 / HIDDEN_SIZE,
    "EPS":     EPS,
    "SS_INIT": 0.0,
}


def parse_buffer_addrs(raw: dict) -> dict:
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])

    addrs = {name: addr_of(name) for name in _FP_CONSTS}
    # No head packing here: hardware_lane_count = 1, so per-row FP slot
    # size is just ROWS (one scalar per row).
    slot_words = ROWS
    addrs["fp_preload_end"] = max(addrs.values()) + slot_words
    return addrs


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x     = torch.randn(BATCH, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32) * 0.5
    scale = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.3 + 1.0
    bias  = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.1

    # LayerNorm golden:
    #   mu  = mean(x)
    #   var = mean((x - mu) ** 2)
    #   y   = (x - mu) * rsqrt(var + eps) * scale + bias
    mu       = x.mean(dim=-1, keepdim=True)
    xc       = x - mu
    var      = (xc * xc).mean(dim=-1, keepdim=True)
    inv      = torch.rsqrt(var + EPS)
    y_golden = xc * inv * scale + bias

    # PLENA doesn't have a VRAM-row broadcast tile op, so the host
    # expands the (H,) affine weights into (rows, H) once.
    scale_full = (
        scale.view(1, 1, 1, HIDDEN_SIZE)
        .expand(BATCH, SEQ_LEN, 1, HIDDEN_SIZE).contiguous()
    )
    bias_full = (
        bias.view(1, 1, 1, HIDDEN_SIZE)
        .expand(BATCH, SEQ_LEN, 1, HIDDEN_SIZE).contiguous()
    )

    golden_flat = y_golden.reshape(BATCH * SEQ_LEN, HIDDEN_SIZE)
    return {
        "hbm_inputs": {
            "X_hbm":     x,
            "SCALE_hbm": scale_full,
            "BIAS_hbm":  bias_full,
            "Y_hbm":     torch.zeros_like(x),
        },
        "golden_flat": golden_flat,
    }


def build_fp_preload(io: dict, addrs: dict):  # noqa: ARG001
    del io
    fp = torch.zeros(addrs["fp_preload_end"], dtype=torch.float16)
    slot_words = ROWS
    for name, value in _FP_CONSTS.items():
        base = addrs[name]
        fp[base : base + slot_words] = float(value)
    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    chunks_per_batch = (HIDDEN_SIZE + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * SEQ_LEN * chunks_per_batch,
        "num_batches": BATCH * SEQ_LEN,
        "elements_per_batch": HIDDEN_SIZE,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="layernorm_min",
    kernel="tilelang_tvm_compiler.kernels.layernorm_min:make_layernorm_min",
    kernel_kwargs={
        "rows": ROWS,
        "hidden_size": HIDDEN_SIZE,
        "num_s_blocks": NUM_S_BLOCKS,
        "batch": BATCH,
    },
    mlen=MLEN,
    stage_output="Y_hbm",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
