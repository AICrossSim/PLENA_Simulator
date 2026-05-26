"""TVM minimal RMSNorm testbench.

Verifies the chain: tile_mul -> row_reduce_sum_at -> fp_mul / fp_add /
fp_sqrt / fp_reci -> tile_mul -> row_mul_fp_at, against a PyTorch
RMSNorm reference (no affine variance, with learnable ``scale``).

FP preload: ``INV_N = 1/hlen`` and ``EPS = 1e-6``.

Verification mode (matches linear / flash_attention / layernorm):
MX-E4M3 round-trip on BOTH golden sides + HBM-direct compare
(``check_hbm=True``) + fingerprint cache (``TB_CACHE=1``, default).
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
ROWS = MLEN  # rows per tile == mlen
HEAD_COUNT = 8
HARDWARE_LANE_COUNT = MLEN // HLEN
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS
EPS = 1e-6


_OUT_LAYOUT = resolve_output_layout(
    num_batches=BATCH * SEQ_LEN,
    elements_per_batch=HEAD_COUNT * HLEN,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull Y_hbm's MXFP-packed byte base for the HBM-direct compare.

    INV_N (= 1/hlen) and EPS are inlined ``T.float16(...)`` literals the
    compiler auto-hoists into ``__const_f16_*`` global.fpram slots;
    ``test_helper`` auto-preloads those from the dump's ``value`` field
    (independent of what this hook returns).
    """
    return {"result_hbm_start_byte": int(raw["Y_hbm"]["address"])}


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x     = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    scale = torch.randn(HLEN, dtype=torch.float32) * 0.3 + 1.0  # ≈ 1

    # Broadcast scale into (..., hlen) per element for the kernel —
    # PLENA doesn't have a VRAM-row broadcast tile_mul yet, so we
    # expand on the host.
    scale_full = scale.view(1, 1, 1, HLEN).expand(BATCH, SEQ_LEN, HEAD_COUNT, HLEN).contiguous()

    # The kernel reads X / SCALE back from HBM already MX-E4M3 quantized,
    # so the golden computes on the MX-round-tripped inputs.
    x_eff     = mx_roundtrip(x)
    scale_eff = mx_roundtrip(scale_full)

    # Golden: matches kernel's ``Y = (X * scale) * INV[row]`` form, the
    # standard RMSNorm output. INV = rsqrt(mean(x²) + eps).
    mean_sq  = (x_eff * x_eff).mean(dim=-1, keepdim=True)
    inv      = torch.rsqrt(mean_sq + EPS)
    y_golden = (x_eff * scale_eff) * inv  # broadcast over hlen

    # The kernel writes Y to HBM as MX-E4M3; HBM-direct reads those bytes
    # back, so the golden's Y is MX-round-tripped too.
    y_golden = mx_roundtrip(y_golden)

    return {
        "hbm_inputs": {
            "X_hbm":     x,
            "SCALE_hbm": scale_full,
            "Y_hbm":     torch.zeros_like(x),
        },
        "golden_flat": _OUT_LAYOUT.flatten_golden(
            y_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * HLEN)
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
    asm_name="rmsnorm_min",
    kernel="tilelang_tvm_compiler.kernels.rmsnorm_min:make_rmsnorm_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def _fingerprint() -> dict:
    return {
        "kernel": "rmsnorm_min",
        "batch": BATCH, "mlen": MLEN, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS, "seq_len": SEQ_LEN, "eps": EPS,
        "seed": 0, "schema": 1,
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
