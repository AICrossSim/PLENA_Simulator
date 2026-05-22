"""TVM minimal LayerNorm testbench.

Verifies the chain: row_reduce_sum_at -> fp_mul -> row_sub_fp_at ->
tile_mul -> row_reduce_sum_at -> fp_mul / fp_add / fp_sqrt / fp_reci
-> tile_mul -> row_mul_fp_at -> tile_add against a PyTorch
LayerNorm reference (no head-packing — D = hidden_size).

FP preload: ``INV_N = 1/hidden_size``, ``EPS = 1e-6``, ``SS_INIT = 0``.

Verification mode (matches linear / flash_attention):
  * MX-E4M3 round-trip on BOTH golden sides — every HBM tensor the
    behave_sim consumes is MX-packed, so the golden computes on
    MX-round-tripped inputs and the output Y is MX-round-tripped too.
  * HBM-direct compare (``check_hbm=True``) — reads Y straight from
    ``hbm_dump.bin`` (no VRAM re-stage / stride remap), diffed against
    the golden flattened through the canonical layout. Same path that
    gives linear cosine=1.0.
  * Fingerprint cache (``TB_CACHE=1``, default) — see ``_tb_cache``.
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

# Shared MX-E4M3 round-trip + fingerprint cache (factored out of the
# flash_attention testbench).
sys.path.insert(0, str(_THIS_FILE.parent))
from _tb_cache import mx_roundtrip, run_cached  # noqa: E402


BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per tile == mlen
HIDDEN_SIZE = 1024
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS
EPS = 1e-6


# Canonical output layout — single source of truth for golden flatten,
# exactly as tvm_linear_min_test.py does it. Y_hbm is logical
# (1, M, 1, HIDDEN_SIZE): rows = M, cols = HIDDEN_SIZE. The golden is
# flattened through this layout and the HBM-direct compare reads Y
# straight off hbm_dump.bin against it with NO reorder — the same
# mechanism that gives linear cosine=1.0.
_OUT_LAYOUT = resolve_output_layout(
    num_batches=BATCH * SEQ_LEN,
    elements_per_batch=HIDDEN_SIZE,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull Y_hbm's MXFP-packed byte base for the HBM-direct compare.

    INV_N (= 1/hidden_size) and EPS are inlined as ``T.float16(...)``
    literals in the kernel body; the compiler auto-hoists them into
    anonymous ``__const_f16_*`` global.fpram slots and ``test_helper``
    auto-preloads them — nothing to introspect there. The one thing we
    DO need is where Y lands in HBM: ``AddressAllocationPass`` sets
    ``Y_hbm["address"]`` to its MXFP-packed byte base, which IS
    ``result_hbm_start_byte``.
    """
    y = raw["Y_hbm"]
    return {"result_hbm_start_byte": int(y["address"])}


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x     = torch.randn(BATCH, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32) * 0.5
    scale = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.3 + 1.0
    bias  = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.1

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

    # The kernel reads X / scale / bias back from HBM already MX-E4M3
    # quantized, so the golden must compute on the MX-round-tripped inputs
    # (matches what the sim actually consumes — see linear's a_eff).
    x_eff     = mx_roundtrip(x)
    scale_eff = mx_roundtrip(scale_full)
    bias_eff  = mx_roundtrip(bias_full)

    # LayerNorm golden on the MX-round-tripped inputs:
    #   mu  = mean(x)
    #   var = mean((x - mu) ** 2)
    #   y   = (x - mu) * rsqrt(var + eps) * scale + bias
    mu       = x_eff.mean(dim=-1, keepdim=True)
    xc       = x_eff - mu
    var      = (xc * xc).mean(dim=-1, keepdim=True)
    inv      = torch.rsqrt(var + EPS)
    y_golden = xc * inv * scale_eff + bias_eff

    # The kernel writes Y to HBM as MX-E4M3, and the HBM-direct comparator
    # reads those exact bytes back — so the golden's Y must be
    # MX-round-tripped too, else we'd diff an fp32 ideal against
    # dequantized MX (spurious cosine~0).
    y_golden = mx_roundtrip(y_golden)

    return {
        "hbm_inputs": {
            "X_hbm":     x,
            "SCALE_hbm": scale_full,
            "BIAS_hbm":  bias_full,
            "Y_hbm":     torch.zeros_like(x),
        },
        # Flatten through the canonical layout so golden's element order
        # matches the HBM-direct comparator's flat read by construction.
        "golden_flat": _OUT_LAYOUT.flatten_golden(
            y_golden.reshape(BATCH * SEQ_LEN, HIDDEN_SIZE)
        ),
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io
    # HBM-direct compare, read straight off hbm_dump.bin — NO VRAM
    # staging, NO stride/reorder. Y is MX-E4M3 (1 byte/elem + 1/8-byte
    # E8M0 scale, block=8); use the comparator's default MX path plus the
    # scale region offset. Geometry comes from the canonical layout.
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
    asm_name="layernorm_min",
    kernel="tilelang_tvm_compiler.kernels.layernorm_min:make_layernorm_min",
    kernel_kwargs={
        "rows": ROWS,
        "hidden_size": HIDDEN_SIZE,
        "num_s_blocks": NUM_S_BLOCKS,
        "batch": BATCH,
    },
    mlen=MLEN,
    # No stage_output: HBM-direct compare reads the kernel's own Y_hbm
    # store from hbm_dump.bin, so there is no re-stage HBM->VRAM step.
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def _fingerprint() -> dict:
    return {
        "kernel": "layernorm_min",
        "batch": BATCH, "mlen": MLEN, "hidden_size": HIDDEN_SIZE,
        "num_s_blocks": NUM_S_BLOCKS, "seq_len": SEQ_LEN, "eps": EPS,
        "seed": 0, "schema": 1,
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
