"""TVM minimal GELU testbench.

Verifies:
  * lower_compound_fp_stores cast-peeling + per-subop tmp lowering
    for FP scalar fragments (chain runs through exp / reci / add /
    sub / mul).
  * Manual hand-expansion of tanh as ``1 - 2/(exp(2u)+1)`` inside the
    kernel against PyTorch's tanh-approx GELU.

Golden: ``torch.nn.functional.gelu(approximate="tanh")``.

FP preload mirrors flash_attention_min: the five GELU scalar constants
(0.5, 1.0, 2.0, sqrt(2/pi), 0.044715) are each declared as a rank-1
``local.fragment`` in the kernel; their FPRAM slot addresses are read
from the ``--dump-buffer-addrs`` JSON and the testbench fills every
``(lane, hlen)`` cell with the corresponding scalar value.

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
    TvmTestbenchSpec,
    resolve_output_layout,
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


_OUT_LAYOUT = resolve_output_layout(
    num_batches=BATCH * SEQ_LEN,
    elements_per_batch=HEAD_COUNT * HLEN,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull Y_hbm's MXFP-packed byte base for the HBM-direct compare.

    The five GELU scalar constants (0.5, 1.0, 2.0, sqrt(2/pi), 0.044715)
    are embedded as ``T.float16(...)`` literals; the compiler auto-hoists
    them into ``__const_f16_*`` global.fpram slots and ``test_helper``
    auto-preloads them from the dump's ``value`` field — which only
    happens because ``parse_buffer_addrs`` is non-None (the flag that
    triggers ``--dump-buffer-addrs``). Returning Y's address keeps that
    flag set AND wires up the HBM-direct compare.
    """
    return {"result_hbm_start_byte": int(raw["Y_hbm"]["address"])}


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    # The kernel reads X back from HBM already MX-E4M3 quantized, so the
    # golden computes GELU on the MX-round-tripped input.
    x_eff = mx_roundtrip(x)
    y_golden = torch.nn.functional.gelu(x_eff, approximate="tanh")

    # The kernel writes Y to HBM as MX-E4M3; HBM-direct reads those bytes
    # back, so the golden's Y is MX-round-tripped too.
    y_golden = mx_roundtrip(y_golden)

    return {
        "hbm_inputs": {
            "X_hbm": x,
            "Y_hbm": torch.zeros_like(x),
        },
        "golden_flat": _OUT_LAYOUT.flatten_golden(y_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * HLEN)),
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
    asm_name="gelu_min",
    kernel="tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
    kernel_kwargs={
        "rows": ROWS,
        "hlen": HLEN,
        "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS,
        "batch": BATCH,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def _fingerprint() -> dict:
    return {
        "kernel": "gelu_min",
        "batch": BATCH,
        "mlen": MLEN,
        "hlen": HLEN,
        "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS,
        "seq_len": SEQ_LEN,
        "seed": 0,
        "schema": 1,
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
