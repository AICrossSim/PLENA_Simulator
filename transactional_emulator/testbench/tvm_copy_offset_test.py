"""TVM copy-offset testbench — pure DMA probe for the o_head_offset path.

Segmented test: ``copy_offset_min`` is gelu_min's offset writeback with
ALL compute stripped out — just HBM -> VRAM -> HBM. If this fails, the
bug is purely in the offset DMA writeback.

``COPY_OFFSET`` env var selects o_head_offset (default 8 = write the
right half of a 2x-wide output). Set COPY_OFFSET=0 for the left-half
control case.
"""

from __future__ import annotations

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
    TvmTestbenchSpec,
    run,
    resolve_output_layout,
)


BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

HLEN = _HW.hlen  # from plena_settings.toml
MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per tile == mlen
HEAD_COUNT = 8
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS

O_HEAD_COUNT = 2 * HEAD_COUNT
O_HEAD_OFFSET = int(os.environ.get("COPY_OFFSET", "8"))
# COMPUTE selects the per-element FPRAM compute stage — see
# copy_offset_min._COMPUTE_STAGES. Default "copy" = plain VRAM->VRAM.
COMPUTE = os.environ.get("COMPUTE", "copy")


def _apply_compute(x: torch.Tensor) -> torch.Tensor:
    """Host reference matching copy_offset_min's ``compute`` stage."""
    if COMPUTE in ("copy", "id"):
        return x
    if COMPUTE == "mul":
        return x * x
    if COMPUTE == "const_mul":
        return 0.5 * x
    if COMPUTE == "exp":
        return torch.exp(x)
    if COMPUTE == "reci":
        return 1.0 / x
    raise ValueError(f"unknown COMPUTE stage {COMPUTE!r}")


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    # "reci" needs x away from 0 to stay numerically sane.
    if COMPUTE == "reci":
        x = x + 2.0

    y = _apply_compute(x)

    # Output is 2x wide; the kernel writes the result into the
    # head-slice [O_HEAD_OFFSET : O_HEAD_OFFSET + HEAD_COUNT].
    y_wide = torch.zeros(BATCH, SEQ_LEN, O_HEAD_COUNT, HLEN, dtype=torch.float32)
    y_wide[:, :, O_HEAD_OFFSET : O_HEAD_OFFSET + HEAD_COUNT, :] = y

    golden_flat = y_wide.reshape(BATCH * SEQ_LEN, O_HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "X_hbm": x,
            "Y_hbm": torch.zeros(BATCH, SEQ_LEN, O_HEAD_COUNT, HLEN, dtype=torch.float32),
        },
        "golden_flat": golden_flat,
    }


def parse_buffer_addrs(raw: dict) -> dict:
    """Trivial passthrough. Its mere PRESENCE (non-None) is what matters:
    it makes test_helper pass --dump-buffer-addrs to the compiler and
    run the hoisted-const auto-preload. The ``const_mul`` stage embeds
    ``T.float16(0.5)``, hoisted into a ``__const_f16_*`` global.fpram
    slot — without auto-preload that slot stays 0 and the kernel
    multiplies by 0. (Stages copy/id/mul have no hoisted const, so they
    pass even without this — which is exactly the false-negative that
    masked the missing hook.)
    """
    del raw
    return {}


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    # Geometry (num_rows / num_batches / elements_per_batch /
    # row_dim / use_stride_mode) from the canonical OutputLayout
    # so it agrees with golden_flat by construction.
    layout = resolve_output_layout(
        num_batches=BATCH * SEQ_LEN,
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
    asm_name="copy_offset",
    kernel="tilelang_tvm_compiler.kernels.copy_offset_min:make_copy_offset_min",
    kernel_kwargs={
        "rows": ROWS,
        "hlen": HLEN,
        "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS,
        "batch": BATCH,
        "o_head_count": O_HEAD_COUNT,
        "o_head_offset": O_HEAD_OFFSET,
        "compute": COMPUTE,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="Y_hbm",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    print(f"[copy_offset] o_head_count={O_HEAD_COUNT}, o_head_offset={O_HEAD_OFFSET}, compute={COMPUTE!r}")
    sys.exit(run(SPEC))
