"""TVM minimal adaLN modulate testbench.

Verifies the VRAM elementwise pipeline (tile_mul + tile_add) by
computing ``y = (1 + scale) * x + shift``. The ``1 + scale`` term is
done host-side (testbench passes ``scale_plus_one`` directly), so the
kernel is two single-op stores over same-shape VRAM tiles.

Golden: PyTorch ``(1 + scale) * x + shift``.
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
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x     = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    scale = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    shift = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    y_golden = (1.0 + scale) * x + shift

    scale_plus_one = (1.0 + scale).to(torch.float32)  # host-side fold

    golden_flat = y_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "X_hbm":       x,
            "SCALE1P_hbm": scale_plus_one,
            "SHIFT_hbm":   shift,
            "Y_hbm":       torch.zeros_like(x),
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    chunks_per_batch = (HEAD_COUNT * HLEN + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * SEQ_LEN * chunks_per_batch,
        "num_batches": BATCH * SEQ_LEN,
        "elements_per_batch": HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="modulate_min",
    kernel="tilelang_tvm_compiler.kernels.modulate_min:make_modulate_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="Y_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
