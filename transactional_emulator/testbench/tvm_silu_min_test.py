"""TVM minimal SiLU testbench.

Verifies sigmoid expansion (1 / (1 + exp(-x))) followed by ``x * sigmoid(x)``
against ``torch.nn.functional.silu``.

FP preload: ``ONE = 1.0`` and ``NEG_ONE = -1.0``, both as rank-1
fragments addressed via the kernel's --dump-buffer-addrs JSON.
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
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS


_FP_CONSTS: dict[str, float] = {
    "ONE":     1.0,
    "NEG_ONE": -1.0,
}


def parse_buffer_addrs(raw: dict) -> dict:
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])

    addrs = {name: addr_of(name) for name in _FP_CONSTS}
    slot_words = HARDWARE_LANE_COUNT * HLEN
    addrs["fp_preload_end"] = max(addrs.values()) + slot_words
    return addrs


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    y_golden = torch.nn.functional.silu(x)
    golden_flat = y_golden.reshape(BATCH * SEQ_LEN, HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "X_hbm": x,
            "Y_hbm": torch.zeros_like(x),
        },
        "golden_flat": golden_flat,
    }


def build_fp_preload(io: dict, addrs: dict):  # noqa: ARG001
    del io
    fp = torch.zeros(addrs["fp_preload_end"], dtype=torch.float16)
    slot_words = HARDWARE_LANE_COUNT * HLEN
    for name, value in _FP_CONSTS.items():
        base = addrs[name]
        fp[base : base + slot_words] = float(value)
    return fp


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
    asm_name="silu_min",
    kernel="tilelang_tvm_compiler.kernels.silu_min:make_silu_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="Y_hbm",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
