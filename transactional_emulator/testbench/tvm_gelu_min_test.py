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
"""

from __future__ import annotations

import math
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

from tilelang_tvm_compiler.test_helper import TvmTestbenchSpec, run  # noqa: E402


BATCH = 1
ROWS = 64
HLEN = 16
MLEN = 64
HEAD_COUNT = 8
HARDWARE_LANE_COUNT = MLEN // HLEN
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS


# (kernel-buffer-name, scalar value) — order is irrelevant; the JSON
# tells the testbench each slot's actual FPRAM address.
_FP_CONSTS: dict[str, float] = {
    "HALF":      0.5,
    "ONE":       1.0,
    "TWO":       2.0,
    "SQRT_2_PI": 0.7978845608028654,  # sqrt(2/pi)
    "COEFF":     0.044715,
}


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull the address of every FP scalar constant slot the kernel
    declared. We also compute ``fp_preload_end`` as the byte offset
    just past the last constant slot so ``build_fp_preload`` knows how
    long its preload tensor must be."""
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])

    addrs = {name: addr_of(name) for name in _FP_CONSTS}
    # Each slot is shape ``(hardware_lane_count, hlen)`` after
    # cluster-expansion, totalling HARDWARE_LANE_COUNT * HLEN words.
    slot_words = HARDWARE_LANE_COUNT * HLEN
    last_addr = max(addrs.values())
    addrs["fp_preload_end"] = last_addr + slot_words
    return addrs


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    y_golden = torch.nn.functional.gelu(x, approximate="tanh")
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
    asm_name="gelu_min",
    kernel="tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
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
